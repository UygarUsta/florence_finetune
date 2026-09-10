import os
import json
import random
from glob import glob
from collections import Counter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename):
    imports = get_imports(filename)
    if str(filename).endswith("modeling_florence2.py") and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

MODEL_ID = "microsoft/Florence-2-large"
DATA_DIR = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_extended/"  # Resim ve JSON'ların bulunduğu klasör
OUTPUT_DIR = "./aug_florence2_quadbox_lora"
EPOCHS = 5
BATCH_SIZE = 2
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Çoklu nesne dengesizliği için ayarlar ---
# Veri setinde çoğunlukla görüntü başına 1 nesne varsa, model çıkarım zamanında
# 2+ nesne gördüğünde de "1 nesne üret, dur" alışkanlığına düşebilir. Bunu
# azaltmak için: (a) rastgele görüntüleri yan yana birleştirip sentetik
# çoklu-nesne örnekleri üretiyoruz, (b) nesne sayısına göre ağırlıklı örnekleme
# yapıyoruz ki nadir olan çoklu-nesne örnekler eğitimde daha sık görülsün.
MULTI_OBJECT_AUG_PROB = 0.35   # bir batch öğesinin sentetik-birleştirilmiş olma olasılığı
MAX_COMPOSE_IMAGES = 3         # birleştirilebilecek maksimum görüntü sayısı (2 veya 3)
USE_WEIGHTED_SAMPLER = True    # nesne sayısına göre ters-frekans ağırlıklı örnekleme


class QuadboxDataset(Dataset):
    def __init__(self, data_dir, multi_object_aug_prob=MULTI_OBJECT_AUG_PROB, max_compose=MAX_COMPOSE_IMAGES):
        json_paths = sorted(glob(os.path.join(data_dir, "*.json")))
        self.samples = []  # (json_path, çözümlenmiş_görsel_yolu)
        self.multi_object_aug_prob = multi_object_aug_prob
        self.max_compose = max(2, max_compose)

        for json_path in json_paths:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            img_path = self._resolve_image_path(os.path.dirname(json_path), data.get("imagePath", ""))
            if img_path is None:
                print(f"[UYARI] Görsel bulunamadı, atlanıyor: {json_path} -> {data.get('imagePath')}")
                continue

            self.samples.append((json_path, img_path))

        skipped = len(json_paths) - len(self.samples)
        print(f"Dataset: {len(json_paths)} json bulundu, {len(self.samples)} geçerli, {skipped} atlandı.")
        self._log_shape_count_distribution()

    def _log_shape_count_distribution(self):
        """Görüntü başına düşen (geçerli 4 noktalı) nesne sayısı dağılımını yazdırır.
        Ne kadar çarpık olduğunu görmeden dengeleme stratejisi seçmeyin."""
        counts = Counter(len(self._load_shapes(jp)) for jp, _ in self.samples)
        print("Görüntü başına nesne sayısı dağılımı:")
        for n in sorted(counts):
            print(f"  {n} nesne: {counts[n]} görüntü")

    @staticmethod
    def _resolve_image_path(dir_path, image_name):
        if not image_name:
            return None

        # 1) JSON'da yazılan isim birebir çalışıyor mu?
        direct = os.path.join(dir_path, image_name)
        if os.path.isfile(direct):
            return direct

        # 2) Aynı isim, farklı uzantı varyasyonlarıyla dene
        base, _ = os.path.splitext(image_name)
        candidate_exts = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP", ".webp", ".WEBP"]
        for ext in candidate_exts:
            candidate = os.path.join(dir_path, base + ext)
            if os.path.isfile(candidate):
                return candidate

        # 3) Klasördeki dosyaları case-insensitive tara (Linux'ta dosya sistemi case-sensitive olduğu için gerekli)
        try:
            for fname in os.listdir(dir_path):
                fbase, _ = os.path.splitext(fname)
                if fname.lower() == image_name.lower() or fbase.lower() == base.lower():
                    return os.path.join(dir_path, fname)
        except FileNotFoundError:
            pass

        return None  # Hiçbir şekilde bulunamadı

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_shapes(json_path):
        """Sadece 4 noktalı (quadbox) şekilleri döndürür."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [s for s in data.get("shapes", []) if len(s["points"]) == 4]

    @staticmethod
    def _shapes_to_target_str(shapes, width, height):
        target_str = ""
        for shape in shapes:
            loc_tokens = ""
            for pt in shape["points"]:
                x_norm = int(round(min(max(pt[0] / width, 0.0), 1.0) * 1000))
                y_norm = int(round(min(max(pt[1] / height, 0.0), 1.0) * 1000))
                loc_tokens += f"<loc_{x_norm}><loc_{y_norm}>"
            target_str += f"{shape['label']}{loc_tokens}"
        return target_str

    def _load_single(self, idx):
        """Tek bir örneği (görsel + geçerli şekiller) yükler, bozuksa bir sonrakine geçer."""
        json_path, img_path = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"[UYARI] Görsel açılamadı, atlanıyor: {img_path} ({e})")
            return self._load_single((idx + 1) % len(self.samples))
        return image, self._load_shapes(json_path)

    def _compose_multi(self, idx):
        """idx'teki örnek dahil, 2-3 rastgele görüntüyü aynı yükseklikte yan yana
        birleştirip tek bir sentetik çoklu-nesne örneği üretir. Koordinatlar
        birleşik tuval üzerine yeniden ölçeklenip kaydırılır."""
        n = random.randint(2, self.max_compose)
        other_idxs = random.sample(
            [i for i in range(len(self.samples)) if i != idx],
            k=min(n - 1, len(self.samples) - 1)
        )
        images, shape_lists = [], []
        for i in [idx] + other_idxs:
            img, shapes = self._load_single(i)
            images.append(img)
            shape_lists.append(shapes)

        target_h = min(img.height for img in images)
        canvas_w = 0
        resized_with_scale = []
        for img in images:
            scale = target_h / img.height
            new_w = max(1, int(round(img.width * scale)))
            resized_with_scale.append((img.resize((new_w, target_h)), scale))
            canvas_w += new_w

        canvas = Image.new("RGB", (canvas_w, target_h))
        merged_shapes = []
        x_offset = 0
        for (resized_img, scale), shapes in zip(resized_with_scale, shape_lists):
            canvas.paste(resized_img, (x_offset, 0))
            for shape in shapes:
                new_points = [[pt[0] * scale + x_offset, pt[1] * scale] for pt in shape["points"]]
                merged_shapes.append({"label": shape["label"], "points": new_points})
            x_offset += resized_img.width

        return canvas, merged_shapes

    def __getitem__(self, idx):
        should_compose = (
            self.multi_object_aug_prob > 0
            and len(self.samples) > 1
            and random.random() < self.multi_object_aug_prob
        )
        if should_compose:
            image, shapes = self._compose_multi(idx)
        else:
            image, shapes = self._load_single(idx)

        width, height = image.size
        target_str = self._shapes_to_target_str(shapes, width, height)

        return {
            "image": image,
            "prefix": "<QUADBOX_DETECTION>",
            "target": target_str
        }

    def shape_count_sample_weights(self):
        """WeightedRandomSampler için: her örneğin nesne sayısına göre ters-frekans
        ağırlığı. Az görülen nesne sayıları (ör. 2, 3 nesneli görüntüler) daha
        sık örneklenir; bu, sentetik augmentation'ı gerçek çoklu-nesne
        örnekleriyle destekler."""
        counts_per_sample = [len(self._load_shapes(jp)) for jp, _ in self.samples]
        freq = Counter(counts_per_sample)
        return [1.0 / freq[c] for c in counts_per_sample]

def collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    prompts = [item["prefix"] for item in batch]
    targets = [item["target"] for item in batch]

    # Girişleri tokenize et
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    # Hedef etiketleri tokenize et
    labels = processor.tokenizer(text=targets, return_tensors="pt", padding=True).input_ids

    # Pad token'ları -100 yaparak loss hesabından çıkar
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs

def main():


    # processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_ID, 
    #     torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32, 
    #     trust_remote_code=True,
    #     attn_implementation="sdpa"
    # )
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            trust_remote_code=True,
            attn_implementation="sdpa"
        )

    

    # LoRA Konfigürasyonu
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            # Dil modeli (mevcut, aynı kalıyor)
            "q_proj", "k_proj", "v_proj", "out_proj",
            # DaViT vision encoder — attention'a özgü tam yollar,
            # "vision_tower.convs.*.proj" (Conv2d, patch embedding) YAKALANMAZ
            "window_attn.fn.qkv", "window_attn.fn.proj",
            "channel_attn.fn.qkv", "channel_attn.fn.proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(DEVICE)

    dataset = QuadboxDataset(DATA_DIR)

    if USE_WEIGHTED_SAMPLER:
        from torch.utils.data import WeightedRandomSampler
        weights = dataset.shape_count_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,  # shuffle=True ile birlikte KULLANILMAZ
            collate_fn=lambda b: collate_fn(b, processor)
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, processor)
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    lr_scheduler = get_scheduler(
        "linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=EPOCHS * len(dataloader)
    )

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            batch = {
                k: (v.to(DEVICE, dtype=DTYPE) if v.is_floating_point() else v.to(DEVICE))
                for k, v in batch.items()
            }
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch + 1} Ortalama Loss: {total_loss / len(dataloader):.4f}")

    # Sadece LoRA ağırlıklarını kaydet
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adaptörü kaydedildi: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()