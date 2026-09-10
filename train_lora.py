import os
import json
from glob import glob
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
DATA_DIR = "/home/uygarusta/datasets/card_merged_datasets/merged_datasets/"  # Resim ve JSON'ların bulunduğu klasör
OUTPUT_DIR = "./florence2_quadbox_lora"
EPOCHS = 5
BATCH_SIZE = 2
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class QuadboxDataset(Dataset):
    def __init__(self, data_dir):
        json_paths = sorted(glob(os.path.join(data_dir, "*.json")))
        self.samples = []  # (json_path, çözümlenmiş_görsel_yolu)

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

    def __getitem__(self, idx):
        json_path, img_path = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"[UYARI] Görsel açılamadı, atlanıyor: {img_path} ({e})")
            return self.__getitem__((idx + 1) % len(self.samples))

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        width, height = image.size
        target_str = ""
        for shape in data.get("shapes", []):
            label = shape["label"]
            points = shape["points"]
            if len(points) == 4:
                loc_tokens = ""
                for pt in points:
                    x_norm = int(round(min(max(pt[0] / width, 0.0), 1.0) * 1000))
                    y_norm = int(round(min(max(pt[1] / height, 0.0), 1.0) * 1000))
                    loc_tokens += f"<loc_{x_norm}><loc_{y_norm}>"
                target_str += f"{label}{loc_tokens}"

        return {
            "image": image,
            "prefix": "<QUADBOX_DETECTION>",
            "target": target_str
        }

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