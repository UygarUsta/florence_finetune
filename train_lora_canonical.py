"""
Canonical quadbox LoRA eğitimi (aug_train_lora.py tabanlı).

Farklar:
  * Hedef dizi: label + 8 loc (GEOMETRİK sıra) + rot{k}  -> bkz. canonical_quad.py
  * Rotasyon augmentation (90° katları + küçük açı). Nokta sırası korunduğu için
    k otomatik doğru hesaplanır ve yön sınıfları dengelenir.
  * Açılışta veri tutarlılık kontrolü: sarım yönü (CW/CCW) çoğunluğa uymayan veya
    dejenere poligon içeren örnekler atlanır ve listelenir.
  * Şekiller init'te belleğe alınır (her __getitem__'da JSON tekrar okunmaz).

LoRA config diğer adaptörlerle BİREBİR aynı ve vocab değişmiyor; multi-adapter
kurulumuna doğrudan eklenebilir.
"""
import os
# DataLoader worker'ları fork edilmeden önce ana süreçte tokenizer kullanıldığı için
# HF tokenizers uyarı basar. Worker'larda paralelliği baştan kapatıyoruz (zararsız).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoProcessor, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from canonical_quad import (
    winding,
    canonical_to_geometric,
    rotate_sample,
    shapes_to_target_str,
)

MODEL_ID = "microsoft/Florence-2-large"
DATA_DIR = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_extended/"
OUTPUT_DIR = "./florence2_canonical_quadbox_lora"
TASK_PROMPT = "<QUADBOX_DETECTION>"
EPOCHS = 5
BATCH_SIZE = 2
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 4

# --- Çoklu nesne ---
# aug_train_lora.py'deki değerler "görüntü başına çoğunlukla 1 nesne" veri seti
# içindi. Görüntü başına ~10 alan olan veride:
#  * Ağırlıklı sampler her nesne-sayısı grubuna EŞİT pay verir; 12 grup varsa
#    2064 görüntülük ana grup örneklemenin sadece ~%8'ini alır, birkaç görüntülük
#    nadir gruplar sürekli tekrarlanır. Bu yüzden kapalı.
#  * Compose, 2-3 belgeyi aynı 768px girdiye sıkıştırır; küçük alanlarda çözünürlük
#    kaybı yaratır. Birden fazla belgeli görüntüler (17-20 nesne) için düşük tutuldu.
MULTI_OBJECT_AUG_PROB = 0.1
MAX_COMPOSE_IMAGES = 2
USE_WEIGHTED_SAMPLER = False

# --- Rotasyon augmentation ---
# ROT90_AUG_PROB=0.75 ve {90,180,270}'den uniform seçim => döndürülmemiş %25 ile
# birlikte 4 yön eşit olasılıklı. Gerçek veride zaten dönük örnek varsa yine de
# dengeli kalır; sadece upright çalışacaksanız düşürün.
ROT90_AUG_PROB = 0.75
SMALL_ANGLE_PROB = 0.5
SMALL_ANGLE_MAX = 15.0      # derece; 45°'ye yaklaştıkça geometrik başlangıç belirsizleşir
ROT_FILL = (0, 0, 0)

# --- Veri kontrolü ---
MIN_POLY_AREA = 1.0         # piksel^2; altındaki poligonlar dejenere sayılır


def resolve_image_path(dir_path, image_name):
    if not image_name:
        return None
    direct = os.path.join(dir_path, image_name)
    if os.path.isfile(direct):
        return direct
    base, _ = os.path.splitext(image_name)
    for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP", ".webp", ".WEBP"]:
        candidate = os.path.join(dir_path, base + ext)
        if os.path.isfile(candidate):
            return candidate
    try:
        for fname in os.listdir(dir_path):
            fbase, _ = os.path.splitext(fname)
            if fname.lower() == image_name.lower() or fbase.lower() == base.lower():
                return os.path.join(dir_path, fname)
    except FileNotFoundError:
        pass
    return None


class CanonicalQuadDataset(Dataset):
    def __init__(self, data_dir, augment=True):
        self.augment = augment
        json_paths = sorted(glob(os.path.join(data_dir, "*.json")))

        raw = []  # (img_path, shapes, windings)
        missing_img = 0
        for json_path in json_paths:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            img_path = resolve_image_path(os.path.dirname(json_path), data.get("imagePath", ""))
            if img_path is None:
                missing_img += 1
                print(f"[UYARI] Görsel bulunamadı, atlanıyor: {json_path}")
                continue
            shapes = [
                {"label": s["label"], "points": [list(map(float, p)) for p in s["points"]]}
                for s in data.get("shapes", []) if len(s["points"]) == 4
            ]
            windings = [winding(s["points"], MIN_POLY_AREA) for s in shapes]
            raw.append((json_path, img_path, shapes, windings))

        # Sarım yönü çoğunluğunu bul (etiketleme konvansiyonun)
        w_counts = Counter(w for *_, ws in raw for w in ws)
        print(f"Sarım yönü dağılımı (poligon): saat yönü={w_counts[1]}, "
              f"ters={w_counts[-1]}, dejenere={w_counts[0]}")
        self.majority_winding = 1 if w_counts[1] >= w_counts[-1] else -1
        print(f"Etiketleme yönü olarak kabul edilen: "
              f"{'saat yönü' if self.majority_winding == 1 else 'saat yönünün tersi'}")

        # Tutarsız örnekleri komple atla: tek bir poligonu düşürmek, görüntüde
        # var olan nesneyi "yok" diye öğretir.
        self.samples = []
        bad = []
        for json_path, img_path, shapes, ws in raw:
            if any(w != self.majority_winding for w in ws):
                bad.append(json_path)
                continue
            self.samples.append((img_path, shapes))

        print(f"Dataset: {len(json_paths)} json, {missing_img} görselsiz, "
              f"{len(bad)} tutarsız sarım/dejenere, {len(self.samples)} geçerli.")
        if bad:
            print("Tutarsız örnekler (ilk 20) — etiketi kontrol edin:")
            for p in bad[:20]:
                print(f"  {p}")

        self._log_distributions()

    def _log_distributions(self):
        counts = Counter(len(shapes) for _, shapes in self.samples)
        print("Görüntü başına nesne sayısı:")
        for n in sorted(counts):
            print(f"  {n} nesne: {counts[n]} görüntü")
        # Ham verideki yön (k) dağılımı — augmentation öncesi ne kadar dengesiz?
        ks = Counter(
            canonical_to_geometric(s["points"])[1]
            for _, shapes in self.samples for s in shapes
        )
        print("Ham verideki rot dağılımı (augmentation öncesi):")
        for k in range(4):
            print(f"  rot{k}: {ks[k]}")

    def __len__(self):
        return len(self.samples)

    def _load_single(self, idx):
        img_path, shapes = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"[UYARI] Görsel açılamadı, atlanıyor: {img_path} ({e})")
            return self._load_single((idx + 1) % len(self.samples))
        return image, shapes

    def _compose_multi(self, idx):
        n = random.randint(2, max(2, MAX_COMPOSE_IMAGES))
        other_idxs = random.sample(
            [i for i in range(len(self.samples)) if i != idx],
            k=min(n - 1, len(self.samples) - 1),
        )
        images, shape_lists = [], []
        for i in [idx] + other_idxs:
            img, shapes = self._load_single(i)
            images.append(img)
            shape_lists.append(shapes)

        target_h = min(img.height for img in images)
        resized = []
        canvas_w = 0
        for img in images:
            scale = target_h / img.height
            new_w = max(1, int(round(img.width * scale)))
            resized.append((img.resize((new_w, target_h)), scale))
            canvas_w += new_w

        canvas = Image.new("RGB", (canvas_w, target_h))
        merged, x_off = [], 0
        for (r_img, scale), shapes in zip(resized, shape_lists):
            canvas.paste(r_img, (x_off, 0))
            for s in shapes:
                merged.append({
                    "label": s["label"],
                    "points": [[x * scale + x_off, y * scale] for x, y in s["points"]],
                })
            x_off += r_img.width
        return canvas, merged

    def _rotate_aug(self, image, shapes):
        angle = 0.0
        if random.random() < ROT90_AUG_PROB:
            angle += random.choice([90, 180, 270])
        if random.random() < SMALL_ANGLE_PROB:
            angle += random.uniform(-SMALL_ANGLE_MAX, SMALL_ANGLE_MAX)
        return rotate_sample(image, shapes, angle, fill=ROT_FILL)

    def __getitem__(self, idx):
        if (self.augment and MULTI_OBJECT_AUG_PROB > 0 and len(self.samples) > 1
                and random.random() < MULTI_OBJECT_AUG_PROB):
            image, shapes = self._compose_multi(idx)
        else:
            image, shapes = self._load_single(idx)

        if self.augment:
            image, shapes = self._rotate_aug(image, shapes)

        width, height = image.size
        return {
            "image": image,
            "prefix": TASK_PROMPT,
            "target": shapes_to_target_str(shapes, width, height),
        }

    def shape_count_sample_weights(self):
        counts = [len(shapes) for _, shapes in self.samples]
        freq = Counter(counts)
        return [1.0 / freq[c] for c in counts]


def collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    prompts = [item["prefix"] for item in batch]
    targets = [item["target"] for item in batch]

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    labels = processor.tokenizer(text=targets, return_tensors="pt", padding=True).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


def main():
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    dataset = CanonicalQuadDataset(DATA_DIR)

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

    # Hedef dizinin tokenizasyonunu bir kez göster (rot token'ı nasıl bölünüyor?)
    sample_target = dataset[0]["target"]
    print("\nÖrnek hedef:", sample_target[:200])
    print("Token'lar   :", processor.tokenizer.tokenize(sample_target)[-12:])

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj",
            "window_attn.fn.qkv", "window_attn.fn.proj",
            "channel_attn.fn.qkv", "channel_attn.fn.proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(DEVICE)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=DEVICE == "cuda",
    )
    if USE_WEIGHTED_SAMPLER:
        weights = dataset.shape_count_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        dataloader = DataLoader(dataset, sampler=sampler, **loader_kwargs)
    else:
        dataloader = DataLoader(dataset, shuffle=True, **loader_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=EPOCHS * len(dataloader),
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
            loss = model(**batch).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch + 1} Ortalama Loss: {total_loss / len(dataloader):.4f}")

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adaptörü kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()