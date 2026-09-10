import os
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
DATA_DIR = "/home/uygarusta/qwen_2_5_vl_3b_finetuning/dataset/ktt_final_1733/train/"  # Resim ve .txt karşılıklarının bulunduğu klasör
OUTPUT_DIR = "./florence2_ocr_lora"
EPOCHS = 5
BATCH_SIZE = 2
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP", ".webp", ".WEBP"]


class OCRDataset(Dataset):
    def __init__(self, data_dir):
        txt_paths = sorted(glob(os.path.join(data_dir, "*.txt")))
        self.samples = []  # (txt_path, çözümlenmiş_görsel_yolu)

        for txt_path in txt_paths:
            img_path = self._resolve_image_path(txt_path)
            if img_path is None:
                print(f"[UYARI] Görsel bulunamadı, atlanıyor: {txt_path}")
                continue
            self.samples.append((txt_path, img_path))

        skipped = len(txt_paths) - len(self.samples)
        print(f"Dataset: {len(txt_paths)} txt bulundu, {len(self.samples)} geçerli, {skipped} atlandı.")

    @staticmethod
    def _resolve_image_path(txt_path):
        base, _ = os.path.splitext(txt_path)

        # 1) Aynı isim, farklı uzantı varyasyonlarıyla dene
        for ext in IMAGE_EXTS:
            candidate = base + ext
            if os.path.isfile(candidate):
                return candidate

        # 2) Klasördeki dosyaları case-insensitive tara (Linux'ta dosya sistemi
        # case-sensitive olduğu için gerekli)
        dir_path = os.path.dirname(txt_path)
        base_name = os.path.basename(base)
        lower_exts = [e.lower() for e in IMAGE_EXTS]
        try:
            for fname in os.listdir(dir_path):
                fbase, fext = os.path.splitext(fname)
                if fbase.lower() == base_name.lower() and fext.lower() in lower_exts:
                    return os.path.join(dir_path, fname)
        except FileNotFoundError:
            pass

        return None  # Hiçbir şekilde bulunamadı

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        txt_path, img_path = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"[UYARI] Görsel açılamadı, atlanıyor: {img_path} ({e})")
            return self.__getitem__((idx + 1) % len(self.samples))

        with open(txt_path, "r", encoding="utf-8") as f:
            target_str = f.read().strip()

        return {
            "image": image,
            "prefix": "<OCR>",
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

    dataset = OCRDataset(DATA_DIR)
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
