import os
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
from transformers import AutoProcessor, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

MODEL_ID = "microsoft/Florence-2-large"
DATA_DIR = "/home/uygarusta/classification/vehicle_text_classification/dataset/"  # Her alt klasörü bir sınıf olan kök klasör
OUTPUT_DIR = "./florence2_classification_lora_vehicle_text"
TASK_PROMPT = "<CLASSIFICATION>"  # quadbox_*/ocr adaptörleriyle aynı base modelde birlikte yaşayacak yeni görev tag'i
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".PNG", ".JPEG"}
EPOCHS = 5
BATCH_SIZE = 2
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ClassificationDataset(Dataset):
    """
    JSON'a gerek yok — etiket doğrudan klasör isminden geliyor. DATA_DIR altındaki
    her alt klasör bir sınıfı temsil eder, o klasördeki tüm görseller o sınıfın
    örneği olarak toplanır:

        DATA_DIR/
          ruhsat/
            img1.jpg
            img2.png
          kimlik/
            img3.jpg
          ...
    """

    def __init__(self, data_dir):
        self.samples = []  # (img_path, label)

        class_dirs = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )

        for label in class_dirs:
            class_dir = os.path.join(data_dir, label)
            for fname in sorted(os.listdir(class_dir)):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                    self.samples.append((os.path.join(class_dir, fname), label))

        print(f"Dataset: {len(class_dirs)} sınıf klasörü, {len(self.samples)} görsel bulundu.")
        print(f"Sınıflar: {class_dirs}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"[UYARI] Görsel açılamadı, atlanıyor: {img_path} ({e})")
            return self.__getitem__((idx + 1) % len(self.samples))

        return {
            "image": image,
            "prefix": TASK_PROMPT,
            "target": label,
        }


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

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            trust_remote_code=True,
            attn_implementation="sdpa"
        )

    # LoRA config quadbox/OCR adaptörlerinle BİREBİR aynı: aynı base model + aynı
    # PeftModel içine yüklenip set_adapter() ile serbestçe geçiş yapabilmen için şart.
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
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(DEVICE)

    dataset = ClassificationDataset(DATA_DIR)
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
