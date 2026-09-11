from unittest.mock import patch
from PIL import Image
import torch
import transformers.utils.import_utils
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

# --- flash_attn workaround'ları (training script ile aynı) ---
transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
transformers.utils.import_utils._is_package_available = (
    lambda pkg_name, *args, **kwargs: False if pkg_name == "flash_attn" else True
)

def fixed_get_imports(filename):
    imports = get_imports(filename)
    if str(filename).endswith("modeling_florence2.py") and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

BASE_MODEL_ID = "microsoft/Florence-2-large"
LORA_PATH = "./florence2_classification_lora_vehicle_text"
IMAGE_PATH = "test_images/notext_image.png"
TASK_PROMPT = "<CLASSIFICATION>"  # eğitimdeki TASK_PROMPT ile birebir aynı olmalı
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32  # eğitimle aynı dtype


def main():
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=DTYPE,
            trust_remote_code=True,
            attn_implementation="sdpa"
        )

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.to(DEVICE)
    model.eval()

    image = Image.open(IMAGE_PATH).convert("RGB")

    inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt")
    inputs = {
        k: (v.to(DEVICE, dtype=DTYPE) if v.is_floating_point() else v.to(DEVICE))
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=32,  # çıktı tek bir sınıf etiketi, quadbox/OCR'daki 512'ye gerek yok
            num_beams=3,
            do_sample=False
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    predicted_label = generated_text.replace(TASK_PROMPT, "").replace("<s>", "").replace("</s>", "").strip()

    print("\n--- Model Çıktısı (Raw) ---")
    print(generated_text)
    print("\n--- Tahmin Edilen Sınıf ---")
    print(predicted_label)


if __name__ == "__main__":
    main()
