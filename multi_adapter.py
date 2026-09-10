from unittest.mock import patch
import torch
import transformers.utils.import_utils
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
import re 

transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
transformers.utils.import_utils._is_package_available = (
    lambda pkg_name, *args, **kwargs: False if pkg_name == "flash_attn" else True
)
 
BOX_COLOR = (255, 0, 0)      # kırmızı çizgi
BOX_WIDTH = 3
LABEL_BG = (255, 0, 0)
LABEL_TEXT_COLOR = (255, 255, 255) 
 
def fixed_get_imports(filename):
    imports = get_imports(filename)
    if str(filename).endswith("modeling_florence2.py") and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def parse_quadbox_predictions(text_output, img_width, img_height):
    """
    Model çıktısındaki 'label<loc_x1><loc_y1>...<loc_x4><loc_y4>' yapısını ayrıştırır.
    """
    pattern = r"([a-zA-Z0-9_\-]+)((?:<loc_\d+>){8})"
    matches = re.findall(pattern, text_output)

    results = []
    for label, coords_str in matches:
        raw_coords = [int(x) for x in re.findall(r"<loc_(\d+)>", coords_str)]

        pixel_points = []
        for i in range(0, 8, 2):
            x = (raw_coords[i] / 1000.0) * img_width
            y = (raw_coords[i+1] / 1000.0) * img_height
            pixel_points.append([round(x, 2), round(y, 2)])

        results.append({
            "label": label,
            "points": pixel_points
        })
    return results

def visualize_detections(image, detections, output_path):
    """
    Quadbox tahminlerini görsel üzerine çizip dosyaya kaydeder.
    """
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=18)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for det in detections:
        points = [tuple(pt) for pt in det["points"]]
        # Poligonu kapat (4 nokta -> döngü halinde çiz)
        draw.polygon(points, outline=BOX_COLOR, width=BOX_WIDTH)

        label_text = det["label"]
        text_x, text_y = points[0]

        # Metin arkaplanı için kutu boyutunu hesapla
        try:
            bbox = draw.textbbox((text_x, text_y), label_text, font=font)
        except AttributeError:  # eski Pillow sürümleri için fallback
            text_w, text_h = draw.textsize(label_text, font=font)
            bbox = (text_x, text_y, text_x + text_w, text_y + text_h)

        draw.rectangle(bbox, fill=LABEL_BG)
        draw.text((text_x, text_y), label_text, fill=LABEL_TEXT_COLOR, font=font)

    vis_image.save(output_path)
    print(f"Görselleştirilmiş çıktı kaydedildi: {output_path}")
 
 
BASE_MODEL_ID = "microsoft/Florence-2-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
 
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
 
# Base modeli SADECE BIR KEZ yukleyin
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
 
# Ilk adaptoru PeftModel'e donusturerek yukleyin
model = PeftModel.from_pretrained(base_model, "./florence2_quadbox_lora_card", adapter_name="quadbox_card")
 
# Diger goreve ozgu LoRA'lari ayni PeftModel'e ekleyin
model.load_adapter("./aug_florence2_quadbox_lora_ruhsat", adapter_name="quadbox_ruhsat")
model.load_adapter("./florence2_quadbox_lora_dikey_kisit", adapter_name="quadbox_dikey_kisit")
# ... kac goreviniz varsa buraya ekleyin
 
model.to(DEVICE)
model.eval()
 
 
def run_task(image, prompt, adapter_name, max_new_tokens=512, num_beams=3):
    model.set_adapter(adapter_name)  # aktif LoRA'yi degistirir, model yeniden yuklenmez
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {
        k: (v.to(DEVICE, dtype=DTYPE) if v.is_floating_point() else v.to(DEVICE))
        for k, v in inputs.items()
    }
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
 
 
# Kullanim ornegi:
OUTPUT_PATH = "multi_adapter_prediction_output.jpg"

IMAGE_PATH = "test_images/2b60699d-a349-4c0e-a44f-75f87976ca6e.jpeg"
image = Image.open(IMAGE_PATH).convert("RGB")
width, height = image.size

prompt = "<QUADBOX_DETECTION>"

result_card = run_task(image, "<QUADBOX_DETECTION>", "quadbox_card")
result_ruhsat   = run_task(image,   "<QUADBOX_DETECTION>", "quadbox_ruhsat")

clean_output = result_ruhsat.replace(prompt, "").replace("<s>", "").replace("</s>", "").strip()
detections = parse_quadbox_predictions(clean_output, width, height)

print("\n--- Model Çıktısı (Raw) ---")
print(clean_output)
print("\n--- Çözümlenmiş Quadbox Koordinatları ---")
for det in detections:
    print(f"Etiket: {det['label']}")
    print(f"Noktalar: {det['points']}\n")

if detections:
    visualize_detections(image, detections, OUTPUT_PATH)



#Ortalama 2.6 gb vram 