"""
Canonical quadbox LoRA inference — tek görsel veya klasör.

Çıktılar:
  * Görselleştirme: poligon + köşe numaraları (1 = canonical başlangıç, büyük nokta)
    + etiket yanında rot{k}. Canonical sıranın doğru gelip gelmediğini gözle
    kontrol etmek için.
  * (Opsiyonel) Düzleştirilmiş kırpıntı: canonical sıra kullanılarak perspektif
    dönüşümü. Ayrı bir rotasyon sınıflandırıcısına gerek kalmadan dik çıkmalı.
"""
import os
from glob import glob
from unittest.mock import patch

import numpy as np
import torch
import transformers.utils.import_utils
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from peft import PeftModel

from canonical_quad import parse_canonical_predictions

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
LORA_PATH = "./florence2_canonical_quadbox_lora"
INPUT_PATH = "test_images/0f973320-492d-4709-8858-30ca2a3a3dbf.jpeg"          # tek dosya ya da klasör
OUTPUT_DIR = "predictions_canonical"
TASK_PROMPT = "<QUADBOX_DETECTION>"
SAVE_WARPED = True
# Etiketleme yönü: 1 = saat yönü, -1 = saat yönünün tersi.
# Eğitim logundaki "Etiketleme yönü olarak kabul edilen" satırıyla aynı olmalı.
LABEL_WINDING = -1
# Dik (upright) belgelerin k değeri = eğitim logundaki "Ham rot dağılımı"nda
# açık ara en sık görülen k. Etiketlemeye hangi köşeden başladığını kodlar;
# düzleştirmede hedef köşeleri ve görselleştirmedeki açı bundan türetilir.
UPRIGHT_K = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

BOX_COLOR = (255, 0, 0)
START_COLOR = (0, 200, 0)
LABEL_TEXT_COLOR = (255, 255, 255)


def load_model():
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True, attn_implementation="sdpa"
        )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.to(DEVICE).eval()
    return processor, model


def predict(processor, model, image, max_new_tokens=512, num_beams=3):
    inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt")
    inputs = {
        k: (v.to(DEVICE, dtype=DTYPE) if v.is_floating_point() else v.to(DEVICE))
        for k, v in inputs.items()
    }
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
    text = processor.batch_decode(ids, skip_special_tokens=False)[0]
    clean = text.replace(TASK_PROMPT, "").replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()
    return clean, parse_canonical_predictions(clean, *image.size)


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except (OSError, IOError):
        return ImageFont.load_default()


def doc_rotation_deg(k):
    """Belgenin dik konuma göre saat yönünün tersine kaba dönüşü (0/90/180/270)."""
    r = (k - UPRIGHT_K) % 4
    return (LABEL_WINDING * r * 90) % 360


def visualize(image, detections, output_path):
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    font, small = _font(18), _font(16)
    r = max(4, min(image.size) // 120)

    for det in detections:
        pts = [tuple(p) for p in det["points"]]
        draw.polygon(pts, outline=BOX_COLOR, width=3)
        for i, (x, y) in enumerate(pts):
            rr = r * 2 if i == 0 else r
            color = START_COLOR if i == 0 else BOX_COLOR
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=color)
            draw.text((x + rr + 2, y - rr), str(i + 1), fill=color, font=small)

        text = f"{det['label']} rot{det['k']} ({doc_rotation_deg(det['k'])}°)" + (
            " (rot yok!)" if det["rot_missing"] else "")
        bbox = draw.textbbox(pts[0], text, font=font)
        draw.rectangle(bbox, fill=BOX_COLOR)
        draw.text(pts[0], text, fill=LABEL_TEXT_COLOR, font=font)

    vis.save(output_path)


def warp(image, canon_points):
    """Canonical sırayı kullanarak belgeyi düzleştirir (cv2 gerekir)."""
    import cv2

    # Canonical sırayı, belgenin kendi sol-üst köşesinden başlayacak şekilde kaydır:
    # p[0]=belge sol-üst, sonra etiketleme yönünde devam.
    p = np.roll(np.asarray(canon_points, dtype=np.float32), -UPRIGHT_K, axis=0)
    if LABEL_WINDING == 1:   # sol-üst, sağ-üst, sağ-alt, sol-alt
        w = int(round(max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3]))))
        h = int(round(max(np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1]))))
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    else:                    # sol-üst, sol-alt, sağ-alt, sağ-üst
        h = int(round(max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3]))))
        w = int(round(max(np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1]))))
        dst = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
    M = cv2.getPerspectiveTransform(p, dst)
    out = cv2.warpPerspective(np.array(image), M, (max(w, 1), max(h, 1)))
    return Image.fromarray(out)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processor, model = load_model()

    if os.path.isdir(INPUT_PATH):
        paths = sorted(p for p in glob(os.path.join(INPUT_PATH, "*"))
                       if os.path.splitext(p)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    else:
        paths = [INPUT_PATH]

    for path in tqdm(paths):
        image = Image.open(path).convert("RGB")
        raw, dets = predict(processor, model, image)
        name, _ = os.path.splitext(os.path.basename(path))

        print(f"\n[{name}] {raw}")
        for d in dets:
            flag = "  [UYARI: rot token yok, k=0 varsayıldı]" if d["rot_missing"] else ""
            print(f"  {d['label']} rot{d['k']} ({doc_rotation_deg(d['k'])}°) -> {d['points']}{flag}")

        if not dets:
            print("  [UYARI] Tespit yok.")
            continue

        visualize(image, dets, os.path.join(OUTPUT_DIR, f"{name}_vis.jpg"))
        if SAVE_WARPED:
            for i, d in enumerate(dets):
                warp(image, d["points"]).save(os.path.join(OUTPUT_DIR, f"{name}_{d['label']}_{i}.jpg"))


if __name__ == "__main__":
    main()