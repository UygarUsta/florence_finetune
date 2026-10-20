"""
Canonical quad yardımcıları — train ve inference ortak kullanır.

Fikir:
  * Etiketlemedeki nokta sırası "canonical"dır: 1. nokta = belgenin kendi
    çerçevesindeki başlangıç köşesi (düzleştirme için gerekli sıra).
  * Hedef dizide koordinatlar GEOMETRİK sırayla yazılır: görüntüde sol-üst
    yönüne (-135°) en yakın köşeden başlayıp etiketleme yönünde devam eder.
  * Arkasına tek bir yön token'ı eklenir: rot{k}, k ∈ {0,1,2,3}.
    geo = roll(canonical, -k)   <=>   canonical = roll(geo, k)

Hedef format (nesneler boşlukla ayrılır):
    ruhsat<loc_..>x8rot2 kimlik<loc_..>x8rot0

"rot0..rot3" düz metindir, normal BPE token'larına bölünür; vocab değişmez,
dolayısıyla adaptör diğer LoRA'larla aynı base model içinde yaşayabilir.
"""
import math
import re

import numpy as np
from PIL import Image

ROT_PREFIX = "rot"
NUM_BINS = 1000  # Florence-2: <loc_0> .. <loc_999>


# ----------------------------------------------------------------------------
# Geometri
# ----------------------------------------------------------------------------
def signed_area(points):
    """Shoelace. Görüntü koordinatında (y aşağı) görsel saat yönü => pozitif."""
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def winding(points, min_area=1.0):
    """+1: saat yönü, -1: saat yönünün tersi, 0: dejenere (alan ~0)."""
    a = signed_area(points)
    if abs(a) < min_area:
        return 0
    return 1 if a > 0 else -1


def geometric_start_index(points):
    """Merkeze göre açısı sol-üst yönüne (-135°) en yakın köşenin indeksi."""
    pts = np.asarray(points, dtype=np.float64)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    diff = np.angle(np.exp(1j * (ang + 3 * np.pi / 4)))  # [-pi, pi] aralığına sar
    return int(np.argmin(np.abs(diff)))


def canonical_to_geometric(points):
    """canonical sıra -> (geometrik sıra, k)"""
    pts = np.asarray(points, dtype=np.float64)
    k = geometric_start_index(pts)
    return np.roll(pts, -k, axis=0), k


def geometric_to_canonical(geo_points, k):
    """(geometrik sıra, k) -> canonical sıra"""
    return np.roll(np.asarray(geo_points, dtype=np.float64), k, axis=0)


# ----------------------------------------------------------------------------
# Kuantizasyon (Florence-2 ile aynı: floor binning, bin merkezine dequantize)
# ----------------------------------------------------------------------------
def quantize(v, size):
    return int(min(max(v / size, 0.0), 1.0 - 1e-9) * NUM_BINS)  # 0..999


def dequantize(q, size):
    return (q + 0.5) / NUM_BINS * size


# ----------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------
def rotate_sample(image, shapes, angle_deg, fill=(0, 0, 0)):
    """
    Görüntüyü angle_deg kadar (görsel olarak saat yönünün TERSİNE, PIL konvansiyonu)
    expand=True ile döndürür, noktaları aynı dönüşümle taşır. Nokta SIRASI korunur
    (canonical bilgi bozulmaz); k değeri hedef üretilirken yeniden hesaplanır.
    Dönüş uygun (proper) bir dönüşüm olduğundan sarım yönü de değişmez.
    """
    if angle_deg % 360 == 0:
        return image, shapes
    W, H = image.size
    rotated = image.rotate(angle_deg, resample=Image.BICUBIC, expand=True, fillcolor=fill)
    W2, H2 = rotated.size
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    cx, cy, cx2, cy2 = W / 2, H / 2, W2 / 2, H2 / 2

    new_shapes = []
    for sh in shapes:
        pts = []
        for x, y in sh["points"]:
            dx, dy = x - cx, y - cy
            pts.append([cx2 + dx * c + dy * s, cy2 - dx * s + dy * c])
        new_shapes.append({**sh, "points": pts})
    return rotated, new_shapes


# ----------------------------------------------------------------------------
# Hedef dizi üretimi / ayrıştırma
# ----------------------------------------------------------------------------
def shapes_to_target_str(shapes, width, height):
    parts = []
    for shape in shapes:
        geo, k = canonical_to_geometric(shape["points"])
        locs = "".join(
            f"<loc_{quantize(x, width)}><loc_{quantize(y, height)}>" for x, y in geo
        )
        parts.append(f"{shape['label']}{locs}{ROT_PREFIX}{k}")
    return " ".join(parts)


_PATTERN = re.compile(
    r"([^\s<>]+?)((?:<loc_\d+>){8})(?:\s*" + ROT_PREFIX + r"([0-3]))?"
)


def parse_canonical_predictions(text_output, img_width, img_height):
    """
    Dönüş: her nesne için
      label, points (canonical sıra, piksel), geo_points (geometrik sıra), k,
      rot_missing (model yön token'ını üretmediyse True; k=0 varsayılır)
    """
    results = []
    for label, coords_str, k_str in _PATTERN.findall(text_output):
        q = [int(v) for v in re.findall(r"<loc_(\d+)>", coords_str)]
        geo = [
            [dequantize(q[i], img_width), dequantize(q[i + 1], img_height)]
            for i in range(0, 8, 2)
        ]
        k = int(k_str) if k_str else 0
        canon = geometric_to_canonical(geo, k)
        results.append({
            "label": label,
            "points": [[round(float(x), 2), round(float(y), 2)] for x, y in canon],
            "geo_points": [[round(x, 2), round(y, 2)] for x, y in geo],
            "k": k,
            "rot_missing": not k_str,
        })
    return results
