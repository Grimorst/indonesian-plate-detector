import os
import sys
import time
import base64
import json
import re
import math
from io import BytesIO
from datetime import datetime

# Load PyTorch DLLs explicitly on Windows to prevent DLL load errors
torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

# Disable mkldnn for PaddleOCR on CPU to avoid crashes
os.environ["FLAGS_use_mkldnn"] = "0"

# Prevent Flask from exiting on Windows when running in background without stdin
try:
    if sys.stdin is None or sys.stdin.closed:
        sys.stdin = open(os.devnull, "r")
except Exception:
    pass

from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION & INITIALIZATION
# ==============================================================================
app = Flask(__name__, static_folder="static", template_folder="templates")

MODEL_PATH = "models/Grimorsbest.pt"
SAMPLE_IMAGE = "images/mobil.jpg"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("  VisionGuard ALPR - Starting Server (High-Accuracy Pipeline)")
print("=" * 60)

# 1. Load YOLO Model (models/best.pt)
print(f"[*] Loading YOLO model from: {MODEL_PATH}")
try:
    yolo_model = YOLO(MODEL_PATH)
    print(f"[+] YOLO model '{MODEL_PATH}' successfully loaded!")
except Exception as e:
    print(f"[!] Error loading YOLO model: {e}")
    yolo_model = None

# 2. Load PaddleOCR (Tuned for Indonesian Plate ALPR)
ocr_engine = None
try:
    print("[*] Initializing PaddleOCR (Tuned for ALPR)...")
    from paddleocr import PaddleOCR
    ocr_engine = PaddleOCR(
        lang="en",
        use_angle_cls=True,
        det_db_thresh=0.01,
        det_db_box_thresh=0.01,
        det_db_unclip_ratio=2.5,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    print("[+] PaddleOCR successfully initialized!")
except Exception as e:
    print(f"[!] Warning: PaddleOCR could not be initialized: {e}")

# Detection history log in memory
detection_history = []

# ==============================================================================
# KODE WILAYAH & SINTAKS PLAT INDONESIA
# ==============================================================================
INDONESIAN_PREFIXES = {
    # Sumatra
    "BL", "BB", "BK", "BA", "BM", "BP", "BG", "BN", "BE", "BD",
    # DKI Jakarta, Banten, Jawa Barat
    "A", "B", "D", "E", "F", "T", "Z",
    # Jawa Tengah & DIY
    "G", "H", "K", "R", "AA", "AD", "AB",
    # Jawa Timur
    "L", "M", "N", "P", "S", "W", "AE", "AG",
    # Bali & Nusa Tenggara
    "DK", "DR", "EA", "DH", "EB", "ED",
    # Kalimantan
    "KB", "DA", "KH", "KT", "KU",
    # Sulawesi
    "DB", "DL", "DM", "DN", "DT", "DD", "DC", "DP",
    # Maluku & Papua
    "DE", "DG", "PA", "PB"
}

PREFIX_DIGIT_TO_LETTER = {
    "0": "D", "1": "I", "2": "Z", "3": "E", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "P"
}

SUFFIX_DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "P"
}

LETTER_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1",
    "Z": "2", "E": "3", "A": "4", "S": "5", "G": "6",
    "T": "7", "B": "8", "P": "9"
}

DISCARD_WORDS = {
    "TOYOTA", "PLAZATOYOTA", "PLAZA", "HONDA", "SUZUKI", "DAIHATSU",
    "MITSUBISHI", "NISSAN", "HYUNDAI", "ISUZU", "HINO", "TERDETEKSI",
    "PLAT", "DETECTED", "AUTO2000", "CINTAMOBIL", "COM", "WWW",
    "INDONESIA", "POLRI", "SAMSAT"
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def clean_token(text):
    """Membersihkan teks hanya menyisakan alfanumerik kapital."""
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()

def parse_indonesian_plate(raw_text):
    """
    Format dan koreksi sintaks plat nomor Indonesia:
    [Kode Wilayah: 1-2 Huruf] [Nomor: 1-4 Angka] [Seri: 1-3 Huruf]
    """
    cleaned = clean_token(raw_text)
    if not cleaned:
        return "", 0.0, None

    for bad in DISCARD_WORDS:
        cleaned = cleaned.replace(bad, "")
    cleaned = cleaned.strip()
    if not cleaned:
        return "", 0.0, None

    # Ekstraksi masa berlaku jika menempel di ujung (contoh: 0927 -> 09.27)
    period_extracted = None
    date_match = re.search(r"(0[1-9]|1[0-2])([2-3]\d)$", cleaned)
    if date_match and len(cleaned) >= 7:
        period_extracted = f"{date_match.group(1)}.{date_match.group(2)}"
        cleaned = cleaned[:date_match.start()]

    chars = list(cleaned)
    n = len(chars)
    if n < 2:
        return cleaned, 5.0, period_extracted

    # Deteksi panjang prefix kode wilayah
    p_len = 0
    if chars[0].isalpha():
        if n >= 2 and chars[1].isalpha():
            p2 = chars[0] + chars[1]
            if p2 in INDONESIAN_PREFIXES:
                p_len = 2
            elif chars[0] in INDONESIAN_PREFIXES and (n > 2 and chars[2].isdigit()):
                p_len = 1
            elif n > 2 and not chars[2].isalpha():
                p_len = 2
            else:
                p_len = 1
        else:
            p_len = 1
    else:
        c0 = PREFIX_DIGIT_TO_LETTER.get(chars[0], "")
        c1 = PREFIX_DIGIT_TO_LETTER.get(chars[1], "") if n > 1 else ""
        if n > 2 and (c0 + c1) in INDONESIAN_PREFIXES and chars[2].isdigit():
            p_len = 2
        else:
            p_len = 0

    if p_len > 0:
        prefix = "".join([PREFIX_DIGIT_TO_LETTER.get(c, c) for c in chars[:p_len]])
        remainder = chars[p_len:]
    else:
        prefix = ""
        remainder = chars

    num_chars = []
    suffix_chars = []
    in_suffix = False

    for i, c in enumerate(remainder):
        if not in_suffix:
            if len(num_chars) >= 1 and (c.isalpha() and c not in ["O", "I", "Z", "S", "B"] or len(num_chars) >= 4):
                in_suffix = True
                suffix_chars.append(SUFFIX_DIGIT_TO_LETTER.get(c, c))
            elif len(num_chars) >= 3 and c.isalpha():
                in_suffix = True
                suffix_chars.append(SUFFIX_DIGIT_TO_LETTER.get(c, c))
            else:
                num_chars.append(LETTER_TO_DIGIT.get(c, c))
        else:
            suffix_chars.append(SUFFIX_DIGIT_TO_LETTER.get(c, c))

    num_str = "".join([c for c in num_chars if c.isdigit()])
    suffix_str = "".join([c for c in suffix_chars if c.isalpha()])

    if not num_str and remainder:
        all_digits = [LETTER_TO_DIGIT.get(c, c) for c in remainder if LETTER_TO_DIGIT.get(c, c).isdigit()]
        num_str = "".join(all_digits[:4])
        suffix_str = "".join([c for c in remainder[len(num_str):] if c.isalpha()])

    parts = []
    if prefix:
        parts.append(prefix)
    if num_str:
        parts.append(num_str)
    if suffix_str:
        parts.append(suffix_str)

    plate_str = " ".join(parts)

    quality = 0.0
    if prefix in INDONESIAN_PREFIXES:
        quality += 40.0
    elif prefix.isalpha():
        quality += 20.0

    if 1 <= len(num_str) <= 4:
        quality += 35.0
        if 3 <= len(num_str) <= 4:
            quality += 10.0

    if 1 <= len(suffix_str) <= 3:
        quality += 25.0
    elif not suffix_str and len(num_str) >= 1:
        quality += 5.0

    if prefix and num_str and suffix_str:
        quality += 20.0

    total_alphanumeric = len(prefix) + len(num_str) + len(suffix_str)
    if total_alphanumeric >= 6:
        quality += 15.0
    elif total_alphanumeric <= 3:
        quality -= 30.0

    return plate_str, quality, period_extracted


# ==============================================================================
# MODULAR PREPROCESSING PIPELINE
# ==============================================================================

# --- Scale helper ---
def _scale_crop(crop_bgr, target_h=90):
    """Resize crop so height = target_h, keep aspect ratio. Scale factor clamped 1.2–5.0."""
    h = max(crop_bgr.shape[0], 1)
    scale = max(1.2, min(5.0, target_h / h))
    return cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def deskew_plate(gray_img):
    """
    Koreksi kemiringan plat menggunakan analisis sudut garis Hough.
    Mengembalikan gambar yang sudah diluruskan.
    """
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=max(20, gray_img.shape[1] // 6),
                            maxLineGap=10)
    if lines is None or len(lines) == 0:
        return gray_img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > 5:
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) < 25:
                angles.append(angle)

    if not angles:
        return gray_img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray_img

    h, w = gray_img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray_img, rotation_matrix, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def adaptive_morphology_clean(gray_img):
    """
    Membersihkan noise dan memperjelas karakter menggunakan
    morphological operations adaptif.
    """
    h, w = gray_img.shape[:2]

    # Gaussian blur ringan untuk mengurangi noise
    denoised = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Adaptive threshold untuk menangani pencahayaan tidak merata
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )

    # Morphological close untuk menyambung karakter yang terputus
    kernel_size = max(1, min(h // 25, 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

    return cleaned


# ==============================================================================
# INDIVIDUAL MODULAR PREPROCESSING FUNCTIONS
# Each returns a BGR image ready for OCR, plus a display-friendly label.
# ==============================================================================

def preprocess_native(crop_bgr):
    """Original crop tanpa perubahan."""
    return crop_bgr


def preprocess_grayscale(crop_bgr, target_h=90):
    """Step 1: Resize → Grayscale → Gaussian Denoising (ringan)."""
    resized = _scale_crop(crop_bgr, target_h)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def preprocess_clahe(crop_bgr, target_h=90):
    """Step 2: Resize → Grayscale → CLAHE contrast enhancement."""
    resized = _scale_crop(crop_bgr, target_h)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def preprocess_otsu(crop_bgr, target_h=90):
    """Step 3: Resize → Grayscale → Otsu global thresholding."""
    resized = _scale_crop(crop_bgr, target_h)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def preprocess_adaptive(crop_bgr, target_h=90):
    """Step 4: Resize → Grayscale → Adaptive Gaussian Threshold."""
    resized = _scale_crop(crop_bgr, target_h)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )
    return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)


def preprocess_bilateral_clahe(crop_bgr, target_h=90):
    """Step 5: Resize → Grayscale → Bilateral Filter (edge-preserving) → CLAHE."""
    resized = _scale_crop(crop_bgr, target_h)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def preprocess_sharp(crop_bgr, target_h=90):
    """Step 6: Resize → Grayscale → Unsharp Masking (high-contrast sharpening)."""
    resized = _scale_crop(crop_bgr, target_h)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.5)
    unsharp = cv2.addWeighted(gray, 1.8, gaussian, -0.8, 0)
    return cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)


def preprocess_deskew_clahe(crop_bgr, target_h=90):
    """Step 7: Resize → Grayscale → Deskew → CLAHE."""
    resized = _scale_crop(crop_bgr, target_h)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    deskewed = deskew_plate(gray)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(deskewed)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def preprocess_inverted(crop_bgr, target_h=90):
    """Step 8: Resize → Grayscale → Invert → CLAHE (untuk plat gelap)."""
    resized = _scale_crop(crop_bgr, target_h)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def preprocess_morph(crop_bgr, target_h=90):
    """Step 9: Resize → Grayscale → Morphological cleaning (close gaps)."""
    resized = _scale_crop(crop_bgr, target_h)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    cleaned = adaptive_morphology_clean(gray)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


# Registry of all named preprocessing functions (for modular comparison)
PREPROCESS_REGISTRY = [
    ("native",         preprocess_native),
    ("grayscale",      preprocess_grayscale),
    ("clahe",          preprocess_clahe),
    ("otsu",           preprocess_otsu),
    ("adaptive",       preprocess_adaptive),
    ("bilateral_clahe",preprocess_bilateral_clahe),
    ("sharp",          preprocess_sharp),
    ("deskew_clahe",   preprocess_deskew_clahe),
    ("inverted",       preprocess_inverted),
    ("morph",          preprocess_morph),
]


def enhance_plate_variants(crop_bgr):
    """Compatibility wrapper: returns list of (name, bgr_img) for all variants."""
    results = []
    for name, fn in PREPROCESS_REGISTRY:
        try:
            img = fn(crop_bgr)
            results.append((name, img))
        except Exception:
            pass
    # Extra: multi-scale native passes for OCR
    h = max(crop_bgr.shape[0], 1)
    for target_h in [70, 120, 160]:
        scale = max(1.2, min(5.0, target_h / h))
        resized = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        results.append((f"scale_{target_h}", resized))
    return results


def mat_to_base64(img_bgr, quality=85):
    """Encode OpenCV BGR image to base64 JPEG data URI."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode('.jpg', img_bgr, encode_params)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

def base64_to_mat(b64_str):
    """Decode base64 data URI to OpenCV BGR image."""
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]
    img_data = base64.b64decode(b64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def box_iou(box_a, box_b):
    """Calculate IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0

def collect_model_boxes(image, offset_x=0, offset_y=0, imgsz=1280, conf=0.15):
    """Run YOLO and return boxes in original-image coordinates."""
    predictions = yolo_model.predict(
        source=image,
        conf=conf,
        imgsz=imgsz,
        augment=True,
        verbose=False
    )
    collected = []
    for box in predictions[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        collected.append({
            "box": [int(x1 + offset_x), int(y1 + offset_y), int(x2 + offset_x), int(y2 + offset_y)],
            "confidence": float(box.conf[0])
        })
    return collected

def suppress_duplicate_boxes(boxes, iou_threshold=0.35):
    """Keep highest-confidence boxes and remove overlapping tile duplicates."""
    selected = []
    for candidate in sorted(boxes, key=lambda item: item["confidence"], reverse=True):
        candidate_box = candidate["box"]
        if not any(box_iou(candidate_box, item["box"]) >= iou_threshold for item in selected):
            selected.append(candidate)
    return selected

def merge_plate_boxes(boxes):
    """Merge split model boxes that sit on the same physical license plate."""
    merged = []
    for item in sorted(boxes, key=lambda value: value["confidence"], reverse=True):
        x1, y1, x2, y2 = item["box"]
        item_w = max(1, x2 - x1)
        item_h = max(1, y2 - y1)
        did_merge = False

        for index, current in enumerate(merged):
            cx1, cy1, cx2, cy2 = current["box"]
            current_w = max(1, cx2 - cx1)
            current_h = max(1, cy2 - cy1)
            vertical_overlap = min(y2, cy2) - max(y1, cy1)
            horizontal_gap = max(cx1 - x2, x1 - cx2, 0)
            center_y_delta = abs(((y1 + y2) / 2) - ((cy1 + cy2) / 2))
            height_ratio = min(item_h, current_h) / max(item_h, current_h)

            same_plate_band = (
                vertical_overlap >= 0.38 * min(item_h, current_h) or
                center_y_delta <= 0.55 * max(item_h, current_h)
            )
            close_enough = horizontal_gap <= 1.35 * max(item_h, current_h)
            contained = (
                (x1 >= cx1 and x2 <= cx2) or
                (cx1 >= x1 and cx2 <= x2)
            )

            if (
                box_iou([x1, y1, x2, y2], [cx1, cy1, cx2, cy2]) >= 0.10 or
                (same_plate_band and height_ratio >= 0.30 and (close_enough or contained))
            ):
                merged[index] = {
                    "box": [min(x1, cx1), min(y1, cy1), max(x2, cx2), max(y2, cy2)],
                    "confidence": max(item["confidence"], current["confidence"])
                }
                did_merge = True
                break

        if not did_merge:
            merged.append(item)

    return merged

def is_plausible_plate_box(box, image_width, image_height):
    """Filter obvious non-plate rectangles before OCR."""
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    ratio = width / height
    area_ratio = (width * height) / max(1, image_width * image_height)
    return 1.20 <= ratio <= 9.5 and 0.0004 <= area_ratio <= 0.80

def refine_plate_box(image_bgr, box):
    """Refine YOLO box using strong horizontal plate-like contours nearby."""
    x1, y1, x2, y2 = box
    height, width = image_bgr.shape[:2]
    box_width = x2 - x1
    box_height = y2 - y1
    search_x1 = max(0, x1 - int(box_width * 0.35))
    search_y1 = max(0, y1 - int(box_height * 0.25))
    search_x2 = min(width, x2 + int(box_width * 0.35))
    search_y2 = min(height, y2 + int(box_height * 0.25))
    roi = image_bgr[search_y1:search_y2, search_x1:search_x2]
    if roi.size == 0:
        return box

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if ch < 4 or cw < 24:
            continue
        ratio = cw / ch
        area = cw * ch
        if ratio < 2.0 or ratio > 6.5 or area < box_width * box_height * 0.18:
            continue
        candidate = [search_x1 + cx, search_y1 + cy, search_x1 + cx + cw, search_y1 + cy + ch]
        overlap = box_iou(candidate, box)
        score = overlap + min(area / max(box_width * box_height, 1), 2.0) * 0.12
        if overlap > 0.15 and score > best_score:
            best = candidate
            best_score = score
    return best or box


# ==============================================================================
# VOTING-BASED OCR WITH CONSENSUS
# ==============================================================================

def normalize_plate_for_voting(plate_text):
    """Normalisasi teks plat untuk perbandingan voting."""
    return plate_text.replace(" ", "").upper()


def vote_best_plate(candidates):
    """
    Voting-based consensus: pilih kandidat yang paling sering muncul
    dengan skor kualitas tertinggi.
    Strategi:
    1. Kelompokkan kandidat berdasarkan teks yang sama (setelah normalisasi)
    2. Hitung voting score = jumlah kemunculan * avg quality score
    3. Pilih kandidat dengan voting score tertinggi
    """
    if not candidates:
        return None

    # Kelompokkan berdasarkan teks normalisasi
    groups = {}
    for c in candidates:
        key = normalize_plate_for_voting(c["text"])
        if key not in groups:
            groups[key] = []
        groups[key].append(c)

    best_group_key = None
    best_voting_score = -1.0

    for key, group in groups.items():
        count = len(group)
        avg_score = sum(c["score"] for c in group) / count
        max_conf = max(c["conf"] for c in group)

        # Voting score: Quality is much more important than raw frequency
        voting_score = (count * 2.0) + avg_score + (max_conf * 25.0)

        if voting_score > best_voting_score:
            best_voting_score = voting_score
            best_group_key = key

    if best_group_key is None:
        return None

    # Dari grup terpilih, ambil yang punya skor individual tertinggi
    best_group = groups[best_group_key]
    best_group.sort(key=lambda c: c["score"], reverse=True)
    return best_group[0]


def _run_ocr_on_image(img_bgr):
    """Run PaddleOCR on a single image, return list of (raw_text, conf, box) tuples."""
    try:
        res = ocr_engine.ocr(img_bgr, cls=False)
    except Exception:
        return []
    if not res or not res[0]:
        return []
    out = []
    for line in res[0]:
        raw = line[1][0].strip()
        conf = float(line[1][1])
        box = line[0]
        out.append((raw, conf, box))
    return out


def run_ocr_on_crop(plate_bgr, return_debug=False):
    """
    Run high-accuracy multi-pass OCR on cropped plate image.
    If return_debug=True, returns extended dict with per-variant debug info.
    Otherwise returns (plate_text, ocr_conf, period) for backward compatibility.
    """
    _EMPTY = ("TERDETEKSI", 0.75, "08.28") if not return_debug else {
        "plate_text": "TERDETEKSI", "ocr_confidence": 0.75, "plate_period": "08.28",
        "per_variant": [], "all_candidates": []
    }
    if ocr_engine is None or plate_bgr is None or plate_bgr.size == 0:
        return _EMPTY

    h, w = plate_bgr.shape[:2]
    if h == 0 or w == 0:
        return _EMPTY

    candidates = []
    extracted_periods = []
    debug_variants = []   # list of {name, image_b64, raw_lines, plate_text, conf}

    # Padding variants to test
    pads = [
        (0, 0),
        (max(2, int(w * 0.08)), max(2, int(h * 0.06))),
        (max(4, int(w * 0.15)), max(4, int(h * 0.12))),
    ]
    crops_to_test = [plate_bgr]
    for px, py in pads[1:]:
        padded = cv2.copyMakeBorder(plate_bgr, py, py, px, px, cv2.BORDER_REPLICATE)
        crops_to_test.append(padded)

    # ── Run each named preprocessing variant ──────────────────────────────────
    for pad_i, c_img in enumerate(crops_to_test):
        for var_name, preprocess_fn in PREPROCESS_REGISTRY:
            try:
                var_img = preprocess_fn(c_img)
            except Exception:
                continue

            vh, vw = var_img.shape[:2]
            ocr_lines = _run_ocr_on_image(var_img)

            # Collect raw text for debug
            raw_lines_debug = [{"text": t, "conf": round(c, 4)} for t, c, _ in ocr_lines]

            frags = []
            for (raw_text_orig, conf, box) in ocr_lines:
                raw_text = clean_token(raw_text_orig)
                if not raw_text or any(bad in raw_text for bad in DISCARD_WORDS):
                    continue
                bx1 = min(p[0] for p in box)
                bx2 = max(p[0] for p in box)
                by1 = min(p[1] for p in box)
                by2 = max(p[1] for p in box)
                bcy = (by1 + by2) / 2.0
                bh_box = max(1.0, by2 - by1)

                if re.match(r"^(0[1-9]|1[0-2])[-.]?([2-3]\d)$", raw_text) and bcy > vh * 0.35:
                    clean_d = raw_text.replace("-", ".").replace(" ", "")
                    if len(clean_d) == 4:
                        clean_d = f"{clean_d[:2]}.{clean_d[2:]}"
                    extracted_periods.append((clean_d, conf))
                    continue

                frags.append({"text": raw_text, "conf": conf, "x1": bx1, "x2": bx2, "cy": bcy, "h": bh_box})

            # Per-fragment candidates
            var_best_text = ""
            var_best_conf = 0.0
            for f in frags:
                plate_str, q_score, p_date = parse_indonesian_plate(f["text"])
                if p_date:
                    extracted_periods.append((p_date, f["conf"]))
                if plate_str:
                    total_score = q_score + (f["conf"] * 30.0)
                    candidates.append({"text": plate_str, "conf": f["conf"], "score": total_score,
                                       "variant": var_name})
                    if total_score > var_best_conf * 30.0 + q_score:
                        var_best_text = plate_str
                        var_best_conf = f["conf"]

            # Horizontal band grouping
            if len(frags) > 1:
                frags_sorted = sorted(frags, key=lambda item: item["x1"])
                bands = []
                for f in frags_sorted:
                    placed = False
                    for band in bands:
                        avg_cy = sum(b["cy"] for b in band) / len(band)
                        avg_h = sum(b["h"] for b in band) / len(band)
                        if abs(f["cy"] - avg_cy) < max(avg_h * 1.5, 30.0):
                            band.append(f)
                            placed = True
                            break
                    if not placed:
                        bands.append([f])
                for band in bands:
                    band_sorted = sorted(band, key=lambda item: item["x1"])
                    joined_text = "".join(b["text"] for b in band_sorted)
                    avg_conf = sum(b["conf"] for b in band_sorted) / len(band_sorted)
                    plate_str, q_score, p_date = parse_indonesian_plate(joined_text)
                    if p_date:
                        extracted_periods.append((p_date, avg_conf))
                    if plate_str:
                        total_score = q_score + (avg_conf * 30.0) + 15.0
                        candidates.append({"text": plate_str, "conf": avg_conf, "score": total_score,
                                           "variant": var_name})
                        if total_score > var_best_conf * 30:
                            var_best_text = plate_str
                            var_best_conf = avg_conf

            # Only record debug for pad=0 (first crop) to keep payload manageable
            if return_debug and pad_i == 0:
                display_text = var_best_text
                if not display_text and raw_lines_debug:
                    display_text = " ".join(r["text"] for r in raw_lines_debug)
                display_conf = var_best_conf if var_best_text else (raw_lines_debug[0]["conf"] if raw_lines_debug else 0.0)
                debug_variants.append({
                    "name": var_name,
                    "image_b64": mat_to_base64(var_img, quality=70),
                    "raw_lines": raw_lines_debug,
                    "plate_text": display_text,
                    "conf": round(display_conf, 4)
                })

    # ── Voting consensus ───────────────────────────────────────────────────────
    if candidates:
        best = vote_best_plate(candidates)
        if best:
            best_text = best["text"]
            best_conf = best["conf"]
        else:
            candidates.sort(key=lambda item: item["score"], reverse=True)
            best_text = candidates[0]["text"]
            best_conf = candidates[0]["conf"]
    else:
        best_text = "TERDETEKSI"
        best_conf = 0.75

    period = "08.28"
    if extracted_periods:
        extracted_periods.sort(key=lambda item: item[1], reverse=True)
        period = extracted_periods[0][0]

    if return_debug:
        # Build unique candidate list for debugging (deduplicated, sorted by score)
        seen = set()
        unique_candidates = []
        for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
            key = c["text"]
            if key not in seen:
                seen.add(key)
                unique_candidates.append({"text": c["text"], "conf": round(c["conf"], 4),
                                          "score": round(c["score"], 1), "variant": c.get("variant", "")})
        return {
            "plate_text": best_text,
            "ocr_confidence": round(best_conf, 4),
            "plate_period": period,
            "per_variant": debug_variants,
            "all_candidates": unique_candidates[:20]   # limit to 20 for payload size
        }

    return best_text, round(best_conf, 4), period


def process_detection(image_bgr, conf_thresh=0.25, return_debug=True):
    """
    Run YOLO detection and PaddleOCR recognition on image.
    Returns structured results, debug variants, and annotated image.
    """
    start_time = time.time()
    
    if yolo_model is None:
        raise RuntimeError("Model YOLO best.pt belum siap atau gagal dimuat.")
    
    h_img, w_img = image_bgr.shape[:2]
    
    # Multi-resolution inference untuk menangkap plat di berbagai ukuran
    all_boxes = []

    # Pass 1: High-res full frame
    all_boxes.extend(collect_model_boxes(
        image_bgr,
        imgsz=1280,
        conf=max(0.12, conf_thresh - 0.05)
    ))

    # Pass 2: Medium-res (lebih sensitif untuk plat dekat)
    all_boxes.extend(collect_model_boxes(
        image_bgr,
        imgsz=960,
        conf=max(0.15, conf_thresh)
    ))

    boxes = merge_plate_boxes(suppress_duplicate_boxes(all_boxes, iou_threshold=0.30))

    image_area = h_img * w_img
    has_small_detection = any(
        ((box["box"][2] - box["box"][0]) *
         (box["box"][3] - box["box"][1])) / image_area < 0.012
        for box in boxes
    ) if image_area else False

    # Use overlapping tiles when no box or only tiny boxes appear.
    if not boxes or has_small_detection:
        tile_width = max(640, int(w_img * 0.62))
        tile_height = max(640, int(h_img * 0.62))
        step_x = max(1, int(tile_width * 0.78))
        step_y = max(1, int(tile_height * 0.78))
        tiled_boxes = []
        for y in range(0, h_img, step_y):
            for x in range(0, w_img, step_x):
                x_end = min(w_img, x + tile_width)
                y_end = min(h_img, y + tile_height)
                tile = image_bgr[y:y_end, x:x_end]
                if tile.size == 0:
                    continue
                tiled_boxes.extend(collect_model_boxes(
                    tile, offset_x=x, offset_y=y, imgsz=1280, conf=0.10
                ))
                if x_end == w_img and y_end == h_img:
                    break
            if y_end == h_img:
                break
        boxes = merge_plate_boxes(suppress_duplicate_boxes(boxes + tiled_boxes, iou_threshold=0.25))

    boxes = merge_plate_boxes(suppress_duplicate_boxes(boxes, iou_threshold=0.25))

    detections = []
    annotated_bgr = image_bgr.copy()

    # Merge full-frame and enhanced predictions, removing duplicate boxes.
    selected_boxes = []
    for i, prediction in enumerate(boxes):
        conf = prediction["confidence"]
        x1, y1, x2, y2 = prediction["box"]
        
        # Clamp coordinates within frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue

        # License plates are wide rectangles; widen narrow model boxes
        box_width = x2 - x1
        box_height = y2 - y1
        target_width = max(box_width, int(box_height * 3.2))
        center_x = (x1 + x2) // 2
        x1 = max(0, center_x - target_width // 2)
        x2 = min(w_img, center_x + target_width // 2)
        if x2 - x1 < target_width:
            if x1 == 0:
                x2 = min(w_img, target_width)
            elif x2 == w_img:
                x1 = max(0, w_img - target_width)

        current_box = refine_plate_box(image_bgr, [x1, y1, x2, y2])
        x1, y1, x2, y2 = current_box
        if not is_plausible_plate_box(current_box, w_img, h_img):
            continue
        if any(box_iou(current_box, previous) >= 0.45 for previous in selected_boxes):
            continue
        selected_boxes.append(current_box)
            
        # Add padding so OCR receives the complete plate border and characters.
        box_width = x2 - x1
        box_height = y2 - y1
        target_width = max(box_width, int(box_height * 3.5))
        extra_width = max(0, target_width - box_width)
        pad_x = max(10, int(extra_width / 2) + int(box_width * 0.08))
        pad_y = max(4, int((y2 - y1) * 0.08))
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w_img, x2 + pad_x)
        crop_y2 = min(h_img, y2 + pad_y)
        plate_crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_b64 = mat_to_base64(plate_crop, quality=90)
        
        # Run OCR
        if return_debug:
            ocr_res = run_ocr_on_crop(plate_crop, return_debug=True)
            plate_text = ocr_res["plate_text"]
            ocr_conf = ocr_res["ocr_confidence"]
            period = ocr_res["plate_period"]
            preprocessed_variants = ocr_res.get("per_variant", [])
            raw_candidates = ocr_res.get("all_candidates", [])
        else:
            plate_text, ocr_conf, period = run_ocr_on_crop(plate_crop, return_debug=False)
            preprocessed_variants = []
            raw_candidates = []
        
        # Draw high-tech HUD bounding box on annotated image
        box_w = x2 - x1
        box_h = y2 - y1
        corner_len = max(8, min(24, box_w // 4))
        
        # Main Cyan Box
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (255, 240, 0), 2)  # BGR cyan
        
        # Neon Emerald Corners
        c_color = (157, 255, 0)  # BGR emerald
        cv2.line(annotated_bgr, (x1, y1), (x1 + corner_len, y1), c_color, 4)
        cv2.line(annotated_bgr, (x1, y1), (x1, y1 + corner_len), c_color, 4)
        
        cv2.line(annotated_bgr, (x2, y1), (x2 - corner_len, y1), c_color, 4)
        cv2.line(annotated_bgr, (x2, y1), (x2, y1 + corner_len), c_color, 4)
        
        cv2.line(annotated_bgr, (x1, y2), (x1 + corner_len, y2), c_color, 4)
        cv2.line(annotated_bgr, (x1, y2), (x1, y2 - corner_len), c_color, 4)
        
        cv2.line(annotated_bgr, (x2, y2), (x2 - corner_len, y2), c_color, 4)
        cv2.line(annotated_bgr, (x2, y2), (x2, y2 - corner_len), c_color, 4)
        
        # Label Badge Background
        label = f"{plate_text} ({int(conf * 100)}%)"
        (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        lbl_y = max(lbl_h + 10, y1 - 8)
        cv2.rectangle(
            annotated_bgr,
            (x1, lbl_y - lbl_h - 6),
            (x1 + lbl_w + 14, lbl_y + 4),
            (10, 15, 25),
            -1
        )
        cv2.rectangle(
            annotated_bgr,
            (x1, lbl_y - lbl_h - 6),
            (x1 + lbl_w + 14, lbl_y + 4),
            c_color,
            1
        )
        cv2.putText(
            annotated_bgr,
            label,
            (x1 + 7, lbl_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            c_color,
            2,
            cv2.LINE_AA
        )
        
        det_item = {
            "id": i + 1,
            "box": [x1, y1, x2, y2],
            "confidence": round(conf, 4),
            "plate_text": plate_text,
            "plate_period": period,
            "ocr_confidence": round(ocr_conf, 4),
            "crop_image": crop_b64,
        }
        if return_debug:
            det_item["preprocessed_variants"] = preprocessed_variants
            det_item["raw_ocr_candidates"] = raw_candidates
        detections.append(det_item)
    
    elapsed_ms = (time.time() - start_time) * 1000.0
    annotated_b64 = mat_to_base64(annotated_bgr, quality=80)
    
    return {
        "success": True,
        "total_plates": len(detections),
        "detections": detections,
        "annotated_image": annotated_b64,
        "inference_time_ms": round(elapsed_ms, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ==============================================================================
# FLASK ROUTES
# ==============================================================================
@app.route("/")
def index():
    """Render main VisionGuard ALPR dashboard."""
    return render_template("index.html")

@app.route("/api/detect/image", methods=["POST"])
def detect_image():
    """Detect license plates from uploaded photo."""
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "Tidak ada file gambar yang diunggah"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Nama file kosong"}), 400
        
        in_memory_file = file.read()
        np_arr = np.frombuffer(in_memory_file, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({"success": False, "error": "Format gambar tidak valid atau korup"}), 400
        
        result = process_detection(img_bgr)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/detect/frame", methods=["POST"])
def detect_frame():
    """Fast detection for real-time CCTV / Webcam frames."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "frame" not in data:
            return jsonify({"success": False, "error": "Data frame base64 tidak ditemukan"}), 400
        
        img_bgr = base64_to_mat(data["frame"])
        if img_bgr is None:
            return jsonify({"success": False, "error": "Gagal membaca frame"}), 400
        
        result = process_detection(img_bgr, conf_thresh=0.25, return_debug=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/sample", methods=["GET"])
def sample_detection():
    """Run detection on local sample image (images/mobil.jpg)."""
    try:
        if not os.path.exists(SAMPLE_IMAGE):
            return jsonify({"success": False, "error": f"File {SAMPLE_IMAGE} tidak ditemukan"}), 404
        
        img_bgr = cv2.imread(SAMPLE_IMAGE)
        if img_bgr is None:
            return jsonify({"success": False, "error": "Gagal membaca file sampel"}), 500
        
        result = process_detection(img_bgr)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/models", methods=["GET"])
def get_model_info():
    """Return model and environment status."""
    return jsonify({
        "model_file": MODEL_PATH,
        "exists": os.path.exists(MODEL_PATH),
        "yolo_loaded": yolo_model is not None,
        "ocr_loaded": ocr_engine is not None,
        "sample_image": SAMPLE_IMAGE,
        "python_version": sys.version
    })

if __name__ == "__main__":
    print("[*] Server running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
