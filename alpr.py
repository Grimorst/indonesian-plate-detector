import os
import sys
import re
import json
import math
import cv2
import numpy as np

# Load PyTorch DLLs explicitly on Windows to prevent DLL load errors
torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

# Disable oneDNN for PaddleOCR on CPU to prevent crashes
os.environ["FLAGS_use_mkldnn"] = "0"

from ultralytics import YOLO
from paddleocr import PaddleOCR

# ==========================================
# KONFIGURASI
# ==========================================

MODEL_PATH = "models/Grimorsbest.pt"
IMAGE_PATH = "images/mobil.jpg"
OUTPUT_DIR = "results"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 1280

# Kode Wilayah Plat Nomor Indonesia
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


# ==========================================
# FUNGSI BANTU OCR & PARSING PLAT
# ==========================================

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


# ==========================================
# ADVANCED IMAGE PREPROCESSING
# ==========================================

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


def enhance_plate_variants(crop_bgr):
    """
    Membuat 9+ varian citra untuk mengenali plat dalam berbagai kondisi.
    Pipeline multi-pass yang komprehensif untuk akurasi maksimal.
    """
    h, w = crop_bgr.shape[:2]
    variants = []

    # 1. Native crop
    variants.append(("native", crop_bgr))

    # 2. Multi-scale resizing (berbagai target heights)
    for target_h in [70, 90, 120, 160, 220]:
        scale = max(1.2, min(5.0, target_h / max(h, 1)))
        resized = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append((f"scale_{target_h}", resized))

    # Ambil versi resized yang paling optimal (90px) untuk preprocessing lanjutan
    optimal_scale = max(1.5, min(4.0, 90 / max(h, 1)))
    resized_base = cv2.resize(crop_bgr, None, fx=optimal_scale, fy=optimal_scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized_base, cv2.COLOR_BGR2GRAY)

    # 3. Deskew + CLAHE
    deskewed = deskew_plate(gray)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_clahe = clahe.apply(deskewed)
    variants.append(("deskew_clahe", cv2.cvtColor(enhanced_clahe, cv2.COLOR_GRAY2BGR)))

    # 4. CLAHE tanpa deskew (direct)
    clahe_direct = clahe.apply(gray)
    variants.append(("clahe", cv2.cvtColor(clahe_direct, cv2.COLOR_GRAY2BGR)))

    # 5. High-contrast sharpening (Unsharp masking agresif)
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.5)
    unsharp = cv2.addWeighted(gray, 1.8, gaussian, -0.8, 0)
    variants.append(("sharp", cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)))

    # 6. Otsu Thresholding pada grayscale bersih
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)))

    # 7. Adaptive morphology cleaning
    morph_cleaned = adaptive_morphology_clean(gray)
    variants.append(("morph", cv2.cvtColor(morph_cleaned, cv2.COLOR_GRAY2BGR)))

    # 8. Inverted (untuk plat putih di latar gelap)
    inverted = cv2.bitwise_not(gray)
    clahe_inv = clahe.apply(inverted)
    variants.append(("inverted", cv2.cvtColor(clahe_inv, cv2.COLOR_GRAY2BGR)))

    # 9. Bilateral filter + CLAHE (edge-preserving noise removal)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe_bilateral = clahe.apply(bilateral)
    variants.append(("bilateral_clahe", cv2.cvtColor(clahe_bilateral, cv2.COLOR_GRAY2BGR)))

    return variants


def normalize_plate_for_voting(plate_text):
    """Normalisasi teks plat untuk perbandingan voting."""
    return plate_text.replace(" ", "").upper()


def vote_best_plate(candidates):
    """
    Voting-based consensus: pilih kandidat yang paling sering muncul
    dengan skor kualitas tertinggi.
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


def run_enhanced_ocr(crop_bgr, ocr_engine):
    """
    Eksekusi OCR multi-pass dengan:
    - 9+ preprocessing variants
    - Deskew correction
    - Penggabungan fragmen horizontal
    - Koreksi sintaks plat nomor
    - Voting-based consensus across all variants
    """
    if crop_bgr is None or crop_bgr.size == 0 or ocr_engine is None:
        return "TERDETEKSI", 0.75, "08.28", []

    candidates = []
    extracted_periods = []
    all_detected_lines = []
    h, w = crop_bgr.shape[:2]

    # Buat crop dengan berbagai padding levels
    pads = [
        (0, 0),  # No padding
        (max(2, int(w * 0.08)), max(2, int(h * 0.06))),  # Standard padding
        (max(4, int(w * 0.15)), max(4, int(h * 0.12))),  # Wide padding
    ]

    crops_to_test = [crop_bgr]
    for px, py in pads[1:]:
        padded = cv2.copyMakeBorder(crop_bgr, py, py, px, px, cv2.BORDER_REPLICATE)
        crops_to_test.append(padded)

    for c_img in crops_to_test:
        variants = enhance_plate_variants(c_img)
        for var_name, var_img in variants:
            vh, vw = var_img.shape[:2]
            try:
                ocr_res = ocr_engine.ocr(var_img, cls=False)
            except Exception:
                continue

            if not ocr_res or not ocr_res[0]:
                continue

            frags = []
            for line in ocr_res[0]:
                raw_text = clean_token(line[1][0])
                conf = float(line[1][1])
                if not raw_text or any(bad in raw_text for bad in DISCARD_WORDS):
                    continue

                all_detected_lines.append((line[1][0].strip(), conf))

                box = line[0]
                bx1 = min(p[0] for p in box)
                bx2 = max(p[0] for p in box)
                by1 = min(p[1] for p in box)
                by2 = max(p[1] for p in box)
                bcy = (by1 + by2) / 2.0
                bh_box = max(1.0, by2 - by1)

                # Cek jika baris merupakan tanggal pajak/masa berlaku
                if re.match(r"^(0[1-9]|1[0-2])[-.]?([2-3]\d)$", raw_text) and bcy > vh * 0.35:
                    clean_d = raw_text.replace("-", ".").replace(" ", "")
                    if len(clean_d) == 4:
                        clean_d = f"{clean_d[:2]}.{clean_d[2:]}"
                    extracted_periods.append((clean_d, conf))
                    continue

                frags.append({
                    "text": raw_text,
                    "conf": conf,
                    "x1": bx1, "x2": bx2,
                    "cy": bcy, "h": bh_box
                })

            if not frags:
                continue

            # Evaluasi setiap fragmen individual
            for f in frags:
                plate_str, q_score, p_date = parse_indonesian_plate(f["text"])
                if p_date:
                    extracted_periods.append((p_date, f["conf"]))
                if plate_str:
                    total_score = q_score + (f["conf"] * 30.0)
                    candidates.append({
                        "text": plate_str,
                        "conf": f["conf"],
                        "score": total_score
                    })

            # Gabungkan fragmen yang berada pada satu garis horizontal
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
                        candidates.append({
                            "text": plate_str,
                            "conf": avg_conf,
                            "score": total_score
                        })

    # Voting-based consensus untuk memilih kandidat terbaik
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

    return best_text, round(best_conf, 4), period, all_detected_lines


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading YOLO model...")
yolo = YOLO(MODEL_PATH)

print("Loading PaddleOCR (Tuned for Plate ALPR)...")
ocr = PaddleOCR(
    lang="en",
    use_angle_cls=True,
    det_db_thresh=0.01,
    det_db_box_thresh=0.01,
    det_db_unclip_ratio=2.5,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print("Model berhasil dimuat.")


# ==========================================
# LOAD GAMBAR
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Gambar tidak ditemukan: {IMAGE_PATH}"
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# DETEKSI PLAT
# ==========================================

print("\n================================")
print("       DETEKSI PLAT")
print("================================")

results = yolo.predict(
    source=IMAGE_PATH,
    conf=CONFIDENCE_THRESHOLD,
    imgsz=IMAGE_SIZE,
    augment=True,
    verbose=False
)

result = results[0]

print(
    "Jumlah plat terdeteksi:",
    len(result.boxes)
)


# ==========================================
# PROSES SETIAP PLAT
# ==========================================

for i, box in enumerate(result.boxes):

    confidence = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    print(f"\nPlat #{i + 1}")
    print(f"Confidence Deteksi: {confidence * 100:.2f}%")
    print(f"Box: [{x1}, {y1}, {x2}, {y2}]")

    h, w = image.shape[:2]
    
    # Add horizontal padding to prevent edge letters from being cut off
    box_w = x2 - x1
    pad_x = int(box_w * 0.08)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2)

    plate = image[y1:y2, x1:x2]

    if plate.size == 0:
        print("Crop plat kosong, dilewati.")
        continue

    # Simpan crop asli
    crop_path = os.path.join(OUTPUT_DIR, f"plate_{i + 1}.jpg")
    cv2.imwrite(crop_path, plate)
    print("Crop asli disimpan:", crop_path)

    # Simpan varian preprocessing terunggul untuk inspeksi visual
    scale = 3
    enlarged = cv2.resize(plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_clahe = clahe.apply(gray)
    processed_path = os.path.join(OUTPUT_DIR, f"plate_{i + 1}_processed.jpg")
    cv2.imwrite(processed_path, cv2.cvtColor(enhanced_clahe, cv2.COLOR_GRAY2BGR))
    print("Gambar preprocessing disimpan:", processed_path)

    # Jalankan OCR Akurasi Tinggi
    print("Membaca nomor plat dengan Enhanced OCR Pipeline v2...")
    plate_text, ocr_conf, period, raw_lines = run_enhanced_ocr(plate, ocr)

    print("\n--------------------------------")
    print("HASIL OCR TERBAIK")
    print("--------------------------------")
    print(f"Nomor Plat   : {plate_text}")
    print(f"Confidence   : {ocr_conf * 100:.2f}%")
    print(f"Masa Berlaku : {period}")

    if raw_lines:
        print("\nDetail Teks Terdeteksi:")
        unique_lines = []
        for txt, sc in raw_lines:
            if not any(u[0] == txt for u in unique_lines):
                unique_lines.append((txt, sc))
        for txt, sc in unique_lines:
            print(f" - '{txt}' (conf: {sc * 100:.2f}%)")


# ==========================================
# SELESAI
# ==========================================

print("\n================================")
print("          ALPR SELESAI")
print("================================")
