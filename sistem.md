# Dokumentasi Struktur Sistem Plat Detector

## Ringkasan Sistem

Proyek ini adalah aplikasi web untuk mendeteksi dan membaca plat kendaraan dari foto atau kamera live. Sistem memakai Flask sebagai backend, YOLO sebagai model deteksi lokasi plat, dan PaddleOCR untuk membaca teks pada plat.

Alur utama aplikasi:

1. Pengguna membuka halaman web.
2. Pengguna memilih mode Foto atau Kamera Live.
3. Gambar dikirim ke backend melalui API.
4. Backend mendeteksi area plat dengan model `models/best.pt`.
5. Area plat dipotong dan diproses agar teks lebih mudah dibaca OCR.
6. OCR membaca nomor plat.
7. Hasil dikirim kembali ke frontend dalam bentuk JSON.
8. Frontend menampilkan gambar hasil anotasi, nomor plat, confidence, crop plat, statistik, dan riwayat deteksi.

## Struktur Folder

```text
plat-detector/
├── app.py
├── alpr.py
├── detect.py
├── run.bat
├── test_detector.py
├── test_pipeline.py
├── skills-lock.json
├── sistem.md
├── images/
├── models/
├── results/
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── templates/
│   └── index.html
├── venv/
└── __pycache__/
```

## Penjelasan File Utama

### `app.py`

File utama aplikasi. Di dalam file ini terdapat konfigurasi Flask, loading model YOLO, loading PaddleOCR, pipeline deteksi, OCR, dan route API.

Kegunaan utama:

- Menjalankan server Flask.
- Meload model YOLO dari `models/best.pt`.
- Meload PaddleOCR untuk membaca teks plat.
- Menerima gambar dari frontend.
- Memproses gambar menjadi hasil deteksi plat.
- Mengirim hasil deteksi ke frontend dalam format JSON.

Route penting:

| Route | Method | Fungsi |
|---|---:|---|
| `/` | GET | Menampilkan halaman utama dari `templates/index.html`. |
| `/api/detect/image` | POST | Mendeteksi plat dari foto upload. |
| `/api/detect/frame` | POST | Mendeteksi plat dari frame kamera live dalam format base64. |
| `/api/sample` | GET | Menjalankan deteksi pada gambar contoh `images/mobil.jpg`. |
| `/api/models` | GET | Mengecek status model YOLO, OCR, dan environment. |

Fungsi penting:

| Fungsi | Kegunaan |
|---|---|
| `clean_token()` | Membersihkan teks OCR agar hanya berisi huruf dan angka. |
| `parse_indonesian_plate()` | Memformat hasil OCR menjadi pola plat Indonesia. |
| `deskew_plate()` | Meluruskan crop plat yang miring. |
| `adaptive_morphology_clean()` | Membersihkan noise gambar plat. |
| `enhance_plate_variants()` | Membuat beberapa versi crop plat agar OCR lebih akurat. |
| `mat_to_base64()` | Mengubah gambar OpenCV menjadi base64 untuk dikirim ke frontend. |
| `base64_to_mat()` | Mengubah base64 dari kamera menjadi gambar OpenCV. |
| `collect_model_boxes()` | Menjalankan YOLO dan mengambil bounding box plat. |
| `suppress_duplicate_boxes()` | Menghapus bounding box duplikat. |
| `merge_plate_boxes()` | Menggabungkan box yang kemungkinan berasal dari satu plat yang sama. |
| `refine_plate_box()` | Menyempurnakan posisi box plat berdasarkan contour. |
| `run_ocr_on_crop()` | Membaca teks dari crop plat menggunakan PaddleOCR. |
| `process_detection()` | Pipeline utama deteksi: YOLO, crop, OCR, anotasi, dan response JSON. |

### `alpr.py`

File ini berisi logika ALPR/OCR tambahan atau versi modular dari beberapa fungsi pemrosesan plat.

Kegunaan:

- Membersihkan teks hasil OCR.
- Memvalidasi format plat Indonesia.
- Membuat variasi preprocessing gambar plat.
- Melakukan voting kandidat OCR untuk memilih hasil terbaik.

File ini berguna jika pipeline OCR ingin dipisahkan dari `app.py` agar kode backend lebih rapi.

### `detect.py`

File kecil untuk menjalankan deteksi model secara sederhana.

Kegunaan:

- Mengecek apakah model YOLO bisa diload.
- Menjalankan prediksi awal pada gambar.
- Cocok untuk debugging model tanpa menjalankan web app lengkap.

### `run.bat`

File batch untuk menjalankan aplikasi di Windows.

Kegunaan:

- Mempermudah menjalankan server tanpa mengetik command panjang.
- Biasanya digunakan untuk mengaktifkan environment lalu menjalankan `app.py`.

### `test_detector.py`

File test sederhana untuk komponen detector.

Kegunaan:

- Mengecek fungsi deteksi dasar.
- Membantu memastikan perubahan kode tidak merusak proses deteksi.

### `test_pipeline.py`

File test untuk pipeline deteksi.

Kegunaan:

- Menguji alur pemrosesan dari input gambar sampai output hasil.
- Berguna untuk memastikan pipeline utama tetap berjalan setelah perubahan kode.

## Folder Frontend

### `templates/index.html`

File HTML utama aplikasi.

Kegunaan:

- Menyusun struktur tampilan dashboard.
- Menyediakan tombol mode Foto dan Kamera Live.
- Menyediakan area upload/drop foto.
- Menyediakan panel hasil utama.
- Menyediakan ringkasan statistik.
- Menyediakan tabel riwayat deteksi.
- Menghubungkan CSS dan JavaScript.

Bagian penting:

- Header aplikasi.
- Mode switch Foto/Kamera Live.
- Area viewport untuk foto atau kamera.
- Sidebar hasil deteksi.
- Riwayat deteksi.
- Modal preview gambar/crop.

### `static/css/style.css`

File styling utama.

Kegunaan:

- Mengatur layout dashboard.
- Mengatur warna, tombol, panel, tabel, modal, dan responsive mobile.
- Membuat tampilan lebih user-friendly.
- Menyediakan style untuk drag-and-drop dan toast notification.

Bagian penting:

- Variabel warna dan radius di `:root`.
- Layout header, main content, sidebar, dan history.
- Komponen tombol.
- Area upload dan kamera.
- Komponen hasil plat.
- Tabel riwayat.
- Responsive design untuk layar kecil.

### `static/js/app.js`

File JavaScript utama untuk interaksi frontend.

Kegunaan:

- Mengatur perpindahan mode Foto dan Kamera Live.
- Mengirim foto upload ke `/api/detect/image`.
- Mengirim frame kamera live ke `/api/detect/frame`.
- Memuat gambar contoh dari `/api/sample`.
- Menampilkan hasil deteksi ke UI.
- Menampilkan crop plat.
- Mengelola statistik dan riwayat deteksi.
- Mengatur modal preview.
- Mengatur toast notification.
- Mengelola akses kamera browser.

Alur penting di frontend:

1. User memilih file atau klik contoh.
2. JavaScript membuat request ke backend.
3. Backend mengirim response JSON.
4. JavaScript membaca `detections`, `annotated_image`, dan `inference_time_ms`.
5. UI diperbarui dengan nomor plat, confidence, crop, statistik, dan log.

## Folder Data dan Model

### `models/`

Berisi file model deteksi.

File penting:

```text
models/best.pt
```

Kegunaan:

- Bobot model YOLO untuk mendeteksi lokasi plat kendaraan.
- Dipakai oleh `app.py` melalui variabel `MODEL_PATH`.

### `images/`

Berisi gambar input contoh.

File penting:

```text
images/mobil.jpg
```

Kegunaan:

- Gambar contoh untuk tombol `Coba Contoh`.
- Dipakai route `/api/sample`.

### `results/`

Berisi hasil pemrosesan atau gambar output testing.

Kegunaan:

- Menyimpan hasil crop/debug dari percobaan.
- Berguna saat mengecek kualitas preprocessing atau output deteksi.

### `venv/`

Virtual environment Python.

Kegunaan:

- Menyimpan library Python yang dibutuhkan proyek.
- Contoh library: Flask, OpenCV, Ultralytics YOLO, PaddleOCR, NumPy.

### `__pycache__/`

Folder cache Python.

Kegunaan:

- Dibuat otomatis oleh Python.
- Berisi file bytecode `.pyc`.
- Tidak perlu diedit manual.

## Alur Backend Detail

### 1. Inisialisasi Server

Saat `app.py` dijalankan:

1. Flask dibuat dengan `Flask(__name__)`.
2. Folder static dan template diatur.
3. Model YOLO diload dari `models/best.pt`.
4. PaddleOCR diinisialisasi.
5. Server berjalan di `http://127.0.0.1:5000`.

### 2. Deteksi Foto Upload

Endpoint:

```text
POST /api/detect/image
```

Alur:

1. Frontend mengirim file gambar lewat `FormData`.
2. Backend membaca file dari `request.files`.
3. Gambar diubah menjadi format OpenCV BGR.
4. Fungsi `process_detection()` dipanggil.
5. Hasil dikirim ke frontend sebagai JSON.

### 3. Deteksi Kamera Live

Endpoint:

```text
POST /api/detect/frame
```

Alur:

1. Frontend mengambil frame dari webcam.
2. Frame diubah menjadi base64.
3. Backend mengubah base64 menjadi gambar OpenCV.
4. Fungsi `process_detection()` memproses frame.
5. Hasil box dan OCR dikirim kembali.

### 4. Deteksi Gambar Contoh

Endpoint:

```text
GET /api/sample
```

Alur:

1. Backend membaca `images/mobil.jpg`.
2. Gambar diproses lewat `process_detection()`.
3. Hasil dikirim ke frontend.

## Format Response JSON

Contoh struktur response sukses:

```json
{
  "success": true,
  "total_plates": 1,
  "detections": [
    {
      "id": 1,
      "box": [10, 20, 200, 80],
      "confidence": 0.88,
      "plate_text": "B 1387 DKC",
      "plate_period": "09.27",
      "ocr_confidence": 0.99,
      "crop_image": "data:image/jpeg;base64,..."
    }
  ],
  "annotated_image": "data:image/jpeg;base64,...",
  "inference_time_ms": 24565,
  "timestamp": "2026-09-15 23:16:02"
}
```

Keterangan:

| Field | Kegunaan |
|---|---|
| `success` | Status proses berhasil atau gagal. |
| `total_plates` | Jumlah plat yang terdeteksi. |
| `detections` | Daftar hasil deteksi plat. |
| `box` | Koordinat bounding box plat. |
| `confidence` | Confidence model YOLO. |
| `plate_text` | Nomor plat hasil OCR. |
| `plate_period` | Masa berlaku plat jika terbaca. |
| `ocr_confidence` | Confidence OCR. |
| `crop_image` | Gambar crop plat dalam base64. |
| `annotated_image` | Gambar penuh yang sudah diberi anotasi. |
| `inference_time_ms` | Waktu proses deteksi dalam milidetik. |
| `timestamp` | Waktu proses backend. |

## Alur Frontend Detail

### Mode Foto

Komponen:

- Tombol `Unggah Foto`.
- Tombol `Kamera`.
- Tombol `Coba Contoh`.
- Area drag-and-drop.

Alur:

1. User memilih file atau drag-and-drop.
2. `handleImageUpload()` membuat `FormData`.
3. Request dikirim ke `/api/detect/image`.
4. Response ditampilkan lewat `displayFotoResults()`.

### Mode Kamera Live

Komponen:

- Tombol `Mulai Kamera`.
- Tombol `Snapshot`.
- Tombol `Demo Live`.

Alur:

1. User klik `Mulai Kamera`.
2. Browser meminta akses webcam.
3. JavaScript mengambil frame berkala.
4. Frame dikirim ke `/api/detect/frame`.
5. Bounding box dan hasil OCR ditampilkan pada overlay.

### Riwayat Deteksi

Setiap hasil deteksi akan masuk ke tabel riwayat.

Data yang disimpan di frontend:

- Waktu deteksi.
- Mode deteksi.
- Crop plat.
- Nomor plat.
- Confidence YOLO.
- Confidence OCR.
- Status.

Riwayat ini bisa diekspor menjadi CSV melalui tombol `Ekspor CSV`.

## Dependensi Utama

Library utama yang digunakan:

| Library | Kegunaan |
|---|---|
| Flask | Web server dan API backend. |
| OpenCV (`cv2`) | Membaca, memproses, dan menggambar pada gambar. |
| NumPy | Manipulasi array gambar. |
| Ultralytics YOLO | Deteksi lokasi plat. |
| PaddleOCR | OCR teks plat. |
| Font Awesome | Ikon UI frontend. |

## Cara Menjalankan

Jalankan dari folder proyek:

```bash
venv\Scripts\python.exe app.py
```

Atau gunakan:

```bash
run.bat
```

Setelah server aktif, buka:

```text
http://127.0.0.1:5000
```

## Catatan Maintenance

- Jika model diganti, ubah nilai `MODEL_PATH` di `app.py`.
- Jika gambar contoh diganti, ubah `SAMPLE_IMAGE` di `app.py`.
- Jika UI ingin diubah, edit `templates/index.html`, `static/css/style.css`, dan `static/js/app.js`.
- Jika akurasi OCR kurang baik, fokus perbaikan ada di fungsi preprocessing dan OCR seperti `enhance_plate_variants()`, `run_ocr_on_crop()`, dan `parse_indonesian_plate()`.
- Folder `venv/` dan `__pycache__/` tidak perlu diedit manual.

## Pembagian Tanggung Jawab File

| Bagian | File/Folder | Tanggung Jawab |
|---|---|---|
| Backend utama | `app.py` | API, deteksi, OCR, response JSON. |
| Logika ALPR tambahan | `alpr.py` | Helper OCR dan parsing plat. |
| Test/model debug | `detect.py` | Uji deteksi sederhana. |
| Tampilan HTML | `templates/index.html` | Struktur halaman web. |
| Styling | `static/css/style.css` | Desain UI/UX. |
| Interaksi frontend | `static/js/app.js` | Upload, kamera, fetch API, render hasil. |
| Model | `models/best.pt` | Bobot YOLO deteksi plat. |
| Sample input | `images/` | Gambar contoh. |
| Output/debug | `results/` | Hasil crop atau proses testing. |
