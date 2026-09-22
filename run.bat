@echo off
title VisionGuard ALPR - Plate Detector
color 0b
echo ================================================================
echo             VISIONGUARD ALPR - AI PLATE DETECTOR
echo         Model: models/Grimorsbest.pt ^| Engine: YOLO + PaddleOCR
echo ================================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment tidak ditemukan di .\venv
    echo Pastikan venv sudah dibuat.
    pause
    exit /b 1
)

echo [*] Memulai server VisionGuard ALPR di http://127.0.0.1:5000 ...
echo [*] Membuka browser otomatis...
start http://127.0.0.1:5000

venv\Scripts\python.exe app.py
pause
