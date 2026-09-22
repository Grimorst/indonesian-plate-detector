# Indonesian Plate Detector

AI-powered Indonesian License Plate Recognition (ALPR) web application using a custom-trained YOLO26 model and PaddleOCR for automatic license plate detection and text recognition.

The application is designed to detect Indonesian vehicle license plates from images or a live camera feed, then recognize the detected plate characters using OCR.

---

## Features

* 🚗 Indonesian license plate detection
* 🤖 Custom-trained YOLO26 object detection model
* 🔤 License plate text recognition using PaddleOCR
* 📷 Image-based license plate detection
* 🎥 Live camera detection
* ✂️ Automatic license plate cropping
* 📊 Detection confidence information
* 🖼️ Annotated detection results
* 🌐 Flask-based web interface
* 💻 Can run locally on Windows
* 🚀 Ready to be deployed to a VPS

---

## Demo

The application provides a web interface where users can upload a vehicle image or use a live camera.

The detection pipeline works as follows:

```text
Image / Camera
      │
      ▼
 YOLO26 Detection
      │
      ▼
License Plate Bounding Box
      │
      ▼
   Plate Crop
      │
      ▼
  PaddleOCR
      │
      ▼
Recognized Plate Number
```

---

## Technology Stack

| Technology          | Purpose                        |
| ------------------- | ------------------------------ |
| Python              | Main programming language      |
| Flask               | Web application backend        |
| YOLO26              | License plate object detection |
| Ultralytics         | YOLO model implementation      |
| PaddleOCR           | License plate text recognition |
| OpenCV              | Image and video processing     |
| NumPy               | Numerical and image processing |
| HTML/CSS/JavaScript | Web interface                  |

---

## Project Structure

```text
indonesian-plate-detector/
│
├── models/
│   └── Grimorsbest.pt
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── templates/
│   └── index.html
│
├── alpr.py
├── app.py
├── run.bat
├── sistem.md
├── .gitignore
└── README.md
```

### Main Files

#### `app.py`

Flask application entry point.

It handles the web application and connects the frontend with the ALPR processing pipeline.

#### `alpr.py`

Contains the main Automatic License Plate Recognition (ALPR) processing logic, including license plate detection, image processing, and OCR.

#### `models/Grimorsbest.pt`

Custom-trained YOLO26 model used to detect Indonesian license plates.

#### `templates/index.html`

Main HTML interface of the web application.

#### `static/css/style.css`

Styling for the web interface.

#### `static/js/app.js`

Frontend JavaScript logic for interacting with the Flask backend and handling detection results.

#### `run.bat`

Windows batch script for starting the application.

#### `sistem.md`

Project/system documentation and technical notes.

---

## Requirements

Recommended environment:

* Windows 10/11
* Python 3.10+
* Git
* At least 8 GB RAM recommended
* Webcam for live camera detection

The application can run using CPU, although GPU acceleration is recommended for faster YOLO inference.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Grimorst/indonesian-plate-detector.git
cd indonesian-plate-detector
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

After activation, the terminal should show something similar to:

```text
(venv) PS C:\...\indonesian-plate-detector>
```

### 3. Install Dependencies

Install the required Python packages:

```powershell
pip install -r requirements.txt
```

> `requirements.txt` should contain the tested dependencies for the current version of the application.

---

## Running the Application

### Windows

The easiest way to start the application is using:

```text
run.bat
```

Or run Flask directly:

```powershell
python app.py
```

After the server starts, open the local address shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

---

## How It Works

The application follows a two-stage ALPR pipeline.

### 1. License Plate Detection

The YOLO26 model detects the location of a vehicle license plate in the input image.

The model returns a bounding box and confidence score.

```text
Input Image
     │
     ▼
   YOLO26
     │
     ▼
License Plate
Bounding Box
```

### 2. License Plate Cropping

After the plate is detected, the detected region is cropped from the original image.

```text
Vehicle Image
     │
     ▼
Detected Bounding Box
     │
     ▼
License Plate Crop
```

### 3. OCR Processing

The cropped license plate is passed to PaddleOCR.

```text
License Plate Crop
        │
        ▼
    PaddleOCR
        │
        ▼
  Plate Characters
```

### 4. Result

The application combines the detection and OCR results and displays the recognized license plate together with the detection information.

---

## Model

The project uses a custom-trained YOLO26 model:

```text
models/Grimorsbest.pt
```

The model is specifically trained for Indonesian vehicle license plate detection.

The model is included in this repository because the current model file is relatively small and makes the project easier to reproduce and deploy.

### Model Purpose

The model performs:

```text
Vehicle Image
      ↓
License Plate Detection
      ↓
Bounding Box
      ↓
License Plate Crop
```

OCR is handled separately by PaddleOCR.

Therefore:

```text
YOLO26
   = Detects WHERE the plate is

PaddleOCR
   = Reads WHAT characters are on the plate
```

---

## ALPR Pipeline

The complete processing pipeline can be represented as:

```text
                ┌─────────────────┐
                │ Image / Camera  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     YOLO26      │
                │ Plate Detection │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Plate Cropping │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    PaddleOCR    │
                │ Text Recognition│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Detection Result│
                │ + Plate Number  │
                └─────────────────┘
```

---

## Input

The application can process vehicle images containing Indonesian license plates.

For best results, images should provide:

* Clearly visible license plates
* Sufficient lighting
* Reasonable image resolution
* Limited motion blur
* Minimal obstruction of the license plate

Performance can decrease when plates are heavily blurred, obstructed, extremely small, tilted significantly, or captured under difficult lighting conditions.

---

## Output

The application can provide information such as:

* Detected license plate
* Recognized license plate text
* Detection confidence
* Cropped license plate
* Annotated image
* Processing statistics

Example:

```text
Plate Number : B 1234 XYZ
Confidence   : 0.95
```

The actual result depends on image quality, detection performance, and OCR accuracy.

---

## YOLO Detection vs OCR

The project separates license plate detection and character recognition.

### YOLO26

YOLO26 is responsible for locating the license plate.

```text
"What part of the image is the license plate?"
```

### PaddleOCR

PaddleOCR is responsible for recognizing the characters.

```text
"What characters are written on the license plate?"
```

This separation allows the detection and OCR components to be improved independently.

---

## Training Data

The training datasets used during model development are **not included in this repository**.

This repository contains the application and the trained model required to run the current ALPR system.

Training datasets, experiments, temporary files, and local development environments are excluded using `.gitignore`.

---

## Development

This project was developed as an Indonesian license plate recognition system combining computer vision, object detection, OCR, and web application technologies.

The development process includes:

```text
Dataset Preparation
       ↓
YOLO Model Training
       ↓
Model Evaluation
       ↓
ALPR Integration
       ↓
OCR Integration
       ↓
Flask Web Application
       ↓
Local Testing
       ↓
Deployment Preparation
```

---

## Limitations

The current system is primarily focused on Indonesian license plate detection and recognition.

Recognition performance may be affected by:

* Low-resolution images
* Poor lighting
* Strong reflections
* Motion blur
* Dirty or damaged plates
* Occluded plates
* Extreme viewing angles
* Very small license plates
* OCR errors

The YOLO detector and OCR system are separate components, so a correct plate detection does not always guarantee correct character recognition.

---

## Future Improvements

Potential improvements include:

* Improved YOLO model training
* Larger and more diverse Indonesian license plate datasets
* Better OCR preprocessing
* OCR result validation
* Indonesian plate format validation
* Multi-frame OCR consensus for video
* Improved real-time camera performance
* GPU acceleration
* Detection history
* Database integration
* REST API
* Authentication
* Production deployment with Nginx and Gunicorn
* VPS deployment
* Docker containerization

---

## Deployment

The application is structured so that it can be deployed to a Linux VPS.

A typical production architecture can be:

```text
                Internet
                   │
                   ▼
             ┌───────────┐
             │   Nginx   │
             └─────┬─────┘
                   │
                   ▼
             ┌───────────┐
             │  Gunicorn │
             └─────┬─────┘
                   │
                   ▼
             ┌───────────┐
             │   Flask   │
             └─────┬─────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
      YOLO26            PaddleOCR
```

The repository can therefore be used as the source code for a future VPS deployment.

---

## Security Notes

Do not commit sensitive information such as:

```text
.env
API keys
passwords
private credentials
access tokens
secret keys
```

These files should remain local and are excluded through `.gitignore`.

---

## License

This project currently does not include a specific open-source license.

If this repository is later intended for public redistribution or commercial use, an appropriate license can be added.

---

## Author

**Alfian Nugroho Jati**

Indonesian License Plate Recognition project using YOLO26, PaddleOCR, OpenCV, and Flask.

---

## Repository

GitHub:

https://github.com/Grimorst/indonesian-plate-detector

---

## Disclaimer

This project is intended for educational, research, and development purposes.

License plate recognition accuracy depends on the quality of the detection model, input images, camera conditions, and OCR processing.
