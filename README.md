# ✋ Handwritten Digit Recognition (OCR)

An AI-powered web application that recognizes **multiple handwritten digits** (0-9) from uploaded images using a deep Convolutional Neural Network (CNN) trained on the MNIST dataset.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Features

- **Multi-Digit Detection** — Upload an image with one *or more* handwritten digits and all of them are detected and recognized individually
- **Deep CNN Model** — 3-block convolutional network with BatchNormalization and Dropout, trained with data augmentation for high accuracy (>99%)
- **Smart Segmentation** — Uses OpenCV contour detection to isolate individual digits and sorts them left-to-right
- **Confidence Filtering** — Predictions below a 70% confidence threshold are rejected. If no digit meets the threshold, a clear message is shown
- **Non-Digit Rejection** — Images without recognizable digits (e.g., photos, text) are handled gracefully with a "Can't identify any digit" message
- **Modern Web UI** — Dark-themed, responsive interface with animated confidence bars and drag-and-drop upload
- **REST API** — JSON endpoint for integration with other applications
- **Docker Support** — Production-ready Dockerfile included

---

## 📋 Prerequisites

- **Python 3.11+**
- **pip** (Python package manager)
- **Git** (to clone the repo)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/OCR.git
cd OCR
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**

| Package | Purpose |
|---------|---------|
| `flask` | Web framework |
| `tensorflow-cpu` | Deep learning (CNN model) |
| `numpy` | Numerical operations |
| `Pillow` | Image processing |
| `opencv-python-headless` | Digit segmentation via contours |
| `gunicorn` | Production WSGI server |
| `python-dotenv` | Environment variables |

### 4. Train the Model

The model must be trained before first use. This trains a CNN on the MNIST handwritten digit dataset:

```bash
python -m handwriting.train_mnist
```

**Training options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 15 | Number of training epochs |
| `--batch` | 64 | Batch size |
| `--save` | `models/digit_cnn.h5` | Path to save the trained model |

**Example with custom settings:**

```bash
python -m handwriting.train_mnist --epochs 20 --batch 128
```

> **Note:** Training on CPU takes approximately 5-8 minutes per epoch. The model typically reaches **>99% accuracy** on the test set. Early stopping will halt training automatically if accuracy plateaus.

### 5. Run the Application

```bash
python main.py
```

The app will start on **<http://localhost:5000>**.

Open your browser and navigate to `http://localhost:5000` to use the web interface.

---

## 🌐 Using the Web Interface

1. **Open** `http://localhost:5000` in your browser
2. **Upload** an image by clicking the upload area or dragging & dropping
3. **Click** "Recognize Digits"
4. **View results** — all detected digits are displayed with individual confidence scores

### Supported Image Formats

PNG, JPG, JPEG, BMP (max file size: 16MB)

### What Happens

| Image Content | Result |
|---------------|--------|
| Single handwritten digit | Shows the digit with confidence % |
| Multiple handwritten digits | Shows all digits left-to-right with individual confidence bars |
| No recognizable digits | Displays "Can't identify any digit in this image" |

---

## 📡 API Documentation

### `POST /api/digit_recognize`

Upload an image to recognize handwritten digits.

**Request:**

```bash
curl -X POST -F "file=@your_image.png" http://localhost:5000/api/digit_recognize
```

**Success Response (digits found):**

```json
{
  "success": true,
  "digits": [
    { "digit": 3, "probability": 0.98, "position": 1 },
    { "digit": 7, "probability": 0.95, "position": 2 }
  ],
  "summary": "37"
}
```

**Success Response (no digits found):**

```json
{
  "success": true,
  "digits": [],
  "message": "Can't identify any digit in this image"
}
```

**Error Response:**

```json
{
  "success": false,
  "error": "Model could not be loaded. Please train the model first."
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the request was processed successfully |
| `digits` | array | List of detected digit objects |
| `digits[].digit` | integer | The recognized digit (0-9) |
| `digits[].probability` | float | Confidence score (0.0 to 1.0) |
| `digits[].position` | integer | Position of the digit (left-to-right, 1-indexed) |
| `summary` | string | All detected digits concatenated as a string |
| `message` | string | Displayed when no digits are found |
| `error` | string | Error description (only on failure) |

---

## 🐳 Docker

### Build the Image

```bash
docker build -t digit-recognizer .
```

### Run the Container

```bash
docker run -p 5000:5000 digit-recognizer
```

> **Important:** Make sure the trained model (`models/digit_cnn.h5`) exists before building the Docker image, as it gets copied into the container.

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

---

## 📁 Project Structure

```
OCR/
├── main.py                     # Flask application entry point
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container configuration
├── .env                        # Environment variables
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── handwriting/
│   ├── __init__.py
│   ├── api.py                  # REST API endpoint (/api/digit_recognize)
│   ├── model.py                # CNN model architecture definition
│   ├── predict.py              # Multi-digit segmentation & prediction logic
│   └── train_mnist.py          # Model training script with data augmentation
├── models/
│   └── digit_cnn.h5            # Trained CNN model (generated by training)
├── templates/
│   └── index.html              # Web UI (upload interface + results display)
└── tests/
    └── test_handwriting.py     # Unit tests
```

---

## 🧠 How It Works

### Model Architecture

The CNN uses a 3-block architecture optimized for handwriting recognition:

```
Input (28×28×1) → [Conv2D×2 + BatchNorm + ReLU + MaxPool + Dropout]×2
                → [Conv2D + BatchNorm + ReLU + Dropout]
                → Dense(256) + BatchNorm + ReLU + Dropout(0.5)
                → Dense(10, softmax) → Output
```

- **1.75M parameters** with BatchNormalization for training stability
- **Dropout** (0.25 on conv layers, 0.5 on dense) prevents overfitting
- Trained with **data augmentation** (random shifts, brightness, contrast, zoom) to generalize to real-world handwriting

### Multi-Digit Detection Pipeline

1. **Convert** uploaded image to grayscale
2. **Auto-invert** if background is light (ensure white-on-black like MNIST)
3. **Threshold** using adaptive Gaussian and Otsu methods
4. **Find contours** to locate individual digit regions
5. **Filter** by area, aspect ratio, and size
6. **Sort** regions left-to-right
7. **Crop & preprocess** each digit to 28×28 MNIST format
8. **Predict** each digit with the CNN
9. **Filter** predictions below 70% confidence

---

## ⚙️ Configuration

| Variable | File | Default | Description |
|----------|------|---------|-------------|
| `SECRET_KEY` | `.env` | — | Flask secret key |
| `DEBUG` | `.env` | `True` | Flask debug mode |
| `MAX_CONTENT_LENGTH` | `main.py` | 16MB | Maximum upload file size |
| `CONFIDENCE_THRESHOLD` | `predict.py` | 0.7 | Minimum confidence to accept a digit |
| `MIN_CONTOUR_AREA` | `predict.py` | 100 | Minimum pixel area for a digit contour |

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
