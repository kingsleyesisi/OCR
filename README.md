# Handwritten Digit Recognition System

A lightweight, modern web application for recognizing handwritten digits (0-9) using a Convolutional Neural Network (CNN). This application supports detecting multiple digits in a single image.

## Overview

This project uses **TensorFlow/Keras** to train a model on the MNIST dataset and **OpenCV** to segment multiple digits from uploaded images. The web interface allows users to upload images of handwritten digits, which are then processed and classified by the model in real-time.

## Prerequisites

- **Python 3.8+**
- **pip** (Python package installer)
- **Docker** (optional, for containerized deployment)

## 🚀 Quick Start (Local)

### 1. Setup Environment

First, clone the repository and install the dependencies.

```bash
# Clone the repository
git clone <repository_url>
cd <repository_folder>

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

You must train the model before running the application. We have provided a helper script for this.

```bash
# Run training script (Linux/Mac)
bash run_training.sh

# Or manually:
# PYTHONPATH=. python handwriting/train_mnist.py --epochs 5 --save models/digit_cnn.h5
```

This will:
- Download the MNIST dataset.
- Train the CNN for 5 epochs.
- Save the trained model to `models/digit_cnn.h5`.

### 3. Run the Application

Start the web server:

```bash
# Run application script (Linux/Mac)
bash run_app.sh

# Or manually:
# python main.py
```

Open your browser to `http://localhost:5000`.

---

## 🐳 Docker Deployment

You can run both training and the web application using Docker.

### 1. Build the Image

```bash
docker build -t digit-recognizer .
```

### 2. Train inside Docker

To train the model inside a container and save the file to your host machine, mount the `models` directory:

```bash
docker run --rm -v $(pwd)/models:/app/models digit-recognizer python handwriting/train_mnist.py --epochs 5
```

### 3. Run the App

Run the container, ensuring the trained model is available (either trained previously or mounted):

```bash
docker run -p 5000:5000 -v $(pwd)/models:/app/models digit-recognizer
```

Open `http://localhost:5000` in your browser.

**Note:** The Dockerfile is optimized based on `python:3.11-slim` but includes necessary system libraries (`libgl1-mesa-glx`, `libglib2.0-0`, etc.) required by OpenCV.

---

## 🛠 Project Structure

- **`handwriting/`**: Core application logic.
  - `model.py`: Neural network architecture.
  - `train_mnist.py`: Training script.
  - `predict.py`: Image preprocessing, segmentation (using OpenCV), and inference.
  - `api.py`: Flask blueprint for API routes.
- **`templates/`**: HTML frontend.
- **`models/`**: Stores the trained `.h5` model files.
- **`main.py`**: Application entry point.
- **`run_training.sh`**: Helper script for training.
- **`run_app.sh`**: Helper script for running the server.

## API Documentation

The application exposes a REST API for programmatic access.

### Predict Digits
**Endpoint:** `POST /api/digit_recognize`

**Parameters:**
- `file`: The image file (form-data).

**Example Request:**
```bash
curl -X POST -F "file=@digits.png" http://localhost:5000/api/digit_recognize
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "digit": 1,
      "probability": 0.992
    },
    {
      "digit": 2,
      "probability": 0.985
    }
  ]
}
```
