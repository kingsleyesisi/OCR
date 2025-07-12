# Flask OCR Web Application

A simple web application built with Flask that performs Optical Character Recognition (OCR) on uploaded images using Tesseract and OpenCV.

## ✨ Features

-   **Image Upload**: Simple web interface to upload an image file.
-   **OCR Processing**: Extracts text from the uploaded image on the backend.
-   **Image Pre-processing**: Uses OpenCV to convert the image to grayscale and applies an adaptive threshold to improve OCR accuracy.
-   **Display Results**: Shows the extracted text on a results page.

## 🛠️ Tech Stack

-   **Backend**: Flask
-   **OCR Engine**: Tesseract (via `pytesseract`)
-   **Image Processing**: OpenCV

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

-   **Python 3.7+**
-   **Tesseract OCR Engine**: This is a system dependency, not a Python package.

    *   **On Debian/Ubuntu:**
        ```bash
        sudo apt update
        sudo apt install tesseract-ocr
        ```
    *   **On macOS (using Homebrew):**
        ```bash
        brew install tesseract
        ```
    *   **On Windows:**
        Download and install from the Tesseract at UB Mannheim page. Make sure to add the Tesseract installation directory to your system's `PATH`.

    **Note**: The application is currently configured in `main.py` to find Tesseract at `/usr/bin/tesseract`. You may need to adjust this path depending on your system.

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♀️ Running the Application

To start the Flask development server, run:

```bash
python main.py
