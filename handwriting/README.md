# Handwritten Digit Recognition Project

This project extends the existing OCR application with a module for recognizing handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Project Structure

The files for this project are located in the `handwriting/` directory:

- `handwriting/model.py`: Defines the CNN architecture using TensorFlow/Keras.
- `handwriting/train_mnist.py`: Script to download MNIST data, train the model, and save it.
- `handwriting/predict.py`: Utility to load the model and predict digits from images.
- `handwriting/api.py`: Flask blueprint exposing the prediction API and UI route.
- `templates/digit_recognize.html`: Web interface for uploading images.
- `tests/test_handwriting.py`: Unit tests for the prediction logic.

## Setup

1.  **Install Dependencies**:
    Ensure you have the required packages installed.
    ```bash
    pip install tensorflow-cpu Pillow
    ```
    (Note: `tensorflow` can be used instead of `tensorflow-cpu` if you have GPU support).

## Training the Model

Before using the recognition feature, you must train the model. The training script downloads the MNIST dataset and trains a simple CNN.

**Command:**
```bash
python handwriting/train_mnist.py --epochs 10 --batch 128 --save models/digit_cnn.h5
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 5).
- `--batch`: Batch size (default: 64).
- `--save`: Path to save the trained model (default: `models/digit_cnn.h5`).

The best model (based on validation accuracy) will be saved to `models/digit_cnn.h5`.

## Running the Application

1.  **Start the Server**:
    Run the main application script from the root directory.
    ```bash
    python main.py
    ```
    The server typically starts at `http://localhost:5000`.

2.  **Access the Web UI**:
    Open your browser and navigate to:
    `http://localhost:5000/digit_recognize`

    Here you can upload an image file containing a handwritten digit to get a prediction.

3.  **Use the API**:
    You can also send a POST request directly to the API endpoint.

    **Endpoint:** `POST /api/digit_recognize`
    **Body:** `form-data` with key `file` containing the image.

    **Example cURL:**
    ```bash
    curl -X POST -F "file=@path/to/digit.png" http://localhost:5000/api/digit_recognize
    ```

    **Response:**
    ```json
    {
      "digit": 7,
      "probability": 0.992,
      "success": true
    }
    ```

## Testing

A unit test is provided to verify the prediction logic (mocking the model).

**Run Test:**
```bash
python tests/test_handwriting.py
```
Or if using pytest:
```bash
pytest tests/test_handwriting.py
```

## Evaluation & Notes

### Model Architecture
The model is a basic Convolutional Neural Network (CNN) consisting of:
- 2 Convolutional layers with ReLU activation and Max Pooling.
- Flatten layer.
- Dense hidden layer (64 units, ReLU).
- Output Dense layer (10 units, Softmax).

### Preprocessing
Real-world photos of digits differ from MNIST (which are white digits on black background, 28x28). The system handles this by:
1.  **Grayscale Conversion**: Converting input to single channel.
2.  **Auto-Inversion**: Detecting if the background is light (common in photos) and inverting the image to match MNIST (white on black).
3.  **Cropping**: Finding the bounding box of the digit to remove excess background.
4.  **Resizing & Padding**: Resizing the digit to fit in a 20x20 box and centering it on a 28x28 canvas to match MNIST format.

### Improvements for Real-World Accuracy
-   **EMNIST Dataset**: Training on EMNIST (Extended MNIST) can improve recognition of letters and more diverse handwriting styles.
-   **Data Augmentation**: Applying rotation, zoom, and shift during training can make the model more robust to imperfectly centered or rotated user inputs.
-   **Advanced Preprocessing**: Using adaptive thresholding (e.g., OpenCV `adaptiveThreshold`) and contour detection can better isolate digits from noisy backgrounds compared to simple global thresholding or inversion.
-   **Deskewing**: Calculating image moments to detect skew and rotating the image to upright the digit.

## Docker Notes
If running in Docker, ensure the Dockerfile includes `tensorflow-cpu`.
```dockerfile
RUN pip install tensorflow-cpu
```
