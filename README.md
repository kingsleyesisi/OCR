# Handwritten Digit Recognition

A simple, lightweight web application for recognizing handwritten digits (0-9).

Built with:
- **Python** & **Flask**
- **TensorFlow/Keras** (CNN Model)
- **MNIST Dataset**

## Project Structure

- `handwriting/`: Contains the core logic.
  - `model.py`: CNN architecture definition.
  - `train_mnist.py`: Script to train the model on MNIST data.
  - `predict.py`: Image preprocessing and prediction logic.
  - `api.py`: API endpoints.
- `templates/`: HTML templates for the web UI.
- `models/`: Directory where trained models are saved.
- `main.py`: Entry point for the web server.

## Installation

1. **Clone the repo**
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

Before running the app, you need to train the model. This script downloads the MNIST dataset and trains a Convolutional Neural Network.

```bash
# Must be run from the root directory
PYTHONPATH=. python handwriting/train_mnist.py --epochs 5 --save models/digit_cnn.h5
```

This will save the best model to `models/digit_cnn.h5`.

## Running the Web App

Start the Flask server:

```bash
python main.py
```

Open your browser to `http://localhost:5000`. You will see a simple interface to upload digit images.

## API Usage

You can also use the API directly:

**Endpoint:** `POST /api/digit_recognize`
**Data:** Form-data with a file field named `file`.

Example:
```bash
curl -X POST -F "file=@my_digit.png" http://localhost:5000/api/digit_recognize
```

Response:
```json
{
  "success": true,
  "digit": 7,
  "probability": 0.99
}
```

## Testing

Run the unit tests:

```bash
PYTHONPATH=. python tests/test_handwriting.py
```
