import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os

MODEL = None

def load_digit_model(model_path='models/digit_cnn.h5'):
    global MODEL
    if MODEL is None:
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return None
        try:
            MODEL = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return MODEL

def preprocess_image(pil_image, target_size=(28, 28)):
    """
    Preprocess the image for digit recognition.
    - Convert to grayscale
    - Auto-invert colors (if bright background)
    - Crop to content (digit)
    - Resize to fit 20x20
    - Center on 28x28 canvas
    """
    # Convert to grayscale
    img = pil_image.convert('L')

    # Auto-invert: if mean pixel value is high (light background), invert.
    img_np = np.array(img)
    if np.mean(img_np) > 127:
        img = ImageOps.invert(img)

    # Threshold to zero out noise for better cropping
    # This helps getbbox() find the actual digit instead of noise
    # We use a temporary image for bbox calculation
    thresholded = img.point(lambda p: 255 if p > 50 else 0)
    bbox = thresholded.getbbox()

    # Crop to bounding box of the content
    if bbox:
        img = img.crop(bbox)

    # Resize to fit in 20x20 (leaving 4px padding on sides like MNIST)
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Create 28x28 black background
    new_img = Image.new('L', target_size, 0)

    # Center paste
    left = (target_size[0] - img.size[0]) // 2
    top = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (left, top))

    # Normalize
    img_array = np.array(new_img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

def predict_from_pil_image(pil_image, model_path='models/digit_cnn.h5'):
    """
    Predict digit from a PIL image.

    Args:
        pil_image: PIL Image object.
        model_path: Path to the .h5 model file.

    Returns:
        dict: {'digit': int, 'probability': float} or {'error': str}
    """
    model = load_digit_model(model_path)
    if model is None:
        return {'error': 'Model could not be loaded. Please train the model first.'}

    # Preprocess
    try:
        processed_img = preprocess_image(pil_image)

        # Predict
        prediction = model.predict(processed_img)
        predicted_digit = int(np.argmax(prediction))
        probability = float(np.max(prediction))

        return {
            'digit': predicted_digit,
            'probability': probability
        }
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}
