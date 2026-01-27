import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os
import cv2

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

def preprocess_for_model(pil_image, target_size=(28, 28)):
    """
    Takes a PIL crop of a single digit, processes it to match MNIST,
    and returns a numpy array ready for prediction.
    """
    # 1. Grayscale
    img = pil_image.convert('L')

    # 2. Invert if needed (MNIST is white on black)
    # Check if background is light
    img_np = np.array(img)
    if np.mean(img_np) > 127:
        img = ImageOps.invert(img)

    # 3. Fit to 20x20 box (preserving aspect ratio)
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # 4. Center on 28x28 canvas
    new_img = Image.new('L', target_size, 0)
    left = (target_size[0] - img.size[0]) // 2
    top = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (left, top))

    # 5. Normalize
    img_array = np.array(new_img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

def find_digit_crops(pil_image):
    """
    Finds bounding boxes of digits in the image using OpenCV.
    Returns list of PIL Images (crops) and their bounding boxes.
    """
    # Convert PIL to OpenCV format (BGR)
    open_cv_image = np.array(pil_image)
    # Handle RGB vs RGBA vs Grayscale input
    if len(open_cv_image.shape) == 2:
        gray = open_cv_image
    elif open_cv_image.shape[2] == 4:
         gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2GRAY)
    else:
         gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

    # Blur and Threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's thresholding usually works well for dark text on light bg or vice versa
    # We assume dark digit on light background for standard photos, or invert
    # Let's verify average brightness to decide inversion for thresholding
    mean_val = np.mean(blurred)
    if mean_val > 127:
        # Light background -> Threshold Binary Inverted
        thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        # Dark background -> Threshold Binary
        thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU

    _, thresh = cv2.threshold(blurred, 0, 255, thresh_type)

    # Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []

    # Filter and sort contours
    valid_contours = []
    min_area = 50 # Ignore small noise

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            valid_contours.append((x, y, w, h))

    # Sort left to right based on x coordinate
    valid_contours.sort(key=lambda b: b[0])

    # Extract crops
    for (x, y, w, h) in valid_contours:
        # Add small padding
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w += 2 * pad
        h += 2 * pad

        # Crop from original PIL image
        crop = pil_image.crop((x, y, x + w, y + h))
        crops.append({
            'image': crop,
            'bbox': (x, y, w, h)
        })

    return crops

def predict_from_pil_image(pil_image, model_path='models/digit_cnn.h5'):
    """
    Predict digit(s) from a PIL image.
    Supports multi-digit recognition.

    Args:
        pil_image: PIL Image object.
        model_path: Path to the .h5 model file.

    Returns:
        dict: {'predictions': [ {'digit': int, 'probability': float} ], 'success': bool}
    """
    model = load_digit_model(model_path)
    if model is None:
        return {'error': 'Model could not be loaded. Please train the model first.', 'success': False}

    try:
        # 1. Segment image into potential digit crops
        crops = find_digit_crops(pil_image)

        # If no contours found, try the whole image as fallback
        if not crops:
             crops = [{'image': pil_image, 'bbox': pil_image.getbbox()}]

        predictions = []

        for item in crops:
            crop_img = item['image']

            # Preprocess
            processed_input = preprocess_for_model(crop_img)

            # Predict
            pred_probs = model.predict(processed_input)
            predicted_digit = int(np.argmax(pred_probs))
            probability = float(np.max(pred_probs))

            # Filter low confidence predictions (likely noise or letters)
            # Threshold of 0.5 is conservative, can be tuned.
            if probability > 0.5:
                predictions.append({
                    'digit': predicted_digit,
                    'probability': probability,
                    # 'bbox': item['bbox'] # Optional: return bbox if UI needs it
                })

        if not predictions:
             return {'predictions': [], 'success': True, 'message': 'No confident digits found.'}

        return {
            'predictions': predictions,
            'success': True
        }

    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}', 'success': False}
