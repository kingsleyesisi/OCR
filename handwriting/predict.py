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

def find_digits_contours(pil_image):
    """
    Find potential digit contours in the image using OpenCV.
    Returns a list of (x, y, w, h) bounding boxes sorted left-to-right.
    """
    # Convert PIL to OpenCV format (BGR) then Grayscale
    img_np = np.array(pil_image)

    # Handle RGBA or RGB
    if len(img_np.shape) == 3:
        if img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    # Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding to handle varying lighting (paper texture)
    # Check if background is light
    if np.mean(gray) > 127:
        thresh_type = cv2.THRESH_BINARY_INV
    else:
        thresh_type = cv2.THRESH_BINARY

    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   thresh_type, 11, 2)

    # Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_boxes = []
    img_area = gray.shape[0] * gray.shape[1]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect_ratio = h / float(w)

        # Filter noise:
        # - Area should be reasonable (e.g., > 0.1% of image and < 50%)
        # - Aspect ratio should be digit-like (taller than wide usually, but 1 is ok for 0, etc.)
        if area > 0.001 * img_area and area < 0.5 * img_area:
             if 0.2 < aspect_ratio < 10:
                digit_boxes.append((x, y, w, h))

    # Sort boxes: Bin Y by 50px lines, then X
    # This handles multi-line somewhat reasonably
    digit_boxes.sort(key=lambda b: (b[1] // 50, b[0]))

    return digit_boxes, thresh

def preprocess_box(thresh_img, box, target_size=(28, 28)):
    """
    Extracts the bounding box from the thresholded image and preprocesses it for the model.
    """
    x, y, w, h = box
    roi = thresh_img[y:y+h, x:x+w]

    # ROI is already binary (white on black)

    # Convert to PIL for easy resizing/padding
    roi_pil = Image.fromarray(roi)

    # Resize to fit in 20x20
    roi_pil.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Create 28x28 black background
    new_img = Image.new('L', target_size, 0)

    # Center paste
    left = (target_size[0] - roi_pil.size[0]) // 2
    top = (target_size[1] - roi_pil.size[1]) // 2
    new_img.paste(roi_pil, (left, top))

    # Convert back to numpy
    img_array = np.array(new_img)

    # Optional: Dilate to thicken thin biro lines
    # Only dilate if it's not too dense?
    # Let's apply a small dilation
    kernel = np.ones((2,2), np.uint8)
    img_array = cv2.dilate(img_array, kernel, iterations=1)

    # Normalize
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

def predict_from_pil_image(pil_image, model_path='models/digit_cnn.h5'):
    """
    Predict digits from a PIL image.
    Returns a list of predictions.
    """
    model = load_digit_model(model_path)
    if model is None:
        return {'error': 'Model could not be loaded. Please train the model first.'}

    try:
        boxes, thresh_img = find_digits_contours(pil_image)

        results = []

        if not boxes:
            # Fallback: Try predicting the whole image as one digit if no contours found
            # This handles cases where the digit fills the image
            h, w = thresh_img.shape
            boxes = [(0, 0, w, h)]

        for box in boxes:
            processed_img = preprocess_box(thresh_img, box)

            prediction = model.predict(processed_img, verbose=0)
            predicted_digit = int(np.argmax(prediction))
            probability = float(np.max(prediction))

            # Confidence threshold
            if probability < 0.6:
                digit_res = 'Unrecognized'
            else:
                digit_res = predicted_digit

            res = {
                'digit': digit_res,
                'probability': probability,
                'box': box
            }
            results.append(res)

        return results

    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}
