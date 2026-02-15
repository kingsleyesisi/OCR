import numpy as np
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
import os

MODEL = None

# Minimum confidence to accept a digit prediction
CONFIDENCE_THRESHOLD = 0.7

# Minimum contour area (in pixels) to consider as a potential digit
MIN_CONTOUR_AREA = 100


def load_digit_model(model_path='models/digit_cnn.h5'):
    """Load the trained CNN model (singleton)."""
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


def segment_digits(pil_image):
    """
    Segment individual digits from an image using contour detection.

    Args:
        pil_image: PIL Image object (any size/color).

    Returns:
        list of PIL Images, each containing one digit, sorted left-to-right.
        Returns empty list if no digit-like regions found.
    """
    # Convert PIL image to OpenCV format (grayscale)
    img_gray = np.array(pil_image.convert('L'))

    # Auto-invert: if background is light, invert so digits are white on black
    if np.mean(img_gray) > 127:
        img_gray = 255 - img_gray

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Adaptive thresholding for better handling of varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Also try Otsu's thresholding and pick the one with more contours
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours on both thresholded images and use the one with more results
    contours_adaptive, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_otsu, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours_adaptive if len(contours_adaptive) >= len(contours_otsu) else contours_otsu

    # Filter contours by minimum area
    digit_regions = []
    img_height, img_width = img_gray.shape

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Filter out regions that are too wide relative to height (likely not digits)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 1.5 or aspect_ratio < 0.1:
            continue

        # Filter out tiny regions relative to image size
        if w < img_width * 0.02 or h < img_height * 0.05:
            continue

        digit_regions.append((x, y, w, h))

    # Sort regions left-to-right by x coordinate
    digit_regions.sort(key=lambda r: r[0])

    # Merge overlapping regions
    merged = []
    for region in digit_regions:
        if merged:
            prev_x, prev_y, prev_w, prev_h = merged[-1]
            curr_x, curr_y, curr_w, curr_h = region
            # Check if current region overlaps with previous
            if curr_x < prev_x + prev_w:
                # Merge: take the bounding box of both
                new_x = min(prev_x, curr_x)
                new_y = min(prev_y, curr_y)
                new_w = max(prev_x + prev_w, curr_x + curr_w) - new_x
                new_h = max(prev_y + prev_h, curr_y + curr_h) - new_y
                merged[-1] = (new_x, new_y, new_w, new_h)
                continue
        merged.append(region)

    # Crop each digit region from the original grayscale image
    digit_images = []
    for (x, y, w, h) in merged:
        # Add small padding around the digit
        pad = max(int(min(w, h) * 0.15), 2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_width, x + w + pad)
        y2 = min(img_height, y + h + pad)

        cropped = img_gray[y1:y2, x1:x2]
        digit_pil = Image.fromarray(cropped)
        digit_images.append(digit_pil)

    # If no contours found, try the whole image as a single digit
    if not digit_images:
        digit_images.append(Image.fromarray(img_gray))

    return digit_images


def preprocess_single_digit(pil_image, target_size=(28, 28)):
    """
    Preprocess a single cropped digit image for CNN prediction.
    Mimics the MNIST format: white digit on black background, centered in 28x28.

    Args:
        pil_image: PIL Image (grayscale) of a single digit.
        target_size: Output size (default 28x28).

    Returns:
        numpy array of shape (1, 28, 28, 1), normalized to [0, 1].
    """
    img = pil_image.convert('L')

    # Ensure digits are white on black (MNIST convention)
    img_np = np.array(img)
    if np.mean(img_np) > 127:
        img = ImageOps.invert(img)

    # Threshold to clean up noise
    thresholded = img.point(lambda p: 255 if p > 50 else 0)
    bbox = thresholded.getbbox()

    # Crop to the content bounding box
    if bbox:
        img = img.crop(bbox)

    # Resize to fit in 20x20 (MNIST-style with 4px padding)
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Center on 28x28 black canvas
    new_img = Image.new('L', target_size, 0)
    left = (target_size[0] - img.size[0]) // 2
    top = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (left, top))

    # Normalize to [0, 1]
    img_array = np.array(new_img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


def predict_from_pil_image(pil_image, model_path='models/digit_cnn.h5'):
    """
    Predict all handwritten digits from a PIL image.

    Segments the image into individual digits, predicts each one,
    and filters by confidence threshold.

    Args:
        pil_image: PIL Image object.
        model_path: Path to the .h5 model file.

    Returns:
        dict with:
          - 'digits': list of {'digit': int, 'probability': float, 'position': int}
          - 'summary': string of all detected digits concatenated
          - 'message': string if no digits found
        or {'error': str} on failure.
    """
    model = load_digit_model(model_path)
    if model is None:
        return {'error': 'Model could not be loaded. Please train the model first.'}

    try:
        # Segment the image into individual digit crops
        digit_images = segment_digits(pil_image)

        results = []
        position = 1

        for digit_img in digit_images:
            # Preprocess each digit
            processed = preprocess_single_digit(digit_img)

            # Predict
            prediction = model.predict(processed, verbose=0)
            predicted_digit = int(np.argmax(prediction))
            probability = float(np.max(prediction))

            # Only accept predictions above confidence threshold
            if probability >= CONFIDENCE_THRESHOLD:
                results.append({
                    'digit': predicted_digit,
                    'probability': probability,
                    'position': position
                })
                position += 1

        if results:
            summary = ''.join(str(r['digit']) for r in results)
            return {
                'digits': results,
                'summary': summary
            }
        else:
            return {
                'digits': [],
                'message': "Can't identify any digit in this image"
            }

    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}
