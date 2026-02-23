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

    # AdaptiveThreshold expects dark text on light background.
    # Force light background for the segmentation phase
    if np.mean(img_gray) < 127:
        img_gray = 255 - img_gray

    # Apply Gaussian blur to reduce noise but preserve edges for pencil
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # THRESH_BINARY_INV makes dark text white, and light background black
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )

    # Use a morphological close to connect broken parts of thin/blurry digits
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_height, img_width = img_gray.shape

    digit_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore objects at the very bottom edge (like watermarks e.g. alamy)
        if y > img_height * 0.9:
            continue

        # Filter out regions that are too wide relative to height (likely not digits)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 4.0 or aspect_ratio < 0.1:
            continue

        # Filter out tiny regions (noise dots) using absolute minimums
        if w < 8 or h < 15:
            continue

        digit_regions.append((x, y, w, h))

    # Sort regions top-to-bottom first, then left-to-right
    # Create "lines" of digits if y-coordinates are close
    digit_regions.sort(key=lambda r: r[1])  # Sort by Y first
    
    lines = []
    current_line = []
    line_y = -1
    
    for r in digit_regions:
        x, y, w, h = r
        if line_y == -1:
            line_y = y
            current_line.append(r)
        elif abs(y - line_y) < img_height * 0.15:  # Same line threshold
            current_line.append(r)
        else:
            lines.append(sorted(current_line, key=lambda k: k[0]))
            current_line = [r]
            line_y = y
            
    if current_line:
        lines.append(sorted(current_line, key=lambda k: k[0]))
        
    sorted_regions = []
    for line in lines:
        sorted_regions.extend(line)

    # Merge overlapping regions within sorted order
    merged = []
    
    for region in sorted_regions:
        if merged:
            prev_x, prev_y, prev_w, prev_h = merged[-1]
            curr_x, curr_y, curr_w, curr_h = region
            
            # Check if current region physically overlaps horizontally
            if curr_x <= prev_x + prev_w:
                # But only merge if they also overlap vertically (they are on the same line)
                if not (curr_y > prev_y + prev_h or curr_y + curr_h < prev_y):
                    # Merge: take the bounding box of both
                    new_x = min(prev_x, curr_x)
                    new_y = min(prev_y, curr_y)
                    new_w = max(prev_x + prev_w, curr_x + curr_w) - new_x
                    new_h = max(prev_y + prev_h, curr_y + curr_h) - new_y
                    merged[-1] = (new_x, new_y, new_w, new_h)
                    continue
                    
        merged.append(region)

    # Crop each digit region from the binary thresholded image
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

    # The image is cropped from a standardized light-background grayscale image.
    # We invert it for the CNN (white text on black background)
    img_np = np.array(img)
    img_np = 255 - img_np
        
    # Apply Otsu's threshold to cleanly separate the pencil stroke from the local background crop
    _, img_clean = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Thicken the clean lines so they match CNN MNIST training
    kernel = np.ones((2, 2), np.uint8)
    img_clean = cv2.dilate(img_clean, kernel, iterations=1)
    
    img = Image.fromarray(img_clean)

    # Find bounding box strictly on the cleaned image
    bbox = img.getbbox()

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
        print(f"Segmented {len(digit_images)} potential digit regions.")

        results = []
        position = 1

        for idx, digit_img in enumerate(digit_images):
            # Preprocess each digit
            processed = preprocess_single_digit(digit_img)
            
            # Debug: save the crop the CNN will see
            import cv2
            cv2.imwrite(f"debug_crop_{idx}.png", (processed[0, :, :, 0] * 255).astype(np.uint8))

            # Predict
            prediction = model.predict(processed, verbose=0)[0] # Extract the 1D array
            
            # Sort indices by probability
            sorted_indices = np.argsort(prediction)[::-1]
            predicted_digit = int(sorted_indices[0])
            probability = float(prediction[predicted_digit])
            
            # If highest class is Noise (10), check the second highest
            if predicted_digit == 10:
                second_best_digit = int(sorted_indices[1])
                second_best_prob = float(prediction[second_best_digit])
                
                # If the second best is a real digit and probability is decent, use it instead
                # (Allows faint pencil digits that look slightly noisy to pass)
                if second_best_prob >= 0.15:
                    predicted_digit = second_best_digit
                    probability = second_best_prob
                else:
                    print(f"Region {idx} rejected as noise. (Noise Prob: {probability:.3f}, Next Best: {second_best_digit} at {second_best_prob:.3f})")
                    continue
            
            print(f"Region {idx}: Predicted {predicted_digit} with prob {probability:.3f} (Was initially {int(sorted_indices[0])})")

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
            
            # Since the user requested any number, we just return the joined summary directly
            return {
                'digits': results,
                'summary': summary
            }

        else:
            return {
                'digits': [],
                'message': "Can't identify any valid digits in this image"
            }

    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}
