from flask import Flask, request, render_template, redirect, url_for, jsonify, flash
import pytesseract as tesseract
import cv2
import numpy as np
import os
import logging
from datetime import datetime
import traceback
from PIL import Image, ImageEnhance
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure Tesseract path
tesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Supported image formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}


class ImageQualityAnalyzer:
    """
    Analyzes image quality to determine the best preprocessing strategy.
    """
    
    @staticmethod
    def calculate_image_quality_metrics(image):
        """
        Calculate various image quality metrics.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Dictionary with quality metrics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance (focus measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate contrast using standard deviation
        contrast = gray.std()
        
        # Calculate brightness (mean intensity)
        brightness = gray.mean()
        
        # Calculate noise estimation using median filter
        median_filtered = cv2.medianBlur(gray, 5)
        noise_estimation = np.mean(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32)))
        
        # Calculate text-like edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        return {
            'sharpness': laplacian_var,
            'contrast': contrast,
            'brightness': brightness,
            'noise': noise_estimation,
            'edge_density': edge_density
        }
    
    @staticmethod
    def is_high_quality_image(metrics):
        """
        Determine if image is high quality based on metrics.
        
        Args:
            metrics: Dictionary of quality metrics
            
        Returns:
            Boolean indicating if image is high quality
        """
        # Define thresholds for high quality
        high_quality_conditions = [
            metrics['sharpness'] > 100,  # Good focus
            metrics['contrast'] > 50,    # Good contrast
            metrics['brightness'] > 50 and metrics['brightness'] < 200,  # Good brightness
            metrics['noise'] < 10,       # Low noise
            metrics['edge_density'] > 0.01  # Reasonable edge density
        ]
        
        # Consider high quality if most conditions are met
        return sum(high_quality_conditions) >= 3


class ImagePreprocessor:
    """
    Advanced image preprocessing class with quality-aware processing.
    """
    
    @staticmethod
    def is_allowed_file(filename):
        """Check if the uploaded file has an allowed extension."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @staticmethod
    def resize_image(image, target_height=1000):
        """
        Resize image to optimal size for OCR while maintaining aspect ratio.
        
        Args:
            image: OpenCV image array
            target_height: Target height in pixels
            
        Returns:
            Resized image array
        """
        height, width = image.shape[:2]
        
        # Only resize if image is significantly larger than target
        if height > target_height * 1.5:
            scale = target_height / height
            new_width = int(width * scale)
            new_height = target_height
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
        
        return image
    
    @staticmethod
    def gentle_enhancement(image, metrics):
        """
        Apply gentle enhancement based on image quality metrics.
        
        Args:
            image: Input image
            metrics: Quality metrics dictionary
            
        Returns:
            Enhanced image
        """
        enhanced = image.copy()
        
        # Only enhance if needed
        if metrics['brightness'] < 100:
            # Slight brightness increase
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.0, beta=20)
            logger.info("Applied brightness enhancement")
        
        if metrics['contrast'] < 40:
            # Slight contrast increase
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=0)
            logger.info("Applied contrast enhancement")
        
        return enhanced
    
    @staticmethod
    def adaptive_preprocessing(image, metrics, is_high_quality):
        """
        Apply preprocessing based on image quality assessment.
        
        Args:
            image: Input image
            metrics: Quality metrics
            is_high_quality: Boolean indicating image quality
            
        Returns:
            Preprocessed image
        """
        if is_high_quality:
            logger.info("High quality image detected - minimal preprocessing")
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Only apply gentle enhancement if really needed
            if metrics['brightness'] < 80 or metrics['contrast'] < 30:
                gray = ImagePreprocessor.gentle_enhancement(gray, metrics)
            
            return gray
        else:
            logger.info("Lower quality image detected - applying comprehensive preprocessing")
            
            # Apply noise reduction for poor quality images
            if metrics['noise'] > 8:
                if len(image.shape) == 3:
                    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                else:
                    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply stronger enhancements for poor quality
            if metrics['brightness'] < 100 or metrics['contrast'] < 40:
                gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
            
            # Apply sharpening if image is blurry
            if metrics['sharpness'] < 50:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                gray = cv2.filter2D(gray, -1, kernel)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            return binary
    
    @staticmethod
    def correct_skew_improved(image):
        """
        Improved skew correction with better angle detection.
        
        Args:
            image: Input image
            
        Returns:
            Skew-corrected image
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Use better edge detection parameters
            edges = cv2.Canny(gray, 30, 100, apertureSize=3)
            
            # Detect lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
            
            if lines is not None and len(lines) > 5:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta) - 90
                    # Filter out vertical lines
                    if abs(angle) < 45:
                        angles.append(angle)
                
                if angles:
                    # Use mode instead of median for better skew detection
                    angle_hist, bins = np.histogram(angles, bins=20, range=(-45, 45))
                    mode_angle = bins[np.argmax(angle_hist)]
                    
                    # Only correct significant skew
                    if abs(mode_angle) > 1.0:
                        (h, w) = image.shape[:2]
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, mode_angle, 1.0)
                        corrected = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                                 flags=cv2.INTER_CUBIC, 
                                                 borderMode=cv2.BORDER_REPLICATE)
                        logger.info(f"Skew corrected by {mode_angle:.2f} degrees")
                        return corrected
            
        except Exception as e:
            logger.warning(f"Skew correction failed: {str(e)}")
        
        return image
    
    @classmethod
    def preprocess_image(cls, image_data, enhancement_level='auto'):
        """
        Main preprocessing pipeline with quality-aware processing.
        
        Args:
            image_data: Raw image data
            enhancement_level: Level of enhancement
            
        Returns:
            Processed image ready for OCR
        """
        try:
            # Convert bytes to numpy array
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Unable to decode image")
            
            logger.info(f"Original image shape: {image.shape}")
            
            # Step 1: Resize to optimal size
            image = cls.resize_image(image)
            
            # Step 2: Analyze image quality
            analyzer = ImageQualityAnalyzer()
            metrics = analyzer.calculate_image_quality_metrics(image)
            is_high_quality = analyzer.is_high_quality_image(metrics)
            
            logger.info(f"Image quality metrics: {metrics}")
            logger.info(f"High quality image: {is_high_quality}")
            
            # Step 3: Apply skew correction if needed
            if enhancement_level != 'light':
                image = cls.correct_skew_improved(image)
            
            # Step 4: Apply quality-aware preprocessing
            processed_image = cls.adaptive_preprocessing(image, metrics, is_high_quality)
            
            logger.info("Image preprocessing completed successfully")
            return processed_image, metrics, is_high_quality
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class OCREngine:
    """
    Enhanced OCR engine with quality-aware configuration selection.
    """
    
    @staticmethod
    def get_optimal_config(is_high_quality, metrics):
        """
        Select optimal OCR configuration based on image quality.
        
        Args:
            is_high_quality: Boolean indicating image quality
            metrics: Quality metrics dictionary
            
        Returns:
            List of OCR configurations to try
        """
        if is_high_quality:
            # For high quality images, use configurations that preserve detail
            configs = [
                '--oem 3 --psm 6 -c preserve_interword_spaces=1',  # Default with word spacing
                '--oem 3 --psm 4 -c preserve_interword_spaces=1',  # Single column
                '--oem 3 --psm 3 -c preserve_interword_spaces=1',  # Fully automatic
                '--oem 1 --psm 6',  # Neural network engine
                '--oem 3 --psm 6',  # Standard config
            ]
        else:
            # For lower quality images, use more aggressive segmentation
            configs = [
                '--oem 3 --psm 6',  # Standard
                '--oem 3 --psm 8',  # Single word
                '--oem 3 --psm 7',  # Single text line
                '--oem 3 --psm 4',  # Single column
                '--oem 3 --psm 3',  # Fully automatic
            ]
        
        return configs
    
    @staticmethod
    def extract_text_with_confidence(image, config):
        """
        Extract text with confidence scores and detailed analysis.
        
        Args:
            image: Processed image
            config: Tesseract configuration string
            
        Returns:
            Tuple of (text, confidence, word_count, line_count)
        """
        try:
            # Extract text
            text = tesseract.image_to_string(image, config=config).strip()
            
            # Get detailed OCR data for confidence calculation
            ocr_data = tesseract.image_to_data(image, config=config, output_type=tesseract.Output.DICT)
            
            # Calculate weighted confidence based on word lengths
            confidences = []
            word_lengths = []
            
            for i, conf in enumerate(ocr_data['conf']):
                if int(conf) > 0 and ocr_data['text'][i].strip():
                    word_length = len(ocr_data['text'][i].strip())
                    confidences.append(int(conf))
                    word_lengths.append(word_length)
            
            if confidences:
                # Weight confidence by word length (longer words are more reliable)
                weighted_conf = sum(c * w for c, w in zip(confidences, word_lengths))
                total_weight = sum(word_lengths)
                avg_confidence = weighted_conf / total_weight if total_weight > 0 else 0
            else:
                avg_confidence = 0
            
            # Count words and lines
            word_count = len([word for word in text.split() if word.strip()])
            line_count = len([line for line in text.split('\n') if line.strip()])
            
            return text, avg_confidence, word_count, line_count
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return "", 0, 0, 0
    
    @staticmethod
    def multi_config_ocr(image, is_high_quality, metrics):
        """
        Try multiple OCR configurations and return the best result.
        
        Args:
            image: Processed image
            is_high_quality: Boolean indicating image quality
            metrics: Quality metrics dictionary
            
        Returns:
            Best OCR result
        """
        configs = OCREngine.get_optimal_config(is_high_quality, metrics)
        
        best_result = ("", 0, 0, 0)
        results = []
        
        for i, config in enumerate(configs):
            try:
                text, confidence, word_count, line_count = OCREngine.extract_text_with_confidence(image, config)
                
                # Calculate result score (combination of confidence and content)
                content_score = min(word_count * 2, 100)  # Bonus for more words
                total_score = (confidence * 0.7) + (content_score * 0.3)
                
                results.append({
                    'text': text,
                    'confidence': confidence,
                    'word_count': word_count,
                    'line_count': line_count,
                    'config': config,
                    'score': total_score
                })
                
                logger.info(f"Config {i+1}: conf={confidence:.1f}, words={word_count}, score={total_score:.1f}")
                
                if total_score > best_result[1]:
                    best_result = (text, confidence, word_count, line_count)
                    
            except Exception as e:
                logger.warning(f"Config {config} failed: {str(e)}")
                continue
        
        # Log the best result
        logger.info(f"Best result: confidence={best_result[1]:.1f}, words={best_result[2]}")
        
        return best_result


# Flask routes
@app.context_processor
def inject_year():
    """Inject current year into templates."""
    return {'year': datetime.now().year}


@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')


@app.route('/ocr', methods=['POST'])
def ocr():
    """
    Main OCR processing route with enhanced quality detection.
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        if not ImagePreprocessor.is_allowed_file(file.filename):
            flash('Invalid file type. Please upload an image file.', 'error')
            return redirect(url_for('index'))
        
        # Get enhancement level from form
        enhancement_level = request.form.get('enhancement_level', 'auto')
        
        logger.info(f"Processing file: {file.filename} with enhancement: {enhancement_level}")
        
        # Read and preprocess image
        image_data = file.read()
        processed_image, metrics, is_high_quality = ImagePreprocessor.preprocess_image(image_data, enhancement_level)
        
        # Extract text using quality-aware OCR
        text, confidence, word_count, line_count = OCREngine.multi_config_ocr(processed_image, is_high_quality, metrics)
        
        # Prepare result data
        result_data = {
            'extracted_text': text,
            'confidence': round(confidence, 2),
            'word_count': word_count,
            'line_count': line_count,
            'filename': file.filename,
            'enhancement_level': enhancement_level,
            'is_high_quality': is_high_quality,
            'quality_metrics': {k: round(v, 2) for k, v in metrics.items()},
            'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"OCR completed - Confidence: {confidence:.1f}%, Words: {word_count}, Quality: {'High' if is_high_quality else 'Low'}")
        
        return render_template('result.html', **result_data)
        
    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    """
    Enhanced API endpoint for OCR processing.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not ImagePreprocessor.is_allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        enhancement_level = request.form.get('enhancement_level', 'auto')
        
        # Process image
        image_data = file.read()
        processed_image, metrics, is_high_quality = ImagePreprocessor.preprocess_image(image_data, enhancement_level)
        
        # Extract text
        text, confidence, word_count, line_count = OCREngine.multi_config_ocr(processed_image, is_high_quality, metrics)
        
        return jsonify({
            'success': True,
            'text': text,
            'confidence': round(confidence, 2),
            'word_count': word_count,
            'line_count': line_count,
            'filename': file.filename,
            'enhancement_level': enhancement_level,
            'is_high_quality': is_high_quality,
            'quality_metrics': {k: round(v, 2) for k, v in metrics.items()},
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"API OCR error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    flash('An unexpected error occurred. Please try again.', 'error')
    return redirect(url_for('index'))


if __name__ == '__main__':
    # Run the application
    logger.info("Starting Enhanced OCR Application with Quality Detection")
    app.run(debug=True, host='0.0.0.0', port=5000)