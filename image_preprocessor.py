"""
Image preprocessing module for the OCR system.

This module contains advanced image enhancement and preprocessing functions
to optimize images for OCR accuracy.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
import skimage
from skimage import restoration, filters, morphology, measure

from config import Config
from utils import Timer

logger = logging.getLogger(__name__)

class ImagePreprocessingError(Exception):
    """Custom exception for image preprocessing errors."""
    pass

class ImagePreprocessor:
    """
    Advanced image preprocessor with quality-aware enhancement techniques.
    
    This class provides comprehensive image preprocessing capabilities
    including noise reduction, contrast enhancement, skew correction,
    and adaptive thresholding.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the image preprocessor.
        
        Args:
            config: Configuration object with preprocessing parameters
        """
        self.config = config or Config()
        self.target_dpi = self.config.TARGET_DPI
        self.max_dimension = self.config.MAX_IMAGE_DIMENSION
    
    def load_image(self, image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """
        Load image from various input types.
        
        Args:
            image_input: Image file path, bytes data, or numpy array
            
        Returns:
            OpenCV image array
            
        Raises:
            ImagePreprocessingError: If image cannot be loaded
        """
        try:
            if isinstance(image_input, str):
                # Load from file path
                image = cv2.imread(image_input)
                if image is None:
                    raise ImagePreprocessingError(f"Cannot load image from path: {image_input}")
            
            elif isinstance(image_input, bytes):
                # Load from bytes data
                np_arr = np.frombuffer(image_input, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ImagePreprocessingError("Cannot decode image from bytes")
            
            elif isinstance(image_input, np.ndarray):
                # Already a numpy array
                image = image_input.copy()
            
            else:
                raise ImagePreprocessingError(f"Unsupported image input type: {type(image_input)}")
            
            logger.debug(f"Image loaded successfully, shape: {image.shape}")
            return image
            
        except Exception as e:
            raise ImagePreprocessingError(f"Failed to load image: {str(e)}")
    
    def resize_image_optimal(self, image: np.ndarray, target_height: int = 1000) -> np.ndarray:
        """
        Resize image to optimal dimensions for OCR while preserving aspect ratio.
        
        Args:
            image: Input image array
            target_height: Target height in pixels
            
        Returns:
            Resized image array
        """
        height, width = image.shape[:2]
        
        # Only resize if significantly larger than target
        if height > target_height * 1.5:
            scale_factor = target_height / height
            new_width = int(width * scale_factor)
            new_height = target_height
            
            # Use appropriate interpolation method
            if scale_factor < 1:
                interpolation = cv2.INTER_AREA  # Better for downscaling
            else:
                interpolation = cv2.INTER_CUBIC  # Better for upscaling
            
            resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
            logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
            return resized
        
        return image
    
    def enhance_contrast_adaptive(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive contrast enhancement using CLAHE.
        
        Args:
            image: Input image array
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Merge channels back
            enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        logger.debug("Applied adaptive contrast enhancement (CLAHE)")
        return enhanced
    
    def reduce_noise_advanced(self, image: np.ndarray, noise_level: float = 10.0) -> np.ndarray:
        """
        Apply advanced noise reduction techniques.
        
        Args:
            image: Input image array
            noise_level: Estimated noise level (0-100)
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            # Color image denoising
            if noise_level > 15:
                # Heavy denoising for very noisy images
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                # Light denoising for moderately noisy images
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
        else:
            # Grayscale image denoising
            if noise_level > 15:
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, 6, 7, 21)
        
        logger.debug(f"Applied noise reduction for noise level: {noise_level}")
        return denoised
    
    def correct_skew_robust(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Robust skew detection and correction using multiple methods.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (corrected_image, skew_angle)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Method 1: Hough Line Transform
            angle_hough = self._detect_skew_hough(gray)
            
            # Method 2: Projection Profile
            angle_projection = self._detect_skew_projection(gray)
            
            # Method 3: Radon Transform (if available)
            try:
                angle_radon = self._detect_skew_radon(gray)
            except Exception:
                angle_radon = None
            
            # Combine results and select best angle
            angles = [angle for angle in [angle_hough, angle_projection, angle_radon] if angle is not None]
            
            if not angles:
                logger.warning("No skew angle detected")
                return image, 0.0
            
            # Use median of detected angles
            final_angle = np.median(angles)
            
            # Only correct if angle is significant
            if abs(final_angle) > 0.5:
                corrected = self._rotate_image(image, final_angle)
                logger.info(f"Skew corrected by {final_angle:.2f} degrees")
                return corrected, final_angle
            else:
                logger.debug("Skew angle too small, no correction needed")
                return image, 0.0
                
        except Exception as e:
            logger.warning(f"Skew correction failed: {str(e)}")
            return image, 0.0
    
    def _detect_skew_hough(self, gray_image: np.ndarray) -> Optional[float]:
        """Detect skew using Hough Line Transform."""
        try:
            # Edge detection
            edges = cv2.Canny(gray_image, 30, 100, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 5:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta) - 90
                    if abs(angle) < 45:  # Filter out vertical lines
                        angles.append(angle)
                
                if angles:
                    return np.median(angles)
            
            return None
        except Exception:
            return None
    
    def _detect_skew_projection(self, gray_image: np.ndarray) -> Optional[float]:
        """Detect skew using projection profile method."""
        try:
            # Binary threshold
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Test angles from -10 to +10 degrees
            angles = np.arange(-10, 11, 0.5)
            scores = []
            
            for angle in angles:
                rotated = self._rotate_image(binary, angle)
                # Calculate horizontal projection variance
                projection = np.sum(rotated, axis=1)
                score = np.var(projection)
                scores.append(score)
            
            # Find angle with maximum variance
            best_angle = angles[np.argmax(scores)]
            return best_angle
            
        except Exception:
            return None
    
    def _detect_skew_radon(self, gray_image: np.ndarray) -> Optional[float]:
        """Detect skew using Radon transform."""
        try:
            # Binary threshold
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Radon transform
            angles = np.arange(-10, 11, 0.5)
            sinogram = skimage.transform.radon(binary, theta=angles, circle=False)
            
            # Find angle with maximum variance in projection
            variances = np.var(sinogram, axis=0)
            best_angle_idx = np.argmax(variances)
            best_angle = angles[best_angle_idx]
            
            return best_angle
            
        except Exception:
            return None
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up the image.
        
        Args:
            image: Input binary image
            
        Returns:
            Cleaned image
        """
        # Define kernels for different operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Remove small noise
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_small)
        
        # Fill small gaps
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        logger.debug("Applied morphological operations")
        return cleaned
    
    def adaptive_threshold_multi(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Apply multiple adaptive thresholding methods and select the best result.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            Best thresholded image
        """
        methods = [
            # (method, block_size, C)
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 3),
            (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2),
            (cv2.ADAPTIVE_THRESH_MEAN_C, 15, 3),
        ]
        
        results = []
        scores = []
        
        for method, block_size, c in methods:
            try:
                thresh = cv2.adaptiveThreshold(gray_image, 255, method, 
                                             cv2.THRESH_BINARY, block_size, c)
                
                # Score based on edge preservation and noise reduction
                edges = cv2.Canny(thresh, 50, 150)
                edge_score = np.sum(edges > 0)
                
                # Penalize excessive noise (too many small components)
                num_labels, labels = cv2.connectedComponents(thresh)
                noise_penalty = max(0, num_labels - 1000) * 0.1
                
                final_score = edge_score - noise_penalty
                
                results.append(thresh)
                scores.append(final_score)
                
            except Exception as e:
                logger.warning(f"Adaptive threshold method failed: {str(e)}")
                continue
        
        if results:
            best_idx = np.argmax(scores)
            best_result = results[best_idx]
            logger.debug(f"Selected adaptive threshold method {best_idx} with score {scores[best_idx]}")
            return best_result
        else:
            # Fallback to simple threshold
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            logger.warning("All adaptive methods failed, using Otsu threshold")
            return thresh
    
    def enhance_text_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance text regions specifically for better OCR.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image with improved text regions
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply unsharp masking for text enhancement
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        # Enhance local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(unsharp_mask)
        
        logger.debug("Applied text region enhancement")
        return enhanced
    
    def preprocess_pipeline(self, image_input: Union[str, bytes, np.ndarray], 
                          quality_metrics: Dict[str, float],
                          enhancement_level: str = 'auto') -> Dict[str, any]:
        """
        Complete preprocessing pipeline with quality-aware processing.
        
        Args:
            image_input: Input image (path, bytes, or array)
            quality_metrics: Image quality assessment results
            enhancement_level: Level of enhancement to apply
            
        Returns:
            Dictionary with processed image and processing information
        """
        with Timer("Image preprocessing pipeline"):
            try:
                # Load image
                original_image = self.load_image(image_input)
                current_image = original_image.copy()
                
                processing_steps = []
                
                # Step 1: Resize if needed
                resized_image = self.resize_image_optimal(current_image)
                if not np.array_equal(resized_image, current_image):
                    processing_steps.append("resized")
                    current_image = resized_image
                
                # Determine processing intensity based on quality and enhancement level
                is_high_quality = quality_metrics.get('overall_quality', 'poor') in ['excellent', 'good']
                
                if enhancement_level == 'auto':
                    if is_high_quality:
                        enhancement_level = 'light'
                    else:
                        enhancement_level = 'medium'
                
                # Step 2: Noise reduction (if needed)
                noise_level = quality_metrics.get('noise_level', 0)
                if noise_level > 8 or enhancement_level in ['medium', 'heavy']:
                    current_image = self.reduce_noise_advanced(current_image, noise_level)
                    processing_steps.append("noise_reduction")
                
                # Step 3: Contrast enhancement
                if (quality_metrics.get('contrast', 0) < 50 or 
                    enhancement_level in ['medium', 'heavy']):
                    current_image = self.enhance_contrast_adaptive(current_image)
                    processing_steps.append("contrast_enhancement")
                
                # Step 4: Skew correction
                if enhancement_level in ['medium', 'heavy']:
                    current_image, skew_angle = self.correct_skew_robust(current_image)
                    if abs(skew_angle) > 0.5:
                        processing_steps.append(f"skew_correction_{skew_angle:.1f}deg")
                
                # Step 5: Convert to grayscale
                if len(current_image.shape) == 3:
                    gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = current_image
                processing_steps.append("grayscale_conversion")
                
                # Step 6: Text enhancement
                if enhancement_level in ['medium', 'heavy']:
                    gray_image = self.enhance_text_regions(gray_image)
                    processing_steps.append("text_enhancement")
                
                # Step 7: Adaptive thresholding (for poor quality images)
                if not is_high_quality or enhancement_level == 'heavy':
                    binary_image = self.adaptive_threshold_multi(gray_image)
                    processing_steps.append("adaptive_thresholding")
                    
                    # Step 8: Morphological operations
                    binary_image = self.apply_morphological_operations(binary_image)
                    processing_steps.append("morphological_operations")
                    
                    final_image = binary_image
                else:
                    final_image = gray_image
                
                # Prepare result
                result = {
                    'processed_image': final_image,
                    'original_shape': original_image.shape,
                    'final_shape': final_image.shape,
                    'processing_steps': processing_steps,
                    'enhancement_level': enhancement_level,
                    'is_binary': not is_high_quality or enhancement_level == 'heavy',
                    'quality_based_processing': True
                }
                
                logger.info(f"Preprocessing completed with {len(processing_steps)} steps: {', '.join(processing_steps)}")
                return result
                
            except Exception as e:
                logger.error(f"Preprocessing pipeline failed: {str(e)}")
                raise ImagePreprocessingError(f"Preprocessing failed: {str(e)}")

def preprocess_for_ocr(image_input: Union[str, bytes, np.ndarray],
                      quality_metrics: Dict[str, float],
                      enhancement_level: str = 'auto') -> Dict[str, any]:
    """
    Convenience function for preprocessing images for OCR.
    
    Args:
        image_input: Input image (path, bytes, or array)
        quality_metrics: Image quality assessment results
        enhancement_level: Level of enhancement to apply
        
    Returns:
        Preprocessing results
    """
    preprocessor = ImagePreprocessor()
    return preprocessor.preprocess_pipeline(image_input, quality_metrics, enhancement_level)