"""
Simplified image preprocessing module for the OCR system.

This module contains essential image enhancement functions to optimize images for OCR accuracy.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageEnhance

from config import Config
from utils import Timer

logger = logging.getLogger(__name__)

class ImagePreprocessingError(Exception):
    """Custom exception for image preprocessing errors."""
    pass

class ImagePreprocessor:
    """
    Simplified image preprocessor with essential enhancement techniques.
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
    
    def resize_image_optimal(self, image: np.ndarray, target_height: int = 1200) -> np.ndarray:
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
        if height > target_height * 1.2:
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
    
    def reduce_noise_basic(self, image: np.ndarray) -> np.ndarray:
        """
        Apply basic noise reduction.
        
        Args:
            image: Input image array
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            # Color image denoising
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
        else:
            # Grayscale image denoising
            denoised = cv2.fastNlMeansDenoising(image, None, 6, 7, 21)
        
        logger.debug("Applied basic noise reduction")
        return denoised
    
    def correct_skew_simple(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Simple skew detection and correction.
        
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
            
            # Edge detection
            edges = cv2.Canny(gray, 30, 100, apertureSize=3)
            
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
                    final_angle = np.median(angles)
                    
                    # Only correct if angle is significant
                    if abs(final_angle) > 0.5:
                        corrected = self._rotate_image(image, final_angle)
                        logger.info(f"Skew corrected by {final_angle:.2f} degrees")
                        return corrected, final_angle
            
            logger.debug("No significant skew detected")
            return image, 0.0
                
        except Exception as e:
            logger.warning(f"Skew correction failed: {str(e)}")
            return image, 0.0
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def adaptive_threshold(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for better text extraction.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Binary image
        """
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        logger.debug("Applied adaptive thresholding")
        return cleaned
    
    def preprocess_pipeline(self, image_input: Union[str, bytes, np.ndarray], 
                           quality_metrics: Dict[str, float],
                           enhancement_level: str = 'auto') -> Dict[str, any]:
        """
        Complete preprocessing pipeline optimized for OCR.
        
        Args:
            image_input: Input image
            quality_metrics: Image quality metrics
            enhancement_level: Enhancement level
            
        Returns:
            Preprocessing results dictionary
        """
        with Timer("Image preprocessing"):
            original_shape = None
            processing_steps = []
            
            try:
                # Load image
                image = self.load_image(image_input)
                original_shape = image.shape
                processing_steps.append('load')
                
                # Resize if needed
                image = self.resize_image_optimal(image)
                if image.shape != original_shape:
                    processing_steps.append('resize')
                
                # Convert to grayscale for OCR
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                processing_steps.append('grayscale')
                
                # Apply enhancement based on quality
                overall_quality = quality_metrics.get('overall_quality', 50)
                
                # Convert string quality to numeric if needed
                if isinstance(overall_quality, str):
                    quality_mapping = {
                        'excellent': 90,
                        'good': 70,
                        'fair': 50,
                        'poor': 30
                    }
                    overall_quality = quality_mapping.get(overall_quality, 50)
                
                if overall_quality < 70:
                    # Apply noise reduction for lower quality images
                    gray = self.reduce_noise_basic(gray)
                    processing_steps.append('noise_reduction')
                
                if overall_quality < 80:
                    # Apply contrast enhancement for lower quality images
                    enhanced = self.enhance_contrast_adaptive(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                    processing_steps.append('contrast_enhancement')
                
                # Correct skew if significant
                corrected, skew_angle = self.correct_skew_simple(gray)
                if abs(skew_angle) > 0.5:
                    processing_steps.append('skew_correction')
                
                # Final processing - use adaptive threshold for better text extraction
                final_image = self.adaptive_threshold(corrected)
                processing_steps.append('thresholding')
                
                # Determine if result is binary
                is_binary = len(np.unique(final_image)) == 2
                
                result = {
                    'processed_image': final_image,
                    'original_shape': original_shape,
                    'final_shape': final_image.shape,
                    'processing_steps': processing_steps,
                    'enhancement_level': enhancement_level,
                    'is_binary': is_binary,
                    'skew_angle': skew_angle,
                    'quality_improvement': self._assess_quality_improvement(original_shape, final_image.shape, processing_steps)
                }
                
                logger.info(f"Preprocessing completed: {len(processing_steps)} steps applied")
                return result
                
            except Exception as e:
                logger.error(f"Preprocessing failed: {str(e)}")
                raise ImagePreprocessingError(f"Preprocessing failed: {str(e)}")
    
    def _assess_quality_improvement(self, original_shape: Tuple, final_shape: Tuple, steps: List[str]) -> str:
        """Assess the quality improvement from preprocessing."""
        if 'resize' in steps:
            return 'optimized_size'
        elif 'contrast_enhancement' in steps or 'noise_reduction' in steps:
            return 'enhanced_quality'
        elif 'skew_correction' in steps:
            return 'corrected_orientation'
        else:
            return 'minimal_processing'

def preprocess_for_ocr(image_input: Union[str, bytes, np.ndarray],
                       quality_metrics: Dict[str, float],
                       enhancement_level: str = 'auto') -> Dict[str, any]:
    """
    Convenience function for preprocessing images for OCR.
    
    Args:
        image_input: Input image
        quality_metrics: Image quality metrics
        enhancement_level: Enhancement level
        
    Returns:
        Preprocessing results
    """
    preprocessor = ImagePreprocessor()
    return preprocessor.preprocess_pipeline(image_input, quality_metrics, enhancement_level)