"""
Enhanced image preprocessing module for the OCR system with Google Lens-like capabilities.

This module contains advanced image enhancement functions to optimize images for OCR accuracy,
including text detection, perspective correction, and intelligent preprocessing.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import math

from config import Config
from utils import Timer

logger = logging.getLogger(__name__)

class ImagePreprocessingError(Exception):
    """Custom exception for image preprocessing errors."""
    pass

class ImagePreprocessor:
    """
    Enhanced image preprocessor with Google Lens-like capabilities.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the image preprocessor.
        
        Args:
            config: Configuration object with preprocessing parameters
        """
        self.config = config if config is not None else Config()
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
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions in the image using MSER and contour detection.
        
        Args:
            image: Input image array
            
        Returns:
            List of bounding boxes (x, y, width, height)
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            # Apply MSER (Maximally Stable Extremal Regions)
            mser = cv2.MSER_create()
            mser.setMinArea(100)
            mser.setMaxArea(2000)
            mser.setDelta(5)
            
            regions, _ = mser.detectRegions(gray)
            
            # Convert regions to bounding boxes
            text_regions = []
            for region in regions:
                x, y, w, h = cv2.boundingRect(region)
                # Filter by aspect ratio and size
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10 and w > 20 and h > 10:
                    text_regions.append((x, y, w, h))
            
            # Merge overlapping regions
            text_regions = self._merge_overlapping_boxes(text_regions)
            
            logger.debug(f"Detected {len(text_regions)} text regions")
            return text_regions
            
        except Exception as e:
            logger.warning(f"Text region detection failed: {str(e)}")
            return []
    
    def _merge_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes."""
        if not boxes:
            return []
        
        # Sort boxes by x coordinate
        boxes = sorted(boxes, key=lambda x: x[0])
        
        merged = []
        current = list(boxes[0])
        
        for box in boxes[1:]:
            # Check if boxes overlap or are close
            if (current[0] + current[2] >= box[0] - 10 and 
                current[1] + current[3] >= box[1] - 10):
                # Merge boxes
                x1 = min(current[0], box[0])
                y1 = min(current[1], box[1])
                x2 = max(current[0] + current[2], box[0] + box[2])
                y2 = max(current[1] + current[3], box[1] + box[3])
                current = [x1, y1, x2 - x1, y2 - y1]
            else:
                merged.append(tuple(current))
                current = list(box)
        
        merged.append(tuple(current))
        return merged
    
    def correct_perspective(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Correct perspective distortion using text region detection.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (corrected_image, was_corrected)
        """
        try:
            # Detect text regions
            text_regions = self.detect_text_regions(image)
            
            if len(text_regions) < 3:
                return image, False
            
            # Find the largest text region for perspective correction
            largest_region = max(text_regions, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_region
            
            # Extract the region
            region = image[y:y+h, x:x+w]
            
            # Convert to grayscale for edge detection
            if len(region.shape) == 3:
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray_region = region.copy()
            
            # Edge detection
            edges = cv2.Canny(gray_region, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image, False
            
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # If we have a quadrilateral, correct perspective
            if len(approx) == 4:
                # Sort points in order: top-left, top-right, bottom-right, bottom-left
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                
                # Top-left point will have the smallest sum
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                
                # Top-right point will have the smallest difference
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                
                # Calculate new width and height
                widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
                widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                
                heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
                heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                
                # Create destination points
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype="float32")
                
                # Calculate perspective transform matrix
                M = cv2.getPerspectiveTransform(rect, dst)
                
                # Apply perspective transform to the entire image
                corrected = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
                
                logger.info("Perspective correction applied")
                return corrected, True
            
            return image, False
            
        except Exception as e:
            logger.warning(f"Perspective correction failed: {str(e)}")
            return image, False
    
    def enhance_for_text_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Apply specialized enhancement for better text detection.
        
        Args:
            image: Input image array
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to PIL Image for advanced processing
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Apply sharpening
            sharpened = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(sharpened)
            enhanced = enhancer.enhance(1.3)
            
            # Enhance brightness slightly
            brightness_enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = brightness_enhancer.enhance(1.1)
            
            # Convert back to OpenCV format
            enhanced_array = np.array(enhanced)
            if len(enhanced_array.shape) == 3:
                enhanced_array = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2BGR)
            
            logger.debug("Applied text-specific enhancement")
            return enhanced_array
            
        except Exception as e:
            logger.warning(f"Text enhancement failed: {str(e)}")
            return image
    
    def adaptive_binarization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive binarization for better text extraction.
        
        Args:
            image: Input image array
            
        Returns:
            Binary image
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            logger.debug("Applied adaptive binarization")
            return cleaned
            
        except Exception as e:
            logger.warning(f"Binarization failed: {str(e)}")
            return image
    
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
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Merge channels back
            enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        logger.debug("Applied adaptive contrast enhancement (CLAHE)")
        return enhanced
    
    def reduce_noise_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        Apply advanced noise reduction preserving text edges.
        
        Args:
            image: Input image array
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            # Color image denoising with edge preservation
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)
        else:
            # Grayscale image denoising
            denoised = cv2.fastNlMeansDenoising(image, None, 8, 7, 21)
        
        logger.debug("Applied advanced noise reduction")
        return denoised
    
    def correct_skew_advanced(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Advanced skew detection and correction using text regions.
        
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
            
            # Detect text regions
            text_regions = self.detect_text_regions(image)
            
            if len(text_regions) < 2:
                # Fallback to traditional method
                return self._correct_skew_traditional(gray)
            
            # Calculate angles from text regions
            angles = []
            for x, y, w, h in text_regions:
                if w > 50 and h > 20:  # Only consider significant text regions
                    # Extract region and find lines
                    region = gray[y:y+h, x:x+w]
                    edges = cv2.Canny(region, 30, 100, apertureSize=3)
                    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
                    
                    if lines is not None:
                        for line in lines:
                            rho, theta = line[0]
                            angle = np.degrees(theta) - 90
                            if abs(angle) < 45:  # Filter out vertical lines
                                angles.append(angle)
            
            if angles:
                # Use median angle for stability
                final_angle = float(np.median(angles))
                
                # Only correct if angle is significant
                if abs(final_angle) > 0.5:
                    corrected = self._rotate_image(image, final_angle)
                    logger.info(f"Advanced skew correction: {final_angle:.2f} degrees")
                    return corrected, final_angle
            
            logger.debug("No significant skew detected")
            return image, 0.0
                
        except Exception as e:
            logger.warning(f"Advanced skew correction failed: {str(e)}")
            return self._correct_skew_traditional(gray)
    
    def _correct_skew_traditional(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """Traditional skew correction using Hough lines."""
        try:
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
                    final_angle = float(np.median(angles))
                    
                    # Only correct if angle is significant
                    if abs(final_angle) > 0.5:
                        corrected = self._rotate_image(gray, final_angle)
                        logger.info(f"Traditional skew correction: {final_angle:.2f} degrees")
                        return corrected, final_angle
            
            return gray, 0.0
                
        except Exception as e:
            logger.warning(f"Traditional skew correction failed: {str(e)}")
            return gray, 0.0
    
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
    
    def preprocess_pipeline(self, image_input: Union[str, bytes, np.ndarray], 
                           quality_metrics: Dict[str, float],
                           enhancement_level: str = 'auto') -> Dict[str, any]:
        """
        Complete preprocessing pipeline optimized for OCR with Google Lens-like capabilities.
        
        Args:
            image_input: Input image
            quality_metrics: Image quality metrics
            enhancement_level: Enhancement level (auto, minimal, moderate, aggressive)
            
        Returns:
            Preprocessing results dictionary
        """
        with Timer("Image preprocessing pipeline"):
            try:
                # Load image
                image = self.load_image(image_input)
                original_shape = image.shape
                
                processing_steps = []
                is_binary = False
                
                # Step 1: Resize to optimal dimensions
                image = self.resize_image_optimal(image)
                if image.shape != original_shape:
                    processing_steps.append('resize')
                
                # Step 2: Detect and correct perspective
                image, perspective_corrected = self.correct_perspective(image)
                if perspective_corrected:
                    processing_steps.append('perspective_correction')
                
                # Step 3: Correct skew
                image, skew_angle = self.correct_skew_advanced(image)
                if abs(skew_angle) > 0.5:
                    processing_steps.append('skew_correction')
                
                # Step 4: Reduce noise
                image = self.reduce_noise_advanced(image)
                processing_steps.append('noise_reduction')
                
                # Step 5: Enhance contrast
                image = self.enhance_contrast_adaptive(image)
                processing_steps.append('contrast_enhancement')
                
                # Step 6: Apply text-specific enhancement
                image = self.enhance_for_text_detection(image)
                processing_steps.append('text_enhancement')
                
                # Step 7: Adaptive binarization for poor quality images
                overall_quality = float(quality_metrics.get('overall_quality', 50))
                if overall_quality < 60:
                    image = self.adaptive_binarization(image)
                    processing_steps.append('binarization')
                    is_binary = True
                
                # Determine enhancement level based on quality
                if enhancement_level == 'auto':
                    if overall_quality >= 80:
                        enhancement_level = 'minimal'
                    elif overall_quality >= 60:
                        enhancement_level = 'moderate'
                    else:
                        enhancement_level = 'aggressive'
                
                logger.info(f"Preprocessing completed: {len(processing_steps)} steps, "
                          f"enhancement: {enhancement_level}")
                
                return {
                    'processed_image': image,
                    'processing_steps': processing_steps,
                    'enhancement_level': enhancement_level,
                    'is_binary': is_binary,
                    'original_shape': original_shape,
                    'final_shape': image.shape,
                    'skew_angle': skew_angle,
                    'perspective_corrected': perspective_corrected
                }
                
            except Exception as e:
                logger.error(f"Preprocessing pipeline failed: {str(e)}")
                raise ImagePreprocessingError(f"Preprocessing failed: {str(e)}")
    
    def _assess_quality_improvement(self, original_shape: Tuple, final_shape: Tuple, steps: List[str]) -> str:
        """Assess the quality improvement from preprocessing."""
        if len(steps) <= 2:
            return 'minimal'
        elif len(steps) <= 4:
            return 'moderate'
        else:
            return 'significant'

# Convenience function
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