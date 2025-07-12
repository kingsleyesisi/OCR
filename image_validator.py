"""
Image validation module for the OCR system.

This module handles all image detection, validation, and quality assessment
before processing. It ensures images meet the requirements for optimal OCR.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ExifTags
import magic
import os

from config import Config
from utils import Timer, format_file_size, bytes_to_mb

logger = logging.getLogger(__name__)

class ImageValidationError(Exception):
    """Custom exception for image validation errors."""
    pass

class ImageValidator:
    """
    Comprehensive image validator for OCR preprocessing.
    
    This class handles image existence verification, format validation,
    dimension checks, and quality assessment.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the image validator.
        
        Args:
            config: Configuration object with validation parameters
        """
        self.config = config or Config()
        self.supported_formats = self.config.ALLOWED_EXTENSIONS
        self.max_dimension = self.config.MAX_IMAGE_DIMENSION
        self.min_dimension = self.config.MIN_IMAGE_DIMENSION
        
    def validate_file_existence(self, file_path: Union[str, Path]) -> bool:
        """
        Verify that the image file exists and is accessible.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            True if file exists and is accessible
            
        Raises:
            ImageValidationError: If file doesn't exist or isn't accessible
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ImageValidationError(f"Image file does not exist: {file_path}")
        
        if not file_path.is_file():
            raise ImageValidationError(f"Path is not a file: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise ImageValidationError(f"File is not readable: {file_path}")
        
        logger.debug(f"File existence validated: {file_path}")
        return True
    
    def validate_file_format(self, file_path: Union[str, Path]) -> str:
        """
        Validate the image file format using multiple methods.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Detected file format
            
        Raises:
            ImageValidationError: If format is not supported
        """
        file_path = Path(file_path)
        
        # Method 1: Check file extension
        file_extension = file_path.suffix.lower().lstrip('.')
        
        # Method 2: Use python-magic for MIME type detection
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
            logger.debug(f"Detected MIME type: {mime_type}")
        except Exception as e:
            logger.warning(f"Could not detect MIME type: {str(e)}")
            mime_type = None
        
        # Method 3: Try to open with PIL
        try:
            with Image.open(file_path) as img:
                pil_format = img.format.lower() if img.format else None
                logger.debug(f"PIL detected format: {pil_format}")
        except Exception as e:
            raise ImageValidationError(f"Cannot open image with PIL: {str(e)}")
        
        # Validate against supported formats
        supported_extensions = {ext.lower() for ext in self.supported_formats}
        
        if file_extension not in supported_extensions:
            raise ImageValidationError(
                f"Unsupported file extension: {file_extension}. "
                f"Supported formats: {', '.join(supported_extensions)}"
            )
        
        # Additional MIME type validation
        valid_mime_types = {
            'image/jpeg', 'image/jpg', 'image/png', 'image/gif',
            'image/bmp', 'image/tiff', 'image/webp'
        }
        
        if mime_type and mime_type not in valid_mime_types:
            logger.warning(f"Unexpected MIME type: {mime_type}")
        
        logger.info(f"File format validated: {file_extension}")
        return file_extension
    
    def validate_file_size(self, file_path: Union[str, Path]) -> Dict[str, Union[int, str]]:
        """
        Validate file size and provide size information.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with size information
            
        Raises:
            ImageValidationError: If file is too large
        """
        file_path = Path(file_path)
        file_size = file_path.stat().st_size
        
        if file_size > self.config.MAX_CONTENT_LENGTH:
            raise ImageValidationError(
                f"File too large: {format_file_size(file_size)}. "
                f"Maximum allowed: {format_file_size(self.config.MAX_CONTENT_LENGTH)}"
            )
        
        size_info = {
            'size_bytes': file_size,
            'size_mb': bytes_to_mb(file_size),
            'size_formatted': format_file_size(file_size)
        }
        
        logger.debug(f"File size validated: {size_info['size_formatted']}")
        return size_info
    
    def validate_image_dimensions(self, file_path: Union[str, Path]) -> Dict[str, int]:
        """
        Validate image dimensions and aspect ratio.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with dimension information
            
        Raises:
            ImageValidationError: If dimensions are invalid
        """
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                
                # Check minimum dimensions
                if width < self.min_dimension or height < self.min_dimension:
                    raise ImageValidationError(
                        f"Image too small: {width}x{height}. "
                        f"Minimum dimension: {self.min_dimension}px"
                    )
                
                # Check maximum dimensions
                if width > self.max_dimension or height > self.max_dimension:
                    logger.warning(
                        f"Large image detected: {width}x{height}. "
                        f"May be resized for processing."
                    )
                
                # Calculate aspect ratio
                aspect_ratio = width / height
                
                dimension_info = {
                    'width': width,
                    'height': height,
                    'aspect_ratio': round(aspect_ratio, 2),
                    'total_pixels': width * height
                }
                
                logger.debug(f"Image dimensions validated: {width}x{height}")
                return dimension_info
                
        except Exception as e:
            raise ImageValidationError(f"Cannot read image dimensions: {str(e)}")
    
    def assess_image_quality(self, file_path: Union[str, Path]) -> Dict[str, float]:
        """
        Assess image quality metrics for OCR suitability.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Load image with OpenCV
            image = cv2.imread(str(file_path))
            if image is None:
                raise ImageValidationError("Cannot load image with OpenCV")
            
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(gray)
            
            # Assess overall quality
            quality_metrics['overall_quality'] = self._assess_overall_quality(quality_metrics)
            
            logger.debug(f"Image quality assessed: {quality_metrics}")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_quality_metrics(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Calculate detailed quality metrics for the image.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        metrics['sharpness'] = float(laplacian.var())
        
        # Contrast (standard deviation)
        metrics['contrast'] = float(gray_image.std())
        
        # Brightness (mean intensity)
        metrics['brightness'] = float(gray_image.mean())
        
        # Noise estimation
        median_filtered = cv2.medianBlur(gray_image, 5)
        noise = np.mean(np.abs(gray_image.astype(np.float32) - median_filtered.astype(np.float32)))
        metrics['noise_level'] = float(noise)
        
        # Edge density (text-like features)
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / (gray_image.shape[0] * gray_image.shape[1])
        metrics['edge_density'] = float(edge_density)
        
        # Dynamic range
        metrics['dynamic_range'] = float(gray_image.max() - gray_image.min())
        
        # Histogram analysis
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist_normalized = hist.flatten() / hist.sum()
        
        # Entropy (information content)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        metrics['entropy'] = float(entropy)
        
        return metrics
    
    def _assess_overall_quality(self, metrics: Dict[str, float]) -> str:
        """
        Assess overall image quality based on individual metrics.
        
        Args:
            metrics: Dictionary of quality metrics
            
        Returns:
            Overall quality assessment string
        """
        score = 0
        max_score = 7
        
        # Sharpness check
        if metrics.get('sharpness', 0) > 100:
            score += 1
        
        # Contrast check
        if metrics.get('contrast', 0) > 50:
            score += 1
        
        # Brightness check (not too dark or too bright)
        brightness = metrics.get('brightness', 0)
        if 50 < brightness < 200:
            score += 1
        
        # Noise check (lower is better)
        if metrics.get('noise_level', float('inf')) < 10:
            score += 1
        
        # Edge density check
        if metrics.get('edge_density', 0) > 0.01:
            score += 1
        
        # Dynamic range check
        if metrics.get('dynamic_range', 0) > 100:
            score += 1
        
        # Entropy check (information content)
        if metrics.get('entropy', 0) > 6:
            score += 1
        
        # Determine quality level
        quality_ratio = score / max_score
        
        if quality_ratio >= 0.8:
            return 'excellent'
        elif quality_ratio >= 0.6:
            return 'good'
        elif quality_ratio >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def get_image_metadata(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """
        Extract comprehensive image metadata.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with image metadata
        """
        metadata = {}
        
        try:
            with Image.open(file_path) as img:
                # Basic information
                metadata['format'] = img.format
                metadata['mode'] = img.mode
                metadata['size'] = img.size
                
                # EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = {}
                    exif = img._getexif()
                    
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_data[tag] = value
                    
                    metadata['exif'] = exif_data
                
                # DPI information
                if 'dpi' in img.info:
                    metadata['dpi'] = img.info['dpi']
                
        except Exception as e:
            logger.warning(f"Could not extract metadata: {str(e)}")
            metadata['error'] = str(e)
        
        return metadata
    
    def validate_image(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """
        Perform comprehensive image validation.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Complete validation report
            
        Raises:
            ImageValidationError: If validation fails
        """
        with Timer("Image validation"):
            validation_report = {
                'file_path': str(file_path),
                'timestamp': Timer().start_time,
                'validation_status': 'pending'
            }
            
            try:
                # Step 1: File existence
                self.validate_file_existence(file_path)
                validation_report['file_exists'] = True
                
                # Step 2: File format
                file_format = self.validate_file_format(file_path)
                validation_report['file_format'] = file_format
                
                # Step 3: File size
                size_info = self.validate_file_size(file_path)
                validation_report['size_info'] = size_info
                
                # Step 4: Image dimensions
                dimension_info = self.validate_image_dimensions(file_path)
                validation_report['dimensions'] = dimension_info
                
                # Step 5: Quality assessment
                quality_metrics = self.assess_image_quality(file_path)
                validation_report['quality_metrics'] = quality_metrics
                
                # Step 6: Metadata extraction
                metadata = self.get_image_metadata(file_path)
                validation_report['metadata'] = metadata
                
                # Final validation status
                validation_report['validation_status'] = 'passed'
                validation_report['is_suitable_for_ocr'] = self._is_suitable_for_ocr(quality_metrics)
                
                logger.info(f"Image validation completed successfully: {file_path}")
                return validation_report
                
            except ImageValidationError as e:
                validation_report['validation_status'] = 'failed'
                validation_report['error'] = str(e)
                logger.error(f"Image validation failed: {str(e)}")
                raise
            
            except Exception as e:
                validation_report['validation_status'] = 'error'
                validation_report['error'] = str(e)
                logger.error(f"Unexpected error during validation: {str(e)}")
                raise ImageValidationError(f"Validation error: {str(e)}")
    
    def _is_suitable_for_ocr(self, quality_metrics: Dict[str, any]) -> bool:
        """
        Determine if image is suitable for OCR based on quality metrics.
        
        Args:
            quality_metrics: Quality assessment results
            
        Returns:
            True if image is suitable for OCR
        """
        if 'error' in quality_metrics:
            return False
        
        overall_quality = quality_metrics.get('overall_quality', 'poor')
        return overall_quality in ['excellent', 'good', 'fair']

def validate_uploaded_file(file_data: bytes, filename: str) -> Dict[str, any]:
    """
    Validate uploaded file data without saving to disk.
    
    Args:
        file_data: Raw file data
        filename: Original filename
        
    Returns:
        Validation report
    """
    import tempfile
    import os
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
        temp_file.write(file_data)
        temp_path = temp_file.name
    
    try:
        validator = ImageValidator()
        report = validator.validate_image(temp_path)
        report['original_filename'] = filename
        return report
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Could not delete temporary file: {str(e)}")