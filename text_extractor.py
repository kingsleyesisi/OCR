"""
Improved text extraction module for the OCR system.

This module provides optimized OCR with better accuracy and text selection capabilities.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
import numpy as np
import pytesseract as tesseract
from PIL import Image
import re
from dataclasses import dataclass
import time

from config import Config
from utils import Timer

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """Data class for OCR results with bounding boxes for text selection."""
    text: str
    confidence: float
    word_count: int
    line_count: int
    character_count: int
    processing_time: float
    config_used: str
    bounding_boxes: List[Dict[str, any]] = None
    word_confidences: List[float] = None

class TextExtractionError(Exception):
    """Custom exception for text extraction errors."""
    pass

class TextExtractor:
    """
    Optimized text extractor with improved accuracy and text selection support.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the text extractor.
        
        Args:
            config: Configuration object with OCR parameters
        """
        self.config = config or Config()
        self._setup_tesseract()
        
    def _setup_tesseract(self):
        """Set up Tesseract OCR engine."""
        try:
            tesseract.pytesseract.tesseract_cmd = self.config.TESSERACT_CMD
            
            # Test Tesseract installation
            version = tesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            
        except Exception as e:
            logger.error(f"Tesseract setup failed: {str(e)}")
            raise TextExtractionError(f"Tesseract not properly configured: {str(e)}")
    
    def get_optimal_configs(self, quality_level: str) -> List[str]:
        """
        Get optimal OCR configurations based on image quality.
        
        Args:
            quality_level: Image quality level (excellent, good, fair, poor)
            
        Returns:
            List of OCR configuration strings
        """
        # Simplified, more effective configurations
        base_configs = [
            '--oem 3 --psm 6 -c preserve_interword_spaces=1',  # Uniform block of text
            '--oem 3 --psm 4 -c preserve_interword_spaces=1',  # Single column of text
            '--oem 3 --psm 3 -c preserve_interword_spaces=1',  # Fully automatic
            '--oem 3 --psm 8 -c preserve_interword_spaces=1',  # Single word
        ]
        
        # Add language-specific configs if needed
        if quality_level == 'poor':
            # For poor quality images, try more aggressive settings
            base_configs.extend([
                '--oem 1 --psm 6',  # Legacy engine
                '--oem 3 --psm 7',  # Single text line
            ])
        
        return base_configs
    
    def extract_text_with_config(self, image: np.ndarray, config: str) -> OCRResult:
        """
        Extract text using a specific OCR configuration.
        
        Args:
            image: Preprocessed image array
            config: Tesseract configuration string
            
        Returns:
            OCR result object
        """
        start_time = time.time()
        
        try:
            # Extract text
            text = tesseract.image_to_string(image, config=config).strip()
            
            # Get detailed OCR data for confidence and bounding boxes
            ocr_data = tesseract.image_to_data(image, config=config, output_type=tesseract.Output.DICT)
            
            # Calculate confidence metrics
            confidence_info = self._calculate_confidence_metrics(ocr_data)
            
            # Extract bounding boxes for text selection
            bounding_boxes = self._extract_bounding_boxes(ocr_data)
            
            # Calculate text statistics
            word_count = len([word for word in text.split() if word.strip()])
            line_count = len([line for line in text.split('\n') if line.strip()])
            character_count = len(text)
            
            processing_time = time.time() - start_time
            
            result = OCRResult(
                text=text,
                confidence=confidence_info['weighted_confidence'],
                word_count=word_count,
                line_count=line_count,
                character_count=character_count,
                processing_time=processing_time,
                config_used=config,
                bounding_boxes=bounding_boxes,
                word_confidences=confidence_info['word_confidences']
            )
            
            logger.debug(f"OCR completed with config '{config}': confidence={result.confidence:.1f}%, words={word_count}")
            return result
            
        except Exception as e:
            logger.warning(f"OCR failed with config '{config}': {str(e)}")
            raise TextExtractionError(f"OCR extraction failed: {str(e)}")
    
    def _calculate_confidence_metrics(self, ocr_data: Dict[str, List]) -> Dict[str, any]:
        """
        Calculate detailed confidence metrics from OCR data.
        
        Args:
            ocr_data: Tesseract OCR data dictionary
            
        Returns:
            Dictionary with confidence metrics
        """
        confidences = []
        word_lengths = []
        word_confidences = []
        
        for i, conf in enumerate(ocr_data['conf']):
            if int(conf) > 0 and ocr_data['text'][i].strip():
                word_text = ocr_data['text'][i].strip()
                word_length = len(word_text)
                word_conf = int(conf)
                
                confidences.append(word_conf)
                word_lengths.append(word_length)
                word_confidences.append(word_conf)
        
        if confidences:
            # Simple average confidence
            avg_confidence = np.mean(confidences)
            
            # Length-weighted confidence (longer words are more reliable)
            if word_lengths:
                weighted_conf = sum(c * w for c, w in zip(confidences, word_lengths))
                total_weight = sum(word_lengths)
                weighted_confidence = weighted_conf / total_weight if total_weight > 0 else 0
            else:
                weighted_confidence = avg_confidence
        else:
            avg_confidence = 0
            weighted_confidence = 0
            word_confidences = []
        
        return {
            'avg_confidence': avg_confidence,
            'weighted_confidence': weighted_confidence,
            'word_confidences': word_confidences,
            'total_words': len(word_confidences)
        }
    
    def _extract_bounding_boxes(self, ocr_data: Dict[str, List]) -> List[Dict[str, any]]:
        """
        Extract bounding box information for text selection.
        
        Args:
            ocr_data: Tesseract OCR data dictionary
            
        Returns:
            List of bounding box dictionaries
        """
        bounding_boxes = []
        
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip() and int(ocr_data['conf'][i]) > 0:
                bbox = {
                    'text': ocr_data['text'][i],
                    'confidence': int(ocr_data['conf'][i]),
                    'left': int(ocr_data['left'][i]),
                    'top': int(ocr_data['top'][i]),
                    'width': int(ocr_data['width'][i]),
                    'height': int(ocr_data['height'][i]),
                    'level': int(ocr_data['level'][i]),
                    'word_num': int(ocr_data['word_num'][i]),
                    'line_num': int(ocr_data['line_num'][i])
                }
                bounding_boxes.append(bbox)
        
        return bounding_boxes
    
    def extract_text_multi_config(self, image: np.ndarray, quality_level: str) -> OCRResult:
        """
        Extract text using multiple configurations and select the best result.
        
        Args:
            image: Preprocessed image array
            quality_level: Image quality level
            
        Returns:
            Best OCR result
        """
        configs = self.get_optimal_configs(quality_level)
        results = []
        
        for config in configs:
            try:
                result = self.extract_text_with_config(image, config)
                if result.text.strip():  # Only consider results with actual text
                    results.append(result)
            except Exception as e:
                logger.debug(f"Config {config} failed: {str(e)}")
                continue
        
        if not results:
            # Fallback to basic config
            try:
                fallback_result = self.extract_text_with_config(image, '--oem 3 --psm 6')
                return fallback_result
            except Exception:
                raise TextExtractionError("All OCR configurations failed")
        
        # Select best result based on confidence and text quality
        best_result = max(results, key=lambda r: self._calculate_result_score(r))
        return best_result
    
    def _calculate_result_score(self, result: OCRResult) -> float:
        """
        Calculate a score for OCR result quality.
        
        Args:
            result: OCR result object
            
        Returns:
            Quality score
        """
        # Base score from confidence
        score = result.confidence
        
        # Bonus for longer, more meaningful text
        if result.word_count > 5:
            score += 10
        elif result.word_count > 2:
            score += 5
        
        # Penalty for very short results
        if result.character_count < 10:
            score -= 20
        
        # Bonus for good text quality
        text_quality = self._assess_text_quality(result.text)
        score += text_quality * 10
        
        return score
    
    def _assess_text_quality(self, text: str) -> float:
        """
        Assess the quality of extracted text.
        
        Args:
            text: Extracted text
            
        Returns:
            Quality score (0-1)
        """
        if not text.strip():
            return 0.0
        
        # Check for common OCR artifacts
        artifacts = ['|', 'l', 'I', '1', '0', 'O']  # Common misrecognitions
        artifact_count = sum(text.count(artifact) for artifact in artifacts)
        
        # Check for reasonable word lengths
        words = text.split()
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Quality score based on artifacts and word length
        artifact_penalty = min(artifact_count / len(text), 0.5)
        length_bonus = min(avg_word_length / 8, 0.3)  # Optimal word length around 8
        
        quality = 1.0 - artifact_penalty + length_bonus
        return max(0.0, min(1.0, quality))

def extract_text_from_image(image: np.ndarray,
                           quality_metrics: Dict[str, any],
                           enhancement_level: str = 'auto') -> OCRResult:
    """
    Extract text from image with optimized processing.
    
    Args:
        image: Preprocessed image array
        quality_metrics: Image quality metrics
        enhancement_level: Enhancement level used
        
    Returns:
        OCR result object
    """
    with Timer("Text extraction"):
        extractor = TextExtractor()
        
        # Determine quality level
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
        
        # Map quality to levels
        if overall_quality >= 80:
            quality_level = 'excellent'
        elif overall_quality >= 60:
            quality_level = 'good'
        elif overall_quality >= 40:
            quality_level = 'fair'
        else:
            quality_level = 'poor'
        
        # Extract text with optimal configuration
        result = extractor.extract_text_multi_config(image, quality_level)
        
        logger.info(f"Text extraction completed: {result.word_count} words, {result.confidence:.1f}% confidence")
        return result