"""
Enhanced text extraction module for the OCR system with Google Lens-like capabilities.

This module provides optimized OCR with better accuracy, multi-line text extraction,
and intelligent text processing for various document types.
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
    bounding_boxes: Optional[List[Dict[str, any]]] = None
    word_confidences: Optional[List[float]] = None
    paragraphs: Optional[List[str]] = None
    detected_language: Optional[str] = None

class TextExtractionError(Exception):
    """Custom exception for text extraction errors."""
    pass

class TextExtractor:
    """
    Enhanced text extractor with Google Lens-like capabilities.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the text extractor.

        Args:
            config: Configuration object with OCR parameters
        """
        self.config = config if config is not None else Config()
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
        # Enhanced configurations for better multi-line text extraction
        base_configs = [
            '--oem 3 --psm 6 -c preserve_interword_spaces=1 -c textord_old_baselines=0',  # Uniform block of text
            '--oem 3 --psm 4 -c preserve_interword_spaces=1 -c textord_old_baselines=0',  # Single column of text
            '--oem 3 --psm 3 -c preserve_interword_spaces=1 -c textord_old_baselines=0',  # Fully automatic
            '--oem 3 --psm 8 -c preserve_interword_spaces=1',  # Single word
            '--oem 3 --psm 7 -c preserve_interword_spaces=1',  # Single text line
            '--oem 3 --psm 13 -c preserve_interword_spaces=1',  # Raw line
        ]

        # Add language-specific configs if needed
        if quality_level == 'poor':
            # For poor quality images, try more aggressive settings
            base_configs.extend([
                '--oem 1 --psm 6',  # Legacy engine
                '--oem 3 --psm 7',  # Single text line
                '--oem 3 --psm 8',  # Single word
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

            # Extract paragraphs
            paragraphs = self._extract_paragraphs(text)

            # Detect language
            detected_language = self._detect_language(text)

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
                word_confidences=confidence_info['word_confidences'],
                paragraphs=paragraphs,
                detected_language=detected_language
            )

            logger.debug(f"OCR completed with config '{config}': confidence={result.confidence:.1f}%, words={word_count}, lines={line_count}")
            return result

        except Exception as e:
            logger.warning(f"OCR failed with config '{config}': {str(e)}")
            raise TextExtractionError(f"OCR extraction failed: {str(e)}")

    def _extract_paragraphs(self, text: str) -> List[str]:
        """
        Extract paragraphs from OCR text.

        Args:
            text: OCR extracted text

        Returns:
            List of paragraphs
        """
        # Split by double newlines to get paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # If no double newlines, try single newlines with some logic
        if not paragraphs:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            current_paragraph = []
            paragraphs = []

            for line in lines:
                # If line is short and ends with punctuation, it might be a continuation
                if len(line) < 50 and line.endswith(('.', '!', '?')):
                    current_paragraph.append(line)
                elif len(line) < 30 and not line.endswith(('.', '!', '?')):
                    # Short line without punctuation, likely continuation
                    current_paragraph.append(line)
                else:
                    if current_paragraph:
                        current_paragraph.append(line)
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                    else:
                        paragraphs.append(line)

            # Add any remaining paragraph
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))

        return paragraphs

    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns.

        Args:
            text: Text to analyze

        Returns:
            Detected language code
        """
        if not text:
            return 'unknown'

        # Simple heuristics for language detection
        # Count non-ASCII characters
        non_ascii_count = sum(1 for char in text if ord(char) > 127)
        total_chars = len(text)

        if total_chars == 0:
            return 'unknown'

        non_ascii_ratio = non_ascii_count / total_chars

        # Check for specific language patterns
        if re.search(r'[а-яё]', text, re.IGNORECASE):
            return 'rus'
        elif re.search(r'[一-龯]', text):
            return 'chi_sim'
        elif re.search(r'[あ-ん]', text):
            return 'jpn'
        elif re.search(r'[가-힣]', text):
            return 'kor'
        elif non_ascii_ratio > 0.1:
            return 'mixed'
        else:
            return 'eng'

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
            except Exception as e:
                logger.error(f"All OCR configurations failed: {str(e)}")
                raise TextExtractionError("All OCR configurations failed")

        # Select the best result based on multiple criteria
        best_result = self._select_best_result(results)

        logger.info(f"Selected best OCR result: confidence={best_result.confidence:.1f}%, "
                   f"words={best_result.word_count}, lines={best_result.line_count}")

        return best_result

    def _select_best_result(self, results: List[OCRResult]) -> OCRResult:
        """
        Select the best OCR result based on multiple criteria.

        Args:
            results: List of OCR results

        Returns:
            Best OCR result
        """
        if len(results) == 1:
            return results[0]

        # Score each result
        scored_results = []
        for result in results:
            score = self._calculate_result_score(result)
            scored_results.append((score, result))

        # Sort by score (higher is better)
        scored_results.sort(key=lambda x: x[0], reverse=True)

        return scored_results[0][1]

    def _calculate_result_score(self, result: OCRResult) -> float:
        """
        Calculate a score for an OCR result.

        Args:
            result: OCR result to score

        Returns:
            Score (higher is better)
        """
        # Base score from confidence
        score = result.confidence

        # Bonus for more words (indicates more complete extraction)
        if result.word_count > 0:
            score += min(result.word_count * 0.5, 20)

        # Bonus for multiple lines (indicates better structure detection)
        if result.line_count > 1:
            score += min(result.line_count * 2, 15)

        # Bonus for paragraphs (indicates good text structure)
        if result.paragraphs and len(result.paragraphs) > 1:
            score += 10

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
        if not text:
            return 0.0

        # Check for common OCR artifacts
        artifacts = 0
        total_chars = len(text)

        # Count suspicious characters
        suspicious_chars = sum(1 for char in text if char in '|[]{}()<>')
        artifacts += suspicious_chars / total_chars if total_chars > 0 else 0

        # Check for excessive whitespace
        consecutive_spaces = len(re.findall(r' {2,}', text))
        artifacts += consecutive_spaces / total_chars if total_chars > 0 else 0

        # Check for mixed case issues
        mixed_case_issues = len(re.findall(r'[a-z][A-Z]', text))
        artifacts += mixed_case_issues / total_chars if total_chars > 0 else 0

        # Calculate quality score
        quality = max(0.0, 1.0 - artifacts)

        return quality

    def extract_text_with_language_detection(self, image: np.ndarray, quality_level: str) -> OCRResult:
        """
        Extract text with automatic language detection and optimization.

        Args:
            image: Preprocessed image array
            quality_level: Image quality level

        Returns:
            Best OCR result with language detection
        """
        # First, try with default language
        result = self.extract_text_multi_config(image, quality_level)

        # If we got some text, try to detect language and re-run if needed
        if result.text.strip() and result.detected_language != 'eng':
            logger.info(f"Detected language: {result.detected_language}")

            # For now, we'll use the best result from default language
            # In a full implementation, you could re-run with specific language packs
            return result

        return result

# Convenience function
def extract_text_from_image(image: np.ndarray,
                           quality_metrics: Dict[str, any],
                           enhancement_level: str = 'auto') -> OCRResult:
    """
    Convenience function for extracting text from preprocessed images.

    Args:
        image: Preprocessed image array
        quality_metrics: Image quality metrics
        enhancement_level: Enhancement level used

    Returns:
        OCR result object
    """
    extractor = TextExtractor()

    # Determine quality level
    overall_quality = quality_metrics.get('overall_quality', 50)
    if overall_quality >= 80:
        quality_level = 'excellent'
    elif overall_quality >= 60:
        quality_level = 'good'
    elif overall_quality >= 40:
        quality_level = 'fair'
    else:
        quality_level = 'poor'

    return extractor.extract_text_with_language_detection(image, quality_level)