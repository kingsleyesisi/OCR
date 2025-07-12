"""
Text extraction module for the OCR system.

This module implements advanced OCR techniques with multiple engines,
confidence scoring, and quality-aware configuration selection.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
import numpy as np
import pytesseract as tesseract
from PIL import Image
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from config import Config
from utils import Timer, retry_on_failure

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """Data class for OCR results."""
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
    Advanced text extractor with multiple OCR engines and quality-aware processing.
    
    This class provides comprehensive text extraction capabilities with
    confidence scoring, multiple configuration testing, and detailed analysis.
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
    
    def get_optimal_configs(self, quality_level: str, image_characteristics: Dict[str, any]) -> List[str]:
        """
        Get optimal OCR configurations based on image quality and characteristics.
        
        Args:
            quality_level: Image quality level (excellent, good, fair, poor)
            image_characteristics: Image analysis results
            
        Returns:
            List of OCR configuration strings
        """
        # Map quality levels to config categories
        quality_mapping = {
            'excellent': 'high_quality',
            'good': 'high_quality',
            'fair': 'medium_quality',
            'poor': 'low_quality'
        }
        
        config_category = quality_mapping.get(quality_level, 'medium_quality')
        base_configs = self.config.OCR_CONFIGS.get(config_category, self.config.OCR_CONFIGS['medium_quality'])
        
        # Customize configs based on image characteristics
        customized_configs = []
        
        for base_config in base_configs:
            # Add language-specific configurations if needed
            if 'preserve_interword_spaces=1' in base_config:
                customized_configs.append(base_config)
            else:
                # Add preserve_interword_spaces for better word separation
                customized_configs.append(base_config + ' -c preserve_interword_spaces=1')
        
        # Add specialized configurations for specific image types
        if image_characteristics.get('edge_density', 0) > 0.05:
            # High edge density suggests complex layout
            customized_configs.extend([
                '--oem 3 --psm 3',  # Fully automatic page segmentation
                '--oem 3 --psm 11', # Sparse text
            ])
        
        if image_characteristics.get('aspect_ratio', 1) > 3:
            # Wide images might be single lines
            customized_configs.insert(0, '--oem 3 --psm 7')  # Single text line
        
        return customized_configs
    
    @retry_on_failure(max_retries=2, delay=0.5)
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
            
            # Extract bounding boxes
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
            
            # Confidence distribution analysis
            high_conf_ratio = sum(1 for c in confidences if c >= 80) / len(confidences)
            low_conf_ratio = sum(1 for c in confidences if c < 50) / len(confidences)
            
        else:
            avg_confidence = 0
            weighted_confidence = 0
            high_conf_ratio = 0
            low_conf_ratio = 1
            word_confidences = []
        
        return {
            'avg_confidence': avg_confidence,
            'weighted_confidence': weighted_confidence,
            'high_conf_ratio': high_conf_ratio,
            'low_conf_ratio': low_conf_ratio,
            'word_confidences': word_confidences,
            'total_words': len(word_confidences)
        }
    
    def _extract_bounding_boxes(self, ocr_data: Dict[str, List]) -> List[Dict[str, any]]:
        """
        Extract bounding box information from OCR data.
        
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
                    'level': int(ocr_data['level'][i])
                }
                bounding_boxes.append(bbox)
        
        return bounding_boxes
    
    def extract_text_multi_config(self, image: np.ndarray, 
                                 quality_level: str,
                                 image_characteristics: Dict[str, any]) -> OCRResult:
        """
        Extract text using multiple configurations and return the best result.
        
        Args:
            image: Preprocessed image array
            quality_level: Image quality assessment
            image_characteristics: Image analysis results
            
        Returns:
            Best OCR result
        """
        with Timer("Multi-config OCR extraction"):
            configs = self.get_optimal_configs(quality_level, image_characteristics)
            
            results = []
            best_result = None
            best_score = -1
            
            for i, config in enumerate(configs):
                try:
                    result = self.extract_text_with_config(image, config)
                    
                    # Calculate result score
                    score = self._calculate_result_score(result)
                    
                    results.append({
                        'result': result,
                        'score': score,
                        'config_index': i
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                    
                    logger.debug(f"Config {i+1}/{len(configs)}: score={score:.1f}, conf={result.confidence:.1f}%, words={result.word_count}")
                    
                except Exception as e:
                    logger.warning(f"Config {i+1} failed: {str(e)}")
                    continue
            
            if best_result is None:
                raise TextExtractionError("All OCR configurations failed")
            
            logger.info(f"Best result: score={best_score:.1f}, confidence={best_result.confidence:.1f}%, words={best_result.word_count}")
            return best_result
    
    def _calculate_result_score(self, result: OCRResult) -> float:
        """
        Calculate a comprehensive score for an OCR result.
        
        Args:
            result: OCR result object
            
        Returns:
            Numerical score for the result
        """
        # Base confidence score (0-100)
        confidence_score = result.confidence
        
        # Content score based on text characteristics
        content_score = 0
        
        # Bonus for reasonable word count
        if result.word_count > 0:
            content_score += min(result.word_count * 2, 50)
        
        # Bonus for reasonable character count
        if result.character_count > 10:
            content_score += min(result.character_count * 0.1, 20)
        
        # Penalty for very short text (likely noise)
        if result.character_count < 5:
            content_score -= 20
        
        # Bonus for multiple lines (structured text)
        if result.line_count > 1:
            content_score += min(result.line_count * 5, 25)
        
        # Text quality assessment
        text_quality_score = self._assess_text_quality(result.text)
        
        # Combine scores with weights
        total_score = (
            confidence_score * 0.5 +      # 50% weight on confidence
            content_score * 0.3 +         # 30% weight on content metrics
            text_quality_score * 0.2      # 20% weight on text quality
        )
        
        return max(0, total_score)
    
    def _assess_text_quality(self, text: str) -> float:
        """
        Assess the quality of extracted text.
        
        Args:
            text: Extracted text string
            
        Returns:
            Quality score (0-100)
        """
        if not text.strip():
            return 0
        
        score = 50  # Base score
        
        # Check for reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if 0.3 <= alpha_ratio <= 0.9:
            score += 20
        
        # Check for reasonable word structure
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if 2 <= avg_word_length <= 15:
                score += 15
        
        # Penalty for excessive special characters
        special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / len(text)
        if special_char_ratio > 0.3:
            score -= 20
        
        # Bonus for proper sentence structure
        if re.search(r'[.!?]', text):
            score += 10
        
        # Penalty for excessive repetition
        if len(set(text.lower().split())) < len(text.split()) * 0.5:
            score -= 15
        
        return max(0, min(100, score))
    
    def extract_text_parallel(self, image: np.ndarray,
                             quality_level: str,
                             image_characteristics: Dict[str, any],
                             max_workers: int = 3) -> OCRResult:
        """
        Extract text using parallel processing for faster results.
        
        Args:
            image: Preprocessed image array
            quality_level: Image quality assessment
            image_characteristics: Image analysis results
            max_workers: Maximum number of parallel workers
            
        Returns:
            Best OCR result
        """
        configs = self.get_optimal_configs(quality_level, image_characteristics)
        
        # Limit configs for parallel processing
        configs = configs[:max_workers]
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit OCR tasks
            future_to_config = {
                executor.submit(self.extract_text_with_config, image, config): config
                for config in configs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    score = self._calculate_result_score(result)
                    results.append((result, score))
                    logger.debug(f"Parallel OCR completed for config: {config}")
                except Exception as e:
                    logger.warning(f"Parallel OCR failed for config {config}: {str(e)}")
        
        if not results:
            raise TextExtractionError("All parallel OCR attempts failed")
        
        # Return best result
        best_result, best_score = max(results, key=lambda x: x[1])
        logger.info(f"Parallel OCR best result: score={best_score:.1f}, confidence={best_result.confidence:.1f}%")
        
        return best_result
    
    def extract_text_with_regions(self, image: np.ndarray,
                                 regions: List[Tuple[int, int, int, int]],
                                 quality_level: str) -> List[OCRResult]:
        """
        Extract text from specific regions of the image.
        
        Args:
            image: Input image array
            regions: List of (x, y, width, height) tuples
            quality_level: Image quality assessment
            
        Returns:
            List of OCR results for each region
        """
        results = []
        
        for i, (x, y, w, h) in enumerate(regions):
            try:
                # Extract region
                region = image[y:y+h, x:x+w]
                
                if region.size == 0:
                    logger.warning(f"Empty region {i}: ({x}, {y}, {w}, {h})")
                    continue
                
                # Extract text from region
                region_characteristics = {'edge_density': 0.02, 'aspect_ratio': w/h}
                result = self.extract_text_multi_config(region, quality_level, region_characteristics)
                
                # Add region information
                result.region_bounds = (x, y, w, h)
                results.append(result)
                
                logger.debug(f"Region {i} OCR: confidence={result.confidence:.1f}%, text_length={len(result.text)}")
                
            except Exception as e:
                logger.error(f"OCR failed for region {i}: {str(e)}")
                continue
        
        return results

def extract_text_from_image(image: np.ndarray,
                           quality_metrics: Dict[str, any],
                           enhancement_level: str = 'auto',
                           use_parallel: bool = False) -> OCRResult:
    """
    Convenience function for extracting text from an image.
    
    Args:
        image: Preprocessed image array
        quality_metrics: Image quality assessment results
        enhancement_level: Level of enhancement applied
        use_parallel: Whether to use parallel processing
        
    Returns:
        OCR result
    """
    extractor = TextExtractor()
    
    quality_level = quality_metrics.get('overall_quality', 'fair')
    image_characteristics = {
        'edge_density': quality_metrics.get('edge_density', 0.02),
        'aspect_ratio': quality_metrics.get('aspect_ratio', 1.0),
        'brightness': quality_metrics.get('brightness', 128),
        'contrast': quality_metrics.get('contrast', 50)
    }
    
    if use_parallel:
        return extractor.extract_text_parallel(image, quality_level, image_characteristics)
    else:
        return extractor.extract_text_multi_config(image, quality_level, image_characteristics)