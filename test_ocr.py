#!/usr/bin/env python3
"""
Simple test script for the OCR system.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_extractor import TextExtractor
from image_preprocessor import ImagePreprocessor
from config import Config

def test_ocr_system():
    """Test the OCR system with a sample image."""
    
    print("Testing OCR System...")
    
    # Initialize components
    config = Config()
    preprocessor = ImagePreprocessor(config)
    extractor = TextExtractor(config)
    
    # Create a simple test image with text
    print("Creating test image...")
    
    # Create a white image with black text
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add text using OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Hello World!", (50, 100), font, 2, (0, 0, 0), 3)
    cv2.putText(img, "OCR Test", (50, 200), font, 1.5, (0, 0, 0), 2)
    cv2.putText(img, "This is a test image", (50, 300), font, 1, (0, 0, 0), 2)
    
    # Save test image
    test_image_path = "snap.png"
    cv2.imwrite(test_image_path, img)
    print(f"Test image saved as: {test_image_path}")
    
    try:
        # Test preprocessing
        print("\nTesting preprocessing...")
        quality_metrics = {'overall_quality': 80}
        preprocessing_result = preprocessor.preprocess_pipeline(
            test_image_path, quality_metrics, 'auto'
        )
        print(f"Preprocessing completed: {len(preprocessing_result['processing_steps'])} steps")
        
        # Test text extraction
        print("\nTesting text extraction...")
        ocr_result = extractor.extract_text_multi_config(
            preprocessing_result['processed_image'], 'good'
        )
        
        print(f"Extracted text: '{ocr_result.text}'")
        print(f"Confidence: {ocr_result.confidence:.1f}%")
        print(f"Word count: {ocr_result.word_count}")
        print(f"Character count: {ocr_result.character_count}")
        
        if ocr_result.bounding_boxes:
            print(f"Bounding boxes: {len(ocr_result.bounding_boxes)} found")
        
        print("\nOCR test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"Cleaned up: {test_image_path}")

if __name__ == "__main__":
    test_ocr_system() 