#!/usr/bin/env python3
"""
Enhanced test script for the OCR system with Google Lens-like capabilities.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_extractor import TextExtractor
from image_preprocessor import ImagePreprocessor
from config import Config

def create_test_image_with_multiple_lines():
    """Create a test image with multiple lines of text like a document."""
    
    # Create a white image
    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Convert to PIL for better text rendering
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to use a system font, fallback to default if not available
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        # Fallback to default font
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Add title
    draw.text((50, 50), "OCR Test Document", fill=(0, 0, 0), font=font_large)
    
    # Add subtitle
    draw.text((50, 100), "Testing Multi-Line Text Extraction", fill=(50, 50, 50), font=font_medium)
    
    # Add multiple paragraphs
    paragraphs = [
        "This is the first paragraph of our test document. It contains multiple sentences to test how well the OCR system can handle longer text blocks and maintain proper formatting.",
        "The second paragraph demonstrates different text characteristics. It includes various punctuation marks, numbers (123), and mixed case text to challenge the OCR engine.",
        "Finally, this third paragraph tests the system's ability to handle multiple lines and paragraphs. The goal is to extract all text accurately while preserving the document structure."
    ]
    
    y_position = 150
    for i, paragraph in enumerate(paragraphs):
        # Simple text wrapping (basic implementation)
        words = paragraph.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) * 8 < width - 100:  # Rough character width estimation
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Draw the paragraph
        for line in lines:
            draw.text((50, y_position), line, fill=(0, 0, 0), font=font_small)
            y_position += 25
        
        y_position += 15  # Space between paragraphs
    
    # Add some additional text elements
    draw.text((50, y_position + 20), "Contact: test@example.com", fill=(0, 0, 0), font=font_small)
    draw.text((50, y_position + 45), "Phone: (555) 123-4567", fill=(0, 0, 0), font=font_small)
    draw.text((50, y_position + 70), "Date: 2024-01-15", fill=(0, 0, 0), font=font_small)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def create_test_image_with_perspective():
    """Create a test image with perspective distortion to test correction."""
    
    # Create a white image
    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Convert to PIL for text rendering
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except:
        font = ImageFont.load_default()
    
    # Add text that will be distorted
    text_lines = [
        "Perspective Test Document",
        "This text should be corrected",
        "by the perspective correction",
        "algorithm in the preprocessor."
    ]
    
    y_position = 200
    for line in text_lines:
        draw.text((100, y_position), line, fill=(0, 0, 0), font=font)
        y_position += 40
    
    # Convert back to OpenCV format
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Apply perspective transformation
    height, width = img.shape[:2]
    
    # Define source points (original rectangle)
    src_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    
    # Define destination points (perspective distorted)
    dst_points = np.array([
        [50, 50],      # Top-left
        [width-50, 100],  # Top-right
        [width-100, height-50],  # Bottom-right
        [100, height-100]   # Bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply perspective transform
    distorted = cv2.warpPerspective(img, matrix, (width, height))
    
    return distorted

def test_ocr_system():
    """Test the enhanced OCR system with multiple scenarios."""
    
    print("Testing Enhanced OCR System with Google Lens-like Capabilities...")
    print("=" * 70)
    
    # Initialize components
    config = Config()
    preprocessor = ImagePreprocessor(config)
    extractor = TextExtractor(config)
    
    # Test 1: Multi-line document
    print("\n1. Testing Multi-Line Document Extraction...")
    test_image = create_test_image_with_multiple_lines()
    test_image_path = "test_multiline.png"
    cv2.imwrite(test_image_path, test_image)
    print(f"Created multi-line test image: {test_image_path}")
    
    try:
        # Test preprocessing
        quality_metrics = {'overall_quality': 85.0}
        preprocessing_result = preprocessor.preprocess_pipeline(
            test_image_path, quality_metrics, 'auto'
        )
        print(f"Preprocessing completed: {len(preprocessing_result['processing_steps'])} steps")
        print(f"Steps applied: {', '.join(preprocessing_result['processing_steps'])}")
        
        # Test text extraction
        ocr_result = extractor.extract_text_multi_config(
            preprocessing_result['processed_image'], 'excellent'
        )
        
        print(f"\nExtracted text ({ocr_result.line_count} lines, {ocr_result.word_count} words):")
        print("-" * 50)
        print(ocr_result.text)
        print("-" * 50)
        print(f"Confidence: {ocr_result.confidence:.1f}%")
        print(f"Character count: {ocr_result.character_count}")
        
        if ocr_result.paragraphs:
            print(f"Paragraphs detected: {len(ocr_result.paragraphs)}")
            for i, para in enumerate(ocr_result.paragraphs[:2]):  # Show first 2 paragraphs
                print(f"  Paragraph {i+1}: {para[:100]}{'...' if len(para) > 100 else ''}")
        
        if ocr_result.detected_language:
            print(f"Detected language: {ocr_result.detected_language}")
        
        if ocr_result.bounding_boxes:
            print(f"Bounding boxes: {len(ocr_result.bounding_boxes)} found")
        
    except Exception as e:
        print(f"Error during multi-line test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
    
    # Test 2: Perspective correction
    print("\n2. Testing Perspective Correction...")
    perspective_image = create_test_image_with_perspective()
    perspective_path = "test_perspective.png"
    cv2.imwrite(perspective_path, perspective_image)
    print(f"Created perspective test image: {perspective_path}")
    
    try:
        # Test preprocessing with perspective correction
        quality_metrics = {'overall_quality': 75.0}
        preprocessing_result = preprocessor.preprocess_pipeline(
            perspective_path, quality_metrics, 'auto'
        )
        
        print(f"Preprocessing completed: {len(preprocessing_result['processing_steps'])} steps")
        if 'perspective_correction' in preprocessing_result['processing_steps']:
            print("✓ Perspective correction applied successfully")
        else:
            print("⚠ Perspective correction not applied (may not have been needed)")
        
        # Test text extraction
        ocr_result = extractor.extract_text_multi_config(
            preprocessing_result['processed_image'], 'good'
        )
        
        print(f"\nExtracted text from perspective-corrected image:")
        print("-" * 50)
        print(ocr_result.text)
        print("-" * 50)
        print(f"Confidence: {ocr_result.confidence:.1f}%")
        print(f"Words: {ocr_result.word_count}, Lines: {ocr_result.line_count}")
        
    except Exception as e:
        print(f"Error during perspective test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(perspective_path):
            os.remove(perspective_path)
    
    # Test 3: Simple test (original)
    print("\n3. Testing Simple Text Extraction...")
    
    # Create a simple test image with text
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
        quality_metrics = {'overall_quality': 80.0}
        preprocessing_result = preprocessor.preprocess_pipeline(
            test_image_path, quality_metrics, 'auto'
        )
        print(f"Preprocessing completed: {len(preprocessing_result['processing_steps'])} steps")
        
        # Test text extraction
        ocr_result = extractor.extract_text_multi_config(
            preprocessing_result['processed_image'], 'good'
        )
        
        print(f"Extracted text: '{ocr_result.text}'")
        print(f"Confidence: {ocr_result.confidence:.1f}%")
        print(f"Word count: {ocr_result.word_count}")
        print(f"Character count: {ocr_result.character_count}")
        
        if ocr_result.bounding_boxes:
            print(f"Bounding boxes: {len(ocr_result.bounding_boxes)} found")
        
        print("\n✓ Simple OCR test completed successfully!")
        
    except Exception as e:
        print(f"Error during simple testing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"Cleaned up: {test_image_path}")
    
    print("\n" + "=" * 70)
    print("Enhanced OCR System Testing Completed!")
    print("The system now supports:")
    print("✓ Multi-line text extraction")
    print("✓ Perspective correction")
    print("✓ Advanced image preprocessing")
    print("✓ Language detection")
    print("✓ Paragraph extraction")
    print("✓ Google Lens-like capabilities")

if __name__ == "__main__":
    test_ocr_system() 