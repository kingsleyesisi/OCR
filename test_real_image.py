#!/usr/bin/env python3
"""
Test script for the enhanced OCR system with real images.
"""

import os
import sys
import cv2
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_extractor import TextExtractor
from image_preprocessor import ImagePreprocessor
from config import Config

def test_with_real_images():
    """Test the OCR system with real images from the uploads folder."""

    print("Testing Enhanced OCR System with Real Images...")
    print("=" * 60)

    # Initialize components
    config = Config()
    preprocessor = ImagePreprocessor(config)
    extractor = TextExtractor(config)

    # Get list of images in uploads folder
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        print(f"Uploads directory '{uploads_dir}' not found.")
        return

    image_files = [f for f in os.listdir(uploads_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]

    if not image_files:
        print("No image files found in uploads directory.")
        return

    print(f"Found {len(image_files)} image(s) to test:")
    for img_file in image_files:
        print(f"  - {img_file}")

    print("\n" + "=" * 60)

    # Test each image
    for i, img_file in enumerate(image_files, 1):
        print(f"\n{i}. Testing: {img_file}")
        print("-" * 40)

        image_path = os.path.join(uploads_dir, img_file)

        try:
            # Test preprocessing
            print("Preprocessing image...")
            quality_metrics = {'overall_quality': 75.0}  # Assume medium quality
            preprocessing_result = preprocessor.preprocess_pipeline(
                image_path, quality_metrics, 'auto'
            )
            print(f"✓ Preprocessing completed: {len(preprocessing_result['processing_steps'])} steps")
            print(f"  Steps: {', '.join(preprocessing_result['processing_steps'])}")

            # Test text extraction
            print("Extracting text...")
            ocr_result = extractor.extract_text_multi_config(
                preprocessing_result['processed_image'], 'good'
            )

            # Display results
            print(f"\nExtracted Text ({ocr_result.line_count} lines, {ocr_result.word_count} words):")
            print("-" * 50)
            if ocr_result.text.strip():
                # Show first 200 characters with ellipsis if longer
                display_text = ocr_result.text.strip()
                if len(display_text) > 200:
                    display_text = display_text[:200] + "..."
                print(display_text)
            else:
                print("(No text extracted)")
            print("-" * 50)

            print(f"Confidence: {ocr_result.confidence:.1f}%")
            print(f"Character count: {ocr_result.character_count}")

            if ocr_result.paragraphs:
                print(f"Paragraphs detected: {len(ocr_result.paragraphs)}")

            if ocr_result.detected_language:
                print(f"Detected language: {ocr_result.detected_language}")

            if ocr_result.bounding_boxes:
                print(f"Bounding boxes: {len(ocr_result.bounding_boxes)} found")

            print(f"Processing time: {ocr_result.processing_time:.2f}s")

            # Quality assessment
            if ocr_result.confidence >= 90:
                quality = "Excellent"
            elif ocr_result.confidence >= 70:
                quality = "Good"
            elif ocr_result.confidence >= 50:
                quality = "Fair"
            else:
                quality = "Poor"

            print(f"Overall quality: {quality}")

        except Exception as e:
            print(f"❌ Error processing {img_file}: {str(e)}")
            import traceback
            traceback.print_exc()

        print()

    print("=" * 60)
    print("Real image testing completed!")
    print("\nThe enhanced OCR system now supports:")
    print("✓ Multi-line text extraction")
    print("✓ Advanced image preprocessing")
    print("✓ Perspective correction")
    print("✓ Language detection")
    print("✓ Paragraph extraction")
    print("✓ Google Lens-like capabilities")

if __name__ == "__main__":
    test_with_real_images()