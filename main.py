"""
Main application module for the Enhanced OCR System.

This module orchestrates the complete OCR workflow including image validation,
preprocessing, text extraction, and post-processing with a modern web interface.
"""

import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, request, render_template, redirect, url_for, jsonify, flash, send_from_directory
import cv2
import numpy as np

# Import our custom modules
from config import get_config, Config
from utils import setup_logging, Timer, ensure_directory, safe_filename, ResultExporter
from image_validator import ImageValidator, validate_uploaded_file, ImageValidationError
from image_preprocessor import ImagePreprocessor, preprocess_for_ocr, ImagePreprocessingError
from text_extractor import TextExtractor, extract_text_from_image, TextExtractionError
from text_processor import TextProcessor, process_ocr_text

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config_class = get_config()
app.config.from_object(config_class)
config = config_class()

# Set up logging
setup_logging(config.LOG_LEVEL, config.LOG_FILE)
logger = logging.getLogger(__name__)

# Ensure required directories exist
ensure_directory(config.UPLOAD_FOLDER)
ensure_directory('outputs')
ensure_directory('logs')

class OCRWorkflow:
    """
    Main OCR workflow orchestrator.

    This class coordinates all components of the OCR system to provide
    a complete image-to-text conversion pipeline.
    """

    def __init__(self, config: Config):
        """
        Initialize the OCR workflow.

        Args:
            config: Configuration object
        """
        self.config = config
        self.validator = ImageValidator(config)
        self.preprocessor = ImagePreprocessor(config)
        self.extractor = TextExtractor(config)
        self.processor = TextProcessor()

    def process_image(self, image_data: bytes, filename: str,
                     enhancement_level: str = 'auto',
                     apply_text_corrections: bool = True,
                     format_type: str = 'standard') -> Dict[str, Any]:
        """
        Complete OCR processing pipeline.

        Args:
            image_data: Raw image data
            filename: Original filename
            enhancement_level: Image enhancement level
            apply_text_corrections: Whether to apply text corrections
            format_type: Text formatting type

        Returns:
            Complete processing results
        """
        with Timer("Complete OCR workflow"):
            results = {
                'filename': filename,
                'timestamp': datetime.now().isoformat(),
                'enhancement_level': enhancement_level,
                'processing_steps': [],
                'success': False
            }

            try:
                # Step 1: Image Validation
                logger.info(f"Starting OCR workflow for: {filename}")

                with Timer("Image validation"):
                    validation_report = validate_uploaded_file(image_data, filename)
                    results['validation'] = validation_report
                    results['processing_steps'].append('validation')

                if validation_report['validation_status'] != 'passed':
                    raise ImageValidationError("Image validation failed")

                # Step 2: Image Preprocessing
                with Timer("Image preprocessing"):
                    quality_metrics = validation_report['quality_metrics']
                    preprocessing_result = preprocess_for_ocr(
                        image_data, quality_metrics, enhancement_level
                    )
                    results['preprocessing'] = {
                        'steps_applied': preprocessing_result['processing_steps'],
                        'enhancement_level': preprocessing_result['enhancement_level'],
                        'is_binary': preprocessing_result['is_binary'],
                        'original_shape': preprocessing_result['original_shape'],
                        'final_shape': preprocessing_result['final_shape']
                    }
                    results['processing_steps'].append('preprocessing')

                # Step 3: Text Extraction
                with Timer("Text extraction"):
                    processed_image = preprocessing_result['processed_image']
                    ocr_result = extract_text_from_image(
                        processed_image, quality_metrics, enhancement_level
                    )
                    results['extraction'] = {
                        'text': ocr_result.text,
                        'confidence': ocr_result.confidence,
                        'word_count': ocr_result.word_count,
                        'line_count': ocr_result.line_count,
                        'character_count': ocr_result.character_count,
                        'processing_time': ocr_result.processing_time,
                        'config_used': ocr_result.config_used
                    }
                    results['processing_steps'].append('extraction')

                # Step 4: Text Processing
                with Timer("Text processing"):
                    processed_text = process_ocr_text(
                        ocr_result.text, apply_text_corrections, format_type
                    )
                    results['text_processing'] = {
                        'original_text': processed_text.original_text,
                        'cleaned_text': processed_text.cleaned_text,
                        'formatted_text': processed_text.formatted_text,
                        'corrections_made': processed_text.corrections_made,
                        'confidence_adjustment': processed_text.confidence_adjustment,
                        'word_count': processed_text.word_count,
                        'line_count': processed_text.line_count,
                        'character_count': processed_text.character_count,
                        'language_detected': processed_text.language_detected,
                        'readability_score': processed_text.readability_score
                    }
                    results['processing_steps'].append('text_processing')

                # Step 5: Final Results Compilation
                final_confidence = max(0, min(100,
                    ocr_result.confidence + processed_text.confidence_adjustment
                ))

                results['final_results'] = {
                    'extracted_text': processed_text.formatted_text,
                    'original_confidence': ocr_result.confidence,
                    'adjusted_confidence': final_confidence,
                    'confidence_adjustment': processed_text.confidence_adjustment,
                    'word_count': processed_text.word_count,
                    'line_count': processed_text.line_count,
                    'character_count': processed_text.character_count,
                    'language': processed_text.language_detected,
                    'readability_score': processed_text.readability_score,
                    'quality_assessment': self._assess_final_quality(results),
                    'processing_summary': self._create_processing_summary(results)
                }

                results['success'] = True
                logger.info(f"OCR workflow completed successfully for {filename}")

                return results

            except Exception as e:
                error_msg = str(e)
                logger.error(f"OCR workflow failed for {filename}: {error_msg}")
                logger.error(traceback.format_exc())

                results['error'] = error_msg
                results['error_type'] = type(e).__name__
                return results

    def _assess_final_quality(self, results: Dict[str, Any]) -> str:
        """
        Assess the overall quality of the OCR results.

        Args:
            results: Complete processing results

        Returns:
            Quality assessment string
        """
        try:
            final_confidence = results['final_results']['adjusted_confidence']
            word_count = results['final_results']['word_count']
            corrections_count = len(results['text_processing']['corrections_made'])

            # Quality assessment logic
            if final_confidence >= 85 and word_count > 5:
                return 'excellent'
            elif final_confidence >= 70 and word_count > 3:
                return 'good'
            elif final_confidence >= 50 and word_count > 1:
                return 'fair'
            else:
                return 'poor'

        except Exception:
            return 'unknown'

    def _create_processing_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of the processing pipeline.

        Args:
            results: Complete processing results

        Returns:
            Processing summary dictionary
        """
        try:
            return {
                'total_steps': len(results['processing_steps']),
                'steps_completed': results['processing_steps'],
                'image_quality': results['validation']['quality_metrics']['overall_quality'],
                'preprocessing_steps': len(results['preprocessing']['steps_applied']),
                'text_corrections': len(results['text_processing']['corrections_made']),
                'confidence_change': results['text_processing']['confidence_adjustment'],
                'final_assessment': results['final_results']['quality_assessment']
            }
        except Exception:
            return {'error': 'Could not create processing summary'}

# Initialize workflow
ocr_workflow = OCRWorkflow(config)

# Flask Routes
@app.context_processor
def inject_globals():
    """Inject global variables into templates."""
    return {
        'year': datetime.now().year,
        'app_name': 'Enhanced OCR System'
    }

@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(config.UPLOAD_FOLDER, filename)

@app.route('/ocr', methods=['POST'])
def ocr():
    """
    Main OCR processing route - returns JSON for AJAX requests.
    """
    try:
        # Validate file upload
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})

        filename = file.filename or 'unknown_image'

        # Get form parameters
        enhancement_level = request.form.get('enhancement_level', 'auto')
        apply_corrections = request.form.get('apply_text_corrections', 'true').lower() == 'true'
        format_type = request.form.get('format_type', 'standard')

        logger.info(f"Processing OCR request: {filename}, enhancement: {enhancement_level}")

        # Read file data
        file_data = file.read()

        # Process image through OCR workflow
        results = ocr_workflow.process_image(
            file_data, filename, enhancement_level, apply_corrections, format_type
        )

        if results['success']:
            # Save the uploaded file for display
            safe_name = safe_filename(filename)
            file_path = os.path.join(config.UPLOAD_FOLDER, safe_name)
            with open(file_path, 'wb') as f:
                f.write(file_data)

            logger.info(f"OCR completed successfully: confidence={results['final_results']['adjusted_confidence']:.1f}%, "
                       f"words={results['final_results']['word_count']}")

            # Return JSON response for AJAX
            return jsonify(results)
        else:
            return jsonify({'success': False, 'error': results.get('error', 'Unknown error')})

    except Exception as e:
        logger.error(f"OCR route error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'})

@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    """
    API endpoint for OCR processing.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = file.filename or 'unknown_image'

        # Get parameters
        enhancement_level = request.form.get('enhancement_level', 'auto')
        apply_corrections = request.form.get('apply_corrections', 'true').lower() == 'true'
        format_type = request.form.get('format_type', 'standard')

        # Process image
        file_data = file.read()
        results = ocr_workflow.process_image(
            file_data, filename, enhancement_level, apply_corrections, format_type
        )

        if results['success']:
            return jsonify({
                'success': True,
                'filename': results['filename'],
                'text': results['final_results']['extracted_text'],
                'confidence': results['final_results']['adjusted_confidence'],
                'original_confidence': results['final_results']['original_confidence'],
                'word_count': results['final_results']['word_count'],
                'line_count': results['final_results']['line_count'],
                'character_count': results['final_results']['character_count'],
                'language': results['final_results']['language'],
                'quality_assessment': results['final_results']['quality_assessment'],
                'processing_summary': results['final_results']['processing_summary'],
                'timestamp': results['timestamp']
            })
        else:
            return jsonify({
                'success': False,
                'error': results.get('error', 'Unknown error'),
                'error_type': results.get('error_type', 'UnknownError')
            }), 500

    except Exception as e:
        logger.error(f"API OCR error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate', methods=['POST'])
def api_validate():
    """
    API endpoint for image validation only.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        filename = file.filename or 'unknown_image'
        file_data = file.read()

        validation_report = validate_uploaded_file(file_data, filename)

        return jsonify({
            'success': True,
            'validation_report': validation_report
        })

    except Exception as e:
        logger.error(f"API validation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<format_type>')
def download_result(format_type):
    """
    Download OCR results in specified format.
    """
    # This would be implemented to download results
    # For now, return a simple response
    return jsonify({'message': f'Download in {format_type} format not yet implemented'})

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    flash('An unexpected error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    logger.info("Starting Enhanced OCR System")
    logger.info(f"Configuration: {config.__class__.__name__}")
    logger.info(f"Upload folder: {config.UPLOAD_FOLDER}")
    logger.info(f"Max file size: {config.MAX_CONTENT_LENGTH / (1024*1024):.1f}MB")

    # Run the application
    app.run(
        debug=config.DEBUG,
        host='0.0.0.0',
        port=5000
    )