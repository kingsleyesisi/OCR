"""
Configuration settings for the OCR system.

This module contains all configuration parameters, file paths, and settings
used throughout the application.
"""

import os
from pathlib import Path

# Application Configuration
class Config:
    """Main configuration class for the OCR application."""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # File Upload Configuration
    UPLOAD_FOLDER = Path('uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'webp'}
    
    # OCR Configuration
    TESSERACT_CMD = '/usr/bin/tesseract'  # Default path for Linux
    
    # Image Processing Configuration
    MAX_IMAGE_DIMENSION = 4000  # Maximum width or height
    MIN_IMAGE_DIMENSION = 50    # Minimum width or height
    TARGET_DPI = 300           # Target DPI for OCR
    
    # Quality Thresholds
    HIGH_QUALITY_THRESHOLD = 80
    MEDIUM_QUALITY_THRESHOLD = 60
    MIN_CONFIDENCE_THRESHOLD = 30
    
    # Preprocessing Parameters
    NOISE_REDUCTION_STRENGTH = 10
    CONTRAST_ENHANCEMENT_FACTOR = 1.2
    BRIGHTNESS_ADJUSTMENT = 10
    
    # OCR Engine Configurations
    OCR_CONFIGS = {
        'high_quality': [
            '--oem 3 --psm 6 -c preserve_interword_spaces=1',
            '--oem 3 --psm 4 -c preserve_interword_spaces=1',
            '--oem 1 --psm 6',
            '--oem 3 --psm 3 -c preserve_interword_spaces=1',
        ],
        'medium_quality': [
            '--oem 3 --psm 6',
            '--oem 3 --psm 4',
            '--oem 3 --psm 8',
            '--oem 3 --psm 7',
        ],
        'low_quality': [
            '--oem 3 --psm 8',
            '--oem 3 --psm 7',
            '--oem 3 --psm 6',
            '--oem 3 --psm 4',
            '--oem 3 --psm 3',
        ]
    }
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'ocr_system.log'
    
    # Output Configuration
    OUTPUT_FORMATS = ['txt', 'json', 'csv']
    DEFAULT_OUTPUT_FORMAT = 'txt'

# Development Configuration
class DevelopmentConfig(Config):
    """Development-specific configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

# Production Configuration
class ProductionConfig(Config):
    """Production-specific configuration."""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("No SECRET_KEY set for production environment")

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get the appropriate configuration based on environment."""
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])