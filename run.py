#!/usr/bin/env python3
"""
Startup script for the Enhanced OCR System.
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Start the OCR system."""

    # Check if required directories exist
    required_dirs = ['uploads', 'outputs', 'logs']
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)

    # Import and run the main application
    try:
        from main import app, config, logger

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

    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()