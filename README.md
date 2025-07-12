# Enhanced OCR System

A modern, accurate OCR (Optical Character Recognition) system with Google Lens-like text selection capabilities.

## Features

- **High Accuracy OCR**: Optimized text extraction with multiple configuration testing
- **Text Selection**: Click on any text in the image to select and copy it (like Google Lens)
- **Smart Preprocessing**: Automatic image enhancement based on quality assessment
- **Modern UI**: Beautiful, responsive web interface
- **Real-time Processing**: Fast text extraction with confidence scoring
- **Multiple Formats**: Support for various image formats (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP)

## Installation

### Prerequisites

- Python 3.7+
- Tesseract OCR engine

### Install Tesseract

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

1. Start the server:
```bash
python main.py
```

2. Open your browser and go to `http://localhost:5000`

3. Upload an image and get OCR results with interactive text selection

### Command Line Testing

Test the OCR system:
```bash
python test_ocr.py
```

## How It Works

### 1. Image Preprocessing
- Automatic quality assessment
- Smart enhancement based on image characteristics
- Noise reduction and contrast enhancement
- Skew correction

### 2. Text Extraction
- Multiple OCR configurations tested automatically
- Best result selection based on confidence and quality
- Bounding box extraction for text selection

### 3. Text Selection Interface
- Interactive overlay showing detected text regions
- Click to select individual text elements
- Copy selected text or entire document
- Visual feedback for selected regions

## Configuration

Edit `config.py` to customize:
- OCR engine settings
- Image processing parameters
- File upload limits
- Logging configuration

## API Endpoints

- `POST /ocr` - Main OCR processing
- `POST /api/ocr` - JSON API for OCR
- `POST /api/validate` - Image validation only

## File Structure

```
OCR/
├── main.py              # Flask application
├── text_extractor.py    # OCR text extraction
├── image_preprocessor.py # Image preprocessing
├── image_validator.py   # Image validation
├── text_processor.py    # Text post-processing
├── config.py           # Configuration
├── utils.py            # Utility functions
├── templates/          # HTML templates
├── uploads/           # Uploaded images
├── outputs/           # Processed results
└── logs/              # Log files
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Ensure Tesseract is installed and the path in `config.py` is correct
2. **Poor OCR results**: Try different enhancement levels or improve image quality
3. **Text selection not working**: Check that bounding boxes are being generated correctly

### Debug Mode

Run with debug logging:
```bash
export FLASK_ENV=development
python main.py
```

## Performance

- **Processing Time**: Typically 1-5 seconds per image
- **Accuracy**: 85-95% for clear, well-formatted text
- **Supported Languages**: English (can be extended)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.