# Enhanced OCR System

A comprehensive, production-ready Optical Character Recognition (OCR) system built with Flask, featuring advanced image preprocessing, quality-aware text extraction, and intelligent post-processing capabilities.

## 🌟 Features

### Advanced Image Processing
- **Intelligent Quality Assessment**: Automatic image quality analysis with metrics for sharpness, contrast, brightness, noise levels, and edge density
- **Adaptive Preprocessing**: Quality-aware image enhancement that applies optimal processing based on image characteristics
- **Robust Skew Correction**: Multi-method skew detection using Hough transforms, projection profiles, and Radon transforms
- **Noise Reduction**: Advanced denoising algorithms with adaptive parameters based on noise level estimation
- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) for optimal text visibility

### Smart OCR Engine
- **Multi-Configuration Processing**: Tests multiple OCR configurations and selects the best result based on confidence and content analysis
- **Quality-Aware Configuration Selection**: Automatically chooses optimal OCR settings based on image quality assessment
- **Confidence Scoring**: Comprehensive confidence calculation with word-length weighting and distribution analysis
- **Parallel Processing**: Optional parallel OCR processing for faster results on multi-core systems
- **Bounding Box Extraction**: Detailed word-level positioning information for advanced applications

### Intelligent Text Processing
- **Error Correction**: Automatic correction of common OCR errors using predefined patterns and heuristics
- **Text Validation**: Comprehensive text quality assessment with readability scoring
- **Format Preservation**: Multiple formatting options to preserve or clean text structure
- **Language Detection**: Basic language identification for extracted text
- **Noise Filtering**: Removal of OCR artifacts and invalid character sequences

### Modern Web Interface
- **Responsive Design**: Mobile-friendly interface with modern UI/UX principles
- **Drag & Drop Upload**: Intuitive file upload with visual feedback
- **Real-time Processing**: Live processing status with detailed progress information
- **Comprehensive Results**: Detailed extraction results with quality metrics and processing information
- **Export Options**: Multiple export formats (TXT, JSON, CSV) with download functionality

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
├── config.py              # Configuration management
├── utils.py               # Common utilities and helpers
├── image_validator.py     # Image validation and quality assessment
├── image_preprocessor.py  # Image enhancement and preprocessing
├── text_extractor.py      # OCR engine and text extraction
├── text_processor.py      # Text cleaning and post-processing
├── main.py               # Flask application and workflow orchestration
├── templates/            # HTML templates
│   ├── index.html        # Upload interface
│   └── result.html       # Results display
└── requirements.txt      # Python dependencies
```

### Component Overview

#### Image Validator (`image_validator.py`)
- File existence and accessibility verification
- Format validation using multiple methods (extension, MIME type, PIL)
- Size and dimension validation
- Comprehensive quality assessment with 7+ metrics
- Metadata extraction including EXIF data

#### Image Preprocessor (`image_preprocessor.py`)
- Quality-aware preprocessing pipeline
- Advanced noise reduction with adaptive parameters
- Multi-method skew detection and correction
- Contrast enhancement using CLAHE
- Morphological operations for text cleanup
- Adaptive thresholding with multiple methods

#### Text Extractor (`text_extractor.py`)
- Multi-configuration OCR processing
- Quality-based configuration selection
- Confidence scoring with multiple metrics
- Parallel processing capabilities
- Bounding box extraction
- Result scoring and selection algorithms

#### Text Processor (`text_processor.py`)
- Common OCR error correction
- Text structure validation
- Multiple formatting options
- Language detection
- Readability assessment
- Confidence adjustment based on processing results

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine
- System dependencies for image processing

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
sudo apt install libgl1-mesa-glx  # For OpenCV
sudo apt install python3-magic    # For file type detection
```

#### macOS
```bash
brew install tesseract
brew install libmagic
```

#### Windows
1. Download and install Tesseract from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Add Tesseract to your system PATH
3. Install Visual C++ redistributables if needed

### Python Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd enhanced-ocr-system
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Tesseract path (if needed):**
   Edit `config.py` and update the `TESSERACT_CMD` path:
   ```python
   TESSERACT_CMD = '/usr/bin/tesseract'  # Linux/macOS
   # TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
   ```

## 🎯 Usage

### Web Interface

1. **Start the application:**
   ```bash
   python main.py
   ```

2. **Access the web interface:**
   Open your browser and navigate to `http://localhost:5000`

3. **Upload and process images:**
   - Drag and drop an image file or click to browse
   - Select processing options (enhancement level, formatting, corrections)
   - Click "Extract Text" to process
   - View results with confidence scores and quality metrics

### API Usage

The system provides RESTful API endpoints for programmatic access:

#### OCR Processing
```bash
curl -X POST \
  http://localhost:5000/api/ocr \
  -F "file=@image.jpg" \
  -F "enhancement_level=auto" \
  -F "apply_corrections=true" \
  -F "format_type=standard"
```

#### Image Validation Only
```bash
curl -X POST \
  http://localhost:5000/api/validate \
  -F "file=@image.jpg"
```

### Python Integration

```python
from main import OCRWorkflow
from config import Config

# Initialize workflow
config = Config()
workflow = OCRWorkflow(config)

# Process image
with open('image.jpg', 'rb') as f:
    image_data = f.read()

results = workflow.process_image(
    image_data=image_data,
    filename='image.jpg',
    enhancement_level='auto',
    apply_text_corrections=True,
    format_type='standard'
)

if results['success']:
    extracted_text = results['final_results']['extracted_text']
    confidence = results['final_results']['adjusted_confidence']
    print(f"Extracted text (confidence: {confidence}%):")
    print(extracted_text)
else:
    print(f"Error: {results['error']}")
```

## ⚙️ Configuration

### Environment Variables

```bash
export FLASK_ENV=development  # or production
export SECRET_KEY=your-secret-key-here
export TESSERACT_CMD=/usr/bin/tesseract
```

### Configuration Options

Edit `config.py` to customize:

- **File Upload**: Maximum file size, allowed extensions
- **Image Processing**: Target DPI, dimension limits, quality thresholds
- **OCR Engine**: Tesseract configurations for different quality levels
- **Text Processing**: Error correction patterns, formatting options
- **Logging**: Log levels, file output, format

### Processing Levels

- **Auto**: Intelligently selects optimal processing based on image quality
- **Light**: Basic noise reduction and contrast enhancement
- **Medium**: Advanced filtering, skew correction, and text enhancement
- **Heavy**: Maximum preprocessing including morphological operations and aggressive thresholding

## 📊 Quality Metrics

The system provides comprehensive quality assessment:

### Image Quality Metrics
- **Sharpness**: Laplacian variance for focus measurement
- **Contrast**: Standard deviation of pixel intensities
- **Brightness**: Mean pixel intensity
- **Noise Level**: Estimated using median filtering
- **Edge Density**: Measure of text-like features
- **Dynamic Range**: Pixel intensity range
- **Entropy**: Information content measure

### OCR Quality Metrics
- **Confidence Score**: Word-length weighted confidence
- **Text Validation**: Structure and content analysis
- **Error Detection**: Common OCR error patterns
- **Readability Score**: Text complexity assessment
- **Language Detection**: Basic language identification

## 🔧 Advanced Features

### Custom OCR Configurations

Add custom Tesseract configurations in `config.py`:

```python
OCR_CONFIGS = {
    'custom_quality': [
        '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789',  # Numbers only
        '--oem 1 --psm 8 -l eng+fra',  # Multiple languages
        '--oem 3 --psm 7 -c preserve_interword_spaces=1',  # Single line
    ]
}
```

### Region-Based OCR

Process specific regions of an image:

```python
from text_extractor import TextExtractor

extractor = TextExtractor()
regions = [(x, y, width, height), ...]  # Define regions
results = extractor.extract_text_with_regions(image, regions, 'good')
```

### Parallel Processing

Enable parallel OCR for faster processing:

```python
result = extract_text_from_image(
    image=processed_image,
    quality_metrics=quality_metrics,
    use_parallel=True
)
```

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t enhanced-ocr-system .

# Run container
docker run -p 5000:5000 enhanced-ocr-system
```

The Dockerfile includes all necessary system dependencies and is optimized for production deployment.

## 📈 Performance Optimization

### Image Size Optimization
- Images are automatically resized to optimal dimensions
- Large images are processed in chunks when possible
- Memory usage is monitored and optimized

### Processing Speed
- Multi-configuration testing is optimized for speed
- Parallel processing available for multi-core systems
- Caching of preprocessing results when applicable

### Quality vs Speed Trade-offs
- **Light processing**: Fastest, good for high-quality images
- **Auto processing**: Balanced approach with intelligent selection
- **Heavy processing**: Slowest but best for difficult images

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=.

# Run specific test categories
pytest tests/test_image_validator.py -v
pytest tests/test_text_extractor.py -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed
- Ensure backward compatibility

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Tesseract OCR**: Google's open-source OCR engine
- **OpenCV**: Computer vision and image processing library
- **Flask**: Lightweight web framework
- **Bootstrap**: Frontend framework for responsive design
- **scikit-image**: Image processing algorithms

## 📞 Support

For support, please:

1. Check the [documentation](README.md)
2. Search [existing issues](../../issues)
3. Create a [new issue](../../issues/new) with detailed information

## 🗺️ Roadmap

### Upcoming Features
- [ ] Support for additional languages
- [ ] PDF processing capabilities
- [ ] Batch processing interface
- [ ] REST API documentation with Swagger
- [ ] Database integration for result storage
- [ ] User authentication and session management
- [ ] Advanced analytics and reporting
- [ ] Machine learning model integration
- [ ] Cloud deployment templates

### Performance Improvements
- [ ] GPU acceleration for image processing
- [ ] Distributed processing capabilities
- [ ] Advanced caching mechanisms
- [ ] Real-time processing optimization

---

**Built with ❤️ for accurate and reliable text extraction**