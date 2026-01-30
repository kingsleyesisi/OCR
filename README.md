# Enhanced OCR System with Google Lens-like Capabilities

A powerful, modern OCR (Optical Character Recognition) system that provides Google Lens-like functionality for extracting text from images with advanced preprocessing and multi-line support.

## 🚀 Features

### ✨ Core Capabilities
- **Multi-line Text Extraction**: Extract complete documents with proper paragraph structure
- **Advanced Image Preprocessing**: Intelligent enhancement with perspective correction, noise reduction, and text-specific optimization
- **Language Detection**: Automatic language detection and optimization
- **Paragraph Extraction**: Smart paragraph detection and formatting preservation
- **High Accuracy**: Multiple OCR configurations for optimal results
- **Real-time Processing**: Fast processing with progress indicators

### 🔧 Advanced Preprocessing
- **Perspective Correction**: Automatically correct skewed or angled images
- **Skew Detection**: Advanced skew detection using text regions
- **Noise Reduction**: Intelligent noise reduction preserving text edges
- **Contrast Enhancement**: Adaptive contrast enhancement using CLAHE
- **Text-specific Enhancement**: Specialized sharpening and enhancement for text
- **Adaptive Binarization**: Smart binarization for poor quality images

### 📱 Modern Web Interface
- **Google Lens-like Design**: Clean, modern interface inspired by Google Lens
- **Drag & Drop Upload**: Easy file upload with drag and drop support
- **Real-time Processing**: Live progress indicators and status updates
- **Responsive Design**: Works perfectly on desktop and mobile devices
- **Text Statistics**: Detailed statistics including word count, line count, and confidence
- **Copy & Download**: Easy text copying and downloading functionality

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, Flask
- **OCR Engine**: Tesseract OCR with multiple configurations
- **Image Processing**: OpenCV, PIL (Pillow)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Additional Libraries**: NumPy, logging, dataclasses

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR engine
- OpenCV and other Python dependencies

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd OCR
   ```

2. **Install Tesseract OCR**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr

   # macOS
   brew install tesseract

   # Windows
   # Download from https://github.com/UB-Mannheim/tesseract/wiki
   ```

3. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**:
   ```bash
   python main.py
   ```

6. **Access the web interface**:
   Open your browser and go to `http://localhost:5000`

## 🎯 Usage

### Web Interface
1. **Upload Image**: Drag and drop an image or click to browse
2. **Configure Options**: Select enhancement level and text correction preferences
3. **Process**: Click "Extract Text" to start processing
4. **View Results**: See extracted text with confidence scores and statistics
5. **Copy/Download**: Copy text to clipboard or download as text file

### Command Line Testing
```bash
# Test with synthetic images
python test_ocr.py

# Test with real images in uploads folder
python test_real_image.py
```

## 🔍 How It Works

### 1. Image Preprocessing Pipeline
The system applies a sophisticated preprocessing pipeline:

1. **Image Loading**: Supports multiple input formats (file path, bytes, numpy array)
2. **Text Region Detection**: Uses MSER (Maximally Stable Extremal Regions) to detect text areas
3. **Perspective Correction**: Automatically corrects skewed images using text region analysis
4. **Skew Correction**: Advanced skew detection using text line analysis
5. **Noise Reduction**: Edge-preserving noise reduction
6. **Contrast Enhancement**: Adaptive contrast enhancement using CLAHE
7. **Text-specific Enhancement**: Specialized sharpening and enhancement
8. **Adaptive Binarization**: Smart binarization for poor quality images

### 2. OCR Processing
Multiple OCR configurations are tested to find the best result:

- **PSM 6**: Uniform block of text
- **PSM 4**: Single column of text
- **PSM 3**: Fully automatic page segmentation
- **PSM 8**: Single word
- **PSM 7**: Single text line
- **PSM 13**: Raw line

### 3. Text Processing
Advanced text processing includes:

- **Paragraph Extraction**: Smart paragraph detection and formatting
- **Language Detection**: Automatic language identification
- **Quality Assessment**: Confidence scoring and quality metrics
- **Text Correction**: Intelligent error correction

## 📊 Performance

### Test Results
The enhanced system shows significant improvements:

- **Multi-line Documents**: Successfully extracts 20+ lines with proper formatting
- **Perspective Correction**: Automatically corrects skewed images
- **Confidence Scores**: 90%+ confidence on high-quality images
- **Processing Speed**: 1-3 seconds for typical documents
- **Language Support**: English with framework for other languages

### Quality Metrics
- **Word Count**: Accurate word counting with bounding boxes
- **Line Count**: Proper line detection and counting
- **Paragraph Detection**: Smart paragraph extraction
- **Confidence Scoring**: Weighted confidence based on word length and quality

## 🎨 Interface Features

### Google Lens-like Design
- **Clean Interface**: Minimalist design with focus on functionality
- **Color Scheme**: Google-inspired blue color palette
- **Modern Typography**: Inter font family for better readability
- **Smooth Animations**: Subtle animations and transitions
- **Responsive Layout**: Works on all screen sizes

### User Experience
- **Drag & Drop**: Intuitive file upload
- **Progress Indicators**: Real-time processing feedback
- **Error Handling**: Clear error messages and recovery options
- **Results Display**: Clean, organized results with statistics
- **Action Buttons**: Easy copy, download, and reprocess options

## 🔧 Configuration

### Environment Variables
```bash
# Flask configuration
FLASK_ENV=development
SECRET_KEY=your-secret-key

# Tesseract path (if not in system PATH)
TESSERACT_CMD=/usr/bin/tesseract
```

### Configuration Options
The system can be configured through `config.py`:

- **Image Processing**: Maximum dimensions, target DPI, quality thresholds
- **OCR Settings**: Multiple engine configurations for different quality levels
- **Preprocessing**: Noise reduction strength, contrast enhancement factors
- **Output Formats**: Supported output formats and default settings

## 🧪 Testing

### Test Scripts
- `test_ocr.py`: Comprehensive testing with synthetic images
- `test_real_image.py`: Testing with real images from uploads folder
- `test_fix.py`: Specific issue testing and debugging

### Test Coverage
- Multi-line document extraction
- Perspective correction
- Skew detection and correction
- Language detection
- Paragraph extraction
- Quality assessment

## 🚀 Future Enhancements

### Planned Features
- **Camera Integration**: Direct camera capture for mobile devices
- **Batch Processing**: Process multiple images simultaneously
- **Cloud Storage**: Integration with cloud storage services
- **API Endpoints**: RESTful API for integration
- **Mobile App**: Native mobile application
- **Advanced Languages**: Support for more languages and scripts

### Technical Improvements
- **Machine Learning**: Integration with deep learning models
- **GPU Acceleration**: CUDA support for faster processing
- **Real-time Processing**: WebSocket support for live processing
- **Advanced Analytics**: Detailed processing analytics and insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Tesseract OCR**: The powerful OCR engine that makes this possible
- **OpenCV**: Computer vision library for image processing
- **Flask**: Web framework for the backend
- **Bootstrap**: Frontend framework for responsive design
- **Google Lens**: Inspiration for the user interface design

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the test scripts for examples

---

**Built with ❤️ for accurate and efficient text extraction from images**