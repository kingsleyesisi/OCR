from flask import Flask, request, render_template, redirect, url_for
import pytesseract as rack
import cv2
import numpy as np

app = Flask(__name__)
rack.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

@app.context_processor
def inject_year():
    from datetime import datetime
    return {'year': datetime.now().year}

@app.route('/')
def index():
    return render_template('index.html')

def get_text_from_image(file_stream):
    data = file_stream.read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 85, 11
    )
    return rack.image_to_string(processed)

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    text = get_text_from_image(request.files['file'])
    return render_template('result.html', extracted_text=text)

if __name__ == '__main__':
    app.run(debug=True)
