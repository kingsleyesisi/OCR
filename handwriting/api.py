from flask import Blueprint, request, jsonify, render_template
from PIL import Image
import io
from .predict import predict_from_pil_image

bp = Blueprint('handwriting', __name__)

@bp.route('/digit_recognize', methods=['GET'])
def digit_recognize_page():
    return render_template('digit_recognize.html')

@bp.route('/api/digit_recognize', methods=['POST'])
def digit_recognize_api():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    try:
        # Read image
        img = Image.open(file.stream)

        # Predict
        result = predict_from_pil_image(img)

        if 'error' in result:
             return jsonify({'success': False, 'error': result['error']})

        return jsonify({
            'success': True,
            'digit': result['digit'],
            'probability': result['probability']
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
