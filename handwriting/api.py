from flask import Blueprint, request, jsonify, render_template
from PIL import Image
import io
from .predict import predict_from_pil_image

bp = Blueprint('handwriting', __name__)


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

        # Predict all digits
        result = predict_from_pil_image(img)

        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']})

        response = {'success': True}

        if 'summary' in result:
            response['digits'] = result['digits']
            response['summary'] = result['summary']
        else:
            response['digits'] = result.get('digits', [])
            response['message'] = result.get('message', "Can't identify any valid digit between 1-100 in this image")

        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
