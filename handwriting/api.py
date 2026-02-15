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

        # Predict
        results = predict_from_pil_image(img)

        # Check if error returned
        if isinstance(results, dict) and 'error' in results:
             return jsonify({'success': False, 'error': results['error']})

        # Return list of predictions
        return jsonify({
            'success': True,
            'predictions': results
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
