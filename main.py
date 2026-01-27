import os
from flask import Flask, render_template
from handwriting.api import bp as handwriting_bp

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Register blueprint
app.register_blueprint(handwriting_bp)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure model exists or provide instructions?
    # For now just start the app.
    app.run(host='0.0.0.0', port=5000, debug=True)
