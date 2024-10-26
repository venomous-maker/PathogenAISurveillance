from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from PIL import Image
import numpy as np
import os
import json
from flask_cors import CORS


# Import your CNN model class
from model import PlantVillageModel, model

# Flask setup
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)
# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the model
model_handler =  model

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/insights')
def insights():
    return render_template('insights.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save uploaded file
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        model_handler.resizeImage(image_path)
        # Predict the class
        predicted_class, confidence = model_handler.predict_image(image_path)

        # Load the image for display
        image = Image.open(image_path)

        # Process the image and return insights
        # Convert non-serializable types to standard Python types
        confidence = float(confidence)  # Convert float32 to float
        predicted_class = str(predicted_class)  # Convert to string if necessary

        # Prepare insights data
        insights = [{
            'image_path': image_path,
            'predicted_class': predicted_class,
            'prediction_confidence': confidence,
            'description': 'Insight based on prediction'
        }, {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'prediction_confidence': confidence,
            'description': 'Insight based on prediction'
        }]
        if request.headers.get('Accept') == 'application/json':
            return  jsonify(insights = insights)
        # Return result with image and prediction
        return render_template('result.html', image_path=image_path, predicted_class=predicted_class, prediction_confidence=confidence)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
