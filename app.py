from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify, abort
from PIL import Image
import os
import numpy as np
from flask_cors import CORS

# Import your CNN model and Memory Network class
from model import model
from MemoryNetWork import network

# Flask setup
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the model handlers
model_handler = model
memory_network_handler = network


@app.route('/')
def home():
    return render_template('index.html', accuracy=model_handler.load_evaluation_results()['Test Accuracy'], round = round,
                           loss= model_handler.load_evaluation_results()['Test Loss'], total_images = model.get_dataset_summary()['total_images'],
                           num_classes = model.get_dataset_summary()['num_classes'],)


@app.route('/insights')
def insights():
    return render_template('insights.html')


@app.route('/predict', methods=['POST'])
def predict():
    insights = []

    # Handle symptoms data
    symptoms = request.form.getlist('symptoms') or request.args.getlist('symptoms')
    if symptoms:
        predicted_diseases = memory_network_handler.predict_disease(symptoms)
        for disease, confidence in predicted_diseases:
            preventions = memory_network_handler.get_preventions(disease)
            cure = memory_network_handler.get_cure(disease)
            insights.append({
                'image_path': 'images/' + model_handler.get_random_image_path(disease),
                'predicted_class': disease,
                'prediction_confidence': float(confidence),
                'description': 'Insight based on symptoms',
                'preventions': preventions,
                'cure': cure
            })

    # Handle image upload
    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)
            model_handler.resizeImage(image_path)
            predicted_class, confidence = model_handler.predict_image(image_path)

            # Prepare insights from the image
            predicted_class = ' '.join(
                word.capitalize() for word in predicted_class.replace('__', ' ').replace('_', ' ').split(' ')).replace(
                '  ', ' ')
            print(predicted_class)
            preventions = memory_network_handler.get_preventions(predicted_class)
            cure = memory_network_handler.get_cure(predicted_class)
            insights.append(
                {
                    "image_path": image_path,
                    "predicted_class": predicted_class,
                    "prediction_confidence": float(confidence),
                    "description": "Insight based on image",
                    "preventions": preventions,
                    "cure": cure,
                }
            )

    # Order Insights according to confidence
    insights.sort(key=lambda x: x["prediction_confidence"])

    # Return JSON response if requested
    if request.headers.get('Accept') == 'application/json':
        return jsonify(insights=insights)

    # Otherwise, return the result template
    return render_template('result.html', image_path=image_path, predicted_class=predicted_class,
                           prediction_confidence=confidence, insights=insights)


# Serve uploaded files
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/images/<path:filename>')
def get_images(filename):
    try:
        return send_from_directory(".", filename)
    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(debug=True)
