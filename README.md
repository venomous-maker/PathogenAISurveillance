# Plant Disease Diagnosis System

## Overview

This project is a comprehensive plant disease diagnosis system designed to identify plant diseases based on signs and symptoms. The system leverages a neural network model (Memory Network) implemented in TensorFlow/Keras, integrated with a Flask web application to provide a user-friendly interface for users to input symptoms and receive predictions.

## Features

- **Flask Web Application**: A RESTful API for users to submit symptoms and receive disease predictions.
- **Memory Network Model**: A custom NLP model that utilizes LSTM layers to process and classify any trained data data.
- **Plant Village Model**: A custome cNN model that utilizes LSTM layers to process and classify images using the trained data.
- **PlantVillage Dataset**: The model is trained on a dataset containing various plant diseases along with their associated signs and symptoms.

## Project Structure

/Plant-Disease-Diagnosis ├── app.py # Main Flask application ├── MemoryNetWork.py # Memory Network implementation ├── PlantVillageModel.py # Model handling, training, and prediction ├── requirements.txt # Project dependencies ├── data/ # Dataset folder (optional) └── README.md # Project documentation

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/venomous-maker/PathogenAISurveillance
   cd PathogenAISurveillance
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Flask Application

To start the Flask web application, run the following command:

```bash
python app.py
```

This will start the Flask server on http://127.0.0.1:5000/ (or the specified port).

## API Endpoints

- POST /predict
  - Description: Predicts plant disease based on symptoms.
  - Request Body:
    ```json
      {
        "symptoms": ["symptom1", "symptom2", ...]
      }
    ```
  - Response:
    ```json
    {
      "predictions": [
        {"disease": "Disease Name", "confidence": 0.85},
        {"disease": "Disease Name", "confidence": 0.10},
        ...
      ]
    }
    ```

## Training the Model

If you need to retrain the model, you can use the MemoryNetWork class in MemoryNetWork.py. Make sure to call the following methods:

1. prepare_data(): Prepares the data for training.
2. split_data(): Splits the data into training and testing sets.
3. build_model(): Builds the neural network model.
4. compile_model(): Compiles the model with the desired loss function and optimizer.
5. train_model(): Trains the model on the prepared data.

## Model Predictions

Use the predict_disease() method from the MemoryNetWork class to make predictions based on an array of symptoms:

```python
predictions = network.predict_disease(["symptom1", "symptom2"])
```

## Requirements

- Python 3.x
- Flask
- TensorFlow / Keras
- Pandas
- NumPy

### Example requirements.txt

```makefile
Flask==2.0.3
tensorflow==2.5.0
pandas==1.3.1
numpy==1.21.0
scikit-learn==0.24.2
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for suggestions or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PlantVillage Dataset for providing the data used for training the model.
- Keras Documentation for guidance on building and training neural network models.

## Contact

For any inquiries, please contact morganokumu8@gmail.com.

