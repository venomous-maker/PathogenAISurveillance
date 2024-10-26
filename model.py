import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


class PlantVillageModel:
    def __init__(self, dataset_path, image_size=(224, 224), batch_size=32):
        self.loaded_from_file = None
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.model = None
        self.class_labels = None
        self.load_data()
        self.prepare_data()
        self.load_model()

    def load_data(self):
        images, labels = [], []
        for subfolder in tqdm(os.listdir(self.dataset_path)):
            subfolder_path = os.path.join(self.dataset_path, subfolder)
            for image_filename in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_filename)
                images.append(image_path)
                labels.append(subfolder)
        df = pd.DataFrame({'image': images, 'label': labels})
        return df

    def prepare_data(self):
        df = self.load_data()
        # Split data into train, validation, and test sets
        X_train, X_test1, y_train, y_test1 = train_test_split(
            df['image'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test1, y_test1, test_size=0.5, random_state=42, stratify=y_test1
        )
        df_train = pd.DataFrame({'image': X_train, 'label': y_train})
        df_test = pd.DataFrame({'image': X_test, 'label': y_test})
        df_val = pd.DataFrame({'image': X_val, 'label': y_val})

        # ImageDataGenerators for data augmentation
        datagen = ImageDataGenerator(rescale=1. / 255)
        train_datagen = ImageDataGenerator(
            rescale=1. / 255, horizontal_flip=True
        )

        # Generators for train, validation, and test sets
        self.train_generator = train_datagen.flow_from_dataframe(
            df_train, x_col='image', y_col='label', target_size=self.image_size,
            batch_size=self.batch_size, shuffle=True
        )
        self.val_generator = datagen.flow_from_dataframe(
            df_val, x_col='image', y_col='label', target_size=self.image_size,
            batch_size=self.batch_size, shuffle=False
        )
        self.test_generator = datagen.flow_from_dataframe(
            df_test, x_col='image', y_col='label', target_size=self.image_size,
            batch_size=self.batch_size, shuffle=False
        )
        self.class_labels = list(self.test_generator.class_indices.keys())

    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_size[0], self.image_size[1], 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(len(self.class_labels), activation='softmax')
        ])
        self.model = model
        self.compile()

    def train_model(self, epochs=20):
        checkpoint_cb = ModelCheckpoint("plant_disease_model.keras", save_best_only=True)
        early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )
        return history

    def evaluate_model(self):
        results = self.model.evaluate(self.test_generator)
        print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

    def save_model(self, filepath="plant_disease_model.h5"):
        self.model.save(filepath)

    def load_model(self, model_json_path="", model_weights_path="", model_path="saved_model.h5"):
        """Loads a pre-trained model from a JSON file."""
        # model_json_path = 'plantvillage_model.json'
        # model_weights_path = 'plantvillage_model_weights.h5'

        if os.path.exists(model_json_path) and os.path.exists(model_weights_path):
            with open(model_json_path, 'r') as json_file:
                model_json = json_file.read()
            self.model = tf.keras.models.model_from_json(model_json)
            self.model.load_weights(model_weights_path)
            self.loaded_from_file = True
            self.compile()
            print("Model loaded from", model_json_path)
        elif model_path is not None and model_path != "" and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.loaded_from_file = True
            print("Model loaded from", model_path)
            self.compile()
        else:
            self.model = None
            pass
    def compile(self):
        if self.model is None:
            raise ValueError("Model must be compiled before training")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    def hasmodel(self):
        return self.model is not None

    def predict_image(self, img_path):
        """Predicts the class of a given image."""
        img = load_img(img_path, target_size=self.image_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_labels[predicted_class_idx]

        return predicted_class, confidence

    def display_prediction(self, img_path):
        """Displays the predicted class and confidence for an image."""
        predicted_class, confidence = self.predict_image(img_path)
        print(f"Predicted class: {predicted_class} with confidence: {confidence:.2f}")

    def resizeImage(self, image_path):
        # Open the image
        if os.path.exists(image_path):
            with Image.open(image_path) as image:
                # Resize if not already the correct size
                if image.size != (self.image_size[0], self.image_size[1]):
                    image = image.resize((self.image_size[0], self.image_size[1]))
                    image.save(image_path)  # Save resized image to the same path
        else:
            raise ValueError("Image path does not exist")


# Instantiate the class
model = PlantVillageModel(dataset_path="./plantvillage dataset/color")

if not model.hasmodel():
    # Build, train, and evaluate the model
    model.build_model()
    model.train_model(epochs=5)
    model.evaluate_model()

    # Save the model
    model.save_model("saved_model.h5")

# if model.hasmodel():
#     # Predict a single image
#     img_path = "/home/v3n0m/Datasets/Hackathon/plantvillage dataset/color/Grape___Black_rot/0a06c482-c94a-44d8-a895-be6fe17b8c06___FAM_B.Rot 5019.JPG"
#     model.display_prediction(img_path)