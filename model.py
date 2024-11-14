import difflib
import os
import random

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
import pickle
import json


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

    def get_dataset_summary(self):
        """Return total images and number of unique classes in the dataset."""
        total_images = len(self.train_generator.filenames) + len(self.val_generator.filenames) + len(
            self.test_generator.filenames)
        num_classes = len(self.class_labels)

        return {
            "total_images": total_images,
            "num_classes": num_classes
        }

    def load_evaluation_results(self, pickle_file="animal_evaluation_results.pkl"):
        with open(pickle_file, "rb") as f:
            evaluation_results = pickle.load(f)

        print(
            f"Loaded Evaluation Results:\nTest Loss: {evaluation_results['Test Loss']}\nTest Accuracy: {evaluation_results['Test Accuracy']}")
        return evaluation_results

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
        checkpoint_cb = ModelCheckpoint("animal_disease_model.keras", save_best_only=True)
        early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )

        # Save history with pickle
        with open("animal_training_history.pkl", "wb") as f:
            pickle.dump(history.history, f)

        return history

    def evaluate_model(self, save_path="animal_evaluation_results.pkl"):
        results = self.model.evaluate(self.test_generator)
        test_loss, test_accuracy = results[0], results[1]
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # Save the evaluation results using pickle
        evaluation_results = {
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy
        }

        with open(save_path, "wb") as f:
            pickle.dump(evaluation_results, f)

    def save_model(self, filepath="animal_disease_model.h5"):
        self.model.save(filepath)

    def load_model(self, model_json_path="", model_weights_path="", model_path="animal_saved_model.h5"):
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
            self.evaluate_model()
            print("Model loaded from", model_json_path)
        elif model_path is not None and model_path != "" and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.loaded_from_file = True
            print("Model loaded from", model_path)
            self.compile()
            self.evaluate_model()
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

    def get_random_image_path(self, class_name):
        class_name = class_name.replace(" ", "_")
        subfolder_path = os.path.join(self.dataset_path, class_name)

        # Check if the exact class folder exists
        if not os.path.exists(subfolder_path):
            # If not, find the closest match
            closest_match = self.find_closest_class_name(class_name)
            if closest_match:
                subfolder_path = os.path.join(self.dataset_path, closest_match)
                print(f"Using closest match: '{closest_match}' instead of '{class_name}'.")
            else:
                raise ValueError(f"Class folder '{class_name}' does not exist in the dataset path.")

        if not os.path.exists(subfolder_path):
            raise ValueError(f"Class folder '{class_name}' does not exist in the dataset path.")

        # Get a list of all files in the class subfolder
        image_filenames = os.listdir(subfolder_path)
        if not image_filenames:
            raise ValueError(f"No images found in class folder '{class_name}'.")

        # Select a random image filename and return its full path
        random_image_filename = random.choice(image_filenames)
        return os.path.join(subfolder_path, random_image_filename)

    def find_closest_class_name(self, class_name):
        # Get all directories in the dataset path
        existing_class_names = os.listdir(self.dataset_path)

        # Use difflib to find the closest match
        closest_matches = difflib.get_close_matches(class_name, existing_class_names, n=1)

        if closest_matches:
            return closest_matches[0]  # Return the closest match
        return None  # Return None if no close matches found

    def load_and_plot_history(self, pickle_file="animal_training_history.pkl", save_path="loaded_training_history_plot.png"):
        """Loads history from a pickle file, plots, and saves accuracy and loss."""
        with open(pickle_file, "rb") as f:
            history_dict = pickle.load(f)

        # Plot accuracy and loss
        epochs = range(1, len(history_dict['accuracy']) + 1)

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, history_dict['accuracy'], 'bo-', label='Training accuracy')
        plt.plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history_dict['loss'], 'bo-', label='Training loss')
        plt.plot(epochs, history_dict['val_loss'], 'r-', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()

        # Save the plot
        plt.savefig(save_path)
        plt.show()


# Instantiate the class
model = PlantVillageModel(dataset_path="./animal/color")
# model.load_evaluation_results()
# model.load_and_plot_history()
if not model.hasmodel():
    # Build, train, and evaluate the model
    model.build_model()
    model.train_model(epochs=5)
    model.evaluate_model()

    # Save the model
    model.save_model("animal_saved_model.h5")
# if model.hasmodel():
#     # Predict a single image
#     img_path = "/home/v3n0m/Datasets/Hackathon/plantvillage dataset/color/Grape___Black_rot/0a06c482-c94a-44d8-a895-be6fe17b8c06___FAM_B.Rot 5019.JPG"
#     model.display_prediction(img_path)
