import os
import pickle

import numpy as np
import pandas as pd
import tensorflow
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import LSTM, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class MemoryNetWork:
    def __init__(self):
        self.max_length = None
        self.accuracy = None
        self.loss = None
        self.vocab_size = None
        self.embedding_dim = None
        self.num_classes = None
        self.X_padded = None
        self.X_encoded = None
        self.tokenizer = None
        self.y_encoded = None
        self.label_encoder = None
        self.signsAndSymptomsDict = {
            "Amoeba": {
                "name": "Amoeba",
                "signs": [
                    "Presence in contaminated water",
                    "Small, gelatinous-like body"
                ],
                "symptoms": [
                    "Diarrhea",
                    "Abdominal pain",
                    "Amoebic dysentery in severe cases"
                ]
            },
            "Aspergillus niger": {
                "name": "Aspergillus niger",
                "signs": [
                    "Black mold appearance on food or surfaces",
                    "Presence in damp environments"
                ],
                "symptoms": [
                    "Respiratory issues",
                    "Fungal infections in immunocompromised individuals",
                    "Allergic reactions"
                ]
            },
            "Candida albicans": {
                "name": "Candida albicans",
                "signs": [
                    "White, creamy colonies in culture",
                    "Thrush in the mouth"
                ],
                "symptoms": [
                    "Oral thrush",
                    "Vaginal yeast infections",
                    "Skin infections",
                    "Digestive issues in severe cases"
                ]
            },
            "covid19": {
                "name": "COVID-19",
                "signs": [
                    "Fever",
                    "Cough",
                    "Positive PCR test"
                ],
                "symptoms": [
                    "Loss of taste and smell",
                    "Shortness of breath",
                    "Fatigue",
                    "Body aches"
                ]
            },
            "Epidermophyton floccosum": {
                "name": "Epidermophyton floccosum",
                "signs": [
                    "Greenish-gray colonies in lab culture",
                    "Skin scaling"
                ],
                "symptoms": [
                    "Athlete's foot",
                    "Jock itch",
                    "Ringworm"
                ]
            },
            "Euglena": {
                "name": "Euglena",
                "signs": [
                    "Green, photosynthetic body",
                    "Eye spot present in cells"
                ],
                "symptoms": [
                    "Potential algae blooms affecting water quality"
                ]
            },
            "Hydra": {
                "name": "Hydra",
                "signs": [
                    "Small, tube-like body in freshwater",
                    "Tentacles around the mouth"
                ],
                "symptoms": [
                    "No direct symptoms in humans",
                    "Part of freshwater ecosystem balance"
                ]
            },
            "Malaria": {
                "name": "Malaria",
                "signs": [
                    "Detection of Plasmodium in blood smear",
                    "Presence of mosquito vectors in the area"
                ],
                "symptoms": [
                    "High fever",
                    "Chills and sweats",
                    "Headache and muscle pain",
                    "Fatigue"
                ]
            },
            "Paramecium": {
                "name": "Paramecium",
                "signs": [
                    "Ciliated body visible under a microscope",
                    "Fast movement in water"
                ],
                "symptoms": [
                    "No direct symptoms in humans",
                    "Important part of freshwater ecosystems"
                ]
            },
            "Rod Bacteria": {
                "name": "Rod Bacteria",
                "signs": [
                    "Rod-shaped cells visible under a microscope",
                    "Commonly found in soil, water, and on surfaces"
                ],
                "symptoms": [
                    "May cause various bacterial infections depending on type",
                    "Possible respiratory or digestive issues"
                ]
            },
            "Spherical Bacteria": {
                "name": "Spherical Bacteria",
                "signs": [
                    "Spherical or round cell shape under a microscope",
                    "Commonly found on skin and surfaces"
                ],
                "symptoms": [
                    "Various infections depending on species",
                    "Can cause skin, respiratory, or gastrointestinal issues"
                ]
            },
            "Spiral Bacteria": {
                "name": "Spiral Bacteria",
                "signs": [
                    "Spiral or corkscrew-shaped cells visible under a microscope",
                    "Commonly found in contaminated water or soil"
                ],
                "symptoms": [
                    "Potential digestive issues if ingested",
                    "Possible infections like Lyme disease"
                ]
            },
            "Trichophyton mentagrophytes": {
                "name": "Trichophyton mentagrophytes",
                "signs": [
                    "White, powdery colonies in culture",
                    "Infects hair, skin, and nails"
                ],
                "symptoms": [
                    "Itchy, scaly skin",
                    "Ringworm",
                    "Athlete's foot"
                ]
            },
            "Trichophyton rubrum": {
                "name": "Trichophyton rubrum",
                "signs": [
                    "Red-pigmented colonies in culture",
                    "Prefers skin and nail infections"
                ],
                "symptoms": [
                    "Chronic skin infections",
                    "Fungal nail infections",
                    "Scaling and redness"
                ]
            },
            "tuberculosis": {
                "name": "Tuberculosis",
                "signs": [
                    "Positive tuberculin skin test",
                    "Presence of tubercles on X-rays"
                ],
                "symptoms": [
                    "Persistent cough",
                    "Weight loss",
                    "Night sweats",
                    "Fever"
                ]
            },
            "Yeast": {
                "name": "Yeast",
                "signs": [
                    "Budding cells under a microscope",
                    "Commonly found on skin and mucous membranes"
                ],
                "symptoms": [
                    "Thrush",
                    "Yeast infections",
                    "Digestive imbalances if overgrown"
                ]
            }
        }

        self.signsAndSymptomsDict_ = {
            "Apple Scab": {
                "name": "Apple Scab",
                "signs": [
                    "Olive-green to brown lesions on leaves",
                    "Dark, velvety spots on fruit",
                ],
                "symptoms": ["Premature leaf drop", "Deformed fruit with scabs"],
            },
            "Apple Black Rot": {
                "name": "Apple Black Rot",
                "signs": [
                    "Brown, circular spots on leaves",
                    "Dark, rotten patches on fruit",
                ],
                "symptoms": ["Leaf yellowing", "Fruit decay and rot"],
            },
            "Apple Cedar Apple Rust": {
                "name": "Apple Cedar Apple Rust",
                "signs": [
                    "Yellow-orange spots on leaves",
                    "Galls on cedar trees nearby",
                ],
                "symptoms": ["Leaf defoliation", "Reduced fruit yield"],
            },
            "Apple Healthy": {
                "name": "Apple Healthy",
                "signs": [],
                "symptoms": ["Healthy foliage", "No visible spots or lesions"],
            },
            "Blueberry Healthy": {
                "name": "Blueberry Healthy",
                "signs": [],
                "symptoms": ["Healthy foliage", "No visible spots or lesions"],
            },
            "Cherry Powdery Mildew": {
                "name": "Cherry Powdery Mildew",
                "signs": ["White, powdery coating on leaves", "Deformed young leaves"],
                "symptoms": ["Leaf curling", "Reduced fruit production"],
            },
            "Cherry Healthy": {
                "name": "Cherry Healthy",
                "signs": [],
                "symptoms": ["Healthy foliage", "No visible spots or lesions"],
            },
            "Corn Cercospora Leaf Spot (Gray Leaf Spot)": {
                "name": "Corn Cercospora Leaf Spot (Gray Leaf Spot)",
                "signs": ["Gray or tan lesions on leaves", "Dark borders around spots"],
                "symptoms": ["Leaf wilting", "Reduced crop yield"],
            },
            "Corn Common Rust": {
                "name": "Corn Common Rust",
                "signs": ["Reddish-brown pustules on leaves", "Small raised spots"],
                "symptoms": ["Reduced photosynthesis", "Leaf death in severe cases"],
            },
            "Corn Northern Leaf Blight": {
                "name": "Corn Northern Leaf Blight",
                "signs": [
                    "Long, grayish lesions on leaves",
                    "Gray-green halos around lesions",
                ],
                "symptoms": ["Premature leaf death", "Reduced crop yield"],
            },
            "Corn Healthy": {
                "name": "Corn Healthy",
                "signs": [],
                "symptoms": ["Healthy foliage", "No visible spots or lesions"],
            },
            "Grape Black Rot": {
                "name": "Grape Black Rot",
                "signs": ["Dark brown lesions on leaves", "Rotten, shriveled fruit"],
                "symptoms": ["Leaf wilting", "Severe fruit loss"],
            },
            "Grape Esca (Black Measles)": {
                "name": "Grape Esca (Black Measles)",
                "signs": ["Black, sunken spots on fruit", "Striped leaves"],
                "symptoms": ["Leaf burn", "Fruit shriveling"],
            },
            "Grape Leaf Blight (Isariopsis Leaf Spot)": {
                "name": "Grape Leaf Blight (Isariopsis Leaf Spot)",
                "signs": ["Brown, angular spots on leaves", "Dark edges around spots"],
                "symptoms": ["Leaf yellowing", "Defoliation"],
            },
            "Grape Healthy": {
                "name": "Grape Healthy",
                "signs": [],
                "symptoms": ["Healthy foliage", "No visible spots or lesions"],
            },
            "Orange Huanglongbing (Citrus Greening)": {
                "name": "Orange Huanglongbing (Citrus Greening)",
                "signs": ["Yellow shoots", "Lopsided, bitter fruit"],
                "symptoms": ["Leaf yellowing", "Fruit deformation"],
            },
            "Peach Bacterial Spot": {
                "name": "Peach Bacterial Spot",
                "signs": [
                    "Dark, water-soaked spots on leaves",
                    "Sunken, scabby spots on fruit",
                ],
                "symptoms": ["Premature leaf drop", "Fruit loss"],
            },
            "Peach Healthy": {
                "name": "Peach Healthy",
                "signs": [],
                "symptoms": ["Healthy foliage", "No visible spots or lesions"],
            },
            "Pepper Bacterial Spot": {
                "name": "Pepper Bacterial Spot",
                "signs": ["Dark spots with yellow halos on leaves", "Fruit lesions"],
                "symptoms": ["Leaf wilting", "Reduced fruit yield"],
            },
            "Pepper Healthy": {
                "name": "Pepper Healthy",
                "signs": [],
                "symptoms": ["Healthy foliage", "No visible spots or lesions"],
            },
            "Potato Early Blight": {
                "name": "Potato Early Blight",
                "signs": [
                    "Brown, circular spots on leaves",
                    "Concentric rings in spots",
                ],
                "symptoms": ["Leaf death", "Reduced yield"],
            },
            "Potato Late Blight": {
                "name": "Potato Late Blight",
                "signs": [
                    "Dark, water-soaked spots on leaves",
                    "White mold on undersides",
                ],
                "symptoms": ["Rapid leaf death", "Potato tuber rot"],
            },
            "Potato Healthy": {
                "name": "Potato Healthy",
                "signs": [],
                "symptoms": ["Healthy foliage", "No visible spots or lesions"],
            },
            "Raspberry Healthy": {
                "name": "Raspberry Healthy",
                "signs": [],
                "symptoms": ["Healthy foliage", "No visible spots or lesions"],
            },
            "Soybean Healthy": {
                "name": "Soybean Healthy",
                "signs": [],
                "symptoms": ["Healthy foliage", "No visible spots or lesions"],
            },
            "Squash Powdery Mildew": {
                "name": "Squash Powdery Mildew",
                "signs": ["White, powdery spots on leaves", "Leaf curl"],
                "symptoms": ["Reduced photosynthesis", "Yield loss"],
            },
            "Strawberry Leaf Scorch": {
                "name": "Strawberry Leaf Scorch",
                "signs": ["Purple spots on leaves", "Margins turning brown"],
                "symptoms": ["Leaf withering", "Reduced fruit yield"],
            },
            "Strawberry Healthy": {
                "name": "Strawberry Healthy",
                "signs": [],
                "symptoms": ["Healthy foliage", "No visible spots or lesions"],
            },
            "Tomato Bacterial Spot": {
                "name": "Tomato Bacterial Spot",
                "signs": [
                    "Dark, water-soaked lesions on leaves",
                    "Fruit spots with scabs",
                ],
                "symptoms": ["Leaf yellowing", "Decreased fruit quality"],
            },
            "Tomato Early Blight": {
                "name": "Tomato Early Blight",
                "signs": [
                    "Brown, circular lesions on leaves",
                    "Target-like rings in spots",
                ],
                "symptoms": ["Leaf drop", "Reduced yield"],
            },
            "Tomato Late Blight": {
                "name": "Tomato Late Blight",
                "signs": ["Water-soaked spots on leaves", "White mold on underside"],
                "symptoms": ["Quick leaf die-off", "Rotting fruit"],
            },
            "Tomato Leaf Mold": {
                "name": "Tomato Leaf Mold",
                "signs": ["Yellow patches on top leaves", "Gray mold underneath"],
                "symptoms": ["Reduced fruit production", "Leaf drop"],
            },
            "Tomato Septoria Leaf Spot": {
                "name": "Tomato Septoria Leaf Spot",
                "signs": ["Small, dark spots on leaves", "Yellow halos around spots"],
                "symptoms": ["Leaf yellowing", "Yield reduction"],
            },
            "Tomato Spider Mites": {
                "name": "Tomato Spider Mites",
                "signs": ["Speckled leaves", "Fine webbing"],
                "symptoms": ["Leaf bronzing", "Stunted growth"],
            },
            "Tomato Target Spot": {
                "name": "Tomato Target Spot",
                "signs": [
                    "Dark, circular spots on leaves",
                    "Spots with target-like rings",
                ],
                "symptoms": ["Leaf defoliation", "Reduced yield"],
            },
            "Tomato Yellow Leaf Curl Virus": {
                "name": "Tomato Yellow Leaf Curl Virus",
                "signs": ["Yellowing, curled leaves", "Stunted plant growth"],
                "symptoms": ["Reduced fruit production", "Deformed growth"],
            },
            "Tomato Mosaic Virus": {
                "name": "Tomato Mosaic Virus",
                "signs": ["Mottled yellow-green leaves", "Wrinkled leaf surface"],
                "symptoms": ["Stunted growth", "Distorted fruit"],
            },
            "Tomato Healthy": {
                "name": "Tomato Healthy",
                "signs": [],
                "symptoms": ["Healthy foliage", "No visible spots or lesions"],
            },
        }
        self.preventionsAndCure = {
            "Amoeba": {
                "preventions": [
                    "Avoid drinking untreated water",
                    "Practice good hygiene, especially in areas with poor sanitation",
                    "Avoid swimming in contaminated water bodies"
                ],
                "cure": [
                    "Use prescribed anti-amoebic medications",
                    "Maintain hydration if symptoms like diarrhea occur",
                    "Seek medical advice for proper treatment"
                ]
            },
            "Aspergillus niger": {
                "preventions": [
                    "Keep indoor environments dry and well-ventilated",
                    "Remove mold from surfaces promptly",
                    "Avoid foods that show signs of mold"
                ],
                "cure": [
                    "Antifungal treatments as prescribed",
                    "Use of air purifiers in mold-prone areas",
                    "Proper cleaning and disinfection of surfaces"
                ]
            },
            "Candida albicans": {
                "preventions": [
                    "Practice good personal hygiene",
                    "Avoid excessive use of antibiotics",
                    "Maintain a balanced diet to support immune health"
                ],
                "cure": [
                    "Use antifungal medications",
                    "Maintain good oral and skin hygiene",
                    "Probiotics may help restore normal microbial balance"
                ]
            },
            "covid19": {
                "preventions": [
                    "Regular hand washing and sanitizing",
                    "Wear masks in crowded places",
                    "Stay updated with vaccinations"
                ],
                "cure": [
                    "Rest and hydration",
                    "Antiviral medications for severe cases as prescribed",
                    "Isolate to prevent spreading to others"
                ]
            },
            "Epidermophyton floccosum": {
                "preventions": [
                    "Avoid sharing personal items like towels",
                    "Keep skin dry and clean",
                    "Wear footwear in public showers or pools"
                ],
                "cure": [
                    "Topical or oral antifungal treatments",
                    "Maintain dryness in affected areas",
                    "Regular cleaning of infected areas"
                ]
            },
            "Euglena": {
                "preventions": [
                    "Avoid stagnant water that can harbor algae blooms",
                    "Proper water filtration for drinking water",
                    "Monitor and maintain water quality in aquariums"
                ],
                "cure": [
                    "Typically no cure needed; harmless to humans",
                    "Use chemical treatment in water if bloom is excessive",
                    "Ensure ecosystem balance in aquariums"
                ]
            },
            "Hydra": {
                "preventions": [
                    "Maintain water quality in aquariums or freshwater sources",
                    "Control water plants and algae to prevent overpopulation"
                ],
                "cure": [
                    "No treatment needed as it is generally non-harmful to humans",
                    "Maintain ecological balance in water systems"
                ]
            },
            "Malaria": {
                "preventions": [
                    "Use mosquito nets and repellents",
                    "Avoid mosquito bites, especially in endemic areas",
                    "Take antimalarial medications if traveling to high-risk areas"
                ],
                "cure": [
                    "Use antimalarial drugs as prescribed",
                    "Seek medical treatment promptly",
                    "Hydration and rest to manage symptoms"
                ]
            },
            "Paramecium": {
                "preventions": [
                    "Avoid drinking untreated water",
                    "Maintain cleanliness in water systems"
                ],
                "cure": [
                    "Generally harmless to humans",
                    "Maintain clean and filtered water in aquariums"
                ]
            },
            "Rod Bacteria": {
                "preventions": [
                    "Wash hands regularly",
                    "Avoid contact with contaminated surfaces",
                    "Disinfect surfaces that may harbor bacteria"
                ],
                "cure": [
                    "Antibiotic treatment for specific infections",
                    "Follow medical guidance for bacterial infections",
                    "Maintain personal and environmental hygiene"
                ]
            },
            "Spherical Bacteria": {
                "preventions": [
                    "Good personal hygiene practices",
                    "Regularly disinfect commonly touched surfaces",
                    "Proper food handling and preparation"
                ],
                "cure": [
                    "Antibiotic treatment as prescribed",
                    "Hydration and rest to support recovery",
                    "Medical consultation for appropriate treatment"
                ]
            },
            "Spiral Bacteria": {
                "preventions": [
                    "Avoid drinking untreated water",
                    "Cook meat thoroughly to avoid infection",
                    "Practice proper sanitation"
                ],
                "cure": [
                    "Antibiotics as prescribed by a healthcare provider",
                    "Supportive care for digestive symptoms",
                    "Maintain hydration if symptoms occur"
                ]
            },
            "Trichophyton mentagrophytes": {
                "preventions": [
                    "Avoid walking barefoot in public places",
                    "Keep skin and nails dry and clean",
                    "Do not share personal items like towels"
                ],
                "cure": [
                    "Topical antifungal treatments",
                    "Oral antifungal medications for severe cases",
                    "Regularly wash and dry affected areas"
                ]
            },
            "Trichophyton rubrum": {
                "preventions": [
                    "Maintain good foot hygiene",
                    "Wear breathable footwear",
                    "Avoid sharing personal items"
                ],
                "cure": [
                    "Apply antifungal creams or ointments",
                    "Use oral antifungals for severe infections",
                    "Keep affected skin areas clean and dry"
                ]
            },
            "Tuberculosis": {
                "preventions": [
                    "Vaccination with BCG (where applicable)",
                    "Avoid close contact with TB-infected individuals",
                    "Good ventilation in living spaces"
                ],
                "cure": [
                    "Long-term antibiotic regimen as prescribed",
                    "Isolation during infectious stages",
                    "Monitor and follow up with healthcare provider"
                ]
            },
            "Yeast": {
                "preventions": [
                    "Maintain balanced diet to support immune health",
                    "Practice good personal hygiene",
                    "Avoid excessive use of antibiotics"
                ],
                "cure": [
                    "Antifungal medications for infections",
                    "Maintain good hygiene in affected areas",
                    "Probiotics may help restore normal microbial balance"
                ]
            }
        }

        self.preventionsAndCure_ = {
            "Apple Scab": {
                "preventions": [
                    "Use disease-resistant apple varieties",
                    "Prune trees for better air circulation",
                    "Apply fungicides during wet seasons",
                ],
                "cure": [
                    "Remove infected leaves and fruits",
                    "Use fungicides as directed",
                    "Clear fallen leaves from around trees",
                ],
            },
            "Apple Black Rot": {
                "preventions": [
                    "Prune trees to remove dead wood",
                    "Use mulch to prevent soil splash onto fruit",
                    "Plant resistant apple varieties",
                ],
                "cure": [
                    "Remove and destroy infected fruits",
                    "Apply fungicides to control spread",
                    "Practice crop rotation if possible",
                ],
            },
            "Apple Cedar Apple Rust": {
                "preventions": [
                    "Plant rust-resistant apple varieties",
                    "Avoid planting apples near cedar trees",
                    "Remove galls from cedar trees",
                ],
                "cure": [
                    "Apply fungicides early in the season",
                    "Regularly inspect for signs of rust and remove affected leaves",
                ],
            },
            "Apple Healthy": {
                "preventions": [
                    "Maintain proper fertilization",
                    "Ensure adequate watering",
                    "Regularly prune and inspect for diseases",
                ],
                "cure": ["N/A"],
            },
            "Blueberry Healthy": {
                "preventions": [
                    "Ensure well-draining soil and acidic conditions",
                    "Regular watering without waterlogging",
                    "Apply mulch to retain moisture",
                ],
                "cure": ["N/A"],
            },
            "Cherry Powdery Mildew": {
                "preventions": [
                    "Avoid overhead watering",
                    "Use resistant cherry varieties",
                    "Provide adequate spacing for airflow",
                ],
                "cure": [
                    "Apply fungicides specific for powdery mildew",
                    "Prune affected leaves",
                    "Clear debris around the tree",
                ],
            },
            "Cherry Healthy": {
                "preventions": [
                    "Plant in sunny areas with good drainage",
                    "Water only at the base of plants",
                    "Prune regularly to improve air circulation",
                ],
                "cure": ["N/A"],
            },
            "Corn Cercospora Leaf Spot (Gray Leaf Spot)": {
                "preventions": [
                    "Rotate crops to avoid pathogen buildup",
                    "Use resistant corn varieties",
                    "Avoid working in fields when plants are wet",
                ],
                "cure": [
                    "Apply fungicides during the early stages",
                    "Remove infected leaves",
                    "Maintain proper plant spacing",
                ],
            },
            "Corn Common Rust": {
                "preventions": [
                    "Plant resistant corn varieties",
                    "Avoid overhead watering",
                    "Use crop rotation",
                ],
                "cure": [
                    "Apply fungicides at the first sign of rust",
                    "Remove severely infected plants",
                ],
            },
            "Corn Northern Leaf Blight": {
                "preventions": [
                    "Plant resistant corn varieties",
                    "Avoid working in wet fields",
                    "Use balanced fertilizer application",
                ],
                "cure": [
                    "Apply foliar fungicides",
                    "Remove infected leaves",
                    "Practice crop rotation",
                ],
            },
            "Corn Healthy": {
                "preventions": [
                    "Use good-quality seeds",
                    "Ensure proper crop spacing",
                    "Water as needed without over-irrigation",
                ],
                "cure": ["N/A"],
            },
            "Grape Black Rot": {
                "preventions": [
                    "Prune vines for airflow",
                    "Use disease-resistant grape varieties",
                    "Apply fungicides before fruit set",
                ],
                "cure": [
                    "Remove infected fruit and leaves",
                    "Spray with fungicide early in the season",
                ],
            },
            "Grape Esca (Black Measles)": {
                "preventions": [
                    "Avoid excessive watering",
                    "Use resistant varieties",
                    "Apply appropriate fungicides",
                ],
                "cure": [
                    "Prune infected parts",
                    "Apply fungicides as per local guidelines",
                ],
            },
            "Grape Leaf Blight (Isariopsis Leaf Spot)": {
                "preventions": [
                    "Ensure good air circulation",
                    "Prune vines",
                    "Use fungicides preventively",
                ],
                "cure": ["Remove infected leaves", "Apply fungicides at the onset"],
            },
            "Grape Healthy": {
                "preventions": [
                    "Proper vineyard sanitation",
                    "Regular pruning",
                    "Use resistant varieties if available",
                ],
                "cure": ["N/A"],
            },
            "Orange Huanglongbing (Citrus Greening)": {
                "preventions": [
                    "Use insect-proof screens",
                    "Practice orchard sanitation",
                    "Apply insecticidal soap or oils",
                ],
                "cure": [
                    "Remove and destroy infected plants",
                    "Apply systemic insecticides if necessary",
                ],
            },
            "Peach Bacterial Spot": {
                "preventions": [
                    "Plant resistant varieties",
                    "Avoid overhead irrigation",
                    "Ensure good airflow around plants",
                ],
                "cure": ["Remove infected leaves", "Apply copper-based fungicides"],
            },
            "Peach Healthy": {
                "preventions": [
                    "Maintain proper irrigation",
                    "Ensure good air circulation",
                    "Regularly inspect for disease symptoms",
                ],
                "cure": ["N/A"],
            },
            "Pepper Bacterial Spot": {
                "preventions": [
                    "Use disease-free seeds",
                    "Rotate crops annually",
                    "Ensure proper spacing between plants",
                ],
                "cure": ["Remove infected plants", "Apply fungicides if needed"],
            },
            "Pepper Healthy": {
                "preventions": [
                    "Practice crop rotation",
                    "Avoid overhead watering",
                    "Maintain soil health",
                ],
                "cure": ["N/A"],
            },
            "Potato Early Blight": {
                "preventions": [
                    "Use certified seed potatoes",
                    "Rotate crops annually",
                    "Destroy infected plant material",
                ],
                "cure": [
                    "Remove and destroy infected plants",
                    "Apply fungicides as directed",
                ],
            },
            "Potato Late Blight": {
                "preventions": [
                    "Rotate crops regularly",
                    "Use disease-free seeds",
                    "Avoid planting in infected soil",
                ],
                "cure": [
                    "Apply fungicides at early stages",
                    "Remove and destroy infected foliage",
                ],
            },
            "Potato Healthy": {
                "preventions": [
                    "Use certified seed potatoes",
                    "Keep soil well-drained",
                    "Avoid overwatering",
                ],
                "cure": ["N/A"],
            },
            "Raspberry Healthy": {
                "preventions": [
                    "Plant in well-draining soil",
                    "Provide good air circulation",
                    "Avoid overhead watering",
                ],
                "cure": ["N/A"],
            },
            "Soybean Healthy": {
                "preventions": [
                    "Plant in well-drained soil",
                    "Practice crop rotation",
                    "Use disease-free seeds",
                ],
                "cure": ["N/A"],
            },
            "Squash Powdery Mildew": {
                "preventions": [
                    "Avoid overhead watering",
                    "Use resistant squash varieties",
                    "Space plants to improve airflow",
                ],
                "cure": [
                    "Apply fungicides specific for powdery mildew",
                    "Remove infected leaves",
                ],
            },
            "Strawberry Leaf Scorch": {
                "preventions": [
                    "Avoid overhead watering",
                    "Provide good airflow around plants",
                    "Mulch to reduce soil splash",
                ],
                "cure": ["Remove infected leaves", "Apply fungicides if needed"],
            },
            "Strawberry Healthy": {
                "preventions": [
                    "Plant in well-draining soil",
                    "Ensure proper air circulation",
                    "Avoid wetting foliage when watering",
                ],
                "cure": ["N/A"],
            },
            "Tomato Bacterial Spot": {
                "preventions": [
                    "Use disease-free seeds",
                    "Avoid overhead watering",
                    "Space plants adequately",
                ],
                "cure": ["Remove infected leaves", "Apply bactericides as needed"],
            },
            "Tomato Early Blight": {
                "preventions": [
                    "Rotate crops to prevent pathogen buildup",
                    "Use resistant varieties",
                    "Avoid planting in infected soil",
                ],
                "cure": [
                    "Apply fungicides as directed",
                    "Remove and destroy infected leaves",
                ],
            },
            "Tomato Late Blight": {
                "preventions": [
                    "Avoid overhead watering",
                    "Use disease-free seeds",
                    "Space plants to ensure good airflow",
                ],
                "cure": [
                    "Remove infected plants",
                    "Apply fungicides at early signs of infection",
                ],
            },
            "Tomato Leaf Mold": {
                "preventions": [
                    "Prune plants to improve air circulation",
                    "Avoid overhead watering",
                    "Use fungicides preventively",
                ],
                "cure": [
                    "Apply fungicides for leaf mold",
                    "Remove infected leaves promptly",
                ],
            },
            "Tomato Septoria Leaf Spot": {
                "preventions": [
                    "Rotate crops to prevent infection buildup",
                    "Use resistant varieties if available",
                    "Water plants at the base",
                ],
                "cure": ["Apply fungicides for leaf spot", "Remove infected leaves"],
            },
            "Tomato Spider Mites": {
                "preventions": [
                    "Monitor plants for early signs",
                    "Avoid over-fertilizing",
                    "Use insecticidal soaps",
                ],
                "cure": [
                    "Apply miticides for spider mites",
                    "Prune and remove affected leaves",
                ],
            },
            "Tomato Target Spot": {
                "preventions": [
                    "Use crop rotation",
                    "Avoid overhead irrigation",
                    "Ensure good air circulation",
                ],
                "cure": ["Apply fungicides as directed", "Remove infected leaves"],
            },
            "Tomato Yellow Leaf Curl Virus": {
                "preventions": [
                    "Use disease-free seeds",
                    "Control whiteflies in the garden",
                    "Use reflective mulches",
                ],
                "cure": [
                    "Remove infected plants",
                    "Apply insecticides for whiteflies if necessary",
                ],
            },
            "Tomato Mosaic Virus": {
                "preventions": [
                    "Use disease-free seeds",
                    "Control aphid populations",
                    "Avoid overwatering",
                ],
                "cure": [
                    "Remove infected plants",
                    "Use virus-resistant varieties if available",
                ],
            },
            "Tomato Healthy": {
                "preventions": [
                    "Ensure proper spacing between plants",
                    "Avoid overhead watering",
                    "Use disease-free seeds",
                ],
                "cure": ["N/A"],
            },
        }

        self.memory = {
            "signs_and_symptoms": {},
            "preventions_and_cure": {},
            "model": None,
        }
        self.model = None
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.load_model()
        self.update_memory()
        self.prepare_data()
        self.split_data()

    def prepare_data(self):
        # Convert to a DataFrame
        data = []
        for disease, details in self.signsAndSymptomsDict.items():
            signs = " ".join(details["signs"])
            symptoms = " ".join(details["symptoms"])
            combined_text = f"{signs} {symptoms}"
            data.append({"text": combined_text, "label": disease})

        self.data = pd.DataFrame(data)

    def update_memory(self):
        self.memory = {
            "signs_and_symptoms": self.signsAndSymptomsDict,
            "preventions_and_cure": self.preventionsAndCure,
        }

    def get_signs(self, item):
        if item in self.signsAndSymptomsDict:
            return self.signsAndSymptomsDict[item]["signs"]
        return []

    def get_symptoms(self, item):
        if item in self.signsAndSymptomsDict:
            return self.signsAndSymptomsDict[item]["symptoms"]
        return []

    def get_cure(self, item):
        if item in self.preventionsAndCure:
            return self.preventionsAndCure[item]["cure"]
        return []

    def get_preventions(self, item):
        if item in self.preventionsAndCure:
            return self.preventionsAndCure[item]["preventions"]
        return []

    def rename_keys_with_names(self):
        # Renaming keys in signsAndSymptomsDict
        new_signs_and_symptoms = {}
        for key, value in self.signsAndSymptomsDict.items():
            name_key = value["name"]  # Use the "name" field as the new key
            new_signs_and_symptoms[name_key] = value

        # Renaming keys in preventionsAndCure
        new_preventions_and_cure = {}
        for key, value in self.preventionsAndCure.items():
            if key in self.signsAndSymptomsDict:
                name_key = self.signsAndSymptomsDict[key][
                    "name"
                ]  # Use the corresponding name
                new_preventions_and_cure[name_key] = value

        # Update the dictionaries with the new ones
        self.signsAndSymptomsDict = new_signs_and_symptoms
        self.preventionsAndCure = new_preventions_and_cure

    def split_data(self):
        # Prepare the data
        X = self.data["text"].values
        y = self.data["label"].values

        # Encode the labels
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(y)

        # Tokenization
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(X)
        self.X_encoded = self.tokenizer.texts_to_sequences(X)

        # Padding sequences
        self.max_length = max(len(x) for x in self.X_encoded)
        self.X_padded = pad_sequences(
            self.X_encoded, maxlen=self.max_length, padding="post"
        )

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_padded, self.y_encoded, test_size=0.2, random_state=42
        )
        # Parameters
        self.vocab_size = (
            len(self.tokenizer.word_index) + 1
        )  # Vocabulary size (+1 for padding token)
        self.embedding_dim = 50
        self.num_classes = len(self.label_encoder.classes_)

    def build_model(self):

        # Build the model
        model = Sequential()
        model.add(
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
            )
        )
        model.add(LSTM(64, return_sequences=True))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.num_classes, activation="softmax"))
        self.model = model

    def compile_model(
        self,
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    ):
        # Compile the model
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def load_model(
        self, model_json_path="", model_weights_path="", model_path="saved_nlp_model.h5"
    ):
        """Loads a pre-trained model from a JSON file."""
        # model_json_path = 'plantvillage_model.json'
        # model_weights_path = 'plantvillage_model_weights.h5'

        if os.path.exists(model_json_path) and os.path.exists(model_weights_path):
            with open(model_json_path, "r") as json_file:
                model_json = json_file.read()
            self.model = tensorflow.keras.models.model_from_json(model_json)
            self.model.load_weights(model_weights_path)
            self.compile_model()
            # if self.model is not None:
            #     self.evaluate_model()
            print("Model loaded from", model_json_path)
        elif model_path is not None and model_path != "" and os.path.exists(model_path):
            self.model = tensorflow.keras.models.load_model(model_path)
            print("Model loaded from", model_path)
            self.compile_model()
            # if self.model is not None:
            #     self.evaluate_model()
        else:
            self.model = None

    def model_summary(self):
        # Print model summary
        self.model.summary()

    def train_model(self):
        # Train the model
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=100,
            batch_size=2,
            validation_data=(self.X_test, self.y_test),
        )

        # Save history with pickle
        with open("nlp_training_history.pkl", "wb") as f:
            pickle.dump(history.history, f)

        return history

    def evaluate_model(self, save_path="nlp_evaluation_results.pkl"):
        # Evaluate the model
        self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test Accuracy: {self.accuracy:.2f}")

        # Save the evaluation results using pickle
        evaluation_results = {
            "Test Loss": self.loss,
            "Test Accuracy": self.accuracy
        }

        with open(save_path, "wb") as f:
            pickle.dump(evaluation_results, f)

    def load_and_plot_history(self, pickle_file="nlp_training_history.pkl", save_path="nlp_loaded_training_history_plot.png"):
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

    def predict_disease(self, signs_and_symptoms):
        # Ensure signs_and_symptoms is a string
        if isinstance(signs_and_symptoms, list):
            signs_and_symptoms = " ".join(signs_and_symptoms)

        # Convert the symptoms into sequences
        sequence = self.tokenizer.texts_to_sequences([signs_and_symptoms])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding="post")

        # Get the prediction probabilities
        prediction = self.model.predict(padded)

        # Get the top 5 predicted classes and their probabilities
        top_n = 5
        predicted_probabilities = prediction[
            0
        ]  # Assuming the prediction shape is (1, num_classes)

        # Get the indices of the top 5 probabilities
        top_indices = np.argsort(predicted_probabilities)[-top_n:][
            ::-1
        ]  # Sort and get top 5 indices

        # Prepare the results with disease names and their corresponding probabilities
        results = []
        for index in top_indices:
            disease_name = self.label_encoder.inverse_transform([index])[0]
            confidence = predicted_probabilities[index]
            results.append((disease_name, confidence))

        return results

    def save_model(self, model_path="saved_nlp_model.h5"):
        """Saves the trained model to the specified path."""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def hasmodel(self):
        return self.model is not None
    
    def get_merged_signs_and_symptoms(self):
        merged_list = []
        
        # Iterate over each pathogen in the dictionary
        for pathogen in self.signsAndSymptomsDict.values():
            # Merge signs and symptoms into one list
            merged_list.extend(pathogen.get('signs', []))
            merged_list.extend(pathogen.get('symptoms', []))
        
        return merged_list

# Instantiate the MemoryNetWork
network = MemoryNetWork()
force = False
if not network.hasmodel() or force:
    # Build and compile the model
    network.build_model()
    network.compile_model()

    # Train the model
    network.train_model()

    # Evaluate the model
    network.evaluate_model()

    network.save_model()

# # Example prediction
# test_input = ['Leaf yellowing', 'Deformed fruit with scabs']  # Example signs and symptoms input
# # Get predictions
# predicted_diseases = network.predict_disease(test_input)
#
# # Print results
# for disease, confidence in predicted_diseases:
#     print(f"Disease: {disease}, Confidence: {confidence:.2f}")
