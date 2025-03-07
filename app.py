from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import io
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import base64
import json
import mediapipe as mp

import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import os
from datetime import datetime
from fpdf import FPDF  # Library for creating PDFs
from fpdf.enums import XPos, YPos
import tempfile

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Image as RLImage, Paragraph
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
import requests

from dotenv import load_dotenv
from linkedin_scraper import scrape_linkedin_profile
from llama_cpp import Llama

import pandas as pd
from sklearn import linear_model
import fitz  # PyMuPDF
from io import BytesIO

import logging

logging.getLogger("pdfminer").setLevel(logging.WARNING)  # Suppress debug logs


app = Flask(__name__)
CORS(app)  # Enable CORS

################################################################

#LINKEDIN ANALYSIS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Initialize LLaMA model
llm = Llama(model_path="D:/models/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048, n_gpu_layers=50)

def analyze_profile(profile_data):
    """Generate an analysis report for the given profile data."""
    prompt = f"Analyze the following LinkedIn profile and suggest improvements, also mention details of the user, and do analysis. At the start of response, rate the profile from 1-10:\n\n{json.dumps(profile_data, indent=2)}"
    output = llm.create_completion(prompt, max_tokens=1000)
    return output["choices"][0]["text"]

@app.route('/api/analyze-linkedin', methods=['POST', 'OPTIONS'])
def analyze_linkedin():
    if request.method == 'OPTIONS':
        # Handle preflight request
        return jsonify({'status': 'success'}), 200

    try:
        print("Received request to /api/analyze-linkedin")  # Debug statement
        data = request.get_json()
        print("Request data:", data)  # Debug statement

        profile_url = data.get('profile_url')
        if not profile_url:
            print("No profile URL provided")  # Debug statement
            return jsonify({'status': 'error', 'message': 'No LinkedIn profile URL provided.'}), 400

        print("Scraping LinkedIn profile...")  # Debug statement
        formatted_text, image_url = scrape_linkedin_profile(profile_url)

        if not formatted_text:
            print("Failed to scrape profile data")  # Debug statement
            return jsonify({'status': 'error', 'message': 'Failed to scrape LinkedIn profile data. Ensure the profile is public.'}), 400

        print("Profile data scraped successfully")  # Debug statement
        profile_data = {"profile_text": formatted_text, "image_url": image_url}

        print("Analyzing profile...")  # Debug statement
        analysis = analyze_profile(profile_data)

        response = {
            'status': 'success',
            'analysis': analysis,
            'profile_image_url': image_url
        }
        print("Response:", response)  # Debug statement
        return jsonify(response)

    except Exception as e:
        print(f"Error analyzing LinkedIn profile: {e}")  # Debug statement
        return jsonify({'status': 'error', 'message': str(e)}), 500




####################################################################

# CV ANALYSIS PART


# Load and train the personality prediction model
class TrainModel:
    def __init__(self):
        self.model = None

    def train(self):
        # Load the training dataset
        data = pd.read_csv('training_dataset.csv')
        array = data.values

        # Convert gender to numerical values
        for i in range(len(array)):
            if array[i][0] == "Male":
                array[i][0] = 1
            else:
                array[i][0] = 0

        # Prepare features and target
        df = pd.DataFrame(array)
        maindf = df[[0, 1, 2, 3, 4, 5, 6]]  # Features: gender, age, openness, neuroticism, conscientiousness, agreeableness, extraversion
        mainarray = maindf.values
        temp = df[7]  # Target: personality
        train_y = temp.values

        # Train the logistic regression model
        self.model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        self.model.fit(mainarray, train_y)

    def predict(self, test_data):
        try:
            test_predict = list(map(int, test_data))  # Convert input to integers
            y_pred = self.model.predict([test_predict])
            return y_pred[0]
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

# Initialize the model
model = TrainModel()
model.train()

def analyze_cv_with_llm(cv_data, predicted_personality):
    """Generate an analysis report for the given CV data and predicted personality. """
    prompt = f"""
    Analyze the following CV and predicted personality traits in detail mentioning details in the CV. Provide suggestions for improvement, highlight strengths, and mention areas for development. At the end of the response, rate the CV from 1-10:

    CV Data:
    {json.dumps(cv_data, indent=2)}

    Predicted Personality:
    {predicted_personality}
    """
    output = llm.create_completion(prompt, max_tokens=3000, stop=["\n\n"])
    return output["choices"][0]["text"]


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF while ensuring all words are captured and sorted correctly."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                words = page.get_text("words")  # Extract all words
                
                if not words:
                    print(f"[WARNING] No words extracted on page {page_num + 1}. Trying alternative methods...")
                
                # Sort words by vertical position first, then horizontal
                words.sort(key=lambda w: (w[1], w[0]))
                
                # Reconstruct the text
                page_text = " ".join(word[4] for word in words)
                text += page_text + "\n\n"
    
        with open("debug_extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(resume_data['text'])


    except Exception as e:
        print(f"[ERROR] Error reading PDF: {e}")

    print("[DEBUG] Full Extracted Text:\n", text[:1000])  # Print first 1000 characters for debugging
    return text.strip()

# New endpoint for CV analysis
@app.route('/api/analyze-cv', methods=['POST'])
def analyze_cv():
    try:
        print("\n[DEBUG] Received request for CV analysis.")

        if 'resume' not in request.files:
            print("[ERROR] No resume file uploaded.")
            return jsonify({'status': 'error', 'message': 'No resume file uploaded.'}), 400

        resume_file = request.files['resume']
        if resume_file.filename == '':
            print("[ERROR] No selected file.")
            return jsonify({'status': 'error', 'message': 'No selected file.'}), 400

        print(f"[DEBUG] Uploaded file name: {resume_file.filename}")

        # Ensure uploads directory exists
        uploads_dir = 'uploads'
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
            print("[DEBUG] Created uploads directory.")

        resume_path = os.path.join(uploads_dir, resume_file.filename)
        resume_file.save(resume_path)
        print(f"[DEBUG] Saved resume at: {resume_path}")

        # Extract text using pdfplumber
        resume_data = {}
        # Extract text using PyMuPDF (fitz)
        try:
            extracted_text = extract_text_from_pdf(resume_path)  # Use the improved function
            resume_data['text'] = extracted_text.strip() if extracted_text else None
            print(f"[DEBUG] Full Extracted Resume Text:\n{resume_data['text']}")
        except Exception as e:
            print(f"[ERROR] Error reading PDF: {e}")
            os.remove(resume_path)
            return jsonify({'status': 'error', 'message': f'Error reading PDF: {str(e)}'}), 500
            if not resume_data.get('text'):
                print("[ERROR] Could not extract text from PDF.")
                os.remove(resume_path)
                return jsonify({'status': 'error', 'message': 'Could not extract text from PDF.'}), 400

        # Get personality prediction inputs
        required_fields = ['gender', 'age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']
        personality_inputs = [request.form.get(field) for field in required_fields]

        print(f"[DEBUG] Received personality inputs: {personality_inputs}")

        if None in personality_inputs:
            print("[ERROR] Missing personality inputs.")
            os.remove(resume_path)
            return jsonify({'status': 'error', 'message': 'All personality inputs are required.'}), 400

        predicted_personality = model.predict(personality_inputs)
        print(f"[DEBUG] Predicted Personality: {predicted_personality}")

        if not predicted_personality:
            print("[ERROR] Failed to predict personality.")
            os.remove(resume_path)
            return jsonify({'status': 'error', 'message': 'Failed to predict personality.'}), 500

        #llm_analysis = analyze_cv_with_llm(resume_data, predicted_personality)
        #print(f"[DEBUG] LLM Analysis Output:\n{llm_analysis[:500]}")  # Print first 500 characters for verification

        os.remove(resume_path)
        print("[DEBUG] Deleted resume file after processing.")


        return jsonify({
            'status': 'success',
            'resume_data': resume_data,
            'predicted_personality': predicted_personality,
            #'llm_analysis': llm_analysis
        })

    except Exception as e:
        print(f"[ERROR] Error analyzing CV: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500




# Constants for audio processing
MAX_LEN = 100
N_MFCC = 13
AUDIO_MODEL_PATH = "C:/Users/HP/OneDrive/Desktop/PEERS/AUDIO/big_five_audio_lstm.h5"  # Update this with your correct model path

# Load the LSTM model for audio processing
if os.path.exists(AUDIO_MODEL_PATH):
    try:
        audio_model = load_model(AUDIO_MODEL_PATH, custom_objects={"mse": MeanSquaredError()})
        print("Audio LSTM model loaded successfully.")
    except Exception as e:
        print(f"Error loading audio model: {str(e)}")
else:
    print(f"Error: Audio model not found at {AUDIO_MODEL_PATH}")




# Define Mediapipe components
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define the CNN model class
class CNN(nn.Module):
    def __init__(self, k):
        super(CNN, self).__init__()
        self.l1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l2 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32)
        self.fc1 = nn.Linear(32 * 48 * 48, k)

    def forward(self, x):
        x = self.pool(F.relu(self.l1(x)))
        x = self.pool(F.relu(self.l2(x)))
        x = x.view(-1, 32 * 48 * 48)
        x = self.fc1(x)
        return x

# Load your models here
eye_model = CNN(3)
eyebrow_model = CNN(3)
jaw_model = CNN(4)
mouth_model = CNN(3)
nose_model = CNN(3)

eye_model.load_state_dict(torch.load("C:/Users/HP/OneDrive/Desktop/gitt/Face2Fate/src/demo/eye_model.pt", weights_only=True))
eyebrow_model.load_state_dict(torch.load("C:/Users/HP/OneDrive/Desktop/gitt/Face2Fate/src/demo/eyebrow_model.pt", weights_only=True))
jaw_model.load_state_dict(torch.load("C:/Users/HP/OneDrive/Desktop/gitt/Face2Fate/src/demo/jaw_model.pt", weights_only=True))
mouth_model.load_state_dict(torch.load("C:/Users/HP/OneDrive/Desktop/gitt/Face2Fate/src/demo/mouth_model.pt", weights_only=True))
nose_model.load_state_dict(torch.load("C:/Users/HP/OneDrive/Desktop/gitt/Face2Fate/src/demo/nose_model.pt", weights_only=True))

# Load the analysis.json file
with open('C:/Users/HP/OneDrive/Desktop/gitt/Face2Fate/src/demo/data/analysis.json', 'r') as f:
    shape_descriptions = json.load(f)

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data provided.'}), 400

        # Decode base64 and convert to RGB
        img_data = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_cv = np.array(img).astype(np.uint8)

        # Initialize Mediapipe Face Mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            # Process the image
            results = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))

            if not results.multi_face_landmarks:
                return jsonify({'status': 'error', 'message': 'No faces detected.'}), 400

            feature_points = []  # To store feature points for drawing
            # Draw landmarks on the image and store the points for model predictions
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    # Get the coordinates of the landmarks
                    x = int(landmark.x * img_cv.shape[1])
                    y = int(landmark.y * img_cv.shape[0])
                    # Draw the landmark points on the image
                    cv2.circle(img_cv, (x, y), 2, (0, 255, 0), -1)  # Green circle for landmarks
                    feature_points.append((x, y))

            # Assuming we want to extract features based on certain landmark points
            if feature_points:
                # Example: extracting the region for eye predictions (you can adjust the indices based on your needs)
                left_eye_region = img_cv[feature_points[36][1]:feature_points[41][1], feature_points[36][0]:feature_points[39][0]]  # Eye region for prediction
                left_eye_img_resized = cv2.resize(left_eye_region, (200, 200))

                preprocess = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = preprocess(Image.fromarray(cv2.cvtColor(left_eye_img_resized, cv2.COLOR_BGR2RGB))).unsqueeze(0)

                # Pass through models
                eyebrow = eyebrow_model(img_tensor)
                eye = eye_model(img_tensor)
                nose = nose_model(img_tensor)
                mouth = mouth_model(img_tensor)
                jaw = jaw_model(img_tensor)

                # Get shapes
                eyebrow_shape = ["Arch", "Circle", "Straight"][torch.argmax(eyebrow).item()]
                eye_shape = ["Big", "Silt", "Small"][torch.argmax(eye).item()]
                nose_shape = ["Long", "Small", "Wide"][torch.argmax(nose).item()]
                mouth_shape = ["Medium", "Small", "Thick"][torch.argmax(mouth).item()]
                jaw_shape = ["Circle", "Oval", "Square", "Triangle"][torch.argmax(jaw).item()]

                # Prepare the results using the loaded JSON data
                def get_shape_analysis(shape_name, feature_name):
                    for region in shape_descriptions['face_regions']:
                        if region['name'] == shape_name:
                            for feature in region['features']:
                                if feature['name'] == feature_name:
                                    return feature['analysis']
                    return "Description not found."

                results = {
                    'eyebrow': {
                        'shape': eyebrow_shape,
                        'description': get_shape_analysis('eyebrows', eyebrow_shape)
                    },
                    'eye': {
                        'shape': eye_shape,
                        'description': get_shape_analysis('eyes', eye_shape)
                    },
                    'nose': {
                        'shape': nose_shape,
                        'description': get_shape_analysis('nose', nose_shape)
                    },
                    'mouth': {
                        'shape': mouth_shape,
                        'description': get_shape_analysis('mouth', mouth_shape)
                    },
                    'jaw': {
                        'shape': jaw_shape,
                        'description': get_shape_analysis('face', jaw_shape)  # Use 'face' for jaw analysis
                    }
                }

        # Convert modified image back to base64
        _, buffer = cv2.imencode('.png', img_cv)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare the response with results and processed image
        return jsonify({'status': 'success', 'results': results, 'image': img_base64})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    try:
        # Check if an audio file is provided
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'No audio file provided.'}), 400

        audio_file = request.files['audio']
        if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.ogg')):
            return jsonify({'status': 'error', 'message': 'Invalid audio file format. Supported formats: .wav, .mp3, .ogg'}), 400

        # Save the uploaded audio temporarily
        audio_path = "temp_audio.wav"
        audio_file.save(audio_path)

        # Function to extract MFCC features
        def extract_mfcc_sequence(audio_path):
            try:
                y, sr = librosa.load(audio_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
                padded_mfcc = np.zeros((N_MFCC, MAX_LEN))
                mfcc_len = min(mfcc.shape[1], MAX_LEN)
                padded_mfcc[:, :mfcc_len] = mfcc[:, :mfcc_len]
                return padded_mfcc.T  # Shape: (MAX_LEN, N_MFCC)
            except Exception as e:
                raise ValueError(f"Error extracting MFCC features: {e}")

        # Extract features and reshape
        features = extract_mfcc_sequence(audio_path).reshape(1, MAX_LEN, N_MFCC)

        # Make predictions
        predictions = audio_model.predict(features)[0]

        # Adjust predictions to increase confidence scores
        adjusted_predictions = 0.5 + (predictions - 0.5) * 1.6
        adjusted_predictions = np.clip(adjusted_predictions, 0, 1)

        # Big Five Traits
        BIG_FIVE = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
        results = {trait: round(float(score), 2) for trait, score in zip(BIG_FIVE, adjusted_predictions)}

        # Return the predictions as JSON
        return jsonify({'status': 'success', 'results': results})
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred. Please try again.'}), 500



# Add Zodiac and Chinese Zodiac logic to the Flask API

# Zodiac signs with date ranges
ZODIAC_SIGNS = [
    ("Capricorn", (12, 22), (1, 19)),
    ("Aquarius", (1, 20), (2, 18)),
    ("Pisces", (2, 19), (3, 20)),
    ("Aries", (3, 21), (4, 19)),
    ("Taurus", (4, 20), (5, 20)),
    ("Gemini", (5, 21), (6, 20)),
    ("Cancer", (6, 21), (7, 22)),
    ("Leo", (7, 23), (8, 22)),
    ("Virgo", (8, 23), (9, 22)),
    ("Libra", (9, 23), (10, 22)),
    ("Scorpio", (10, 23), (11, 21)),
    ("Sagittarius", (11, 22), (12, 21)),
]

# Chinese Zodiac signs
CHINESE_ZODIAC = [
    "Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake", 
    "Horse", "Goat", "Monkey", "Rooster", "Dog", "Pig"
]

# Load zodiac and Chinese zodiac data from JSON
JSON_FILE_PATH = os.path.join('C:\\Projects\\brainwave\\frontend\\src\\components', 'zodiac_chinese_signs.json')

try:
    with open(JSON_FILE_PATH, 'r') as file:
        zodiac_data = json.load(file)
        ZODIAC_SIGNS = zodiac_data.get('zodiac_signs', [])
        CHINESE_ZODIAC_SIGNS = zodiac_data.get('chinese_signs', [])  # Corrected key here
except Exception as e:
    print(f"Error loading JSON file: {e}")
    ZODIAC_SIGNS = []
    CHINESE_ZODIAC_SIGNS = []

    
@app.route('/api/calculate-signs', methods=['POST'])
def calculate_signs():
    try:
        data = request.get_json()
        dob = data.get('dob')  # Expected format: 'MM/DD/YYYY'

        if not dob:
            return jsonify({'status': 'error', 'message': 'DOB not provided.'}), 400

        # Validate and parse DOB format (MM/DD/YYYY)
        try:
            dob_obj = datetime.strptime(dob, '%m/%d/%Y')  # Parse the DOB
            dob = dob_obj.strftime('%Y-%m-%d')  # Convert to 'YYYY-MM-DD' format
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid DOB format. Use MM/DD/YYYY.'}), 400

        # Parse the DOB into year, month, and day
        year, month, day = map(int, dob.split('-'))

        # Zodiac sign calculation
        def get_zodiac_sign(month, day):
            for sign_data in ZODIAC_SIGNS:
                start_month, start_day = map(int, sign_data['date_range']['start'].split('-'))
                end_month, end_day = map(int, sign_data['date_range']['end'].split('-'))
                if (month == start_month and day >= start_day) or \
                   (month == end_month and day <= end_day) or \
                   (start_month < month < end_month):
                    return sign_data
            # Default to Capricorn if no match found
            return next(sign for sign in ZODIAC_SIGNS if sign['name'] == 'Capricorn')

        # Chinese Zodiac calculation
        def get_chinese_zodiac(year):
            if not CHINESE_ZODIAC_SIGNS:
                raise ValueError("Chinese zodiac signs not properly loaded.")

            index = year % 12
            if index < 0 or index >= len(CHINESE_ZODIAC_SIGNS):
                raise IndexError("Calculated index for Chinese Zodiac is out of range.")
            return CHINESE_ZODIAC_SIGNS[index]


        # Get signs and descriptions
        zodiac_sign_data = get_zodiac_sign(month, day)
        chinese_zodiac_data = get_chinese_zodiac(year)

        # Return the results
        return jsonify({
            'status': 'success',
            'zodiac_sign': zodiac_sign_data,
            'chinese_zodiac': chinese_zodiac_data
        })

    except Exception as e:
        print(f"Error calculating signs: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
@app.route('/api/find-lookalike', methods=['POST'])
def find_lookalike():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data provided.'}), 400

        # Decode base64 and convert to RGB
        img_data = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_cv = np.array(img).astype(np.uint8)

        # Initialize Mediapipe Face Mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            # Process the image
            results = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))

            if not results.multi_face_landmarks:
                return jsonify({'status': 'error', 'message': 'No faces detected.'}), 400

            feature_points = []
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * img_cv.shape[1])
                    y = int(landmark.y * img_cv.shape[0])
                    feature_points.append((x, y))

            # Predict feature shapes
            if feature_points:
                # Extract a representative region (example: left eye region)
                left_eye_region = img_cv[
                    feature_points[36][1]:feature_points[41][1],
                    feature_points[36][0]:feature_points[39][0]
                ]
                left_eye_img_resized = cv2.resize(left_eye_region, (200, 200))

                preprocess = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = preprocess(Image.fromarray(cv2.cvtColor(left_eye_img_resized, cv2.COLOR_BGR2RGB))).unsqueeze(0)

                eyebrow = eyebrow_model(img_tensor)
                eye = eye_model(img_tensor)
                nose = nose_model(img_tensor)
                mouth = mouth_model(img_tensor)
                jaw = jaw_model(img_tensor)

                features = {
                    'eyebrow': ["Arch", "Circle", "Straight"][torch.argmax(eyebrow).item()],
                    'eye': ["Big", "Silt", "Small"][torch.argmax(eye).item()],
                    'nose': ["Long", "Small", "Wide"][torch.argmax(nose).item()],
                    'mouth': ["Medium", "Small", "Thick"][torch.argmax(mouth).item()],
                    'jaw': ["Circle", "Oval", "Square", "Triangle"][torch.argmax(jaw).item()]
                }

            # Match with celebrity database
            with open('C:\\Projects\\brainwave\\frontend\\src\\components\\celebrity_data.json') as f:
                celeb_data = json.load(f)

            matches = []
            for celeb in celeb_data:
                if celeb['gender'].lower() != data.get('gender', '').lower():
                    continue  # Skip if gender doesn't match

                score = 0
                total_weight = 0
                weights = {'eyebrow': 1, 'eye': 2, 'nose': 2, 'mouth': 1, 'jaw': 3}

                for feature, weight in weights.items():
                    if celeb['features'][feature] == features[feature]:
                        score += weight
                    total_weight += weight

                match_percentage = (score / total_weight) * 100
                matches.append({
                    'name': celeb['name'],
                    'gender': celeb['gender'],
                    'features': celeb['features'],
                    'match_percentage': round(match_percentage, 2)
                })

            # Sort matches by percentage (descending) and name (alphabetically)
            matches = sorted(matches, key=lambda x: (-x['match_percentage'], x['name']))

            # Return top 3 matches
            if matches:
                return jsonify({'status': 'success', 'matches': matches[:3]})
            else:
                return jsonify({'status': 'error', 'message': 'No matching celebrities found.'})

    except Exception as e:
        print(f"Error finding lookalike: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500




def get_celebrity_image(celebrity_name):
    """Fetches the celebrity's image URL from Wikipedia API."""
    try:
        search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{celebrity_name.replace(' ', '_')}"
        response = requests.get(search_url).json()
        
        if 'thumbnail' in response and 'source' in response['thumbnail']:
            return response['thumbnail']['source']  # Return image URL
        return None
    except Exception as e:
        print(f"Error fetching image for {celebrity_name}: {e}")
        return None

def generate_pdf_report(name, email, number, zodiac_results, facial_features, lookalike_results, labeled_image_base64):
    try:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=landscape(letter))  # Set landscape mode
        width, height = landscape(letter)

        # **Set a custom AI-styled background**
        background_path = "C:/Projects/brainwave/backend/ai_robotic_background.jpg"
        if os.path.exists(background_path):
            pdf.drawImage(background_path, 0, 0, width=width, height=height, mask='auto')

        # **Title**
        pdf.setFont("Helvetica-Bold", 20)
        pdf.setFillColor(colors.white)  # White text on dark background
        pdf.drawCentredString(width / 2, height - 40, "AI Personality & Lookalike Report")

        pdf.setFillColor(colors.black)  # Reset text color
        pdf.setFont("Helvetica", 12)
        pdf.drawString(50, height - 80, f"Name: {name}")
        pdf.drawString(50, height - 100, f"Email: {email}")
        pdf.drawString(50, height - 120, f"Phone: {number}")

        # **Zodiac Analysis**
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, height - 150, "Zodiac Analysis")

        zodiac_table_data = [
            ["Zodiac Sign", zodiac_results["zodiac_sign"]["name"]],
            ["Element", zodiac_results["zodiac_sign"]["element"]],
            ["Description", Paragraph(zodiac_results["zodiac_sign"]["description"], getSampleStyleSheet()["Normal"])],
            ["Chinese Zodiac", zodiac_results["chinese_zodiac"]["name"]],
            ["Description", Paragraph(zodiac_results["chinese_zodiac"]["description"], getSampleStyleSheet()["Normal"])],
        ]
        zodiac_table = Table(zodiac_table_data, colWidths=[150, 400])
        zodiac_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        zodiac_table.wrapOn(pdf, width, height)
        zodiac_table.drawOn(pdf, 50, height - 250)

        # **Facial Features Table**
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, height - 280, "Facial Feature Analysis")

        facial_table_data = [["Feature", "Shape", "Personality Trait"]]
        for feature, values in facial_features.items():
            facial_table_data.append([
                feature.capitalize(), 
                values["shape"], 
                Paragraph(values["description"], getSampleStyleSheet()["Normal"])  # Auto-wrap text
            ])

        facial_table = Table(facial_table_data, colWidths=[100, 100, 400])
        facial_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        facial_table.wrapOn(pdf, width, height)
        facial_table.drawOn(pdf, 50, height - 420)

        # **Lookalike Results**
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, height - 450, "Celebrity Lookalike Matches")

        x_position = 50
        y_position = height - 520

        for match in lookalike_results:
            celeb_name = match["name"]
            match_percentage = match["match_percentage"]
            celeb_image_url = get_celebrity_image(celeb_name)

            if celeb_image_url:
                try:
                    image_response = requests.get(celeb_image_url, stream=True)
                    img = ImageReader(image_response.raw)
                    pdf.drawImage(img, x_position, y_position, width=100, height=100)  # Display horizontally
                except Exception as e:
                    print(f"Error displaying image for {celeb_name}: {e}")

            # Display Name & Match %
            pdf.drawString(x_position, y_position - 20, f"{celeb_name} ({match_percentage}%)")
            x_position += 180  # Move right for next image

        # **Personality Analysis**
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y_position - 150, "Personalized Personality Analysis")
        pdf.setFont("Helvetica", 12)

        personality_text = (
            f"Based on your facial features, you are {facial_features['jaw']['description'].lower()}. "
            f"Your eyes suggest that you are {facial_features['eye']['description'].lower()}, "
            f"while your nose indicates {facial_features['nose']['description'].lower()}. "
            f"Overall, you have a {zodiac_results['zodiac_sign']['description'].lower()} personality, "
            f"blended with the characteristics of a {zodiac_results['chinese_zodiac']['description'].lower()}."
        )
        
        paragraph = Paragraph(personality_text, getSampleStyleSheet()["Normal"])
        paragraph.wrapOn(pdf, width - 100, height)
        paragraph.drawOn(pdf, 50, y_position - 200)

        # **Save PDF**
        pdf.save()
        buffer.seek(0)
        return buffer

    except Exception as e:
        raise ValueError(f"Error generating PDF: {str(e)}")


@app.route('/api/process-all', methods=['POST'])
def process_all():
    try:
        data = request.get_json()
        
        # Extract inputs
        name = data.get('name')
        email = data.get('email')
        dob = data.get('dob')  # Can be 'YYYY-MM-DD' or 'MM/DD/YYYY'
        number = data.get('number')
        image_data = data.get('image')
        gender = data.get('gender')
        
        if not dob or not image_data or not name or not email or not gender:
            return jsonify({'status': 'error', 'message': 'Missing required inputs.'}), 400

        # Process DOB for zodiac signs
        def calculate_zodiac_signs(dob):
            try:
                # Support both 'MM/DD/YYYY' and 'YYYY-MM-DD' formats
                try:
                    dob_obj = datetime.strptime(dob, '%m/%d/%Y')
                except ValueError:
                    dob_obj = datetime.strptime(dob, '%Y-%m-%d')
                
                dob_str = dob_obj.strftime('%Y-%m-%d')  # Standardize to 'YYYY-MM-DD'
                year, month, day = map(int, dob_str.split('-'))

                # Zodiac sign calculation
                def get_zodiac_sign(month, day):
                    for sign_data in ZODIAC_SIGNS:
                        start_month, start_day = map(int, sign_data['date_range']['start'].split('-'))
                        end_month, end_day = map(int, sign_data['date_range']['end'].split('-'))
                        if (month == start_month and day >= start_day) or \
                           (month == end_month and day <= end_day) or \
                           (start_month < month < end_month):
                            return sign_data
                    return next(sign for sign in ZODIAC_SIGNS if sign['name'] == 'Capricorn')

                # Chinese Zodiac calculation
                def get_chinese_zodiac(year):
                    if not CHINESE_ZODIAC_SIGNS:
                        raise ValueError("Chinese zodiac signs not properly loaded.")

                    index = year % 12
                    if index < 0 or index >= len(CHINESE_ZODIAC_SIGNS):
                        raise IndexError("Calculated index for Chinese Zodiac is out of range.")
                    return CHINESE_ZODIAC_SIGNS[index]

                zodiac_sign_data = get_zodiac_sign(month, day)
                chinese_zodiac_data = get_chinese_zodiac(year)

                return {
                    'zodiac_sign': {
                        'name': zodiac_sign_data['name'],
                        'description': f"{zodiac_sign_data['name']} - {zodiac_sign_data.get('description', 'Description not available.')}",
                        'element': zodiac_sign_data.get('element', 'N/A')
                    },
                    'chinese_zodiac': {
                        'name': chinese_zodiac_data['name'],
                        'description': chinese_zodiac_data.get('description', 'Description not available.')
                    }
                }

            except Exception as e:
                raise ValueError(f"Error processing DOB: {str(e)}")

        zodiac_results = calculate_zodiac_signs(dob)

                # Process image for facial features
        def process_image_data(image_data):
            try:
                img_data = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                img_cv = np.array(img).astype(np.uint8)

                with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
                    results = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
                    if not results.multi_face_landmarks:
                        raise ValueError("No faces detected.")

                    feature_points = []
                    # Draw landmarks on the image
                    for face_landmarks in results.multi_face_landmarks:
                        for landmark in face_landmarks.landmark:
                            x = int(landmark.x * img_cv.shape[1])
                            y = int(landmark.y * img_cv.shape[0])
                            cv2.circle(img_cv, (x, y), 2, (0, 255, 0), -1)  # Draw green landmarks
                            feature_points.append((x, y))

                    if feature_points:
                        left_eye_region = img_cv[
                            feature_points[36][1]:feature_points[41][1],
                            feature_points[36][0]:feature_points[39][0]
                        ]
                        left_eye_img_resized = cv2.resize(left_eye_region, (200, 200))
                        preprocess = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        img_tensor = preprocess(Image.fromarray(cv2.cvtColor(left_eye_img_resized, cv2.COLOR_BGR2RGB))).unsqueeze(0)

                        eyebrow = eyebrow_model(img_tensor)
                        eye = eye_model(img_tensor)
                        nose = nose_model(img_tensor)
                        mouth = mouth_model(img_tensor)
                        jaw = jaw_model(img_tensor)

                        # Get shapes and descriptions
                        def get_shape_analysis(shape_name, feature_name):
                            for region in shape_descriptions['face_regions']:
                                if region['name'] == shape_name:
                                    for feature in region['features']:
                                        if feature['name'] == feature_name:
                                            return feature['analysis']
                            return "Description not found."

                        return {
                            'eyebrow': {
                                'shape': ["Arch", "Circle", "Straight"][torch.argmax(eyebrow).item()],
                                'description': get_shape_analysis('eyebrows', ["Arch", "Circle", "Straight"][torch.argmax(eyebrow).item()])
                            },
                            'eye': {
                                'shape': ["Big", "Silt", "Small"][torch.argmax(eye).item()],
                                'description': get_shape_analysis('eyes', ["Big", "Silt", "Small"][torch.argmax(eye).item()])
                            },
                            'nose': {
                                'shape': ["Long", "Small", "Wide"][torch.argmax(nose).item()],
                                'description': get_shape_analysis('nose', ["Long", "Small", "Wide"][torch.argmax(nose).item()])
                            },
                            'mouth': {
                                'shape': ["Medium", "Small", "Thick"][torch.argmax(mouth).item()],
                                'description': get_shape_analysis('mouth', ["Medium", "Small", "Thick"][torch.argmax(mouth).item()])
                            },
                            'jaw': {
                                'shape': ["Circle", "Oval", "Square", "Triangle"][torch.argmax(jaw).item()],
                                'description': get_shape_analysis('face', ["Circle", "Oval", "Square", "Triangle"][torch.argmax(jaw).item()])
                            },
                            'labeled_image': img_cv  # Add the labeled image
                        }
            except Exception as e:
                raise ValueError(f"Error processing image: {str(e)}")

        facial_features = process_image_data(image_data)

        # Generate labeled image base64
        _, buffer = cv2.imencode('.png', facial_features.pop('labeled_image'))
        labeled_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Find lookalikes
        def find_lookalike(facial_features, gender):
            try:
                with open('C:\\Projects\\brainwave\\frontend\\src\\components\\celebrity_data.json') as f:
                    celeb_data = json.load(f)

                matches = []
                weights = {'eyebrow': 1, 'eye': 2, 'nose': 2, 'mouth': 1, 'jaw': 3}
                for celeb in celeb_data:
                    if celeb['gender'].lower() != gender.lower():
                        continue

                    score = sum(weights[feature] for feature in weights if celeb['features'][feature] == facial_features[feature])
                    total_weight = sum(weights.values())
                    match_percentage = (score / total_weight) * 100
                    matches.append({
                        'name': celeb['name'],
                        'gender': celeb['gender'],
                        'features': celeb['features'],
                        'match_percentage': round(match_percentage, 2)
                    })

                matches.sort(key=lambda x: (-x['match_percentage'], x['name']))
                return matches[:3]
            except Exception as e:
                raise ValueError(f"Error finding lookalike: {str(e)}")

        lookalike_results = find_lookalike(facial_features, gender)

        PDF_STORAGE_PATH = "C:/Projects/brainwave/backend/pdfs"
        os.makedirs(PDF_STORAGE_PATH, exist_ok=True)  # Create 'pdfs' directory if not exists

        pdf_stream = generate_pdf_report(name, email, number, zodiac_results, facial_features, lookalike_results, labeled_image_base64)
        if not pdf_stream:
            raise ValueError("Failed to generate PDF report.")


        # Generate PDF and save it to a file
        pdf_stream = generate_pdf_report(name, email, number, zodiac_results, facial_features, lookalike_results, labeled_image_base64)
        pdf_filename = f"{name.replace(' ', '_')}_{email.replace('@', '_')}.pdf"
        pdf_path = os.path.join(PDF_STORAGE_PATH, pdf_filename)

        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(pdf_stream.read())
        pdf_stream.close()

        app.logger.info(f"PDF saved at {pdf_path}")

        try:
            # Generate and send the PDF
            app.logger.info("Sending PDF...")
            return send_file(pdf_path, as_attachment=True, mimetype="application/pdf")
        except Exception as e:
            app.logger.error(f"Error while sending PDF: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    

    except Exception as e:
        app.logger.error(f"Error in process-all: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# Endpoint to serve PDF files
@app.route('/files/<path:filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_from_directory(PDF_STORAGE_PATH, filename, as_attachment=True, mimetype='application/pdf')
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'File not found.'}), 404

        
if __name__ == '__main__':
    app.run(debug=True)
