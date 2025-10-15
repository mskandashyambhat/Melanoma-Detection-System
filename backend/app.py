"""
Flask Backend API for Melanoma Detection System
Handles image upload, prediction, report generation, and doctor consultation
"""

import sys
import flask
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not installed. Using mock predictions.")
    TF_AVAILABLE = False
    
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("⚠️  OpenCV not installed. Some features may be limited.")
    CV2_AVAILABLE = False
    
from datetime import datetime
import json
from report_generator import generate_patient_report

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Load models (will be loaded once on startup)
classifier_model = None
segmentation_model = None
# Binary classification: Melanoma or Not Melanoma
class_names = [
    'Melanoma', 'Not Melanoma'
]

# Disease descriptions and recommendations - ONLY Melanoma Stages and Not Melanoma
disease_info = {
    'Melanoma Stage 1': {
        'description': 'Early-stage melanoma confined to the skin, typically ≤1mm thick with no ulceration. Highly treatable with good prognosis.',
        'severity': 'Moderate',
        'stage': 1,
        'recommendations': [
            'Schedule surgical removal (wide excision) within next 2 weeks',
            'Sentinel lymph node biopsy may be recommended',
            'Regular skin checks every 3-6 months for the next 2 years',
            'Daily sun protection with SPF 50+ sunscreen'
        ]
    },
    'Melanoma Stage 2': {
        'description': 'Intermediate melanoma, typically thicker (>1mm) and/or with ulceration, but still confined to the skin with no spread.',
        'severity': 'High',
        'stage': 2,
        'recommendations': [
            'Immediate surgical removal with wider margins',
            'Sentinel lymph node biopsy strongly recommended',
            'Consider adjuvant therapy options',
            'Regular imaging and follow-ups every 3 months'
        ]
    },
    'Melanoma Stage 3': {
        'description': 'Advanced melanoma that has spread to nearby lymph nodes but not to distant sites. Requires aggressive treatment.',
        'severity': 'Very High',
        'stage': 3,
        'recommendations': [
            'Urgent surgical removal of primary tumor and affected lymph nodes',
            'Immunotherapy and/or targeted therapy strongly advised',
            'Radiation therapy may be necessary',
            'Clinical trial participation should be considered'
        ]
    },
    'Melanoma Stage 4': {
        'description': 'Advanced melanoma that has metastasized to distant organs such as lungs, liver, brain, or bones. Most serious stage requiring immediate intervention.',
        'severity': 'Critical',
        'stage': 4,
        'recommendations': [
            'Immediate oncology consultation with melanoma specialist',
            'Systemic therapies including immunotherapy and targeted therapy',
            'Comprehensive mutation testing for treatment planning',
            'Consider clinical trials for innovative treatment options'
        ]
    },
    'Not Melanoma': {
        'description': 'The analyzed skin lesion does not show characteristics typical of melanoma. However, continue monitoring for any changes.',
        'severity': 'Low',
        'recommendations': [
            'Continue regular skin self-examinations',
            'Use sunscreen daily (SPF 30+) to prevent skin damage',
            'Consult a dermatologist if the lesion changes in size, color, or shape',
            'Monitor for any new symptoms like itching, bleeding, or pain',
            'Schedule routine dermatology check-ups annually'
        ]
    }
}

# Mock doctor database
doctors = [
    {
        'id': 1,
        'name': 'Dr. Sarah Johnson',
        'specialization': 'Dermatology & Skin Cancer',
        'experience': '15 years',
        'rating': 4.9,
        'location': 'New York Medical Center',
        'availability': 'Mon-Fri, 9 AM - 5 PM',
        'image': 'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=300&h=300&fit=crop&crop=faces',
        'email': 'sarah.johnson@hospital.com',
        'phone': '+1-555-0101'
    },
    {
        'id': 2,
        'name': 'Dr. Michael Chen',
        'specialization': 'Dermatologist',
        'experience': '12 years',
        'rating': 4.8,
        'location': 'City Dermatology Clinic',
        'availability': 'Mon-Sat, 10 AM - 6 PM',
        'image': 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=300&h=300&fit=crop&crop=faces',
        'email': 'michael.chen@clinic.com',
        'phone': '+1-555-0102'
    },
    {
        'id': 3,
        'name': 'Dr. Emily Rodriguez',
        'specialization': 'Oncologist & Dermatology',
        'experience': '18 years',
        'rating': 5.0,
        'location': 'Cancer Care Institute',
        'availability': 'Tue-Sat, 8 AM - 4 PM',
        'image': 'https://images.unsplash.com/photo-1594824476967-48c8b964273f?w=300&h=300&fit=crop&crop=faces',
        'email': 'emily.rodriguez@cci.com',
        'phone': '+1-555-0103'
    },
    {
        'id': 4,
        'name': 'Dr. James Williams',
        'specialization': 'Dermatology',
        'experience': '10 years',
        'rating': 4.7,
        'location': 'Skin Health Center',
        'availability': 'Mon-Fri, 11 AM - 7 PM',
        'image': 'https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=300&h=300&fit=crop&crop=faces',
        'email': 'james.williams@skincenter.com',
        'phone': '+1-555-0104'
    },
    {
        'id': 5,
        'name': 'Dr. Priya Patel',
        'specialization': 'Pediatric Dermatology',
        'experience': '8 years',
        'rating': 4.9,
        'location': 'Children\'s Skin Clinic',
        'availability': 'Mon-Thu, 9 AM - 5 PM',
        'image': 'https://images.unsplash.com/photo-1651008376811-b90baee60c1f?w=300&h=300&fit=crop&crop=faces',
        'email': 'priya.patel@childclinic.com',
        'phone': '+1-555-0105'
    },
    {
        'id': 6,
        'name': 'Dr. Robert Anderson',
        'specialization': 'Mohs Surgery Specialist',
        'experience': '20 years',
        'rating': 4.8,
        'location': 'Advanced Dermatology',
        'availability': 'Wed-Sun, 8 AM - 3 PM',
        'image': 'https://images.unsplash.com/photo-1537368910025-700350fe46c7?w=300&h=300&fit=crop&crop=faces',
        'email': 'robert.anderson@advderm.com',
        'phone': '+1-555-0106'
    }
]


def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def advanced_mock_prediction(img_array):
    """Advanced mock prediction based on image array features
    
    Returns:
        tuple: (predicted_disease, confidence, image_features)
    """
    # Extract image features
    img_avg_color = np.mean(img_array)
    color_variation = np.std(img_array)
    
    # Calculate additional features for melanoma staging
    try:
        # Convert to grayscale for edge detection
        if img_array.shape[-1] == 3:  # RGB image
            img_gray = np.mean(img_array, axis=-1)
        else:
            img_gray = img_array
            
        # Detect edges (normalized to 0-1 range)
        edges = np.abs(np.gradient(img_gray)[0]) + np.abs(np.gradient(img_gray)[1])
        border_irregularity_raw = np.std(edges) / np.mean(edges) if np.mean(edges) > 0 else 0
        border_irregularity = min(1.0, border_irregularity_raw / 3)  # Normalize
        
        # Estimate diameter (simplified approximation)
        non_zero = np.count_nonzero(img_gray > 0.1)
        diameter_estimate = np.sqrt(non_zero / np.pi) * 2  # Convert to diameter in relative units
        diameter_mm = diameter_estimate * 0.15 + np.random.uniform(-3, 3)  # Add variation
        
        # Calculate asymmetry (normalized to 0-1 range)
        h, w = img_gray.shape[1:3]
        left_half = img_gray[:, :w//2]
        right_half = img_gray[:, w//2:]
        right_half_flipped = np.flip(right_half, axis=1)
        asymmetry_raw = np.mean(np.abs(left_half - right_half_flipped))
        asymmetry = min(1.0, asymmetry_raw)  # Already normalized since img_gray is 0-1
    except:
        border_irregularity = 0.3 + np.random.uniform(-0.15, 0.15)
        diameter_mm = 6 + np.random.uniform(-3, 8)
        asymmetry = 0.25 + np.random.uniform(-0.1, 0.2)
    
    # Compile features
    img_features = {
        'darkness': float(255 - (img_avg_color * 255)),  # Convert to 0-255 scale
        'color_variation': float(color_variation * 255),
        'border_irregularity': float(border_irregularity),
        'asymmetry': float(asymmetry),
        'diameter_estimate': float(diameter_mm)  # in mm
    }
    
    # Disease prediction based on features
    # Dark spots with high variation and irregular borders are more likely to be melanoma
    # Adjusted scoring to be more conservative and produce varied stages
    melanoma_score = (img_features['darkness'] / 255) * 0.20 + \
                    (img_features['color_variation'] / 60) * 0.20 + \
                    min(img_features['border_irregularity'], 0.6) * 0.25 + \
                    min(img_features['asymmetry'], 0.6) * 0.20 + \
                    min(1.0, img_features['diameter_estimate'] / 20) * 0.15
    
    # Add image-based randomness for variation
    hash_seed = int(img_features['darkness'] * 1000) % 100
    np.random.seed(hash_seed)
    random_offset = np.random.uniform(-0.12, 0.12)
    melanoma_score = max(0, min(1, melanoma_score + random_offset))
    
    # Decision logic - ONLY Melanoma or Not Melanoma
    # Threshold for melanoma detection
    if melanoma_score > 0.35:
        # Detected as Melanoma - will be staged later
        confidence = 75 + melanoma_score * 20
        return 'Melanoma', min(confidence, 98.0), img_features
    else:
        # Not melanoma
        confidence = 70 + (1 - melanoma_score) * 20
        return 'Not Melanoma', min(confidence, 95.0), img_features


def advanced_mock_prediction_from_image(img_array_simple):
    """Advanced mock prediction from raw image data
    
    Returns:
        tuple: (predicted_disease, confidence, image_features)
    """
    # Calculate basic image statistics
    color_variation = np.std(img_array_simple)
    darkness = 255 - np.mean(img_array_simple)
    
    # Calculate more advanced features
    try:
        # Convert to grayscale if it's a color image
        if len(img_array_simple.shape) > 2 and img_array_simple.shape[2] == 3:
            gray_img = np.mean(img_array_simple, axis=2)
        else:
            gray_img = img_array_simple
            
        # Estimate border irregularity (normalize to 0-1 range)
        edges = np.abs(np.gradient(gray_img)[0]) + np.abs(np.gradient(gray_img)[1])
        border_irregularity_raw = np.std(edges) / np.mean(edges) if np.mean(edges) > 0 else 0
        border_irregularity = min(1.0, border_irregularity_raw / 3)  # Normalize: typical values 0-3
        
        # Estimate asymmetry (normalize to 0-1 range)
        h, w = gray_img.shape
        left_half = gray_img[:, :w//2]
        right_half = gray_img[:, w//2:]
        flipped_right = np.flip(right_half, axis=1)
        # If sizes don't match exactly, use the minimum size
        min_w = min(left_half.shape[1], flipped_right.shape[1])
        asymmetry_raw = np.mean(np.abs(left_half[:, :min_w] - flipped_right[:, :min_w]))
        asymmetry = min(1.0, asymmetry_raw / 128)  # Normalize: pixel differences 0-255, typical < 128
        
        # Estimate diameter
        non_zero = np.count_nonzero(gray_img > np.mean(gray_img) * 0.2)
        total_pixels = gray_img.size
        relative_size = non_zero / total_pixels
        diameter_mm = relative_size * 30 + np.random.uniform(-5, 5)  # 0-30mm range with variation
    except Exception as e:
        print(f"Error in advanced feature extraction: {e}")
        border_irregularity = 0.3 + np.random.uniform(-0.1, 0.1)
        asymmetry = 0.25 + np.random.uniform(-0.1, 0.1)
        diameter_mm = 5 + np.random.uniform(0, 10)
    
    # Compile image features
    img_features = {
        'darkness': float(darkness),
        'color_variation': float(color_variation),
        'border_irregularity': float(border_irregularity),
        'asymmetry': float(asymmetry),
        'diameter_estimate': float(diameter_mm)
    }
    
    # Melanoma detection logic - adjusted for more balanced staging
    # All features now properly normalized to 0-1 range
    melanoma_score = (darkness / 255) * 0.20 + \
                     (color_variation / 60) * 0.20 + \
                     min(border_irregularity, 0.6) * 0.25 + \
                     min(asymmetry, 0.6) * 0.20 + \
                     min(1.0, diameter_mm / 20) * 0.15
    
    # Add deterministic randomness based on image characteristics
    hash_seed = int(darkness * color_variation * 1000) % 100
    np.random.seed(hash_seed)
    random_offset = np.random.uniform(-0.12, 0.12)
    melanoma_score = max(0, min(1, melanoma_score + random_offset))
    
    # Decision logic - ONLY Melanoma or Not Melanoma
    # Threshold for melanoma detection
    if melanoma_score > 0.35:
        # Detected as Melanoma - will be staged later
        confidence = 75 + melanoma_score * 20
        return 'Melanoma', min(confidence, 98.0), img_features
    else:
        # Not melanoma
        confidence = 70 + (1 - melanoma_score) * 20
        return 'Not Melanoma', min(confidence, 95.0), img_features


def determine_melanoma_stage(img_features):
    """Determine melanoma stage based on image features
    
    Staging criteria based on melanoma characteristics:
    - Stage 1: Early, localized (≤1mm thick, no ulceration)
    - Stage 2: Intermediate (>1mm thick and/or ulcerated)
    - Stage 3: Regional spread (lymph node involvement)
    - Stage 4: Distant metastasis
    
    Args:
        img_features: Dictionary of image features
        
    Returns:
        int: Melanoma stage (1-4)
    """
    # Extract and normalize relevant features properly
    darkness = img_features.get('darkness', 100) / 255  # Normalize to 0-1
    color_variation = img_features.get('color_variation', 20) / 100  # Normalize to 0-1
    border_irregularity = min(1.0, img_features.get('border_irregularity', 0.5))  # Cap at 1.0
    asymmetry = min(1.0, img_features.get('asymmetry', 0.3))  # Cap at 1.0
    diameter_mm = img_features.get('diameter_estimate', 6)
    
    # Calculate ABCDE criteria scores (Asymmetry, Border, Color, Diameter, Evolution)
    # Evolution cannot be determined from a single image, so we exclude it
    
    # Create a combined severity score with balanced weights
    # Using conservative scoring to produce varied stages
    severity_score = (
        min(asymmetry, 0.5) * 0.20 +           # Asymmetry (capped contribution)
        min(border_irregularity, 0.5) * 0.20 + # Border irregularity (capped)
        min(color_variation, 0.5) * 0.25 +     # Color variation (capped)
        min(darkness, 0.5) * 0.15 +            # Darkness (capped)
        min(1.0, diameter_mm / 20) * 0.20      # Diameter relative to 20mm
    )
    
    # Add some image-based randomness for natural variation
    hash_value = (darkness * 100 + color_variation * 50 + asymmetry * 30) % 1
    variation = (hash_value - 0.5) * 0.15  # ±7.5% variation
    severity_score = max(0, min(1, severity_score + variation))
    
    # Determine stage with realistic distribution
    # Most melanomas (60-70%) are caught in stage 1-2
    if severity_score > 0.75:
        return 4  # Very severe features - possible metastatic (10-15%)
    elif severity_score > 0.55:
        return 3  # Severe features - possible lymph node involvement (15-20%)
    elif severity_score > 0.35:
        return 2  # Moderate features - thicker lesion (25-30%)
    else:
        return 1  # Mild features - early stage (35-40%)


def load_models():
    """Load pre-trained models"""
    global classifier_model, segmentation_model
    
    try:
        # Try to load saved models
        if os.path.exists('models/classifier_model.h5'):
            try:
                classifier_model = keras.models.load_model('models/classifier_model.h5')
                print("✓ Classifier model loaded successfully")
            except Exception as e:
                print(f"⚠ Error loading classifier model: {e}")
                print("⚠ Falling back to mock predictions for classifier.")
                classifier_model = None
        else:
            print("⚠ Classifier model file not found at models/classifier_model.h5")
            print("⚠ Using mock predictions. Place trained model in the models directory.")
            
        if os.path.exists('models/segmentation_model.h5'):
            try:
                segmentation_model = keras.models.load_model('models/segmentation_model.h5')
                print("✓ Segmentation model loaded successfully")
            except Exception as e:
                print(f"⚠ Error loading segmentation model: {e}")
                print("⚠ Falling back to mock predictions for segmentation.")
                segmentation_model = None
        else:
            print("⚠ Segmentation model file not found at models/segmentation_model.h5")
            print("⚠ Using mock predictions. Place trained model in the models directory.")
            
        # Create the models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Return information on loaded models for logging
        return {
            'classifier': classifier_model is not None,
            'segmentation': segmentation_model is not None
        }
    except Exception as e:
        print(f"⚠ Critical error loading models: {e}")
        print("⚠ All predictions will use mock data.")
        return {'classifier': False, 'segmentation': False}


def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model prediction with enhanced preprocessing"""
    try:
        # Check if file exists and is accessible
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")
            
        # Check file size
        file_stats = os.stat(image_path)
        if file_stats.st_size > 16 * 1024 * 1024:  # 16MB limit
            raise ValueError("Image file is too large (>16MB)")
            
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Store original image characteristics for analysis
        img_np = np.array(img)
        
        # Advanced preprocessing for better melanoma detection
        if CV2_AVAILABLE:
            try:
                # Convert to OpenCV format
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Apply contrast enhancement
                lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                
                # Convert back to RGB for model input
                enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(enhanced_img_rgb)
                
                print("✓ Applied advanced image preprocessing")
            except Exception as e:
                print(f"Warning: Error in advanced preprocessing: {e}")
                # Fall back to original image if enhancement fails
                pass
            
        # Resize image
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise ValueError(f"Invalid or corrupted image file: {str(e)}")


def predict_disease(image_path):
    """Predict disease from image with advanced melanoma staging"""
    try:
        # Preprocess image
        img_array = preprocess_image(image_path)
        
        # Track if we're using real or mock predictions
        using_mock = False
        model_status = "Real model prediction"
        
        # If models are loaded, use them
        if classifier_model is not None:
            try:
                predictions = classifier_model.predict(img_array, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class]) * 100
                
                # For demonstration purposes, we'll use the original prediction
                base_disease = class_names[predicted_class]
            except Exception as e:
                print(f"Error during model prediction: {e}")
                print("Falling back to mock predictions")
                using_mock = True
                model_status = f"Mock prediction (model error: {str(e)[:100]})"
                # Call our advanced mock prediction function
                base_disease, confidence, img_features = advanced_mock_prediction(img_array)
        else:
            # Mock prediction when model isn't loaded
            using_mock = True
            model_status = "Mock prediction (no model loaded)"
            
            # Try to analyze basic image characteristics for better mock predictions
            try:
                img = Image.open(image_path)
                img_array_simple = np.array(img.resize((50, 50)))
                
                # Advanced image analysis for detailed features
                base_disease, confidence, img_features = advanced_mock_prediction_from_image(img_array_simple)
            except Exception as e:
                # Fallback if image analysis fails
                print(f"Error during mock image analysis: {e}")
                base_disease = np.random.choice(['Melanoma', 'Basal Cell Carcinoma', 'Acne', 'Ringworm', 
                                               'Burns', 'Eczema', 'Psoriasis', 'Normal Skin'])
                confidence = np.random.uniform(75.0, 92.0)
                img_features = {
                    'darkness': 100,
                    'color_variation': 20,
                    'border_irregularity': 0.5,
                    'asymmetry': 0.3,
                    'diameter_estimate': 6  # in mm
                }
        
        # Determine melanoma stage if melanoma is detected
        if base_disease == 'Melanoma':
            # Use image features to determine melanoma stage
            if using_mock:
                melanoma_stage = determine_melanoma_stage(img_features)
                disease = f"Melanoma Stage {melanoma_stage}"
            else:
                # If using real model, use confidence to estimate stage
                # In a production system, this would use a dedicated staging model
                if confidence > 95:
                    disease = "Melanoma Stage 4"
                elif confidence > 90:
                    disease = "Melanoma Stage 3"
                elif confidence > 85:
                    disease = "Melanoma Stage 2"
                else:
                    disease = "Melanoma Stage 1"
        else:
            disease = base_disease
        
        # Get disease info or fallback to basic info if not found
        if disease in disease_info:
            info = disease_info[disease]
        else:
            # Basic info for other diseases or if staging info not found
            info = {
                'description': f"{disease} detected with {round(confidence)}% confidence.",
                'severity': 'Medium',
                'recommendations': [
                    'Consult with a dermatologist',
                    'Protect the affected area',
                    'Follow up with healthcare provider'
                ]
            }
        
        # Log the prediction
        print(f"Prediction: {disease} (Confidence: {round(confidence, 2)}%) - {model_status}")
        
        # Add features to the result for comprehensive analysis
        result = {
            'disease': disease,
            'confidence': round(confidence, 2),
            'description': info['description'],
            'severity': info['severity'],
            'recommendations': info['recommendations'],
            'using_mock_prediction': using_mock,
            'features': img_features if using_mock else {},
        }
        
        # If melanoma, add stage information
        if disease.startswith('Melanoma Stage'):
            result['stage'] = int(disease[-1])
            
        return result
    except Exception as e:
        print(f"Critical error during prediction: {e}")
        raise Exception(f"Error analyzing image: {str(e)}. Please try with a different image or contact support.")


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Melanoma Detection System API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Upload image for prediction',
            '/generate-report': 'POST - Generate patient report',
            '/doctors': 'GET - Get list of doctors',
            '/consult-doctor': 'POST - Send report to doctor'
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'details': 'Please ensure you selected an image file before uploading',
                'code': 'MISSING_FILE'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'details': 'An empty file was submitted. Please select a valid image file',
                'code': 'EMPTY_FILE'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'details': f'Only {", ".join(ALLOWED_EXTENSIONS)} files are supported',
                'code': 'INVALID_FILE_TYPE'
            }), 400
        
        # Check file size before saving
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        
        # Check if file is empty
        if len(file_content) == 0:
            return jsonify({
                'error': 'Empty file uploaded',
                'details': 'The uploaded file contains no data',
                'code': 'EMPTY_FILE_CONTENT'
            }), 400
            
        # Check file size (16MB limit)
        if len(file_content) > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({
                'error': 'File too large',
                'details': f'Maximum file size is {app.config["MAX_CONTENT_LENGTH"] / (1024*1024)}MB',
                'code': 'FILE_TOO_LARGE'
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Image saved to {filepath}, proceeding with analysis...")
        
        # Predict disease
        result = predict_disease(filepath)
        result['image_path'] = filepath
        result['filename'] = filename
        
        # Add system status info to help with debugging
        result['system_info'] = {
            'models_available': {
                'classifier': classifier_model is not None,
                'segmentation': segmentation_model is not None
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        error_message = str(e)
        error_code = 'ANALYSIS_ERROR'
        
        if 'Invalid or corrupted' in error_message:
            error_code = 'CORRUPTED_IMAGE'
        elif 'too large' in error_message:
            error_code = 'IMAGE_TOO_LARGE'
        
        return jsonify({
            'error': 'Failed to analyze image',
            'details': error_message,
            'code': error_code
        }), 500


@app.route('/generate-report', methods=['POST'])
def generate_report():
    """Generate patient report PDF"""
    try:
        data = request.get_json()
        
        # Extract patient and prediction data
        patient_info = data.get('patient_info', {})
        prediction_data = data.get('prediction_data', {})
        
        # Generate PDF report
        report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        
        generate_patient_report(
            patient_info=patient_info,
            prediction_data=prediction_data,
            output_path=report_path
        )
        
        return jsonify({
            'message': 'Report generated successfully',
            'report_filename': report_filename,
            'download_url': f'/download-report/{report_filename}'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download-report/<filename>', methods=['GET'])
def download_report(filename):
    """Download generated report"""
    try:
        report_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
        if os.path.exists(report_path):
            return send_file(report_path, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/doctors', methods=['GET'])
def get_doctors():
    """Get list of available doctors"""
    return jsonify({'doctors': doctors}), 200


@app.route('/consult-doctor', methods=['POST'])
def consult_doctor():
    """Send report to selected doctor"""
    try:
        data = request.get_json()
        doctor_id = data.get('doctor_id')
        report_filename = data.get('report_filename')
        patient_info = data.get('patient_info', {})
        
        # Find doctor
        doctor = next((d for d in doctors if d['id'] == doctor_id), None)
        
        if not doctor:
            return jsonify({'error': 'Doctor not found'}), 404
        
        # In a real application, this would send an email or notification
        # For demo purposes, we'll just return success
        
        return jsonify({
            'message': f'Report successfully sent to {doctor["name"]}',
            'doctor': doctor,
            'consultation_id': f'CONS{datetime.now().strftime("%Y%m%d%H%M%S")}'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 80)
    print(" Melanoma Detection System API ")
    print("=" * 80)
    
    # Log system info
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Flask version: {flask.__version__}")
    
    if TF_AVAILABLE:
        print(f"TensorFlow version: {tf.__version__}")
    else:
        print("TensorFlow: Not installed")
        
    if CV2_AVAILABLE:
        print(f"OpenCV version: {cv2.__version__}")
    else:
        print("OpenCV: Not installed")
    
    print("\nLoading models...")
    model_status = load_models()
    
    if model_status['classifier'] and model_status['segmentation']:
        print("\n✅ All models loaded successfully!")
    else:
        print("\n⚠️ Some models were not loaded. Using mock predictions.")
        if not model_status['classifier']:
            print("   - Classifier model not available")
        if not model_status['segmentation']:
            print("   - Segmentation model not available")
        print("\nTo use real predictions, place model files in the models/ directory:")
        print("  - models/classifier_model.h5")
        print("  - models/segmentation_model.h5")
    
    print("\nInitializing server on port 5001...")
    print("=" * 80)
    app.run(debug=True, host='0.0.0.0', port=5001)
