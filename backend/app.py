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
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    print("✅ TensorFlow imported successfully")
except ImportError:
    print("⚠️  TensorFlow not installed. Using mock predictions.")
    TF_AVAILABLE = False
    keras = None
    load_model = None
    
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("⚠️  OpenCV not installed. Some features may be limited.")
    CV2_AVAILABLE = False
    
from datetime import datetime
import json

try:
    from gemini_validator import validate_image_with_gemini
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠️  Gemini API not available. Advanced validation disabled.")
    GEMINI_AVAILABLE = False
    validate_image_with_gemini = None

try:
    from report_generator import generate_patient_report
    REPORT_GEN_AVAILABLE = True
except ImportError:
    print("⚠️  Report generator not available.")
    REPORT_GEN_AVAILABLE = False
    generate_patient_report = None

try:
    from chatbot_service import get_chatbot_response
    CHATBOT_AVAILABLE = True
except ImportError:
    print("⚠️  Chatbot service not available.")
    CHATBOT_AVAILABLE = False
    get_chatbot_response = None

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Try to import email configuration
try:
    from email_config import SENDER_EMAIL, SENDER_PASSWORD, SMTP_SERVER, SMTP_PORT
    EMAIL_CONFIGURED = True
except ImportError:
    print("⚠️  Email configuration not found. Email sending will be disabled.")
    SENDER_EMAIL = None
    SENDER_PASSWORD = None
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    EMAIL_CONFIGURED = False

# Try to import WhatsApp configuration
try:
    from email_config import WHATSAPP_SENDER_NAME, REAL_DOCTOR_WHATSAPP
    WHATSAPP_CONFIGURED = True
    # Try to import pywhatkit (WhatsApp Web API - no extra number needed!)
    try:
        import pywhatkit as kit
        PYWHATKIT_AVAILABLE = True
        print("✅ WhatsApp integration ready (via WhatsApp Web)")
    except ImportError:
        print("⚠️  pywhatkit not installed. WhatsApp sending will be disabled.")
        print("    Install with: pip install pywhatkit")
        PYWHATKIT_AVAILABLE = False
except ImportError:
    print("⚠️  WhatsApp configuration not found. WhatsApp sending will be disabled.")
    WHATSAPP_SENDER_NAME = "Melanoma Detection Lab"
    REAL_DOCTOR_WHATSAPP = {}
    WHATSAPP_CONFIGURED = False
    PYWHATKIT_AVAILABLE = False

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

# Binary classification: Melanoma vs Benign
class_names = ['Benign', 'Melanoma']

# Disease descriptions and recommendations - Binary Classification
disease_info = {
    'Melanoma': {
        'description': 'Melanoma detected. This is a serious form of skin cancer that requires immediate medical attention. Early detection and treatment are crucial for positive outcomes.',
        'severity': 'High',
        'recommendations': [
            '⚠️ URGENT: Schedule appointment with dermatologist within 1-2 days',
            'Avoid sun exposure and use SPF 50+ sunscreen',
            'Take photos to track any changes',
            'Prepare list of questions for your doctor consultation',
            'Do not attempt self-treatment',
            'Bring this report to your medical appointment'
        ],
        'next_steps': [
            'Professional biopsy for confirmation',
            'Dermoscopic examination by specialist',
            'Possible imaging tests to check spread',
            'Discussion of treatment options if confirmed'
        ]
    },
    'Benign': {
        'description': 'Good news! The analyzed skin lesion appears benign and does not show characteristics typical of melanoma. This is a low-risk finding, but regular monitoring is always recommended.',
        'severity': 'Low',
        'recommendations': [
            '✅ No immediate medical action required',
            'Continue regular skin self-examinations monthly',
            'Use sunscreen daily (SPF 30+) to prevent skin damage',
            'Monitor for any changes in size, color, or shape',
            'Photograph the lesion for future comparison',
            'Stay aware of the ABCDE warning signs'
        ],
        'next_steps': [
            'Self-monitor using ABCDE rule (Asymmetry, Border, Color, Diameter, Evolving)',
            'Consult dermatologist only if you notice changes',
            'Maintain healthy skin care routine',
            'Consider annual skin check-up as part of routine wellness'
        ]
    }
}

# Mock doctor database
doctors = [
    {
        'id': 1,
        'name': 'Dr. Rajesh Kumar',
        'specialization': 'Dermatology & Skin Cancer',
        'experience': '15 years',
        'rating': 4.9,
        'location': 'Apollo Hospital, Bangalore',
        'availability': 'Mon-Fri, 9 AM - 5 PM',
        'email': 'dr.rajesh@apollohospital.com',
        'phone': '+91-80-2222-3333'
    },
    {
        'id': 2,
        'name': 'Dr. Priya Sharma',
        'specialization': 'Clinical Dermatologist',
        'experience': '12 years',
        'rating': 4.8,
        'location': 'Fortis Hospital, Mumbai',
        'availability': 'Mon-Sat, 10 AM - 6 PM',
        'email': 'dr.priya@fortishospital.com',
        'phone': '+91-22-4444-5555'
    },
    {
        'id': 3,
        'name': 'Dr. Anil Desai',
        'specialization': 'Oncology & Dermatology',
        'experience': '18 years',
        'rating': 5.0,
        'location': 'Tata Memorial Hospital, Mumbai',
        'availability': 'Tue-Sat, 8 AM - 4 PM',
        'email': 'dr.anil@tatamemorial.com',
        'phone': '+91-22-6666-7777'
    },
    {
        'id': 4,
        'name': 'Dr. Kavita Reddy',
        'specialization': 'Dermatology',
        'experience': '10 years',
        'rating': 4.7,
        'location': 'Manipal Hospital, Bangalore',
        'availability': 'Mon-Fri, 11 AM - 7 PM',
        'email': 'dr.kavita@manipalhospital.com',
        'phone': '+91-80-8888-9999'
    },
    {
        'id': 5,
        'name': 'Dr. Suresh Menon',
        'specialization': 'Pediatric Dermatology',
        'experience': '8 years',
        'rating': 4.9,
        'location': 'Max Hospital, Delhi',
        'availability': 'Mon-Thu, 9 AM - 5 PM',
        'email': 'dr.suresh@maxhospital.com',
        'phone': '+91-11-1111-2222'
    },
    {
        'id': 6,
        'name': 'Dr. Meera Iyer',
        'specialization': 'Cosmetic & Medical Dermatology',
        'experience': '14 years',
        'rating': 4.8,
        'location': 'AIIMS, New Delhi',
        'availability': 'Wed-Sun, 8 AM - 3 PM',
        'email': 'dr.meera@aiims.edu',
        'phone': '+91-11-3333-4444'
    },
    {
        'id': 107,
        'name': 'Dr. Shyam',
        'specialization': 'MD Dermatology, MBBS',
        'experience': '14 years',
        'rating': 4.9,
        'location': 'Digital Health Institute',
        'availability': 'Mon-Fri, 10 AM - 6 PM',
        'email': 'skandashyam102@gmail.com',
        'phone': '+1-555-0107'
    },
    {
        'id': 108,
        'name': 'Dr. Kaushik',
        'specialization': 'Surgical Oncology & Dermatology',
        'experience': '16 years',
        'rating': 4.8,
        'location': 'Advanced Cancer Care',
        'availability': 'Tue-Sat, 9 AM - 5 PM',
        'email': 'kaushikmuliya@gmail.com',
        'phone': '+1-555-0108'
    },
    {
        'id': 109,
        'name': 'Dr. Paavani',
        'specialization': 'Cosmetic & Medical Dermatology',
        'experience': '11 years',
        'rating': 5.0,
        'location': 'Skin Wellness Center',
        'availability': 'Mon-Thu, 11 AM - 7 PM',
        'email': 'kpaavani20@gmail.com',
        'phone': '+1-555-0109'
    },
    {
        'id': 110,
        'name': 'Dr. Deepa',
        'specialization': 'Dermatopathology & Melanoma Research',
        'experience': '19 years',
        'rating': 4.9,
        'location': 'Research Dermatology Clinic',
        'availability': 'Mon-Fri, 8 AM - 4 PM',
        'email': 'deepamk725@gmail.com',
        'phone': '+1-555-0110'
    }
]

# Email configuration for real doctors
REAL_DOCTOR_EMAILS = {
    107: 'skandashyam102@gmail.com',
    108: 'kaushikmuliya@gmail.com',
    109: 'kpaavani20@gmail.com',
    110: 'deepamk725@gmail.com'
}


def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def send_whatsapp_to_doctor(doctor_whatsapp, doctor_name, patient_info, report_path, consultation_id):
    """Send WhatsApp message to doctor with consultation details via WhatsApp Web"""
    try:
        # Check if WhatsApp is configured
        if not WHATSAPP_CONFIGURED or not PYWHATKIT_AVAILABLE:
            print("⚠️  WhatsApp not configured. Skipping WhatsApp send.")
            return False
        
        # Create WhatsApp message body
        message_body = f"""🏥 *{WHATSAPP_SENDER_NAME}*

*New Patient Consultation Request*

*Consultation ID:* {consultation_id}

*Patient Information:*
• Name: {patient_info.get('name', 'N/A')}
• Age: {patient_info.get('age', 'N/A')}
• Gender: {patient_info.get('gender', 'N/A')}
• Phone: {patient_info.get('phone', 'N/A')}

*Medical History:* {patient_info.get('medical_history', 'N/A')}

The detailed medical report has been sent to your email: {patient_info.get('doctor_email', 'your registered email')}

Please review at your earliest convenience.

_Thank you for your consultation_"""
        
        # Get current time and schedule message immediately (next minute)
        from datetime import datetime
        now = datetime.now()
        hour = now.hour
        minute = now.minute + 1
        
        # If minute is 60, adjust to next hour
        if minute >= 60:
            minute = 0
            hour = (hour + 1) % 24
        
        # Send WhatsApp message via WhatsApp Web
        # This will open WhatsApp Web and send the message automatically
        kit.sendwhatmsg(
            phone_no=doctor_whatsapp,
            message=message_body.strip(),
            time_hour=hour,
            time_min=minute,
            wait_time=15,  # Wait 15 seconds for WhatsApp Web to load
            tab_close=True,  # Close tab after sending
            close_time=3  # Close after 3 seconds
        )
        
        print(f"✅ WhatsApp message scheduled to {doctor_name} at {hour}:{minute:02d}")
        return True
        
    except Exception as e:
        print(f"⚠️ WhatsApp sending failed: {str(e)}")
        print(f"   Note: Make sure you're logged into WhatsApp Web in your default browser")
        return False


def send_email_to_doctor(doctor_email, doctor_name, patient_info, report_path, consultation_id):
    """Send email to doctor with patient report attached"""
    try:
        # Check if email is configured
        if not EMAIL_CONFIGURED or not SENDER_EMAIL or not SENDER_PASSWORD:
            print("⚠️  Email not configured. Skipping email send.")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = doctor_email
        msg['Subject'] = f"New Patient Consultation Request - {consultation_id}"
        
        # Email body
        body = f"""
Dear {doctor_name},

You have received a new patient consultation request through the Melanoma Detection Lab.

Patient Information:

Name: {patient_info.get('name', 'N/A')}

Age: {patient_info.get('age', 'N/A')}

Gender: {patient_info.get('gender', 'N/A')}

Phone: {patient_info.get('phone', 'N/A')}

Email: {patient_info.get('email', 'N/A')}

Medical History: {patient_info.get('medical_history', 'N/A')}

Consultation ID: {consultation_id}

Please find the detailed medical report attached to this email.

Your evaluation will help guide the next steps in this patient’s care.

Best regards,
Melanoma Detection Lab
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach report PDF if it exists
        if os.path.exists(report_path):
            with open(report_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(report_path)}'
                )
                msg.attach(part)
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        
        return True
    except Exception as e:
        print(f"Email sending failed: {str(e)}")
        return False


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


# Staging function removed - Binary classification only (Melanoma vs No Melanoma)


def load_models():
    """Load pre-trained models"""
    global classifier_model, segmentation_model
    
    try:
        # Try multiple model paths in order of preference
        model_paths = [
            'models/resnet50_melanoma_20251112_172626.h5',
            'models/best_model_20251103_225237.h5',
            '../models/hybrid_balanced_20251112_154433.h5',
            '../models/hybrid_balanced_20251112_152820.h5'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    print(f"🔄 Attempting to load model: {model_path}")
                    classifier_model = load_model(model_path, compile=False)
                    
                    # Recompile with binary crossentropy
                    classifier_model.compile(
                        optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    print("=" * 60)
                    print("✅ TRAINED MODEL LOADED SUCCESSFULLY")
                    print(f"   Model: {model_path}")
                    print(f"   Layers: {len(classifier_model.layers)}")
                    print(f"   Input shape: {classifier_model.input_shape}")
                    print(f"   Output shape: {classifier_model.output_shape}")
                    print("=" * 60)
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"❌ Error loading {model_path}: {e}")
                    continue
        
        if not model_loaded:
            print(f"❌ NO TRAINED MODEL FOUND!")
            print(f"   Searched paths: {model_paths}")
            raise Exception("No trained model available")
            classifier_model = None
            
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
    """Predict disease from image"""
    try:
        # Preprocess image (224x224 RGB, normalized to 0-1)
        img_array = preprocess_image(image_path, target_size=(224, 224))
        
        # Track if we're using real or mock predictions
        using_mock = False
        model_status = "Real model prediction"
        
        # If models are loaded, use them
        if classifier_model is not None:
            try:
                # Model outputs 2 probabilities: [Benign, Melanoma] - class_names order
                predictions = classifier_model.predict(img_array, verbose=0)
                benign_prob = float(predictions[0][0])
                melanoma_prob = float(predictions[0][1])
                
                print(f"   Raw predictions - Benign: {benign_prob:.4f}, Melanoma: {melanoma_prob:.4f}")
                
                # Require strong confidence (>0.95) for melanoma classification to reduce false positives
                if melanoma_prob > 0.95:
                    disease = 'Melanoma'
                    confidence = melanoma_prob * 100
                else:
                    disease = 'Benign'
                    confidence = max(benign_prob, 1 - melanoma_prob) * 100
                
                print(f"✅ Model prediction: {disease} ({confidence:.2f}% confidence)")
                
            except Exception as e:
                print(f"❌ Error during model prediction: {e}")
                print("⚠️  Falling back to mock predictions")
                using_mock = True
                model_status = f"Mock prediction (model error: {str(e)[:100]})"
                # Simple mock for binary classification
                disease = np.random.choice(['Benign', 'Melanoma'], p=[0.85, 0.15])
                confidence = np.random.uniform(75.0, 95.0)
        else:
            # Mock prediction when model isn't loaded
            using_mock = True
            model_status = "Mock prediction (no model loaded)"
            
            # Binary mock prediction (weighted toward benign)
            disease = np.random.choice(['Benign', 'Melanoma'], p=[0.85, 0.15])
            confidence = np.random.uniform(75.0, 95.0)
        
        # Get disease info
        info = disease_info.get(disease, disease_info['Benign'])
        
        # Log the prediction
        print(f"Prediction: {disease} (Confidence: {round(confidence, 2)}%) - {model_status}")
        
        # Return result for binary classification
        result = {
            'disease': disease,
            'confidence': round(confidence, 2),
            'description': info['description'],
            'severity': info['severity'],
            'recommendations': info['recommendations'],
            'next_steps': info.get('next_steps', []),
            'using_mock_prediction': using_mock,
            'model_status': model_status
        }
        
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
            '/health': 'GET - Health check',
            '/predict': 'POST - Upload image for prediction',
            '/generate-report': 'POST - Generate patient report',
            '/doctors': 'GET - Get list of doctors',
            '/consult-doctor': 'POST - Send report to doctor'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier_model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/validate', methods=['POST'])
def validate_image():
    """Validate if uploaded image is a valid medical image using Gemini AI"""
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
        
        # Save file temporarily for validation
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            # Use Gemini AI to validate the image if available
            if GEMINI_AVAILABLE and validate_image_with_gemini:
                is_valid, validation_message = validate_image_with_gemini(filepath)
            else:
                # Skip validation if Gemini is not available
                is_valid = True
                validation_message = "Image uploaded successfully (AI validation unavailable)"
            
            if not is_valid:
                # Delete the file if validation fails
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
                return jsonify({
                    'valid': False,
                    'error': 'Invalid image for medical analysis',
                    'details': validation_message,
                    'code': 'VALIDATION_FAILED'
                }), 400
            
            # Image is valid - return success (keep file for prediction)
            return jsonify({
                'valid': True,
                'message': validation_message,
                'filename': filename
            }), 200
            
        except Exception as e:
            # Clean up file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
            
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        return jsonify({
            'valid': False,
            'error': 'Validation failed',
            'details': str(e),
            'code': 'VALIDATION_ERROR'
        }), 500


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
        
        print(f"Image saved to {filepath}, proceeding with prediction...")
        
        # Predict disease (validation already done at /validate endpoint)
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
        
        consultation_id = f'CONS{datetime.now().strftime("%Y%m%d%H%M%S")}'
        
        # Check if this doctor should receive real emails/WhatsApp
        if doctor_id in REAL_DOCTOR_EMAILS:
            report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
            doctor_email = REAL_DOCTOR_EMAILS[doctor_id]
            
            # Add doctor email to patient info for WhatsApp message
            patient_info['doctor_email'] = doctor_email
            
            # Send real email
            email_sent = send_email_to_doctor(
                doctor_email,
                doctor['name'],
                patient_info,
                report_path,
                consultation_id
            )
            
            # Send WhatsApp notification if available
            whatsapp_sent = False
            if doctor_id in REAL_DOCTOR_WHATSAPP:
                doctor_whatsapp = REAL_DOCTOR_WHATSAPP[doctor_id]
                whatsapp_sent = send_whatsapp_to_doctor(
                    doctor_whatsapp,
                    doctor['name'],
                    patient_info,
                    report_path,
                    consultation_id
                )
            
            if email_sent or whatsapp_sent:
                notification_methods = []
                if email_sent:
                    notification_methods.append('email')
                if whatsapp_sent:
                    notification_methods.append('WhatsApp')
                
                return jsonify({
                    'message': f'Report successfully sent to {doctor["name"]} via {" and ".join(notification_methods)}',
                    'doctor': doctor,
                    'consultation_id': consultation_id,
                    'email_sent': email_sent,
                    'whatsapp_sent': whatsapp_sent,
                    'notification_methods': notification_methods
                }), 200
            else:
                return jsonify({
                    'message': f'Report logged for {doctor["name"]} (delivery pending)',
                    'doctor': doctor,
                    'consultation_id': consultation_id,
                    'email_sent': False,
                    'whatsapp_sent': False,
                    'warning': 'Email/WhatsApp configuration required'
                }), 200
        else:
            # For demo doctors, just return success without sending email
            return jsonify({
                'message': f'Report successfully sent to {doctor["name"]}',
                'doctor': doctor,
                'consultation_id': consultation_id,
                'email_sent': False
            }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chatbot', methods=['POST'])
def chatbot():
    """AI Chatbot endpoint for answering patient queries"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        context = data.get('context', {})
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get AI response
        response = get_chatbot_response(user_message, context)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"Chatbot error: {str(e)}")
        return jsonify({
            'response': 'I apologize, but I encountered an error. Please try again or consult with a medical professional.',
            'error': str(e)
        }), 500


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
    
    print("\nLoading trained melanoma detection model...")
    model_status = load_models()
    
    if model_status and model_status.get('classifier'):
        print("\n✅ Trained model loaded successfully! Using real predictions.")
    else:
        print("\n⚠️ Trained model not loaded. Using mock predictions.")
        print("\nTo use real predictions, ensure the trained model is at:")
        print("  - models/best_model_20251103_225237.h5")
    
    print("\nInitializing server on port 5001...")
    print("=" * 80)
    app.run(debug=True, host='0.0.0.0', port=5001)
