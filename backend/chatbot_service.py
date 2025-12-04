"""
AI Medical Chatbot using Gemini AI
Provides medical information and answers patient queries about skin conditions
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file in parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configure Gemini API with the provided key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

def get_chatbot_response(user_message, context):
    """
    Generate AI chatbot response for medical queries
    
    Args:
        user_message: User's question/message
        context: Dictionary containing diagnosis context (disease, severity, etc.)
        
    Returns:
        str: AI-generated response
    """
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Build context-aware system prompt
        disease = context.get('disease', 'Unknown')
        severity = context.get('severity', 'Unknown')
        confidence = context.get('confidence', 0)
        recommendations = context.get('recommendations', [])
        melanoma_stage = context.get('melanoma_stage', None)
        
        system_context = f"""You are a helpful and empathetic AI Medical Assistant specializing in dermatology and skin conditions. 

PATIENT CONTEXT:
- Detected Condition: {disease}
- Severity Level: {severity}
- Confidence: {confidence}%
{f'- Melanoma Stage: {melanoma_stage}' if melanoma_stage else ''}

IMPORTANT GUIDELINES:
1. You are NOT a replacement for professional medical advice
2. You CANNOT prescribe medications or provide specific treatment plans
3. You CAN provide general medical information about skin conditions
4. You CAN suggest over-the-counter products for benign conditions:
   - For benign lesions: Gentle moisturizers, soothing creams (aloe vera, vitamin E)
   - Sun protection: SPF 50+ sunscreen, protective clothing
   - General skin care: Gentle cleansers, hypoallergenic products
5. If asked about prescription medications, politely explain you cannot prescribe but can explain what types of treatments exist in general
6. Always emphasize the importance of consulting qualified dermatologists
7. Be compassionate and reassuring while being medically accurate
8. For melanoma cases, stress the importance of urgent medical attention
9. Answer questions related to:
   - Understanding the diagnosis
   - What the condition means
   - General prevention tips
   - Lifestyle modifications
   - When to seek immediate medical help
   - How to monitor the condition
   - General skin care advice

CURRENT RECOMMENDATIONS:
{chr(10).join(f'- {rec}' for rec in recommendations)}

USER QUESTION: {user_message}

Provide a helpful, accurate, and compassionate response. Keep it conversational and easy to understand. If the question is not related to medical/health topics, politely redirect the conversation to health-related matters."""

        # Generate response
        response = model.generate_content(system_context)
        
        return response.text.strip()
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"❌ Chatbot error: {str(e)}")
        
        # Handle quota exceeded - provide helpful fallback responses
        if "quota" in error_msg or "429" in error_msg or "exceeded" in error_msg:
            print("⚠️  Chatbot API quota exceeded - using fallback responses")
            return get_fallback_chatbot_response(user_message, context)
        
        # Generic fallback for other errors
        return """I apologize, but I'm having trouble processing your request right now. 

For immediate medical concerns, please:
- Contact your healthcare provider
- Visit an urgent care center
- Call emergency services if it's an emergency

Is there anything else I can help clarify about your diagnosis?"""


def get_fallback_chatbot_response(user_message, context):
    """
    Provide helpful fallback responses when AI is unavailable
    
    Args:
        user_message: User's question
        context: Diagnosis context
        
    Returns:
        str: Helpful fallback response
    """
    disease = context.get('disease', 'Unknown').lower()
    severity = context.get('severity', 'Unknown').lower()
    user_msg_lower = user_message.lower()
    
    # Handle common questions with pre-written responses
    if any(word in user_msg_lower for word in ['what', 'mean', 'is', 'tell me about']):
        if 'melanoma' in disease:
            return """Melanoma is a serious form of skin cancer that develops in the cells that produce melanin (the pigment that gives skin its color). It's the most dangerous type of skin cancer because it can spread to other parts of the body if not caught early.

**Key points:**
- Early detection is crucial for successful treatment
- Regular skin checks and professional dermatologist visits are essential
- Protect your skin from UV radiation with sunscreen and protective clothing

Please consult a dermatologist for personalized information about your specific case."""
        
        elif 'benign' in disease:
            return """A benign skin lesion means the growth is non-cancerous and not harmful. Most skin lesions are benign, and while they may need monitoring, they typically don't require urgent treatment.

**General care tips:**
- Continue regular skin checks
- Use sunscreen to protect your skin
- Monitor for any changes in size, color, or shape
- Consult a dermatologist if you notice any concerning changes

Your condition appears to be non-cancerous, which is good news!"""
    
    elif any(word in user_msg_lower for word in ['treatment', 'cure', 'medicine', 'doctor']):
        if 'melanoma' in disease:
            return """For melanoma treatment, you should consult a dermatologist or oncologist immediately. Treatment options may include:

- Surgical removal of the lesion
- Additional testing to determine if cancer has spread
- Possible chemotherapy, immunotherapy, or radiation therapy

**Important:** I'm not a doctor and cannot provide specific treatment recommendations. Please seek professional medical care promptly for melanoma cases."""
        
        else:
            return """For treatment recommendations, please consult a qualified healthcare professional. They can provide personalized advice based on your specific condition.

In the meantime, general skin health tips include:
- Use broad-spectrum SPF 30+ sunscreen daily
- Wear protective clothing when outdoors
- Avoid tanning beds
- Perform regular skin self-examinations
- See a dermatologist annually for professional skin checks"""
    
    elif any(word in user_msg_lower for word in ['symptom', 'sign', 'look for', 'check']):
        return """**ABCDE rule for skin cancer detection:**
- **A**symmetry: One half doesn't match the other
- **B**order: Irregular, scalloped, or poorly defined border
- **C**olor: Varied colors within the lesion
- **D**iameter: Larger than 6mm (pencil eraser)
- **E**volving: Changes in size, shape, or color over time

**When to see a doctor:**
- Any new or changing skin lesions
- Lesions that bleed, itch, or don't heal
- Family history of skin cancer
- Fair skin, many moles, or history of sunburns

Regular professional skin examinations are recommended."""
    
    elif any(word in user_msg_lower for word in ['prevention', 'prevent', 'avoid', 'risk']):
        return """**Skin Cancer Prevention:**
- Use broad-spectrum SPF 30+ sunscreen daily
- Apply sunscreen 15-30 minutes before sun exposure
- Reapply every 2 hours, or immediately after swimming/sweating
- Wear protective clothing, hats, and sunglasses
- Seek shade during peak sun hours (10 AM - 4 PM)
- Avoid tanning beds completely
- Perform monthly skin self-examinations
- Get annual professional skin checks

**Additional risk factors to be aware of:**
- Fair skin, light hair, and blue eyes
- History of sunburns, especially in childhood
- Family history of skin cancer
- Many moles or atypical moles
- Weakened immune system"""
    
    else:
        # Generic helpful response
        return f"""I understand you're asking about: "{user_message}"

While I'm currently experiencing technical difficulties with my AI assistant, I can still provide some general information about skin health and when to seek medical attention.

**General advice:**
- Monitor your skin regularly for any changes
- Use sun protection daily
- Consult healthcare professionals for personalized advice
- Early detection is key for skin conditions

For your specific question about {disease}, I recommend consulting with a dermatologist who can provide detailed, personalized information.

Is there a specific aspect of your skin condition you'd like general information about?"""

def get_initial_greeting(context):
    """
    Generate personalized initial greeting based on diagnosis
    
    Args:
        context: Dictionary containing diagnosis context
        
    Returns:
        str: Personalized greeting message
    """
    disease = context.get('disease', 'Unknown')
    severity = context.get('severity', 'Unknown')
    
    if disease == 'Melanoma':
        return """Hello! I'm your AI Medical Assistant. I understand this diagnosis may be concerning. I'm here to help answer your questions and provide information about melanoma and next steps.

Please remember that while I can provide general information, it's crucial to consult with a dermatologist or oncologist for personalized medical advice.

What would you like to know?"""
    elif disease == 'Benign':
        return """Hello! Good news - your skin lesion appears to be benign. I'm here to answer any questions you might have about maintaining healthy skin and monitoring your condition.

What would you like to know?"""
    else:
        return """Hello! I'm your AI Medical Assistant. I'm here to help answer questions about your skin condition. 

How can I assist you today?"""
