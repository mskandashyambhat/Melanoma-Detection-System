"""
AI Medical Chatbot using Gemini AI
Provides medical information and answers patient queries about skin conditions
"""

import google.generativeai as genai

# Configure Gemini API with the provided key
GEMINI_API_KEY = "AIzaSyB4hxXg2ta6cw6f1FmBlIjG6wB6dxpCwKc"
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
        print(f"Error in chatbot: {str(e)}")
        return """I apologize, but I'm having trouble processing your request right now. 

For immediate medical concerns, please:
- Contact your healthcare provider
- Visit an urgent care center
- Call emergency services if it's an emergency

Is there anything else I can help clarify about your diagnosis?"""

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
