"""
Gemini AI Image Validator
Uses Google's Gemini AI to validate if an image is a valid medical/skin lesion image
"""

import google.generativeai as genai
from PIL import Image
import os

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyCFkVSiMG9BoQ1QmH-A7-Bg9P8XH71bCVo"
genai.configure(api_key=GEMINI_API_KEY)

def validate_image_with_gemini(image_path):
    """
    Use Gemini AI to validate if the image is a valid skin lesion/medical image
    
    Args:
        image_path: Path to the uploaded image
        
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Initialize Gemini model - use the latest available model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Create prompt for validation
        prompt = """Is this image a SKIN LESION or MOLE? Answer ONLY with YES or NO.

If it shows skin with a lesion, mole, rash, or any skin condition: say YES
If it shows shoes, furniture, objects, or anything else: say NO

Answer:"""
        
        # Generate response
        response = model.generate_content([prompt, img])
        result = response.text.strip().upper()
        
        print(f"🤖 Gemini AI validation result: {result}")
        
        if "YES" in result:
            return True, "Valid skin lesion image"
        else:
            return False, "This does not appear to be a valid skin lesion image. Please upload a photo of a skin lesion, mole, or skin condition."
            
    except Exception as e:
        print(f"❌ Gemini validation error: {str(e)}")
        # FAIL CLOSED - reject if we can't validate
        return False, "Unable to validate image with AI. Please try a clearer image of a skin lesion."

if __name__ == "__main__":
    # Test the validator
    test_image = "uploads/test.jpg"
    if os.path.exists(test_image):
        is_valid, message = validate_image_with_gemini(test_image)
        print(f"Valid: {is_valid}")
        print(f"Message: {message}")
    else:
        print("Test image not found")
