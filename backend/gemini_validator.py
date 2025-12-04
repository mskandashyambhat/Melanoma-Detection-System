"""
Gemini AI Image Validator
Uses Google's Gemini AI to validate if an image is a valid medical/skin lesion image
"""

import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file in parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")
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
        error_msg = str(e).lower()
        print(f"❌ Gemini validation error: {str(e)}")
        
        # Handle quota exceeded - fall back to basic validation
        if "quota" in error_msg or "429" in error_msg or "exceeded" in error_msg:
            print("⚠️  Gemini API quota exceeded - using fallback validation")
            return fallback_image_validation(image_path)
        
        # FAIL CLOSED - reject if we can't validate
        return False, "Unable to validate image with AI. Please try a clearer image of a skin lesion."


def fallback_image_validation(image_path):
    """
    Basic fallback validation when Gemini API is unavailable
    
    Args:
        image_path: Path to the uploaded image
        
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    try:
        from PIL import Image
        import os
        
        # Check if file exists and is a valid image
        if not os.path.exists(image_path):
            return False, "Image file not found"
            
        img = Image.open(image_path)
        
        # Check image dimensions (too small images might not be useful)
        width, height = img.size
        if width < 100 or height < 100:
            return False, "Image is too small. Please upload a higher resolution image."
            
        # Check file size (reasonable limit)
        file_size = os.path.getsize(image_path)
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            return False, "Image file is too large. Please upload a smaller image (max 10MB)."
            
        # Accept common image formats
        valid_formats = ['JPEG', 'JPG', 'PNG', 'BMP', 'TIFF']
        if img.format not in valid_formats:
            return False, f"Unsupported image format. Please use: {', '.join(valid_formats)}"
            
        # Basic validation passed
        return True, "Image accepted for analysis (AI validation temporarily unavailable)"
        
    except Exception as e:
        print(f"❌ Fallback validation error: {str(e)}")
        return False, "Unable to process the uploaded image. Please try again."

if __name__ == "__main__":
    # Test the validator
    test_image = "uploads/test.jpg"
    if os.path.exists(test_image):
        is_valid, message = validate_image_with_gemini(test_image)
        print(f"Valid: {is_valid}")
        print(f"Message: {message}")
    else:
        print("Test image not found")
