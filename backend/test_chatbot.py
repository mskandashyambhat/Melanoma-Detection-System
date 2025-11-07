"""
Test script for the AI Medical Chatbot
"""

import sys
sys.path.append('/Users/skandashyam/Documents/Mini-Project/melanoma-detection/backend')

from chatbot_service import get_chatbot_response, get_initial_greeting

# Test context for Melanoma
melanoma_context = {
    'disease': 'Melanoma',
    'severity': 'High',
    'confidence': 87.5,
    'recommendations': [
        'Schedule appointment with dermatologist within 1-2 days',
        'Avoid sun exposure and use SPF 50+ sunscreen',
        'Take photos to track any changes'
    ],
    'melanoma_stage': 2
}

# Test context for Benign
benign_context = {
    'disease': 'Benign',
    'severity': 'Low',
    'confidence': 92.3,
    'recommendations': [
        'No immediate medical action required',
        'Continue regular skin self-examinations monthly',
        'Use sunscreen daily (SPF 30+)'
    ]
}

def test_chatbot():
    print("=" * 80)
    print("AI MEDICAL CHATBOT TEST")
    print("=" * 80)
    
    # Test 1: Initial greeting for Melanoma
    print("\n[Test 1] Initial Greeting for Melanoma Patient:")
    print("-" * 80)
    greeting = get_initial_greeting(melanoma_context)
    print(greeting)
    
    # Test 2: Question about melanoma
    print("\n[Test 2] Question: 'What does melanoma mean?'")
    print("-" * 80)
    response = get_chatbot_response("What does melanoma mean?", melanoma_context)
    print(response)
    
    # Test 3: Question about medications
    print("\n[Test 3] Question: 'Can you prescribe me some medication?'")
    print("-" * 80)
    response = get_chatbot_response("Can you prescribe me some medication?", melanoma_context)
    print(response)
    
    # Test 4: Benign condition - skincare advice
    print("\n[Test 4] Benign - Question: 'What cream should I use?'")
    print("-" * 80)
    response = get_chatbot_response("What cream should I use for my benign mole?", benign_context)
    print(response)
    
    # Test 5: Sun protection
    print("\n[Test 5] Question: 'How can I protect myself from sun?'")
    print("-" * 80)
    response = get_chatbot_response("How can I protect myself from the sun?", benign_context)
    print(response)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

if __name__ == '__main__':
    test_chatbot()
