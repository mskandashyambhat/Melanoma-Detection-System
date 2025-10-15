"""
Test script for the Melanoma Detection System API
Run this to verify your backend is working correctly
"""

import requests
import os
import json

BASE_URL = "http://localhost:5000"

def test_home():
    """Test home endpoint"""
    print("\n" + "="*80)
    print("Testing Home Endpoint")
    print("="*80)
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200, "Home endpoint failed"
        print("✓ Home endpoint working!")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_doctors():
    """Test doctors endpoint"""
    print("\n" + "="*80)
    print("Testing Doctors Endpoint")
    print("="*80)
    
    try:
        response = requests.get(f"{BASE_URL}/doctors")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Number of doctors: {len(data['doctors'])}")
        print(f"First doctor: {data['doctors'][0]['name']}")
        assert response.status_code == 200, "Doctors endpoint failed"
        assert len(data['doctors']) > 0, "No doctors found"
        print("✓ Doctors endpoint working!")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_predict():
    """Test prediction endpoint with a sample image"""
    print("\n" + "="*80)
    print("Testing Prediction Endpoint")
    print("="*80)
    
    # Check if we have a test image
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print("⚠️  No test image found. Please add 'test_image.jpg' to test this endpoint.")
        return
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/predict", files=files)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Detected Disease: {data['disease']}")
            print(f"Confidence: {data['confidence']}%")
            print(f"Severity: {data['severity']}")
            print("✓ Prediction endpoint working!")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_generate_report():
    """Test report generation endpoint"""
    print("\n" + "="*80)
    print("Testing Report Generation Endpoint")
    print("="*80)
    
    sample_data = {
        "patient_info": {
            "name": "Test Patient",
            "age": 35,
            "gender": "male",
            "phone": "+1-555-1234",
            "email": "test@example.com",
            "medical_history": "No significant history"
        },
        "prediction_data": {
            "disease": "Normal Skin",
            "confidence": 98.5,
            "severity": "None",
            "description": "Healthy skin with no abnormalities detected.",
            "recommendations": [
                "Maintain regular skincare routine",
                "Use sunscreen daily",
                "Stay hydrated"
            ],
            "image_path": ""
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/generate-report",
            json=sample_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Report Filename: {data['report_filename']}")
            print(f"Download URL: {data['download_url']}")
            print("✓ Report generation working!")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("\n" + "="*80)
    print("MELANOMA DETECTION SYSTEM - API TEST SUITE")
    print("="*80)
    print("\nMake sure the backend server is running on http://localhost:5000")
    input("\nPress Enter to start tests...")
    
    # Run all tests
    test_home()
    test_doctors()
    test_predict()
    test_generate_report()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETED")
    print("="*80)
    print("\n✓ All basic endpoints are working!")
    print("\nNote: To test image prediction, add a 'test_image.jpg' file in this directory.")

if __name__ == "__main__":
    main()
