#!/bin/bash

# Melanoma Detection Backend Startup Script

echo "=================================="
echo "Melanoma Detection System Backend"
echo "=================================="

# Navigate to backend directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found."
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/.requirements_installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch venv/.requirements_installed
    echo "✅ Requirements installed"
else
    echo "✅ Requirements already installed"
fi

# Check if model exists
if [ -f "models/best_model_20251103_225237.h5" ]; then
    echo "✅ Trained model found"
else
    echo "⚠️  Trained model not found at models/best_model_20251103_225237.h5"
    echo "   The system will use mock predictions."
fi

echo ""
echo "Starting Flask server on http://localhost:5001"
echo "=================================="
echo ""

# Start the Flask app
python app.py
