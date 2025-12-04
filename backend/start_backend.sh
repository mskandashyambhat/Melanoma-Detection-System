#!/bin/bash

# Backend Startup Script for Melanoma Detection System

echo "=========================================="
echo "     STARTING MELANOMA DETECTION API     "
echo "=========================================="
echo ""

# Get the backend directory
BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Backend Directory: $BACKEND_DIR"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run setup first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
echo "📦 Checking dependencies..."
python3 -c "import flask, tensorflow, cv2, PIL, google.generativeai, reportlab" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependencies not installed!"
    echo "   Installing requirements..."
    pip install -r requirements.txt
fi

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1

# Start the Flask application
echo ""
echo "🚀 Starting Flask API server on port 5001..."
echo "   API will be available at: http://localhost:5001"
echo "   Press Ctrl+C to stop the server"
echo ""

python3 app.py