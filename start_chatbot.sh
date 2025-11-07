#!/bin/bash

# AI Medical Chatbot - Quick Start Script
# This script helps you get the chatbot up and running

echo "=========================================="
echo "AI Medical Chatbot - Quick Start"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the melanoma-detection root directory"
    exit 1
fi

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd backend

# Check if google-generativeai is installed
if python3 -c "import google.generativeai" 2>/dev/null; then
    echo "✅ google-generativeai already installed"
else
    echo "📥 Installing google-generativeai..."
    pip3 install google-generativeai==0.8.3
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To start the application:"
echo ""
echo "1. Start Backend (Terminal 1):"
echo "   cd backend"
echo "   python3 app.py"
echo ""
echo "2. Start Frontend (Terminal 2):"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "3. Access the application:"
echo "   http://localhost:5173"
echo ""
echo "4. Test the chatbot:"
echo "   - Upload a skin lesion image"
echo "   - View results"
echo "   - Click 'Ask AI Assistant' button"
echo ""
echo "=========================================="
echo "📚 Documentation:"
echo "   - CHATBOT_README.md (detailed docs)"
echo "   - CHATBOT_IMPLEMENTATION.md (summary)"
echo "=========================================="
