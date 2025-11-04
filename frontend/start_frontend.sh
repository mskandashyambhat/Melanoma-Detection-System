#!/bin/bash

# Melanoma Detection Frontend Startup Script

echo "==================================="
echo "Melanoma Detection System Frontend"
echo "==================================="

# Navigate to frontend directory
cd "$(dirname "$0")"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "⚠️  node_modules not found."
    echo "Installing dependencies..."
    npm install
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

echo ""
echo "Starting development server..."
echo "Frontend will be available at http://localhost:5173"
echo "==================================="
echo ""

# Start the Vite dev server
npm run dev
