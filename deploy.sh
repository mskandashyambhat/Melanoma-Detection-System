#!/bin/bash

# Melanoma Detection System - Production Deployment Script
# This script sets up and runs both frontend and backend for web hosting

set -e  # Exit on error

echo "=========================================="
echo "  MELANOMA DETECTION - DEPLOYMENT SETUP  "
echo "=========================================="
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PROJECT_ROOT

echo "📁 Project Directory: $PROJECT_ROOT"
echo ""

# Check if .env file exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "❌ Error: .env file not found!"
    echo "   Please create .env file with your GEMINI_API_KEY"
    echo "   You can copy .env.example and update it:"
    echo "   cp .env.example .env"
    exit 1
fi

echo "✅ Environment file found"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."
echo ""

if ! command_exists python3; then
    echo "❌ Python3 is not installed!"
    exit 1
fi
echo "✅ Python3: $(python3 --version)"

if ! command_exists node; then
    echo "❌ Node.js is not installed!"
    exit 1
fi
echo "✅ Node.js: $(node --version)"

if ! command_exists npm; then
    echo "❌ npm is not installed!"
    exit 1
fi
echo "✅ npm: $(npm --version)"

echo ""
echo "=========================================="
echo "         BACKEND SETUP                    "
echo "=========================================="
echo ""

cd "$PROJECT_ROOT/backend"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet

# Install backend dependencies
echo "📦 Installing backend dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo "✅ Backend dependencies installed"
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "         FRONTEND SETUP                   "
echo "=========================================="
echo ""

cd "$PROJECT_ROOT/frontend"

# Install frontend dependencies
if [ -f "package.json" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install --silent
    echo "✅ Frontend dependencies installed"
else
    echo "❌ package.json not found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "    STARTING APPLICATION SERVERS         "
echo "=========================================="
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        echo "   Backend stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo "   Frontend stopped"
    fi
    echo "👋 Goodbye!"
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup INT TERM

# Check if ports are in use
check_port() {
    lsof -i :$1 > /dev/null 2>&1
    return $?
}

# Kill process on port if exists
kill_port() {
    if check_port $1; then
        echo "⚠️  Port $1 is in use. Stopping existing process..."
        lsof -ti:$1 | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

# Clean up ports
kill_port 5001
kill_port 5173

# Start Backend
echo "🚀 Starting Backend Server..."
cd "$PROJECT_ROOT/backend"
source venv/bin/activate

# Start backend in background
python3 app.py > "$PROJECT_ROOT/backend/backend.log" 2>&1 &
BACKEND_PID=$!

echo "   Backend PID: $BACKEND_PID"
echo "   Backend logs: $PROJECT_ROOT/backend/backend.log"

# Wait for backend to start
echo "   Waiting for backend to initialize..."
sleep 8

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start! Check logs at: $PROJECT_ROOT/backend/backend.log"
    tail -n 20 "$PROJECT_ROOT/backend/backend.log"
    exit 1
fi

echo "✅ Backend started successfully on port 5001"
echo ""

# Start Frontend
echo "🚀 Starting Frontend Server..."
cd "$PROJECT_ROOT/frontend"

# Start frontend in background
npm run dev > "$PROJECT_ROOT/frontend/frontend.log" 2>&1 &
FRONTEND_PID=$!

echo "   Frontend PID: $FRONTEND_PID"
echo "   Frontend logs: $PROJECT_ROOT/frontend/frontend.log"

# Wait for frontend to start
echo "   Waiting for frontend to initialize..."
sleep 5

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Frontend failed to start! Check logs at: $PROJECT_ROOT/frontend/frontend.log"
    tail -n 20 "$PROJECT_ROOT/frontend/frontend.log"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Frontend started successfully on port 5173"
echo ""

# Save PIDs to file for easy cleanup later
echo $BACKEND_PID > "$PROJECT_ROOT/.backend.pid"
echo $FRONTEND_PID > "$PROJECT_ROOT/.frontend.pid"

echo "=========================================="
echo "   ✅ DEPLOYMENT SUCCESSFUL!              "
echo "=========================================="
echo ""
echo "🌐 Application URLs:"
echo "   Frontend:  http://localhost:5173"
echo "   Backend:   http://localhost:5001"
echo ""
echo "📊 Process Information:"
echo "   Backend PID:  $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo ""
echo "📝 Log Files:"
echo "   Backend:  $PROJECT_ROOT/backend/backend.log"
echo "   Frontend: $PROJECT_ROOT/frontend/frontend.log"
echo ""
echo "🛑 To stop the servers:"
echo "   Press Ctrl+C (will stop both servers)"
echo "   Or run: ./stop.sh"
echo ""
echo "📖 View logs in real-time:"
echo "   Backend:  tail -f $PROJECT_ROOT/backend/backend.log"
echo "   Frontend: tail -f $PROJECT_ROOT/frontend/frontend.log"
echo ""
echo "=========================================="
echo ""

# Keep script running and show logs
echo "📊 Monitoring servers (Press Ctrl+C to stop)..."
echo ""

# Monitor both processes
while true; do
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend process died! Check logs."
        kill $FRONTEND_PID 2>/dev/null || true
        exit 1
    fi
    
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "❌ Frontend process died! Check logs."
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    
    sleep 5
done
