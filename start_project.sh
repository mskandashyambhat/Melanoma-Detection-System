#!/bin/bash

# Melanoma Detection System - Complete Startup Script

echo "=========================================="
echo "   MELANOMA DETECTION SYSTEM LAUNCHER    "
echo "=========================================="
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Project Directory: $PROJECT_ROOT"
echo ""

# Function to check if a port is in use
check_port() {
    lsof -i :$1 > /dev/null 2>&1
    return $?
}

# Ask user what to start
echo "What would you like to start?"
echo "  1) Backend only (Flask API on port 5001)"
echo "  2) Frontend only (Vite dev server on port 5173)"
echo "  3) Both Backend and Frontend"
echo "  4) Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Starting Backend only..."
        echo "=========================================="
        
        # Check if backend port is already in use
        if check_port 5001; then
            echo "⚠️  Port 5001 is already in use!"
            echo "   Stopping existing process..."
            lsof -ti:5001 | xargs kill -9 2>/dev/null
            sleep 2
        fi
        
        cd "$PROJECT_ROOT/backend"
        chmod +x start_backend.sh
        ./start_backend.sh
        ;;
        
    2)
        echo ""
        echo "Starting Frontend only..."
        echo "=========================================="
        
        # Check if frontend port is already in use
        if check_port 5173; then
            echo "⚠️  Port 5173 is already in use!"
            echo "   Stopping existing process..."
            lsof -ti:5173 | xargs kill -9 2>/dev/null
            sleep 2
        fi
        
        cd "$PROJECT_ROOT/frontend"
        chmod +x start_frontend.sh
        ./start_frontend.sh
        ;;
        
    3)
        echo ""
        echo "Starting Both Backend and Frontend..."
        echo "=========================================="
        
        # Check and clean ports
        if check_port 5001; then
            echo "⚠️  Port 5001 is already in use! Stopping..."
            lsof -ti:5001 | xargs kill -9 2>/dev/null
            sleep 2
        fi
        
        if check_port 5173; then
            echo "⚠️  Port 5173 is already in use! Stopping..."
            lsof -ti:5173 | xargs kill -9 2>/dev/null
            sleep 2
        fi
        
        # Start backend in background
        echo ""
        echo "1️⃣  Starting Backend..."
        cd "$PROJECT_ROOT/backend"
        chmod +x start_backend.sh
        ./start_backend.sh > backend.log 2>&1 &
        BACKEND_PID=$!
        echo "   Backend started with PID: $BACKEND_PID"
        echo "   Logs: backend/backend.log"
        
        # Wait for backend to start
        echo "   Waiting for backend to start..."
        sleep 5
        
        # Start frontend
        echo ""
        echo "2️⃣  Starting Frontend..."
        cd "$PROJECT_ROOT/frontend"
        chmod +x start_frontend.sh
        
        echo ""
        echo "=========================================="
        echo "✅ SYSTEM STARTED SUCCESSFULLY"
        echo "=========================================="
        echo ""
        echo "Backend API:  http://localhost:5001"
        echo "Frontend App: http://localhost:5173"
        echo ""
        echo "Backend PID: $BACKEND_PID"
        echo "Backend Logs: $PROJECT_ROOT/backend/backend.log"
        echo ""
        echo "Press Ctrl+C to stop the frontend"
        echo "To stop backend: kill $BACKEND_PID"
        echo "=========================================="
        echo ""
        
        ./start_frontend.sh
        ;;
        
    4)
        echo "Exiting..."
        exit 0
        ;;
        
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac
