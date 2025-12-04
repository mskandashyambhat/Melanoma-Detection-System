#!/bin/bash

# Quick status check for Melanoma Detection System

echo "=========================================="
echo "   MELANOMA DETECTION SYSTEM STATUS      "
echo "=========================================="
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Function to check if a port is in use and get PID
check_port_status() {
    local port=$1
    local service_name=$2
    
    echo "🔍 $service_name (Port $port):"
    if lsof -i :$port > /dev/null 2>&1; then
        local pid=$(lsof -ti:$port)
        echo "   ✅ RUNNING (PID: $pid)"
        
        # Try to curl the endpoint
        if [ "$port" == "5001" ]; then
            if curl -s http://localhost:$port/api/health > /dev/null 2>&1; then
                echo "   ✅ Health check: OK"
            else
                echo "   ⚠️  Health check: Not responding"
            fi
        elif [ "$port" == "5173" ]; then
            if curl -s http://localhost:$port > /dev/null 2>&1; then
                echo "   ✅ Responding to requests"
            else
                echo "   ⚠️  Not responding"
            fi
        fi
    else
        echo "   ❌ NOT RUNNING"
    fi
    echo ""
}

# Check Backend
check_port_status 5001 "Backend API"

# Check Frontend
check_port_status 5173 "Frontend Server"

# Check PID files
echo "📋 PID Files:"
if [ -f "$PROJECT_ROOT/.backend.pid" ]; then
    echo "   Backend PID file: $(cat $PROJECT_ROOT/.backend.pid)"
else
    echo "   Backend PID file: Not found"
fi

if [ -f "$PROJECT_ROOT/.frontend.pid" ]; then
    echo "   Frontend PID file: $(cat $PROJECT_ROOT/.frontend.pid)"
else
    echo "   Frontend PID file: Not found"
fi
echo ""

# Check log files
echo "📝 Log Files:"
if [ -f "$PROJECT_ROOT/backend/backend.log" ]; then
    local lines=$(wc -l < "$PROJECT_ROOT/backend/backend.log")
    echo "   Backend log: $lines lines"
    echo "   Last 3 lines:"
    tail -n 3 "$PROJECT_ROOT/backend/backend.log" | sed 's/^/      /'
else
    echo "   Backend log: Not found"
fi
echo ""

if [ -f "$PROJECT_ROOT/frontend/frontend.log" ]; then
    local lines=$(wc -l < "$PROJECT_ROOT/frontend/frontend.log")
    echo "   Frontend log: $lines lines"
else
    echo "   Frontend log: Not found"
fi
echo ""

# Check .env file
echo "🔐 Environment:"
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "   ✅ .env file found"
    if grep -q "GEMINI_API_KEY=" "$PROJECT_ROOT/.env" 2>/dev/null; then
        if grep -q "GEMINI_API_KEY=.*[A-Za-z0-9]" "$PROJECT_ROOT/.env" 2>/dev/null; then
            echo "   ✅ GEMINI_API_KEY is set"
        else
            echo "   ⚠️  GEMINI_API_KEY is empty"
        fi
    else
        echo "   ❌ GEMINI_API_KEY not found in .env"
    fi
else
    echo "   ❌ .env file not found"
fi

echo ""
echo "=========================================="
