#!/bin/bash

# Stop script for Melanoma Detection System

echo "🛑 Stopping Melanoma Detection System..."
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Function to stop process by PID file
stop_by_pid_file() {
    local pid_file=$1
    local service_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            echo "   Stopping $service_name (PID: $pid)..."
            kill $pid 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if kill -0 $pid 2>/dev/null; then
                kill -9 $pid 2>/dev/null || true
            fi
            echo "   ✅ $service_name stopped"
        else
            echo "   ⚠️  $service_name process not running"
        fi
        rm -f "$pid_file"
    else
        echo "   ⚠️  No PID file found for $service_name"
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    local service_name=$2
    
    if lsof -i :$port > /dev/null 2>&1; then
        echo "   Stopping $service_name on port $port..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        echo "   ✅ Port $port freed"
    fi
}

# Stop using PID files first
stop_by_pid_file "$PROJECT_ROOT/.backend.pid" "Backend"
stop_by_pid_file "$PROJECT_ROOT/.frontend.pid" "Frontend"

# Also check and kill by port (backup method)
kill_port 5001 "Backend"
kill_port 5173 "Frontend"

echo ""
echo "✅ All services stopped successfully"
