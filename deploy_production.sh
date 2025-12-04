#!/bin/bash

# Production deployment script for Melanoma Detection System
# Optimized for cloud hosting (AWS, Google Cloud, Azure, DigitalOcean, etc.)

set -e

echo "=========================================="
echo " MELANOMA DETECTION - PRODUCTION DEPLOY  "
echo "=========================================="
echo ""

# Configuration
HOST="${HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-5001}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
NODE_ENV="${NODE_ENV:-production}"

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PROJECT_ROOT

echo "📋 Configuration:"
echo "   Host: $HOST"
echo "   Backend Port: $BACKEND_PORT"
echo "   Frontend Port: $FRONTEND_PORT"
echo "   Environment: $NODE_ENV"
echo ""

# Check .env file
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "❌ Error: .env file not found!"
    echo "   Create .env file with required variables"
    exit 1
fi

# Load environment variables
export $(cat "$PROJECT_ROOT/.env" | grep -v '^#' | xargs)

# Verify required environment variables
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY not set in .env file"
    exit 1
fi

echo "✅ Environment variables loaded"
echo ""

# Backend Setup
echo "=========================================="
echo "         BACKEND SETUP (PRODUCTION)       "
echo "=========================================="
echo ""

cd "$PROJECT_ROOT/backend"

# Setup virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install gunicorn --quiet  # Production WSGI server

echo "✅ Backend setup complete"
echo ""

# Frontend Setup
echo "=========================================="
echo "        FRONTEND SETUP (PRODUCTION)       "
echo "=========================================="
echo ""

cd "$PROJECT_ROOT/frontend"

# Install dependencies
echo "📦 Installing Node.js dependencies..."
npm install --silent

# Build frontend for production
echo "🏗️  Building frontend for production..."
npm run build --silent

echo "✅ Frontend built successfully"
echo ""

# Create production start script
cat > "$PROJECT_ROOT/start_production.sh" << 'EOF'
#!/bin/bash

# Start production servers
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
HOST="${HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-5001}"

# Start backend with Gunicorn
cd "$PROJECT_ROOT/backend"
source venv/bin/activate

echo "🚀 Starting production backend with Gunicorn..."
gunicorn -w 4 -b $HOST:$BACKEND_PORT --timeout 120 --access-logfile - --error-logfile - app:app &
BACKEND_PID=$!

echo $BACKEND_PID > "$PROJECT_ROOT/.backend.pid"
echo "✅ Backend started (PID: $BACKEND_PID)"

# Serve frontend build
cd "$PROJECT_ROOT/frontend"
echo "🚀 Starting frontend server..."
npm run preview -- --host $HOST --port ${FRONTEND_PORT:-5173} &
FRONTEND_PID=$!

echo $FRONTEND_PID > "$PROJECT_ROOT/.frontend.pid"
echo "✅ Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "=========================================="
echo "   PRODUCTION DEPLOYMENT COMPLETE         "
echo "=========================================="
echo "   Backend:  http://$HOST:$BACKEND_PORT"
echo "   Frontend: http://$HOST:${FRONTEND_PORT:-5173}"
echo "=========================================="

# Keep running
wait
EOF

chmod +x "$PROJECT_ROOT/start_production.sh"

echo "=========================================="
echo "   PRODUCTION BUILD COMPLETE              "
echo "=========================================="
echo ""
echo "📦 Production files ready:"
echo "   Backend: $PROJECT_ROOT/backend/"
echo "   Frontend build: $PROJECT_ROOT/frontend/dist/"
echo ""
echo "🚀 To start production servers:"
echo "   ./start_production.sh"
echo ""
echo "🌐 Or use individual commands:"
echo "   Backend:  cd backend && source venv/bin/activate && gunicorn -w 4 -b 0.0.0.0:5001 app:app"
echo "   Frontend: cd frontend && npm run preview -- --host 0.0.0.0 --port 5173"
echo ""
echo "=========================================="
