# Melanoma Detection System

AI-powered melanoma detection system using deep learning models with React frontend and Flask backend.

## Features

- 🔬 Advanced melanoma detection using hybrid CNN models
- 🤖 AI-powered medical chatbot using Gemini AI
- 📊 Comprehensive medical reports with visualizations
- 🖼️ Image validation using Gemini AI
- 📱 Responsive web interface

## Prerequisites

- Python 3.8+
- Node.js 14+
- Gemini API Key (Google AI)

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/mskandashyambhat/Melanoma-Detection-System.git
cd melanoma-detection
```

### 2. Configure Environment

Create `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

**Get Gemini API Key:** Visit [Google AI Studio](https://makersuite.google.com/app/apikey)

### 3. Deploy the Application

#### Option A: Development Mode (Recommended for local testing)

```bash
./deploy.sh
```

This will:
- ✅ Install all dependencies automatically
- ✅ Start backend on http://localhost:5001
- ✅ Start frontend on http://localhost:5173
- ✅ Monitor both servers
- ✅ Create log files for debugging

#### Option B: Production Mode (For web hosting)

```bash
./deploy_production.sh
./start_production.sh
```

This will:
- ✅ Build optimized production bundles
- ✅ Use Gunicorn for backend (production-ready)
- ✅ Serve optimized frontend build
- ✅ Configure for cloud hosting

### 4. Control the Application

**Check status:**
```bash
./status.sh
```

**Stop servers:**
```bash
./stop.sh
```

**View logs:**
```bash
# Backend logs
tail -f backend/backend.log

# Frontend logs
tail -f frontend/frontend.log
```

## Deployment Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `deploy.sh` | Development deployment | Local development & testing |
| `deploy_production.sh` | Production build | Preparing for web hosting |
| `start_production.sh` | Start production servers | After running deploy_production.sh |
| `status.sh` | Check server status | Monitoring deployment |
| `stop.sh` | Stop all servers | Shutdown or restart |

## Manual Setup (Alternative)

### Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend
python3 app.py
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Development mode
npm run dev

# Production build
npm run build
npm run preview
```

## Project Structure

```
melanoma-detection/
├── backend/
│   ├── app.py                    # Main Flask application
│   ├── chatbot_service.py        # AI chatbot with Gemini
│   ├── gemini_validator.py       # Image validation
│   ├── models/                   # Trained ML models (.h5)
│   ├── uploads/                  # User uploaded images
│   ├── reports/                  # Generated PDF reports
│   ├── requirements.txt          # Python dependencies
│   └── venv/                     # Virtual environment
│
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── pages/               # Page components
│   │   ├── App.jsx              # Main application
│   │   └── config.js            # Configuration
│   ├── dist/                    # Production build
│   ├── package.json             # Node dependencies
│   └── vite.config.js           # Vite configuration
│
├── models/                       # Additional trained models
├── data/                         # Preprocessed datasets
│   └── ham10000_binary/         # Binary numpy arrays
│
├── .env                         # Environment variables (create this!)
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
│
├── deploy.sh                    # Development deployment
├── deploy_production.sh         # Production build script
├── start_production.sh          # Production start script
├── status.sh                    # Status checker
├── stop.sh                      # Stop all servers
└── README.md                    # This file
```

## Cloud Deployment

### Deploy to AWS EC2, Google Cloud, DigitalOcean, etc.

1. **Upload project to server:**
```bash
scp -r melanoma-detection/ user@your-server:/path/
```

2. **SSH into server:**
```bash
ssh user@your-server
cd /path/melanoma-detection
```

3. **Setup environment:**
```bash
# Copy and configure .env
cp .env.example .env
nano .env  # Add your GEMINI_API_KEY
```

4. **Deploy:**
```bash
./deploy_production.sh
./start_production.sh
```

5. **Configure firewall:**
```bash
# Allow ports (adjust based on your firewall)
sudo ufw allow 5001/tcp
sudo ufw allow 5173/tcp
```

### Using Docker (Optional)

Create `Dockerfile` in backend:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app:app"]
```

Build and run:
```bash
docker build -t melanoma-backend ./backend
docker run -d -p 5001:5001 --env-file .env melanoma-backend
```

## API Endpoints

### Health Check
```
GET /api/health
```

### Image Analysis
```
POST /api/analyze
Content-Type: multipart/form-data
Body: image file

Response: {
  "disease": "melanoma",
  "confidence": 0.95,
  "severity": "high",
  "recommendations": [...]
}
```

### Chatbot
```
POST /api/chatbot
Content-Type: application/json
Body: {
  "message": "What is melanoma?",
  "context": {...}
}

Response: {
  "response": "AI-generated response..."
}
```

### Image Validation
```
POST /api/validate-image
Content-Type: multipart/form-data
Body: image file

Response: {
  "is_valid": true,
  "message": "Valid skin lesion image"
}
```

## Models

The system uses:
- **ResNet50** - Melanoma classification
- **U-Net** - Lesion segmentation
- **Hybrid Model** - Ensemble predictions
- **Gemini AI** - Image validation & chatbot

## Environment Variables

Required in `.env` file:

```bash
# Gemini API (Required)
GEMINI_API_KEY=your_api_key_here

# Server Configuration (Optional)
HOST=0.0.0.0
BACKEND_PORT=5001
FRONTEND_PORT=5173
NODE_ENV=production

# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_ADDRESS=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
```

## Troubleshooting

### Backend Issues

**Port already in use:**
```bash
./stop.sh
# Or manually: lsof -ti:5001 | xargs kill -9
```

**Gemini API errors:**
- Check API key in `.env`
- Verify API quota at [Google AI Studio](https://makersuite.google.com/)
- Ensure `python-dotenv` is installed

**Models not loading:**
- Verify `.h5` model files exist in `backend/models/`
- Check file permissions
- Ensure sufficient disk space

**Dependencies error:**
```bash
cd backend
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Frontend Issues

**Build fails:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Port 5173 in use:**
```bash
./stop.sh
# Or: lsof -ti:5173 | xargs kill -9
```

### Logs

Check logs for detailed error information:
```bash
# Backend
tail -f backend/backend.log

# Frontend
tail -f frontend/frontend.log

# Check status
./status.sh
```

## Performance Optimization

### Backend
- Uses Gunicorn with 4 workers in production
- TensorFlow optimized for inference
- Image preprocessing with OpenCV

### Frontend
- Vite for fast builds
- Code splitting and lazy loading
- Optimized production bundle

## Security Notes

- ✅ `.env` file is in `.gitignore` (never committed)
- ✅ API keys stored in environment variables
- ✅ CORS configured for security
- ✅ Input validation on all endpoints
- ⚠️  Change default ports for production
- ⚠️  Use HTTPS in production (nginx/Apache reverse proxy)
- ⚠️  Set up proper firewall rules

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/mskandashyambhat/Melanoma-Detection-System/issues)
- Email: support@example.com

## Authors

- **Skanda Shyam Bhat M** - *Initial work* - [mskandashyambhat](https://github.com/mskandashyambhat)

## Acknowledgments

- HAM10000 Dataset for training data
- Google Gemini AI for image validation and chatbot
- TensorFlow/Keras for deep learning framework
- React and Vite for frontend framework
