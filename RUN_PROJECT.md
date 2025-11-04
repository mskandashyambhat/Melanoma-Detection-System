# 🚀 Melanoma Detection System - Quick Start Guide

## Overview
This is a full-stack melanoma detection system using:
- **Backend**: Flask API with trained TensorFlow model
- **Frontend**: React + Vite application
- **Model**: UNet + ResNet50 binary classifier (89.42% validation accuracy)

## 📋 Prerequisites

### System Requirements
- **macOS** (current setup)
- **Python 3.8+**
- **Node.js 16+** and npm
- **8GB+ RAM** (for model loading)

### Required Software
```bash
# Check Python version
python3 --version  # Should be 3.8 or higher

# Check Node.js version
node --version     # Should be 16 or higher
npm --version
```

## 🎯 Quick Start (Easiest Method)

### Option 1: Start Everything with One Command
```bash
./start_project.sh
```

Then select option 3 to start both backend and frontend.

### Option 2: Start Backend and Frontend Separately

**Terminal 1 - Backend:**
```bash
cd backend
./start_backend.sh
```

**Terminal 2 - Frontend:**
```bash
cd frontend
./start_frontend.sh
```

## 🔧 Manual Setup (If Needed)

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model exists:**
   ```bash
   ls -lh models/best_model_20251103_225237.h5
   ```
   Should show: `~486M` file

5. **Start the Flask server:**
   ```bash
   python app.py
   ```
   
   ✅ Backend will be available at: `http://localhost:5001`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```
   
   ✅ Frontend will be available at: `http://localhost:5173`

## 🧪 Testing the System

### 1. Check Backend Health
```bash
curl http://localhost:5001/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-04T..."
}
```

### 2. Access Frontend
Open browser: `http://localhost:5173`

### 3. Test Image Upload
1. Click "Upload Image" on the homepage
2. Select a skin lesion image
3. View prediction results
4. Check confidence score and recommendations

## 📊 Model Information

- **Model File**: `backend/models/best_model_20251103_225237.h5`
- **Architecture**: UNet encoder + ResNet50 backbone
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Melanoma vs Benign)
- **Performance**: 
  - Validation Accuracy: 89.42%
  - Test Accuracy: ~88.89%
  - ROC AUC: 0.843

## 🔍 Troubleshooting

### Backend Issues

**Port 5001 already in use:**
```bash
# Kill process on port 5001
lsof -ti:5001 | xargs kill -9
```

**Model not loading:**
- Check file exists: `ls backend/models/best_model_20251103_225237.h5`
- Check file size: Should be ~486MB
- Check permissions: `chmod 644 backend/models/best_model_20251103_225237.h5`

**Dependencies not installing:**
```bash
# Upgrade pip
pip install --upgrade pip

# Try installing individually
pip install flask flask-cors tensorflow numpy pillow opencv-python
```

### Frontend Issues

**Port 5173 already in use:**
```bash
# Kill process on port 5173
lsof -ti:5173 | xargs kill -9
```

**node_modules issues:**
```bash
# Clean install
rm -rf node_modules package-lock.json
npm install
```

**Backend connection failed:**
1. Ensure backend is running: `curl http://localhost:5001/health`
2. Check CORS settings in `backend/app.py`
3. Verify `frontend/src/config.js` has correct API URL

## 🌐 API Endpoints

### Health Check
```bash
GET http://localhost:5001/health
```

### Upload & Predict
```bash
POST http://localhost:5001/upload
Content-Type: multipart/form-data
Body: image file
```

### Get Doctors List
```bash
GET http://localhost:5001/api/doctors
```

## 📁 Project Structure

```
melanoma-detection/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── requirements.txt       # Python dependencies
│   ├── start_backend.sh       # Backend startup script
│   ├── models/
│   │   └── best_model_20251103_225237.h5  # Trained model
│   ├── uploads/               # Temporary image uploads
│   └── reports/               # Generated reports
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main React component
│   │   ├── config.js         # API configuration
│   │   └── pages/            # Page components
│   ├── package.json          # Node dependencies
│   └── start_frontend.sh     # Frontend startup script
│
├── data/
│   └── ham10000_binary/      # Preprocessed training data
│
└── start_project.sh          # Master startup script
```

## 🎨 Features

- ✅ Real-time melanoma detection using trained CNN model
- ✅ Image preprocessing and enhancement
- ✅ Confidence scoring and risk assessment
- ✅ Detailed recommendations and next steps
- ✅ Doctor consultation system
- ✅ PDF report generation
- ✅ Responsive modern UI with Tailwind CSS
- ✅ ABCDE rule information for self-examination

## 🔐 Security Notes

- Images are temporarily stored and should be deleted after processing
- No patient data is permanently stored (demo version)
- CORS is enabled for development (restrict in production)
- Model predictions are for educational/demo purposes only

## 📝 Next Steps for Production

1. **Security Enhancements:**
   - Add authentication and authorization
   - Implement rate limiting
   - Secure file upload validation
   - HTTPS/SSL certificates

2. **Database Integration:**
   - Store user profiles and prediction history
   - Track model performance metrics
   - Save consultation records

3. **Model Improvements:**
   - Regularly retrain with new data
   - Implement model monitoring
   - Add explainability features (Grad-CAM)

4. **Deployment:**
   - Dockerize the application
   - Set up CI/CD pipeline
   - Deploy to cloud (AWS/GCP/Azure)
   - Configure production environment variables

## 📞 Support

For issues or questions:
- Check logs: `backend/backend.log`
- Review console output in both terminals
- Ensure all dependencies are installed
- Verify model file integrity

## ⚠️ Disclaimer

This system is for educational and demonstration purposes only. It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for actual medical advice.

---

**Model Training Date**: November 3, 2025  
**Best Epoch**: 11/20 (89.42% validation accuracy)  
**Dataset**: HAM10000 (Binary: Melanoma vs Benign)
