# Melanoma Detection System - Setup Instructions

## Quick Start Guide

### 1. Backend Setup (5 minutes)

Open a terminal and run:

```bash
# Navigate to backend folder
cd melanoma-detection/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

The backend will start on http://localhost:5000

### 2. Frontend Setup (3 minutes)

Open a new terminal and run:

```bash
# Navigate to frontend folder
cd melanoma-detection/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will start on http://localhost:3000

### 3. Access the Application

Open your browser and go to: http://localhost:3000

## Common Issues & Solutions

### Issue: "Module not found" errors in Python
**Solution**: Make sure you're in the virtual environment
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### Issue: Port already in use
**Solution**: Change the port in the respective config files
- Backend: Edit `app.py`, change `port=5000` to another port
- Frontend: Edit `vite.config.js`, change `port: 3000` to another port

### Issue: CORS errors
**Solution**: Make sure both backend and frontend are running and the backend URL in frontend matches (check src/pages/*.jsx files)

### Issue: Model not found warnings
**Solution**: This is normal! The app will work with mock predictions. To use real models:
1. Train your models using `train_model.py`
2. Place trained models in `backend/models/` folder

## Using the Application

### Step 1: Upload Image
1. Go to home page
2. Drag and drop or click to upload a skin lesion image
3. Click "Analyze Image"

### Step 2: View Results
1. See the analysis results with confidence scores
2. Read the description and recommendations
3. Click "Download Report" to get PDF

### Step 3: Fill Patient Information
1. Enter patient details in the form
2. Click "Generate Report"
3. Report will be downloaded automatically

### Step 4: Consult a Doctor
1. Click "Consult a Doctor" button
2. Browse available doctors
3. Select a doctor to send your report
4. Doctor will receive your report and contact you

## Training Your Own Model (Optional)

If you want to train the model on your own dataset:

1. Prepare your dataset:
```
dataset/
├── train/
│   ├── melanoma/
│   ├── acne/
│   ├── ringworm/
│   ├── burns/
│   ├── eczema/
│   ├── psoriasis/
│   ├── basal_cell_carcinoma/
│   └── normal/
└── validation/
    └── (same structure as train)
```

2. Update paths in `train_model.py`

3. Run training:
```bash
cd backend
python train_model.py
```

## Production Deployment

### Backend (Flask)
Use a production WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Frontend (React)
Build and serve:
```bash
cd frontend
npm run build
# Serve the dist folder with any static server
```

### Docker Deployment (Recommended)
Create `Dockerfile` for each service and use docker-compose.

## Support

If you encounter any issues:
1. Check the console for error messages
2. Ensure all dependencies are installed
3. Verify both servers are running
4. Check the README.md for detailed documentation

Happy detecting! 🔬
