# 🚀 Quick Start Guide - Binary Melanoma Detection

## ⚡ One-Page Quick Reference

### 📋 Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended, not required)
- HAM10000 dataset downloaded

---

## 🎯 5-Step Workflow

### Step 1️⃣: Install Dependencies (5 minutes)
```bash
cd backend
pip install -r requirements_training.txt
```

### Step 2️⃣: Place Dataset (2 minutes)
```
backend/data/HAM10000/
├── HAM10000_metadata.csv
└── HAM10000_images/
    └── [10,015 .jpg files]
```

### Step 3️⃣: Preprocess Data (5-10 minutes)
```bash
cd backend
python preprocess_ham10000_binary.py
```
**Output**: `data/ham10000_binary/` with train/val/test .npy files

### Step 4️⃣: Train Model (15-30 minutes GPU, 2-4 hours CPU)
```bash
cd backend/models_training
python train_binary_model.py
```
**Output**: `models/` folder with 5 files:
- `combined_model_final_*.h5` ← Main model
- `unet_model_final_*.h5`
- `resnet_model_final_*.h5`
- `training_history_final_*.json`
- `model_metadata_final_*.json`

### Step 5️⃣: Evaluate & Visualize (10-15 minutes)
```bash
cd backend
python run_complete_evaluation.py \
    models/combined_model_final_20231101_120000.h5 \
    models/unet_model_final_20231101_120000.h5 \
    models/resnet_model_final_20231101_120000.h5 \
    models/training_history_final_20231101_120000.json
```
*(Replace timestamps with your actual file names)*

**Output**: `visualization/outputs/` with 13+ PNG files and reports

---

## 📊 What You'll Get

### Trained Models (3)
| Model | Purpose | Size |
|-------|---------|------|
| Combined | Full pipeline (UNet→ResNet50) | ~150 MB |
| UNet | Segmentation only | ~60 MB |
| ResNet50 | Classification only | ~90 MB |

### Evaluation Results (3 JSON files)
- UNet efficiency metrics
- ResNet50 classification metrics
- Combined end-to-end metrics

### Visualizations (13+ images)
1. **Confusion Matrix** - Classification accuracy breakdown
2. **ROC Curve** - True vs False positive rates
3. **Metrics Bar Chart** - All metrics compared
4. **Metrics Summary** - Comprehensive overview
5. **Accuracy Curve** - Training progress
6. **Loss Curve** - Training convergence
7. **Training History** - Combined view
8. **All Metrics** - Precision, recall, etc.
9. **Grad-CAM** - Model attention maps (8 samples)
10. **Sample Outputs** - Correct predictions (16 samples)
11. **Misclassified** - Error analysis (16 samples)
12. **Model Comparison** - UNet vs ResNet50 vs Combined
13. **Comparison Report** - Text summary

---

## 🎯 Performance Targets

| Metric | Target | What It Means |
|--------|--------|---------------|
| Accuracy | >90% | Overall correctness |
| Precision | >85% | When it says "Melanoma", it's usually right |
| Recall | >85% | Catches most melanoma cases |
| F1-Score | >85% | Balance of precision & recall |
| AUC | >0.90 | Overall discrimination ability |
| Specificity | >90% | Correctly identifies benign cases |
| Inference | <100ms | Fast enough for real-time use |

---

## 📂 Key Files Reference

### Training & Evaluation
```
backend/
├── preprocess_ham10000_binary.py     # Run 1st: Prepare data
├── models_training/
│   └── train_binary_model.py         # Run 2nd: Train models
└── run_complete_evaluation.py        # Run 3rd: Evaluate all
```

### Individual Evaluators (run separately if needed)
```
backend/evaluation/
├── evaluate_unet.py           # UNet segmentation analysis
├── evaluate_resnet50.py       # ResNet50 classification analysis
└── evaluate_combined.py       # Full pipeline analysis
```

### Individual Visualizers (run separately if needed)
```
backend/visualization/
├── visualize_metrics.py       # Confusion matrix, ROC, metrics
├── visualize_training.py      # Accuracy/loss curves
├── visualize_gradcam.py       # Grad-CAM, samples, errors
└── visualize_comparison.py    # Model comparison charts
```

---

## 🔧 Common Commands

### Check GPU Availability
```bash
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

### Monitor Training Progress
Training outputs live metrics. Look for:
- `val_accuracy` (validation accuracy)
- `val_loss` (validation loss)
- Best epoch will be saved automatically

### Quick Test Model
```bash
cd backend
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('models/combined_model_final_*.h5')
print('Model loaded successfully!')
print(f'Parameters: {model.count_params():,}')
"
```

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of memory** | Reduce batch_size in `train_binary_model.py` (line 100) from 64 to 32 or 16 |
| **Training too slow** | Increase batch_size to 128 (if GPU allows), or reduce epochs to 15 |
| **Dataset not found** | Check `backend/data/HAM10000/` has metadata.csv and images folder |
| **Import errors** | Run `pip install -r requirements_training.txt` |
| **GPU not detected** | Install CUDA toolkit, or train on CPU (slower but works) |

---

## 📈 Customization Quick Tips

### For Better Accuracy (Slower)
In `train_binary_model.py`:
```python
trainer = BinaryMelanomaTrainer(
    batch_size=32,    # Smaller batches
    epochs=50         # More epochs
)
```

### For Faster Training (May Reduce Accuracy)
```python
trainer = BinaryMelanomaTrainer(
    batch_size=128,   # Larger batches
    epochs=15         # Fewer epochs
)
```

### For Smaller Model (Faster Inference)
In `unet_resnet50_binary.py` line ~200:
```python
def build_resnet50_classifier(self, trainable_layers=5):  # Reduce from 10
```

---

## 📝 Output Locations

After running everything, find your results here:

```
backend/
├── data/ham10000_binary/           # Preprocessed data
├── models/                          # Trained models
├── evaluation/results/              # Evaluation JSON files
└── visualization/outputs/           # All visualizations
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── gradcam_visualizations.png
    ├── model_comparison.png
    ├── EVALUATION_SUMMARY.txt       # ← Read this first!
    └── [10+ more visualizations]
```

---

## ✅ Success Checklist

- [ ] Dependencies installed (`pip install -r requirements_training.txt`)
- [ ] HAM10000 dataset in `backend/data/HAM10000/`
- [ ] Preprocessing completed (`preprocess_ham10000_binary.py`)
- [ ] Model training completed (`train_binary_model.py`)
- [ ] Models saved in `backend/models/`
- [ ] Evaluation completed (`run_complete_evaluation.py`)
- [ ] Visualizations generated in `backend/visualization/outputs/`
- [ ] Reviewed `EVALUATION_SUMMARY.txt`
- [ ] All metrics meet target thresholds
- [ ] Ready to deploy to production

---

## 🎓 Understanding Your Results

### Confusion Matrix
```
              Predicted
              No Mel  Melanoma
Actual No Mel   TN      FP     ← FP = False alarms
Actual Melanoma FN      TP     ← FN = Missed cases
```
- **TN**: Correctly identified benign (good!)
- **TP**: Correctly identified melanoma (good!)
- **FP**: False alarm - said melanoma but was benign (not ideal)
- **FN**: Missed melanoma - said benign but was melanoma (dangerous!)

### Medical Metrics
- **Sensitivity (Recall)**: % of melanomas caught
- **Specificity**: % of benign cases correctly identified
- **PPV (Precision)**: When test says melanoma, how often is it correct?
- **NPV**: When test says benign, how often is it correct?

---

## 🚀 Ready to Go!

Everything is set up. Just run the 5 steps above and you'll have:
- ✅ Binary melanoma classifier (no staging)
- ✅ Time-efficient model (batch 64)
- ✅ Comprehensive evaluation
- ✅ 13+ visualizations
- ✅ Full performance analysis

**Need help? Check the detailed guides:**
- `TRAINING_GUIDE.md` - Comprehensive guide
- `PROJECT_SUMMARY.md` - What changed & why
- `CHECKLIST.md` - Complete task list

**Happy training! 🎉**
