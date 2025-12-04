"""
COMBINE OLD + NEW MODELS FOR ENSEMBLE
=====================================
Combines best_model_20251103_225237.h5 + resnet50_resumed model
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score
import os

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for imbalanced data"""
    def __init__(self, gamma=3.0, alpha=0.8, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        melanoma_pred = y_pred[:, 1]
        melanoma_true = y_true
        cross_entropy = -melanoma_true * tf.math.log(melanoma_pred) - (1 - melanoma_true) * tf.math.log(1 - melanoma_pred)
        weight = melanoma_true * self.alpha + (1 - melanoma_true) * (1 - self.alpha)
        focal_weight = tf.pow(tf.abs(melanoma_true - melanoma_pred), self.gamma)
        return tf.reduce_mean(weight * focal_weight * cross_entropy)

class CombinedEnsemblePredictor:
    """Combines old and new models"""

    def __init__(self):
        self.models = []
        self.model_names = []

    def load_models(self):
        """Load both old and new models"""
        script_dir = Path(__file__).parent
        model_dir = script_dir / 'models'

        # Old model
        old_path = model_dir / 'best_model_20251103_225237.h5'
        if old_path.exists():
            print(f"Loading old model: {old_path}")
            self.models.append(tf.keras.models.load_model(old_path))
            self.model_names.append('Old Model')
        else:
            print("❌ Old model not found")

        # New high-recall model
        new_models = list(model_dir.glob('resnet50_*.h5'))
        if new_models:
            latest_new = max(new_models, key=lambda x: x.stat().st_mtime)
            print(f"Loading new model: {latest_new}")
            self.models.append(tf.keras.models.load_model(
                latest_new,
                custom_objects={'FocalLoss': FocalLoss, 'focal_loss': FocalLoss}
            ))
            self.model_names.append('New High-Recall Model')
        else:
            print("❌ New model not found")

        print(f"✅ Loaded {len(self.models)} models")

    def ensemble_predict(self, X, method='weighted_average'):
        """Ensemble prediction on batch"""
        if not self.models:
            raise ValueError("No models loaded")

        # Get predictions from all models
        all_predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X, batch_size=32, verbose=0)
            
            # Handle multi-output models (hybrid model outputs [segmentation, classification])
            if isinstance(pred, list):
                pred = pred[1]  # Take classification output
            
            # Normalize predictions to 2 classes [benign, melanoma]
            if pred.shape[1] == 1:
                # Sigmoid output: convert to 2-class probability
                pred = np.hstack([1 - pred, pred])  # [benign_prob, melanoma_prob]
            
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples, n_classes)

        if method == 'average':
            # Simple average
            ensemble_pred = np.mean(all_predictions, axis=0)

        elif method == 'weighted_average':
            # Weight new model more (higher recall)
            weights = np.array([1.0, 2.0])  # New model gets 2x weight
            weights = weights / np.sum(weights[:len(self.models)])

            ensemble_pred = np.average(all_predictions, axis=0, weights=weights)

        elif method == 'majority_vote':
            # Majority voting
            class_preds = np.argmax(all_predictions, axis=2)  # (n_models, n_samples)
            ensemble_classes = []
            for sample_idx in range(class_preds.shape[1]):
                votes = class_preds[:, sample_idx]
                majority_class = np.bincount(votes).argmax()
                ensemble_classes.append(majority_class)

            # Convert to probabilities (simplified)
            ensemble_pred = np.zeros((len(ensemble_classes), 2))
            for i, cls in enumerate(ensemble_classes):
                ensemble_pred[i, cls] = 0.6  # Fake confidence
                ensemble_pred[i, 1-cls] = 0.4

        return ensemble_pred

    def evaluate_ensemble(self):
        """Evaluate ensemble on validation data"""
        print("\n" + "="*80)
        print("🎯 ENSEMBLE EVALUATION: OLD + NEW MODELS")
        print("="*80)

        # Load validation data
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / 'data' / 'ham10000_binary'
        val_X = np.load(data_dir / 'val_X.npy')
        val_y = np.load(data_dir / 'val_y.npy')

        print(f"Evaluating on {len(val_X)} validation samples...")

        methods = ['average', 'weighted_average', 'majority_vote']

        for method in methods:
            print(f"\n{'='*60}")
            print(f"📊 METHOD: {method.upper()}")
            print('='*60)

            # Get ensemble predictions
            ensemble_pred = self.ensemble_predict(val_X, method=method)
            y_pred = np.argmax(ensemble_pred, axis=1)

            # Classification report
            print("\n📋 Classification Report:")
            print(classification_report(
                val_y, y_pred,
                target_names=['Benign', 'Melanoma'],
                digits=4
            ))

            # Confusion matrix
            cm = confusion_matrix(val_y, y_pred)
            print("\n🔢 Confusion Matrix:")
            print(f"                 Predicted")
            print(f"                 Benign  Melanoma")
            print(f"Actual Benign    {cm[0][0]:6d}  {cm[0][1]:6d}")
            print(f"       Melanoma  {cm[1][0]:6d}  {cm[1][1]:6d}")

            # Key metrics
            melanoma_recall = recall_score(val_y, y_pred, pos_label=1)
            overall_acc = np.mean(y_pred == val_y)

            print(f"\n🎯 KEY METRICS:")
            print(f"   Overall Accuracy: {overall_acc:.4f}")
            print(f"   Melanoma Recall:  {melanoma_recall:.4f}")
            print(f"   Detected: {cm[1][1]}/{cm[1][0] + cm[1][1]} melanomas")
            print(f"   Missed: {cm[1][0]} (False Negatives)")

            if melanoma_recall >= 0.95:
                print(f"\n✅ EXCELLENT! Near-perfect recall")
            elif melanoma_recall >= 0.80:
                print(f"\n⚡ VERY GOOD! High recall")
            else:
                print(f"\n⚠️  Moderate recall")

# Simple wrapper for API usage
class EnsemblePredictor:
    """Simple wrapper for Flask API usage"""
    
    def __init__(self, model_paths):
        """Load models from given paths"""
        self.models = []
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = tf.keras.models.load_model(
                        path,
                        custom_objects={'FocalLoss': FocalLoss, 'focal_loss': FocalLoss}
                    )
                    self.models.append(model)
                    print(f"✅ Loaded: {path}")
                except Exception as e:
                    print(f"❌ Failed to load {path}: {e}")
        
        if not self.models:
            raise ValueError("No models loaded successfully")
    
    def ensemble_predict(self, X, method='weighted_average'):
        """Make ensemble prediction"""
        all_predictions = []
        for model in self.models:
            pred = model.predict(X, batch_size=32, verbose=0)
            
            # Handle multi-output models
            if isinstance(pred, list):
                pred = pred[1]
            
            # Normalize to 2-class format
            if pred.shape[1] == 1:
                pred = np.hstack([1 - pred, pred])
            
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        if method == 'weighted_average':
            # Weight new model 2x
            weights = np.array([1.0, 2.0])[:len(self.models)]
            weights = weights / np.sum(weights)
            return np.average(all_predictions, axis=0, weights=weights)
        else:
            return np.mean(all_predictions, axis=0)


if __name__ == '__main__':
    print("="*80)
    print("🔄 COMBINING OLD + NEW MODELS")
    print("="*80)

    ensemble = CombinedEnsemblePredictor()
    ensemble.load_models()

    if ensemble.models:
        ensemble.evaluate_ensemble()
    else:
        print("❌ No models loaded!")
