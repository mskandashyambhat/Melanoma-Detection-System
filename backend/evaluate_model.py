"""
═══════════════════════════════════════════════════════════════════
    MELANOMA DETECTION MODEL - COMPREHENSIVE EVALUATION
═══════════════════════════════════════════════════════════════════

Generates comprehensive evaluation metrics and visualizations:
- Accuracy, Precision, Recall, F1 Score
- Loss Function curves (Training & Validation)
- Confusion Matrix
- ROC Curve and AUC Score
- Precision-Recall Curve
- Classification Report

Results saved to: inference/ directory
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations"""
    
    def __init__(self, model_path, data_dir='data/ham10000_binary', output_dir='inference'):
        """
        Initialize the model evaluator
        
        Args:
            model_path: Path to trained model (.h5 file)
            data_dir: Directory containing test data (.npy files)
            output_dir: Directory to save evaluation results
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model
        print(f"Loading model from: {self.model_path}")
        self.model = keras.models.load_model(str(self.model_path))
        print("✅ Model loaded successfully")
        
        # Load test data
        print(f"\nLoading test data from: {self.data_dir}")
        self.X_test = np.load(self.data_dir / 'test_X.npy')
        self.y_test = np.load(self.data_dir / 'test_y.npy')
        print(f"✅ Test data loaded: {self.X_test.shape[0]} samples")
        
        # Load dataset info
        with open(self.data_dir / 'dataset_info.json', 'r') as f:
            self.dataset_info = json.load(f)
        
        self.class_names = ['Benign (No Melanoma)', 'Melanoma']
        
        # Evaluation results
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
        
    def predict(self):
        """Generate predictions on test set"""
        print("\n" + "="*70)
        print("GENERATING PREDICTIONS")
        print("="*70)
        
        # Get predicted probabilities
        self.y_pred_proba = self.model.predict(self.X_test, verbose=1)
        
        # Get predicted classes
        self.y_pred = (self.y_pred_proba > 0.5).astype(int).flatten()
        
        print(f"✅ Predictions generated for {len(self.y_pred)} samples")
        
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        print("\n" + "="*70)
        print("CALCULATING METRICS")
        print("="*70)
        
        # Basic metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, zero_division=0)
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall AUC
        avg_precision = average_precision_score(self.y_test, self.y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'roc_auc': roc_auc,
            'pr_auc': avg_precision,
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'total_samples': int(len(self.y_test)),
            'melanoma_samples': int(np.sum(self.y_test)),
            'benign_samples': int(len(self.y_test) - np.sum(self.y_test))
        }
        
        # Print metrics
        print("\n" + "="*70)
        print("EVALUATION METRICS")
        print("="*70)
        print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision:    {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:       {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1 Score:     {f1:.4f} ({f1*100:.2f}%)")
        print(f"Specificity:  {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"ROC AUC:      {roc_auc:.4f}")
        print(f"PR AUC:       {avg_precision:.4f}")
        print("\n" + "="*70)
        print("CONFUSION MATRIX")
        print("="*70)
        print(f"True Negatives:  {tn:4d} (Correctly identified as Benign)")
        print(f"False Positives: {fp:4d} (Benign misclassified as Melanoma)")
        print(f"False Negatives: {fn:4d} (Melanoma misclassified as Benign)")
        print(f"True Positives:  {tp:4d} (Correctly identified as Melanoma)")
        print("="*70)
        
    def plot_confusion_matrix(self):
        """Plot and save confusion matrix"""
        print("\nGenerating confusion matrix plot...")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Melanoma Detection', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
        
    def plot_roc_curve(self):
        """Plot and save ROC curve"""
        print("Generating ROC curve...")
        
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve - Melanoma Detection', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
        
    def plot_precision_recall_curve(self):
        """Plot and save Precision-Recall curve"""
        print("Generating Precision-Recall curve...")
        
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkgreen', lw=2, 
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve - Melanoma Detection', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'precision_recall_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
        
    def plot_metrics_comparison(self):
        """Plot bar chart comparing all metrics"""
        print("Generating metrics comparison chart...")
        
        metrics_to_plot = {
            'Accuracy': self.metrics['accuracy'],
            'Precision': self.metrics['precision'],
            'Recall': self.metrics['recall'],
            'F1 Score': self.metrics['f1_score'],
            'Specificity': self.metrics['specificity'],
            'ROC AUC': self.metrics['roc_auc']
        }
        
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(metrics_to_plot.keys(), metrics_to_plot.values(), 
                      color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'],
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}\n({height*100:.2f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics - Melanoma Detection', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'metrics_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
        
    def plot_class_distribution(self):
        """Plot class distribution in predictions"""
        print("Generating class distribution chart...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # True distribution
        true_counts = [np.sum(self.y_test == 0), np.sum(self.y_test == 1)]
        colors = ['#3498db', '#e74c3c']
        ax1.pie(true_counts, labels=self.class_names, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('True Class Distribution', fontsize=14, fontweight='bold', pad=15)
        
        # Predicted distribution
        pred_counts = [np.sum(self.y_pred == 0), np.sum(self.y_pred == 1)]
        ax2.pie(pred_counts, labels=self.class_names, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold', pad=15)
        
        plt.suptitle('Class Distribution Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'class_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
        
    def plot_prediction_confidence(self):
        """Plot distribution of prediction confidence scores"""
        print("Generating prediction confidence distribution...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Confidence for Benign predictions
        benign_conf = self.y_pred_proba[self.y_test == 0]
        ax1.hist(benign_conf, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax1.set_xlabel('Predicted Probability (Melanoma)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('True Benign Cases - Prediction Confidence', 
                     fontsize=12, fontweight='bold', pad=10)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Confidence for Melanoma predictions
        melanoma_conf = self.y_pred_proba[self.y_test == 1]
        ax2.hist(melanoma_conf, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax2.set_xlabel('Predicted Probability (Melanoma)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('True Melanoma Cases - Prediction Confidence', 
                     fontsize=12, fontweight='bold', pad=10)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.suptitle('Prediction Confidence Distribution', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'prediction_confidence.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
        
    def generate_classification_report(self):
        """Generate and save classification report"""
        print("Generating classification report...")
        
        report = classification_report(self.y_test, self.y_pred, 
                                       target_names=self.class_names,
                                       digits=4)
        
        report_path = self.output_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MELANOMA DETECTION MODEL - CLASSIFICATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(report)
            f.write("\n\n" + "="*70 + "\n")
            f.write("DETAILED METRICS\n")
            f.write("="*70 + "\n")
            for key, value in self.metrics.items():
                if key != 'confusion_matrix':
                    if isinstance(value, float):
                        f.write(f"{key.replace('_', ' ').title():20s}: {value:.4f} ({value*100:.2f}%)\n")
                    else:
                        f.write(f"{key.replace('_', ' ').title():20s}: {value}\n")
        
        print(f"✅ Saved: {report_path}")
        
        # Also print to console
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(report)
        
    def save_metrics_json(self):
        """Save metrics to JSON file"""
        print("Saving metrics to JSON...")
        
        # Prepare metrics dictionary
        metrics_output = {
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': str(self.model_path),
            'test_samples': int(len(self.y_test)),
            'metrics': {
                'accuracy': float(self.metrics['accuracy']),
                'precision': float(self.metrics['precision']),
                'recall': float(self.metrics['recall']),
                'f1_score': float(self.metrics['f1_score']),
                'specificity': float(self.metrics['specificity']),
                'roc_auc': float(self.metrics['roc_auc']),
                'pr_auc': float(self.metrics['pr_auc'])
            },
            'confusion_matrix': self.metrics['confusion_matrix'],
            'class_distribution': {
                'benign_samples': int(self.metrics['benign_samples']),
                'melanoma_samples': int(self.metrics['melanoma_samples'])
            }
        }
        
        json_path = self.output_dir / 'evaluation_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(metrics_output, f, indent=4)
        
        print(f"✅ Saved: {json_path}")
        
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "="*70)
        print("MELANOMA DETECTION MODEL - COMPREHENSIVE EVALUATION")
        print("="*70)
        print(f"Model: {self.model_path.name}")
        print(f"Output Directory: {self.output_dir}")
        print("="*70)
        
        # Generate predictions
        self.predict()
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Generate all plots
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_metrics_comparison()
        self.plot_class_distribution()
        self.plot_prediction_confidence()
        
        # Save reports
        print("\n" + "="*70)
        print("SAVING REPORTS")
        print("="*70)
        self.generate_classification_report()
        self.save_metrics_json()
        
        # Final summary
        print("\n" + "="*70)
        print("EVALUATION COMPLETE ✅")
        print("="*70)
        print(f"All results saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  📊 confusion_matrix.png")
        print("  📈 roc_curve.png")
        print("  📉 precision_recall_curve.png")
        print("  📊 metrics_comparison.png")
        print("  🥧 class_distribution.png")
        print("  📊 prediction_confidence.png")
        print("  📄 classification_report.txt")
        print("  📄 evaluation_metrics.json")
        print("="*70)
        
        return self.metrics


def main():
    """Main execution function"""
    # Find the latest model
    models_dir = Path('models')
    model_files = list(models_dir.glob('*.h5'))
    
    if not model_files:
        print("❌ No trained models found in 'models/' directory")
        return
    
    # Use the latest model
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    print("\n" + "="*70)
    print("MELANOMA DETECTION MODEL EVALUATION")
    print("="*70)
    print(f"Using model: {latest_model.name}")
    print("="*70)
    
    # Run evaluation
    evaluator = ModelEvaluator(
        model_path=latest_model,
        data_dir='../data/ham10000_binary',
        output_dir='../inference'
    )
    
    metrics = evaluator.run_full_evaluation()
    
    return metrics


if __name__ == "__main__":
    main()
