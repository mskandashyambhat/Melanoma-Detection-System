"""
MELANOMA DETECTION MODEL TRAINING - SIMPLE & FAST
=================================================

EfficientNetB0 Classifier (~5.3M parameters)

Features:
- 10 epochs (fast training)
- Simple EfficientNetB0 architecture (no UNet, no ResNet50)
- Class weights for imbalanced dataset
- Early stopping and learning rate scheduling
- 5-10x faster than previous UNet+ResNet50 pipeline

Binary Classification: Melanoma vs Benign
Dataset: HAM10000 (preprocessed .npy files)
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import simple model architecture
from model_simple import SimpleMelanomaModel as MelanomaDetectionModel


class ModelTrainer:
    """Handles training of the melanoma detection model"""
    
    def __init__(self, data_dir='data/ham10000/augmented', model_dir='models'):
        """
        Initialize the model trainer
        
        Args:
            data_dir: Directory containing preprocessed data (use 'augmented' for 3x dataset)
            model_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize model builder
        self.model_builder = MelanomaDetectionModel(input_shape=(224, 224, 3))
        
        # Training history
        self.history = None
        self.model = None
        self.unet_model = None
        self.resnet_model = None
        
    def load_data(self):
        """Load preprocessed data from .npy files"""
        print("\n📂 Loading preprocessed data...")
        
        # Load training data
        self.train_X = np.load(self.data_dir / 'train_X.npy')
        self.train_y = np.load(self.data_dir / 'train_y.npy')
        
        # Load validation data
        self.val_X = np.load(self.data_dir / 'validation_X.npy')
        self.val_y = np.load(self.data_dir / 'validation_y.npy')
        
        # Load test data
        self.test_X = np.load(self.data_dir / 'test_X.npy')
        self.test_y = np.load(self.data_dir / 'test_y.npy')
        
        print(f"✅ Training set: {self.train_X.shape[0]} images")
        print(f"✅ Validation set: {self.val_X.shape[0]} images")
        print(f"✅ Test set: {self.test_X.shape[0]} images")
        
        # Print class distribution
        train_melanoma = np.sum(self.train_y)
        train_benign = len(self.train_y) - train_melanoma
        print(f"\n📊 Training set distribution:")
        print(f"   Benign: {train_benign} ({100*train_benign/len(self.train_y):.1f}%)")
        print(f"   Melanoma: {train_melanoma} ({100*train_melanoma/len(self.train_y):.1f}%)")
        
        return True
    
    def calculate_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.train_y),
            y=self.train_y
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print(f"\n⚖️  Class weights calculated:")
        print(f"   Benign (0): {class_weight_dict[0]:.3f}")
        print(f"   Melanoma (1): {class_weight_dict[1]:.3f}")
        
        return class_weight_dict
    
    def train(self, epochs=50, batch_size=32):
        """
        Train the combined UNet + ResNet50 model
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        print("\n" + "="*70)
        print("🚀 STARTING MELANOMA DETECTION MODEL TRAINING")
        print("="*70)
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔢 Epochs: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        print("="*70 + "\n")
        
        # Calculate class weights
        class_weights = self.calculate_class_weights()
        
        # Build the simple model
        print("\n🏗️  Building EfficientNetB0 classifier...")
        self.model = self.model_builder.build_model()
        
        print("\n📋 Model Summary:")
        print("-" * 70)
        self.model.summary()
        print("-" * 70)
        
        # Compile the model
        print("\n⚙️  Compiling model...")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',  # For integer labels with softmax output
            metrics=['accuracy']  # Keep it simple to avoid shape mismatch issues
        )
        
        # Setup callbacks
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / f'melanoma_model_{timestamp}_best.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=str(self.model_dir / 'logs' / timestamp),
                histogram_freq=1
            )
        ]
        
        print("\n🎯 Starting training with enhanced augmentation...")
        print("   ✨ Augmentations: Flip, Rotation (±30%), Zoom (±30%), Translation (±20%)")
        print("   ✨ Color: Brightness (±30%), Contrast (±30%), Hue, Saturation")
        print("   ✨ Noise: Gaussian noise added")
        print("")
        
        # Train the model
        self.history = self.model.fit(
            self.train_X, self.train_y,
            validation_data=(self.val_X, self.val_y),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = self.model_dir / f'melanoma_model_final_{timestamp}.keras'
        self.model.save(final_model_path)
        print(f"\n💾 Final model saved to: {final_model_path}")
        
        # Save ResNet50 classifier separately (for API use)
        resnet_model = self.model_builder.build_resnet_classifier()
        resnet_model.set_weights(self.model.get_layer('resnet_classifier').get_weights())
        resnet_path = self.model_dir / f'resnet_classifier_{timestamp}.keras'
        resnet_model.save(resnet_path)
        print(f"💾 ResNet50 classifier saved to: {resnet_path}")
        
        # Save UNet separately
        unet_model = self.model_builder.build_unet_segmentation()
        unet_model.set_weights(self.model.get_layer('unet_segmentation').get_weights())
        unet_path = self.model_dir / f'unet_segmentation_{timestamp}.keras'
        unet_model.save(unet_path)
        print(f"💾 UNet segmentation saved to: {unet_path}")
        
        # Save model info
        model_info = {
            'timestamp': timestamp,
            'epochs': epochs,
            'batch_size': batch_size,
            'input_shape': [224, 224, 3],
            'architecture': 'UNet + ResNet50',
            'final_model': str(final_model_path),
            'resnet_model': str(resnet_path),
            'unet_model': str(unet_path),
            'training_samples': int(len(self.train_X)),
            'validation_samples': int(len(self.val_X)),
            'test_samples': int(len(self.test_X)),
            'class_weights': {str(k): float(v) for k, v in class_weights.items()}
        }
        
        with open(self.model_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        return self.history
    
    def evaluate(self):
        """Evaluate the trained model on test set"""
        print("\n" + "="*70)
        print("📊 EVALUATING MODEL ON TEST SET")
        print("="*70 + "\n")
        
        # Evaluate on test set
        test_loss, test_acc = self.model.evaluate(
            self.test_X, self.test_y,
            batch_size=32,
            verbose=1
        )
        
        # Get predictions
        print("\n🔮 Generating predictions...")
        y_pred_probs = self.model.predict(self.test_X, batch_size=32, verbose=1)
        y_pred = (y_pred_probs[:, 1] > 0.5).astype(int)  # Get melanoma probability
        
        # Classification report
        print("\n� Detailed Classification Report:")
        print("-" * 70)
        target_names = ['Benign', 'Melanoma']
        report_dict = classification_report(self.test_y, y_pred, target_names=target_names, output_dict=True)
        print(classification_report(self.test_y, y_pred, target_names=target_names))
        
        # Confusion matrix
        print("\n🔢 Confusion Matrix:")
        print("-" * 70)
        cm = confusion_matrix(self.test_y, y_pred)
        print(f"                 Predicted")
        print(f"               Benign  Melanoma")
        print(f"Actual Benign    {cm[0][0]:5d}   {cm[0][1]:5d}")
        print(f"     Melanoma    {cm[1][0]:5d}   {cm[1][1]:5d}")
        print("-" * 70)
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = sensitivity  # Same as sensitivity
        
        # Calculate metrics
        print("\n📈 Test Set Results:")
        print("-" * 70)
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f} ({100*test_acc:.2f}%)")
        print(f"  Precision (Melanoma): {precision:.4f}")
        print(f"  Recall (Melanoma): {recall:.4f}")
        print(f"  Sensitivity (True Positive Rate): {sensitivity:.4f}")
        print(f"  Specificity (True Negative Rate): {specificity:.4f}")
        print(f"  False Positive Rate: {fp/(fp+tn) if (fp+tn) > 0 else 0:.4f}")
        print(f"  False Negative Rate: {fn/(fn+tp) if (fn+tp) > 0 else 0:.4f}")
        print("-" * 70)
        
        # Save test results
        test_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'confusion_matrix': cm.tolist(),
            'classification_report': report_dict
        }
        
        with open(self.model_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print("\n💾 Test results saved to: models/test_results.json")
        
        return test_results
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("⚠️  No training history available. Train the model first.")
            return
        
        print("\n📊 Generating training history plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Melanoma Detection Model Training History', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.model_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"💾 Training plots saved to: {plot_path}")
        
        plt.close()


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("🔬 MELANOMA DETECTION SYSTEM - MODEL TRAINING")
    print("="*70)
    print("Architecture: UNet → ResNet50 Pipeline")
    print("Dataset: HAM10000 (Binary Classification: Melanoma vs Benign)")
    print("="*70 + "\n")
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_dir='data/ham10000/augmented',  # Using augmented dataset (21,630 images)
        model_dir='models'
    )
    
    # Load data
    trainer.load_data()
    
    # Train model (10 epochs, batch 64, no runtime aug for faster training)
    trainer.train(epochs=10, batch_size=64)
    
    # Evaluate model
    trainer.evaluate()    # Plot training history
    trainer.plot_training_history()
    
    print("\n" + "="*70)
    print("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("1. Check models/ directory for trained model files")
    print("2. Review training_history.png for training curves")
    print("3. Check test_results.json for detailed metrics")
    print("4. Update app.py to use the trained model")
    print("5. Test the complete pipeline with frontend")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
