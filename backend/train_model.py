"""
Model Training Script for Melanoma Detection System
====================================================

This script demonstrates how to train the UNet + ResNet50 models on medical imaging datasets.

RECOMMENDED DATASETS:
=====================

1. HAM10000 (Human Against Machine with 10,000 training images)
   - Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
   - Size: 10,015 dermatoscopic images
   - Classes: 7 skin conditions (Melanoma, Basal Cell Carcinoma, etc.)
   - Format: JPG images with CSV metadata
   - Citation: Tschandl et al., "The HAM10000 dataset" (2018)
   
2. ISIC Archive (International Skin Imaging Collaboration)
   - Source: https://www.isic-archive.com/
   - Size: 100,000+ dermoscopic images
   - Classes: Multiple skin lesion types with expert annotations
   - Format: JPG/PNG with JSON metadata
   - Benefits: Largest publicly available skin lesion dataset
   
3. PH² Dataset (Pedro Hispano Hospital)
   - Source: https://www.fc.up.pt/addi/ph2%20database.html
   - Size: 200 dermoscopic images
   - Classes: Common nevi, Atypical nevi, Melanoma
   - Benefits: Includes ground truth segmentation masks (useful for UNet training)

DATASET PREPARATION:
====================
1. Download one of the datasets above
2. Organize into the following structure:
   dataset/
   ├── train/
   │   ├── melanoma/
   │   ├── basal_cell_carcinoma/
   │   ├── acne/
   │   ├── ringworm/
   │   ├── burns/
   │   ├── eczema/
   │   ├── psoriasis/
   │   └── normal_skin/
   └── validation/
       ├── melanoma/
       ├── basal_cell_carcinoma/
       └── ... (same structure)

3. Split ratio: 80% training, 20% validation
4. Balance classes or use class weights for imbalanced data
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from model_architecture import MelanomaDetectionModel

# Configuration
IMG_SIZE = (224, 224)  # Input size for ResNet50
BATCH_SIZE = 32        # Adjust based on GPU memory
EPOCHS = 100           # Number of training epochs
TRAIN_DIR = 'dataset/train'       # Path to training data
VAL_DIR = 'dataset/validation'    # Path to validation data
MODEL_SAVE_PATH = 'models/'       # Where to save trained models

def create_data_generators():
    """
    Create data generators for training and validation
    
    Data Augmentation Strategy:
    ===========================
    Training data is augmented to:
    1. Increase dataset diversity
    2. Prevent overfitting
    3. Improve model generalization
    
    Augmentations applied:
    - Rotation: ±20° (lesions can appear at any angle)
    - Width/Height shifts: ±20% (lesions at different positions)
    - Shear: ±20% (perspective variations)
    - Zoom: ±20% (different camera distances)
    - Horizontal/Vertical flip: Mirror images (common in dermatoscopy)
    
    Note: For HAM10000 and ISIC datasets, ensure images are already centered
    and properly cropped around the lesion before training.
    """
    
    # Training data generator with extensive augmentation
    # These augmentations simulate real-world variations in dermoscopic imaging
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalize pixel values to [0, 1]
        rotation_range=20,           # Random rotation
        width_shift_range=0.2,       # Horizontal shift
        height_shift_range=0.2,      # Vertical shift
        shear_range=0.2,             # Shear transformation
        zoom_range=0.2,              # Random zoom
        horizontal_flip=True,        # Mirror horizontally
        vertical_flip=True,          # Mirror vertically
        fill_mode='nearest'          # Fill empty pixels after transformation
    )
    
    # Validation data generator - NO augmentation for consistent evaluation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator


def train_classifier():
    """
    Train the ResNet50 classifier
    
    Training Strategy:
    ==================
    1. Transfer Learning: Use pre-trained ResNet50 weights from ImageNet
    2. Fine-tuning: Unfreeze last 20 layers for domain adaptation
    3. Early Stopping: Prevent overfitting by monitoring validation accuracy
    4. Learning Rate Reduction: Adaptive learning rate for better convergence
    5. Model Checkpointing: Save best model based on validation accuracy
    
    Expected Performance (with proper dataset):
    ===========================================
    - HAM10000: 85-90% validation accuracy
    - ISIC: 80-88% validation accuracy
    - Training time: 2-4 hours on GPU (NVIDIA RTX 3090 or better)
    - Best results with balanced classes or class weights
    """
    print("=" * 80)
    print("Training Melanoma Classifier (ResNet50)")
    print("=" * 80)
    
    # Create data generators
    train_gen, val_gen = create_data_generators()
    
    # Get number of classes
    num_classes = train_gen.num_classes
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class indices: {train_gen.class_indices}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    # Build model
    model_builder = MelanomaDetectionModel(num_classes=num_classes)
    classifier, _ = model_builder.build_combined_model()
    classifier, _ = model_builder.compile_models(classifier, None)
    
    print("\nModel built successfully!")
    print(f"Total parameters: {classifier.count_params():,}")
    
    # Get callbacks
    callbacks = model_builder.get_callbacks()
    
    # Add TensorBoard callback
    callbacks.append(
        keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1,
            write_graph=True
        )
    )
    
    # Train model
    print("\nStarting training...")
    history = classifier.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, 'classifier_model.h5')
    classifier.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Print final metrics
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    # Find best epoch
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    return classifier, history


def evaluate_model(model, val_generator):
    """Evaluate model performance"""
    print("\n" + "=" * 80)
    print("Evaluating Model")
    print("=" * 80)
    
    # Evaluate
    results = model.evaluate(val_generator, verbose=1)
    
    print("\nEvaluation Results:")
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]:.4f}")
    if len(results) > 2:
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")
        print(f"AUC: {results[4]:.4f}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(val_generator, verbose=1)
    
    # Get true labels
    true_labels = val_generator.classes
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    class_names = list(val_generator.class_indices.keys())
    for i, class_name in enumerate(class_names):
        class_mask = true_labels == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predicted_labels[class_mask] == true_labels[class_mask])
            print(f"{class_name}: {class_acc:.4f} ({np.sum(class_mask)} samples)")


if __name__ == "__main__":
    # Check if dataset directories exist
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("⚠️  Dataset directories not found!")
        print("\n" + "=" * 80)
        print("DATASET SETUP REQUIRED")
        print("=" * 80)
        print("\nPlease download and prepare one of these datasets:")
        print("\n1. HAM10000 Dataset (Recommended for beginners)")
        print("   Download: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        print("   - 10,015 images")
        print("   - Well-balanced classes")
        print("   - Includes metadata CSV")
        print("\n2. ISIC Archive (Best for production)")
        print("   Download: https://www.isic-archive.com/")
        print("   - 100,000+ images")
        print("   - Most comprehensive dataset")
        print("   - Requires registration")
        print("\n3. PH² Dataset (Best for UNet segmentation)")
        print("   Download: https://www.fc.up.pt/addi/ph2%20database.html")
        print("   - 200 images with segmentation masks")
        print("   - Perfect for training UNet model")
        print("\nAfter downloading, organize as:")
        print(f"  - {TRAIN_DIR}/")
        print("    ├── melanoma/")
        print("    ├── basal_cell_carcinoma/")
        print("    ├── acne/")
        print("    ├── ringworm/")
        print("    ├── burns/")
        print("    ├── eczema/")
        print("    ├── psoriasis/")
        print("    └── normal_skin/")
        print(f"  - {VAL_DIR}/")
        print("    └── (same structure)")
        print("\nUse 80-20 train-validation split")
        print("=" * 80)
        exit(1)
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Train classifier
    try:
        classifier, history = train_classifier()
        
        # Evaluate model
        _, val_gen = create_data_generators()
        evaluate_model(classifier, val_gen)
        
        print("\n✓ Training completed successfully!")
        print(f"Model saved to: {MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
