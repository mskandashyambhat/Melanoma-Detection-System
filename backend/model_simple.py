"""
═══════════════════════════════════════════════════════════════════
    MELANOMA DETECTION - SIMPLE EFFICIENTNET CLASSIFIER
═══════════════════════════════════════════════════════════════════

Simple, fast architecture using only MobileNetV2 (~3.5M parameters)

Why this is better for training on Mac:
- 15x fewer parameters than UNet+ResNet50 (3.5M vs 56M)
- 5-10x faster training
- Still excellent accuracy for binary classification
- No segmentation needed - direct image classification

Dataset: HAM10000
Output: Binary classification (Melanoma vs Benign)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


class SimpleMelanomaModel:
    """Simple EfficientNet-based melanoma classifier"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = ['Benign', 'Melanoma']
        
        print(f"\n{'='*70}")
        print(f"  SIMPLE MELANOMA CLASSIFIER (MobileNetV2)")
        print(f"{'='*70}")
        print(f"  Input Shape: {input_shape}")
        print(f"  Classes: {num_classes} (Binary: Benign vs Melanoma)")
        print(f"  Base Model: MobileNetV2 (~3.5M parameters)")
        print(f"{'='*70}\n")
    
    def build_model(self):
        """Build simple MobileNetV2 classifier"""
        print("🏗️  Building MobileNetV2 model...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            alpha=1.0  # Width multiplier
        )
        
        # Fine-tuning: freeze early layers, train last 20
        print("   Applying fine-tuning strategy (freezing early layers)...")
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        
        # MobileNetV2 feature extraction
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        
        # Classification head
        x = layers.Dense(256, activation='relu', name='fc1')(x)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        
        x = layers.Dense(128, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SimpleMelanomaClassifier')
        
        # Print summary
        total_params = model.count_params()
        trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
        
        print(f"✅ Model built successfully!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Non-trainable parameters: {total_params - trainable_params:,}")
        
        return model
    
    def predict(self, images):
        """Make predictions on images"""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not built. Call build_model() first.")
        
        predictions = self.model.predict(images)
        return predictions


# For backward compatibility with existing code
class MelanomaDetectionModel(SimpleMelanomaModel):
    """Alias for SimpleMelanomaModel to maintain compatibility"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        super().__init__(input_shape, num_classes)
    
    def build_combined_model(self):
        """Build model (renamed for compatibility)"""
        model = self.build_model()
        # Return single model (no separate unet/resnet)
        return model, None, None


def build_simple_model(input_shape=(224, 224, 3), num_classes=2):
    """Helper function to build model"""
    builder = SimpleMelanomaModel(input_shape, num_classes)
    return builder.build_model()


if __name__ == '__main__':
    print("Testing SimpleMelanomaModel...")
    builder = SimpleMelanomaModel()
    model = builder.build_model()
    model.summary()
