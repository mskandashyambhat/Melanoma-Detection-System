"""
Advanced Melanoma Detection System using UNet + ResNet50 Pipeline
================================================
Architecture: UNet (Segmentation) → ResNet50 (Classification)

This system uses a two-stage approach:
1. UNet Model: Performs lesion segmentation to isolate the region of interest
2. ResNet50 Model: Classifies the segmented lesion into specific skin conditions

Dataset Requirements:
- HAM10000 (Human Against Machine with 10,000 training images)
- ISIC Archive (International Skin Imaging Collaboration)
- Both datasets provide diverse skin lesion images with expert annotations
"""

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, UpSampling2D, Concatenate,
        BatchNormalization, Activation, Dropout, Dense, GlobalAveragePooling2D,
        Input, Multiply
    )
    import numpy as np
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not installed. Models will use mock predictions.")
    TF_AVAILABLE = False
    import numpy as np


class MelanomaDetectionModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=8):
        """
        Initialize the UNet + ResNet50 pipeline for melanoma detection
        
        Architecture Overview:
        1. Input Image (224x224x3) → UNet Segmentation → Segmentation Mask
        2. Input Image × Segmentation Mask → Focused Region → ResNet50 → Classification
        
        Classes:
        0: Melanoma
        1: Basal Cell Carcinoma
        2: Acne
        3: Ringworm
        4: Burns
        5: Eczema
        6: Psoriasis
        7: Normal/Healthy Skin
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = [
            'Melanoma', 'Basal Cell Carcinoma', 'Acne', 'Ringworm',
            'Burns', 'Eczema', 'Psoriasis', 'Normal Skin'
        ]
        
    def build_resnet_classifier(self):
        """
        Build ResNet50-based classifier for skin condition detection
        
        ResNet50 Architecture:
        - Pre-trained on ImageNet for transfer learning
        - Deep residual learning framework with skip connections
        - Prevents vanishing gradient problem in deep networks
        
        Input: Segmented/masked image (224x224x3)
        Output: Probability distribution over 8 skin condition classes
        
        Transfer Learning Strategy:
        - Freeze early layers (pre-trained feature extractors)
        - Fine-tune last 20 layers for skin-specific features
        - Custom classification head for multi-class prediction
        """
        # Load ResNet50 pre-trained on ImageNet
        # include_top=False removes the original classification head
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Fine-tuning strategy: Freeze early layers, train later layers
        # Early layers learn generic features (edges, textures)
        # Later layers learn domain-specific features (melanoma patterns)
        for layer in base_model.layers[:-20]:
            layer.trainable = False
            
        inputs = keras.Input(shape=self.input_shape)
        
        # Data augmentation for robustness and preventing overfitting
        # Applied during training only
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.2)(x)  # ±20% rotation
        x = layers.RandomZoom(0.2)(x)      # ±20% zoom
        x = layers.RandomContrast(0.2)(x)  # Contrast variation
        
        # ResNet50 feature extraction backbone
        x = base_model(x, training=False)
        
        # Custom classification head for 8-class skin condition prediction
        x = GlobalAveragePooling2D()(x)  # Spatial pooling
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)  # Prevent overfitting
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        # Softmax for multi-class probability distribution
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_unet_segmentation(self):
        """
        Build UNet model for lesion segmentation
        
        UNet Architecture:
        - Encoder: Extracts features at multiple scales via convolution + pooling
        - Bridge: Bottleneck layer with maximum feature compression
        - Decoder: Reconstructs spatial information via upsampling + skip connections
        - Output: Binary mask highlighting the lesion area
        
        Purpose: Isolate the skin lesion from surrounding healthy tissue
        Output Shape: (224, 224, 1) - Binary segmentation mask
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # ENCODER PATH - Downsampling to capture context
        # Level 1: 224x224x64
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 112x112
        
        # Level 2: 112x112x128
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 56x56
        
        # Level 3: 56x56x256
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 28x28
        
        # Level 4: 28x28x512
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 14x14
        
        # BRIDGE - Bottleneck layer: 14x14x1024
        # Maximum feature compression at lowest spatial resolution
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        
        # DECODER PATH - Upsampling to reconstruct spatial information
        # Skip connections from encoder preserve fine-grained details
        
        # Level 4 Decoder: 28x28x512
        up6 = UpSampling2D(size=(2, 2))(conv5)  # 28x28
        up6 = Conv2D(512, 2, activation='relu', padding='same')(up6)
        merge6 = Concatenate()([conv4, up6])  # Skip connection from encoder
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        
        # Level 3 Decoder: 56x56x256
        up7 = UpSampling2D(size=(2, 2))(conv6)  # 56x56
        up7 = Conv2D(256, 2, activation='relu', padding='same')(up7)
        merge7 = Concatenate()([conv3, up7])  # Skip connection
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        
        # Level 2 Decoder: 112x112x128
        up8 = UpSampling2D(size=(2, 2))(conv7)  # 112x112
        up8 = Conv2D(128, 2, activation='relu', padding='same')(up8)
        merge8 = Concatenate()([conv2, up8])  # Skip connection
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)
        
        # Level 1 Decoder: 224x224x64
        up9 = UpSampling2D(size=(2, 2))(conv8)  # 224x224
        up9 = Conv2D(64, 2, activation='relu', padding='same')(up9)
        merge9 = Concatenate()([conv1, up9])  # Skip connection
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)
        
        # Output layer: Binary segmentation mask (224x224x1)
        # Sigmoid activation outputs probability for each pixel being part of lesion
        outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_combined_model(self):
        """
        Build combined UNet → ResNet50 pipeline
        
        Two-Stage Architecture:
        ========================
        Stage 1 (UNet): Lesion Segmentation
        - Input: Original image (224x224x3)
        - Process: Identifies and segments the lesion region
        - Output: Binary mask (224x224x1) highlighting lesion area
        
        Stage 2 (ResNet50): Classification
        - Input: Original image × Segmentation mask (focused on lesion)
        - Process: Classifies the segmented lesion
        - Output: Disease probability distribution (8 classes)
        
        Pipeline Flow:
        1. Original Image → UNet → Segmentation Mask
        2. Original Image × Mask → Masked Image (lesion only)
        3. Masked Image → ResNet50 → Disease Classification
        
        Benefits:
        - UNet isolates lesion from background noise
        - ResNet50 focuses only on relevant tissue
        - Improved accuracy by reducing false features
        """
        # Build both models independently
        classifier = self.build_resnet_classifier()
        segmentation = self.build_unet_segmentation()
        
        return classifier, segmentation
    
    def build_integrated_pipeline(self):
        """
        Build fully integrated UNet → ResNet50 pipeline
        
        This creates a single end-to-end model where:
        - UNet segments the lesion
        - The segmentation mask is applied to the input
        - ResNet50 classifies the masked region
        
        Note: For training, it's often better to train separately then combine.
        This integrated approach is useful for inference/deployment.
        """
        # Input image
        inputs = Input(shape=self.input_shape, name='input_image')
        
        # Stage 1: UNet Segmentation
        segmentation_model = self.build_unet_segmentation()
        segmentation_mask = segmentation_model(inputs)  # Output: (224, 224, 1)
        
        # Apply mask to input image (element-wise multiplication)
        # This isolates the lesion region for classification
        masked_image = Multiply()([inputs, segmentation_mask])
        
        # Stage 2: ResNet50 Classification on masked image
        classifier_model = self.build_resnet_classifier()
        classification_output = classifier_model(masked_image)
        
        # Create end-to-end model
        integrated_model = models.Model(
            inputs=inputs,
            outputs=[segmentation_mask, classification_output],
            name='unet_resnet50_pipeline'
        )
        
        return integrated_model
    
    def compile_models(self, classifier, segmentation):
        """Compile both models with optimized parameters for high accuracy"""
        # Classifier compilation
        classifier.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(),
                    keras.metrics.Recall(),
                    keras.metrics.AUC()]
        )
        
        # Segmentation model compilation
        segmentation.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.MeanIoU(num_classes=2)]
        )
        
        return classifier, segmentation
    
    def get_callbacks(self):
        """Get training callbacks for optimal performance"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_classifier_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        return callbacks


def create_model():
    """Factory function to create and return the complete model"""
    model_builder = MelanomaDetectionModel()
    classifier, segmentation = model_builder.build_combined_model()
    classifier, segmentation = model_builder.compile_models(classifier, segmentation)
    
    return classifier, segmentation, model_builder.class_names


if __name__ == "__main__":
    # Test model creation
    print("Building Melanoma Detection Models...")
    classifier, segmentation, class_names = create_model()
    
    print("\n=== Classifier Model Summary ===")
    classifier.summary()
    
    print("\n=== Segmentation Model Summary ===")
    segmentation.summary()
    
    print("\n=== Classes ===")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")
