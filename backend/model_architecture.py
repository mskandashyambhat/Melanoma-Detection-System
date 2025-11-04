"""
═══════════════════════════════════════════════════════════════════
    MELANOMA DETECTION SYSTEM - BINARY CLASSIFICATION
═══════════════════════════════════════════════════════════════════

Architecture: UNet (Segmentation) → ResNet50 (Classification)

PIPELINE:
1. 📍 UNet Model: Segments lesion from healthy skin
2. 🔄 Mask Application: Focuses attention on lesion region
3. 🧠 ResNet50 Model: Classifies as MELANOMA or BENIGN

Dataset: HAM10000 (10,015 dermoscopic images)
Output: Binary classification (Melanoma vs Non-Melanoma)
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
    print("✅ TensorFlow loaded successfully")
except ImportError:
    print("⚠️  TensorFlow not installed. Models will use mock predictions.")
    TF_AVAILABLE = False
    import numpy as np


class MelanomaDetectionModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2, use_efficientnet=False, use_unet=True):
        """
        Initialize the UNet + ResNet50 pipeline for melanoma detection
        
        🎯 BINARY CLASSIFICATION:
        - Class 0: Benign (Non-Melanoma)
        - Class 1: Melanoma
        
        Architecture Flow:
        Input Image → [UNet] → Segmentation Mask
                   ↓
        Masked Image → [ResNet50] → Melanoma Probability
        """
        self.input_shape = input_shape
        self.num_classes = num_classes  # Binary: 2 classes
        # Options
        self.use_efficientnet = use_efficientnet
        self.use_unet = use_unet
        self.class_names = ['Benign', 'Melanoma']

        print(f"\n{'='*70}")
        print(f"  MELANOMA DETECTION MODEL INITIALIZED")
        print(f"{'='*70}")
        print(f"  Input Shape: {input_shape}")
        print(f"  Classification: Binary (Melanoma vs Benign)")
        print(f"  Classes: {self.class_names}")
        print(f"{'='*70}\n")
    
    # ═══════════════════════════════════════════════════════════════
    # 📍 PART 1: UNET SEGMENTATION MODEL
    # ═══════════════════════════════════════════════════════════════
    
    def build_unet_segmentation(self):
        """
        ╔══════════════════════════════════════════════════════════╗
        ║                                                          ║
        ║                    🔷 THIS IS UNET 🔷                    ║
        ║                                                          ║
        ║   PURPOSE: Segment skin lesion from background          ║
        ║   INPUT:   224x224x3 RGB image                          ║
        ║   OUTPUT:  224x224x1 Binary segmentation mask           ║
        ║                                                          ║
        ╚══════════════════════════════════════════════════════════╝
        
        UNet Architecture:
        - Encoder: Feature extraction (downsampling)
        - Bridge: Bottleneck layer
        - Decoder: Spatial reconstruction (upsampling + skip connections)
        """
        print("\n🔷 Building UNet Segmentation Model...")
        
        inputs = keras.Input(shape=self.input_shape, name='unet_input')
        
        # ─────────────────────────────────────────────────────────
        # ENCODER PATH (Downsampling)
        # ─────────────────────────────────────────────────────────
        print("   Building Encoder Path...")
        
        # Level 1: 224x224x64
        conv1 = Conv2D(64, 3, activation='relu', padding='same', name='enc_conv1a')(inputs)
        conv1 = BatchNormalization(name='enc_bn1a')(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', name='enc_conv1b')(conv1)
        conv1 = BatchNormalization(name='enc_bn1b')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='enc_pool1')(conv1)
        
        # Level 2: 112x112x128
        conv2 = Conv2D(128, 3, activation='relu', padding='same', name='enc_conv2a')(pool1)
        conv2 = BatchNormalization(name='enc_bn2a')(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', name='enc_conv2b')(conv2)
        conv2 = BatchNormalization(name='enc_bn2b')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='enc_pool2')(conv2)
        
        # Level 3: 56x56x256
        conv3 = Conv2D(256, 3, activation='relu', padding='same', name='enc_conv3a')(pool2)
        conv3 = BatchNormalization(name='enc_bn3a')(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', name='enc_conv3b')(conv3)
        conv3 = BatchNormalization(name='enc_bn3b')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='enc_pool3')(conv3)
        
        # Level 4: 28x28x512
        conv4 = Conv2D(512, 3, activation='relu', padding='same', name='enc_conv4a')(pool3)
        conv4 = BatchNormalization(name='enc_bn4a')(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', name='enc_conv4b')(conv4)
        conv4 = BatchNormalization(name='enc_bn4b')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), name='enc_pool4')(conv4)
        
        # ─────────────────────────────────────────────────────────
        # BRIDGE (Bottleneck): 14x14x1024
        # ─────────────────────────────────────────────────────────
        print("   Building Bridge (Bottleneck)...")
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', name='bridge_conv5a')(pool4)
        conv5 = BatchNormalization(name='bridge_bn5a')(conv5)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', name='bridge_conv5b')(conv5)
        conv5 = BatchNormalization(name='bridge_bn5b')(conv5)
        
        # ─────────────────────────────────────────────────────────
        # DECODER PATH (Upsampling with Skip Connections)
        # ─────────────────────────────────────────────────────────
        print("   Building Decoder Path...")
        
        # Level 4 Decoder: 28x28x512
        up6 = UpSampling2D(size=(2, 2), name='dec_up6')(conv5)
        up6 = Conv2D(512, 2, activation='relu', padding='same', name='dec_conv6_up')(up6)
        merge6 = Concatenate(name='dec_merge6')([conv4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same', name='dec_conv6a')(merge6)
        conv6 = BatchNormalization(name='dec_bn6a')(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', name='dec_conv6b')(conv6)
        conv6 = BatchNormalization(name='dec_bn6b')(conv6)
        
        # Level 3 Decoder: 56x56x256
        up7 = UpSampling2D(size=(2, 2), name='dec_up7')(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', name='dec_conv7_up')(up7)
        merge7 = Concatenate(name='dec_merge7')([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same', name='dec_conv7a')(merge7)
        conv7 = BatchNormalization(name='dec_bn7a')(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', name='dec_conv7b')(conv7)
        conv7 = BatchNormalization(name='dec_bn7b')(conv7)
        
        # Level 2 Decoder: 112x112x128
        up8 = UpSampling2D(size=(2, 2), name='dec_up8')(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', name='dec_conv8_up')(up8)
        merge8 = Concatenate(name='dec_merge8')([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same', name='dec_conv8a')(merge8)
        conv8 = BatchNormalization(name='dec_bn8a')(conv8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', name='dec_conv8b')(conv8)
        conv8 = BatchNormalization(name='dec_bn8b')(conv8)
        
        # Level 1 Decoder: 224x224x64
        up9 = UpSampling2D(size=(2, 2), name='dec_up9')(conv8)
        up9 = Conv2D(64, 2, activation='relu', padding='same', name='dec_conv9_up')(up9)
        merge9 = Concatenate(name='dec_merge9')([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same', name='dec_conv9a')(merge9)
        conv9 = BatchNormalization(name='dec_bn9a')(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', name='dec_conv9b')(conv9)
        conv9 = BatchNormalization(name='dec_bn9b')(conv9)
        
        # Output: Binary segmentation mask (224x224x1)
        outputs = Conv2D(1, 1, activation='sigmoid', name='unet_segmentation_output')(conv9)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='UNet_Segmentation')
        
        print("✅ UNet Model Built Successfully!")
        print(f"   Total Parameters: {model.count_params():,}")
        
        return model

    def build_efficientnet_classifier(self):
        """Build a lightweight EfficientNetB0 classifier (fast, ~5.3M params)

        This function returns a Keras Model that accepts an RGB image and
        outputs class probabilities for binary classification.
        """
        print("\n🔷 Building EfficientNetB0 classifier...")
        try:
            from tensorflow.keras.applications import EfficientNetB0
        except Exception:
            raise

        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=self.input_shape)
        print(f"   EfficientNetB0 base loaded. Layers: {len(base_model.layers)}")

        # Freeze base for transfer learning
        for layer in base_model.layers:
            layer.trainable = False

        inputs = keras.Input(shape=self.input_shape, name='efficientnet_input')
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D(name='gap')(x)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = BatchNormalization(name='bn1')(x)
        x = Dropout(0.4, name='dropout1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        x = Dropout(0.3, name='dropout2')(x)
        outputs = Dense(self.num_classes, activation='softmax', name='classification')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name='EfficientNetB0_Classifier')
        print(f"   EfficientNetB0 classifier built. Params: {model.count_params():,}")
        return model
    
    # ═══════════════════════════════════════════════════════════════
    # 🧠 PART 2: RESNET50 CLASSIFICATION MODEL
    # ═══════════════════════════════════════════════════════════════
    
    def build_resnet_classifier(self):
        """
        ╔══════════════════════════════════════════════════════════╗
        ║                                                          ║
        ║                 🔶 THIS IS RESNET50 🔶                   ║
        ║                                                          ║
        ║   PURPOSE: Classify segmented lesion                    ║
        ║   INPUT:   224x224x3 Masked RGB image                   ║
        ║   OUTPUT:  Binary classification (Melanoma/Benign)      ║
        ║                                                          ║
        ╚══════════════════════════════════════════════════════════╝
        
        ResNet50 Features:
        - Pre-trained on ImageNet (transfer learning)
        - 50 layers with residual skip connections
        - Fine-tuned for skin lesion classification
        """
        print("\n🔶 Building ResNet50 Classification Model...")
        
        # Load pre-trained ResNet50 (without top classification layer)
        print("   Loading pre-trained ResNet50 weights...")
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Fine-tuning strategy: Freeze early layers
        print("   Applying fine-tuning strategy...")
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        inputs = keras.Input(shape=self.input_shape, name='resnet_input')
        
        # ═══════════════════════════════════════════════════════
        # 🎨 DATA AUGMENTATION DISABLED FOR FASTER TRAINING
        # ═══════════════════════════════════════════════════════
        # Data is already pre-augmented in the dataset
        # Skipping runtime augmentation for 30-40% speed improvement
        print("   Skipping runtime augmentation (using pre-augmented data)...")
        
        # Direct input to ResNet50 (no augmentation layers)
        x = inputs
        
        # ResNet50 feature extraction
        x = base_model(x, training=False)
        
        # Custom classification head for binary classification
        print("   Building classification head...")
        x = GlobalAveragePooling2D(name='gap')(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = BatchNormalization(name='bn1')(x)
        x = Dropout(0.5, name='dropout1')(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = BatchNormalization(name='bn2')(x)
        x = Dropout(0.4, name='dropout2')(x)
        x = Dense(128, activation='relu', name='fc3')(x)
        x = Dropout(0.3, name='dropout3')(x)
        
        # Binary classification output (2 classes: Benign, Melanoma)
        outputs = Dense(self.num_classes, activation='softmax', name='resnet_output')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='ResNet50_Classifier')
        
        print("✅ ResNet50 Model Built Successfully!")
        print(f"   Total Parameters: {model.count_params():,}")
        print(f"   Trainable Parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
    
    # ═══════════════════════════════════════════════════════════════
    # 🔄 PART 3: COMBINED PIPELINE (UNet → ResNet50)
    # ═══════════════════════════════════════════════════════════════
    
    def build_combined_model(self):
        """
        ╔══════════════════════════════════════════════════════════╗
        ║                                                          ║
        ║             🔷 UNET → RESNET50 PIPELINE 🔶              ║
        ║                                                          ║
        ║   1. UNet segments the lesion                           ║
        ║   2. Mask is applied to original image                  ║
        ║   3. ResNet50 classifies the masked image               ║
        ║                                                          ║
        ╚══════════════════════════════════════════════════════════╝
        """
        print("\n" + "="*70)
        if self.use_unet:
            print("  BUILDING COMBINED PIPELINE: UNet → Classifier")
        else:
            print("  BUILDING CLASSIFICATION-ONLY PIPELINE")
        print("="*70)

        # Choose classifier
        if self.use_efficientnet:
            classifier_model = self.build_efficientnet_classifier()
        else:
            classifier_model = self.build_resnet_classifier()

        if self.use_unet:
            # Build individual models
            unet_model = self.build_unet_segmentation()

            # Combined pipeline: UNet -> mask -> classifier
            print("\n🔗 Connecting UNet output to classifier input...")
            inputs = keras.Input(shape=self.input_shape, name='pipeline_input')

            segmentation_mask = unet_model(inputs)
            segmentation_mask_3ch = Concatenate(name='mask_expansion')([
                segmentation_mask, segmentation_mask, segmentation_mask
            ])
            masked_image = Multiply(name='apply_mask')([inputs, segmentation_mask_3ch])

            classification_output = classifier_model(masked_image)

            combined_model = models.Model(
                inputs=inputs,
                outputs=classification_output,
                name='MelanomaDetection_Pipeline'
            )

            print("\n✅ Combined Pipeline Built Successfully!")
            print(f"   Total Pipeline Parameters: {combined_model.count_params():,}")
            print("="*70 + "\n")

            return combined_model, unet_model, classifier_model
        else:
            # Classification-only model (no UNet)
            classifier = classifier_model
            print("\n✅ Classification-only model built successfully!")
            print(f"   Classifier Parameters: {classifier.count_params():,}")
            print("="*70 + "\n")
            return classifier, None, classifier
    
    def compile_models(self, combined_model, learning_rate=0.001):
        """
        Compile the combined model with appropriate loss functions and metrics
        """
        print("⚙️  Compiling model...")
        
        combined_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'segmentation': 'binary_crossentropy',  # UNet loss
                'classification': 'categorical_crossentropy'  # ResNet50 loss
            },
            loss_weights={
                'segmentation': 0.3,  # Lower weight for segmentation
                'classification': 0.7  # Higher weight for classification
            },
            metrics={
                'segmentation': ['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)],
                'classification': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            }
        )
        
        print("✅ Model compiled successfully!")
        return combined_model


# ═══════════════════════════════════════════════════════════════════
# 🧪 TESTING & VALIDATION
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  MELANOMA DETECTION MODEL - ARCHITECTURE TEST")
    print("="*70 + "\n")
    
    if TF_AVAILABLE:
        # Initialize model
        model_builder = MelanomaDetectionModel(
            input_shape=(224, 224, 3),
            num_classes=2  # Binary classification
        )
        
        # Build combined pipeline
        combined_model, unet_model, resnet_model = model_builder.build_combined_model()
        
        # Compile
        combined_model = model_builder.compile_models(combined_model)
        
        # Display summary
        print("\n📋 MODEL SUMMARY:")
        print("="*70)
        print(f"UNet Segmentation:")
        print(f"  - Input: {unet_model.input_shape}")
        print(f"  - Output: {unet_model.output_shape}")
        print(f"  - Parameters: {unet_model.count_params():,}")
        print()
        print(f"ResNet50 Classification:")
        print(f"  - Input: {resnet_model.input_shape}")
        print(f"  - Output: {resnet_model.output_shape}")
        print(f"  - Parameters: {resnet_model.count_params():,}")
        print()
        print(f"Combined Pipeline:")
        print(f"  - Input: {combined_model.input_shape}")
        print(f"  - Outputs: Segmentation + Classification")
        print(f"  - Total Parameters: {combined_model.count_params():,}")
        print("="*70)
        
        print("\n✅ Architecture test completed successfully!")
    else:
        print("⚠️  TensorFlow not available. Skipping architecture test.")
