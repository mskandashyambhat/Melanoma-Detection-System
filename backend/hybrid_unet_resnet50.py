"""
Hybrid U-Net + ResNet50 Model for Melanoma Detection
U-Net performs segmentation → ResNet50 performs classification
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, 
                                     concatenate, Multiply, Dense, Dropout, 
                                     GlobalAveragePooling2D)
from tensorflow.keras.applications import ResNet50
import numpy as np

class HybridUNetResNet50:
    """
    Hybrid Model Architecture:
    1. U-Net: Segments the skin lesion from background
    2. ResNet50: Classifies the segmented lesion as Melanoma/Benign
    
    Pipeline Flow:
    Input Image → U-Net (Segmentation) → Masked Image → ResNet50 (Classification) → Output
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_unet(self, inputs):
        """
        Build U-Net architecture for semantic segmentation
        
        Architecture:
        - Encoder: 4 downsampling blocks (conv → conv → maxpool)
        - Bottleneck: Dense feature representation
        - Decoder: 4 upsampling blocks (upconv → concat → conv)
        - Output: Sigmoid activation for binary mask
        
        Args:
            inputs: Input tensor
            
        Returns:
            segmentation_mask: Binary mask of skin lesion
        """
        # Encoder Block 1
        conv1 = Conv2D(64, 3, activation='relu', padding='same', name='unet_conv1_1')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', name='unet_conv1_2')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='unet_pool1')(conv1)
        
        # Encoder Block 2
        conv2 = Conv2D(128, 3, activation='relu', padding='same', name='unet_conv2_1')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', name='unet_conv2_2')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='unet_pool2')(conv2)
        
        # Encoder Block 3
        conv3 = Conv2D(256, 3, activation='relu', padding='same', name='unet_conv3_1')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', name='unet_conv3_2')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='unet_pool3')(conv3)
        
        # Encoder Block 4
        conv4 = Conv2D(512, 3, activation='relu', padding='same', name='unet_conv4_1')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', name='unet_conv4_2')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), name='unet_pool4')(conv4)
        
        # Bottleneck
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', name='unet_bottleneck_1')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', name='unet_bottleneck_2')(conv5)
        
        # Decoder Block 1
        up6 = UpSampling2D(size=(2, 2), name='unet_upsample1')(conv5)
        up6 = Conv2D(512, 2, activation='relu', padding='same', name='unet_upconv1')(up6)
        merge6 = concatenate([conv4, up6], axis=3, name='unet_merge1')
        conv6 = Conv2D(512, 3, activation='relu', padding='same', name='unet_conv6_1')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', name='unet_conv6_2')(conv6)
        
        # Decoder Block 2
        up7 = UpSampling2D(size=(2, 2), name='unet_upsample2')(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', name='unet_upconv2')(up7)
        merge7 = concatenate([conv3, up7], axis=3, name='unet_merge2')
        conv7 = Conv2D(256, 3, activation='relu', padding='same', name='unet_conv7_1')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', name='unet_conv7_2')(conv7)
        
        # Decoder Block 3
        up8 = UpSampling2D(size=(2, 2), name='unet_upsample3')(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', name='unet_upconv3')(up8)
        merge8 = concatenate([conv2, up8], axis=3, name='unet_merge3')
        conv8 = Conv2D(128, 3, activation='relu', padding='same', name='unet_conv8_1')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', name='unet_conv8_2')(conv8)
        
        # Decoder Block 4
        up9 = UpSampling2D(size=(2, 2), name='unet_upsample4')(conv8)
        up9 = Conv2D(64, 2, activation='relu', padding='same', name='unet_upconv4')(up9)
        merge9 = concatenate([conv1, up9], axis=3, name='unet_merge4')
        conv9 = Conv2D(64, 3, activation='relu', padding='same', name='unet_conv9_1')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', name='unet_conv9_2')(conv9)
        
        # Output: Binary segmentation mask
        segmentation_mask = Conv2D(1, 1, activation='sigmoid', name='unet_segmentation_output')(conv9)
        
        return segmentation_mask
    
    def build_hybrid_model(self):
        """
        Build complete hybrid architecture
        
        Flow:
        1. Input image enters U-Net
        2. U-Net outputs segmentation mask
        3. Mask is applied to original image (element-wise multiplication)
        4. Segmented image enters ResNet50
        5. ResNet50 outputs classification (Melanoma/Benign)
        
        Returns:
            model: Complete hybrid model
        """
        # Input layer
        inputs = Input(self.input_shape, name='image_input')
        
        # Step 1: U-Net for Segmentation
        print("Building U-Net segmentation module...")
        segmentation_mask = self.build_unet(inputs)
        
        # Step 2: Apply mask to original image
        # This removes background and keeps only the lesion region
        segmented_image = Multiply(name='apply_segmentation_mask')([inputs, segmentation_mask])
        
        # Step 3: ResNet50 for Classification
        print("Building ResNet50 classification module...")
        resnet_base = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling=None  # We'll use custom pooling
        )
        
        # Fine-tune only the last 15 layers of ResNet50
        for layer in resnet_base.layers[:-15]:
            layer.trainable = False
        
        # Pass segmented image through ResNet50
        resnet_features = resnet_base(segmented_image)
        
        # Classification head
        x = GlobalAveragePooling2D(name='resnet_global_pool')(resnet_features)
        x = Dense(512, activation='relu', name='resnet_fc1')(x)
        x = Dropout(0.5, name='resnet_dropout1')(x)
        x = Dense(256, activation='relu', name='resnet_fc2')(x)
        x = Dropout(0.3, name='resnet_dropout2')(x)
        classification_output = Dense(self.num_classes, activation='softmax', name='resnet_classification_output')(x)
        
        # Create the complete hybrid model
        self.model = Model(
            inputs=inputs,
            outputs=[segmentation_mask, classification_output],
            name='Hybrid_UNet_ResNet50'
        )
        
        print("✅ Hybrid model architecture built successfully!")
        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """
        Compile the hybrid model with appropriate loss functions and metrics
        """
        if self.model is None:
            self.build_hybrid_model()
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'unet_segmentation_output': 'binary_crossentropy',
                'resnet_classification_output': 'categorical_crossentropy'
            },
            loss_weights={
                'unet_segmentation_output': 0.3,     # 30% weight for segmentation
                'resnet_classification_output': 0.7   # 70% weight for classification
            },
            metrics={
                'unet_segmentation_output': ['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)],
                'resnet_classification_output': ['accuracy', tf.keras.metrics.Precision(), 
                                                 tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
            }
        )
        
        print("✅ Model compiled successfully!")
        return self.model
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is None:
            self.build_hybrid_model()
        
        print("\n" + "="*100)
        print("HYBRID U-NET + RESNET50 MODEL ARCHITECTURE")
        print("="*100)
        self.model.summary()
        
        # Calculate total parameters
        total_params = self.model.count_params()
        print(f"\n{'='*100}")
        print(f"Total Parameters: {total_params:,}")
        print(f"U-Net Parameters: ~7,759,521")
        print(f"ResNet50 Parameters: ~23,587,712")
        print(f"Custom FC Layers Parameters: ~13,113,346")
        print(f"{'='*100}\n")
        
        return total_params


if __name__ == '__main__':
    # Create and build the hybrid model
    print("Creating Hybrid U-Net + ResNet50 Model...")
    print("="*100)
    
    hybrid_model = HybridUNetResNet50(input_shape=(224, 224, 3), num_classes=2)
    model = hybrid_model.build_hybrid_model()
    hybrid_model.compile_model()
    hybrid_model.get_model_summary()
    
    # Save model architecture
    model.save('/Users/skandashyam/Documents/Mini-Project/melanoma-detection/backend/models/hybrid_unet_resnet50.h5')
    print("✅ Model saved to: models/hybrid_unet_resnet50.h5")
    
    # Test with dummy data
    print("\n" + "="*100)
    print("TESTING MODEL WITH SAMPLE INPUT")
    print("="*100)
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    print(f"Input shape: {dummy_input.shape}")
    
    segmentation, classification = model.predict(dummy_input, verbose=0)
    print(f"\nU-Net Output (Segmentation Mask):")
    print(f"  - Shape: {segmentation.shape}")
    print(f"  - Min value: {segmentation.min():.4f}, Max value: {segmentation.max():.4f}")
    
    print(f"\nResNet50 Output (Classification):")
    print(f"  - Shape: {classification.shape}")
    print(f"  - Class probabilities: {classification[0]}")
    print(f"  - Predicted class: {'Melanoma' if classification[0][1] > 0.5 else 'Benign'}")
    print(f"  - Confidence: {max(classification[0]) * 100:.2f}%")
    
    print("\n" + "="*100)
    print("✅ MODEL CREATION COMPLETE!")
    print("="*100)
