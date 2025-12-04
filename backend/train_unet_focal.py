"""
TRAIN U-NET COMPONENT WITH FOCAL LOSS FOR IMPROVED RECALL
==========================================================
This script trains a U-Net adapted for binary classification with focal loss
to improve recall while maintaining accuracy, then uses it in ensemble.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, 
                                     concatenate, Dense, Dropout, GlobalAveragePooling2D,
                                     BatchNormalization, Activation)
from tensorflow.keras import regularizers

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for imbalanced data - same as ResNet50"""
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

class UNetClassifierTrainer:
    """U-Net adapted for classification with focal loss"""
    
    def __init__(self):
        script_dir = Path(__file__).parent
        self.data_dir = script_dir.parent / 'data' / 'ham10000_binary'
        self.model_dir = script_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)

        print("Loading data...")
        self.train_X = np.load(self.data_dir / 'train_X.npy')
        self.train_y = np.load(self.data_dir / 'train_y.npy')
        self.val_X = np.load(self.data_dir / 'val_X.npy')
        self.val_y = np.load(self.data_dir / 'val_y.npy')

        print(f"Train: {self.train_X.shape}, Val: {self.val_X.shape}")
        print(f"Train: {np.bincount(self.train_y)} | Val: {np.bincount(self.val_y)}")
        
    def create_balanced_dataset(self):
        """Create 1:1 balanced dataset like ResNet50"""
        print("\n🔧 Creating BALANCED 1:1 dataset...")

        melanoma_mask = self.train_y == 1
        benign_mask = self.train_y == 0

        X_melanoma = self.train_X[melanoma_mask]
        X_benign = self.train_X[benign_mask]

        print(f"Original: {len(X_melanoma)} melanoma, {len(X_benign)} benign")

        # Strong augmentation for melanoma
        datagen = ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.6, 1.4],
            zoom_range=0.3,
            shear_range=0.2,
            fill_mode='nearest'
        )

        # Target: 3500 samples each (1:1 balanced)
        target_samples = 3500
        
        # Augment melanoma
        melanoma_factor = target_samples // len(X_melanoma)
        print(f"Augmenting melanoma {melanoma_factor}x...")
        
        augmented_melanoma = list(X_melanoma)
        for img in X_melanoma:
            img_exp = np.expand_dims(img, 0)
            aug_iter = datagen.flow(img_exp, batch_size=1)
            for _ in range(melanoma_factor - 1):
                augmented_melanoma.append(next(aug_iter)[0])
        
        X_melanoma_aug = np.array(augmented_melanoma[:target_samples])
        y_melanoma_aug = np.ones(len(X_melanoma_aug), dtype=int)

        # Sample benign to match
        benign_indices = np.random.choice(len(X_benign), target_samples, replace=False)
        X_benign_sample = X_benign[benign_indices]
        y_benign_sample = np.zeros(len(X_benign_sample), dtype=int)

        # Combine and shuffle
        self.train_X_balanced = np.concatenate([X_melanoma_aug, X_benign_sample])
        self.train_y_balanced = np.concatenate([y_melanoma_aug, y_benign_sample])

        indices = np.arange(len(self.train_X_balanced))
        np.random.shuffle(indices)
        self.train_X_balanced = self.train_X_balanced[indices]
        self.train_y_balanced = self.train_y_balanced[indices]

        print(f"✅ Balanced: {np.bincount(self.train_y_balanced)} (1:1 ratio)")
        print(f"Total: {len(self.train_X_balanced)} samples\n")
    
    def build_unet_classifier(self):
        """Build U-Net adapted for classification by adding global pooling"""
        inputs = Input(shape=(224, 224, 3), name='input')
        
        # Encoder
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Bottleneck
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        
        # For classification, add global pooling instead of decoder
        gap = GlobalAveragePooling2D()(conv4)
        dense1 = Dense(256, activation='relu')(gap)
        dense1 = Dropout(0.5)(dense1)
        dense2 = Dense(128, activation='relu')(dense1)
        dense2 = Dropout(0.3)(dense2)
        outputs = Dense(2, activation='softmax')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs, name='UNet_Classifier')
        return model
    
    def train(self):
        """Train the U-Net classifier with focal loss"""
        print("🏗️ Building U-Net classifier model...")
        model = self.build_unet_classifier()
        
        # Create balanced dataset
        self.create_balanced_dataset()
        
        # Convert labels to categorical
        train_y_cat = tf.keras.utils.to_categorical(self.train_y_balanced, 2)
        val_y_cat = tf.keras.utils.to_categorical(self.val_y, 2)
        
        # Compile with focal loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=FocalLoss(gamma=3.0, alpha=0.8),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Training
        print("🚀 Starting training with focal loss...")
        history = model.fit(
            self.train_X_balanced, train_y_cat,
            validation_data=(self.val_X, val_y_cat),
            epochs=30,  # Increased epochs
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    self.model_dir / 'unet_focal_recall.h5',
                    save_best_only=True,
                    monitor='val_recall_1'  # Save based on recall
                )
            ]
        )
        
        return model, history

# Usage
if __name__ == "__main__":
    trainer = UNetClassifierTrainer()
    model, history = trainer.train()
    
    print("✅ U-Net classifier trained with focal loss for improved recall!")
    print("💡 Use this model in your ensemble with adjusted weights for better balance.")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, 
                                     concatenate, Dense, Dropout, GlobalAveragePooling2D,
                                     BatchNormalization, Activation)
from tensorflow.keras import regularizers

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for imbalanced data - same as ResNet50"""
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

class UNetClassifierTrainer:
    """U-Net adapted for classification with focal loss"""
    
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.num_classes = 2
        
    def build_unet_classifier(self):
        """Build U-Net adapted for classification by adding global pooling"""
        inputs = Input(shape=self.input_shape, name='input')
        
        # Encoder
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Bottleneck
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        
        # For classification, add global pooling instead of decoder
        gap = GlobalAveragePooling2D()(conv4)
        dense1 = Dense(256, activation='relu')(gap)
        dense1 = Dropout(0.5)(dense1)
        dense2 = Dense(128, activation='relu')(dense1)
        dense2 = Dropout(0.3)(dense2)
        outputs = Dense(self.num_classes, activation='softmax')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs, name='UNet_Classifier')
        return model
    
    def train(self):
        """Train the U-Net classifier with focal loss"""
        print("🏗️ Building U-Net classifier model...")
        model = self.build_unet_classifier()
        
        # Compile with focal loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=FocalLoss(gamma=3.0, alpha=0.8),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Data loading (assuming same as other scripts)
        # You'll need to add your data loading code here
        # For now, placeholder
        print("📂 Loading data...")
        # X_train, y_train, X_val, y_val = load_your_data()
        
        # Training
        print("🚀 Starting training with focal loss...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,  # Increased epochs
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    'models/unet_focal_recall.h5',
                    save_best_only=True,
                    monitor='val_recall_1'  # Save based on recall
                )
            ]
        )
        
        return model, history

# Usage
if __name__ == "__main__":
    trainer = UNetClassifierTrainer()
    model, history = trainer.train()
    
    print("✅ U-Net classifier trained with focal loss for improved recall!")
    print("💡 Use this model in your ensemble with adjusted weights for better balance.")