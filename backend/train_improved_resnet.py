"""
IMPROVED TRAINING WITH MIXUP FOR BETTER GENERALIZATION
======================================================

Enhances model accuracy and precision while maintaining high recall
Uses MixUp data augmentation for robust feature learning
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class MixUpDataGenerator:
    """Data generator with MixUp augmentation"""

    def __init__(self, X, y, batch_size=32, alpha=0.2):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(X))

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        # Apply MixUp
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = len(X_batch)

        # Random shuffle for mixing
        mix_indices = np.random.permutation(batch_size)
        X_mixed = lam * X_batch + (1 - lam) * X_batch[mix_indices]
        y_mixed = lam * tf.keras.utils.to_categorical(y_batch, 2) + \
                 (1 - lam) * tf.keras.utils.to_categorical(y_batch[mix_indices], 2)

        return X_mixed, y_mixed

class ImprovedResNet50Trainer:
    """Improved ResNet50 trainer with MixUp and advanced techniques"""

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

    def create_augmented_dataset(self):
        """Create augmented dataset with MixUp-ready data"""
        print("\n🔧 Creating AUGMENTED dataset with advanced techniques...")

        melanoma_mask = self.train_y == 1
        benign_mask = self.train_y == 0

        X_melanoma = self.train_X[melanoma_mask]
        X_benign = self.train_X[benign_mask]

        print(f"Original: {len(X_melanoma)} melanoma, {len(X_benign)} benign")

        # Advanced augmentation for melanoma
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.25,
            height_shift_range=0.25,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            zoom_range=0.2,
            shear_range=0.15,
            fill_mode='reflect'
        )

        # Target: balanced dataset
        target_samples = min(4000, len(X_benign))  # Don't oversample benign too much

        # Augment melanoma more aggressively
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

        # Sample benign
        benign_indices = np.random.choice(len(X_benign), target_samples, replace=False)
        X_benign_sample = X_benign[benign_indices]
        y_benign_sample = np.zeros(len(X_benign_sample), dtype=int)

        # Combine
        self.train_X_aug = np.concatenate([X_melanoma_aug, X_benign_sample])
        self.train_y_aug = np.concatenate([y_melanoma_aug, y_benign_sample])

        # Shuffle
        indices = np.arange(len(self.train_X_aug))
        np.random.shuffle(indices)
        self.train_X_aug = self.train_X_aug[indices]
        self.train_y_aug = self.train_y_aug[indices]

        print(f"✅ Augmented: {np.bincount(self.train_y_aug)}")
        print(f"Total: {len(self.train_X_aug)} samples\n")

    def build_improved_model(self):
        """Build improved ResNet50 with better regularization"""
        print("🏗️ Building IMPROVED ResNet50 model...")

        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )

        # Fine-tune more layers
        for layer in base_model.layers[:-60]:  # Fine-tune more layers
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=outputs)

        return model

    def focal_loss(self, y_true, y_pred, gamma=2.0, alpha=0.75):
        """Focal loss for imbalanced data"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # For binary classification
        p_t = y_true * y_pred[:, 1] + (1 - y_true) * (1 - y_pred[:, 1])
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, gamma)

        bce = -tf.reduce_sum(y_true * tf.math.log(y_pred[:, 1] + 1e-7) +
                           (1 - y_true) * tf.math.log(1 - y_pred[:, 1] + 1e-7), axis=-1)

        return tf.reduce_mean(focal_weight * bce)

    def train_model(self):
        """Train the improved model"""
        self.create_augmented_dataset()
        model = self.build_improved_model()

        # Compile with focal loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: self.focal_loss(y_true, y_pred),
            metrics=['accuracy']
        )

        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.model_dir / f'improved_model_{timestamp}.h5'

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
        ]

        # Data generators
        train_gen = MixUpDataGenerator(self.train_X_aug, self.train_y_aug, batch_size=32, alpha=0.4)
        val_gen = MixUpDataGenerator(self.val_X, self.val_y, batch_size=32, alpha=0.1)  # Light MixUp for validation

        # Train
        print("🚀 Training improved model...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=50,
            callbacks=callbacks
        )

        # Save final model
        final_path = self.model_dir / f'improved_final_{timestamp}.h5'
        model.save(final_path)
        print(f"✅ Model saved: {final_path}")

        return model, history

if __name__ == '__main__':
    trainer = ImprovedResNet50Trainer()
    model, history = trainer.train_model()

    print("🎉 Training completed! Model should have improved accuracy and precision while maintaining high recall.")