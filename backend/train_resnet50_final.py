"""
FIXED RESNET50 TRAINING FOR HIGH MELANOMA RECALL
=================================================
Using YOUR project's ResNet50 architecture
Key fixes:
1. Focal Loss for class imbalance
2. Balanced 1:1 dataset
3. Save based on MELANOMA RECALL (not accuracy)
4. Higher melanoma penalty
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

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for imbalanced data"""
    def __init__(self, gamma=3.0, alpha=0.8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Melanoma prediction (class 1)
        melanoma_pred = y_pred[:, 1]
        melanoma_true = y_true
        
        # Focal loss
        cross_entropy = -melanoma_true * tf.math.log(melanoma_pred) - (1 - melanoma_true) * tf.math.log(1 - melanoma_pred)
        weight = melanoma_true * self.alpha + (1 - melanoma_true) * (1 - self.alpha)
        focal_weight = tf.pow(tf.abs(melanoma_true - melanoma_pred), self.gamma)
        
        return tf.reduce_mean(weight * focal_weight * cross_entropy)

class ResNet50MelanomaTrainer:
    """ResNet50 trainer for your project"""

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
        """Create 1:1 balanced dataset"""
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

    def build_resnet50_model(self):
        """Build ResNet50 model for YOUR project"""
        print("🏗️ Building ResNet50 model...")
        
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Fine-tune last layers
        for layer in base_model.layers[:-40]:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)  # Strong dropout
        x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.4)(x)
        outputs = Dense(2, activation='softmax', dtype='float32')(x)
        
        model = Model(inputs=base_model.input, outputs=outputs)
        print(f"✅ ResNet50 model built\n")
        
        return model

    def train(self):
        """Train with focal loss and melanoma recall focus"""
        model = self.build_resnet50_model()

        # Compile with Focal Loss (high gamma and alpha for melanoma)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=FocalLoss(gamma=3.0, alpha=0.8),  # Aggressive focal loss
            metrics=['accuracy']
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f'resnet50_melanoma_{timestamp}.h5'
        
        best_melanoma_recall = [0.0]
        
        # Melanoma recall callback
        class MelanomaCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                val_pred = self.model.predict(trainer.val_X, batch_size=32, verbose=0)
                y_pred = np.argmax(val_pred, axis=1)
                
                melanoma_mask = trainer.val_y == 1
                melanoma_detected = np.sum((y_pred == 1) & melanoma_mask)
                melanoma_total = np.sum(melanoma_mask)
                melanoma_recall = melanoma_detected / melanoma_total
                
                benign_mask = trainer.val_y == 0
                benign_correct = np.sum((y_pred == 0) & benign_mask)
                benign_total = np.sum(benign_mask)
                benign_recall = benign_correct / benign_total
                
                overall_acc = np.mean(y_pred == trainer.val_y)
                
                print(f"\n{'='*80}")
                print(f"📊 EPOCH {epoch + 1}/20:")
                print(f"   Val Accuracy:       {overall_acc:.4f}")
                print(f"   🎯 MELANOMA RECALL:  {melanoma_recall:.4f} ({melanoma_detected}/{melanoma_total})")
                print(f"   Benign Recall:      {benign_recall:.4f}")
                
                if melanoma_recall > best_melanoma_recall[0]:
                    best_melanoma_recall[0] = melanoma_recall
                    print(f"   ✅ BEST MELANOMA RECALL! Saving model...")
                    self.model.save(model_path)
                
                print(f"{'='*80}\n")

        trainer = self
        melanoma_callback = MelanomaCallback()
        
        # Save checkpoint every epoch
        checkpoint_path = self.model_dir / f'checkpoint_resnet50_{timestamp}.h5'
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                save_freq='epoch',  # Save after every epoch
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=12,
                restore_best_weights=False,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-7
            ),
            melanoma_callback
        ]

        print("🚀 Starting ResNet50 training for HIGH MELANOMA RECALL...")
        print("="*80)
        
        history = model.fit(
            self.train_X_balanced,
            self.train_y_balanced,
            validation_data=(self.val_X, self.val_y),
            epochs=20,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        print(f"\n✅ Best model saved to: {model_path}")
        
        # Load and evaluate best model
        print("\n" + "="*80)
        print("📊 FINAL EVALUATION WITH BEST MODEL")
        print("="*80)
        
        best_model = tf.keras.models.load_model(
            model_path,
            custom_objects={'FocalLoss': FocalLoss}
        )
        
        val_pred = best_model.predict(self.val_X, batch_size=32)
        y_pred = np.argmax(val_pred, axis=1)
        
        print("\n📋 Classification Report:")
        print(classification_report(
            self.val_y, y_pred,
            target_names=['Benign', 'Melanoma'],
            digits=4
        ))
        
        cm = confusion_matrix(self.val_y, y_pred)
        print("\n🔢 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Benign  Melanoma")
        print(f"Actual Benign    {cm[0][0]:6d}  {cm[0][1]:6d}")
        print(f"       Melanoma  {cm[1][0]:6d}  {cm[1][1]:6d}")
        
        melanoma_recall = recall_score(self.val_y, y_pred, pos_label=1)
        print(f"\n🎯 FINAL MELANOMA RECALL: {melanoma_recall:.4f}")
        print(f"   ✅ Detected: {cm[1][1]}/{cm[1][0] + cm[1][1]} melanomas")
        print(f"   ❌ Missed: {cm[1][0]} (False Negatives)")
        
        if melanoma_recall >= 0.75:
            print(f"\n✅ EXCELLENT! Melanoma recall ≥ 75%")
        elif melanoma_recall >= 0.60:
            print(f"\n⚡ GOOD! Melanoma recall ≥ 60%")
        else:
            print(f"\n⚠️  Target: 60%+ melanoma recall")
        
        print("="*80)
        
        return model_path

if __name__ == '__main__':
    print("="*80)
    print("🔬 RESNET50 MELANOMA TRAINING (YOUR PROJECT)")
    print("="*80 + "\n")
    
    trainer = ResNet50MelanomaTrainer()
    trainer.create_balanced_dataset()
    model_path = trainer.train()
    
    print(f"\n✅ Training complete!")
    print(f"📁 Model: {model_path}")
    print("="*80)
