"""
ENSEMBLE PREDICTION FOR MELANOMA DETECTION
==========================================

Combines existing best model + new balanced model for improved F1 and Recall
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

class EnsemblePredictor:
    """Ensemble of existing and new models"""

    def __init__(self, existing_model_path='../models/best_model_20251103_225237.h5',
                 new_model_path='../models/hybrid_balanced_latest.h5'):
        self.existing_model_path = Path(existing_model_path)
        self.new_model_path = Path(new_model_path)

        # Load models
        print("Loading models...")
        self.existing_model = tf.keras.models.load_model(self.existing_model_path)
        self.new_model = tf.keras.models.load_model(self.new_model_path)

        # Weights based on validation performance (higher weight for better recall)
        # Adjust these based on your validation results
        self.existing_weight = 0.4  # Lower weight if lower recall
        self.new_weight = 0.6       # Higher weight for better recall

        print("Models loaded successfully!")

    def preprocess_image(self, image):
        """Preprocess image for both models"""
        # Assuming image is numpy array or PIL
        if isinstance(image, np.ndarray):
            if image.dtype != np.float32:
                image = image.astype(np.float32)
        else:
            # Convert PIL to numpy
            image = np.array(image).astype(np.float32)

        # Resize if needed
        if image.shape[:2] != (224, 224):
            image = tf.image.resize(image, (224, 224))

        # Normalize (assuming models expect 0-1 or -1 to 1)
        # Adjust based on your preprocessing
        image = image / 255.0  # Assuming 0-255 input

        return image

    def predict(self, image, threshold=0.4):
        """
        Ensemble prediction with adjusted threshold for higher recall

        Args:
            image: Input image (numpy array or PIL)
            threshold: Classification threshold (lower for higher recall)

        Returns:
            dict: Prediction results
        """
        processed_image = self.preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)

        # Get predictions from both models
        try:
            # Existing model (assuming binary classification output)
            existing_pred = self.existing_model.predict(processed_image, verbose=0)
            if isinstance(existing_pred, list):
                existing_prob = existing_pred[0][0] if len(existing_pred[0].shape) > 1 else existing_pred[0]
            else:
                existing_prob = existing_pred[0][0] if len(existing_pred.shape) > 1 else existing_pred[0]

        except Exception as e:
            print(f"Error with existing model: {e}")
            existing_prob = 0.5  # Fallback

        try:
            # New hybrid model (classification output is second)
            new_pred = self.new_model.predict(processed_image, verbose=0)
            if isinstance(new_pred, list):
                new_prob = new_pred[1][0][1]  # Classification output, class 1 prob
            else:
                new_prob = new_pred[0][1] if len(new_pred.shape) > 1 else new_pred[1]

        except Exception as e:
            print(f"Error with new model: {e}")
            new_prob = 0.5  # Fallback

        # Ensemble: weighted average based on recall performance
        ensemble_prob = (self.existing_weight * existing_prob + self.new_weight * new_prob) / (self.existing_weight + self.new_weight)

        # Classification with adjusted threshold
        predicted_class = 1 if ensemble_prob > threshold else 0
        confidence = ensemble_prob * 100 if predicted_class == 1 else (1 - ensemble_prob) * 100

        disease = 'Melanoma' if predicted_class == 1 else 'Benign'

        return {
            'disease': disease,
            'confidence': confidence,
            'ensemble_prob': ensemble_prob,
            'existing_prob': existing_prob,
            'new_prob': new_prob
        }

# Convenience function for easy use
def predict_melanoma(image, threshold=0.4):
    """
    Quick prediction function

    Args:
        image: Input image
        threshold: Classification threshold

    Returns:
        dict: Results
    """
    predictor = EnsemblePredictor()
    return predictor.predict(image, threshold)

if __name__ == '__main__':
    # Test with dummy image
    dummy_image = np.random.rand(224, 224, 3).astype(np.float32)

    predictor = EnsemblePredictor()
    result = predictor.predict(dummy_image)

    print("Ensemble Prediction Test:")
    print(f"Disease: {result['disease']}")
    print(".2f")
    print(".3f")
    print(".3f")
    print(".3f")