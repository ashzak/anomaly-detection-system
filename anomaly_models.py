import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
import joblib
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class IsolationForestDetector:
    """Isolation Forest for anomaly detection"""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit the Isolation Forest model"""
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for anomaly, 0 for normal)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Isolation Forest returns 1 for normal, -1 for anomaly
        predictions = self.model.predict(X)
        # Convert to 0 for normal, 1 for anomaly
        return (predictions == -1).astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return -self.model.decision_function(X)  # Negative for higher anomaly score
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = joblib.load(filepath)
        self.is_fitted = True

class OneClassSVMDetector:
    """One-Class SVM for anomaly detection"""
    
    def __init__(self, kernel: str = 'rbf', gamma: str = 'scale', nu: float = 0.1):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'OneClassSVMDetector':
        """Fit the One-Class SVM model"""
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for anomaly, 0 for normal)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # One-Class SVM returns 1 for normal, -1 for anomaly
        predictions = self.model.predict(X)
        # Convert to 0 for normal, 1 for anomaly
        return (predictions == -1).astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return -self.model.decision_function(X)  # Negative for higher anomaly score
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = joblib.load(filepath)
        self.is_fitted = True

class AutoencoderDetector:
    """Autoencoder neural network for anomaly detection"""
    
    def __init__(self, 
                 encoding_dim: int = 8,
                 hidden_layers: list = [32, 16],
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100):
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.encoder = None
        self.decoder = None
        self.is_fitted = False
        self.input_dim = None
        self.threshold = None
    
    def _build_model(self, input_dim: int):
        """Build the autoencoder model"""
        self.input_dim = input_dim
        
        # Input layer
        input_layer = layers.Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for hidden_dim in self.hidden_layers:
            encoded = layers.Dense(hidden_dim, activation='relu')(encoded)
        
        # Bottleneck
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck')(encoded)
        
        # Decoder
        decoded = encoded
        for hidden_dim in reversed(self.hidden_layers):
            decoded = layers.Dense(hidden_dim, activation='relu')(decoded)
        
        # Output layer
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Create models
        self.model = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        
        # Compile
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X: np.ndarray, validation_split: float = 0.2) -> 'AutoencoderDetector':
        """Fit the autoencoder model"""
        if self.model is None:
            self._build_model(X.shape[1])
        
        # Train on normal data only (assuming most data is normal)
        history = self.model.fit(
            X, X,  # Autoencoder tries to reconstruct input
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            shuffle=True,
            verbose=1
        )
        
        # Calculate reconstruction errors for threshold
        reconstructed = self.model.predict(X)
        mse = np.mean(np.square(X - reconstructed), axis=1)
        
        # Set threshold as 95th percentile of training reconstruction errors
        self.threshold = np.percentile(mse, 95)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for anomaly, 0 for normal)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        reconstructed = self.model.predict(X)
        mse = np.mean(np.square(X - reconstructed), axis=1)
        
        return (mse > self.threshold).astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (reconstruction errors)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        reconstructed = self.model.predict(X)
        mse = np.mean(np.square(X - reconstructed), axis=1)
        
        return mse
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode data to latent space"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.encoder.predict(X)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        self.model.save(filepath)
        # Save threshold separately
        np.save(f"{filepath}_threshold.npy", self.threshold)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        self.threshold = np.load(f"{filepath}_threshold.npy")
        
        # Recreate encoder
        bottleneck_layer = self.model.get_layer('bottleneck')
        self.encoder = Model(self.model.input, bottleneck_layer.output)
        
        self.is_fitted = True

class AnomalyDetectionEnsemble:
    """Ensemble of multiple anomaly detection models"""
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """Fit all models in the ensemble"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, voting: str = 'majority') -> np.ndarray:
        """Predict using ensemble voting"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        predictions = np.array([model.predict(X) for model in self.models.values()])
        
        if voting == 'majority':
            return (np.sum(predictions, axis=0) > len(self.models) / 2).astype(int)
        elif voting == 'unanimous':
            return (np.sum(predictions, axis=0) == len(self.models)).astype(int)
        else:
            raise ValueError("Voting must be 'majority' or 'unanimous'")
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        return {name: model.predict(X) for name, model in self.models.items()}