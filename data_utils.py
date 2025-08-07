import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyDataGenerator:
    """Generate synthetic datasets for anomaly detection"""
    
    @staticmethod
    def generate_gaussian_data(n_samples: int = 1000, 
                             n_features: int = 2, 
                             contamination: float = 0.1,
                             random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Gaussian distributed data with outliers"""
        np.random.seed(random_state)
        
        # Generate normal samples
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal
        
        # Normal data from standard distribution
        normal_data = np.random.randn(n_normal, n_features)
        
        # Anomalous data with higher variance and shifted mean
        anomaly_data = np.random.randn(n_anomalies, n_features) * 3 + 5
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
        
        return X, y
    
    @staticmethod
    def generate_multimodal_data(n_samples: int = 1000,
                               n_features: int = 2,
                               n_centers: int = 3,
                               contamination: float = 0.1,
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate multimodal data with outliers"""
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal
        
        # Generate normal clusters
        X_normal, _ = make_blobs(n_samples=n_normal, 
                                centers=n_centers, 
                                n_features=n_features,
                                cluster_std=1.0,
                                random_state=random_state)
        
        # Generate anomalous points far from clusters
        np.random.seed(random_state + 1)
        X_anomaly = np.random.uniform(-10, 10, (n_anomalies, n_features))
        
        # Combine data
        X = np.vstack([X_normal, X_anomaly])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
        
        return X, y
    
    @staticmethod
    def generate_time_series_data(n_samples: int = 1000,
                                contamination: float = 0.1,
                                random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate time series data with anomalies"""
        np.random.seed(random_state)
        
        # Generate time points
        t = np.linspace(0, 4*np.pi, n_samples)
        
        # Normal pattern: sine wave with noise
        normal_pattern = np.sin(t) + 0.1 * np.random.randn(n_samples)
        
        # Add anomalies
        n_anomalies = int(n_samples * contamination)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        # Create anomalous spikes
        data = normal_pattern.copy()
        data[anomaly_indices] += np.random.choice([-1, 1], n_anomalies) * np.random.uniform(2, 4, n_anomalies)
        
        # Labels
        y = np.zeros(n_samples)
        y[anomaly_indices] = 1
        
        # Reshape for sklearn compatibility
        X = data.reshape(-1, 1)
        
        return X, y

class DataPreprocessor:
    """Preprocess data for anomaly detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data"""
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        return self.scaler.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        return self.scaler.inverse_transform(X)

def load_sample_dataset(dataset_type: str = 'gaussian', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Load a sample dataset for testing"""
    generator = AnomalyDataGenerator()
    
    if dataset_type == 'gaussian':
        return generator.generate_gaussian_data(**kwargs)
    elif dataset_type == 'multimodal':
        return generator.generate_multimodal_data(**kwargs)
    elif dataset_type == 'timeseries':
        return generator.generate_time_series_data(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def visualize_data(X: np.ndarray, y: np.ndarray, title: str = "Anomaly Detection Data"):
    """Visualize 2D data with anomalies highlighted"""
    plt.figure(figsize=(10, 8))
    
    if X.shape[1] == 1:
        # Time series visualization
        plt.subplot(2, 1, 1)
        plt.plot(X.flatten(), 'b-', alpha=0.7, label='Data')
        anomaly_indices = np.where(y == 1)[0]
        plt.scatter(anomaly_indices, X[anomaly_indices].flatten(), 
                   color='red', s=50, label='Anomalies', zorder=5)
        plt.title(title)
        plt.legend()
        plt.ylabel('Value')
        
        plt.subplot(2, 1, 2)
        plt.hist(X.flatten(), bins=50, alpha=0.7, color='blue', label='Normal')
        plt.hist(X[y == 1].flatten(), bins=20, alpha=0.7, color='red', label='Anomalies')
        plt.title('Distribution')
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    
    elif X.shape[1] == 2:
        # 2D scatter plot
        normal_mask = y == 0
        anomaly_mask = y == 1
        
        plt.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                   c='blue', alpha=0.6, s=20, label='Normal')
        plt.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], 
                   c='red', alpha=0.8, s=50, label='Anomalies')
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    else:
        # High-dimensional data - show first two dimensions
        plt.scatter(X[y == 0, 0], X[y == 0, 1], 
                   c='blue', alpha=0.6, s=20, label='Normal')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], 
                   c='red', alpha=0.8, s=50, label='Anomalies')
        plt.title(f"{title} (First 2 dimensions)")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()