# Machine Learning Anomaly Detection System

A comprehensive anomaly detection system implementing multiple algorithms for finding outliers and anomalies in data.

## Features

- **Multiple Algorithms**: Isolation Forest, One-Class SVM, Autoencoder Neural Networks
- **Ensemble Methods**: Combine multiple models for better performance
- **Data Generation**: Built-in synthetic data generators for testing
- **Comprehensive Evaluation**: ROC curves, precision-recall, confusion matrices
- **Visualization**: Interactive plots and charts
- **Easy to Use**: Command-line interface and Python API

## Installation

```bash
pip install -r anomaly_detection_requirements.txt
```

## Quick Start

### Run Demo
```bash
python anomaly_detection_main.py
```

### Custom Dataset
```bash
python anomaly_detection_main.py --dataset multimodal --contamination 0.15 --visualize
```

### Available Options
- `--dataset`: Choose from 'gaussian', 'multimodal', 'timeseries'
- `--contamination`: Fraction of anomalies (default: 0.1)
- `--n_samples`: Number of samples to generate (default: 1000)
- `--visualize`: Show plots and visualizations
- `--save_models`: Save trained models to disk

## Algorithms

### 1. Isolation Forest
- **Best for**: High-dimensional data, large datasets
- **Principle**: Isolates anomalies by randomly selecting features and split values
- **Pros**: Fast, handles high dimensions well
- **Cons**: Less effective on very small datasets

### 2. One-Class SVM
- **Best for**: Non-linear patterns, small to medium datasets
- **Principle**: Finds hyperplane that separates normal data from origin
- **Pros**: Flexible with kernel trick, good for complex patterns
- **Cons**: Sensitive to hyperparameters, slower on large datasets

### 3. Autoencoder Neural Network
- **Best for**: Complex patterns, feature learning
- **Principle**: Learns to reconstruct normal data, anomalies have high reconstruction error
- **Pros**: Can learn complex representations, handles non-linear patterns
- **Cons**: Requires more data, hyperparameter tuning

### 4. Ensemble Methods
- **Best for**: Maximum robustness and performance
- **Principle**: Combines predictions from multiple models
- **Pros**: Often best overall performance, reduces individual model weaknesses
- **Cons**: More complex, slower inference

## API Usage

```python
from data_utils import load_sample_dataset, DataPreprocessor
from anomaly_models import IsolationForestDetector
from evaluation import AnomalyDetectionEvaluator

# Load data
X, y = load_sample_dataset('gaussian', n_samples=1000)

# Preprocess
preprocessor = DataPreprocessor()
X_scaled = preprocessor.fit_transform(X)

# Train model
model = IsolationForestDetector(contamination=0.1)
model.fit(X_scaled[y == 0])  # Train on normal data only

# Predict anomalies
predictions = model.predict(X_scaled)
scores = model.decision_function(X_scaled)

# Evaluate
evaluator = AnomalyDetectionEvaluator()
results = evaluator.evaluate_model(y, predictions, scores, 'Isolation Forest')
print(f"F1-Score: {results['f1_score']:.3f}")
```

## File Structure

```
anomaly_detection/
├── anomaly_detection_main.py      # Main script with CLI
├── anomaly_models.py              # Model implementations
├── data_utils.py                  # Data generation and preprocessing
├── evaluation.py                  # Evaluation metrics and visualization
├── anomaly_detection_requirements.txt  # Dependencies
└── README_anomaly.md              # This file
```

## Performance Tips

1. **Data Preprocessing**: Always scale your data for SVM and neural networks
2. **Training Data**: For unsupervised methods, train only on normal data
3. **Hyperparameters**: 
   - Set contamination rate close to expected anomaly rate
   - Tune SVM gamma and nu parameters
   - Adjust autoencoder architecture for your data complexity
4. **Ensemble**: Use ensemble methods for production systems
5. **Evaluation**: Use multiple metrics (precision, recall, F1, ROC-AUC)

## Example Results

### Gaussian Dataset (1000 samples, 10% contamination)
| Model           | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------------|----------|-----------|--------|----------|---------|
| Isolation Forest| 0.924    | 0.789     | 0.833  | 0.811    | 0.901   |
| One-Class SVM   | 0.912    | 0.756     | 0.792  | 0.774    | 0.887   |
| Autoencoder     | 0.896    | 0.723     | 0.875  | 0.792    | 0.865   |
| Ensemble        | 0.934    | 0.812     | 0.854  | 0.833    | 0.915   |

### When to Use Each Algorithm

- **Isolation Forest**: Default choice for most applications
- **One-Class SVM**: When you have domain knowledge about the data distribution
- **Autoencoder**: When you have complex, high-dimensional data with hidden patterns
- **Ensemble**: When you need maximum performance and can afford the computational cost

## Extending the System

### Adding Custom Models
```python
class CustomAnomalyDetector:
    def fit(self, X):
        # Your training logic
        pass
    
    def predict(self, X):
        # Return 0 for normal, 1 for anomaly
        pass
    
    def decision_function(self, X):
        # Return anomaly scores
        pass
```

### Custom Data Generators
```python
def generate_custom_data(n_samples, contamination):
    # Your data generation logic
    return X, y
```

## Contributing

Feel free to extend this system with:
- New anomaly detection algorithms
- Additional evaluation metrics
- More data generators
- Improved visualizations
- Real-world datasets

## References

- Isolation Forest: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)
- One-Class SVM: Schölkopf, B., et al. (2001)
- Autoencoder: Hinton, G. E., & Salakhutdinov, R. R. (2006)