# Anomaly Detection System

A comprehensive machine learning system for detecting anomalies using multiple algorithms with both command-line and web interfaces.

## 🚀 Features

- **Multiple ML Algorithms**: Isolation Forest, One-Class SVM, Autoencoder Neural Networks
- **Interactive Web UI**: Built with Streamlit for easy use
- **Synthetic Data Generation**: Built-in generators for testing
- **Comprehensive Evaluation**: ROC curves, confusion matrices, detailed metrics
- **Ensemble Methods**: Combine multiple models for better performance
- **File Upload Support**: Use your own CSV datasets

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/ashzak/anomaly-detection-system.git
cd anomaly-detection-system
```

### 2. Install dependencies
```bash
# Core ML dependencies
pip3 install -r anomaly_detection_requirements.txt

# Web UI dependencies  
pip3 install -r webui_requirements.txt
```

## 🎯 Quick Start

### Web Interface (Recommended)
```bash
# Launch the web UI
./start_app.sh

# Or manually
python3 -m streamlit run anomaly_detection_app.py
```
Then open `http://localhost:8504` in your browser.

### Command Line Interface
```bash
# Run with default settings
python3 anomaly_detection_main.py

# Custom parameters
python3 anomaly_detection_main.py --dataset multimodal --contamination 0.15 --visualize
```

## 📁 Project Structure

```
├── anomaly_detection_main.py      # CLI interface
├── anomaly_detection_app.py       # Streamlit web interface
├── anomaly_models.py              # ML model implementations
├── data_utils.py                  # Data generation and preprocessing
├── evaluation.py                  # Evaluation metrics and visualization
├── gmail_agent.py                 # Gmail integration example
├── start_app.sh                   # Web UI launcher script
├── requirements files             # Dependencies
└── documentation/                 # README files and guides
```

## 🤖 Supported Algorithms

### 1. Isolation Forest
- **Best for**: Large datasets, high-dimensional data
- **Principle**: Isolates anomalies through random partitioning
- **Pros**: Fast, scalable, handles mixed data types

### 2. One-Class SVM  
- **Best for**: Non-linear patterns, medium datasets
- **Principle**: Finds decision boundary around normal data
- **Pros**: Flexible with kernels, good for complex patterns

### 3. Autoencoder Neural Network
- **Best for**: Complex patterns, feature learning
- **Principle**: Learns to reconstruct normal data, anomalies have high error
- **Pros**: Can capture complex non-linear relationships

### 4. Ensemble Methods
- **Best for**: Maximum robustness
- **Principle**: Combines multiple models using voting
- **Pros**: Often achieves best overall performance

## 📊 Web Interface Guide

### 1. Data Generation Page
- Create synthetic datasets with customizable parameters
- Support for Gaussian, multimodal, and time series data
- Interactive visualization with Plotly
- Real-time statistics

### 2. Model Training Page
- Configure hyperparameters for each algorithm
- Train multiple models simultaneously
- Progress tracking and status updates
- Automatic ensemble creation

### 3. Results & Evaluation Page
- Performance comparison with highlighting
- Interactive ROC and Precision-Recall curves
- Confusion matrices and score distributions
- Detailed model analysis

### 4. Upload Dataset Page
- CSV file support with automatic label detection
- Data preview and validation
- Sample dataset downloads

## 💡 Usage Examples

### Python API
```python
from data_utils import load_sample_dataset, DataPreprocessor
from anomaly_models import IsolationForestDetector
from evaluation import AnomalyDetectionEvaluator

# Load data
X, y = load_sample_dataset('gaussian', n_samples=1000, contamination=0.1)

# Preprocess
preprocessor = DataPreprocessor()
X_scaled = preprocessor.fit_transform(X)

# Train model (unsupervised - use only normal data)
normal_data = X_scaled[y == 0]
model = IsolationForestDetector(contamination=0.1)
model.fit(normal_data)

# Predict
predictions = model.predict(X_scaled)
scores = model.decision_function(X_scaled)

# Evaluate
evaluator = AnomalyDetectionEvaluator()
results = evaluator.evaluate_model(y, predictions, scores, 'Isolation Forest')
print(f"F1-Score: {results['f1_score']:.3f}")
```

### Command Line
```bash
# Generate gaussian data and compare all models
python3 anomaly_detection_main.py --dataset gaussian --contamination 0.1 --visualize

# Time series anomaly detection
python3 anomaly_detection_main.py --dataset timeseries --n_samples 2000 --contamination 0.05

# Save trained models
python3 anomaly_detection_main.py --save_models
```

## 📈 Performance Tips

1. **Data Preprocessing**: Always scale features for SVM and neural networks
2. **Training Data**: For unsupervised learning, train only on normal data
3. **Hyperparameters**: Set contamination rate close to expected anomaly rate
4. **Ensemble**: Use ensemble methods for production systems
5. **Evaluation**: Consider multiple metrics (precision, recall, F1, ROC-AUC)

## 🔧 Customization

### Adding New Models
```python
class CustomDetector:
    def fit(self, X):
        # Training logic
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
    # Custom data generation logic
    return X, y
```

## 🛠️ Development

### Running Tests
```bash
# Test model imports
python3 -c "from anomaly_models import *; print('Models OK')"

# Test web app
python3 -c "from anomaly_detection_app import *; print('Web app OK')"

# Generate sample results
python3 anomaly_detection_main.py --dataset gaussian --n_samples 500
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📋 Dependencies

### Core Dependencies
- scikit-learn >= 1.0.0
- tensorflow >= 2.8.0  
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0

### Web UI Dependencies
- streamlit >= 1.28.0
- plotly >= 5.0.0
- altair >= 5.0.0

## 🚀 Deployment

### Local Development
```bash
./start_app.sh
```

### Production
```bash
streamlit run anomaly_detection_app.py --server.port 8080 --server.address 0.0.0.0
```

### Docker (Optional)
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r anomaly_detection_requirements.txt -r webui_requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "anomaly_detection_app.py", "--server.address", "0.0.0.0"]
```

## 📞 Support

- Check the documentation in the `README_*.md` files
- Review the troubleshooting section in `README_webui.md`
- Ensure all dependencies are installed correctly
- Verify data format matches requirements

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with scikit-learn, TensorFlow, and Streamlit
- Inspired by various anomaly detection research papers
- Uses Plotly for interactive visualizations
