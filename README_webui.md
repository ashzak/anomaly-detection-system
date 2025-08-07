# Anomaly Detection Web UI

A comprehensive web interface for the Anomaly Detection System built with Streamlit.

## 🚀 Quick Start

### Option 1: Using the launcher script
```bash
python3 run_app.py
```

### Option 2: Direct Streamlit command
```bash
python3 -m streamlit run anomaly_detection_app.py
```

The web interface will open automatically in your browser at `http://localhost:8501`

## 📱 Features

### 🎲 Data Generation Page
- **Interactive Dataset Creation**: Generate synthetic datasets with customizable parameters
- **Multiple Dataset Types**: 
  - Gaussian (normal distribution with outliers)
  - Multimodal (multiple clusters with anomalies)
  - Time Series (temporal data with spikes)
- **Real-time Visualization**: Interactive plots powered by Plotly
- **Parameter Control**: Adjust sample count, contamination rate, features, etc.

### 🤖 Model Training Page
- **Multiple Algorithms**: Train Isolation Forest, One-Class SVM, and Autoencoder simultaneously
- **Hyperparameter Tuning**: Adjust model-specific parameters through intuitive sliders
- **Progress Tracking**: Real-time training progress with status updates
- **Ensemble Methods**: Automatic ensemble creation for improved performance

### 📊 Results & Evaluation Page
- **Performance Comparison**: Side-by-side model comparison with highlighted best performers
- **Interactive Visualizations**:
  - ROC curves with AUC scores
  - Precision-Recall curves
  - Score distribution histograms
  - Confusion matrices
- **Detailed Metrics**: Comprehensive evaluation including accuracy, precision, recall, F1-score
- **Model Selection**: Detailed view of individual model performance

### 📁 Custom Dataset Upload
- **CSV File Support**: Upload your own datasets for analysis
- **Automatic Detection**: Smart detection of label columns
- **Sample Downloads**: Get example datasets for testing
- **Data Preview**: Inspect your data before training

## 🎯 Key Features

### Interactive Interface
- **Responsive Design**: Optimized for desktop and tablet viewing
- **Real-time Updates**: Dynamic content that updates as you work
- **Progress Indicators**: Visual feedback during long-running operations
- **Error Handling**: User-friendly error messages and recovery suggestions

### Advanced Visualizations
- **Plotly Integration**: Interactive charts you can zoom, pan, and explore
- **Multi-dimensional Display**: Smart visualization for 1D, 2D, and high-dimensional data
- **Color-coded Results**: Easy identification of normal vs. anomalous data points
- **Statistical Summaries**: Key metrics and distributions at a glance

### Session Management
- **State Persistence**: Your work is saved as you navigate between pages
- **Data Continuity**: Generated datasets and trained models persist across sessions
- **Progress Tracking**: The app remembers where you left off

## 📋 Usage Workflow

1. **📊 Start with Data**
   - Go to "Data Generation" page
   - Choose dataset type and parameters
   - Generate and visualize your data

2. **🤖 Train Models**
   - Navigate to "Model Training" page
   - Adjust hyperparameters as needed
   - Click "Train Models" and wait for completion

3. **📈 Analyze Results**
   - Visit "Results & Evaluation" page
   - Compare model performance
   - Explore detailed visualizations

4. **📁 Upload Custom Data** (Optional)
   - Use "Upload Dataset" page for your own data
   - Follow CSV format guidelines
   - Return to step 2 with your data

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit with custom CSS styling
- **Visualization**: Plotly for interactive charts
- **Backend**: Original anomaly detection modules
- **State Management**: Streamlit session state

### Performance Considerations
- **Chunked Processing**: Large datasets processed in batches
- **Progress Feedback**: Real-time updates during training
- **Memory Management**: Efficient data handling for large datasets
- **Caching**: Smart caching of expensive computations

### Browser Compatibility
- **Chrome**: Full support (recommended)
- **Firefox**: Full support
- **Safari**: Full support
- **Edge**: Full support

## 🎨 Customization

### Themes
The interface uses a clean, professional theme with:
- Blue accent colors for primary actions
- Color-coded results (blue for normal, red for anomalies)
- Responsive layout that adapts to screen size

### Configuration
Key settings can be adjusted in `anomaly_detection_app.py`:
- Default parameters for each algorithm
- Visualization colors and styles
- Page layout and organization

## 🔧 Troubleshooting

### Common Issues

**App won't start:**
```bash
# Check if Streamlit is installed
python3 -c "import streamlit; print('Streamlit OK')"

# Install if missing
pip3 install streamlit
```

**Import errors:**
- Ensure all required files are in the same directory
- Check that dependencies are installed: `pip3 install -r anomaly_detection_requirements.txt`

**Performance issues:**
- Reduce dataset size for faster training
- Use fewer epochs for autoencoder
- Close other browser tabs to free memory

**Browser not opening:**
- Manually navigate to `http://localhost:8501`
- Try a different port: `streamlit run anomaly_detection_app.py --server.port 8502`

### Debug Mode
Run with debug output:
```bash
streamlit run anomaly_detection_app.py --logger.level=debug
```

## 📈 Advanced Usage

### Custom Model Integration
To add new anomaly detection models:

1. Implement the model in `anomaly_models.py` following the existing pattern
2. Add configuration controls in the Model Training page
3. Include the model in the training loop

### Dataset Extensions
Support for new data formats:

1. Extend the file upload functionality in `file_upload_page()`
2. Add preprocessing for your specific format
3. Update visualization logic if needed

### Export Functionality
The app supports:
- CSV download of results
- Model serialization for later use
- Visualization export (PNG, PDF)

## 🚀 Production Deployment

For production deployment:

```bash
# Install production dependencies
pip3 install streamlit gunicorn

# Run with gunicorn (optional)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker anomaly_detection_app:app
```

Or deploy to cloud platforms:
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Use provided Procfile
- **AWS/GCP**: Container deployment options

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure data format matches requirements
4. Check browser console for JavaScript errors

The web interface makes anomaly detection accessible to users of all technical levels while maintaining the full power of the underlying machine learning algorithms.