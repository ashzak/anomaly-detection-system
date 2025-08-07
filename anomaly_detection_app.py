import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_utils import AnomalyDataGenerator, DataPreprocessor, visualize_data
from anomaly_models import IsolationForestDetector, OneClassSVMDetector, AutoencoderDetector, AnomalyDetectionEnsemble
from evaluation import AnomalyDetectionEvaluator, save_results

# Page config
st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'X' not in st.session_state:
        st.session_state.X = None
    if 'y' not in st.session_state:
        st.session_state.y = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = None

def create_plotly_scatter(X, y, title="Anomaly Detection Data"):
    """Create interactive plotly scatter plot"""
    if X.shape[1] == 1:
        # Time series plot
        fig = go.Figure()
        
        # Normal points
        normal_idx = np.where(y == 0)[0]
        fig.add_trace(go.Scatter(
            x=normal_idx,
            y=X[normal_idx].flatten(),
            mode='markers+lines',
            name='Normal',
            marker=dict(color='blue', size=4),
            line=dict(color='blue', width=1)
        ))
        
        # Anomaly points
        anomaly_idx = np.where(y == 1)[0]
        fig.add_trace(go.Scatter(
            x=anomaly_idx,
            y=X[anomaly_idx].flatten(),
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=8, symbol='diamond')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='closest'
        )
    
    elif X.shape[1] >= 2:
        # 2D scatter plot
        df = pd.DataFrame({
            'Feature_1': X[:, 0],
            'Feature_2': X[:, 1],
            'Label': ['Anomaly' if label == 1 else 'Normal' for label in y]
        })
        
        fig = px.scatter(
            df, x='Feature_1', y='Feature_2', color='Label',
            title=title,
            color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'}
        )
        
        fig.update_traces(marker=dict(size=8))
    
    return fig

def data_generation_page():
    """Data Generation and Visualization Page"""
    st.markdown("<h1 class='main-header'>🔍 Anomaly Detection System</h1>", unsafe_allow_html=True)
    
    st.header("📊 Data Generation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Dataset Configuration")
        
        dataset_type = st.selectbox(
            "Dataset Type",
            ["gaussian", "multimodal", "timeseries"],
            help="Choose the type of synthetic dataset to generate"
        )
        
        n_samples = st.slider("Number of Samples", 100, 5000, 1000, 100)
        contamination = st.slider("Contamination Rate", 0.01, 0.3, 0.1, 0.01)
        
        if dataset_type == "multimodal":
            n_centers = st.slider("Number of Centers", 2, 8, 3)
        else:
            n_centers = 3
        
        n_features = st.slider("Number of Features", 1, 10, 2) if dataset_type != "timeseries" else 1
        
        generate_button = st.button("🎲 Generate Dataset", type="primary")
    
    with col2:
        st.subheader("Dataset Information")
        
        if st.session_state.data_generated:
            st.success(f"✅ Dataset generated successfully!")
            st.info(f"""
            - **Samples**: {st.session_state.X.shape[0]}
            - **Features**: {st.session_state.X.shape[1]}
            - **Anomalies**: {np.sum(st.session_state.y)} ({np.mean(st.session_state.y)*100:.1f}%)
            - **Normal**: {np.sum(st.session_state.y == 0)} ({np.mean(st.session_state.y == 0)*100:.1f}%)
            """)
        else:
            st.info("👆 Configure and generate a dataset to get started")
    
    # Generate dataset
    if generate_button:
        with st.spinner("Generating dataset..."):
            try:
                generator = AnomalyDataGenerator()
                
                if dataset_type == "gaussian":
                    X, y = generator.generate_gaussian_data(
                        n_samples=n_samples,
                        n_features=n_features,
                        contamination=contamination,
                        random_state=42
                    )
                elif dataset_type == "multimodal":
                    X, y = generator.generate_multimodal_data(
                        n_samples=n_samples,
                        n_features=n_features,
                        n_centers=n_centers,
                        contamination=contamination,
                        random_state=42
                    )
                else:  # timeseries
                    X, y = generator.generate_time_series_data(
                        n_samples=n_samples,
                        contamination=contamination,
                        random_state=42
                    )
                
                # Store in session state
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.data_generated = True
                st.session_state.models_trained = False
                
                st.success("Dataset generated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating dataset: {str(e)}")
    
    # Visualize data
    if st.session_state.data_generated:
        st.header("📈 Data Visualization")
        
        # Interactive plot
        fig = create_plotly_scatter(st.session_state.X, st.session_state.y, "Generated Dataset")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", st.session_state.X.shape[0])
        
        with col2:
            st.metric("Features", st.session_state.X.shape[1])
        
        with col3:
            st.metric("Anomalies", int(np.sum(st.session_state.y)))
        
        with col4:
            st.metric("Anomaly Rate", f"{np.mean(st.session_state.y)*100:.1f}%")

def model_training_page():
    """Model Training and Configuration Page"""
    st.header("🤖 Model Training")
    
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate a dataset first on the Data Generation page.")
        return
    
    # Model Configuration
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Isolation Forest**")
        if_contamination = st.slider("IF Contamination", 0.01, 0.5, 0.1, key="if_cont")
        if_estimators = st.slider("IF Estimators", 50, 500, 100, key="if_est")
        
        st.markdown("**One-Class SVM**")
        svm_nu = st.slider("SVM Nu", 0.01, 0.5, 0.1, key="svm_nu")
        svm_gamma = st.selectbox("SVM Gamma", ["scale", "auto", 0.1, 0.01, 0.001], key="svm_gamma")
    
    with col2:
        st.markdown("**Autoencoder**")
        ae_encoding_dim = st.slider("Encoding Dimension", 2, 32, 8, key="ae_dim")
        ae_epochs = st.slider("Training Epochs", 10, 200, 50, key="ae_epochs")
        ae_batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1, key="ae_batch")
        
        st.markdown("**General Settings**")
        train_test_split = st.slider("Training Split", 0.5, 0.9, 0.8, key="split")
    
    # Train Models Button
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Preprocess data
                preprocessor = DataPreprocessor()
                X_scaled = preprocessor.fit_transform(st.session_state.X)
                st.session_state.preprocessor = preprocessor
                
                # Split data
                normal_indices = np.where(st.session_state.y == 0)[0]
                anomaly_indices = np.where(st.session_state.y == 1)[0]
                
                n_train_normal = int(train_test_split * len(normal_indices))
                train_normal_idx = normal_indices[:n_train_normal]
                test_normal_idx = normal_indices[n_train_normal:]
                
                X_train = X_scaled[train_normal_idx]
                test_indices = np.concatenate([test_normal_idx, anomaly_indices])
                X_test = X_scaled[test_indices]
                y_test = st.session_state.y[test_indices]
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                # Initialize models
                models = {}
                
                # Isolation Forest
                models['Isolation Forest'] = IsolationForestDetector(
                    contamination=if_contamination,
                    n_estimators=if_estimators,
                    random_state=42
                )
                
                # One-Class SVM
                models['One-Class SVM'] = OneClassSVMDetector(
                    kernel='rbf',
                    nu=svm_nu,
                    gamma=svm_gamma
                )
                
                # Autoencoder
                models['Autoencoder'] = AutoencoderDetector(
                    encoding_dim=ae_encoding_dim,
                    epochs=ae_epochs,
                    batch_size=ae_batch_size,
                    hidden_layers=[32, 16] if st.session_state.X.shape[1] > 5 else [8, 4]
                )
                
                # Train models and collect results
                results = {}
                evaluator = AnomalyDetectionEvaluator()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, (model_name, model) in enumerate(models.items()):
                    status_text.text(f"Training {model_name}...")
                    
                    # Train model
                    model.fit(X_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    y_scores = model.decision_function(X_test)
                    
                    # Evaluate
                    result = evaluator.evaluate_model(y_test, y_pred, y_scores, model_name)
                    results[model_name] = result
                    
                    progress_bar.progress((idx + 1) / len(models))
                
                # Ensemble
                status_text.text("Creating ensemble...")
                ensemble_models = {
                    'isolation_forest': models['Isolation Forest'],
                    'one_class_svm': models['One-Class SVM']
                }
                
                ensemble = AnomalyDetectionEnsemble(ensemble_models)
                ensemble.fit(X_train)
                
                y_pred_ensemble = ensemble.predict(X_test, voting='majority')
                y_scores_ensemble = np.mean([
                    models['Isolation Forest'].decision_function(X_test),
                    models['One-Class SVM'].decision_function(X_test)
                ], axis=0)
                
                result_ensemble = evaluator.evaluate_model(
                    y_test, y_pred_ensemble, y_scores_ensemble, 'Ensemble'
                )
                results['Ensemble'] = result_ensemble
                models['Ensemble'] = ensemble
                
                # Store results
                st.session_state.models = models
                st.session_state.results = results
                st.session_state.models_trained = True
                
                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                
                st.success("✅ All models trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
    
    # Show training status
    if st.session_state.models_trained:
        st.success("✅ Models are trained and ready for evaluation!")
        
        # Quick results preview
        if st.session_state.results:
            st.subheader("📊 Quick Results Preview")
            
            # Create comparison dataframe
            evaluator = AnomalyDetectionEvaluator()
            comparison_df = evaluator.compare_models(st.session_state.results)
            
            st.dataframe(comparison_df, use_container_width=True)

def results_page():
    """Results and Evaluation Page"""
    st.header("📈 Model Evaluation Results")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first on the Model Training page.")
        return
    
    # Model comparison table
    st.subheader("🏆 Model Performance Comparison")
    
    evaluator = AnomalyDetectionEvaluator()
    comparison_df = evaluator.compare_models(st.session_state.results)
    
    # Highlight best performing model
    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_max]
    
    styled_df = comparison_df.style.apply(highlight_best, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Best model summary
    best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
    best_f1 = comparison_df['F1-Score'].max()
    
    st.success(f"🏆 **Best performing model**: {best_model} (F1-Score: {best_f1:.4f})")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Metrics", "📈 ROC Curves", "🎯 Precision-Recall", "📉 Score Distributions"])
    
    with tab1:
        # Performance metrics bar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metrics,
            specs=[[{"secondary_y": False}] * len(metrics)]
        )
        
        colors = px.colors.qualitative.Set3[:len(comparison_df)]
        
        for idx, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    name=metric,
                    marker_color=colors,
                    showlegend=False,
                    text=np.round(comparison_df[metric], 3),
                    textposition='auto'
                ),
                row=1, col=idx+1
            )
            
            fig.update_yaxes(range=[0, 1], row=1, col=idx+1)
        
        fig.update_layout(height=400, title_text="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # ROC Curves
        fig = go.Figure()
        
        from sklearn.metrics import roc_curve, auc
        
        for model_name, model in st.session_state.models.items():
            if model_name != 'Ensemble':
                y_scores = model.decision_function(st.session_state.X_test)
            else:
                y_scores = np.mean([
                    st.session_state.models['Isolation Forest'].decision_function(st.session_state.X_test),
                    st.session_state.models['One-Class SVM'].decision_function(st.session_state.X_test)
                ], axis=0)
            
            fpr, tpr, _ = roc_curve(st.session_state.y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.3f})',
                line=dict(width=2)
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Precision-Recall Curves
        fig = go.Figure()
        
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        for model_name, model in st.session_state.models.items():
            if model_name != 'Ensemble':
                y_scores = model.decision_function(st.session_state.X_test)
            else:
                y_scores = np.mean([
                    st.session_state.models['Isolation Forest'].decision_function(st.session_state.X_test),
                    st.session_state.models['One-Class SVM'].decision_function(st.session_state.X_test)
                ], axis=0)
            
            precision, recall, _ = precision_recall_curve(st.session_state.y_test, y_scores)
            avg_precision = average_precision_score(st.session_state.y_test, y_scores)
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{model_name} (AP = {avg_precision:.3f})',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Precision-Recall Curves Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Score Distributions
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Isolation Forest', 'One-Class SVM', 'Autoencoder', 'Ensemble']
        )
        
        model_positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for idx, (model_name, model) in enumerate(st.session_state.models.items()):
            if idx >= 4:
                break
                
            if model_name != 'Ensemble':
                y_scores = model.decision_function(st.session_state.X_test)
            else:
                y_scores = np.mean([
                    st.session_state.models['Isolation Forest'].decision_function(st.session_state.X_test),
                    st.session_state.models['One-Class SVM'].decision_function(st.session_state.X_test)
                ], axis=0)
            
            normal_scores = y_scores[st.session_state.y_test == 0]
            anomaly_scores = y_scores[st.session_state.y_test == 1]
            
            row, col = model_positions[idx]
            
            fig.add_trace(go.Histogram(
                x=normal_scores,
                name='Normal',
                opacity=0.7,
                marker_color='blue',
                showlegend=(idx == 0)
            ), row=row, col=col)
            
            fig.add_trace(go.Histogram(
                x=anomaly_scores,
                name='Anomaly',
                opacity=0.7,
                marker_color='red',
                showlegend=(idx == 0)
            ), row=row, col=col)
        
        fig.update_layout(height=600, title_text="Score Distributions by Model")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results
    st.subheader("📋 Detailed Results")
    
    selected_model = st.selectbox("Select model for detailed view:", list(st.session_state.results.keys()))
    
    if selected_model:
        result = st.session_state.results[selected_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Classification Metrics:**")
            st.write(f"- **Accuracy**: {result['accuracy']:.4f}")
            st.write(f"- **Precision**: {result['precision']:.4f}")
            st.write(f"- **Recall**: {result['recall']:.4f}")
            st.write(f"- **F1-Score**: {result['f1_score']:.4f}")
            st.write(f"- **ROC-AUC**: {result['roc_auc']:.4f}")
        
        with col2:
            st.markdown("**Confusion Matrix:**")
            cm = result['confusion_matrix']
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Normal', 'Anomaly'],
                y=['Normal', 'Anomaly'],
                text=cm,
                texttemplate="%{text}",
                colorscale='Blues'
            ))
            
            fig.update_layout(
                title=f"{selected_model} - Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

def file_upload_page():
    """File Upload and Custom Dataset Page"""
    st.header("📁 Custom Dataset Upload")
    
    st.markdown("""
    Upload your own dataset for anomaly detection. The dataset should be in CSV format with:
    - Numerical features only
    - Optional: A 'label' column with 0 (normal) and 1 (anomaly)
    - If no label column is provided, the system will treat all data as normal for unsupervised training
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📊 Dataset Preview")
            st.write(f"Shape: {df.shape}")
            st.dataframe(df.head(10))
            
            # Check for label column
            has_labels = 'label' in df.columns.str.lower()
            
            if has_labels:
                label_col = [col for col in df.columns if col.lower() == 'label'][0]
                y = df[label_col].values
                X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
                
                st.success(f"✅ Found label column: {label_col}")
                st.write(f"- Normal samples: {np.sum(y == 0)}")
                st.write(f"- Anomalous samples: {np.sum(y == 1)}")
            else:
                X = df.select_dtypes(include=[np.number]).values
                y = np.zeros(len(X))  # Treat all as normal
                
                st.info("ℹ️ No label column found. Treating all data as normal for unsupervised training.")
            
            # Store in session state
            if st.button("📥 Load Dataset"):
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.data_generated = True
                st.session_state.models_trained = False
                
                st.success("Dataset loaded successfully!")
                
                # Show basic statistics
                st.subheader("📈 Dataset Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Samples", X.shape[0])
                
                with col2:
                    st.metric("Features", X.shape[1])
                
                with col3:
                    st.metric("Anomalies", int(np.sum(y)))
                
                with col4:
                    st.metric("Anomaly Rate", f"{np.mean(y)*100:.1f}%")
                
                # Visualize if possible
                if X.shape[1] >= 2:
                    fig = create_plotly_scatter(X, y, "Uploaded Dataset")
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Sample dataset download
    st.subheader("📥 Sample Datasets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download Gaussian Sample"):
            generator = AnomalyDataGenerator()
            X, y = generator.generate_gaussian_data(n_samples=500, contamination=0.1)
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            df['label'] = y
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name="gaussian_sample.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Download Multimodal Sample"):
            generator = AnomalyDataGenerator()
            X, y = generator.generate_multimodal_data(n_samples=500, contamination=0.1)
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            df['label'] = y
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name="multimodal_sample.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("Download Time Series Sample"):
            generator = AnomalyDataGenerator()
            X, y = generator.generate_time_series_data(n_samples=500, contamination=0.1)
            df = pd.DataFrame(X, columns=['value'])
            df['label'] = y
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name="timeseries_sample.csv",
                mime="text/csv"
            )

def main():
    """Main Streamlit app"""
    initialize_session_state()
    
    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    
    pages = {
        "📊 Data Generation": data_generation_page,
        "🤖 Model Training": model_training_page,
        "📈 Results & Evaluation": results_page,
        "📁 Upload Dataset": file_upload_page
    }
    
    selected_page = st.sidebar.radio("Select Page", list(pages.keys()))
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Quick Guide")
    st.sidebar.markdown("""
    1. **Generate** or upload data
    2. **Train** multiple models
    3. **Evaluate** and compare results
    4. **Download** results and models
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This tool implements multiple anomaly detection algorithms:
    - **Isolation Forest**: Fast, tree-based
    - **One-Class SVM**: Kernel-based
    - **Autoencoder**: Neural network
    - **Ensemble**: Combined approach
    """)
    
    # Run selected page
    pages[selected_page]()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🔍 Anomaly Detection System | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()