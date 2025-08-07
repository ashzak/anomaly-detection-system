#!/usr/bin/env python3
"""
Comprehensive Anomaly Detection System
=====================================

This script demonstrates multiple anomaly detection algorithms:
- Isolation Forest
- One-Class SVM  
- Autoencoder Neural Network
- Ensemble Methods

Usage:
    python anomaly_detection_main.py [--dataset gaussian|multimodal|timeseries] [--contamination 0.1]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_utils import (
    AnomalyDataGenerator, DataPreprocessor, 
    load_sample_dataset, visualize_data
)
from anomaly_models import (
    IsolationForestDetector, OneClassSVMDetector, 
    AutoencoderDetector, AnomalyDetectionEnsemble
)
from evaluation import AnomalyDetectionEvaluator, save_results

def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection System')
    parser.add_argument('--dataset', choices=['gaussian', 'multimodal', 'timeseries'], 
                       default='gaussian', help='Dataset type to generate')
    parser.add_argument('--contamination', type=float, default=0.1,
                       help='Contamination rate (fraction of anomalies)')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualizations')
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained models')
    
    args = parser.parse_args()
    
    print("=== Anomaly Detection System ===")
    print(f"Dataset: {args.dataset}")
    print(f"Contamination rate: {args.contamination}")
    print(f"Number of samples: {args.n_samples}")
    print("=" * 40)
    
    # Generate dataset
    print("\\n1. Generating dataset...")
    X, y = load_sample_dataset(
        dataset_type=args.dataset,
        n_samples=args.n_samples,
        contamination=args.contamination,
        random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Anomalies: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    
    # Visualize original data
    if args.visualize:
        visualize_data(X, y, f"Original {args.dataset.title()} Dataset")
    
    # Preprocess data
    print("\\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.fit_transform(X)
    
    # Split into train/test (use only normal data for training unsupervised models)
    normal_indices = np.where(y == 0)[0]
    anomaly_indices = np.where(y == 1)[0]
    
    # Use 80% of normal data for training
    n_train_normal = int(0.8 * len(normal_indices))
    train_normal_idx = normal_indices[:n_train_normal]
    test_normal_idx = normal_indices[n_train_normal:]
    
    # Training set (only normal data)
    X_train = X_scaled[train_normal_idx]
    
    # Test set (remaining normal + all anomalies)
    test_indices = np.concatenate([test_normal_idx, anomaly_indices])
    X_test = X_scaled[test_indices]
    y_test = y[test_indices]
    
    print(f"Training samples: {len(X_train)} (all normal)")
    print(f"Test samples: {len(X_test)} ({np.sum(y_test)} anomalies)")
    
    # Initialize models
    print("\\n3. Initializing models...")
    
    models = {
        'Isolation Forest': IsolationForestDetector(
            contamination=args.contamination,
            n_estimators=100,
            random_state=42
        ),
        'One-Class SVM': OneClassSVMDetector(
            kernel='rbf',
            nu=args.contamination,
            gamma='scale'
        ),
        'Autoencoder': AutoencoderDetector(
            encoding_dim=max(2, X.shape[1] // 2),
            hidden_layers=[32, 16] if X.shape[1] > 10 else [8, 4],
            epochs=50,  # Reduced for demo
            batch_size=32
        )
    }
    
    # Train models and evaluate
    print("\\n4. Training and evaluating models...")
    results = {}
    model_scores = {}
    
    evaluator = AnomalyDetectionEvaluator()
    
    for model_name, model in models.items():
        print(f"\\n--- {model_name} ---")
        
        # Train
        print("Training...")
        model.fit(X_train)
        
        # Predict on test set
        print("Predicting...")
        y_pred = model.predict(X_test)
        y_scores = model.decision_function(X_test)
        
        # Evaluate
        result = evaluator.evaluate_model(y_test, y_pred, y_scores, model_name)
        results[model_name] = result
        model_scores[model_name] = y_scores
        
        print(f"Accuracy: {result['accuracy']:.3f}")
        print(f"Precision: {result['precision']:.3f}")
        print(f"Recall: {result['recall']:.3f}")
        print(f"F1-Score: {result['f1_score']:.3f}")
        print(f"ROC-AUC: {result['roc_auc']:.3f}")
        
        # Save model if requested
        if args.save_models:
            model.save_model(f"{model_name.lower().replace('-', '_').replace(' ', '_')}_model")
            print(f"Model saved!")
    
    # Ensemble method
    print("\\n--- Ensemble Model ---")
    ensemble_models = {
        'isolation_forest': models['Isolation Forest'],
        'one_class_svm': models['One-Class SVM']
    }
    
    ensemble = AnomalyDetectionEnsemble(ensemble_models)
    ensemble.fit(X_train)  # Already trained individual models
    
    # Ensemble predictions
    y_pred_ensemble = ensemble.predict(X_test, voting='majority')
    
    # For ensemble scoring, use average of individual model scores
    y_scores_ensemble = np.mean([
        models['Isolation Forest'].decision_function(X_test),
        models['One-Class SVM'].decision_function(X_test)
    ], axis=0)
    
    result_ensemble = evaluator.evaluate_model(
        y_test, y_pred_ensemble, y_scores_ensemble, 'Ensemble'
    )
    results['Ensemble'] = result_ensemble
    model_scores['Ensemble'] = y_scores_ensemble
    
    print(f"Accuracy: {result_ensemble['accuracy']:.3f}")
    print(f"Precision: {result_ensemble['precision']:.3f}")
    print(f"Recall: {result_ensemble['recall']:.3f}")
    print(f"F1-Score: {result_ensemble['f1_score']:.3f}")
    print(f"ROC-AUC: {result_ensemble['roc_auc']:.3f}")
    
    # Compare models
    print("\\n5. Model Comparison:")
    print("=" * 80)
    comparison_df = evaluator.compare_models(results)
    print(comparison_df.to_string(index=False, float_format='{:.4f}'.format))
    
    # Find best model
    best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
    best_f1 = comparison_df['F1-Score'].max()
    print(f"\\nBest performing model: {best_model} (F1-Score: {best_f1:.4f})")
    
    # Visualizations
    if args.visualize:
        print("\\n6. Generating visualizations...")
        
        # Confusion matrices
        evaluator.plot_confusion_matrices(results)
        
        # ROC curves
        evaluator.plot_roc_curves(y_test, model_scores)
        
        # Precision-Recall curves
        evaluator.plot_precision_recall_curves(y_test, model_scores)
        
        # Score distributions
        evaluator.plot_score_distributions(y_test, model_scores)
    
    # Save results
    print("\\n7. Saving results...")
    save_results(results, f"anomaly_detection_results_{args.dataset}.txt")
    
    # Generate summary report
    print("\\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    report = evaluator.generate_report(results)
    print(report)
    
    print("\\n=== Analysis Complete ===")
    
    return results, models, X_test, y_test

def run_demo():
    """Run a quick demo with different datasets"""
    print("Running Anomaly Detection Demo on Multiple Datasets...")
    
    datasets = ['gaussian', 'multimodal', 'timeseries']
    
    for dataset in datasets:
        print(f"\\n{'='*60}")
        print(f"TESTING ON {dataset.upper()} DATASET")
        print(f"{'='*60}")
        
        # Set args manually for demo
        class Args:
            dataset = dataset
            contamination = 0.1
            n_samples = 500  # Smaller for demo
            visualize = False
            save_models = False
        
        args = Args()
        
        try:
            results, models, X_test, y_test = main_with_args(args)
            
            # Print best model for this dataset
            evaluator = AnomalyDetectionEvaluator()
            comparison_df = evaluator.compare_models(results)
            best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
            best_f1 = comparison_df['F1-Score'].max()
            
            print(f"\\n🏆 Best model for {dataset}: {best_model} (F1: {best_f1:.3f})")
            
        except Exception as e:
            print(f"Error with {dataset} dataset: {e}")

def main_with_args(args):
    """Main function that accepts args object instead of parsing"""
    # Same logic as main() but with args object instead of parser
    
    print("=== Anomaly Detection System ===")
    print(f"Dataset: {args.dataset}")
    print(f"Contamination rate: {args.contamination}")
    print(f"Number of samples: {args.n_samples}")
    
    # Generate dataset
    X, y = load_sample_dataset(
        dataset_type=args.dataset,
        n_samples=args.n_samples,
        contamination=args.contamination,
        random_state=42
    )
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.fit_transform(X)
    
    # Train/test split
    normal_indices = np.where(y == 0)[0]
    anomaly_indices = np.where(y == 1)[0]
    
    n_train_normal = int(0.8 * len(normal_indices))
    train_normal_idx = normal_indices[:n_train_normal]
    test_normal_idx = normal_indices[n_train_normal:]
    
    X_train = X_scaled[train_normal_idx]
    test_indices = np.concatenate([test_normal_idx, anomaly_indices])
    X_test = X_scaled[test_indices]
    y_test = y[test_indices]
    
    # Initialize models
    models = {
        'Isolation Forest': IsolationForestDetector(
            contamination=args.contamination,
            n_estimators=50,  # Reduced for demo
            random_state=42
        ),
        'One-Class SVM': OneClassSVMDetector(
            kernel='rbf',
            nu=args.contamination,
            gamma='scale'
        ),
        'Autoencoder': AutoencoderDetector(
            encoding_dim=max(2, X.shape[1] // 2),
            hidden_layers=[16, 8] if X.shape[1] > 5 else [4, 2],
            epochs=20,  # Reduced for demo
            batch_size=32
        )
    }
    
    # Train and evaluate
    results = {}
    evaluator = AnomalyDetectionEvaluator()
    
    for model_name, model in models.items():
        model.fit(X_train)
        y_pred = model.predict(X_test)
        y_scores = model.decision_function(X_test)
        result = evaluator.evaluate_model(y_test, y_pred, y_scores, model_name)
        results[model_name] = result
    
    return results, models, X_test, y_test

if __name__ == "__main__":
    # Check if running as demo
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Running demo mode...")
        run_demo()
    else:
        main()