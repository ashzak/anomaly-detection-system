import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, average_precision_score
)
import pandas as pd
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class AnomalyDetectionEvaluator:
    """Comprehensive evaluation suite for anomaly detection models"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_scores: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Evaluate a single anomaly detection model"""
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except:
            roc_auc = 0.5
        
        # Average Precision
        try:
            avg_precision = average_precision_score(y_true, y_scores)
        except:
            avg_precision = 0.0
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        self.results[model_name] = results
        return results
    
    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models and return results dataframe"""
        comparison_data = []
        
        for model_name, results in models_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'Avg Precision': results['avg_precision']
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_confusion_matrices(self, models_results: Dict[str, Dict]):
        """Plot confusion matrices for all models"""
        n_models = len(models_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(models_results.items()):
            ax = axes[idx] if n_models > 1 else axes[0]
            
            sns.heatmap(results['confusion_matrix'], 
                       annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'],
                       ax=ax)
            ax.set_title(f'{model_name}\nConfusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, y_true: np.ndarray, models_scores: Dict[str, np.ndarray]):
        """Plot ROC curves for multiple models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, y_scores in models_scores.items():
            try:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc = roc_auc_score(y_true, y_scores)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
            except:
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, models_scores: Dict[str, np.ndarray]):
        """Plot Precision-Recall curves for multiple models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, y_scores in models_scores.items():
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                avg_precision = average_precision_score(y_true, y_scores)
                plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})', linewidth=2)
            except:
                continue
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_score_distributions(self, y_true: np.ndarray, models_scores: Dict[str, np.ndarray]):
        """Plot score distributions for normal vs anomalous samples"""
        n_models = len(models_scores)
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_models > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, scores) in enumerate(models_scores.items()):
            ax = axes[idx] if n_models > 1 else axes[0]
            
            normal_scores = scores[y_true == 0]
            anomaly_scores = scores[y_true == 1]
            
            ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True, color='blue')
            ax.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', density=True, color='red')
            
            ax.set_title(f'{model_name}\nScore Distribution')
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_comparison(self, comparison_df: pd.DataFrame):
        """Create interactive plotly comparison of models"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Avg Precision']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metrics,
            specs=[[{"secondary_y": False}]*3, [{"secondary_y": False}]*3]
        )
        
        colors = px.colors.qualitative.Set3[:len(comparison_df)]
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
        
        for idx, metric in enumerate(metrics):
            row, col = positions[idx]
            
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
                row=row, col=col
            )
            
            fig.update_yaxes(range=[0, 1], row=row, col=col)
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            showlegend=False,
            height=600
        )
        
        return fig
    
    def generate_report(self, models_results: Dict[str, Dict]) -> str:
        """Generate a comprehensive text report"""
        report = "=== ANOMALY DETECTION MODEL EVALUATION REPORT ===\n\n"
        
        # Summary table
        comparison_df = self.compare_models(models_results)
        report += "Model Performance Summary:\n"
        report += "=" * 80 + "\n"
        report += comparison_df.to_string(index=False, float_format='{:.4f}'.format)
        report += "\n\n"
        
        # Best performing model for each metric
        report += "Best Performing Models by Metric:\n"
        report += "=" * 40 + "\n"
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Avg Precision']:
            best_model = comparison_df.loc[comparison_df[metric].idxmax(), 'Model']
            best_score = comparison_df[metric].max()
            report += f"{metric:15}: {best_model:20} ({best_score:.4f})\n"
        
        report += "\n"
        
        # Detailed results for each model
        for model_name, results in models_results.items():
            report += f"\nDetailed Results for {model_name}:\n"
            report += "-" * 50 + "\n"
            report += f"Accuracy:        {results['accuracy']:.4f}\n"
            report += f"Precision:       {results['precision']:.4f}\n"
            report += f"Recall:          {results['recall']:.4f}\n"
            report += f"F1-Score:        {results['f1_score']:.4f}\n"
            report += f"ROC-AUC:         {results['roc_auc']:.4f}\n"
            report += f"Avg Precision:   {results['avg_precision']:.4f}\n"
            report += f"\nConfusion Matrix:\n"
            report += f"True Negatives:  {results['true_negatives']}\n"
            report += f"False Positives: {results['false_positives']}\n"
            report += f"False Negatives: {results['false_negatives']}\n"
            report += f"True Positives:  {results['true_positives']}\n"
        
        return report

def save_results(results: Dict, filename: str = "anomaly_detection_results.txt"):
    """Save evaluation results to file"""
    evaluator = AnomalyDetectionEvaluator()
    report = evaluator.generate_report(results)
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"Results saved to {filename}")