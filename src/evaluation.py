"""
Model evaluation metrics and statistical significance testing
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from scipy.stats import wilcoxon, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import config

# Color palette
COLORS = ["#7400ff", "#a788e4", "#d216d2", "#ffb500", "#36c9dd"]

class ModelEvaluator:
    """Evaluate and compare model performance"""
    
    def __init__(self, output_dir=config.OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
        # Set style and colors
        sns.set_style("whitegrid")
        sns.despine(left=True, bottom=True)
        sns.set_palette(sns.color_palette(COLORS))
        plt.rc("xtick", labelsize=12)
        plt.rc("ytick", labelsize=12)
        
    def evaluate_single_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model on test set"""
        print(f"\n{'=' * 80}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'=' * 80}")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probabilities (if available)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred  # Fallback
        
        # Calculate metrics
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
        }
        
        # AUC-ROC (only if probabilities available)
        try:
            metrics['auc_roc'] = roc_auc_score(y_test, y_proba)
        except:
            metrics['auc_roc'] = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Class-specific metrics (for imbalanced dataset)
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Store results
        self.results[model_name] = metrics
        
        # Print results
        print(f"\nMetrics for {model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def compare_models(self):
        """Create comparison table of all evaluated models"""
        if not self.results:
            print("No models evaluated yet")
            return None
        
        # Create comparison DataFrame
        comparison = []
        for model_name, metrics in self.results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC-ROC': metrics['auc_roc'],
                'Specificity': metrics['specificity']
            })
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.round(4)
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(df_comparison.to_string(index=False))
        
        # Save to CSV
        output_file = self.output_dir / 'model_comparison.csv'
        df_comparison.to_csv(output_file, index=False)
        print(f"\nComparison saved to: {output_file}")
        
        return df_comparison
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        if n_models == 0:
            print("No models to plot")
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[idx], 
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        output_file = self.output_dir / 'confusion_matrices.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to: {output_file}")
        plt.close()
    
    def plot_roc_curves(self, models_dict, X_test, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models_dict.items():
            if model is None:
                continue
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                continue
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        output_file = self.output_dir / 'roc_curves.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {output_file}")
        plt.close()
    
    def plot_metrics_comparison(self):
        """Bar plot comparing key metrics across models"""
        if not self.results:
            print("No models to compare")
            return
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(self.results)
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            values = [metrics[m] for m in metrics_to_plot]
            color = COLORS[idx % len(COLORS)]
            ax.bar(x + idx*width, values, width, label=model_name, color=color)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(self.results) - 1) / 2)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'metrics_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {output_file}")
        plt.close()
    
    def statistical_significance_test(self, cv_scores_dict, test='wilcoxon'):
        """
        Test statistical significance between model performances
        
        Parameters:
        -----------
        cv_scores_dict : dict
            Dictionary with model names as keys and arrays of CV scores as values
        test : str
            'wilcoxon' for Wilcoxon signed-rank test or 'ttest' for paired t-test
        """
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("=" * 80)
        
        model_names = list(cv_scores_dict.keys())
        results = []
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                scores1 = cv_scores_dict[model1]
                scores2 = cv_scores_dict[model2]
                
                # Perform test
                if test == 'wilcoxon':
                    statistic, p_value = wilcoxon(scores1, scores2)
                    test_name = "Wilcoxon"
                else:  # t-test
                    statistic, p_value = ttest_rel(scores1, scores2)
                    test_name = "Paired t-test"
                
                # Determine significance
                significant = "Yes" if p_value < 0.05 else "No"
                
                results.append({
                    'Model 1': model1,
                    'Model 2': model2,
                    'Mean Diff': np.mean(scores1) - np.mean(scores2),
                    'Test': test_name,
                    'p-value': p_value,
                    'Significant (α=0.05)': significant
                })
                
                print(f"\n{model1} vs {model2}:")
                print(f"  {test_name} p-value: {p_value:.6f}")
                print(f"  Statistically significant: {significant}")
                print(f"  Mean difference: {np.mean(scores1) - np.mean(scores2):.6f}")
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / 'statistical_significance.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\n\nStatistical test results saved to: {output_file}")
        
        return df_results


if __name__ == "__main__":
    # Example usage
    pass
