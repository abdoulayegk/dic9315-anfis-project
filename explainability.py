"""
SHAP-based explainability analysis for credit risk models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
import config


class SHAPExplainer:
    """SHAP analysis for model explainability"""

    def __init__(self, output_dir=config.PLOTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.explainers = {}
        self.shap_values = {}

        # Set style
        sns.set_style("whitegrid")
        sns.despine(left=True, bottom=True)
        plt.rc("xtick", labelsize=12)
        plt.rc("ytick", labelsize=12)

    def create_explainer(self, model, X_train, model_name, model_type="tree"):
        """
        Create SHAP explainer for a given model

        Parameters:
        -----------
        model : trained model
            The model to explain
        X_train : array-like
            Training data for background samples
        model_name : str
            Name of the model
        model_type : str
            Type of explainer: 'tree', 'kernel', or 'linear'
        """
        print(f"\n{'='*80}")
        print(f"Creating SHAP explainer for {model_name}")
        print(f"{'='*80}")

        # Convert to DataFrame if needed
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)

        try:
            if model_type == "tree":
                # For tree-based models (Random Forest)
                explainer = shap.TreeExplainer(model)
            elif model_type == "kernel":
                # For any model (SVM, ANFIS) - slower but universal
                # Use a sample of background data for efficiency
                background = shap.sample(X_train, min(100, len(X_train)))
                explainer = shap.KernelExplainer(model.predict_proba, background)
            elif model_type == "linear":
                # For linear models
                explainer = shap.LinearExplainer(model, X_train)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.explainers[model_name] = explainer
            print(f"SHAP explainer created for {model_name}")

            return explainer

        except Exception as e:
            print(f"Error creating explainer for {model_name}: {e}")
            return None

    def calculate_shap_values(self, model_name, X_test):
        """Calculate SHAP values for test set"""
        print(f"\nCalculating SHAP values for {model_name}...")

        if model_name not in self.explainers:
            print(f"No explainer found for {model_name}")
            return None

        explainer = self.explainers[model_name]

        # Convert to DataFrame if needed
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)

        try:
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test)

            # For binary classification, some explainers return values for both classes
            if isinstance(shap_values, list):
                # Take the positive class (index 1)
                shap_values = shap_values[1]

            self.shap_values[model_name] = {
                "values": shap_values,
                "data": X_test,
                "feature_names": (
                    X_test.columns.tolist() if hasattr(X_test, "columns") else None
                ),
            }

            print(f"SHAP values calculated: {shap_values.shape}")
            return shap_values

        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return None

    def plot_summary(self, model_name, max_display=20):
        """
        Create SHAP summary plot showing global feature importance

        Parameters:
        -----------
        model_name : str
            Name of the model
        max_display : int
            Maximum number of features to display
        """
        if model_name not in self.shap_values:
            print(f"No SHAP values found for {model_name}")
            return

        print(f"\nGenerating SHAP summary plot for {model_name}...")

        shap_data = self.shap_values[model_name]

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_data["values"],
            shap_data["data"],
            max_display=max_display,
            show=False,
            color_bar_label="Feature Value",
        )
        plt.title(
            f"SHAP Summary Plot - {model_name}", fontsize=14, fontweight="bold", pad=20
        )
        plt.tight_layout()

        output_file = (
            self.output_dir / f'shap_summary_{model_name.lower().replace(" ", "_")}.png'
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Summary plot saved to: {output_file}")
        plt.close()

    def plot_bar(self, model_name, max_display=20):
        """
        Create SHAP bar plot showing mean absolute feature importance

        Parameters:
        -----------
        model_name : str
            Name of the model
        max_display : int
            Maximum number of features to display
        """
        if model_name not in self.shap_values:
            print(f"No SHAP values found for {model_name}")
            return

        print(f"\nGenerating SHAP bar plot for {model_name}...")

        shap_data = self.shap_values[model_name]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_data["values"],
            shap_data["data"],
            plot_type="bar",
            max_display=max_display,
            show=False,
            color=config.COLORS[0],
        )
        plt.title(
            f"SHAP Feature Importance - {model_name}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()

        output_file = (
            self.output_dir / f'shap_bar_{model_name.lower().replace(" ", "_")}.png'
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Bar plot saved to: {output_file}")
        plt.close()

    def plot_force(self, model_name, instance_idx=0):
        """
        Create SHAP force plot for individual prediction

        Parameters:
        -----------
        model_name : str
            Name of the model
        instance_idx : int
            Index of the instance to explain
        """
        if model_name not in self.shap_values:
            print(f"No SHAP values found for {model_name}")
            return

        print(
            f"\nGenerating SHAP force plot for {model_name} (instance {instance_idx})..."
        )

        shap_data = self.shap_values[model_name]
        explainer = self.explainers[model_name]

        # Get expected value
        if hasattr(explainer, "expected_value"):
            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = (
                    expected_value[1] if len(expected_value) > 1 else expected_value[0]
                )
        else:
            expected_value = shap_data["values"].mean()

        # Create force plot
        shap.force_plot(
            expected_value,
            shap_data["values"][instance_idx],
            shap_data["data"].iloc[instance_idx],
            matplotlib=True,
            show=False,
        )

        output_file = (
            self.output_dir
            / f'shap_force_{model_name.lower().replace(" ", "_")}_inst{instance_idx}.png'
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Force plot saved to: {output_file}")
        plt.close()

    def plot_waterfall(self, model_name, instance_idx=0, max_display=20):
        """
        Create SHAP waterfall plot for individual prediction

        Parameters:
        -----------
        model_name : str
            Name of the model
        instance_idx : int
            Index of the instance to explain
        max_display : int
            Maximum number of features to display
        """
        if model_name not in self.shap_values:
            print(f"No SHAP values found for {model_name}")
            return

        print(
            f"\nGenerating SHAP waterfall plot for {model_name} (instance {instance_idx})..."
        )

        shap_data = self.shap_values[model_name]
        explainer = self.explainers[model_name]

        # Get expected value
        if hasattr(explainer, "expected_value"):
            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = (
                    expected_value[1] if len(expected_value) > 1 else expected_value[0]
                )
        else:
            expected_value = shap_data["values"].mean()

        # Create Explanation object
        explanation = shap.Explanation(
            values=shap_data["values"][instance_idx],
            base_values=expected_value,
            data=shap_data["data"].iloc[instance_idx].values,
            feature_names=shap_data["feature_names"],
        )

        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, max_display=max_display, show=False)
        plt.title(
            f"SHAP Waterfall Plot - {model_name} (Instance {instance_idx})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()

        output_file = (
            self.output_dir
            / f'shap_waterfall_{model_name.lower().replace(" ", "_")}_inst{instance_idx}.png'
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Waterfall plot saved to: {output_file}")
        plt.close()

    def plot_dependence(self, model_name, feature_name, interaction_feature="auto"):
        """
        Create SHAP dependence plot showing feature interactions

        Parameters:
        -----------
        model_name : str
            Name of the model
        feature_name : str
            Main feature to plot
        interaction_feature : str or 'auto'
            Feature to color by (auto-detected if 'auto')
        """
        if model_name not in self.shap_values:
            print(f"No SHAP values found for {model_name}")
            return

        print(f"\nGenerating SHAP dependence plot for {model_name} - {feature_name}...")

        shap_data = self.shap_values[model_name]

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            shap_data["values"],
            shap_data["data"],
            interaction_index=interaction_feature,
            show=False,
            dot_size=30,
            alpha=0.6,
        )
        plt.title(
            f"SHAP Dependence Plot - {model_name}\n{feature_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        output_file = (
            self.output_dir
            / f'shap_dependence_{model_name.lower().replace(" ", "_")}_{feature_name}.png'
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Dependence plot saved to: {output_file}")
        plt.close()

    def get_feature_importance_df(self, model_name):
        """
        Get feature importance as DataFrame

        Returns:
        --------
        pd.DataFrame with features and their mean absolute SHAP values
        """
        if model_name not in self.shap_values:
            print(f"No SHAP values found for {model_name}")
            return None

        shap_data = self.shap_values[model_name]

        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_data["values"]).mean(axis=0)

        # Create DataFrame
        importance_df = pd.DataFrame(
            {"Feature": shap_data["feature_names"], "Mean_Absolute_SHAP": mean_shap}
        )

        # Sort by importance
        importance_df = importance_df.sort_values("Mean_Absolute_SHAP", ascending=False)

        return importance_df

    def compare_models_importance(self, model_names, top_n=15):
        """
        Compare feature importance across multiple models

        Parameters:
        -----------
        model_names : list
            List of model names to compare
        top_n : int
            Number of top features to display
        """
        print(f"\n{'='*80}")
        print("COMPARING SHAP FEATURE IMPORTANCE ACROSS MODELS")
        print(f"{'='*80}")

        # Collect importance for each model
        all_importance = {}
        for model_name in model_names:
            if model_name in self.shap_values:
                df = self.get_feature_importance_df(model_name)
                if df is not None:
                    all_importance[model_name] = df

        if not all_importance:
            print("No SHAP values available for comparison")
            return

        # Get union of top features from all models
        all_features = set()
        for df in all_importance.values():
            all_features.update(df.head(top_n)["Feature"].tolist())

        # Create comparison DataFrame
        comparison_data = []
        for feature in all_features:
            row = {"Feature": feature}
            for model_name, df in all_importance.items():
                feat_importance = df[df["Feature"] == feature][
                    "Mean_Absolute_SHAP"
                ].values
                row[model_name] = feat_importance[0] if len(feat_importance) > 0 else 0
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by mean importance across models
        model_cols = [col for col in comparison_df.columns if col != "Feature"]
        comparison_df["Mean_Importance"] = comparison_df[model_cols].mean(axis=1)
        comparison_df = comparison_df.sort_values(
            "Mean_Importance", ascending=False
        ).head(top_n)

        # Plot comparison
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(comparison_df))
        width = 0.8 / len(model_cols)

        for idx, model_name in enumerate(model_cols):
            color = config.COLORS[idx % len(config.COLORS)]
            ax.bar(
                x + idx * width,
                comparison_df[model_name],
                width,
                label=model_name,
                color=color,
                alpha=0.8,
            )

        ax.set_xlabel("Features", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Absolute SHAP Value", fontsize=12, fontweight="bold")
        ax.set_title(
            f"SHAP Feature Importance Comparison (Top {top_n})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x + width * (len(model_cols) - 1) / 2)
        ax.set_xticklabels(comparison_df["Feature"], rotation=45, ha="right")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "shap_comparison_models.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\nModel comparison plot saved to: {output_file}")
        plt.close()

        # Save comparison table
        output_csv = (
            self.output_dir.parent
            / "results"
            / "shap_feature_importance_comparison.csv"
        )
        output_csv.parent.mkdir(exist_ok=True)
        comparison_df.drop("Mean_Importance", axis=1).to_csv(output_csv, index=False)
        print(f"Comparison table saved to: {output_csv}")

        return comparison_df


if __name__ == "__main__":
    # Example usage
    pass
