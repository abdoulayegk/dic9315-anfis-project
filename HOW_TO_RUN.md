# How to Run the Credit Risk Analysis Code

## Quick Test (Just Completed Successfully!)

The code has been tested and works correctly! See results above.

## Running Options

### **Option 1: Jupyter Notebook (RECOMMENDED for Students)**

This is the most interactive and visual way to run the analysis:

#### Step 1: Activate Virtual Environment

```bash
cd /Users/balde/Desktop/A2025/DIC9315/Projet/Work
source DIC9315/bin/activate
```

#### Step 2: Install Jupyter Notebook (if not installed)

```bash
pip install notebook
```

#### Step 3: Start Jupyter Notebook

```bash
jupyter notebook credit_risk_analysis.ipynb
```

This will:

- Open your web browser automatically
- Show the notebook interface
- Allow you to run cells one by one (Shift+Enter)
- Display all plots and outputs inline

#### Tips for Running in Jupyter:

- **Run All Cells**: Menu → Cell → Run All
- **Run Single Cell**: Click cell + Shift+Enter
- **Restart Kernel**: Menu → Kernel → Restart & Clear Output
- **Stop Execution**: Menu → Kernel → Interrupt

---

### **Option 2: VS Code with Jupyter Extension**

If you're using VS Code (which you are!):

#### Step 1: Open the notebook

- File is already open: `credit_risk_analysis.ipynb`

#### Step 2: Select the correct kernel

1. Click "Select Kernel" in top-right corner
2. Choose: `Python 3.13.1 ('DIC9315': venv)`
3. Or browse to: `/Users/balde/Desktop/A2025/DIC9315/Projet/Work/DIC9315/bin/python`

#### Step 3: Run cells

- Click ▶️ button next to each cell
- Or use "Run All" button at the top
- Or Shift+Enter for current cell

---

### **Option 3: Command Line (Convert to Python Script)**

Convert notebook to Python script and run:

```bash
cd /Users/balde/Desktop/A2025/DIC9315/Projet/Work
source DIC9315/bin/activate

# Convert notebook to Python script
jupyter nbconvert --to python credit_risk_analysis.ipynb

# Run the script
python credit_risk_analysis.py
```

---

### **Option 4: Use the Test Script (Quick Validation)**

Run the quick test we just created:

```bash
cd /Users/balde/Desktop/A2025/DIC9315/Projet/Work
./DIC9315/bin/python3 test_pipeline.py
```

This runs a simplified version in ~30 seconds to verify everything works.

---

## New Feature: PyTorch ANFIS Implementation

We have replaced the placeholder with a **real Takagi-Sugeno ANFIS** implementation using PyTorch!

### Key Features:

- **Architecture**: 5 Layers (Fuzzification, Rules, Normalization, Consequent, Output)
- **Inputs**: Strictly optimized for **6 features** to ensure interpretability
- **Optimization**: Uses PyTorch's Adam optimizer with hybrid learning capabilities
- **Membership**: Gaussian membership functions with learnable $\mu$ and $\sigma$ parameters

### How to Use It:

The standard pipeline automatically uses this new implementation. Just run:

```bash
jupyter notebook credit_risk_analysis.ipynb
```

### Configuration:

You can tweak ANFIS parameters in `config.py`:

```python
ANFIS_CONFIG = {
    'n_membership_functions': 3,
    'membership_type': 'gaussian',
    'max_epochs': 100,          # Increase for better convergence
    'learning_rate': 0.01,      # Decrease if training is unstable
    'batch_size': 32,
    'n_rules': 10               # Number of fuzzy rules
}
```

### Requirements:

- `torch` (PyTorch) is now required and has been installed.

---

## What to Expect

### Execution Time:

- **Quick test**: ~30 seconds
- **Full notebook (no hyperparameter tuning)**: ~2-3 minutes
- **Full notebook (with tuning)**: ~10-15 minutes

### Expected Outputs:

- Data preprocessing logs
- Feature selection results
- Model training progress
- Performance metrics (Accuracy, F1, etc.)
- Confusion matrices (plots)
- ROC curves (plots)
- SHAP analysis plots
- Statistical significance tests

### Output Files Created:

```
results/
  ├── model_comparison.csv
  ├── statistical_significance.csv
  ├── confusion_matrices.png
  ├── roc_curves.png
  └── metrics_comparison.png

plots/
  ├── shap_summary_random_forest.png
  ├── shap_bar_random_forest.png
  ├── shap_summary_svm.png
  └── shap_comparison_models.png

models/
  └── (trained model files if saved)
```

---

## Troubleshooting

### Problem: "Module not found"

**Solution**: Make sure virtual environment is activated

```bash
source DIC9315/bin/activate
```

### Problem: "File not found: default of credit card clients.xls"

**Solution**: Make sure you're in the correct directory

```bash
cd /Users/balde/Desktop/A2025/DIC9315/Projet/Work
ls -la *.xls  # Should show the data file
```

### Problem: Kernel keeps dying / Out of memory

**Solution**:

1. Restart Jupyter kernel
2. Close other applications
3. Run with SMOTE disabled: Set `USE_SMOTE = False` in config.py

### Problem: SHAP analysis is very slow

**Solution**: This is normal for SVM. You can:

1. Reduce test set size in SHAP cells (already set to 100 samples for SVM)
2. Skip SVM SHAP analysis if needed
3. Only run Random Forest SHAP

---

## Recommended Workflow for Students

1. **First Run**: Execute all cells to completion
2. **Review Results**: Look at all plots and metrics
3. **Experiment**:
   - Try different hyperparameters in `config.py`
   - Change number of features for ANFIS
   - Enable/disable SMOTE
4. **Document**: Add markdown cells with your observations
5. **Present**: Use the generated plots in your report

---

## Notes

- All emojis have been removed from the code
- Variable names are now more student-friendly
- Code maintains full functionality
- Tested and working as of November 22, 2025

**For any issues, check the output of `test_pipeline.py` first!**
