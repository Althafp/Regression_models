import os
import io
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Path to your Excel file and persistence files
EXCEL_PATH = 'ilovepdf_merged_all_resistance_data.xlsx'
METRICS_FILE = 'metrics.json'
PARAMS_FILE = 'params.json'

# Updated features
FEATURE_COLS = [
    'Length BP [m]', 'Breadth [m]', 'Avg Draft [m]',
    'Displacement [m^3]', 'Wetted Surface Area [m^2]', 'Block Coefficient', 'Vs'
]
TARGET_COL = 'Rts'

# Model descriptions with importance
MODEL_DESCRIPTIONS = {
    "Linear Regression": {
        "description": "Fits a linear equation to predict ship resistance based on input features. It minimizes the sum of squared errors, assuming a linear relationship.",
        "importance": "Simple and interpretable, ideal for initial analysis of ship resistance when relationships are approximately linear. Less effective for complex, non-linear patterns.",
        "parameters": [],
        "overfit_risk": 0.2,
        "plot_types": ["Predicted vs Actual"],
        "impact_on_metrics": "No tunable parameters, so R2, MSE, and MAE depend solely on data linearity. Limited by linear assumptions, struggles with non-linear relationships (e.g., drag vs. speed).",
        "when_to_adjust": "No Parameters: If R2 is low (<0.9) or MSE/MAE are high, switch to non-linear models (e.g., Polynomial Regression, Random Forest). Use When: Data relationships are approximately linear (check Predicted vs Actual plot for linear trend). Avoid When: Residual Plot shows systematic patterns, indicating non-linearities.",
        "good_fit": "R2 > 0.9, MSE < 1000, MAE < 20 (assuming Rts range 0-1000 kN). Predicted vs Actual plot shows points near diagonal. Residual Plot shows random scatter.",
        "strategy": "Use as a baseline. If metrics are poor, try models with non-linear capabilities."
    },
    "Polynomial Regression": {
        "description": "Extends linear regression by adding polynomial features to capture non-linear relationships. Higher degrees increase model complexity.",
        "importance": "Useful for modeling non-linear effects in ship resistance, such as drag variations with speed. Requires careful degree selection to avoid overfitting.",
        "parameters": [{"name": "degree", "type": "number", "default": 2, "min": 1, "max": 4, "step": 1}],
        "overfit_risk": 0.8,
        "plot_types": ["Predicted vs Actual"],
        "impact_on_metrics": "Increasing degree: Adds higher-order features, increasing model complexity. Raises R2 and lowers MSE/MAE by capturing non-linearities, but risks overfitting. Decreasing degree: Simplifies model, reducing overfitting but may lower R2 and increase MSE/MAE if underfitting.",
        "when_to_adjust": "Increase degree (e.g., from 2 to 3): R2 < 0.9 or MSE/MAE are high (e.g., MSE > 1000). Predicted vs Actual plot shows systematic deviations from diagonal. Residual Plot shows curved patterns. Decrease degree (e.g., from 4 to 3): R2 ≈ 1.0 and MSE/MAE are very low (e.g., MSE < 10), indicating possible overfitting. Residual Plot shows patterns or overfitting risk (0.8) is high. Use When: Data has moderate non-linearities (e.g., quadratic or cubic trends in Vs vs. Rts). Avoid When: Dataset is small, as high degree leads to overfitting.",
        "good_fit": "R2 > 0.95, MSE < 100, MAE < 10. Residual Plot shows random scatter. Predicted vs Actual plot aligns closely with diagonal.",
        "strategy": "Start with degree=2. Increase to 3 if R2 is low. Reduce to 2 if overfitting is suspected. Monitor overfitting with Residual Plot."
    },
    "Random Forest Regression": {
        "description": "An ensemble of decision trees that averages predictions to reduce variance. Handles non-linear data and feature interactions well.",
        "importance": "Robust for ship resistance prediction due to its ability to handle complex relationships and reduce overfitting through averaging. Good for noisy maritime data.",
        "parameters": [
            {"name": "n_estimators", "type": "number", "default": 100, "min": 10, "max": 300, "step": 10},
            {"name": "max_depth", "type": "number", "default": 10, "min": 1, "max": 30, "step": 1}
        ],
        "overfit_risk": 0.3,
        "plot_types": ["Feature Importance", "Training Error vs Trees"],
        "impact_on_metrics": "Increasing n_estimators: More trees improve stability, slightly raising R2 and lowering MSE/MAE, but with diminishing returns. Decreasing n_estimators: Reduces computation but may lower R2 and increase MSE/MAE due to less averaging. Increasing max_depth: Deeper trees raise R2 and lower MSE/MAE by capturing complex patterns, but risk overfitting. Decreasing max_depth: Simplifies trees, reducing overfitting but may lower R2 and increase MSE/MAE.",
        "when_to_adjust": "Increase n_estimators (e.g., to 200): R2 < 0.95 or MSE > 100. Training Error vs Trees plot shows error decreasing with more trees. Decrease n_estimators (e.g., to 50): R2 ≈ 0.99 and MSE < 50, to reduce computation with minimal accuracy loss. Increase max_depth (e.g., to 15): R2 < 0.95 or MSE > 100. Feature Importance plot shows key features underutilized. Decrease max_depth (e.g., to 5): R2 ≈ 1.0 and MSE < 10, indicating overfitting. Training Error vs Trees plot shows overfitting. Use When: Data has complex, non-linear relationships and interactions. Avoid When: Interpretability is critical (use Decision Tree instead).",
        "good_fit": "R2 > 0.95, MSE < 100, MAE < 10. Feature Importance plot highlights relevant features (e.g., Vs, Displacement). Training Error vs Trees plot stabilizes.",
        "strategy": "Start with n_estimators=100, max_depth=10. Increase max_depth if R2 is low, increase n_estimators for stability. Reduce max_depth if overfitting is detected."
    },
    "Decision Tree Regression": {
        "description": "Splits data into regions based on feature thresholds to predict resistance. Simple but prone to overfitting without constraints.",
        "importance": "Useful for quick insights into feature importance in ship resistance, but less reliable for predictions due to high variance.",
        "parameters": [
            {"name": "max_depth", "type": "number", "default": 10, "min": 1, "max": 30, "step": 1},
            {"name": "min_samples_split", "type": "number", "default": 2, "min": 2, "max": 20, "step": 1}
        ],
        "overfit_risk": 0.7,
        "plot_types": ["Feature Importance"],
        "impact_on_metrics": "Increasing max_depth: Deeper trees increase R2 and lower MSE/MAE, but high overfitting risk (0.7). Decreasing max_depth: Simplifies model, reducing overfitting but may lower R2 and increase MSE/MAE. Increasing min_samples_split: Reduces overfitting by requiring more samples per split, but may lower R2 if too restrictive. Decreasing min_samples_split: Allows finer splits, raising R2 and lowering MSE/MAE, but risks overfitting.",
        "when_to_adjust": "Increase max_depth (e.g., to 15): R2 < 0.9 or MSE > 1000. Feature Importance plot shows key features underutilized. Decrease max_depth (e.g., to 5): R2 ≈ 1.0 and MSE < 10, indicating overfitting. Increase min_samples_split (e.g., to 5): R2 ≈ 1.0 and Feature Importance plot shows noise. Decrease min_samples_split (e.g., to 2): R2 < 0.9 and MSE > 1000. Use When: Quick insights into feature importance are needed. Avoid When: High variance is a concern (use Random Forest).",
        "good_fit": "R2 > 0.9, MSE < 1000, MAE < 20. Feature Importance plot shows meaningful features. Residual Plot shows random scatter.",
        "strategy": "Start with max_depth=10, min_samples_split=2. Increase max_depth if R2 is low, increase min_samples_split to reduce overfitting."
    },
    "Support Vector Regression (SVR)": {
        "description": "Fits a hyperplane within a margin of tolerance, using an RBF kernel for non-linear relationships. Requires feature scaling.",
        "importance": "Effective for small datasets with complex patterns in ship resistance. Sensitive to parameter tuning and computational cost.",
        "parameters": [
            {"name": "C", "type": "number", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
            {"name": "epsilon", "type": "number", "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}
        ],
        "overfit_risk": 0.4,
        "plot_types": ["Residual Plot"],
        "impact_on_metrics": "Increasing C: Higher penalty for errors, increasing R2 and lowering MSE/MAE, but may overfit. Decreasing C: Lower penalty, reducing overfitting but may lower R2 and increase MSE/MAE. Increasing epsilon: Wider margin, reducing sensitivity to noise, which may lower R2 but stabilize MSE/MAE. Decreasing epsilon: Narrower margin, increasing R2 and lowering MSE/MAE, but risks overfitting.",
        "when_to_adjust": "Increase C (e.g., to 10.0): R2 < 0.9 or MSE > 1000. Residual Plot shows large errors. Decrease C (e.g., to 0.5): R2 > 0.95 and MSE < 100, to prevent overfitting. Increase epsilon (e.g., to 0.5): Data is noisy, MSE/MAE fluctuate. Decrease epsilon (e.g., to 0.05): R2 < 0.9 and MSE > 1000. Use When: Dataset is small with complex patterns. Avoid When: Large datasets (computationally expensive).",
        "good_fit": "R2 > 0.9, MSE < 1000, MAE < 20. Residual Plot shows random scatter.",
        "strategy": "Start with C=1.0, epsilon=0.1. Increase C and decrease epsilon if R2 is low. Increase epsilon if noise is suspected."
    },
    "XGBoost Regression": {
        "description": "A gradient boosting method that builds trees sequentially, optimizing a loss function with regularization to prevent overfitting.",
        "importance": "Highly accurate for ship resistance prediction due to its ability to model complex interactions and handle noisy data. Widely used in maritime engineering.",
        "parameters": [
            {"name": "n_estimators", "type": "number", "default": 100, "min": 10, "max": 300, "step": 10},
            {"name": "learning_rate", "type": "number", "default": 0.1, "min": 0.01, "max": 0.3, "step": 0.01},
            {"name": "max_depth", "type": "number", "default": 6, "min": 1, "max": 15, "step": 1}
        ],
        "overfit_risk": 0.3,
        "plot_types": ["Loss Curve", "Feature Importance"],
        "impact_on_metrics": "Increasing n_estimators: More trees raise R2 and lower MSE/MAE, but with diminishing returns. Decreasing n_estimators: Reduces computation but may lower R2 and increase MSE/MAE. Increasing learning_rate: Faster learning, potentially raising R2 but risking overfitting. Decreasing learning_rate: Slower learning, reducing overfitting but may lower R2 if too low. Increasing max_depth: Deeper trees raise R2 and lower MSE/MAE, but risk overfitting. Decreasing max_depth: Simplifies model, reducing overfitting but may lower R2.",
        "when_to_adjust": "Increase n_estimators (e.g., to 200): R2 < 0.95 or MSE > 100. Decrease n_estimators (e.g., to 50): R2 ≈ 0.99 and MSE < 10, to reduce computation. Increase learning_rate (e.g., to 0.2): R2 < 0.95 and training is slow. Decrease learning_rate (e.g., to 0.03): R2 ≈ 1.0 and MSE < 10, to prevent overfitting. Increase max_depth (e.g., to 8): R2 < 0.95. Decrease max_depth (e.g., to 3): R2 ≈ 1.0 and MSE < 10. Use When: High accuracy is needed for complex data. Avoid When: Interpretability is critical.",
        "good_fit": "R2 > 0.95, MSE < 100, MAE < 10. Loss Curve stabilizes; Feature Importance highlights key features.",
        "strategy": "Start with n_estimators=100, learning_rate=0.1, max_depth=6. Adjust learning_rate and max_depth to balance fit and generalization."
    },
    "Artificial Neural Network (ANN)": {
        "description": "A multi-layer neural network that learns complex patterns through backpropagation. Requires feature scaling and extensive tuning.",
        "importance": "Powerful for capturing intricate non-linear relationships in ship resistance, but computationally intensive and less interpretable.",
        "parameters": [
            {"name": "hidden_layer_size1", "type": "number", "default": 100, "min": 10, "max": 200, "step": 10},
            {"name": "hidden_layer_size2", "type": "number", "default": 100, "min": 10, "max": 200, "step": 10}
        ],
        "overfit_risk": 0.6,
        "plot_types": ["Loss Curve"],
        "impact_on_metrics": "Increasing hidden_layer_size1/2: More neurons increase capacity, potentially raising R2 and lowering MSE/MAE, but risk overfitting or convergence issues. Decreasing hidden_layer_size1/2: Simplifies model, aiding convergence, which may raise R2 and lower MSE/MAE for small datasets.",
        "when_to_adjust": "Increase hidden_layer_size1/2 (e.g., to 150): R2 > 0.5 and MSE < 5000, to capture more complexity. Decrease hidden_layer_size1/2 (e.g., to 30, 20): R2 < 0 or MSE > 10000, to improve convergence. Loss Curve shows no improvement. Use When: Data has intricate non-linear patterns. Avoid When: Dataset is small or convergence is difficult.",
        "good_fit": "R2 > 0.9, MSE < 1000, MAE < 20. Loss Curve shows decreasing trend.",
        "strategy": "Start with hidden_layer_size1=100, hidden_layer_size2=100. Reduce sizes if convergence fails, increase if R2 improves."
    },
    "Gaussian Process Regression": {
        "description": "A probabilistic model that predicts distributions over functions, suitable for small datasets but computationally expensive.",
        "importance": "Useful for uncertainty quantification in ship resistance, especially with limited data. Less practical for large datasets.",
        "parameters": [],
        "overfit_risk": 0.4,
        "plot_types": ["Residual Plot"],
        "impact_on_metrics": "No tunable parameters. High R2 and low MSE/MAE may indicate overfitting on small datasets.",
        "when_to_adjust": "No Parameters: If R2 ≈ 1.0 and MSE ≈ 0, suspect overfitting and switch to Random Forest or XGBoost. Use When: Small dataset with need for uncertainty quantification. Avoid When: Large datasets (computationally expensive).",
        "good_fit": "R2 > 0.95, MSE < 100, MAE < 10 (on test data). Residual Plot shows random scatter.",
        "strategy": "Use as a benchmark. Switch to other models if overfitting is suspected."
    },
    "Lasso Regression": {
        "description": "Linear regression with L1 regularization, which can zero out coefficients for feature selection and prevent overfitting.",
        "importance": "Valuable for ship resistance when feature selection is needed, as it simplifies models by eliminating less important predictors.",
        "parameters": [{"name": "alpha", "type": "number", "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}],
        "overfit_risk": 0.2,
        "plot_types": ["Predicted vs Actual"],
        "impact_on_metrics": "Increasing alpha: Stronger regularization, reducing overfitting but may lower R2 and increase MSE/MAE by zeroing out features. Decreasing alpha: Weaker regularization, raising R2 and lowering MSE/MAE but risking overfitting.",
        "when_to_adjust": "Increase alpha (e.g., to 0.5): R2 > 0.9 and MSE < 1000, to simplify model. Decrease alpha (e.g., to 0.01): R2 < 0.9 or MSE > 1000. Use When: Feature selection is needed. Avoid When: Non-linear relationships dominate.",
        "good_fit": "R2 > 0.9, MSE < 1000, MAE < 20. Predicted vs Actual plot shows good alignment.",
        "strategy": "Start with alpha=0.1. Decrease if R2 is low, increase for sparsity."
    },
    "Ridge Regression": {
        "description": "Linear regression with L2 regularization, penalizing large coefficients to improve generalization.",
        "importance": "Effective for ship resistance when features are correlated, as it stabilizes predictions without eliminating features.",
        "parameters": [{"name": "alpha", "type": "number", "default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}],
        "overfit_risk": 0.2,
        "plot_types": ["Predicted vs Actual"],
        "impact_on_metrics": "Increasing alpha: Stronger regularization, reducing overfitting but may lower R2 and increase MSE/MAE. Decreasing alpha: Weaker regularization, raising R2 and lowering MSE/MAE but risking overfitting.",
        "when_to_adjust": "Increase alpha (e.g., to 5.0): R2 > 0.9 and MSE < 1000. Decrease alpha (e.g., to 0.01): R2 < 0.9 or MSE > 1000. Use When: Features are correlated. Avoid When: Non-linear relationships dominate.",
        "good_fit": "R2 > 0.9, MSE < 1000, MAE < 20. Predicted vs Actual plot shows good alignment.",
        "strategy": "Start with alpha=1.0. Adjust based on R2 and MSE."
    },
    "Elastic Net Regression": {
        "description": "Combines L1 and L2 regularization, balancing feature selection and coefficient shrinkage.",
        "importance": "Ideal for ship resistance when both feature selection and handling correlated features are needed, offering a robust compromise.",
        "parameters": [
            {"name": "alpha", "type": "number", "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01},
            {"name": "l1_ratio", "type": "number", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}
        ],
        "overfit_risk": 0.2,
        "plot_types": ["Predicted vs Actual"],
        "impact_on_metrics": "Increasing alpha: Stronger regularization, reducing overfitting but may lower R2 and increase MSE/MAE. Decreasing alpha: Weaker regularization, raising R2 and lowering MSE/MAE. Increasing l1_ratio: More L1 regularization, promoting sparsity, which may lower R2 if too many features are dropped. Decreasing l1_ratio: More L2 regularization, stabilizing coefficients but may not improve R2 much.",
        "when_to_adjust": "Increase alpha (e.g., to 0.5): R2 > 0.9 and MSE < 1000. Decrease alpha (e.g., to 0.01): R2 < 0.9 or MSE > 1000. Increase l1_ratio (e.g., to 0.8): Feature selection is desired. Decrease l1_ratio (e.g., to 0.2): Features are correlated. Use When: Both feature selection and correlated features are concerns. Avoid When: Non-linear relationships dominate.",
        "good_fit": "R2 > 0.9, MSE < 1000, MAE < 20. Predicted vs Actual plot shows good alignment.",
        "strategy": "Start with alpha=0.1, l1_ratio=0.5. Adjust based on metrics and feature needs."
    }
}

# Model dictionary with default configurations
MODELS = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": lambda params: make_pipeline(PolynomialFeatures(degree=int(params.get("degree", 2))), LinearRegression()),
    "Random Forest Regression": lambda params: RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=int(params.get("max_depth", 10)),
        random_state=42
    ),
    "Decision Tree Regression": lambda params: DecisionTreeRegressor(
        max_depth=int(params.get("max_depth", 10)),
        min_samples_split=int(params.get("min_samples_split", 2)),
        random_state=42
    ),
    "Support Vector Regression (SVR)": lambda params: make_pipeline(
        StandardScaler(),
        SVR(C=float(params.get("C", 1.0)), epsilon=float(params.get("epsilon", 0.1)))
    ),
    "XGBoost Regression": lambda params: XGBRegressor(
        n_estimators=int(params.get("n_estimators", 100)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        max_depth=int(params.get("max_depth", 6)),
        random_state=42,
        verbosity=0
    ),
    "Artificial Neural Network (ANN)": lambda params: MLPRegressor(
        hidden_layer_sizes=(
            int(params.get("hidden_layer_size1", 100)),
            int(params.get("hidden_layer_size2", 100))
        ),
        max_iter=1000,
        random_state=42,
        alpha=0.001,
        solver='adam',
        verbose=True,
        warm_start=True
    ),
    "Gaussian Process Regression": GaussianProcessRegressor(random_state=42),
    "Lasso Regression": lambda params: Lasso(alpha=float(params.get("alpha", 0.1))),
    "Ridge Regression": lambda params: Ridge(alpha=float(params.get("alpha", 1.0))),
    "Elastic Net Regression": lambda params: ElasticNet(
        alpha=float(params.get("alpha", 0.1)),
        l1_ratio=float(params.get("l1_ratio", 0.5))
    )
}

MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Polynomial Regression": "polynomial_regression.pkl",
    "Random Forest Regression": "random_forest.pkl",
    "Decision Tree Regression": "decision_tree.pkl",
    "Support Vector Regression (SVR)": "svr.pkl",
    "XGBoost Regression": "xgboost.pkl",
    "Artificial Neural Network (ANN)": "ann.pkl",
    "Gaussian Process Regression": "gaussian_process.pkl",
    "Lasso Regression": "lasso.pkl",
    "Ridge Regression": "ridge.pkl",
    "Elastic Net Regression": "elastic_net.pkl"
}

# Initialize global variables
trained_metrics = {}
best_model_name = None
training_plots = {}  # Store plot paths for each model
current_params = {}  # Store user-entered parameters

# Load persisted metrics, plots, and parameters on app start
def load_persisted_data():
    global trained_metrics, best_model_name, training_plots, current_params
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r') as f:
                data = json.load(f)
                trained_metrics = data.get('metrics', {})
                best_model_name = data.get('best_model', None)
                training_plots = data.get('plots', {})
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, 'r') as f:
                current_params = json.load(f)
    except Exception as e:
        logger.error(f"Error loading persisted data: {e}")

# Save metrics, plots, and parameters to files
def save_persisted_data():
    try:
        with open(METRICS_FILE, 'w') as f:
            json.dump({
                'metrics': trained_metrics,
                'best_model': best_model_name,
                'plots': training_plots
            }, f)
        with open(PARAMS_FILE, 'w') as f:
            json.dump(current_params, f)
    except Exception as e:
        logger.error(f"Error saving persisted data: {e}")

# Load persisted data on app start
load_persisted_data()

def load_data():
    """Load data, preprocess, and split into training and validation sets"""
    try:
        df = pd.read_excel(EXCEL_PATH)
        if 'Avg Draft [m]' not in df.columns:
            if 'Draft Aft [m]' in df.columns and 'Draft Fwd [m]' in df.columns:
                df['Avg Draft [m]'] = (df['Draft Aft [m]'] + df['Draft Fwd [m]']) / 2
            else:
                raise ValueError("Cannot find 'Draft Aft [m]' and 'Draft Fwd [m]' to calculate 'Avg Draft [m]'")
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]
        # Split into 90% training and 10% validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        return X_train, X_val, y_train, y_val
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def sanitize_columns(df):
    """Sanitize column names for XGBoost"""
    df = df.copy()
    df.columns = [col.replace('[', '_').replace(']', '_').replace('^', '_').replace(' ', '_') for col in df.columns]
    return df

def sanitize_model_name(model_name):
    """Sanitize model names for HTML IDs"""
    return model_name.replace(' ', '_').replace('(', '').replace(')', '')

def generate_training_plots(model, model_name, X_train, y_train, X_val, y_val, model_params):
    """Generate training and validation plots based on model type"""
    os.makedirs('static/plots', exist_ok=True)
    plot_paths = {}
    
    try:
        for plot_type in MODEL_DESCRIPTIONS[model_name]["plot_types"]:
            logger.info(f"Generating {plot_type} for {model_name}")
            
            if plot_type == "Loss Curve" and model_name in ["Artificial Neural Network (ANN)", "XGBoost Regression"]:
                if model_name == "Artificial Neural Network (ANN)":
                    model_instance = model
                    model_instance.max_iter = 1
                    train_losses = []
                    val_losses = []
                    epochs = range(1, 101)
                    for _ in epochs:
                        model_instance.partial_fit(X_train, y_train)
                        train_preds = model_instance.predict(X_train)
                        val_preds = model_instance.predict(X_val)
                        train_loss = mean_squared_error(y_train, train_preds)
                        val_loss = mean_squared_error(y_val, val_preds)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(epochs, train_losses, label='Training Loss')
                    plt.plot(epochs, val_losses, label='Validation Loss')
                    plt.title(f'{model_name} Loss Curve')
                    plt.xlabel('Epoch')
                    plt.ylabel('Mean Squared Error')
                    plt.grid(True)
                    plt.legend()
                
                elif model_name == "XGBoost Regression":
                    train_losses = []
                    val_losses = []
                    n_estimators = int(model_params.get("n_estimators", 100))
                    for i in range(1, n_estimators + 1):
                        model.set_params(n_estimators=i)
                        model.fit(X_train, y_train)
                        train_preds = model.predict(X_train)
                        val_preds = model.predict(X_val)
                        train_losses.append(mean_squared_error(y_train, train_preds))
                        val_losses.append(mean_squared_error(y_val, val_preds))
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(range(1, n_estimators + 1), train_losses, label='Training Loss')
                    plt.plot(range(1, n_estimators + 1), val_losses, label='Validation Loss')
                    plt.title(f'{model_name} Loss Curve')
                    plt.xlabel('Number of Trees')
                    plt.ylabel('Mean Squared Error')
                    plt.grid(True)
                    plt.legend()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                plot_path = f'static/plots/{sanitize_model_name(model_name)}_loss.png'
                with open(plot_path, 'wb') as f:
                    f.write(buf.getbuffer())
                plot_paths["Loss Curve"] = '/' + plot_path
                logger.info(f"Saved Loss Curve plot for {model_name} at {plot_path}")
            
            if plot_type == "Feature Importance" and model_name in ["Random Forest Regression", "Decision Tree Regression", "XGBoost Regression"]:
                feature_importance = model.feature_importances_
                plt.figure(figsize=(8, 6))
                plt.bar(FEATURE_COLS, feature_importance)
                plt.title(f'{model_name} Feature Importance')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                plot_path = f'static/plots/{sanitize_model_name(model_name)}_feature_importance.png'
                with open(plot_path, 'wb') as f:
                    f.write(buf.getbuffer())
                plot_paths["Feature Importance"] = '/' + plot_path
                logger.info(f"Saved Feature Importance plot for {model_name} at {plot_path}")
            
            if plot_type == "Predicted vs Actual":
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)
                plt.figure(figsize=(8, 6))
                plt.scatter(y_train, train_preds, alpha=0.5, label='Training')
                plt.scatter(y_val, val_preds, alpha=0.5, label='Validation')
                plt.plot([min(y_train.min(), y_val.min()), max(y_train.max(), y_val.max())], 
                         [min(y_train.min(), y_val.min()), max(y_train.max(), y_val.max())], 'r--', lw=2)
                plt.title(f'{model_name} Predicted vs Actual')
                plt.xlabel('Actual Rts (kN)')
                plt.ylabel('Predicted Rts (kN)')
                plt.grid(True)
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                plot_path = f'static/plots/{sanitize_model_name(model_name)}_pred_actual.png'
                with open(plot_path, 'wb') as f:
                    f.write(buf.getbuffer())
                plot_paths["Predicted vs Actual"] = '/' + plot_path
                logger.info(f"Saved Predicted vs Actual plot for {model_name} at {plot_path}")
            
            if plot_type == "Residual Plot":
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)
                train_residuals = y_train - train_preds
                val_residuals = y_val - val_preds
                plt.figure(figsize=(8, 6))
                plt.scatter(train_preds, train_residuals, alpha=0.5, label='Training')
                plt.scatter(val_preds, val_residuals, alpha=0.5, label='Validation')
                plt.axhline(0, color='r', linestyle='--')
                plt.title(f'{model_name} Residual Plot')
                plt.xlabel('Predicted Rts (kN)')
                plt.ylabel('Residuals (kN)')
                plt.grid(True)
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                plot_path = f'static/plots/{sanitize_model_name(model_name)}_residual.png'
                with open(plot_path, 'wb') as f:
                    f.write(buf.getbuffer())
                plot_paths["Residual Plot"] = '/' + plot_path
                logger.info(f"Saved Residual Plot for {model_name} at {plot_path}")
            
            if plot_type == "Training Error vs Trees" and model_name == "Random Forest Regression":
                n_estimators = int(model_params.get("n_estimators", 100))
                train_errors = []
                val_errors = []
                for i in range(1, n_estimators + 1):
                    model.set_params(n_estimators=i)
                    model.fit(X_train, y_train)
                    train_preds = model.predict(X_train)
                    val_preds = model.predict(X_val)
                    train_errors.append(mean_squared_error(y_train, train_preds))
                    val_errors.append(mean_squared_error(y_val, val_preds))
                plt.figure(figsize=(8, 6))
                plt.plot(range(1, n_estimators + 1), train_errors, label='Training Error')
                plt.plot(range(1, n_estimators + 1), val_errors, label='Validation Error')
                plt.title(f'{model_name} Training Error vs Number of Trees')
                plt.xlabel('Number of Trees')
                plt.ylabel('Mean Squared Error')
                plt.grid(True)
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                plot_path = f'static/plots/{sanitize_model_name(model_name)}_error_vs_trees.png'
                with open(plot_path, 'wb') as f:
                    f.write(buf.getbuffer())
                plot_paths["Training Error vs Trees"] = '/' + plot_path
                logger.info(f"Saved Training Error vs Trees plot for {model_name} at {plot_path}")
    
    except Exception as e:
        logger.error(f"Error generating plots for {model_name}: {e}")
        return {}
    
    return plot_paths

def calculate_model_score(val_r2, val_mse, val_mae, overfit_risk):
    """Calculate a composite score based on validation R2, MSE, MAE, and overfit risk"""
    if isinstance(val_r2, str):  # Handle error cases
        return -np.inf
    
    r2_norm = val_r2
    mse_norm = 1 / (1 + val_mse)
    mae_norm = 1 / (1 + val_mae)
    
    score = 0.5 * r2_norm + 0.25 * mse_norm + 0.25 * mae_norm
    score *= (1 - overfit_risk)
    return score

def train_single_model(model_name, params=None):
    """Train a single model and update metrics, plots, and best model"""
    global trained_metrics, best_model_name, training_plots, current_params
    params = params or {}
    current_params.update(params)  # Store user-entered parameters
    try:
        X_train, X_val, y_train, y_val = load_data()
        os.makedirs('models', exist_ok=True)

        if model_name not in MODELS:
            return f"Error: Model {model_name} not found"

        logger.info(f"Training single model: {model_name}")
        X_train_copy = sanitize_columns(X_train.copy())  # Sanitize feature names
        X_val_copy = sanitize_columns(X_val.copy())      # Sanitize feature names
        
        model_params = {k: v for k, v in params.items() if k in [p["name"] for p in MODEL_DESCRIPTIONS[model_name]["parameters"]]}
        model_func = MODELS[model_name]
        model = model_func(model_params) if callable(model_func) else model_func
        
        model.fit(X_train_copy, y_train)
        train_preds = model.predict(X_train_copy)
        val_preds = model.predict(X_val_copy)
        
        # Training metrics (for display)
        train_r2 = r2_score(y_train, train_preds)
        train_mse = mean_squared_error(y_train, train_preds)
        train_mae = mean_absolute_error(y_train, train_preds)
        
        # Validation metrics (for model selection)
        val_r2 = r2_score(y_val, val_preds)
        val_mse = mean_squared_error(y_val, val_preds)
        val_mae = mean_absolute_error(y_val, val_preds)

        # Update metrics for this model
        trained_metrics[model_name] = {
            'R2 Score': round(train_r2, 4),
            'MSE': round(train_mse, 4),
            'MAE': round(train_mae, 4)
        }

        # Generate and store plots
        plot_paths = generate_training_plots(model, model_name, X_train_copy, y_train, X_val_copy, y_val, model_params)
        training_plots[model_name] = plot_paths

        # Save the model
        with open(os.path.join('models', MODEL_FILES[model_name]), 'wb') as f:
            pickle.dump(model, f)

        # Update best model if this model is better
        score = calculate_model_score(val_r2, val_mse, val_mae, MODEL_DESCRIPTIONS[model_name]["overfit_risk"])
        logger.info(f"Trained {model_name} with Training R2: {train_r2:.4f}, Validation R2: {val_r2:.4f}, Score: {score:.4f}, Plots: {list(plot_paths.keys())}")

        # Check if this model is the best among all trained models
        best_score = -np.inf
        for name in trained_metrics:
            if name in trained_metrics and 'R2 Score' in trained_metrics[name] and isinstance(trained_metrics[name]['R2 Score'], (int, float)):
                try:
                    model_func = MODELS[name]
                    model = model_func({}) if callable(model_func) else model_func
                    model.fit(X_train_copy, y_train)  # Use sanitized features
                    val_preds_temp = model.predict(X_val_copy)
                    temp_score = calculate_model_score(
                        r2_score(y_val, val_preds_temp),
                        mean_squared_error(y_val, val_preds_temp),
                        mean_absolute_error(y_val, val_preds_temp),
                        MODEL_DESCRIPTIONS[name]["overfit_risk"]
                    )
                    if temp_score > best_score:
                        best_score = temp_score
                        best_model_name = name
                except Exception as e:
                    logger.error(f"Error evaluating {name} for best model: {e}")
                    continue
        
        # Save metrics, plots, and parameters
        save_persisted_data()
        
        return f"Model {model_name} trained successfully"
    
    except Exception as e:
        logger.error(f"Error training {model_name}: {str(e)}")
        trained_metrics[model_name] = {
            'R2 Score': 'Error',
            'MSE': 'Error',
            'MAE': str(e)
        }
        training_plots[model_name] = {}
        save_persisted_data()
        return f"Error training {model_name}: {e}"

def train_all_models(params=None):
    global trained_metrics, best_model_name, training_plots, current_params
    params = params or {}
    logger.info(f"Starting train_all_models with params: {params}")
    current_params.update(params)  # Store user-entered parameters
    try:
        X_train, X_val, y_train, y_val = load_data()
        logger.info("Data loaded successfully")
        trained_metrics = {}
        training_plots = {}
        best_score = -np.inf
        best_model_name = None
        os.makedirs('models', exist_ok=True)

        for name, model_func in MODELS.items():
            try:
                logger.info(f"Training model: {name}")
                X_train_copy = sanitize_columns(X_train.copy())  # Sanitize feature names
                X_val_copy = sanitize_columns(X_val.copy())      # Sanitize feature names
                
                model_params = {k: v for k, v in params.items() if k in [p["name"] for p in MODEL_DESCRIPTIONS[name]["parameters"]]}
                logger.debug(f"Model {name} using params: {model_params}")
                model = model_func(model_params) if callable(model_func) else model_func
                
                model.fit(X_train_copy, y_train)
                train_preds = model.predict(X_train_copy)
                val_preds = model.predict(X_val_copy)
                
                # Training metrics (for display)
                train_r2 = r2_score(y_train, train_preds)
                train_mse = mean_squared_error(y_train, train_preds)
                train_mae = mean_absolute_error(y_train, train_preds)
                
                # Validation metrics (for model selection)
                val_r2 = r2_score(y_val, val_preds)
                val_mse = mean_squared_error(y_val, val_preds)
                val_mae = mean_absolute_error(y_val, val_preds)

                # Store only training metrics for display
                trained_metrics[name] = {
                    'R2 Score': round(train_r2, 4),
                    'MSE': round(train_mse, 4),
                    'MAE': round(train_mae, 4)
                }

                plot_paths = generate_training_plots(model, name, X_train_copy, y_train, X_val_copy, y_val, model_params)
                training_plots[name] = plot_paths

                with open(os.path.join('models', MODEL_FILES[name]), 'wb') as f:
                    pickle.dump(model, f)

                score = calculate_model_score(val_r2, val_mse, val_mae, MODEL_DESCRIPTIONS[name]["overfit_risk"])
                if score > best_score:
                    best_score = score
                    best_model_name = name
                
                logger.info(f"Successfully trained {name} with Training R2: {train_r2:.4f}, Validation R2: {val_r2:.4f}, Score: {score:.4f}, Plots: {list(plot_paths.keys())}")

            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                trained_metrics[name] = {
                    'R2 Score': 'Error',
                    'MSE': 'Error',
                    'MAE': str(e)
                }
                training_plots[name] = {}
        
        # Save metrics, plots, and parameters after training
        logger.info("Saving persisted data")
        save_persisted_data()
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return f"Error: {e}"
    
    logger.info("train_all_models completed successfully")
    return "Models trained successfully"

# Update index route
@app.route('/', methods=['GET', 'POST'])
def index():
    global trained_metrics, training_plots
    prediction_table = None
    plot_url = None
    power_plot_url = None
    model_options = list(MODELS.keys())
    train_result = None
    selected_plot = None

    if request.method == 'GET':
        # Load persisted metrics and plots on page load
        load_persisted_data()

    if request.method == 'POST':
        logger.info(f"POST request received with form data: {request.form.to_dict()}")
        if 'train' in request.form:
            params = {}
            for model_name in MODELS.keys():
                sanitized_model_name = sanitize_model_name(model_name)
                for param in MODEL_DESCRIPTIONS[model_name]["parameters"]:
                    param_name = f"{sanitized_model_name}_{param['name']}"
                    if param_name in request.form:
                        params[param["name"]] = request.form[param_name]
            logger.debug(f"Collected params for train_all: {params}")
            train_result = train_all_models(params)
        elif 'train_single' in request.form:
            model_name = request.form['model_name']
            params = {}
            sanitized_model_name = sanitize_model_name(model_name)
            for param in MODEL_DESCRIPTIONS[model_name]["parameters"]:
                param_name = f"{sanitized_model_name}_{param['name']}"
                if param_name in request.form:
                    params[param["name"]] = request.form[param_name]
            logger.debug(f"Collected params for train_single ({model_name}): {params}")
            train_result = train_single_model(model_name, params)
        elif 'show_plot' in request.form:
            selected_model = request.form['selected_model']
            selected_plot_type = request.form['plot_type']
            selected_plot = training_plots.get(selected_model, {}).get(selected_plot_type)
        elif 'predict' in request.form:
            selected_model = request.form['selected_model']
            try:
                length = float(request.form['length'])
                breadth = float(request.form['breadth'])
                avg_draft = float(request.form['avg_draft'])
                displacement = float(request.form['displacement'])
                wetted_area = float(request.form['wetted_area'])
                block_coeff = float(request.form['block_coeff'])
                vs_start = float(request.form['vs_start'])
                vs_end = float(request.form['vs_end'])
                vs_step = float(request.form['vs_step'])

                vs_values = np.arange(vs_start, vs_end + vs_step, vs_step)
                input_data = pd.DataFrame({
                    'Length BP [m]': [length]*len(vs_values),
                    'Breadth [m]': [breadth]*len(vs_values),
                    'Avg Draft [m]': [avg_draft]*len(vs_values),
                    'Displacement [m^3]': [displacement]*len(vs_values),
                    'Wetted Surface Area [m^2]': [wetted_area]*len(vs_values),
                    'Block Coefficient': [block_coeff]*len(vs_values),
                    'Vs': vs_values
                })

                model_path = os.path.join('models', MODEL_FILES[selected_model])
                if not os.path.exists(model_path):
                    return render_template('index.html',
                                          trained_metrics=trained_metrics,
                                          best_model_name=best_model_name,
                                          model_options=model_options,
                                          model_descriptions=MODEL_DESCRIPTIONS,
                                          training_plots=training_plots,
                                          prediction_table="<p style='color:red;'>Error: Models need to be trained first</p>",
                                          plot_url=None,
                                          power_plot_url=None,
                                          train_result=train_result,
                                          selected_plot=selected_plot,
                                          current_params=current_params,
                                          sanitize_model_name=sanitize_model_name)

                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                input_data = sanitize_columns(input_data)  # Sanitize input data
                predictions = model.predict(input_data)
                display_data = pd.DataFrame({
                    'Vs': vs_values,
                    'Predicted Rts': predictions,
                    'PE (Power)': vs_values * predictions
                })

                prediction_table = display_data.to_html(classes='table table-bordered', index=False)
                os.makedirs('static', exist_ok=True)

                plt.figure(figsize=(8,6))
                plt.plot(vs_values, predictions, marker='o')
                plt.title(f'Vs vs Predicted Rts ({selected_model})')
                plt.xlabel('Vs (knots)')
                plt.ylabel('Predicted Rts (kN)')
                plt.grid(True)
                plot_buf = io.BytesIO()
                plt.savefig(plot_buf, format='png')
                plt.close()
                plot_buf.seek(0)
                plot_path = 'static/plot.png'
                with open(plot_path, 'wb') as f:
                    f.write(plot_buf.getbuffer())
                plot_url = '/' + plot_path

                plt.figure(figsize=(8,6))
                plt.plot(vs_values, vs_values * predictions, marker='o', color='green')
                plt.title(f'Vs vs PE (Power) ({selected_model})')
                plt.xlabel('Vs (knots)')
                plt.ylabel('PE (Power) (kW)')
                plt.grid(True)
                power_plot_buf = io.BytesIO()
                plt.savefig(power_plot_buf, format='png')
                plt.close()
                power_plot_buf.seek(0)
                power_plot_path = 'static/power_plot.png'
                with open(power_plot_path, 'wb') as f:
                    f.write(power_plot_buf.getbuffer())
                power_plot_url = '/' + power_plot_path

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                prediction_table = f'<p style="color:red;">Error: {e}</p>'

    return render_template('index.html',
                          trained_metrics=trained_metrics,
                          best_model_name=best_model_name,
                          model_options=model_options,
                          model_descriptions=MODEL_DESCRIPTIONS,
                          training_plots=training_plots,
                          prediction_table=prediction_table,
                          plot_url=plot_url,
                          power_plot_url=power_plot_url,
                          train_result=train_result,
                          selected_plot=selected_plot,
                          current_params=current_params,
                          sanitize_model_name=sanitize_model_name)
if __name__ == '__main__':
    app.run(debug=True)