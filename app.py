import os
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt

from flask import Flask, request, render_template

from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

app = Flask(__name__)

# Path to your Excel file (change this if needed)
EXCEL_PATH = 'ilovepdf_merged_all_resistance_data.xlsx'

# Updated features with Avg Draft instead of Draft Aft and Draft Fwd
FEATURE_COLS = [
    'Length BP [m]', 'Breadth [m]', 'Avg Draft [m]',
    'Displacement [m^3]', 'Wetted Surface Area [m^2]', 'Block Coefficient', 'Vs'
]
TARGET_COL = 'Rts'

# Model dictionary
MODELS = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
    "Support Vector Regression (SVR)": make_pipeline(StandardScaler(), SVR()),
    "XGBoost Regression": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "Artificial Neural Network (ANN)": MLPRegressor(hidden_layer_sizes=(100,100), max_iter=1000, random_state=42),
    "Gaussian Process Regression": GaussianProcessRegressor(random_state=42),
    "Lasso Regression": Lasso(alpha=0.1),
    "Bayesian Ridge Regression": BayesianRidge()
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
    "Bayesian Ridge Regression": "bayesian_ridge.pkl"
}

trained_metrics = {}
best_model_name = None


def load_data():
    """Load data and preprocess to calculate Avg Draft if needed"""
    df = pd.read_excel(EXCEL_PATH)
    
    # Check if Avg Draft already exists, if not calculate it
    if 'Avg Draft [m]' not in df.columns:
        if 'Draft Aft [m]' in df.columns and 'Draft Fwd [m]' in df.columns:
            df['Avg Draft [m]'] = (df['Draft Aft [m]'] + df['Draft Fwd [m]']) / 2
        else:
            raise ValueError("Cannot find 'Draft Aft [m]' and 'Draft Fwd [m]' to calculate 'Avg Draft [m]'")
    
    # Select only the required columns
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    return X, y


def sanitize_columns(df):
    """Sanitize column names for XGBoost"""
    df = df.copy()
    df.columns = [col.replace('[', '_').replace(']', '_').replace('^', '_').replace(' ', '_') for col in df.columns]
    return df


def train_all_models():
    global trained_metrics, best_model_name
    
    try:
        X, y = load_data()
        trained_metrics = {}

        best_score = -np.inf
        best_model_name = None

        os.makedirs('models', exist_ok=True)

        for name, model in MODELS.items():
            try:
                X_train = X.copy()

                if name == "XGBoost Regression":
                    X_train = sanitize_columns(X_train)

                model.fit(X_train, y)

                preds = model.predict(X_train)  # Use X_train to match the sanitized columns if needed
                r2 = r2_score(y, preds)
                mse = mean_squared_error(y, preds)
                mae = mean_absolute_error(y, preds)

                trained_metrics[name] = {
                    'R2 Score': round(r2, 4),
                    'MSE': round(mse, 4),
                    'MAE': round(mae, 4)
                }

                # Save model
                with open(os.path.join('models', MODEL_FILES[name]), 'wb') as f:
                    pickle.dump(model, f)

                # Track best model
                if r2 > best_score:
                    best_score = r2
                    best_model_name = name

            except Exception as e:
                trained_metrics[name] = {'R2 Score': 'Error', 'MSE': 'Error', 'MAE': str(e)}
    
    except Exception as e:
        print(f"Error during model training: {e}")
        return f"Error: {e}"
    
    return "Models trained successfully"


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_table = None
    plot_url = None
    power_plot_url=None
    model_options = list(MODELS.keys())
    train_result = None

    selected_model = None
    if request.method == 'POST':
        if 'train' in request.form:
            train_result = train_all_models()
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

                # Check if model file exists
                model_path = os.path.join('models', MODEL_FILES[selected_model])
                if not os.path.exists(model_path):
                    return render_template('index.html',
                                          trained_metrics=trained_metrics,
                                          best_model_name=best_model_name,
                                          model_options=model_options,
                                          prediction_table="<p style='color:red;'>Error: Models need to be trained first</p>",
                                          plot_url=None,
                                          power_plot_url=None,
                                          train_result=train_result)

                # Load selected model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                # Prepare input data for XGBoost if needed
                if selected_model == "XGBoost Regression":
                    input_data = sanitize_columns(input_data)

                predictions = model.predict(input_data)
                
                # Add this code after calculating predictions
                # Create display DataFrame with original column names and power calculation
                display_data = pd.DataFrame({
                    'Vs': vs_values,
                    'Predicted Rts': predictions,
                    'PE (Power)': vs_values * predictions  # Calculate power as Vs * Rts
                })

                prediction_table = display_data.to_html(classes='table table-bordered', index=False)

                # Plot
                plt.figure(figsize=(8,6))
                plt.plot(vs_values, predictions, marker='o')
                plt.title(f'Vs vs Predicted Rts ({selected_model})')
                plt.xlabel('Vs')
                plt.ylabel('Predicted Rts')
                plt.grid(True)

                plot_buf = io.BytesIO()
                plt.savefig(plot_buf, format='png')
                plt.close()
                plot_buf.seek(0)

                os.makedirs('static', exist_ok=True)
                plot_path = 'static/plot.png'
                with open(plot_path, 'wb') as f:
                    f.write(plot_buf.getbuffer())

                plot_url = '/' + plot_path

                # Add this code after creating the first plot
                # Create a second plot for Vs vs PE
                plt.figure(figsize=(8,6))
                plt.plot(vs_values, vs_values * predictions, marker='o', color='green')
                plt.title(f'Vs vs PE (Power) ({selected_model})')
                plt.xlabel('Vs')
                plt.ylabel('PE (Power)')
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
                prediction_table = f'<p style="color:red;">Error: {e}</p>'

    return render_template('index.html',
                        trained_metrics=trained_metrics,
                        best_model_name=best_model_name,
                        model_options=model_options,
                        prediction_table=prediction_table,
                        plot_url=plot_url,
                        power_plot_url=power_plot_url,  # Add this line
                        train_result=train_result)


# Create a basic template if it doesn't exist
@app.before_first_request
def create_template():
    os.makedirs('templates', exist_ok=True)
    
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Ship Resistance Prediction</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body { padding: 20px; }
        .container { max-width: 1200px; }
        .model-metrics { margin-top: 20px; }
        .prediction-form { margin-top: 30px; }
        .plot-container { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Ship Resistance Prediction</h1>
        
        <div class="card">
            <div class="card-header">Model Training</div>
            <div class="card-body">
                <form method="post">
                    <button type="submit" name="train" class="btn btn-primary">Train All Models</button>
                </form>
                
                {% if train_result %}
                <div class="alert {% if 'Error' in train_result %}alert-danger{% else %}alert-success{% endif %} mt-3">
                    {{ train_result }}
                </div>
                {% endif %}
                
                {% if trained_metrics %}
                <div class="model-metrics">
                    <h3>Model Performance</h3>
                    {% if best_model_name %}
                    <div class="alert alert-success">
                        Best Model: <strong>{{ best_model_name }}</strong>
                    </div>
                    {% endif %}
                    <table class="table table-bordered">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>R2 Score</th>
                                <th>MSE</th>
                                <th>MAE</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for model_name, metrics in trained_metrics.items() %}
                            <tr>
                                <td>{{ model_name }}</td>
                                <td>{{ metrics['R2 Score'] }}</td>
                                <td>{{ metrics['MSE'] }}</td>
                                <td>{{ metrics['MAE'] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                {% endif %}
            </div>
        </div>
        
        <div class="card prediction-form mt-4">
            <div class="card-header">Make Predictions</div>
            <div class="card-body">
                <form method="post">
                    <div class="form-row">
                        <div class="form-group col-md-6">
                            <label for="selected_model">Select Model</label>
                            <select name="selected_model" id="selected_model" class="form-control" required>
                                {% for model in model_options %}
                                <option value="{{ model }}">{{ model }}</option>
                                {% endfor %}
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group col-md-4">
                            <label for="length">Length BP [m]</label>
                            <input type="number" step="0.01" name="length" id="length" class="form-control" required>
                        </div>
                        <div class="form-group col-md-4">
                            <label for="breadth">Breadth [m]</label>
                            <input type="number" step="0.01" name="breadth" id="breadth" class="form-control" required>
                        </div>
                        <div class="form-group col-md-4">
                            <label for="avg_draft">Avg Draft [m]</label>
                            <input type="number" step="0.01" name="avg_draft" id="avg_draft" class="form-control" required>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group col-md-4">
                            <label for="displacement">Displacement [m^3]</label>
                            <input type="number" step="0.01" name="displacement" id="displacement" class="form-control" required>
                        </div>
                        <div class="form-group col-md-4">
                            <label for="wetted_area">Wetted Surface Area [m^2]</label>
                            <input type="number" step="0.01" name="wetted_area" id="wetted_area" class="form-control" required>
                        </div>
                        <div class="form-group col-md-4">
                            <label for="block_coeff">Block Coefficient</label>
                            <input type="number" step="0.001" name="block_coeff" id="block_coeff" class="form-control" required>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group col-md-4">
                            <label for="vs_start">Vs Start</label>
                            <input type="number" step="0.5" name="vs_start" id="vs_start" class="form-control" value="5" required>
                        </div>
                        <div class="form-group col-md-4">
                            <label for="vs_end">Vs End</label>
                            <input type="number" step="0.5" name="vs_end" id="vs_end" class="form-control" value="15" required>
                        </div>
                        <div class="form-group col-md-4">
                            <label for="vs_step">Vs Step</label>
                            <input type="number" step="0.1" name="vs_step" id="vs_step" class="form-control" value="0.5" required>
                        </div>
                    </div>
                    
                    <button type="submit" name="predict" class="btn btn-success">Predict</button>
                </form>
            </div>
        </div>
        
        {% if prediction_table %}
        <div class="card mt-4">
            <div class="card-header">Prediction Results</div>
            <div class="card-body">
                {{ prediction_table|safe }}
                
                {% if plot_url %}
                <div class="plot-container text-center mt-4">
                    <img src="{{ plot_url }}" alt="Resistance Plot" class="img-fluid">
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}
    </div>
    
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
            ''')

if __name__ == '__main__':
    app.run(debug=True)