from flask import Flask, jsonify
from flask_cors import CORS
from bmw import load_data, preprocess_data, train_linear_regression, scale_data, train_arima_model, optimize_portfolio

app = Flask(__name__)
CORS(app)

@app.route('/load-preprocess')
def load_and_preprocess():
    df = load_data('/Users/azrabano/Downloads/projects_bmw/data/BMW_Data.csv')
    preprocessed_df = preprocess_data(df)
    return jsonify(preprocessed_df.to_json(orient="split")), 200

@app.route('/train-lr')
def train_lr():
    df = load_data('/Users/azrabano/Downloads/projects_bmw/data/BMW_Data.csv')
    df_preprocessed = preprocess_data(df)
    scaled_df, _ = scale_data(df_preprocessed)
    _, mse, predictions = train_linear_regression(scaled_df)
    return jsonify({"MSE": mse, "Predictions": predictions.tolist()}), 200

@app.route('/forecast-arima')
def forecast_arima():
    df = load_data('/Users/azrabano/Downloads/projects_bmw/data/BMW_Data.csv')
    _, forecast = train_arima_model(df)
    return jsonify({"Forecast": forecast.tolist()}), 200

@app.route('/optimize-portfolio')
def optimize_portfolio_route():
    df = load_data('/Users/azrabano/Downloads/projects_bmw/data/BMW_Data.csv')
    weights, performance = optimize_portfolio(df)
    return jsonify({"Weights": weights, "Performance": performance}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
