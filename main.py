from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model

app = Flask(__name__)

# Load the model
MODEL_PATH = 'model.h5'
SMALL_MODEL_PATH = 'small_model.h5'
small_model = load_model(SMALL_MODEL_PATH)
model = load_model(MODEL_PATH)

def make_predictions(model, sales_and_spend, window_size=360, predict_days=30):
    if sales_and_spend.shape[0] < window_size:
        raise ValueError(f"Input data must have at least {window_size} rows. Provided data has only {sales_and_spend.shape[0]} rows.")
    
    future_sales_and_spend = np.zeros((predict_days, 2))  # Initialize array to store predicted values
    x_input_predict = sales_and_spend[-window_size:]
    
    for i in range(predict_days):
        x_input_predict = x_input_predict.reshape((1, window_size, 2))  # Reshape input data for model
        prediction = model.predict(x_input_predict)  # Make prediction
        future_sales_and_spend[i] = prediction  # Store predicted values
        x_input_predict = np.roll(x_input_predict, -1)  # Remove the first data from x_input_predict
        x_input_predict[:, -1, :] = prediction  # Update x_input_data with the previous prediction
    
    return future_sales_and_spend

@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = data['features']
        
        # Convert features to numpy array and combine sales and spend
        sales_and_spend = np.array(features)
        
        # Determine the predict_days from the input or use a default value
        predict_days = 30  # You can adjust this value as needed
        
        # Make predictions
        if sales_and_spend.shape[0] < 360:
            print("Using small model")
            res = make_predictions(small_model, sales_and_spend, window_size=30, predict_days=predict_days)
        else :
            print("Using Big model")
            res = make_predictions(model, sales_and_spend, window_size=360, predict_days=predict_days)
        # Convert prediction results to list for JSON serialization
        return jsonify({'prediction': res.tolist()})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Invalid input or internal error.'}), 400

if __name__ == '__main__':
    print(" Starting app...")
    app.run(debug=True)

