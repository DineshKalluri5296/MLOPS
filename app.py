#This server loads the saved model.pkl and exposes a /predict endpoint to receive input data and return a prediction.

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# 1. Initialize FastAPI app
app = FastAPI()

# 2. Define the input data structure for the API
class PredictionRequest(BaseModel):
    area: float

# 3. Load the model globally when the app starts
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("ML Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl not found. Run train.py first.")
    model = None

# 4. Define a root endpoint (for health check)
@app.get("/")
def read_root():
    return {"status": "Model service is running"}

# 5. Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        return {"error": "Model not loaded"}, 500

    # Get the area from the request body
    area = request.area
    
    # FastAPI expects a 2D array for single-feature models
    input_data = np.array([[area]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the prediction value
    return {"predicted_price": float(prediction[0])}