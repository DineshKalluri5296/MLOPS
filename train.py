import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

def train_and_save_model():
    """
    Simulates training a Linear Regression model and saving it.
    """
    print("Starting model training...")

    # 1. Create dummy data for demonstration (Area vs. Price)
    # Area (X) is a 2D array as required by Scikit-learn's fit method
    X = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
    # Price (y) - simple linear relationship: Price = 50 * Area + 10000
    y = 50 * X.flatten() + 10000 + np.random.randn(5) * 5000 

    # 2. Initialize and train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # 3. Save the trained model to model.pkl
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
        
    print(f"Model trained (Coeff: {model.coef_[0]:.2f}, Intercept: {model.intercept_:.2f}) and saved as model.pkl")

if __name__ == "__main__":
    train_and_save_model()