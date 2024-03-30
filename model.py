import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

def train_model(X, y):

    model = LinearRegression()
    model.fit(X, y)
    
    return model

def save_model(model, filename):
    # Save the trained model to a file
    joblib.dump(model, filename)

def load_model(filename):
    # Load the trained model from a file
    model = joblib.load(filename)
    return model

if __name__ == "__main__":
    # Example usage
    # Create sample dataset (replace this with your actual dataset)
    X_train = np.array([[1], [2], [3], [4], [5]])
    y_train = np.array([2, 4, 6, 8, 10])
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Save the model to a file
    model_filename = 'model.pkl'
    save_model(model, model_filename)
    
    print("Model trained and saved to", model_filename)
