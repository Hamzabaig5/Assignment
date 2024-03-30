from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return 'Hello, welcome to the Flask app!'

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()
    
    # Ensure the input data is in the expected format
    if 'input' not in data:
        return jsonify({'error': 'Input data is missing'}), 400
    
    # Make prediction using the loaded model
    input_data = data['input']
    prediction = model.predict([[input_data]])
    
    # Return the prediction as JSON response
    return jsonify({'prediction': prediction.tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True)
