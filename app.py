from flask import Flask, request, jsonify
import pickle

# Load the trained model
with open('crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Crop Recommendation API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the POST request
        data = request.get_json()

        # Extract features
        features = [
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]

        # Predict the crop
        prediction = model.predict([features])[0]

        # Return the prediction
        return jsonify({'crop': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app (for local testing)
if __name__ == '__main__':
    app.run(debug=True)
