from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# 🔹 Load Model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return "🏆 ML Model is Running! Use /predict for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_features = np.array([list(data.values())]).reshape(1, -1)
    
    prediction = model.predict(input_features)
    result = "Approved" if prediction[0] == 1 else "Rejected"
    
    return jsonify({"Loan Prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
