from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
# Ensure the 'models' folder exists in your directory
model = pickle.load(open("models/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # Converting dictionary values to a numpy array for the model
        features = np.array([list(data.values())]).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        return jsonify({
            "PlacementStatus": "Placed" if prediction == 1 else "NotPlaced"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)