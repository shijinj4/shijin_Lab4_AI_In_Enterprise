from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load models and scaler
reg_model = joblib.load("fish_regressor.pkl")
clf_model = joblib.load("fish_classifier.pkl")
scaler = joblib.load("fish_scaler.pkl")
label_encoder = joblib.load("fish_label_encoder.pkl")

app = Flask(__name__, template_folder="templates")  # Ensure templates folder is set

@app.route("/")  # Serve index.html
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input from form submission
        data = request.form.to_dict()

        # Convert string inputs to floats in proper order
        features = [float(data[x]) for x in ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]]  # Now includes 'Weight'

        # Scale features
        scaled_features = scaler.transform([features])

        # Predict regression (fish weight)
        weight_pred = reg_model.predict(scaled_features)[0]

        # Predict classification (fish species)
        species_pred = clf_model.predict(scaled_features)[0]
        species_name = label_encoder.inverse_transform([species_pred])[0]

        return render_template("index.html", prediction=f"Predicted Weight: {round(weight_pred, 2)} g, Species: {species_name}")

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
