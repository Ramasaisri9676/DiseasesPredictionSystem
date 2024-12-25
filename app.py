from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import pandas as pd

app = Flask(__name__)

model = joblib.load('multi_output_disease_predict_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
df = pd.read_csv('disease_dataset.csv')
df.columns = df.columns.str.strip()

@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    symptoms = df['symptoms'].str.split(',').explode().str.strip().unique().tolist()
    return jsonify({"symptoms": sorted(symptoms)})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected_symptoms = data.get('symptoms', [])
    
    if not selected_symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    symptoms_string = ",".join(selected_symptoms)
    symptoms_vectorized = vectorizer.transform([symptoms_string])
    y_pred = model.predict(symptoms_vectorized)

    try:
        response = {
            'disease': y_pred[:, 0][0],
            'cures': y_pred[:, 1][0],
            'doctor': y_pred[:, 2][0],
            'risk level': y_pred[:, 3][0]
        }
        return jsonify(response)
    except IndexError:
        return jsonify({"error": "Prediction failed. Ensure the model and dataset are valid."}), 500

if __name__ == '__main__':
    app.run(debug=True)
