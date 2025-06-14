
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[x]) for x in ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'dpf', 'age']]
    prediction = model.predict([features])
    result = 'Positive for Diabetes' if prediction[0] == 1 else 'Negative for Diabetes'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
