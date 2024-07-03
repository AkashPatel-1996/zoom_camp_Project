from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
pickle_file_path = "../../SavedModel/lin_beg.bin"
with open(pickle_file_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.json
    df = pd.DataFrame([json_data])
    predictions = model.predict(df)
    return jsonify({'predictions': predictions.tolist()})

@app.route('/submit', methods=['POST'])
def submit():
    data = {
        "mean radius": request.form['mean_radius'],
        "mean texture": request.form['mean_texture'],
        "mean perimeter": request.form['mean_perimeter'],
        "mean area": request.form['mean_area']
    }
    df = pd.DataFrame([data])
    predictions = model.predict(df)
    if predictions[0] == 0:
        result = "Malignant"
    else:
        result = "Benign"
    return render_template('result.html', prediction= result)

if __name__ == "__main__":
    app.run(debug=True)
