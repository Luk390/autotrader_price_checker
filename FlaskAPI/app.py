import flask
from flask import Flask, jsonify, request
import json
import pandas as pd
import numpy as np
import pickle
import os
app = Flask(__name__)

columns = ['body_type',
 'co2Emissions',
 'condition',
 'doors',
 'engine_size',
 'isTradeSeller',
 'make',
 'manufactured_year',
 'mileage',
 'model',
 'seats',
 'transmission',
 'location']

row = [['Hatchback',
 101,
 'Used',
 5.0,
 1.0,
 True,
 'Volkswagen',
 2018.0,
 14254,
 'Polo',
 5.0,
 'Manual',
 'Leeds']]

data_in = pd.DataFrame(row, columns=columns)

import os.path
from os import path

print(path.exists("models/model_file.p"))


def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

@app.route('/predict', methods=['GET'])
def predict():
    # stub input features
    x = data_in
    # load model
    model = load_models()
    prediction = model.predict(x).tolist()
    response = json.dumps({'response': prediction})
    return response['response'], 200


if __name__ == '__main__':
    application.run(debug=True)