# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 14:07:39 2025

@author: sabra
"""

#app/routes.py (Handles API Requests)
from flask import request, jsonify
import pickle
import numpy as np

# Load the model
model = pickle.load(open('app/model/model.pkl', 'rb'))


def configure_routes(app):
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            
            data = request.get_json(force=True)
            features = [list(data.values())]
            prediction = model.predict(features)
            # Convert NumPy int64 to Python int
            prediction_value = int(prediction[0])
            return jsonify({'prediction': prediction_value})

        except Exception as e:
            return jsonify({'error': str(e)}), 400
