# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:15:29 2025

@author: m
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 14:07:39 2025

@author: sabra
"""

#app/routes.py (Handles API Requests)
from flask import request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from flask_cors import CORS

# Load the model
model = pickle.load(open('app/model/model.pkl', 'rb'))


def configure_routes(app):
    CORS(app)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
                        
            
             data = request.get_json()
             print("Received Data:", data)
             
             # Convert to DataFrame (ensure keys match your features)
             input_data = pd.DataFrame([data])
             
                       
             # Predict
             prediction = model.predict(input_data)
             proba = model.predict_proba(input_data)[:, 1]  # Probability of "Fatal"                     
             return jsonify({
                 "prediction": int(prediction[0]),
                 "message": "Fatal" if prediction[0] == 1 else "Non-Fatal"
             })


        except Exception as e:
            return jsonify({'error': str(e)}), 400
