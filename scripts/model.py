# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:10:28 2025

@author: m
"""

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load a sample dataset (Iris)
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to a pickle file
pickle_file_path = './model.pkl'
with open(pickle_file_path, 'wb') as file:
    pickle.dump(model, file)

pickle_file_path