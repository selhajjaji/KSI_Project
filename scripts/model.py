# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 13:49:01 2025

@author: m
"""

# ==============================
# KSI Project Full Pipeline File
# ==============================
# Combines: Exploration, Modeling, Training, Evaluation
#  Group Members: 
#   Sabra Elhajjaji 
#   Lissette Gorrin Rodriguez
#   Sara Hamoleila
#   Yunlong Liu
#   Darpan Nayyar
# =============================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_curve, auc, 
                            classification_report, roc_auc_score)
import pickle
import logging
import os
# --------------------------------
# 1. Load Dataset & Exploration
# -------------------------------
print("\n  1. Load Dataset & Exploration ")
# Load the Dataset
file_path = "../data/PASSENGER_KSI.csv"
df = pd.read_csv(file_path)

# Display First Rows and Dataset Info
print("\n First 5 Rows of the Dataset:")
print(df.head())  # Shows first 5 rows

# This dataset contains records of road collisions, focusing on passengers.
# Here are the first five rows. As you can see, it includes variables like 
# OBJECTID, ACCNUM, DATE, TIME, VEHICLE TYPE, INJURY details, and more.

# Dataset Summary & Structure
print("\n Dataset Summary:")
print(df.info())  # Shows column names, types, and missing values

# Check Data Types and Missing Values
print("\n Column Names & Data Types:")
print(df.dtypes)

# Here is the summary of the dataset structure.
# There are 54 columns and 7,183 entries.
# Most columns are of object type, meaning categorical, 
# and a few are numerical like LATITUDE, LONGITUDE, TIME, etc.

# Missing Values Inspection
print("\n Missing Values in Dataset:")
print(df.isnull().sum())  # Counts missing values per column

# This output shows the count of missing values in each column.
#Notice that some columns like OFFSET, INJURY, FATAL_NO, 
# and many driver behavior columns have a large number of missing values.


# Statistical Summary of Numeric Features
print("\n Statistical Summary of Numeric Features:")
print(df.describe())  # Gives min, max, mean, etc.
# This table shows statistical metrics such as mean, min, max, 
# and standard deviation for numeric columns.
# For instance, LATITUDE and LONGITUDE represent accident locations.


# -------------------------------
# Visualizations
# -------------------------------

# Display Dataset Columns
#   Print column names to verify available data
print("\nDataset Columns:")
print(df.columns)
# These are all the dataset columns. 
# Many of them are location-based, accident characteristics, or driver/passenger-related.


# 1. Visualization - Distribution of Fatal vs. Serious Injuries
plt.figure(figsize=(8,6))
sns.countplot(x="ACCLASS", data=df, palette="coolwarm")  # ACCLASS = Accident Class (Fatal vs Injury)
plt.title("Distribution of Accident Class")
plt.xlabel("Accident Class")
plt.ylabel("Count")
plt.savefig("accident_class_distribution.png")  # Save figure
plt.show()
# This bar chart shows the distribution of accident classes.
# The majority of cases are Non-Fatal Injuries, followed by Fatal cases.
# Property Damage cases are almost non-existent in this dataset.


# 2. Visualization - Accident Count by Visibility Condition
plt.figure(figsize=(10,6))
sns.countplot(y="VISIBILITY", data=df, palette="viridis", order=df["VISIBILITY"].value_counts().index)
plt.title("Accident Count by Visibility Conditions")
plt.xlabel("Count")
plt.ylabel("Visibility Condition")
plt.savefig("visibility_condition_distribution.png")  # Save figure
plt.show()
# This visualization shows accident counts based on visibility conditions.
# The majority of accidents happened in clear visibility, followed by rainy conditions.
# Snow, fog, and other conditions had relatively fewer accidents.

# 3. Correlation Heatmap (Numeric Features)
df_corr = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12,8))
sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("feature_correlation_heatmap.png")  # Save figure
plt.show()
# This heatmap shows correlation between numeric variables.
#As we can see, LATITUDE, LONGITUDE, X, and Y are highly correlated – 
# which is expected since they represent geographic coordinates.
# FATAL_NO has weak correlation with other features.

# 4. Visualization - Missing Data Visualization
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.savefig("missing_values_heatmap.png")  # Save figure
plt.show()
# This heatmap visually represents missing values in the dataset.
# Yellow bars represent missing values.
# It's clear that several important columns have many missing entries, 
# which needs attention during data preparation.

# Summary of Findings
print("\n Data Exploration Completed!")
print(" Key Findings:")
print(" The dataset includes serious injuries and fatalities.")
print(" Visibility conditions impact the number of accidents.")
print(" Missing values exist in several columns and need handling before modeling.")
print(" Some variables have strong correlations, which may help in prediction.")
# To summarize:
# -The dataset includes serious injuries and fatalities.
# -Visibility conditions have a clear impact on accidents.
# -Many columns have significant missing values.
# -Some variables, especially location data, are highly correlated.

# ----------------------------
# 2. Data Cleaning & Modeling
# ----------------------------

print("\n 2. Data Cleaning & Modeling\n")
# =======================
# Drop columns with excessive missing values
# =======================
drop_cols = ['FATAL_NO', 'PEDTYPE', 'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT', 'CYCCOND']
df.drop(columns=drop_cols, inplace=True)

# =======================
# Convert 'INVAGE' from string range to midpoint integer
# =======================
def convert_age_range(age_range):
    try:
        low, high = age_range.split(' to ')
        return (int(low) + int(high)) // 2
    except:
        return np.nan

df['INVAGE'] = df['INVAGE'].apply(convert_age_range)

df['INVAGE'] = df['INVAGE'].fillna(df['INVAGE'].median())

# =======================
# Feature Engineering
# =======================
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df['Month'] = df['DATE'].dt.month
df['DayOfWeek'] = df['DATE'].dt.dayofweek

# Fix TIME parsing from integer (e.g., 236) to datetime
# Step 1: Pad with zeros to get 4-digit time (e.g., '0236')
df['TIME'] = df['TIME'].astype(str).str.zfill(4)
# Step 2: Parse into datetime using %H%M format
df['TIME'] = pd.to_datetime(df['TIME'], format='%H%M', errors='coerce')
df = df[df['TIME'].notna()]
df['Hour'] = df['TIME'].dt.hour
df['RushHour'] = df['Hour'].apply(lambda x: 1 if x in [7, 8, 9, 16, 17, 18] else 0)

# =======================
# Select features and target
# =======================
target = 'ACCLASS'
features = ['TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE',
            'INVAGE', 'INVTYPE', 'Month', 'DayOfWeek', 'Hour',
            'SPEEDING', 'ALCOHOL']

# Drop rows with missing values
df = df.dropna(subset=features + [target])
print("\n Final dataset shape after dropping nulls:", df.shape)

if df.shape[0] == 0:
    raise ValueError(" No data left after preprocessing! Please check for over-cleaning or missing values.")

# =======================
# Prepare data
# =======================
X = df[features]
y = df[target].apply(lambda val: 1 if val == 'Fatal' else 0)

# =======================
# Preprocessing pipeline
# =======================
numeric_features = ['INVAGE', 'Month', 'DayOfWeek', 'Hour']
categorical_features = ['TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE',
                        'INVTYPE','SPEEDING', 'ALCOHOL']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# =======================
# Apply preprocessing
# =======================
X_processed = preprocessor.fit_transform(X)

# =======================
# Train/test split BEFORE SMOTE
# =======================
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
 
# =======================
# Handle class imbalance (SMOTE applied only to training data)
# =======================
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
 # =======================
# 3.1 Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),    
    "SVM": SVC(probability=True, random_state=42),
    "Neural Network": MLPClassifier(random_state=42)
}
param_grids = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
    "Decision Tree": {"max_depth": [5, 10, 15]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "Neural Network": {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.0001, 0.001]}
}

best_models = {}
model_scores = {}

print("\n--- Model Training and Tuning ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=3, n_jobs=-1, scoring='f1')
    grid.fit(X_train_balanced, y_train_balanced)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    y_pred = best_model.predict(X_test)
    score = f1_score(y_test, y_pred)
    model_scores[name] = score
    print(f"Best Params for {name}: {grid.best_params_} | F1 Score: {score:.4f}")
    # Save each model
    print('Saving',name,'....')
    with open(f"../app/model/{name.replace(' ', '_').lower()}_model.pkl", "wb") as f:
       pickle.dump(best_model, f)

# Select best performing model based on F1 Score
best_model_name = max(model_scores, key=model_scores.get)
best_model = best_models[best_model_name]
print('best model is : ',best_model)

print("\nSaving the model for deployment in progress......\n")
# Save model for deployment    

print("\nCreating full pipeline with best model...")
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])
print("\nSaving the full pipeline for deployment...")
with open("../app/model/model.pkl", "wb") as f:
   pickle.dump(full_pipeline, f)
    
print("\nThe full pipeline has been saved")
# --------------------------
# 4. Evaluation (on all models)
# --------------------------

print("\n 4. Model Evaluation\n")
for name, model in best_models.items():
    print(f"\n=== Evaluation: {name} ===")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    print('Classification Report of',name,':')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix of',name,':')
    print(confusion_matrix(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"roc_curve_{name.replace(' ', '_').lower()}.png")
    plt.show()

print(f"\nAll models trained and evaluated. All models and best model saved ({best_model_name}).")
