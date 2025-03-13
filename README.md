Fish Market Prediction System - Lab#4 (Heroku Deployment)
AI in Enterprise - Lab 4

Overview
This project is a web-based Fish Market Prediction System that performs two machine learning tasks:

Regression Task: Predicts the weight of a fish based on its physical measurements.
Classification Task: Identifies the species of a fish based on its given measurements.
The system is deployed on Heroku and provides a simple web interface for users to input fish attributes and obtain predictions.

Dataset
The system is trained on the Fish Market Dataset, which contains multiple fish species and their corresponding measurements:

Species: Categorical variable representing different fish species.
Weight: The weight of the fish in grams.
Length1, Length2, Length3: Different length measurements in cm.
Height: The height of the fish in cm.
Width: The width of the fish in cm.
Approach
Data Preprocessing

Encoded categorical variables
Applied feature scaling
Handled missing data
Model Selection

Random Forest Regressor: Predicts fish weight
Random Forest Classifier: Classifies fish species
Web Development

Built a Flask API to serve the ML model
Designed a simple HTML/CSS frontend for user input
Deployment

Hosted the model on Heroku for public access
