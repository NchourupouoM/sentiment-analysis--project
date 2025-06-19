# scripts/evaluate.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import json
import os

# Créer un dossier pour les métriques
os.makedirs('metrics', exist_ok=True)

# 1. Charger le modèle et les données de test
print("Loading model and test data...")
model = joblib.load('models/logistic_regression_model.joblib')
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
print("Data loaded.")

# 2. Faire des prédictions
y_pred = model.predict(X_test)

# 3. Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# 4. Sauvegarder la métrique dans un fichier JSON
# C'est crucial pour que le workflow GitHub Actions puisse lire le score
metrics = {'accuracy': accuracy}
with open('metrics/metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Metrics saved to metrics/metrics.json")