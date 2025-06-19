# scripts/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Créer le dossier pour les modèles si il n'existe pas
os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# 1. Charger les données
df = pd.read_csv('data/heart.csv')

# 2. Prétraitement simple
# Séparer les features (X) et la target (y)
X = df.drop('target', axis=1)
y = df['target']

# Diviser les données en ensembles d'entraînement et de test
# Nous utilisons un random_state pour la reproductibilité
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarder les données de test pour l'évaluation séparée
X_test.to_csv('data/processed/X_test.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)
X_train.to_csv('data/processed/X_train.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)

# 3. Entraîner le modèle
print("Training model...")
model = LogisticRegression(max_iter=1000) # Augmenter max_iter pour la convergence
model.fit(X_train, y_train)
print("Model trained.")

# 4. Sauvegarder le modèle entraîné
joblib.dump(model, 'models/logistic_regression_model.joblib')
print("Model saved to models/logistic_regression_model.joblib")