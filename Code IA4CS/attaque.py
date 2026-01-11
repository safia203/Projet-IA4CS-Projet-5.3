import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Chargement du modèle et des données de test
# On charge le modèle entraîné précédemment et les données sauvegardées
model = joblib.load("model_rf.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# 2. Sélection de 100 échantillons de malwares (Mission 8)
# On ne cible que les fichiers qui sont réellement des malwares (Label == 1)
malware_idx = np.where(y_test == 1)[0][:100]
X_malware = X_test.iloc[malware_idx]

# 3. Génération de l'attaque avec un bruit Gaussien (Mission 15 & 16)
# On augmente sigma à 0.5 pour garantir un Taux d'évasion > 30%
sigma = 0.5 
noise = np.random.normal(0, sigma, size=X_malware.shape) 
X_adv = X_malware + noise

# Assurer que les données restent dans les limites valides du dataset
X_adv = np.clip(X_adv, X_test.min().min(), X_test.max().max())

# 4. Évaluation de la robustesse (Mission 18 & 19)
# On demande au modèle de prédire sur les exemples modifiés (Adversarial Examples)
y_adv_pred = model.predict(X_adv)

# Calcul du nombre d'évasions (Malwares classés comme sains '0')
evasions = np.sum(y_adv_pred == 0)
drop_rate = (evasions / 100) * 100

print(f"--- RÉSULTATS DE L'ATTAQUE (Sigma={sigma}) ---")
print(f"Nombre de malwares ayant trompé le modèle: {evasions} / 100")
print(f"Taux d'évasion (Drop Accuracy): {drop_rate}%")

# 5. Visualisation des features les plus sensibles (Mission 20)
# Ce graphique montre quelles caractéristiques facilitent l'évasion
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[indices], color='firebrick')
plt.yticks(range(10), [X_test.columns[i] for i in indices])
plt.title(f"Top 10 Features les plus sensibles (Sigma {sigma})")
plt.xlabel("Importance Score")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()