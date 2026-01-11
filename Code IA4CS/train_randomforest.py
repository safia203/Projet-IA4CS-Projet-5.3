import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Étape 2: Entraînement du modèle Random Forest
data = pd.read_csv("ember_subset.csv")
X = data.drop(columns=["Label"])
y = data["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle configuré pour PC à ressources limitées
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Sauvegarde pour l'attaque
joblib.dump(model, "model_rf.pkl")
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print(f" Modèle entraîné. Accuracy: {model.score(X_test, y_test)*100:.2f}%")