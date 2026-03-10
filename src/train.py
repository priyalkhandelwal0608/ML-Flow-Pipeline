import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
import os
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved at models/model.pkl")