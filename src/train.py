import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess

# Start MLflow tracking
mlflow.set_experiment("Student_Placement_Prediction")

with mlflow.start_run():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Model definition
    n_estimators = 100
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Log parameters and model to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.sklearn.log_model(model, "model")

    # Save local copy for the Flask app
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(" Model trained, tracked in MLflow, and saved at models/model.pkl")