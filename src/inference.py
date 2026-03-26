import pickle
import pandas as pd
import os

def load_model(model_path="models/model.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_from_csv(csv_path, model_path="models/model.pkl"):
    if not os.path.exists(csv_path):
        return "Error: CSV file not found."
    
    model = load_model(model_path)
    df = pd.read_csv(csv_path)
    
    # Ensure features match training columns
    predictions = model.predict(df)
    df["PlacementPrediction"] = ["Placed" if p == 1 else "NotPlaced" for p in predictions]
    return df

if __name__ == "__main__":
    # Example: Run batch inference
    # results = predict_from_csv("data/new_students.csv")
    # print(results)
    pass