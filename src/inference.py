import pickle
import pandas as pd

def load_model(model_path="models/model.pkl"):
    """Load the trained model from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_from_csv(csv_path, model_path="models/model.pkl"):
    """Run inference on a CSV file of new student data."""
    model = load_model(model_path)
    df = pd.read_csv(csv_path)

    predictions = model.predict(df)
    df["PlacementPrediction"] = ["Placed" if p == 1 else "NotPlaced" for p in predictions]

    return df

def predict_single(sample_dict, model_path="models/model.pkl"):
    """Run inference on a single student record (dict of features)."""
    model = load_model(model_path)
    sample_df = pd.DataFrame([sample_dict])
    prediction = model.predict(sample_df)[0]
    return "Placed" if prediction == 1 else "NotPlaced"

if __name__ == "__main__":
    # Example usage: batch inference
    results = predict_from_csv("data/new_students.csv")
    print(results)

    # Example usage: single inference
    sample = {
        "CGPA": 8.2,
        "Internships": 1,
        "Projects": 2,
        "Workshops/Certifications": 2,
        "AptitudeTestScore": 85,
        "SoftSkillsRating": 4.5,
        "ExtracurricularActivities": 1,
        "PlacementTraining": 1,
        "SSC_Marks": 75,
        "HSC_Marks": 82
    }
    print("Single Prediction:", predict_single(sample))