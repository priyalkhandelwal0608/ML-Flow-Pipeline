import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path="data/students.csv"):
    df = pd.read_csv(path)

    # Drop StudentID if present
    if "StudentID" in df.columns:
        df = df.drop("StudentID", axis=1)

    # Encode categorical features
    le = LabelEncoder()
    df['ExtracurricularActivities'] = le.fit_transform(df['ExtracurricularActivities'])
    df['PlacementTraining'] = le.fit_transform(df['PlacementTraining'])
    df['PlacementStatus'] = le.fit_transform(df['PlacementStatus'])  # target

    X = df.drop("PlacementStatus", axis=1)
    y = df["PlacementStatus"]

    return train_test_split(X, y, test_size=0.2, random_state=42)