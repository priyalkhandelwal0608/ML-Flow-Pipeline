import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path="data/students.csv"):
    df = pd.read_csv(path)

    # Drop StudentID if present to avoid noise
    if "StudentID" in df.columns:
        df = df.drop("StudentID", axis=1)

    # Encode categorical features into numerical values
    le = LabelEncoder()
    # Note: In a production environment, you should save the 'le' object 
    # to use the same mapping during inference.
    for col in ['ExtracurricularActivities', 'PlacementTraining']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            
    # Encode target variable
    if 'PlacementStatus' in df.columns:
        df['PlacementStatus'] = le.fit_transform(df['PlacementStatus'])
        X = df.drop("PlacementStatus", axis=1)
        y = df["PlacementStatus"]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    return df # Return full df if used for inference