from sklearn.metrics import classification_report
from preprocess import load_and_preprocess
import pickle

# Load the latest trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

X_train, X_test, y_train, y_test = load_and_preprocess()

# Print detailed metrics
print("Model Performance Report:")
print(classification_report(y_test, model.predict(X_test)))