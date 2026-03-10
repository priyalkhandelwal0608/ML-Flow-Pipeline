from sklearn.metrics import classification_report
from preprocess import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = load_and_preprocess()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))