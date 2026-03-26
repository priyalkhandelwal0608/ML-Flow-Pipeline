# Student Placement Prediction System

This repository contains an end-to-end MLOps pipeline designed to predict student placement status using a Random Forest Classifier. The system includes data preprocessing, model training with experiment tracking, and a containerized Flask web application for real-time inference.

---
## 🚀 Features

### 🔹 Automated Preprocessing
- Handles categorical encoding for features like `ExtracurricularActivities` and `PlacementTraining`.

### 🔹 Machine Learning Pipeline
- Trains a `RandomForestClassifier` model.
- Exports the trained model for production use.

### 🔹 Dual Inference Modes
- **Web UI**  
  - Interactive form for single-student predictions.

- **Batch Processing**  
  - Processes entire CSV files using `inference.py`.

### 🔹 Containerized Deployment
- Ready for deployment using **Docker** and **Gunicorn**.
---
## 📦 Dependencies

### 🔹 Data Processing
- `pandas`
- `numpy`

### 🔹 Machine Learning
- `scikit-learn`

### 🔹 Web Framework
- `flask`
- `gunicorn`

### 🔹 MLOps & Experiment Tracking
- `mlflow`
- `dvc`
---
##  Installation and run
- pip install -r requirements.txt
- python src/train.py
- python app.py
---
##  Project Structure

The project is organized into the following directory structure:

```text
.
├── app.py                # Flask Web Application
├── Dockerfile            # Containerization configuration
├── requirements.txt      # Python dependencies
├── data/                 # Dataset storage
│   ├── students.csv      # Training data
│   └── new_students.csv  # Batch inference data
├── models/               # Serialized model artifacts
│   └── model.pkl         # Trained Random Forest model
├── src/                  # Core logic scripts
│   ├── preprocess.py     # Data cleaning and encoding
│   ├── train.py          # Model training logic
│   ├── evaluate.py       # Performance metrics
│   └── inference.py      # Batch processing logic
├── static/               # CSS and frontend assets
│   └── style.css
└── templates/            # HTML templates
    └── index.html        # Main web interface
