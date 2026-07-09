import torch
import joblib
import pandas as pd
from src.model import BreastCancerNN


MODEL_PATH = "artifacts/model.pth"
SCALER_PATH = "artifacts/scaler.pkl"
NUM_FEATURES = 30

FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

def load_model():
    model = BreastCancerNN(NUM_FEATURES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    return model

def predict(features:list) -> str:
    X_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    scaler = joblib.load(SCALER_PATH)
    model = load_model()
    
    X_scaled = scaler.transform(X_df)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(X_tensor)
        pred = torch.argmax(output, dim=1).item()
        
    return "Malignant" if pred == 1 else "Benign"


if __name__ == "__main__":
    user_input = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
                  0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
                  0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
                  184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]

    result = predict(user_input)
    print("Prediction:", result)
    