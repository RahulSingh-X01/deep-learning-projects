import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean(path):
    data = pd.read_csv(path)
    data = data.drop(columns=['Unnamed: 32', 'id'])
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test, save_path=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if not os.path.exists("ann-tumor-classifier/artifacts/scaler.pkl"): 
        joblib.dump(scaler, "ann-tumor-classifier/artifacts/scaler.pkl")

    return X_train_scaled, X_test_scaled

