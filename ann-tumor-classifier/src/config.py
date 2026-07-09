# config.py
import torch

DATA_PATH = "data/breast_data.csv"
MODEL_SAVE_PATH = "artifacts/model.pth"
SCALER_SAVE_PATH = "artifacts/scaler.pkl"

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
SEED = 42