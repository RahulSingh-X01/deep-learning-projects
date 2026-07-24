import os
import torch
import random
import numpy as np

def save_model(model, model_name="model.pth"):
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    save_path = os.path.join(artifacts_dir, model_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved → {save_path}")
    
    
def load_model(model, model_name="model.pth"):
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    load_path = os.path.join(artifacts_dir, model_name)

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No model found at: {load_path}")

    model.load_state_dict(torch.load(load_path, map_location="cpu"))
    model.eval()
    print(f"Model loaded ← {load_path}")
    
    return model

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed set to {seed}")
    

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")