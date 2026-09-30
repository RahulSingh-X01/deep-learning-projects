import torch
import random
import numpy as np
import os

class Utils:
    
    @staticmethod
    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)      # for multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"Seed set to: {seed}")
    
    @staticmethod
    def get_device():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        return device
    
    @staticmethod
    def save_model(model, path: str = 'checkpoints/model.pth'):
        # Create checkpoints folder if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Model saved to: {path}")
    
    @staticmethod
    def load_model(model, path: str = 'checkpoints/model.pth'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at: {path}")
        
        device = Utils.get_device()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"Model loaded from: {path}")
        return model