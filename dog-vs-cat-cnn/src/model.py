import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def load_model():
    """Load a pretrained EfficientNet-B0 model."""
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    
    return model
    

def create_model(model, num_classes=2):
    """Freeze the backbone and replace the classifier."""
    
    # dynamically extract the input features from the existing classifier
    input_features = model.classifier[-1].in_features 
    
    # freeze the feature parameters
    for param in model.features.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(
        nn.Linear(input_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.3),

        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        
        nn.Linear(128, num_classes)
    )
        
    return model    