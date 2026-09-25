import torch
import torch.nn as nn

class FashionMnistModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(config.in_channel, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
         
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(128*7*7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            
            nn.Linear(64, config.num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x