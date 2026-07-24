import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

DEFAULT_CONFIG = {
    "learning_rate": 0.0001,
    "epochs": 10,
    "weight_decay": 1e-4,
    "batch_size": 32
}


def define_hyperparameter(model, config=DEFAULT_CONFIG):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    return criterion, optimizer, config

def data_loaders(train_dataset, test_dataset, device, config=DEFAULT_CONFIG):
    use_cuda = device.type == 'cuda'
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=use_cuda, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=use_cuda, num_workers=4, persistent_workers=True)
    
    return train_loader, test_loader
    

def train(model, criterion, optimizer, train_loader, device, config=DEFAULT_CONFIG):
    
    model = model.to(device)
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.00
        
        for images, labels in train_loader:
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            y_pred = model(images)
            
            loss = criterion(y_pred, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['epochs']}]  Loss: {avg_loss:.4f}")
        
    return model