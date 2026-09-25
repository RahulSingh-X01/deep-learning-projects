import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, config, train_loader, device):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.device = device
        
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = self.config.learning_rate,
            weight_decay = self.config.weight_deacy
        )
        
        self.loss_history = []
            
            
        
    def _train_one_epoch(self):
        
        self.model.train()
        
        epoch_loss = 0
        
        for images, labels in self.train_loader:
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss/len(self.train_loader)
    
    def fit(self):
        
        print(f"Training on: {self.device}")
        
        for epoch in range(self.config.epochs):
            
            avg_loss = self._train_one_epoch()
            
            self.loss_history.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{self.config.epochs}  Loss: {avg_loss:.4f}")
            
        print("Training complete!")
            
            
        
            
        