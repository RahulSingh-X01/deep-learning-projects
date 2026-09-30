import torch
from sklearn.metrics import classification_report, confusion_matrix

class Evaluator:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        self.all_preds = []
        self.all_labels = []
        
    def _collect_predictions(self, loader):
        
        self.all_preds = []
        self.all_labels = []
        
        self.model.eval()
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                
                _, predicted = torch.max(outputs, 1)
                
                self.all_preds.extend(predicted.cpu().numpy())
                self.all_labels.extend(labels.cpu().numpy())
                
    def accuracy(self, loader):
        self._collect_predictions(loader)
        
        correct = sum(p==l for p, l in zip(self.all_preds, self.all_labels))
        
        return (correct/len(self.all_labels))*100
    
    def report(self, loader):
        self._collect_predictions(loader)
        print(classification_report(
            self.all_labels,
            self.all_preds,
            target_names = self.config.class_names
        ))
        
    def evaluate(self, loader, label=""):
        acc = self.accuracy(loader)
        print(f"\n{label} Accuracy: {acc:.2f}%")
        self.report(loader)
        
    
        
        
        