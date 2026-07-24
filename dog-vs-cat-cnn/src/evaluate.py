import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, test_loader):
    model = model.to(device)
    model.eval()
    
    total = 0
    correct = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc