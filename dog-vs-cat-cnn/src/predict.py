import torch
from PIL import Image


def predict(model, image, transform, classes, device): 
    
    model = model.to(device)
    
    image = Image.open(image).convert('RGB')
    
    image = transform(image)
    
    image = image.unsqueeze(0)
    
    image = image.to(device)
    
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=1).item()
        
    return classes[predicted]