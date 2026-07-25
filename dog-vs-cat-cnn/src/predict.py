import torch
from PIL import Image
import torch.nn.functional as F


def predict(model, image, transform, classes, device):

    model = model.to(device)
    
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert('RGB')

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    model.eval()

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    return classes[predicted.item()], confidence.item() * 100