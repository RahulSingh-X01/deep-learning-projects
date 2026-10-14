import torch
from PIL import Image
import numpy as np
from src.config import Config
from src.model import FashionMnistModel
from src.transforms import FashionMnistTransforms
from src.utils import Utils

CHECKPOINT_PTH = "/home/rahul/Projects/deep-learning-projects/convolution-fashion-classifier/checkpoints/model.pth"

class Inference:
    def __init__(self, checkpoint_pth: str = CHECKPOINT_PTH):
        
        self.config = Config()
        self.transforms = FashionMnistTransforms()
        self.device = Utils.get_device()
        
        self.model = FashionMnistModel(self.config)
        self.model = Utils.load_model(self.model, checkpoint_pth)
        self.model.eval() # set eval for inference
        
    def _preprocess(self, image_pth: str):
            
            image = Image.open(image_pth).convert('L') # convert into grayscale
            
            image = image.resize((28, 28))
            
            image = np.array(image, dtype=np.float32) / 255.0
            
            image = torch.tensor(image).reshape(1, 1, 28, 28)
            
            image = self.transforms.eval(image)
            
            return image.to(self.device)
        
    def predict(self, image_pth: str):
            
            image = self._preprocess(image_pth)
            
            with torch.no_grad():
                outputs = self.model(image)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
            class_idx = predicted.item()
            class_name = self.config.class_names[class_idx] 
            confidence  = confidence.item() * 100
            
            return {
            'class_index' : class_idx,
            'class_name'  : class_name,
            'confidence'  : f"{confidence:.2f}%"
        }
            
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        print("Example: python inference.py data/sample.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    inferencer = Inference()
    result     = inferencer.predict(image_path)
    
    print(f"\nPrediction  : {result['class_name']}")
    print(f"Confidence  : {result['confidence']}")
    print(f"Class Index : {result['class_index']}")
        
        
        