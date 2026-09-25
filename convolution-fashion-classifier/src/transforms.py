from torchvision.transforms import transforms

class FashionMnistTransforms:
    
    MEAN = [0.2860]
    STD  = [0.3530]
    
    def __init__(self):
        
        self.train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Normalize(self.MEAN, self.STD)
        ])
        
        self.eval = transforms.Compose([
            transforms.Normalize(self.MEAN, self.STD)
        ])