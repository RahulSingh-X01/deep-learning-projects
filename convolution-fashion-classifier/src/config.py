from dataclasses import dataclass

@dataclass
class Config:
    # Data
    train_path: str = r'/home/rahul/Projects/deep-learning-projects/convolution-fashion-classifier/data/test_data.csv'
    test_path: str  = r'/home/rahul/Projects/deep-learning-projects/convolution-fashion-classifier/data/train_data.csv'
    
    # Dataloader
    batch_size: int = 32
    num_workers: int = 4
    
    # Model
    in_channels: int = 1   # grayscale = 1 channel
    num_classes: int = 10
    
    # Training
    epochs: int       = 100
    learning_rate: float = 0.0001
    weight_decay: float  = 1e-4
    
    # Classes
    class_names: tuple = (
        'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal',  'Shirt',   'Sneaker',  'Bag',   'Ankle Boot'
    )