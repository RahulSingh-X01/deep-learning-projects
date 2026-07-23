from pathlib import Path
from torchvision.datasets import ImageFolder

def load_data(train_transform, test_transform):
    
    project_dir = Path(__file__).resolve().parent.parent
    train_path = project_dir / "data" / "train"
    test_path = project_dir / "data" / "test"
    
    train_dataset = ImageFolder(
        root=train_path,
        transform=train_transform
    )
    
    test_dataset = ImageFolder(
        root=test_path,
        transform=test_transform
    )
    
    return train_dataset, test_dataset