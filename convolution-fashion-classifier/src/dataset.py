import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class FashionMnistDataset(Dataset):
    def __init__(self, csv_path, transforms, augment=False):
        
        self.transforms = transforms
        self.augment = augment
        
        df = pd.read_csv(csv_path)
        
        X = df.iloc[:, 1:].to_numpy() / 225.0
        y = df.iloc[:, 0].to_numpy()
        
        self.features = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.labels = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        images = self.features[idx]
        labels = self.labels[idx]
        
        if self.augment:
            images = self.transforms.train(images)
        else:
            images = self.transforms.eval(images)
            
        return images, labels
    

class FashionMnistDataModule:
    def __init__(self, config, transforms):
        self.config = config
        self.transforms = transforms
        
        self.train_loader = None
        self.test_loader = None
        
    def setup(self):
        # Build Datasets
        train_dataset = FashionMnistDataset(
            csv_path = self.config.train_path,
            transforms = self.transforms,
            augment = True
        )
        test_dataset = FashionMnistDataset(
            csv_path = self.config.test_path,
            transforms = self.transforms,
            augment = False
        )
        
        # Build DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size  = self.config.batch_size,
            shuffle     = True,
            num_workers = self.config.num_workers,
            pin_memory  = True,
            persistent_workers = True,
            drop_last   = True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size  = self.config.batch_size,
            shuffle     = False,
            num_workers = self.config.num_workers,
            pin_memory  = True,
            persistent_workers = True
        )