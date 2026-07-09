import torch 
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

def get_loaders(X_train, X_test, y_train, y_test, batch_size):
    
    train_data = CustomDataset(
        torch.from_numpy(X_train.astype('float32')),
        torch.tensor(y_train.to_numpy(), dtype=torch.long)
    )
    test_data = CustomDataset(
        torch.from_numpy(X_test.astype('float32')),
        torch.tensor(y_test.to_numpy(), dtype=torch.long)
    )
    X_train_loaders = DataLoader(train_data, batch_size=32, shuffle=True)
    X_test_loaders = DataLoader(test_data, batch_size=32, shuffle=False)
    
    return X_train_loaders, X_test_loaders
