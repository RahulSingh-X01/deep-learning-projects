from src.config import Config
from src.transforms import FashionMnistTransforms
from src.dataset import FashionMnistDataModule
from src.model import FashionMnistModel
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.utils import Utils
import torch

def main():
    
    Utils.set_seed(42)
    
    device = Utils.get_device()
    
    config = Config()
    
    transform = FashionMnistTransforms()
    
    dm = FashionMnistDataModule(config, transform)
    dm.setup()
    
    model = FashionMnistModel(config)
    
    trainer = Trainer(model, config, dm.train_loader, device)
    trainer.fit()
    
    Utils.save_model(model, 'checkpoints/model.pth')
    
    evaluator = Evaluator(model, config, device)
    evaluator.evaluate(dm.train_loader, "Train")
    evaluator.evaluate(dm.test_loader, "Test")
    
if __name__ == '__main__':
    main()