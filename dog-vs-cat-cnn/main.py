from src.utils import set_seed, save_model, get_device
from src.data_transformation import transform_data
from src.data_ingestion import load_data
from src.train import data_loaders, define_hyperparameter, train
from src.model import load_pretrained_model, create_model
from src.evaluate import evaluate

# set random_seed = 42 for reproducibility
set_seed()

# get gpu drivers
device = get_device()

# load train/test transform
train_transform, test_transform = transform_data()

# load datasets
train_dataset, test_dataset = load_data(train_transform, test_transform)

# create data loaders
train_loader, test_loader = data_loaders(train_dataset, test_dataset, device)

# load pretrained efficientnet_b0 model
model = load_pretrained_model()

# replace classifier
model = create_model(model)

# move model to device
model = model.to(device)

# create criterion and optimizer
criterion, optimizer, config = define_hyperparameter(model)

# train model
model = train(model, criterion, optimizer, train_loader, device)

# save model
save_model(model)

# evaluate on test data
test_accuracy = evaluate(model, test_loader, device)