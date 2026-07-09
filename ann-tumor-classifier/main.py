from src import config, preprocess, dataset, model as model_module, train, evaluate
import torch

X_train, X_test, y_train, y_test = preprocess.load_and_clean(config.DATA_PATH)
X_train_s, X_test_s = preprocess.scale_data(X_train, X_test, config.SCALER_SAVE_PATH)
train_loader, test_loader = dataset.get_loaders(X_train_s, X_test_s, y_train, y_test, config.BATCH_SIZE)

net = model_module.BreastCancerNN(X_train_s.shape[1])
net = train.train_model(net, train_loader, config.EPOCHS, config.LEARNING_RATE)

torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
evaluate.evaluate_model(net, test_loader)