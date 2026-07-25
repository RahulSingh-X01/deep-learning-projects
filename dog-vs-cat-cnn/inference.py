from src.model import load_pretrained_model, create_model
from src.predict import predict
from src.utils import load_model, get_device
from src.data_transformation import transform_data
from PIL import Image

CLASS_NAMES = ['Cat', 'Dog']


def load_pipeline():
    device = get_device()
    model = load_pretrained_model()
    model = create_model(model)
    model = load_model(model)
    _, transform = transform_data()
    return model, transform, device

def run_inference(image):
    model, transform, device = load_pipeline()
    prediction = predict(model, image, transform, CLASS_NAMES, device)
    return prediction