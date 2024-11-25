import torch
from torchvision import transforms
from utils.transforms import image_transforms
from utils.config import device

def preprocess_image(image):
    try:
        transformed_image = image_transforms(image)
        return transformed_image.unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")