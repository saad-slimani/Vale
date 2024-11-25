from torchvision import transforms
from transformers import AutoFeatureExtractor
from utils.config import model_name

# Feature extractor for mean and std values
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Define image preprocessing transforms
image_transforms = transforms.Compose([
    transforms.Resize(size=(384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=feature_extractor.image_mean,
        std=feature_extractor.image_std
    ),
])