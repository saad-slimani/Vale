from PIL import Image
import numpy as np
import pydicom

def load_dicom(file):
    try:
        dicom = pydicom.dcmread(file)
        image = dicom.pixel_array
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255  # Normalize to [0, 255]
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")  # Convert to RGB
    except Exception as e:
        raise ValueError(f"Error loading DICOM file: {e}")