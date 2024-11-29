import streamlit as st
import torch
from transformers import AutoFeatureExtractor, AutoModel, PreTrainedTokenizerFast
from PIL import Image
import pydicom
import numpy as np
from torchvision import transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CXRMate-Single-TF model, tokenizer, and feature extractor
model_name = "aehrc/cxrmate-single-tf"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Define image preprocessing transforms
image_transforms = transforms.Compose([
    transforms.Resize(size=(384, 384)),  # Resize to fixed dimensions
    transforms.ToTensor(),
    transforms.Normalize(
        mean=feature_extractor.image_mean,  # Use feature extractor for normalization
        std=feature_extractor.image_std
    ),
])

# Function to load and preprocess DICOM files
def load_dicom(file):
    try:
        dicom = pydicom.dcmread(file)
        image = dicom.pixel_array
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255  # Normalize to [0, 255]
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")  # Convert to RGB
        return image
    except Exception as e:
        raise ValueError(f"Error loading DICOM file: {e}")

# Function to preprocess uploaded image files
def preprocess_image(image):
    try:
        transformed_image = image_transforms(image)
        return transformed_image.unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

# Function to generate the report using CXRMate-Single-TF
def generate_report(image_tensor):
    try:
        pixel_values = image_tensor

        # Generate the report
        outputs = model.generate(
            pixel_values=pixel_values,
            special_token_ids=[tokenizer.sep_token_id],  # Separator token
            bos_token_id=tokenizer.bos_token_id,         # Beginning of sequence token
            eos_token_id=tokenizer.eos_token_id,         # End of sequence token
            pad_token_id=tokenizer.pad_token_id,         # Padding token
            return_dict_in_generate=True,
            use_cache=True,
            max_length=256,  # Limit maximum length for concise reports
            num_beams=4      # Use beam search for better results
        )

        # Decode the generated sequences into a readable report
        decoded_report = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return decoded_report
    except Exception as e:
        raise ValueError(f"Error generating report: {e}")

# Streamlit app interface
st.title("Vale - CXRMate-Single-TF Chest Radiograph Report Generator")

# Upload DICOM or image files
uploaded_file = st.file_uploader("Upload a DICOM or Image File", type=["dcm", "jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Load and preprocess the image
        if uploaded_file.name.endswith(".dcm"):
            image = load_dicom(uploaded_file)
        else:
            image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)
        preprocessed_image = preprocess_image(image)

        # Generate and display the report
        report = generate_report(preprocessed_image)
        st.write("Generated Report:")
        st.markdown(f"**Report:** {report}")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
