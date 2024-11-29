import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import pydicom
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CXRMate-Single-TF model and processor
model_name = "aehrc/cxrmate-single-tf"
model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Function to load DICOM files
def load_dicom(file):
    try:
        dicom = pydicom.dcmread(file)
        image = dicom.pixel_array
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255  # Normalize to [0, 255]
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")  # Convert to RGB
        return image
    except Exception as e:
        raise ValueError(f"Error loading DICOM file: {e}")

# Function to generate the report using CXRMate-Single-TF
def generate_report(image, clinical_info=None):
    try:
        # Preprocess the inputs
        inputs = processor(images=image, text=clinical_info, return_tensors="pt").to(device)
        
        # Generate the report
        outputs = model.generate(**inputs)
        
        # Decode the generated sequences into a readable report
        report = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return report
    except Exception as e:
        raise ValueError(f"Error generating report: {e}")

# Streamlit app interface
st.title("Vale - CXRMate-Single-TF Chest Radiograph Report Generator")

# Upload DICOM or image files
uploaded_file = st.file_uploader("Upload a DICOM or Image File", type=["dcm", "jpg", "jpeg", "png"])

# Optional Clinical Information
clinical_info = st.text_area("Optional Clinical Information", placeholder="Enter any relevant clinical information here...")

if uploaded_file:
    try:
        # Load the image
        if uploaded_file.name.endswith(".dcm"):
            image = load_dicom(uploaded_file)
        else:
            image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate and display the report
        report = generate_report(image, clinical_info)
        st.write("Generated Report:")
        st.markdown(f"**Report:** {report}")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
