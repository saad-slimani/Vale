import streamlit as st
from PIL import Image
from utils.dicom_utils import load_dicom
from utils.preprocess import preprocess_image
from utils.model import generate_report
import openai  # OpenAI integration
import os  # For accessing environment variables

# Load OpenAI API key securely from environment variables
api_key = os.getenv("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")
if not api_key:
    st.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
else:
    openai.api_key = api_key

st.title("Vale - CXRMate-Single-TF Chest Radiograph Report Generator")

# Function to format the report using GPT-4
def format_with_gpt4(report):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert radiologist. Rewrite this report in a structured and precise fashion."},
                {"role": "user", "content": report}
            ]
        )
        # Extract and return the response content
        return response.choices[0].message["content"]
    except openai.error.OpenAIError as e:
        # Generic OpenAI error handling
        st.error(f"OpenAI API Error: {str(e)}")
    except Exception as e:
        # Catch-all for unexpected errors
        st.error(f"Unexpected error formatting the report: {str(e)}")
    return report

# Upload DICOM or image files
uploaded_file = st.file_uploader("Upload a DICOM or Image File", type=["dcm", "jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Processing..."):
        try:
            # Load and preprocess the image
            if uploaded_file.name.lower().endswith(".dcm"):
                image = load_dicom(uploaded_file)
            else:
                image = Image.open(uploaded_file).convert("RGB")

            st.image(image, caption="Uploaded Image", use_column_width=True)
            preprocessed_image = preprocess_image(image)

            # Generate and display the report
            raw_report = generate_report(preprocessed_image)
            st.markdown("### Raw Report:")
            st.markdown(f"**{raw_report}**")

            # Format the report using GPT-4
            if api_key:
                formatted_report = format_with_gpt4(raw_report)
                st.success("Report formatted successfully!")
                st.markdown("### Formatted Report:")
                st.markdown(f"**{formatted_report}**")
            else:
                st.warning("Cannot format the report without a valid OpenAI API key.")

        except FileNotFoundError:
            st.error("File not found. Please upload a valid file.")
        except ValueError as e:
            st.error(f"Value Error: {str(e)}")
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")