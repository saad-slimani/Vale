import torch
from transformers import AutoModel, PreTrainedTokenizerFast

from utils.config import device, model_name

# Load model and tokenizer
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

def generate_report(image_tensor):
    try:
        # Convert the preprocessed image into input IDs
        input_ids = tokenizer.encode("Report:", return_tensors="pt").to(device)

        # Generate the report
        outputs = model.generate(
            input_ids=input_ids,               # Use tokenized input IDs
            max_length=256,                   # Set maximum output length
            num_beams=4,                      # Use beam search for better results
            use_cache=True                    # Enable cache for efficiency
        )

        # Decode the generated sequence into text
        decoded_report = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_report
    except Exception as e:
        raise ValueError(f"Error generating report: {e}")
