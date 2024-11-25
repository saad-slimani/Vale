import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from utils.config import device, model_name

# Load the model and tokenizer
# Use AutoModelForCausalLM for text generation tasks
try:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
except Exception as e:
    raise ValueError(f"Error loading model or tokenizer: {e}")

def generate_report(image_tensor):
    """
    Generates a report from the given preprocessed image tensor using a language model.

    Args:
        image_tensor (torch.Tensor): The preprocessed image tensor (optional in this context).

    Returns:
        str: The generated report as a string.

    Raises:
        ValueError: If there is an error during the report generation process.
    """
    try:
        # Convert the input into tokenized format
        # Starting prompt for the report
        input_ids = tokenizer.encode("Report:", return_tensors="pt").to(device)

        # Ensure the model configuration includes all necessary token IDs
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        # Generate the report
        outputs = model.generate(
            input_ids=input_ids,  # Tokenized input IDs
            max_length=256,       # Maximum output length
            num_beams=4,          # Beam search for better results
            use_cache=True        # Enable caching
        )

        # Decode the output into human-readable text
        decoded_report = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_report

    except Exception as e:
        raise ValueError(f"Error generating report: {e}")
