import torch
from transformers import AutoModel, PreTrainedTokenizerFast

from utils.config import device, model_name

# Load model and tokenizer
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

def generate_report(image_tensor):
    try:
        outputs = model.generate(
            pixel_values=image_tensor,
            special_token_ids=[tokenizer.sep_token_id],
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
            max_length=256,
            num_beams=4
        )
        # Decode the generated report
        return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    except Exception as e:
        raise ValueError(f"Error generating report: {e}")