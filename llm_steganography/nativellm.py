# %%
import hashlib
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# %%
def custom_token_selection(
    logits,
    index_needed: int,
    tokenizer,
    top_k=500,
    temperature=1.0,
    base=16,
    char_per_index=8,
):
    """
    Custom token selection algorithm.
    Selects tokens based on a combination of top-k sampling and temperature scaling.
    """
    logits = logits / temperature  # Apply temperature scaling
    probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
    # Perform top-k filtering
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
    normalized_probs = top_k_probs / torch.sum(
        top_k_probs
    )  # Normalize top-k probabilities

    for i in range(top_k):
        token = top_k_indices[i].item()
        text = tokenizer.decode(token, skip_special_tokens=True)
        # get first n characters
        text = text[:char_per_index]
        # get hash
        hash = hashlib.sha256(text.encode(encoding="utf-8")).hexdigest()
        hash_int = int(hash, 16)
        n = hash_int % base
        if n == index_needed:
            return token, text
    # If no token is selected, raise an exception
    raise ValueError("No token selected")


def native_generate_text(
    prompt: str,
    index_list: list[int],
    model_name="Qwen/Qwen2.5-0.5B",
    top_k=500,
    temperature=1.0,
    base=16,
    char_per_index=8,
) -> str:
    """
    Generate text from the model using a custom token selection algorithm.
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_built()
        else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Load a tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.to(device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.eval()

    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    pbar = tqdm(index_list, desc="Native Endoding", leave=False)
    for index in pbar:
        with torch.no_grad():
            # Perform forward pass
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]  # Extract logits of the last token
        # Select the next token using the custom algorithm
        next_token, text = custom_token_selection(
            logits.squeeze(),
            index,
            tokenizer,
            top_k=top_k,
            temperature=temperature,
            base=base,
            char_per_index=char_per_index,
        )
        generated_text += text
        generated_ids = tokenizer.encode(generated_text, return_tensors="pt").tolist()
        # Append the new token and update the input
        input_ids = torch.tensor([generated_ids], device=device)

        pbar.set_postfix_str(f"got: {text}, index:{index}")
        # Stop if end-of-sequence token is generated
        if next_token == tokenizer.eos_token_id:
            break
    return generated_text
