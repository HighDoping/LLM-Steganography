# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_built()
    else "cpu"
)
print(device)

# %%
# Load a tokenizer and model
model_name = "gpt2"  # Replace with the model you want to use
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Use MPS (Metal Performance Shaders) backend for Mac M-series acceleration

model.to(device)


# %%
def custom_token_selection(logits, top_k=5, temperature=1.0):
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

    # Sample a token from the top-k probabilities
    sampled_index = torch.multinomial(normalized_probs, num_samples=1).item()
    selected_token = top_k_indices[sampled_index].item()

    return selected_token


def generate_text(prompt, max_length=50, top_k=5, temperature=1.0):
    """
    Generate text from the model using a custom token selection algorithm.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.eval()

    generated_ids = input_ids[0].tolist()
    for _ in range(max_length):
        with torch.no_grad():
            # Perform forward pass
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]  # Extract logits of the last token

        # Select the next token using the custom algorithm
        next_token = custom_token_selection(
            logits.squeeze(), top_k=top_k, temperature=temperature
        )
        generated_ids.append(next_token)

        # Append the new token and update the input
        input_ids = torch.tensor([generated_ids], device=device)

        # Stop if end-of-sequence token is generated
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# Example usage
if __name__ == "__main__":
    prompt = "Once upon a time"
    generated_text = generate_text(prompt, max_length=100, top_k=10, temperature=0.7)
    print("Generated Text:")
    print(generated_text)
