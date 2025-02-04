# %%
import gc
import hashlib
import logging
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def single_token_selection(logits, top_k=5, temperature=1.0):
    logits = logits / temperature  # Apply temperature scaling
    probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities

    # Perform top-k filtering
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
    normalized_probs = top_k_probs / torch.sum(
        top_k_probs
    )  # Normalize top-k probabilities

    # Perform multinomial sampling
    selected_token = top_k_indices[
        torch.multinomial(normalized_probs, num_samples=1)
    ].item()
    return selected_token


def multi_token_selection(logits, top_k=5, temperature=1.0):
    logits = logits / temperature  # Apply temperature scaling
    probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities

    # Perform top-k filtering
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
    # normalized_probs = top_k_probs / torch.sum(
    #     top_k_probs
    # )  # Normalize top-k probabilities
    # Perform multinomial sampling for all samples at once
    selected_tokens = top_k_indices.tolist()
    return selected_tokens


def generate_multiple_token(
    prompt_tokens,
    model,
    device,
    max_length=50,
    top_k=50,
    temperature=1.0,
    start_top_k=50,
):
    input_ids = prompt_tokens.to(device)
    model.eval()

    generated_ids = input_ids[0].tolist()
    generated_ids_list = []
    with torch.no_grad():
        # Perform forward pass
        input_ids = torch.tensor([generated_ids], device=device)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]  # Extract logits of the last token

        next_tokens = multi_token_selection(
            logits.squeeze(), top_k=start_top_k, temperature=temperature
        )
        for next_token in next_tokens:
            generated_ids_list.append(generated_ids + [next_token])
        for _ in range(max_length):
            for i, generated_ids in enumerate(generated_ids_list):
                input_ids = torch.tensor([generated_ids], device=device)
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]
                next_token = single_token_selection(
                    logits.squeeze(), top_k=top_k, temperature=temperature
                )
                generated_ids_list[i].append(next_token)
    # return [
    #     tokenizer.decode(generated_ids, skip_special_tokens=True)
    #     for generated_ids in generated_ids_list
    # ]
    return generated_ids_list


# %%


def multi_token_encoding(
    text_list,
    index_needed: int,
    base=16,
    char_per_index=8,
):
    for text in text_list:
        if len(text) < char_per_index:
            logging.debug("Text shorter than char_per_index")
            continue
        text = text[:char_per_index]
        hash = hashlib.sha256(text.encode(encoding="utf-8")).hexdigest()
        hash_int = int(hash, 16)
        n = hash_int % base
        logging.debug(f"index_needed:{index_needed},hash:{n},text:{text}")
        if n == index_needed:
            return text
    raise ValueError("No token found")


def native_generate_text(
    prompt: str,
    index_list: list[int],
    model="Qwen/Qwen2.5-0.5B",
    top_k=500,
    temperature=1.0,
    base=16,
    char_per_index=8,
    retry_limit=20,
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
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)

    model.to(device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.eval()

    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    for index in tqdm(index_list, desc="Native Encoding", leave=False):
        for retry in range(retry_limit):
            try:
                # run gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.mps.empty_cache()
                # Generate multiple token sequences
                input_ids = tokenizer.encode(generated_text, return_tensors="pt").to(
                    device
                )
                tokens_list = generate_multiple_token(
                    input_ids,
                    model=model,
                    device=device,
                    max_length=char_per_index + 2,
                    top_k=top_k,
                    temperature=temperature,
                    start_top_k=top_k,
                )

                # Remove prompt tokens from sequences
                # get count of elements from tensor
                input_ids_length = len(input_ids[0])
                tokens_list = [tokens[input_ids_length:] for tokens in tokens_list]
                text_list = [
                    tokenizer.decode(tokens, skip_special_tokens=True)
                    for tokens in tokens_list
                ]
                # Find token matching desired index
                selected_text = multi_token_encoding(
                    text_list,
                    index,
                    base=base,
                    char_per_index=char_per_index,
                )
                break
            except ValueError:
                if retry == retry_limit - 1:
                    raise ValueError(
                        f"No valid token found after {retry_limit} retries"
                    )
                logging.info(f"No valid token found, attempt {retry + 1}")

        generated_text += selected_text

        logging.info(f"Generated: {generated_text}")
    return generated_text
