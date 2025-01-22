import hashlib

import requests
from tqdm import tqdm


def create_completion(
    prompt,
    base_url,
    api_key,
    model,
    max_tokens=20,
    temperature=0.99,
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(
        f"{base_url}/completions",
        headers=headers,
        json=data,
    )

    if response.status_code == 200:
        res = response.json()
        return res["choices"][0]["text"]
    else:
        raise Exception(
            f"API call failed with status {response.status_code}: {response.text}"
        )


# %%
# Function to split the sentence
def split_sentence(text: str):
    punctuation = {".", "!", "?", ",", "。", "，", ";", "；", "？"}
    result = []
    temp = ""

    for char in text:
        temp += char  # Add character to the current segment
        if char in punctuation:  # If it's a punctuation mark
            # Add the segment to the result and reset temp
            result.append(temp.strip())
            temp = ""

    # Add any remaining part after the last punctuation
    if temp.strip():
        result.append(temp.strip())

    return result


def api_generate_text(
    encoded_index, starter, base=16, char_per_index=8, **api_args
) -> str:
    def api_generate_sentence(string, cut=4, **api_args) -> str:
        while True:
            new_string = create_completion(string, **api_args)
            if cut <= -1:
                return new_string
            if cut == 0:
                sentences = split_sentence(new_string)
                # if len(sentences) > 1:
                return sentences[0]
            # return first 4 char
            else:
                if len(new_string) >= cut:
                    return new_string[:cut]

    result = []
    result.append(starter)

    pbar = tqdm(encoded_index, desc="API Encoding", leave=False)
    for index in pbar:
        result_str = "".join(result)
        n = -1
        while n != index:
            new_string = api_generate_sentence(
                result_str, cut=char_per_index, **api_args
            )
            hash = hashlib.sha256(new_string.encode(encoding="utf-8")).hexdigest()
            hash_int = int(hash, 16)
            n = hash_int % base
            pbar.set_postfix_str(f"trying: {new_string}, n:{n}, index:{index}")
        result.append(new_string)
    return "".join(result)
