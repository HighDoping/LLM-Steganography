import os

import requests

BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"
MODEL = "qwen2.5-0.5b"


def create_completion(prompt, model=MODEL, max_tokens=20, temperature=0.7):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # "stop": [",", "，", ".", "。"]
    }

    response = requests.post(f"{BASE_URL}/completions", headers=headers, json=data)

    if response.status_code == 200:
        res = response.json()
        return res["choices"][0]["text"]
        # return res
    else:
        raise Exception(
            f"API call failed with status {response.status_code}: {response.text}"
        )
