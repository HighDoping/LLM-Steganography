import requests


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
