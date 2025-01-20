# %%
import argparse
import hashlib
import logging

import reedsolo
from aes import aes256_decrypt, aes256_encrypt
from codec import ReedSolomonCodec, byte_to_index, bytes_raw, index_to_byte
from llmapi import create_completion
from tqdm import tqdm


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


def generate_sentence(string, cut=4, **api_args) -> str:
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


def llm_encode(
    plaintxt: str, starter: str, password=None, base=16, char_per_index=8, **api_args
):
    # Encrypt the plaintext with a password
    byte_stream = bytes_raw(plaintxt)
    if password is not None:
        byte_stream = aes256_encrypt(password, byte_stream, mode="CBC")
    # use reed solomon to encode
    codec = ReedSolomonCodec(8, 6)
    encoded_list = codec.encode_byte_stream(byte_stream)
    # convert to base64 index
    encoded_index = []
    for chunk in encoded_list:
        encoded_index.extend(byte_to_index(chunk, base=base))

    result = []
    result.append(starter)

    pbar = tqdm(encoded_index, desc="Encoding", leave=False)
    for index in pbar:
        result_str = "".join(result)
        n = -1
        while n != index:
            new_string = generate_sentence(result_str, cut=char_per_index, **api_args)
            hash = hashlib.sha256(new_string.encode(encoding="utf-8")).hexdigest()
            hash_int = int(hash, 16)
            n = hash_int % base
            pbar.set_postfix_str(f"trying: {new_string}, n:{n}, index:{index}")
        result.append(new_string)
        print("".join(result))
    return "".join(result)


def llm_decode(encoded: str, password=None, base=16, char_per_index=8):
    codec = ReedSolomonCodec(8, 6)
    offset = 0
    while offset < len(encoded):
        try:
            # seperate the string to a list
            encoded_list = []
            for i in range(offset, len(encoded), char_per_index):
                encoded_list.append(encoded[i : i + char_per_index])
            # convert to index
            for i in range(len(encoded_list)):
                encoded_list[i] = (
                    int(
                        hashlib.sha256(
                            encoded_list[i].encode(encoding="utf-8")
                        ).hexdigest(),
                        16,
                    )
                    % base
                )
            # basex to byte
            byte_stream = index_to_byte(encoded_list, base=base)
            decoded_byte_stream = codec.decode_byte_stream(byte_stream)
            if password is not None:
                decoded_byte_stream = aes256_decrypt(
                    password, decoded_byte_stream, mode="CBC"
                )
            decrype_str = decoded_byte_stream.decode(encoding="utf-8")
            break
        except Exception as e:
            offset += 1
    return decrype_str


def save_to_file(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to :{filename}"


# %%
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["encode", "decode"], help="encode or decode")
    parser.add_argument("filename", help="file to encode or decode")
    parser.add_argument("output", help="output file")
    parser.add_argument("--starter", default="我", help="starter text")
    parser.add_argument("--password", help="password for encryption")
    parser.add_argument("--base", type=int, default=16, help="base for encoding")
    parser.add_argument("--char_per_index", type=int, default=8, help="char per index")
    parser.add_argument(
        "--baseurl", default="http://localhost:1234/v1", help="base url"
    )
    parser.add_argument("--apikey", default="lm-studio", help="api key")
    parser.add_argument("--model", default="qwen2.5-0.5b", help="model for completion")
    args = parser.parse_args()
    if args.mode == "encode":
        with open(args.filename, "r", encoding="utf-8") as f:
            text = f.read()
        result = llm_encode(
            text,
            password=args.password,
            base=args.base,
            char_per_index=args.char_per_index,
            starter=args.starter,
            base_url=args.baseurl,
            api_key=args.apikey,
            model=args.model,
        )
        print(result)
        print(save_to_file(result, args.output))
    elif args.mode == "decode":
        with open(args.filename, "r", encoding="utf-8") as f:
            text = f.read()
        result = llm_decode(
            text,
            password=args.password,
            base=args.base,
            char_per_index=args.char_per_index,
        )
        print(f"Decoded message: {result}")
        print(save_to_file(result, args.output))
