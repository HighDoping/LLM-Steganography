# %%
import argparse
import hashlib
import logging

from .aes import aes256_decrypt, aes256_encrypt
from .codec import ReedSolomonCodec, byte_to_index, bytes_raw, index_to_byte


def llm_encode(
    plaintxt: str,
    starter: str,
    mode="api",
    password=None,
    base=16,
    char_per_index=8,
    api_args={},
    native_args={},
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
    # generate text
    if mode == "api":
        from .llmapi import api_generate_text

        return api_generate_text(
            encoded_index, starter, base=base, char_per_index=char_per_index, **api_args
        )
    else:
        if mode == "native":
            from .nativellm import native_generate_text

            return native_generate_text(
                starter,
                encoded_index,
                base=base,
                char_per_index=char_per_index,
                **native_args,
            )


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
        "--encode_mode", default="api", help="encode mode, api or native"
    )
    # api args
    parser.add_argument(
        "--baseurl", default="http://localhost:1234/v1", help="api base url"
    )
    parser.add_argument("--apikey", default="lm-studio", help="api key")
    parser.add_argument(
        "--model", default="qwen2.5-0.5b", help="api model for completion"
    )
    # native args
    parser.add_argument(
        "--model_name", default="Qwen/Qwen2.5-0.5B", help="native model for completion"
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--top_k", type=int, default=500, help="top k")
    parser.add_argument(
        "--random_choice", action="store_true", help="random choice"
    )

    args = parser.parse_args()

    # set logging
    logging.basicConfig(level=logging.INFO)

    if args.mode == "encode":
        with open(args.filename, "r", encoding="utf-8") as f:
            text = f.read()
        if args.encode_mode == "api":
            api_args = {
                "baseurl": args.baseurl,
                "apikey": args.apikey,
                "model": args.model,
            }
            result = llm_encode(
                text,
                password=args.password,
                base=args.base,
                char_per_index=args.char_per_index,
                starter=args.starter,
                mode="api",
                api_args=api_args,
            )
        elif args.encode_mode == "native":
            native_args = {
                "model_name": args.model_name,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "random_choice": args.random_choice,
            }
            result = llm_encode(
                text,
                password=args.password,
                base=args.base,
                char_per_index=args.char_per_index,
                starter=args.starter,
                mode="native",
                native_args=native_args,
            )
        else:
            raise ValueError("Unknown encode mode")
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
        print(save_to_file(result, args.output))
        print(save_to_file(result, args.output))
