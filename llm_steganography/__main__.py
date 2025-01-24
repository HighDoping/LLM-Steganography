import argparse
import hashlib
import logging

from llm_steganography.aes import aes256_decrypt, aes256_encrypt
from llm_steganography.codec import (
    ReedSolomonCodec,
    byte_to_index,
    bytes_raw,
    index_to_byte,
)


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
    """Encodes a plaintext message into steganographic text using an LLM model.
    This function takes a plaintext message, optionally encrypts it, applies Reed-Solomon encoding,
    and generates natural-looking text that contains the encoded message using either an API-based
    or native LLM model.
    Args:
        plaintxt (str): The plaintext message to encode.
        starter (str): Initial text prompt to start the LLM generation.
        mode (str, optional): Generation mode - either "api" or "native". Defaults to "api".
        password (str, optional): Password for AES-256 encryption. If None, no encryption is used.
        base (int, optional): Base for encoding indices. Defaults to 16.
        char_per_index (int, optional): Number of characters used per encoded index. Defaults to 8.
        api_args (dict, optional): Additional arguments to pass to the API generation function.
        native_args (dict, optional): Additional arguments to pass to the native generation function.
    Returns:
        str: Generated text containing the encoded message.
    Raises:
        ValueError: If an invalid mode is specified.
    Example:
        >>> encoded_text = llm_encode("secret message", "Once upon a time", password="mypass")
    """
    # Convert text to bytes and encrypt if password provided
    byte_stream = (
        aes256_encrypt(password, bytes_raw(plaintxt), mode="CBC")
        if password
        else bytes_raw(plaintxt)
    )

    # Encode using Reed-Solomon
    codec = ReedSolomonCodec(8, 6)
    encoded_index = []
    encoded_index = byte_to_index(
        b"".join(codec.encode_byte_stream(byte_stream)), base=base
    )

    # Generate text using specified mode
    if mode == "api":
        from .llmapi import api_generate_text

        return api_generate_text(
            encoded_index, starter, base=base, char_per_index=char_per_index, **api_args
        )
    elif mode == "native":
        from .nativellm import native_generate_text

        return native_generate_text(
            starter,
            encoded_index,
            base=base,
            char_per_index=char_per_index,
            **native_args,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def llm_decode(encoded: str, password=None, base=16, char_per_index=8):
    """Decode a steganographically encoded message from a text string.
    This function attempts to decode a message hidden in the input string using Reed-Solomon
    error correction and optional AES-256 encryption.
    Args:
        encoded (str): The text string containing the steganographically encoded message
        password (str, optional): Password for AES-256 decryption. Defaults to None.
        base (int, optional): The base used for encoding indices. Defaults to 16.
        char_per_index (int, optional): Number of characters per encoded index. Defaults to 8.
    Returns:
        str: The decoded message as a UTF-8 string
    Raises:
        ValueError: If the message cannot be successfully decoded after trying all offsets
    Example:
        >>> encoded_text = "some encoded message"
        >>> decoded = llm_decode(encoded_text, password="mypass")
        >>> print(decoded)
        "original message"
    """
    codec = ReedSolomonCodec(8, 6)

    for offset in range(len(encoded)):
        try:
            # Split encoded string into chunks and convert to indices
            chunks = [
                encoded[i : i + char_per_index]
                for i in range(offset, len(encoded), char_per_index)
            ]
            indices = [
                int(hashlib.sha256(c.encode()).hexdigest(), 16) % base for c in chunks
            ]
            # Decode byte stream
            byte_stream = index_to_byte(indices, base=base)
            decoded = codec.decode_byte_stream(byte_stream)
            # Decrypt if password provided
            if password:
                decoded = aes256_decrypt(password, decoded, mode="CBC")

            return decoded.decode(encoding="utf-8")

        except Exception as e:
            logging.debug(e, exc_info=True)
            continue

    raise ValueError("Failed to decode message")


def save_to_file(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to :{filename}"


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
    parser.add_argument("--debug", action="store_true", help="debug mode")
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

    args = parser.parse_args()

    # set logging
    if args.debug == True:
        logging.basicConfig(level=logging.DEBUG)
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
        print(f"Encoded message: {result}")
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
