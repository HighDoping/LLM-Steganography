# %%
import base64
import hashlib
import os
import random
import re
import zlib

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from llmapi import create_completion
from tqdm import tqdm

# %%
plaintxt = "This should be hidden.这是密文。"


# %%
def encrypt_string(password: str, plaintext: bytes) -> str:
    # Generate a random salt
    salt = os.urandom(16)

    # Derive the encryption key using PBKDF2
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    key = kdf.derive(password.encode())

    # Generate a random IV (Initialization Vector) for CBC mode
    iv = os.urandom(16)

    # Create the AES cipher in CBC mode
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Make the plaintext length a multiple of block size by padding
    pad_length = 16 - len(plaintext) % 16
    padded_text = plaintext + bytes([pad_length] * pad_length)

    # Encrypt the data
    ciphertext = encryptor.update(padded_text) + encryptor.finalize()

    # Return the base64-encoded salt, IV, and ciphertext as a single string
    encrypted_data = base64.b64encode(salt + iv + ciphertext).decode("utf-8")

    return encrypted_data


def decrypt_string(password: str, encrypted_data: str) -> bytes:
    # Decode the base64-encoded data
    encrypted_data = base64.b64decode(encrypted_data)

    # Extract the salt, IV, and ciphertext
    salt = encrypted_data[:16]
    iv = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]

    # Derive the encryption key using PBKDF2
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    key = kdf.derive(password.encode())

    # Create the AES cipher in CBC mode for decryption
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt the ciphertext
    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()

    # Remove padding
    pad_length = decrypted_data[-1]
    return decrypted_data[:-pad_length]


# %%
def bytes_raw(input_data: str) -> bytes:
    return input_data.encode(encoding="utf-8")


def deflate_raw(input_data):
    # Compress the data with raw DEFLATE (no headers or checksums)
    compressed_data = zlib.compress(
        input_data.encode(), level=9
    )  # Max compression level
    # Remove the zlib header and checksum
    return compressed_data[2:-4]


# %%
raw = bytes_raw(plaintxt)
draw = deflate_raw(plaintxt)
print(raw)
print(draw)
print(len(raw))
print(len(draw))
print(encrypt_string("password", raw))
print(decrypt_string("password", encrypt_string("password", raw)))

# %%


def byte_stream_to_base16_index_list(byte_stream):
    base16_encoded = base64.b16encode(byte_stream).decode("utf-8").rstrip("=")
    base16_alphabet = "0123456789ABCDEF"
    return [base16_alphabet.index(c) for c in base16_encoded]


def byte_stream_to_base32_index_list(byte_stream):
    base32_encoded = base64.b32encode(byte_stream).decode("utf-8").rstrip("=")
    base32_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    return [base32_alphabet.index(c) for c in base32_encoded]


def byte_stream_to_base64_index_list(byte_stream):
    base64_encoded = base64.b64encode(byte_stream).decode("utf-8").rstrip("=")
    base64_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    return [base64_alphabet.index(c) for c in base64_encoded]


def byte_stream_to_base85_index_list(byte_stream):
    try:
        base85_encoded = base64.b85encode(byte_stream).decode("ascii")
        # Use the standard ASCII85/base85 alphabet
        base85_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"
        return [base85_alphabet.index(c) for c in base85_encoded]
    except ValueError as e:
        raise ValueError(f"Invalid character in base85 encoding: {e}")
    except Exception as e:
        raise Exception(f"Error processing byte stream: {e}")


# %%
index_list = byte_stream_to_base85_index_list(
    bytes_raw(encrypt_string("password", draw))
)
print(index_list)
print(len(index_list))
# index_list=byte_stream_to_base85_index_list(raw)
# print(index_list)
# %%
result_string = "白い雪"


# separate the string with punctuation using re and return with punctuation
def generate_sentence(string):
    while True:
        new_string = create_completion(string)
        # sentences = re.split(r"(?<=[.?!。？！;\n,])\s*", new_string)
        # if len(sentences) > 1:
        # return sentences[0]
        # return first 4 char
        if len(new_string) >= 4:
            return new_string[:4]


for index in tqdm(index_list, desc="Processing indices"):
    n = -1
    while n != index:
        new_string = generate_sentence(result_string)
        # hash new_string with sha-256 and mod 64
        hash = int(hashlib.sha256(new_string.encode("utf-8")).hexdigest(), 16)
        n = hash % 85
    result_string += new_string
    print(f"Get token:{new_string},index:{n}")
print(result_string)

decrypt_str = result_string
# %%
