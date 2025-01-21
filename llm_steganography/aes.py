import base64
import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def aes256_encrypt(password: str, plaintext: bytes, mode="CBC") -> bytes:
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

    if mode == "CBC":
        # Generate a random IV for CBC mode
        iv = os.urandom(16)

        # Create the AES cipher in CBC mode
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Pad the plaintext to a multiple of the block size (16 bytes)
        pad_length = 16 - len(plaintext) % 16
        padded_text = plaintext + bytes([pad_length] * pad_length)

        # Encrypt the data
        ciphertext = encryptor.update(padded_text) + encryptor.finalize()

        # Return the base64-encoded salt, IV, and ciphertext as a single string
        encrypted_data = base64.b64encode(salt + iv + ciphertext)

    elif mode == "GCM":
        # Generate a random nonce (12 bytes is standard for GCM)
        nonce = os.urandom(12)

        # Create the AES cipher in GCM mode
        cipher = Cipher(
            algorithms.AES(key), modes.GCM(nonce), backend=default_backend()
        )
        encryptor = cipher.encryptor()

        # Encrypt the data
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        # Include the GCM authentication tag
        tag = encryptor.tag

        # Return the base64-encoded salt, nonce, tag, and ciphertext as a single string
        encrypted_data = base64.b64encode(salt + nonce + tag + ciphertext)

    else:
        raise ValueError("Invalid mode selected. Use 'CBC' or 'GCM'.")

    return encrypted_data


def aes256_decrypt(password: str, encrypted_data: str, mode="CBC") -> bytes:
    # Decode the base64-encoded data
    encrypted_data = base64.b64decode(encrypted_data)

    # Extract the salt and other components based on the mode
    salt = encrypted_data[:16]

    if mode == "CBC":
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
        decrypted_data = decrypted_data[:-pad_length]

    elif mode == "GCM":
        nonce = encrypted_data[16:28]  # Nonce is 12 bytes
        tag = encrypted_data[28:44]  # Tag is 16 bytes
        ciphertext = encrypted_data[44:]

        # Derive the encryption key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        key = kdf.derive(password.encode())

        # Create the AES cipher in GCM mode for decryption
        cipher = Cipher(
            algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend()
        )
        decryptor = cipher.decryptor()

        # Decrypt the ciphertext
        decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()

    else:
        raise ValueError("Invalid mode selected. Use 'CBC' or 'GCM'.")

    return decrypted_data
