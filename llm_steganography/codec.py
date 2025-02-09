import base64
import zlib

import reedsolo


def bytes_raw(input_data: str) -> bytes:
    return input_data.encode(encoding="utf-8")


def deflate_raw(input_data):
    # Compress the data with raw DEFLATE (no headers or checksums)
    compressed_data = zlib.compress(
        input_data.encode(encoding="utf-8"), level=9
    )  # Max compression level
    # Remove the zlib header and checksum
    return compressed_data[2:-4]


def byte_to_index(byte_stream: bytes, base=16) -> list[int]:
    if base == 2:
        # to binary
        base2_encoded = "".join([f"{b:0>8b}" for b in byte_stream])
        base2_alphabet = "01"
        return [base2_alphabet.index(c) for c in base2_encoded]
    if base == 4:
        # to quaternary
        base4_encoded = ""
        for b in byte_stream:
            # Convert each byte to three base-4 digits (since 8 bits require 3 base-4 digits)
            base4_encoded += (
                f"{(b >> 6) & 0x3}{(b >> 4) & 0x3}{(b >> 2) & 0x3}{b & 0x3}"
            )
        base4_alphabet = "0123"
        return [base4_alphabet.index(c) for c in base4_encoded]
    if base == 8:
        # to octal
        base8_encoded = "".join([f"{oct(b)[2:]:0>3}" for b in byte_stream])
        base8_alphabet = "01234567"
        return [base8_alphabet.index(c) for c in base8_encoded]
    if base == 16:
        base16_encoded = base64.b16encode(byte_stream).decode("utf-8").rstrip("=")
        base16_alphabet = "0123456789ABCDEF"
        return [base16_alphabet.index(c) for c in base16_encoded]
    elif base == 32:
        base32_encoded = base64.b32encode(byte_stream).decode("utf-8").rstrip("=")
        base32_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
        return [base32_alphabet.index(c) for c in base32_encoded]
    elif base == 64:
        base64_encoded = base64.b64encode(byte_stream).decode("utf-8").rstrip("=")
        base64_alphabet = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        )
        return [base64_alphabet.index(c) for c in base64_encoded]
    elif base == 85:
        base85_encoded = base64.b85encode(byte_stream).decode("ascii")
        # Use the standard ASCII85/base85 alphabet
        base85_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"
        return [base85_alphabet.index(c) for c in base85_encoded]
    else:
        raise ValueError(
            "Invalid base selected. Available options: 2, 4, 8, 16, 32, 64, 85"
        )


def index_to_byte(indices: list[int], base=16) -> bytes:
    if base == 2:
        # Convert indices back to binary-encoded string
        base2_alphabet = "01"
        base2_encoded = "".join(base2_alphabet[i] for i in indices)
        # Pad to multiple of 8 bits
        padding_length = (8 - len(base2_encoded) % 8) % 8
        base2_encoded = base2_encoded + "0" * padding_length
        # Split into 8-bit chunks and convert to bytes
        byte_stream = bytes(
            int(base2_encoded[i : i + 8], 2) for i in range(0, len(base2_encoded), 8)
        )
        return byte_stream
    elif base == 4:
        # Convert indices back to quaternary-encoded string
        base4_alphabet = "0123"
        base4_encoded = "".join(base4_alphabet[i] for i in indices)
        # Pad to multiple of 4 chars
        padding_length = (4 - len(base4_encoded) % 4) % 4
        base4_encoded = base4_encoded + "0" * padding_length
        # Combine four base-4 digits to form bytes
        byte_stream = bytes(
            (int(base4_encoded[i], 4) << 6) | (int(base4_encoded[i + 1], 4) << 4) |
            (int(base4_encoded[i + 2], 4) << 2) | int(base4_encoded[i + 3], 4)
            for i in range(0, len(base4_encoded), 4)
        )
        return byte_stream
    elif base == 8:
        # Convert indices back to octal-encoded string
        base8_alphabet = "01234567"
        base8_encoded = "".join(base8_alphabet[i] for i in indices)
        # Pad to multiple of 3 chars
        padding_length = (3 - len(base8_encoded) % 3) % 3
        base8_encoded = base8_encoded + "0" * padding_length
        # Split into 3-character chunks and convert to bytes
        byte_stream = bytes(
            int(base8_encoded[i : i + 3], 8) for i in range(0, len(base8_encoded), 3)
        )
        return byte_stream
    elif base == 16:
        # Convert indices back to base16-encoded string
        base16_alphabet = "0123456789ABCDEF"
        base16_encoded = "".join(base16_alphabet[i] for i in indices)
        # Pad to multiple of 2 chars
        padding_length = (2 - len(base16_encoded) % 2) % 2
        base16_encoded = base16_encoded + "0" * padding_length
        byte_stream = base64.b16decode(base16_encoded.upper())
        return byte_stream
    elif base == 32:
        # Convert indices back to base32-encoded string
        base32_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
        base32_encoded = "".join(base32_alphabet[i] for i in indices)
        # Pad to multiple of 8 chars
        padding_length = (8 - len(base32_encoded) % 8) % 8
        base32_encoded = base32_encoded + "=" * padding_length
        byte_stream = base64.b32decode(base32_encoded)
        return byte_stream
    elif base == 64:
        # Convert indices back to base64-encoded string
        base64_alphabet = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        )
        base64_encoded = "".join(base64_alphabet[i] for i in indices)
        # Pad to multiple of 4 chars
        padding_length = (4 - len(base64_encoded) % 4) % 4
        base64_encoded = base64_encoded + "=" * padding_length
        byte_stream = base64.b64decode(base64_encoded)
        return byte_stream
    elif base == 85:
        # Convert indices back to base85-encoded string
        base85_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"
        base85_encoded = "".join(base85_alphabet[i] for i in indices)
        # Pad to multiple of 5 chars
        padding_length = (5 - len(base85_encoded) % 5) % 5
        base85_encoded = (
            base85_encoded + "~" * padding_length
        )  # Using ~ as padding for base85
        byte_stream = base64.b85decode(base85_encoded)
        return byte_stream
    else:
        raise ValueError(
            "Invalid base selected. Available options: 2, 4, 8, 16, 32, 64, 85"
        )


class ReedSolomonCodec:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.rs = reedsolo.RSCodec(n - k)

    def encode_byte_stream(self, byte_stream: bytes) -> list[bytes]:
        """
        Encode a byte stream using Reed-Solomon codes.
        """
        encoded_stream = []
        for i in range(0, len(byte_stream), self.k):
            chunk = byte_stream[i : i + self.k]  # Take a chunk of 'k' data symbols
            # Pad the chunk if it is smaller than 'k'
            chunk = chunk.ljust(self.k, b"\x00")
            # Encode the chunk with RS
            encoded_chunk = self.rs.encode(chunk)
            encoded_stream.append(encoded_chunk)
        return encoded_stream

    def decode_byte_stream(self, encoded_stream: bytes) -> bytes:
        """
        Decode a Reed-Solomon encoded stream using bit-level sliding window.
        Handles broken blocks, bit shifts, and trailing garbage data.
        """
        decoded_stream = []
        # Convert bytes to bits for finer granularity
        bits = "".join([f"{b:08b}" for b in encoded_stream])
        pos = 0
        while pos <= len(bits) - self.n * 8:  # Ensure enough bits for a chunk
            # Convert window of bits back to bytes
            chunk_bits = bits[pos : pos + self.n * 8]
            chunk_bytes = bytes(
                int(chunk_bits[i : i + 8], 2) for i in range(0, len(chunk_bits), 8)
            )
            try:
                # Attempt to decode this chunk
                decoded_chunk = self.rs.decode(chunk_bytes)[0]
                # Only append non-empty chunks after removing padding
                cleaned_chunk = decoded_chunk.rstrip(b"\x00")
                if cleaned_chunk:
                    decoded_stream.append(cleaned_chunk)
                # Move to next chunk
                pos += self.n * 8
            except reedsolo.ReedSolomonError:
                # If decoding fails, slide forward one bit
                pos += 1
                continue

        if not decoded_stream:
            raise reedsolo.ReedSolomonError("No valid data found")

        return b"".join(decoded_stream)
