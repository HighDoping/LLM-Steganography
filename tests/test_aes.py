from llm_steganography import aes


def test_aes256_encrypt_decrypt():
    password = "password"
    plaintext = b"Hello, World!"
    encrypted = aes.aes256_encrypt(password, plaintext)
    decrypted = aes.aes256_decrypt(password, encrypted)
    assert decrypted == plaintext


def test_aes256_encrypt_decrypt_wrong_password():
    password = "password"
    wrong_password = "wrong_password"
    plaintext = b"Hello, World!"
    encrypted = aes.aes256_encrypt(password, plaintext)
    decrypted = aes.aes256_decrypt(wrong_password, encrypted)
    assert decrypted != plaintext
