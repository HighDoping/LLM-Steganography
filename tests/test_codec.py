from llm_steganography import codec


def test_bytes_raw():
    plaintext = "Hello, World!"
    byte_stream = codec.bytes_raw(plaintext)
    assert byte_stream == b"Hello, World!"


def test_byte_to_index_base2():
    byte_stream = b"\xab"
    base = 2
    indices = codec.byte_to_index(byte_stream, base=base)
    assert indices == [1, 0, 1, 0, 1, 0, 1, 1]


def test_index_to_byte_base2():
    indices = [1, 0, 1, 0, 1, 0, 1, 1]
    base = 2
    byte_stream = codec.index_to_byte(indices, base=base)
    assert byte_stream == b"\xab"


def test_byte_to_index_base4():
    byte_stream = b"\xab"
    base = 4
    indices = codec.byte_to_index(byte_stream, base=base)
    assert indices == [2, 2, 2, 3]


def test_index_to_byte_base4():
    indices = [2, 2, 2, 3]
    base = 4
    byte_stream = codec.index_to_byte(indices, base=base)
    assert byte_stream == b"\xab"


def test_byte_to_index_base8():
    byte_stream = b"\xab"
    base = 8
    indices = codec.byte_to_index(byte_stream, base=base)
    assert indices == [2, 5, 3]


def test_index_to_byte_base8():
    indices = [2, 5, 3]
    base = 8
    byte_stream = codec.index_to_byte(indices, base=base)
    assert byte_stream == b"\xab"


def test_byte_to_index_base16():
    byte_stream = b"\xab"
    base = 16
    indices = codec.byte_to_index(byte_stream, base=base)
    assert indices == [10, 11]


def test_index_to_byte_base16():
    indices = [10, 11]
    base = 16
    byte_stream = codec.index_to_byte(indices, base=base)
    assert byte_stream == b"\xab"


def test_byte_to_index_base32():
    byte_stream = b"\xab\xcd"
    base = 32
    indices = codec.byte_to_index(byte_stream, base=base)
    assert indices == [21, 15, 6, 16]


def test_index_to_byte_base32():
    indices = [21, 15, 6, 16]
    base = 32
    byte_stream = codec.index_to_byte(indices, base=base)
    assert byte_stream == b"\xab\xcd"


def test_byte_to_index_base64():
    byte_stream = b"\xab\xcd"
    base = 64
    indices = codec.byte_to_index(byte_stream, base=base)
    assert indices == [42, 60, 52]


def test_index_to_byte_base64():
    indices = [42, 60, 52]
    base = 64
    byte_stream = codec.index_to_byte(indices, base=base)
    assert byte_stream == b"\xab\xcd"


def test_byte_to_index_base85():
    byte_stream = b"\xab\xcd\xef\xab"
    base = 85
    indices = codec.byte_to_index(byte_stream, base=base)
    assert indices == [55, 18, 43, 10, 21]


def test_index_to_byte_base85():
    indices = [55, 18, 43, 10, 21]
    base = 85
    byte_stream = codec.index_to_byte(indices, base=base)
    assert byte_stream == b"\xab\xcd\xef\xab"


def test_reed_solomon_codec():
    rs_codec = codec.ReedSolomonCodec(8, 6)
    data = b"Hello, World!"
    encoded_data = rs_codec.encode_byte_stream(data)
    encoded_data = b"".join(encoded_data)
    decoded_data = rs_codec.decode_byte_stream(encoded_data)
    assert decoded_data == data
