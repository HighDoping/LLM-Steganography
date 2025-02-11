# %%
import tiktoken

enc = tiktoken.get_encoding("o200k_base")
# %%
input_tokens = [28630, 18257, 14757, 14552]
processed_tokens = enc.encode(enc.decode(input_tokens))
try:
    assert input_tokens == processed_tokens
except AssertionError:
    print("Tokens->Text->Tokens mismatch")
