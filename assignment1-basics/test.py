import json
import bpe_cpp

print(bpe_cpp.hello_from_bin())

# def bytes_to_unicode():
#     """
#     **FROM https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gpt2/tokenization_gpt2.py#L37**
#     Returns list of utf-8 byte and a mapping to unicode strings.
#     We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

#     The reversible bpe codes work on unicode strings.
#     This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
#     When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
#     This is a signficant percentage of your normal, say, 32K bpe vocab.
#     To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
#     """
#     # bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
#     bs = (
#         list(range(ord("!"), ord("~") + 1)) +
#         list(range(ord("¡"), ord("¬") + 1)) +
#         list(range(ord("®"), ord("ÿ") + 1))
#     )
#     cs = bs[:]
#     n = 0
#     for b in range(256):
#         if b not in bs:
#             bs.append(b)
#             cs.append(256 + n)
#             n += 1
#     cs = [chr(n) for n in cs]
#     return dict(zip(bs, cs))

# with open("model/my_tokenizer/vocab.json", encoding="utf-8") as f:
#     vocab = {k: v for k, v in json.load(f).items()}

# longest_key = max(vocab, key=len)

# b2u = bytes_to_unicode()
# u2b = {v: k for k, v in b2u.items()}

# longest_key_bytes = bytes(u2b[c] for c in longest_key)
# print(longest_key_bytes)
