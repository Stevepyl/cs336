import json

with open("model/my_tokenizer/vocab.json", encoding="utf-8") as f:
    vocab = {k.encode("latin-1"): v for k, v in json.load(f).items()}


