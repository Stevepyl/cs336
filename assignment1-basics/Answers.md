# Answers of the Questions in Assignment 1

## 1.Byte-Pair Encoding (BPE) Tokenizer
### 1.1 Problem (unicode1): Understanding Unicode
(1) `chr(0)` returns `'\x00'`, which represents the character U+0000, which is the NULL character (also called "NUL").
(2) The string representation `str.__repr()__` shows the code needed to create the string and tell the interpreter its not bare characters, but the string literal is what a human wanna read it, like 'abc' and ''abc''.
(3)`chr(0)` can't be printed, but if inserted into a string, it will be string literal `\x00`

### 1.2 Problem (unicode2): Unicode Encodings
(1) Because using UTF-16/UTF-32 need much more space than UTF-8, especially for ASCII characters. And UTF-16/32 introduce more zero bytes than UTF-8, which may introduce less robustness
(2) In UTF-8, there are four byte-classes:
    - 0xxxxxxx -> 1-byte char
    - 110xxxxx -> start of 2-byte char, starts with '\xc3'
    - 1110xxxx -> start of 3-byte char. starts with '\xe3'
    - 11110xxx -> start of 4-byte char, starts with '\xf0'
    So if a string contains 2/3/4-byte char, directly decoding it byte-by-byte will raise an error as these byte themselves are incomplete.
    For example, for this `test_string = "hello! こんにちは!"`, its UTF-8 encoded string is `b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'`. Specifically,
    - `こ` is \xe3\x81\x93
    - `ん` is \xe3\x82\x93
    - `に` is \xe3\x81\xab
    - `ち` is \xe3\x81\xa1
    - `は` is \xe3\x81\xaf
    When decoding this string byte-by-byte and the function reading up to `\xe3`, it will raise an UnicodeDecodeError
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    i = 0
    chars = []
    while i < len(bytestring):
        b0 = bytestring[i] # b0 = 11100011
        # Determine number of bytes
        if b0 < 0x80:
            n = 1
            code = b0
        elif 0xC0 <= b0 < 0xE0:
            n = 2
            code = b0 & 0x1F
        elif 0xE0 <= b0 < 0xF0:  # 11100011 10000001 10010011
            n = 3
            code = b0 & 0x0F # code = 00000011
        elif 0xF0 <= b0 < 0xF8:
            n = 4
            code = b0 & 0x07
        else:
            raise ValueError("Invalid UTF-8 lead byte")

        if i + n > len(bytestring):
            raise ValueError("Truncated UTF-8 sequence")

        # Add continuation bytes
        for j in range(1, n):
            cont = bytestring[i + j]  # cont = \x81 1100 0000
            # Every non-leading UTF-8 byte (i.e., bytes after the first one in a multi-byte sequence) must begin with the bit pattern 10xxxxxx.
            if cont & 0xC0 != 0x80:
                raise ValueError("Invalid UTF-8 continuation byte")
            # appends 6 bits to the right of the existing ones.
            code = (code << 6) | (cont & 0x3F)

        chars.append(chr(code))
        i += n
    return "".join(chars)
```
(3) b"\xe3\x82", because in UTF-8, 2-byte characters must begin with 110xxxxx, and a byte seq starting with 0xe3 is 3-byte UTF-8 character
    
### 1.3 Problem (train_bpe_tinystories) BPE Training on TinyStories
(1)On Apple M1 Pro with 8 processes, it takes 150.69 seconds (140.95 seconds with 16 processes). The longest token is `'Ġaccomplishment'`. Considering of usage of `bytes_to_unicode`, it's b' accomplishment' actually
(2) The part of find the token pair which having the most occurency frequencies in the merging process takes the most time.


