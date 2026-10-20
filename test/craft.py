"""
Hand-crafted Brotli streams, to reach paths that encoders and random mutations
rarely reach: every dictionary transform on words of every length, and
header edge cases. Validity doesn't matter: expected results come from the
reference decoder.
"""
import os, re, random

class BitWriter:
    def __init__(self):
        self.bits = []

    def write(self, n, v):
        for i in range(n):
            self.bits.append((v >> i) & 1)

    def pad(self, value=0):
        while len(self.bits) % 8:
            self.bits.append(value & 1)
            value >>= 1

    def bytes(self):
        self.pad()
        out = bytearray()
        for i in range(0, len(self.bits), 8):
            b = 0
            for j in range(8):
                b |= self.bits[i + j] << j
            out.append(b)
        return bytes(out)

def window_bits(w, n):
    """Stream header for WBITS = n (10..24), or raw codes for invalid ones."""
    if n == 16:
        w.write(1, 0)
    elif n >= 18:
        w.write(1, 1); w.write(3, n - 17)
    elif n == 17:
        w.write(1, 1); w.write(3, 0); w.write(3, 0)
    else:
        w.write(1, 1); w.write(3, 0); w.write(3, n - 8)

def mlen(w, length, nibbles=4):
    w.write(2, nibbles - 4)
    w.write(4 * nibbles, length - 1)

def simple_code(w, alphabet_bits, symbols, tree_select=0):
    w.write(2, 1)
    w.write(2, len(symbols) - 1)
    for s in symbols:
        w.write(alphabet_bits, s)
    if len(symbols) == 4:
        w.write(1, tree_select)

# Insert-and-copy lengths.
INS_BITS = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 24]
COPY_BITS = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 24]
INS_OFF = [0]
COPY_OFF = [2]
for i in range(23):
    INS_OFF.append(INS_OFF[-1] + (1 << INS_BITS[i]))
    COPY_OFF.append(COPY_OFF[-1] + (1 << COPY_BITS[i]))

def length_code(offsets, bits, n):
    for c in range(len(offsets)):
        if offsets[c] <= n < offsets[c] + (1 << bits[c]):
            return c, n - offsets[c]
    raise ValueError(n)

def command_symbol(ins_code, copy_code):
    # explicit distance cells
    cells = {(0, 0): 2, (0, 8): 3, (8, 0): 4, (8, 8): 5, (0, 16): 6,
             (16, 0): 7, (8, 16): 8, (16, 8): 9, (16, 16): 10}
    cell = cells[(ins_code & ~7, copy_code & ~7)]
    return cell * 64 + ((ins_code & 7) << 3) + (copy_code & 7)

def distance_code(d):
    """(dcode, nbits, extra) for distance d, with NPOSTFIX = NDIRECT = 0."""
    for x in range(48):
        ndistbits = 1 + (x >> 1)
        offset = ((2 + (x & 1)) << ndistbits) - 4
        if offset + 1 <= d <= offset + (1 << ndistbits):
            return 16 + x, ndistbits, d - offset - 1
    raise ValueError(d)

def load_transform_lengths():
    """(prefix length, type, suffix length) of each transform, from the D source."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'source', 'brotli.d'),
               encoding='utf-8').read()
    res = []
    for m in re.finditer(r'    Transform\("((?:[^"\\]|\\.)*)", (\d+), "((?:[^"\\]|\\.)*)"\),', src):
        def dlen(s):
            return len(re.sub(r'\\x[0-9A-F]{2}|\\.', 'x', s))
        res.append((dlen(m.group(1)), int(m.group(2)), dlen(m.group(3))))
    assert len(res) == 121
    return res

SIZE_BITS = [0, 0, 0, 0, 10, 10, 11, 11, 10, 10, 10, 10, 10, 9, 9, 8, 7, 7, 8, 7, 7, 6, 6, 5, 5]

def transformed_length(t, wordlen):
    p, typ, s = t
    if 1 <= typ <= 9:
        wordlen = max(0, wordlen - typ)
    elif 12 <= typ <= 20:
        wordlen = max(0, wordlen - (typ - 11))
    return p + wordlen + s

def dictionary_stream(transforms, wordlen, word, tid, literal=0x58):
    """One meta-block: a dictionary reference, then one literal."""
    out_len = transformed_length(transforms[tid], wordlen)
    w = BitWriter()
    window_bits(w, 16)
    w.write(1, 1); w.write(1, 0)       # ISLAST, not empty
    mlen(w, out_len + 1)
    w.write(1, 0); w.write(1, 0); w.write(1, 0)  # NBLTYPES L, I, D = 1
    w.write(2, 0); w.write(4, 0)       # NPOSTFIX, NDIRECT
    w.write(2, 0)                      # context mode
    w.write(1, 0); w.write(1, 0)       # NTREESL, NTREESD = 1
    simple_code(w, 8, [literal])
    copy_code, copy_extra = length_code(COPY_OFF, COPY_BITS, wordlen)
    dict_cmd = command_symbol(0, copy_code)
    lit_cmd = command_symbol(1, 2)     # insert 1, copy 4 (ignored)
    simple_code(w, 10, [dict_cmd, lit_cmd])
    address = (tid << SIZE_BITS[wordlen]) + word
    dcode, nbits, extra = distance_code(address + 1)  # max distance is 0
    simple_code(w, 6, [dcode])
    # command 1: dictionary word
    w.write(1, 0 if dict_cmd < lit_cmd else 1)
    w.write(COPY_BITS[copy_code], copy_extra)
    w.write(nbits, extra)
    # command 2: one literal, ends the meta-block
    w.write(1, 1 if dict_cmd < lit_cmd else 0)
    return w.bytes()

def dictionary_streams(rng):
    transforms = load_transform_lengths()
    res = []
    for wordlen in range(4, 25):
        nwords = 1 << SIZE_BITS[wordlen]
        words = sorted(set([0, nwords - 1] + [rng.randrange(nwords) for _ in range(2)]))
        for tid in range(121):
            for word in words:
                res.append(dictionary_stream(transforms, wordlen, word, tid))
    return res

def header_streams():
    res = []
    # Window sizes, valid and invalid, followed by an empty last meta-block.
    for n in range(10, 25):
        w = BitWriter(); window_bits(w, n); w.write(2, 3); res.append(w.bytes())
    for raw in range(8):  # 1 000 xxx: includes the large-window marker
        w = BitWriter(); w.write(1, 1); w.write(3, 0); w.write(3, raw); w.write(2, 3); res.append(w.bytes())
    # Metadata blocks: reserved bit, skip bytes count, exuberant last byte, padding.
    for reserved in (0, 1):
        for nbytes in range(4):
            for value in (0, 1, 255, 256, 65535, 65536):
                if nbytes == 0 and value != 0:
                    continue
                for padding in (0, 1):
                    for truncate in (0, 1):
                        if value >= 65535 and (reserved or padding):
                            continue  # keep the suite small
                        w = BitWriter(); window_bits(w, 16)
                        w.write(1, 0); w.write(2, 3); w.write(1, reserved); w.write(2, nbytes)
                        length = 0
                        for i in range(nbytes):
                            b = (value >> (8 * i)) & 255
                            w.write(8, b)
                            length |= b << (8 * i)
                        skip = length + 1 if nbytes else 0
                        w.pad(padding)
                        body = w.bytes() + bytes(range(256)) * (skip // 256) + bytes(skip % 256)
                        if truncate:
                            body = body[:-1]
                        w2 = BitWriter(); w2.write(1, 1); w2.write(1, 1)
                        res.append(body + w2.bytes())
    # Uncompressed blocks: size nibbles, exuberant nibble, padding, truncation, then last.
    for nibbles in (4, 5, 6):
        for length in (1, 2, 16, 65536, 65537):
            if length - 1 >= 1 << (4 * nibbles):
                continue
            for padding in (0, 1):
                for truncate in (0, 1):
                    if length > 16 and padding:
                        continue  # keep the suite small
                    w = BitWriter(); window_bits(w, 16)
                    w.write(1, 0); mlen(w, length, nibbles); w.write(1, 1)
                    w.pad(padding)
                    body = w.bytes() + bytes(i & 255 for i in range(length))
                    if truncate:
                        body = body[:-1]
                    w2 = BitWriter(); w2.write(1, 1); w2.write(1, 1)
                    res.append(body + w2.bytes())
    # Last empty meta-block: padding bits, trailing bytes.
    for padding in range(4):
        w = BitWriter(); window_bits(w, 16); w.write(2, 3); w.pad(padding)
        res.append(w.bytes())
        res.append(w.bytes() + b'\x00')
    return res

def crafted(rng):
    return header_streams() + dictionary_streams(rng)

if __name__ == '__main__':
    r = crafted(random.Random(1))
    print(len(r), sum(len(x) for x in r))
