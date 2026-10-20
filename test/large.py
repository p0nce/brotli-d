"""
Large input test: multiple meta-blocks, long distances, all window sizes.

    python large.py <tester executable>
"""
import os, sys, random, struct, subprocess, tempfile, hashlib
import brotli

TESTER = sys.argv[1]
rng = random.Random(42)

def fnv1a_fast(d):
    # Same as tester's FNV-1a 64, but written to be bearable on big inputs.
    h = 0xcbf29ce484222325
    p = 0x100000001b3
    m = 0xFFFFFFFFFFFFFFFF
    for b in d:
        h = ((h ^ b) * p) & m
    return h

# ~20 MB: random blocks repeated at long distances, text and zeros.
blocks = [bytes(rng.getrandbits(8) for _ in range(4096)) for _ in range(64)]
parts = []
size = 0
while size < 20 * 1024 * 1024:
    k = rng.randrange(4)
    if k == 0:
        b = rng.choice(blocks)
    elif k == 1:
        b = bytes(rng.randrange(1, 60000))
    elif k == 2:
        b = (b'The quick brown fox jumps over the lazy dog. ' * rng.randrange(1, 3000))
    else:
        b = bytes(rng.getrandbits(8) for _ in range(rng.randrange(1, 3000)))
    parts.append(b)
    size += len(b)
data = b''.join(parts)
expected = '0 %d %016x' % (len(data), fnv1a_fast(data))

inputs = []
for q, lgwin in ((1, 24), (5, 24), (9, 18), (2, 10), (0, 16)):
    inputs.append(brotli.compress(data, quality=q, lgwin=lgwin))
    print('compressed q=%d lgwin=%d: %d bytes' % (q, lgwin, len(inputs[-1])))

with tempfile.TemporaryDirectory() as tmp:
    fin = os.path.join(tmp, 'in.bin')
    fout = os.path.join(tmp, 'out.txt')
    with open(fin, 'wb') as f:
        for x in inputs:
            f.write(struct.pack('<I', len(x)))
            f.write(x)
    subprocess.run([TESTER, 'batch', fin, fout], check=True)
    got = open(fout).read().split('\n')
bad = 0
for i in range(len(inputs)):
    ok = got[i] == expected
    bad += not ok
    print('large', i, 'OK' if ok else 'MISMATCH ' + got[i] + ' vs ' + expected)
sys.exit(1 if bad else 0)
