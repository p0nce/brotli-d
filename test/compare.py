"""
Differential test of the D brotli decoder against the reference decoder
(the `brotli` Python package, which wraps Google's C decoder).

    python compare.py <tester executable> <google brotli repo checkout> [seed] [fuzz iterations]

Requires: pip install brotli
"""
import os, sys, random, struct, subprocess, tempfile, glob
import brotli

TESTER = sys.argv[1]
REPO = sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
FUZZ = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
LIMIT = 64 * 1024 * 1024
rng = random.Random(SEED)

def fnv1a(d):
    h = 0xcbf29ce484222325
    for b in d:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h

def reference(data):
    try:
        d = brotli.decompress(data)
    except brotli.error:
        return '1 0 0'
    if len(d) > LIMIT:
        return None  # out of scope of the test
    return '0 %d %016x' % (len(d), fnv1a(d))

def run_batch(inputs, label):
    with tempfile.TemporaryDirectory() as tmp:
        fin = os.path.join(tmp, 'in.bin')
        fout = os.path.join(tmp, 'out.txt')
        with open(fin, 'wb') as f:
            for x in inputs:
                f.write(struct.pack('<I', len(x)))
                f.write(x)
        r = subprocess.run([TESTER, 'batch', fin, fout], capture_output=True, text=True)
        if r.returncode != 0:
            print('TESTER CRASHED on batch', label, r.stdout, r.stderr)
            # find the culprit
            for i, x in enumerate(inputs):
                with open(fin, 'wb') as f:
                    f.write(struct.pack('<I', len(x)))
                    f.write(x)
                r = subprocess.run([TESTER, 'batch', fin, fout], capture_output=True, text=True)
                if r.returncode != 0:
                    open('crash.br', 'wb').write(x)
                    print('culprit saved to crash.br', r.stderr[-2000:])
                    break
            sys.exit(1)
        got = open(fout).read().split('\n')
    bad = 0
    nsuccess = 0
    for i, x in enumerate(inputs):
        ref = reference(x)
        if ref is None:
            continue
        if ref.startswith('0'):
            nsuccess += 1
        if got[i] != ref:
            bad += 1
            name = 'mismatch_%s_%d.br' % (label, i)
            open(name, 'wb').write(x)
            print('MISMATCH', label, i, 'ref:', ref, 'got:', got[i], '->', name)
    print('%-12s %6d inputs, %6d valid, %d mismatches' % (label, len(inputs), nsuccess, bad))
    return bad

# ---------------------------------------------------------------------------
# Corpus

def corpus():
    out = []
    td = os.path.join(REPO, 'tests', 'testdata')
    for f in sorted(os.listdir(td)):
        p = os.path.join(td, f)
        if '.compressed' not in f and os.path.getsize(p) < 4 * 1024 * 1024:
            out.append(open(p, 'rb').read())
    # synthetic
    out.append(b'')
    out.append(b'a')
    out.append(b'ab' * 1000)
    out.append(bytes(rng.getrandbits(8) for _ in range(5000)))
    out.append(bytes(100000))
    out.append(bytes(range(256)) * 300)
    words = ['the', 'of', 'and', 'Hello', 'WORLD', 'brotli', 'été', 'Straße', 'ДОБРО', 'пожаловать',
             '日本語', 'テキスト', 'el', 'niño', 'http://', '.com/', 'function', '{', '}', '<div>',
             'Ünïcödé', 'ÀÉÎÕÜ', 'ŒUVRE', '\t', '\n', ' ', ',', '. The ', 'ing', 'ly ']
    for n in (10, 100, 1000, 30000):
        out.append(''.join(rng.choice(words) + rng.choice(['', ' ', ' ', '\n', ', ', '. '])
                           for _ in range(n)).encode('utf-8'))
    # all bytes, low entropy
    out.append(bytes(rng.choice(b'abc\x00\xff') for _ in range(20000)))
    # long distance repeat
    blk = bytes(rng.getrandbits(8) for _ in range(3000))
    out.append(blk + bytes(rng.getrandbits(8) for _ in range(70000)) + blk)
    return out

def compressed_variants(data, many):
    res = []
    qualities = range(12) if many else (0, 1, 5, 9, 11)
    for q in qualities:
        for mode in (brotli.MODE_GENERIC, brotli.MODE_TEXT, brotli.MODE_FONT):
            if not many and mode != brotli.MODE_GENERIC:
                continue
            lgwins = (10, 16, 22, 24) if many else (22,)
            for lgwin in lgwins:
                res.append(brotli.compress(data, quality=q, lgwin=lgwin, mode=mode))
    return res

def mutate(x):
    x = bytearray(x)
    k = rng.randrange(7)
    if len(x) == 0:
        return bytes([rng.getrandbits(8)])
    if k == 0:  # bit flip
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(len(x))
            x[i] ^= 1 << rng.randrange(8)
    elif k == 1:  # byte set
        i = rng.randrange(len(x))
        x[i] = rng.getrandbits(8)
    elif k == 2:  # truncate
        x = x[:rng.randrange(len(x))]
    elif k == 3:  # append
        x += bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 4)))
    elif k == 4:  # delete
        i = rng.randrange(len(x))
        del x[i:i + rng.randint(1, 4)]
    elif k == 5:  # insert
        i = rng.randrange(len(x) + 1)
        x[i:i] = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 4)))
    else:  # flip in the first bytes (headers)
        i = rng.randrange(min(len(x), 16))
        x[i] ^= 1 << rng.randrange(8)
    return bytes(x)

def main():
    total_bad = 0

    # 1. Official test vectors.
    td = os.path.join(REPO, 'tests', 'testdata')
    vecs = [open(p, 'rb').read() for p in sorted(glob.glob(os.path.join(td, '*.compressed*')))]
    total_bad += run_batch(vecs, 'vectors')

    # 2. Reference encoder output, many settings.
    data = corpus()
    enc = []
    for i, d in enumerate(data):
        enc += compressed_variants(d, many=len(d) < 200000)
    total_bad += run_batch(enc, 'encoded')

    # 3. Mutation fuzzing.
    small = [e for e in enc + vecs if len(e) < 20000]
    done = 0
    while done < FUZZ:
        n = min(2000, FUZZ - done)
        batch = []
        for _ in range(n):
            x = rng.choice(small)
            for _ in range(rng.randint(1, 3)):
                x = mutate(x)
            batch.append(x)
        total_bad += run_batch(batch, 'fuzz')
        done += n

    # 4. Random garbage.
    batch = [bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 64))) for _ in range(5000)]
    total_bad += run_batch(batch, 'random')

    print('TOTAL MISMATCHES:', total_bad)
    sys.exit(1 if total_bad else 0)

main()
