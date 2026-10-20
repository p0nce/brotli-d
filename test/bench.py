"""
Decoding speed benchmark.

    python bench.py <tester executable built with optimizations> <google brotli repo checkout>

Requires: pip install brotli (used to create the inputs, and as the C reference).
"""
import os, sys, glob, time, subprocess, tempfile
import brotli

TESTER = sys.argv[1]
REPO = sys.argv[2]
T = os.path.join(REPO, 'tests', 'testdata')

def read(p):
    return open(p, 'rb').read()

text = b''.join(read(os.path.join(T, f)) for f in
                ['alice29.txt', 'asyoulik.txt', 'lcet10.txt', 'plrabn12.txt'])
csrc = b''.join(read(f) for f in sorted(glob.glob(os.path.join(REPO, 'c', '**', '*.[ch]'), recursive=True))
                if 'dictionary' not in f)
maps = read(os.path.join(T, 'mapsdatazrh'))

inputs = []
for q in (1, 5, 11):
    inputs.append(('text q%d' % q, brotli.compress(text, quality=q)))
for q in (1, 5, 11):
    inputs.append(('csrc q%d' % q, brotli.compress(csrc, quality=q)))
inputs.append(('maps q11', brotli.compress(maps, quality=11)))

def ref_speed(data):
    out = brotli.decompress(data)
    n = 0
    t = time.perf_counter()
    while time.perf_counter() - t < 1.0:
        brotli.decompress(data)
        n += 1
    return len(out) * n / (time.perf_counter() - t) / 1e6

total_d = 0
total_c = 0
with tempfile.TemporaryDirectory() as tmp:
    print('%-10s %10s %10s %8s' % ('input', 'D MB/s', 'C MB/s', 'D/C'))
    for name, data in inputs:
        p = os.path.join(tmp, 'x.br')
        open(p, 'wb').write(data)
        out = len(brotli.decompress(data))
        iters = max(3, int(200e6 / out))
        best = 0
        for _ in range(3):
            r = subprocess.run([TESTER, 'bench', p, str(iters)], capture_output=True, text=True, check=True)
            best = max(best, float(r.stdout.split()[0]))
        c = ref_speed(data)
        total_d += best
        total_c += c
        print('%-10s %10.1f %10.1f %8.2f' % (name, best, c, best / c))
    print('%-10s %10.1f %10.1f %8.2f' % ('mean', total_d / len(inputs), total_c / len(inputs), total_d / total_c))
