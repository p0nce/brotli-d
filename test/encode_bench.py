"""
Compression ratio and speed of the D encoder, against the reference encoder.

    python encode_bench.py <tester executable built with optimizations> <google brotli repo checkout> [levels]

Requires: pip install brotli (the reference encoder, Google's C code).
"""
import os, sys, glob, time, subprocess, tempfile
import brotli

TESTER = sys.argv[1]
REPO = sys.argv[2]
LEVELS = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0, 1, 3, 5, 6, 9, 11]
T = os.path.join(REPO, 'tests', 'testdata')
HERE = os.path.dirname(os.path.abspath(__file__))

def read(p):
    return open(p, 'rb').read()

inputs = [
    ('text', b''.join(read(os.path.join(T, f)) for f in
                      ['alice29.txt', 'asyoulik.txt', 'lcet10.txt', 'plrabn12.txt'])),
    ('csrc', b''.join(read(f) for f in sorted(glob.glob(os.path.join(REPO, 'c', '**', '*.[ch]'), recursive=True))
                      if 'dictionary' not in f)),
    ('maps', read(os.path.join(T, 'mapsdatazrh'))),
    ('qoi', b''.join(read(f) for f in sorted(glob.glob(os.path.join(HERE, '..', 'qoi_test_images', '*.qoi'))))),
    ('small', read(os.path.join(T, 'asyoulik.txt'))[:5000]),
]

def ref_speed(data, q, seconds):
    n = 0
    t = time.perf_counter()
    while True:
        c = brotli.compress(data, quality=q)
        n += 1
        if time.perf_counter() - t > seconds:
            break
    return len(c), len(data) * n / (time.perf_counter() - t) / 1e6

with tempfile.TemporaryDirectory() as tmp:
    print('%-6s %5s %10s %10s %8s %8s %8s' % ('input', 'level', 'D size', 'C size', 'D/C', 'D MB/s', 'C MB/s'))
    for name, data in inputs:
        p = os.path.join(tmp, name)
        open(p, 'wb').write(data)
        for level in LEVELS:
            slow = level >= 10
            iters = 1 if slow else max(2, min(20, int(20e6 / len(data))))
            r = subprocess.run([TESTER, 'enc', p, str(level), '24', str(iters)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print('TESTER FAILED', r.stdout, r.stderr)
                sys.exit(1)
            dsize, dspeed = r.stdout.split()
            dsize = int(dsize)
            dspeed = float(dspeed)
            csize, cspeed = ref_speed(data, level, 0.05 if slow else 0.5)
            print('%-6s %5d %10d %10d %8.3f %8.1f %8.1f' % (name, level, dsize, csize, dsize / csize, dspeed, cspeed))
