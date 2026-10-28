"""
Compression ratio and speed of the D encoder, against the reference encoder.

    python encode_bench.py <tester executable built with optimizations> <google brotli repo checkout> [levels]

Requires: pip install brotli (the reference encoder, Google's C code).
"""
import os, sys, glob, time, random, struct, subprocess, tempfile
import brotli

TESTER = sys.argv[1]
REPO = sys.argv[2]
LEVELS = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0, 1, 3, 5, 6, 9, 11]
T = os.path.join(REPO, 'tests', 'testdata')
HERE = os.path.dirname(os.path.abspath(__file__))

def read(p):
    return open(p, 'rb').read()

text = b''.join(read(os.path.join(T, f)) for f in ['alice29.txt', 'asyoulik.txt', 'lcet10.txt', 'plrabn12.txt'])
csrc = b''.join(read(f) for f in sorted(glob.glob(os.path.join(REPO, 'c', '**', '*.[ch]'), recursive=True))
                if 'dictionary' not in f)

# Small inputs: 100 chunks at random (fixed) offsets, each compressed on its own.
def chunks(data, size):
    rng = random.Random(size)
    return [data[o:o + size] for o in (rng.randrange(len(data) - size) for _ in range(100))]

inputs = [
    ('text', [text]),
    ('csrc', [csrc]),
    ('maps', [read(os.path.join(T, 'mapsdatazrh'))]),
    ('qoi', [b''.join(read(f) for f in sorted(glob.glob(os.path.join(HERE, '..', 'qoi_test_images', '*.qoi'))))]),
]
for size, label in [(100, '100'), (1000, '1k'), (5000, '5k')]:
    inputs.append(('text' + label, chunks(text, size)))
    inputs.append(('csrc' + label, chunks(csrc, size)))

# More kinds of data, when present (not in the repository): web, CJK text,
# x86-64 code, 16-bit PCM audio, small files, compressed image formats.
MORE = os.path.join(HERE, '..', 'more-test-files')
EXAMPLES = os.path.join(HERE, '..', 'example', 'encode')
for name, path in [('js', os.path.join(MORE, 'htmx.min.js')),
                   ('css', os.path.join(MORE, 'tailwind-min.reference.css')),
                   ('html', os.path.join(MORE, 'web-page.html')),
                   ('cjk', os.path.join(MORE, 'chinese markdown.md')),
                   ('exe', os.path.join(MORE, 'executable.exe')),
                   ('wav', os.path.join(MORE, 'vive-la-lune.wav')),
                   ('svg', os.path.join(MORE, 'icon-linux.svg')),
                   ('png', os.path.join(MORE, 'pnggrad16rgb.png')),
                   ('sqz', os.path.join(EXAMPLES, 'skybox.sqz')),
                   ('qoix', os.path.join(EXAMPLES, 'knob-A.qoix'))]:
    if os.path.exists(path):
        inputs.append((name, [read(path)]))

def ref_speed(parts, q, seconds):
    n = 0
    t = time.perf_counter()
    while True:
        size = sum(len(brotli.compress(d, quality=q)) for d in parts)
        n += 1
        if time.perf_counter() - t > seconds:
            break
    return size, sum(map(len, parts)) * n / (time.perf_counter() - t) / 1e6

with tempfile.TemporaryDirectory() as tmp:
    print('%-8s %5s %10s %10s %8s %8s %8s' % ('input', 'level', 'D size', 'C size', 'D/C', 'D MB/s', 'C MB/s'))
    for name, parts in inputs:
        p = os.path.join(tmp, name)
        length = sum(map(len, parts))
        if len(parts) == 1:
            open(p, 'wb').write(parts[0])
        else:
            open(p, 'wb').write(b''.join(struct.pack('<I', len(d)) + d for d in parts))
        for level in LEVELS:
            slow = level >= 10
            iters = 1 if slow else max(2, min(20, int(20e6 / length)))
            r = subprocess.run([TESTER, 'enc' if len(parts) == 1 else 'encchunks', p, str(level), '24', str(iters)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print('TESTER FAILED', r.stdout, r.stderr)
                sys.exit(1)
            dsize, dspeed = r.stdout.split()
            dsize = int(dsize)
            dspeed = float(dspeed)
            csize, cspeed = ref_speed(parts, level, 0.05 if slow else 0.5)
            print('%-8s %5d %10d %10d %8.3f %8.1f %8.1f' % (name, level, dsize, csize, dsize / csize, dspeed, cspeed))
