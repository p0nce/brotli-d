"""
Round-trip test of the D brotli encoder, checked with the reference decoder
(the `brotli` Python package, which wraps Google's C decoder).

    python encode_test.py <tester executable> <google brotli repo checkout> [long]

The tester itself checks that every output decodes back with the D decoder,
and that the fixed-size API agrees. This script then decodes every output
with the reference decoder.

Inputs are sized for what each level exercises: around 64 KB (the minimum
meta-block length) and the level's meta-block length, small windows for
the window limits. Inputs larger than the 16 MB window only at a few
levels, or at every level with `long`. Jobs run in parallel: 2 at a time,
or the number in the environment variable JOBS.

Requires: pip install brotli
"""
import os, sys, glob, random, struct, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import brotli

TESTER = sys.argv[1]
REPO = sys.argv[2]
LONG = len(sys.argv) > 3 and sys.argv[3] == 'long'
T = os.path.join(REPO, 'tests', 'testdata')
HERE = os.path.dirname(os.path.abspath(__file__))

def read(p):
    return open(p, 'rb').read()

def text_like(rng, n):
    words = [b'the', b'brotli', b'compression', b'of', b'data', b'and', b'a', b'stream',
             b'window', b'distance', b'literal', b'context', b'\n', b'. ', b', ', b'The', b'WORLD']
    out = bytearray()
    while len(out) < n:
        out += rng.choice(words) + b' '
    return bytes(out[:n])

def small_inputs(rng):
    """Many small edge cases."""
    r = []
    for n in list(range(0, 70)) + [100, 127, 128, 255, 256, 257, 1000, 1023, 1024, 1025, 4095, 4096]:
        r.append(bytes(rng.getrandbits(8) for _ in range(n)))
        r.append(b'\0' * n)
        r.append(b'a' * n)
        r.append(text_like(rng, n))
        r.append(bytes(rng.choice(b'ab') for _ in range(n)))
        r.append(bytes((i * 7) & 0xFF for i in range(n)))
    # Runs that the short distance codes catch.
    for period in range(1, 40):
        r.append(bytes((i % period) for i in range(3000)))
    # Every byte value, increasing runs.
    r.append(bytes(range(256)) * 20)
    r.append(b''.join(bytes([i]) * i for i in range(256)))
    return r

def medium_inputs(rng, block):
    """Around the meta-block length limits: `block` is the level's meta-block
    length, 64 KB the shortest meta-block."""
    r = []
    for n in [65535, 65536, 65537, block - 1, block, block + 1,
              block + 65535, block + 65536, block + 65537, 2 * block + 12345]:
        r.append(text_like(rng, n))
    # Incompressible, then compressible: the fallback to uncompressed
    # meta-blocks, then back to compressed ones (the distance cache must be right).
    rnd = rng.randbytes(block)
    r.append(rnd + text_like(rng, block // 2) + rnd[:5000] + text_like(rng, 100000))
    # Matches at the window limit, for small windows.
    b = rng.randbytes(70000)
    r.append(b + b + b)
    b = rng.randbytes(1 << 16)
    r.append(b + b[:1000] + rng.randbytes(5000) + b)
    return r

def optimal_inputs(rng):
    """For the slow levels with optimal parsing (512 KB meta-blocks): what
    the faster levels don't already test. Several meta-blocks, the fallback
    to uncompressed meta-blocks and back, window limits. In 3 jobs."""
    rnd = rng.randbytes(100000)
    b = rng.randbytes(2000)  # for window 10: 1008 bytes
    return [
        [text_like(rng, n) for n in (65535, 65536, 65537)],
        [text_like(rng, (1 << 19) + 70000)],
        [rnd + text_like(rng, 500000) + rnd[:5000] + text_like(rng, 30000)],
        [b + b + b, b + b[:100] + rng.randbytes(500) + b + text_like(rng, 20000)],
    ]

def file_inputs(limit):
    """Test files, C sources, images, each truncated to `limit` bytes."""
    r = []
    for p in sorted(glob.glob(os.path.join(T, '*'))):
        if os.path.isfile(p) and '.compressed' not in p:
            r.append(read(p)[:limit])
    r.append(b''.join(read(f) for f in sorted(glob.glob(os.path.join(REPO, 'c', '**', '*.[ch]'),
                                                        recursive=True)))[:limit])
    for p in sorted(glob.glob(os.path.join(HERE, '..', 'qoi_test_images', '*'))):
        r.append(read(p)[:limit])
    return r

def big_inputs(rng):
    """Several meta-blocks, longer than the largest window."""
    text = b''.join(read(os.path.join(T, f)) for f in
                    ['alice29.txt', 'asyoulik.txt', 'lcet10.txt', 'plrabn12.txt'])
    return [text * 15, rng.randbytes(17 << 20)]

def run_job(job):
    name, level, window, inputs = job
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        fin = os.path.join(tmp, 'in.bin')
        fout = os.path.join(tmp, 'out.bin')
        with open(fin, 'wb') as f:
            for x in inputs:
                f.write(struct.pack('<I', len(x)))
                f.write(x)
        r = subprocess.run([TESTER, 'encbatch', fin, fout, str(level), str(window)],
                           capture_output=True, text=True, cwd=tmp)
        if r.returncode != 0:
            failed = os.path.join(tmp, 'encfail.bin')
            if os.path.exists(failed):
                os.replace(failed, 'encfail.bin')
            return 'TESTER FAILED %s level %d window %d: %s %s' % (name, level, window, r.stdout, r.stderr)
        data = read(fout)
    outs = []
    p = 0
    while p < len(data):
        n = struct.unpack('<I', data[p:p + 4])[0]
        outs.append(data[p + 4:p + 4 + n])
        p += 4 + n
    if len(outs) != len(inputs):
        return 'WRONG NUMBER OF OUTPUTS %s level %d window %d' % (name, level, window)
    total = 0
    for i, (x, c) in enumerate(zip(inputs, outs)):
        try:
            d = brotli.decompress(c)
        except brotli.error as e:
            open('encfail.bin', 'wb').write(x)
            return 'REFERENCE DECODER REJECTS: %s input %d (%d bytes), level %d window %d: %s' % (
                name, i, len(x), level, window, e)
        if d != x:
            open('encfail.bin', 'wb').write(x)
            return 'MISMATCH: %s input %d (%d bytes), level %d window %d' % (name, i, len(x), level, window)
        total += len(c)
    return None, '%-6s level %2d window %2d: %4d inputs ok, %9d bytes, %.1fs' % (
        name, level, window, len(inputs), total, time.time() - t0)

def main():
    t0 = time.time()
    rng = random.Random(7)
    small = small_inputs(rng)
    medium = {1 << 20: medium_inputs(rng, 1 << 20), 1 << 19: medium_inputs(rng, 1 << 19)}
    files = file_inputs(1 << 30)
    medium_files = file_inputs(1 << 18)
    short_files = file_inputs(1 << 14)
    optimal = optimal_inputs(rng)
    big = big_inputs(rng)

    jobs = []
    for level in range(12):
        for window in [10, 16, 24]:
            jobs.append(('small', level, window, small))
        if level < 10 or LONG:
            for window in [10, 24]:
                jobs.append(('medium', level, window, medium[1 << 19 if level >= 10 else 1 << 20]))
            jobs.append(('files', level, 24, files if LONG or level in (1, 5) else medium_files))
        else:
            # Optimal parsing is slow: less data, and what is specific to it.
            for k, inputs in enumerate(optimal):
                jobs.append(('opt%d' % k, level, 10 if k == 3 else 24, inputs))
            jobs.append(('files', level, 24, short_files))
        if LONG or level in (1, 5):
            jobs.append(('big', level, 24, big))
    # Slowest first, for a better use of the workers.
    jobs.sort(key=lambda j: -sum(len(x) for x in j[3]) * (20 if j[1] >= 10 else 1))

    failures = []
    with ThreadPoolExecutor(max_workers=int(os.environ.get('JOBS', '2'))) as pool:
        for result in pool.map(run_job, jobs):
            if isinstance(result, str):
                failures.append(result)
                print(result, flush=True)
            else:
                print(result[1], flush=True)
    if failures:
        print('%d FAILED' % len(failures))
        sys.exit(1)
    print('ALL OK, %d jobs in %.0fs' % (len(jobs), time.time() - t0))

main()
