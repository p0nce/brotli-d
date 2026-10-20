"""
Generates the frozen regression suite: test/suite.bin and test/suite.txt.

    python make_suite.py <google brotli repo checkout>

suite.bin is a sequence of records: uint32 little-endian length, then bytes.
suite.txt has one line per record, the reference decoder's result:
"<status> <decoded length> <fnv1a64 of decoded data>", status 0 = success, 1 = error.

Inputs are:
  - Google's official test vectors,
  - outputs of the reference encoder for every quality, several windows and modes,
  - hand-crafted streams (see craft.py): every dictionary transform on words
    of every length, and header edge cases,
  - mutated streams (bit flips, truncations, insertions...), half of them still
    valid but decoding to something else, half of them invalid.

Only needs to be re-run if you want a different suite.
Requires: pip install brotli
"""
import os, sys, glob, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # helpers
import craft

def main():
    compare.REPO = sys.argv[1]
    compare.rng.seed(1234)
    rng = compare.rng
    here = os.path.dirname(os.path.abspath(__file__))
    td = os.path.join(compare.REPO, 'tests', 'testdata')

    inputs = []

    # 1. Official vectors, except the big ones.
    for p in sorted(glob.glob(os.path.join(td, '*.compressed*'))):
        if os.path.getsize(p) < 64 * 1024:
            inputs.append(open(p, 'rb').read())

    # 2. Reference encoder output, on small data.
    data = [d for d in compare.corpus() if len(d) < 40000]
    encoded = []
    for d in data:
        encoded += compare.compressed_variants(d, many=True)
    small = [e for e in encoded if len(e) < 2048]
    inputs += [e for e in encoded if len(e) < 8192]

    # 3. Hand-crafted streams: all dictionary transforms, header edge cases.
    crafted = craft.crafted(rng)
    inputs += crafted
    small += [c for c in crafted if len(c) < 2048]

    # 4. Mutations of small streams, keep as many valid as invalid ones.
    valid = []
    invalid = []
    while len(valid) < 2500 or len(invalid) < 2500:
        x = rng.choice(small)
        for _ in range(rng.randint(1, 3)):
            x = compare.mutate(x)
        ref = compare.reference(x)
        if ref is None:
            continue
        if ref.startswith('0'):
            if len(valid) < 2500:
                valid.append(x)
        elif len(invalid) < 2500:
            invalid.append(x)
    inputs += valid + invalid

    with open(os.path.join(here, 'suite.bin'), 'wb') as f:
        for x in inputs:
            f.write(struct.pack('<I', len(x)))
            f.write(x)
    with open(os.path.join(here, 'suite.txt'), 'w', newline='\n') as f:
        for x in inputs:
            f.write(compare.reference(x) + '\n')
    print('%d inputs, %d bytes' % (len(inputs), os.path.getsize(os.path.join(here, 'suite.bin'))))

main()
