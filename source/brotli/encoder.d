/**
Brotli encoder (RFC 7932).

One-shot compression from memory to memory, levels 0 to 11.
Levels 0-9 use greedy and lazy matching, levels 10-11 optimal parsing.
Uses the decoder's tables and static dictionary, and numem for allocations.
Everything is `@safe pure nothrow @nogc`. Cost estimates use integer
arithmetic only: the output is the same on every platform.

Authored by Claude Opus 5.5.

Copyright:
    Copyright (c) 2026 Guillaume Piolat.

License:
    MIT, see bottom of file.

Example:
---
import brotli;

// 1. Compress into a buffer you own: brotliCompressBound bytes are always enough.
ubyte[] buf = new ubyte[brotliCompressBound(data.length)];
size_t len;
if (brotliCompress(data, buf, len, BROTLI_DEFAULT_LEVEL) == BrotliStatus.success)
    use(buf[0..len]);

// 2. Compress into a malloc'ed buffer.
ubyte[] compressed;
if (brotliCompressAlloc(data, compressed, 11) == BrotliStatus.success)
{
    use(compressed);
    brotliFree(compressed); // @system
}
---
*/
module brotli.encoder;

import core.bitop : bsf, bsr;
import inteli.emmintrin;
import brotli.decoder;

@safe pure nothrow @nogc:

/// Compression levels: 0 is the fastest, 11 compresses the most.
enum BROTLI_MIN_LEVEL = 0;
enum BROTLI_MAX_LEVEL = 11;     /// ditto
enum BROTLI_DEFAULT_LEVEL = 6;  /// ditto

/// Window sizes: the decoder needs (1 << windowBits) bytes of history.
/// The encoder uses less than asked when the input is smaller.
enum BROTLI_MIN_WINDOW_BITS = 10;
enum BROTLI_MAX_WINDOW_BITS = 24;     /// ditto
enum BROTLI_DEFAULT_WINDOW_BITS = 24; /// ditto

/**
    Maximum compressed size of an input of `inputLength` bytes, for any level
    and window. `brotliCompress` never fails with `outputTooSmall` when given
    that much output space. Returns `size_t.max` on overflow.
*/
size_t brotliCompressBound(size_t inputLength)
{
    // Worst case: every meta-block is uncompressed, 4 bytes of header each,
    // plus 1 byte for the stream header and 1 for the final empty meta-block.
    // Meta-blocks are at least MIN_METABLOCK_LENGTH long, except the last one.
    size_t numBlocks = inputLength / MIN_METABLOCK_LENGTH + 1;
    size_t overhead = 4 * numBlocks + 2;
    if (inputLength > size_t.max - overhead)
        return size_t.max;
    return inputLength + overhead;
}

/**
    Compress `input` into a caller-provided buffer.

    Params:
        input       = Data to compress.
        output      = Where to write the compressed stream. Its content is
                      unspecified on error.
        encodedSize = On success, number of bytes written to `output`. 0 on error.
        level       = Compression level, clamped to [BROTLI_MIN_LEVEL, BROTLI_MAX_LEVEL].
        windowBits  = Maximum window size, clamped to
                      [BROTLI_MIN_WINDOW_BITS, BROTLI_MAX_WINDOW_BITS].

    Returns: `BrotliStatus.success`, `outputTooSmall` (cannot happen with an
             output of `brotliCompressBound(input.length)` bytes) or `outOfMemory`.
*/
BrotliStatus brotliCompress(const(ubyte)[] input, ubyte[] output, out size_t encodedSize,
                            int level = BROTLI_DEFAULT_LEVEL,
                            int windowBits = BROTLI_DEFAULT_WINDOW_BITS)
{
    BitWriter w;
    w.buf = output;
    w.growable = false;
    BrotliStatus r = encodeStream(input, w, level, windowBits);
    if (r == BrotliStatus.success)
        encodedSize = w.pos;
    return r;
}

/**
    Compress `input` into a newly allocated buffer.

    Params:
        input      = Data to compress.
        output     = On success, the compressed stream, to be released with
                     `brotliFree`. `null` on error.
        level      = Compression level, clamped to [BROTLI_MIN_LEVEL, BROTLI_MAX_LEVEL].
        windowBits = Maximum window size, clamped to
                     [BROTLI_MIN_WINDOW_BITS, BROTLI_MAX_WINDOW_BITS].

    Returns: `BrotliStatus.success` or `BrotliStatus.outOfMemory`.
*/
BrotliStatus brotliCompressAlloc(const(ubyte)[] input, out ubyte[] output,
                                 int level = BROTLI_DEFAULT_LEVEL,
                                 int windowBits = BROTLI_DEFAULT_WINDOW_BITS)
{
    BitWriter w;
    w.growable = true;
    BrotliStatus r = encodeStream(input, w, level, windowBits);
    if (r != BrotliStatus.success)
        return r;
    assert(w.pos > 0);
    // Shrink to fit. If that fails, the larger buffer is still fine.
    w.owned.resize(w.pos);
    output = w.owned.take()[0 .. w.pos];
    return r;
}

private:

// Marks the constants that tune the encoder (as opposed to format constants):
// the speed / compression trade-offs worth revisiting.
struct tuning {}

// ---------------------------------------------------------------------------
// Encoder: bit writer
// ---------------------------------------------------------------------------

// Meta-blocks are at least that long, except the last one (see brotliCompressBound).
enum MIN_METABLOCK_LENGTH = 1 << 16;
// MLEN can't be larger.
enum MAX_METABLOCK_LENGTH = 1 << 24;

struct BitWriter
{
pure nothrow @nogc @safe:
    ubyte[] buf;    // Caller's buffer, or `owned` if `growable`.
    MallocArray!ubyte owned;
    size_t pos;     // Bytes written. Keeps counting past buf.length after an error.
    ulong acc;      // Bits not written yet, LSB first.
    uint nbits;     // Number of bits in acc, < 32 between calls.
    bool growable;
    BrotliStatus error;

    static struct Mark
    {
        size_t pos;
        ulong acc;
        uint nbits;
    }

    Mark mark() const
    {
        return Mark(pos, acc, nbits);
    }

    // Discard what was written after `m`. This also discards an
    // `outputTooSmall` error that happened after `m`.
    void rewind(Mark m)
    {
        pos = m.pos;
        acc = m.acc;
        nbits = m.nbits;
        if (error == BrotliStatus.outputTooSmall)
            error = BrotliStatus.success;
    }

    ulong bitPosition() const
    {
        return cast(ulong) pos * 8 + nbits;
    }

    // Write the n low bits of v, n <= 32.
    pragma(inline, true)
    void writeBits(uint n, ulong v)
    {
        assert(n <= 32 && (v >> n) == 0);
        acc |= v << nbits;
        nbits += n;
        if (nbits >= 32)
            flush32();
    }

    void flush32()
    {
        if (pos + 4 <= buf.length || makeRoom(4))
        {
            uint v = cast(uint) acc;
            buf[pos]     = cast(ubyte) v;
            buf[pos + 1] = cast(ubyte)(v >> 8);
            buf[pos + 2] = cast(ubyte)(v >> 16);
            buf[pos + 3] = cast(ubyte)(v >> 24);
        }
        pos += 4;
        acc >>= 32;
        nbits -= 32;
    }

    // Pad with zero bits up to the next byte boundary.
    void alignToByte()
    {
        while (nbits > 0)
        {
            if (pos + 1 <= buf.length || makeRoom(1))
                buf[pos] = cast(ubyte) acc;
            pos += 1;
            acc >>= 8;
            nbits = nbits > 8 ? nbits - 8 : 0;
        }
        acc = 0;
    }

    // Append the first `numBits` bits of `bytes`, as written by a BitWriter.
    void writeBitString(scope const(ubyte)[] bytes, ulong numBits)
    {
        size_t whole = cast(size_t)(numBits / 8);
        size_t i = 0;
        for (; i + 4 <= whole; i += 4)
            writeBits(32, bytes[i] | (bytes[i + 1] << 8) | (bytes[i + 2] << 16) | (cast(uint) bytes[i + 3] << 24));
        for (; i < whole; ++i)
            writeBits(8, bytes[i]);
        uint rest = numBits % 8;
        if (rest != 0)
            writeBits(rest, bytes[whole] & ((1u << rest) - 1));
    }

    // The bits written, as bytes: buf[0 .. pos], then the pending bits.
    // Returns false if out of memory.
    bool copyTo(ref MallocArray!ubyte dst) const
    {
        size_t numBytes = pos + (nbits + 7) / 8;
        if (!dst.ensure(numBytes))
            return false;
        dst[0 .. pos] = buf[0 .. pos];
        foreach (k; pos .. numBytes)
            dst[k] = cast(ubyte)(acc >> (8 * (k - pos)));
        return true;
    }

    // Write whole bytes, must be on a byte boundary.
    void writeBytes(scope const(ubyte)[] data)
    {
        assert(nbits == 0);
        if (pos + data.length <= buf.length || makeRoom(data.length))
            buf[pos .. pos + data.length] = data[];
        pos += data.length;
    }

    bool makeRoom(size_t n)
    {
        if (error != BrotliStatus.success)
            return false;
        if (!growable)
        {
            error = BrotliStatus.outputTooSmall;
            return false;
        }
        size_t need = pos + n;
        size_t cap = buf.length < 4096 ? 4096 : buf.length;
        while (cap < need)
            cap = (cap > size_t.max / 2) ? need : cap * 2;
        if (!owned.resize(cap))
        {
            error = BrotliStatus.outOfMemory;
            return false;
        }
        buf = owned[];
        return true;
    }
}

// Inverse of readVarLenUint8, n in [0..255].
void writeVarLenUint8(ref BitWriter w, uint n)
{
    assert(n <= 255);
    if (n == 0)
    {
        w.writeBits(1, 0);
        return;
    }
    uint nbits = bsr(n);
    w.writeBits(1, 1);
    w.writeBits(3, nbits);
    w.writeBits(nbits, n - (1u << nbits));
}

// ---------------------------------------------------------------------------
// Encoder: prefix codes
// ---------------------------------------------------------------------------

void siftDown(scope ulong[] a, size_t root, size_t end)
{
    for (;;)
    {
        size_t child = 2 * root + 1;
        if (child >= end)
            return;
        if (child + 1 < end && a[child] < a[child + 1])
            child++;
        if (a[root] >= a[child])
            return;
        ulong t = a[root];
        a[root] = a[child];
        a[child] = t;
        root = child;
    }
}

void heapSort(scope ulong[] a)
{
    size_t n = a.length;
    if (n < 2)
        return;
    for (size_t i = n / 2; i-- > 0; )
        siftDown(a, i, n);
    for (size_t end = n - 1; end > 0; --end)
    {
        ulong t = a[0];
        a[0] = a[end];
        a[end] = t;
        siftDown(a, 0, end);
    }
}

// Huffman code lengths of the leaves, given as (weight << 16 | symbol) sorted
// by increasing weight. Returns false if a length exceeds `maxBits`.
bool huffmanLengths(scope const(ulong)[] leaves, uint maxBits, scope ubyte[] depth)
{
    size_t n = leaves.length;
    assert(n >= 2 && n <= MAX_ALPHABET_SIZE);
    ulong[2 * MAX_ALPHABET_SIZE] weight;
    ushort[2 * MAX_ALPHABET_SIZE] parent;
    ubyte[2 * MAX_ALPHABET_SIZE] nodeDepth;
    foreach (i; 0 .. n)
        weight[i] = leaves[i] >> 16;

    // Two queues: the sorted leaves, and the internal nodes, which are
    // created in order of increasing weight.
    size_t nextLeaf = 0;
    size_t nextNode = n;
    size_t numNodes = n;
    while (numNodes < 2 * n - 1)
    {
        size_t[2] pick;
        foreach (j; 0 .. 2)
        {
            if (nextLeaf < n && (nextNode >= numNodes || weight[nextLeaf] <= weight[nextNode]))
                pick[j] = nextLeaf++;
            else
                pick[j] = nextNode++;
        }
        weight[numNodes] = weight[pick[0]] + weight[pick[1]];
        parent[pick[0]] = cast(ushort) numNodes;
        parent[pick[1]] = cast(ushort) numNodes;
        numNodes++;
    }

    // Children have smaller indices than their parent, the root is last.
    nodeDepth[numNodes - 1] = 0;
    for (size_t k = numNodes - 1; k-- > 0; )
        nodeDepth[k] = cast(ubyte)(nodeDepth[parent[k]] + 1);
    foreach (i; 0 .. n)
        if (nodeDepth[i] > maxBits)
            return false;
    foreach (i; 0 .. n)
        depth[cast(size_t)(leaves[i] & 0xFFFF)] = nodeDepth[i];
    return true;
}

// Code lengths of a prefix code for `counts`, limited to `maxBits`.
// Unused symbols get 0. A single used symbol also gets 0, since it is
// encoded with no bits at all.
void buildHuffmanLengths(scope const(uint)[] counts, uint maxBits, scope ubyte[] depth)
{
    assert(counts.length <= MAX_ALPHABET_SIZE && depth.length >= counts.length);
    depth[0 .. counts.length] = 0;
    ushort[MAX_ALPHABET_SIZE] symbols;
    size_t n = 0;
    foreach (i, c; counts)
        if (c != 0)
            symbols[n++] = cast(ushort) i;
    if (n <= 1)
        return;

    // If the code is too deep, flatten the distribution and retry.
    // Terminates: with all weights equal, depths are <= ceil(log2(n)).
    ulong[MAX_ALPHABET_SIZE] leaves;
    for (ulong minWeight = 1; ; minWeight *= 2)
    {
        foreach (k; 0 .. n)
        {
            ulong weight = counts[symbols[k]];
            if (weight < minWeight)
                weight = minWeight;
            leaves[k] = (weight << 16) | symbols[k];
        }
        heapSort(leaves[0 .. n]);
        if (huffmanLengths(leaves[0 .. n], maxBits, depth))
            return;
    }
}

uint reverseBits(uint v, uint n)
{
    uint r = 0;
    foreach (i; 0 .. n)
    {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    return r;
}

// Canonical codes for the code lengths `depth`, bit-reversed since they are
// written LSB first. Same assignment as buildHuffmanTable: codes ordered by
// length, then by symbol.
void buildHuffmanCodes(scope const(ubyte)[] depth, scope ushort[] bits)
{
    uint[MAX_CODE_LENGTH + 1] count;
    foreach (d; depth)
        count[d]++;
    count[0] = 0;
    uint[MAX_CODE_LENGTH + 1] next;
    uint code = 0;
    foreach (len; 1 .. MAX_CODE_LENGTH + 1)
    {
        code = (code + count[len - 1]) << 1;
        next[len] = code;
    }
    foreach (sym, d; depth)
    {
        if (d == 0)
            bits[sym] = 0;
        else
            bits[sym] = cast(ushort) reverseBits(next[d]++, d);
    }
}

// Static code for code length code lengths 0..5, see readCodeLengthCodeLength.
static immutable ubyte[6] kCodeLengthCodeLengthBits = [0, 7, 3, 2, 1, 15];
static immutable ubyte[6] kCodeLengthCodeLengthNbits = [2, 4, 3, 2, 2, 4];

enum REPEAT_PREVIOUS_CODE_LENGTH = 16;
enum REPEAT_ZERO_CODE_LENGTH = 17;
enum INITIAL_REPEATED_CODE_LENGTH = 8;

// Run-length code of `count` repetitions of the code length `value`, as
// symbols 0..17 and their extra bits. Successive repeat codes combine:
// the repeat count of each one multiplies the previous one (RFC 7932 3.5),
// so a repeat count is written as digits, most significant first.
void rleRepeat(ubyte value, size_t count, scope ubyte[] tokens, scope ubyte[] extras, ref size_t n)
{
    ubyte code = value == 0 ? REPEAT_ZERO_CODE_LENGTH : REPEAT_PREVIOUS_CODE_LENGTH;
    uint shift = value == 0 ? 3 : 2;
    size_t special = value == 0 ? 11 : 7; // costs one more code than value + repeat
    if (count == special)
    {
        tokens[n] = value;
        extras[n++] = 0;
        count--;
    }
    if (count < 3)
    {
        foreach (i; 0 .. count)
        {
            tokens[n] = value;
            extras[n++] = 0;
        }
        return;
    }
    size_t start = n;
    count -= 3;
    for (;;)
    {
        tokens[n] = code;
        extras[n++] = cast(ubyte)(count & ((1u << shift) - 1));
        count >>= shift;
        if (count == 0)
            break;
        count--;
    }
    // Most significant digit first.
    for (size_t i = start, j = n - 1; i < j; ++i, --j)
    {
        ubyte t = extras[i];
        extras[i] = extras[j];
        extras[j] = t;
    }
}

// Run-length code the code lengths `depth`, returns the number of tokens.
// Trailing zeros are not coded: the decoder stops when the code is complete.
size_t rleCodeLengths(scope const(ubyte)[] depth, bool rleNonZero, bool rleZero,
                      scope ubyte[] tokens, scope ubyte[] extras)
{
    size_t len = depth.length;
    while (len > 0 && depth[len - 1] == 0)
        len--;
    size_t n = 0;
    ubyte previous = INITIAL_REPEATED_CODE_LENGTH;
    size_t i = 0;
    while (i < len)
    {
        ubyte value = depth[i];
        size_t reps = 1;
        if ((value != 0 && rleNonZero) || (value == 0 && rleZero))
            while (i + reps < len && depth[i + reps] == value)
                reps++;
        i += reps;
        if (value == 0)
            rleRepeat(0, reps, tokens, extras, n);
        else
        {
            if (value != previous)
            {
                // Code 16 repeats the previous non-zero length.
                tokens[n] = value;
                extras[n++] = 0;
                reps--;
                previous = value;
            }
            if (reps > 0)
                rleRepeat(value, reps, tokens, extras, n);
        }
    }
    return n;
}

// Write a complex prefix code from its run-length coded code lengths.
// Returns its size in bits. Writes nothing if `dryRun`.
size_t storeCodeLengths(bool dryRun)(ref BitWriter w, scope const(ubyte)[] tokens,
                                     scope const(ubyte)[] extras)
{
    uint[18] histogram;
    foreach (t; tokens)
        histogram[t]++;
    ubyte[18] clDepth;
    buildHuffmanLengths(histogram[], 5, clDepth[]);
    uint numCodes = 0;
    uint lastCode = 0;
    foreach (i; 0 .. 18)
    {
        if (histogram[i] != 0)
        {
            numCodes++;
            lastCode = i;
        }
    }
    assert(numCodes > 0);
    // With a single code length symbol, the decoder accepts an incomplete
    // code, reads all 18 code length code lengths, then decodes that symbol
    // with zero bits. Any non-zero length will do.
    if (numCodes == 1)
        clDepth[lastCode] = 1;
    ushort[18] clBits;
    buildHuffmanCodes(clDepth[], clBits[]);

    uint hskip = 0;
    if (clDepth[kCodeLengthCodeOrder[0]] == 0 && clDepth[kCodeLengthCodeOrder[1]] == 0)
    {
        hskip = 2;
        if (clDepth[kCodeLengthCodeOrder[2]] == 0)
            hskip = 3;
    }
    // The decoder stops reading once the code is complete.
    uint numToStore = 18;
    if (numCodes > 1)
        while (clDepth[kCodeLengthCodeOrder[numToStore - 1]] == 0)
            numToStore--;

    size_t bits = 2;
    static if (!dryRun)
        w.writeBits(2, hskip);
    foreach (i; hskip .. numToStore)
    {
        uint v = clDepth[kCodeLengthCodeOrder[i]];
        bits += kCodeLengthCodeLengthNbits[v];
        static if (!dryRun)
            w.writeBits(kCodeLengthCodeLengthNbits[v], kCodeLengthCodeLengthBits[v]);
    }
    foreach (k, t; tokens)
    {
        uint n = numCodes == 1 ? 0 : clDepth[t];
        uint e = t == REPEAT_PREVIOUS_CODE_LENGTH ? 2 : (t == REPEAT_ZERO_CODE_LENGTH ? 3 : 0);
        bits += n + e;
        static if (!dryRun)
        {
            w.writeBits(n, clBits[t]);
            w.writeBits(e, extras[k]);
        }
    }
    return bits;
}

// Write the prefix code with code lengths `depth`, as computed by
// buildHuffmanLengths from `counts`. Returns its size in bits.
// Writes nothing if `dryRun`.
size_t storePrefixCode(bool dryRun)(ref BitWriter w, scope const(uint)[] counts,
                                    scope const(ubyte)[] depth)
{
    size_t alphabetSize = counts.length;
    assert(alphabetSize >= 2 && alphabetSize <= MAX_ALPHABET_SIZE && depth.length == alphabetSize);

    uint[4] symbols;
    uint numSymbols = 0;
    foreach (i, c; counts)
    {
        if (c != 0)
        {
            if (numSymbols < 4)
                symbols[numSymbols] = cast(uint) i;
            numSymbols++;
        }
    }

    if (numSymbols <= 4)
    {
        // Simple prefix code. The decoder gives code lengths to the symbols
        // in the order they are written: shortest codes first.
        if (numSymbols == 0)
            numSymbols = 1; // symbol 0, never used
        foreach (i; 1 .. numSymbols)
        {
            for (size_t j = i; j > 0 && depth[symbols[j]] < depth[symbols[j - 1]]; --j)
            {
                uint t = symbols[j];
                symbols[j] = symbols[j - 1];
                symbols[j - 1] = t;
            }
        }
        uint nbits = 0;
        for (size_t x = alphabetSize - 1; x != 0; x >>= 1)
            ++nbits;
        size_t bits = 4 + numSymbols * nbits + (numSymbols == 4 ? 1 : 0);
        static if (!dryRun)
        {
            w.writeBits(2, 1);
            w.writeBits(2, numSymbols - 1);
            foreach (i; 0 .. numSymbols)
                w.writeBits(nbits, symbols[i]);
            if (numSymbols == 4)
                w.writeBits(1, depth[symbols[0]] == 1 ? 1 : 0); // lengths 1,2,3,3 or 2,2,2,2
        }
        return bits;
    }

    // Complex prefix code: keep the cheapest run-length coding.
    ubyte[MAX_ALPHABET_SIZE] tokens;
    ubyte[MAX_ALPHABET_SIZE] extras;
    size_t bestBits = size_t.max;
    uint bestVariant = 0;
    foreach (variant; 0 .. 4)
    {
        size_t n = rleCodeLengths(depth, (variant & 1) != 0, (variant & 2) != 0, tokens[], extras[]);
        size_t bits = storeCodeLengths!true(w, tokens[0 .. n], extras[0 .. n]);
        if (bits < bestBits)
        {
            bestBits = bits;
            bestVariant = variant;
        }
    }
    static if (!dryRun)
    {
        size_t n = rleCodeLengths(depth, (bestVariant & 1) != 0, (bestVariant & 2) != 0,
                                  tokens[], extras[]);
        storeCodeLengths!false(w, tokens[0 .. n], extras[0 .. n]);
    }
    return bestBits;
}

// Build a prefix code for `counts` into `depth` and `bits`, and write it.
void buildAndStorePrefixCode(ref BitWriter w, scope const(uint)[] counts,
                             scope ubyte[] depth, scope ushort[] bits)
{
    buildHuffmanLengths(counts, MAX_CODE_LENGTH, depth);
    buildHuffmanCodes(depth[0 .. counts.length], bits);
    storePrefixCode!false(w, counts, depth[0 .. counts.length]);
}

// Size in bits of the symbols `counts` coded with code lengths `depth`.
ulong dataBits(scope const(uint)[] counts, scope const(ubyte)[] depth)
{
    ulong r = 0;
    foreach (i, c; counts)
        r += cast(ulong) c * depth[i];
    return r;
}

// ---------------------------------------------------------------------------
// Encoder: context maps
// ---------------------------------------------------------------------------

enum MAX_CONTEXT_MAP_LENGTH = 256 * 64;

// Run-length code a context map into symbols (extra bits << 9) | symbol.
// `symbols` must be as long as `values`. Returns the number of symbols.
size_t rleContextMap(scope const(ubyte)[] values, uint maxRunLengthPrefix, scope uint[] symbols)
{
    size_t n = 0;
    size_t i = 0;
    while (i < values.length)
    {
        if (values[i] != 0)
        {
            symbols[n++] = values[i] + maxRunLengthPrefix;
            i++;
            continue;
        }
        size_t run = 1;
        while (i + run < values.length && values[i + run] == 0)
            run++;
        i += run;
        while (run > 0)
        {
            uint prefix = bsr(run);
            if (prefix > maxRunLengthPrefix)
                prefix = maxRunLengthPrefix;
            if (prefix == 0)
            {
                symbols[n++] = 0;
                run--;
                continue;
            }
            size_t take = (2u << prefix) - 1; // longest run for that prefix
            if (take > run)
                take = run;
            symbols[n++] = cast(uint)((take - (1u << prefix)) << 9) | prefix;
            run -= take;
        }
    }
    return n;
}

void moveToFront(scope const(ubyte)[] map, scope ubyte[] result)
{
    ubyte[256] mtf;
    foreach (i; 0 .. 256)
        mtf[i] = cast(ubyte) i;
    foreach (k, v; map)
    {
        uint index = 0;
        while (mtf[index] != v)
            index++;
        result[k] = cast(ubyte) index;
        for (uint j = index; j > 0; --j)
            mtf[j] = mtf[j - 1];
        mtf[0] = v;
    }
}

// Write a context map, with the cheapest run-length and move-to-front options.
// Values must be < numTrees. `scratch` must have 2 * map.length entries.
void storeContextMap(ref BitWriter w, scope const(ubyte)[] map, uint numTrees, scope uint[] scratch)
{
    assert(numTrees >= 1 && numTrees <= 256 && map.length <= MAX_CONTEXT_MAP_LENGTH);
    assert(scratch.length >= 2 * map.length);
    writeVarLenUint8(w, numTrees - 1);
    if (numTrees == 1)
        return;

    ubyte[MAX_CONTEXT_MAP_LENGTH] transformed;
    moveToFront(map, transformed[0 .. map.length]);
    uint[] symbols = scratch[0 .. map.length];
    uint[] bestSymbols = scratch[map.length .. 2 * map.length];

    ulong bestBits = ulong.max;
    bool bestMtf;
    uint bestPrefix;
    size_t bestNum;
    foreach (useMtf; 0 .. 2)
    {
        const(ubyte)[] values = useMtf ? transformed[0 .. map.length] : map;
        // The longest zero run bounds the useful run length prefixes.
        size_t longestRun = 0;
        size_t run = 0;
        foreach (v; values)
        {
            run = v == 0 ? run + 1 : 0;
            if (run > longestRun)
                longestRun = run;
        }
        uint maxPrefix = longestRun > 1 ? bsr(longestRun) : 0;
        if (maxPrefix > 16)
            maxPrefix = 16;
        foreach (prefix; 0 .. maxPrefix + 1)
        {
            size_t n = rleContextMap(values, prefix, symbols);
            uint[256 + 16] histogram;
            uint alphabetSize = numTrees + prefix;
            ulong bits = 1 + (prefix > 0 ? 4 : 0) + 1;
            foreach (s; symbols[0 .. n])
            {
                uint sym = s & 0x1FF;
                histogram[sym]++;
                if (sym > 0 && sym <= prefix)
                    bits += sym;
            }
            ubyte[256 + 16] depth;
            buildHuffmanLengths(histogram[0 .. alphabetSize], MAX_CODE_LENGTH, depth[]);
            bits += storePrefixCode!true(w, histogram[0 .. alphabetSize], depth[0 .. alphabetSize]);
            bits += dataBits(histogram[0 .. alphabetSize], depth[0 .. alphabetSize]);
            if (bits < bestBits)
            {
                bestBits = bits;
                bestMtf = useMtf != 0;
                bestPrefix = prefix;
                bestNum = n;
                bestSymbols[0 .. n] = symbols[0 .. n];
            }
        }
    }

    uint alphabetSize = numTrees + bestPrefix;
    uint[256 + 16] histogram;
    foreach (s; bestSymbols[0 .. bestNum])
        histogram[s & 0x1FF]++;
    w.writeBits(1, bestPrefix > 0 ? 1 : 0);
    if (bestPrefix > 0)
        w.writeBits(4, bestPrefix - 1);
    ubyte[256 + 16] depth;
    ushort[256 + 16] bits;
    buildAndStorePrefixCode(w, histogram[0 .. alphabetSize], depth[], bits[]);
    foreach (s; bestSymbols[0 .. bestNum])
    {
        uint sym = s & 0x1FF;
        w.writeBits(depth[sym], bits[sym]);
        if (sym > 0 && sym <= bestPrefix)
            w.writeBits(sym, s >> 9);
    }
    w.writeBits(1, bestMtf ? 1 : 0);
}

// ---------------------------------------------------------------------------
// Encoder: commands and distances
// ---------------------------------------------------------------------------

uint insertLengthCode(size_t len)
{
    if (len < 6)
        return cast(uint) len;
    if (len < 130)
    {
        uint nbits = bsr(len - 2) - 1;
        return (nbits << 1) + cast(uint)((len - 2) >> nbits) + 2;
    }
    if (len < 2114)
        return bsr(len - 66) + 10;
    if (len < 6210)
        return 21;
    if (len < 22594)
        return 22;
    return 23;
}

uint copyLengthCode(size_t len)
{
    assert(len >= 2);
    if (len < 10)
        return cast(uint)(len - 2);
    if (len < 134)
    {
        uint nbits = bsr(len - 6) - 1;
        return (nbits << 1) + cast(uint)((len - 6) >> nbits) + 4;
    }
    if (len < 2118)
        return bsr(len - 70) + 12;
    return 23;
}

// copyLengthCode(len) for len < 2118, longer lengths have code 23.
static immutable ubyte[2118] kCopyLengthCode = makeCopyLengthCodes();

ubyte[2118] makeCopyLengthCodes()
{
    ubyte[2118] t;
    foreach (len; 2 .. t.length)
        t[len] = cast(ubyte) copyLengthCode(len);
    return t;
}

pragma(inline, true)
uint fastCopyLengthCode(size_t len)
{
    return len < kCopyLengthCode.length ? kCopyLengthCode[len] : 23;
}

// Cell (symbol >> 6) of insert-and-copy symbols with an explicit distance,
// indexed by (insert code >> 3) * 3 + (copy code >> 3).
static immutable ubyte[9] kCommandCell = makeCommandCells();

ubyte[9] makeCommandCells()
{
    ubyte[9] r;
    foreach (cell; 2 .. 11)
        r[(kInsertCodeBase[cell] >> 3) * 3 + (kCopyCodeBase[cell] >> 3)] = cast(ubyte) cell;
    return r;
}

// Insert-and-copy symbol. `implicitDistance`: use the cells where the
// distance code is an implicit 0, if possible.
uint commandSymbol(uint insertCode, uint copyCode, bool implicitDistance)
{
    uint low = ((insertCode & 7) << 3) | (copyCode & 7);
    if (implicitDistance && insertCode < 8 && copyCode < 16)
        return ((copyCode >> 3) << 6) | low; // cell 0 or 1
    return (kCommandCell[(insertCode >> 3) * 3 + (copyCode >> 3)] << 6) | low;
}

// Distance symbol and extra bits for a distance code: 0..15 for the short
// codes, else distance + 15.
void distanceSymbol(size_t distanceCode, uint npostfix, uint ndirect,
                    out uint symbol, out uint nbits, out uint extra)
{
    if (distanceCode < 16 + ndirect)
    {
        symbol = cast(uint) distanceCode;
        return;
    }
    size_t dist = (size_t(1) << (npostfix + 2)) + (distanceCode - 16 - ndirect);
    uint bucket = bsr(dist) - 1;
    uint postfix = cast(uint)(dist & ((1u << npostfix) - 1));
    uint prefix = (dist >> bucket) & 1;
    size_t offset = size_t(2 + prefix) << bucket;
    nbits = bucket - npostfix;
    symbol = 16 + ndirect + (((2 * (nbits - 1) + prefix) << npostfix) | postfix);
    extra = cast(uint)((dist - offset) >> npostfix);
}

struct Command
{
    uint insertLength;
    uint copyLength;    // as coded, 0 for a final insert-only command
    uint outputLength;  // bytes produced by the copy: copyLength, or less for a dictionary word
    uint distanceCode;  // 0..15: short code, else distance + 15
}

// Distances of the last copies, most recent first.
alias DistanceCache = size_t[4];

static immutable DistanceCache kInitialDistanceCache = [4, 11, 15, 16];

// Distance code for `distance`, given the last distances.
uint distanceCode(size_t distance, ref const(DistanceCache) last)
{
    if (distance == last[0])
        return 0;
    if (distance == last[1])
        return 1;
    foreach (code; 4 .. 16)
    {
        size_t base = code < 10 ? last[0] : last[1];
        if (base + kDistanceShortCodeValueOffset[code] == distance)
            return code;
    }
    if (distance == last[2])
        return 2;
    if (distance == last[3])
        return 3;
    return cast(uint)(distance + 15);
}

// The decoder pushes every distance but those of code 0 (and dictionary words).
void updateDistanceCache(ref DistanceCache last, size_t distance, uint code)
{
    if (code == 0)
        return;
    last[3] = last[2];
    last[2] = last[1];
    last[1] = last[0];
    last[0] = distance;
}

// The 16 distances that short codes 0..15 give, 0 when not valid.
void shortCodeDistances(ref const(DistanceCache) last, ref size_t[16] r)
{
    foreach (code; 0 .. 16)
    {
        size_t base = last[kDistanceShortCodeIndexOffset[code] == 0 ? 0
                         : 4 - kDistanceShortCodeIndexOffset[code]];
        long d = cast(long) base + kDistanceShortCodeValueOffset[code];
        r[code] = d > 0 ? cast(size_t) d : 0;
    }
}

// Growable array. If an allocation fails, push returns false and sets `failed`.
struct GrowableArray(T)
{
pure nothrow @nogc @safe:
    MallocArray!T storage;
    size_t length;
    bool failed;

    pragma(inline, true)
    bool push(T value)
    {
        if (length == storage.length && !grow())
            return false;
        storage[length++] = value;
        return true;
    }

    bool grow()
    {
        size_t cap = storage.length < 1024 ? 1024 : storage.length * 2;
        if (!storage.resize(cap))
        {
            failed = true;
            return false;
        }
        return true;
    }

    inout(T)[] opSlice() inout
    {
        return storage[0 .. length];
    }

    void release()
    {
        storage.release();
        length = 0;
    }
}

alias CommandBuffer = GrowableArray!Command;
alias MatchBuffer = GrowableArray!BackwardMatch;

// ---------------------------------------------------------------------------
// Encoder: static dictionary
// ---------------------------------------------------------------------------

// Transforms that only omit the last n bytes of a word, n = 0..9 (0: identity).
static immutable ubyte[10] kCutoffTransforms = findCutoffTransforms();

ubyte[10] findCutoffTransforms()
{
    ubyte[10] r;
    foreach (n; 0 .. 10)
    {
        uint type = n == 0 ? TRANSFORM_IDENTITY : TRANSFORM_OMIT_LAST_1 + n - 1;
        bool found = false;
        foreach (i, ref const t; kTransforms)
        {
            if (t.prefix.length == 0 && t.suffix.length == 0 && t.type == type)
            {
                r[n] = cast(ubyte) i;
                found = true;
                break;
            }
        }
        assert(found);
    }
    return r;
}

enum DICTIONARY_HASH_BITS = 14;
enum NUM_DICTIONARY_WORDS = 13504;

// Dictionary words by hash of their first 4 bytes, ASCII case folded (to
// find uppercased words too): the words of hash h are
// words[bucketStart[h] .. bucketStart[h + 1]], as (length << 11) | index,
// longest first. Indices have up to 11 bits (lengths 6 and 7: 2048 words).
struct DictionaryHashTable
{
    ushort[(1 << DICTIONARY_HASH_BITS) + 1] bucketStart;
    ushort[NUM_DICTIONARY_WORDS] words;
}

// The values of buildDictionaryHashTable(), see the unittest: computing them
// at compile time takes too long.
static immutable DictionaryHashTable kDictionaryHashTable =
    DictionaryHashTable(kDictionaryBucketStart, kDictionaryWords);

// Hash of 4 bytes, ASCII case folded.
pragma(inline, true)
uint dictionaryHash(uint fourBytes)
{
    uint folded = 0;
    foreach (k; 0 .. 4)
    {
        uint b = (fourBytes >> (8 * k)) & 0xFF;
        if (b >= 'A' && b <= 'Z')
            b += 32;
        folded |= b << (8 * k);
    }
    return (folded * 0x1E35A7BDu) >> (32 - DICTIONARY_HASH_BITS);
}

uint dictionaryWordHash(size_t offset)
{
    uint v = 0;
    foreach (k; 0 .. 4)
        v |= cast(uint)(cast(ubyte) kDictionaryData[offset + k]) << (8 * k);
    return dictionaryHash(v);
}

// Transforms that keep the first bytes of words (all but "omit first n"),
// grouped by prefix: group g is transforms[groupStart[g] .. groupStart[g + 1]].
struct TransformGroups
{
    uint numGroups;
    ubyte[16] groupStart;
    ubyte[121] transforms;
}

static immutable TransformGroups kTransformGroups = groupTransforms();

bool keepsWordStart(ref const(Transform) t)
{
    return t.type < TRANSFORM_OMIT_FIRST_1 || t.type > TRANSFORM_OMIT_FIRST_9;
}

TransformGroups groupTransforms()
{
    TransformGroups g;
    size_t n = 0;
    bool[kTransforms.length] done;
    foreach (i, ref const t; kTransforms)
    {
        if (done[i] || !keepsWordStart(t))
            continue;
        g.groupStart[g.numGroups++] = cast(ubyte) n;
        foreach (j; i .. kTransforms.length)
        {
            if (!done[j] && keepsWordStart(kTransforms[j]) && kTransforms[j].prefix == t.prefix)
            {
                done[j] = true;
                g.transforms[n++] = cast(ubyte) j;
            }
        }
    }
    g.groupStart[g.numGroups] = cast(ubyte) n;
    return g;
}

// Static dictionary references at data[pos .. pos + maxLength] that produce
// more than minLength bytes, with any transform: for each
// output length, the one with the smallest distance. Returns true if any.
// Dictionary references are the distances after `maxDistance`.
struct DictionaryReferences
{
    size_t[MAX_TRANSFORMED_WORD_LENGTH + 1] distance = size_t.max;
    ubyte[MAX_TRANSFORMED_WORD_LENGTH + 1] wordLength;
    ushort[MAX_TRANSFORMED_WORD_LENGTH + 1] wordIndex;
    ubyte[MAX_TRANSFORMED_WORD_LENGTH + 1] transform;
}

bool findDictionaryReferences(scope const(ubyte)[] data, size_t pos, size_t maxLength,
                              size_t maxDistance, size_t minLength, ref const(OmitFirstIndex) omitFirst,
                              ref DictionaryReferences r)
{
    immutable(ubyte)[] dict = dictionary();
    bool found = false;
    foreach (g; 0 .. kTransformGroups.numGroups)
    {
        const(ubyte)[] group = kTransformGroups.transforms[kTransformGroups.groupStart[g]
                                                          .. kTransformGroups.groupStart[g + 1]];
        string prefix = kTransforms[group[0]].prefix;
        if (prefix.length + MIN_DICTIONARY_WORD_LENGTH > maxLength)
            continue;
        bool prefixMatches = true;
        foreach (k, c; prefix)
        {
            if (data[pos + k] != cast(ubyte) c)
            {
                prefixMatches = false;
                break;
            }
        }
        if (!prefixMatches)
            continue;
        size_t q = pos + prefix.length;
        size_t room = maxLength - prefix.length;
        uint h = dictionaryHash(load32(data, q));
        foreach (entry; kDictionaryHashTable.words[kDictionaryHashTable.bucketStart[h]
                                                   .. kDictionaryHashTable.bucketStart[h + 1]])
        {
            size_t len = entry >> 11;
            size_t idx = entry & 2047;
            size_t offset = kDictionaryOffsets[len] + len * idx;
            const(ubyte)[] word = dict[offset .. offset + len];
            size_t limit = len < room ? len : room;
            size_t exact = 0;
            while (exact < limit && word[exact] == data[q + exact])
                exact++;
            // Matched lengths of the uppercased word, computed when needed.
            size_t upperFirst = size_t.max;
            size_t upperAll = size_t.max;

            foreach (ti; group)
            {
                const(Transform)* t = &kTransforms[ti];
                size_t body;
                if (t.type == TRANSFORM_IDENTITY)
                {
                    if (exact < len)
                        continue;
                    body = len;
                }
                else if (t.type >= TRANSFORM_OMIT_LAST_1 && t.type <= TRANSFORM_OMIT_LAST_9)
                {
                    size_t omit = t.type - TRANSFORM_OMIT_LAST_1 + 1;
                    if (omit >= len || exact < len - omit)
                        continue;
                    body = len - omit;
                }
                else if (t.type == TRANSFORM_UPPERCASE_FIRST || t.type == TRANSFORM_UPPERCASE_ALL)
                {
                    if (len > room)
                        continue;
                    bool firstOnly = t.type == TRANSFORM_UPPERCASE_FIRST;
                    size_t matched = firstOnly ? upperFirst : upperAll;
                    if (matched == size_t.max)
                    {
                        ubyte[MAX_DICTIONARY_WORD_LENGTH] upper;
                        upper[0 .. len] = word[];
                        toUpperCase(upper[0 .. len], firstOnly);
                        matched = 0;
                        while (matched < len && upper[matched] == data[q + matched])
                            matched++;
                        if (firstOnly)
                            upperFirst = matched;
                        else
                            upperAll = matched;
                    }
                    if (matched < len)
                        continue;
                    body = len;
                }
                else
                    continue;

                size_t outLength = prefix.length + body + t.suffix.length;
                if (outLength > maxLength || outLength <= minLength)
                    continue;
                bool suffixMatches = true;
                foreach (k, c; t.suffix)
                {
                    if (data[q + body + k] != cast(ubyte) c)
                    {
                        suffixMatches = false;
                        break;
                    }
                }
                if (!suffixMatches)
                    continue;
                size_t distance = maxDistance + 1 + idx + (size_t(ti) << kDictionarySizeBits[len]);
                if (distance < r.distance[outLength])
                {
                    r.distance[outLength] = distance;
                    r.wordLength[outLength] = cast(ubyte) len;
                    r.wordIndex[outLength] = cast(ushort) idx;
                    r.transform[outLength] = ti;
                    found = true;
                }
            }
        }
    }

    // Words with their first bytes omitted.
    foreach (entry; omitFirst.bucket(omitFirstHash(load32(data, pos))))
    {
        const(ubyte)[] output = omitFirstOutput(entry);
        if (output.length > maxLength || output.length <= minLength || output != data[pos .. pos + output.length])
            continue;
        size_t len = (entry >> 11) & 31;
        size_t idx = entry & 2047;
        ubyte ti = kOmitFirstTransforms[entry >> 16];
        size_t distance = maxDistance + 1 + idx + (size_t(ti) << kDictionarySizeBits[len]);
        if (distance < r.distance[output.length])
        {
            r.distance[output.length] = distance;
            r.wordLength[output.length] = cast(ubyte) len;
            r.wordIndex[output.length] = cast(ushort) idx;
            r.transform[output.length] = ti;
            found = true;
        }
    }
    return found;
}

DictionaryHashTable buildDictionaryHashTable()
{
    DictionaryHashTable t;
    uint[1 << DICTIONARY_HASH_BITS] count;
    size_t numWords = 0;
    foreach (len; MIN_DICTIONARY_WORD_LENGTH .. MAX_DICTIONARY_WORD_LENGTH + 1)
    {
        foreach (idx; 0 .. 1u << kDictionarySizeBits[len])
        {
            count[dictionaryWordHash(kDictionaryOffsets[len] + len * idx)]++;
            numWords++;
        }
    }
    assert(numWords == NUM_DICTIONARY_WORDS);
    uint sum = 0;
    foreach (h; 0 .. 1 << DICTIONARY_HASH_BITS)
    {
        t.bucketStart[h] = cast(ushort) sum;
        sum += count[h];
    }
    t.bucketStart[1 << DICTIONARY_HASH_BITS] = cast(ushort) sum;
    uint[1 << DICTIONARY_HASH_BITS] fill;
    foreach (h; 0 .. 1 << DICTIONARY_HASH_BITS)
        fill[h] = t.bucketStart[h];
    for (size_t len = MAX_DICTIONARY_WORD_LENGTH; len >= MIN_DICTIONARY_WORD_LENGTH; --len)
    {
        foreach (idx; 0 .. 1u << kDictionarySizeBits[len])
        {
            uint h = dictionaryWordHash(kDictionaryOffsets[len] + len * idx);
            t.words[fill[h]++] = cast(ushort)((len << 11) | idx);
        }
    }
    return t;
}

// Transforms that only omit the first n bytes of a word, n = 1..9; 0 if none (n = 8).
static immutable ubyte[10] kOmitFirstTransforms = findOmitFirstTransforms();

ubyte[10] findOmitFirstTransforms()
{
    ubyte[10] r;
    foreach (i, ref const t; kTransforms)
        if (t.prefix.length == 0 && t.suffix.length == 0
            && t.type >= TRANSFORM_OMIT_FIRST_1 && t.type <= TRANSFORM_OMIT_FIRST_9)
            r[t.type - TRANSFORM_OMIT_FIRST_1 + 1] = cast(ubyte) i;
    return r;
}

@tuning enum OMIT_FIRST_MIN_LENGTH = 5; // shortest "omit first n" output tried
const(ubyte)[] omitFirstOutput(uint entry)
{
    size_t n = entry >> 16;
    size_t len = (entry >> 11) & 31;
    size_t offset = kDictionaryOffsets[len] + len * (entry & 2047);
    return dictionary()[offset + n .. offset + len];
}

// Hash of the first 4 bytes of an "omit first n" output: exact, unlike
// dictionaryHash, the transforms don't uppercase.
pragma(inline, true)
uint omitFirstHash(uint fourBytes)
{
    return (fourBytes * 0x1E35A7BDu) >> (32 - DICTIONARY_HASH_BITS);
}

// Dictionary words with their first n bytes omitted (n with a transform, at
// least OMIT_FIRST_MIN_LENGTH bytes left), by hash of their first 4 bytes
// left: the words of hash h are words[bucketStart[h] .. bucketStart[h + 1]],
// as (n << 16) | (length << 11) | index. Built at run time: as a table, it
// would add 190 KB to the binary.
struct OmitFirstIndex
{
pure nothrow @nogc @safe:
    MallocArray!uint bucketStart;
    MallocArray!uint words;

    bool initialize()
    {
        enum numBuckets = 1 << DICTIONARY_HASH_BITS;
        if (!bucketStart.resize(numBuckets + 1))
            return false;
        bucketStart[] = 0;
        foreach (pass; 0 .. 2)
        {
            foreach (len; MIN_DICTIONARY_WORD_LENGTH .. MAX_DICTIONARY_WORD_LENGTH + 1)
            {
                foreach (idx; 0 .. 1u << kDictionarySizeBits[len])
                {
                    foreach (n; 1 .. 10)
                    {
                        if (kOmitFirstTransforms[n] == 0 || len < n + OMIT_FIRST_MIN_LENGTH)
                            continue;
                        uint entry = cast(uint)((n << 16) | (len << 11) | idx);
                        uint h = omitFirstHash(load32(omitFirstOutput(entry), 0));
                        if (pass == 0)
                            bucketStart[h + 1]++;
                        else
                            words[bucketStart[h]++] = entry; // ends at the start of h + 1
                    }
                }
            }
            if (pass == 0)
            {
                foreach (h; 0 .. numBuckets)
                    bucketStart[h + 1] += bucketStart[h];
                if (!words.resize(bucketStart[numBuckets]))
                    return false;
            }
        }
        for (size_t h = numBuckets; h > 0; --h)
            bucketStart[h] = bucketStart[h - 1];
        bucketStart[0] = 0;
        return true;
    }

    void release()
    {
        bucketStart.release();
        words.release();
    }

    const(uint)[] bucket(uint h) const return
    {
        return words[bucketStart[h] .. bucketStart[h + 1]];
    }
}

// Find a static dictionary word, maybe with its last bytes omitted, at
// data[pos .. pos + maxLength]. Dictionary references are the distances
// after `maxDistance`. Improves on `m`, returns true if it did.
bool findDictionaryMatch(scope const(ubyte)[] data, size_t pos, size_t maxLength,
                         size_t maxDistance, ref Match m)
{
    if (maxLength < MIN_DICTIONARY_WORD_LENGTH)
        return false;
    immutable(ubyte)[] dict = dictionary();
    uint h = dictionaryHash(load32(data, pos));
    bool found = false;
    foreach (entry; kDictionaryHashTable.words[kDictionaryHashTable.bucketStart[h]
                                               .. kDictionaryHashTable.bucketStart[h + 1]])
    {
        size_t len = entry >> 11;
        size_t idx = entry & 2047;
        size_t limit = len < maxLength ? len : maxLength;
        size_t offset = kDictionaryOffsets[len] + len * idx;
        const(ubyte)[] word = dict[offset .. offset + limit];
        size_t matched = 0;
        while (matched < limit && word[matched] == data[pos + matched])
            matched++;
        if (matched < MIN_DICTIONARY_WORD_LENGTH || len - matched >= kCutoffTransforms.length)
            continue;
        size_t transform = kCutoffTransforms[len - matched];
        size_t distance = maxDistance + 1 + idx + (transform << kDictionarySizeBits[len]);
        size_t score = backwardReferenceScore(matched, distance);
        if (score > m.score)
        {
            m.len = matched;
            m.distance = distance;
            m.score = score;
            m.wordLength = len;
            found = true;
        }
    }
    return found;
}

// ---------------------------------------------------------------------------
// Encoder: match finder
// ---------------------------------------------------------------------------

pragma(inline, true)
uint load32(scope const(ubyte)[] data, size_t pos)
{
    const(ubyte)[] p = data[pos .. pos + 4];
    return p[0] | (p[1] << 8) | (p[2] << 16) | (cast(uint) p[3] << 24);
}

pragma(inline, true)
ulong load64(const(ubyte)* p) @system
{
    return p[0]
         | (cast(ulong) p[1] << 8)
         | (cast(ulong) p[2] << 16)
         | (cast(ulong) p[3] << 24)
         | (cast(ulong) p[4] << 32)
         | (cast(ulong) p[5] << 40)
         | (cast(ulong) p[6] << 48)
         | (cast(ulong) p[7] << 56);
}

// Length of the common prefix of data[a .. a + limit] and data[b .. b + limit], a < b.
// Halts if they are not in `data`, even in release builds.
pragma(inline, true)
size_t matchLength(scope const(ubyte)[] data, size_t a, size_t b, size_t limit) @trusted
{
    if (!(a < b && b <= data.length && limit <= data.length - b))
        assert(0);
    const(ubyte)* pa = data.ptr + a;
    const(ubyte)* pb = data.ptr + b;
    size_t n = 0;
    while (n + 8 <= limit)
    {
        ulong x = load64(pa + n) ^ load64(pb + n);
        if (x != 0)
            return n + (bsf(x) >> 3);
        n += 8;
    }
    while (n < limit && pa[n] == pb[n])
        n++;
    return n;
}

// Heuristic match scores, in 1/135 of literal bytes (same as the reference encoder).
@tuning enum LITERAL_BYTE_SCORE = 135;
@tuning enum DISTANCE_BIT_PENALTY = 30;
enum SCORE_BASE = DISTANCE_BIT_PENALTY * 8 * 8;
@tuning enum MIN_SCORE_MARGIN = 100;   // a match must score more than SCORE_BASE + this
enum MIN_SCORE = SCORE_BASE + MIN_SCORE_MARGIN;
@tuning enum LAST_DISTANCE_BONUS = 15;
@tuning enum SHORT_CODE_PENALTY = 39;  // plus 0 to 14, depending on the short code
@tuning enum LAZY_MATCH_MARGIN = 175;  // a match one byte later must score that much more
@tuning enum DICTIONARY_THROTTLE_SHIFT = 7; // stop looking up the dictionary below 1 hit in 2^this lookups

// Incompressible data: after a while without matches, look for matches
// every NEAR_STEP bytes, NEAR_JUMP bytes at a time; and after
// FAR_FACTOR * randomWindow more, every FAR_STEP bytes, FAR_JUMP at a time.
@tuning enum SKIP_NEAR_STEP = 2;
@tuning enum SKIP_NEAR_JUMP = 8;
@tuning enum SKIP_FAR_STEP = 4;
@tuning enum SKIP_FAR_JUMP = 16;
@tuning enum SKIP_FAR_FACTOR = 4;

pragma(inline, true)
size_t backwardReferenceScore(size_t len, size_t distance)
{
    return SCORE_BASE + LITERAL_BYTE_SCORE * len - DISTANCE_BIT_PENALTY * bsr(distance);
}

pragma(inline, true)
size_t lastDistanceScore(size_t len, uint shortCode)
{
    size_t score = LITERAL_BYTE_SCORE * len + SCORE_BASE + LAST_DISTANCE_BONUS;
    if (shortCode != 0)
        score -= SHORT_CODE_PENALTY + ((0x1CA10 >> (shortCode & 0xE)) & 0xE);
    return score;
}

struct Match
{
    size_t len;         // bytes produced
    size_t distance;
    size_t score = MIN_SCORE;
    size_t wordLength;  // for a static dictionary word: its length, the coded copy length
}

// Parameters of a compression level.
struct LevelParams
{
    ubyte bucketBits;       // log2 of the number of hash buckets
    ubyte blockBits;        // log2 of the number of positions per bucket
    ubyte numShortCodes;    // short code distances tried: 4, 10 or 16
    bool lazyMatching;
    ushort randomWindow;    // literals without a match before skipping
    bool contextModeling;   // literal and distance context maps
    bool tryAllContextModes;
    bool useDictionary;     // static dictionary words
    ubyte blockSplitting;   // 0: none, 1: fast, 2: best
    ubyte distanceParamSearch; // NPOSTFIX and NDIRECT. 0: none, 1: NPOSTFIX, 2: both
    bool optimizeHistograms; // for run-length coding of prefix codes
    ubyte optimalPasses;    // optimal parsing instead of greedy, with that many passes
    ushort optimalMaxLength; // longer copies are only tried at their full length
    ubyte optimalStarts;    // positions tried for the start of each insert
    bool optimalKeepBest;   // keep the pass that compresses best, else the last
    bool clusterLiteralsByContext; // cheaper literal clustering when there are many histograms
    ubyte hashLength;       // bytes hashed by the hash table: 4, or 5 for text (see textHashLength)
}

@tuning static immutable LevelParams[BROTLI_MAX_LEVEL + 1] kLevelParams =
[
    LevelParams(14, 0,  4, false,  64, false, false, false, 0, 0, false, 0, 0, 0, false, false, 4), // 0
    LevelParams(15, 0,  4, false,  64, false, false, false, 0, 0, false, 0, 0, 0, false, false, 4), // 1
    LevelParams(15, 1,  4, true,   64, false, false, true, 0, 0, false, 0, 0, 0, false, false, 4), // 2
    LevelParams(15, 2,  4, true,   64, false, false, true, 0, 0, false, 0, 0, 0, false, false, 4), // 3
    LevelParams(15, 3,  4, true,   64, true, false, true, 0, 0, true, 0, 0, 0, false, false, 4), // 4
    LevelParams(15, 4,  4, true,   64, true, false, true, 1, 1, true, 0, 0, 0, false, true, 5), // 5
    LevelParams(15, 5, 10, true,   64, true, false, true, 1, 1, true, 0, 0, 0, false, true, 5), // 6
    LevelParams(16, 5, 10, true,   64, true, false, true, 1, 1, true, 0, 0, 0, false, true, 5), // 7
    LevelParams(16, 6, 16, true,   64, true, false, true, 1, 1, true, 0, 0, 0, false, true, 5), // 8
    LevelParams(16, 6, 16, true,  512, true, true, true, 1, 2, true, 0, 0, 0, false, true, 5), // 9
    LevelParams(16, 6, 16, true,  512, true, true, true, 2, 2, true, 2, 150, 1, false, false, 4), // 10
    LevelParams(16, 6, 16, true,  512, true, true, true, 2, 2, true, 4, 325, 5, true, false, 4), // 11
];

// Inputs from that long use tags in the hash table, see Hasher.
@tuning enum size_t TAGGED_MIN_LENGTH = 1 << 20;

// Hash table of recent positions, by hash of their next 4 bytes.
// Each bucket is a ring of the most recent positions.
struct Hasher
{
pure nothrow @nogc @safe:
    MallocArray!uint positions; // bucket * blockSize + i, relative to `base`
    MallocArray!uint counts;    // insertions in each bucket
    // With `tagged`, each position also has an 8-bit tag, the hash bits
    // below those of the bucket: positions with another tag don't have the
    // same 4 bytes, they are skipped without reading their data. Worth it
    // for buckets of 16 positions or more, with a 4 byte hash (binary data),
    // and long inputs: many collisions.
    MallocArray!ubyte tags;     // same indices as positions
    bool tagged;
    uint bucketBits;
    uint blockBits;
    uint blockMask;
    uint hashLength;    // 4 or 5 bytes
    size_t base;
    size_t dictionaryLookups;
    size_t dictionaryMatches;

    bool initialize(uint bucketBits, uint blockBits, uint hashLength, size_t inputLength)
    {
        assert(hashLength == 4 || hashLength == 5);
        this.hashLength = hashLength;
        this.bucketBits = bucketBits;
        this.blockBits = blockBits;
        blockMask = (1u << blockBits) - 1;
        if (!positions.resize(size_t(1) << (bucketBits + blockBits))
            || !counts.resize(size_t(1) << bucketBits))
            return false;
        tagged = hashLength == 4 && blockBits >= 4 && inputLength >= TAGGED_MIN_LENGTH;
        if (tagged && !tags.resize(size_t(1) << (bucketBits + blockBits)))
            return false;
        counts[] = 0; // positions are only read once written
        base = 0;
        return true;
    }

    void release()
    {
        positions.release();
        counts.release();
        tags.release();
    }

    // Bucket and tag of the 4 bytes at `pos`.
    pragma(inline, true)
    void bucketAndTag(scope const(ubyte)[] data, size_t pos, out uint key, out ubyte tag) const
    {
        uint hash = load32(data, pos) * 0x1E35A7BDu;
        key = hash >> (32 - bucketBits);
        tag = cast(ubyte)(hash >> (24 - bucketBits));
    }

    // Hash of the hashLength (4 or 5) bytes at `pos`, which needs 4 bytes
    // of data (a missing 5th byte counts as zero).
    pragma(inline, true)
    uint hashAt(uint hashLength)(scope const(ubyte)[] data, size_t pos) const
    {
        static if (hashLength == 4)
            return (load32(data, pos) * 0x1E35A7BDu) >> (32 - bucketBits);
        else
        {
            ulong v = load32(data, pos);
            if (pos + 4 < data.length)
                v |= ulong(data[pos + 4]) << 32;
            return cast(uint)(((v << 24) * 0x1FE35A7BD3579BD3UL) >> (64 - bucketBits));
        }
    }

    // Remember position `pos`, which needs 4 bytes of data.
    pragma(inline, true)
    void store(uint hashLength, bool isTagged)(scope const(ubyte)[] data, size_t pos)
    {
        static if (isTagged)
        {
            uint key;
            ubyte tag;
            bucketAndTag(data, pos, key, tag);
        }
        else
            uint key = hashAt!hashLength(data, pos);
        uint n = counts[key];
        size_t slot = (size_t(key) << blockBits) + (n & blockMask);
        positions[slot] = cast(uint)(pos - base);
        static if (isTagged)
            tags[slot] = tag;
        counts[key] = n + 1;
    }

    // Positions are 32-bit relative to `base`: move `base` forward when
    // they would overflow. Positions older than the window are useless.
    void rebase(size_t pos, size_t window)
    {
        if (pos - base < 0x8000_0000)
            return;
        size_t shift = pos - base - window;
        foreach (ref p; positions)
            p = (p > shift) ? cast(uint)(p - shift) : 0;
        base += shift;
    }
}

// Bit i is set if tags[i] == tag. Halts unless tags.length is a multiple of
// 16, at most 64, even in release builds.
pragma(inline, true)
ulong tagMask(scope const(ubyte)[] tags, ubyte tag) @trusted
{
    if (tags.length % 16 != 0 || tags.length > 64)
        assert(0);
    __m128i t = _mm_set1_epi8(cast(byte) tag);
    ulong mask = 0;
    for (size_t i = 0; i < tags.length; i += 16)
    {
        __m128i v = _mm_loadu_si128(cast(const(__m128i)*) &tags[i]);
        mask |= cast(ulong)(cast(uint) _mm_movemask_epi8(_mm_cmpeq_epi8(v, t))) << i;
    }
    return mask;
}

// Find the best match at data[pos .. pos + maxLength], at distances <= maxDistance.
// Improves on `m`. Also remembers `pos` in the hash table.
void findLongestMatch(uint hashLength, bool isTagged)(ref Hasher h, scope const(ubyte)[] data, size_t pos, size_t maxLength,
                      size_t maxDistance, ref const(size_t[16]) shortDistances,
                      ref const(LevelParams) params, ref Match m)
{
    uint numShortCodes = params.numShortCodes;
    assert(maxLength >= 4 && pos + maxLength <= data.length);
    size_t bestLen = m.len;

    // Distances of the short codes are cheap, try them first.
    foreach (code; 0 .. numShortCodes)
    {
        if (bestLen >= maxLength)
            break;
        size_t d = shortDistances[code];
        if (d - 1 >= maxDistance) // d == 0 or too far
            continue;
        size_t prev = pos - d;
        if (data[prev + bestLen] != data[pos + bestLen])
            continue;
        size_t len = matchLength(data, prev, pos, maxLength);
        if (len >= 3 || (len == 2 && code < 2))
        {
            size_t score = lastDistanceScore(len, code);
            if (score > m.score)
            {
                m.len = bestLen = len;
                m.distance = d;
                m.score = score;
            }
        }
    }

    static if (isTagged)
    {
        uint key;
        ubyte tag;
        h.bucketAndTag(data, pos, key, tag);
    }
    else
        uint key = h.hashAt!hashLength(data, pos);
    size_t bucket = size_t(key) << h.blockBits;
    uint n = h.counts[key];
    uint blockSize = 1u << h.blockBits;
    uint count = n < blockSize ? n : blockSize;
    static if (isTagged)
    {
        // Only the candidates with the same tag: a mask of the bucket,
        // rotated so that bit b is candidate blockSize - 1 - b (the newest
        // is the top bit), without the slots not written yet.
        uint r = n & h.blockMask;
        ulong mask = tagMask(h.tags[bucket .. bucket + blockSize], tag);
        if (r != 0)
            mask = (mask >> r) | (mask << (blockSize - r));
        if (blockSize < 64)
            mask &= (1UL << blockSize) - 1;
        if (count == 0)
            mask = 0; // also avoids shifting by 64 below
        else if (count < blockSize)
            mask &= ~((1UL << (blockSize - count)) - 1);
        while (mask != 0 && bestLen < maxLength)
        {
            uint b = bsr(mask);
            mask ^= 1UL << b;
            size_t prev = h.base + h.positions[bucket + ((b + r) & h.blockMask)];
            size_t d = pos - prev;
            if (d - 1 >= maxDistance) // older positions are even further
                break;
            if (data[prev + bestLen] != data[pos + bestLen])
                continue;
            size_t len = matchLength(data, prev, pos, maxLength);
            if (len >= 4)
            {
                size_t score = backwardReferenceScore(len, d);
                if (score > m.score)
                {
                    m.len = bestLen = len;
                    m.distance = d;
                    m.score = score;
                }
            }
        }
        h.tags[bucket + (n & h.blockMask)] = tag;
    }
    else
    {
        foreach (k; 0 .. count)
        {
            if (bestLen >= maxLength)
                break;
            size_t prev = h.base + h.positions[bucket + ((n - 1 - k) & h.blockMask)];
            size_t d = pos - prev;
            if (d - 1 >= maxDistance) // older positions are even further
                break;
            if (data[prev + bestLen] != data[pos + bestLen])
                continue;
            size_t len = matchLength(data, prev, pos, maxLength);
            if (len >= 4)
            {
                size_t score = backwardReferenceScore(len, d);
                if (score > m.score)
                {
                    m.len = bestLen = len;
                    m.distance = d;
                    m.score = score;
                }
            }
        }
    }
    h.positions[bucket + (n & h.blockMask)] = cast(uint)(pos - h.base);
    h.counts[key] = n + 1;

    // Static dictionary, if nothing was found. Stop looking when it rarely
    // finds anything.
    if (params.useDictionary && m.score == MIN_SCORE
        && h.dictionaryMatches >= (h.dictionaryLookups >> DICTIONARY_THROTTLE_SHIFT))
    {
        h.dictionaryLookups++;
        if (findDictionaryMatch(data, pos, maxLength, maxDistance, m))
            h.dictionaryMatches++;
    }
}

// LZ77 parse of data[start .. end]: appends commands to `cmds`, the last one
// is insert-only if the block doesn't end with a copy. Updates `last`.
bool createCommands(uint hashLength, bool isTagged)(ref Hasher h, scope const(ubyte)[] data, size_t start, size_t end,
                    size_t maxBackward, ref const(LevelParams) params,
                    ref DistanceCache last, ref CommandBuffer cmds)
{
    size_t[16] shortDistances;
    shortCodeDistances(last, shortDistances);
    size_t storeEnd = data.length >= 4 ? data.length - 3 : 0; // positions with 4 bytes of data
    size_t pos = start;
    size_t insert = 0;
    size_t stored = 0; // positions below were stored already
    size_t applyRandomHeuristics = pos + params.randomWindow;

    while (end - pos >= 4)
    {
        size_t maxLength = end - pos;
        size_t maxDistance = pos < maxBackward ? pos : maxBackward;
        Match m;
        findLongestMatch!(hashLength, isTagged)(h, data, pos, maxLength, maxDistance, shortDistances, params, m);
        stored = pos + 1;
        if (m.score > MIN_SCORE)
        {
            if (params.lazyMatching)
            {
                // Is there something better at the next position?
                for (int delayed = 0; end - pos >= 5; )
                {
                    Match m2;
                    size_t maxDistance2 = pos + 1 < maxBackward ? pos + 1 : maxBackward;
                    findLongestMatch!(hashLength, isTagged)(h, data, pos + 1, maxLength - 1, maxDistance2, shortDistances,
                                     params, m2);
                    stored = pos + 2;
                    if (m2.score < m.score + LAZY_MATCH_MARGIN)
                        break;
                    pos++;
                    insert++;
                    maxLength--;
                    m = m2;
                    if (++delayed >= 4)
                        break;
                }
            }
            applyRandomHeuristics = pos + 2 * m.len + params.randomWindow;
            uint code = distanceCode(m.distance, last);
            Command c;
            c.insertLength = cast(uint) insert;
            c.copyLength = cast(uint)(m.wordLength != 0 ? m.wordLength : m.len);
            c.outputLength = cast(uint) m.len;
            c.distanceCode = code;
            if (!cmds.push(c))
                return false;
            // Distances of dictionary words are not remembered.
            assert((m.wordLength != 0) == (m.distance > (pos < maxBackward ? pos : maxBackward)));
            if (m.wordLength == 0)
            {
                updateDistanceCache(last, m.distance, code);
                shortCodeDistances(last, shortDistances);
            }
            insert = 0;
            size_t matchEnd = pos + m.len;
            size_t storeTo = matchEnd < storeEnd ? matchEnd : storeEnd;
            for (size_t q = stored; q < storeTo; ++q)
                h.store!(hashLength, isTagged)(data, q);
            pos = matchEnd;
        }
        else
        {
            insert++;
            pos++;
            // Many literals in a row: probably incompressible data, look
            // for matches less often.
            if (pos > applyRandomHeuristics)
            {
                size_t margin = 4;
                if (pos > applyRandomHeuristics + SKIP_FAR_FACTOR * params.randomWindow)
                {
                    size_t jump = (pos + SKIP_FAR_JUMP < end - margin) ? pos + SKIP_FAR_JUMP : end - margin;
                    for (; pos < jump; pos += SKIP_FAR_STEP)
                    {
                        h.store!(hashLength, isTagged)(data, pos);
                        insert += SKIP_FAR_STEP;
                    }
                }
                else
                {
                    size_t jump = (pos + SKIP_NEAR_JUMP < end - margin) ? pos + SKIP_NEAR_JUMP : end - margin;
                    for (; pos < jump; pos += SKIP_NEAR_STEP)
                    {
                        h.store!(hashLength, isTagged)(data, pos);
                        insert += SKIP_NEAR_STEP;
                    }
                }
            }
        }
    }
    insert += end - pos;
    if (insert > 0)
    {
        Command c;
        c.insertLength = cast(uint) insert;
        if (!cmds.push(c))
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Encoder: entropy model
// ---------------------------------------------------------------------------

// Costs are in 1/65536 of a bit (16.16 fixed point): integer arithmetic,
// the same on every platform.
enum uint COST_ONE_BIT = 1 << 16;

// log2(x) in fixed point with `fractionalBits` bits, truncated: by
// repeated squaring of the mantissa. Slow, for checking the tables below.
uint log2Squaring(size_t x, uint fractionalBits)
{
    assert(x != 0 && fractionalBits <= 20);
    uint e = bsr(x);
    // Mantissa in [1, 2), in 1.31 fixed point.
    ulong m = e >= 31 ? x >> (e - 31) : cast(ulong) x << (31 - e);
    uint fraction = 0;
    foreach (i; 0 .. fractionalBits)
    {
        m = (m * m) >> 31; // in [1, 4)
        fraction <<= 1;
        if (m >= (2UL << 31))
        {
            fraction |= 1;
            m >>= 1;
        }
    }
    return (e << fractionalBits) | fraction;
}

// log2(1 + i / 256) in 16.16, rounded, for i in [0 .. 256]
// (from log2Squaring, see the unittest).
static immutable uint[257] kLog2Mantissa =
[
          0,     369,     736,    1102,    1466,    1829,    2190,    2551,
       2909,    3267,    3623,    3978,    4331,    4683,    5034,    5384,
       5732,    6079,    6425,    6769,    7112,    7454,    7795,    8134,
       8473,    8810,    9146,    9480,    9814,   10146,   10477,   10807,
      11136,   11464,   11791,   12116,   12440,   12764,   13086,   13407,
      13727,   14046,   14363,   14680,   14996,   15310,   15624,   15937,
      16248,   16559,   16868,   17177,   17484,   17791,   18096,   18401,
      18704,   19007,   19308,   19609,   19909,   20207,   20505,   20802,
      21098,   21393,   21687,   21980,   22272,   22564,   22854,   23144,
      23433,   23720,   24007,   24293,   24579,   24863,   25146,   25429,
      25711,   25992,   26272,   26551,   26830,   27108,   27384,   27660,
      27936,   28210,   28484,   28757,   29029,   29300,   29571,   29840,
      30109,   30378,   30645,   30912,   31178,   31443,   31707,   31971,
      32234,   32496,   32758,   33019,   33279,   33538,   33797,   34055,
      34312,   34569,   34825,   35080,   35334,   35588,   35841,   36094,
      36346,   36597,   36847,   37097,   37346,   37595,   37842,   38090,
      38336,   38582,   38827,   39072,   39316,   39559,   39802,   40044,
      40286,   40527,   40767,   41006,   41246,   41484,   41722,   41959,
      42196,   42432,   42667,   42902,   43137,   43370,   43603,   43836,
      44068,   44300,   44530,   44761,   44990,   45220,   45448,   45676,
      45904,   46131,   46357,   46583,   46809,   47034,   47258,   47482,
      47705,   47928,   48150,   48372,   48593,   48813,   49034,   49253,
      49472,   49691,   49909,   50127,   50344,   50560,   50776,   50992,
      51207,   51422,   51636,   51850,   52063,   52276,   52488,   52700,
      52911,   53122,   53332,   53542,   53751,   53960,   54169,   54377,
      54584,   54791,   54998,   55204,   55410,   55615,   55820,   56025,
      56229,   56432,   56635,   56838,   57040,   57242,   57443,   57644,
      57845,   58045,   58245,   58444,   58643,   58841,   59039,   59237,
      59434,   59631,   59827,   60023,   60219,   60414,   60609,   60803,
      60997,   61190,   61384,   61576,   61769,   61961,   62152,   62343,
      62534,   62725,   62915,   63104,   63294,   63483,   63671,   63859,
      64047,   64234,   64421,   64608,   64794,   64980,   65166,   65351,
      65536,
];

// log2(x) in 16.16, rounded, for x in [0 .. 255], log2(0) is 0
// (from log2Squaring, see the unittest).
static immutable uint[256] kLog2Small =
[
          0,       0,   65536,  103872,  131072,  152170,  169408,  183983,
     196608,  207744,  217706,  226717,  234944,  242512,  249519,  256042,
     262144,  267876,  273280,  278392,  283242,  287855,  292253,  296456,
     300480,  304340,  308048,  311616,  315055,  318373,  321578,  324678,
     327680,  330589,  333412,  336153,  338816,  341407,  343928,  346384,
     348778,  351113,  353391,  355616,  357789,  359914,  361992,  364026,
     366016,  367966,  369876,  371748,  373584,  375385,  377152,  378887,
     380591,  382264,  383909,  385525,  387114,  388677,  390214,  391727,
     393216,  394682,  396125,  397547,  398948,  400328,  401689,  403030,
     404352,  405656,  406943,  408212,  409464,  410700,  411920,  413125,
     414314,  415488,  416649,  417795,  418927,  420046,  421152,  422245,
     423325,  424394,  425450,  426495,  427528,  428550,  429562,  430562,
     431552,  432532,  433502,  434462,  435412,  436353,  437284,  438206,
     439120,  440025,  440921,  441809,  442688,  443560,  444423,  445279,
     446127,  446967,  447800,  448626,  449445,  450256,  451061,  451859,
     452650,  453435,  454213,  454985,  455750,  456510,  457263,  458010,
     458752,  459488,  460218,  460942,  461661,  462375,  463083,  463786,
     464484,  465177,  465864,  466547,  467225,  467898,  468566,  469229,
     469888,  470543,  471192,  471838,  472479,  473115,  473748,  474376,
     475000,  475620,  476236,  476848,  477456,  478060,  478661,  479257,
     479850,  480439,  481024,  481606,  482185,  482759,  483331,  483898,
     484463,  485024,  485582,  486136,  486688,  487236,  487781,  488323,
     488861,  489397,  489930,  490459,  490986,  491510,  492031,  492549,
     493064,  493577,  494086,  494593,  495098,  495599,  496098,  496594,
     497088,  497579,  498068,  498554,  499038,  499519,  499998,  500474,
     500948,  501419,  501889,  502355,  502820,  503282,  503742,  504200,
     504656,  505109,  505561,  506010,  506457,  506902,  507345,  507786,
     508224,  508661,  509096,  509528,  509959,  510388,  510815,  511240,
     511663,  512084,  512503,  512921,  513336,  513750,  514162,  514572,
     514981,  515387,  515792,  516195,  516597,  516997,  517395,  517791,
     518186,  518579,  518971,  519361,  519749,  520136,  520521,  520904,
     521286,  521667,  522046,  522423,  522799,  523173,  523546,  523918,
];

// log2(x) in 16.16 fixed point, within 1/65536 of the exact value.
// log2(0) is 0. At most 64 << 16, it fits in a uint.
pragma(inline, true)
uint log2Fixed(size_t x)
{
    if (x < 256)
        return kLog2Small[x]; // the most common case, exact
    uint e = bsr(x);
    // Mantissa in [1, 2), in 2.30 fixed point: the 8 bits after the leading
    // one index the table, the next 16 interpolate.
    uint m = cast(uint)(e >= 30 ? x >> (e - 30) : x << (30 - e));
    uint i = (m >> 22) & 255;
    uint f = (m >> 6) & 0xFFFF;
    uint a = kLog2Mantissa[i];
    uint b = kLog2Mantissa[i + 1];
    return (e << 16) + a + (((b - a) * f + 32768) >> 16);
}

bool isEmpty(scope const(uint)[] h)
{
    foreach (c; h)
        if (c != 0)
            return false;
    return true;
}

// Shannon entropy of the symbols of `h`, at least 1 bit per symbol.
ulong bitsEntropy(scope const(uint)[] h)
{
    ulong total = 0;
    ulong sum = 0; // of c * log2(c)
    foreach (c; h)
    {
        total += c;
        sum += cast(ulong) c * log2Fixed(c);
    }
    ulong totalBits = total * log2Fixed(cast(size_t) total);
    ulong bits = totalBits > sum ? totalBits - sum : 0;
    ulong minimum = total * COST_ONE_BIT;
    return bits < minimum ? minimum : bits;
}

// Estimated sizes of prefix codes in bits: simple codes of 1 to 4 symbols,
// complex codes (plus the entropy of their code lengths).
@tuning enum ulong ONE_SYMBOL_CODE_BITS = 12;
@tuning enum ulong TWO_SYMBOLS_CODE_BITS = 20;
@tuning enum ulong THREE_SYMBOLS_CODE_BITS = 28;
@tuning enum ulong FOUR_SYMBOLS_CODE_BITS = 37;
@tuning enum ulong COMPLEX_CODE_BITS = 18;
@tuning enum ulong COMPLEX_CODE_BITS_PER_DEPTH = 2; // per bit of the longest code

// Estimated size of symbols coded with a prefix code, including the
// prefix code itself (same model as the reference encoder). The non-zero
// counts are given in increasing symbol order, with `add`.
struct PopulationCost
{
pure nothrow @nogc @safe:
    uint log2Total;
    ulong bits;
    uint maxDepth = 1;
    uint[18] depthHistogram;
    uint[4] firstCounts;
    size_t count;
    size_t next;    // symbol after the last one added

    this(ulong total)
    {
        log2Total = log2Fixed(cast(size_t) total);
    }

    pragma(inline, true)
    void add(size_t symbol, uint c)
    {
        assert(c != 0 && symbol >= next);
        if (count < 4)
            firstCounts[count] = c;
        count++;

        // The zero run before: the prefix code would use code 17 for it.
        size_t reps = symbol - next;
        next = symbol + 1;
        if (reps > 0)
        {
            if (reps < 3)
                depthHistogram[0] += reps;
            else
            {
                reps -= 2;
                while (reps > 0)
                {
                    depthHistogram[REPEAT_ZERO_CODE_LENGTH]++;
                    bits += 3 * COST_ONE_BIT;
                    reps >>= 3;
                }
            }
        }

        // Entropy of the symbol, and its approximate code length.
        uint log2c = log2Fixed(c);
        uint log2p = log2Total > log2c ? log2Total - log2c : 0;
        uint depth = (log2p + COST_ONE_BIT / 2) >> 16;
        bits += cast(ulong) c * log2p;
        if (depth > 15)
            depth = 15;
        if (depth > maxDepth)
            maxDepth = depth;
        depthHistogram[depth]++;
    }

    ulong result()
    {
        alias s = firstCounts;
        if (count <= 1)
            return ONE_SYMBOL_CODE_BITS * COST_ONE_BIT;
        if (count == 2)
            return (TWO_SYMBOLS_CODE_BITS + s[0] + s[1]) * COST_ONE_BIT;
        if (count == 3)
        {
            ulong max = s[0] > s[1] ? s[0] : s[1];
            if (s[2] > max)
                max = s[2];
            return (THREE_SYMBOLS_CODE_BITS + 2 * (cast(ulong) s[0] + s[1] + s[2]) - max) * COST_ONE_BIT;
        }
        if (count == 4)
        {
            // Sort decreasing.
            foreach (i; 1 .. 4)
                for (size_t j = i; j > 0 && s[j] > s[j - 1]; --j)
                {
                    uint t = s[j];
                    s[j] = s[j - 1];
                    s[j - 1] = t;
                }
            ulong h23 = cast(ulong) s[2] + s[3];
            ulong max = h23 > s[0] ? h23 : s[0];
            return (FOUR_SYMBOLS_CODE_BITS + 3 * h23 + 2 * (cast(ulong) s[0] + s[1]) - max) * COST_ONE_BIT;
        }
        return bits + (COMPLEX_CODE_BITS + COMPLEX_CODE_BITS_PER_DEPTH * maxDepth) * COST_ONE_BIT
             + bitsEntropy(depthHistogram[]);
    }
}

ulong populationCost(scope const(uint)[] h)
{
    ulong total = 0;
    foreach (c; h)
        total += c;
    auto pc = PopulationCost(total);
    foreach (i, c; h)
        if (c != 0)
            pc.add(i, c);
    return pc.result();
}

// Clustering works on at most that many histograms.
enum MAX_CLUSTER_INPUTS = 256;

// Cluster `n` histograms of `alphabetSize` symbols each (histogram i is
// histograms[i * alphabetSize .. (i + 1) * alphabetSize]), into at most
// `maxClusters`, merging greedily as long as it saves bits.
// Writes the cluster of each histogram to `map`, numbered in order of first
// use, and the summed histograms of the clusters to `clusters`. Returns the
// number of clusters, and their estimated cost in bits.
// `clusters` and `scratch` must have room for `n` histograms, `deltaScratch`
// for n * n values.
uint clusterHistograms(scope const(uint)[] histograms, size_t n, size_t alphabetSize,
                       uint maxClusters, scope ubyte[] map, scope uint[] clusters,
                       scope ushort[] scratch, scope long[] deltaScratch, out ulong totalCost)
{
    assert(n >= 1 && n <= MAX_CLUSTER_INPUTS && maxClusters >= 1);
    assert(alphabetSize <= MAX_ALPHABET_SIZE);
    assert(histograms.length >= n * alphabetSize && clusters.length >= n * alphabetSize);
    assert(scratch.length >= n * alphabetSize && deltaScratch.length >= n * n);

    long[MAX_CLUSTER_INPUTS] cost;
    ulong[MAX_CLUSTER_INPUTS] total;
    bool[MAX_CLUSTER_INPUTS] active;
    ubyte[MAX_CLUSTER_INPUTS] owner;
    size_t[MAX_CLUSTER_INPUTS] numSymbols;
    ushort[MAX_ALPHABET_SIZE] tmpSymbols;

    // Cost change if clusters i < j were merged.
    ref long delta(size_t i, size_t j) { return deltaScratch[i * n + j]; }

    const(uint)[] input(size_t i) { return histograms[i * alphabetSize .. (i + 1) * alphabetSize]; }
    uint[] cluster(size_t i) { return clusters[i * alphabetSize .. (i + 1) * alphabetSize]; }
    // Non-zero symbols of cluster i, in increasing order.
    ushort[] symbols(size_t i) { return scratch[i * alphabetSize .. i * alphabetSize + numSymbols[i]]; }

    // Cost of the sum of cluster `a` (or of `aCounts` with non-zero symbols
    // `aSymbols`) and cluster `b`.
    long mergedCost(scope const(uint)[] aCounts, scope const(ushort)[] aSymbols, ulong aTotal, size_t b)
    {
        const(uint)[] bCounts = cluster(b);
        const(ushort)[] bSymbols = symbols(b);
        auto pc = PopulationCost(aTotal + total[b]);
        size_t i = 0, j = 0;
        while (i < aSymbols.length || j < bSymbols.length)
        {
            size_t s;
            if (j == bSymbols.length || (i < aSymbols.length && aSymbols[i] < bSymbols[j]))
                s = aSymbols[i++];
            else if (i == aSymbols.length || bSymbols[j] < aSymbols[i])
                s = bSymbols[j++];
            else
            {
                s = aSymbols[i++];
                j++;
            }
            pc.add(s, aCounts[s] + bCounts[s]);
        }
        return pc.result();
    }

    uint numActive = 0;
    foreach (i; 0 .. n)
    {
        uint[] c = cluster(i);
        c[] = input(i)[];
        size_t k = 0;
        total[i] = 0;
        foreach (sym, v; c)
        {
            if (v != 0)
            {
                scratch[i * alphabetSize + k++] = cast(ushort) sym;
                total[i] += v;
            }
        }
        numSymbols[i] = k;
        active[i] = k > 0;
        cost[i] = active[i] ? populationCost(c) : 0;
        owner[i] = cast(ubyte) i;
        if (active[i])
            numActive++;
    }

    if (numActive == 0)
    {
        map[0 .. n] = 0;
        cluster(0)[] = 0;
        totalCost = populationCost(cluster(0));
        return 1;
    }
    bool[MAX_CLUSTER_INPUTS] empty;
    foreach (i; 0 .. n)
        empty[i] = !active[i];

    foreach (i; 0 .. n)
        foreach (j; i + 1 .. n)
            if (active[i] && active[j])
                delta(i, j) = mergedCost(cluster(i), symbols(i), total[i], j) - cost[i] - cost[j];

    // Greedy merging.
    while (numActive > 1)
    {
        size_t bi = 0, bj = 0;
        long best = long.max;
        foreach (i; 0 .. n)
        {
            if (!active[i])
                continue;
            foreach (j; i + 1 .. n)
                if (active[j] && delta(i, j) < best)
                {
                    best = delta(i, j);
                    bi = i;
                    bj = j;
                }
        }
        if (best >= 0 && numActive <= maxClusters)
            break;

        // Merge bj into bi.
        uint[] ci = cluster(bi);
        foreach (s; symbols(bj))
            ci[s] += cluster(bj)[s];
        size_t k = 0;
        foreach (sym, v; ci)
            if (v != 0)
                tmpSymbols[k++] = cast(ushort) sym;
        numSymbols[bi] = k;
        symbols(bi)[] = tmpSymbols[0 .. k];
        total[bi] += total[bj];
        cost[bi] += cost[bj] + best;
        active[bj] = false;
        numActive--;
        foreach (i; 0 .. n)
            if (owner[i] == bj)
                owner[i] = cast(ubyte) bi;
        foreach (j; 0 .. n)
        {
            if (!active[j] || j == bi)
                continue;
            long d = mergedCost(ci, symbols(bi), total[bi], j) - cost[bi] - cost[j];
            if (j < bi)
                delta(j, bi) = d;
            else
                delta(bi, j) = d;
        }
    }

    // Move each histogram to the cluster where it fits best.
    if (numActive > 1)
    {
        foreach (i; 0 .. n)
        {
            if (empty[i])
                continue;
            const(uint)[] h = input(i);
            size_t k = 0;
            ulong t = 0;
            foreach (sym, v; h)
            {
                if (v != 0)
                {
                    tmpSymbols[k++] = cast(ushort) sym;
                    t += v;
                }
            }
            long bestCost = long.max;
            foreach (j; 0 .. n)
            {
                if (!active[j])
                    continue;
                long d = mergedCost(h, tmpSymbols[0 .. k], t, j) - cost[j];
                if (d < bestCost)
                {
                    bestCost = d;
                    owner[i] = cast(ubyte) j;
                }
            }
        }
    }

    // Number the clusters by first use. Empty histograms take the cluster of
    // the previous one, which codes well in the context map.
    ushort[MAX_CLUSTER_INPUTS] number = ushort.max;
    uint numClusters = 0;
    foreach (i; 0 .. n)
    {
        if (empty[i])
            continue;
        if (number[owner[i]] == ushort.max)
            number[owner[i]] = cast(ushort) numClusters++;
        map[i] = cast(ubyte) number[owner[i]];
    }
    ubyte previous = 0;
    foreach (i; 0 .. n)
    {
        if (!empty[i])
        {
            previous = map[i];
            break;
        }
    }
    foreach (i; 0 .. n)
    {
        if (empty[i])
            map[i] = previous;
        previous = map[i];
    }

    clusters[0 .. numClusters * alphabetSize] = 0;
    foreach (i; 0 .. n)
    {
        uint[] c = cluster(map[i]);
        foreach (k, v; input(i))
            c[k] += v;
    }
    totalCost = 0;
    foreach (k; 0 .. numClusters)
        totalCost += populationCost(cluster(k));
    return numClusters;
}

// ---------------------------------------------------------------------------
// Encoder: optimal parsing
// ---------------------------------------------------------------------------

// Binary tree match finder: a binary search tree of the window positions
// with the same 4-byte hash, sorted by the data that follows them. Finds
// every match length at a position, with the smallest distance found for it.
@tuning enum TREE_BUCKET_BITS = 17;
// The trees cover at most that much history: 8 bytes of memory per byte.
@tuning enum MAX_TREE_WINDOW = 1 << 22;
@tuning enum MAX_TREE_SEARCH_DEPTH = 64;
@tuning enum MAX_TREE_COMP_LENGTH = 128;
@tuning enum SHORT_MATCH_SEARCH_DISTANCE = 64; // for matches of 2 and 3 bytes
@tuning enum MISSING_SYMBOL_EXTRA_BITS = 2; // cost of an absent symbol, above the rarest
@tuning enum INITIAL_COMMAND_COST_OFFSET = 11; // first pass: symbol i costs log2(this + i)
@tuning enum INITIAL_DISTANCE_COST_OFFSET = 20;
@tuning enum OPTIMAL_MATCH_STARTS = 2; // starts tried with all the matches, the others only with short codes
@tuning enum long START_MAX_COST_DIFF = 0; // command ends worse than literals don't start inserts
@tuning enum LONG_COPY_QUICK_STEP = 16384; // skip the parse inside copies at least that long
// Literal costs after the first pass: weighted average of the costs with
// and without contexts.
@tuning enum uint LITERAL_CONTEXT_WEIGHT = 1;
@tuning enum uint LITERAL_ORDER0_WEIGHT = 1;
enum uint TREE_INVALID = uint.max;

struct TreeHasher
{
pure nothrow @nogc @safe:
    MallocArray!uint buckets; // root of each tree, relative to `base`
    MallocArray!uint forest;  // children of position p: forest[2 * (p & windowMask)], [+ 1]
    size_t windowMask;
    size_t base;
    size_t dictionaryLookups;
    size_t dictionaryMatches;
    OmitFirstIndex omitFirst;

    // windowSize: power of 2, at most MAX_TREE_WINDOW. Matches are found
    // up to windowSize - 16 bytes back.
    bool initialize(size_t windowSize, bool useDictionary)
    {
        assert(windowSize <= MAX_TREE_WINDOW);
        windowMask = windowSize - 1;
        if (!buckets.resize(size_t(1) << TREE_BUCKET_BITS) || !forest.resize(2 * windowSize))
            return false;
        if (useDictionary && !omitFirst.initialize())
            return false;
        buckets[] = TREE_INVALID; // forest entries are written when a position is inserted
        base = 0;
        return true;
    }

    void release()
    {
        buckets.release();
        forest.release();
        omitFirst.release();
    }

    // Positions are 32-bit relative to `base`: move `base` forward when they
    // would overflow. Positions older than the window are useless.
    void rebase(size_t pos, size_t window)
    {
        if (pos - base < 0x8000_0000)
            return;
        size_t shift = pos - base - window;
        foreach (ref p; buckets)
            p = (p != TREE_INVALID && p >= shift) ? cast(uint)(p - shift) : TREE_INVALID;
        foreach (ref p; forest)
            p = (p != TREE_INVALID && p >= shift) ? cast(uint)(p - shift) : TREE_INVALID;
        base += shift;
    }

    pragma(inline, true)
    static uint hashAt(scope const(ubyte)[] data, size_t pos)
    {
        return (load32(data, pos) * 0x1E35A7BDu) >> (32 - TREE_BUCKET_BITS);
    }

    // Insert `pos` in its tree, which needs 4 bytes of data. If `record`,
    // also append the matches better than `bestLen` (up to `maxLength`
    // bytes, at distances <= maxDistance) to `matches`, by increasing length.
    void storeAndFindMatches(bool record)(scope const(ubyte)[] data, size_t pos, size_t maxLength,
                                          size_t maxDistance, ref size_t bestLen, ref MatchBuffer matches)
    {
        if (maxDistance > windowMask - 15)
            maxDistance = windowMask - 15;
        // Tree comparisons may look past maxLength, the tree is for the whole data.
        size_t limit = data.length - pos;
        size_t maxCompLength = limit < MAX_TREE_COMP_LENGTH ? limit : MAX_TREE_COMP_LENGTH;
        bool reroot = limit >= MAX_TREE_COMP_LENGTH; // else the tree order can't be kept
        // Comparisons need maxCompLength bytes, reports need maxLength: more
        // would make long repeats quadratic.
        static if (record)
            size_t scanLimit = maxLength > maxCompLength ? maxLength : maxCompLength;
        else
            size_t scanLimit = maxCompLength;
        if (scanLimit > limit)
            scanLimit = limit;
        uint key = hashAt(data, pos);
        uint prev = buckets[key];
        size_t nodeLeft = 2 * (pos & windowMask);
        size_t nodeRight = nodeLeft + 1;
        size_t bestLenLeft = 0;
        size_t bestLenRight = 0;
        if (reroot)
            buckets[key] = cast(uint)(pos - base);
        for (size_t depth = MAX_TREE_SEARCH_DEPTH; ; --depth)
        {
            size_t prevPos = base + prev;
            size_t backward = pos - prevPos;
            if (prev == TREE_INVALID || prevPos >= pos || backward > maxDistance || depth == 0)
            {
                if (reroot)
                {
                    forest[nodeLeft] = TREE_INVALID;
                    forest[nodeRight] = TREE_INVALID;
                }
                break;
            }
            size_t common = bestLenLeft < bestLenRight ? bestLenLeft : bestLenRight;
            size_t len = common + matchLength(data, prevPos + common, pos + common, scanLimit - common);
            static if (record)
            {
                size_t reported = len < maxLength ? len : maxLength;
                if (reported > bestLen)
                {
                    bestLen = reported;
                    matches.push(BackwardMatch(cast(uint) backward, cast(uint) reported, 0));
                }
            }
            if (len >= maxCompLength)
            {
                // Same data as far as the tree cares: take over its children.
                if (reroot)
                {
                    forest[nodeLeft] = forest[2 * (prevPos & windowMask)];
                    forest[nodeRight] = forest[2 * (prevPos & windowMask) + 1];
                }
                break;
            }
            if (data[pos + len] > data[prevPos + len])
            {
                bestLenLeft = len;
                if (reroot)
                    forest[nodeLeft] = prev;
                nodeLeft = 2 * (prevPos & windowMask) + 1;
                prev = forest[nodeLeft];
            }
            else
            {
                bestLenRight = len;
                if (reroot)
                    forest[nodeRight] = prev;
                nodeRight = 2 * (prevPos & windowMask);
                prev = forest[nodeRight];
            }
        }
    }
}

// A match: distance, bytes produced, and for a dictionary word its length.
struct BackwardMatch
{
    uint distance;
    uint length;
    uint wordLength; // 0 for backward references
}

// All the matches at data[pos .. pos + maxLength], by increasing length:
// very close ones, from the tree, and static dictionary words.
void findAllMatches(ref TreeHasher t, scope const(ubyte)[] data, size_t pos, size_t maxLength,
                    size_t maxDistance, ref const(LevelParams) params, ref MatchBuffer matches)
{
    size_t bestLen = 1;

    // Short distances, for lengths 2 and 3 that the tree doesn't find.
    size_t stop = pos > SHORT_MATCH_SEARCH_DISTANCE ? pos - SHORT_MATCH_SEARCH_DISTANCE : 0;
    for (size_t prev = pos - 1; prev > stop && bestLen <= 2; --prev)
    {
        size_t backward = pos - prev;
        if (backward > maxDistance)
            break;
        if (data[prev] != data[pos] || data[prev + 1] != data[pos + 1])
            continue;
        size_t len = matchLength(data, prev, pos, maxLength);
        if (len > bestLen)
        {
            bestLen = len;
            matches.push(BackwardMatch(cast(uint) backward, cast(uint) len, 0));
        }
    }
    if (bestLen < maxLength)
        t.storeAndFindMatches!true(data, pos, maxLength, maxDistance, bestLen, matches);
    else
        t.storeAndFindMatches!false(data, pos, maxLength, maxDistance, bestLen, matches);

    // Dictionary words that produce more than that.
    if (!params.useDictionary || bestLen >= MAX_TRANSFORMED_WORD_LENGTH || maxLength < 4
        || t.dictionaryMatches < (t.dictionaryLookups >> DICTIONARY_THROTTLE_SHIFT))
        return;
    t.dictionaryLookups++;
    DictionaryReferences refs;
    if (!findDictionaryReferences(data, pos, maxLength, maxDistance, bestLen, t.omitFirst, refs))
        return;
    t.dictionaryMatches++;
    immutable(ubyte)[] dict = dictionary();
    foreach (l; bestLen + 1 .. MAX_TRANSFORMED_WORD_LENGTH + 1)
    {
        if (refs.distance[l] == size_t.max)
            continue;
        // Check with the decoder's transform: this is what it will produce.
        size_t len = refs.wordLength[l];
        size_t offset = kDictionaryOffsets[len] + len * refs.wordIndex[l];
        ubyte[MAX_TRANSFORMED_WORD_LENGTH] transformed;
        size_t n = transformWord(transformed[], dict[offset .. offset + len], kTransforms[refs.transform[l]]);
        bool same = n == l && transformed[0 .. n] == data[pos .. pos + n];
        assert(same);
        if (same)
            matches.push(BackwardMatch(cast(uint) refs.distance[l], cast(uint) l, cast(uint) len));
    }
}

// Cost model of the optimal parser, in bits.
struct ZopfliCostModel
{
    uint[NUM_COMMAND_CODES] commandCost;
    uint[16 + 48] distanceCost; // NPOSTFIX = 0, NDIRECT = 0
    uint minCommandCost;
}

// Costs from a histogram: Shannon bits, at least 1, a bit more than the
// rarest for absent symbols.
void setCosts(scope const(uint)[] histogram, scope uint[] cost, bool literals)
{
    size_t sum = 0;
    size_t missing = 0;
    foreach (c; histogram)
    {
        sum += c;
        if (c == 0)
            missing++;
    }
    uint log2Sum = log2Fixed(sum);
    uint missingCost = log2Fixed(literals ? sum : sum + missing) + MISSING_SYMBOL_EXTRA_BITS * COST_ONE_BIT;
    foreach (i, c; histogram)
    {
        if (c == 0)
        {
            cost[i] = missingCost;
            continue;
        }
        uint log2c = log2Fixed(c);
        uint bits = log2Sum > log2c ? log2Sum - log2c : 0;
        cost[i] = bits < COST_ONE_BIT ? COST_ONE_BIT : bits;
    }
}

enum long UNREACHED = long.max;

struct ZopfliNode
{
    long cost = UNREACHED;
    uint start;         // where the insert of the command ending here starts
    uint copyLength;    // as coded, 0 if not reached by a command
    uint outputLength;
    uint distance;
    uint[4] cache;      // last distances, set once the node is a start
}

// Possible start of the insert of the next command: a position where a
// command ends, with its cost minus the cost of the literals before it.
struct StartPosition
{
    size_t pos;
    long costDiff;
}

// Up to 8 start positions, sorted by costDiff. When full, a push replaces
// the most expensive one (even if the new one is more expensive).
struct StartPosQueue
{
pure nothrow @nogc @safe:
    StartPosition[8] q;
    size_t count;

    size_t size() const
    {
        return count < 8 ? count : 8;
    }

    void push(StartPosition p)
    {
        size_t offset = ~(count++) & 7;
        size_t len = size();
        q[offset] = p;
        foreach (i; 1 .. len)
        {
            if (q[offset & 7].costDiff > q[(offset + 1) & 7].costDiff)
            {
                StartPosition t = q[offset & 7];
                q[offset & 7] = q[(offset + 1) & 7];
                q[(offset + 1) & 7] = t;
            }
            ++offset;
        }
    }

    StartPosition opIndex(size_t k) const
    {
        return q[(k - count) & 7];
    }
}

// Last distances after the command ending at node `e` (relative to the block start).
void computeNodeCache(scope ZopfliNode[] nodes, size_t e, size_t blockStart, size_t maxBackward)
{
    ZopfliNode* node = &nodes[e];
    uint[4] cache = nodes[node.start].cache;
    size_t copyPos = blockStart + e - node.outputLength;
    size_t maxDistance = copyPos < maxBackward ? copyPos : maxBackward;
    if (node.distance <= maxDistance) // not a dictionary word
    {
        DistanceCache last = [cache[0], cache[1], cache[2], cache[3]];
        uint code = distanceCode(node.distance, last);
        if (code != 0)
            cache = [node.distance, cache[0], cache[1], cache[2]];
    }
    node.cache = cache;
}

// Optimal parse of data[start .. end] with the matches of each position, for
// the cost model. literalCosts[i] is the cost of the literals of
// data[start .. start + i]. Fills nodes[0 .. end - start + 1].
void zopfliShortestPath(scope const(ubyte)[] data, size_t start, size_t end, size_t maxBackward,
                        ref const(DistanceCache) last, scope const(uint)[] numMatches,
                        scope const(BackwardMatch)[] matches, ref const(ZopfliCostModel) model,
                        scope const(long)[] literalCosts, ref const(LevelParams) params,
                        scope ZopfliNode[] nodes)
{
    size_t n = end - start;
    size_t maxZopfliLen = params.optimalMaxLength;
    size_t maxStarts = params.optimalStarts;
    foreach (ref node; nodes[0 .. n + 1])
        node = ZopfliNode.init;
    nodes[0].cost = 0;
    foreach (k; 0 .. 4)
        nodes[0].cache[k] = cast(uint) last[k];

    // Cost of the insert-and-copy symbol and copy extra bits, for each
    // (insert code, copy code): with an explicit distance, and with the last
    // distance (uint.max if it can't be implicit).
    uint[24][24] withDistance;
    uint[24][24] withLastDistance;
    foreach (ic; 0 .. 24)
    {
        foreach (cc; 0 .. 24)
        {
            uint extra = kCopyLengthBits[cc] * COST_ONE_BIT;
            withDistance[ic][cc] = model.commandCost[commandSymbol(ic, cc, false)] + extra;
            uint implicit = commandSymbol(ic, cc, true);
            withLastDistance[ic][cc] = implicit < 128 ? model.commandCost[implicit] + extra : uint.max;
        }
    }

    StartPosQueue queue;
    size_t matchIndex = 0;
    size_t i = 0;
    while (i < n)
    {
        const(BackwardMatch)[] here = matches[matchIndex .. matchIndex + numMatches[i]];
        matchIndex += numMatches[i];

        // A command ends here: the next one may start its insert here.
        if (nodes[i].cost != UNREACHED && (i == 0 || nodes[i].copyLength != 0))
        {
            if (i != 0)
                computeNodeCache(nodes, i, start, maxBackward);
            long costDiff = nodes[i].cost - literalCosts[i];
            if (costDiff <= START_MAX_COST_DIFF || i == 0)
                queue.push(StartPosition(i, costDiff));
        }

        size_t skip = 0;
        if (queue.size() > 0 && n - i >= 2)
        {
            size_t pos = start + i;
            size_t maxLength = n - i;
            size_t maxDistance = pos < maxBackward ? pos : maxBackward;

            // Lengths that can't reach a node cheaper than it already is
            // are not worth trying.
            long minCost = queue[0].costDiff + literalCosts[i] + model.minCommandCost;
            size_t minLen = 2;
            {
                size_t bucket = 4, nextOffset = 10;
                while (i + minLen <= n && nodes[i + minLen].cost <= minCost)
                {
                    minLen++;
                    if (minLen == nextOffset)
                    {
                        minCost += COST_ONE_BIT;
                        nextOffset += bucket;
                        bucket *= 2;
                    }
                }
            }

            size_t longest = 0;
            foreach (k; 0 .. queue.size() < maxStarts ? queue.size() : maxStarts)
            {
                StartPosition sp = queue[k];
                size_t insertLength = i - sp.pos;
                uint insertCode = insertLengthCode(insertLength);
                long baseCost = sp.costDiff + literalCosts[i] + kInsertLengthBits[insertCode] * COST_ONE_BIT;
                const(uint)[] cache = nodes[sp.pos].cache[];
                const(uint)[] explicitCost = withDistance[insertCode][];
                const(uint)[] lastDistanceCost = withLastDistance[insertCode][];

                void update(size_t len, size_t codedLength, size_t distance, long cost)
                {
                    ZopfliNode* node = &nodes[i + len];
                    if (cost < node.cost)
                    {
                        node.cost = cost;
                        node.start = cast(uint) sp.pos;
                        node.copyLength = cast(uint) codedLength;
                        node.outputLength = cast(uint) len;
                        node.distance = cast(uint) distance;
                        if (len > longest)
                            longest = len;
                    }
                }

                // Distances of the short codes.
                size_t bestLen = minLen - 1;
                foreach (code; 0 .. 16)
                {
                    if (bestLen >= maxLength)
                        break;
                    size_t ci = kDistanceShortCodeIndexOffset[code] == 0 ? 0 : 4 - kDistanceShortCodeIndexOffset[code];
                    long d = cast(long) cache[ci] + kDistanceShortCodeValueOffset[code];
                    if (d <= 0 || d > maxDistance)
                        continue;
                    size_t prev = pos - cast(size_t) d;
                    if (data[prev + bestLen] != data[pos + bestLen])
                        continue;
                    size_t len = matchLength(data, prev, pos, maxLength);
                    if (len <= bestLen)
                        continue;
                    long distanceCost = baseCost + model.distanceCost[code];
                    size_t l = bestLen + 1;
                    for (; l <= len; ++l)
                    {
                        if (l > maxZopfliLen && l < len)
                            l = len; // long copies: only the full length
                        uint copyCode = fastCopyLengthCode(l);
                        long cost;
                        if (code == 0 && lastDistanceCost[copyCode] != uint.max)
                            cost = baseCost + lastDistanceCost[copyCode];
                        else
                            cost = distanceCost + explicitCost[copyCode];
                        update(l, l, cast(size_t) d, cost);
                    }
                    bestLen = len;
                }

                // Other matches, from the best starts only.
                if (k >= OPTIMAL_MATCH_STARTS)
                    continue;
                size_t len = minLen;
                foreach (ref const m; here)
                {
                    bool isDictionary = m.wordLength != 0;
                    uint dsym, dbits, dextra;
                    distanceSymbol(m.distance + 15, 0, 0, dsym, dbits, dextra);
                    long distanceCost = baseCost + dbits * COST_ONE_BIT + model.distanceCost[dsym];
                    size_t maxMatchLen = m.length;
                    if (len < maxMatchLen && (isDictionary || maxMatchLen > maxZopfliLen))
                        len = maxMatchLen;
                    for (; len <= maxMatchLen; ++len)
                    {
                        size_t codedLength = isDictionary ? m.wordLength : len;
                        long cost = distanceCost + explicitCost[fastCopyLengthCode(codedLength)];
                        update(len, codedLength, m.distance, cost);
                    }
                }
            }

            if (longest >= LONG_COPY_QUICK_STEP)
                skip = longest;
            if (here.length == 1 && here[0].length > maxZopfliLen && here[0].length > skip)
                skip = here[0].length;
        }
        i++;

        // Inside a long copy: no new commands start there, only commands
        // may end there.
        if (skip > 1)
        {
            for (size_t s = skip - 1; s > 0 && i < n; --s)
            {
                matchIndex += numMatches[i];
                if (nodes[i].cost != UNREACHED && nodes[i].copyLength != 0)
                {
                    computeNodeCache(nodes, i, start, maxBackward);
                    long costDiff = nodes[i].cost - literalCosts[i];
                    if (costDiff <= START_MAX_COST_DIFF)
                        queue.push(StartPosition(i, costDiff));
                }
                i++;
            }
        }
    }
}

// Commands of the cheapest path to the end of the block, from the nodes of
// zopfliShortestPath. Appends them to `cmds` and updates `last`.
bool zopfliCreateCommands(scope ZopfliNode[] nodes, size_t start, size_t end, size_t maxBackward,
                          scope const(long)[] literalCosts, ref const(ZopfliCostModel) model,
                          ref DistanceCache last, ref CommandBuffer cmds)
{
    size_t n = end - start;
    // End of the last command, the rest is inserted.
    size_t bestEnd = 0;
    long bestCost = long.max;
    foreach (e; 0 .. n + 1)
    {
        if (nodes[e].cost == UNREACHED || (e != 0 && nodes[e].copyLength == 0))
            continue;
        long cost = nodes[e].cost;
        if (e < n)
        {
            uint insertCode = insertLengthCode(n - e);
            cost += literalCosts[n] - literalCosts[e] + kInsertLengthBits[insertCode] * COST_ONE_BIT
                  + model.commandCost[commandSymbol(insertCode, 0, true)];
        }
        if (cost < bestCost)
        {
            bestCost = cost;
            bestEnd = e;
        }
    }

    // Walk back, linking each node to the next one with `start`.
    size_t numCommands = 0;
    size_t next = n + 1; // sentinel
    for (size_t e = bestEnd; e != 0; )
    {
        size_t s = nodes[e].start;
        nodes[e].start = cast(uint) next;
        next = e;
        e = s;
        numCommands++;
    }

    // Forward: commands, with their real distance codes.
    size_t pos = 0;
    for (size_t e = next; e != n + 1; e = nodes[e].start)
    {
        ZopfliNode node = nodes[e];
        size_t copyPos = e - node.outputLength;
        Command c;
        c.insertLength = cast(uint)(copyPos - pos);
        c.copyLength = node.copyLength;
        c.outputLength = node.outputLength;
        size_t maxDistance = start + copyPos < maxBackward ? start + copyPos : maxBackward;
        c.distanceCode = distanceCode(node.distance, last);
        if (!cmds.push(c))
            return false;
        if (node.distance <= maxDistance)
            updateDistanceCache(last, node.distance, c.distanceCode);
        pos = e;
    }
    assert(pos == bestEnd);
    if (pos < n)
    {
        Command c;
        c.insertLength = cast(uint)(n - pos);
        if (!cmds.push(c))
            return false;
    }
    return true;
}

// Histograms of the literals of `cmds` (or of all the bytes of
// data[start .. start + n] if there are no commands) in each context of each
// mode, into ws.modeHistograms[mode][context][symbol]. `bestMode` is the
// cheapest mode, from a quick estimate: the cost of each context, without
// clustering. Returns false if out of memory.
bool literalModeHistograms(scope const(ubyte)[] data, size_t start, size_t n,
                           scope const(Command)[] cmds, ref EncoderWorkspace ws, out uint bestMode)
{
    enum A = NUM_LITERAL_CODES;
    if (!ws.modeHistograms.ensure(4 * 64 * A))
        return false;
    uint[] mh = ws.modeHistograms[0 .. 4 * 64 * A];
    mh[] = 0;
    uint[4 * 64] totals; // literals in each histogram

    void count(size_t from, size_t to)
    {
        ubyte p1 = from >= 1 ? data[from - 1] : 0;
        ubyte p2 = from >= 2 ? data[from - 2] : 0;
        foreach (b; data[from .. to])
        {
            foreach (mode; 0 .. 4)
            {
                uint k = mode * 64 + (kContextLookupTables[mode * 2][p1] | kContextLookupTables[mode * 2 + 1][p2]);
                mh[k * A + b]++;
                totals[k]++;
            }
            p2 = p1;
            p1 = b;
        }
    }

    if (cmds.length == 0)
        count(start, start + n);
    else
    {
        size_t pos = start;
        foreach (ref const c; cmds)
        {
            count(pos, pos + c.insertLength);
            pos += c.insertLength + c.outputLength;
        }
    }

    ulong best = ulong.max;
    foreach (mode; 0 .. 4)
    {
        ulong estimate = 0;
        foreach (context; 0 .. 64)
        {
            size_t k = mode * 64 + context;
            if (totals[k] != 0)
                estimate += populationCost(mh[k * A .. (k + 1) * A]);
        }
        if (estimate < best)
        {
            best = estimate;
            bestMode = mode;
        }
    }
    return true;
}

// Literal costs of data[start .. start + n] for the optimal parser, as
// prefix sums: costs[i] is the cost of data[start .. start + i].
// Without commands yet, from the statistics of all the bytes, ignoring
// contexts: with contexts, they would be much too optimistic, since copied
// bytes are counted. Else from the literals of `cmds`, the average of the
// costs with and without contexts (of the cheapest mode).
bool computeLiteralCosts(scope const(ubyte)[] data, size_t start, size_t n,
                         scope const(Command)[] cmds, ref EncoderWorkspace ws, scope long[] costs)
{
    enum A = NUM_LITERAL_CODES;
    uint bestMode;
    if (!literalModeHistograms(data, start, n, cmds, ws, bestMode))
        return false;
    uint[] mh = ws.modeHistograms[0 .. 4 * 64 * A];

    // Costs of each context. Contexts without statistics use those of all contexts.
    if (!ws.literalContextCosts.ensure(64 * A))
        return false;
    uint[] contextCosts = ws.literalContextCosts;
    uint[A] all;
    foreach (context; 0 .. 64)
        all[] += mh[(bestMode * 64 + context) * A .. (bestMode * 64 + context + 1) * A];
    uint[A] allCosts;
    setCosts(all[], allCosts[], true);
    foreach (context; 0 .. 64)
    {
        const(uint)[] h = mh[(bestMode * 64 + context) * A .. (bestMode * 64 + context + 1) * A];
        uint[] c = contextCosts[context * A .. (context + 1) * A];
        if (isEmpty(h) || cmds.length == 0)
            c[] = allCosts[];
        else
        {
            setCosts(h, c, true);
            foreach (k; 0 .. A)
                c[k] = (LITERAL_CONTEXT_WEIGHT * c[k] + LITERAL_ORDER0_WEIGHT * allCosts[k])
                     / (LITERAL_CONTEXT_WEIGHT + LITERAL_ORDER0_WEIGHT);
        }
    }

    costs[0] = 0;
    ubyte p1 = start >= 1 ? data[start - 1] : 0;
    ubyte p2 = start >= 2 ? data[start - 2] : 0;
    foreach (i, b; data[start .. start + n])
    {
        uint context = kContextLookupTables[bestMode * 2][p1] | kContextLookupTables[bestMode * 2 + 1][p2];
        costs[i + 1] = costs[i] + contextCosts[context * A + b];
        p2 = p1;
        p1 = b;
    }
    return true;
}

// Command and distance costs from the commands of a previous parse, or a
// rough guess if there are none.
void computeCommandCosts(scope const(Command)[] cmds, ref ZopfliCostModel model)
{
    if (cmds.length == 0)
    {
        foreach (i, ref c; model.commandCost)
            c = log2Fixed(INITIAL_COMMAND_COST_OFFSET + i);
        foreach (i, ref c; model.distanceCost)
            c = log2Fixed(INITIAL_DISTANCE_COST_OFFSET + i);
    }
    else
    {
        uint[NUM_COMMAND_CODES] commandHistogram;
        uint[16 + 48] distanceHistogram;
        foreach (ref const c; cmds)
        {
            uint insertCode = insertLengthCode(c.insertLength);
            uint copyCode = c.copyLength == 0 ? 0 : copyLengthCode(c.copyLength);
            uint cmd = commandSymbol(insertCode, copyCode, c.copyLength == 0 || c.distanceCode == 0);
            commandHistogram[cmd]++;
            if (hasExplicitDistance(c))
            {
                uint dsym, dbits, dextra;
                distanceSymbol(c.distanceCode, 0, 0, dsym, dbits, dextra);
                distanceHistogram[dsym]++;
            }
        }
        setCosts(commandHistogram[], model.commandCost[], false);
        setCosts(distanceHistogram[], model.distanceCost[], false);
    }
    model.minCommandCost = uint.max;
    foreach (c; model.commandCost)
        if (c < model.minCommandCost)
            model.minCommandCost = c;
}

// Matches of every position of data[start .. end], for optimal parsing, into
// ws.matches and ws.numMatches. After a long match, the next positions only
// go in the tree.
bool findOptimalMatches(ref TreeHasher t, scope const(ubyte)[] data, size_t start, size_t end,
                        size_t maxBackward, ref const(LevelParams) params, ref EncoderWorkspace ws)
{
    size_t n = end - start;
    if (!ws.numMatches.ensure(n))
        return false;
    uint[] numMatches = ws.numMatches[0 .. n];
    ws.matches.length = 0;
    size_t i = 0;
    while (i < n)
    {
        size_t pos = start + i;
        size_t maxDistance = pos < maxBackward ? pos : maxBackward;
        numMatches[i] = 0;
        if (pos + 4 > data.length)
        {
            i++;
            continue;
        }
        if (n - i < 4)
        {
            size_t unused;
            t.storeAndFindMatches!false(data, pos, n - i, maxDistance, unused, ws.matches);
            i++;
            continue;
        }
        size_t before = ws.matches.length;
        findAllMatches(t, data, pos, n - i, maxDistance, params, ws.matches);
        if (ws.matches.failed)
            return false;
        numMatches[i] = cast(uint)(ws.matches.length - before);
        i++;
        if (numMatches[i - 1] > 0)
        {
            BackwardMatch longest = ws.matches.storage[ws.matches.length - 1];
            if (longest.length > params.optimalMaxLength)
            {
                ws.matches.storage[before] = longest;
                ws.matches.length = before + 1;
                numMatches[i - 1] = 1;
                size_t skipEnd = pos + longest.length;
                for (; start + i < skipEnd && i < n; ++i)
                {
                    numMatches[i] = 0;
                    size_t p = start + i;
                    if (p + 4 <= data.length)
                    {
                        size_t unused;
                        t.storeAndFindMatches!false(data, p, n - i, p < maxBackward ? p : maxBackward,
                                                    unused, ws.matches);
                    }
                }
            }
        }
    }
    return true;
}

// One optimal parse of data[start .. end], with the matches of
// findOptimalMatches, and costs from the commands of a previous parse
// (or estimated, if there are none). Appends commands to `cmds`, the last
// one is insert-only if the block doesn't end with a copy. Updates `last`.
bool optimalParse(scope const(ubyte)[] data, size_t start, size_t end, size_t maxBackward,
                  ref const(LevelParams) params, scope const(Command)[] previous,
                  ref DistanceCache last, ref CommandBuffer cmds, ref EncoderWorkspace ws)
{
    size_t n = end - start;
    if (!ws.literalCosts.ensure(n + 1) || !ws.nodes.ensure(n + 1))
        return false;
    long[] literalCosts = ws.literalCosts[0 .. n + 1];
    ZopfliNode[] nodes = ws.nodes[0 .. n + 1];
    ZopfliCostModel model;
    computeCommandCosts(previous, model);
    if (!computeLiteralCosts(data, start, n, previous, ws, literalCosts))
        return false;
    zopfliShortestPath(data, start, end, maxBackward, last, ws.numMatches[0 .. n],
                       ws.matches[], model, literalCosts, params, nodes);
    return zopfliCreateCommands(nodes, start, end, maxBackward, literalCosts, model, last, cmds);
}

// ---------------------------------------------------------------------------
// Encoder: block splitting
// ---------------------------------------------------------------------------

uint blockLengthCode(size_t len)
{
    assert(len >= 1 && len < kBlockLengthOffset[25] + (1u << kBlockLengthBits[25]));
    uint code = 0;
    while (code < 25 && len >= kBlockLengthOffset[code + 1])
        code++;
    return code;
}

// Blocks of one category of symbols: block i has type types[i], and
// lengths[i] symbols. Types are numbered by first use, consecutive blocks
// have different types.
struct BlockSplit
{
pure nothrow @nogc @safe:
    MallocArray!ubyte types;
    MallocArray!uint lengths;
    size_t numBlocks;
    uint numTypes = 1;

    // A single block of n symbols, n may be 0.
    bool setSingle(size_t n)
    {
        numTypes = 1;
        numBlocks = 0;
        if (n == 0)
            return true;
        if (!types.ensure(1) || !lengths.ensure(1))
            return false;
        types[0] = 0;
        lengths[0] = cast(uint) n;
        numBlocks = 1;
        return true;
    }

    void release()
    {
        types.release();
        lengths.release();
    }
}

// Walks the blocks of a split, symbol by symbol.
struct BlockCursor
{
pure nothrow @nogc @safe:
    const(ubyte)[] types;
    const(uint)[] lengths;
    size_t index;
    uint remaining;
    uint type;

    this(ref const(BlockSplit) split)
    {
        types = split.types[0 .. split.numBlocks];
        lengths = split.lengths[0 .. split.numBlocks];
        remaining = split.numBlocks > 0 ? lengths[0] : 0;
    }

    // Type of the next symbol. Returns true if a new block starts.
    pragma(inline, true)
    bool next()
    {
        bool switched = false;
        if (remaining == 0)
        {
            index++;
            remaining = lengths[index];
            type = types[index];
            switched = true;
        }
        remaining--;
        return switched;
    }
}

// Parameters of the block splitter, for a category of symbols.
struct SplitParams
{
    uint maxHistograms;         // at most 256
    uint symbolsPerHistogram;
    uint stride;                // length of samples of the initial histograms
    uint minLength;             // shorter is not split
    uint switchCost;            // in 1/65536 bit
    uint iterations;
    bool clusterBlocks;         // cluster each block on its own, not the histograms (slower)
}

@tuning enum ABSENT_SYMBOL_EXTRA_BITS = 2; // cost of a symbol absent from a histogram, above log2(total)
// Switches are cheaper at the beginning, where histograms are less reliable:
// from START% of the switch cost, + SLOPE% each WARMUP symbols, until WARMUP.
@tuning enum SPLIT_WARMUP = 2000;
@tuning enum SPLIT_WARMUP_START = 77;
@tuning enum SPLIT_WARMUP_SLOPE = 7;
@tuning enum SPLIT_REFINE_FACTOR = 2;  // random samples: this * n / stride + SPLIT_REFINE_MIN
@tuning enum SPLIT_REFINE_MIN = 100;

@tuning static immutable SplitParams[2][3] kSplitParams =
[
    // Literals, commands, distances: [fast, best]. Switch costs: 28.1, 13.5, 14.6 bits.
    [SplitParams(16, 544, 70, 128, 1841562, 2, false), SplitParams(100, 544, 70, 128, 1841562, 10, true)],
    [SplitParams(16, 530, 40, 128, 884736, 2, false), SplitParams(50, 530, 40, 128, 884736, 10, true)],
    [SplitParams(16, 544, 40, 128, 956826, 2, false), SplitParams(50, 544, 40, 128, 956826, 10, true)],
];

// Deterministic pseudo-random numbers for sampling.
pragma(inline, true)
uint nextRandom(ref uint seed)
{
    seed *= 16807;
    return seed;
}

// Assign each symbol to the histogram that codes it best, with a cost for
// switching histograms (Viterbi). Writes the histogram of each symbol to
// blockIds.
void findBlocks(scope const(ushort)[] symbols, size_t alphabetSize, scope const(uint)[] histograms,
                size_t numHistograms, uint switchCost, scope ubyte[] blockIds,
                scope ubyte[] switchSignal, scope uint[] insertCost, scope uint[] cost)
{
    size_t n = symbols.length;
    if (numHistograms == 1)
    {
        blockIds[0 .. n] = 0;
        return;
    }
    // Cost of each symbol with each histogram: absent symbols cost a bit more
    // than the rarest.
    foreach (k; 0 .. numHistograms)
    {
        const(uint)[] h = histograms[k * alphabetSize .. (k + 1) * alphabetSize];
        ulong total = 0;
        foreach (c; h)
            total += c;
        uint log2Total = log2Fixed(cast(size_t) total);
        foreach (s, c; h)
            insertCost[s * numHistograms + k] = c == 0 ? log2Total + ABSENT_SYMBOL_EXTRA_BITS * COST_ONE_BIT
                                                        : log2Total - log2Fixed(c);
    }

    // cost[k]: cost of the best path ending with histogram k, relative to
    // the best path. Capped at switchCost: then the path switches from the
    // best one, and switchSignal records it.
    // (The loops are branch-free, for vectorization.)
    size_t bitmapLength = (numHistograms + 7) >> 3;
    uint[] costs = cost[0 .. numHistograms];
    costs[] = 0;
    foreach (i, sym; symbols)
    {
        const(uint)[] ic = insertCost[sym * numHistograms .. (sym + 1) * numHistograms];
        uint minCost = uint.max;
        foreach (k, ref c; costs)
        {
            c += ic[k];
            minCost = c < minCost ? c : minCost;
        }
        uint best = 0;
        while (costs[best] != minCost)
            best++;
        blockIds[i] = cast(ubyte) best;
        uint sc = switchCost;
        if (i < SPLIT_WARMUP)
            sc = cast(uint)(cast(ulong) switchCost * (SPLIT_WARMUP_START * SPLIT_WARMUP + SPLIT_WARMUP_SLOPE * i)
                            / (100 * SPLIT_WARMUP));
        ubyte[] signal = switchSignal[i * bitmapLength .. (i + 1) * bitmapLength];
        for (size_t k8 = 0; k8 < numHistograms; k8 += 8)
        {
            uint[] group = costs[k8 .. (k8 + 8 < numHistograms ? k8 + 8 : numHistograms)];
            uint bits = 0;
            foreach (j, ref c; group)
            {
                uint d = c - minCost;
                uint capped = d >= sc;
                c = capped ? sc : d;
                bits |= capped << j;
            }
            signal[k8 >> 3] = cast(ubyte) bits;
        }
    }

    // Trace back the best path.
    size_t i = n - 1;
    ubyte current = blockIds[i];
    while (i > 0)
    {
        --i;
        if (switchSignal[i * bitmapLength + (current >> 3)] & (1 << (current & 7)))
            current = blockIds[i];
        blockIds[i] = current;
    }
}

// Split `symbols` (each < alphabetSize) into blocks of similar statistics.
BrotliStatus splitSymbols(scope const(ushort)[] symbols, size_t alphabetSize,
                          ref const(SplitParams) p, ref BlockSplit split, ref EncoderWorkspace ws)
{
    enum oom = BrotliStatus.outOfMemory;
    size_t n = symbols.length;
    size_t numHistograms = n / p.symbolsPerHistogram + 1;
    if (numHistograms > p.maxHistograms)
        numHistograms = p.maxHistograms;
    if (n < p.minLength || numHistograms <= 1)
        return split.setSingle(n) ? BrotliStatus.success : oom;

    size_t bitmapLength = (numHistograms + 7) >> 3;
    if (!ws.splitHistograms.ensure(numHistograms * alphabetSize)
        || !ws.splitBlockIds.ensure(n)
        || !ws.splitSignal.ensure(n * bitmapLength)
        || !ws.splitInsertCost.ensure(numHistograms * alphabetSize)
        || !ws.reserveClustering(numHistograms, alphabetSize))
        return oom;
    uint[] histograms = ws.splitHistograms;
    ubyte[] blockIds = ws.splitBlockIds[0 .. n];
    uint[256] cost;

    // Initial histograms: a sample of each part of the data, then more
    // random samples.
    histograms[0 .. numHistograms * alphabetSize] = 0;
    uint seed = 7;
    foreach (k; 0 .. numHistograms)
    {
        size_t partLength = n / numHistograms;
        size_t pos = n * k / numHistograms;
        if (partLength > p.stride)
            pos += nextRandom(seed) % (partLength - p.stride);
        size_t end = pos + p.stride < n ? pos + p.stride : n;
        foreach (s; symbols[pos .. end])
            histograms[k * alphabetSize + s]++;
    }
    size_t refinements = SPLIT_REFINE_FACTOR * n / p.stride + SPLIT_REFINE_MIN;
    refinements = (refinements + numHistograms - 1) / numHistograms * numHistograms;
    foreach (r; 0 .. refinements)
    {
        size_t pos = 0;
        size_t len = n;
        if (p.stride < n)
        {
            pos = nextRandom(seed) % (n - p.stride + 1);
            len = p.stride;
        }
        size_t k = r % numHistograms;
        foreach (s; symbols[pos .. pos + len])
            histograms[k * alphabetSize + s]++;
    }

    // Assign symbols to histograms, and recompute histograms from the
    // assignment.
    foreach (iteration; 0 .. p.iterations)
    {
        findBlocks(symbols, alphabetSize, histograms, numHistograms, p.switchCost, blockIds,
                   ws.splitSignal, ws.splitInsertCost, cost[]);
        ushort[256] newId = ushort.max;
        size_t numUsed = 0;
        foreach (ref id; blockIds)
        {
            if (newId[id] == ushort.max)
                newId[id] = cast(ushort) numUsed++;
            id = cast(ubyte) newId[id];
        }
        numHistograms = numUsed;
        histograms[0 .. numHistograms * alphabetSize] = 0;
        foreach (i, s; symbols)
            histograms[blockIds[i] * alphabetSize + s]++;
        if (numHistograms == 1)
            break;
    }

    // Runs of the same histogram.
    size_t numRuns = 1;
    foreach (i; 1 .. n)
        if (blockIds[i] != blockIds[i - 1])
            numRuns++;
    MallocArray!uint runLength;
    MallocArray!uint runType;
    MallocArray!uint runHistograms;
    scope(exit)
    {
        runLength.release();
        runType.release();
        runHistograms.release();
    }
    if (!runLength.resize(numRuns) || !runType.resize(numRuns))
        return oom;
    {
        size_t r = 0;
        runLength[0] = 0;
        runType[0] = blockIds[0];
        foreach (i; 1 .. n)
        {
            runLength[r]++;
            if (blockIds[i] != blockIds[i - 1])
            {
                runLength[++r] = 0;
                runType[r] = blockIds[i];
            }
        }
        runLength[r]++;
    }

    if (!p.clusterBlocks)
    {
        // Merge similar histograms into block types.
        ubyte[256] map;
        ulong clusterCost;
        clusterHistograms(histograms, numHistograms, alphabetSize, 256, map[], ws.clusters,
                          ws.clusterScratch, ws.clusterDelta, clusterCost);
        foreach (ref t; runType[0 .. numRuns])
            t = map[t];
    }
    else
    {
        // Merge similar runs into block types, then move each run to the
        // type that codes it best (merging can leave some in a worse one).
        if (!runHistograms.resize(numRuns * alphabetSize))
            return oom;
        runHistograms[] = 0;
        {
            size_t pos = 0;
            foreach (r; 0 .. numRuns)
            {
                foreach (s; symbols[pos .. pos + runLength[r]])
                    runHistograms[r * alphabetSize + s]++;
                pos += runLength[r];
            }
        }
        uint numClusters;
        ulong clusterCost;
        BrotliStatus st = clusterMany(runHistograms[0 .. numRuns * alphabetSize], numRuns, alphabetSize, 256,
                                      runType[0 .. numRuns], ws, numClusters, clusterCost);
        if (st != BrotliStatus.success)
            return st;
        const(uint)[] clustered = ws.clusters[0 .. numClusters * alphabetSize];
        ulong[256] clusteredCost;
        foreach (k; 0 .. numClusters)
            clusteredCost[k] = populationCost(clustered[k * alphabetSize .. (k + 1) * alphabetSize]);
        // Bits added by run r to type k.
        uint[MAX_ALPHABET_SIZE] sum;
        long addedCost(size_t r, size_t k)
        {
            foreach (j; 0 .. alphabetSize)
                sum[j] = runHistograms[r * alphabetSize + j] + clustered[k * alphabetSize + j];
            return cast(long) populationCost(sum[0 .. alphabetSize]) - cast(long) clusteredCost[k];
        }
        foreach (r; 0 .. numRuns)
        {
            // On ties, the type of the previous run: no switch.
            uint best = runType[r == 0 ? 0 : r - 1];
            long bestBits = addedCost(r, best);
            foreach (k; 0 .. numClusters)
            {
                long bits = addedCost(r, k);
                if (bits < bestBits)
                {
                    bestBits = bits;
                    best = k;
                }
            }
            runType[r] = best;
        }
    }

    // Blocks, types numbered by first use, neighbor runs of the same type merged.
    ushort[256] typeNumber = ushort.max;
    uint numTypes = 0;
    size_t numBlocks = 0;
    uint previous = uint.max;
    foreach (t; runType[0 .. numRuns])
    {
        if (typeNumber[t] == ushort.max)
            typeNumber[t] = cast(ushort) numTypes++;
        if (typeNumber[t] != previous)
            numBlocks++;
        previous = typeNumber[t];
    }
    if (!split.types.ensure(numBlocks) || !split.lengths.ensure(numBlocks))
        return oom;
    size_t b = 0;
    previous = uint.max;
    foreach (r; 0 .. numRuns)
    {
        uint t = typeNumber[runType[r]];
        if (t != previous)
        {
            split.types[b] = cast(ubyte) t;
            split.lengths[b] = 0;
            b++;
            previous = t;
        }
        split.lengths[b - 1] += runLength[r];
    }
    assert(b == numBlocks && split.types[0] == 0);
    split.numBlocks = numBlocks;
    split.numTypes = numTypes;
    return BrotliStatus.success;
}

// Block types and lengths of a category, as written in the meta-block.
struct BlockEncoder
{
pure nothrow @nogc @safe:
    BlockCursor cursor;
    uint numTypes;
    ushort[] typeCodes; // of each block
    ubyte[256 + 2] typeDepth;
    ushort[256 + 2] typeBits;
    ubyte[NUM_BLOCK_LENGTH_CODES] lengthDepth;
    ushort[NUM_BLOCK_LENGTH_CODES] lengthBits;

    // Write NBLTYPES, and if there are several types, their prefix codes and
    // the first block length. `typeCodes` must have a slot per block.
    void initialize(ref BitWriter w, ref const(BlockSplit) split, ushort[] typeCodes)
    {
        cursor = BlockCursor(split);
        numTypes = split.numTypes;
        writeVarLenUint8(w, numTypes - 1);
        if (numTypes < 2)
        {
            cursor.remaining = uint.max; // never switches
            return;
        }
        this.typeCodes = typeCodes;
        uint[256 + 2] typeHistogram;
        uint[NUM_BLOCK_LENGTH_CODES] lengthHistogram;
        // Same initial state as the decoder: the last type is 0, the one before 1.
        uint last = 1;
        uint secondLast = 0;
        foreach (i; 0 .. split.numBlocks)
        {
            uint t = split.types[i];
            uint code = (t == last + 1) ? 1 : (t == secondLast ? 0 : t + 2);
            secondLast = last;
            last = t;
            typeCodes[i] = cast(ushort) code;
            if (i > 0)
                typeHistogram[code]++;
            lengthHistogram[blockLengthCode(split.lengths[i])]++;
        }
        buildAndStorePrefixCode(w, typeHistogram[0 .. numTypes + 2], typeDepth[], typeBits[]);
        buildAndStorePrefixCode(w, lengthHistogram[], lengthDepth[], lengthBits[]);
        writeLength(w, split.lengths[0]);
    }

    void writeLength(ref BitWriter w, uint len)
    {
        uint code = blockLengthCode(len);
        w.writeBits(lengthDepth[code], lengthBits[code]);
        w.writeBits(kBlockLengthBits[code], len - kBlockLengthOffset[code]);
    }

    // Type of the next symbol, writes a block switch if needed.
    pragma(inline, true)
    uint next(ref BitWriter w)
    {
        if (cursor.next())
        {
            uint code = typeCodes[cursor.index];
            w.writeBits(typeDepth[code], typeBits[code]);
            writeLength(w, cursor.lengths[cursor.index]);
        }
        return cursor.type;
    }
}

@tuning enum CLUSTER_BATCH = 64; // clusterMany clusters histograms by batches of that many

// Like clusterHistograms, for any number of histograms (`map` has a uint per
// histogram): clusters batches of them first. The cluster histograms are
// written to ws.clusters.
BrotliStatus clusterMany(scope const(uint)[] histograms, size_t n, size_t alphabetSize,
                         uint maxClusters, scope uint[] map, ref EncoderWorkspace ws,
                         out uint numClusters, out ulong totalCost)
{
    assert(n >= 1 && maxClusters <= MAX_CLUSTER_INPUTS);
    enum batch = CLUSTER_BATCH;
    if (!ws.reserveClustering(n < MAX_CLUSTER_INPUTS ? n : MAX_CLUSTER_INPUTS, alphabetSize))
        return BrotliStatus.outOfMemory;
    if (n <= MAX_CLUSTER_INPUTS)
    {
        ubyte[MAX_CLUSTER_INPUTS] m;
        numClusters = clusterHistograms(histograms, n, alphabetSize, maxClusters, m[], ws.clusters,
                                        ws.clusterScratch, ws.clusterDelta, totalCost);
        foreach (i; 0 .. n)
            map[i] = m[i];
        return BrotliStatus.success;
    }

    // Cluster each batch, keep the batch clusters. The first pass only merges
    // when it saves bits, the next ones halve the number of histograms.
    MallocArray!uint reduced;
    MallocArray!uint reducedMap;
    if (!reduced.resize(n * alphabetSize) || !reducedMap.resize(n))
        return BrotliStatus.outOfMemory;
    reduced[0 .. n * alphabetSize] = histograms[0 .. n * alphabetSize];
    foreach (i; 0 .. n)
        map[i] = cast(uint) i;
    size_t count = n;
    for (int pass = 0; count > MAX_CLUSTER_INPUTS; ++pass)
    {
        size_t newCount = 0;
        for (size_t b = 0; b < count; b += batch)
        {
            size_t bn = count - b < batch ? count - b : batch;
            ubyte[batch] m;
            ulong cost;
            uint k = clusterHistograms(reduced[b * alphabetSize .. $], bn, alphabetSize,
                                       pass == 0 ? batch : batch / 2, m[], ws.clusters,
                                       ws.clusterScratch, ws.clusterDelta, cost);
            // Batches only move histograms down: reduced[newCount ..] does not
            // overlap the batches not yet clustered.
            reduced[newCount * alphabetSize .. (newCount + k) * alphabetSize] = ws.clusters[0 .. k * alphabetSize];
            foreach (j; 0 .. bn)
                reducedMap[b + j] = cast(uint)(newCount + m[j]);
            newCount += k;
        }
        foreach (i; 0 .. n)
            map[i] = reducedMap[map[i]];
        count = newCount;
    }
    ubyte[MAX_CLUSTER_INPUTS] m;
    numClusters = clusterHistograms(reduced, count, alphabetSize, maxClusters, m[], ws.clusters,
                                    ws.clusterScratch, ws.clusterDelta, totalCost);
    foreach (i; 0 .. n)
        map[i] = m[map[i]];
    return BrotliStatus.success;
}

// ---------------------------------------------------------------------------
// Encoder: meta-blocks
// ---------------------------------------------------------------------------

enum NUM_DISTANCE_SYMBOLS_MAX = 16 + 120 + (48 << 3);

enum : uint
{
    CONTEXT_LSB6   = 0,
    CONTEXT_MSB6   = 1,
    CONTEXT_UTF8   = 2,
    CONTEXT_SIGNED = 3,
}

// Number of nibbles of MLEN - 1, as small as possible (the decoder requires it).
uint metaBlockNibbles(size_t length)
{
    assert(length >= 1 && length <= MAX_METABLOCK_LENGTH);
    size_t v = length - 1;
    if (v < (1 << 16))
        return 4;
    if (v < (1 << 20))
        return 5;
    return 6;
}

void writeMetaBlockLength(ref BitWriter w, size_t length)
{
    uint nibbles = metaBlockNibbles(length);
    w.writeBits(2, nibbles - 4);
    w.writeBits(nibbles * 4, length - 1);
}

void writeUncompressedMetaBlock(ref BitWriter w, scope const(ubyte)[] block)
{
    w.writeBits(1, 0); // ISLAST
    writeMetaBlockLength(w, block.length);
    w.writeBits(1, 1); // ISUNCOMPRESSED
    w.alignToByte();
    w.writeBytes(block);
}

@tuning enum size_t RLE_STREAK_LIMIT = 1240; // optimizeHistogramForRle, in 1/256

// Make the code lengths of a histogram cheaper to store, by smoothing
// counts into runs of equal values (the reference encoder's heuristic).
// Used symbols stay used.
void optimizeHistogramForRle(scope uint[] counts)
{
    size_t length = counts.length;
    size_t nonzeros = 0;
    uint smallestNonzero = uint.max;
    foreach (c; counts)
    {
        if (c != 0)
        {
            nonzeros++;
            if (c < smallestNonzero)
                smallestNonzero = c;
        }
    }
    if (nonzeros < 16)
        return;
    while (length > 0 && counts[length - 1] == 0)
        length--;
    // Fill isolated holes of a nearly full histogram.
    if (smallestNonzero < 4 && length - nonzeros < 6)
        foreach (i; 1 .. length - 1)
            if (counts[i - 1] != 0 && counts[i] == 0 && counts[i + 1] != 0)
                counts[i] = 1;
    if (nonzeros < 28)
        return;

    // Runs that are already good for run-length coding: zeros of length >= 5,
    // others of length >= 7.
    bool[MAX_ALPHABET_SIZE] goodForRle;
    {
        uint symbol = counts[0];
        size_t step = 0;
        foreach (i; 0 .. length + 1)
        {
            if (i == length || counts[i] != symbol)
            {
                if ((symbol == 0 && step >= 5) || (symbol != 0 && step >= 7))
                    goodForRle[i - step .. i] = true;
                step = 1;
                if (i != length)
                    symbol = counts[i];
            }
            else
                step++;
        }
    }

    // Replace runs of similar counts by their average (24.8 fixed point).
    enum size_t streakLimit = RLE_STREAK_LIMIT;
    size_t stride = 0;
    size_t limit = 256 * (cast(size_t) counts[0] + counts[1] + counts[2]) / 3 + 420;
    size_t sum = 0;
    foreach (i; 0 .. length + 1)
    {
        if (i == length || goodForRle[i] || (i != 0 && goodForRle[i - 1])
            || (256 * cast(size_t) counts[i] - limit + streakLimit) >= 2 * streakLimit)
        {
            if (stride >= 4 || (stride >= 3 && sum == 0))
            {
                // End of a stride: replace it by its average, never by zero
                // if it isn't all zeros.
                size_t count = (sum + stride / 2) / stride;
                if (count == 0)
                    count = 1;
                if (sum == 0)
                    count = 0;
                counts[i - stride .. i] = cast(uint) count;
            }
            stride = 0;
            sum = 0;
            if (i + 2 < length)
                limit = 256 * (cast(size_t) counts[i] + counts[i + 1] + counts[i + 2]) / 3 + 420;
            else if (i < length)
                limit = 256 * cast(size_t) counts[i];
            else
                limit = 0;
        }
        stride++;
        if (i != length)
        {
            sum += counts[i];
            if (stride >= 4)
                limit = (256 * sum + stride / 2) / stride;
            if (stride == 4)
                limit += 120;
        }
    }
}

// Estimated size of the distances of `cmds`, with these distance parameters.
ulong distanceCost(scope const(Command)[] cmds, uint npostfix, uint ndirect)
{
    uint[NUM_DISTANCE_SYMBOLS_MAX] histogram;
    ulong extraBits = 0;
    foreach (ref const c; cmds)
    {
        if (!hasExplicitDistance(c))
            continue;
        uint dsym, dbits, dextra;
        distanceSymbol(c.distanceCode, npostfix, ndirect, dsym, dbits, dextra);
        histogram[dsym]++;
        extraBits += dbits;
    }
    return populationCost(histogram[0 .. 16 + ndirect + (48 << npostfix)]) + extraBits * COST_ONE_BIT;
}

// Does the command have a distance symbol?
pragma(inline, true)
bool hasExplicitDistance(ref const(Command) c)
{
    if (c.copyLength == 0)
        return false;
    return c.distanceCode != 0 || insertLengthCode(c.insertLength) >= 8 || copyLengthCode(c.copyLength) >= 16;
}

// Choose NPOSTFIX and NDIRECT.
void chooseDistanceParams(scope const(Command)[] cmds, uint search,
                          out uint npostfix, out uint ndirect)
{
    npostfix = 0;
    ndirect = 0;
    if (search == 0)
        return;
    ulong best = distanceCost(cmds, 0, 0);
    foreach (p; 0 .. 4)
    {
        ulong previous = ulong.max;
        foreach (msb; 0 .. (search >= 2 ? 16 : 1))
        {
            if (p == 0 && msb == 0)
                continue;
            ulong cost = distanceCost(cmds, p, msb << p);
            if (cost < best)
            {
                best = cost;
                npostfix = p;
                ndirect = msb << p;
            }
            if (cost > previous)
                break; // usually gets worse from there
            previous = cost;
        }
    }
}

// Memory reused across meta-blocks.
struct EncoderWorkspace
{
pure nothrow @nogc @safe:
    CommandBuffer commands;

    // Symbols of a meta-block.
    MallocArray!ushort literals;
    MallocArray!ushort commandSymbols;
    MallocArray!ushort distanceSymbols;

    // Block splits.
    BlockSplit[3] splits;
    MallocArray!ushort typeCodes;
    MallocArray!uint splitHistograms;
    MallocArray!ubyte splitBlockIds;
    MallocArray!ubyte splitSignal;
    MallocArray!uint splitInsertCost;

    // Histograms and prefix codes.
    MallocArray!uint modeHistograms;      // [mode][context][symbol], to choose the literal context mode
    MallocArray!uint histograms;
    MallocArray!uint treeHistograms;
    MallocArray!uint clusterMap;
    MallocArray!uint contextHistograms;
    MallocArray!uint typeHistograms;
    MallocArray!ubyte depths;
    MallocArray!ushort bits;

    // Clustering.
    MallocArray!uint clusters;
    MallocArray!ushort clusterScratch;
    MallocArray!long clusterDelta;
    size_t clusteringCapacity; // histograms of MAX_ALPHABET_SIZE

    MallocArray!uint scratch;

    // Optimal parsing.
    CommandBuffer previousCommands;
    CommandBuffer bestCommands;
    BitWriter trial;
    MallocArray!ubyte bestMetaBlock; // when keeping the best pass, its meta-block
    ulong bestMetaBlockBits;
    MatchBuffer matches;
    MallocArray!uint numMatches;
    MallocArray!long literalCosts;
    MallocArray!uint literalContextCosts;
    MallocArray!ZopfliNode nodes;

    bool initialize()
    {
        return scratch.resize(2 * MAX_CONTEXT_MAP_LENGTH);
    }

    // Room to cluster n histograms of alphabetSize symbols.
    bool reserveClustering(size_t n, size_t alphabetSize)
    {
        return clusters.ensure(n * alphabetSize)
            && clusterScratch.ensure(n * alphabetSize)
            && clusterDelta.ensure(n * n);
    }

    void release()
    {
        commands.release();
        literals.release();
        commandSymbols.release();
        distanceSymbols.release();
        foreach (ref s; splits)
            s.release();
        typeCodes.release();
        splitHistograms.release();
        splitBlockIds.release();
        splitSignal.release();
        splitInsertCost.release();
        modeHistograms.release();
        histograms.release();
        treeHistograms.release();
        clusterMap.release();
        depths.release();
        bits.release();
        clusters.release();
        clusterScratch.release();
        clusterDelta.release();
        scratch.release();
        previousCommands.release();
        bestCommands.release();
        matches.release();
        numMatches.release();
        literalCosts.release();
        literalContextCosts.release();
        nodes.release();
    }
}

enum : uint
{
    LITERALS  = 0,
    COMMANDS  = 1,
    DISTANCES = 2,
}

// Build the prefix codes of `numTrees` histograms (after optimizing them for
// RLE if asked), write them, and keep their code lengths and codes in
// depths/bits.
void storeTrees(ref BitWriter w, scope uint[] histograms, size_t numTrees, size_t alphabetSize,
                bool optimize, scope ubyte[] depths, scope ushort[] bits)
{
    foreach (t; 0 .. numTrees)
    {
        size_t o = t * alphabetSize;
        uint[] h = histograms[o .. o + alphabetSize];
        if (optimize)
            optimizeHistogramForRle(h);
        buildAndStorePrefixCode(w, h, depths[o .. o + alphabetSize], bits[o .. o + alphabetSize]);
    }
}

// Literal trees from the histograms of each (block type, context), cheaper
// than clustering all of them: the 64 contexts are clustered for all the
// block types together, then for each cluster of contexts, the block types
// are clustered. Writes the tree of each (block type, context) to `map`,
// the trees to `trees`.
BrotliStatus clusterLiteralsByContext(scope const(uint)[] histograms, uint numTypes, scope ubyte[] map,
                                      scope uint[] trees, ref EncoderWorkspace ws, out uint numTrees,
                                      out ulong totalCost)
{
    enum A = NUM_LITERAL_CODES;
    if (!ws.reserveClustering(64 > numTypes ? 64 : numTypes, A)
        || !ws.contextHistograms.ensure(64 * A) || !ws.typeHistograms.ensure(numTypes * A))
        return BrotliStatus.outOfMemory;

    // Clusters of contexts.
    uint[] contexts = ws.contextHistograms[0 .. 64 * A];
    contexts[] = 0;
    foreach (t; 0 .. numTypes)
        foreach (k; 0 .. 64 * A)
            contexts[k] += histograms[t * 64 * A + k];
    ubyte[64] contextCluster;
    ulong cost;
    uint numContextClusters = clusterHistograms(contexts, 64, A, 64, contextCluster[], ws.clusters,
                                                ws.clusterScratch, ws.clusterDelta, cost);

    // For each cluster of contexts, the histograms of the block types, clustered.
    uint maxTrees = 256 / numContextClusters;
    uint[] byType = ws.typeHistograms[0 .. numTypes * A];
    numTrees = 0;
    totalCost = 0;
    foreach (j; 0 .. numContextClusters)
    {
        byType[] = 0;
        foreach (t; 0 .. numTypes)
            foreach (context; 0 .. 64)
                if (contextCluster[context] == j)
                    byType[t * A .. (t + 1) * A] += histograms[(t * 64 + context) * A .. (t * 64 + context + 1) * A];
        ubyte[256] typeTree;
        uint n = clusterHistograms(byType, numTypes, A, maxTrees, typeTree[],
                                   ws.clusters, ws.clusterScratch, ws.clusterDelta, cost);
        trees[numTrees * A .. (numTrees + n) * A] = ws.clusters[0 .. n * A];
        foreach (t; 0 .. numTypes)
            foreach (context; 0 .. 64)
                if (contextCluster[context] == j)
                    map[t * 64 + context] = cast(ubyte)(numTrees + typeTree[t]);
        numTrees += n;
        totalCost += cost;
    }
    return BrotliStatus.success;
}

// Above this (used histograms x their non-zero symbols), literals are
// clustered with clusterLiteralsByContext, if the level allows.
@tuning enum size_t FULL_CLUSTERING_MAX_WORK = 1 << 22;

// Write data[start .. start + length] as a compressed meta-block made of `cmds`.
BrotliStatus writeCompressedMetaBlock(ref BitWriter w, scope const(ubyte)[] data, size_t start,
                                      size_t length, scope const(Command)[] cmds, bool isLast,
                                      ref const(LevelParams) params, ref EncoderWorkspace ws)
{
    enum oom = BrotliStatus.outOfMemory;
    enum A = NUM_LITERAL_CODES;

    // Distance parameters.
    uint npostfix, ndirect;
    chooseDistanceParams(cmds, params.distanceParamSearch, npostfix, ndirect);
    uint numDistanceSymbols = 16 + ndirect + (48u << npostfix);

    // The symbols of each category, in order.
    size_t numLiterals = 0;
    size_t numDistances = 0;
    foreach (ref const c; cmds)
    {
        numLiterals += c.insertLength;
        if (hasExplicitDistance(c))
            numDistances++;
    }
    if (!ws.literals.ensure(numLiterals) || !ws.commandSymbols.ensure(cmds.length)
        || !ws.distanceSymbols.ensure(numDistances))
        return oom;
    {
        size_t pos = start, nl = 0, nd = 0;
        foreach (k, ref const c; cmds)
        {
            foreach (b; data[pos .. pos + c.insertLength])
                ws.literals[nl++] = b;
            pos += c.insertLength + c.outputLength;
            uint insertCode = insertLengthCode(c.insertLength);
            uint copyCode = c.copyLength == 0 ? 0 : copyLengthCode(c.copyLength);
            ws.commandSymbols[k] = cast(ushort) commandSymbol(insertCode, copyCode,
                                                               c.copyLength == 0 || c.distanceCode == 0);
            if (hasExplicitDistance(c))
            {
                uint dsym, dbits, dextra;
                distanceSymbol(c.distanceCode, npostfix, ndirect, dsym, dbits, dextra);
                ws.distanceSymbols[nd++] = cast(ushort) dsym;
            }
            assert((ws.commandSymbols[k] >= 128) == hasExplicitDistance(c) || c.copyLength == 0);
        }
        assert(pos == start + length && nl == numLiterals && nd == numDistances);
    }
    const(ushort)[] literals = ws.literals[0 .. numLiterals];
    const(ushort)[] commandSymbols = ws.commandSymbols[0 .. cmds.length];
    const(ushort)[] distanceSymbols = ws.distanceSymbols[0 .. numDistances];

    // Block splits.
    BlockSplit[] splits = ws.splits[];
    if (params.blockSplitting > 0)
    {
        BrotliStatus r = splitSymbols(literals, A, kSplitParams[LITERALS][params.blockSplitting - 1],
                                      splits[LITERALS], ws);
        if (r == BrotliStatus.success)
            r = splitSymbols(commandSymbols, NUM_COMMAND_CODES,
                             kSplitParams[COMMANDS][params.blockSplitting - 1], splits[COMMANDS], ws);
        if (r == BrotliStatus.success)
            r = splitSymbols(distanceSymbols, numDistanceSymbols,
                             kSplitParams[DISTANCES][params.blockSplitting - 1], splits[DISTANCES], ws);
        if (r != BrotliStatus.success)
            return r;
    }
    else
    {
        if (!splits[LITERALS].setSingle(numLiterals) || !splits[COMMANDS].setSingle(cmds.length)
            || !splits[DISTANCES].setSingle(numDistances))
            return oom;
    }
    uint numLiteralTypes = splits[LITERALS].numTypes;
    uint numCommandTypes = splits[COMMANDS].numTypes;
    uint numDistanceTypes = splits[DISTANCES].numTypes;

    // Literal context mode: the same for all literal block types.
    uint contextMode = CONTEXT_LSB6;
    // Clustering is slow: unless the level is high, only the mode of a cheap
    // estimate is clustered.
    if (params.contextModeling && !params.tryAllContextModes)
    {
        if (!literalModeHistograms(data, start, length, cmds, ws, contextMode))
            return oom;
    }

    // Literal histograms of each (block type, context), clustered into trees.
    // ws.histograms: literal histograms, then distance trees.
    // ws.treeHistograms: literal trees, followed by distance histograms, then
    // command trees.
    size_t numLiteralHistograms = numLiteralTypes * 64;
    size_t numDistanceHistograms = numDistanceTypes * 4;
    size_t literalSize = numLiteralHistograms * A;
    size_t distanceSize = numDistanceHistograms * numDistanceSymbols;
    size_t commandSize = numCommandTypes * NUM_COMMAND_CODES;
    if (!ws.histograms.ensure(literalSize > distanceSize ? literalSize : distanceSize)
        || !ws.treeHistograms.ensure(literalSize + (distanceSize > commandSize ? distanceSize : commandSize)
                                     + literalSize)
        || !ws.clusterMap.ensure(numLiteralHistograms > numDistanceHistograms
                         ? numLiteralHistograms : numDistanceHistograms))
        return oom;

    ubyte[256 * 64] literalMap;
    uint numLiteralTrees;
    uint[] literalTrees = ws.treeHistograms;
    {
        uint bestTrees;
        ulong bestCost = ulong.max;
        foreach (mode; 0 .. 4)
        {
            bool tryMode = params.tryAllContextModes ? params.contextModeling || mode == 0 : mode == contextMode;
            if (!tryMode)
                continue;
            uint[] h = ws.histograms[0 .. numLiteralHistograms * A];
            h[] = 0;
            BlockCursor cursor = BlockCursor(splits[LITERALS]);
            size_t pos = start;
            foreach (ref const c; cmds)
            {
                ubyte p1 = pos >= 1 ? data[pos - 1] : 0;
                ubyte p2 = pos >= 2 ? data[pos - 2] : 0;
                foreach (b; data[pos .. pos + c.insertLength])
                {
                    cursor.next();
                    uint context = 0;
                    if (params.contextModeling)
                        context = kContextLookupTables[mode * 2][p1] | kContextLookupTables[mode * 2 + 1][p2];
                    h[(cursor.type * 64 + context) * A + b]++;
                    p2 = p1;
                    p1 = b;
                }
                pos += c.insertLength + c.outputLength;
            }
            uint numTrees;
            ulong cost;
            // Clustering all the histograms costs about (number of histograms) x
            // (their non-zero symbols): too slow for many dense histograms.
            bool byContext = false;
            if (params.clusterLiteralsByContext)
            {
                size_t used = 0;
                size_t nonZero = 0;
                foreach (k; 0 .. numLiteralHistograms)
                {
                    size_t nz = 0;
                    foreach (c; h[k * A .. (k + 1) * A])
                        nz += c != 0;
                    nonZero += nz;
                    used += nz != 0;
                }
                byContext = numLiteralHistograms > MAX_CLUSTER_INPUTS || used * nonZero > FULL_CLUSTERING_MAX_WORK;
            }
            if (byContext)
            {
                ubyte[256 * 64] map;
                uint[] trees = ws.treeHistograms[literalSize .. 2 * literalSize];
                BrotliStatus r = clusterLiteralsByContext(h, numLiteralTypes, map[], trees, ws, numTrees, cost);
                if (r != BrotliStatus.success)
                    return r;
                if (cost < bestCost)
                {
                    bestCost = cost;
                    contextMode = mode;
                    numLiteralTrees = numTrees;
                    literalMap[0 .. numLiteralHistograms] = map[0 .. numLiteralHistograms];
                    literalTrees[0 .. numTrees * A] = trees[0 .. numTrees * A];
                }
                continue;
            }
            BrotliStatus r = clusterMany(h, numLiteralHistograms, A, 256, ws.clusterMap, ws,
                                         numTrees, cost);
            if (r != BrotliStatus.success)
                return r;
            if (cost < bestCost)
            {
                bestCost = cost;
                contextMode = mode;
                numLiteralTrees = numTrees;
                foreach (i; 0 .. numLiteralHistograms)
                    literalMap[i] = cast(ubyte) ws.clusterMap[i];
                literalTrees[0 .. numTrees * A] = ws.clusters[0 .. numTrees * A];
            }
        }
    }

    // Distance histograms of each (block type, context), clustered into trees.
    ubyte[256 * 4] distanceMap;
    uint numDistanceTrees;
    uint[] distanceTrees = ws.histograms; // the literal histograms are not needed anymore
    {
        uint[] h = ws.treeHistograms[numLiteralTrees * A .. numLiteralTrees * A + numDistanceHistograms * numDistanceSymbols];
        h[] = 0;
        BlockCursor cursor = BlockCursor(splits[DISTANCES]);
        size_t nd = 0;
        foreach (ref const c; cmds)
        {
            if (!hasExplicitDistance(c))
                continue;
            cursor.next();
            uint context = c.copyLength > 4 ? 3 : c.copyLength - 2;
            h[(cursor.type * 4 + context) * numDistanceSymbols + distanceSymbols[nd++]]++;
        }
        ulong cost;
        BrotliStatus r = clusterMany(h, numDistanceHistograms, numDistanceSymbols, 256, ws.clusterMap, ws,
                                     numDistanceTrees, cost);
        if (r != BrotliStatus.success)
            return r;
        foreach (i; 0 .. numDistanceHistograms)
            distanceMap[i] = cast(ubyte) ws.clusterMap[i];
        distanceTrees[0 .. numDistanceTrees * numDistanceSymbols] = ws.clusters[0 .. numDistanceTrees * numDistanceSymbols];
    }

    // Command histograms of each block type.
    uint[] commandTrees = ws.treeHistograms[numLiteralTrees * A .. numLiteralTrees * A + numCommandTypes * NUM_COMMAND_CODES];
    {
        commandTrees[] = 0;
        BlockCursor cursor = BlockCursor(splits[COMMANDS]);
        foreach (s; commandSymbols)
        {
            cursor.next();
            commandTrees[cursor.type * NUM_COMMAND_CODES + s]++;
        }
    }

    // Header.
    w.writeBits(1, isLast ? 1 : 0);
    if (isLast)
        w.writeBits(1, 0); // ISLASTEMPTY
    writeMetaBlockLength(w, length);
    if (!isLast)
        w.writeBits(1, 0); // ISUNCOMPRESSED

    size_t numBlocks = splits[LITERALS].numBlocks + splits[COMMANDS].numBlocks + splits[DISTANCES].numBlocks;
    if (!ws.typeCodes.ensure(numBlocks))
        return oom;
    BlockEncoder[3] blocks;
    {
        size_t o = 0;
        foreach (k; 0 .. 3)
        {
            size_t nb = splits[k].numBlocks;
            blocks[k].initialize(w, splits[k], ws.typeCodes[o .. o + nb]);
            o += nb;
        }
    }
    w.writeBits(2, npostfix);
    w.writeBits(4, ndirect >> npostfix);
    foreach (t; 0 .. numLiteralTypes)
        w.writeBits(2, contextMode);
    storeContextMap(w, literalMap[0 .. numLiteralHistograms], numLiteralTrees, ws.scratch);
    storeContextMap(w, distanceMap[0 .. numDistanceHistograms], numDistanceTrees, ws.scratch);

    // Prefix codes.
    size_t literalCodes = numLiteralTrees * A;
    size_t commandCodes = numCommandTypes * NUM_COMMAND_CODES;
    size_t distanceCodes = numDistanceTrees * numDistanceSymbols;
    if (!ws.depths.ensure(literalCodes + commandCodes + distanceCodes)
        || !ws.bits.ensure(literalCodes + commandCodes + distanceCodes))
        return oom;
    ubyte[] literalDepth = ws.depths[0 .. literalCodes];
    ushort[] literalBits = ws.bits[0 .. literalCodes];
    ubyte[] commandDepth = ws.depths[literalCodes .. literalCodes + commandCodes];
    ushort[] commandBits = ws.bits[literalCodes .. literalCodes + commandCodes];
    ubyte[] distanceDepth = ws.depths[literalCodes + commandCodes .. literalCodes + commandCodes + distanceCodes];
    ushort[] distanceBits = ws.bits[literalCodes + commandCodes .. literalCodes + commandCodes + distanceCodes];
    bool optimize = params.optimizeHistograms;
    storeTrees(w, literalTrees, numLiteralTrees, A, optimize, literalDepth, literalBits);
    storeTrees(w, commandTrees, numCommandTypes, NUM_COMMAND_CODES, optimize, commandDepth, commandBits);
    storeTrees(w, distanceTrees, numDistanceTrees, numDistanceSymbols, optimize, distanceDepth, distanceBits);

    // Commands.
    const(ubyte)[256]* lut1 = &kContextLookupTables[contextMode * 2];
    const(ubyte)[256]* lut2 = &kContextLookupTables[contextMode * 2 + 1];
    size_t pos = start;
    foreach (k, ref const c; cmds)
    {
        uint commandType = blocks[COMMANDS].next(w);
        uint insertCode = insertLengthCode(c.insertLength);
        uint copyCode = c.copyLength == 0 ? 0 : copyLengthCode(c.copyLength);
        uint symbol = commandSymbols[k];
        size_t co = commandType * NUM_COMMAND_CODES + symbol;
        w.writeBits(commandDepth[co], commandBits[co]);
        w.writeBits(kInsertLengthBits[insertCode], c.insertLength - kInsertLengthOffset[insertCode]);
        w.writeBits(kCopyLengthBits[copyCode],
                    c.copyLength == 0 ? 0 : c.copyLength - kCopyLengthOffset[copyCode]);
        ubyte p1 = pos >= 1 ? data[pos - 1] : 0;
        ubyte p2 = pos >= 2 ? data[pos - 2] : 0;
        foreach (b; data[pos .. pos + c.insertLength])
        {
            uint literalType = blocks[LITERALS].next(w);
            size_t lo = literalMap[literalType * 64 + ((*lut1)[p1] | (*lut2)[p2])] * A + b;
            w.writeBits(literalDepth[lo], literalBits[lo]);
            p2 = p1;
            p1 = b;
        }
        pos += c.insertLength;
        if (c.copyLength == 0)
            break; // must be the last command
        if (symbol >= 128)
        {
            uint distanceType = blocks[DISTANCES].next(w);
            uint dsym, dbits, dextra;
            distanceSymbol(c.distanceCode, npostfix, ndirect, dsym, dbits, dextra);
            uint context = c.copyLength > 4 ? 3 : c.copyLength - 2;
            size_t o = distanceMap[distanceType * 4 + context] * numDistanceSymbols + dsym;
            w.writeBits(distanceDepth[o], distanceBits[o]);
            w.writeBits(dbits, dextra);
        }
        pos += c.outputLength;
    }
    assert(pos == start + length);
    return w.error;
}

void writeWindowBits(ref BitWriter w, uint lgwin)
{
    if (lgwin == 16)
        w.writeBits(1, 0);
    else if (lgwin == 17)
        w.writeBits(7, 1);
    else if (lgwin > 17)
        w.writeBits(4, ((lgwin - 17) << 1) | 1);
    else
        w.writeBits(7, ((lgwin - 8) << 4) | 1);
}

// With optimalKeepBest, the passes measured to keep the best one.
@tuning enum uint OPTIMAL_FIRST_TRIED_PASS = 1;

// Commands of data[start .. start + length] with optimal parsing, into
// ws.commands. Each pass uses the statistics of the previous one. Updates `last`.
BrotliStatus optimalCommands(ref TreeHasher t, scope const(ubyte)[] data, size_t start, size_t length,
                             bool isLast, size_t maxBackward, ref const(LevelParams) params,
                             ref DistanceCache last, ref EncoderWorkspace ws)
{
    enum oom = BrotliStatus.outOfMemory;
    size_t end = start + length;
    if (!findOptimalMatches(t, data, start, end, maxBackward, params, ws))
        return oom;
    DistanceCache initialLast = last;
    DistanceCache bestLast;
    ulong bestBits = ulong.max;
    ws.previousCommands.length = 0;
    foreach (pass; 0 .. params.optimalPasses)
    {
        ws.commands.length = 0;
        DistanceCache passLast = initialLast;
        if (!optimalParse(data, start, end, maxBackward, params, ws.previousCommands[], passLast,
                          ws.commands, ws))
            return oom;
        // The first passes, from rough statistics, never compress best.
        uint firstTried = OPTIMAL_FIRST_TRIED_PASS < params.optimalPasses ? OPTIMAL_FIRST_TRIED_PASS
                                                                         : params.optimalPasses - 1;
        if (params.optimalKeepBest && pass >= firstTried)
        {
            // Size of the meta-block with these commands, written aside.
            ws.trial.growable = true;
            ws.trial.rewind(BitWriter.Mark.init);
            BrotliStatus r = writeCompressedMetaBlock(ws.trial, data, start, length, ws.commands[],
                                                      isLast, params, ws);
            if (r != BrotliStatus.success)
                return r;
            if (ws.trial.bitPosition() < bestBits)
            {
                bestBits = ws.trial.bitPosition();
                bestLast = passLast;
                if (!copyCommands(ws.commands, ws.bestCommands) || !ws.trial.copyTo(ws.bestMetaBlock))
                    return oom;
                ws.bestMetaBlockBits = bestBits;
            }
        }
        else
            bestLast = passLast;
        if (!copyCommands(ws.commands, ws.previousCommands))
            return oom;
    }
    if (params.optimalKeepBest && !copyCommands(ws.bestCommands, ws.commands))
        return oom;
    last = bestLast;
    return BrotliStatus.success;
}

bool copyCommands(ref const(CommandBuffer) from, ref CommandBuffer to)
{
    to.length = 0;
    foreach (ref const c; from[])
        if (!to.push(c))
            return false;
    return true;
}

// Inputs at least that long, and mostly UTF-8 text (at least
// TEXT_UTF8_PERCENT of the bytes), are hashed on 5 bytes at the levels that allow it.
@tuning enum size_t TEXT_HASH_MIN_LENGTH = 1 << 14;
@tuning enum uint TEXT_UTF8_PERCENT = 75;
@tuning enum size_t TEXT_SAMPLE_LENGTH = 1 << 16; // bytes checked for UTF-8, at the start

// Is most of `data` UTF-8? (Bytes in well formed sequences, not checking
// for overlong encodings: a heuristic.)
bool isMostlyUTF8(scope const(ubyte)[] data)
{
    size_t utf8 = 0;
    size_t i = 0;
    while (i < data.length)
    {
        uint b = data[i];
        size_t len = b < 0x80 ? 1 : (b >= 0xC2 && b < 0xE0 ? 2 : (b >= 0xE0 && b < 0xF0 ? 3 : (b >= 0xF0 && b < 0xF5 ? 4 : 0)));
        bool valid = len != 0 && i + len <= data.length;
        foreach (k; 1 .. len)
            if (valid && (data[i + k] & 0xC0) != 0x80)
                valid = false;
        if (valid)
        {
            utf8 += len;
            i += len;
        }
        else
            i += 1;
    }
    return utf8 * 100 > data.length * TEXT_UTF8_PERCENT;
}

// Smaller meta-blocks adapt better, but cost more headers.
@tuning enum size_t METABLOCK_LENGTH = 1 << 20;
@tuning enum size_t OPTIMAL_METABLOCK_LENGTH = 3 << 17; // levels with optimal parsing

BrotliStatus encodeStream(const(ubyte)[] input, ref BitWriter w, int level, int windowBits)
{
    if (level < BROTLI_MIN_LEVEL)
        level = BROTLI_MIN_LEVEL;
    if (level > BROTLI_MAX_LEVEL)
        level = BROTLI_MAX_LEVEL;
    if (windowBits < BROTLI_MIN_WINDOW_BITS)
        windowBits = BROTLI_MIN_WINDOW_BITS;
    if (windowBits > BROTLI_MAX_WINDOW_BITS)
        windowBits = BROTLI_MAX_WINDOW_BITS;
    const(LevelParams) params = kLevelParams[level];

    // No larger window than needed.
    uint lgwin = BROTLI_MIN_WINDOW_BITS;
    while (lgwin < windowBits && (size_t(1) << lgwin) - 16 < input.length)
        lgwin++;
    size_t maxBackward = (size_t(1) << lgwin) - 16;

    writeWindowBits(w, lgwin);
    if (input.length == 0)
    {
        w.writeBits(2, 3); // ISLAST, ISLASTEMPTY
        w.alignToByte();
        return w.error;
    }

    // Match finder: hash table for greedy parsing, binary trees for
    // optimal parsing. No larger than useful.
    bool optimal = params.optimalPasses > 0;
    Hasher h;
    scope(exit) h.release();
    TreeHasher t;
    scope(exit) t.release();
    if (optimal)
    {
        size_t windowSize = 16;
        while (windowSize < input.length && windowSize < (size_t(1) << lgwin)
               && windowSize < MAX_TREE_WINDOW)
            windowSize *= 2;
        if (!t.initialize(windowSize, params.useDictionary))
            return BrotliStatus.outOfMemory;
    }
    else
    {
        uint bucketBits = params.bucketBits;
        while (bucketBits > 8 && (size_t(1) << (bucketBits - 1)) >= input.length)
            bucketBits--;
        // Longer hashes are better for text, worse for binary data.
        uint hashLength = params.hashLength;
        if (hashLength == 5 && (input.length < TEXT_HASH_MIN_LENGTH
                                || !isMostlyUTF8(input[0 .. input.length < TEXT_SAMPLE_LENGTH
                                                            ? input.length : TEXT_SAMPLE_LENGTH])))
            hashLength = 4;
        if (!h.initialize(bucketBits, params.blockBits, hashLength, input.length))
            return BrotliStatus.outOfMemory;
    }

    EncoderWorkspace ws;
    scope(exit) ws.release();
    if (!ws.initialize())
        return BrotliStatus.outOfMemory;

    DistanceCache last = kInitialDistanceCache;
    size_t blockLength = optimal ? OPTIMAL_METABLOCK_LENGTH : METABLOCK_LENGTH;
    size_t pos = 0;
    while (pos < input.length)
    {
        // Meta-blocks are >= MIN_METABLOCK_LENGTH, except the last one.
        size_t length = input.length - pos;
        if (length > blockLength && length - blockLength >= MIN_METABLOCK_LENGTH)
            length = blockLength;
        assert(length <= blockLength + MIN_METABLOCK_LENGTH && length <= MAX_METABLOCK_LENGTH);
        bool isLast = pos + length == input.length;

        ws.commands.length = 0;
        DistanceCache newLast = last;
        if (optimal)
        {
            t.rebase(pos, maxBackward + MAX_METABLOCK_LENGTH);
            BrotliStatus r = optimalCommands(t, input, pos, length, isLast, maxBackward, params,
                                             newLast, ws);
            if (r != BrotliStatus.success)
                return r;
        }
        else
        {
            h.rebase(pos, maxBackward + MAX_METABLOCK_LENGTH);
            bool ok;
            if (h.hashLength == 5)
                ok = createCommands!(5, false)(h, input, pos, pos + length, maxBackward, params, newLast, ws.commands);
            else if (h.tagged)
                ok = createCommands!(4, true)(h, input, pos, pos + length, maxBackward, params, newLast, ws.commands);
            else
                ok = createCommands!(4, false)(h, input, pos, pos + length, maxBackward, params, newLast, ws.commands);
            if (!ok)
                return BrotliStatus.outOfMemory;
        }

        BitWriter.Mark mark = w.mark();
        BrotliStatus r;
        if (optimal && params.optimalKeepBest)
        {
            // Already written aside, by optimalCommands.
            w.writeBitString(ws.bestMetaBlock[], ws.bestMetaBlockBits);
            r = w.error;
        }
        else
            r = writeCompressedMetaBlock(w, input, pos, length, ws.commands[], isLast, params, ws);
        if (r == BrotliStatus.outOfMemory)
            return r;

        // Keep it compressed only if not larger than uncompressed.
        ulong headerBits = 1 + 2 + 4 * metaBlockNibbles(length) + 1;
        ulong markBits = cast(ulong) mark.pos * 8 + mark.nbits;
        ulong uncompressedEnd = ((markBits + headerBits + 7) / 8 + length) * 8 + (isLast ? 8 : 0);
        ulong compressedEnd = w.bitPosition();
        if (isLast)
            compressedEnd = (compressedEnd + 7) & ~7UL;
        if (r != BrotliStatus.success || compressedEnd >= uncompressedEnd)
        {
            // The decoder doesn't update the last distances for uncompressed meta-blocks.
            w.rewind(mark);
            writeUncompressedMetaBlock(w, input[pos .. pos + length]);
            if (isLast)
                w.writeBits(2, 3); // ISLAST, ISLASTEMPTY
        }
        else
            last = newLast;
        if (w.error != BrotliStatus.success)
            return w.error;
        pos += length;
    }
    w.alignToByte();
    return w.error;
}


// ---------------------------------------------------------------------------
// Encoder tests
// ---------------------------------------------------------------------------

version(unittest)
{
    // xorshift64*, deterministic.
    struct TestRandom
    {
    pure nothrow @nogc @safe:
        ulong state = 0x9E3779B97F4A7C15UL;

        ulong next()
        {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            return state * 0x2545F4914F6CDD1DUL;
        }

        uint below(uint n)
        {
            return cast(uint)(next() % n);
        }
    }

    // Writer with its own buffer, for tests.
    BitWriter testWriter()
    {
        BitWriter w;
        w.growable = true;
        return w;
    }
}

unittest
{
    // The dictionary index has the values of buildDictionaryHashTable().
    MallocArray!DictionaryHashTable t; // too large for the stack
    assert(t.resize(1));
    t[0] = buildDictionaryHashTable();
    assert(t[0].bucketStart == kDictionaryHashTable.bucketStart);
    assert(t[0].words == kDictionaryHashTable.words);
}

unittest
{
    // Every "omit first n" output long enough is in its bucket, once.
    OmitFirstIndex index;
    scope(exit) index.release();
    assert(index.initialize());
    size_t total = 0;
    foreach (len; MIN_DICTIONARY_WORD_LENGTH .. MAX_DICTIONARY_WORD_LENGTH + 1)
    {
        foreach (idx; 0 .. 1u << kDictionarySizeBits[len])
        {
            foreach (n; 1 .. 10)
            {
                if (kOmitFirstTransforms[n] == 0 || len < n + OMIT_FIRST_MIN_LENGTH)
                    continue;
                uint entry = cast(uint)((n << 16) | (len << 11) | idx);
                size_t found = 0;
                foreach (other; index.bucket(omitFirstHash(load32(omitFirstOutput(entry), 0))))
                    if (other == entry)
                        found++;
                assert(found == 1);
                total++;
            }
        }
    }
    assert(total == index.words.length);
}

unittest
{
    // Every dictionary word is in the bucket of its first 4 bytes, once.
    immutable(ubyte)[] dict = dictionary();
    size_t total = 0;
    foreach (len; MIN_DICTIONARY_WORD_LENGTH .. MAX_DICTIONARY_WORD_LENGTH + 1)
    {
        foreach (idx; 0 .. 1u << kDictionarySizeBits[len])
        {
            size_t offset = kDictionaryOffsets[len] + len * idx;
            uint h = dictionaryHash(load32(dict, offset));
            size_t found = 0;
            foreach (entry; kDictionaryHashTable.words[kDictionaryHashTable.bucketStart[h]
                                                       .. kDictionaryHashTable.bucketStart[h + 1]])
                if ((entry >> 11) == len && (entry & 2047) == idx)
                    found++;
            assert(found == 1);
            total++;
        }
    }
    assert(total == NUM_DICTIONARY_WORDS && kDictionaryHashTable.bucketStart[$ - 1] == NUM_DICTIONARY_WORDS);
}

unittest
{
    // The log2 tables.
    foreach (x; 1 .. 256)
        assert(kLog2Small[x] == (log2Squaring(x, 20) + 8) >> 4);
    assert(kLog2Small[0] == 0);
    foreach (i; 0 .. 257)
        assert(kLog2Mantissa[i] == ((log2Squaring(256 + i, 20) - (8 << 20)) + 8) >> 4);
}

unittest
{
    // log2Fixed is within 1/65536 of log2 (here, of the squaring method with
    // 20 fractional bits, rounded to 16), and never decreases.
    void check(size_t x)
    {
        uint exact = (log2Squaring(x, 20) + 8) >> 4;
        uint v = log2Fixed(x);
        assert(v + 1 >= exact && v <= exact + 1);
    }
    uint previous = 0;
    foreach (size_t x; 1 .. 1 << 20)
    {
        check(x);
        uint v = log2Fixed(x);
        assert(v >= previous);
        previous = v;
    }
    for (ulong x = 1 << 20; x <= size_t.max / 2; x += x / 300_000 + 1)
        check(cast(size_t) x);
    assert(log2Fixed(0) == 0 && log2Fixed(1) == 0 && log2Fixed(2) == COST_ONE_BIT);
    assert(log2Fixed(1 << 20) == 20 * COST_ONE_BIT);
}

unittest
{
    // Length codes cover every length.
    foreach (len; 0 .. 70_000)
    {
        uint c = insertLengthCode(len);
        assert(kInsertLengthOffset[c] <= len && len - kInsertLengthOffset[c] < (1u << kInsertLengthBits[c]));
        if (len >= 2)
        {
            c = copyLengthCode(len);
            assert(kCopyLengthOffset[c] <= len && len - kCopyLengthOffset[c] < (1u << kCopyLengthBits[c]));
        }
    }
    static immutable size_t[2] kLongLengths = [(1 << 24) - 1, 1 << 24];
    foreach (len; kLongLengths)
    {
        uint c = insertLengthCode(len);
        assert(c == 23 && len - kInsertLengthOffset[c] < (1u << kInsertLengthBits[c]));
        c = copyLengthCode(len);
        assert(c == 23 && len - kCopyLengthOffset[c] < (1u << kCopyLengthBits[c]));
    }

    // Insert-and-copy symbols agree with the decoder.
    foreach (insertCode; 0 .. 24)
        foreach (copyCode; 0 .. 24)
            foreach (implicitInt; 0 .. 2)
            {
                bool implicit = implicitInt != 0;
                uint sym = commandSymbol(insertCode, copyCode, implicit);
                assert(sym < NUM_COMMAND_CODES);
                assert(kCommandCodes[sym].insertOffset == kInsertLengthOffset[insertCode]);
                assert(kCommandCodes[sym].copyOffset == kCopyLengthOffset[copyCode]);
                assert((sym < 128) == (implicit && insertCode < 8 && copyCode < 16));
            }
}

unittest
{
    // Distance symbols decode back, as in decodeCompressedMetaBlock.
    foreach (npostfix; 0 .. 4)
        foreach (ndirectHigh; 0 .. 16)
        {
            uint ndirect = ndirectHigh << npostfix;
            uint numSymbols = 16 + ndirect + (48u << npostfix);
            void check(size_t distance)
            {
                uint sym, nbits, extra;
                distanceSymbol(distance + 15, npostfix, ndirect, sym, nbits, extra);
                assert(sym >= 16 && sym < numSymbols);
                assert(nbits <= 24 && (extra >> nbits) == 0);
                size_t decoded;
                if (sym < 16 + ndirect)
                    decoded = sym - 15;
                else
                {
                    uint x = sym - ndirect - 16;
                    uint ndistbits = 1 + (x >> (npostfix + 1));
                    assert(ndistbits == nbits);
                    uint hcode = x >> npostfix;
                    uint lcode = x & ((1u << npostfix) - 1);
                    size_t offset = (size_t(2 + (hcode & 1)) << ndistbits) - 4;
                    decoded = ((offset + extra) << npostfix) + lcode + ndirect + 1;
                }
                assert(decoded == distance);
            }
            foreach (d; 1 .. 20_000)
                check(d);
            for (size_t d = 20_000; d < (1 << 26); d = d * 9 / 8 + 1)
                check(d);
        }

    // Short codes, as in decodeCompressedMetaBlock.
    DistanceCache last = [100, 7, 3, 2];
    size_t[16] dists;
    shortCodeDistances(last, dists);
    foreach (code; 0 .. 16)
    {
        static immutable ubyte[4] kAge = [0, 3, 2, 1];
        size_t expected = last[kAge[kDistanceShortCodeIndexOffset[code]]]
                        + kDistanceShortCodeValueOffset[code];
        assert(dists[code] == expected);
        assert(distanceCode(dists[code], last) < 16);
    }
    assert(distanceCode(100, last) == 0);
    assert(distanceCode(7, last) == 1);
    assert(distanceCode(97, last) == 8);
    assert(distanceCode(1000, last) == 1015);
}

version(unittest)
{
    // Write a prefix code for `counts` followed by every used symbol, read
    // them back with the decoder.
    void checkPrefixCodeRoundTrip(scope const(uint)[] counts)
    {
        uint alphabetSize = cast(uint) counts.length;
        ubyte[MAX_ALPHABET_SIZE] depth;
        ushort[MAX_ALPHABET_SIZE] bits;
        BitWriter w = testWriter();
        buildAndStorePrefixCode(w, counts, depth[], bits[]);
        size_t treeBits = cast(size_t) w.bitPosition();

        // Lengths are limited, the code is complete.
        uint used = 0;
        ulong kraft = 0;
        foreach (i, c; counts)
        {
            assert(depth[i] <= MAX_CODE_LENGTH);
            assert(c != 0 || depth[i] == 0);
            if (c != 0)
                used++;
            if (depth[i] != 0)
                kraft += 1UL << (MAX_CODE_LENGTH - depth[i]);
        }
        if (used >= 2)
            assert(kraft == 1UL << MAX_CODE_LENGTH);

        // The dry run gives the same size.
        BitWriter dry;
        assert(storePrefixCode!true(dry, counts, depth[0 .. alphabetSize]) == treeBits);

        foreach (i, c; counts)
            if (c != 0)
                w.writeBits(depth[i], bits[i]);
        w.writeBits(8, 0xA5);
        w.alignToByte();
        assert(w.error == BrotliStatus.success);

        BitReader br;
        br.data = w.buf[0 .. w.pos];
        uint[kMaxHuffmanTableSize[$ - 1]] table;
        assert(readPrefixCode(br, alphabetSize, table[0 .. maxHuffmanTableSize(alphabetSize)]));
        foreach (i, c; counts)
            if (c != 0)
                assert(readSymbol(br, table) == i);
        assert(br.readBits(8) == 0xA5);
        assert(!br.failed);
    }
}

unittest
{
    TestRandom rng;
    uint[MAX_ALPHABET_SIZE] counts;
    foreach (iter; 0 .. 3000)
    {
        static immutable uint[12] kSizes = [2, 3, 4, 5, 18, 26, 64, 256, 258, 272, 520, 704];
        uint alphabetSize = kSizes[rng.below(12)];
        uint[] h = counts[0 .. alphabetSize];
        h[] = 0;
        switch (iter % 7)
        {
            case 0: // few symbols
                foreach (k; 0 .. rng.below(6))
                    h[rng.below(alphabetSize)] = 1 + rng.below(1000);
                break;
            case 1: // dense, uniform
                foreach (ref c; h)
                    c = 1 + rng.below(100);
                break;
            case 2: // very skewed, needs length limiting
                foreach (i, ref c; h)
                    c = (rng.below(3) == 0) ? 0 : cast(uint)(1UL << rng.below(31));
                break;
            case 3: // Fibonacci-like, deepest possible trees
            {
                uint a = 1, b = 1;
                foreach (i, ref c; h)
                {
                    c = a;
                    uint t = a + b;
                    a = b;
                    b = t > 0x4000_0000 ? 0x4000_0000 : t;
                }
                break;
            }
            case 4: // runs of equal and zero counts, for the run-length coding
            {
                size_t i = 0;
                while (i < alphabetSize)
                {
                    size_t run = 1 + rng.below(40);
                    uint v = rng.below(2) ? 0 : 1u << rng.below(12);
                    for (size_t k = 0; k < run && i < alphabetSize; ++k)
                        h[i++] = v;
                }
                break;
            }
            case 5: // all equal
                h[] = 7;
                break;
            default: // sparse random
                foreach (ref c; h)
                    c = rng.below(4) == 0 ? rng.below(50000) : 0;
                break;
        }
        checkPrefixCodeRoundTrip(h);
    }

    // Exact run lengths that the run-length coding treats specially.
    static immutable uint[3] kRunSizes = [64, 256, 704];
    foreach (alphabetSize; kRunSizes)
        foreach (run; 1 .. 40)
            foreach (zeroInt; 0 .. 2)
            {
                bool zero = zeroInt != 0;
                uint[] h = counts[0 .. alphabetSize];
                foreach (i, ref c; h)
                    c = cast(uint)(i % 5 + 1) * 100;
                foreach (i; 10 .. 10 + run)
                    if (i < alphabetSize)
                        h[i] = zero ? 0 : 3;
                checkPrefixCodeRoundTrip(h);
            }
}

unittest
{
    // Context maps round-trip.
    TestRandom rng;
    MallocArray!uint scratch;
    assert(scratch.resize(2 * MAX_CONTEXT_MAP_LENGTH));
    MallocArray!ubyte map;
    assert(map.resize(MAX_CONTEXT_MAP_LENGTH));
    MallocArray!ubyte decodedStorage;
    assert(decodedStorage.resize(MAX_CONTEXT_MAP_LENGTH));
    foreach (iter; 0 .. 400)
    {
        size_t length = 64 * (1 + rng.below(256));
        if (iter % 5 == 0)
            length = 4 * (1 + rng.below(256));
        else if (iter % 5 == 1)
            length = MAX_CONTEXT_MAP_LENGTH;
        uint numTrees = 1 + rng.below(iter % 3 == 0 ? 256 : 8);
        foreach (i, ref v; map[0 .. length])
        {
            if (iter % 4 == 0)
                v = cast(ubyte) rng.below(numTrees);
            else if (iter % 4 == 1)
                v = cast(ubyte)((i / (1 + iter % 13)) % numTrees);
            else
                v = rng.below(10) == 0 ? cast(ubyte) rng.below(numTrees) : 0;
        }
        BitWriter w = testWriter();
        storeContextMap(w, map[0 .. length], numTrees, scratch);
        w.writeBits(5, 0x15);
        w.alignToByte();

        BitReader br;
        br.data = w.buf[0 .. w.pos];
        ubyte[] decoded = decodedStorage[0 .. length];
        uint decodedTrees;
        assert(decodeContextMap(br, decoded, decodedTrees));
        assert(decodedTrees == numTrees);
        assert(decoded == map[0 .. length]);
        assert(br.readBits(5) == 0x15);
        assert(!br.failed);
    }
}

unittest
{
    // Clustering invariants.
    TestRandom rng;
    enum A = 64;
    uint[MAX_CLUSTER_INPUTS * A] histograms;
    uint[MAX_CLUSTER_INPUTS * A] clusters;
    ushort[MAX_CLUSTER_INPUTS * A] scratch;
    MallocArray!long delta;
    assert(delta.resize(MAX_CLUSTER_INPUTS * MAX_CLUSTER_INPUTS));
    foreach (iter; 0 .. 300)
    {
        size_t n = 1 + rng.below(MAX_CLUSTER_INPUTS);
        uint maxClusters = 1 + rng.below(iter % 2 ? 4 : 64);
        foreach (i; 0 .. n)
        {
            uint kind = rng.below(4); // empty, sparse, dense, copy of a previous one
            foreach (k; 0 .. A)
            {
                uint v = 0;
                if (kind == 1 && rng.below(8) == 0)
                    v = rng.below(1000);
                else if (kind == 2)
                    v = rng.below(50);
                else if (kind == 3 && i > 0)
                    v = histograms[(i - 1) * A + k];
                histograms[i * A + k] = v;
            }
        }
        ubyte[MAX_CLUSTER_INPUTS] map;
        ulong cost;
        uint k = clusterHistograms(histograms[], n, A, maxClusters, map[], clusters[], scratch[], delta, cost);
        assert(k >= 1 && k <= maxClusters && k <= n);
        bool[MAX_CLUSTER_INPUTS] used;
        foreach (i; 0 .. n)
        {
            assert(map[i] < k);
            if (!isEmpty(histograms[i * A .. (i + 1) * A]))
                used[map[i]] = true;
        }
        uint[A] sum;
        foreach (c; 0 .. k)
        {
            sum[] = 0;
            foreach (i; 0 .. n)
                if (map[i] == c)
                    sum[] += histograms[i * A .. (i + 1) * A];
            assert(sum[] == clusters[c * A .. (c + 1) * A]);
        }
        bool anyUsed = false;
        foreach (c; 0 .. k)
            anyUsed |= used[c];
        if (anyUsed)
            foreach (c; 0 .. k)
                assert(used[c]);
    }
}

unittest
{
    // Block length codes cover every length.
    foreach (len; 1 .. 100_000)
    {
        uint c = blockLengthCode(len);
        assert(kBlockLengthOffset[c] <= len && len - kBlockLengthOffset[c] < (1u << kBlockLengthBits[c]));
    }
    assert(blockLengthCode(1 << 24) == 25);

    // Histogram smoothing keeps used symbols used.
    TestRandom rng;
    uint[MAX_ALPHABET_SIZE] h, original;
    foreach (iter; 0 .. 2000)
    {
        size_t n = 2 + rng.below(MAX_ALPHABET_SIZE - 1);
        foreach (ref c; h[0 .. n])
        {
            uint kind = rng.below(4);
            c = kind == 0 ? 0 : (kind == 1 ? 1 + rng.below(4) : (kind == 2 ? 100 + rng.below(10) : rng.below(100000)));
        }
        original[0 .. n] = h[0 .. n];
        optimizeHistogramForRle(h[0 .. n]);
        foreach (i; 0 .. n)
            assert(original[i] == 0 || h[i] != 0);
    }

    // Block splits cover the symbols.
    EncoderWorkspace ws;
    scope(exit) ws.release();
    assert(ws.initialize());
    BlockSplit split;
    scope(exit) split.release();
    MallocArray!ushort symbols;
    assert(symbols.resize(50_000));
    foreach (iter; 0 .. 30)
    {
        size_t n = iter < 5 ? iter * 50 : 1 + rng.below(50_000);
        uint alphabet = iter % 2 ? 256 : 704;
        // Several regions with different statistics.
        foreach (i, ref s; symbols[0 .. n])
            s = cast(ushort)(((i / 3000) % 3) * 50 + rng.below((i / 7000) % 2 ? 40 : alphabet - 100));
        foreach (q; 0 .. 2)
        {
            assert(splitSymbols(symbols[0 .. n], alphabet, kSplitParams[LITERALS][q], split, ws) == BrotliStatus.success);
            size_t total = 0;
            uint seen = 0;
            foreach (b; 0 .. split.numBlocks)
            {
                assert(split.lengths[b] > 0);
                assert(split.types[b] < split.numTypes && split.types[b] <= seen);
                if (split.types[b] == seen)
                    seen++;
                if (b > 0)
                    assert(split.types[b] != split.types[b - 1]);
                total += split.lengths[b];
            }
            assert(total == n && split.numTypes >= 1);
            assert(n == 0 ? split.numBlocks == 0 : (seen == split.numTypes && split.types[0] == 0));
        }
    }
}

unittest
{
    // The matches of the optimal parser's match finder are real matches.
    static immutable string text = "The quick brown fox jumps over the lazy dog. Then, the house of the "
        ~ "World: information about the people. THE TIME AND THE PLACE, the time and the place. ";
    TestRandom rng;
    MallocArray!ubyte data;
    assert(data.resize(30_000));
    foreach (i, ref b; data)
    {
        if ((i / 2000) % 3 == 0)
            b = cast(ubyte) text[i % text.length];
        else if ((i / 2000) % 3 == 1)
            b = cast(ubyte)(rng.below(4) == 0 ? rng.below(256) : 'a' + rng.below(3));
        else
            b = cast(ubyte) rng.below(256);
    }
    foreach (windowBits; [10, 16])
    {
        TreeHasher t;
        scope(exit) t.release();
        assert(t.initialize(size_t(1) << windowBits, true));
        MatchBuffer matches;
        scope(exit) matches.release();
        size_t maxBackward = (size_t(1) << windowBits) - 16;
        immutable(ubyte)[] dict = dictionary();
        size_t numDictionary = 0;
        foreach (pos; 0 .. data.length - 4)
        {
            size_t maxLength = data.length - pos < 300 ? data.length - pos : 300;
            size_t maxDistance = pos < maxBackward ? pos : maxBackward;
            matches.length = 0;
            findAllMatches(t, data, pos, maxLength, maxDistance, kLevelParams[11], matches);
            assert(!matches.failed);
            size_t previous = 0;
            foreach (ref m; matches[])
            {
                assert(m.length > previous && m.length <= maxLength);
                previous = m.length;
                if (m.wordLength == 0)
                {
                    assert(m.distance >= 1 && m.distance <= maxDistance);
                    assert(data[pos - m.distance .. pos - m.distance + m.length] == data[pos .. pos + m.length]);
                }
                else
                {
                    numDictionary++;
                    size_t address = m.distance - maxDistance - 1;
                    size_t len = m.wordLength;
                    size_t idx = address & ((1u << kDictionarySizeBits[len]) - 1);
                    size_t transform = address >> kDictionarySizeBits[len];
                    assert(transform < kTransforms.length);
                    ubyte[MAX_TRANSFORMED_WORD_LENGTH] out_;
                    size_t offset = kDictionaryOffsets[len] + len * idx;
                    size_t n = transformWord(out_[], dict[offset .. offset + len], kTransforms[transform]);
                    assert(n == m.length && out_[0 .. n] == data[pos .. pos + n]);
                }
            }
        }
        assert(numDictionary > 0);
    }

    // Start positions: when full, a push replaces the most expensive one.
    StartPosQueue q;
    StartPosition[8] model;
    size_t modelSize = 0;
    foreach (i; 0 .. 200)
    {
        StartPosition p = StartPosition(i, rng.below(20));
        q.push(p);
        if (modelSize == 8)
            modelSize--; // the last one is the most expensive
        model[modelSize++] = p;
        for (size_t k = modelSize - 1; k > 0 && model[k].costDiff < model[k - 1].costDiff; --k)
        {
            StartPosition tmp = model[k];
            model[k] = model[k - 1];
            model[k - 1] = tmp;
        }
        assert(q.size() == modelSize);
        foreach (k; 0 .. modelSize)
            assert(q[k].costDiff == model[k].costDiff);
    }
}

unittest
{
    // Whole streams.
    static immutable string[] samples =
    [
        "", "a", "ab", "hello", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
        "abcabcabcabcabcabcabcabcabcabcabcabcabcabcxyzabcabcabcabcabcabcabcabc",
    ];
    foreach (level; BROTLI_MIN_LEVEL .. BROTLI_MAX_LEVEL + 1)
        foreach (s; samples)
        {
            const(ubyte)[] input = cast(const(ubyte)[]) s;
            ubyte[512] compressedStorage;
            ubyte[] compressed = compressedStorage[0 .. brotliCompressBound(input.length)];
            size_t clen;
            assert(brotliCompress(input, compressed, clen, level) == BrotliStatus.success);
            assert(clen <= brotliCompressBound(input.length));
            ubyte[256] decoded;
            size_t dlen;
            assert(brotliDecompress(compressed[0 .. clen], decoded[], dlen) == BrotliStatus.success);
            assert(decoded[0 .. dlen] == input);
            if (clen > 0)
                assert(brotliCompress(input, compressed[0 .. clen - 1], clen, level)
                       == BrotliStatus.outputTooSmall);
        }
    ubyte[] empty;
    size_t len;
    assert(brotliCompress(null, empty, len) == BrotliStatus.outputTooSmall);
}


// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

// Static dictionary index, see DictionaryHashTable.
static immutable ushort[(1 << DICTIONARY_HASH_BITS) + 1] kDictionaryBucketStart =
[
    0, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5,
    7, 7, 9, 35, 37, 37, 37, 37, 37, 37, 37, 37, 37, 39, 39, 39,
    39, 40, 41, 42, 42, 42, 42, 42, 44, 44, 44, 45, 45, 45, 47, 48,
    48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49,
    49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 54, 56, 56, 57, 57,
    57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 59, 59, 59, 59, 61, 61, 61, 61, 61, 67, 68, 82, 83,
    83, 83, 83, 83, 83, 83, 83, 83, 83, 84, 84, 84, 84, 84, 84, 84,
    84, 86, 86, 91, 91, 91, 96, 96, 96, 98, 99, 103, 103, 104, 105, 107,
    107, 107, 107, 107, 108, 108, 108, 108, 109, 109, 110, 111, 111, 111, 111, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 113, 113, 118, 118, 126, 129, 129,
    129, 130, 131, 131, 131, 133, 134, 134, 139, 139, 139, 139, 139, 139, 141, 141,
    147, 147, 150, 151, 153, 153, 153, 153, 153, 154, 154, 154, 154, 154, 154, 156,
    157, 157, 157, 176, 177, 180, 180, 180, 180, 188, 188, 196, 196, 197, 197, 197,
    197, 197, 197, 197, 197, 198, 199, 200, 201, 201, 202, 203, 203, 203, 203, 203,
    203, 203, 203, 204, 205, 205, 205, 208, 208, 208, 208, 208, 208, 209, 217, 217,
    217, 217, 217, 217, 217, 220, 225, 225, 226, 227, 229, 229, 229, 229, 229, 229,
    229, 230, 230, 230, 230, 240, 240, 240, 241, 241, 241, 251, 251, 261, 261, 261,
    261, 261, 264, 264, 264, 264, 264, 264, 264, 265, 266, 266, 267, 267, 267, 317,
    317, 320, 320, 320, 320, 321, 322, 323, 324, 324, 324, 324, 324, 324, 324, 324,
    324, 326, 326, 339, 352, 352, 352, 352, 352, 352, 352, 352, 352, 352, 355, 355,
    361, 366, 366, 366, 366, 366, 366, 369, 370, 387, 387, 387, 387, 387, 387, 408,
    408, 408, 408, 408, 408, 408, 408, 409, 409, 410, 410, 411, 411, 411, 411, 411,
    412, 412, 419, 419, 419, 419, 419, 419, 420, 420, 421, 421, 421, 421, 422, 422,
    422, 422, 423, 429, 429, 429, 437, 438, 439, 439, 439, 439, 439, 441, 442, 442,
    442, 442, 442, 442, 443, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444,
    447, 447, 447, 447, 448, 449, 449, 449, 449, 449, 456, 456, 458, 466, 473, 473,
    473, 473, 474, 474, 475, 475, 475, 475, 479, 480, 480, 481, 481, 482, 483, 486,
    486, 486, 486, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 492, 492, 492,
    492, 492, 494, 494, 494, 495, 495, 495, 497, 497, 498, 498, 499, 499, 499, 499,
    499, 501, 501, 501, 501, 501, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502,
    502, 506, 506, 506, 508, 509, 509, 509, 509, 509, 511, 511, 512, 512, 512, 512,
    512, 512, 514, 514, 514, 514, 514, 514, 515, 516, 516, 516, 516, 518, 519, 519,
    519, 520, 520, 520, 522, 522, 523, 527, 527, 527, 527, 527, 531, 531, 531, 531,
    531, 531, 537, 537, 537, 537, 538, 551, 551, 551, 553, 556, 556, 556, 559, 560,
    561, 561, 561, 561, 561, 561, 561, 561, 561, 562, 563, 564, 564, 564, 564, 564,
    564, 564, 574, 575, 575, 576, 576, 576, 576, 576, 576, 576, 576, 577, 578, 578,
    578, 579, 579, 582, 586, 586, 586, 586, 586, 586, 587, 589, 589, 592, 592, 592,
    592, 592, 594, 594, 594, 594, 594, 594, 594, 594, 595, 595, 598, 605, 605, 605,
    606, 607, 607, 607, 607, 607, 607, 610, 610, 610, 610, 610, 610, 610, 610, 611,
    617, 617, 617, 617, 633, 633, 633, 633, 633, 635, 635, 635, 635, 635, 635, 635,
    635, 635, 636, 636, 636, 641, 641, 641, 641, 642, 654, 654, 654, 654, 654, 654,
    666, 667, 667, 669, 670, 670, 670, 671, 671, 674, 675, 675, 675, 675, 675, 675,
    675, 675, 675, 675, 675, 675, 675, 675, 675, 675, 675, 675, 675, 675, 675, 675,
    675, 677, 677, 678, 678, 678, 679, 679, 679, 679, 679, 679, 684, 684, 685, 685,
    685, 686, 686, 686, 686, 686, 686, 686, 687, 688, 690, 690, 690, 693, 693, 693,
    693, 702, 702, 702, 702, 702, 704, 704, 706, 706, 708, 708, 709, 709, 710, 710,
    710, 711, 711, 711, 711, 713, 713, 713, 713, 713, 717, 717, 717, 718, 719, 719,
    719, 719, 719, 719, 720, 720, 720, 720, 720, 721, 721, 721, 733, 733, 733, 733,
    733, 733, 735, 735, 735, 735, 735, 741, 741, 743, 743, 743, 743, 743, 743, 743,
    745, 745, 745, 745, 745, 745, 745, 746, 746, 746, 747, 747, 747, 748, 748, 749,
    753, 753, 753, 753, 753, 753, 754, 770, 770, 770, 770, 770, 770, 770, 770, 770,
    770, 770, 770, 770, 770, 770, 770, 770, 772, 772, 772, 776, 776, 776, 776, 776,
    776, 776, 776, 776, 776, 776, 776, 776, 777, 778, 778, 778, 778, 778, 778, 779,
    779, 780, 780, 780, 780, 782, 782, 782, 783, 788, 789, 789, 789, 789, 809, 809,
    809, 809, 809, 809, 809, 811, 811, 813, 813, 813, 813, 813, 813, 871, 871, 873,
    873, 873, 873, 873, 873, 873, 873, 873, 873, 873, 873, 883, 883, 883, 883, 885,
    885, 885, 885, 885, 886, 886, 886, 886, 886, 887, 887, 888, 890, 890, 890, 890,
    890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 892,
    892, 892, 895, 896, 896, 898, 898, 898, 898, 898, 898, 898, 898, 898, 898, 898,
    898, 898, 898, 905, 906, 906, 906, 906, 906, 906, 906, 906, 906, 906, 906, 906,
    906, 906, 906, 906, 908, 908, 908, 908, 909, 910, 910, 910, 910, 910, 910, 911,
    911, 912, 913, 913, 913, 913, 913, 913, 913, 913, 914, 916, 916, 916, 916, 916,
    917, 917, 917, 918, 918, 918, 918, 918, 920, 920, 920, 920, 920, 921, 925, 925,
    925, 925, 925, 926, 926, 928, 928, 928, 928, 928, 929, 929, 930, 930, 930, 930,
    931, 931, 931, 931, 931, 931, 931, 934, 935, 936, 936, 936, 937, 937, 937, 937,
    937, 937, 937, 937, 937, 945, 945, 945, 945, 946, 946, 946, 947, 949, 949, 949,
    949, 949, 949, 949, 949, 949, 949, 951, 951, 951, 951, 951, 952, 955, 955, 955,
    956, 956, 956, 956, 956, 956, 956, 960, 960, 960, 960, 960, 960, 960, 960, 960,
    960, 960, 960, 960, 960, 960, 960, 961, 962, 962, 962, 964, 964, 964, 964, 969,
    969, 969, 971, 971, 971, 972, 972, 972, 972, 972, 972, 972, 972, 973, 973, 974,
    977, 977, 977, 977, 978, 978, 978, 978, 978, 979, 980, 980, 996, 996, 996, 996,
    997, 1005, 1005, 1005, 1005, 1007, 1007, 1007, 1007, 1007, 1007, 1007, 1007, 1010, 1012, 1012,
    1012, 1012, 1012, 1012, 1012, 1012, 1012, 1012, 1012, 1013, 1013, 1014, 1014, 1014, 1014, 1015,
    1015, 1016, 1016, 1016, 1016, 1016, 1016, 1016, 1018, 1018, 1018, 1020, 1020, 1020, 1020, 1020,
    1020, 1021, 1021, 1033, 1035, 1035, 1036, 1051, 1052, 1053, 1053, 1055, 1055, 1056, 1057, 1067,
    1077, 1077, 1077, 1081, 1081, 1081, 1081, 1082, 1083, 1085, 1086, 1086, 1086, 1086, 1087, 1088,
    1088, 1088, 1089, 1091, 1091, 1091, 1091, 1091, 1091, 1105, 1106, 1106, 1106, 1106, 1106, 1106,
    1106, 1106, 1106, 1106, 1106, 1106, 1106, 1106, 1106, 1106, 1106, 1106, 1106, 1106, 1106, 1106,
    1106, 1106, 1106, 1108, 1108, 1108, 1108, 1108, 1108, 1108, 1110, 1110, 1110, 1110, 1110, 1111,
    1111, 1113, 1113, 1113, 1113, 1117, 1117, 1117, 1119, 1119, 1119, 1119, 1119, 1119, 1119, 1120,
    1121, 1121, 1121, 1122, 1125, 1125, 1131, 1131, 1132, 1133, 1136, 1136, 1137, 1137, 1137, 1138,
    1138, 1138, 1138, 1138, 1138, 1138, 1138, 1138, 1138, 1139, 1139, 1139, 1144, 1144, 1144, 1171,
    1171, 1171, 1171, 1176, 1177, 1181, 1181, 1183, 1183, 1183, 1185, 1186, 1186, 1186, 1186, 1191,
    1191, 1191, 1199, 1199, 1199, 1200, 1200, 1200, 1201, 1201, 1203, 1203, 1203, 1205, 1207, 1207,
    1207, 1207, 1208, 1208, 1209, 1209, 1209, 1209, 1209, 1209, 1209, 1210, 1210, 1210, 1212, 1212,
    1212, 1212, 1212, 1212, 1212, 1212, 1212, 1212, 1213, 1213, 1213, 1216, 1223, 1224, 1224, 1228,
    1228, 1230, 1230, 1231, 1231, 1231, 1231, 1232, 1232, 1232, 1232, 1232, 1232, 1232, 1232, 1232,
    1243, 1243, 1243, 1243, 1243, 1244, 1244, 1252, 1254, 1254, 1254, 1255, 1255, 1255, 1255, 1255,
    1255, 1255, 1259, 1261, 1261, 1261, 1261, 1262, 1264, 1264, 1264, 1264, 1264, 1264, 1264, 1264,
    1264, 1264, 1264, 1264, 1264, 1265, 1266, 1266, 1266, 1266, 1266, 1267, 1267, 1267, 1271, 1271,
    1271, 1271, 1271, 1271, 1271, 1272, 1272, 1274, 1287, 1287, 1287, 1287, 1287, 1290, 1290, 1292,
    1292, 1293, 1307, 1307, 1313, 1314, 1317, 1317, 1317, 1317, 1317, 1318, 1319, 1319, 1323, 1323,
    1323, 1323, 1323, 1323, 1323, 1323, 1323, 1323, 1323, 1324, 1324, 1324, 1324, 1324, 1324, 1325,
    1326, 1326, 1326, 1326, 1326, 1327, 1327, 1327, 1327, 1327, 1327, 1327, 1332, 1332, 1333, 1338,
    1338, 1338, 1339, 1340, 1340, 1340, 1340, 1340, 1340, 1340, 1340, 1340, 1342, 1342, 1342, 1342,
    1342, 1352, 1352, 1352, 1352, 1353, 1353, 1355, 1355, 1355, 1355, 1355, 1356, 1356, 1357, 1367,
    1368, 1368, 1369, 1369, 1369, 1369, 1369, 1369, 1369, 1370, 1370, 1370, 1370, 1370, 1370, 1370,
    1377, 1378, 1378, 1378, 1378, 1378, 1378, 1379, 1382, 1383, 1383, 1383, 1383, 1385, 1388, 1388,
    1389, 1389, 1389, 1389, 1389, 1391, 1391, 1391, 1391, 1391, 1391, 1391, 1391, 1391, 1391, 1392,
    1398, 1398, 1398, 1398, 1398, 1398, 1399, 1400, 1404, 1406, 1406, 1409, 1409, 1409, 1409, 1409,
    1409, 1410, 1417, 1425, 1425, 1425, 1425, 1425, 1430, 1430, 1430, 1439, 1439, 1439, 1439, 1439,
    1439, 1441, 1441, 1441, 1441, 1441, 1441, 1441, 1442, 1442, 1442, 1442, 1443, 1443, 1452, 1452,
    1461, 1461, 1462, 1467, 1467, 1467, 1467, 1468, 1468, 1469, 1469, 1474, 1474, 1474, 1474, 1481,
    1481, 1485, 1487, 1487, 1487, 1487, 1489, 1490, 1490, 1490, 1490, 1493, 1493, 1493, 1495, 1495,
    1497, 1497, 1497, 1497, 1497, 1498, 1498, 1498, 1498, 1500, 1500, 1500, 1500, 1500, 1500, 1500,
    1500, 1500, 1500, 1503, 1503, 1503, 1503, 1503, 1503, 1503, 1503, 1503, 1503, 1503, 1503, 1503,
    1503, 1503, 1503, 1504, 1507, 1507, 1510, 1512, 1514, 1514, 1514, 1514, 1514, 1514, 1514, 1514,
    1516, 1516, 1516, 1516, 1516, 1517, 1517, 1517, 1517, 1517, 1523, 1523, 1523, 1524, 1524, 1524,
    1524, 1524, 1527, 1527, 1527, 1527, 1527, 1527, 1527, 1527, 1527, 1527, 1527, 1527, 1527, 1528,
    1528, 1531, 1531, 1531, 1531, 1531, 1531, 1531, 1532, 1532, 1532, 1532, 1532, 1532, 1532, 1534,
    1535, 1535, 1536, 1539, 1539, 1539, 1539, 1539, 1539, 1539, 1539, 1544, 1544, 1546, 1547, 1548,
    1548, 1548, 1549, 1549, 1549, 1549, 1561, 1561, 1561, 1561, 1561, 1563, 1563, 1563, 1565, 1566,
    1566, 1566, 1566, 1566, 1566, 1566, 1567, 1567, 1567, 1567, 1567, 1567, 1571, 1572, 1574, 1574,
    1575, 1575, 1575, 1575, 1576, 1576, 1576, 1579, 1579, 1579, 1579, 1579, 1579, 1597, 1597, 1598,
    1598, 1599, 1600, 1600, 1600, 1602, 1603, 1604, 1604, 1604, 1604, 1604, 1604, 1604, 1604, 1604,
    1604, 1604, 1604, 1604, 1608, 1608, 1610, 1610, 1610, 1610, 1611, 1611, 1611, 1612, 1612, 1612,
    1612, 1612, 1612, 1613, 1613, 1613, 1613, 1614, 1614, 1616, 1616, 1618, 1623, 1623, 1623, 1623,
    1625, 1625, 1625, 1625, 1625, 1625, 1627, 1627, 1627, 1627, 1627, 1627, 1627, 1627, 1628, 1628,
    1633, 1633, 1634, 1634, 1635, 1635, 1635, 1635, 1635, 1637, 1637, 1637, 1637, 1637, 1637, 1637,
    1638, 1638, 1638, 1638, 1638, 1638, 1638, 1639, 1641, 1648, 1649, 1649, 1649, 1649, 1651, 1651,
    1651, 1651, 1651, 1651, 1651, 1651, 1651, 1651, 1651, 1652, 1655, 1655, 1655, 1655, 1655, 1677,
    1677, 1677, 1677, 1677, 1681, 1682, 1682, 1682, 1682, 1686, 1692, 1692, 1692, 1706, 1706, 1706,
    1706, 1707, 1713, 1713, 1713, 1714, 1716, 1716, 1716, 1726, 1726, 1726, 1726, 1726, 1726, 1726,
    1726, 1726, 1726, 1726, 1737, 1750, 1750, 1751, 1753, 1753, 1753, 1753, 1753, 1753, 1754, 1754,
    1755, 1755, 1755, 1756, 1759, 1759, 1760, 1760, 1760, 1760, 1760, 1762, 1762, 1762, 1762, 1762,
    1762, 1762, 1762, 1769, 1770, 1770, 1770, 1770, 1770, 1770, 1770, 1771, 1771, 1771, 1771, 1771,
    1772, 1772, 1772, 1772, 1772, 1772, 1772, 1772, 1773, 1773, 1773, 1776, 1776, 1776, 1776, 1776,
    1776, 1776, 1776, 1777, 1779, 1782, 1799, 1799, 1799, 1800, 1800, 1801, 1811, 1811, 1811, 1811,
    1811, 1811, 1811, 1811, 1811, 1814, 1814, 1815, 1815, 1815, 1815, 1815, 1815, 1815, 1815, 1816,
    1818, 1819, 1822, 1823, 1823, 1828, 1830, 1830, 1831, 1832, 1832, 1833, 1833, 1833, 1833, 1834,
    1834, 1834, 1834, 1840, 1842, 1842, 1842, 1845, 1845, 1845, 2010, 2010, 2010, 2010, 2012, 2013,
    2013, 2013, 2013, 2013, 2013, 2013, 2013, 2013, 2013, 2013, 2013, 2013, 2014, 2014, 2014, 2014,
    2014, 2018, 2018, 2018, 2022, 2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023, 2023, 2025, 2025,
    2025, 2025, 2026, 2026, 2027, 2027, 2032, 2032, 2040, 2040, 2040, 2042, 2043, 2043, 2043, 2043,
    2043, 2045, 2045, 2046, 2046, 2046, 2047, 2047, 2047, 2047, 2047, 2049, 2049, 2049, 2052, 2052,
    2052, 2052, 2056, 2057, 2057, 2057, 2058, 2058, 2059, 2059, 2059, 2059, 2059, 2059, 2059, 2062,
    2063, 2063, 2065, 2065, 2065, 2065, 2065, 2066, 2068, 2070, 2070, 2070, 2070, 2072, 2072, 2072,
    2083, 2083, 2083, 2093, 2096, 2096, 2096, 2096, 2096, 2096, 2096, 2096, 2096, 2097, 2097, 2097,
    2097, 2097, 2097, 2097, 2097, 2097, 2097, 2097, 2097, 2099, 2101, 2103, 2103, 2103, 2103, 2103,
    2103, 2103, 2112, 2112, 2112, 2114, 2114, 2114, 2114, 2116, 2116, 2116, 2116, 2116, 2123, 2123,
    2125, 2126, 2126, 2126, 2144, 2144, 2144, 2146, 2147, 2147, 2147, 2147, 2148, 2151, 2152, 2152,
    2152, 2152, 2152, 2152, 2152, 2160, 2161, 2161, 2161, 2162, 2162, 2162, 2164, 2178, 2178, 2178,
    2178, 2178, 2181, 2181, 2181, 2182, 2182, 2182, 2186, 2186, 2186, 2186, 2186, 2186, 2186, 2186,
    2186, 2186, 2186, 2186, 2186, 2186, 2186, 2191, 2193, 2193, 2193, 2193, 2193, 2193, 2193, 2193,
    2193, 2193, 2193, 2193, 2193, 2193, 2193, 2193, 2193, 2194, 2194, 2194, 2194, 2194, 2194, 2194,
    2194, 2194, 2194, 2194, 2194, 2194, 2229, 2230, 2231, 2239, 2239, 2240, 2240, 2241, 2241, 2241,
    2243, 2243, 2243, 2243, 2243, 2243, 2243, 2244, 2244, 2250, 2251, 2252, 2252, 2252, 2252, 2252,
    2252, 2252, 2252, 2252, 2253, 2253, 2253, 2254, 2265, 2265, 2265, 2265, 2265, 2270, 2271, 2271,
    2271, 2276, 2283, 2283, 2283, 2283, 2285, 2285, 2285, 2286, 2286, 2286, 2286, 2286, 2286, 2286,
    2286, 2286, 2286, 2286, 2287, 2289, 2289, 2289, 2289, 2289, 2290, 2290, 2290, 2290, 2293, 2299,
    2299, 2299, 2300, 2300, 2300, 2300, 2300, 2300, 2300, 2300, 2301, 2301, 2301, 2303, 2307, 2315,
    2315, 2315, 2315, 2316, 2318, 2318, 2318, 2319, 2319, 2320, 2320, 2320, 2320, 2320, 2320, 2320,
    2320, 2320, 2321, 2321, 2321, 2321, 2321, 2321, 2322, 2322, 2322, 2322, 2322, 2322, 2322, 2322,
    2323, 2325, 2325, 2330, 2330, 2331, 2332, 2332, 2332, 2332, 2332, 2333, 2334, 2335, 2336, 2336,
    2341, 2341, 2341, 2341, 2341, 2341, 2341, 2341, 2341, 2341, 2341, 2341, 2341, 2341, 2341, 2341,
    2343, 2343, 2345, 2345, 2345, 2345, 2345, 2345, 2345, 2345, 2347, 2347, 2347, 2347, 2347, 2347,
    2347, 2347, 2347, 2347, 2347, 2348, 2348, 2348, 2348, 2352, 2352, 2352, 2352, 2353, 2353, 2353,
    2353, 2354, 2354, 2355, 2355, 2357, 2358, 2358, 2358, 2358, 2358, 2361, 2361, 2361, 2361, 2361,
    2361, 2361, 2361, 2361, 2367, 2367, 2367, 2368, 2369, 2370, 2370, 2370, 2370, 2370, 2370, 2370,
    2370, 2370, 2373, 2373, 2373, 2373, 2373, 2373, 2373, 2373, 2373, 2373, 2373, 2373, 2373, 2384,
    2385, 2385, 2385, 2385, 2387, 2387, 2387, 2387, 2387, 2387, 2387, 2387, 2387, 2387, 2387, 2387,
    2388, 2388, 2388, 2388, 2388, 2388, 2388, 2388, 2388, 2388, 2388, 2388, 2388, 2389, 2392, 2393,
    2393, 2402, 2402, 2402, 2402, 2402, 2403, 2404, 2404, 2404, 2404, 2404, 2404, 2404, 2405, 2405,
    2405, 2406, 2406, 2408, 2408, 2408, 2408, 2408, 2409, 2409, 2409, 2409, 2409, 2411, 2412, 2412,
    2412, 2412, 2412, 2413, 2413, 2413, 2414, 2414, 2414, 2415, 2416, 2418, 2418, 2418, 2418, 2419,
    2420, 2420, 2420, 2420, 2420, 2420, 2424, 2424, 2424, 2425, 2425, 2425, 2425, 2425, 2426, 2426,
    2426, 2426, 2426, 2427, 2427, 2429, 2430, 2430, 2430, 2430, 2430, 2430, 2431, 2431, 2431, 2431,
    2433, 2433, 2433, 2433, 2433, 2433, 2433, 2434, 2434, 2434, 2435, 2435, 2435, 2436, 2436, 2437,
    2437, 2437, 2437, 2437, 2437, 2437, 2437, 2437, 2438, 2438, 2438, 2438, 2438, 2457, 2459, 2459,
    2459, 2459, 2459, 2459, 2459, 2460, 2460, 2460, 2461, 2462, 2462, 2462, 2462, 2462, 2467, 2467,
    2467, 2469, 2469, 2469, 2469, 2470, 2470, 2470, 2470, 2470, 2470, 2470, 2471, 2471, 2471, 2471,
    2471, 2471, 2471, 2473, 2473, 2473, 2474, 2474, 2474, 2474, 2481, 2481, 2481, 2481, 2482, 2482,
    2484, 2485, 2485, 2485, 2485, 2485, 2485, 2485, 2485, 2485, 2488, 2488, 2488, 2488, 2488, 2489,
    2489, 2491, 2492, 2493, 2493, 2494, 2494, 2494, 2494, 2494, 2494, 2495, 2495, 2495, 2495, 2495,
    2495, 2505, 2506, 2506, 2506, 2513, 2513, 2513, 2513, 2513, 2513, 2514, 2514, 2514, 2514, 2514,
    2516, 2516, 2516, 2516, 2516, 2516, 2516, 2516, 2518, 2518, 2529, 2530, 2534, 2534, 2534, 2534,
    2535, 2535, 2535, 2535, 2535, 2535, 2542, 2543, 2543, 2543, 2543, 2543, 2543, 2543, 2543, 2543,
    2555, 2555, 2555, 2555, 2556, 2556, 2557, 2557, 2557, 2557, 2557, 2557, 2557, 2557, 2557, 2557,
    2558, 2558, 2558, 2558, 2558, 2558, 2561, 2561, 2561, 2561, 2564, 2564, 2564, 2564, 2565, 2565,
    2567, 2567, 2568, 2570, 2570, 2570, 2572, 2572, 2572, 2572, 2572, 2572, 2574, 2574, 2574, 2574,
    2574, 2574, 2574, 2574, 2574, 2574, 2574, 2574, 2574, 2574, 2574, 2574, 2574, 2574, 2576, 2577,
    2577, 2577, 2582, 2582, 2582, 2582, 2582, 2582, 2584, 2584, 2584, 2585, 2586, 2586, 2586, 2586,
    2591, 2591, 2591, 2591, 2591, 2591, 2591, 2591, 2591, 2591, 2591, 2591, 2593, 2595, 2595, 2595,
    2595, 2597, 2597, 2597, 2598, 2598, 2599, 2599, 2599, 2600, 2600, 2600, 2600, 2600, 2600, 2600,
    2600, 2604, 2604, 2604, 2607, 2607, 2607, 2607, 2607, 2607, 2607, 2607, 2607, 2607, 2607, 2609,
    2609, 2609, 2609, 2610, 2610, 2610, 2611, 2612, 2623, 2623, 2634, 2638, 2638, 2638, 2639, 2641,
    2641, 2642, 2642, 2642, 2642, 2646, 2646, 2646, 2646, 2646, 2646, 2646, 2646, 2646, 2646, 2646,
    2646, 2648, 2651, 2651, 2651, 2651, 2651, 2651, 2651, 2651, 2651, 2651, 2652, 2652, 2652, 2652,
    2652, 2660, 2660, 2660, 2665, 2665, 2665, 2665, 2665, 2666, 2668, 2668, 2668, 2668, 2668, 2668,
    2668, 2668, 2672, 2672, 2672, 2672, 2673, 2673, 2673, 2673, 2673, 2673, 2673, 2674, 2674, 2674,
    2674, 2674, 2677, 2678, 2678, 2678, 2678, 2679, 2679, 2679, 2681, 2681, 2681, 2681, 2681, 2700,
    2700, 2700, 2700, 2700, 2700, 2700, 2704, 2712, 2712, 2712, 2712, 2712, 2712, 2714, 2714, 2725,
    2728, 2728, 2728, 2728, 2728, 2728, 2728, 2728, 2728, 2737, 2737, 2737, 2737, 2737, 2739, 2753,
    2753, 2753, 2753, 2754, 2754, 2754, 2754, 2755, 2755, 2765, 2770, 2770, 2770, 2770, 2770, 2770,
    2799, 2799, 2799, 2799, 2799, 2799, 2799, 2799, 2833, 2833, 2835, 2835, 2835, 2835, 2835, 2835,
    2835, 2835, 2835, 2835, 2835, 2835, 2835, 2836, 2847, 2847, 2848, 2849, 2849, 2850, 2850, 2850,
    2850, 2850, 2856, 2856, 2856, 2857, 2859, 2859, 2859, 2859, 2871, 2871, 2871, 2871, 2871, 2871,
    2873, 2873, 2873, 2873, 2873, 2873, 2873, 2873, 2873, 2873, 2873, 2875, 2877, 2877, 2880, 2886,
    2886, 2886, 2886, 2886, 2886, 2886, 2886, 2888, 2888, 2888, 2888, 2888, 2888, 2889, 2889, 2891,
    2891, 2892, 2892, 2898, 2898, 2898, 2898, 2900, 2900, 2901, 2902, 2902, 2902, 2902, 2902, 2903,
    2903, 2903, 2904, 2917, 2919, 2925, 2925, 2925, 2926, 2926, 2926, 2926, 2926, 2928, 2929, 2929,
    2929, 2929, 2929, 2929, 2929, 2929, 2929, 2929, 2929, 2930, 2930, 2930, 2930, 2930, 2931, 2931,
    2931, 2932, 2932, 2933, 2933, 2934, 2936, 2937, 2937, 2937, 2939, 2941, 2941, 2947, 2947, 2948,
    2949, 2949, 2949, 2949, 2950, 2950, 2950, 2950, 2950, 2950, 2950, 2950, 2950, 2950, 2950, 2951,
    2952, 2953, 2953, 2953, 2953, 2953, 2953, 2953, 2953, 2955, 2955, 2955, 2955, 2956, 2956, 2956,
    2956, 2972, 2973, 2973, 2973, 2973, 2974, 2975, 2975, 2975, 2975, 2978, 2978, 2978, 2979, 2980,
    2980, 2980, 2980, 2980, 2980, 2981, 2981, 2981, 2985, 2985, 2985, 2985, 2996, 2996, 2996, 2996,
    2996, 2996, 3002, 3002, 3005, 3005, 3005, 3005, 3005, 3007, 3008, 3008, 3008, 3008, 3008, 3009,
    3009, 3009, 3009, 3011, 3015, 3016, 3016, 3017, 3017, 3018, 3018, 3018, 3018, 3020, 3020, 3020,
    3030, 3030, 3030, 3030, 3030, 3030, 3030, 3031, 3031, 3031, 3037, 3039, 3039, 3040, 3041, 3041,
    3041, 3041, 3041, 3041, 3041, 3041, 3041, 3041, 3048, 3048, 3048, 3048, 3048, 3048, 3048, 3048,
    3048, 3049, 3049, 3053, 3053, 3064, 3064, 3064, 3077, 3077, 3077, 3077, 3077, 3078, 3078, 3078,
    3078, 3078, 3078, 3080, 3082, 3082, 3083, 3083, 3083, 3083, 3085, 3085, 3085, 3085, 3085, 3085,
    3085, 3085, 3085, 3086, 3086, 3086, 3086, 3086, 3087, 3090, 3092, 3092, 3092, 3092, 3092, 3092,
    3092, 3092, 3101, 3101, 3101, 3101, 3101, 3101, 3102, 3103, 3103, 3104, 3104, 3110, 3110, 3110,
    3110, 3110, 3111, 3111, 3113, 3114, 3115, 3116, 3120, 3121, 3121, 3121, 3122, 3123, 3123, 3123,
    3123, 3123, 3123, 3123, 3123, 3123, 3126, 3126, 3126, 3126, 3127, 3127, 3128, 3128, 3130, 3130,
    3130, 3130, 3130, 3131, 3131, 3141, 3141, 3142, 3142, 3142, 3142, 3142, 3142, 3142, 3142, 3142,
    3142, 3142, 3142, 3142, 3161, 3161, 3161, 3161, 3161, 3161, 3161, 3163, 3163, 3163, 3163, 3163,
    3164, 3164, 3172, 3172, 3172, 3172, 3172, 3173, 3173, 3173, 3173, 3173, 3173, 3180, 3180, 3180,
    3180, 3181, 3181, 3181, 3181, 3181, 3181, 3183, 3183, 3183, 3190, 3190, 3190, 3190, 3190, 3190,
    3190, 3192, 3193, 3195, 3198, 3200, 3200, 3200, 3201, 3201, 3201, 3201, 3201, 3215, 3217, 3217,
    3217, 3217, 3217, 3217, 3217, 3217, 3217, 3217, 3217, 3217, 3217, 3217, 3217, 3217, 3217, 3217,
    3217, 3218, 3218, 3218, 3218, 3218, 3218, 3219, 3219, 3219, 3219, 3219, 3219, 3219, 3219, 3220,
    3220, 3221, 3221, 3221, 3221, 3221, 3224, 3224, 3224, 3225, 3225, 3225, 3225, 3225, 3225, 3232,
    3232, 3232, 3232, 3232, 3232, 3235, 3236, 3236, 3237, 3237, 3237, 3238, 3238, 3238, 3238, 3238,
    3238, 3238, 3238, 3238, 3239, 3239, 3239, 3240, 3244, 3244, 3249, 3249, 3261, 3261, 3261, 3261,
    3261, 3261, 3261, 3261, 3264, 3264, 3264, 3264, 3264, 3264, 3264, 3264, 3264, 3265, 3265, 3265,
    3266, 3269, 3269, 3269, 3269, 3269, 3269, 3270, 3270, 3270, 3270, 3270, 3270, 3270, 3273, 3273,
    3273, 3276, 3276, 3277, 3277, 3277, 3277, 3277, 3277, 3277, 3282, 3284, 3285, 3285, 3285, 3286,
    3287, 3289, 3289, 3289, 3289, 3289, 3289, 3295, 3296, 3296, 3296, 3296, 3296, 3296, 3296, 3296,
    3296, 3296, 3296, 3297, 3297, 3297, 3297, 3297, 3297, 3297, 3297, 3297, 3298, 3298, 3298, 3299,
    3299, 3299, 3299, 3302, 3302, 3302, 3303, 3303, 3303, 3323, 3323, 3323, 3323, 3323, 3328, 3329,
    3329, 3329, 3329, 3329, 3329, 3329, 3329, 3329, 3332, 3333, 3341, 3341, 3341, 3341, 3341, 3341,
    3341, 3341, 3341, 3341, 3341, 3341, 3341, 3341, 3341, 3341, 3341, 3341, 3341, 3341, 3341, 3341,
    3341, 3341, 3341, 3341, 3341, 3346, 3346, 3346, 3346, 3350, 3350, 3350, 3350, 3350, 3352, 3352,
    3352, 3352, 3352, 3352, 3354, 3354, 3354, 3354, 3354, 3354, 3356, 3357, 3357, 3358, 3358, 3358,
    3358, 3358, 3358, 3358, 3358, 3360, 3360, 3361, 3361, 3361, 3361, 3361, 3362, 3380, 3380, 3380,
    3380, 3380, 3382, 3382, 3385, 3385, 3385, 3387, 3387, 3387, 3390, 3390, 3390, 3390, 3390, 3390,
    3392, 3392, 3392, 3392, 3392, 3392, 3392, 3392, 3394, 3394, 3395, 3395, 3395, 3395, 3395, 3395,
    3395, 3398, 3398, 3398, 3398, 3398, 3398, 3399, 3399, 3400, 3400, 3401, 3401, 3401, 3401, 3401,
    3401, 3401, 3401, 3404, 3404, 3405, 3405, 3409, 3409, 3410, 3410, 3410, 3410, 3410, 3410, 3411,
    3411, 3411, 3412, 3412, 3414, 3414, 3414, 3414, 3416, 3416, 3416, 3417, 3417, 3417, 3417, 3417,
    3417, 3424, 3427, 3427, 3429, 3429, 3430, 3430, 3430, 3430, 3430, 3430, 3430, 3432, 3433, 3433,
    3433, 3433, 3433, 3434, 3434, 3434, 3435, 3435, 3435, 3435, 3435, 3435, 3440, 3441, 3441, 3441,
    3445, 3445, 3445, 3445, 3445, 3446, 3446, 3446, 3447, 3452, 3452, 3453, 3453, 3453, 3456, 3458,
    3458, 3458, 3460, 3460, 3461, 3461, 3461, 3463, 3463, 3463, 3463, 3463, 3463, 3463, 3464, 3469,
    3471, 3471, 3471, 3472, 3472, 3472, 3473, 3474, 3475, 3475, 3475, 3475, 3475, 3475, 3475, 3475,
    3475, 3475, 3478, 3478, 3478, 3481, 3481, 3481, 3481, 3483, 3483, 3483, 3483, 3484, 3485, 3485,
    3485, 3485, 3485, 3485, 3485, 3485, 3486, 3486, 3486, 3486, 3486, 3486, 3486, 3486, 3486, 3487,
    3487, 3487, 3487, 3487, 3487, 3487, 3487, 3487, 3487, 3487, 3501, 3501, 3501, 3503, 3503, 3503,
    3503, 3503, 3503, 3505, 3505, 3505, 3505, 3505, 3505, 3505, 3505, 3506, 3506, 3508, 3508, 3509,
    3511, 3511, 3511, 3511, 3511, 3511, 3511, 3511, 3521, 3522, 3522, 3522, 3522, 3523, 3523, 3523,
    3523, 3537, 3537, 3547, 3547, 3547, 3547, 3547, 3547, 3548, 3548, 3566, 3566, 3566, 3566, 3567,
    3567, 3567, 3567, 3569, 3572, 3572, 3572, 3572, 3572, 3572, 3572, 3572, 3572, 3573, 3574, 3577,
    3577, 3577, 3577, 3577, 3577, 3584, 3585, 3585, 3585, 3585, 3588, 3588, 3589, 3589, 3590, 3590,
    3590, 3590, 3591, 3591, 3592, 3592, 3592, 3592, 3592, 3592, 3592, 3592, 3592, 3598, 3598, 3598,
    3598, 3598, 3598, 3599, 3599, 3602, 3602, 3602, 3602, 3602, 3602, 3602, 3602, 3603, 3603, 3603,
    3604, 3604, 3604, 3604, 3604, 3604, 3604, 3604, 3612, 3612, 3612, 3612, 3612, 3612, 3612, 3612,
    3612, 3612, 3612, 3612, 3612, 3616, 3616, 3616, 3618, 3618, 3618, 3618, 3618, 3618, 3618, 3618,
    3618, 3626, 3626, 3626, 3626, 3626, 3626, 3629, 3630, 3630, 3632, 3635, 3635, 3635, 3635, 3635,
    3650, 3650, 3650, 3650, 3650, 3650, 3650, 3650, 3656, 3656, 3658, 3658, 3658, 3658, 3658, 3658,
    3661, 3662, 3665, 3665, 3665, 3665, 3665, 3665, 3665, 3665, 3666, 3666, 3666, 3666, 3666, 3666,
    3666, 3666, 3666, 3666, 3666, 3666, 3671, 3672, 3674, 3674, 3674, 3674, 3674, 3674, 3675, 3676,
    3676, 3676, 3676, 3677, 3677, 3677, 3677, 3677, 3677, 3677, 3677, 3677, 3693, 3693, 3693, 3693,
    3693, 3693, 3693, 3693, 3693, 3693, 3693, 3693, 3693, 3697, 3698, 3698, 3698, 3698, 3698, 3698,
    3698, 3704, 3705, 3705, 3705, 3706, 3706, 3706, 3706, 3706, 3712, 3712, 3712, 3712, 3712, 3712,
    3740, 3740, 3740, 3740, 3740, 3740, 3740, 3740, 3740, 3741, 3741, 3741, 3745, 3757, 3757, 3757,
    3758, 3758, 3761, 3762, 3762, 3762, 3762, 3762, 3762, 3762, 3762, 3762, 3762, 3762, 3762, 3763,
    3763, 3764, 3764, 3765, 3765, 3765, 3765, 3769, 3769, 3770, 3770, 3771, 3771, 3771, 3771, 3771,
    3771, 3771, 3772, 3772, 3773, 3773, 3773, 3774, 3774, 3774, 3774, 3775, 3775, 3775, 3775, 3775,
    3776, 3776, 3776, 3776, 3776, 3776, 3777, 3777, 3777, 3777, 3777, 3777, 3777, 3777, 3777, 3777,
    3777, 3778, 3778, 3778, 3778, 3779, 3779, 3779, 3779, 3781, 3781, 3781, 3781, 3794, 3796, 3796,
    3796, 3796, 3796, 3796, 3796, 3796, 3796, 3796, 3797, 3797, 3797, 3799, 3807, 3812, 3821, 3822,
    3822, 3822, 3822, 3823, 3824, 3824, 3824, 3825, 3825, 3825, 3828, 3828, 3828, 3829, 3829, 3831,
    3831, 3831, 3831, 3831, 3831, 3831, 3832, 3832, 3832, 3833, 3833, 3833, 3833, 3833, 3833, 3833,
    3833, 3833, 3833, 3833, 3834, 3834, 3834, 3834, 3834, 3834, 3834, 3836, 3836, 3836, 3840, 3840,
    3854, 3854, 3856, 3856, 3856, 3857, 3857, 3857, 3857, 3857, 3857, 3859, 3859, 3859, 3859, 3859,
    3859, 3860, 3860, 3861, 3862, 3862, 3863, 3863, 3863, 3863, 3863, 3863, 3863, 3863, 3863, 3863,
    3864, 3864, 3864, 3864, 3864, 3864, 3864, 3864, 3864, 3864, 3864, 3864, 3864, 3869, 3869, 3870,
    3871, 3871, 3871, 3871, 3871, 3871, 3871, 3871, 3872, 3872, 3872, 3872, 3872, 3872, 3874, 3878,
    3878, 3879, 3880, 3880, 3881, 3882, 3882, 3882, 3882, 3882, 3882, 3882, 3882, 3882, 3882, 3882,
    3882, 3882, 3882, 3883, 3883, 3883, 3883, 3885, 3885, 3885, 3885, 3889, 3889, 3902, 3903, 3903,
    3903, 3908, 3909, 3909, 3909, 3910, 3910, 3910, 3910, 3915, 3917, 3917, 3917, 3917, 3917, 3917,
    3917, 3918, 3918, 3919, 3919, 3919, 3921, 3921, 3922, 3922, 3923, 3924, 3924, 3929, 3929, 3929,
    3929, 3929, 3929, 3929, 3929, 3929, 3935, 3936, 3936, 3936, 3936, 3938, 3938, 3939, 3939, 3939,
    3939, 3942, 3942, 3942, 3942, 3942, 3946, 3947, 3947, 3947, 3947, 3950, 3950, 3950, 3950, 3950,
    3950, 3950, 3950, 3950, 3950, 3950, 3950, 3950, 3950, 3950, 3952, 3952, 3952, 3953, 3954, 3963,
    3963, 3963, 3963, 3963, 3964, 3964, 3964, 3965, 3965, 3965, 3965, 3967, 3967, 3969, 3969, 3969,
    3981, 3981, 3981, 3981, 3981, 3982, 3985, 3985, 3985, 3987, 3988, 3988, 3989, 3989, 3989, 3990,
    3992, 3992, 3993, 3993, 3993, 3993, 3993, 3994, 3994, 3994, 3994, 3994, 3997, 3997, 3997, 3997,
    3997, 3997, 3997, 3997, 3997, 3997, 3997, 3997, 3998, 3998, 3998, 3999, 3999, 4000, 4000, 4000,
    4001, 4001, 4001, 4001, 4002, 4002, 4002, 4002, 4002, 4003, 4003, 4003, 4003, 4006, 4006, 4006,
    4008, 4008, 4008, 4008, 4008, 4008, 4008, 4008, 4008, 4008, 4008, 4008, 4009, 4009, 4010, 4012,
    4012, 4012, 4012, 4019, 4019, 4019, 4019, 4019, 4023, 4023, 4023, 4023, 4023, 4023, 4024, 4035,
    4040, 4041, 4193, 4196, 4197, 4197, 4197, 4197, 4197, 4197, 4197, 4197, 4197, 4197, 4197, 4197,
    4197, 4197, 4198, 4198, 4198, 4198, 4198, 4198, 4200, 4201, 4201, 4201, 4201, 4201, 4203, 4203,
    4203, 4203, 4206, 4207, 4209, 4209, 4210, 4210, 4210, 4210, 4210, 4210, 4210, 4210, 4210, 4210,
    4210, 4211, 4211, 4211, 4211, 4211, 4211, 4211, 4211, 4211, 4213, 4213, 4213, 4213, 4213, 4214,
    4217, 4217, 4217, 4217, 4218, 4218, 4218, 4218, 4218, 4218, 4220, 4220, 4222, 4222, 4222, 4226,
    4228, 4228, 4230, 4231, 4231, 4231, 4231, 4231, 4231, 4238, 4238, 4238, 4238, 4241, 4241, 4243,
    4243, 4243, 4245, 4245, 4245, 4245, 4250, 4251, 4251, 4251, 4251, 4251, 4251, 4251, 4253, 4254,
    4254, 4254, 4254, 4254, 4254, 4255, 4255, 4257, 4259, 4259, 4259, 4259, 4259, 4259, 4259, 4267,
    4268, 4268, 4269, 4274, 4274, 4274, 4291, 4291, 4292, 4292, 4292, 4292, 4292, 4292, 4293, 4293,
    4293, 4293, 4293, 4295, 4296, 4296, 4296, 4297, 4297, 4297, 4297, 4297, 4309, 4309, 4309, 4310,
    4310, 4310, 4311, 4311, 4311, 4312, 4312, 4312, 4312, 4312, 4312, 4313, 4313, 4313, 4313, 4313,
    4313, 4314, 4314, 4314, 4314, 4314, 4314, 4314, 4317, 4317, 4317, 4317, 4317, 4320, 4322, 4322,
    4322, 4323, 4323, 4324, 4327, 4328, 4328, 4328, 4335, 4336, 4336, 4336, 4336, 4341, 4341, 4341,
    4341, 4341, 4341, 4342, 4342, 4342, 4342, 4342, 4345, 4345, 4345, 4348, 4361, 4361, 4361, 4361,
    4362, 4364, 4364, 4364, 4364, 4364, 4364, 4364, 4364, 4364, 4364, 4364, 4364, 4364, 4364, 4364,
    4364, 4366, 4372, 4372, 4372, 4373, 4373, 4374, 4374, 4374, 4374, 4374, 4374, 4374, 4374, 4376,
    4377, 4377, 4377, 4378, 4379, 4379, 4379, 4381, 4381, 4381, 4381, 4381, 4381, 4381, 4381, 4381,
    4381, 4381, 4385, 4388, 4388, 4388, 4391, 4391, 4391, 4391, 4392, 4402, 4402, 4402, 4402, 4402,
    4402, 4402, 4404, 4404, 4404, 4404, 4406, 4407, 4407, 4407, 4409, 4409, 4409, 4409, 4409, 4409,
    4409, 4409, 4411, 4411, 4411, 4411, 4412, 4412, 4412, 4412, 4415, 4415, 4415, 4415, 4415, 4415,
    4415, 4418, 4418, 4418, 4418, 4420, 4423, 4427, 4428, 4432, 4432, 4432, 4432, 4434, 4434, 4434,
    4434, 4435, 4435, 4437, 4437, 4437, 4437, 4437, 4438, 4438, 4438, 4438, 4438, 4438, 4438, 4438,
    4438, 4439, 4440, 4442, 4443, 4445, 4445, 4445, 4445, 4446, 4446, 4449, 4449, 4450, 4450, 4450,
    4451, 4451, 4468, 4468, 4468, 4468, 4469, 4470, 4470, 4471, 4473, 4473, 4473, 4473, 4473, 4473,
    4473, 4473, 4473, 4473, 4486, 4487, 4488, 4489, 4494, 4494, 4494, 4494, 4494, 4494, 4496, 4499,
    4499, 4499, 4501, 4501, 4501, 4501, 4501, 4501, 4501, 4501, 4501, 4501, 4501, 4501, 4501, 4501,
    4501, 4501, 4502, 4502, 4502, 4503, 4503, 4503, 4504, 4504, 4504, 4505, 4505, 4505, 4514, 4514,
    4514, 4514, 4514, 4514, 4517, 4517, 4517, 4523, 4523, 4523, 4523, 4526, 4526, 4527, 4529, 4531,
    4533, 4533, 4533, 4535, 4535, 4535, 4535, 4535, 4535, 4535, 4535, 4539, 4539, 4539, 4550, 4550,
    4550, 4550, 4552, 4552, 4552, 4552, 4552, 4552, 4552, 4552, 4552, 4553, 4553, 4553, 4553, 4553,
    4554, 4554, 4554, 4554, 4555, 4555, 4555, 4555, 4556, 4559, 4559, 4559, 4559, 4559, 4559, 4559,
    4559, 4559, 4559, 4559, 4559, 4559, 4559, 4559, 4559, 4559, 4559, 4559, 4559, 4559, 4559, 4559,
    4560, 4560, 4560, 4560, 4560, 4560, 4561, 4562, 4562, 4562, 4562, 4562, 4563, 4563, 4565, 4565,
    4565, 4565, 4565, 4565, 4566, 4566, 4566, 4566, 4566, 4566, 4566, 4566, 4566, 4576, 4576, 4576,
    4576, 4577, 4577, 4577, 4578, 4578, 4578, 4578, 4578, 4578, 4578, 4578, 4578, 4578, 4578, 4578,
    4580, 4580, 4580, 4582, 4582, 4582, 4582, 4585, 4585, 4585, 4585, 4585, 4585, 4585, 4585, 4585,
    4585, 4585, 4585, 4588, 4588, 4594, 4594, 4594, 4594, 4596, 4597, 4597, 4597, 4598, 4598, 4598,
    4598, 4598, 4598, 4598, 4598, 4598, 4598, 4598, 4599, 4599, 4600, 4600, 4600, 4600, 4600, 4604,
    4605, 4605, 4605, 4606, 4607, 4609, 4609, 4628, 4628, 4630, 4630, 4630, 4631, 4648, 4648, 4648,
    4648, 4651, 4652, 4652, 4652, 4654, 4654, 4655, 4655, 4655, 4655, 4655, 4655, 4655, 4655, 4655,
    4655, 4656, 4656, 4658, 4658, 4658, 4658, 4658, 4658, 4658, 4658, 4658, 4658, 4668, 4668, 4673,
    4673, 4673, 4673, 4673, 4673, 4673, 4673, 4673, 4673, 4674, 4674, 4674, 4674, 4674, 4676, 4676,
    4676, 4676, 4676, 4676, 4676, 4676, 4676, 4677, 4677, 4677, 4687, 4688, 4688, 4689, 4690, 4690,
    4693, 4693, 4693, 4693, 4693, 4693, 4695, 4695, 4695, 4695, 4697, 4697, 4697, 4698, 4698, 4708,
    4708, 4708, 4708, 4708, 4708, 4708, 4708, 4710, 4710, 4711, 4711, 4711, 4711, 4715, 4715, 4715,
    4715, 4715, 4715, 4715, 4715, 4715, 4715, 4716, 4716, 4717, 4718, 4718, 4718, 4718, 4718, 4718,
    4719, 4719, 4719, 4719, 4719, 4720, 4720, 4721, 4722, 4722, 4722, 4723, 4723, 4723, 4724, 4724,
    4724, 4724, 4724, 4726, 4727, 4727, 4727, 4727, 4727, 4729, 4729, 4729, 4729, 4732, 4733, 4739,
    4739, 4739, 4740, 4740, 4740, 4742, 4742, 4745, 4745, 4745, 4745, 4745, 4746, 4746, 4746, 4746,
    4746, 4746, 4746, 4747, 4749, 4749, 4749, 4749, 4754, 4754, 4754, 4754, 4754, 4754, 4754, 4755,
    4755, 4755, 4755, 4755, 4757, 4757, 4757, 4757, 4765, 4765, 4767, 4767, 4767, 4767, 4767, 4772,
    4772, 4773, 4773, 4773, 4773, 4773, 4775, 4777, 4777, 4777, 4777, 4778, 4778, 4778, 4778, 4778,
    4779, 4780, 4780, 4780, 4781, 4781, 4781, 4782, 4782, 4782, 4783, 4783, 4785, 4785, 4785, 4785,
    4786, 4786, 4786, 4786, 4786, 4786, 4786, 4786, 4787, 4787, 4788, 4788, 4788, 4788, 4788, 4788,
    4788, 4788, 4788, 4788, 4790, 4790, 4790, 4790, 4790, 4790, 4790, 4790, 4790, 4790, 4790, 4790,
    4793, 4793, 4793, 4793, 4794, 4794, 4794, 4798, 4798, 4798, 4798, 4798, 4798, 4800, 4800, 4800,
    4801, 4801, 4801, 4803, 4803, 4803, 4803, 4803, 4803, 4803, 4804, 4804, 4804, 4805, 4806, 4806,
    4806, 4809, 4809, 4812, 4812, 4819, 4819, 4819, 4819, 4819, 4819, 4819, 4820, 4820, 4820, 4820,
    4820, 4820, 4820, 4820, 4820, 4820, 4820, 4820, 4822, 4822, 4822, 4822, 4823, 4823, 4823, 4823,
    4823, 4823, 4823, 4823, 4824, 4824, 4824, 4824, 4826, 4826, 4826, 4832, 4832, 4832, 4833, 4833,
    4833, 4833, 4833, 4835, 4835, 4835, 4836, 4836, 4836, 4836, 4836, 4837, 4837, 4837, 4837, 4837,
    4837, 4837, 4837, 4837, 4837, 4839, 4839, 4839, 4840, 4840, 4902, 4902, 4902, 4902, 4903, 4908,
    4908, 4910, 4910, 4910, 4912, 4912, 4912, 4913, 4934, 4934, 4934, 4938, 4938, 4938, 4939, 4939,
    4939, 4940, 4940, 4940, 4940, 4940, 4940, 4941, 4941, 4941, 4941, 4941, 4941, 4941, 4941, 4956,
    4956, 4957, 4958, 4958, 4959, 4959, 4960, 4971, 4971, 4971, 4971, 4971, 4971, 4971, 4971, 4971,
    4971, 4971, 4971, 4973, 4974, 4974, 4974, 4981, 4982, 4982, 4982, 4982, 4995, 4996, 4996, 4996,
    4997, 4997, 4997, 4997, 4997, 4998, 5004, 5004, 5005, 5005, 5005, 5005, 5005, 5006, 5007, 5007,
    5007, 5008, 5008, 5008, 5008, 5008, 5008, 5008, 5008, 5008, 5008, 5008, 5009, 5009, 5009, 5009,
    5009, 5009, 5009, 5009, 5009, 5009, 5013, 5014, 5022, 5022, 5022, 5022, 5022, 5022, 5022, 5034,
    5034, 5035, 5035, 5035, 5036, 5036, 5036, 5036, 5040, 5040, 5040, 5040, 5040, 5040, 5040, 5042,
    5042, 5043, 5043, 5043, 5043, 5043, 5043, 5043, 5044, 5045, 5045, 5045, 5045, 5045, 5046, 5046,
    5048, 5048, 5048, 5048, 5048, 5049, 5055, 5056, 5056, 5056, 5056, 5056, 5057, 5057, 5057, 5057,
    5057, 5057, 5058, 5058, 5058, 5058, 5058, 5058, 5058, 5058, 5058, 5058, 5058, 5058, 5058, 5058,
    5058, 5059, 5061, 5061, 5061, 5061, 5062, 5062, 5062, 5062, 5062, 5062, 5062, 5063, 5063, 5063,
    5063, 5063, 5063, 5064, 5064, 5065, 5065, 5065, 5065, 5065, 5065, 5065, 5068, 5068, 5073, 5073,
    5073, 5073, 5074, 5077, 5077, 5077, 5077, 5077, 5077, 5077, 5080, 5082, 5082, 5082, 5082, 5086,
    5088, 5088, 5088, 5088, 5090, 5090, 5090, 5090, 5090, 5090, 5090, 5091, 5091, 5091, 5091, 5091,
    5092, 5092, 5092, 5092, 5092, 5093, 5093, 5093, 5093, 5094, 5095, 5095, 5096, 5098, 5099, 5099,
    5099, 5099, 5099, 5099, 5099, 5099, 5099, 5099, 5102, 5102, 5102, 5102, 5102, 5102, 5103, 5103,
    5104, 5104, 5106, 5106, 5106, 5106, 5109, 5111, 5111, 5115, 5115, 5115, 5115, 5115, 5124, 5124,
    5124, 5140, 5141, 5141, 5141, 5141, 5142, 5143, 5143, 5143, 5144, 5144, 5144, 5144, 5144, 5144,
    5144, 5145, 5145, 5145, 5145, 5145, 5146, 5147, 5147, 5154, 5156, 5160, 5163, 5173, 5173, 5176,
    5176, 5177, 5177, 5177, 5177, 5177, 5177, 5177, 5177, 5178, 5178, 5178, 5178, 5178, 5178, 5179,
    5180, 5181, 5182, 5182, 5182, 5184, 5184, 5186, 5186, 5186, 5186, 5189, 5189, 5189, 5190, 5190,
    5191, 5191, 5191, 5191, 5191, 5191, 5191, 5191, 5191, 5196, 5196, 5196, 5196, 5196, 5196, 5197,
    5197, 5197, 5202, 5202, 5206, 5206, 5206, 5206, 5206, 5206, 5207, 5207, 5208, 5208, 5208, 5208,
    5217, 5217, 5217, 5217, 5219, 5223, 5223, 5228, 5228, 5228, 5228, 5230, 5230, 5230, 5230, 5230,
    5230, 5230, 5231, 5233, 5233, 5235, 5236, 5237, 5238, 5238, 5238, 5241, 5245, 5245, 5245, 5245,
    5245, 5245, 5245, 5245, 5245, 5245, 5245, 5245, 5245, 5245, 5247, 5254, 5254, 5257, 5257, 5258,
    5258, 5259, 5263, 5263, 5263, 5263, 5263, 5263, 5264, 5264, 5265, 5268, 5268, 5268, 5270, 5270,
    5270, 5272, 5272, 5273, 5273, 5273, 5276, 5281, 5281, 5281, 5281, 5283, 5283, 5286, 5286, 5286,
    5291, 5291, 5291, 5291, 5291, 5291, 5291, 5291, 5291, 5291, 5292, 5292, 5292, 5293, 5293, 5300,
    5300, 5301, 5301, 5301, 5301, 5301, 5301, 5302, 5302, 5302, 5303, 5303, 5303, 5303, 5303, 5303,
    5303, 5303, 5303, 5303, 5303, 5303, 5309, 5313, 5313, 5313, 5313, 5313, 5313, 5313, 5319, 5319,
    5320, 5320, 5320, 5321, 5322, 5323, 5323, 5323, 5323, 5323, 5324, 5324, 5324, 5337, 5338, 5338,
    5338, 5339, 5339, 5339, 5339, 5340, 5341, 5344, 5344, 5344, 5344, 5344, 5344, 5344, 5344, 5344,
    5370, 5370, 5370, 5370, 5371, 5371, 5371, 5371, 5371, 5372, 5372, 5372, 5372, 5372, 5372, 5373,
    5373, 5373, 5373, 5373, 5374, 5374, 5375, 5375, 5375, 5375, 5375, 5375, 5375, 5376, 5376, 5376,
    5376, 5376, 5376, 5376, 5376, 5381, 5381, 5381, 5383, 5384, 5384, 5384, 5387, 5388, 5388, 5388,
    5388, 5394, 5395, 5395, 5395, 5405, 5405, 5405, 5405, 5405, 5405, 5413, 5414, 5414, 5414, 5414,
    5414, 5414, 5414, 5414, 5414, 5414, 5414, 5414, 5422, 5422, 5422, 5423, 5425, 5425, 5425, 5425,
    5429, 5429, 5430, 5430, 5430, 5430, 5430, 5430, 5430, 5430, 5430, 5430, 5431, 5431, 5431, 5431,
    5431, 5431, 5431, 5431, 5431, 5431, 5431, 5431, 5431, 5431, 5431, 5431, 5431, 5431, 5431, 5431,
    5455, 5455, 5455, 5455, 5456, 5456, 5456, 5466, 5466, 5466, 5467, 5467, 5468, 5468, 5468, 5468,
    5472, 5472, 5472, 5473, 5473, 5473, 5473, 5473, 5473, 5473, 5473, 5473, 5474, 5474, 5475, 5475,
    5475, 5475, 5475, 5475, 5475, 5475, 5475, 5475, 5475, 5475, 5475, 5475, 5475, 5476, 5476, 5476,
    5476, 5476, 5476, 5478, 5478, 5478, 5478, 5478, 5478, 5478, 5485, 5488, 5488, 5488, 5488, 5488,
    5488, 5488, 5488, 5488, 5488, 5488, 5490, 5490, 5490, 5490, 5490, 5490, 5490, 5490, 5490, 5495,
    5495, 5495, 5495, 5495, 5495, 5496, 5496, 5496, 5497, 5506, 5506, 5506, 5506, 5506, 5506, 5506,
    5506, 5506, 5506, 5506, 5507, 5507, 5507, 5507, 5507, 5507, 5507, 5507, 5508, 5509, 5510, 5514,
    5516, 5516, 5517, 5517, 5517, 5547, 5547, 5548, 5548, 5548, 5548, 5550, 5550, 5550, 5550, 5550,
    5550, 5552, 5552, 5552, 5561, 5561, 5561, 5561, 5565, 5565, 5565, 5565, 5565, 5565, 5565, 5565,
    5596, 5596, 5608, 5610, 5610, 5614, 5614, 5614, 5614, 5614, 5614, 5614, 5614, 5614, 5614, 5614,
    5614, 5614, 5614, 5614, 5614, 5614, 5614, 5615, 5615, 5615, 5615, 5615, 5616, 5619, 5619, 5622,
    5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5627, 5627, 5631,
    5631, 5631, 5631, 5632, 5632, 5632, 5632, 5632, 5632, 5632, 5643, 5643, 5644, 5644, 5644, 5644,
    5644, 5644, 5645, 5645, 5645, 5645, 5645, 5645, 5645, 5645, 5645, 5653, 5655, 5655, 5655, 5656,
    5656, 5656, 5656, 5656, 5656, 5656, 5656, 5656, 5656, 5657, 5657, 5657, 5657, 5658, 5658, 5662,
    5663, 5663, 5663, 5663, 5663, 5663, 5663, 5663, 5663, 5663, 5663, 5664, 5664, 5664, 5664, 5665,
    5675, 5678, 5678, 5678, 5678, 5679, 5679, 5681, 5681, 5681, 5681, 5683, 5683, 5683, 5683, 5683,
    5683, 5683, 5683, 5683, 5683, 5683, 5685, 5685, 5685, 5685, 5685, 5688, 5688, 5688, 5688, 5688,
    5691, 5691, 5691, 5691, 5695, 5695, 5695, 5695, 5696, 5696, 5698, 5698, 5699, 5700, 5700, 5700,
    5700, 5700, 5700, 5700, 5700, 5704, 5705, 5705, 5705, 5705, 5706, 5706, 5706, 5706, 5706, 5707,
    5708, 5708, 5708, 5709, 5710, 5710, 5711, 5712, 5712, 5712, 5712, 5713, 5713, 5713, 5713, 5718,
    5718, 5719, 5719, 5744, 5745, 5745, 5745, 5746, 5749, 5749, 5749, 5749, 5749, 5749, 5749, 5749,
    5749, 5749, 5749, 5750, 5750, 5750, 5750, 5750, 5751, 5751, 5751, 5751, 5753, 5753, 5754, 5754,
    5754, 5754, 5754, 5754, 5754, 5754, 5755, 5755, 5755, 5756, 5756, 5756, 5756, 5756, 5756, 5756,
    5756, 5763, 5763, 5772, 5772, 5772, 5776, 5776, 5776, 5777, 5777, 5778, 5778, 5782, 5782, 5782,
    5782, 5782, 5782, 5782, 5782, 5782, 5784, 5784, 5786, 5787, 5787, 5788, 5788, 5788, 5788, 5788,
    5789, 5789, 5789, 5789, 5790, 5790, 5790, 5790, 5790, 5790, 5790, 5790, 5790, 5795, 5797, 5799,
    5799, 5800, 5804, 5805, 5805, 5806, 5806, 5806, 5809, 5811, 5811, 5811, 5812, 5812, 5812, 5812,
    5814, 5814, 5814, 5815, 5815, 5817, 5817, 5817, 5819, 5819, 5819, 5819, 5819, 5821, 5821, 5821,
    5827, 5827, 5829, 5829, 5829, 5829, 5829, 5830, 5830, 5830, 5830, 5830, 5830, 5831, 5832, 5833,
    5834, 5835, 5835, 5836, 5838, 5838, 5838, 5838, 5838, 5838, 5839, 5840, 5840, 5840, 5840, 5840,
    5840, 5840, 5840, 5840, 5840, 5843, 5848, 5848, 5848, 5854, 5856, 5856, 5858, 5859, 5859, 5859,
    5859, 5859, 5859, 5859, 5861, 5861, 5861, 5861, 5861, 5861, 5868, 5868, 5868, 5869, 5869, 5869,
    5869, 5872, 5872, 5872, 5872, 5872, 5872, 5872, 5872, 5874, 5874, 5874, 5874, 5874, 5874, 5874,
    5874, 5874, 5874, 5874, 5877, 5877, 5877, 5877, 5877, 5882, 5882, 5883, 5883, 5884, 5884, 5884,
    5884, 5886, 5887, 5887, 5887, 5888, 5888, 5888, 5888, 5888, 5890, 5892, 5892, 5892, 5892, 5892,
    5896, 5896, 5896, 5896, 5896, 5896, 5896, 5896, 5898, 5898, 5898, 5898, 5898, 5898, 5898, 5899,
    5899, 5899, 5900, 5900, 5900, 5900, 5900, 5900, 5901, 5902, 5902, 5902, 5902, 5902, 5902, 5903,
    5904, 5904, 5904, 5904, 5906, 5907, 5907, 5907, 5907, 5907, 5909, 5909, 5915, 5915, 5916, 5916,
    5917, 5917, 5917, 5918, 5919, 5921, 5921, 5923, 5923, 5923, 5923, 5923, 5926, 5926, 5926, 5926,
    5926, 5927, 5927, 5927, 5927, 5928, 5929, 5929, 5933, 5933, 5933, 5935, 5935, 5935, 5936, 5937,
    5937, 5937, 5937, 5937, 5937, 5940, 5940, 5941, 5942, 5942, 5942, 5942, 5942, 5942, 5942, 5942,
    5942, 5944, 5945, 5945, 5946, 5946, 5949, 5949, 5950, 5950, 5951, 5956, 5956, 5956, 5956, 5956,
    5956, 5956, 5958, 5960, 5960, 5960, 5961, 5963, 5963, 5963, 5963, 5963, 5963, 5963, 5963, 5963,
    5963, 5963, 5976, 5976, 5976, 5978, 5978, 5978, 5979, 5980, 5985, 5985, 5985, 5985, 5985, 5985,
    5985, 5985, 5985, 5985, 5988, 5988, 5988, 5988, 5988, 5988, 5988, 5989, 5989, 5989, 5989, 5989,
    5990, 5990, 5993, 5993, 5993, 5993, 5993, 5993, 5993, 5993, 5993, 5993, 5994, 5994, 5995, 5995,
    5995, 5996, 5996, 5996, 5996, 5996, 5996, 5996, 5998, 5998, 5998, 5998, 5999, 5999, 5999, 5999,
    5999, 6009, 6009, 6010, 6010, 6010, 6010, 6011, 6011, 6011, 6011, 6011, 6011, 6011, 6011, 6011,
    6036, 6036, 6036, 6037, 6037, 6042, 6042, 6042, 6042, 6042, 6044, 6065, 6065, 6065, 6065, 6065,
    6065, 6065, 6065, 6065, 6065, 6065, 6065, 6066, 6066, 6066, 6066, 6066, 6066, 6068, 6068, 6072,
    6072, 6074, 6075, 6075, 6075, 6075, 6075, 6075, 6079, 6079, 6079, 6080, 6081, 6084, 6085, 6087,
    6087, 6087, 6088, 6088, 6088, 6088, 6088, 6088, 6088, 6088, 6090, 6090, 6090, 6090, 6090, 6090,
    6091, 6091, 6091, 6092, 6092, 6092, 6092, 6092, 6092, 6092, 6093, 6093, 6093, 6093, 6093, 6095,
    6096, 6096, 6096, 6101, 6102, 6102, 6102, 6103, 6103, 6104, 6104, 6104, 6104, 6104, 6106, 6107,
    6107, 6107, 6107, 6107, 6107, 6107, 6107, 6109, 6110, 6110, 6112, 6113, 6113, 6113, 6114, 6114,
    6114, 6115, 6115, 6115, 6115, 6115, 6115, 6115, 6115, 6116, 6116, 6116, 6116, 6121, 6121, 6121,
    6122, 6122, 6122, 6122, 6122, 6122, 6122, 6122, 6122, 6122, 6122, 6122, 6122, 6123, 6139, 6139,
    6139, 6139, 6139, 6139, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6149, 6149, 6155, 6155, 6155,
    6155, 6155, 6156, 6156, 6156, 6157, 6157, 6157, 6157, 6158, 6159, 6159, 6167, 6169, 6169, 6170,
    6170, 6170, 6171, 6171, 6171, 6171, 6171, 6171, 6171, 6171, 6171, 6172, 6172, 6172, 6172, 6179,
    6180, 6180, 6180, 6182, 6182, 6182, 6182, 6182, 6182, 6183, 6183, 6183, 6183, 6183, 6183, 6184,
    6184, 6184, 6184, 6184, 6185, 6185, 6185, 6185, 6186, 6186, 6186, 6186, 6186, 6186, 6186, 6186,
    6188, 6188, 6188, 6189, 6189, 6193, 6193, 6194, 6195, 6199, 6200, 6200, 6200, 6201, 6201, 6201,
    6202, 6202, 6203, 6203, 6203, 6203, 6203, 6203, 6203, 6203, 6203, 6203, 6203, 6203, 6203, 6206,
    6206, 6206, 6206, 6206, 6206, 6206, 6206, 6210, 6210, 6210, 6212, 6212, 6212, 6213, 6213, 6213,
    6213, 6213, 6213, 6213, 6213, 6213, 6216, 6219, 6220, 6220, 6220, 6220, 6220, 6220, 6220, 6220,
    6225, 6225, 6225, 6225, 6225, 6226, 6226, 6226, 6226, 6227, 6227, 6228, 6230, 6231, 6235, 6238,
    6238, 6239, 6239, 6239, 6239, 6239, 6239, 6243, 6248, 6249, 6249, 6249, 6249, 6249, 6249, 6250,
    6250, 6257, 6259, 6261, 6261, 6263, 6264, 6265, 6265, 6266, 6266, 6266, 6266, 6275, 6275, 6276,
    6276, 6276, 6276, 6276, 6278, 6278, 6278, 6278, 6278, 6280, 6280, 6280, 6280, 6281, 6281, 6281,
    6281, 6281, 6281, 6281, 6281, 6281, 6281, 6281, 6281, 6282, 6282, 6282, 6282, 6282, 6282, 6282,
    6282, 6282, 6282, 6282, 6282, 6282, 6282, 6284, 6284, 6284, 6297, 6297, 6297, 6298, 6301, 6301,
    6301, 6301, 6302, 6302, 6302, 6302, 6302, 6302, 6302, 6302, 6302, 6303, 6303, 6303, 6304, 6304,
    6305, 6312, 6312, 6312, 6312, 6312, 6312, 6312, 6312, 6312, 6312, 6312, 6312, 6312, 6312, 6312,
    6312, 6312, 6313, 6314, 6315, 6315, 6316, 6316, 6316, 6316, 6316, 6316, 6317, 6317, 6317, 6318,
    6319, 6319, 6319, 6319, 6319, 6319, 6319, 6326, 6326, 6326, 6327, 6327, 6327, 6327, 6328, 6328,
    6328, 6328, 6328, 6328, 6328, 6330, 6331, 6331, 6331, 6331, 6339, 6339, 6339, 6342, 6343, 6343,
    6343, 6343, 6344, 6344, 6345, 6346, 6346, 6346, 6346, 6346, 6348, 6350, 6350, 6350, 6350, 6351,
    6351, 6351, 6353, 6353, 6353, 6353, 6358, 6359, 6359, 6359, 6360, 6360, 6360, 6361, 6361, 6361,
    6361, 6364, 6364, 6364, 6365, 6365, 6365, 6365, 6367, 6367, 6367, 6367, 6367, 6368, 6371, 6372,
    6372, 6372, 6378, 6378, 6378, 6378, 6378, 6378, 6378, 6381, 6381, 6381, 6381, 6382, 6382, 6382,
    6382, 6382, 6410, 6413, 6413, 6413, 6413, 6413, 6413, 6413, 6425, 6427, 6427, 6440, 6440, 6440,
    6440, 6440, 6440, 6449, 6449, 6449, 6449, 6450, 6452, 6453, 6453, 6453, 6454, 6454, 6454, 6454,
    6454, 6456, 6458, 6458, 6458, 6458, 6469, 6469, 6469, 6469, 6469, 6469, 6469, 6469, 6470, 6470,
    6470, 6470, 6470, 6471, 6471, 6471, 6471, 6478, 6479, 6479, 6479, 6479, 6479, 6479, 6479, 6480,
    6480, 6480, 6480, 6495, 6496, 6496, 6496, 6496, 6496, 6496, 6496, 6496, 6496, 6496, 6496, 6497,
    6497, 6499, 6499, 6499, 6501, 6501, 6501, 6502, 6502, 6502, 6502, 6502, 6502, 6503, 6506, 6506,
    6506, 6506, 6509, 6510, 6510, 6510, 6510, 6510, 6510, 6510, 6511, 6518, 6518, 6518, 6518, 6518,
    6519, 6519, 6519, 6520, 6520, 6520, 6520, 6521, 6521, 6521, 6521, 6521, 6528, 6534, 6534, 6534,
    6534, 6534, 6534, 6534, 6534, 6534, 6535, 6536, 6536, 6536, 6536, 6536, 6536, 6536, 6536, 6536,
    6536, 6538, 6538, 6540, 6540, 6540, 6540, 6540, 6542, 6542, 6542, 6542, 6542, 6542, 6542, 6542,
    6542, 6542, 6542, 6542, 6542, 6542, 6542, 6544, 6544, 6544, 6544, 6544, 6544, 6544, 6544, 6545,
    6545, 6545, 6545, 6545, 6545, 6545, 6546, 6546, 6558, 6558, 6558, 6559, 6559, 6559, 6559, 6559,
    6559, 6559, 6559, 6560, 6560, 6560, 6560, 6560, 6560, 6562, 6562, 6562, 6562, 6563, 6563, 6563,
    6563, 6564, 6564, 6564, 6564, 6565, 6565, 6565, 6565, 6565, 6565, 6565, 6568, 6569, 6569, 6571,
    6571, 6571, 6571, 6571, 6572, 6572, 6572, 6572, 6572, 6572, 6572, 6572, 6572, 6572, 6572, 6576,
    6580, 6580, 6581, 6581, 6581, 6581, 6582, 6582, 6582, 6582, 6582, 6582, 6582, 6582, 6583, 6590,
    6590, 6590, 6590, 6590, 6590, 6590, 6590, 6590, 6590, 6591, 6592, 6592, 6592, 6592, 6593, 6609,
    6609, 6609, 6609, 6609, 6609, 6609, 6609, 6609, 6609, 6609, 6609, 6610, 6610, 6610, 6610, 6610,
    6610, 6611, 6612, 6612, 6615, 6615, 6615, 6615, 6616, 6616, 6616, 6631, 6631, 6631, 6631, 6633,
    6636, 6637, 6637, 6637, 6638, 6638, 6638, 6638, 6646, 6648, 6648, 6648, 6648, 6651, 6652, 6652,
    6654, 6654, 6654, 6655, 6655, 6655, 6657, 6657, 6659, 6659, 6659, 6661, 6661, 6661, 6675, 6675,
    6678, 6678, 6679, 6681, 6681, 6681, 6681, 6681, 6685, 6691, 6692, 6692, 6692, 6692, 6692, 6700,
    6700, 6700, 6700, 6700, 6700, 6701, 6704, 6704, 6707, 6707, 6708, 6708, 6712, 6713, 6713, 6721,
    6722, 6722, 6722, 6723, 6723, 6723, 6723, 6723, 6723, 6723, 6724, 6724, 6724, 6724, 6724, 6724,
    6724, 6725, 6725, 6725, 6729, 6729, 6729, 6729, 6730, 6731, 6731, 6731, 6731, 6731, 6731, 6731,
    6731, 6731, 6731, 6732, 6732, 6733, 6733, 6733, 6733, 6733, 6733, 6738, 6742, 6742, 6743, 6743,
    6743, 6743, 6745, 6745, 6745, 6745, 6745, 6745, 6745, 6745, 6746, 6746, 6746, 6746, 6746, 6752,
    6752, 6753, 6756, 6756, 6756, 6756, 6756, 6761, 6761, 6761, 6762, 6762, 6762, 6762, 6762, 6762,
    6762, 6762, 6762, 6762, 6762, 6763, 6763, 6763, 6765, 6765, 6765, 6765, 6765, 6765, 6765, 6765,
    6765, 6766, 6766, 6767, 6767, 6767, 6767, 6767, 6768, 6770, 6771, 6772, 6772, 6775, 6780, 6781,
    6782, 6782, 6782, 6782, 6782, 6782, 6783, 6796, 6796, 6801, 6801, 6802, 6803, 6804, 6804, 6804,
    6804, 6804, 6804, 6805, 6805, 6805, 6805, 6805, 6805, 6806, 6806, 6806, 6806, 6806, 6806, 6806,
    6806, 6806, 6806, 6806, 6807, 6807, 6822, 6822, 6826, 6826, 6826, 6826, 6826, 6826, 6826, 6826,
    6827, 6827, 6828, 6828, 6828, 6828, 6828, 6828, 6831, 6831, 6831, 6832, 6832, 6832, 6835, 6839,
    6839, 6839, 6839, 6839, 6839, 6839, 6840, 6840, 6840, 6845, 6846, 6846, 6846, 6846, 6848, 6850,
    6850, 6851, 6851, 6851, 6851, 6851, 6851, 6856, 6856, 6856, 6856, 6856, 6857, 6862, 6862, 6863,
    6863, 6863, 6864, 6864, 6864, 6864, 6864, 6864, 6864, 6864, 6866, 6866, 6866, 6867, 6867, 6868,
    6868, 6868, 6869, 6869, 6869, 6878, 6891, 6891, 6891, 6891, 6891, 6891, 6891, 6891, 6891, 6891,
    6893, 6893, 6894, 6894, 6894, 6894, 6894, 6895, 6895, 6895, 6895, 6895, 6896, 6896, 6897, 6897,
    6898, 6898, 6898, 6898, 6898, 6898, 6898, 6898, 6898, 6898, 6898, 6898, 6898, 6898, 6898, 6900,
    6901, 6901, 6901, 6901, 6901, 6902, 6910, 6910, 6911, 6911, 6913, 6913, 6913, 6913, 6913, 6913,
    6914, 6918, 6918, 6918, 6920, 6920, 6920, 6920, 6920, 6920, 6921, 6922, 6922, 6922, 6922, 6922,
    6922, 6922, 6922, 6923, 6923, 6923, 6923, 6925, 6925, 6925, 6925, 6927, 6930, 6930, 6931, 6931,
    6931, 6931, 6931, 6940, 6940, 6940, 6941, 6941, 6942, 6942, 6942, 6943, 6943, 6943, 6943, 6943,
    6943, 6943, 6943, 6944, 6945, 6945, 6945, 6945, 6945, 6945, 6945, 6945, 6946, 6946, 6946, 6946,
    6947, 6947, 6947, 6947, 6948, 6948, 6952, 6953, 6953, 6953, 6954, 6957, 6957, 6959, 6959, 6959,
    6961, 6962, 6962, 6962, 6962, 6962, 6962, 6962, 6962, 6962, 6962, 6962, 6962, 6962, 6962, 6962,
    6962, 6964, 6971, 6971, 6971, 6971, 6971, 6971, 6971, 6971, 6971, 6971, 6971, 6973, 6973, 6973,
    6973, 6976, 6976, 6976, 6976, 6976, 6976, 6977, 6985, 6988, 6988, 6988, 6991, 6991, 6991, 6992,
    6992, 6992, 6992, 6992, 6992, 6992, 6992, 6993, 6993, 6993, 6995, 6995, 6995, 6997, 6997, 7000,
    7000, 7000, 7000, 7001, 7002, 7003, 7003, 7003, 7003, 7003, 7004, 7004, 7004, 7004, 7004, 7004,
    7004, 7004, 7004, 7004, 7007, 7007, 7007, 7007, 7007, 7009, 7009, 7009, 7010, 7010, 7013, 7013,
    7013, 7015, 7015, 7015, 7015, 7015, 7015, 7015, 7016, 7017, 7017, 7017, 7017, 7018, 7018, 7018,
    7018, 7018, 7018, 7018, 7018, 7020, 7020, 7020, 7020, 7020, 7020, 7020, 7021, 7024, 7024, 7026,
    7026, 7026, 7027, 7028, 7031, 7032, 7032, 7032, 7032, 7032, 7036, 7036, 7036, 7036, 7037, 7037,
    7037, 7046, 7046, 7046, 7046, 7047, 7047, 7047, 7047, 7049, 7049, 7049, 7051, 7051, 7051, 7051,
    7052, 7053, 7053, 7054, 7056, 7056, 7056, 7056, 7056, 7056, 7056, 7056, 7056, 7056, 7056, 7056,
    7056, 7056, 7057, 7058, 7058, 7059, 7060, 7060, 7064, 7069, 7069, 7072, 7072, 7072, 7072, 7072,
    7073, 7073, 7074, 7074, 7074, 7074, 7074, 7075, 7075, 7075, 7075, 7075, 7075, 7075, 7077, 7077,
    7078, 7078, 7078, 7078, 7078, 7078, 7078, 7083, 7104, 7104, 7104, 7104, 7108, 7108, 7108, 7109,
    7109, 7109, 7114, 7114, 7114, 7114, 7114, 7114, 7115, 7115, 7116, 7116, 7121, 7121, 7121, 7122,
    7122, 7122, 7125, 7125, 7125, 7133, 7133, 7133, 7134, 7134, 7135, 7136, 7137, 7137, 7137, 7139,
    7139, 7139, 7139, 7142, 7142, 7142, 7142, 7143, 7144, 7144, 7144, 7144, 7144, 7144, 7144, 7144,
    7144, 7144, 7144, 7146, 7147, 7147, 7147, 7147, 7147, 7147, 7149, 7149, 7149, 7149, 7150, 7150,
    7150, 7150, 7150, 7154, 7154, 7154, 7154, 7154, 7154, 7155, 7155, 7155, 7155, 7159, 7159, 7159,
    7159, 7159, 7161, 7161, 7161, 7161, 7161, 7161, 7161, 7168, 7169, 7171, 7171, 7171, 7173, 7174,
    7174, 7174, 7175, 7175, 7176, 7176, 7176, 7177, 7177, 7177, 7177, 7191, 7191, 7191, 7191, 7191,
    7191, 7191, 7191, 7191, 7191, 7191, 7191, 7194, 7194, 7210, 7210, 7210, 7210, 7210, 7210, 7210,
    7210, 7210, 7210, 7210, 7210, 7210, 7210, 7210, 7210, 7210, 7210, 7210, 7213, 7213, 7214, 7216,
    7216, 7216, 7216, 7216, 7216, 7216, 7216, 7216, 7216, 7216, 7216, 7216, 7216, 7216, 7216, 7216,
    7217, 7219, 7219, 7231, 7231, 7231, 7231, 7232, 7232, 7232, 7235, 7235, 7237, 7237, 7239, 7249,
    7249, 7249, 7255, 7256, 7256, 7258, 7258, 7258, 7258, 7260, 7262, 7262, 7262, 7263, 7265, 7265,
    7265, 7266, 7266, 7266, 7266, 7266, 7266, 7267, 7268, 7268, 7268, 7273, 7273, 7274, 7274, 7274,
    7274, 7274, 7276, 7276, 7276, 7277, 7277, 7277, 7277, 7278, 7279, 7280, 7280, 7280, 7280, 7281,
    7284, 7285, 7292, 7292, 7292, 7293, 7293, 7295, 7295, 7295, 7295, 7298, 7298, 7298, 7298, 7299,
    7299, 7299, 7299, 7300, 7300, 7300, 7300, 7300, 7300, 7300, 7300, 7300, 7300, 7301, 7301, 7301,
    7301, 7301, 7301, 7301, 7301, 7301, 7301, 7301, 7303, 7303, 7303, 7303, 7303, 7303, 7303, 7303,
    7303, 7303, 7303, 7303, 7303, 7303, 7303, 7303, 7303, 7304, 7309, 7309, 7309, 7310, 7310, 7310,
    7312, 7312, 7313, 7313, 7313, 7313, 7314, 7314, 7314, 7326, 7327, 7327, 7327, 7327, 7328, 7329,
    7329, 7329, 7329, 7329, 7330, 7330, 7332, 7332, 7332, 7342, 7342, 7347, 7347, 7351, 7351, 7355,
    7355, 7355, 7356, 7356, 7357, 7357, 7357, 7357, 7357, 7358, 7358, 7358, 7358, 7358, 7358, 7366,
    7367, 7367, 7369, 7369, 7369, 7371, 7371, 7372, 7373, 7383, 7383, 7383, 7383, 7383, 7385, 7388,
    7389, 7389, 7389, 7389, 7392, 7392, 7393, 7405, 7405, 7408, 7408, 7411, 7411, 7411, 7413, 7419,
    7419, 7422, 7422, 7422, 7422, 7422, 7422, 7422, 7422, 7422, 7422, 7422, 7422, 7422, 7422, 7422,
    7422, 7423, 7427, 7427, 7428, 7429, 7430, 7430, 7430, 7431, 7431, 7431, 7431, 7447, 7447, 7447,
    7447, 7447, 7450, 7450, 7453, 7453, 7454, 7455, 7455, 7455, 7455, 7455, 7455, 7455, 7455, 7455,
    7455, 7455, 7458, 7458, 7459, 7466, 7466, 7467, 7467, 7467, 7467, 7467, 7468, 7468, 7469, 7469,
    7470, 7470, 7470, 7470, 7470, 7471, 7471, 7471, 7471, 7471, 7471, 7471, 7473, 7477, 7477, 7482,
    7482, 7482, 7483, 7483, 7483, 7483, 7483, 7483, 7491, 7495, 7497, 7497, 7497, 7499, 7499, 7499,
    7499, 7502, 7502, 7502, 7506, 7506, 7507, 7507, 7512, 7512, 7512, 7512, 7512, 7512, 7512, 7513,
    7513, 7513, 7513, 7513, 7513, 7513, 7515, 7516, 7518, 7518, 7519, 7519, 7519, 7519, 7519, 7519,
    7519, 7519, 7519, 7519, 7519, 7519, 7519, 7519, 7519, 7522, 7522, 7522, 7522, 7525, 7525, 7525,
    7525, 7526, 7526, 7526, 7526, 7526, 7526, 7526, 7526, 7526, 7526, 7526, 7527, 7527, 7531, 7531,
    7532, 7532, 7532, 7533, 7533, 7534, 7534, 7534, 7534, 7534, 7535, 7541, 7541, 7541, 7541, 7541,
    7541, 7541, 7541, 7541, 7541, 7541, 7541, 7541, 7541, 7541, 7541, 7541, 7541, 7543, 7543, 7544,
    7544, 7544, 7544, 7544, 7544, 7545, 7545, 7545, 7550, 7550, 7550, 7550, 7550, 7550, 7550, 7550,
    7550, 7550, 7556, 7556, 7556, 7557, 7557, 7557, 7557, 7559, 7559, 7559, 7561, 7561, 7561, 7561,
    7561, 7561, 7562, 7562, 7563, 7565, 7565, 7565, 7565, 7565, 7565, 7566, 7566, 7566, 7566, 7566,
    7566, 7567, 7567, 7567, 7567, 7567, 7567, 7567, 7568, 7568, 7568, 7568, 7568, 7569, 7570, 7574,
    7574, 7574, 7577, 7577, 7577, 7577, 7578, 7579, 7579, 7597, 7597, 7597, 7601, 7601, 7601, 7601,
    7603, 7604, 7604, 7605, 7606, 7606, 7606, 7606, 7607, 7607, 7607, 7608, 7608, 7609, 7609, 7609,
    7611, 7612, 7612, 7612, 7612, 7612, 7612, 7613, 7613, 7614, 7614, 7614, 7615, 7615, 7615, 7615,
    7615, 7615, 7616, 7617, 7621, 7622, 7622, 7622, 7622, 7622, 7622, 7659, 7659, 7659, 7659, 7659,
    7659, 7660, 7660, 7662, 7662, 7662, 7668, 7668, 7680, 7681, 7681, 7681, 7681, 7681, 7681, 7686,
    7686, 7686, 7686, 7686, 7687, 7687, 7687, 7687, 7687, 7687, 7687, 7687, 7687, 7687, 7687, 7688,
    7688, 7688, 7688, 7688, 7688, 7688, 7688, 7690, 7690, 7692, 7692, 7692, 7696, 7697, 7697, 7697,
    7698, 7700, 7703, 7703, 7703, 7703, 7703, 7703, 7703, 7705, 7705, 7705, 7705, 7705, 7705, 7705,
    7705, 7705, 7705, 7705, 7706, 7706, 7706, 7706, 7708, 7709, 7709, 7709, 7709, 7709, 7709, 7709,
    7709, 7710, 7710, 7710, 7710, 7710, 7710, 7710, 7710, 7710, 7711, 7711, 7712, 7713, 7713, 7716,
    7716, 7716, 7716, 7718, 7719, 7719, 7719, 7719, 7719, 7719, 7719, 7719, 7726, 7726, 7729, 7729,
    7729, 7729, 7729, 7729, 7729, 7730, 7730, 7730, 7732, 7732, 7734, 7734, 7734, 7735, 7737, 7738,
    7738, 7738, 7738, 7739, 7739, 7739, 7740, 7740, 7740, 7740, 7740, 7741, 7741, 7742, 7743, 7743,
    7743, 7758, 7758, 7758, 7758, 7758, 7758, 7758, 7758, 7759, 7760, 7762, 7762, 7762, 7762, 7762,
    7762, 7762, 7762, 7762, 7762, 7762, 7764, 7764, 7764, 7764, 7764, 7764, 7764, 7764, 7765, 7765,
    7766, 7766, 7766, 7767, 7770, 7770, 7775, 7775, 7775, 7775, 7775, 7778, 7778, 7787, 7787, 7788,
    7788, 7791, 7791, 7791, 7791, 7792, 7792, 7792, 7792, 7796, 7796, 7800, 7800, 7800, 7801, 7802,
    7803, 7803, 7803, 7811, 7811, 7819, 7819, 7819, 7819, 7819, 7819, 7819, 7821, 7822, 7822, 7822,
    7822, 7822, 7822, 7822, 7822, 7823, 7823, 7823, 7823, 7824, 7827, 7827, 7827, 7827, 7829, 7829,
    7830, 7830, 7830, 7830, 7830, 7830, 7830, 7830, 7830, 7831, 7831, 7831, 7832, 7837, 7837, 7837,
    7837, 7837, 7838, 7838, 7838, 7838, 7838, 7838, 7840, 7840, 7840, 7840, 7840, 7842, 7842, 7843,
    7843, 7843, 7846, 7846, 7846, 7846, 7846, 7846, 7847, 7848, 7848, 7848, 7856, 7856, 7856, 7856,
    7856, 7856, 7856, 7856, 7861, 7861, 7861, 7861, 7861, 7861, 7863, 7863, 7863, 7863, 7863, 7863,
    7863, 7865, 7870, 7870, 7870, 7870, 7870, 7871, 7871, 7871, 7871, 7871, 7871, 7872, 7873, 7873,
    7874, 7879, 7879, 7881, 7881, 7881, 7881, 7881, 7881, 7881, 7881, 7881, 7881, 7881, 7881, 7882,
    7883, 7883, 7883, 7883, 7883, 7884, 7885, 7887, 7887, 7888, 7888, 7888, 7888, 7888, 7888, 7889,
    7890, 7890, 7890, 7890, 7892, 7892, 7892, 7892, 7892, 7892, 7892, 7892, 7892, 7892, 7892, 7892,
    7895, 7895, 7896, 7896, 7897, 7897, 7897, 7897, 7897, 7897, 7898, 7898, 7898, 7898, 7898, 7899,
    7899, 7899, 7899, 7899, 7899, 7899, 7900, 7903, 7903, 7903, 7903, 7904, 7904, 7906, 7906, 7906,
    7906, 7908, 7908, 7908, 7908, 7908, 7908, 7913, 7913, 7913, 7923, 7923, 7924, 7924, 7925, 7926,
    7927, 7927, 7927, 7927, 7927, 7930, 7930, 7930, 7930, 7930, 7930, 7935, 7936, 7936, 7938, 7939,
    7939, 7939, 7939, 7939, 7939, 7939, 7946, 7946, 7946, 7946, 7946, 7946, 7946, 7946, 7947, 7947,
    7947, 7947, 7947, 7947, 7947, 7947, 7947, 7947, 7947, 7947, 7947, 7947, 7948, 7948, 7948, 7948,
    7948, 7948, 7948, 7948, 7951, 7956, 7956, 7956, 7956, 7957, 7957, 7957, 7957, 7957, 7957, 7957,
    7957, 7957, 7958, 7959, 7961, 7961, 7962, 7962, 7975, 7975, 7975, 7975, 7975, 7975, 7975, 7975,
    7976, 7977, 7979, 7980, 7981, 7985, 7987, 7987, 7988, 7988, 7988, 7989, 7989, 7989, 7992, 7992,
    7992, 7992, 7992, 7992, 7992, 8004, 8004, 8004, 8006, 8006, 8007, 8008, 8008, 8008, 8008, 8008,
    8008, 8008, 8008, 8009, 8009, 8010, 8010, 8011, 8012, 8012, 8012, 8012, 8013, 8013, 8013, 8013,
    8013, 8013, 8013, 8013, 8013, 8013, 8015, 8015, 8017, 8017, 8017, 8018, 8018, 8018, 8018, 8033,
    8033, 8033, 8033, 8033, 8033, 8033, 8033, 8033, 8033, 8033, 8037, 8037, 8037, 8037, 8037, 8037,
    8040, 8040, 8040, 8040, 8040, 8041, 8041, 8041, 8042, 8042, 8043, 8043, 8043, 8044, 8044, 8044,
    8046, 8046, 8046, 8046, 8046, 8052, 8052, 8052, 8054, 8054, 8054, 8055, 8055, 8055, 8056, 8056,
    8056, 8058, 8058, 8060, 8060, 8060, 8060, 8060, 8060, 8060, 8061, 8064, 8067, 8067, 8067, 8067,
    8067, 8067, 8067, 8067, 8067, 8067, 8067, 8067, 8067, 8067, 8068, 8068, 8068, 8068, 8068, 8068,
    8076, 8077, 8077, 8077, 8077, 8077, 8078, 8078, 8079, 8079, 8084, 8087, 8092, 8092, 8092, 8092,
    8093, 8093, 8093, 8094, 8094, 8094, 8094, 8094, 8094, 8094, 8094, 8094, 8094, 8094, 8094, 8095,
    8095, 8095, 8095, 8095, 8095, 8095, 8097, 8097, 8097, 8097, 8097, 8097, 8097, 8097, 8097, 8097,
    8097, 8097, 8097, 8097, 8097, 8097, 8097, 8099, 8100, 8100, 8100, 8100, 8100, 8101, 8103, 8117,
    8118, 8118, 8129, 8129, 8129, 8129, 8130, 8130, 8132, 8132, 8132, 8132, 8132, 8132, 8135, 8136,
    8136, 8136, 8136, 8136, 8136, 8136, 8136, 8137, 8137, 8137, 8137, 8137, 8137, 8137, 8137, 8137,
    8137, 8137, 8137, 8137, 8137, 8141, 8141, 8141, 8143, 8143, 8143, 8143, 8143, 8143, 8144, 8152,
    8152, 8152, 8152, 8152, 8152, 8152, 8152, 8152, 8155, 8155, 8160, 8166, 8166, 8166, 8166, 8166,
    8166, 8166, 8166, 8166, 8166, 8167, 8167, 8168, 8168, 8168, 8168, 8168, 8168, 8168, 8168, 8168,
    8168, 8169, 8169, 8169, 8170, 8170, 8170, 8170, 8171, 8171, 8171, 8171, 8171, 8171, 8171, 8177,
    8177, 8178, 8178, 8178, 8178, 8179, 8180, 8180, 8181, 8190, 8190, 8190, 8190, 8190, 8190, 8197,
    8197, 8198, 8198, 8199, 8199, 8199, 8201, 8212, 8213, 8213, 8213, 8216, 8216, 8216, 8216, 8216,
    8216, 8216, 8216, 8217, 8217, 8218, 8218, 8220, 8221, 8221, 8221, 8221, 8224, 8224, 8224, 8224,
    8224, 8224, 8247, 8247, 8247, 8247, 8247, 8248, 8248, 8248, 8257, 8257, 8257, 8264, 8264, 8264,
    8264, 8264, 8264, 8264, 8264, 8266, 8266, 8266, 8266, 8267, 8267, 8267, 8269, 8269, 8272, 8272,
    8272, 8273, 8274, 8274, 8286, 8286, 8286, 8286, 8286, 8286, 8286, 8286, 8286, 8286, 8293, 8299,
    8299, 8299, 8299, 8299, 8299, 8299, 8301, 8301, 8301, 8301, 8301, 8301, 8301, 8301, 8301, 8301,
    8301, 8301, 8301, 8301, 8301, 8302, 8302, 8302, 8302, 8302, 8302, 8302, 8303, 8303, 8305, 8305,
    8305, 8305, 8305, 8305, 8305, 8305, 8305, 8305, 8305, 8307, 8307, 8307, 8308, 8308, 8308, 8308,
    8308, 8308, 8308, 8308, 8308, 8309, 8309, 8309, 8309, 8309, 8309, 8309, 8309, 8309, 8309, 8309,
    8309, 8309, 8309, 8310, 8310, 8321, 8322, 8322, 8337, 8337, 8337, 8337, 8337, 8337, 8337, 8337,
    8337, 8337, 8337, 8354, 8354, 8354, 8354, 8354, 8354, 8356, 8356, 8356, 8360, 8361, 8361, 8361,
    8361, 8362, 8362, 8362, 8362, 8362, 8364, 8364, 8364, 8364, 8364, 8366, 8366, 8366, 8367, 8368,
    8370, 8371, 8371, 8372, 8372, 8372, 8372, 8374, 8374, 8374, 8374, 8374, 8374, 8376, 8376, 8376,
    8376, 8376, 8377, 8377, 8382, 8382, 8382, 8382, 8382, 8382, 8382, 8383, 8383, 8383, 8385, 8390,
    8390, 8392, 8392, 8392, 8392, 8396, 8396, 8396, 8396, 8396, 8396, 8396, 8399, 8399, 8399, 8399,
    8399, 8399, 8400, 8400, 8407, 8407, 8407, 8407, 8407, 8407, 8407, 8409, 8409, 8411, 8411, 8411,
    8411, 8411, 8412, 8413, 8413, 8414, 8414, 8415, 8415, 8415, 8415, 8415, 8415, 8415, 8415, 8415,
    8415, 8415, 8415, 8415, 8416, 8424, 8426, 8426, 8427, 8427, 8427, 8427, 8427, 8427, 8427, 8427,
    8427, 8427, 8429, 8429, 8429, 8429, 8429, 8430, 8431, 8433, 8433, 8433, 8433, 8435, 8435, 8445,
    8445, 8445, 8445, 8445, 8446, 8447, 8449, 8449, 8449, 8449, 8449, 8449, 8449, 8450, 8450, 8452,
    8454, 8454, 8454, 8454, 8455, 8455, 8455, 8455, 8455, 8456, 8456, 8456, 8456, 8457, 8457, 8457,
    8460, 8461, 8461, 8461, 8462, 8462, 8463, 8464, 8464, 8464, 8472, 8488, 8489, 8489, 8489, 8489,
    8489, 8489, 8489, 8489, 8489, 8490, 8501, 8501, 8503, 8503, 8503, 8503, 8517, 8519, 8519, 8521,
    8523, 8523, 8523, 8526, 8526, 8531, 8531, 8531, 8531, 8531, 8531, 8532, 8532, 8532, 8532, 8532,
    8532, 8534, 8534, 8534, 8537, 8537, 8539, 8539, 8539, 8546, 8548, 8549, 8549, 8549, 8549, 8549,
    8550, 8553, 8553, 8553, 8553, 8553, 8553, 8553, 8554, 8554, 8556, 8556, 8557, 8558, 8558, 8558,
    8558, 8558, 8558, 8558, 8558, 8558, 8558, 8558, 8558, 8558, 8558, 8558, 8558, 8558, 8558, 8561,
    8561, 8561, 8562, 8562, 8562, 8562, 8562, 8562, 8571, 8571, 8571, 8571, 8571, 8571, 8571, 8571,
    8571, 8571, 8572, 8573, 8573, 8573, 8580, 8580, 8582, 8584, 8584, 8585, 8585, 8585, 8585, 8585,
    8585, 8586, 8586, 8586, 8591, 8592, 8592, 8592, 8592, 8592, 8592, 8592, 8594, 8599, 8601, 8646,
    8646, 8646, 8646, 8646, 8648, 8648, 8648, 8648, 8648, 8648, 8648, 8648, 8648, 8648, 8650, 8650,
    8650, 8650, 8651, 8651, 8652, 8652, 8652, 8652, 8652, 8652, 8652, 8660, 8662, 8662, 8662, 8669,
    8669, 8669, 8671, 8671, 8671, 8671, 8671, 8671, 8671, 8675, 8675, 8675, 8675, 8675, 8677, 8677,
    8677, 8677, 8678, 8681, 8685, 8685, 8685, 8685, 8685, 8685, 8686, 8688, 8688, 8688, 8689, 8689,
    8689, 8689, 8689, 8690, 8690, 8690, 8690, 8690, 8690, 8690, 8700, 8700, 8700, 8700, 8700, 8700,
    8700, 8700, 8700, 8707, 8707, 8707, 8707, 8707, 8707, 8707, 8707, 8709, 8710, 8710, 8711, 8712,
    8719, 8719, 8719, 8719, 8719, 8719, 8719, 8719, 8719, 8719, 8719, 8720, 8720, 8720, 8722, 8722,
    8722, 8722, 8722, 8723, 8723, 8724, 8724, 8724, 8725, 8726, 8726, 8727, 8728, 8728, 8728, 8728,
    8728, 8728, 8728, 8728, 8728, 8728, 8728, 8728, 8728, 8728, 8728, 8728, 8728, 8729, 8729, 8732,
    8732, 8732, 8732, 8732, 8732, 8733, 8733, 8733, 8737, 8737, 8741, 8742, 8742, 8742, 8742, 8742,
    8751, 8751, 8751, 8751, 8751, 8751, 8751, 8751, 8752, 8752, 8752, 8752, 8752, 8753, 8753, 8771,
    8771, 8771, 8773, 8774, 8774, 8774, 8774, 8774, 8774, 8774, 8774, 8776, 8776, 8776, 8792, 8792,
    8793, 8794, 8794, 8794, 8794, 8794, 8794, 8794, 8794, 8794, 8795, 8795, 8797, 8798, 8798, 8798,
    8801, 8801, 8801, 8801, 8801, 8802, 8802, 8802, 8802, 8802, 8802, 8804, 8804, 8879, 8879, 8879,
    8879, 8879, 8879, 8879, 8879, 8879, 8879, 8880, 8880, 8880, 8881, 8881, 8882, 8882, 8882, 8882,
    8882, 8882, 8883, 8884, 8885, 8885, 8885, 8885, 8885, 8889, 8889, 8897, 8897, 8899, 8899, 8902,
    8902, 8902, 8902, 8902, 8902, 8902, 8902, 8902, 8902, 8902, 8903, 8903, 8903, 8903, 8903, 8903,
    8904, 8904, 8904, 8904, 8904, 8904, 8904, 8904, 8906, 8906, 8906, 8907, 8907, 8907, 8907, 8912,
    8912, 8912, 8913, 8913, 8914, 8914, 8914, 8917, 8917, 8917, 8917, 8917, 8918, 8921, 8921, 8921,
    8921, 8921, 8922, 8922, 8923, 8923, 8923, 8923, 8938, 8938, 8940, 8940, 8941, 8941, 8941, 8941,
    8941, 8941, 8942, 8946, 8946, 8946, 8948, 8948, 8948, 8948, 8949, 8949, 8949, 8949, 8949, 8949,
    8949, 8949, 8949, 8949, 8950, 8951, 8951, 8951, 8952, 8952, 8952, 8954, 8954, 8956, 8960, 8960,
    8960, 8960, 8960, 8960, 8960, 8960, 8960, 8961, 8962, 8965, 8965, 8965, 8965, 8967, 8967, 8970,
    8970, 8970, 8970, 8970, 8982, 8982, 8985, 8986, 8986, 8986, 8986, 8986, 8986, 8986, 8986, 8987,
    8987, 8987, 8988, 8988, 8988, 8988, 8988, 8988, 8989, 8989, 8989, 8989, 8995, 8996, 8996, 8997,
    8997, 8997, 8997, 8997, 8997, 8997, 8997, 8997, 8997, 9005, 9005, 9005, 9006, 9006, 9007, 9007,
    9007, 9007, 9007, 9008, 9009, 9009, 9009, 9009, 9009, 9010, 9011, 9011, 9011, 9011, 9014, 9014,
    9014, 9014, 9015, 9019, 9019, 9019, 9019, 9020, 9021, 9022, 9022, 9023, 9023, 9023, 9023, 9023,
    9023, 9023, 9023, 9023, 9023, 9023, 9023, 9024, 9024, 9024, 9024, 9025, 9025, 9025, 9025, 9026,
    9026, 9028, 9028, 9028, 9028, 9029, 9029, 9029, 9029, 9030, 9030, 9031, 9031, 9031, 9031, 9032,
    9035, 9035, 9035, 9035, 9035, 9035, 9035, 9035, 9037, 9040, 9040, 9045, 9045, 9046, 9046, 9046,
    9051, 9052, 9055, 9055, 9055, 9055, 9055, 9055, 9058, 9058, 9058, 9058, 9059, 9059, 9059, 9060,
    9060, 9061, 9061, 9061, 9061, 9061, 9061, 9061, 9061, 9061, 9061, 9061, 9062, 9067, 9067, 9067,
    9068, 9068, 9068, 9068, 9068, 9069, 9078, 9078, 9078, 9079, 9079, 9079, 9079, 9080, 9080, 9080,
    9082, 9082, 9082, 9082, 9082, 9082, 9082, 9083, 9083, 9084, 9084, 9084, 9084, 9084, 9084, 9086,
    9086, 9086, 9086, 9090, 9090, 9099, 9099, 9108, 9109, 9109, 9109, 9109, 9109, 9109, 9109, 9109,
    9109, 9109, 9109, 9109, 9109, 9109, 9109, 9110, 9124, 9126, 9126, 9128, 9128, 9128, 9128, 9128,
    9157, 9157, 9158, 9158, 9159, 9159, 9159, 9159, 9159, 9159, 9159, 9160, 9160, 9163, 9163, 9163,
    9163, 9163, 9163, 9165, 9165, 9165, 9165, 9165, 9167, 9167, 9167, 9167, 9167, 9167, 9167, 9167,
    9167, 9167, 9167, 9167, 9167, 9168, 9168, 9172, 9172, 9173, 9173, 9173, 9173, 9173, 9173, 9173,
    9173, 9173, 9173, 9173, 9173, 9173, 9173, 9173, 9173, 9173, 9175, 9175, 9175, 9176, 9178, 9178,
    9178, 9178, 9178, 9178, 9178, 9180, 9180, 9180, 9180, 9180, 9180, 9180, 9180, 9180, 9180, 9180,
    9180, 9180, 9180, 9181, 9181, 9181, 9181, 9182, 9182, 9186, 9186, 9194, 9194, 9195, 9198, 9198,
    9199, 9201, 9201, 9201, 9201, 9211, 9211, 9217, 9218, 9220, 9220, 9222, 9222, 9222, 9222, 9222,
    9222, 9223, 9224, 9224, 9224, 9224, 9224, 9224, 9225, 9225, 9225, 9225, 9225, 9225, 9225, 9225,
    9225, 9225, 9226, 9231, 9231, 9231, 9231, 9231, 9231, 9231, 9231, 9233, 9233, 9233, 9233, 9233,
    9233, 9233, 9233, 9233, 9233, 9233, 9233, 9233, 9244, 9244, 9245, 9245, 9245, 9245, 9245, 9245,
    9245, 9245, 9245, 9245, 9251, 9251, 9251, 9251, 9258, 9258, 9258, 9261, 9262, 9262, 9262, 9263,
    9264, 9274, 9274, 9274, 9284, 9284, 9284, 9284, 9285, 9285, 9285, 9285, 9285, 9285, 9285, 9285,
    9285, 9285, 9285, 9285, 9285, 9285, 9285, 9285, 9287, 9287, 9287, 9290, 9300, 9301, 9301, 9301,
    9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301,
    9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301, 9301, 9303, 9303, 9303, 9303, 9303,
    9303, 9303, 9303, 9303, 9303, 9304, 9304, 9304, 9304, 9305, 9305, 9305, 9305, 9305, 9305, 9306,
    9307, 9307, 9315, 9315, 9315, 9316, 9316, 9317, 9317, 9317, 9321, 9323, 9323, 9325, 9326, 9326,
    9327, 9327, 9327, 9327, 9327, 9327, 9327, 9330, 9331, 9331, 9331, 9331, 9331, 9331, 9331, 9331,
    9334, 9334, 9334, 9334, 9340, 9340, 9340, 9341, 9341, 9343, 9345, 9348, 9350, 9353, 9353, 9354,
    9355, 9355, 9355, 9356, 9356, 9356, 9356, 9356, 9357, 9357, 9357, 9362, 9367, 9367, 9367, 9367,
    9367, 9367, 9368, 9368, 9368, 9370, 9371, 9371, 9371, 9371, 9371, 9371, 9371, 9371, 9371, 9371,
    9371, 9371, 9371, 9371, 9371, 9372, 9372, 9372, 9372, 9372, 9375, 9375, 9375, 9375, 9378, 9378,
    9378, 9378, 9378, 9378, 9378, 9378, 9378, 9380, 9380, 9384, 9384, 9384, 9384, 9386, 9389, 9389,
    9389, 9389, 9389, 9389, 9391, 9391, 9391, 9391, 9392, 9392, 9392, 9392, 9392, 9392, 9392, 9392,
    9392, 9392, 9392, 9393, 9394, 9394, 9396, 9396, 9396, 9396, 9396, 9396, 9396, 9396, 9396, 9396,
    9396, 9396, 9397, 9398, 9398, 9398, 9398, 9402, 9402, 9402, 9403, 9403, 9406, 9406, 9406, 9406,
    9413, 9413, 9413, 9413, 9414, 9414, 9415, 9415, 9415, 9415, 9415, 9415, 9415, 9422, 9427, 9427,
    9427, 9427, 9427, 9427, 9428, 9428, 9428, 9428, 9428, 9429, 9429, 9429, 9429, 9432, 9432, 9432,
    9432, 9432, 9444, 9444, 9444, 9444, 9444, 9446, 9446, 9446, 9446, 9447, 9455, 9455, 9455, 9455,
    9455, 9455, 9455, 9455, 9456, 9456, 9460, 9460, 9460, 9460, 9460, 9460, 9460, 9460, 9461, 9462,
    9462, 9462, 9462, 9462, 9462, 9463, 9463, 9464, 9464, 9464, 9464, 9464, 9464, 9464, 9464, 9464,
    9476, 9476, 9478, 9479, 9479, 9479, 9479, 9479, 9479, 9479, 9480, 9483, 9483, 9483, 9483, 9485,
    9488, 9488, 9488, 9488, 9488, 9491, 9491, 9491, 9491, 9491, 9497, 9497, 9499, 9499, 9504, 9504,
    9504, 9504, 9504, 9504, 9506, 9506, 9523, 9523, 9523, 9524, 9524, 9524, 9525, 9531, 9531, 9532,
    9539, 9539, 9539, 9539, 9539, 9539, 9539, 9540, 9540, 9540, 9540, 9540, 9540, 9540, 9540, 9540,
    9540, 9540, 9543, 9543, 9543, 9545, 9545, 9545, 9547, 9548, 9548, 9548, 9550, 9550, 9550, 9551,
    9552, 9552, 9552, 9552, 9552, 9552, 9552, 9552, 9552, 9553, 9554, 9554, 9554, 9554, 9554, 9554,
    9554, 9555, 9555, 9557, 9557, 9557, 9572, 9572, 9572, 9572, 9574, 9574, 9574, 9574, 9574, 9574,
    9574, 9574, 9574, 9574, 9574, 9576, 9576, 9576, 9584, 9584, 9584, 9584, 9584, 9584, 9584, 9584,
    9584, 9598, 9599, 9599, 9600, 9600, 9600, 9600, 9600, 9600, 9600, 9600, 9600, 9600, 9600, 9600,
    9600, 9602, 9602, 9602, 9602, 9603, 9603, 9603, 9607, 9607, 9607, 9609, 9609, 9609, 9609, 9610,
    9610, 9610, 9610, 9610, 9610, 9610, 9610, 9610, 9615, 9615, 9615, 9616, 9618, 9618, 9618, 9618,
    9618, 9618, 9618, 9618, 9618, 9619, 9619, 9619, 9619, 9619, 9621, 9621, 9621, 9623, 9623, 9623,
    9623, 9623, 9623, 9624, 9624, 9624, 9624, 9624, 9625, 9625, 9625, 9625, 9625, 9625, 9625, 9625,
    9625, 9625, 9627, 9627, 9627, 9627, 9627, 9644, 9644, 9644, 9644, 9644, 9644, 9645, 9645, 9646,
    9657, 9659, 9659, 9659, 9659, 9662, 9665, 9665, 9666, 9666, 9666, 9666, 9666, 9667, 9667, 9667,
    9668, 9668, 9669, 9670, 9670, 9670, 9670, 9670, 9670, 9670, 9670, 9671, 9671, 9671, 9671, 9672,
    9672, 9672, 9672, 9672, 9672, 9672, 9672, 9672, 9675, 9682, 9682, 9682, 9682, 9682, 9682, 9684,
    9684, 9684, 9686, 9686, 9686, 9687, 9691, 9691, 9691, 9691, 9691, 9692, 9692, 9692, 9692, 9692,
    9696, 9697, 9700, 9700, 9703, 9703, 9711, 9711, 9711, 9711, 9711, 9711, 9711, 9713, 9713, 9714,
    9714, 9714, 9714, 9714, 9714, 9714, 9715, 9726, 9728, 9730, 9730, 9730, 9730, 9730, 9730, 9730,
    9730, 9731, 9731, 9731, 9731, 9738, 9738, 9738, 9738, 9739, 9739, 9743, 9745, 9745, 9745, 9745,
    9746, 9746, 9746, 9746, 9746, 9746, 9747, 9747, 9748, 9748, 9748, 9749, 9750, 9750, 9750, 9751,
    9751, 9751, 9751, 9752, 9752, 9753, 9759, 9759, 9759, 9759, 9759, 9760, 9760, 9760, 9767, 9767,
    9769, 9770, 9770, 9770, 9770, 9770, 9771, 9780, 9780, 9782, 9782, 9782, 9782, 9784, 9784, 9784,
    9784, 9785, 9793, 9793, 9795, 9795, 9796, 9796, 9796, 9796, 9796, 9796, 9797, 9797, 9797, 9797,
    9797, 9797, 9797, 9798, 9798, 9798, 9798, 9798, 9798, 9798, 9798, 9798, 9798, 9798, 9799, 9799,
    9799, 9815, 9815, 9815, 9815, 9815, 9816, 9833, 9833, 9835, 9835, 9835, 9835, 9835, 9835, 9836,
    9836, 9836, 9836, 9836, 9836, 9836, 9836, 9836, 9840, 9841, 9843, 9847, 9847, 9847, 9848, 9849,
    9907, 9907, 9908, 9909, 9909, 9909, 9919, 9919, 9919, 9920, 9920, 9920, 9920, 9920, 9920, 9920,
    9920, 9920, 9920, 9920, 9927, 9927, 9927, 9927, 9927, 9927, 9927, 9928, 9928, 9935, 9935, 9936,
    9936, 9936, 9936, 9936, 9936, 9936, 9937, 9937, 9942, 9942, 9942, 9942, 9942, 9944, 9944, 9945,
    9945, 9945, 9945, 9945, 9945, 9945, 9952, 9952, 9952, 9952, 9954, 9954, 9954, 9954, 9954, 9957,
    9957, 9957, 9957, 9957, 9958, 9958, 9958, 9958, 9958, 9958, 9958, 9960, 9960, 9960, 9968, 9968,
    9974, 9974, 9974, 9974, 9974, 9979, 9980, 9980, 9986, 9986, 9986, 9988, 9991, 9995, 9995, 9995,
    9995, 9996, 9996, 9996, 9997, 9997, 9997, 9998, 9998, 10003, 10003, 10003, 10003, 10003, 10003, 10003,
    10004, 10004, 10004, 10004, 10004, 10004, 10004, 10008, 10009, 10009, 10009, 10013, 10018, 10019, 10019, 10020,
    10020, 10020, 10020, 10020, 10020, 10020, 10020, 10020, 10020, 10020, 10021, 10021, 10028, 10028, 10028, 10030,
    10033, 10033, 10033, 10036, 10037, 10039, 10039, 10039, 10039, 10039, 10039, 10040, 10040, 10041, 10041, 10041,
    10041, 10041, 10041, 10041, 10041, 10041, 10041, 10041, 10042, 10042, 10042, 10042, 10042, 10042, 10042, 10042,
    10044, 10045, 10057, 10057, 10057, 10057, 10057, 10057, 10057, 10058, 10060, 10060, 10060, 10060, 10060, 10060,
    10060, 10061, 10062, 10062, 10062, 10062, 10062, 10062, 10063, 10064, 10064, 10064, 10067, 10067, 10067, 10067,
    10067, 10067, 10067, 10067, 10069, 10069, 10069, 10069, 10069, 10078, 10078, 10078, 10078, 10080, 10080, 10080,
    10081, 10081, 10082, 10088, 10088, 10090, 10090, 10090, 10091, 10091, 10093, 10093, 10093, 10093, 10093, 10093,
    10093, 10093, 10093, 10097, 10097, 10097, 10097, 10097, 10097, 10097, 10098, 10098, 10116, 10116, 10116, 10116,
    10117, 10117, 10117, 10117, 10117, 10117, 10117, 10117, 10118, 10118, 10119, 10119, 10119, 10119, 10119, 10119,
    10119, 10119, 10119, 10119, 10119, 10122, 10122, 10123, 10123, 10125, 10125, 10127, 10127, 10131, 10131, 10131,
    10133, 10133, 10134, 10134, 10134, 10135, 10135, 10135, 10136, 10136, 10138, 10139, 10139, 10139, 10145, 10145,
    10145, 10145, 10145, 10145, 10145, 10145, 10145, 10147, 10147, 10147, 10148, 10148, 10148, 10149, 10149, 10149,
    10149, 10150, 10150, 10152, 10152, 10152, 10152, 10152, 10152, 10152, 10152, 10153, 10155, 10155, 10155, 10155,
    10157, 10157, 10157, 10158, 10158, 10158, 10158, 10158, 10158, 10158, 10158, 10158, 10158, 10158, 10159, 10160,
    10160, 10160, 10160, 10160, 10160, 10160, 10160, 10160, 10160, 10160, 10160, 10160, 10161, 10174, 10178, 10180,
    10183, 10189, 10189, 10193, 10193, 10194, 10194, 10194, 10194, 10194, 10194, 10194, 10194, 10194, 10195, 10195,
    10195, 10195, 10200, 10200, 10200, 10200, 10200, 10200, 10201, 10202, 10202, 10202, 10202, 10202, 10202, 10202,
    10202, 10202, 10202, 10202, 10206, 10206, 10207, 10209, 10210, 10210, 10212, 10212, 10212, 10212, 10213, 10213,
    10213, 10213, 10213, 10217, 10217, 10218, 10218, 10218, 10218, 10218, 10224, 10224, 10225, 10225, 10225, 10225,
    10225, 10226, 10226, 10226, 10226, 10227, 10227, 10227, 10227, 10227, 10228, 10228, 10229, 10229, 10230, 10233,
    10233, 10233, 10234, 10235, 10235, 10243, 10247, 10247, 10253, 10253, 10254, 10254, 10261, 10262, 10262, 10262,
    10262, 10265, 10268, 10268, 10268, 10270, 10270, 10270, 10270, 10270, 10270, 10277, 10277, 10277, 10277, 10278,
    10279, 10280, 10280, 10280, 10280, 10280, 10280, 10285, 10287, 10287, 10287, 10289, 10293, 10295, 10295, 10296,
    10297, 10297, 10297, 10297, 10297, 10297, 10297, 10297, 10297, 10298, 10298, 10298, 10298, 10300, 10300, 10302,
    10302, 10302, 10302, 10303, 10303, 10305, 10305, 10305, 10305, 10305, 10305, 10305, 10305, 10305, 10305, 10305,
    10305, 10305, 10305, 10305, 10305, 10305, 10305, 10306, 10314, 10314, 10314, 10316, 10318, 10318, 10318, 10318,
    10319, 10319, 10319, 10321, 10321, 10321, 10321, 10321, 10322, 10322, 10322, 10322, 10322, 10322, 10322, 10324,
    10324, 10324, 10324, 10348, 10354, 10354, 10354, 10354, 10371, 10393, 10393, 10394, 10394, 10395, 10395, 10395,
    10396, 10396, 10396, 10396, 10396, 10396, 10398, 10398, 10398, 10402, 10402, 10408, 10409, 10409, 10409, 10409,
    10409, 10409, 10409, 10410, 10410, 10410, 10410, 10413, 10416, 10416, 10416, 10416, 10416, 10416, 10417, 10417,
    10420, 10420, 10420, 10421, 10421, 10421, 10421, 10421, 10421, 10422, 10422, 10422, 10422, 10422, 10422, 10423,
    10423, 10423, 10423, 10423, 10423, 10423, 10423, 10423, 10423, 10424, 10424, 10424, 10426, 10426, 10426, 10426,
    10426, 10426, 10426, 10426, 10426, 10426, 10426, 10426, 10426, 10426, 10427, 10427, 10427, 10427, 10433, 10433,
    10433, 10433, 10433, 10434, 10435, 10435, 10442, 10442, 10442, 10442, 10442, 10442, 10442, 10442, 10442, 10443,
    10443, 10443, 10447, 10447, 10447, 10447, 10447, 10447, 10447, 10447, 10447, 10447, 10447, 10447, 10448, 10448,
    10448, 10448, 10448, 10448, 10448, 10448, 10448, 10449, 10457, 10457, 10457, 10457, 10457, 10461, 10462, 10463,
    10463, 10464, 10465, 10466, 10466, 10466, 10466, 10466, 10466, 10466, 10466, 10466, 10466, 10466, 10466, 10466,
    10466, 10466, 10466, 10466, 10466, 10466, 10466, 10467, 10468, 10468, 10468, 10468, 10468, 10468, 10469, 10470,
    10470, 10470, 10471, 10471, 10471, 10471, 10471, 10471, 10471, 10471, 10471, 10471, 10471, 10471, 10472, 10472,
    10474, 10474, 10474, 10474, 10476, 10476, 10476, 10476, 10477, 10478, 10478, 10478, 10479, 10479, 10479, 10479,
    10486, 10486, 10487, 10487, 10487, 10487, 10498, 10498, 10498, 10498, 10498, 10498, 10498, 10498, 10499, 10501,
    10501, 10501, 10501, 10501, 10502, 10503, 10503, 10503, 10505, 10505, 10505, 10506, 10506, 10506, 10508, 10509,
    10509, 10509, 10509, 10509, 10509, 10511, 10511, 10513, 10513, 10513, 10513, 10514, 10514, 10514, 10514, 10515,
    10515, 10515, 10523, 10523, 10523, 10523, 10524, 10524, 10524, 10524, 10524, 10525, 10525, 10525, 10527, 10528,
    10528, 10528, 10528, 10528, 10529, 10529, 10530, 10530, 10530, 10530, 10530, 10530, 10530, 10531, 10531, 10531,
    10532, 10532, 10532, 10532, 10532, 10532, 10532, 10532, 10532, 10532, 10532, 10532, 10532, 10532, 10532, 10534,
    10534, 10535, 10535, 10535, 10535, 10535, 10543, 10543, 10543, 10544, 10544, 10544, 10547, 10548, 10548, 10549,
    10549, 10549, 10549, 10549, 10549, 10549, 10549, 10549, 10549, 10549, 10549, 10549, 10550, 10550, 10550, 10550,
    10553, 10553, 10553, 10557, 10560, 10560, 10560, 10560, 10560, 10560, 10560, 10560, 10560, 10560, 10560, 10561,
    10563, 10563, 10563, 10563, 10563, 10563, 10563, 10563, 10564, 10578, 10578, 10593, 10593, 10593, 10593, 10593,
    10596, 10596, 10596, 10597, 10597, 10597, 10612, 10613, 10613, 10613, 10613, 10613, 10626, 10628, 10628, 10628,
    10628, 10628, 10633, 10633, 10633, 10633, 10633, 10633, 10633, 10635, 10637, 10637, 10643, 10643, 10643, 10644,
    10644, 10644, 10646, 10646, 10646, 10646, 10646, 10646, 10651, 10653, 10653, 10654, 10663, 10663, 10663, 10663,
    10672, 10672, 10674, 10674, 10674, 10677, 10677, 10691, 10691, 10692, 10692, 10692, 10692, 10692, 10692, 10692,
    10692, 10692, 10694, 10696, 10696, 10696, 10696, 10697, 10698, 10700, 10700, 10700, 10700, 10700, 10700, 10700,
    10700, 10700, 10700, 10702, 10702, 10704, 10704, 10705, 10706, 10706, 10708, 10708, 10708, 10708, 10710, 10710,
    10710, 10714, 10714, 10714, 10714, 10714, 10716, 10716, 10717, 10717, 10717, 10717, 10717, 10717, 10718, 10718,
    10718, 10718, 10718, 10718, 10718, 10718, 10718, 10721, 10721, 10721, 10731, 10732, 10732, 10732, 10732, 10732,
    10732, 10732, 10732, 10732, 10732, 10732, 10732, 10732, 10732, 10733, 10733, 10733, 10733, 10733, 10733, 10734,
    10736, 10737, 10739, 10739, 10740, 10740, 10740, 10740, 10742, 10742, 10742, 10742, 10743, 10747, 10747, 10748,
    10748, 10748, 10748, 10748, 10748, 10750, 10750, 10750, 10760, 10765, 10765, 10817, 10817, 10818, 10818, 10818,
    10818, 10818, 10818, 10818, 10818, 10819, 10819, 10819, 10819, 10819, 10820, 10820, 10820, 10821, 10821, 10821,
    10821, 10821, 10821, 10821, 10821, 10842, 10844, 10844, 10844, 10844, 10844, 10844, 10847, 10847, 10849, 10851,
    10859, 10859, 10861, 10861, 10862, 10862, 10862, 10869, 10869, 10869, 10869, 10869, 10869, 10869, 10871, 10871,
    10871, 10871, 10871, 10871, 10871, 10873, 10873, 10882, 10882, 10882, 10884, 10884, 10884, 10884, 10884, 10885,
    10885, 10885, 10885, 10885, 10888, 10888, 10890, 10890, 10890, 10890, 10890, 10891, 10891, 10892, 10892, 10894,
    10894, 10894, 10894, 10895, 10897, 10897, 10898, 10898, 10898, 10900, 10907, 10908, 10908, 10908, 10908, 10911,
    10911, 10911, 10912, 10912, 10912, 10913, 10914, 10914, 10914, 10914, 10914, 10914, 10918, 10918, 10918, 10918,
    10918, 10918, 10918, 10918, 10918, 10918, 10918, 10918, 10931, 10931, 10931, 10931, 10939, 10939, 10939, 10939,
    10939, 10939, 10939, 10939, 10939, 10939, 10939, 10939, 10939, 10939, 10940, 10948, 10949, 10951, 10951, 10953,
    10953, 10953, 10954, 10954, 10954, 10955, 10955, 10973, 10975, 10975, 10976, 10976, 10976, 10976, 10976, 10976,
    10976, 10976, 10976, 10976, 10976, 10976, 10976, 10976, 10976, 10976, 10976, 10976, 10977, 10977, 10978, 10979,
    10979, 10981, 10981, 10982, 11004, 11007, 11007, 11007, 11007, 11007, 11011, 11017, 11019, 11019, 11020, 11020,
    11035, 11035, 11035, 11035, 11035, 11035, 11036, 11042, 11044, 11044, 11067, 11070, 11072, 11072, 11072, 11072,
    11072, 11074, 11074, 11074, 11078, 11078, 11079, 11082, 11082, 11082, 11082, 11084, 11086, 11086, 11089, 11089,
    11089, 11089, 11089, 11089, 11090, 11090, 11090, 11090, 11091, 11091, 11091, 11092, 11092, 11092, 11092, 11092,
    11093, 11093, 11098, 11098, 11098, 11098, 11098, 11098, 11098, 11098, 11098, 11099, 11099, 11101, 11101, 11104,
    11104, 11104, 11104, 11116, 11116, 11116, 11132, 11132, 11133, 11133, 11133, 11133, 11133, 11133, 11133, 11133,
    11135, 11135, 11135, 11135, 11135, 11135, 11135, 11137, 11137, 11137, 11137, 11138, 11144, 11144, 11144, 11146,
    11146, 11146, 11166, 11166, 11166, 11167, 11168, 11169, 11169, 11170, 11170, 11170, 11172, 11172, 11172, 11172,
    11173, 11173, 11173, 11173, 11173, 11173, 11175, 11175, 11175, 11175, 11188, 11188, 11188, 11190, 11190, 11193,
    11193, 11193, 11193, 11193, 11193, 11193, 11193, 11198, 11198, 11198, 11199, 11199, 11200, 11200, 11201, 11201,
    11201, 11201, 11201, 11201, 11201, 11201, 11203, 11203, 11203, 11205, 11208, 11208, 11208, 11208, 11208, 11208,
    11208, 11210, 11210, 11210, 11210, 11210, 11210, 11219, 11219, 11219, 11219, 11219, 11219, 11219, 11219, 11219,
    11219, 11219, 11220, 11220, 11221, 11221, 11221, 11221, 11222, 11226, 11226, 11226, 11226, 11227, 11227, 11239,
    11241, 11241, 11241, 11242, 11244, 11244, 11244, 11244, 11246, 11246, 11246, 11246, 11246, 11251, 11252, 11252,
    11252, 11255, 11256, 11256, 11256, 11256, 11256, 11256, 11256, 11256, 11258, 11258, 11258, 11259, 11259, 11259,
    11259, 11260, 11260, 11274, 11278, 11278, 11278, 11278, 11278, 11279, 11279, 11279, 11279, 11279, 11279, 11279,
    11279, 11287, 11289, 11291, 11291, 11291, 11291, 11291, 11292, 11308, 11308, 11308, 11308, 11308, 11311, 11312,
    11312, 11312, 11313, 11313, 11313, 11314, 11314, 11314, 11314, 11314, 11314, 11314, 11315, 11315, 11316, 11316,
    11317, 11317, 11317, 11317, 11317, 11317, 11317, 11324, 11324, 11324, 11324, 11324, 11324, 11363, 11363, 11366,
    11366, 11366, 11366, 11382, 11387, 11387, 11387, 11387, 11387, 11387, 11387, 11388, 11389, 11392, 11393, 11393,
    11393, 11393, 11393, 11393, 11393, 11393, 11393, 11393, 11397, 11397, 11397, 11397, 11399, 11399, 11399, 11399,
    11401, 11401, 11401, 11402, 11402, 11403, 11404, 11404, 11404, 11404, 11406, 11406, 11406, 11406, 11406, 11406,
    11406, 11406, 11406, 11409, 11409, 11409, 11409, 11409, 11410, 11410, 11410, 11410, 11410, 11412, 11412, 11412,
    11412, 11413, 11414, 11414, 11414, 11414, 11414, 11414, 11414, 11414, 11415, 11415, 11415, 11415, 11415, 11415,
    11415, 11416, 11416, 11416, 11416, 11416, 11416, 11416, 11416, 11417, 11417, 11419, 11419, 11420, 11420, 11422,
    11423, 11423, 11423, 11424, 11424, 11425, 11425, 11437, 11437, 11439, 11439, 11455, 11455, 11457, 11457, 11457,
    11458, 11460, 11460, 11460, 11460, 11460, 11460, 11461, 11461, 11461, 11461, 11463, 11468, 11468, 11472, 11472,
    11472, 11472, 11472, 11474, 11474, 11476, 11476, 11476, 11476, 11476, 11476, 11476, 11476, 11476, 11476, 11477,
    11477, 11477, 11480, 11480, 11480, 11481, 11481, 11481, 11483, 11483, 11488, 11488, 11489, 11491, 11491, 11492,
    11494, 11495, 11495, 11498, 11499, 11499, 11519, 11519, 11519, 11519, 11521, 11521, 11521, 11529, 11530, 11530,
    11531, 11531, 11531, 11531, 11532, 11532, 11532, 11533, 11533, 11533, 11533, 11533, 11533, 11534, 11534, 11535,
    11535, 11535, 11535, 11535, 11536, 11536, 11536, 11536, 11536, 11536, 11536, 11537, 11537, 11539, 11539, 11539,
    11539, 11553, 11554, 11555, 11555, 11555, 11555, 11555, 11555, 11556, 11556, 11556, 11556, 11557, 11557, 11557,
    11557, 11557, 11558, 11562, 11562, 11563, 11563, 11563, 11563, 11563, 11565, 11571, 11571, 11571, 11571, 11571,
    11571, 11572, 11576, 11576, 11576, 11576, 11576, 11576, 11576, 11578, 11580, 11582, 11582, 11583, 11584, 11584,
    11584, 11584, 11584, 11585, 11585, 11585, 11586, 11586, 11586, 11588, 11588, 11588, 11588, 11594, 11594, 11594,
    11594, 11594, 11594, 11594, 11594, 11594, 11594, 11594, 11594, 11594, 11594, 11594, 11594, 11595, 11597, 11597,
    11597, 11598, 11598, 11600, 11600, 11600, 11600, 11600, 11600, 11601, 11601, 11601, 11601, 11604, 11604, 11606,
    11606, 11606, 11606, 11606, 11606, 11606, 11607, 11607, 11607, 11607, 11607, 11607, 11612, 11612, 11613, 11616,
    11616, 11616, 11616, 11616, 11619, 11619, 11619, 11619, 11619, 11619, 11619, 11619, 11619, 11619, 11620, 11620,
    11620, 11620, 11620, 11620, 11620, 11620, 11622, 11622, 11622, 11622, 11625, 11625, 11625, 11626, 11626, 11626,
    11626, 11626, 11626, 11626, 11626, 11626, 11628, 11628, 11628, 11628, 11628, 11628, 11628, 11628, 11628, 11628,
    11630, 11630, 11630, 11630, 11630, 11633, 11633, 11633, 11633, 11633, 11633, 11634, 11634, 11634, 11634, 11634,
    11634, 11634, 11634, 11634, 11634, 11636, 11636, 11636, 11636, 11636, 11637, 11637, 11638, 11638, 11638, 11638,
    11639, 11639, 11639, 11639, 11639, 11639, 11648, 11650, 11650, 11650, 11650, 11650, 11655, 11655, 11655, 11655,
    11655, 11656, 11656, 11656, 11657, 11657, 11657, 11657, 11658, 11658, 11658, 11670, 11670, 11670, 11670, 11670,
    11670, 11670, 11670, 11670, 11670, 11670, 11670, 11670, 11670, 11670, 11671, 11671, 11671, 11672, 11672, 11672,
    11673, 11674, 11674, 11678, 11678, 11681, 11681, 11681, 11681, 11681, 11681, 11682, 11682, 11682, 11682, 11684,
    11684, 11685, 11685, 11685, 11685, 11685, 11685, 11685, 11685, 11685, 11693, 11693, 11693, 11693, 11694, 11694,
    11694, 11694, 11694, 11694, 11694, 11694, 11694, 11694, 11694, 11695, 11696, 11696, 11703, 11703, 11703, 11703,
    11713, 11715, 11715, 11715, 11715, 11715, 11715, 11719, 11719, 11719, 11719, 11720, 11720, 11720, 11720, 11721,
    11721, 11721, 11723, 11728, 11728, 11728, 11734, 11734, 11734, 11735, 11735, 11736, 11736, 11736, 11736, 11738,
    11738, 11738, 11738, 11738, 11743, 11743, 11743, 11748, 11749, 11750, 11767, 11767, 11767, 11767, 11768, 11775,
    11775, 11775, 11775, 11775, 11775, 11775, 11775, 11775, 11775, 11775, 11775, 11775, 11775, 11776, 11776, 11777,
    11778, 11778, 11778, 11778, 11778, 11784, 11785, 11786, 11786, 11786, 11786, 11786, 11786, 11786, 11786, 11786,
    11786, 11786, 11796, 11796, 11796, 11796, 11796, 11796, 11796, 11796, 11796, 11796, 11797, 11797, 11798, 11798,
    11798, 11798, 11798, 11798, 11798, 11799, 11799, 11799, 11799, 11799, 11804, 11804, 11806, 11807, 11807, 11807,
    11807, 11807, 11807, 11807, 11809, 11809, 11814, 11814, 11814, 11814, 11814, 11814, 11814, 11814, 11814, 11814,
    11814, 11815, 11815, 11815, 11815, 11815, 11815, 11815, 11817, 11817, 11818, 11818, 11819, 11819, 11820, 11824,
    11824, 11824, 11824, 11825, 11825, 11825, 11825, 11825, 11825, 11825, 11825, 11826, 11826, 11826, 11826, 11828,
    11828, 11829, 11830, 11830, 11830, 11830, 11830, 11831, 11831, 11859, 11859, 11859, 11859, 11863, 11863, 11864,
    11865, 11865, 11865, 11865, 11865, 11865, 11865, 11866, 11872, 11874, 11874, 11875, 11875, 11876, 11876, 11876,
    11876, 11876, 11876, 11877, 11877, 11877, 11877, 11883, 11883, 11884, 11885, 11886, 11886, 11886, 11888, 11888,
    11888, 11888, 11888, 11889, 11890, 11890, 11890, 11890, 11903, 11903, 11903, 11903, 11903, 11937, 11939, 11939,
    11941, 11941, 11946, 11947, 11947, 11947, 11947, 11947, 11947, 11948, 11948, 11948, 11948, 11948, 11948, 11948,
    11948, 11948, 11948, 11948, 11948, 11958, 11958, 11963, 11963, 11963, 11963, 11963, 11963, 11963, 11967, 11967,
    11973, 11973, 11973, 11976, 11976, 11976, 11976, 11976, 11977, 11977, 11977, 11978, 11979, 11984, 11984, 11984,
    11989, 11989, 11989, 11991, 11991, 11991, 11991, 11993, 11993, 11995, 11995, 11995, 11995, 11999, 11999, 11999,
    11999, 11999, 11999, 11999, 12000, 12001, 12001, 12001, 12001, 12001, 12001, 12001, 12001, 12001, 12004, 12004,
    12006, 12006, 12006, 12008, 12008, 12009, 12011, 12011, 12011, 12011, 12011, 12011, 12012, 12012, 12012, 12012,
    12012, 12012, 12012, 12012, 12012, 12012, 12012, 12016, 12016, 12017, 12020, 12020, 12020, 12020, 12022, 12022,
    12022, 12022, 12022, 12022, 12022, 12027, 12027, 12027, 12027, 12027, 12027, 12027, 12027, 12027, 12027, 12028,
    12028, 12028, 12028, 12028, 12028, 12028, 12028, 12029, 12029, 12029, 12031, 12032, 12032, 12034, 12034, 12036,
    12036, 12037, 12037, 12038, 12038, 12038, 12038, 12038, 12040, 12040, 12040, 12040, 12040, 12040, 12046, 12046,
    12047, 12047, 12047, 12048, 12048, 12048, 12048, 12050, 12051, 12052, 12052, 12052, 12053, 12053, 12055, 12055,
    12055, 12056, 12056, 12056, 12057, 12057, 12057, 12057, 12057, 12057, 12057, 12057, 12057, 12057, 12061, 12062,
    12062, 12065, 12065, 12066, 12066, 12066, 12066, 12066, 12066, 12066, 12066, 12066, 12067, 12067, 12067, 12068,
    12068, 12068, 12068, 12069, 12069, 12069, 12069, 12069, 12069, 12069, 12070, 12071, 12071, 12072, 12072, 12072,
    12072, 12076, 12076, 12076, 12076, 12081, 12081, 12089, 12089, 12091, 12091, 12091, 12091, 12091, 12091, 12091,
    12091, 12091, 12091, 12091, 12091, 12091, 12091, 12091, 12091, 12091, 12092, 12092, 12093, 12093, 12094, 12094,
    12094, 12094, 12094, 12094, 12095, 12095, 12095, 12096, 12096, 12096, 12096, 12096, 12096, 12121, 12127, 12127,
    12127, 12127, 12127, 12127, 12128, 12128, 12130, 12134, 12134, 12134, 12134, 12137, 12137, 12143, 12144, 12149,
    12149, 12150, 12150, 12152, 12152, 12155, 12155, 12155, 12155, 12155, 12155, 12155, 12155, 12155, 12155, 12155,
    12155, 12155, 12155, 12155, 12156, 12156, 12156, 12165, 12165, 12167, 12167, 12167, 12167, 12167, 12167, 12167,
    12168, 12168, 12168, 12168, 12174, 12174, 12174, 12174, 12174, 12174, 12174, 12174, 12174, 12176, 12176, 12176,
    12176, 12176, 12176, 12178, 12178, 12178, 12178, 12178, 12178, 12178, 12178, 12180, 12180, 12181, 12181, 12181,
    12181, 12181, 12182, 12182, 12187, 12190, 12193, 12197, 12197, 12198, 12198, 12198, 12203, 12203, 12203, 12203,
    12203, 12204, 12205, 12205, 12205, 12205, 12205, 12205, 12207, 12207, 12208, 12208, 12208, 12208, 12208, 12208,
    12208, 12208, 12208, 12208, 12208, 12208, 12217, 12217, 12217, 12217, 12217, 12218, 12218, 12218, 12218, 12218,
    12218, 12221, 12221, 12221, 12221, 12221, 12223, 12223, 12224, 12224, 12224, 12225, 12225, 12226, 12228, 12228,
    12228, 12228, 12228, 12228, 12228, 12228, 12228, 12228, 12228, 12228, 12228, 12228, 12228, 12228, 12228, 12228,
    12228, 12228, 12228, 12228, 12228, 12228, 12228, 12228, 12228, 12229, 12231, 12232, 12234, 12234, 12234, 12236,
    12236, 12236, 12237, 12237, 12239, 12239, 12239, 12239, 12239, 12240, 12240, 12240, 12240, 12240, 12240, 12240,
    12240, 12240, 12240, 12241, 12241, 12243, 12243, 12243, 12243, 12243, 12243, 12246, 12255, 12255, 12255, 12255,
    12255, 12255, 12256, 12257, 12257, 12258, 12258, 12260, 12260, 12260, 12262, 12262, 12262, 12262, 12262, 12262,
    12262, 12262, 12262, 12262, 12262, 12262, 12265, 12265, 12265, 12265, 12266, 12266, 12267, 12267, 12280, 12280,
    12280, 12280, 12280, 12280, 12280, 12280, 12281, 12281, 12281, 12283, 12283, 12283, 12283, 12291, 12291, 12291,
    12291, 12291, 12294, 12294, 12294, 12294, 12294, 12294, 12295, 12296, 12297, 12298, 12298, 12298, 12301, 12303,
    12303, 12303, 12304, 12304, 12304, 12304, 12304, 12306, 12306, 12307, 12307, 12308, 12308, 12308, 12308, 12308,
    12312, 12313, 12313, 12313, 12313, 12313, 12313, 12313, 12313, 12313, 12313, 12313, 12313, 12313, 12313, 12313,
    12313, 12313, 12313, 12313, 12313, 12313, 12313, 12313, 12315, 12315, 12322, 12322, 12323, 12323, 12323, 12323,
    12323, 12323, 12323, 12323, 12323, 12325, 12325, 12325, 12325, 12325, 12325, 12325, 12325, 12325, 12325, 12327,
    12327, 12328, 12328, 12330, 12330, 12335, 12335, 12340, 12340, 12340, 12340, 12340, 12343, 12343, 12343, 12350,
    12361, 12361, 12361, 12361, 12361, 12361, 12361, 12361, 12361, 12361, 12361, 12363, 12363, 12363, 12365, 12365,
    12365, 12365, 12372, 12380, 12381, 12381, 12381, 12381, 12381, 12381, 12381, 12381, 12383, 12389, 12389, 12389,
    12389, 12389, 12391, 12391, 12391, 12391, 12391, 12391, 12398, 12412, 12412, 12412, 12412, 12465, 12465, 12465,
    12465, 12466, 12466, 12466, 12466, 12466, 12466, 12466, 12466, 12466, 12473, 12473, 12477, 12477, 12477, 12479,
    12479, 12479, 12479, 12479, 12479, 12479, 12479, 12479, 12479, 12480, 12480, 12482, 12482, 12482, 12482, 12482,
    12482, 12482, 12483, 12485, 12485, 12485, 12488, 12488, 12488, 12488, 12488, 12488, 12488, 12491, 12492, 12492,
    12492, 12492, 12492, 12493, 12493, 12493, 12493, 12493, 12493, 12493, 12493, 12493, 12494, 12494, 12494, 12494,
    12494, 12494, 12494, 12494, 12494, 12494, 12494, 12495, 12495, 12496, 12496, 12497, 12497, 12497, 12497, 12504,
    12504, 12504, 12505, 12505, 12505, 12505, 12505, 12505, 12516, 12516, 12516, 12516, 12516, 12516, 12521, 12521,
    12521, 12521, 12524, 12524, 12524, 12524, 12525, 12525, 12526, 12526, 12526, 12528, 12528, 12529, 12529, 12529,
    12529, 12529, 12529, 12529, 12529, 12541, 12541, 12541, 12541, 12543, 12543, 12545, 12545, 12545, 12546, 12552,
    12554, 12554, 12554, 12554, 12554, 12554, 12554, 12554, 12554, 12554, 12554, 12554, 12557, 12570, 12570, 12574,
    12574, 12574, 12574, 12575, 12580, 12580, 12580, 12581, 12584, 12587, 12587, 12587, 12587, 12588, 12588, 12589,
    12589, 12591, 12591, 12591, 12591, 12592, 12592, 12592, 12592, 12592, 12593, 12593, 12593, 12593, 12593, 12593,
    12593, 12595, 12595, 12595, 12596, 12597, 12598, 12599, 12599, 12600, 12600, 12600, 12600, 12600, 12600, 12600,
    12600, 12600, 12600, 12601, 12601, 12601, 12601, 12601, 12602, 12602, 12605, 12612, 12612, 12612, 12612, 12612,
    12612, 12612, 12612, 12612, 12612, 12612, 12613, 12613, 12613, 12613, 12613, 12613, 12613, 12613, 12613, 12614,
    12614, 12616, 12616, 12619, 12621, 12621, 12621, 12621, 12621, 12625, 12625, 12625, 12628, 12632, 12632, 12632,
    12632, 12632, 12632, 12632, 12632, 12632, 12632, 12632, 12635, 12635, 12635, 12635, 12635, 12635, 12637, 12637,
    12637, 12637, 12637, 12637, 12637, 12638, 12638, 12638, 12638, 12638, 12639, 12639, 12639, 12644, 12644, 12644,
    12644, 12644, 12648, 12648, 12648, 12648, 12648, 12652, 12652, 12653, 12653, 12655, 12655, 12655, 12655, 12656,
    12656, 12656, 12656, 12656, 12656, 12656, 12656, 12656, 12656, 12656, 12656, 12659, 12660, 12660, 12660, 12660,
    12660, 12660, 12660, 12660, 12660, 12662, 12662, 12662, 12662, 12662, 12663, 12663, 12666, 12668, 12668, 12668,
    12668, 12669, 12669, 12669, 12669, 12669, 12669, 12669, 12669, 12669, 12669, 12669, 12669, 12671, 12671, 12671,
    12671, 12671, 12671, 12671, 12671, 12689, 12689, 12689, 12689, 12690, 12691, 12691, 12691, 12692, 12692, 12692,
    12692, 12693, 12696, 12696, 12698, 12698, 12698, 12699, 12700, 12705, 12705, 12706, 12706, 12706, 12706, 12706,
    12706, 12706, 12707, 12707, 12722, 12725, 12725, 12726, 12726, 12726, 12728, 12728, 12728, 12728, 12728, 12728,
    12728, 12728, 12728, 12729, 12729, 12729, 12729, 12729, 12729, 12729, 12729, 12729, 12730, 12730, 12755, 12755,
    12755, 12755, 12755, 12755, 12758, 12759, 12759, 12759, 12759, 12763, 12763, 12763, 12763, 12764, 12764, 12764,
    12764, 12766, 12766, 12766, 12766, 12766, 12766, 12766, 12766, 12766, 12766, 12766, 12788, 12790, 12799, 12799,
    12799, 12799, 12799, 12799, 12799, 12799, 12799, 12801, 12801, 12801, 12801, 12801, 12801, 12801, 12801, 12801,
    12801, 12801, 12803, 12803, 12809, 12810, 12810, 12810, 12810, 12810, 12810, 12810, 12814, 12824, 12824, 12824,
    12824, 12824, 12824, 12826, 12826, 12826, 12826, 12827, 12827, 12827, 12827, 12827, 12827, 12837, 12837, 12837,
    12837, 12837, 12837, 12840, 12840, 12840, 12840, 12840, 12841, 12841, 12841, 12842, 12844, 12849, 12849, 12849,
    12849, 12849, 12852, 12852, 12852, 12855, 12860, 12864, 12883, 12883, 12883, 12883, 12883, 12883, 12885, 12885,
    12887, 12896, 12896, 12896, 12896, 12896, 12896, 12896, 12896, 12898, 12909, 12909, 12909, 12909, 12909, 12909,
    12910, 12910, 12910, 12912, 12913, 12913, 12913, 12918, 12934, 12934, 12934, 12935, 12935, 12935, 12935, 12936,
    12937, 12938, 12943, 12944, 12945, 12945, 12945, 12945, 12945, 12945, 12945, 12945, 12945, 12945, 12945, 12946,
    12946, 12946, 12946, 12946, 12946, 12948, 12948, 12948, 12948, 12949, 12949, 12950, 12955, 12955, 12955, 12956,
    12958, 12958, 12958, 12958, 12958, 12958, 12958, 12958, 12959, 12959, 12964, 12964, 12964, 12964, 12964, 12964,
    12964, 12964, 12964, 12971, 12971, 12971, 12973, 12973, 12973, 12973, 12973, 12973, 12973, 12975, 12980, 12980,
    12980, 12980, 12980, 12981, 12981, 12981, 12982, 12982, 12982, 12982, 12984, 12984, 12984, 12984, 12984, 12984,
    12984, 12984, 12984, 12989, 12989, 12989, 12989, 12989, 12989, 12989, 12989, 12989, 12989, 12996, 12996, 12997,
    12997, 12997, 12997, 12997, 12999, 12999, 13000, 13002, 13002, 13002, 13003, 13003, 13003, 13003, 13010, 13028,
    13029, 13029, 13029, 13029, 13031, 13031, 13031, 13043, 13045, 13049, 13050, 13051, 13052, 13052, 13052, 13058,
    13058, 13058, 13058, 13058, 13064, 13064, 13064, 13064, 13064, 13064, 13064, 13064, 13065, 13065, 13066, 13066,
    13066, 13066, 13067, 13067, 13067, 13067, 13067, 13068, 13079, 13081, 13081, 13081, 13081, 13081, 13081, 13081,
    13081, 13081, 13083, 13083, 13083, 13090, 13090, 13093, 13093, 13093, 13094, 13095, 13098, 13098, 13099, 13099,
    13099, 13099, 13100, 13100, 13100, 13100, 13101, 13101, 13104, 13104, 13105, 13105, 13105, 13107, 13107, 13107,
    13107, 13107, 13124, 13124, 13124, 13124, 13124, 13124, 13124, 13128, 13128, 13129, 13129, 13130, 13130, 13130,
    13130, 13130, 13131, 13131, 13132, 13132, 13133, 13139, 13140, 13140, 13140, 13142, 13142, 13144, 13144, 13144,
    13144, 13145, 13145, 13145, 13146, 13146, 13146, 13146, 13146, 13146, 13148, 13148, 13148, 13148, 13148, 13148,
    13148, 13148, 13151, 13151, 13151, 13151, 13151, 13154, 13155, 13155, 13155, 13155, 13155, 13173, 13183, 13183,
    13183, 13186, 13186, 13186, 13186, 13186, 13187, 13187, 13187, 13187, 13187, 13187, 13187, 13187, 13187, 13189,
    13189, 13189, 13189, 13189, 13189, 13189, 13189, 13189, 13190, 13193, 13194, 13194, 13195, 13197, 13197, 13197,
    13200, 13200, 13200, 13200, 13200, 13200, 13200, 13201, 13201, 13201, 13201, 13201, 13201, 13201, 13201, 13201,
    13201, 13201, 13201, 13201, 13201, 13201, 13201, 13201, 13201, 13201, 13201, 13204, 13204, 13209, 13211, 13212,
    13212, 13213, 13215, 13215, 13215, 13215, 13215, 13215, 13215, 13217, 13220, 13221, 13221, 13222, 13222, 13222,
    13223, 13224, 13224, 13224, 13224, 13234, 13234, 13234, 13234, 13234, 13234, 13235, 13235, 13235, 13240, 13240,
    13240, 13240, 13240, 13240, 13240, 13240, 13240, 13241, 13241, 13241, 13241, 13241, 13242, 13244, 13244, 13245,
    13245, 13245, 13245, 13253, 13253, 13253, 13253, 13254, 13254, 13254, 13254, 13254, 13254, 13254, 13255, 13255,
    13255, 13261, 13261, 13268, 13268, 13268, 13284, 13284, 13285, 13286, 13286, 13286, 13286, 13286, 13286, 13286,
    13286, 13286, 13286, 13286, 13286, 13286, 13288, 13288, 13288, 13288, 13288, 13290, 13290, 13291, 13291, 13291,
    13291, 13291, 13291, 13293, 13294, 13310, 13310, 13310, 13310, 13313, 13313, 13313, 13313, 13313, 13313, 13313,
    13313, 13313, 13313, 13313, 13313, 13313, 13313, 13313, 13313, 13313, 13313, 13313, 13313, 13313, 13313, 13319,
    13322, 13322, 13322, 13322, 13322, 13322, 13322, 13322, 13326, 13326, 13326, 13326, 13326, 13329, 13329, 13329,
    13329, 13330, 13330, 13330, 13330, 13330, 13330, 13330, 13330, 13330, 13330, 13330, 13330, 13330, 13330, 13333,
    13333, 13333, 13333, 13333, 13333, 13333, 13334, 13334, 13334, 13334, 13336, 13337, 13337, 13338, 13338, 13338,
    13338, 13338, 13345, 13345, 13345, 13345, 13345, 13345, 13356, 13357, 13360, 13360, 13360, 13360, 13360, 13360,
    13360, 13360, 13360, 13361, 13363, 13364, 13364, 13364, 13364, 13364, 13365, 13365, 13365, 13369, 13369, 13369,
    13370, 13370, 13371, 13378, 13378, 13378, 13378, 13378, 13378, 13378, 13378, 13378, 13378, 13379, 13379, 13379,
    13379, 13382, 13382, 13382, 13382, 13382, 13382, 13382, 13382, 13382, 13382, 13382, 13382, 13382, 13386, 13389,
    13390, 13390, 13390, 13391, 13395, 13395, 13395, 13395, 13395, 13396, 13396, 13403, 13405, 13405, 13405, 13405,
    13405, 13405, 13405, 13405, 13406, 13406, 13406, 13410, 13413, 13413, 13413, 13413, 13413, 13413, 13413, 13413,
    13413, 13413, 13414, 13414, 13414, 13414, 13414, 13414, 13414, 13420, 13421, 13421, 13421, 13421, 13421, 13421,
    13421, 13422, 13423, 13423, 13423, 13424, 13424, 13424, 13424, 13424, 13424, 13424, 13424, 13426, 13426, 13427,
    13429, 13429, 13439, 13439, 13439, 13439, 13440, 13441, 13444, 13444, 13444, 13444, 13446, 13446, 13446, 13448,
    13448, 13448, 13448, 13448, 13448, 13449, 13449, 13449, 13449, 13449, 13449, 13449, 13450, 13450, 13450, 13450,
    13450, 13453, 13453, 13453, 13453, 13453, 13453, 13453, 13455, 13455, 13455, 13455, 13455, 13457, 13457, 13457,
    13457, 13457, 13462, 13462, 13482, 13482, 13495, 13495, 13495, 13495, 13497, 13498, 13500, 13500, 13504, 13504,
    13504,
];

static immutable ushort[NUM_DICTIONARY_WORDS] kDictionaryWords =
[
    17386, 17399, 8875, 13553, 13719, 24616, 22666, 16025, 8347, 45064, 45066, 45070, 45087, 43012, 40992, 41016,
    41029, 41062, 38948, 36870, 36910, 37015, 34927, 28708, 29063, 24789, 14834, 15206, 15878, 13435, 10391, 10970,
    10981, 8356, 8437, 16088, 8816, 10691, 8684, 15171, 26694, 8731, 14772, 12401, 17102, 14856, 13793, 15406,
    13803, 36973, 32827, 27055, 25186, 19184, 13818, 8942, 8499, 11204, 38977, 13287, 13598, 21125, 14394, 15046,
    12788, 10420, 8267, 10889, 30750, 30852, 26676, 26925, 24704, 25070, 23102, 19031, 19186, 16843, 14618, 15215,
    15260, 12564, 8392, 15957, 18643, 12513, 22630, 22902, 17071, 15054, 15308, 34888, 26737, 23337, 19156, 16442,
    15583, 12633, 8607, 18733, 19335, 17151, 17187, 22731, 13442, 17058, 16285, 12547, 15611, 13519, 16638, 22805,
    10488, 23010, 23262, 23317, 21173, 19232, 23037, 23443, 18704, 16413, 16024, 10335, 10685, 8308, 30795, 26724,
    14826, 15977, 10783, 16425, 10421, 10897, 21016, 21189, 16586, 14361, 14525, 24665, 18855, 39021, 24712, 22611,
    21034, 19309, 16497, 29114, 21407, 9155, 11216, 15082, 8398, 14290, 12984, 13296, 13458, 39022, 34844, 34929,
    28749, 28750, 26736, 26854, 26985, 24661, 25092, 25199, 23098, 23133, 23327, 23332, 21017, 16610, 14349, 14663,
    8961, 19110, 13832, 8743, 34867, 26982, 24760, 24947, 22647, 20905, 18489, 18891, 23217, 21089, 18853, 19161,
    16394, 10428, 10912, 11054, 12912, 8570, 18701, 8720, 13434, 13393, 13904, 12944, 14228, 15967, 12461, 10344,
    10697, 24935, 23411, 20673, 18516, 16522, 16726, 15146, 15672, 17351, 12383, 14222, 24676, 22572, 20889, 20927,
    12694, 8669, 13556, 17328, 14229, 13924, 36884, 34826, 28909, 24577, 24655, 25120, 22926, 23162, 15370, 8474,
    9016, 14936, 15224, 15432, 12764, 12953, 13464, 13484, 10358, 10484, 10739, 34938, 28784, 26713, 24622, 22832,
    19044, 16944, 16970, 14632, 14932, 19441, 14060, 11134, 13762, 22894, 18753, 43040, 41056, 34834, 34854, 30815,
    30858, 30887, 28678, 28742, 28835, 28944, 28949, 29042, 26649, 26722, 26900, 27050, 27132, 24603, 24674, 24707,
    24821, 24867, 24888, 25022, 25030, 22585, 22613, 22648, 22705, 22824, 22910, 22982, 23286, 23527, 20661, 20886,
    18551, 18665, 18730, 18851, 18903, 16463, 16490, 16574, 16686, 16767, 17097, 15935, 16059, 47126, 38914, 20680,
    14047, 20852, 34832, 17327, 13007, 10498, 49172, 37027, 37034, 37035, 32874, 25274, 25308, 25346, 21354, 21366,
    17284, 14086, 14127, 41022, 24999, 25014, 25078, 22854, 22867, 19139, 19220, 16518, 14831, 15217, 10270, 10564,
    16798, 16284, 16340, 26629, 22538, 16695, 12695, 10909, 8307, 15116, 15376, 12727, 10290, 10757, 21000, 12369,
    10241, 13026, 38984, 28794, 26686, 26952, 24626, 24703, 24972, 22830, 23306, 20702, 20840, 21180, 21302, 18604,
    19145, 16482, 10722, 24640, 24836, 25064, 25266, 22646, 22754, 22779, 22958, 23229, 23502, 23510, 20481, 20609,
    20705, 18509, 16637, 16973, 15133, 15830, 12800, 10242, 15110, 18779, 10812, 13120, 26712, 17181, 14756, 14840,
    14858, 15145, 8204, 10768, 13149, 41069, 13475, 28717, 23134, 20773, 18722, 18796, 10506, 23362, 20529, 20892,
    18553, 12798, 12882, 8242, 8266, 17132, 8658, 26786, 18810, 18951, 8859, 13190, 13101, 14284, 9061, 8452,
    12602, 25099, 22672, 16791, 16179, 10276, 10413, 8231, 21067, 8702, 36873, 20967, 21289, 14365, 14402, 14984,
    15210, 16316, 38957, 36927, 28754, 28966, 26884, 24687, 21069, 12858, 13752, 38922, 38930, 30722, 18804, 8979,
    18632, 8708, 9036, 29037, 29047, 23415, 24620, 17205, 23314, 15253, 16138, 8202, 14319, 9208, 10924, 13309,
    13768, 12315, 12683, 14821, 12843, 9005, 18638, 16659, 12630, 12784, 11180, 9101, 9083, 13884, 8879, 13630,
    26784, 12329, 14006, 17162, 36915, 12650, 13898, 10355, 28986, 8486, 9175, 34900, 26754, 24754, 10433, 14510,
    13282, 13339, 13727, 25279, 21335, 17232, 17251, 17291, 14084, 24797, 24857, 22884, 23210, 23437, 21166, 19119,
    19174, 16951, 13948, 13959, 11036, 11192, 8314, 32846, 22528, 15431, 15549, 10419, 15058, 15240, 8258, 13061,
    14688, 26659, 18584, 13485, 26867, 26942, 24893, 24931, 25041, 25045, 25136, 25154, 21072, 21233, 10804, 13085,
    12844, 13107, 18535, 20541, 17103, 14880, 18953, 17021, 10738, 8197, 11167, 15777, 13441, 11020, 11133, 9085,
    15686, 10478, 13584, 17353, 14017, 14287, 23015, 19132, 16698, 15096, 15277, 16136, 8252, 13432, 15483, 20917,
    16803, 16189, 11068, 14512, 14514, 15446, 15576, 16312, 11046, 22553, 22565, 22942, 21234, 18866, 14927, 15185,
    15287, 15465, 15949, 16096, 16324, 12292, 10279, 10581, 8243, 15264, 11142, 12558, 34920, 19045, 19233, 15361,
    12665, 8608, 28782, 28961, 29044, 24579, 24595, 22534, 21028, 21061, 14676, 15203, 10392, 10437, 24600, 24962,
    25152, 22862, 18900, 16412, 16525, 16828, 17195, 14370, 14526, 14819, 10997, 17282, 15889, 10961, 17396, 15438,
    10887, 8981, 9090, 26871, 10517, 8627, 13599, 14364, 14796, 12500, 13102, 10505, 21230, 16919, 14300, 8894,
    16381, 11048, 25087, 17123, 10406, 25062, 25172, 18789, 19151, 16044, 16225, 10410, 10687, 8311, 13097, 13788,
    15494, 8642, 15654, 12885, 13890, 8946, 13656, 17183, 11090, 29138, 25277, 25333, 21388, 36883, 25035, 14090,
    13685, 30812, 28918, 27127, 24918, 25164, 20946, 19030, 16504, 16572, 17043, 14574, 16369, 13606, 13612, 34881,
    22775, 23354, 16986, 16080, 8393, 36926, 16548, 36901, 10855, 15579, 17352, 13170, 9099, 43037, 24992, 20809,
    10340, 8368, 28943, 26887, 27083, 25102, 25183, 23083, 20690, 20772, 21013, 18749, 18790, 16410, 14345, 15844,
    10876, 11238, 26651, 20607, 16503, 17357, 14283, 14289, 13870, 15914, 13083, 14768, 16468, 14462, 17174, 28813,
    29101, 26933, 24953, 12352, 13817, 30828, 28741, 29081, 26921, 26991, 27109, 24824, 24887, 25010, 25176, 25198,
    22797, 23067, 23135, 23155, 23234, 23503, 15108, 15440, 15983, 27131, 23505, 23012, 18466, 41065, 36983, 32839,
    30782, 30805, 28906, 26685, 26893, 26935, 26943, 27110, 24699, 24916, 24982, 25219, 22548, 22576, 22735, 22819,
    22826, 22870, 22893, 22922, 23148, 23233, 23545, 20535, 20540, 20712, 20812, 20864, 20876, 20984, 21022, 21132,
    18462, 18631, 18663, 18683, 18766, 18793, 18920, 19144, 16408, 16498, 16636, 16719, 16980, 17111, 17113, 14362,
    14629, 14709, 14718, 15362, 15423, 16363, 13988, 21398, 21406, 36989, 24823, 25193, 18864, 19093, 16560, 16567,
    14606, 15452, 12667, 11032, 11140, 16256, 13059, 16315, 11016, 9093, 41001, 28812, 15589, 10574, 8229, 8700,
    10753, 8832, 49169, 45080, 40968, 41055, 38976, 30882, 24822, 8725, 25328, 11160, 20871, 16288, 28798, 12928,
    8675, 20960, 36968, 18927, 9066, 13236, 23537, 8940, 16577, 38926, 26717, 16856, 17037, 9176, 14259, 10412,
    13856, 22912, 10500, 36885, 36985, 28715, 9020, 16030, 13829, 21100, 21250, 18514, 18993, 16551, 16759, 14393,
    14564, 13214, 19259, 16578, 17140, 14358, 9000, 34880, 23424, 15585, 10879, 13902, 23492, 15151, 16035, 8294,
    9091, 8761, 17179, 14333, 38999, 24884, 18536, 18991, 12340, 8493, 8796, 17163, 38944, 12782, 12965, 12970,
    13332, 14844, 13302, 13245, 25248, 22895, 23127, 23214, 21043, 21110, 21276, 21290, 18444, 18783, 18808, 16456,
    16656, 16815, 16185, 12598, 16833, 27008, 25086, 20976, 18559, 16448, 14873, 12469, 12601, 14569, 8436, 22773,
    21182, 13938, 41009, 27087, 18770, 13567, 11136, 37042, 19240, 12983, 16923, 15453, 25579, 36877, 34820, 32822,
    26627, 22529, 22569, 16628, 12736, 10826, 10916, 8399, 8980, 16219, 10756, 25536, 30729, 26678, 24701, 16532,
    17080, 17175, 14835, 15395, 16155, 12301, 12826, 13978, 14014, 10244, 11050, 15604, 13036, 23070, 23380, 24582,
    9040, 36980, 27037, 27125, 24808, 24812, 23110, 23178, 23429, 23477, 15721, 41077, 37024, 25358, 25364, 17246,
    14099, 14133, 14141, 14146, 9137, 21308, 17082, 11043, 9117, 13704, 23520, 13166, 13765, 16183, 10623, 13692,
    17370, 20827, 19092, 36933, 37009, 29069, 26941, 26994, 27093, 27095, 24932, 25126, 23208, 23370, 23452, 19024,
    14908, 15747, 14117, 14118, 23228, 15084, 10367, 24659, 12486, 21344, 15705, 12343, 10516, 17110, 14040, 15402,
    11004, 23211, 21030, 14667, 12597, 21416, 17258, 17266, 15398, 13500, 14140, 13652, 15322, 20861, 13011, 13696,
    15909, 13313, 15895, 23192, 20973, 21050, 14469, 12482, 32801, 25063, 25107, 22687, 22781, 23034, 23406, 20628,
    20808, 20932, 18475, 19016, 19192, 16808, 14631, 14935, 15265, 16022, 16190, 12291, 12407, 12431, 12728, 12949,
    13805, 10264, 10809, 16921, 14570, 12571, 10254, 11132, 23069, 38946, 17019, 15587, 8318, 15897, 8508, 19293,
    9190, 17393, 22887, 12656, 10592, 10960, 8256, 28705, 26844, 26864, 24941, 22952, 20784, 21029, 15913, 13395,
    13217, 15386, 15478, 18721, 12463, 10672, 11153, 15121, 23123, 24656, 13986, 11171, 13611, 26914, 23054, 23467,
    26733, 16943, 13068, 13925, 13974, 14039, 11014, 26751, 34892, 27015, 23462, 15503, 15820, 12548, 12861, 12632,
    36932, 28902, 18847, 19012, 15660, 16280, 12459, 13977, 11111, 8484, 8928, 13452, 43055, 37053, 37071, 30893,
    30898, 30920, 19414, 14210, 15256, 12351, 11033, 15727, 12350, 12662, 10307, 24623, 19222, 14732, 15505, 8583,
    8463, 24890, 13428, 18772, 14636, 10784, 8445, 21370, 13410, 11058, 22849, 17172, 14566, 15189, 12382, 12745,
    12787, 10265, 10293, 10625, 10781, 11030, 9112, 10818, 8660, 8937, 13158, 13526, 14053, 41081, 32861, 32870,
    32878, 29109, 29111, 29115, 29118, 29124, 29141, 29143, 29145, 21333, 21423, 22746, 20660, 14854, 15618, 16254,
    8797, 11062, 29091, 23444, 8826, 13186, 27103, 38998, 16838, 14994, 12324, 19238, 13614, 8296, 13619, 21208,
    18974, 18998, 14381, 15167, 13804, 36881, 34902, 32791, 30762, 28818, 20948, 17389, 28960, 14423, 38989, 26636,
    26858, 24583, 24598, 24610, 18496, 18934, 15324, 16275, 14477, 17256, 14130, 8811, 22616, 26965, 22671, 22944,
    23387, 20604, 20933, 18787, 18944, 16573, 10486, 8792, 8860, 10918, 20659, 16669, 15198, 16222, 12575, 10275,
    10813, 8709, 25116, 36947, 22931, 17051, 24650, 13200, 11001, 24648, 20611, 18589, 15050, 20662, 16717, 15674,
    34821, 32772, 32845, 32848, 30785, 28766, 10689, 10886, 34885, 16487, 14422, 14799, 21121, 16364, 28674, 12289,
    12759, 13157, 36951, 24705, 15610, 15722, 15728, 12565, 12842, 43036, 28789, 16764, 16926, 14375, 15139, 15539,
    12475, 23466, 15129, 12585, 10608, 8977, 41035, 30822, 29058, 24807, 22633, 22891, 16822, 16942, 14800, 25307,
    17261, 13896, 13717, 36909, 36993, 28773, 29099, 26658, 25139, 23261, 23287, 15346, 26701, 26898, 22682, 23073,
    23303, 20818, 20970, 18801, 15992, 13326, 21431, 21491, 17363, 17369, 14298, 13293, 39033, 18651, 14995, 15702,
    12586, 10260, 29039, 27096, 25127, 23382, 23400, 23449, 15776, 15019, 12903, 12908, 10730, 13358, 13885, 24818,
    8631, 17053, 13054, 13431, 13533, 12956, 13506, 16931, 8425, 8674, 14682, 9078, 14670, 14029, 11035, 14958,
    14961, 15846, 16086, 14675, 16272, 8395, 32806, 30749, 15592, 10578, 24883, 23206, 16380, 24831, 18740, 14647,
    13058, 13090, 13100, 15602, 14568, 10507, 8592, 13278, 16167, 13064, 13203, 8805, 10972, 11003, 9070, 9169,
    16697, 15015, 8292, 23151, 15355, 12606, 10602, 10683, 16680, 10703, 28809, 8646, 34874, 26642, 27074, 25052,
    22577, 23076, 20935, 21086, 16981, 14794, 12845, 10288, 10644, 16320, 10682, 32808, 12737, 9045, 13779, 24737,
    14712, 15975, 10295, 10690, 13462, 13850, 24783, 15205, 16464, 14559, 8405, 36907, 28778, 28870, 28992, 29085,
    26699, 24738, 23442, 20538, 20819, 21229, 19089, 16738, 16821, 17171, 14717, 15142, 15156, 17131, 17338, 13361,
    13683, 10932, 12479, 8996, 17215, 16296, 12450, 8898, 45091, 20906, 15571, 13522, 26720, 37013, 19334, 8396,
    17217, 17333, 24775, 25132, 25153, 18719, 14702, 17376, 14308, 14560, 10596, 13072, 26743, 22678, 22972, 18999,
    14966, 12827, 28699, 13043, 13351, 23342, 17371, 14591, 8427, 23288, 23379, 21241, 14919, 12672, 10824, 8386,
    13374, 17400, 17402, 10802, 18703, 13201, 10431, 30837, 30847, 28927, 28934, 26727, 26908, 24617, 24862, 22660,
    22696, 22755, 22848, 22964, 23053, 23383, 23401, 21181, 15839, 16259, 12441, 10713, 8662, 13001, 13257, 13578,
    13839, 10567, 19036, 16090, 12840, 13580, 32790, 28761, 26732, 22531, 12647, 8415, 26789, 25006, 23249, 23522,
    20789, 18919, 19285, 19321, 16840, 14739, 15838, 15962, 12693, 10320, 14304, 16390, 16927, 17210, 12823, 11127,
    8645, 17359, 21441, 21442, 43059, 30946, 25377, 25435, 25459, 19348, 19397, 19404, 14179, 14181, 30768, 30804,
    28841, 28974, 28985, 21157, 18532, 16013, 12376, 12831, 10341, 30820, 28787, 28958, 24825, 22979, 20870, 21213,
    19227, 15044, 15332, 15353, 8502, 8858, 10827, 17367, 14230, 12933, 13139, 8690, 21090, 16524, 14553, 13736,
    11145, 8958, 23011, 17027, 14441, 14694, 15781, 15973, 8648, 13080, 45112, 10775, 16084, 13164, 10779, 10994,
    13853, 14771, 12379, 25368, 13687, 9131, 43029, 40964, 38915, 38919, 36867, 32783, 32802, 29090, 24782, 24927,
    23340, 16550, 12658, 14066, 10556, 11082, 8208, 32863, 14887, 40980, 30727, 25177, 22650, 21027, 21167, 14955,
    14254, 14330, 9191, 13172, 13307, 13438, 16063, 9080, 20848, 8341, 8737, 16489, 15470, 8346, 15319, 23219,
    18890, 15298, 12717, 8223, 12931, 13390, 8504, 10980, 15978, 8428, 38965, 14395, 14608, 13968, 11076, 11097,
    23022, 20548, 23395, 15131, 8273, 41085, 41086, 41087, 37106, 37107, 37108, 37109, 37112, 37113, 37114, 37115,
    37116, 37117, 37118, 37119, 32881, 32882, 32883, 32885, 32886, 32887, 32889, 32890, 32891, 32892, 32893, 32894,
    32895, 29147, 29149, 29150, 29151, 29152, 29154, 29155, 29156, 29157, 29158, 29159, 29160, 29161, 29162, 29163,
    29164, 29165, 29166, 29167, 29168, 29169, 29170, 29171, 29172, 29173, 29174, 29175, 29176, 29178, 29179, 29180,
    29181, 29182, 29183, 25535, 25537, 25538, 25539, 25540, 25541, 25543, 25544, 25545, 25546, 25547, 25548, 25549,
    25550, 25551, 25552, 25553, 25555, 25557, 25558, 25559, 25560, 25561, 25562, 25563, 25565, 25566, 25568, 25569,
    25571, 25574, 25575, 25576, 25577, 25578, 25580, 25581, 25582, 25583, 25584, 25585, 25586, 25587, 25588, 25590,
    25591, 25592, 25593, 25594, 25595, 25598, 25599, 20923, 21430, 21433, 21434, 21436, 21438, 21439, 21444, 21450,
    21452, 21453, 21455, 21456, 21460, 21461, 21464, 21468, 21470, 21475, 21479, 21482, 21483, 21485, 21486, 21488,
    21489, 21492, 21494, 21495, 21499, 21500, 21502, 18727, 19339, 17034, 17303, 17304, 17307, 17310, 17311, 17315,
    17323, 17331, 17340, 17381, 14220, 14265, 14275, 14280, 10522, 9193, 13436, 13874, 8967, 17354, 21145, 14455,
    10616, 8263, 30868, 28706, 24632, 23355, 8671, 16483, 14943, 14248, 8985, 25195, 23399, 15963, 10314, 8233,
    41023, 30742, 29026, 24817, 22857, 23493, 14430, 15861, 18641, 18732, 13785, 13604, 10338, 8655, 28963, 15449,
    11224, 14556, 12493, 8374, 21139, 18637, 19218, 8234, 13089, 8827, 23061, 15172, 8464, 8563, 13570, 13928,
    11065, 14693, 19311, 12755, 14653, 14905, 13212, 13876, 21332, 21422, 17228, 17233, 17238, 17254, 17271, 14083,
    14123, 14124, 14144, 25349, 21328, 21355, 21375, 21379, 21382, 18739, 14087, 14113, 9136, 26696, 26766, 22571,
    10765, 12816, 10733, 15247, 8366, 19214, 16801, 27105, 23350, 23482, 20496, 16118, 16228, 12557, 10455, 8224,
    16861, 16085, 15424, 10548, 27116, 24629, 24950, 18798, 18877, 16681, 10366, 22889, 15887, 16867, 37031, 37032,
    37036, 32875, 29126, 29137, 25298, 25337, 25339, 25340, 21334, 21350, 21368, 21377, 21400, 17255, 14095, 9138,
    15643, 8367, 8877, 11037, 13033, 13146, 13251, 10805, 41047, 36970, 26855, 22756, 22954, 20914, 15404, 8444,
    19258, 8983, 20989, 10471, 29100, 26723, 24689, 25051, 23059, 23072, 23113, 20658, 20737, 20898, 21051, 18709,
    19106, 14937, 32811, 15927, 16115, 16019, 24854, 25113, 22573, 16673, 18988, 15087, 10359, 10734, 8591, 10900,
    8590, 13903, 41032, 36976, 30853, 28735, 29052, 29056, 26719, 26725, 26795, 26939, 27002, 27049, 27099, 24880,
    24881, 24899, 25059, 25148, 25169, 25178, 22774, 22960, 23045, 23175, 23207, 23255, 20826, 21192, 18843, 19006,
    19098, 19169, 14930, 15187, 16194, 15429, 8871, 32880, 29127, 25224, 25225, 21383, 21395, 18635, 17297, 13780,
    19146, 17209, 13799, 13240, 17120, 15018, 16328, 14048, 10383, 11009, 8892, 13828, 14723, 11100, 26671, 26890,
    24717, 24811, 25029, 23518, 19009, 19037, 14336, 15880, 12710, 37003, 23490, 18563, 15941, 10755, 8838, 41011,
    26826, 18549, 15064, 8588, 23183, 16546, 15474, 15631, 12409, 12412, 12645, 20733, 20837, 10477, 8905, 20828,
    13824, 10674, 20846, 14907, 16213, 24698, 24763, 20502, 20505, 20519, 18945, 8772, 19305, 12875, 12913, 20881,
    16934, 14443, 15174, 30880, 29106, 25289, 23116, 23392, 23496, 14138, 9135, 11008, 28672, 22631, 8991, 9086,
    16009, 13340, 12560, 13709, 8633, 40963, 23008, 14704, 10368, 8201, 15518, 13634, 17066, 13292, 30744, 13256,
    14881, 15230, 13030, 13081, 8194, 24761, 19201, 19010, 14810, 28785, 10770, 10796, 25144, 19257, 16400, 13605,
    21183, 13180, 13859, 18858, 12474, 13389, 19369, 14158, 8777, 23172, 18982, 14627, 12517, 10492, 8285, 19181,
    18594, 8573, 13050, 13084, 13244, 19060, 15310, 15637, 15675, 15826, 16209, 12365, 12445, 10261, 10415, 9049,
    49160, 27033, 23447, 15825, 9024, 23160, 19113, 16782, 17029, 40961, 40972, 40974, 38924, 36999, 34839, 32781,
    20573, 19223, 12308, 8992, 14281, 17153, 22818, 23193, 13714, 19073, 16210, 12932, 8745, 16093, 15273, 15988,
    12942, 13764, 14863, 30854, 15296, 10466, 10563, 8771, 13774, 24861, 13633, 38982, 36955, 16788, 11213, 15512,
    12423, 17360, 32789, 32819, 14650, 36868, 28923, 27124, 24651, 24983, 25184, 22675, 22923, 20623, 20649, 20691,
    20897, 21054, 18630, 16666, 16864, 16883, 14342, 14654, 12569, 13381, 13693, 16112, 13816, 8611, 18508, 15348,
    15459, 15616, 8330, 25214, 10269, 15754, 15399, 16374, 13998, 15532, 25036, 25108, 22911, 22965, 15331, 10263,
    8209, 11102, 21426, 12880, 14962, 26896, 24747, 16454, 19244, 22604, 20713, 13276, 12730, 15232, 13997, 37005,
    30739, 28685, 28734, 28849, 29093, 26837, 24681, 24693, 22597, 18741, 28825, 26842, 20686, 21076, 18578, 14944,
    15312, 8960, 15472, 8710, 19295, 17073, 32792, 24736, 20939, 18503, 18682, 16433, 16784, 14481, 14780, 15614,
    8210, 25596, 14234, 14272, 14286, 9189, 12946, 41074, 37030, 30756, 24658, 24682, 21411, 14115, 11250, 36972,
    28796, 26910, 24961, 22917, 23341, 20592, 20688, 21062, 14649, 15439, 10327, 8459, 10943, 15457, 18874, 15385,
    12703, 22737, 20729, 8753, 8952, 25061, 18586, 16639, 24995, 8515, 14977, 8560, 29000, 16906, 39003, 18524,
    20631, 24860, 19308, 17190, 12813, 14264, 15851, 12588, 23486, 10834, 26630, 21184, 15396, 12767, 12850, 17204,
    13929, 13844, 13906, 16102, 8494, 15649, 19108, 13962, 20574, 18574, 14485, 12677, 16475, 16217, 12296, 13035,
    13303, 16082, 14360, 13640, 36897, 28783, 28897, 28925, 28976, 29011, 29089, 26637, 26761, 26882, 20638, 25112,
    23129, 23347, 23474, 21198, 19042, 15650, 15997, 13133, 10370, 8695, 29008, 22808, 20629, 13689, 16282, 32776,
    17047, 13895, 21362, 21363, 21394, 21408, 15933, 8584, 23157, 19065, 10673, 12522, 39000, 23345, 20866, 17064,
    14476, 15389, 15558, 16286, 34842, 32834, 24889, 22777, 12293, 13504, 14061, 11203, 16893, 17176, 14516, 15356,
    9194, 9123, 25323, 23142, 15279, 16173, 12929, 20614, 18712, 30841, 26703, 24930, 24938, 25037, 25171, 23041,
    23105, 23138, 23417, 23488, 23489, 20552, 21262, 18618, 15090, 10577, 10785, 8282, 22782, 18761, 14487, 15982,
    49170, 25317, 25318, 21352, 17237, 17265, 17295, 17296, 13209, 13222, 36928, 36938, 34860, 23544, 20516, 20779,
    18619, 16901, 14501, 15169, 15255, 16016, 10535, 8698, 28681, 28696, 28736, 24713, 25142, 23316, 21002, 19120,
    15323, 37059, 14063, 28910, 24830, 24997, 22635, 22676, 23450, 20598, 18548, 18967, 16898, 14725, 15036, 15276,
    15569, 8935, 22820, 38961, 36911, 30864, 28852, 22853, 22936, 20821, 18659, 18931, 15619, 25350, 14093, 14102,
    14150, 9145, 34831, 28945, 24695, 24776, 24859, 22758, 23185, 23227, 23276, 20942, 21293, 18529, 18547, 18611,
    19179, 19326, 16437, 16945, 17078, 17138, 14482, 15678, 15782, 13077, 13165, 13541, 13701, 10321, 10648, 43068,
    37044, 37085, 37088, 37091, 37093, 30906, 30907, 30915, 30916, 30929, 30935, 30956, 30961, 30969, 30973, 25388,
    25395, 25407, 25409, 25414, 25485, 25504, 25507, 25526, 19345, 19360, 19382, 19386, 19410, 19425, 14198, 14203,
    14211, 16162, 16274, 13450, 26872, 24937, 22838, 23416, 23432, 19008, 19290, 16470, 16699, 17084, 14366, 13009,
    21319, 19049, 21126, 18912, 16553, 16618, 14860, 12716, 13936, 21117, 15541, 45083, 45099, 45108, 40970, 25211,
    23446, 20646, 18471, 16388, 16449, 8458, 8874, 13842, 9054, 16663, 14493, 16775, 12892, 23241, 23272, 15281,
    25287, 23550, 21357, 21417, 21418, 9165, 19291, 8759, 11002, 26815, 21264, 10538, 25212, 19199, 14969, 15241,
    15245, 15874, 8809, 8987, 22829, 16306, 13524, 14603, 38967, 32788, 32838, 32858, 24670, 25220, 25221, 23478,
    18636, 16837, 14428, 14518, 14705, 43035, 38927, 18750, 16544, 15181, 15644, 12359, 12839, 13735, 21215, 13543,
    16460, 21321, 13124, 15958, 25311, 13889, 19236, 13678, 29066, 15762, 12384, 10529, 8843, 39030, 21247, 18943,
    16419, 15024, 15411, 12955, 8705, 14322, 8896, 13812, 15414, 10546, 8912, 10671, 34857, 32832, 30733, 28725,
    27073, 25141, 20494, 18447, 18996, 14612, 15032, 15250, 15444, 15784, 12524, 10704, 13706, 16237, 13195, 34923,
    24770, 14055, 11080, 13188, 13724, 28865, 22764, 19074, 12963, 22622, 22882, 20585, 20651, 18995, 16657, 15130,
    12674, 10868, 10874, 8260, 23358, 16595, 14752, 13585, 13892, 8287, 30731, 26644, 15067, 25273, 21330, 10976,
    9053, 13159, 13288, 30766, 23199, 21128, 11104, 11168, 16716, 13343, 25258, 23514, 38973, 37008, 30842, 22927,
    23422, 21060, 19316, 16428, 14412, 15066, 26669, 43024, 25347, 22703, 20601, 20820, 10758, 19263, 13396, 9028,
    16188, 25343, 17226, 17227, 17235, 17239, 14092, 9149, 20726, 16994, 10378, 10442, 9126, 34926, 30810, 28871,
    24885, 22951, 22970, 21087, 21284, 21312, 18468, 16480, 26681, 26881, 24646, 24765, 23501, 20555, 20664, 20710,
    21044, 21049, 18445, 18830, 19159, 12691, 28939, 22603, 26990, 13103, 19063, 15352, 10585, 15436, 10913, 43043,
    19282, 12950, 11258, 8479, 49157, 47128, 47135, 36896, 26978, 25020, 21204, 10274, 9076, 16811, 12670, 8521,
    45074, 28756, 20887, 21307, 18711, 19278, 9098, 16750, 13131, 13731, 15708, 21472, 16420, 15845, 10845, 8315,
    8841, 11007, 13306, 10898, 11235, 8465, 12360, 15681, 13581, 10829, 14276, 23088, 21168, 16604, 16763, 17032,
    15500, 15772, 12802, 12829, 10357, 10926, 39017, 32856, 30752, 30783, 28746, 28848, 28862, 29070, 29084, 26870,
    26894, 27019, 24839, 22875, 23080, 23104, 23318, 21245, 14532, 16128, 10621, 13740, 49152, 40977, 28751, 20893,
    21240, 16751, 14952, 16226, 15565, 23120, 23258, 19099, 16670, 15524, 16046, 12408, 15482, 15023, 8544, 14824,
    15487, 12488, 12581, 13984, 10289, 8384, 13223, 13439, 13338, 8339, 8842, 24638, 24664, 22722, 15259, 12627,
    11174, 49167, 38935, 38947, 38951, 38963, 38983, 38987, 32778, 30784, 30789, 30834, 28805, 28876, 26835, 14841,
    14859, 16677, 8829, 13447, 13135, 21317, 19299, 13232, 16145, 26771, 22653, 16779, 16871, 15416, 15670, 12458,
    17002, 13667, 8727, 15961, 19204, 45079, 8970, 8238, 16368, 13967, 14002, 11051, 43045, 25227, 22543, 18839,
    15794, 41058, 21108, 18795, 19075, 17092, 17105, 14405, 14542, 14976, 15425, 15617, 12322, 30723, 20794, 16825,
    8654, 16909, 12311, 12626, 10313, 8868, 22796, 16882, 15061, 47111, 24597, 16134, 15109, 49178, 37063, 30934,
    19368, 8775, 23187, 15798, 10460, 23331, 8736, 41079, 32866, 26807, 22709, 12725, 12852, 10416, 10867, 15528,
    16214, 43034, 15280, 12682, 10513, 8982, 13143, 36962, 34836, 30838, 28845, 29057, 24709, 24912, 25071, 25162,
    25236, 25244, 22533, 22714, 16997, 15521, 15847, 15948, 12415, 8327, 8442, 16846, 15297, 15421, 16290, 12518,
    19068, 16452, 10402, 10769, 8846, 22947, 23130, 18480, 19124, 15334, 15381, 16196, 8348, 26714, 27029, 25049,
    20780, 18629, 21443, 21481, 17342, 17383, 15194, 8798, 17060, 16314, 10555, 8553, 8653, 8664, 21227, 13193,
    10395, 23257, 38988, 28891, 28994, 24829, 24986, 22728, 22825, 23086, 23095, 20504, 21161, 18662, 18833, 18834,
    16703, 14499, 16116, 10715, 16292, 13136, 43044, 19270, 12919, 14057, 11184, 13130, 13688, 8336, 36936, 29041,
    18987, 12535, 17380, 20760, 21277, 8812, 16910, 22873, 17401, 20922, 21131, 10668, 15498, 21026, 16053, 16105,
    8984, 13491, 20669, 10339, 14634, 13341, 16939, 8589, 13445, 45097, 43019, 40999, 24974, 14689, 15859, 8677,
    15720, 15937, 12654, 20640, 21267, 18714, 10711, 8349, 13229, 16500, 13075, 26679, 20650, 14699, 14925, 10400,
    14103, 13177, 13355, 13471, 9058, 8881, 19252, 23026, 20971, 14741, 15140, 8300, 13407, 25185, 22658, 15341,
    10821, 8604, 11012, 9087, 16234, 11223, 9118, 8730, 27120, 17008, 15603, 15661, 12582, 20781, 21231, 8949,
    13551, 9012, 10962, 13327, 14316, 9183, 22652, 16156, 16211, 13636, 13833, 13199, 9146, 9008, 18465, 29072,
    26840, 26912, 22990, 23043, 20524, 20653, 20974, 19025, 19066, 16987, 14442, 14511, 10390, 12841, 10988, 26915,
    24684, 16113, 32823, 8685, 22609, 16705, 13268, 28829, 23177, 18946, 16746, 16794, 17214, 14448, 15005, 16140,
    12856, 13126, 8817, 28807, 26927, 25031, 23344, 21003, 16660, 16761, 14379, 14465, 14517, 14744, 12304, 12446,
    12563, 23364, 18639, 18822, 18899, 16445, 16600, 14617, 15479, 15799, 8193, 15655, 28763, 26770, 24768, 22757,
    23180, 20501, 20539, 20560, 20704, 21318, 18585, 18647, 16583, 16724, 15367, 11135, 8487, 8915, 18483, 24787,
    10549, 20495, 16461, 10365, 21269, 10679, 24652, 22692, 22907, 23176, 20958, 18806, 15488, 15489, 16051, 8200,
    15435, 15037, 12642, 10815, 12806, 12990, 16471, 20918, 20944, 19088, 13921, 10362, 11089, 8261, 23027, 20567,
    14812, 15928, 9032, 13908, 24773, 23137, 21143, 18723, 18871, 16491, 15496, 11099, 28995, 16241, 12660, 8309,
    22999, 8914, 30859, 26853, 24792, 24870, 24874, 24976, 25094, 18437, 20957, 15640, 8706, 14062, 22798, 18520,
    18870, 14711, 10252, 39027, 36971, 25093, 22929, 23322, 20693, 20787, 20943, 21151, 21226, 18715, 18952, 15152,
    15577, 13285, 21057, 17023, 15239, 16299, 12310, 8450, 17166, 10921, 23541, 11044, 8856, 13230, 13637, 13641,
    13738, 10272, 24791, 14965, 16034, 16047, 8481, 14263, 16493, 12738, 17089, 10904, 15574, 43046, 43070, 37098,
    37102, 30950, 25370, 25452, 25456, 25494, 19366, 19388, 19392, 19428, 14160, 14194, 14195, 30732, 24805, 24815,
    15113, 12314, 26999, 24635, 20931, 19086, 10707, 8606, 14353, 15291, 49171, 37025, 37026, 25329, 21367, 9152,
    45073, 43017, 43018, 43042, 40985, 40988, 40989, 34825, 34849, 32853, 28726, 28771, 28786, 29060, 26731, 26827,
    24741, 24891, 22637, 22783, 20734, 20776, 21200, 18860, 19140, 16262, 10950, 8409, 13207, 24614, 22872, 20982,
    12568, 43015, 36930, 36952, 36987, 34911, 30736, 26763, 27031, 22976, 23408, 15497, 10285, 17161, 14857, 12961,
    8602, 16114, 15509, 15123, 8628, 23032, 21008, 15923, 16043, 22624, 8643, 12968, 21396, 13225, 22866, 10979,
    14999, 36992, 13375, 14839, 16221, 30848, 26762, 27034, 24666, 21435, 18441, 19203, 17309, 15345, 14253, 10403,
    10975, 8305, 20584, 8516, 15823, 11063, 9127, 34916, 23122, 23246, 19134, 16723, 14770, 15227, 10509, 26888,
    18917, 15214, 15350, 14001, 38986, 34883, 27119, 24679, 23394, 21429, 21493, 16146, 8615, 15236, 14306, 25150,
    15369, 13252, 13494, 13590, 15544, 40998, 28885, 13310, 28815, 16365, 45090, 16907, 14611, 15083, 15953, 10865,
    30763, 30850, 24637, 22801, 22835, 22928, 23001, 20860, 21118, 19152, 16809, 17077, 14373, 14520, 14534, 16279,
    8957, 28831, 23079, 8360, 17394, 14282, 14427, 16220, 39005, 20720, 14406, 12303, 12489, 13515, 13822, 22920,
    14991, 16180, 18994, 14731, 12464, 8337, 8693, 17403, 12797, 22795, 13387, 21428, 21448, 28887, 27086, 18924,
    8306, 29030, 26868, 24755, 24834, 22807, 14749, 15051, 16052, 12427, 12689, 12692, 10601, 8269, 14005, 26808,
    24996, 22778, 16466, 15397, 15060, 13421, 25597, 21473, 13129, 13330, 13482, 14325, 9213, 45093, 8756, 15388,
    12801, 9017, 19331, 13066, 49164, 34864, 24958, 15753, 8854, 25194, 23170, 20594, 14901, 16005, 12833, 10245,
    14528, 12989, 17147, 38941, 21235, 10999, 14706, 15343, 15513, 15901, 15573, 26922, 10490, 8279, 36895, 24683,
    15150, 25567, 23047, 21084, 19189, 15531, 15774, 15884, 16171, 16240, 8249, 8732, 16992, 14317, 9206, 15238,
    8401, 43026, 23271, 18990, 16937, 17036, 17040, 14671, 15773, 16004, 12591, 12701, 14318, 8751, 38936, 36924,
    16911, 20991, 15135, 11257, 22684, 10949, 12986, 13175, 14033, 10954, 16590, 14344, 14626, 8968, 19053, 15608,
    13347, 16790, 9139, 20648, 21053, 16559, 13855, 9062, 25130, 11024, 23410, 17292, 34824, 32816, 28826, 20743,
    16780, 15729, 12646, 49168, 40962, 36949, 28704, 9037, 30830, 30883, 26966, 27026, 24863, 22537, 23494, 20786,
    20847, 18446, 18873, 25292, 13866, 14108, 14135, 9159, 16978, 43030, 41038, 38952, 38991, 36914, 36918, 36946,
    36950, 36954, 36975, 36979, 37020, 34818, 34882, 34908, 34919, 34928, 34936, 32818, 32831, 32841, 30781, 30794,
    30807, 30808, 30814, 30816, 30836, 30869, 30874, 28727, 28729, 28753, 28762, 28820, 28911, 28938, 28993, 28998,
    28999, 29002, 29019, 29027, 29073, 29076, 29083, 29086, 29088, 29092, 26683, 26684, 26769, 26863, 26874, 26897,
    26953, 26968, 26989, 26996, 27011, 27027, 27038, 27080, 27094, 27102, 27107, 27113, 24724, 24771, 24796, 24813,
    24894, 24915, 24969, 25021, 25026, 25050, 25081, 25085, 25109, 25167, 25170, 25173, 25197, 25209, 25218, 22711,
    22729, 22772, 22790, 22880, 22967, 23021, 23100, 23119, 23136, 23139, 23174, 23202, 23239, 23279, 23289, 23304,
    23313, 23378, 23381, 23390, 23393, 23396, 23403, 23426, 23430, 23431, 23438, 23460, 23470, 23485, 23495, 23499,
    23500, 20678, 20694, 20747, 20755, 20936, 21031, 21041, 21093, 21107, 21148, 21155, 21171, 21175, 21187, 21195,
    18601, 18614, 18799, 18910, 18938, 18983, 18989, 19080, 19083, 19129, 19137, 19157, 19219, 19226, 15764, 15976,
    16203, 15175, 15223, 10401, 8974, 21478, 13331, 13862, 13784, 29133, 25357, 19249, 12927, 13402, 16150, 25534,
    21477, 9004, 16523, 16620, 14979, 13460, 19076, 19131, 14917, 13626, 15849, 10294, 15824, 8448, 25531, 14173,
    14186, 14187, 10709, 8906, 13558, 13744, 8936, 39016, 36919, 27076, 20882, 20956, 15266, 15305, 19056, 16434,
    16535, 13643, 8764, 29117, 21327, 20993, 18936, 16632, 14374, 12331, 10609, 15119, 11006, 10603, 12966, 24985,
    12546, 41042, 10489, 29054, 21165, 21221, 18464, 19023, 16317, 12666, 10633, 11028, 13569, 27133, 23439, 20499,
    16304, 13583, 30759, 30831, 24849, 24879, 23051, 15986, 12579, 12854, 12994, 13571, 10298, 10411, 10875, 10923,
    10940, 8253, 8552, 12877, 19256, 11245, 9109, 15934, 9111, 28981, 26716, 23453, 20532, 20994, 14464, 14619,
    15635, 12420, 13657, 10542, 8447, 12616, 18605, 13699, 27066, 16918, 21325, 21342, 12498, 32873, 25295, 21372,
    12511, 8476, 16913, 19239, 30857, 14862, 8410, 13732, 28776, 28983, 26633, 15658, 12620, 10432, 8462, 25237,
    40993, 28748, 29007, 23125, 15853, 10658, 18809, 16865, 14715, 19178, 12437, 11243, 28872, 26954, 27025, 22742,
    22932, 22985, 23308, 23377, 23441, 20498, 20630, 19059, 16646, 15078, 16688, 8334, 19231, 15261, 24735, 20675,
    16868, 12890, 10291, 8663, 17072, 8889, 13457, 13529, 25327, 10461, 13725, 17339, 9180, 34871, 19225, 12480,
    12534, 15578, 10309, 8324, 14638, 15201, 8303, 8959, 41063, 26876, 19104, 16515, 16850, 15752, 15926, 16341,
    12330, 12749, 16752, 12332, 10360, 8762, 24605, 15925, 12515, 13408, 10762, 15865, 29120, 29139, 25315, 21080,
    16372, 13955, 24708, 12799, 15629, 12426, 10808, 18640, 18791, 16642, 9212, 14829, 16407, 16543, 17211, 8757,
    11263, 9114, 12528, 22673, 15338, 8932, 16786, 13042, 26655, 8908, 11222, 17075, 17115, 17272, 21283, 19272,
    11087, 19255, 13073, 49154, 45057, 38932, 36900, 34856, 30737, 30753, 28718, 26698, 22716, 20998, 21309, 18485,
    18700, 19333, 16536, 16831, 16916, 15017, 25253, 16768, 8876, 28959, 26949, 24581, 22599, 22683, 22841, 18814,
    19081, 16674, 16975, 14446, 14524, 14783, 14106, 17391, 20801, 21124, 18498, 18710, 14696, 14761, 11101, 9092,
    16457, 15321, 10494, 17067, 16295, 8917, 12484, 11071, 17107, 34890, 26704, 27061, 24866, 17378, 16233, 16273,
    12915, 9184, 17148, 12354, 11088, 26895, 23283, 20685, 20807, 15723, 8228, 13238, 13334, 13773, 9079, 41071,
    29122, 14956, 16227, 20868, 12453, 14491, 8911, 21260, 15091, 13753, 10773, 28884, 23421, 19097, 19330, 14624,
    14763, 15326, 15684, 12434, 14077, 8545, 15593, 8237, 17200, 15875, 10607, 12849, 14922, 12847, 13979, 12960,
    12744, 11165, 18451, 13110, 13761, 22641, 28919, 29003, 29049, 26706, 22938, 20531, 21120, 14576, 16018, 12481,
    17017, 15484, 13005, 9067, 14850, 8429, 14401, 16355, 12680, 24585, 20626, 8636, 23428, 20523, 20732, 18654,
    15596, 16126, 21451, 19193, 12987, 13404, 9021, 12881, 16585, 16989, 14372, 14497, 24800, 13718, 14094, 10878,
    8715, 43057, 37047, 37051, 37058, 37076, 37096, 30936, 30952, 25372, 25375, 25384, 25411, 25412, 25490, 25522,
    19390, 19403, 19408, 14177, 13115, 13722, 14331, 41046, 36959, 26628, 26828, 26940, 23107, 23551, 20554, 20616,
    20719, 18545, 18652, 18859, 16488, 14563, 15917, 12530, 26687, 16668, 14484, 28682, 15197, 13537, 8999, 39006,
    16878, 15664, 28890, 29107, 26643, 24923, 25025, 22753, 20606, 18914, 16889, 15642, 34943, 26663, 24696, 21244,
    10508, 15807, 15358, 8576, 16039, 45116, 45119, 29110, 29116, 29128, 25179, 22556, 23189, 18591, 9156, 16967,
    8865, 8707, 19286, 17016, 17081, 13411, 13653, 10791, 8328, 14070, 28797, 26715, 24850, 24851, 24948, 23367,
    21032, 21122, 18476, 15623, 28988, 20841, 12883, 20748, 19185, 15222, 12435, 20617, 10654, 16098, 12887, 14037,
    13915, 8680, 8275, 25252, 47110, 18439, 12566, 32782, 13373, 23309, 14724, 14980, 9041, 24767, 20858, 18622,
    18689, 16443, 14565, 13403, 32862, 32876, 12988, 13105, 13290, 13702, 8852, 21217, 15371, 25097, 18649, 18865,
    19019, 15026, 20752, 14032, 9203, 32842, 26752, 24660, 24663, 22726, 20572, 18626, 8925, 20537, 20757, 38916,
    36865, 36875, 24675, 22876, 12584, 17320, 14269, 45096, 43041, 13063, 14642, 22897, 11219, 10851, 10418, 14257,
    9202, 12952, 8582, 13798, 13012, 13117, 30793, 28768, 15873, 10345, 26834, 25203, 19126, 10678, 49163, 13544,
    8971, 21306, 16409, 8500, 15077, 8863, 17033, 17404, 14486, 18477, 18531, 12809, 38978, 30761, 30886, 28675,
    28694, 15778, 10579, 13646, 24810, 12443, 13916, 15165, 15685, 10726, 24578, 20697, 20775, 20925, 16401, 15971,
    21038, 13650, 13749, 10839, 13664, 23002, 18525, 12735, 41048, 37000, 30748, 30801, 30843, 30872, 28731, 28732,
    28811, 28824, 28948, 29025, 29074, 26690, 26755, 26791, 26860, 26992, 27035, 27058, 27077, 27085, 27098, 24720,
    24901, 24905, 24965, 24979, 25027, 25091, 25180, 25192, 25202, 22617, 22730, 22800, 22908, 22943, 23033, 23065,
    23078, 23090, 23168, 23195, 23216, 23375, 23455, 23458, 23468, 21036, 21154, 21164, 18907, 18937, 18958, 19041,
    19121, 19158, 19167, 19170, 14913, 15768, 28979, 32793, 22941, 18976, 15092, 8801, 11187, 9122, 28693, 11179,
    11202, 36997, 28677, 28712, 28834, 28874, 24672, 24706, 24820, 22532, 22715, 22809, 23020, 23031, 23205, 20503,
    20517, 21314, 18493, 18495, 16554, 10891, 45071, 25068, 10819, 8598, 14933, 9173, 13846, 30839, 28817, 29059,
    27091, 24896, 23310, 20969, 21280, 19318, 14352, 14657, 15337, 16318, 12305, 13592, 34833, 21387, 16494, 10698,
    38933, 36864, 36866, 23184, 23526, 20953, 18552, 19164, 16384, 16555, 16319, 21404, 14132, 13520, 43020, 30754,
    28689, 28850, 24602, 20550, 8711, 9185, 32785, 32840, 30792, 28686, 28808, 24743, 24987, 22581, 23307, 16976,
    14827, 14866, 12319, 11147, 13021, 16439, 21052, 18620, 16792, 12819, 10346, 8400, 13981, 25017, 13216, 8433,
    8451, 15621, 10552, 10752, 8365, 26809, 39014, 28683, 24855, 25024, 22723, 18896, 15939, 16027, 23414, 20793,
    18716, 16545, 14929, 15056, 15714, 12721, 10823, 11057, 8644, 8739, 17106, 45069, 47108, 24633, 16006, 12577,
    22721, 16804, 14745, 20788, 14151, 13078, 17129, 11242, 14625, 22596, 23145, 15943, 12765, 10744, 8431, 43021,
    10849, 8559, 8886, 18490, 14884, 25263, 21484, 41034, 14549, 13048, 13122, 13536, 30771, 22717, 22957, 20952,
    16008, 13836, 43064, 30912, 19400, 24608, 22918, 8943, 10318, 8601, 26931, 24788, 21178, 15938, 34921, 12761,
    30876, 14237, 13051, 28701, 15219, 12776, 10857, 22578, 23222, 15854, 19217, 28905, 16378, 8738, 34838, 13304,
    10817, 8505, 27057, 16568, 15271, 18546, 13008, 36934, 28772, 24762, 24868, 41013, 32771, 28847, 28889, 26803,
    26830, 24685, 25007, 22930, 28769, 24954, 22662, 23275, 21078, 21249, 19324, 16479, 16711, 17000, 14350, 15068,
    15990, 12334, 14078, 11126, 9161, 13015, 11261, 13478, 8345, 11038, 8779, 34901, 28868, 22686, 20692, 17079,
    14378, 12886, 14217, 14221, 14585, 13607, 11197, 11248, 22575, 14801, 8321, 36871, 25103, 23434, 19187, 14806,
    15178, 15244, 15393, 15537, 15995, 16834, 10388, 10502, 13454, 23531, 18478, 9071, 13094, 16204, 14125, 14134,
    13018, 13169, 15301, 15796, 15940, 16890, 30730, 25572, 21440, 21459, 17312, 14302, 9046, 47124, 38962, 34850,
    30865, 23164, 30776, 20512, 15318, 15697, 17325, 10675, 25095, 16455, 14470, 14641, 14685, 14697, 15673, 12707,
    10325, 25264, 15076, 24745, 18526, 14681, 10931, 25011, 21159, 18776, 12870, 10423, 14847, 13233, 18971, 17068,
    16329, 12947, 13270, 12871, 8995, 10944, 10438, 10937, 8371, 21191, 16565, 14683, 15307, 14817, 8435, 28912,
    22885, 20885, 14355, 14407, 14449, 10443, 22627, 15786, 13114, 15931, 12780, 15929, 12696, 12783, 10374, 28676,
    8857, 18781, 14473, 8440, 23140, 15606, 14899, 10611, 15162, 47125, 47127, 13823, 23386, 14883, 16036, 16169,
    8566, 26666, 18743, 21471, 14256, 14315, 18926, 18941, 15888, 16346, 13935, 13057, 12930, 30826, 22874, 23190,
    20637, 19160, 14550, 12411, 15771, 41039, 13713, 38953, 36913, 26730, 12681, 10828, 8899, 24945, 23014, 16026,
    12499, 23311, 21236, 16205, 10424, 10930, 8379, 13670, 13781, 13295, 14828, 12507, 34866, 28724, 28765, 25196,
    18455, 18923, 16899, 17157, 14435, 16298, 16322, 13934, 10822, 11091, 13566, 9192, 21462, 19342, 17146, 17196,
    37037, 37039, 32868, 32877, 26650, 27081, 27100, 25326, 25331, 22738, 21255, 21374, 21389, 21390, 21415, 18854,
    19102, 17243, 17281, 17302, 15701, 14107, 14122, 14129, 8290, 9132, 21386, 14018, 13473, 20520, 18579, 25226,
    22584, 22833, 12318, 10449, 10642, 28692, 10837, 15950, 12615, 10543, 8342, 32817, 39007, 26983, 22836, 20558,
    21035, 16576, 15122, 28977, 24988, 25016, 22540, 22914, 20816, 18570, 14415, 14832, 13926, 43009, 25147, 22561,
    23092, 20565, 16426, 14540, 14751, 8806, 37011, 15080, 15543, 15561, 16245, 10487, 10884, 8195, 17326, 14311,
    9211, 30802, 27097, 18499, 10525, 47112, 13597, 41057, 32809, 28853, 28881, 28904, 28908, 28972, 26740, 26857,
    27090, 24865, 24911, 24924, 25182, 22625, 23085, 23273, 20913, 18735, 18782, 15964, 15984, 12820, 8221, 16028,
    30775, 30800, 28996, 26873, 25133, 23048, 18985, 16040, 13815, 10741, 17372, 28895, 32825, 28774, 28898, 25084,
    16100, 14303, 13887, 12417, 13099, 8726, 24576, 14950, 15491, 12792, 10300, 8217, 8799, 23149, 23230, 15584,
    15733, 8571, 27065, 14411, 12562, 12649, 8310, 19273, 25269, 34863, 29005, 26993, 24732, 22924, 21130, 14755,
    12509, 10554, 9105, 21360, 16859, 11256, 24702, 21261, 16563, 14938, 20591, 12617, 16200, 43049, 37055, 37073,
    37075, 30902, 30939, 30975, 25394, 25422, 25423, 25437, 25438, 25449, 25451, 25453, 25455, 25457, 25460, 25479,
    25491, 25502, 25515, 19385, 19417, 19423, 19433, 14176, 14180, 14189, 14192, 19212, 13975, 11061, 13468, 9148,
    30877, 23075, 23147, 20741, 21099, 18820, 19136, 15743, 10311, 20997, 18769, 16344, 12673, 37048, 30891, 30901,
    30944, 30955, 30971, 25389, 25393, 25416, 25419, 25431, 25440, 25447, 25448, 25475, 25481, 25500, 19347, 19358,
    19372, 19375, 19398, 19412, 19421, 19437, 19442, 14153, 14167, 14169, 14182, 14196, 25042, 23109, 19153, 19210,
    16641, 16766, 14628, 14897, 12374, 12891, 10256, 10656, 10731, 8678, 15183, 15636, 12553, 8264, 10861, 18666,
    13976, 14043, 11055, 17127, 16323, 16376, 16589, 12485, 20699, 14648, 12740, 26811, 24929, 18608, 8525, 13398,
    25303, 21346, 21348, 21353, 21391, 21392, 21424, 17247, 17248, 17252, 17268, 13525, 13973, 22984, 23284, 21138,
    21177, 18675, 16391, 16446, 14471, 20559, 18827, 18540, 14967, 21315, 23254, 16990, 14742, 14795, 13267, 10852,
    8697, 22776, 23334, 20725, 21158, 21224, 19034, 19047, 16542, 14818, 8564, 16327, 11094, 8830, 15828, 13121,
    13286, 16075, 13038, 16021, 10435, 23534, 17114, 16333, 45117, 29135, 29142, 24779, 23106, 16650, 10991, 15157,
    13003, 13613, 13156, 13247, 12639, 13412, 11190, 11234, 13369, 41019, 14383, 19116, 13894, 13065, 12812, 15430,
    8542, 39026, 28914, 26883, 26980, 16148, 11207, 45056, 36922, 28828, 28875, 28896, 28928, 26765, 26821, 26851,
    27030, 27111, 24688, 24793, 24819, 25138, 25239, 25259, 22566, 22642, 22959, 19303, 19325, 17192, 14959, 8665,
    20766, 8990, 17013, 14622, 8414, 26930, 16530, 15999, 8657, 8717, 13352, 8788, 22766, 19105, 14474, 14853,
    16077, 12370, 12596, 45115, 45118, 41054, 29123, 25284, 25301, 17278, 14112, 9154, 22906, 19165, 15170, 10735,
    13492, 10799, 16608, 15780, 12309, 14025, 13017, 13241, 16949, 8919, 14236, 10759, 13481, 25366, 16707, 16863,
    14338, 15029, 8594, 28873, 23108, 15008, 10299, 36898, 36912, 17280, 16076, 9153, 13573, 41006, 14885, 14307,
    8277, 10334, 8855, 8975, 14523, 12299, 8714, 14997, 8244, 10760, 8575, 28799, 22871, 18829, 14530, 15004,
    16229, 10278, 8473, 23087, 19067, 9167, 41076, 28723, 9050, 13363, 10688, 47105, 14946, 15837, 16575, 18893,
    21153, 10939, 8511, 23096, 20579, 18717, 16936, 12624, 26846, 18650, 15001, 15546, 15742, 12711, 20549, 10496,
    17336, 14268, 9025, 40996, 34855, 45110, 38923, 37010, 37016, 34869, 29024, 14970, 8810, 34889, 14595, 8531,
    26793, 8951, 16866, 12937, 12958, 16770, 12629, 12766, 14051, 10576, 13323, 18583, 11164, 11209, 14409, 16307,
    15680, 13746, 12789, 10636, 16430, 17124, 11039, 9121, 11010, 9084, 14494, 13720, 47129, 16337, 22992, 8670,
    24667, 15797, 8794, 16208, 13356, 24981, 23049, 18556, 15834, 10807, 8205, 15342, 16119, 8927, 21292, 14659,
    8362, 15739, 10544, 10982, 8340, 8568, 10763, 17365, 10351, 22719, 22886, 22988, 20612, 13178, 13708, 10684,
    16123, 32775, 30726, 16507, 14329, 17316, 14436, 11177, 13283, 20735, 21343, 21378, 9157, 25207, 13682, 45062,
    45072, 16704, 8610, 8844, 13185, 13621, 10561, 8884, 15258, 10859, 10938, 37061, 37064, 37077, 37078, 37090,
    30900, 25404, 25420, 25473, 25477, 25519, 19389, 19450, 13947, 11157, 13461, 14218, 37006, 18646, 16652, 10491,
    11196, 16774, 10586, 10951, 13032, 10864, 16334, 16383, 11023, 13413, 8786, 15527, 15870, 8460, 10803, 32770,
    17046, 14554, 15003, 15445, 15502, 16023, 12327, 12739, 10277, 13620, 12796, 29017, 26810, 26988, 24607, 24642,
    24798, 24845, 25175, 25260, 25265, 25267, 22639, 22657, 22768, 20599, 20715, 18760, 18762, 19297, 16715, 14789,
    14971, 15132, 15316, 16170, 18959, 14921, 15038, 15327, 12386, 12884, 26708, 26746, 28710, 28713, 28882, 27040,
    27043, 22583, 22677, 23301, 20582, 20842, 21265, 18610, 18708, 18771, 18968, 16579, 16700, 14635, 15209, 16002,
    12325, 10885, 15692, 8482, 22816, 21242, 15628, 12358, 15651, 8785, 14243, 25149, 16594, 14467, 14758, 23325,
    30758, 45067, 30871, 28851, 9128, 12729, 10693, 14931, 13279, 13470, 8845, 13191, 36944, 28806, 29104, 8506,
    21162, 21222, 16616, 16745, 15704, 18966, 18459, 28684, 22792, 15072, 13945, 13132, 13405, 14463, 13472, 13496,
    13582, 16748, 13703, 19224, 21285, 19281, 11149, 8822, 8891, 9160, 13734, 36969, 30840, 28827, 26758, 26919,
    24925, 22568, 20744, 20975, 18737, 16508, 17003, 14529, 14733, 14957, 15441, 26647, 22799, 14678, 10881, 8454,
    20561, 16511, 12538, 12634, 8291, 26848, 24731, 20950, 18949, 10315, 9107, 28755, 13319, 13758, 11194, 26664,
    22869, 23203, 20825, 20920, 16802, 15450, 12592, 22953, 18842, 13022, 14110, 12302, 34915, 32805, 22689, 16201,
    12531, 13996, 10353, 24645, 13210, 8466, 8733, 14502, 13137, 10957, 18736, 16984, 10894, 36941, 27001, 20668,
    12561, 8735, 14746, 26711, 22626, 22761, 23052, 9133, 14073, 13627, 8972, 19084, 16858, 15612, 40984, 39008,
    25044, 11188, 25233, 13041, 23469, 14998, 10946, 8724, 23513, 13939, 11077, 16197, 16676, 15269, 16348, 8522,
    9055, 15103, 8833, 13521, 34930, 18674, 19315, 23058, 20754, 21105, 14296, 14543, 14082, 8895, 8763, 19313,
    17207, 11113, 11200, 24592, 24726, 23029, 20978, 14953, 8623, 22811, 30721, 23250, 20482, 20507, 20813, 18566,
    13069, 12758, 10273, 15627, 16342, 25359, 16658, 16218, 8986, 16184, 16832, 17212, 14426, 15000, 15166, 15716,
    16361, 12335, 12550, 13487, 18849, 18925, 12583, 13104, 11259, 12496, 25354, 21410, 20762, 20824, 21202, 19142,
    16771, 16826, 16900, 14734, 14784, 15965, 16149, 12356, 12559, 13416, 24680, 25201, 23372, 11027, 16875, 12999,
    32777, 27054, 24769, 22629, 21320, 14786, 10681, 8595, 10919, 18758, 14664, 8741, 15464, 13025, 16132, 23036,
    21288, 19109, 16722, 14686, 15360, 8766, 13384, 15417, 16012, 16373, 13495, 30897, 30913, 25489, 25512, 19376,
    19401, 19454, 10594, 20921, 19096, 16982, 18729, 16108, 26661, 14116, 15858, 12545, 18623, 8774, 21270, 20527,
    16853, 20929, 16517, 15476, 15676, 12594, 23539, 18731, 9029, 20490, 16644, 10569, 13608, 37022, 8849, 49159,
    21476, 17358, 14227, 16001, 47104, 45059, 43039, 41014, 36903, 30728, 16672, 16232, 10306, 13931, 40990, 34940,
    32851, 28711, 26660, 23038, 23256, 21091, 21291, 18510, 18940, 16954, 14651, 15085, 15683, 15763, 16236, 12342,
    12400, 12690, 14020, 10331, 10810, 10888, 10990, 11021, 8213, 8498, 22935, 20802, 21199, 28719, 26738, 26916,
    22541, 22909, 23264, 23294, 23373, 20966, 16509, 16753, 16908, 15095, 13535, 27115, 24980, 20557, 20926, 21211,
    18657, 18788, 16438, 16605, 14692, 16268, 13972, 14019, 29140, 25270, 21376, 17230, 17250, 17283, 14097, 14137,
    9143, 8603, 10284, 8272, 30751, 15009, 13591, 8713, 15419, 10539, 24795, 25137, 20645, 18518, 18969, 14703,
    15193, 15545, 16049, 16097, 8411, 25004, 10754, 28721, 12859, 13298, 10541, 10695, 8247, 8723, 36966, 10604,
    30851, 28930, 29038, 24910, 25018, 22955, 23281, 21163, 21401, 18738, 18862, 19095, 15886, 8268, 9163, 18484,
    13546, 13346, 13647, 15115, 8848, 14480, 15709, 26634, 14890, 14941, 13937, 11078, 11081, 11252, 32821, 26905,
    16538, 14701, 15693, 15833, 12539, 12878, 10580, 13098, 23023, 29020, 26768, 25046, 20850, 18576, 18898, 16645,
    26759, 23209, 21015, 18978, 19329, 12536, 12855, 16829, 12985, 13284, 13624, 8769, 21397, 12462, 11118, 11212,
    8543, 8478, 37012, 23497, 23523, 20510, 20739, 20908, 18488, 18726, 16424, 14774, 14963, 15852, 14238, 8930,
    21449, 14335, 13106, 11106, 13755, 14139, 14145, 9164, 12643, 10356, 8649, 20803, 16651, 15195, 12477, 10767,
    30780, 30845, 22632, 23082, 13901, 12404, 13174, 15282, 12604, 12743, 10727, 10907, 10993, 8278, 45101, 13151,
    14241, 41050, 39002, 39015, 39039, 28855, 29006, 29034, 29040, 29048, 29051, 26956, 27117, 23117, 20930, 16109,
    8271, 29053, 13430, 13477, 14250, 14291, 10903, 11086, 39001, 36929, 36960, 26936, 25032, 22559, 22921, 20665,
    21042, 21115, 18718, 16849, 15738, 15750, 10618, 28997, 15235, 23326, 19112, 16216, 14058, 8840, 22713, 20855,
    18823, 17208, 14503, 13944, 14075, 11185, 15609, 14334, 18681, 16736, 15981, 8344, 16069, 12899, 11246, 15785,
    10660, 14767, 15168, 14600, 10591, 34822, 28844, 28989, 26689, 22743, 18598, 19072, 19117, 19205, 16462, 16597,
    14934, 15974, 10280, 23297, 18775, 16444, 13642, 19177, 8916, 25247, 17149, 17156, 16287, 25040, 22803, 19207,
    16603, 16772, 14444, 19071, 28954, 26774, 22859, 19162, 16884, 16094, 12344, 10255, 14174, 19143, 12726, 10445,
    17349, 13153, 14313, 10774, 36984, 30788, 14571, 15811, 19123, 27045, 24993, 23243, 20619, 18669, 19003, 19127,
    12731, 13843, 13698, 9187, 8770, 25316, 25319, 25335, 21361, 9014, 11163, 9116, 10905, 15143, 15286, 13380,
    13705, 8457, 32794, 27122, 24843, 10333, 15787, 19260, 13076, 15330, 16679, 16999, 14666, 12428, 10473, 10560,
    11218, 19275, 13989, 9097, 25285, 25344, 25362, 17234, 9166, 49158, 8701, 14609, 13595, 15377, 13328, 10933,
    22762, 13062, 8528, 15735, 17189, 16357, 11042, 30786, 19209, 12698, 10407, 11139, 19221, 13993, 14346, 47117,
    41051, 26710, 25111, 22549, 23346, 20562, 19014, 19026, 15469, 12405, 12773, 10251, 23481, 16588, 16769, 15403,
    8456, 20536, 14205, 10710, 15534, 23543, 10955, 43069, 30927, 30948, 30958, 25466, 25483, 25506, 19351, 19354,
    19359, 19365, 19394, 19432, 19434, 19453, 28900, 20778, 18832, 17178, 9059, 17108, 15104, 15506, 12478, 13354,
    38972, 28739, 25125, 37038, 37040, 15359, 9140, 8956, 20674, 20924, 18507, 18685, 12425, 8948, 17392, 9034,
    24691, 23171, 22595, 16519, 16880, 14878, 11000, 9013, 25312, 34859, 20869, 14655, 14691, 15099, 21266, 14332,
    23312, 20518, 13318, 13371, 13514, 24984, 16740, 14738, 15590, 16239, 12607, 10526, 10947, 8250, 26805, 24643,
    22839, 23089, 20488, 20509, 20533, 21296, 18613, 18687, 18942, 8581, 9007, 14843, 10714, 8619, 10613, 16308,
    16129, 8514, 34905, 15987, 22530, 8864, 25275, 25322, 25348, 21324, 21326, 21338, 17289, 14148, 10977, 13034,
    13900, 10385, 24700, 24711, 15220, 15647, 15624, 8579, 12967, 14886, 15641, 15599, 8768, 12974, 13851, 12895,
    13961, 11159, 40997, 22748, 21258, 16485, 14661, 16062, 12495, 10573, 11144, 8274, 9150, 11143, 24723, 12978,
    8897, 12378, 13196, 13549, 26680, 25073, 15149, 13523, 14765, 21384, 47116, 15613, 12413, 10626, 9033, 13148,
    13357, 17330, 23315, 21248, 24989, 22640, 21098, 19202, 15442, 12448, 12705, 34925, 24847, 26961, 24612, 23247,
    14815, 39020, 30799, 24784, 23224, 20742, 19078, 19294, 14640, 15462, 15872, 8823, 21349, 21393, 17264, 15317,
    8626, 14988, 12838, 21304, 10793, 16682, 16836, 15073, 8964, 10532, 20615, 13377, 22734, 21083, 14729, 18664,
    9104, 13265, 17100, 14434, 16309, 28775, 10973, 10908, 11141, 25232, 49161, 15062, 18504, 13557, 13854, 13909,
    21075, 18895, 15790, 13575, 26869, 23128, 23198, 17373, 24801, 18539, 14500, 15907, 13349, 16386, 16584, 15336,
    15434, 12722, 10724, 10750, 8219, 8323, 13415, 13140, 8652, 11131, 9124, 12876, 17009, 12364, 18653, 16476,
    8955, 8814, 10493, 14882, 34935, 28857, 20590, 18505, 22615, 23231, 20736, 14960, 15252, 27006, 24611, 10452,
    13754, 13587, 13806, 15511, 13517, 11056, 17118, 17168, 13914, 11109, 11208, 26750, 26760, 26794, 26814, 24875,
    23213, 23225, 23461, 20530, 20903, 19320, 19340, 16440, 16502, 16796, 16948, 15418, 15881, 14004, 14015, 8734,
    25009, 20954, 18634, 8424, 16596, 16495, 16727, 15339, 15517, 12396, 14000, 28888, 15047, 15634, 15921, 16265,
    8325, 12938, 34873, 16144, 12387, 28922, 24876, 23013, 23368, 21149, 14764, 15586, 12862, 17407, 16207, 8920,
    18684, 15900, 12611, 20915, 17159, 14797, 22700, 8656, 36937, 15063, 9215, 16961, 8687, 13325, 24856, 16816,
    10917, 8605, 13087, 26802, 22974, 14982, 16066, 23290, 8587, 43028, 30818, 22968, 23042, 23369, 20644, 18606,
    12714, 23530, 21088, 19409, 19415, 20730, 22628, 14270, 10716, 26824, 24596, 24748, 24914, 22701, 22989, 23094,
    21074, 19090, 15100, 15204, 12746, 12785, 10316, 15365, 15390, 8718, 47118, 38934, 37046, 34847, 30899, 30910,
    25383, 25401, 19431, 14996, 15460, 12521, 13987, 14163, 14208, 8828, 18794, 14637, 8471, 16079, 10639, 8370,
    9064, 13119, 13980, 41027, 36891, 34933, 34934, 24936, 22736, 23158, 20928, 21116, 19264, 12490, 13294, 14242,
    22695, 20996, 15208, 18745, 8416, 24746, 22552, 28915, 25101, 22694, 21282, 18747, 18965, 16395, 16396, 14371,
    10786, 25005, 23126, 21147, 18883, 16626, 10481, 45102, 19262, 13561, 13113, 13281, 13311, 9198, 15916, 18948,
    8558, 14024, 14223, 9018, 22667, 22699, 17007, 12619, 13893, 14842, 13204, 13880, 16224, 13759, 13858, 8586,
    16267, 18577, 14910, 16055, 8850, 41067, 26785, 26852, 26892, 22913, 20624, 19213, 16758, 15071, 16064, 22851,
    12786, 8536, 18587, 11146, 13427, 22644, 20588, 8808, 25302, 21359, 17241, 17286, 17294, 17076, 13111, 13271,
    8497, 12969, 28886, 26624, 26668, 25039, 25166, 22987, 23024, 20568, 19118, 16995, 14939, 10454, 17388, 28695,
    13991, 15779, 34835, 13002, 29102, 26688, 27004, 24636, 25013, 22822, 20578, 20992, 12605, 8929, 29022, 18777,
    18835, 16904, 14046, 20722, 21059, 10287, 10766, 21019, 15504, 13747, 10379, 27012, 12541, 25321, 23005, 20863,
    18933, 16742, 16823, 16888, 14949, 8299, 9063, 16988, 8421, 16152, 9095, 24647, 10789, 22785, 22919, 18438,
    14788, 14879, 16050, 16164, 16192, 12661, 8239, 14042, 11220, 19038, 17249, 10945, 40960, 16527, 14872, 15667,
    13676, 43013, 38966, 36882, 34819, 34843, 34894, 24609, 24697, 24957, 22679, 20785, 10964, 36990, 12821, 10259,
    14064, 10635, 11191, 16244, 9038, 38955, 29045, 26790, 26934, 24786, 23191, 20655, 18554, 16756, 8887, 38993,
    18819, 17321, 14833, 28955, 13006, 16230, 13869, 27051, 23440, 18593, 19017, 19103, 19271, 16402, 16757, 17062,
    14337, 14367, 14713, 16029, 12337, 12638, 10524, 15832, 8550, 8789, 13382, 14261, 9200, 9003, 23436, 24618,
    16708, 16271, 16350, 25163, 23064, 20546, 21322, 18533, 18882, 16343, 23114, 15413, 13509, 22590, 14054, 45081,
    45084, 20979, 14592, 15724, 13953, 16481, 15184, 15912, 10250, 8214, 10721, 32828, 28957, 29043, 18469, 14773,
    15295, 16160, 12422, 24740, 22621, 16539, 14822, 13679, 13730, 14673, 10831, 16694, 10620, 10890, 18841, 15492,
    15523, 8556, 12770, 23028, 16647, 15262, 12775, 8973, 13092, 15601, 8286, 8609, 12810, 10588, 9204, 49173,
    37029, 14091, 14836, 8600, 8933, 13609, 13426, 14513, 15607, 12361, 8441, 10816, 28903, 18720, 13277, 24917,
    25047, 25145, 23473, 15600, 16014, 25355, 14874, 10653, 16879, 25143, 20671, 21094, 16696, 14390, 28962, 16914,
    14903, 15211, 15814, 12756, 13772, 19247, 12897, 15158, 12772, 19332, 15671, 16358, 14012, 10880, 14119, 8666,
    13391, 15840, 25231, 13249, 13322, 13864, 26832, 18992, 10545, 14802, 15572, 39013, 26913, 26967, 21123, 18660,
    16813, 14466, 14522, 14992, 15173, 15657, 15883, 15993, 12455, 12505, 12590, 13930, 8672, 26972, 23236, 21219,
    14614, 12996, 13632, 22594, 14128, 24809, 14305, 13776, 13878, 13547, 14088, 32796, 34852, 16762, 13795, 22567,
    13342, 17109, 15525, 16291, 12651, 14324, 49174, 43062, 43063, 43066, 43067, 37045, 37057, 37060, 37065, 37066,
    37074, 37081, 37099, 37100, 37101, 30890, 30896, 30908, 30921, 30925, 30932, 30954, 30964, 25381, 25428, 25429,
    25445, 25458, 25472, 25482, 25493, 25516, 19367, 19393, 19436, 19448, 14159, 19237, 16382, 11053, 32835, 26945,
    22545, 16191, 13528, 10952, 28842, 26781, 23524, 23525, 20666, 20753, 21144, 21281, 18487, 15117, 16257, 12608,
    16655, 34878, 28971, 22712, 14439, 15199, 19091, 12964, 28975, 22865, 24940, 25082, 38985, 21323, 15791, 10317,
    8650, 15756, 20862, 12397, 22670, 15420, 12618, 34903, 32859, 13527, 24721, 22739, 18880, 13337, 16321, 16506,
    8426, 13264, 13386, 13729, 16933, 14589, 15248, 27014, 22842, 18954, 14753, 12419, 10670, 11172, 30777, 28821,
    29071, 8699, 19243, 12918, 10751, 8535, 8640, 21071, 14791, 15717, 12863, 13463, 12941, 14049, 13486, 39032,
    36982, 34887, 28819, 28901, 26726, 26924, 27108, 24840, 23292, 23338, 20777, 20902, 15463, 15908, 8870, 15137,
    37028, 32867, 28707, 28863, 8317, 13910, 8913, 38968, 16529, 15112, 23498, 23549, 21301, 16531, 15128, 20880,
    18680, 16335, 30813, 26665, 25076, 25155, 19055, 16458, 10408, 10771, 11170, 32833, 26729, 15437, 12508, 13248,
    22956, 15749, 12307, 9197, 19328, 16785, 17087, 14432, 13883, 34932, 17405, 24942, 16534, 17086, 17121, 17134,
    16297, 16301, 16366, 26891, 26947, 22837, 23529, 18580, 19304, 19322, 17184, 28747, 26923, 13671, 19211, 8790,
    14760, 13258, 10787, 13510, 13673, 9068, 10373, 16571, 23516, 20480, 21298, 16392, 16749, 17069, 16516, 12760,
    13226, 13348, 19306, 24627, 21150, 17158, 9022, 19133, 22840, 20727, 19004, 16810, 12644, 13503, 10590, 8319,
    27021, 19035, 15221, 12347, 14050, 28728, 26917, 14777, 15850, 21046, 16547, 17160, 14340, 12399, 8998, 16330,
    14027, 16777, 41007, 38964, 29031, 15374, 10537, 14645, 15898, 16747, 12794, 17050, 16011, 12943, 13888, 15144,
    8472, 24630, 10788, 8613, 20583, 18541, 12542, 32769, 10969, 13093, 13918, 10911, 17356, 17382, 9181, 9081,
    15718, 16246, 11015, 8819, 24964, 22890, 23476, 14538, 15347, 39037, 36996, 26640, 26839, 26975, 25066, 16773,
    14787, 12795, 12848, 15177, 16950, 15550, 17322, 21097, 13983, 11152, 32824, 23146, 16876, 12860, 10476, 14142,
    16702, 10430, 15712, 29148, 21427, 21454, 21498, 17335, 14260, 14301, 9042, 13198, 32797, 28691, 28700, 20643,
    20796, 16969, 14421, 14583, 10606, 20625, 38995, 14321, 9205, 16258, 30797, 28859, 27013, 24790, 22649, 22697,
    22950, 23398, 23402, 20937, 15793, 10797, 8316, 15529, 32850, 20832, 10694, 36921, 15669, 18828, 14974, 15968,
    12395, 14044, 10610, 26691, 10444, 13593, 13994, 11210, 16778, 14339, 15540, 12777, 13070, 13320, 10361, 10723,
    10893, 10925, 8227, 8304, 11247, 9113, 16032, 13807, 19413, 13360, 8904, 13917, 13016, 13465, 13559, 15591,
    15944, 10725, 26767, 25043, 22996, 21246, 16537, 14343, 12589, 12804, 12879, 13750, 10267, 10534, 10557, 10883,
    8477, 14233, 14274, 14279, 14326, 40981, 41004, 14904, 17203, 23445, 15120, 30855, 24900, 20934, 19284, 17221,
    16352, 16356, 14007, 14023, 15188, 15626, 9144, 21048, 22674, 22940, 20899, 16451, 14114, 28969, 29033, 21137,
    12954, 13055, 13830, 22680, 41066, 25057, 25146, 21313, 19300, 16435, 14656, 15226, 8637, 16048, 10479, 23248,
    16514, 14419, 14792, 16326, 10790, 10992, 8335, 14684, 15755, 16360, 12818, 10363, 13246, 8402, 8622, 20487,
    10453, 13224, 13577, 10348, 9077, 23115, 10459, 36978, 30833, 26937, 24718, 24864, 25254, 22845, 23154, 20878,
    21206, 14700, 13329, 11156, 8385, 15501, 36963, 27022, 22812, 22973, 23179, 23391, 21209, 18774, 17216, 15243,
    12529, 19166, 13219, 13742, 23407, 18686, 8754, 8978, 15630, 16877, 15237, 15679, 12504, 22747, 14677, 16070,
    30774, 30832, 29018, 27028, 24730, 25001, 22948, 20595, 17085, 10844, 11251, 29119, 25278, 17240, 17242, 17263,
    20782, 18627, 15375, 15466, 10906, 8246, 13957, 14079, 15564, 8815, 15134, 27003, 25117, 22998, 20564, 18691,
    14458, 13125, 17397, 12865, 12748, 32871, 29129, 25291, 25304, 25353, 21381, 21399, 21412, 17275, 23397, 18600,
    19196, 16415, 16091, 10322, 10371, 12893, 13534, 18500, 10518, 32854, 16549, 14620, 15996, 12623, 13220, 13372,
    10441, 10547, 10550, 8355, 17165, 15580, 12321, 12995, 8729, 13401, 17104, 16289, 20683, 18857, 15013, 12614,
    41012, 41068, 36894, 32844, 30870, 28860, 26654, 26673, 26764, 27005, 24716, 24922, 25055, 22769, 23204, 23339,
    23419, 18624, 19021, 14891, 15652, 15748, 16175, 13588, 34914, 26836, 26928, 26957, 21142, 14809, 14871, 15520,
    10598, 25283, 25351, 21420, 17223, 17236, 17287, 14111, 25230, 13810, 22691, 26938, 9170, 22744, 22868, 14323,
    11186, 16818, 36935, 30827, 29080, 26697, 24719, 24978, 23019, 20835, 18755, 17015, 14747, 14926, 34899, 26944,
    24589, 23483, 21024, 16712, 15803, 36967, 28970, 28990, 29082, 27018, 23039, 12771, 11075, 8869, 18502, 15741,
    12706, 8523, 8969, 10987, 28740, 10566, 36880, 30884, 26981, 22892, 23293, 21160, 21273, 18467, 18562, 12916,
    8883, 21134, 28980, 24590, 23517, 23538, 20642, 21275, 18973, 16421, 16478, 16613, 17095, 14721, 15093, 16252,
    12320, 28830, 26776, 26843, 24816, 22708, 23365, 20770, 20799, 21295, 18867, 19100, 16496, 15588, 12440, 12637,
    12697, 13970, 13014, 8773, 28716, 16781, 14909, 12414, 20569, 13819, 13455, 11166, 18963, 15568, 13231, 10910,
    22759, 9072, 13905, 16293, 13263, 13273, 14578, 16278, 10661, 18913, 19312, 16398, 15538, 11074, 20945, 19266,
    12976, 18845, 14479, 12460, 8549, 9047, 36940, 8782, 18725, 19175, 16739, 14623, 16429, 14669, 10409, 8390,
    23093, 23366, 15836, 10380, 10425, 10825, 8216, 12979, 13791, 10833, 8526, 15249, 24692, 13648, 8994, 8901,
    37049, 25405, 25464, 25467, 19426, 19438, 14164, 14183, 11137, 8420, 16176, 10737, 8800, 13861, 16375, 16520,
    12828, 11026, 9089, 30974, 25471, 25503, 19370, 19380, 19387, 19405, 19444, 14166, 14172, 13757, 11239, 15340,
    8491, 10696, 16015, 13745, 24739, 16564, 8947, 15475, 13379, 41073, 25271, 25280, 15800, 15311, 26959, 14267,
    28687, 28804, 28899, 18644, 19150, 16653, 15902, 8807, 43032, 28964, 26849, 24752, 22536, 23547, 19062, 16303,
    16377, 12684, 13919, 13966, 11045, 11059, 11201, 9082, 8821, 16151, 34942, 27069, 24934, 25023, 23320, 20670,
    21114, 14940, 15841, 12824, 10240, 21503, 17350, 25056, 25122, 22760, 23405, 20817, 18628, 18784, 18889, 16393,
    16541, 17170, 15410, 12520, 12700, 26662, 15294, 16963, 8574, 15542, 8404, 21445, 17355, 8873, 36908, 28688,
    28947, 23112, 23144, 10895, 13039, 14255, 16807, 14431, 12519, 21496, 10540, 36965, 24749, 22863, 18848, 16484,
    14947, 12525, 37014, 28698, 13658, 13507, 30735, 28801, 27041, 10439, 36902, 25003, 13237, 17049, 14875, 15048,
    10928, 13474, 26670, 26778, 24835, 24871, 23074, 23220, 21064, 12600, 14003, 8780, 49156, 24990, 15065, 15300,
    15994, 12392, 10301, 8423, 21237, 12551, 10801, 8804, 8512, 8417, 36988, 34853, 28730, 15153, 8597, 11214,
    14314, 9188, 24946, 22659, 15575, 10475, 8719, 27010, 18448, 34937, 28846, 28892, 28956, 29062, 26813, 26880,
    26964, 27056, 27060, 27064, 27078, 24902, 24963, 25060, 25104, 25128, 25189, 23046, 23141, 23223, 23244, 23267,
    23298, 23299, 23376, 23385, 23459, 23475, 21047, 21188, 18756, 19168, 19206, 19208, 19216, 16974, 15292, 15493,
    15842, 15942, 15951, 15955, 16072, 16111, 20701, 18569, 20620, 14726, 15480, 12752, 45094, 41033, 38949, 24869,
    22732, 22771, 23017, 10647, 16687, 15554, 25190, 21065, 18537, 16389, 16611, 15819, 10565, 20492, 15136, 32787,
    32857, 27128, 12909, 13208, 13392, 12506, 25156, 22916, 17025, 25262, 17152, 14790, 15118, 32768, 25038, 21156,
    12452, 45106, 36904, 30767, 26744, 26745, 26951, 24766, 24903, 23277, 17038, 12573, 23030, 23427, 20677, 21055,
    18658, 14531, 12631, 45085, 38920, 14100, 43016, 34872, 45077, 29077, 22589, 22693, 18575, 18692, 15461, 13615,
    19007, 13886, 13183, 11199, 16095, 16557, 12554, 28933, 13715, 16684, 13321, 13505, 12872, 16436, 14814, 10667,
    10974, 28837, 22745, 14575, 16266, 13417, 21174, 18838, 19343, 17116, 14356, 15392, 12375, 12456, 14022, 10853,
    8824, 41000, 32795, 30885, 28680, 26812, 27059, 24907, 25191, 22786, 23535, 21279, 19011, 19279, 19288, 16469,
    17093, 14368, 15320, 17293, 10934, 22535, 14418, 8359, 24858, 22780, 20999, 18690, 16664, 16925, 17070, 14410,
    15014, 15020, 15379, 15408, 12312, 12570, 12822, 10302, 20513, 8551, 10451, 26960, 12433, 13418, 17224, 17225,
    8455, 9108, 32847, 30875, 41045, 38958, 39010, 39018, 34879, 30773, 28869, 29055, 26632, 26749, 26818, 26911,
    24690, 24758, 24802, 24908, 25015, 25151, 25174, 25251, 22587, 22727, 22763, 22881, 23003, 23118, 23159, 23457,
    23491, 23515, 20783, 20815, 20904, 20941, 21023, 21239, 21294, 18443, 18678, 18916, 19005, 19138, 19194, 19230,
    19287, 19302, 19344, 16591, 16598, 16615, 16678, 16720, 16812, 16835, 16844, 16855, 17006, 17010, 17063, 17143,
    17213, 14341, 14391, 14403, 14429, 14433, 14546, 15027, 15315, 15559, 15936, 16033, 13922, 10832, 8791, 13680,
    39038, 13467, 13053, 10427, 15622, 12923, 13192, 13498, 13777, 22551, 22655, 23324, 20483, 21403, 18470, 17274,
    8407, 13453, 13594, 26721, 26875, 15052, 17041, 16172, 25234, 13370, 16338, 24920, 23361, 14584, 10680, 8220,
    13942, 21011, 12922, 13086, 13128, 11093, 23101, 15570, 12537, 21007, 13707, 26845, 22821, 23360, 23448, 23512,
    20528, 20990, 18815, 19277, 16404, 16556, 17074, 14385, 14728, 15053, 11098, 11232, 10968, 12595, 16881, 16957,
    14861, 15915, 18494, 12556, 13489, 20586, 13147, 15767, 19245, 13213, 16607, 12901, 20811, 17337, 15711, 15802,
    8862, 24631, 22664, 18472, 14552, 24846, 22787, 28968, 18742, 19114, 18506, 16447, 17044, 14737, 15737, 15746,
    15812, 15930, 16010, 12640, 12763, 8488, 24764, 8331, 8580, 8989, 13860, 43047, 13882, 14985, 15815, 15919,
    12578, 12803, 8939, 10472, 19319, 27082, 21228, 16905, 14808, 15481, 16137, 8206, 8679, 13029, 13280, 32879,
    8617, 13881, 14056, 14067, 11183, 8835, 9147, 16609, 14918, 12864, 10485, 28913, 12898, 14231, 13572, 15045,
    16938, 13852, 13197, 8635, 28952, 8921, 8540, 13046, 18915, 19280, 8490, 19261, 13710, 28861, 14457, 10404,
    24928, 19064, 15148, 15597, 15696, 23084, 26879, 25249, 20916, 19043, 8364, 11084, 36899, 26856, 13272, 20627,
    20972, 18695, 10777, 9051, 19274, 13770, 25188, 19000, 12708, 10470, 10632, 28703, 8945, 23433, 18981, 15795,
    15805, 15954, 16195, 10281, 8192, 8338, 10936, 9026, 13364, 8963, 11161, 10914, 16692, 12732, 23333, 17135,
    15473, 12468, 41059, 29096, 26902, 26963, 27106, 23352, 23480, 14607, 8375, 26838, 24785, 23435, 15557, 12793,
    10717, 10836, 8211, 8861, 15551, 13933, 26656, 24634, 24694, 22586, 23548, 20522, 20708, 17045, 14545, 15605,
    15860, 16212, 12380, 12476, 11206, 8651, 24906, 13845, 30743, 28920, 24654, 24727, 24780, 24968, 25242, 22681,
    22879, 22903, 20593, 20597, 20716, 20795, 20910, 21040, 21102, 21278, 18486, 18497, 18964, 19276, 16932, 17141,
    14515, 14577, 15057, 15268, 12306, 14030, 11205, 12470, 20545, 20968, 20977, 13261, 13446, 20740, 19020, 14074,
    21497, 17346, 13616, 14273, 14644, 16972, 14759, 10854, 25369, 16671, 19176, 8632, 13424, 15059, 17180, 17193,
    17218, 15055, 22663, 20844, 16710, 16874, 14396, 12501, 10450, 10847, 8893, 19190, 14716, 10712, 16283, 20758,
    13367, 32852, 30757, 22638, 22888, 18929, 16789, 17112, 14377, 14475, 8578, 25048, 17018, 15788, 15816, 10286,
    10528, 17345, 17194, 8716, 26657, 20718, 10772, 11237, 13589, 13586, 12959, 13444, 14026, 11022, 11069, 23221,
    16159, 34904, 29012, 23060, 20874, 20951, 16060, 16161, 13545, 10336, 10422, 8270, 16300, 24715, 22883, 20985,
    21212, 19033, 19051, 21259, 18856, 14920, 15364, 15809, 16057, 12503, 17055, 15254, 10568, 8721, 13004, 13797,
    34896, 28752, 21310, 18786, 19107, 14594, 14983, 15966, 12355, 13400, 41018, 26997, 27118, 25131, 23215, 20792,
    18491, 19039, 14561, 12429, 17167, 13243, 13786, 29015, 22574, 14766, 18534, 18850, 19195, 19235, 15519, 16122,
    16157, 16354, 12394, 10842, 17061, 13645, 13668, 14665, 8661, 29132, 14031, 21487, 16760, 17191, 17347, 17364,
    12628, 11169, 9182, 13891, 24826, 20563, 17030, 15946, 10458, 15848, 16130, 16178, 10811, 8647, 13300, 19029,
    14708, 12449, 8541, 22850, 19183, 12720, 30824, 24921, 23384, 20805, 10463, 10708, 8758, 15730, 8251, 23009,
    12724, 15289, 10511, 8240, 16841, 10850, 19188, 10858, 8965, 10995, 15553, 8389, 10840, 16928, 10457, 10732,
    10896, 8548, 28779, 14750, 15190, 13518, 10499, 9199, 32773, 30734, 13868, 14131, 15734, 16165, 8357, 19241,
    13173, 13297, 24757, 18930, 24882, 14633, 15302, 10249, 13775, 8629, 21201, 12699, 10386, 14803, 12465, 13665,
    10440, 9151, 14216, 14245, 13723, 12769, 20556, 20622, 15818, 8276, 11255, 15706, 10747, 8507, 49153, 41002,
    36888, 32799, 32807, 32843, 19191, 13000, 14010, 36964, 30829, 23044, 23103, 18670, 16606, 16860, 21465, 17318,
    14246, 14320, 9178, 47120, 18875, 21018, 15042, 15394, 34898, 32813, 30835, 27009, 27017, 24838, 22620, 22724,
    22937, 22978, 23194, 23305, 16117, 12328, 20577, 20657, 20756, 18474, 18698, 18812, 19077, 17048, 14573, 18887,
    25276, 25282, 25305, 17290, 23152, 16958, 8546, 15894, 37033, 32865, 25272, 25293, 25299, 25314, 25325, 25356,
    21364, 17301, 14101, 9134, 14805, 15293, 11070, 36887, 14539, 12982, 10736, 12835, 13837, 18671, 10328, 9056,
    14596, 14986, 15646, 25121, 22606, 22949, 23062, 18512, 15412, 28790, 25129, 24853, 22791, 23007, 23035, 14610,
    15920, 8689, 26929, 25105, 23055, 23188, 23197, 20728, 20829, 21127, 18897, 14597, 15443, 15451, 15690, 15740,
    15924, 16041, 16238, 16397, 13079, 28767, 26672, 22651, 22665, 16634, 14978, 15765, 29121, 25336, 21365, 21371,
    21380, 21419, 9141, 9010, 17366, 17377, 9196, 24877, 21025, 13235, 13350, 13476, 11034, 9096, 13660, 10929,
    8885, 22905, 15031, 21193, 14722, 38945, 23509, 20500, 18560, 19298, 16427, 16592, 14483, 15405, 15904, 15911,
    16253, 12295, 12323, 13579, 27048, 15229, 25363, 17288, 34886, 30846, 28757, 26833, 27123, 22802, 18984, 12610,
    30863, 30866, 26859, 26878, 23161, 23268, 20867, 18460, 18461, 19317, 16467, 17126, 17201, 14096, 24625, 14120,
    14757, 15231, 22817, 28878, 20895, 21077, 8302, 14541, 16336, 15114, 47123, 45092, 38942, 26829, 23040, 13628,
    15530, 8475, 14008, 14870, 10559, 41072, 15516, 13255, 16276, 16941, 17119, 28917, 28935, 28936, 29065, 29075,
    24848, 25000, 25115, 25134, 22901, 23196, 23323, 23425, 20938, 18918, 19147, 10372, 9030, 10619, 21106, 19002,
    16602, 16806, 15905, 15959, 16186, 12377, 12688, 10248, 10510, 21140, 9209, 26735, 15218, 12675, 19079, 15213,
    12512, 15126, 29130, 13242, 14730, 13655, 9074, 15745, 16422, 14460, 15638, 25223, 20822, 18558, 17199, 14414,
    14535, 10501, 25300, 8820, 17188, 14602, 14544, 15357, 15855, 12367, 13466, 17125, 12664, 12894, 13635, 8902,
    13782, 19253, 13138, 13841, 12832, 10645, 8659, 26682, 16562, 17022, 14580, 14968, 15945, 12388, 12555, 10915,
    8673, 13666, 11031, 28940, 26841, 26877, 25554, 22995, 23409, 21096, 16417, 16622, 15022, 15563, 18908, 19289,
    16851, 17390, 8639, 22614, 21190, 14630, 14914, 12750, 10655, 8320, 16971, 21172, 15354, 15707, 12715, 15138,
    8387, 13834, 12972, 16729, 10605, 13971, 29105, 47132, 25573, 26903, 24959, 21185, 18550, 16962, 10243, 10469,
    34922, 28793, 27032, 25072, 21058, 21152, 16486, 13154, 13368, 17406, 45111, 38992, 34895, 28702, 24949, 22834,
    20608, 18921, 16499, 16734, 12945, 13221, 10942, 8422, 14016, 49166, 47131, 43008, 41003, 38954, 36890, 8354,
    8676, 14521, 15039, 13091, 13550, 8825, 13118, 34858, 28894, 26850, 26979, 24778, 24956, 22554, 22741, 23165,
    23235, 18603, 15368, 12587, 12622, 10310, 10862, 30937, 27112, 24759, 23269, 18892, 19215, 17024, 14478, 14720,
    14895, 15801, 12338, 12339, 12753, 10553, 11254, 8215, 8333, 26984, 15922, 15102, 21014, 18572, 19087, 10462,
    19338, 13301, 13538, 32888, 17375, 14244, 14297, 20983, 15732, 49179, 49183, 43050, 43051, 43052, 43053, 43054,
    43061, 37054, 37068, 37079, 37080, 37086, 37089, 37094, 37103, 37105, 30895, 30923, 30933, 30938, 30945, 30951,
    30959, 30968, 30972, 25382, 25386, 25390, 25392, 25400, 25408, 25413, 25439, 25444, 25446, 25465, 25468, 25476,
    25478, 25484, 25509, 25518, 25525, 25530, 20859, 20965, 19349, 19357, 19364, 19383, 19439, 19445, 15827, 14155,
    14202, 14213, 10394, 10956, 15180, 16648, 17005, 14472, 14867, 12494, 12540, 12653, 10628, 11085, 8283, 19180,
    47106, 47113, 45060, 40966, 38974, 36872, 24734, 14912, 12936, 13920, 13932, 14068, 11011, 11103, 11138, 10571,
    25208, 16902, 14537, 16153, 12734, 10347, 27134, 19283, 8696, 40967, 26797, 26918, 22582, 16776, 11123, 11215,
    13995, 11162, 18523, 14851, 11233, 49162, 18797, 15769, 20508, 16964, 16977, 17014, 15257, 14292, 10343, 10527,
    17182, 17197, 16370, 11107, 11253, 9103, 26889, 27063, 24615, 16261, 10701, 13141, 34870, 22789, 20751, 16058,
    12807, 10375, 41041, 29010, 14820, 10676, 8492, 18902, 15200, 10659, 10927, 19268, 14225, 8585, 38969, 34910,
    29098, 21225, 16741, 8783, 28758, 25123, 14893, 10627, 16083, 16459, 15125, 10948, 8198, 25160, 22981, 21119,
    19013, 15687, 8495, 21252, 13353, 23245, 21170, 14646, 15879, 16121, 12514, 8298, 11130, 11228, 29103, 18932,
    12914, 45100, 34848, 21186, 18764, 26675, 8890, 13116, 20497, 12372, 38939, 15207, 25074, 37041, 32872, 25310,
    20581, 20798, 21303, 21425, 16842, 15918, 12418, 14081, 11226, 21446, 18561, 15980, 15333, 16886, 9015, 16263,
    18530, 16092, 8954, 22688, 15382, 26823, 27075, 25075, 22607, 20872, 18453, 16513, 14769, 12333, 16737, 14894,
    11025, 13433, 26962, 23006, 23240, 18621, 18633, 16465, 47130, 10593, 14312, 10846, 8803, 32800, 15191, 15594,
    12313, 14299, 45103, 40994, 38921, 36869, 30809, 26652, 26783, 24926, 24973, 23252, 20489, 20566, 19054, 16661,
    15454, 15656, 16042, 8196, 13737, 13733, 13456, 12543, 10615, 8438, 18648, 22814, 18615, 47107, 23484, 22899,
    21203, 15477, 8728, 14621, 15212, 14896, 8882, 15582, 38975, 37021, 21079, 36876, 28916, 27007, 25067, 25096,
    16198, 10794, 8382, 11241, 10686, 13601, 19269, 12924, 28926, 28745, 22806, 13260, 9060, 21490, 15567, 8496,
    45082, 34851, 26950, 23121, 23291, 20877, 21109, 18869, 16416, 16477, 15744, 12341, 12393, 12487, 20986, 21274,
    18752, 9035, 27020, 23232, 26909, 10354, 10495, 39028, 36892, 16869, 15234, 15428, 12907, 21311, 18521, 16403,
    16371, 16269, 17130, 17379, 14252, 14262, 14278, 9179, 13838, 13162, 14226, 14309, 14327, 9210, 9006, 22654,
    22815, 13552, 17198, 11195, 13269, 25352, 17276, 17299, 10296, 41060, 41017, 24753, 23166, 20484, 20493, 8831,
    10978, 10998, 14888, 15448, 22856, 18565, 26820, 26861, 14504, 13440, 15581, 24794, 18616, 18956, 16922, 14736,
    12457, 13990, 10350, 25345, 25361, 21351, 14104, 36943, 25204, 23521, 18785, 16797, 12421, 13266, 24619, 18961,
    17057, 15758, 16187, 12381, 13796, 18803, 20947, 21039, 18457, 25330, 15278, 8527, 28867, 22710, 28673, 24781,
    22612, 21253, 16423, 10841, 11110, 8373, 16754, 13940, 26800, 15698, 12574, 10387, 8289, 15124, 12825, 13031,
    13511, 17164, 14505, 14590, 12663, 17387, 13184, 15783, 15196, 25320, 14089, 9177, 15885, 10869, 18836, 12889,
    10533, 8612, 30941, 30965, 28709, 25406, 25454, 25470, 13422, 14188, 25118, 15906, 17317, 17341, 13659, 20605,
    8517, 13835, 15552, 8910, 49177, 49181, 43056, 43058, 37052, 30918, 30926, 30940, 30947, 30967, 25385, 25426,
    25434, 25450, 25463, 25480, 25505, 25524, 25528, 19379, 19406, 14184, 14193, 14200, 19061, 14351, 14489, 12836,
    10364, 10384, 32826, 30740, 30745, 28738, 28879, 27092, 24662, 23186, 23507, 20580, 18672, 18707, 19018, 16800,
    15867, 12363, 12791, 34868, 34897, 28733, 28800, 28973, 26904, 27052, 25240, 23071, 20856, 20911, 18515, 18713,
    18997, 14547, 15034, 15548, 15689, 14071, 10381, 10843, 11155, 14869, 43014, 10778, 10640, 8530, 34827, 23423,
    14420, 12288, 23420, 20764, 15033, 15547, 10899, 8413, 11229, 20987, 30798, 15141, 8380, 23519, 13020, 11124,
    13941, 17306, 17362, 14277, 20980, 24742, 14754, 17144, 13565, 13826, 17154, 20723, 14889, 16081, 12603, 10729,
    8281, 13684, 10800, 26709, 26825, 26970, 23169, 20634, 18435, 13946, 21066, 18542, 13958, 10745, 8358, 12925,
    14235, 24777, 22765, 23150, 20521, 20919, 16691, 17088, 14781, 16441, 16313, 10953, 8381, 38928, 20851, 8744,
    13171, 13211, 16124, 34865, 21141, 16281, 13082, 13423, 10985, 8976, 12451, 10324, 10650, 9168, 12991, 36994,
    28880, 20667, 18694, 16570, 16277, 10562, 10467, 38929, 38943, 27072, 22623, 20698, 18668, 19310, 16870, 17056,
    15731, 16379, 13716, 13728, 10652, 13406, 16946, 22784, 18748, 14059, 19052, 15526, 18564, 15615, 8509, 10303,
    10749, 13763, 8480, 30844, 24898, 23156, 21254, 18481, 19040, 16473, 17004, 13513, 13756, 12977, 13739, 20600,
    29001, 14562, 15682, 12373, 39031, 13459, 9172, 25090, 23266, 21197, 19069, 16830, 14804, 12523, 10519, 8485,
    18960, 12659, 10257, 18813, 45061, 45104, 26977, 27000, 21081, 13943, 13964, 11019, 11047, 10798, 11189, 8388,
    13610, 23265, 8944, 14748, 30747, 28722, 28795, 28864, 28921, 28967, 24994, 20774, 20959, 16940, 14416, 10399,
    10649, 8419, 41044, 38940, 34875, 32829, 26639, 24584, 22608, 22740, 22896, 22946, 19070, 16450, 16521, 17145,
    15806, 12971, 13540, 9065, 13123, 28760, 28866, 29009, 27023, 27089, 22733, 23068, 23274, 21218, 18454, 18763,
    16569, 16640, 15556, 12326, 21466, 39035, 36981, 28907, 24951, 20571, 21095, 18816, 16453, 14588, 14593, 16264,
    12403, 8351, 16662, 11151, 18754, 16735, 17133, 15903, 8343, 25532, 21469, 18693, 11230, 24756, 25140, 18522,
    15998, 13215, 10434, 10665, 11211, 8966, 25222, 25281, 25288, 25290, 9019, 23280, 23413, 19094, 22550, 20746,
    16599, 16897, 14357, 14382, 16349, 12552, 10312, 30724, 28832, 27067, 23131, 20575, 14454, 16003, 12527, 12853,
    12719, 12957, 30720, 13965, 11114, 43027, 27062, 24975, 22793, 23260, 23456, 23465, 21176, 18972, 15275, 15508,
    16007, 12353, 12668, 13638, 16689, 15991, 16223, 10922, 9001, 8888, 13362, 13760, 10870, 8918, 17186, 17222,
    13625, 9162, 19229, 17324, 14239, 14328, 24587, 14551, 15857, 12406, 20731, 14951, 15380, 10465, 15662, 10996,
    8562, 41028, 39004, 30888, 28777, 28856, 26728, 20596, 17155, 14384, 13950, 20689, 13218, 13574, 13047, 13182,
    8312, 43025, 18817, 20909, 15267, 11060, 20679, 47114, 14719, 12439, 13963, 13049, 34862, 15422, 29032, 26886,
    27068, 24998, 22539, 23111, 13308, 13548, 13877, 13912, 45114, 29144, 10584, 10597, 8235, 49175, 49180, 49182,
    43048, 43060, 43065, 37067, 30904, 30911, 30919, 30957, 30960, 30963, 30966, 25371, 25373, 25374, 25376, 25379,
    25387, 25396, 25397, 25415, 25417, 25424, 25487, 25497, 25498, 25501, 25513, 25520, 25521, 25529, 19352, 19353,
    19356, 19378, 19391, 19395, 19402, 19429, 19440, 19449, 13013, 14152, 14154, 14156, 14157, 14162, 14165, 14191,
    14199, 11128, 8903, 14707, 18872, 43071, 37062, 30892, 30894, 25403, 25410, 25421, 25488, 25511, 25527, 19346,
    19371, 19384, 19411, 19422, 19430, 19435, 19455, 14201, 14214, 14247, 11198, 11227, 47109, 47133, 45078, 14662,
    9039, 28965, 22804, 26646, 21092, 18734, 16701, 16249, 10719, 11244, 8257, 14846, 8453, 8934, 29029, 27016,
    24728, 25019, 18947, 14437, 8483, 22810, 8781, 15179, 8322, 32820, 26987, 22991, 23000, 14581, 14660, 15868,
    10282, 11119, 10612, 8326, 10920, 17319, 17361, 14285, 15970, 9002, 15726, 8836, 21347, 9174, 14876, 14459,
    14495, 14266, 17173, 12911, 14679, 16332, 16362, 14011, 10599, 10748, 9100, 13508, 14816, 15863, 8533, 8748,
    13176, 24833, 25250, 23418, 16683, 16331, 45113, 41070, 30765, 30806, 20663, 21268, 18581, 16765, 16857, 16339,
    12467, 10743, 8430, 39019, 29014, 21085, 18568, 18863, 10396, 10589, 8313, 13661, 45095, 28690, 21210, 18979,
    14509, 16310, 12368, 13982, 24772, 13999, 9115, 12934, 13335, 13899, 23454, 28883, 27070, 23487, 18433, 17042,
    14348, 14450, 14838, 14855, 15285, 15458, 16087, 16089, 16141, 12747, 10480, 10677, 8787, 17012, 16182, 8638,
    12778, 12779, 13872, 16891, 14586, 8350, 41010, 28941, 29028, 26816, 26976, 24828, 25159, 22855, 22878, 23081,
    23253, 23295, 23359, 21169, 18442, 18592, 18759, 18935, 19046, 16593, 14698, 8332, 45065, 38913, 38938, 20823,
    18840, 15677, 12516, 21340, 21358, 14945, 12768, 10523, 8565, 13067, 13497, 8749, 30861, 29095, 25238, 25257,
    22602, 22843, 20830, 20833, 21037, 18582, 19028, 16526, 16709, 14533, 8740, 13778, 32779, 30803, 14714, 15010,
    15760, 10820, 13345, 13623, 47122, 45068, 43022, 36889, 32849, 24580, 24873, 25028, 20613, 18826, 16787, 15028,
    15182, 15351, 16135, 12687, 12867, 13394, 13801, 10323, 10515, 8254, 8361, 29079, 22827, 15882, 13480, 8369,
    14989, 10651, 22788, 23212, 21207, 14775, 18876, 15089, 15864, 10448, 30778, 23143, 20962, 19050, 29177, 14240,
    14258, 10482, 11193, 23330, 10530, 23319, 18800, 14613, 10292, 8572, 13316, 23004, 20618, 16147, 10965, 10967,
    41083, 29131, 25296, 21414, 17260, 17267, 17273, 17277, 17279, 14109, 14126, 9171, 39012, 27044, 24628, 22844,
    18846, 16649, 16755, 14643, 15335, 15817, 16068, 16074, 12544, 12723, 8230, 8941, 15632, 12939, 10963, 13488,
    13618, 13274, 24872, 24878, 25098, 16231, 14072, 10349, 12948, 13088, 38996, 30787, 28816, 28838, 28839, 26817,
    27047, 25114, 25135, 22725, 22966, 23404, 20639, 21033, 18573, 18702, 19111, 17329, 16215, 10456, 14848, 13378,
    18513, 12679, 16633, 14658, 13700, 21006, 10872, 39011, 34893, 20834, 18885, 18962, 19228, 16619, 16845, 15284,
    15378, 16193, 12432, 12609, 13228, 13516, 14388, 14639, 8554, 16693, 16730, 14548, 12774, 10436, 16248, 8878,
    17090, 10283, 8222, 26635, 15303, 29113, 29134, 25334, 15871, 8412, 41084, 25286, 25294, 25324, 25341, 17257,
    14347, 14147, 9142, 13867, 26626, 16110, 20873, 16061, 12702, 10503, 19250, 41053, 26779, 23226, 20635, 20676,
    20839, 20853, 19122, 16635, 16824, 12466, 14038, 18825, 10337, 12962, 24744, 22751, 37023, 13927, 45058, 38971,
    34913, 17031, 15270, 8383, 21135, 18778, 10405, 11117, 22707, 16512, 15633, 28836, 47119, 30825, 25069, 14468,
    15384, 15490, 15751, 15843, 12389, 12868, 12874, 13951, 10297, 10393, 16625, 16718, 15409, 10699, 16713, 23278,
    18501, 16612, 14674, 15952, 16065, 12612, 12808, 45109, 43038, 15070, 15246, 15041, 45088, 41049, 30878, 24842,
    22579, 23321, 20888, 18904, 19197, 19198, 16601, 15263, 15666, 16158, 12900, 8265, 18661, 17020, 10629, 13697,
    13181, 16405, 25297, 17343, 13721, 26707, 26885, 24750, 16665, 14778, 14779, 10429, 41024, 34876, 30790, 28720,
    28803, 26641, 26780, 26958, 24677, 24897, 25080, 25106, 25119, 22600, 22858, 22983, 20636, 20696, 20791, 21001,
    21068, 21103, 18436, 18596, 18599, 18744, 18868, 19058, 16406, 16654, 16996, 14527, 14782, 15094, 15216, 15598,
    15695, 16073, 12290, 23343, 20890, 14690, 25187, 22750, 22852, 23357, 20865, 19085, 19172, 16795, 16819, 14710,
    15006, 15510, 15663, 12348, 12454, 12655, 41036, 38925, 30764, 26638, 15947, 13766, 23124, 12686, 10446, 10583,
    14386, 22720, 21101, 15495, 12805, 23173, 20703, 16629, 14400, 28950, 13790, 20745, 24886, 16099, 15283, 12349,
    10369, 13644, 27129, 27130, 23528, 13254, 13127, 10877, 19022, 15562, 10520, 13674, 13314, 10521, 15400, 13205,
    43023, 37072, 30962, 25399, 25418, 25442, 25510, 19377, 19446, 19452, 14170, 14171, 14185, 13253, 13787, 43010,
    32815, 30772, 30881, 28697, 26741, 27036, 24955, 25255, 23532, 21300, 18751, 16873, 14369, 14456, 14906, 11041,
    9129, 9069, 12981, 13112, 8557, 23302, 10247, 15426, 12718, 10902, 8926, 8938, 39036, 20711, 16621, 10600,
    19254, 13028, 17313, 15821, 9044, 20771, 18449, 8834, 13289, 30755, 9023, 24960, 14793, 12781, 10848, 8469,
    13639, 14363, 14652, 17368, 18696, 10792, 8593, 12998, 13399, 13631, 13499, 47115, 43011, 40986, 41025, 38950,
    32786, 28924, 28987, 26747, 25100, 22558, 22636, 23181, 21004, 18975, 19148, 16903, 14440, 10863, 8397, 16924,
    15710, 23389, 23479, 23508, 20485, 18473, 18676, 19101, 14915, 8909, 17395, 8802, 8997, 8681, 13312, 15228,
    18905, 25235, 13152, 41064, 38960, 30817, 27084, 24852, 25002, 25165, 22547, 22592, 20632, 20706, 18458, 18824,
    15155, 14387, 29046, 14942, 8866, 8962, 22580, 15363, 15813, 10319, 13954, 12483, 11029, 20845, 18519, 14849,
    15620, 8443, 8880, 11129, 16558, 15770, 10570, 8352, 13160, 13187, 13206, 13414, 16953, 14604, 13512, 13649,
    10614, 10971, 30769, 22718, 22669, 20682, 12713, 10746, 10882, 8818, 17219, 47121, 15862, 14813, 21474, 13662,
    13677, 24624, 16839, 10624, 12472, 10504, 17385, 14555, 15659, 12316, 12567, 10873, 9043, 24751, 16627, 14695,
    24714, 20767, 8297, 13145, 19251, 13690, 36916, 26798, 10464, 15299, 15985, 8950, 17039, 8765, 22656, 14687,
    16031, 11158, 12975, 13825, 13813, 9201, 8532, 25360, 25367, 21341, 21356, 21369, 21402, 21409, 21413, 17231,
    22704, 21113, 34912, 14845, 14865, 15233, 11225, 16255, 13419, 8439, 23238, 20633, 16411, 16983, 14359, 14579,
    14830, 14954, 15792, 12625, 10630, 8376, 9207, 13376, 13873, 40982, 24991, 23025, 23237, 20707, 16862, 16000,
    12648, 17083, 15455, 12869, 8931, 34924, 19173, 19301, 15373, 15625, 15891, 10662, 8259, 16848, 28714, 20576,
    17253, 17259, 17285, 14143, 10582, 8704, 9158, 28877, 28893, 26757, 26901, 26926, 27088, 25058, 23351, 23374,
    16587, 10958, 8641, 20884, 18688, 12599, 8547, 21238, 11067, 25157, 10304, 15383, 15688, 15979, 13027, 8634,
    14864, 15111, 15401, 12576, 12593, 11217, 13663, 36945, 15645, 8446, 24952, 15555, 12300, 12502, 8467, 14461,
    14490, 13239, 13397, 13469, 13808, 14041, 30860, 28932, 28942, 26819, 26831, 26865, 24671, 25205, 22752, 22915,
    23300, 20506, 18884, 16732, 15372, 12657, 12790, 8461, 29068, 22877, 16675, 14868, 15533, 15694, 10720, 40975,
    13629, 13305, 32864, 29112, 21337, 17270, 14993, 14098, 16917, 13741, 26866, 18939, 17220, 14605, 15147, 16133,
    12526, 10575, 8403, 8524, 16540, 11249, 13108, 18697, 16731, 14492, 12491, 10468, 15467, 15804, 13564, 21329,
    14080, 25124, 18821, 15101, 10531, 10558, 23050, 22610, 21005, 29125, 15002, 8614, 34816, 19350, 14161, 10595,
    19337, 11105, 13913, 11154, 10474, 15719, 16690, 37082, 37083, 37097, 30903, 30905, 30909, 30917, 30922, 30953,
    25378, 25391, 25432, 25461, 25469, 25474, 25495, 25496, 25517, 25523, 19355, 19362, 19373, 19381, 19419, 19420,
    19424, 14207, 14209, 40995, 36942, 26787, 26804, 15648, 19246, 15829, 20491, 20610, 20981, 14397, 15876, 12709,
    25309, 25342, 18773, 28770, 17398, 38979, 30862, 14408, 12390, 13259, 13409, 47134, 15159, 13333, 16206, 13800,
    13531, 18677, 40987, 22685, 22864, 23218, 23371, 18805, 19296, 16432, 16472, 16966, 17065, 14404, 15075, 41015,
    41037, 41052, 38990, 39023, 37002, 29108, 26705, 26862, 26998, 27042, 24644, 24668, 25083, 25268, 22570, 22846,
    23259, 20790, 21316, 18955, 19182, 16510, 16947, 14413, 14762, 15653, 15691, 13952, 14013, 10377, 10414, 11017,
    8280, 19057, 9130, 20831, 17001, 20654, 18746, 16501, 15835, 13956, 15043, 10643, 41043, 18440, 18894, 14975,
    15098, 16163, 12444, 12751, 10551, 8236, 20843, 18612, 18888, 14451, 8538, 18977, 15097, 15387, 16250, 19267,
    13040, 13056, 13483, 13542, 13907, 17314, 17344, 14224, 25229, 16728, 13771, 23201, 19307, 16235, 12473, 13163,
    24803, 17137, 15668, 10718, 8519, 11260, 11262, 10641, 8295, 15757, 8691, 26788, 27024, 20542, 20687, 8288,
    15304, 16915, 17298, 15566, 14735, 14892, 18656, 12851, 15866, 12992, 13502, 13847, 25306, 25365, 21385, 17245,
    13234, 29004, 24909, 10986, 11072, 9102, 12920, 12940, 13560, 13672, 13848, 13681, 13802, 28991, 20681, 13669,
    18780, 16912, 10856, 8212, 13539, 13451, 15427, 16101, 15040, 15715, 15810, 12533, 10669, 8218, 9073, 8813,
    13024, 10782, 12636, 11108, 16817, 21251, 15831, 20695, 26748, 39024, 24621, 14398, 14811, 9048, 19130, 16956,
    16968, 13227, 8839, 36874, 20700, 29136, 8301, 14034, 16733, 14599, 14902, 14964, 22645, 23296, 23463, 20547,
    8232, 30738, 28833, 26653, 24639, 19154, 16685, 15328, 8993, 23451, 21012, 18679, 15161, 24904, 13651, 13568,
    45063, 45076, 40965, 41005, 41061, 38980, 38997, 36906, 36917, 36948, 37004, 34846, 32830, 32837, 30879, 28744,
    28946, 28984, 29016, 27126, 20800, 21063, 12492, 10268, 10483, 21133, 18792, 14498, 15025, 12678, 8750, 13052,
    19265, 13150, 13960, 14045, 14069, 11096, 24678, 21216, 10332, 21447, 21458, 17308, 17332, 14215, 14232, 13767,
    34907, 20543, 18645, 16744, 10830, 13809, 12997, 13179, 41021, 28792, 27101, 13501, 34845, 32884, 29153, 25533,
    25556, 25589, 21480, 16887, 9214, 15007, 12754, 8923, 37017, 28843, 29061, 25033, 23282, 20964, 20721, 8432,
    17348, 14293, 13388, 13530, 15290, 16294, 20955, 18597, 16418, 16667, 12837, 25542, 25570, 21432, 14785, 10892,
    10941, 37001, 34891, 15186, 8847, 17054, 18950, 14852, 15775, 12447, 12652, 26775, 13315, 21463, 17334, 15329,
    34918, 28802, 29087, 26667, 27053, 22860, 20621, 18543, 14837, 10692, 16617, 10389, 11175, 36995, 25158, 17185,
    8262, 20849, 32804, 12741, 14105, 32784, 13366, 21501, 14310, 10646, 14453, 12357, 16166, 15160, 8624, 11066,
    19248, 16017, 12398, 19015, 15069, 11240, 25215, 25261, 17094, 17101, 14488, 15049, 16181, 12817, 8489, 16894,
    13336, 12906, 14911, 8712, 21457, 17374, 22939, 12857, 8513, 15471, 17305, 32798, 25213, 23063, 23200, 23546,
    20515, 18456, 18757, 16827, 16993, 15012, 12442, 10637, 13144, 26906, 11221, 27104, 24913, 22993, 23363, 16991,
    17035, 16127, 8248, 24837, 20891, 18590, 13691, 13493, 15456, 23091, 15989, 12888, 8784, 12621, 8434, 8561,
    15366, 12910, 13695, 26792, 28951, 21104, 14496, 15309, 8555, 22702, 12580, 20511, 21136, 18957, 15163, 15822,
    12904, 10417, 12973, 10871, 8353, 13010, 13555, 14085, 15893, 16020, 34828, 28737, 25088, 18901, 19141, 22601,
    22904, 22961, 23335, 20709, 28791, 24844, 25216, 23536, 20514, 18432, 16805, 16892, 14987, 12846, 41040, 30873,
    22542, 22598, 23097, 23263, 20589, 16056, 16260, 12704, 10305, 16979, 16120, 20854, 16168, 22544, 20647, 20652,
    21257, 18538, 18595, 18728, 20901, 18837, 14417, 15074, 15107, 16067, 12294, 12811, 8793, 11121, 11125, 13142,
    13420, 13783, 13814, 13821, 13865, 12834, 10382, 30741, 30823, 28858, 26907, 24774, 23099, 10706, 38918, 28822,
    26974, 24599, 23542, 20738, 20875, 21214, 18831, 14507, 10497, 11079, 11176, 8377, 38970, 36957, 34939, 30791,
    30856, 28743, 28929, 29067, 26674, 26677, 26756, 26822, 26899, 26946, 27039, 27135, 24601, 24649, 24653, 24733,
    24832, 24892, 24919, 24966, 24967, 24970, 24971, 25054, 25079, 25168, 25256, 22767, 22823, 22975, 23056, 23167,
    23328, 23464, 20602, 21045, 21129, 21263, 18906, 19027, 16533, 16623, 16896, 17177, 17202, 14740, 15713, 15766,
    16353, 17096, 25338, 21336, 21339, 21405, 21421, 17244, 14136, 25228, 19242, 12917, 13194, 49165, 8569, 13789,
    16242, 8372, 8683, 20714, 8363, 21111, 15522, 8778, 41031, 20814, 19363, 22698, 13875, 11005, 10664, 17098,
    15021, 34906, 26799, 22925, 20749, 20765, 21056, 18852, 8621, 36961, 20551, 20907, 20988, 18705, 18844, 19135,
    17136, 14354, 15035, 16302, 28788, 15088, 15164, 10426, 8245, 22847, 20769, 16959, 8867, 13712, 12935, 13383,
    13622, 34829, 28679, 26645, 26973, 24604, 25206, 22564, 18642, 16852, 12497, 13429, 13576, 15515, 8534, 11150,
    9120, 16174, 41075, 41078, 41082, 32869, 15192, 8688, 23182, 8510, 38912, 16631, 15251, 28781, 23506, 23540,
    21223, 21272, 18807, 19327, 17142, 15272, 16139, 12742, 10622, 8394, 25089, 15486, 15761, 15789, 21271, 45075,
    36905, 24686, 24729, 23349, 8988, 19341, 14668, 12733, 29097, 26969, 21256, 12866, 13675, 10795, 8630, 17150,
    14877, 13168, 13479, 13794, 16177, 23511, 19200, 8776, 25313, 22588, 16106, 8418, 8518, 39034, 23077, 18434,
    16414, 13449, 13617, 10666, 39029, 15325, 14798, 8537, 14601, 14923, 8577, 13096, 10638, 15665, 10663, 11236,
    8539, 22593, 18557, 15595, 17028, 12424, 12613, 10376, 15703, 10705, 8470, 10959, 8625, 13711, 13037, 16930,
    14981, 15030, 15699, 12669, 12993, 13071, 13563, 13769, 23533, 20768, 16920, 8747, 21467, 20570, 18802, 9052,
    24641, 16367, 8449, 16104, 13359, 13849, 12873, 15725, 12371, 14021, 16154, 9027, 14294, 19128, 16885, 38959,
    36886, 36953, 36986, 29021, 29078, 26773, 26971, 24710, 24943, 22831, 23336, 20759, 20797, 20900, 18527, 18706,
    16054, 10266, 8742, 13189, 17206, 12410, 11049, 9094, 13095, 13654, 14598, 10860, 16985, 14615, 15960, 10814,
    8226, 18602, 10572, 49176, 37069, 37070, 37084, 37087, 30889, 30928, 30931, 25398, 25430, 25441, 25486, 25514,
    19396, 14197, 23471, 16566, 12549, 14009, 11120, 9125, 13897, 11052, 37043, 37050, 37056, 30914, 30924, 30930,
    30942, 30943, 30949, 30970, 25380, 25427, 25443, 25492, 25499, 19361, 19374, 19407, 19416, 19418, 19443, 19451,
    14178, 14190, 14204, 15514, 15856, 8329, 13602, 16311, 16345, 16347, 13949, 13879, 45107, 23348, 39009, 39025,
    36977, 34909, 29013, 29036, 26625, 27046, 22668, 23388, 20544, 20750, 20883, 20949, 21009, 21070, 18617, 18667,
    18928, 16251, 10329, 10935, 13275, 13448, 36893, 26796, 24806, 15016, 15314, 15349, 15391, 12471, 10246, 22591,
    12317, 14028, 8503, 28953, 23242, 16474, 16561, 16799, 14506, 14249, 34941, 30821, 26753, 22619, 40971, 30725,
    24593, 16935, 14380, 14743, 14898, 15468, 12362, 8686, 10536, 8468, 13811, 22971, 23285, 21020, 18922, 14557,
    15910, 12830, 10330, 10398, 8199, 21010, 18607, 14924, 13074, 14916, 17011, 14424, 18879, 18886, 14807, 12345,
    11112, 17229, 17300, 13023, 34877, 12902, 10271, 24591, 25008, 25053, 22986, 16714, 20940, 15536, 10761, 8722,
    40991, 38931, 36879, 36920, 36923, 36991, 37018, 34817, 34837, 34840, 32855, 28931, 26693, 24588, 24657, 24895,
    19001, 16202, 12712, 16199, 10835, 12430, 10253, 26948, 23412, 21194, 18980, 16706, 17139, 14452, 12385, 12436,
    15808, 15890, 32812, 30746, 25181, 22690, 22706, 22770, 22980, 15407, 15877, 16103, 16142, 34931, 13694, 13831,
    15447, 20806, 18544, 19082, 16385, 8922, 24977, 25217, 22794, 22900, 23472, 18625, 18673, 18768, 19314, 16505,
    16528, 16743, 14825, 15972, 12815, 10728, 14271, 13443, 13863, 13726, 27121, 24725, 20724, 21179, 18463, 10617,
    34917, 16359, 32803, 11083, 8746, 10776, 34823, 23153, 18511, 10447, 10657, 8616, 17169, 14672, 24673, 20838,
    18450, 15105, 14065, 9088, 36925, 28937, 25200, 22555, 22994, 18655, 15086, 37019, 24814, 10901, 8795, 21205,
    14567, 15869, 12416, 8203, 8668, 13840, 16143, 8872, 22963, 23016, 20804, 21287, 18452, 40973, 20961, 18724,
    15639, 12391, 12438, 10352, 20836, 14251, 14288, 13532, 24594, 20684, 13155, 24804, 22546, 23057, 18482, 18818,
    16399, 8900, 41020, 36974, 26955, 25161, 22962, 20553, 18492, 18767, 19336, 16820, 16847, 14445, 14587, 14928,
    14973, 15106, 16037, 8755, 8694, 20641, 12635, 22861, 18986, 19155, 16854, 15415, 15759, 16038, 12572, 10342,
    10740, 8225, 8406, 14582, 15176, 30811, 19171, 16955, 12298, 10866, 16998, 20717, 38917, 38956, 26648, 26734,
    12814, 10258, 45086, 45089, 15306, 15433, 10702, 8501, 13344, 13992, 13324, 13603, 37007, 29050, 26742, 24827,
    23018, 23353, 20761, 20912, 18517, 15079, 10326, 10764, 10984, 25012, 20587, 36958, 28814, 26995, 25246, 20603,
    20894, 9119, 21437, 18588, 15485, 11018, 10742, 22934, 20763, 16580, 13600, 11095, 15932, 20525, 18878, 15736,
    9195, 13385, 13820, 40969, 32774, 32780, 30770, 28810, 29064, 26777, 26920, 25065, 22977, 21021, 21243, 19149,
    16929, 16952, 17026, 14948, 32836, 30819, 30849, 27079, 14508, 13317, 8760, 8924, 13262, 28780, 28978, 26692,
    26718, 24939, 20857, 17117, 16125, 8620, 25241, 16351, 21286, 13827, 26806, 15899, 38994, 13911, 13969, 17262,
    14121, 14149, 13857, 45098, 43031, 43033, 40976, 40979, 41026, 38937, 38981, 36878, 36998, 34841, 32814, 30760,
    30779, 28823, 26986, 24669, 25077, 36931, 29023, 29035, 26932, 20526, 21299, 19234, 17099, 14036, 9186, 12762,
    10587, 8255, 10989, 14425, 8837, 13751, 15313, 13060, 13425, 21373, 41008, 11064, 11181, 22634, 16614, 14776,
    8408, 11040, 11092, 9106, 28840, 26739, 20534, 20896, 14900, 11116, 9110, 14052, 49155, 10514, 8284, 21331,
    10983, 14616, 15969, 8241, 13167, 16431, 9057, 13490, 23066, 23270, 18479, 16581, 16814, 15344, 15896, 10512,
    11148, 11182, 13743, 20672, 21220, 18881, 19323, 16247, 8953, 8599, 14519, 8752, 27114, 29094, 26631, 26695,
    26801, 22563, 22618, 13019, 8667, 9011, 8767, 40978, 36939, 34830, 34861, 32860, 15154, 22605, 22828, 19125,
    16793, 14972, 15956, 12510, 37092, 37095, 37104, 25402, 25425, 25433, 25436, 25462, 25508, 19399, 19427, 19447,
    14168, 14175, 14206, 14212, 12921, 13596, 8907, 9009, 12951, 11122, 13202, 21345, 17269, 8618, 34884, 22560,
    22661, 22749, 20810, 21073, 21297, 18911, 19163, 16624, 16783, 17059, 14990, 15081, 16305, 12346, 18970, 14438,
    14727, 26847, 25210, 22997, 20879, 18571, 10806, 14399, 15288, 12336, 16387, 15560, 15892, 12402, 18699, 16960,
    8853, 11173, 12980, 13109, 13365, 21232, 12926, 8567, 13299, 18528, 41030, 36956, 32810, 28759, 28854, 26772,
    24606, 28764, 27071, 24613, 24944, 20963, 19032, 16721, 17122, 14392, 16325, 13686, 21146, 16630, 14823, 10634,
    13291, 15202, 15499, 8520, 17384, 17128, 15507, 11073, 8293, 10966, 20656, 24722, 23356, 21112, 14389, 16078,
    12297, 12685, 9031, 24799, 24841, 22945, 37110, 37111, 14219, 14295, 25110, 22557, 16582, 16107, 13134, 20486,
    16552, 10397, 8378, 11115, 26702, 18909, 13562, 13923, 14076, 11013, 11231, 16965, 14447, 14035, 26782, 15700,
    16243, 13161, 15011, 8596, 8851, 10631, 16725, 17052, 14558, 13985, 10780, 9075, 12366, 13748, 13250, 10700,
    22562, 8391, 12676, 41080, 29146, 24933, 22969, 23251, 23504, 21082, 18765, 18811, 19115, 12896, 8703, 45105,
    25564, 17091, 12905, 11178, 13044, 13437, 16895, 13871, 22813, 14680, 20995, 16045, 12757, 15274, 8692, 12671,
    10308, 18861, 16872, 16270, 12641, 8529, 40983, 30796, 30867, 28982, 26700, 24586, 25243, 25245, 22643, 22898,
    22933, 23132, 21305, 18567, 18609, 19292, 16643, 14572, 15225, 16131, 25034, 23163, 23329, 21196, 19048, 14376,
    14536, 15242, 15535, 12532, 10262, 8207, 8682, 25332, 13792, 13045, 16071, 13554, 18555, 16492, 15127, 10838,
];

/*
Copyright (c) 2026 Guillaume Piolat.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
