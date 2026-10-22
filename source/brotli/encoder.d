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
// more than minLength bytes, with any transform but "omit first n": for each
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
                              size_t maxDistance, size_t minLength, ref DictionaryReferences r)
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
    uint h = dictionaryHash(load32(data, pos));
    foreach (entry; kOmitFirstHashTable.words[kOmitFirstHashTable.bucketStart[h]
                                              .. kOmitFirstHashTable.bucketStart[h + 1]])
    {
        const(ubyte)[] output = omitFirstOutput(entry);
        if (output.length < OMIT_FIRST_MIN_LENGTH || output.length > maxLength || output.length <= minLength
            || output != data[pos .. pos + output.length])
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

@tuning enum OMIT_FIRST_MIN_LENGTH = 5; // shortest "omit first n" output tried (the index has 4 and up)
enum NUM_OMIT_FIRST_CANDIDATES = 58880;
enum NUM_OMIT_FIRST_WORDS = 40035;

// Dictionary words with their first n bytes omitted (n with a transform, at
// least 4 bytes left), by hash of their first 4 bytes left: the words of hash
// h are words[bucketStart[h] .. bucketStart[h + 1]], as
// (n << 16) | (length << 11) | index. Of the ones that produce the same
// bytes, only the cheapest (smallest distance, then length) is kept, and
// none that a dictionary word produces as is.
struct OmitFirstHashTable
{
    ushort[(1 << DICTIONARY_HASH_BITS) + 1] bucketStart;
    uint[NUM_OMIT_FIRST_WORDS] words;
}

// The values of buildOmitFirstHashTable(), see the unittest.
static immutable OmitFirstHashTable kOmitFirstHashTable =
    OmitFirstHashTable(kOmitFirstBucketStart, kOmitFirstWords);

const(ubyte)[] omitFirstOutput(uint entry)
{
    size_t n = entry >> 16;
    size_t len = (entry >> 11) & 31;
    size_t offset = kDictionaryOffsets[len] + len * (entry & 2047);
    return dictionary()[offset + n .. offset + len];
}

// (distance, length) of an "omit first n" entry, to compare costs.
size_t omitFirstCost(uint entry)
{
    size_t len = (entry >> 11) & 31;
    size_t distance = (entry & 2047) + (size_t(kOmitFirstTransforms[entry >> 16]) << kDictionarySizeBits[len]);
    return (distance << 5) | len;
}

// `candidates`: scratch space for NUM_OMIT_FIRST_CANDIDATES entries.
void buildOmitFirstHashTable(ref OmitFirstHashTable t, scope uint[] candidates)
{
    // All candidates, by hash.
    uint[(1 << DICTIONARY_HASH_BITS) + 1] start;
    size_t num = 0;
    foreach (pass; 0 .. 2)
    {
        uint[1 << DICTIONARY_HASH_BITS] fill;
        foreach (len; MIN_DICTIONARY_WORD_LENGTH .. MAX_DICTIONARY_WORD_LENGTH + 1)
        {
            foreach (idx; 0 .. 1u << kDictionarySizeBits[len])
            {
                foreach (n; 1 .. 10)
                {
                    if (kOmitFirstTransforms[n] == 0 || len - n < 4)
                        continue;
                    uint entry = cast(uint)((n << 16) | (len << 11) | idx);
                    uint h = dictionaryHash(load32(omitFirstOutput(entry), 0));
                    if (pass == 0)
                    {
                        fill[h]++;
                        num++;
                    }
                    else
                        candidates[start[h] + fill[h]++] = entry;
                }
            }
        }
        if (pass == 0)
        {
            assert(num == NUM_OMIT_FIRST_CANDIDATES);
            uint sum = 0;
            foreach (h; 0 .. 1 << DICTIONARY_HASH_BITS)
            {
                start[h] = sum;
                sum += fill[h];
            }
            start[1 << DICTIONARY_HASH_BITS] = sum;
        }
    }

    // Keep the cheapest of each output, if not a dictionary word.
    immutable(ubyte)[] dict = dictionary();
    size_t n = 0;
    foreach (h; 0 .. 1 << DICTIONARY_HASH_BITS)
    {
        t.bucketStart[h] = cast(ushort) n;
        const(uint)[] bucket = candidates[start[h] .. start[h + 1]];
        foreach (entry; bucket)
        {
            const(ubyte)[] output = omitFirstOutput(entry);
            bool keep = true;
            foreach (other; bucket)
                if (omitFirstCost(other) < omitFirstCost(entry) && omitFirstOutput(other) == output)
                    keep = false;
            foreach (word; kDictionaryHashTable.words[kDictionaryHashTable.bucketStart[h]
                                                      .. kDictionaryHashTable.bucketStart[h + 1]])
            {
                size_t len = word >> 11;
                size_t offset = kDictionaryOffsets[len] + len * (word & 2047);
                if (dict[offset .. offset + len] == output)
                    keep = false;
            }
            if (keep)
                t.words[n++] = entry;
        }
    }
    assert(n == NUM_OMIT_FIRST_WORDS);
    t.bucketStart[1 << DICTIONARY_HASH_BITS] = cast(ushort) n;
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

    // windowSize: power of 2, at most MAX_TREE_WINDOW. Matches are found
    // up to windowSize - 16 bytes back.
    bool initialize(size_t windowSize)
    {
        assert(windowSize <= MAX_TREE_WINDOW);
        windowMask = windowSize - 1;
        if (!buckets.resize(size_t(1) << TREE_BUCKET_BITS) || !forest.resize(2 * windowSize))
            return false;
        buckets[] = TREE_INVALID; // forest entries are written when a position is inserted
        base = 0;
        return true;
    }

    void release()
    {
        buckets.release();
        forest.release();
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
    if (!findDictionaryReferences(data, pos, maxLength, maxDistance, bestLen, refs))
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
        if (!t.initialize(windowSize))
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
    // The "omit first n" index has the values of buildOmitFirstHashTable().
    MallocArray!OmitFirstHashTable t;
    MallocArray!uint candidates;
    assert(t.resize(1) && candidates.resize(NUM_OMIT_FIRST_CANDIDATES));
    buildOmitFirstHashTable(t[0], candidates[]);
    assert(t[0].bucketStart == kOmitFirstHashTable.bucketStart);
    assert(t[0].words == kOmitFirstHashTable.words);
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
        assert(t.initialize(size_t(1) << windowBits));
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

// "Omit first n" dictionary index, see OmitFirstHashTable.
static immutable ushort[(1 << DICTIONARY_HASH_BITS) + 1] kOmitFirstBucketStart =
[
    0, 5, 10, 13, 24, 37, 42, 42, 46, 48, 48, 48, 49, 49, 49, 49,
    51, 51, 51, 77, 77, 89, 93, 96, 99, 99, 99, 100, 100, 108, 108, 112,
    113, 113, 119, 119, 119, 119, 120, 122, 124, 125, 125, 125, 126, 128, 128, 129,
    130, 130, 133, 135, 138, 139, 139, 143, 172, 172, 172, 173, 175, 175, 177, 178,
    179, 179, 181, 181, 182, 183, 183, 184, 185, 185, 185, 187, 194, 196, 197, 197,
    200, 200, 202, 203, 206, 234, 240, 240, 243, 245, 254, 254, 257, 258, 259, 261,
    261, 261, 261, 262, 262, 262, 262, 264, 265, 265, 271, 276, 280, 282, 284, 284,
    285, 286, 286, 287, 290, 290, 291, 294, 294, 296, 296, 296, 301, 301, 301, 303,
    303, 308, 308, 310, 312, 312, 312, 313, 313, 314, 314, 321, 321, 324, 324, 326,
    326, 326, 326, 327, 336, 336, 338, 341, 346, 349, 351, 362, 373, 375, 381, 404,
    405, 405, 405, 407, 409, 410, 410, 411, 412, 416, 416, 416, 417, 420, 421, 423,
    433, 433, 433, 439, 439, 440, 444, 444, 444, 445, 445, 445, 445, 447, 447, 449,
    451, 451, 453, 453, 453, 454, 457, 459, 459, 460, 463, 463, 465, 467, 468, 469,
    470, 471, 479, 481, 481, 481, 481, 483, 485, 485, 485, 487, 488, 488, 488, 495,
    497, 497, 497, 498, 498, 502, 502, 502, 502, 502, 504, 509, 512, 517, 522, 523,
    523, 523, 523, 524, 524, 524, 525, 529, 529, 530, 530, 531, 534, 534, 535, 535,
    536, 538, 546, 547, 547, 556, 556, 556, 556, 562, 562, 564, 567, 567, 568, 568,
    568, 570, 570, 570, 572, 576, 576, 576, 576, 577, 577, 579, 579, 581, 581, 581,
    584, 584, 587, 588, 589, 589, 592, 593, 593, 594, 604, 605, 605, 605, 608, 616,
    619, 623, 624, 624, 625, 625, 626, 626, 627, 627, 628, 628, 628, 629, 630, 634,
    652, 652, 652, 660, 669, 672, 672, 672, 672, 673, 676, 681, 684, 684, 685, 685,
    685, 686, 694, 695, 698, 702, 702, 716, 717, 724, 724, 724, 727, 727, 727, 730,
    730, 734, 735, 743, 743, 743, 743, 754, 756, 757, 767, 768, 768, 772, 775, 776,
    786, 787, 787, 792, 793, 802, 803, 804, 804, 809, 809, 809, 811, 812, 812, 812,
    812, 812, 812, 813, 813, 813, 814, 821, 822, 822, 823, 824, 825, 829, 829, 830,
    928, 932, 932, 933, 935, 935, 938, 940, 941, 942, 944, 944, 970, 973, 973, 975,
    975, 976, 979, 979, 982, 982, 983, 985, 985, 987, 999, 1004, 1007, 1007, 1009, 1013,
    1013, 1013, 1014, 1016, 1017, 1021, 1021, 1022, 1024, 1024, 1025, 1028, 1029, 1030, 1031, 1031,
    1033, 1036, 1041, 1056, 1057, 1061, 1064, 1064, 1064, 1064, 1066, 1066, 1067, 1074, 1074, 1075,
    1075, 1076, 1077, 1082, 1082, 1083, 1083, 1084, 1084, 1099, 1102, 1103, 1105, 1106, 1111, 1111,
    1112, 1112, 1112, 1112, 1112, 1115, 1117, 1117, 1117, 1117, 1117, 1117, 1118, 1118, 1120, 1122,
    1122, 1122, 1122, 1126, 1127, 1131, 1133, 1133, 1139, 1139, 1139, 1139, 1141, 1150, 1151, 1151,
    1152, 1154, 1161, 1163, 1163, 1164, 1164, 1165, 1165, 1165, 1166, 1167, 1170, 1170, 1170, 1172,
    1177, 1177, 1178, 1178, 1181, 1183, 1189, 1190, 1190, 1191, 1191, 1192, 1197, 1197, 1200, 1203,
    1203, 1203, 1205, 1208, 1212, 1218, 1218, 1220, 1220, 1220, 1222, 1227, 1227, 1228, 1228, 1231,
    1250, 1254, 1254, 1262, 1262, 1262, 1263, 1269, 1269, 1288, 1292, 1292, 1292, 1292, 1293, 1293,
    1296, 1297, 1298, 1300, 1301, 1301, 1301, 1301, 1302, 1302, 1304, 1304, 1305, 1306, 1306, 1307,
    1307, 1307, 1307, 1308, 1317, 1317, 1317, 1318, 1323, 1324, 1324, 1334, 1346, 1351, 1365, 1365,
    1368, 1369, 1371, 1375, 1376, 1377, 1377, 1380, 1380, 1383, 1384, 1384, 1391, 1391, 1392, 1393,
    1393, 1393, 1395, 1402, 1404, 1404, 1404, 1404, 1404, 1404, 1405, 1405, 1405, 1406, 1407, 1411,
    1422, 1422, 1423, 1423, 1433, 1434, 1434, 1437, 1438, 1445, 1448, 1448, 1448, 1448, 1448, 1448,
    1448, 1449, 1453, 1460, 1461, 1461, 1462, 1462, 1464, 1469, 1472, 1472, 1477, 1477, 1478, 1491,
    1497, 1521, 1523, 1529, 1529, 1529, 1540, 1540, 1545, 1545, 1548, 1548, 1549, 1551, 1551, 1552,
    1552, 1555, 1558, 1563, 1563, 1565, 1568, 1576, 1580, 1586, 1586, 1586, 1586, 1586, 1588, 1588,
    1588, 1589, 1591, 1592, 1593, 1593, 1595, 1614, 1614, 1614, 1615, 1619, 1626, 1627, 1633, 1634,
    1634, 1634, 1636, 1645, 1645, 1646, 1647, 1647, 1647, 1648, 1659, 1660, 1661, 1661, 1662, 1664,
    1665, 1666, 1666, 1668, 1669, 1669, 1678, 1678, 1679, 1679, 1685, 1686, 1690, 1691, 1695, 1699,
    1717, 1718, 1724, 1725, 1731, 1731, 1731, 1731, 1731, 1732, 1740, 1740, 1742, 1744, 1752, 1752,
    1752, 1763, 1763, 1767, 1767, 1767, 1768, 1770, 1770, 1771, 1771, 1771, 1772, 1772, 1774, 1777,
    1777, 1778, 1779, 1780, 1781, 1781, 1783, 1784, 1784, 1784, 1786, 1794, 1796, 1800, 1802, 1808,
    1827, 1829, 1830, 1832, 1844, 1844, 1844, 1853, 1853, 1854, 1855, 1855, 1856, 1858, 1858, 1858,
    1869, 1874, 1874, 1875, 1875, 1876, 1882, 1886, 1888, 1889, 1903, 1906, 1906, 1908, 1914, 1914,
    1917, 1925, 1925, 1940, 1942, 1942, 1945, 1945, 1947, 1947, 1947, 1950, 1950, 1950, 1951, 1951,
    1971, 1971, 1975, 1976, 1989, 1992, 1992, 1994, 1994, 1994, 2001, 2001, 2002, 2002, 2002, 2004,
    2011, 2011, 2011, 2011, 2012, 2013, 2013, 2014, 2031, 2036, 2043, 2044, 2048, 2053, 2056, 2056,
    2057, 2061, 2061, 2061, 2067, 2067, 2067, 2070, 2070, 2077, 2078, 2078, 2079, 2097, 2101, 2106,
    2107, 2107, 2107, 2107, 2109, 2110, 2110, 2110, 2113, 2114, 2115, 2125, 2126, 2134, 2137, 2138,
    2138, 2138, 2139, 2140, 2140, 2142, 2142, 2143, 2143, 2144, 2144, 2161, 2161, 2166, 2169, 2170,
    2175, 2176, 2185, 2185, 2199, 2199, 2199, 2209, 2209, 2209, 2209, 2209, 2209, 2211, 2211, 2219,
    2219, 2219, 2220, 2221, 2221, 2222, 2222, 2230, 2230, 2230, 2230, 2230, 2230, 2231, 2231, 2233,
    2238, 2240, 2240, 2264, 2265, 2266, 2266, 2267, 2268, 2273, 2273, 2274, 2274, 2274, 2276, 2280,
    2284, 2286, 2287, 2288, 2290, 2290, 2296, 2299, 2315, 2319, 2320, 2321, 2322, 2324, 2325, 2329,
    2338, 2338, 2345, 2345, 2347, 2348, 2348, 2348, 2348, 2348, 2361, 2363, 2366, 2369, 2369, 2370,
    2370, 2370, 2370, 2370, 2370, 2371, 2372, 2372, 2373, 2374, 2374, 2375, 2382, 2384, 2427, 2427,
    2432, 2440, 2447, 2447, 2447, 2461, 2461, 2463, 2463, 2463, 2463, 2463, 2466, 2466, 2470, 2470,
    2470, 2470, 2470, 2473, 2473, 2473, 2473, 2478, 2490, 2491, 2491, 2496, 2496, 2497, 2497, 2497,
    2497, 2497, 2500, 2500, 2500, 2505, 2508, 2508, 2509, 2511, 2518, 2519, 2519, 2519, 2533, 2543,
    2547, 2560, 2561, 2563, 2567, 2568, 2568, 2575, 2581, 2581, 2582, 2582, 2594, 2597, 2598, 2598,
    2617, 2625, 2629, 2631, 2631, 2631, 2639, 2643, 2646, 2648, 2648, 2655, 2658, 2661, 2661, 2661,
    2661, 2661, 2662, 2664, 2666, 2667, 2667, 2669, 2669, 2669, 2674, 2676, 2676, 2676, 2676, 2678,
    2680, 2681, 2682, 2688, 2688, 2689, 2689, 2693, 2694, 2695, 2709, 2712, 2713, 2716, 2716, 2716,
    2717, 2717, 2717, 2717, 2721, 2723, 2725, 2725, 2728, 2734, 2735, 2736, 2742, 2742, 2748, 2748,
    2748, 2750, 2750, 2750, 2750, 2750, 2750, 2751, 2752, 2752, 2752, 2752, 2752, 2761, 2765, 2766,
    2786, 2787, 2787, 2788, 2789, 2790, 2793, 2793, 2794, 2797, 2797, 2797, 2797, 2798, 2800, 2801,
    2810, 2810, 2821, 2821, 2821, 2821, 2823, 2829, 2829, 2833, 2848, 2858, 2858, 2858, 2858, 2866,
    2866, 2867, 2867, 2868, 2869, 2874, 2880, 2883, 2885, 2885, 2886, 2894, 2897, 2897, 2898, 2900,
    2902, 2902, 2910, 2915, 2916, 2919, 2920, 2924, 2925, 2925, 2927, 2936, 2937, 2937, 2937, 2937,
    2938, 2938, 2939, 2941, 2946, 2952, 2952, 2954, 2954, 2959, 2959, 2961, 2961, 2964, 2966, 2967,
    2976, 2976, 2976, 2977, 2978, 2978, 2981, 2981, 2981, 2982, 2982, 2982, 2982, 2983, 2985, 2990,
    2996, 2997, 2997, 3002, 3003, 3003, 3003, 3006, 3009, 3009, 3037, 3041, 3042, 3042, 3043, 3045,
    3045, 3047, 3053, 3054, 3055, 3055, 3055, 3055, 3073, 3074, 3080, 3080, 3083, 3086, 3086, 3087,
    3098, 3098, 3098, 3101, 3102, 3102, 3107, 3118, 3118, 3123, 3123, 3124, 3132, 3132, 3133, 3136,
    3136, 3146, 3146, 3146, 3157, 3164, 3164, 3165, 3165, 3165, 3169, 3173, 3186, 3186, 3187, 3193,
    3193, 3203, 3203, 3223, 3226, 3234, 3234, 3234, 3234, 3238, 3238, 3238, 3241, 3241, 3241, 3248,
    3248, 3255, 3267, 3267, 3267, 3267, 3273, 3273, 3275, 3275, 3275, 3281, 3282, 3310, 3311, 3321,
    3325, 3326, 3330, 3331, 3335, 3335, 3335, 3335, 3335, 3338, 3338, 3338, 3342, 3342, 3343, 3347,
    3355, 3356, 3357, 3358, 3358, 3358, 3361, 3361, 3361, 3361, 3362, 3363, 3365, 3365, 3369, 3375,
    3377, 3378, 3378, 3378, 3385, 3387, 3391, 3394, 3398, 3398, 3399, 3399, 3399, 3400, 3410, 3413,
    3420, 3422, 3422, 3424, 3430, 3430, 3430, 3439, 3440, 3440, 3440, 3440, 3440, 3440, 3440, 3440,
    3441, 3441, 3446, 3449, 3453, 3456, 3456, 3456, 3457, 3458, 3458, 3458, 3458, 3459, 3459, 3466,
    3466, 3470, 3475, 3492, 3492, 3495, 3497, 3499, 3500, 3502, 3505, 3506, 3508, 3508, 3523, 3523,
    3523, 3529, 3529, 3529, 3529, 3544, 3544, 3545, 3547, 3549, 3550, 3551, 3551, 3552, 3552, 3554,
    3556, 3556, 3561, 3562, 3563, 3608, 3609, 3612, 3612, 3613, 3613, 3613, 3618, 3618, 3628, 3631,
    3631, 3631, 3634, 3638, 3638, 3640, 3641, 3641, 3642, 3647, 3648, 3649, 3650, 3651, 3651, 3652,
    3658, 3658, 3658, 3658, 3658, 3659, 3659, 3660, 3661, 3662, 3663, 3663, 3663, 3667, 3667, 3672,
    3676, 3683, 3684, 3688, 3689, 3689, 3689, 3692, 3692, 3692, 3695, 3695, 3696, 3698, 3698, 3699,
    3699, 3704, 3704, 3704, 3705, 3709, 3713, 3723, 3726, 3726, 3726, 3744, 3745, 3746, 3747, 3747,
    3761, 3761, 3761, 3761, 3763, 3781, 3788, 3794, 3796, 3797, 3797, 3806, 3806, 3825, 3825, 3826,
    3829, 3833, 3833, 3834, 3835, 3835, 3835, 3860, 3865, 3865, 3866, 3866, 3867, 3884, 3884, 3884,
    3889, 3889, 3890, 3892, 3905, 3905, 3912, 3912, 3913, 3913, 3913, 3913, 3913, 3913, 3917, 3917,
    3929, 3929, 3929, 3933, 3934, 3934, 3934, 3937, 3937, 3938, 3940, 3941, 3943, 3944, 3948, 3950,
    3950, 3952, 3973, 3980, 3980, 3980, 3983, 3983, 3983, 3983, 3983, 3988, 3988, 3989, 3989, 3989,
    3989, 3990, 3990, 3992, 3992, 3998, 3998, 3999, 3999, 4000, 4002, 4002, 4002, 4003, 4004, 4007,
    4010, 4012, 4012, 4014, 4016, 4017, 4017, 4017, 4018, 4018, 4030, 4035, 4035, 4042, 4043, 4047,
    4047, 4051, 4057, 4058, 4064, 4069, 4074, 4075, 4075, 4081, 4084, 4086, 4086, 4089, 4090, 4092,
    4095, 4097, 4098, 4106, 4106, 4106, 4112, 4113, 4113, 4114, 4114, 4117, 4117, 4117, 4117, 4134,
    4135, 4135, 4136, 4151, 4156, 4159, 4160, 4160, 4161, 4161, 4162, 4163, 4163, 4164, 4166, 4167,
    4173, 4173, 4173, 4178, 4179, 4179, 4200, 4208, 4209, 4211, 4211, 4211, 4213, 4213, 4213, 4214,
    4215, 4215, 4215, 4215, 4217, 4227, 4229, 4230, 4230, 4230, 4232, 4240, 4242, 4242, 4243, 4248,
    4250, 4266, 4269, 4270, 4271, 4272, 4276, 4279, 4284, 4284, 4285, 4288, 4289, 4290, 4295, 4295,
    4295, 4300, 4301, 4301, 4301, 4301, 4301, 4301, 4304, 4312, 4313, 4316, 4317, 4328, 4329, 4329,
    4331, 4331, 4332, 4332, 4332, 4334, 4335, 4343, 4343, 4343, 4343, 4343, 4343, 4344, 4344, 4349,
    4349, 4349, 4354, 4354, 4354, 4362, 4363, 4365, 4365, 4367, 4367, 4369, 4370, 4370, 4371, 4371,
    4371, 4373, 4373, 4374, 4379, 4382, 4382, 4383, 4383, 4386, 4386, 4393, 4395, 4395, 4395, 4395,
    4397, 4397, 4399, 4399, 4399, 4399, 4400, 4400, 4400, 4401, 4403, 4403, 4403, 4406, 4407, 4407,
    4409, 4415, 4428, 4428, 4428, 4432, 4433, 4436, 4437, 4437, 4437, 4438, 4438, 4439, 4441, 4441,
    4449, 4451, 4451, 4451, 4452, 4452, 4455, 4455, 4465, 4465, 4465, 4479, 4485, 4491, 4496, 4496,
    4496, 4496, 4500, 4505, 4507, 4510, 4510, 4511, 4511, 4513, 4513, 4514, 4517, 4518, 4518, 4520,
    4521, 4528, 4528, 4534, 4534, 4534, 4539, 4540, 4541, 4542, 4542, 4542, 4544, 4546, 4546, 4546,
    4549, 4549, 4552, 4552, 4553, 4555, 4555, 4555, 4567, 4570, 4571, 4572, 4578, 4578, 4595, 4595,
    4605, 4605, 4606, 4609, 4614, 4614, 4614, 4614, 4614, 4627, 4629, 4640, 4640, 4641, 4646, 4647,
    4649, 4650, 4650, 4651, 4654, 4654, 4655, 4655, 4655, 4655, 4655, 4656, 4656, 4662, 4665, 4670,
    4685, 4686, 4691, 4697, 4697, 4698, 4698, 4698, 4698, 4703, 4717, 4722, 4722, 4729, 4730, 4731,
    4731, 4736, 4737, 4738, 4739, 4739, 4741, 4742, 4745, 4758, 4758, 4758, 4759, 4761, 4761, 4761,
    4761, 4763, 4768, 4776, 4786, 4796, 4800, 4802, 4804, 4804, 4808, 4808, 4809, 4811, 4811, 4815,
    4815, 4815, 4815, 4823, 4823, 4834, 4836, 4836, 4840, 4841, 4843, 4858, 4858, 4861, 4861, 4861,
    4861, 4861, 4865, 4871, 4874, 4874, 4875, 4878, 4878, 4878, 4880, 4881, 4884, 4888, 4888, 4889,
    4891, 4892, 4892, 4892, 4899, 4899, 4904, 4909, 4915, 4916, 4918, 4918, 4918, 4919, 4919, 4920,
    4923, 4923, 4923, 4923, 4923, 4927, 4944, 4948, 4948, 4948, 4948, 4950, 4953, 4954, 4954, 4956,
    4960, 4961, 4962, 4963, 4964, 4964, 4969, 4970, 4970, 4972, 4972, 4972, 4972, 4972, 4972, 4972,
    4974, 4974, 4977, 4983, 4985, 4994, 4994, 4994, 4995, 4995, 5000, 5003, 5005, 5039, 5039, 5039,
    5039, 5040, 5041, 5047, 5047, 5053, 5053, 5060, 5060, 5064, 5070, 5074, 5074, 5077, 5078, 5081,
    5081, 5082, 5085, 5088, 5089, 5089, 5089, 5105, 5107, 5108, 5111, 5127, 5129, 5137, 5138, 5138,
    5146, 5150, 5150, 5150, 5158, 5158, 5158, 5158, 5164, 5165, 5167, 5170, 5172, 5173, 5177, 5180,
    5184, 5184, 5184, 5184, 5184, 5188, 5189, 5191, 5193, 5193, 5202, 5202, 5202, 5202, 5204, 5209,
    5209, 5210, 5210, 5213, 5213, 5217, 5228, 5228, 5228, 5228, 5228, 5229, 5229, 5229, 5230, 5230,
    5230, 5231, 5234, 5238, 5238, 5242, 5247, 5250, 5250, 5253, 5254, 5254, 5257, 5260, 5260, 5262,
    5262, 5262, 5270, 5271, 5276, 5281, 5285, 5292, 5297, 5299, 5300, 5301, 5301, 5301, 5301, 5307,
    5308, 5309, 5309, 5314, 5322, 5324, 5328, 5329, 5329, 5330, 5331, 5331, 5331, 5333, 5334, 5334,
    5334, 5335, 5335, 5337, 5337, 5337, 5337, 5338, 5341, 5342, 5343, 5346, 5347, 5347, 5348, 5350,
    5350, 5352, 5356, 5357, 5357, 5358, 5358, 5361, 5361, 5361, 5362, 5376, 5376, 5382, 5383, 5384,
    5384, 5384, 5385, 5387, 5387, 5388, 5390, 5420, 5421, 5421, 5423, 5424, 5428, 5428, 5428, 5428,
    5428, 5429, 5436, 5436, 5436, 5440, 5440, 5447, 5447, 5448, 5448, 5449, 5451, 5451, 5451, 5454,
    5457, 5460, 5462, 5465, 5470, 5472, 5475, 5477, 5477, 5480, 5480, 5491, 5491, 5495, 5504, 5511,
    5511, 5515, 5515, 5516, 5517, 5517, 5517, 5517, 5517, 5517, 5520, 5521, 5529, 5529, 5529, 5530,
    5534, 5536, 5538, 5539, 5544, 5544, 5544, 5546, 5547, 5547, 5549, 5549, 5549, 5575, 5575, 5575,
    5576, 5576, 5579, 5581, 5585, 5585, 5592, 5597, 5602, 5602, 5602, 5603, 5603, 5603, 5603, 5603,
    5608, 5611, 5613, 5613, 5613, 5614, 5614, 5616, 5616, 5630, 5630, 5631, 5631, 5631, 5631, 5635,
    5635, 5637, 5637, 5637, 5639, 5639, 5642, 5642, 5642, 5642, 5642, 5643, 5647, 5647, 5647, 5647,
    5666, 5668, 5668, 5675, 5675, 5676, 5676, 5679, 5679, 5682, 5685, 5685, 5686, 5687, 5687, 5687,
    5688, 5688, 5698, 5699, 5700, 5701, 5701, 5713, 5715, 5833, 5833, 5833, 5836, 5838, 5843, 5845,
    5845, 5846, 5847, 5850, 5853, 5865, 5865, 5865, 5866, 5866, 5867, 5868, 5868, 5869, 5875, 5893,
    5893, 5896, 5897, 5897, 5905, 5906, 5908, 5909, 5910, 5914, 5914, 5918, 5920, 5921, 5921, 5922,
    5922, 5922, 5923, 5923, 5923, 5942, 5942, 5942, 5942, 5943, 5947, 5949, 5962, 5963, 5964, 5964,
    5969, 5975, 5975, 5981, 5983, 5986, 5987, 5987, 5987, 5987, 5988, 5990, 5990, 5991, 6003, 6003,
    6003, 6006, 6011, 6013, 6013, 6030, 6040, 6040, 6043, 6044, 6045, 6051, 6051, 6052, 6053, 6059,
    6060, 6060, 6063, 6063, 6070, 6071, 6072, 6083, 6083, 6084, 6107, 6108, 6114, 6114, 6115, 6115,
    6116, 6120, 6124, 6132, 6132, 6132, 6135, 6135, 6137, 6142, 6142, 6144, 6144, 6148, 6150, 6150,
    6150, 6150, 6150, 6151, 6155, 6157, 6159, 6160, 6163, 6168, 6170, 6170, 6178, 6180, 6180, 6181,
    6183, 6185, 6185, 6187, 6194, 6195, 6196, 6196, 6197, 6205, 6209, 6210, 6211, 6211, 6211, 6211,
    6215, 6220, 6220, 6227, 6227, 6227, 6228, 6228, 6231, 6239, 6241, 6241, 6242, 6249, 6254, 6264,
    6267, 6269, 6269, 6272, 6284, 6285, 6287, 6288, 6289, 6290, 6291, 6291, 6291, 6291, 6291, 6291,
    6291, 6291, 6292, 6293, 6298, 6301, 6302, 6303, 6303, 6303, 6303, 6306, 6308, 6313, 6319, 6321,
    6321, 6325, 6330, 6337, 6340, 6340, 6343, 6353, 6357, 6357, 6362, 6362, 6363, 6366, 6369, 6370,
    6370, 6373, 6373, 6374, 6374, 6377, 6377, 6378, 6382, 6382, 6382, 6382, 6383, 6383, 6388, 6389,
    6389, 6389, 6390, 6390, 6394, 6394, 6395, 6398, 6398, 6402, 6402, 6402, 6404, 6405, 6405, 6406,
    6407, 6409, 6409, 6412, 6415, 6415, 6415, 6416, 6416, 6417, 6419, 6421, 6421, 6421, 6425, 6425,
    6427, 6428, 6428, 6441, 6441, 6444, 6447, 6448, 6452, 6456, 6476, 6476, 6477, 6478, 6478, 6479,
    6485, 6485, 6486, 6486, 6488, 6488, 6489, 6492, 6492, 6492, 6493, 6495, 6495, 6495, 6500, 6502,
    6506, 6507, 6507, 6507, 6507, 6507, 6507, 6509, 6510, 6510, 6510, 6510, 6511, 6515, 6515, 6517,
    6521, 6521, 6522, 6522, 6523, 6525, 6525, 6525, 6526, 6533, 6541, 6541, 6543, 6543, 6548, 6548,
    6549, 6550, 6550, 6550, 6550, 6551, 6551, 6555, 6556, 6556, 6556, 6566, 6566, 6574, 6575, 6577,
    6577, 6580, 6585, 6585, 6585, 6585, 6588, 6588, 6591, 6591, 6605, 6607, 6607, 6607, 6619, 6620,
    6620, 6620, 6621, 6621, 6624, 6627, 6637, 6637, 6637, 6671, 6671, 6671, 6671, 6672, 6673, 6674,
    6676, 6677, 6678, 6680, 6682, 6682, 6684, 6684, 6685, 6686, 6686, 6686, 6687, 6688, 6690, 6690,
    6691, 6691, 6691, 6691, 6691, 6691, 6695, 6697, 6697, 6699, 6699, 6699, 6699, 6699, 6699, 6699,
    6704, 6705, 6708, 6709, 6710, 6710, 6711, 6711, 6714, 6714, 6743, 6747, 6764, 6764, 6765, 6766,
    6766, 6768, 6768, 6770, 6770, 6771, 6777, 6777, 6777, 6777, 6777, 6779, 6779, 6782, 6782, 6782,
    6786, 6788, 6788, 6788, 6798, 6801, 6803, 6806, 6809, 6813, 6813, 6815, 6818, 6818, 6819, 6824,
    6832, 6835, 6835, 6835, 6835, 6845, 6847, 6847, 6847, 6851, 6859, 6860, 6861, 6861, 6862, 6863,
    6864, 6864, 6864, 6866, 6869, 6869, 6869, 6877, 6878, 6878, 6879, 6895, 6904, 6920, 6920, 6920,
    6921, 6921, 6921, 6931, 6931, 6939, 6940, 6941, 6941, 6953, 6954, 6957, 6961, 6961, 6967, 6967,
    6967, 6968, 6982, 6985, 6985, 6986, 6988, 6988, 6988, 6988, 7006, 7009, 7011, 7011, 7012, 7012,
    7020, 7020, 7020, 7029, 7034, 7035, 7049, 7059, 7059, 7060, 7062, 7062, 7063, 7077, 7079, 7079,
    7083, 7088, 7091, 7091, 7097, 7100, 7114, 7114, 7114, 7115, 7121, 7121, 7122, 7123, 7123, 7127,
    7128, 7134, 7135, 7136, 7136, 7146, 7148, 7149, 7150, 7151, 7151, 7152, 7155, 7155, 7157, 7158,
    7163, 7163, 7165, 7169, 7169, 7169, 7172, 7172, 7174, 7175, 7177, 7179, 7180, 7182, 7183, 7184,
    7184, 7185, 7187, 7187, 7190, 7200, 7202, 7202, 7204, 7206, 7206, 7224, 7224, 7224, 7226, 7226,
    7228, 7229, 7231, 7231, 7231, 7232, 7240, 7240, 7243, 7246, 7252, 7253, 7253, 7253, 7254, 7259,
    7262, 7269, 7269, 7269, 7271, 7271, 7272, 7274, 7274, 7274, 7274, 7274, 7282, 7290, 7290, 7307,
    7321, 7321, 7323, 7323, 7323, 7323, 7326, 7326, 7334, 7334, 7334, 7334, 7336, 7337, 7337, 7338,
    7342, 7342, 7349, 7350, 7350, 7350, 7350, 7351, 7355, 7362, 7362, 7362, 7363, 7363, 7363, 7369,
    7369, 7369, 7369, 7370, 7371, 7371, 7373, 7383, 7384, 7385, 7385, 7389, 7389, 7396, 7396, 7401,
    7406, 7406, 7406, 7406, 7406, 7406, 7406, 7407, 7408, 7411, 7411, 7412, 7412, 7417, 7417, 7418,
    7418, 7424, 7424, 7424, 7424, 7433, 7443, 7451, 7451, 7453, 7462, 7462, 7463, 7465, 7465, 7465,
    7478, 7478, 7481, 7484, 7485, 7486, 7487, 7487, 7506, 7531, 7532, 7533, 7533, 7541, 7541, 7550,
    7551, 7553, 7553, 7558, 7558, 7558, 7558, 7558, 7567, 7567, 7568, 7584, 7584, 7585, 7585, 7586,
    7589, 7591, 7605, 7606, 7607, 7607, 7608, 7611, 7616, 7617, 7622, 7622, 7626, 7630, 7631, 7631,
    7631, 7632, 7633, 7634, 7637, 7640, 7641, 7645, 7647, 7647, 7649, 7649, 7670, 7673, 7675, 7677,
    7677, 7677, 7679, 7679, 7686, 7686, 7688, 7692, 7694, 7695, 7697, 7697, 7697, 7700, 7700, 7707,
    7710, 7730, 7731, 7732, 7732, 7732, 7733, 7737, 7745, 7753, 7753, 7761, 7761, 7762, 7762, 7762,
    7763, 7763, 7770, 7772, 7776, 7782, 7783, 7784, 7785, 7785, 7786, 7789, 7789, 7790, 7793, 7793,
    7797, 7798, 7798, 7802, 7805, 7810, 7810, 7811, 7811, 7813, 7814, 7814, 7814, 7814, 7814, 7815,
    7815, 7815, 7817, 7820, 7820, 7820, 7823, 7825, 7826, 7826, 7827, 7835, 7835, 7839, 7846, 7847,
    7847, 7855, 7856, 7856, 7858, 7858, 7859, 7862, 7863, 7863, 7867, 7868, 7875, 7875, 7875, 7875,
    7879, 7879, 7881, 7881, 7884, 7886, 7887, 7887, 7887, 7887, 7892, 7896, 7917, 7918, 7918, 7918,
    7919, 7930, 7939, 7943, 7944, 7944, 7949, 7950, 7951, 7951, 7951, 7951, 7951, 7964, 7965, 7976,
    7980, 7986, 7987, 7989, 7990, 7991, 7991, 7992, 7993, 8006, 8007, 8007, 8035, 8035, 8036, 8036,
    8036, 8038, 8044, 8050, 8051, 8053, 8054, 8056, 8058, 8071, 8071, 8075, 8075, 8078, 8078, 8078,
    8078, 8080, 8087, 8087, 8089, 8089, 8091, 8096, 8098, 8103, 8104, 8105, 8110, 8111, 8111, 8111,
    8117, 8120, 8120, 8122, 8122, 8123, 8125, 8129, 8129, 8132, 8137, 8137, 8140, 8140, 8146, 8147,
    8149, 8150, 8151, 8156, 8156, 8156, 8156, 8158, 8161, 8161, 8162, 8162, 8162, 8162, 8162, 8168,
    8170, 8180, 8180, 8185, 8185, 8185, 8186, 8187, 8187, 8190, 8190, 8190, 8207, 8208, 8208, 8208,
    8209, 8211, 8213, 8214, 8216, 8219, 8219, 8219, 8220, 8224, 8228, 8236, 8236, 8236, 8240, 8241,
    8242, 8242, 8251, 8254, 8254, 8255, 8255, 8256, 8256, 8259, 8262, 8262, 8270, 8270, 8270, 8270,
    8270, 8270, 8275, 8275, 8275, 8277, 8277, 8277, 8283, 8286, 8287, 8291, 8291, 8293, 8294, 8294,
    8300, 8300, 8302, 8302, 8302, 8305, 8305, 8309, 8309, 8314, 8315, 8317, 8317, 8320, 8322, 8322,
    8324, 8325, 8327, 8330, 8339, 8356, 8360, 8362, 8368, 8368, 8369, 8370, 8370, 8370, 8374, 8377,
    8377, 8377, 8377, 8377, 8381, 8390, 8390, 8391, 8393, 8393, 8399, 8410, 8411, 8417, 8417, 8419,
    8420, 8420, 8420, 8420, 8422, 8423, 8427, 8428, 8428, 8428, 8428, 8434, 8434, 8438, 8441, 8442,
    8442, 8446, 8451, 8461, 8462, 8462, 8463, 8471, 8471, 8471, 8473, 8477, 8477, 8478, 8481, 8485,
    8485, 8487, 8494, 8494, 8494, 8495, 8495, 8500, 8500, 8501, 8503, 8504, 8507, 8516, 8516, 8522,
    8523, 8524, 8524, 8524, 8524, 8524, 8524, 8525, 8526, 8530, 8536, 8536, 8536, 8547, 8547, 8548,
    8548, 8554, 8557, 8560, 8561, 8561, 8563, 8563, 8565, 8566, 8569, 8569, 8571, 8572, 8572, 8572,
    8572, 8573, 8582, 8585, 8586, 8586, 8592, 8593, 8607, 8608, 8611, 8616, 8617, 8618, 8628, 8628,
    8629, 8636, 8637, 8637, 8642, 8643, 8643, 8645, 8645, 8645, 8649, 8650, 8650, 8654, 8654, 8654,
    8660, 8660, 8660, 8665, 8665, 8667, 8668, 8669, 8671, 8671, 8671, 8675, 8675, 8677, 8677, 8679,
    8679, 8686, 8686, 8686, 8693, 8694, 8694, 8694, 8695, 8695, 8696, 8698, 8707, 8708, 8709, 8710,
    8710, 8715, 8718, 8719, 8722, 8722, 8722, 8724, 8724, 8724, 8724, 8727, 8727, 8728, 8729, 8729,
    8732, 8735, 8736, 8739, 8741, 8742, 8747, 8747, 8751, 8757, 8758, 8758, 8758, 8763, 8763, 8763,
    8763, 8765, 8767, 8767, 8768, 8768, 8769, 8769, 8769, 8773, 8779, 8780, 8793, 8793, 8793, 8795,
    8797, 8798, 8800, 8800, 8802, 8802, 8802, 8802, 8806, 8807, 8813, 8813, 8817, 8819, 8819, 8822,
    8824, 8824, 8828, 8828, 8829, 8832, 8832, 8837, 8838, 8838, 8839, 8842, 8842, 8842, 8842, 8845,
    8846, 8846, 8855, 8864, 8868, 8870, 8870, 8871, 8871, 8874, 8874, 8874, 8876, 8876, 8876, 8878,
    8882, 8882, 8886, 8889, 8889, 8889, 8890, 8890, 8899, 8903, 8906, 8906, 8906, 8918, 8918, 8919,
    8921, 8924, 8924, 8925, 8927, 8927, 8929, 8929, 8930, 8930, 8932, 8933, 8933, 8934, 8937, 8937,
    8937, 8939, 8942, 8942, 8943, 8945, 8945, 8946, 8946, 8949, 8950, 8951, 8967, 8968, 8969, 8972,
    8972, 8975, 8975, 8975, 8977, 8977, 8977, 8977, 8980, 8981, 8983, 8996, 8999, 9001, 9002, 9002,
    9003, 9006, 9021, 9022, 9022, 9025, 9025, 9036, 9041, 9041, 9043, 9044, 9044, 9045, 9058, 9059,
    9059, 9059, 9059, 9070, 9070, 9070, 9070, 9070, 9070, 9073, 9123, 9124, 9127, 9127, 9127, 9130,
    9144, 9152, 9152, 9155, 9159, 9160, 9160, 9162, 9163, 9163, 9173, 9174, 9176, 9178, 9181, 9182,
    9182, 9186, 9189, 9194, 9194, 9194, 9194, 9194, 9194, 9200, 9201, 9203, 9207, 9221, 9222, 9224,
    9227, 9230, 9230, 9242, 9250, 9252, 9254, 9259, 9260, 9260, 9263, 9263, 9263, 9263, 9266, 9266,
    9268, 9268, 9269, 9269, 9269, 9270, 9271, 9271, 9273, 9278, 9279, 9279, 9279, 9279, 9281, 9283,
    9283, 9288, 9288, 9297, 9299, 9299, 9299, 9300, 9305, 9307, 9309, 9309, 9309, 9309, 9309, 9309,
    9315, 9316, 9319, 9323, 9328, 9328, 9328, 9328, 9328, 9330, 9335, 9336, 9337, 9339, 9339, 9339,
    9339, 9339, 9340, 9343, 9343, 9343, 9343, 9344, 9345, 9349, 9356, 9362, 9373, 9377, 9385, 9385,
    9399, 9400, 9402, 9402, 9403, 9405, 9405, 9405, 9405, 9412, 9412, 9428, 9430, 9431, 9434, 9435,
    9440, 9443, 9446, 9448, 9451, 9452, 9452, 9456, 9457, 9457, 9457, 9462, 9462, 9462, 9463, 9465,
    9466, 9466, 9469, 9471, 9471, 9475, 9475, 9479, 9480, 9480, 9485, 9485, 9487, 9487, 9487, 9487,
    9487, 9487, 9490, 9490, 9490, 9492, 9493, 9494, 9496, 9498, 9498, 9498, 9498, 9498, 9498, 9498,
    9498, 9499, 9499, 9506, 9508, 9508, 9508, 9509, 9512, 9512, 9517, 9532, 9532, 9539, 9549, 9549,
    9549, 9550, 9550, 9553, 9556, 9557, 9559, 9560, 9560, 9562, 9562, 9562, 9567, 9568, 9572, 9572,
    9587, 9587, 9588, 9592, 9592, 9593, 9600, 9610, 9611, 9639, 9639, 9640, 9641, 9657, 9660, 9661,
    9665, 9674, 9674, 9674, 9674, 9674, 9674, 9674, 9678, 9681, 9686, 9688, 9697, 9698, 9698, 9699,
    9699, 9699, 9699, 9700, 9701, 9702, 9702, 9702, 9703, 9707, 9708, 9709, 9710, 9712, 9715, 9717,
    9717, 9717, 9718, 9718, 9719, 9724, 9725, 9725, 9729, 9730, 9731, 9735, 9735, 9741, 9742, 9743,
    9743, 9745, 9745, 9745, 9748, 9750, 9750, 9750, 9751, 9753, 9755, 9755, 9756, 9784, 9787, 9788,
    9789, 9789, 9797, 9797, 9799, 9806, 9808, 9808, 9809, 9810, 9825, 9825, 9827, 9827, 9827, 9827,
    9830, 9831, 9837, 9837, 9843, 9843, 9861, 9861, 9864, 9864, 9864, 9865, 9865, 9868, 9868, 9869,
    9869, 9869, 9869, 9883, 9883, 9893, 9894, 9896, 9906, 9909, 9910, 9910, 9915, 9915, 9915, 9915,
    9915, 9919, 9921, 9921, 9921, 9921, 9921, 9922, 9922, 9922, 9925, 9925, 9929, 9930, 9935, 9945,
    9967, 9969, 9969, 9970, 9971, 9988, 9991, 9995, 9996, 9997, 10002, 10002, 10002, 10002, 10003, 10003,
    10003, 10009, 10009, 10009, 10011, 10012, 10012, 10013, 10017, 10018, 10020, 10022, 10029, 10029, 10040, 10040,
    10050, 10053, 10055, 10056, 10058, 10059, 10064, 10064, 10066, 10066, 10067, 10073, 10074, 10079, 10090, 10091,
    10092, 10092, 10093, 10094, 10094, 10094, 10094, 10095, 10099, 10102, 10102, 10102, 10128, 10128, 10130, 10134,
    10136, 10136, 10143, 10146, 10164, 10165, 10166, 10166, 10167, 10168, 10183, 10186, 10186, 10186, 10186, 10187,
    10187, 10190, 10193, 10195, 10197, 10197, 10197, 10199, 10199, 10199, 10205, 10209, 10212, 10220, 10228, 10228,
    10233, 10236, 10236, 10236, 10245, 10245, 10253, 10262, 10264, 10274, 10274, 10274, 10276, 10278, 10287, 10287,
    10291, 10295, 10296, 10300, 10302, 10305, 10307, 10307, 10307, 10307, 10310, 10313, 10314, 10324, 10324, 10326,
    10333, 10333, 10334, 10336, 10339, 10339, 10339, 10350, 10350, 10355, 10357, 10358, 10358, 10359, 10363, 10363,
    10363, 10363, 10365, 10365, 10365, 10368, 10372, 10374, 10374, 10374, 10374, 10375, 10383, 10383, 10384, 10385,
    10385, 10385, 10385, 10385, 10385, 10385, 10385, 10385, 10387, 10387, 10390, 10390, 10390, 10390, 10392, 10392,
    10393, 10397, 10399, 10399, 10401, 10401, 10402, 10418, 10418, 10432, 10432, 10432, 10435, 10440, 10442, 10443,
    10448, 10449, 10449, 10449, 10449, 10450, 10450, 10450, 10455, 10455, 10457, 10461, 10462, 10464, 10466, 10468,
    10468, 10474, 10474, 10476, 10477, 10479, 10479, 10486, 10486, 10487, 10487, 10487, 10487, 10487, 10494, 10496,
    10496, 10498, 10500, 10500, 10502, 10502, 10502, 10505, 10506, 10512, 10514, 10518, 10519, 10519, 10521, 10526,
    10526, 10526, 10534, 10535, 10539, 10541, 10545, 10547, 10548, 10551, 10551, 10558, 10560, 10563, 10565, 10567,
    10570, 10570, 10574, 10575, 10576, 10576, 10576, 10576, 10577, 10580, 10580, 10581, 10587, 10589, 10589, 10592,
    10594, 10594, 10596, 10613, 10616, 10616, 10620, 10620, 10623, 10624, 10624, 10626, 10626, 10626, 10627, 10628,
    10628, 10629, 10630, 10630, 10631, 10649, 10650, 10651, 10651, 10651, 10656, 10656, 10660, 10661, 10664, 10666,
    10666, 10667, 10670, 10670, 10671, 10673, 10680, 10681, 10693, 10696, 10712, 10713, 10715, 10723, 10723, 10725,
    10726, 10728, 10729, 10733, 10734, 10737, 10738, 10745, 10746, 10746, 10760, 10765, 10765, 10770, 10770, 10770,
    10771, 10776, 10776, 10778, 10778, 10779, 10779, 10781, 10790, 10790, 10790, 10790, 10790, 10798, 10799, 10799,
    10799, 10816, 10817, 10818, 10819, 10830, 10830, 10831, 10831, 10832, 10835, 10835, 10842, 10842, 10842, 10846,
    10846, 10850, 10850, 10851, 10859, 10859, 10859, 10859, 10868, 10871, 10872, 10875, 10875, 10891, 10894, 10894,
    10895, 10895, 10895, 10896, 10896, 10896, 10896, 10900, 10900, 10900, 10900, 10901, 10901, 10914, 10923, 10923,
    10923, 10924, 10924, 10924, 10924, 10924, 10927, 10927, 10932, 10932, 10936, 10937, 10939, 10939, 10945, 10948,
    10948, 10952, 10955, 10957, 10958, 10962, 10969, 10969, 10969, 10970, 10997, 10998, 11005, 11005, 11007, 11008,
    11008, 11008, 11008, 11010, 11016, 11017, 11017, 11030, 11030, 11031, 11032, 11041, 11045, 11046, 11046, 11046,
    11050, 11050, 11050, 11050, 11050, 11050, 11051, 11054, 11054, 11058, 11063, 11063, 11063, 11064, 11064, 11064,
    11064, 11064, 11064, 11066, 11068, 11070, 11071, 11071, 11074, 11074, 11075, 11075, 11077, 11077, 11078, 11079,
    11079, 11079, 11079, 11080, 11080, 11080, 11081, 11082, 11084, 11087, 11090, 11091, 11092, 11093, 11101, 11103,
    11110, 11115, 11122, 11123, 11123, 11124, 11128, 11128, 11129, 11147, 11148, 11148, 11149, 11154, 11154, 11155,
    11158, 11169, 11170, 11176, 11192, 11193, 11200, 11201, 11201, 11204, 11204, 11215, 11221, 11221, 11221, 11235,
    11235, 11239, 11240, 11242, 11242, 11242, 11242, 11250, 11269, 11269, 11269, 11272, 11272, 11272, 11272, 11278,
    11280, 11286, 11334, 11339, 11339, 11341, 11341, 11343, 11351, 11351, 11351, 11357, 11358, 11358, 11367, 11371,
    11373, 11383, 11392, 11394, 11406, 11409, 11412, 11416, 11416, 11418, 11418, 11419, 11420, 11422, 11430, 11430,
    11430, 11433, 11433, 11452, 11456, 11456, 11458, 11458, 11458, 11459, 11460, 11460, 11461, 11461, 11462, 11462,
    11472, 11475, 11475, 11483, 11486, 11487, 11487, 11487, 11498, 11498, 11501, 11503, 11503, 11503, 11508, 11508,
    11508, 11509, 11509, 11510, 11510, 11510, 11511, 11511, 11511, 11512, 11514, 11515, 11515, 11518, 11524, 11530,
    11532, 11532, 11532, 11532, 11535, 11535, 11537, 11537, 11538, 11539, 11539, 11540, 11546, 11546, 11547, 11547,
    11548, 11549, 11550, 11557, 11562, 11574, 11577, 11579, 11580, 11585, 11587, 11588, 11592, 11592, 11593, 11596,
    11598, 11598, 11599, 11600, 11604, 11604, 11604, 11604, 11604, 11604, 11604, 11605, 11609, 11614, 11615, 11615,
    11615, 11616, 11624, 11626, 11641, 11641, 11654, 11655, 11655, 11655, 11657, 11658, 11665, 11677, 11677, 11678,
    11678, 11678, 11678, 11678, 11680, 11684, 11686, 11688, 11690, 11697, 11697, 11698, 11711, 11714, 11714, 11718,
    11720, 11720, 11720, 11723, 11724, 11724, 11724, 11724, 11724, 11730, 11730, 11731, 11744, 11745, 11746, 11748,
    11749, 11750, 11750, 11753, 11759, 11759, 11761, 11761, 11761, 11767, 11768, 11768, 11769, 11771, 11771, 11772,
    11774, 11780, 11783, 11783, 11785, 11791, 11791, 11793, 11800, 11800, 11801, 11808, 11808, 11810, 11817, 11817,
    11817, 11817, 11829, 11829, 11829, 11829, 11831, 11833, 11833, 11833, 11833, 11833, 11838, 11838, 11838, 11845,
    11846, 11852, 11856, 11856, 11857, 11864, 11897, 11901, 11905, 11907, 11911, 11912, 11912, 11922, 11922, 11943,
    11944, 11965, 11966, 11966, 11966, 11966, 11969, 11971, 11971, 11977, 11979, 11979, 11980, 11980, 11980, 11980,
    11982, 11984, 11986, 11986, 11987, 11994, 11998, 12000, 12000, 12000, 12001, 12001, 12001, 12001, 12001, 12002,
    12002, 12003, 12003, 12004, 12005, 12008, 12008, 12011, 12012, 12012, 12012, 12014, 12015, 12015, 12015, 12015,
    12026, 12026, 12042, 12043, 12045, 12047, 12047, 12048, 12048, 12049, 12049, 12051, 12053, 12054, 12054, 12055,
    12055, 12055, 12056, 12061, 12061, 12061, 12074, 12074, 12074, 12076, 12085, 12086, 12088, 12088, 12088, 12088,
    12088, 12088, 12088, 12089, 12091, 12093, 12109, 12109, 12110, 12115, 12115, 12121, 12121, 12126, 12128, 12128,
    12132, 12143, 12151, 12151, 12151, 12151, 12151, 12151, 12151, 12151, 12151, 12153, 12153, 12171, 12171, 12188,
    12189, 12200, 12200, 12203, 12203, 12203, 12203, 12225, 12225, 12225, 12225, 12227, 12227, 12227, 12227, 12228,
    12228, 12229, 12242, 12243, 12243, 12246, 12249, 12249, 12250, 12253, 12253, 12254, 12259, 12259, 12259, 12260,
    12261, 12268, 12281, 12284, 12304, 12304, 12305, 12305, 12305, 12308, 12311, 12324, 12329, 12329, 12343, 12345,
    12347, 12347, 12349, 12349, 12349, 12351, 12351, 12354, 12355, 12360, 12360, 12360, 12363, 12364, 12364, 12365,
    12366, 12366, 12371, 12372, 12372, 12372, 12372, 12373, 12375, 12380, 12381, 12392, 12398, 12399, 12399, 12399,
    12399, 12402, 12403, 12405, 12429, 12429, 12430, 12435, 12435, 12435, 12437, 12437, 12446, 12446, 12449, 12452,
    12459, 12459, 12459, 12459, 12464, 12464, 12464, 12464, 12464, 12467, 12470, 12471, 12473, 12473, 12476, 12476,
    12476, 12476, 12499, 12501, 12501, 12501, 12501, 12503, 12504, 12517, 12518, 12519, 12526, 12526, 12526, 12526,
    12528, 12529, 12532, 12532, 12532, 12534, 12534, 12536, 12539, 12541, 12541, 12541, 12541, 12541, 12543, 12544,
    12544, 12545, 12545, 12547, 12549, 12549, 12549, 12550, 12552, 12557, 12560, 12561, 12565, 12565, 12574, 12577,
    12577, 12584, 12585, 12587, 12591, 12592, 12592, 12593, 12594, 12594, 12596, 12600, 12601, 12601, 12603, 12606,
    12607, 12610, 12611, 12612, 12617, 12617, 12617, 12618, 12618, 12618, 12618, 12623, 12629, 12635, 12638, 12647,
    12649, 12651, 12651, 12653, 12654, 12654, 12654, 12654, 12654, 12664, 12666, 12668, 12669, 12671, 12675, 12676,
    12677, 12680, 12680, 12682, 12690, 12690, 12692, 12693, 12693, 12694, 12694, 12696, 12696, 12696, 12706, 12728,
    12728, 12728, 12733, 12736, 12736, 12737, 12740, 12740, 12741, 12745, 12745, 12758, 12758, 12758, 12759, 12761,
    12761, 12761, 12761, 12761, 12763, 12765, 12765, 12767, 12768, 12768, 12769, 12770, 12770, 12770, 12770, 12773,
    12780, 12780, 12780, 12783, 12783, 12790, 12790, 12791, 12792, 12795, 12809, 12810, 12811, 12812, 12813, 12813,
    12816, 12819, 12822, 12824, 12824, 12825, 12825, 12826, 12826, 12827, 12828, 12830, 12830, 12830, 12830, 12830,
    12830, 12838, 12844, 12847, 12849, 12850, 12850, 12887, 12887, 12888, 12894, 12901, 12907, 12909, 12910, 12914,
    12914, 12914, 12915, 12918, 12918, 12918, 12918, 12919, 12919, 12920, 12922, 12981, 12983, 12983, 12986, 12989,
    12989, 12989, 12995, 12997, 12997, 12997, 12997, 13000, 13000, 13001, 13013, 13021, 13021, 13021, 13021, 13021,
    13021, 13022, 13024, 13024, 13024, 13026, 13026, 13029, 13031, 13038, 13038, 13041, 13041, 13041, 13041, 13048,
    13048, 13049, 13049, 13051, 13059, 13059, 13059, 13064, 13066, 13068, 13069, 13070, 13070, 13073, 13073, 13075,
    13080, 13089, 13093, 13093, 13093, 13093, 13093, 13093, 13093, 13109, 13125, 13125, 13130, 13131, 13131, 13133,
    13133, 13133, 13134, 13136, 13137, 13137, 13138, 13145, 13147, 13148, 13148, 13149, 13150, 13151, 13154, 13156,
    13156, 13156, 13156, 13156, 13171, 13171, 13173, 13173, 13173, 13173, 13176, 13179, 13186, 13186, 13187, 13187,
    13187, 13188, 13206, 13207, 13207, 13208, 13209, 13209, 13210, 13234, 13234, 13235, 13235, 13235, 13235, 13241,
    13242, 13242, 13242, 13256, 13257, 13267, 13267, 13271, 13271, 13272, 13272, 13272, 13273, 13274, 13274, 13274,
    13287, 13296, 13296, 13305, 13317, 13317, 13321, 13323, 13324, 13331, 13331, 13331, 13331, 13331, 13333, 13334,
    13337, 13339, 13341, 13341, 13341, 13341, 13344, 13344, 13349, 13350, 13350, 13350, 13350, 13352, 13352, 13352,
    13353, 13357, 13357, 13359, 13359, 13361, 13361, 13361, 13363, 13363, 13367, 13374, 13385, 13385, 13385, 13386,
    13386, 13386, 13387, 13388, 13388, 13388, 13399, 13404, 13404, 13404, 13404, 13404, 13410, 13410, 13411, 13415,
    13416, 13417, 13422, 13422, 13423, 13424, 13424, 13424, 13425, 13426, 13426, 13426, 13428, 13430, 13430, 13431,
    13439, 13439, 13439, 13439, 13447, 13450, 13450, 13451, 13454, 13463, 13465, 13467, 13472, 13489, 13490, 13492,
    13495, 13496, 13497, 13500, 13501, 13501, 13501, 13507, 13510, 13510, 13514, 13517, 13522, 13535, 13535, 13536,
    13536, 13537, 13538, 13538, 13545, 13549, 13549, 13549, 13549, 13553, 13554, 13554, 13556, 13556, 13556, 13556,
    13562, 13566, 13567, 13567, 13568, 13568, 13575, 13575, 13575, 13575, 13578, 13578, 13584, 13584, 13594, 13595,
    13602, 13604, 13604, 13607, 13607, 13608, 13609, 13610, 13614, 13614, 13620, 13621, 13621, 13621, 13623, 13625,
    13635, 13635, 13635, 13636, 13636, 13651, 13651, 13652, 13656, 13659, 13659, 13659, 13660, 13674, 13674, 13674,
    13679, 13680, 13693, 13694, 13709, 13709, 13710, 13710, 13710, 13712, 13717, 13717, 13719, 13720, 13720, 13731,
    13731, 13736, 13737, 13741, 13741, 13741, 13741, 13741, 13745, 13749, 13749, 13759, 13761, 13761, 13764, 13767,
    13767, 13770, 13770, 13773, 13777, 13777, 13778, 13780, 13781, 13782, 13783, 13788, 13790, 13792, 13794, 13802,
    13803, 13811, 13813, 13813, 13813, 13813, 13814, 13825, 13825, 13827, 13835, 13836, 13837, 13839, 13847, 13849,
    13856, 13856, 13856, 13865, 13865, 13866, 13870, 13870, 13877, 13877, 13878, 13885, 13896, 13898, 13899, 13902,
    13902, 13902, 13902, 13903, 13903, 13903, 13905, 13905, 13905, 13908, 13912, 13913, 13913, 13913, 13913, 13917,
    13919, 13925, 13926, 13926, 13932, 13932, 13933, 13934, 13936, 13936, 13936, 13939, 13945, 13963, 13964, 13964,
    13964, 13965, 13967, 13983, 13983, 13985, 13995, 14007, 14014, 14022, 14022, 14022, 14033, 14033, 14051, 14057,
    14058, 14058, 14060, 14069, 14069, 14069, 14069, 14071, 14073, 14074, 14074, 14074, 14074, 14076, 14081, 14121,
    14125, 14130, 14130, 14135, 14135, 14135, 14135, 14144, 14148, 14154, 14154, 14154, 14155, 14155, 14155, 14156,
    14156, 14157, 14160, 14160, 14161, 14163, 14164, 14167, 14168, 14187, 14188, 14189, 14189, 14196, 14201, 14201,
    14201, 14202, 14202, 14202, 14205, 14213, 14214, 14214, 14214, 14214, 14216, 14216, 14217, 14218, 14222, 14227,
    14231, 14231, 14234, 14234, 14235, 14235, 14239, 14240, 14240, 14264, 14266, 14267, 14267, 14270, 14272, 14273,
    14273, 14273, 14276, 14277, 14288, 14294, 14294, 14298, 14298, 14299, 14300, 14301, 14301, 14301, 14302, 14304,
    14307, 14307, 14308, 14321, 14322, 14322, 14322, 14322, 14325, 14327, 14327, 14327, 14328, 14328, 14328, 14330,
    14333, 14335, 14335, 14336, 14340, 14340, 14345, 14346, 14346, 14347, 14347, 14347, 14348, 14360, 14366, 14366,
    14370, 14378, 14382, 14382, 14383, 14395, 14396, 14396, 14396, 14396, 14398, 14406, 14406, 14406, 14409, 14411,
    14411, 14411, 14412, 14414, 14415, 14417, 14424, 14431, 14436, 14437, 14446, 14447, 14447, 14447, 14448, 14448,
    14450, 14451, 14451, 14455, 14455, 14455, 14456, 14456, 14457, 14467, 14480, 14490, 14491, 14495, 14499, 14500,
    14505, 14517, 14518, 14529, 14534, 14534, 14534, 14534, 14534, 14534, 14539, 14539, 14542, 14545, 14547, 14547,
    14548, 14551, 14551, 14551, 14558, 14558, 14561, 14561, 14562, 14562, 14565, 14566, 14575, 14586, 14586, 14588,
    14588, 14588, 14590, 14592, 14597, 14597, 14597, 14598, 14599, 14603, 14604, 14615, 14641, 14645, 14645, 14650,
    14652, 14654, 14668, 14668, 14669, 14669, 14669, 14670, 14670, 14672, 14672, 14673, 14677, 14678, 14678, 14682,
    14685, 14685, 14688, 14690, 14691, 14696, 14697, 14698, 14702, 14703, 14703, 14703, 14703, 14703, 14705, 14707,
    14707, 14710, 14722, 14723, 14729, 14730, 14730, 14730, 14730, 14730, 14730, 14731, 14742, 14743, 14743, 14743,
    14749, 14750, 14752, 14752, 14752, 14754, 14754, 14758, 14760, 14763, 14763, 14764, 14766, 14766, 14766, 14770,
    14770, 14770, 14770, 14770, 14778, 14784, 14785, 14790, 14791, 14791, 14791, 14792, 14792, 14794, 14795, 14800,
    14801, 14802, 14802, 14802, 14808, 14808, 14808, 14809, 14810, 14813, 14814, 14818, 14818, 14819, 14821, 14823,
    14824, 14829, 14829, 14836, 14836, 14836, 14836, 14836, 14837, 14840, 14840, 14843, 14844, 14844, 14853, 14853,
    14853, 14854, 14856, 14856, 14856, 14858, 14860, 14873, 14874, 14874, 14874, 14876, 14876, 14890, 14892, 14892,
    14894, 14895, 14902, 14903, 14903, 14904, 14906, 14906, 14907, 14909, 14909, 14909, 14909, 14910, 14912, 14913,
    14917, 14918, 14919, 14931, 14932, 14934, 14937, 14951, 14955, 14955, 14955, 14960, 14960, 14964, 14964, 14964,
    14965, 14965, 14966, 14969, 14971, 14971, 15019, 15019, 15020, 15021, 15034, 15035, 15037, 15039, 15048, 15048,
    15048, 15048, 15050, 15053, 15054, 15054, 15054, 15055, 15056, 15059, 15061, 15063, 15063, 15066, 15069, 15070,
    15071, 15073, 15073, 15073, 15082, 15086, 15089, 15094, 15094, 15095, 15095, 15100, 15100, 15103, 15103, 15113,
    15124, 15129, 15131, 15132, 15135, 15135, 15136, 15138, 15140, 15141, 15141, 15141, 15142, 15142, 15143, 15143,
    15143, 15144, 15144, 15147, 15147, 15150, 15150, 15153, 15153, 15159, 15162, 15170, 15172, 15186, 15186, 15186,
    15188, 15193, 15195, 15196, 15197, 15198, 15199, 15199, 15200, 15200, 15201, 15201, 15203, 15203, 15203, 15203,
    15203, 15205, 15206, 15207, 15208, 15208, 15212, 15214, 15214, 15214, 15214, 15214, 15216, 15217, 15226, 15227,
    15227, 15227, 15228, 15228, 15230, 15230, 15232, 15235, 15238, 15238, 15239, 15242, 15247, 15247, 15247, 15247,
    15250, 15250, 15252, 15253, 15255, 15258, 15259, 15260, 15261, 15261, 15263, 15266, 15276, 15278, 15278, 15278,
    15278, 15278, 15278, 15281, 15283, 15288, 15288, 15288, 15288, 15288, 15288, 15288, 15289, 15289, 15289, 15289,
    15296, 15296, 15296, 15303, 15311, 15314, 15314, 15315, 15317, 15324, 15326, 15331, 15338, 15338, 15342, 15342,
    15345, 15345, 15345, 15355, 15355, 15355, 15356, 15356, 15356, 15358, 15359, 15363, 15363, 15363, 15364, 15364,
    15364, 15364, 15364, 15367, 15367, 15368, 15368, 15368, 15371, 15371, 15371, 15379, 15380, 15381, 15383, 15386,
    15390, 15390, 15390, 15391, 15391, 15391, 15393, 15394, 15396, 15396, 15401, 15402, 15406, 15406, 15408, 15410,
    15415, 15416, 15417, 15417, 15425, 15425, 15425, 15426, 15429, 15432, 15432, 15434, 15435, 15436, 15436, 15440,
    15441, 15446, 15446, 15446, 15448, 15474, 15475, 15475, 15475, 15476, 15478, 15478, 15479, 15479, 15481, 15484,
    15505, 15511, 15511, 15512, 15513, 15514, 15514, 15521, 15522, 15522, 15522, 15526, 15531, 15534, 15537, 15542,
    15547, 15547, 15555, 15557, 15564, 15575, 15583, 15584, 15585, 15585, 15585, 15585, 15586, 15589, 15589, 15590,
    15590, 15593, 15593, 15595, 15606, 15609, 15609, 15609, 15609, 15613, 15614, 15615, 15617, 15617, 15620, 15620,
    15643, 15647, 15655, 15655, 15657, 15665, 15667, 15679, 15679, 15679, 15679, 15683, 15683, 15685, 15693, 15695,
    15698, 15698, 15698, 15703, 15706, 15706, 15706, 15709, 15712, 15712, 15715, 15716, 15723, 15727, 15731, 15731,
    15739, 15739, 15739, 15739, 15739, 15740, 15740, 15749, 15750, 15756, 15756, 15757, 15758, 15763, 15765, 15768,
    15773, 15773, 15773, 15774, 15774, 15776, 15776, 15776, 15777, 15778, 15778, 15781, 15782, 15782, 15783, 15784,
    15788, 15789, 15789, 15796, 15796, 15806, 15811, 15813, 15813, 15819, 15824, 15827, 15828, 15833, 15833, 15837,
    15837, 15838, 15838, 15838, 15839, 15839, 15839, 15839, 15846, 15850, 15851, 15851, 15852, 15853, 15853, 15869,
    15892, 15898, 15902, 15906, 15911, 15917, 15917, 15917, 15917, 15919, 15925, 15925, 15925, 15926, 15926, 15926,
    15927, 15927, 15928, 15929, 15937, 15939, 15941, 15943, 15943, 15944, 15944, 15944, 15949, 15950, 15954, 15955,
    15955, 15957, 15959, 15959, 15965, 15967, 15975, 15975, 15980, 15982, 15983, 15983, 15984, 15984, 15984, 15984,
    15997, 15998, 16003, 16004, 16005, 16005, 16026, 16032, 16033, 16034, 16036, 16037, 16042, 16056, 16056, 16058,
    16058, 16058, 16061, 16065, 16065, 16074, 16077, 16079, 16080, 16082, 16086, 16091, 16093, 16093, 16100, 16102,
    16113, 16137, 16138, 16141, 16141, 16144, 16145, 16146, 16147, 16148, 16150, 16153, 16154, 16154, 16154, 16161,
    16162, 16162, 16162, 16174, 16175, 16175, 16197, 16198, 16198, 16199, 16199, 16200, 16200, 16201, 16201, 16202,
    16202, 16203, 16207, 16212, 16213, 16214, 16216, 16216, 16216, 16220, 16222, 16231, 16231, 16243, 16243, 16247,
    16247, 16252, 16253, 16260, 16261, 16267, 16267, 16267, 16268, 16277, 16278, 16279, 16279, 16282, 16282, 16285,
    16287, 16287, 16303, 16327, 16329, 16353, 16369, 16373, 16376, 16377, 16377, 16378, 16385, 16386, 16386, 16392,
    16392, 16393, 16393, 16393, 16393, 16398, 16398, 16399, 16399, 16399, 16406, 16406, 16407, 16410, 16410, 16411,
    16414, 16414, 16414, 16416, 16417, 16419, 16425, 16425, 16431, 16434, 16434, 16434, 16435, 16435, 16440, 16441,
    16442, 16446, 16450, 16451, 16453, 16457, 16457, 16457, 16457, 16459, 16465, 16465, 16466, 16471, 16475, 16483,
    16483, 16483, 16483, 16484, 16484, 16484, 16489, 16489, 16490, 16491, 16507, 16507, 16511, 16512, 16512, 16517,
    16517, 16519, 16523, 16525, 16525, 16525, 16529, 16530, 16530, 16530, 16532, 16532, 16539, 16543, 16543, 16544,
    16545, 16546, 16546, 16548, 16557, 16557, 16557, 16558, 16560, 16560, 16560, 16560, 16560, 16560, 16560, 16561,
    16583, 16584, 16587, 16589, 16595, 16606, 16619, 16621, 16621, 16624, 16626, 16630, 16630, 16630, 16631, 16634,
    16636, 16638, 16640, 16643, 16656, 16662, 16662, 16663, 16664, 16665, 16723, 16724, 16736, 16736, 16736, 16744,
    16747, 16765, 16765, 16765, 16765, 16766, 16767, 16767, 16767, 16769, 16774, 16780, 16795, 16796, 16808, 16821,
    16821, 16821, 16821, 16821, 16821, 16821, 16823, 16823, 16823, 16827, 16827, 16827, 16827, 16827, 16827, 16827,
    16830, 16830, 16831, 16834, 16834, 16834, 16835, 16835, 16836, 16844, 16845, 16851, 16851, 16852, 16852, 16852,
    16852, 16852, 16852, 16852, 16852, 16857, 16858, 16870, 16872, 16872, 16872, 16872, 16874, 16874, 16875, 16879,
    16879, 16881, 16881, 16881, 16893, 16893, 16894, 16895, 16896, 16902, 16908, 16909, 16909, 16914, 16919, 16919,
    16919, 16919, 16919, 16920, 16923, 16923, 16924, 16924, 16926, 16927, 16929, 16929, 16932, 16938, 16943, 16946,
    16947, 16949, 16949, 16949, 16949, 16949, 16949, 16949, 16950, 16950, 16960, 16961, 16963, 16964, 16964, 16964,
    16964, 16971, 16971, 16978, 16979, 16982, 16983, 16989, 16990, 16993, 16994, 16995, 16995, 16999, 17003, 17003,
    17003, 17006, 17007, 17008, 17009, 17019, 17019, 17022, 17027, 17027, 17027, 17028, 17029, 17030, 17032, 17039,
    17039, 17039, 17052, 17057, 17058, 17060, 17066, 17069, 17069, 17111, 17111, 17114, 17114, 17122, 17122, 17131,
    17143, 17153, 17153, 17157, 17161, 17161, 17169, 17170, 17171, 17173, 17173, 17173, 17173, 17175, 17176, 17176,
    17178, 17179, 17183, 17184, 17184, 17186, 17191, 17192, 17193, 17195, 17196, 17197, 17202, 17202, 17207, 17207,
    17209, 17209, 17209, 17218, 17219, 17219, 17221, 17221, 17225, 17225, 17229, 17229, 17235, 17235, 17235, 17235,
    17238, 17238, 17239, 17240, 17244, 17244, 17244, 17244, 17244, 17246, 17247, 17248, 17250, 17250, 17250, 17252,
    17252, 17256, 17256, 17256, 17257, 17258, 17259, 17259, 17260, 17262, 17263, 17263, 17263, 17263, 17263, 17263,
    17267, 17281, 17285, 17287, 17288, 17289, 17289, 17289, 17289, 17289, 17291, 17292, 17295, 17297, 17297, 17297,
    17301, 17301, 17306, 17307, 17311, 17311, 17311, 17313, 17316, 17316, 17323, 17324, 17324, 17325, 17325, 17325,
    17325, 17325, 17326, 17332, 17333, 17333, 17333, 17339, 17339, 17345, 17345, 17347, 17349, 17350, 17352, 17355,
    17356, 17356, 17358, 17358, 17359, 17359, 17364, 17369, 17369, 17372, 17372, 17374, 17375, 17377, 17384, 17385,
    17385, 17386, 17389, 17392, 17392, 17393, 17394, 17394, 17394, 17399, 17400, 17400, 17400, 17401, 17402, 17403,
    17411, 17412, 17412, 17412, 17414, 17419, 17419, 17422, 17425, 17425, 17429, 17429, 17429, 17430, 17430, 17442,
    17442, 17446, 17447, 17451, 17460, 17460, 17461, 17464, 17464, 17466, 17466, 17467, 17469, 17471, 17472, 17472,
    17472, 17474, 17474, 17477, 17478, 17478, 17478, 17478, 17478, 17478, 17480, 17493, 17493, 17501, 17501, 17501,
    17502, 17504, 17504, 17504, 17508, 17513, 17515, 17516, 17518, 17519, 17519, 17523, 17523, 17540, 17543, 17543,
    17548, 17550, 17552, 17552, 17552, 17556, 17556, 17559, 17559, 17559, 17563, 17575, 17576, 17576, 17577, 17578,
    17581, 17584, 17585, 17587, 17589, 17589, 17590, 17594, 17595, 17602, 17602, 17604, 17605, 17609, 17609, 17610,
    17610, 17610, 17611, 17615, 17615, 17615, 17617, 17622, 17622, 17622, 17622, 17623, 17636, 17636, 17636, 17637,
    17649, 17668, 17669, 17669, 17669, 17671, 17671, 17672, 17676, 17684, 17684, 17684, 17707, 17707, 17707, 17710,
    17724, 17724, 17724, 17724, 17738, 17738, 17741, 17741, 17741, 17742, 17742, 17742, 17742, 17743, 17743, 17744,
    17748, 17752, 17754, 17754, 17754, 17754, 17754, 17755, 17755, 17755, 17755, 17756, 17758, 17758, 17763, 17788,
    17792, 17792, 17792, 17820, 17820, 17824, 17824, 17838, 17838, 17839, 17862, 17862, 17862, 17863, 17863, 17863,
    17863, 17863, 17863, 17865, 17867, 17868, 17870, 17871, 17871, 17873, 17874, 17874, 17875, 17875, 17876, 17877,
    17878, 17878, 17878, 17903, 17904, 17910, 17913, 17919, 17924, 17925, 17925, 17926, 17928, 17931, 17932, 17935,
    17942, 17962, 17962, 17962, 17962, 17962, 17962, 17963, 17963, 17969, 17971, 17971, 17975, 17976, 17976, 17977,
    17994, 17996, 17996, 18001, 18002, 18002, 18002, 18005, 18009, 18017, 18020, 18020, 18047, 18047, 18048, 18050,
    18052, 18057, 18057, 18057, 18061, 18062, 18062, 18062, 18063, 18063, 18063, 18077, 18077, 18077, 18078, 18079,
    18084, 18086, 18086, 18087, 18087, 18087, 18093, 18101, 18109, 18109, 18111, 18114, 18114, 18120, 18120, 18120,
    18121, 18123, 18125, 18125, 18125, 18138, 18138, 18139, 18139, 18139, 18159, 18171, 18171, 18173, 18179, 18190,
    18190, 18190, 18191, 18191, 18196, 18199, 18204, 18211, 18213, 18213, 18213, 18213, 18221, 18221, 18222, 18225,
    18225, 18226, 18226, 18226, 18227, 18230, 18234, 18244, 18250, 18251, 18254, 18254, 18254, 18256, 18261, 18261,
    18261, 18261, 18263, 18264, 18264, 18265, 18277, 18289, 18289, 18295, 18295, 18297, 18299, 18299, 18300, 18301,
    18301, 18311, 18311, 18317, 18317, 18317, 18317, 18317, 18319, 18319, 18325, 18326, 18327, 18330, 18331, 18332,
    18332, 18332, 18332, 18334, 18335, 18345, 18348, 18349, 18349, 18350, 18352, 18352, 18353, 18353, 18355, 18363,
    18367, 18367, 18367, 18367, 18367, 18368, 18368, 18378, 18378, 18400, 18400, 18404, 18405, 18405, 18405, 18405,
    18407, 18407, 18408, 18409, 18410, 18411, 18412, 18413, 18419, 18420, 18426, 18426, 18430, 18434, 18435, 18435,
    18436, 18444, 18445, 18455, 18466, 18467, 18468, 18469, 18481, 18481, 18482, 18487, 18487, 18489, 18489, 18491,
    18496, 18496, 18514, 18514, 18515, 18517, 18523, 18525, 18525, 18526, 18527, 18527, 18554, 18554, 18554, 18575,
    18585, 18586, 18591, 18601, 18604, 18605, 18610, 18611, 18612, 18612, 18613, 18613, 18613, 18625, 18629, 18629,
    18629, 18632, 18633, 18633, 18634, 18634, 18634, 18634, 18644, 18645, 18645, 18646, 18646, 18647, 18649, 18651,
    18651, 18652, 18684, 18685, 18685, 18688, 18689, 18689, 18689, 18690, 18690, 18692, 18692, 18697, 18698, 18704,
    18707, 18707, 18721, 18735, 18735, 18735, 18736, 18744, 18745, 18749, 18771, 18778, 18780, 18780, 18780, 18780,
    18780, 18781, 18781, 18790, 18791, 18791, 18796, 18796, 18800, 18802, 18803, 18803, 18806, 18806, 18809, 18811,
    18811, 18811, 18811, 18811, 18814, 18814, 18814, 18816, 18817, 18832, 18834, 18834, 18834, 18837, 18837, 18840,
    18844, 18847, 18851, 18852, 18853, 18853, 18853, 18853, 18855, 18855, 18855, 18858, 18860, 18866, 18870, 18870,
    18871, 18873, 18874, 18874, 18876, 18876, 18877, 18877, 18877, 18878, 18878, 18904, 18913, 18914, 18915, 18915,
    18916, 18918, 18918, 18918, 18918, 18925, 18925, 18932, 18932, 18937, 18937, 18937, 18938, 18942, 18942, 18947,
    18947, 18949, 18954, 18957, 18957, 18957, 18957, 18962, 18962, 18962, 18962, 18962, 18965, 18966, 18969, 18970,
    18970, 18970, 18976, 18976, 18979, 18979, 18981, 18982, 18983, 18986, 18986, 18986, 18986, 19002, 19002, 19003,
    19013, 19014, 19016, 19018, 19018, 19018, 19023, 19023, 19023, 19024, 19036, 19036, 19036, 19036, 19036, 19038,
    19038, 19038, 19041, 19044, 19044, 19046, 19047, 19049, 19055, 19058, 19059, 19059, 19059, 19061, 19061, 19062,
    19062, 19062, 19063, 19063, 19063, 19063, 19065, 19066, 19069, 19070, 19078, 19080, 19080, 19082, 19082, 19083,
    19089, 19092, 19093, 19093, 19094, 19097, 19097, 19097, 19098, 19098, 19099, 19099, 19115, 19117, 19117, 19117,
    19117, 19118, 19124, 19126, 19126, 19128, 19128, 19128, 19129, 19129, 19141, 19141, 19142, 19142, 19149, 19150,
    19151, 19151, 19151, 19151, 19152, 19152, 19153, 19153, 19153, 19153, 19153, 19154, 19156, 19156, 19156, 19159,
    19162, 19165, 19165, 19165, 19166, 19166, 19166, 19168, 19169, 19173, 19173, 19181, 19182, 19185, 19188, 19188,
    19188, 19188, 19192, 19192, 19197, 19213, 19216, 19217, 19218, 19262, 19263, 19270, 19274, 19275, 19294, 19300,
    19301, 19302, 19302, 19302, 19308, 19308, 19308, 19308, 19308, 19308, 19308, 19309, 19309, 19309, 19310, 19310,
    19310, 19310, 19310, 19313, 19314, 19314, 19315, 19325, 19325, 19325, 19327, 19329, 19329, 19329, 19330, 19335,
    19335, 19335, 19338, 19339, 19340, 19340, 19341, 19344, 19346, 19348, 19350, 19350, 19351, 19351, 19351, 19351,
    19351, 19351, 19352, 19355, 19356, 19358, 19362, 19362, 19363, 19375, 19376, 19381, 19381, 19381, 19385, 19385,
    19386, 19388, 19388, 19390, 19390, 19390, 19393, 19394, 19394, 19399, 19399, 19399, 19402, 19410, 19410, 19419,
    19421, 19421, 19422, 19423, 19427, 19428, 19432, 19432, 19443, 19443, 19445, 19446, 19450, 19451, 19451, 19456,
    19456, 19456, 19458, 19459, 19459, 19459, 19461, 19462, 19462, 19465, 19465, 19473, 19473, 19474, 19474, 19475,
    19480, 19480, 19482, 19482, 19483, 19484, 19485, 19487, 19495, 19502, 19511, 19516, 19517, 19528, 19530, 19530,
    19530, 19530, 19531, 19531, 19531, 19531, 19534, 19535, 19539, 19540, 19542, 19546, 19547, 19547, 19547, 19547,
    19548, 19548, 19549, 19550, 19550, 19554, 19554, 19555, 19557, 19562, 19565, 19565, 19576, 19577, 19577, 19586,
    19588, 19589, 19589, 19599, 19599, 19599, 19600, 19601, 19603, 19603, 19604, 19606, 19607, 19620, 19630, 19634,
    19634, 19634, 19634, 19634, 19635, 19635, 19636, 19636, 19636, 19636, 19636, 19636, 19636, 19636, 19636, 19643,
    19666, 19672, 19672, 19672, 19673, 19674, 19674, 19689, 19689, 19689, 19695, 19704, 19704, 19711, 19723, 19723,
    19748, 19748, 19749, 19750, 19750, 19750, 19752, 19764, 19767, 19795, 19795, 19798, 19800, 19804, 19804, 19807,
    19808, 19808, 19808, 19809, 19810, 19823, 19823, 19823, 19823, 19826, 19826, 19827, 19829, 19829, 19829, 19835,
    19841, 19841, 19843, 19843, 19844, 19844, 19901, 19902, 19902, 19904, 19905, 19905, 19908, 19908, 19916, 19916,
    19916, 19916, 19918, 19921, 19921, 19921, 19930, 19930, 19942, 19943, 19943, 19943, 19943, 19946, 19949, 19956,
    19956, 19957, 19957, 19969, 19973, 19973, 19973, 19973, 19973, 19977, 19978, 19978, 19978, 19984, 19988, 19995,
    19996, 19996, 19996, 19996, 20000, 20002, 20002, 20007, 20013, 20013, 20013, 20014, 20018, 20024, 20025, 20029,
    20029, 20031, 20033, 20033, 20055, 20068, 20068, 20073, 20074, 20078, 20086, 20086, 20086, 20087, 20087, 20088,
    20088, 20088, 20088, 20088, 20089, 20094, 20100, 20100, 20101, 20102, 20108, 20108, 20108, 20108, 20110, 20110,
    20112, 20113, 20114, 20115, 20117, 20117, 20118, 20151, 20151, 20151, 20151, 20151, 20153, 20154, 20166, 20174,
    20174, 20174, 20174, 20175, 20176, 20176, 20176, 20178, 20181, 20182, 20190, 20192, 20194, 20194, 20194, 20200,
    20201, 20202, 20202, 20223, 20223, 20226, 20233, 20233, 20234, 20252, 20253, 20254, 20256, 20259, 20259, 20259,
    20259, 20259, 20260, 20262, 20265, 20265, 20265, 20265, 20271, 20276, 20279, 20279, 20282, 20285, 20288, 20288,
    20289, 20293, 20293, 20293, 20294, 20295, 20295, 20300, 20303, 20318, 20318, 20318, 20330, 20330, 20337, 20339,
    20346, 20348, 20353, 20356, 20356, 20362, 20363, 20363, 20364, 20364, 20364, 20365, 20383, 20383, 20384, 20387,
    20389, 20395, 20395, 20396, 20399, 20400, 20400, 20402, 20412, 20414, 20416, 20424, 20425, 20427, 20427, 20430,
    20441, 20442, 20443, 20451, 20451, 20456, 20458, 20459, 20459, 20463, 20463, 20463, 20465, 20465, 20465, 20466,
    20470, 20476, 20476, 20477, 20484, 20486, 20488, 20488, 20488, 20488, 20490, 20492, 20492, 20492, 20492, 20494,
    20498, 20510, 20510, 20510, 20511, 20519, 20519, 20520, 20522, 20523, 20523, 20523, 20531, 20531, 20532, 20533,
    20533, 20533, 20533, 20540, 20540, 20543, 20544, 20545, 20547, 20549, 20551, 20552, 20552, 20559, 20559, 20564,
    20568, 20568, 20569, 20571, 20571, 20571, 20572, 20583, 20584, 20585, 20591, 20591, 20591, 20594, 20596, 20598,
    20599, 20599, 20601, 20601, 20601, 20601, 20612, 20634, 20634, 20635, 20636, 20637, 20637, 20637, 20640, 20642,
    20642, 20653, 20656, 20656, 20666, 20666, 20670, 20672, 20672, 20678, 20678, 20678, 20678, 20683, 20686, 20688,
    20689, 20691, 20693, 20693, 20693, 20694, 20694, 20696, 20697, 20697, 20697, 20701, 20702, 20708, 20708, 20709,
    20709, 20709, 20709, 20709, 20709, 20710, 20710, 20711, 20715, 20715, 20715, 20716, 20718, 20720, 20720, 20721,
    20721, 20722, 20722, 20724, 20726, 20730, 20731, 20733, 20735, 20735, 20739, 20739, 20743, 20743, 20754, 20755,
    20755, 20755, 20757, 20766, 20766, 20768, 20768, 20769, 20769, 20806, 20816, 20818, 20822, 20823, 20823, 20826,
    20830, 20838, 20851, 20852, 20852, 20857, 20859, 20861, 20863, 20863, 20864, 20864, 20864, 20870, 20873, 20875,
    20878, 20880, 20890, 20891, 20894, 20894, 20900, 20902, 20905, 20907, 20913, 20913, 20913, 20913, 20913, 20916,
    20916, 20916, 20919, 20922, 20922, 20933, 20934, 20936, 20936, 20937, 20937, 20937, 20942, 20942, 20950, 20951,
    20952, 20952, 20952, 20955, 20956, 20956, 20962, 20963, 20967, 20968, 20972, 20977, 20979, 20979, 20979, 20981,
    20981, 20982, 20994, 20994, 20994, 20996, 21002, 21004, 21005, 21006, 21009, 21010, 21010, 21014, 21016, 21017,
    21018, 21019, 21019, 21019, 21020, 21021, 21021, 21021, 21021, 21021, 21021, 21021, 21022, 21022, 21022, 21022,
    21023, 21023, 21023, 21027, 21030, 21031, 21032, 21033, 21035, 21035, 21037, 21037, 21037, 21039, 21039, 21039,
    21054, 21055, 21055, 21055, 21056, 21056, 21061, 21062, 21062, 21068, 21068, 21070, 21071, 21075, 21075, 21079,
    21082, 21083, 21084, 21088, 21088, 21091, 21091, 21092, 21092, 21093, 21093, 21093, 21097, 21099, 21106, 21109,
    21128, 21136, 21137, 21141, 21141, 21142, 21142, 21143, 21143, 21148, 21150, 21155, 21167, 21169, 21169, 21169,
    21173, 21173, 21186, 21188, 21188, 21188, 21188, 21193, 21193, 21193, 21196, 21203, 21203, 21203, 21203, 21207,
    21207, 21210, 21210, 21212, 21216, 21217, 21224, 21224, 21224, 21280, 21282, 21282, 21293, 21293, 21294, 21296,
    21296, 21318, 21321, 21321, 21322, 21322, 21325, 21326, 21326, 21327, 21327, 21327, 21328, 21328, 21328, 21328,
    21332, 21332, 21340, 21341, 21343, 21344, 21348, 21353, 21353, 21362, 21363, 21369, 21369, 21378, 21378, 21382,
    21383, 21384, 21384, 21393, 21394, 21394, 21404, 21404, 21404, 21405, 21405, 21407, 21411, 21430, 21430, 21449,
    21449, 21454, 21458, 21458, 21459, 21459, 21459, 21459, 21459, 21459, 21462, 21463, 21463, 21466, 21466, 21480,
    21481, 21488, 21488, 21489, 21494, 21494, 21497, 21499, 21499, 21501, 21504, 21506, 21506, 21507, 21507, 21510,
    21510, 21514, 21514, 21515, 21516, 21523, 21523, 21525, 21525, 21525, 21527, 21529, 21531, 21531, 21532, 21534,
    21534, 21534, 21535, 21535, 21535, 21539, 21540, 21541, 21542, 21549, 21553, 21553, 21553, 21564, 21567, 21571,
    21571, 21571, 21572, 21575, 21575, 21579, 21583, 21583, 21583, 21583, 21585, 21592, 21593, 21596, 21598, 21603,
    21604, 21605, 21606, 21606, 21607, 21607, 21608, 21610, 21610, 21615, 21615, 21618, 21618, 21620, 21628, 21630,
    21633, 21645, 21645, 21645, 21659, 21659, 21660, 21660, 21660, 21661, 21670, 21671, 21671, 21674, 21676, 21679,
    21679, 21680, 21680, 21680, 21682, 21682, 21683, 21683, 21683, 21700, 21706, 21706, 21706, 21707, 21708, 21715,
    21721, 21721, 21722, 21722, 21722, 21722, 21728, 21729, 21730, 21731, 21734, 21755, 21755, 21770, 21771, 21778,
    21781, 21783, 21788, 21788, 21789, 21789, 21789, 21790, 21790, 21802, 21806, 21806, 21808, 21808, 21808, 21808,
    21810, 21810, 21810, 21811, 21811, 21811, 21814, 21815, 21818, 21824, 21825, 21826, 21827, 21827, 21829, 21833,
    21834, 21835, 21836, 21839, 21841, 21841, 21844, 21845, 21846, 21846, 21846, 21846, 21847, 21853, 21854, 21858,
    21858, 21861, 21867, 21867, 21867, 21868, 21872, 21875, 21875, 21876, 21876, 21878, 21878, 21879, 21880, 21882,
    21882, 21883, 21900, 21900, 21903, 21903, 21904, 21904, 21904, 21908, 21924, 21924, 21924, 21930, 21933, 21934,
    21934, 21937, 21942, 21942, 21942, 21942, 21942, 21944, 21944, 21944, 21946, 21946, 21946, 21947, 21947, 21950,
    21963, 21964, 21971, 21978, 21986, 21988, 21991, 21995, 22001, 22001, 22001, 22002, 22002, 22003, 22004, 22004,
    22004, 22004, 22004, 22006, 22006, 22006, 22006, 22006, 22007, 22007, 22007, 22007, 22010, 22016, 22016, 22020,
    22028, 22030, 22030, 22032, 22041, 22043, 22044, 22044, 22044, 22044, 22045, 22045, 22047, 22061, 22061, 22063,
    22069, 22070, 22073, 22076, 22076, 22077, 22082, 22083, 22083, 22086, 22088, 22101, 22103, 22109, 22111, 22121,
    22122, 22122, 22122, 22123, 22124, 22124, 22126, 22139, 22141, 22146, 22146, 22149, 22149, 22149, 22150, 22150,
    22151, 22151, 22163, 22164, 22164, 22164, 22166, 22167, 22173, 22214, 22215, 22229, 22231, 22239, 22239, 22242,
    22242, 22244, 22244, 22245, 22247, 22247, 22247, 22253, 22255, 22265, 22266, 22266, 22269, 22274, 22277, 22281,
    22287, 22291, 22292, 22295, 22296, 22296, 22296, 22296, 22296, 22297, 22297, 22307, 22309, 22309, 22312, 22314,
    22315, 22315, 22316, 22316, 22319, 22320, 22322, 22322, 22322, 22322, 22330, 22332, 22332, 22332, 22338, 22341,
    22351, 22356, 22356, 22356, 22375, 22375, 22375, 22375, 22377, 22377, 22381, 22381, 22381, 22382, 22383, 22383,
    22383, 22383, 22390, 22391, 22391, 22391, 22394, 22394, 22396, 22397, 22397, 22397, 22397, 22399, 22399, 22401,
    22401, 22402, 22402, 22402, 22402, 22402, 22407, 22421, 22426, 22427, 22428, 22429, 22433, 22433, 22433, 22434,
    22434, 22439, 22439, 22439, 22440, 22440, 22440, 22442, 22443, 22449, 22453, 22455, 22460, 22462, 22466, 22466,
    22466, 22472, 22473, 22473, 22473, 22473, 22479, 22483, 22484, 22517, 22518, 22518, 22518, 22520, 22522, 22522,
    22522, 22525, 22528, 22533, 22541, 22542, 22544, 22544, 22545, 22545, 22548, 22549, 22550, 22554, 22555, 22555,
    22556, 22557, 22559, 22560, 22562, 22563, 22564, 22567, 22569, 22569, 22575, 22576, 22583, 22583, 22583, 22583,
    22583, 22583, 22584, 22584, 22597, 22598, 22599, 22602, 22602, 22609, 22609, 22629, 22630, 22630, 22630, 22631,
    22631, 22633, 22633, 22649, 22651, 22651, 22658, 22658, 22660, 22682, 22684, 22684, 22685, 22687, 22689, 22691,
    22691, 22691, 22692, 22692, 22694, 22697, 22698, 22698, 22698, 22705, 22705, 22707, 22708, 22715, 22716, 22718,
    22718, 22718, 22719, 22723, 22723, 22724, 22727, 22734, 22735, 22736, 22743, 22745, 22745, 22760, 22768, 22769,
    22779, 22782, 22789, 22790, 22791, 22796, 22797, 22798, 22799, 22799, 22801, 22802, 22803, 22804, 22808, 22809,
    22810, 22810, 22810, 22812, 22812, 22813, 22813, 22813, 22815, 22818, 22818, 22824, 22824, 22824, 22832, 22837,
    22837, 22837, 22839, 22839, 22839, 22843, 22845, 22845, 22851, 22854, 22854, 22858, 22859, 22860, 22861, 22861,
    22862, 22862, 22862, 22862, 22869, 22869, 22869, 22869, 22869, 22871, 22871, 22871, 22886, 22896, 22898, 22899,
    22899, 22903, 22903, 22905, 22908, 22912, 22912, 22912, 22913, 22913, 22924, 22926, 22926, 22926, 22932, 22934,
    22938, 22939, 22939, 22939, 22939, 22948, 22949, 22949, 22949, 22952, 22953, 22961, 22961, 22962, 22969, 22982,
    22983, 22985, 22985, 22985, 22985, 22985, 22986, 22987, 22988, 22988, 22991, 22996, 22997, 22997, 23002, 23005,
    23006, 23007, 23008, 23010, 23011, 23015, 23016, 23017, 23024, 23025, 23036, 23036, 23036, 23037, 23037, 23038,
    23038, 23039, 23051, 23051, 23051, 23051, 23056, 23056, 23063, 23066, 23067, 23068, 23069, 23070, 23072, 23089,
    23089, 23090, 23090, 23090, 23117, 23119, 23119, 23121, 23124, 23127, 23137, 23139, 23140, 23141, 23141, 23141,
    23144, 23147, 23148, 23153, 23153, 23160, 23163, 23163, 23173, 23173, 23174, 23176, 23177, 23177, 23179, 23180,
    23188, 23188, 23189, 23194, 23196, 23196, 23201, 23202, 23203, 23204, 23209, 23211, 23215, 23227, 23242, 23243,
    23245, 23247, 23248, 23252, 23269, 23269, 23275, 23275, 23276, 23278, 23281, 23282, 23282, 23293, 23294, 23298,
    23299, 23300, 23307, 23309, 23309, 23312, 23312, 23312, 23312, 23314, 23314, 23317, 23318, 23319, 23321, 23325,
    23325, 23325, 23325, 23325, 23331, 23332, 23332, 23333, 23335, 23335, 23335, 23337, 23354, 23354, 23354, 23354,
    23362, 23362, 23362, 23364, 23364, 23365, 23365, 23365, 23366, 23366, 23369, 23370, 23377, 23378, 23378, 23378,
    23379, 23379, 23379, 23380, 23387, 23388, 23388, 23388, 23388, 23389, 23390, 23390, 23393, 23393, 23393, 23394,
    23394, 23424, 23425, 23430, 23430, 23430, 23432, 23434, 23443, 23448, 23449, 23452, 23459, 23459, 23460, 23466,
    23466, 23466, 23467, 23467, 23469, 23475, 23475, 23476, 23476, 23476, 23476, 23478, 23478, 23478, 23478, 23482,
    23492, 23503, 23507, 23522, 23522, 23526, 23527, 23528, 23528, 23528, 23529, 23534, 23535, 23537, 23538, 23538,
    23540, 23541, 23543, 23543, 23544, 23545, 23545, 23547, 23559, 23573, 23574, 23578, 23578, 23578, 23579, 23583,
    23583, 23585, 23585, 23585, 23587, 23588, 23588, 23591, 23593, 23593, 23593, 23608, 23609, 23613, 23613, 23613,
    23618, 23618, 23618, 23618, 23620, 23621, 23622, 23624, 23625, 23633, 23649, 23650, 23654, 23654, 23656, 23658,
    23665, 23665, 23669, 23669, 23676, 23676, 23682, 23683, 23686, 23687, 23691, 23693, 23694, 23700, 23700, 23700,
    23703, 23705, 23705, 23705, 23705, 23705, 23711, 23713, 23715, 23715, 23715, 23716, 23717, 23718, 23725, 23726,
    23729, 23729, 23729, 23729, 23730, 23732, 23732, 23733, 23733, 23733, 23733, 23733, 23746, 23746, 23746, 23746,
    23747, 23747, 23749, 23750, 23750, 23751, 23751, 23751, 23752, 23752, 23757, 23758, 23760, 23761, 23761, 23761,
    23762, 23765, 23766, 23778, 23782, 23784, 23790, 23791, 23794, 23794, 23794, 23795, 23800, 23800, 23800, 23800,
    23812, 23815, 23818, 23818, 23819, 23833, 23850, 23850, 23850, 23851, 23851, 23852, 23852, 23855, 23865, 23865,
    23866, 23867, 23867, 23869, 23869, 23872, 23873, 23873, 23874, 23875, 23879, 23881, 23882, 23882, 23882, 23899,
    23899, 23903, 23903, 23903, 23903, 23903, 23904, 23910, 23910, 23914, 23914, 23918, 23920, 23920, 23920, 23920,
    23920, 23923, 23923, 23923, 23923, 23929, 23930, 23930, 23930, 23935, 23951, 23953, 23954, 23959, 23960, 23961,
    23965, 23967, 23967, 23968, 23968, 23968, 23970, 23970, 23970, 23990, 23994, 23994, 24001, 24001, 24002, 24002,
    24004, 24004, 24004, 24005, 24006, 24006, 24010, 24011, 24014, 24014, 24022, 24022, 24024, 24027, 24027, 24032,
    24036, 24036, 24036, 24039, 24045, 24045, 24045, 24052, 24055, 24055, 24055, 24059, 24063, 24065, 24065, 24066,
    24067, 24068, 24070, 24077, 24078, 24078, 24080, 24080, 24081, 24082, 24082, 24083, 24083, 24083, 24087, 24090,
    24092, 24093, 24093, 24093, 24100, 24100, 24100, 24100, 24101, 24102, 24104, 24107, 24107, 24115, 24115, 24117,
    24126, 24131, 24131, 24141, 24143, 24144, 24147, 24148, 24154, 24157, 24162, 24163, 24163, 24163, 24165, 24167,
    24167, 24167, 24167, 24167, 24168, 24168, 24175, 24184, 24184, 24185, 24185, 24186, 24186, 24187, 24187, 24190,
    24190, 24191, 24193, 24201, 24203, 24203, 24216, 24230, 24231, 24242, 24249, 24251, 24251, 24251, 24252, 24252,
    24253, 24256, 24258, 24262, 24262, 24263, 24270, 24270, 24271, 24273, 24274, 24275, 24275, 24278, 24278, 24286,
    24289, 24291, 24291, 24303, 24303, 24303, 24303, 24304, 24304, 24305, 24307, 24307, 24308, 24308, 24315, 24318,
    24318, 24320, 24324, 24325, 24326, 24326, 24326, 24332, 24333, 24333, 24334, 24363, 24396, 24396, 24398, 24398,
    24399, 24399, 24406, 24406, 24406, 24406, 24406, 24421, 24421, 24422, 24423, 24423, 24425, 24425, 24428, 24429,
    24429, 24429, 24429, 24432, 24432, 24433, 24434, 24434, 24435, 24437, 24444, 24445, 24446, 24446, 24447, 24448,
    24448, 24448, 24448, 24456, 24456, 24457, 24459, 24459, 24464, 24465, 24481, 24486, 24486, 24488, 24488, 24491,
    24491, 24491, 24491, 24491, 24495, 24496, 24496, 24496, 24496, 24497, 24497, 24505, 24505, 24527, 24527, 24530,
    24535, 24535, 24535, 24539, 24539, 24548, 24548, 24558, 24559, 24573, 24575, 24576, 24581, 24581, 24588, 24589,
    24590, 24593, 24594, 24597, 24597, 24597, 24601, 24606, 24607, 24607, 24607, 24607, 24607, 24608, 24620, 24620,
    24620, 24624, 24626, 24626, 24628, 24628, 24628, 24628, 24628, 24629, 24629, 24629, 24629, 24630, 24630, 24633,
    24654, 24657, 24668, 24670, 24673, 24676, 24679, 24679, 24680, 24682, 24685, 24685, 24685, 24695, 24695, 24699,
    24699, 24699, 24700, 24700, 24706, 24706, 24706, 24722, 24724, 24724, 24724, 24724, 24733, 24733, 24734, 24734,
    24734, 24740, 24742, 24742, 24744, 24746, 24747, 24748, 24750, 24750, 24761, 24765, 24766, 24766, 24779, 24782,
    24789, 24795, 24796, 24796, 24797, 24797, 24798, 24800, 24800, 24803, 24803, 24806, 24806, 24807, 24809, 24810,
    24810, 24811, 24811, 24811, 24811, 24811, 24816, 24819, 24822, 24825, 24839, 24840, 24840, 24843, 24847, 24847,
    24848, 24849, 24849, 24849, 24849, 24849, 24850, 24850, 24850, 24856, 24856, 24856, 24857, 24860, 24862, 24862,
    24867, 24867, 24867, 24867, 24868, 24872, 24872, 24875, 24876, 24876, 24877, 24877, 24877, 24877, 24880, 24896,
    24896, 24896, 24901, 24902, 24902, 24909, 24909, 24910, 24910, 24911, 24919, 24919, 24920, 24920, 24922, 24922,
    24927, 24928, 24928, 24931, 24932, 24932, 24934, 24934, 24936, 24937, 24939, 24939, 24946, 24946, 24946, 24967,
    24976, 24976, 24977, 24977, 24981, 24981, 24981, 24981, 24982, 24989, 24993, 24994, 24994, 24998, 24998, 25001,
    25003, 25003, 25003, 25004, 25006, 25006, 25007, 25007, 25014, 25017, 25017, 25018, 25019, 25034, 25035, 25036,
    25036, 25039, 25043, 25044, 25047, 25047, 25047, 25048, 25048, 25048, 25050, 25053, 25054, 25054, 25054, 25056,
    25056, 25058, 25063, 25067, 25068, 25071, 25073, 25073, 25082, 25082, 25085, 25085, 25097, 25098, 25098, 25101,
    25109, 25109, 25117, 25117, 25121, 25128, 25128, 25129, 25129, 25188, 25200, 25202, 25203, 25203, 25206, 25208,
    25208, 25209, 25209, 25212, 25213, 25222, 25222, 25223, 25224, 25224, 25237, 25238, 25240, 25240, 25254, 25254,
    25254, 25254, 25255, 25255, 25339, 25341, 25343, 25346, 25348, 25350, 25352, 25352, 25352, 25352, 25354, 25358,
    25358, 25364, 25376, 25376, 25376, 25376, 25376, 25376, 25377, 25382, 25383, 25384, 25384, 25384, 25388, 25403,
    25404, 25405, 25411, 25411, 25412, 25431, 25440, 25442, 25449, 25449, 25455, 25456, 25456, 25458, 25460, 25461,
    25461, 25463, 25464, 25465, 25465, 25465, 25465, 25467, 25468, 25474, 25484, 25488, 25488, 25490, 25494, 25494,
    25495, 25497, 25499, 25505, 25505, 25505, 25505, 25507, 25507, 25522, 25524, 25532, 25535, 25537, 25538, 25538,
    25538, 25541, 25541, 25541, 25541, 25544, 25546, 25547, 25551, 25553, 25553, 25553, 25553, 25553, 25555, 25556,
    25557, 25557, 25557, 25561, 25561, 25567, 25567, 25567, 25571, 25571, 25572, 25576, 25576, 25590, 25590, 25591,
    25591, 25595, 25596, 25596, 25596, 25596, 25596, 25599, 25600, 25606, 25608, 25608, 25612, 25612, 25620, 25620,
    25620, 25625, 25628, 25645, 25647, 25655, 25655, 25656, 25661, 25662, 25662, 25663, 25663, 25663, 25666, 25666,
    25666, 25666, 25666, 25666, 25667, 25667, 25668, 25668, 25670, 25676, 25679, 25679, 25680, 25680, 25680, 25681,
    25681, 25681, 25687, 25692, 25693, 25693, 25694, 25695, 25695, 25696, 25696, 25696, 25696, 25696, 25696, 25697,
    25699, 25699, 25702, 25703, 25703, 25703, 25711, 25722, 25725, 25745, 25745, 25746, 25754, 25754, 25762, 25773,
    25773, 25777, 25778, 25790, 25799, 25801, 25802, 25803, 25806, 25806, 25806, 25810, 25836, 25848, 25858, 25867,
    25867, 25867, 25867, 25868, 25868, 25868, 25871, 25876, 25876, 25877, 25877, 25877, 25877, 25878, 25879, 25879,
    25879, 25880, 25880, 25882, 25884, 25888, 25889, 25890, 25892, 25892, 25894, 25903, 25906, 25906, 25908, 25924,
    25926, 25927, 25929, 25930, 25931, 25935, 25935, 25935, 25935, 25943, 25945, 25945, 25945, 25945, 25949, 25950,
    25954, 25954, 25954, 25955, 25955, 25955, 25955, 25957, 25958, 25958, 25959, 25969, 25970, 25971, 25976, 25976,
    25977, 25979, 25979, 25984, 25984, 25984, 25984, 25989, 25989, 25991, 25993, 26019, 26019, 26019, 26029, 26030,
    26030, 26031, 26034, 26034, 26034, 26035, 26035, 26038, 26039, 26039, 26047, 26051, 26061, 26061, 26071, 26072,
    26076, 26076, 26080, 26081, 26098, 26098, 26109, 26112, 26136, 26136, 26147, 26149, 26150, 26151, 26151, 26152,
    26154, 26155, 26155, 26160, 26165, 26165, 26168, 26171, 26171, 26173, 26173, 26176, 26178, 26178, 26178, 26182,
    26184, 26203, 26205, 26206, 26209, 26222, 26223, 26223, 26238, 26238, 26239, 26240, 26241, 26242, 26242, 26244,
    26244, 26247, 26247, 26249, 26249, 26249, 26252, 26255, 26256, 26256, 26259, 26260, 26260, 26260, 26261, 26267,
    26270, 26282, 26282, 26282, 26282, 26282, 26289, 26289, 26289, 26291, 26292, 26293, 26294, 26294, 26294, 26301,
    26301, 26302, 26304, 26304, 26306, 26308, 26324, 26326, 26326, 26326, 26326, 26331, 26331, 26338, 26343, 26344,
    26346, 26346, 26346, 26346, 26346, 26346, 26346, 26346, 26347, 26347, 26347, 26348, 26350, 26351, 26351, 26354,
    26364, 26365, 26365, 26366, 26366, 26369, 26369, 26370, 26370, 26373, 26373, 26379, 26384, 26394, 26394, 26401,
    26402, 26402, 26402, 26405, 26405, 26406, 26406, 26406, 26406, 26407, 26415, 26416, 26417, 26417, 26417, 26424,
    26439, 26440, 26442, 26444, 26446, 26447, 26447, 26448, 26453, 26454, 26455, 26460, 26460, 26462, 26462, 26471,
    26472, 26472, 26478, 26478, 26481, 26483, 26483, 26483, 26483, 26484, 26484, 26486, 26486, 26487, 26487, 26490,
    26494, 26505, 26505, 26505, 26505, 26509, 26509, 26512, 26512, 26512, 26515, 26517, 26517, 26517, 26521, 26540,
    26540, 26540, 26544, 26544, 26550, 26550, 26550, 26550, 26550, 26552, 26552, 26554, 26557, 26559, 26601, 26601,
    26601, 26602, 26605, 26610, 26611, 26611, 26611, 26612, 26616, 26621, 26623, 26624, 26624, 26624, 26624, 26626,
    26629, 26630, 26630, 26635, 26649, 26649, 26652, 26653, 26653, 26653, 26653, 26653, 26653, 26653, 26653, 26653,
    26653, 26655, 26656, 26656, 26659, 26659, 26659, 26659, 26661, 26661, 26666, 26667, 26669, 26674, 26676, 26677,
    26677, 26710, 26710, 26710, 26718, 26718, 26718, 26718, 26720, 26721, 26724, 26727, 26727, 26732, 26732, 26737,
    26742, 26743, 26743, 26758, 26761, 26761, 26765, 26765, 26765, 26766, 26766, 26769, 26770, 26770, 26782, 26783,
    26783, 26785, 26794, 26794, 26794, 26795, 26795, 26797, 26798, 26806, 26809, 26809, 26809, 26809, 26816, 26819,
    26821, 26822, 26822, 26823, 26826, 26827, 26827, 26829, 26829, 26831, 26831, 26831, 26921, 26921, 26921, 26922,
    26923, 26923, 26923, 26926, 26927, 26927, 26927, 26937, 26940, 26940, 26944, 26945, 26953, 26959, 26959, 26961,
    26961, 26968, 26969, 26976, 26976, 26976, 26976, 26976, 26979, 26981, 26985, 26985, 26990, 26990, 26990, 26991,
    26992, 26993, 26993, 26993, 26993, 27000, 27002, 27002, 27003, 27003, 27005, 27008, 27012, 27014, 27014, 27017,
    27022, 27023, 27025, 27026, 27027, 27028, 27029, 27029, 27031, 27033, 27038, 27039, 27039, 27047, 27051, 27052,
    27052, 27052, 27052, 27052, 27056, 27056, 27063, 27063, 27064, 27069, 27070, 27072, 27075, 27076, 27076, 27090,
    27090, 27091, 27095, 27096, 27097, 27097, 27097, 27097, 27098, 27102, 27102, 27106, 27110, 27112, 27113, 27116,
    27116, 27116, 27116, 27144, 27147, 27154, 27155, 27155, 27156, 27159, 27159, 27161, 27162, 27163, 27163, 27163,
    27169, 27169, 27173, 27175, 27176, 27177, 27187, 27187, 27187, 27208, 27213, 27213, 27214, 27214, 27215, 27215,
    27216, 27218, 27253, 27253, 27254, 27258, 27258, 27259, 27262, 27263, 27263, 27264, 27264, 27264, 27264, 27268,
    27268, 27270, 27270, 27270, 27270, 27272, 27272, 27286, 27286, 27290, 27292, 27295, 27295, 27295, 27304, 27307,
    27308, 27309, 27316, 27318, 27318, 27322, 27322, 27343, 27346, 27354, 27354, 27355, 27355, 27355, 27356, 27356,
    27360, 27362, 27362, 27371, 27372, 27375, 27377, 27378, 27379, 27379, 27383, 27388, 27388, 27389, 27397, 27404,
    27409, 27409, 27414, 27415, 27418, 27418, 27424, 27424, 27425, 27425, 27432, 27432, 27433, 27433, 27433, 27438,
    27438, 27442, 27443, 27444, 27445, 27446, 27447, 27452, 27456, 27458, 27460, 27461, 27461, 27465, 27465, 27465,
    27478, 27480, 27484, 27484, 27485, 27485, 27489, 27489, 27489, 27489, 27490, 27490, 27493, 27493, 27494, 27494,
    27495, 27495, 27499, 27501, 27504, 27505, 27505, 27505, 27505, 27505, 27509, 27511, 27511, 27511, 27511, 27515,
    27516, 27516, 27517, 27519, 27519, 27521, 27521, 27521, 27522, 27522, 27522, 27536, 27537, 27550, 27561, 27563,
    27564, 27564, 27567, 27569, 27570, 27591, 27592, 27605, 27606, 27612, 27615, 27616, 27622, 27624, 27625, 27626,
    27629, 27629, 27629, 27629, 27629, 27629, 27632, 27632, 27634, 27640, 27640, 27640, 27640, 27644, 27644, 27644,
    27645, 27649, 27649, 27651, 27652, 27653, 27655, 27655, 27655, 27655, 27655, 27656, 27657, 27660, 27662, 27663,
    27664, 27666, 27669, 27669, 27670, 27685, 27688, 27689, 27691, 27691, 27694, 27694, 27697, 27697, 27698, 27710,
    27711, 27712, 27715, 27716, 27717, 27717, 27718, 27718, 27730, 27730, 27730, 27743, 27753, 27758, 27758, 27762,
    27762, 27762, 27766, 27768, 27768, 27769, 27769, 27770, 27771, 27777, 27777, 27779, 27781, 27782, 27783, 27783,
    27783, 27783, 27783, 27793, 27794, 27799, 27800, 27800, 27807, 27808, 27818, 27821, 27824, 27825, 27827, 27827,
    27832, 27833, 27834, 27834, 27835, 27837, 27837, 27838, 27838, 27838, 27840, 27841, 27850, 27850, 27850, 27852,
    27852, 27852, 27861, 27861, 27862, 27865, 27866, 27866, 27867, 27867, 27870, 27870, 27871, 27872, 27873, 27875,
    27875, 27883, 27885, 27885, 27896, 27898, 27901, 27909, 27909, 27914, 27914, 27918, 27922, 27922, 27929, 27930,
    27932, 27932, 27933, 27939, 27941, 27941, 27941, 27941, 27941, 27943, 27943, 27943, 27943, 27944, 27944, 27946,
    27947, 27950, 27952, 27953, 27956, 27957, 27958, 27958, 27958, 27962, 27963, 27964, 27964, 27964, 27965, 27965,
    27967, 27970, 27970, 27982, 27984, 27986, 27990, 27990, 27990, 28014, 28018, 28024, 28024, 28026, 28026, 28029,
    28031, 28031, 28034, 28035, 28035, 28035, 28035, 28035, 28038, 28038, 28039, 28041, 28046, 28049, 28050, 28052,
    28053, 28053, 28053, 28060, 28061, 28063, 28079, 28079, 28079, 28079, 28084, 28084, 28084, 28093, 28093, 28094,
    28097, 28099, 28099, 28104, 28104, 28111, 28113, 28117, 28118, 28118, 28122, 28123, 28123, 28125, 28127, 28127,
    28127, 28136, 28136, 28139, 28139, 28143, 28143, 28147, 28147, 28149, 28150, 28151, 28151, 28154, 28154, 28154,
    28155, 28155, 28155, 28162, 28168, 28168, 28168, 28170, 28177, 28178, 28179, 28179, 28180, 28181, 28181, 28183,
    28185, 28185, 28186, 28191, 28192, 28194, 28202, 28202, 28209, 28209, 28210, 28210, 28210, 28210, 28218, 28221,
    28230, 28231, 28233, 28233, 28234, 28239, 28241, 28242, 28244, 28244, 28245, 28245, 28257, 28261, 28261, 28263,
    28264, 28264, 28264, 28264, 28265, 28266, 28269, 28276, 28278, 28278, 28292, 28293, 28293, 28293, 28293, 28296,
    28296, 28296, 28299, 28299, 28305, 28306, 28307, 28307, 28308, 28314, 28314, 28314, 28314, 28324, 28328, 28329,
    28329, 28331, 28336, 28347, 28348, 28353, 28354, 28374, 28375, 28378, 28405, 28405, 28409, 28409, 28409, 28411,
    28413, 28413, 28415, 28416, 28416, 28417, 28419, 28419, 28422, 28428, 28430, 28433, 28433, 28434, 28467, 28468,
    28469, 28469, 28469, 28474, 28474, 28474, 28475, 28475, 28475, 28476, 28480, 28480, 28484, 28484, 28484, 28485,
    28500, 28500, 28504, 28506, 28509, 28509, 28510, 28512, 28513, 28518, 28522, 28522, 28523, 28523, 28523, 28526,
    28528, 28528, 28528, 28528, 28529, 28531, 28533, 28536, 28537, 28540, 28541, 28541, 28542, 28545, 28547, 28547,
    28547, 28552, 28553, 28555, 28557, 28559, 28560, 28561, 28561, 28562, 28562, 28565, 28566, 28574, 28574, 28575,
    28581, 28581, 28581, 28600, 28600, 28600, 28601, 28601, 28602, 28603, 28603, 28603, 28605, 28614, 28614, 28615,
    28615, 28615, 28615, 28617, 28617, 28622, 28622, 28627, 28627, 28629, 28635, 28635, 28636, 28637, 28641, 28645,
    28648, 28649, 28650, 28651, 28654, 28666, 28667, 28669, 28672, 28673, 28678, 28679, 28679, 28679, 28680, 28684,
    28684, 28685, 28685, 28687, 28687, 28694, 28710, 28710, 28713, 28715, 28716, 28716, 28718, 28718, 28723, 28728,
    28729, 28730, 28734, 28737, 28737, 28741, 28741, 28742, 28745, 28747, 28747, 28748, 28753, 28755, 28755, 28757,
    28760, 28765, 28767, 28768, 28768, 28768, 28769, 28769, 28774, 28775, 28787, 28787, 28789, 28791, 28791, 28793,
    28794, 28794, 28795, 28796, 28796, 28797, 28798, 28800, 28800, 28807, 28810, 28812, 28819, 28820, 28822, 28824,
    28824, 28826, 28826, 28827, 28828, 28828, 28832, 28832, 28833, 28833, 28833, 28833, 28841, 28843, 28860, 28878,
    28880, 28886, 28886, 28886, 28886, 28886, 28890, 28891, 28898, 28899, 28915, 28915, 28916, 28918, 28918, 28919,
    28919, 28921, 28921, 28921, 28926, 28926, 28926, 28928, 28930, 28934, 28936, 28937, 28937, 28938, 28939, 28939,
    28945, 28945, 28959, 28959, 28963, 28964, 28977, 28984, 28984, 28984, 28986, 28986, 28993, 28993, 28998, 28998,
    29003, 29011, 29016, 29018, 29042, 29042, 29043, 29048, 29052, 29052, 29057, 29058, 29058, 29062, 29065, 29066,
    29066, 29067, 29069, 29069, 29073, 29073, 29073, 29074, 29074, 29074, 29074, 29074, 29076, 29076, 29077, 29077,
    29077, 29077, 29078, 29080, 29085, 29085, 29086, 29086, 29087, 29087, 29090, 29090, 29091, 29091, 29091, 29092,
    29092, 29093, 29123, 29123, 29125, 29125, 29132, 29144, 29147, 29147, 29152, 29153, 29155, 29156, 29156, 29156,
    29157, 29157, 29157, 29159, 29159, 29159, 29161, 29161, 29161, 29161, 29171, 29182, 29182, 29182, 29185, 29185,
    29185, 29185, 29185, 29187, 29187, 29187, 29201, 29203, 29206, 29207, 29219, 29219, 29219, 29220, 29220, 29221,
    29222, 29222, 29222, 29222, 29222, 29225, 29228, 29228, 29232, 29232, 29233, 29233, 29234, 29238, 29240, 29241,
    29247, 29247, 29250, 29254, 29255, 29255, 29256, 29257, 29265, 29265, 29267, 29268, 29272, 29276, 29276, 29281,
    29281, 29282, 29283, 29285, 29285, 29286, 29294, 29295, 29296, 29299, 29299, 29300, 29300, 29306, 29306, 29307,
    29307, 29309, 29309, 29309, 29309, 29309, 29314, 29325, 29325, 29330, 29330, 29330, 29330, 29330, 29330, 29330,
    29330, 29330, 29347, 29349, 29349, 29349, 29351, 29351, 29351, 29352, 29352, 29352, 29352, 29372, 29372, 29379,
    29392, 29393, 29395, 29396, 29398, 29398, 29401, 29405, 29407, 29407, 29411, 29411, 29411, 29412, 29413, 29413,
    29415, 29429, 29429, 29433, 29435, 29447, 29453, 29465, 29466, 29467, 29468, 29468, 29469, 29469, 29471, 29478,
    29479, 29479, 29479, 29480, 29480, 29483, 29484, 29487, 29489, 29493, 29496, 29504, 29505, 29505, 29506, 29507,
    29523, 29523, 29560, 29561, 29561, 29562, 29567, 29568, 29570, 29571, 29572, 29572, 29572, 29576, 29576, 29576,
    29599, 29599, 29599, 29599, 29602, 29604, 29604, 29604, 29619, 29620, 29620, 29620, 29620, 29620, 29620, 29620,
    29620, 29620, 29622, 29622, 29625, 29627, 29627, 29630, 29631, 29632, 29632, 29632, 29632, 29632, 29632, 29632,
    29632, 29632, 29633, 29633, 29636, 29636, 29643, 29644, 29644, 29644, 29644, 29645, 29651, 29654, 29654, 29700,
    29701, 29719, 29723, 29724, 29724, 29725, 29727, 29728, 29728, 29739, 29743, 29754, 29755, 29756, 29762, 29764,
    29764, 29764, 29765, 29766, 29766, 29766, 29766, 29778, 29782, 29782, 29784, 29784, 29785, 29786, 29786, 29788,
    29790, 29790, 29791, 29791, 29795, 29795, 29805, 29805, 29805, 29825, 29825, 29825, 29831, 29831, 29831, 29842,
    29843, 29843, 29843, 29843, 29843, 29843, 29844, 29849, 29854, 29855, 29858, 29864, 29867, 29867, 29869, 29870,
    29871, 29874, 29874, 29874, 29878, 29879, 29893, 29893, 29898, 29898, 29898, 29899, 29907, 29909, 29909, 29913,
    29921, 29921, 29923, 29953, 29956, 29956, 29957, 29957, 29957, 29957, 29958, 29961, 29963, 29972, 29972, 29975,
    29975, 29982, 29982, 29982, 29984, 29987, 29995, 29998, 30000, 30000, 30004, 30006, 30006, 30008, 30008, 30030,
    30032, 30036, 30048, 30048, 30048, 30048, 30049, 30050, 30050, 30068, 30068, 30068, 30071, 30071, 30072, 30072,
    30072, 30075, 30079, 30079, 30082, 30083, 30083, 30085, 30085, 30105, 30105, 30105, 30110, 30110, 30115, 30116,
    30116, 30116, 30116, 30116, 30116, 30116, 30118, 30119, 30119, 30119, 30119, 30119, 30120, 30120, 30127, 30127,
    30135, 30135, 30153, 30155, 30155, 30178, 30178, 30179, 30180, 30180, 30180, 30180, 30183, 30184, 30184, 30184,
    30184, 30187, 30187, 30196, 30201, 30202, 30208, 30210, 30211, 30211, 30218, 30224, 30227, 30227, 30233, 30241,
    30241, 30241, 30242, 30242, 30243, 30243, 30244, 30244, 30244, 30246, 30246, 30258, 30258, 30258, 30260, 30265,
    30268, 30268, 30268, 30269, 30269, 30270, 30271, 30281, 30281, 30281, 30286, 30307, 30310, 30318, 30321, 30322,
    30334, 30335, 30337, 30337, 30337, 30339, 30339, 30342, 30343, 30345, 30356, 30365, 30365, 30366, 30373, 30374,
    30376, 30376, 30376, 30376, 30377, 30379, 30380, 30385, 30385, 30406, 30406, 30408, 30410, 30416, 30417, 30419,
    30428, 30434, 30465, 30465, 30466, 30469, 30469, 30471, 30471, 30471, 30480, 30480, 30481, 30484, 30484, 30484,
    30485, 30486, 30486, 30487, 30488, 30488, 30489, 30489, 30490, 30493, 30510, 30519, 30525, 30526, 30539, 30539,
    30539, 30541, 30541, 30541, 30543, 30543, 30543, 30545, 30545, 30548, 30566, 30566, 30567, 30576, 30579, 30585,
    30590, 30593, 30594, 30602, 30604, 30604, 30604, 30606, 30606, 30607, 30607, 30607, 30607, 30607, 30614, 30617,
    30617, 30617, 30618, 30618, 30664, 30664, 30664, 30664, 30665, 30665, 30675, 30675, 30676, 30681, 30682, 30686,
    30690, 30692, 30692, 30696, 30702, 30706, 30708, 30708, 30709, 30711, 30713, 30715, 30715, 30715, 30715, 30717,
    30727, 30727, 30727, 30729, 30729, 30732, 30739, 30750, 30753, 30754, 30770, 30773, 30787, 30797, 30821, 30823,
    30823, 30833, 30837, 30838, 30841, 30842, 30846, 30846, 30848, 30852, 30856, 30858, 30861, 30863, 30863, 30863,
    30867, 30870, 30870, 30879, 30879, 30879, 30879, 30881, 30882, 30882, 30884, 30884, 30884, 30884, 30893, 30893,
    30893, 30899, 30900, 30900, 30906, 30907, 30907, 30908, 30912, 30920, 30923, 30923, 30923, 30923, 30927, 30928,
    30946, 30946, 30948, 30949, 30949, 30949, 30949, 30952, 30956, 30958, 30958, 30958, 30962, 30967, 30975, 30995,
    31001, 31003, 31005, 31006, 31011, 31018, 31018, 31019, 31023, 31023, 31023, 31023, 31024, 31026, 31026, 31029,
    31029, 31031, 31032, 31033, 31033, 31033, 31033, 31037, 31044, 31044, 31045, 31051, 31062, 31062, 31064, 31083,
    31084, 31084, 31084, 31088, 31088, 31092, 31098, 31101, 31102, 31102, 31104, 31106, 31110, 31111, 31111, 31113,
    31113, 31115, 31123, 31141, 31142, 31143, 31148, 31151, 31152, 31153, 31154, 31154, 31154, 31154, 31160, 31164,
    31164, 31164, 31165, 31178, 31194, 31195, 31196, 31196, 31200, 31223, 31223, 31301, 31302, 31303, 31303, 31303,
    31304, 31304, 31304, 31308, 31309, 31309, 31309, 31310, 31316, 31329, 31330, 31330, 31330, 31332, 31332, 31336,
    31336, 31336, 31337, 31340, 31340, 31340, 31342, 31350, 31360, 31361, 31362, 31363, 31367, 31367, 31367, 31367,
    31368, 31369, 31376, 31381, 31382, 31383, 31393, 31393, 31397, 31401, 31402, 31405, 31411, 31411, 31411, 31411,
    31414, 31415, 31417, 31417, 31421, 31421, 31421, 31424, 31433, 31434, 31434, 31444, 31444, 31444, 31447, 31448,
    31448, 31448, 31448, 31454, 31454, 31461, 31462, 31462, 31463, 31463, 31464, 31464, 31464, 31465, 31466, 31466,
    31466, 31472, 31473, 31473, 31473, 31473, 31485, 31485, 31492, 31493, 31494, 31497, 31497, 31497, 31499, 31501,
    31501, 31501, 31502, 31502, 31502, 31502, 31502, 31506, 31517, 31518, 31520, 31523, 31530, 31532, 31534, 31537,
    31537, 31537, 31538, 31543, 31544, 31545, 31545, 31550, 31550, 31555, 31556, 31556, 31557, 31557, 31562, 31565,
    31565, 31565, 31565, 31577, 31577, 31583, 31584, 31584, 31584, 31587, 31587, 31588, 31588, 31594, 31595, 31596,
    31599, 31599, 31599, 31600, 31615, 31615, 31615, 31623, 31627, 31630, 31635, 31635, 31635, 31635, 31635, 31635,
    31639, 31639, 31646, 31646, 31654, 31660, 31660, 31660, 31660, 31663, 31663, 31666, 31667, 31668, 31668, 31668,
    31668, 31669, 31669, 31669, 31670, 31672, 31676, 31676, 31676, 31677, 31677, 31678, 31680, 31680, 31681, 31681,
    31684, 31684, 31690, 31691, 31694, 31694, 31695, 31697, 31704, 31704, 31710, 31711, 31714, 31715, 31716, 31717,
    31717, 31717, 31717, 31719, 31720, 31720, 31720, 31722, 31724, 31724, 31724, 31724, 31724, 31724, 31724, 31734,
    31734, 31735, 31735, 31735, 31738, 31742, 31742, 31751, 31751, 31751, 31752, 31762, 31762, 31774, 31774, 31779,
    31779, 31779, 31780, 31780, 31782, 31782, 31784, 31785, 31795, 31796, 31796, 31797, 31797, 31798, 31798, 31800,
    31801, 31805, 31805, 31805, 31805, 31812, 31815, 31815, 31815, 31817, 31818, 31819, 31819, 31820, 31820, 31821,
    31823, 31826, 31859, 31859, 31870, 31870, 31870, 31870, 31870, 31874, 31874, 31874, 31875, 31877, 31877, 31878,
    31878, 31878, 31880, 31884, 31884, 31884, 31888, 31892, 31894, 31894, 31894, 31897, 31898, 31899, 31899, 31899,
    31899, 31900, 31901, 31901, 31901, 31902, 31902, 31903, 31904, 31905, 31909, 31909, 31933, 31934, 31934, 31935,
    31943, 31947, 31949, 31951, 31953, 31953, 31959, 31959, 31959, 31961, 31964, 31964, 31965, 31966, 31968, 31968,
    32045, 32045, 32046, 32052, 32052, 32054, 32058, 32058, 32058, 32063, 32063, 32063, 32063, 32066, 32071, 32077,
    32078, 32078, 32079, 32079, 32080, 32080, 32086, 32086, 32086, 32086, 32086, 32086, 32087, 32092, 32092, 32093,
    32093, 32093, 32096, 32096, 32097, 32098, 32098, 32100, 32100, 32100, 32107, 32107, 32107, 32107, 32108, 32111,
    32112, 32118, 32119, 32119, 32123, 32123, 32123, 32123, 32123, 32133, 32133, 32138, 32145, 32145, 32145, 32145,
    32148, 32151, 32151, 32151, 32153, 32156, 32156, 32163, 32163, 32163, 32164, 32169, 32171, 32171, 32177, 32182,
    32182, 32184, 32192, 32193, 32193, 32200, 32200, 32201, 32207, 32207, 32207, 32207, 32207, 32212, 32212, 32212,
    32213, 32213, 32215, 32215, 32216, 32216, 32218, 32220, 32220, 32220, 32226, 32227, 32231, 32231, 32231, 32232,
    32232, 32232, 32233, 32233, 32233, 32233, 32234, 32235, 32235, 32237, 32238, 32240, 32240, 32241, 32241, 32247,
    32249, 32249, 32250, 32252, 32253, 32255, 32255, 32256, 32256, 32256, 32277, 32277, 32277, 32277, 32278, 32280,
    32281, 32288, 32289, 32289, 32289, 32292, 32301, 32301, 32302, 32307, 32319, 32319, 32320, 32321, 32322, 32322,
    32332, 32333, 32334, 32334, 32336, 32337, 32340, 32340, 32364, 32364, 32364, 32383, 32383, 32396, 32396, 32396,
    32406, 32407, 32407, 32408, 32408, 32412, 32412, 32417, 32417, 32440, 32443, 32462, 32463, 32463, 32472, 32472,
    32477, 32477, 32481, 32482, 32482, 32482, 32485, 32485, 32488, 32488, 32488, 32489, 32490, 32495, 32495, 32496,
    32499, 32504, 32504, 32504, 32504, 32520, 32520, 32520, 32520, 32520, 32520, 32520, 32521, 32521, 32521, 32525,
    32528, 32529, 32529, 32533, 32535, 32535, 32540, 32571, 32572, 32573, 32573, 32574, 32578, 32578, 32578, 32578,
    32578, 32578, 32579, 32579, 32582, 32601, 32601, 32602, 32603, 32606, 32606, 32607, 32607, 32607, 32607, 32607,
    32607, 32609, 32609, 32609, 32610, 32613, 32615, 32615, 32615, 32616, 32618, 32619, 32619, 32620, 32620, 32623,
    32623, 32627, 32628, 32631, 32635, 32638, 32643, 32644, 32647, 32648, 32650, 32656, 32663, 32663, 32665, 32667,
    32667, 32667, 32669, 32669, 32671, 32673, 32676, 32676, 32676, 32682, 32682, 32682, 32687, 32695, 32695, 32696,
    32698, 32699, 32699, 32701, 32702, 32703, 32703, 32703, 32738, 32739, 32740, 32740, 32746, 32746, 32746, 32760,
    32761, 32761, 32763, 32765, 32769, 32769, 32772, 32772, 32772, 32779, 32779, 32789, 32816, 32820, 32820, 32823,
    32825, 32825, 32825, 32825, 32825, 32829, 32833, 32841, 32849, 32849, 32849, 32850, 32857, 32857, 32864, 32865,
    32868, 32869, 32879, 32880, 32880, 32881, 32883, 32885, 32893, 32903, 32903, 32903, 32903, 32905, 32905, 32910,
    32910, 32911, 32922, 32922, 32933, 32937, 32937, 32940, 32954, 32956, 32956, 32957, 32957, 32958, 32960, 32960,
    32968, 32968, 32969, 32969, 32970, 32970, 32972, 32973, 32973, 32978, 32991, 32993, 32993, 32998, 33000, 33001,
    33002, 33002, 33002, 33004, 33006, 33007, 33011, 33013, 33013, 33013, 33013, 33016, 33016, 33016, 33022, 33029,
    33029, 33030, 33031, 33033, 33039, 33048, 33051, 33052, 33054, 33057, 33057, 33057, 33057, 33058, 33059, 33059,
    33061, 33063, 33086, 33086, 33086, 33087, 33087, 33087, 33088, 33089, 33092, 33092, 33094, 33094, 33094, 33107,
    33109, 33110, 33115, 33126, 33128, 33129, 33148, 33149, 33149, 33152, 33154, 33154, 33154, 33154, 33154, 33154,
    33155, 33170, 33173, 33176, 33177, 33179, 33186, 33186, 33187, 33189, 33189, 33190, 33196, 33196, 33197, 33197,
    33197, 33205, 33239, 33242, 33242, 33280, 33281, 33288, 33288, 33291, 33292, 33292, 33293, 33294, 33294, 33294,
    33295, 33295, 33296, 33296, 33298, 33300, 33310, 33311, 33313, 33313, 33314, 33316, 33316, 33327, 33328, 33334,
    33336, 33336, 33337, 33337, 33337, 33339, 33354, 33362, 33362, 33362, 33362, 33364, 33365, 33365, 33379, 33379,
    33381, 33382, 33382, 33383, 33385, 33385, 33388, 33389, 33398, 33398, 33401, 33401, 33405, 33409, 33415, 33417,
    33418, 33422, 33423, 33426, 33431, 33431, 33432, 33432, 33433, 33467, 33468, 33474, 33485, 33485, 33490, 33491,
    33491, 33515, 33525, 33526, 33527, 33527, 33527, 33534, 33534, 33535, 33539, 33539, 33539, 33545, 33545, 33545,
    33545, 33545, 33545, 33546, 33547, 33548, 33550, 33551, 33554, 33554, 33557, 33559, 33559, 33571, 33573, 33574,
    33575, 33575, 33581, 33584, 33586, 33586, 33595, 33595, 33597, 33598, 33598, 33598, 33598, 33601, 33612, 33612,
    33612, 33630, 33630, 33642, 33645, 33649, 33653, 33653, 33653, 33656, 33661, 33663, 33663, 33663, 33666, 33666,
    33670, 33679, 33681, 33683, 33683, 33685, 33685, 33685, 33685, 33696, 33697, 33700, 33701, 33702, 33702, 33702,
    33706, 33706, 33706, 33709, 33709, 33709, 33709, 33710, 33714, 33715, 33716, 33722, 33730, 33730, 33737, 33739,
    33744, 33746, 33747, 33748, 33749, 33749, 33749, 33750, 33753, 33755, 33761, 33767, 33767, 33776, 33776, 33779,
    33780, 33781, 33782, 33783, 33785, 33787, 33788, 33792, 33793, 33793, 33793, 33796, 33799, 33806, 33807, 33807,
    33807, 33809, 33809, 33809, 33810, 33811, 33812, 33813, 33814, 33817, 33820, 33822, 33822, 33830, 33830, 33830,
    33831, 33839, 33839, 33840, 33840, 33841, 33842, 33843, 33849, 33849, 33851, 33852, 33853, 33853, 33853, 33853,
    33853, 33853, 33853, 33853, 33853, 33853, 33853, 33853, 33854, 33857, 33857, 33861, 33879, 33883, 33885, 33888,
    33888, 33888, 33888, 33894, 33897, 33897, 33898, 33898, 33899, 33900, 33912, 33916, 33918, 33919, 33919, 33921,
    33925, 33925, 33925, 33926, 33926, 33926, 33935, 33936, 33943, 33944, 33944, 33947, 33950, 33951, 33953, 33955,
    33965, 33967, 33969, 33972, 33973, 33974, 33974, 33986, 33989, 33991, 33993, 33995, 34002, 34002, 34002, 34039,
    34040, 34041, 34041, 34043, 34043, 34043, 34043, 34045, 34046, 34055, 34056, 34057, 34057, 34061, 34065, 34066,
    34066, 34068, 34070, 34070, 34072, 34072, 34073, 34079, 34079, 34082, 34082, 34082, 34082, 34083, 34084, 34086,
    34087, 34100, 34101, 34102, 34103, 34104, 34105, 34105, 34107, 34107, 34110, 34113, 34115, 34119, 34119, 34128,
    34134, 34134, 34134, 34136, 34136, 34137, 34146, 34154, 34154, 34154, 34158, 34159, 34174, 34182, 34184, 34185,
    34186, 34186, 34187, 34188, 34190, 34192, 34192, 34193, 34201, 34201, 34201, 34201, 34201, 34201, 34201, 34201,
    34204, 34204, 34204, 34205, 34205, 34205, 34206, 34206, 34208, 34211, 34213, 34213, 34213, 34213, 34215, 34216,
    34220, 34222, 34233, 34240, 34240, 34241, 34241, 34241, 34242, 34244, 34245, 34245, 34245, 34245, 34245, 34247,
    34248, 34248, 34248, 34252, 34252, 34252, 34253, 34256, 34256, 34256, 34261, 34261, 34262, 34265, 34267, 34267,
    34268, 34268, 34269, 34269, 34271, 34271, 34271, 34272, 34273, 34273, 34274, 34274, 34274, 34276, 34277, 34277,
    34278, 34279, 34279, 34282, 34282, 34292, 34293, 34298, 34298, 34306, 34306, 34306, 34306, 34325, 34326, 34329,
    34332, 34332, 34336, 34336, 34337, 34338, 34339, 34342, 34343, 34343, 34343, 34343, 34347, 34347, 34354, 34354,
    34361, 34363, 34363, 34369, 34369, 34369, 34370, 34372, 34372, 34374, 34377, 34382, 34383, 34388, 34389, 34389,
    34398, 34399, 34399, 34399, 34400, 34400, 34400, 34401, 34403, 34404, 34404, 34405, 34406, 34415, 34415, 34420,
    34424, 34428, 34428, 34428, 34430, 34434, 34445, 34445, 34445, 34447, 34447, 34448, 34448, 34448, 34450, 34452,
    34453, 34453, 34454, 34454, 34458, 34465, 34465, 34465, 34466, 34468, 34481, 34483, 34483, 34483, 34483, 34485,
    34488, 34488, 34488, 34489, 34492, 34495, 34501, 34504, 34505, 34511, 34514, 34516, 34517, 34519, 34521, 34534,
    34535, 34536, 34538, 34538, 34540, 34540, 34543, 34543, 34546, 34546, 34546, 34546, 34547, 34547, 34550, 34550,
    34551, 34551, 34552, 34563, 34564, 34565, 34565, 34567, 34570, 34570, 34572, 34572, 34575, 34577, 34581, 34582,
    34582, 34586, 34588, 34589, 34589, 34590, 34598, 34603, 34603, 34607, 34608, 34609, 34610, 34617, 34619, 34620,
    34620, 34627, 34636, 34646, 34646, 34647, 34647, 34647, 34648, 34649, 34651, 34678, 34678, 34683, 34688, 34688,
    34688, 34688, 34688, 34689, 34689, 34689, 34690, 34690, 34693, 34694, 34700, 34705, 34710, 34711, 34711, 34711,
    34713, 34724, 34725, 34726, 34741, 34743, 34743, 34744, 34747, 34748, 34750, 34752, 34755, 34756, 34756, 34758,
    34760, 34760, 34760, 34760, 34771, 34771, 34771, 34771, 34773, 34775, 34777, 34790, 34790, 34791, 34793, 34794,
    34797, 34799, 34800, 34801, 34803, 34803, 34804, 34805, 34805, 34806, 34813, 34814, 34816, 34816, 34817, 34821,
    34826, 34828, 34828, 34829, 34832, 34832, 34832, 34834, 34835, 34837, 34838, 34844, 34844, 34845, 34846, 34858,
    34867, 34867, 34868, 34870, 34871, 34874, 34880, 34880, 34883, 34884, 34892, 34892, 34893, 34894, 34895, 34897,
    34899, 34899, 34901, 34901, 34902, 34902, 34911, 34963, 34964, 34965, 34974, 34974, 34974, 34974, 34978, 34978,
    34978, 34980, 34980, 34982, 34982, 34982, 34982, 34990, 34994, 34995, 34995, 35011, 35013, 35024, 35030, 35030,
    35030, 35032, 35032, 35032, 35033, 35038, 35039, 35039, 35043, 35043, 35043, 35043, 35044, 35047, 35047, 35060,
    35060, 35060, 35067, 35067, 35067, 35068, 35070, 35070, 35070, 35070, 35072, 35072, 35072, 35072, 35075, 35075,
    35075, 35075, 35077, 35078, 35082, 35083, 35087, 35097, 35098, 35098, 35102, 35103, 35110, 35110, 35113, 35113,
    35113, 35119, 35122, 35122, 35123, 35123, 35125, 35127, 35140, 35140, 35141, 35141, 35141, 35141, 35141, 35143,
    35144, 35149, 35150, 35150, 35150, 35150, 35150, 35150, 35150, 35152, 35153, 35156, 35158, 35161, 35165, 35166,
    35166, 35166, 35166, 35167, 35167, 35167, 35168, 35169, 35170, 35170, 35170, 35171, 35180, 35182, 35182, 35182,
    35183, 35183, 35187, 35187, 35189, 35189, 35189, 35190, 35192, 35203, 35204, 35221, 35222, 35228, 35228, 35233,
    35238, 35239, 35242, 35245, 35247, 35247, 35248, 35256, 35259, 35264, 35264, 35264, 35266, 35267, 35267, 35267,
    35268, 35274, 35276, 35279, 35281, 35286, 35286, 35289, 35289, 35301, 35313, 35313, 35313, 35313, 35320, 35320,
    35321, 35321, 35326, 35331, 35331, 35332, 35335, 35335, 35335, 35335, 35335, 35342, 35345, 35356, 35361, 35364,
    35372, 35385, 35397, 35398, 35399, 35399, 35419, 35427, 35427, 35428, 35428, 35428, 35429, 35429, 35431, 35431,
    35431, 35431, 35432, 35434, 35435, 35441, 35441, 35447, 35447, 35502, 35504, 35511, 35512, 35512, 35513, 35514,
    35518, 35524, 35526, 35529, 35531, 35531, 35531, 35532, 35532, 35533, 35550, 35550, 35550, 35552, 35552, 35556,
    35559, 35561, 35564, 35566, 35566, 35566, 35566, 35586, 35589, 35595, 35595, 35606, 35606, 35606, 35610, 35610,
    35610, 35616, 35617, 35617, 35636, 35638, 35638, 35638, 35639, 35644, 35651, 35651, 35651, 35651, 35651, 35651,
    35658, 35658, 35662, 35664, 35664, 35668, 35669, 35669, 35670, 35670, 35671, 35671, 35677, 35678, 35690, 35692,
    35693, 35694, 35694, 35694, 35694, 35694, 35695, 35698, 35699, 35703, 35721, 35721, 35721, 35721, 35724, 35758,
    35759, 35759, 35759, 35765, 35766, 35766, 35766, 35766, 35766, 35768, 35772, 35777, 35777, 35777, 35777, 35781,
    35783, 35783, 35783, 35789, 35789, 35790, 35794, 35794, 35798, 35798, 35798, 35800, 35821, 35821, 35822, 35827,
    35827, 35827, 35829, 35829, 35830, 35830, 35839, 35841, 35849, 35863, 35864, 35864, 35867, 35872, 35872, 35874,
    35890, 35891, 35891, 35893, 35893, 35893, 35893, 35897, 35897, 35900, 35900, 35900, 35900, 35900, 35902, 35902,
    35904, 35907, 35907, 35911, 35915, 35915, 35915, 35915, 35915, 35915, 35915, 35915, 35915, 35916, 35916, 35918,
    35918, 35929, 35930, 35935, 35938, 35939, 35939, 35940, 35952, 35953, 35956, 35962, 35962, 35962, 35962, 35963,
    35971, 35975, 35975, 35975, 35975, 35975, 35975, 35975, 35976, 35979, 35986, 35992, 35993, 35999, 36014, 36014,
    36014, 36017, 36020, 36020, 36021, 36030, 36034, 36037, 36037, 36074, 36076, 36076, 36079, 36080, 36110, 36110,
    36111, 36111, 36113, 36117, 36117, 36117, 36117, 36125, 36125, 36126, 36127, 36129, 36131, 36131, 36134, 36137,
    36137, 36138, 36141, 36164, 36165, 36167, 36169, 36169, 36170, 36170, 36170, 36171, 36171, 36199, 36199, 36201,
    36204, 36205, 36212, 36214, 36214, 36214, 36216, 36218, 36220, 36225, 36228, 36231, 36235, 36239, 36239, 36239,
    36241, 36241, 36241, 36242, 36243, 36254, 36258, 36264, 36265, 36265, 36271, 36271, 36271, 36271, 36272, 36275,
    36275, 36275, 36276, 36280, 36280, 36283, 36283, 36289, 36289, 36294, 36299, 36299, 36302, 36308, 36308, 36308,
    36308, 36308, 36309, 36309, 36327, 36329, 36330, 36331, 36331, 36345, 36347, 36347, 36350, 36350, 36350, 36350,
    36357, 36357, 36359, 36361, 36361, 36362, 36363, 36364, 36364, 36365, 36365, 36372, 36372, 36373, 36374, 36377,
    36377, 36380, 36383, 36383, 36383, 36391, 36391, 36396, 36397, 36400, 36400, 36400, 36403, 36404, 36405, 36415,
    36415, 36416, 36419, 36421, 36421, 36424, 36424, 36425, 36426, 36426, 36427, 36427, 36428, 36428, 36429, 36429,
    36429, 36429, 36430, 36430, 36431, 36431, 36433, 36435, 36435, 36441, 36441, 36442, 36442, 36445, 36445, 36445,
    36456, 36458, 36458, 36458, 36458, 36458, 36461, 36464, 36469, 36470, 36494, 36501, 36501, 36502, 36504, 36504,
    36504, 36504, 36504, 36508, 36508, 36510, 36511, 36517, 36521, 36523, 36524, 36524, 36524, 36524, 36525, 36525,
    36525, 36525, 36525, 36526, 36529, 36529, 36529, 36529, 36535, 36535, 36536, 36537, 36543, 36543, 36545, 36548,
    36548, 36548, 36563, 36569, 36570, 36570, 36570, 36570, 36570, 36577, 36579, 36579, 36580, 36581, 36581, 36581,
    36581, 36581, 36589, 36592, 36594, 36594, 36594, 36594, 36594, 36595, 36596, 36605, 36605, 36605, 36606, 36609,
    36609, 36609, 36612, 36612, 36612, 36614, 36615, 36616, 36617, 36617, 36619, 36620, 36621, 36623, 36624, 36625,
    36625, 36626, 36626, 36626, 36628, 36628, 36629, 36629, 36629, 36629, 36635, 36637, 36639, 36648, 36657, 36657,
    36658, 36661, 36661, 36661, 36694, 36695, 36695, 36695, 36695, 36703, 36703, 36707, 36707, 36715, 36718, 36726,
    36727, 36743, 36746, 36752, 36754, 36754, 36755, 36755, 36757, 36760, 36760, 36761, 36762, 36764, 36764, 36766,
    36771, 36774, 36777, 36778, 36779, 36779, 36779, 36781, 36781, 36782, 36783, 36791, 36791, 36791, 36792, 36793,
    36795, 36796, 36796, 36796, 36796, 36796, 36796, 36797, 36798, 36798, 36800, 36800, 36801, 36806, 36806, 36806,
    36807, 36807, 36811, 36812, 36813, 36813, 36813, 36813, 36813, 36814, 36817, 36820, 36822, 36834, 36836, 36838,
    36842, 36842, 36849, 36850, 36851, 36852, 36860, 36860, 36860, 36863, 36866, 36866, 36866, 36867, 36868, 36868,
    36869, 36874, 36874, 36874, 36877, 36882, 36884, 36884, 36885, 36887, 36887, 36893, 36895, 36896, 36897, 36898,
    36901, 36901, 36903, 36903, 36905, 36907, 36915, 36918, 36918, 36918, 36918, 36918, 36918, 36921, 36922, 36923,
    36926, 36926, 36930, 36930, 36936, 36936, 36937, 36937, 36938, 36939, 36939, 36940, 36944, 36944, 36949, 36951,
    36965, 36965, 36967, 36967, 36970, 36970, 36971, 36971, 36987, 36987, 36987, 36988, 36989, 36994, 36995, 37005,
    37005, 37005, 37005, 37007, 37007, 37008, 37009, 37009, 37009, 37015, 37015, 37015, 37016, 37017, 37017, 37038,
    37038, 37039, 37042, 37046, 37046, 37046, 37054, 37054, 37055, 37055, 37057, 37063, 37063, 37063, 37065, 37071,
    37071, 37071, 37081, 37082, 37082, 37082, 37083, 37083, 37085, 37087, 37087, 37087, 37089, 37092, 37092, 37100,
    37101, 37107, 37108, 37109, 37113, 37113, 37121, 37131, 37131, 37131, 37138, 37138, 37164, 37172, 37172, 37172,
    37178, 37178, 37178, 37183, 37187, 37191, 37196, 37200, 37200, 37202, 37202, 37204, 37204, 37207, 37207, 37211,
    37212, 37213, 37213, 37213, 37213, 37213, 37217, 37218, 37222, 37222, 37223, 37234, 37235, 37249, 37256, 37256,
    37258, 37260, 37261, 37261, 37262, 37264, 37266, 37266, 37266, 37266, 37268, 37270, 37277, 37277, 37278, 37281,
    37282, 37282, 37283, 37286, 37286, 37290, 37290, 37290, 37291, 37292, 37304, 37308, 37308, 37308, 37309, 37312,
    37315, 37315, 37315, 37317, 37321, 37321, 37330, 37330, 37332, 37338, 37341, 37347, 37348, 37350, 37350, 37354,
    37354, 37356, 37356, 37356, 37367, 37368, 37368, 37371, 37380, 37383, 37383, 37383, 37383, 37383, 37387, 37391,
    37394, 37394, 37398, 37399, 37401, 37401, 37402, 37402, 37402, 37411, 37412, 37412, 37416, 37419, 37419, 37419,
    37433, 37433, 37433, 37433, 37433, 37433, 37454, 37456, 37457, 37465, 37469, 37469, 37469, 37470, 37470, 37471,
    37471, 37471, 37472, 37484, 37486, 37486, 37487, 37491, 37492, 37492, 37492, 37497, 37497, 37500, 37500, 37505,
    37505, 37506, 37508, 37508, 37508, 37509, 37513, 37513, 37513, 37514, 37521, 37525, 37526, 37526, 37528, 37546,
    37547, 37550, 37551, 37553, 37553, 37563, 37563, 37563, 37565, 37565, 37565, 37567, 37573, 37576, 37577, 37579,
    37579, 37584, 37587, 37589, 37592, 37593, 37593, 37593, 37594, 37595, 37595, 37597, 37605, 37608, 37614, 37617,
    37618, 37621, 37621, 37630, 37632, 37632, 37632, 37634, 37634, 37647, 37647, 37647, 37649, 37651, 37667, 37671,
    37671, 37672, 37678, 37678, 37679, 37679, 37683, 37686, 37691, 37696, 37697, 37704, 37705, 37713, 37715, 37715,
    37716, 37720, 37729, 37729, 37730, 37730, 37731, 37731, 37750, 37750, 37751, 37751, 37756, 37759, 37761, 37763,
    37765, 37765, 37765, 37765, 37766, 37766, 37770, 37772, 37777, 37779, 37779, 37782, 37783, 37783, 37792, 37792,
    37793, 37793, 37797, 37797, 37798, 37799, 37804, 37805, 37805, 37805, 37805, 37809, 37810, 37810, 37814, 37817,
    37819, 37819, 37819, 37823, 37824, 37824, 37824, 37825, 37825, 37828, 37840, 37841, 37842, 37845, 37845, 37847,
    37856, 37856, 37857, 37860, 37862, 37862, 37862, 37863, 37863, 37864, 37870, 37870, 37870, 37873, 37873, 37873,
    37877, 37894, 37902, 37902, 37903, 37904, 37904, 37905, 37905, 37905, 37923, 37934, 37934, 37940, 37948, 37954,
    37954, 37954, 37955, 37955, 37956, 37958, 37960, 37960, 37961, 37964, 37964, 37967, 37970, 37970, 37980, 37983,
    37983, 37985, 37990, 37991, 37993, 37993, 37993, 37999, 37999, 38004, 38007, 38009, 38015, 38016, 38018, 38020,
    38020, 38023, 38024, 38024, 38024, 38033, 38033, 38033, 38035, 38035, 38036, 38038, 38050, 38050, 38052, 38052,
    38052, 38052, 38054, 38054, 38054, 38055, 38056, 38059, 38065, 38065, 38071, 38071, 38073, 38073, 38073, 38073,
    38073, 38075, 38076, 38080, 38081, 38082, 38082, 38084, 38085, 38086, 38086, 38086, 38087, 38090, 38095, 38096,
    38099, 38104, 38108, 38108, 38109, 38109, 38113, 38113, 38114, 38114, 38114, 38116, 38133, 38136, 38154, 38158,
    38161, 38161, 38161, 38161, 38161, 38162, 38162, 38170, 38170, 38170, 38170, 38170, 38170, 38180, 38182, 38182,
    38182, 38182, 38182, 38182, 38182, 38183, 38185, 38185, 38189, 38193, 38194, 38195, 38195, 38197, 38198, 38198,
    38201, 38202, 38202, 38202, 38204, 38205, 38205, 38205, 38207, 38211, 38216, 38216, 38218, 38218, 38218, 38219,
    38219, 38220, 38220, 38224, 38227, 38227, 38229, 38229, 38238, 38240, 38244, 38244, 38244, 38246, 38247, 38247,
    38247, 38248, 38265, 38268, 38273, 38275, 38276, 38276, 38276, 38278, 38280, 38280, 38281, 38281, 38283, 38283,
    38283, 38288, 38293, 38300, 38300, 38301, 38301, 38301, 38301, 38301, 38302, 38303, 38307, 38307, 38309, 38309,
    38309, 38309, 38314, 38314, 38314, 38321, 38322, 38326, 38328, 38328, 38329, 38329, 38329, 38330, 38330, 38331,
    38333, 38333, 38339, 38340, 38347, 38348, 38351, 38353, 38353, 38353, 38354, 38354, 38356, 38362, 38363, 38365,
    38365, 38366, 38367, 38367, 38369, 38369, 38369, 38374, 38374, 38384, 38384, 38385, 38386, 38387, 38387, 38387,
    38389, 38390, 38390, 38393, 38393, 38393, 38393, 38393, 38393, 38397, 38397, 38397, 38405, 38406, 38408, 38409,
    38409, 38410, 38414, 38416, 38416, 38416, 38416, 38418, 38418, 38418, 38418, 38420, 38423, 38423, 38432, 38441,
    38448, 38449, 38449, 38449, 38449, 38450, 38456, 38460, 38464, 38464, 38466, 38469, 38469, 38473, 38474, 38501,
    38503, 38504, 38507, 38507, 38543, 38548, 38548, 38568, 38568, 38573, 38573, 38574, 38578, 38579, 38580, 38580,
    38582, 38585, 38592, 38595, 38595, 38596, 38597, 38611, 38617, 38618, 38619, 38621, 38621, 38625, 38626, 38627,
    38643, 38644, 38644, 38655, 38659, 38659, 38659, 38663, 38666, 38667, 38667, 38672, 38685, 38686, 38686, 38688,
    38689, 38689, 38702, 38702, 38704, 38704, 38704, 38704, 38704, 38713, 38713, 38714, 38721, 38724, 38732, 38733,
    38734, 38735, 38740, 38743, 38745, 38746, 38746, 38750, 38750, 38752, 38752, 38758, 38758, 38762, 38763, 38763,
    38767, 38767, 38771, 38778, 38778, 38784, 38787, 38787, 38789, 38789, 38789, 38802, 38804, 38810, 38821, 38823,
    38823, 38823, 38826, 38826, 38827, 38835, 38839, 38839, 38840, 38853, 38856, 38856, 38856, 38856, 38857, 38860,
    38862, 38862, 38862, 38868, 38868, 38869, 38870, 38871, 38871, 38871, 38871, 38872, 38873, 38889, 38895, 38900,
    38900, 38900, 38902, 38907, 38907, 38907, 38907, 38909, 38910, 38912, 38913, 38914, 38918, 38919, 38919, 38921,
    38921, 38925, 38925, 38927, 38929, 38929, 38931, 38954, 38954, 38954, 38959, 38959, 38965, 38965, 38968, 38969,
    38970, 38970, 38971, 38972, 38976, 38980, 38980, 38982, 38984, 38991, 39001, 39004, 39004, 39007, 39046, 39047,
    39047, 39047, 39047, 39048, 39049, 39049, 39049, 39061, 39062, 39062, 39063, 39071, 39073, 39074, 39075, 39075,
    39077, 39079, 39083, 39087, 39087, 39088, 39088, 39091, 39092, 39094, 39101, 39101, 39108, 39115, 39120, 39120,
    39120, 39121, 39121, 39121, 39121, 39126, 39131, 39131, 39132, 39140, 39140, 39141, 39142, 39143, 39153, 39153,
    39154, 39167, 39168, 39169, 39173, 39173, 39175, 39179, 39183, 39183, 39191, 39192, 39192, 39192, 39192, 39202,
    39207, 39208, 39208, 39214, 39214, 39215, 39215, 39215, 39218, 39222, 39223, 39238, 39239, 39242, 39268, 39268,
    39272, 39276, 39276, 39277, 39281, 39281, 39281, 39282, 39282, 39282, 39306, 39309, 39310, 39310, 39312, 39313,
    39314, 39337, 39338, 39339, 39339, 39340, 39341, 39343, 39343, 39347, 39349, 39349, 39350, 39355, 39355, 39357,
    39359, 39367, 39367, 39369, 39372, 39374, 39374, 39405, 39409, 39410, 39410, 39410, 39410, 39410, 39411, 39411,
    39411, 39412, 39412, 39415, 39416, 39419, 39419, 39420, 39421, 39421, 39425, 39427, 39429, 39434, 39434, 39434,
    39436, 39436, 39441, 39441, 39441, 39441, 39454, 39454, 39457, 39458, 39459, 39461, 39467, 39486, 39487, 39490,
    39490, 39495, 39496, 39496, 39496, 39496, 39497, 39500, 39500, 39508, 39510, 39511, 39512, 39513, 39514, 39520,
    39521, 39521, 39521, 39524, 39524, 39524, 39529, 39538, 39538, 39539, 39540, 39540, 39543, 39543, 39544, 39554,
    39554, 39566, 39569, 39574, 39574, 39574, 39579, 39579, 39606, 39610, 39610, 39610, 39610, 39610, 39610, 39610,
    39611, 39611, 39611, 39614, 39616, 39617, 39619, 39620, 39622, 39623, 39628, 39629, 39634, 39642, 39644, 39657,
    39657, 39657, 39660, 39661, 39661, 39661, 39661, 39662, 39662, 39662, 39662, 39662, 39663, 39668, 39671, 39671,
    39672, 39678, 39678, 39678, 39694, 39694, 39698, 39698, 39698, 39698, 39700, 39700, 39700, 39705, 39707, 39710,
    39719, 39719, 39721, 39722, 39748, 39750, 39760, 39762, 39762, 39764, 39764, 39765, 39769, 39769, 39771, 39773,
    39773, 39773, 39776, 39776, 39776, 39776, 39782, 39783, 39791, 39792, 39792, 39794, 39794, 39794, 39794, 39797,
    39798, 39803, 39817, 39817, 39817, 39817, 39822, 39830, 39830, 39834, 39845, 39845, 39845, 39852, 39852, 39852,
    39852, 39857, 39857, 39858, 39860, 39862, 39865, 39869, 39870, 39871, 39871, 39888, 39889, 39889, 39892, 39893,
    39901, 39902, 39903, 39903, 39906, 39906, 39911, 39914, 39915, 39915, 39915, 39919, 39919, 39920, 39921, 39922,
    39922, 39923, 39924, 39925, 39926, 39926, 39926, 39926, 39927, 39928, 39928, 39928, 39945, 39945, 39946, 39948,
    39949, 39951, 39953, 39957, 39957, 39957, 39959, 39959, 39962, 39962, 39962, 39966, 39968, 39969, 39969, 39972,
    39972, 39989, 39998, 40003, 40008, 40010, 40012, 40012, 40013, 40014, 40016, 40020, 40022, 40022, 40028, 40028,
    40035,
];

static immutable uint[NUM_OMIT_FIRST_WORDS] kOmitFirstWords =
[
    145931, 82922, 148458, 213994, 279530, 150526, 618827, 424022, 426068, 165985, 79119, 358406, 360455, 213646, 150432, 219885,
    223353, 616775, 356379, 227375, 162045, 364596, 168116, 438297, 78667, 147718, 150131, 151570, 153814, 154004, 154057, 154380,
    154449, 154513, 158026, 158097, 159944, 145234, 150467, 156445, 156451, 156592, 76650, 78966, 79501, 96256, 487884, 98415,
    350771, 81109, 217204, 80942, 90358, 159884, 358408, 620588, 98305, 364583, 104512, 497782, 106504, 368694, 434277, 370699,
    370719, 372739, 438279, 110616, 374784, 243713, 505867, 309260, 505870, 440344, 374810, 440351, 114705, 283120, 481280, 350464,
    284940, 287078, 352822, 420153, 289152, 616867, 356571, 225703, 491598, 78896, 90114, 419840, 422102, 145968, 283269, 430114,
    78690, 290832, 231472, 76313, 279506, 152558, 156615, 156641, 156663, 156666, 156667, 160247, 143442, 145546, 147200, 150177,
    153801, 348258, 350502, 352693, 354368, 354684, 356782, 432212, 84385, 84461, 78971, 217520, 279443, 283190, 415916, 350768,
    146206, 483943, 75996, 94658, 106607, 153730, 221810, 144730, 150344, 152355, 150521, 78729, 144270, 80874, 213386, 76065,
    146073, 146618, 81129, 81228, 149722, 150209, 151827, 284755, 153783, 219733, 350815, 418008, 90422, 156041, 418192, 418310,
    157720, 157806, 616580, 157918, 420069, 487594, 225615, 489534, 229384, 297056, 169989, 170028, 110627, 239651, 632867, 75798,
    78462, 281001, 82602, 211229, 618950, 212195, 79073, 76171, 78598, 219580, 223351, 79243, 278774, 149775, 283234, 155878,
    360534, 493585, 79312, 347062, 352436, 278826, 84556, 86276, 81440, 348682, 416463, 215683, 282987, 284827, 212529, 215298,
    215380, 215705, 152157, 285003, 418187, 419931, 485474, 354409, 420271, 354771, 225384, 422009, 225591, 291133, 160086, 422316,
    292940, 491538, 426043, 493570, 493658, 362597, 362601, 628814, 497755, 499738, 286982, 352926, 352928, 485670, 487892, 294957,
    149872, 280983, 284687, 283642, 362565, 280904, 281441, 284801, 351008, 286751, 221302, 288954, 289132, 356423, 148076, 149553,
    152326, 80697, 211138, 148148, 416767, 144893, 346566, 415992, 484316, 150433, 347057, 150490, 156452, 162015, 168122, 352633,
    90604, 497669, 301180, 110607, 79614, 146997, 86031, 167970, 79347, 288803, 80351, 152546, 146558, 144972, 279490, 144019,
    81650, 84412, 78157, 86896, 221885, 360555, 346387, 352843, 152295, 287148, 159853, 422157, 434226, 80482, 158149, 217653,
    153704, 154154, 161852, 161917, 151986, 154026, 279502, 422365, 144854, 81828, 79471, 81424, 81882, 84462, 89064, 428132,
    430188, 79117, 211312, 217456, 146866, 151672, 76652, 143682, 144753, 150056, 153874, 157905, 158195, 159829, 161918, 165958,
    217197, 237648, 143688, 281395, 227406, 79018, 346372, 346834, 86267, 422132, 288809, 292957, 161986, 80866, 282817, 79751,
    79768, 82844, 82868, 213991, 86983, 86994, 353243, 354526, 487918, 495867, 76131, 79023, 212809, 152134, 219297, 350912,
    156033, 229413, 622637, 100352, 231523, 280666, 286900, 80358, 82075, 152520, 88157, 416479, 90276, 211536, 146084, 146148,
    147899, 148065, 149528, 149829, 151587, 217131, 86547, 154135, 154272, 220036, 352453, 286948, 156285, 288884, 616774, 160051,
    163870, 428043, 430175, 434274, 237685, 147064, 151890, 147184, 150309, 616871, 279343, 145408, 75839, 146821, 154028, 620553,
    148178, 154428, 155794, 156161, 212755, 76542, 493623, 82783, 82784, 82788, 82804, 86884, 86889, 86927, 86960, 90839,
    231512, 147543, 286842, 286907, 354501, 487642, 620681, 78648, 145580, 147498, 86302, 354669, 147002, 145616, 217291, 79407,
    147850, 227344, 172047, 211230, 213621, 79741, 301057, 301082, 307209, 143510, 144075, 153745, 156537, 174133, 180251, 215378,
    352887, 217407, 223384, 82690, 154465, 285569, 364797, 80572, 147807, 84084, 149826, 88171, 88986, 90533, 94446, 362585,
    233535, 76599, 144646, 76763, 84804, 485387, 366719, 420325, 76615, 146707, 81223, 84549, 152152, 154525, 89078, 350755,
    352558, 278831, 215078, 217091, 219159, 219263, 146319, 146362, 146473, 81628, 84066, 151673, 174096, 81319, 348471, 285650,
    152243, 154208, 354482, 356527, 161853, 213559, 346525, 221465, 354464, 362551, 153920, 146100, 149985, 76746, 79530, 279362,
    153978, 618774, 283150, 219918, 223699, 227447, 147088, 78832, 77871, 79962, 77980, 80511, 81012, 146688, 81699, 83976,
    84430, 106579, 144809, 221672, 157881, 225289, 356403, 225344, 291159, 295034, 231436, 303204, 144947, 79859, 279079, 148248,
    152092, 285418, 144445, 618750, 222169, 225773, 233725, 86828, 491613, 491630, 218058, 284740, 213575, 149884, 88287, 161988,
    212710, 283353, 417859, 144497, 225573, 143788, 215652, 219440, 215253, 418648, 231515, 279102, 76788, 76336, 211366, 106499,
    292937, 144237, 76452, 143467, 144373, 146046, 147040, 149900, 88105, 98358, 100356, 102413, 84309, 229441, 296974, 299083,
    77929, 145811, 147956, 153694, 153931, 286940, 287336, 354754, 82878, 86979, 87017, 80559, 80977, 154377, 294928, 88206,
    81390, 144321, 213890, 81340, 76497, 144518, 84006, 86019, 88087, 88191, 80008, 82541, 150028, 215641, 217593, 217820,
    88276, 88432, 88482, 285178, 221323, 92175, 288818, 288838, 618549, 620557, 100411, 104476, 146542, 283140, 287428, 156372,
    291267, 295012, 299172, 172148, 354322, 159935, 227393, 161947, 229380, 294940, 229453, 163919, 428107, 82857, 217968, 221977,
    147109, 79353, 212940, 414127, 84481, 86038, 86055, 90234, 90299, 484053, 495777, 495778, 82514, 144585, 146415, 151989,
    220006, 616719, 223621, 491576, 432181, 432184, 149756, 77825, 94210, 356643, 78199, 80041, 80126, 82700, 143597, 146019,
    80933, 82002, 213136, 213351, 88314, 219402, 220080, 90532, 221825, 92344, 358563, 168037, 147217, 148026, 84077, 219662,
    289042, 303214, 436253, 307257, 86296, 352635, 354421, 81149, 82084, 102462, 211668, 110614, 112645, 112669, 213753, 146560,
    280581, 281248, 286936, 287014, 287120, 287238, 288997, 75973, 144303, 151695, 88914, 88967, 90260, 288974, 92504, 94535,
    96410, 102486, 144189, 80707, 165942, 79917, 148322, 84510, 84534, 86744, 152396, 152398, 156347, 618554, 626812, 76440,
    213284, 84841, 351225, 286938, 144065, 422178, 102445, 144556, 79671, 213917, 217995, 218043, 349139, 618955, 487900, 491628,
    229487, 634938, 144973, 146312, 146743, 281122, 283159, 223442, 283348, 84149, 84223, 86024, 86045, 86832, 88375, 88625,
    90179, 92341, 80413, 221466, 78337, 80924, 219417, 221280, 358636, 147575, 150053, 144939, 150140, 79315, 79494, 81915,
    82592, 84045, 90600, 419960, 225281, 144026, 81410, 213126, 80356, 217282, 282963, 88912, 354491, 495703, 143960, 147061,
    82432, 281215, 281493, 281499, 281506, 281510, 281517, 281527, 281537, 281566, 281575, 281586, 281592, 281597, 86138, 88511,
    90428, 90502, 90503, 90507, 90704, 287521, 287527, 287529, 287536, 287545, 484165, 287559, 484170, 287567, 287570, 287573,
    287579, 287582, 287586, 287594, 287598, 287601, 287604, 287605, 287607, 484223, 287620, 287621, 287623, 287624, 287645, 287648,
    287654, 484267, 484275, 92210, 92292, 92358, 92575, 290824, 94603, 96327, 96392, 489643, 293042, 293044, 489653, 293046,
    489655, 293050, 293054, 293059, 293061, 293064, 489674, 293071, 293083, 293090, 293092, 293095, 489704, 293110, 294939, 495719,
    299190, 495806, 299204, 299216, 299232, 495841, 299241, 301099, 303144, 305191, 305193, 305198, 305203, 305212, 311319, 311326,
    76158, 82039, 86759, 94317, 78556, 215406, 221684, 349108, 218037, 618985, 278925, 90434, 287392, 144971, 144244, 78805,
    76557, 76618, 77845, 77878, 144627, 79556, 79602, 81316, 81772, 82144, 84046, 217137, 152273, 86827, 284684, 219970,
    221317, 287132, 221679, 221770, 291121, 225665, 227446, 98387, 231444, 233570, 145689, 86182, 217270, 145673, 281304, 483495,
    78084, 145048, 81356, 349081, 349082, 353047, 159827, 491546, 628768, 82869, 217346, 144543, 150285, 151696, 219405, 153985,
    285639, 155960, 157857, 157924, 157964, 368721, 172139, 79861, 347133, 353134, 353157, 354362, 215339, 287131, 497750, 157723,
    616915, 76162, 80398, 414595, 96393, 215143, 213042, 215664, 280634, 213539, 279160, 219635, 485495, 212242, 77952, 286724,
    626722, 88342, 90319, 94281, 76233, 144307, 278766, 146463, 219862, 81154, 346540, 88282, 84048, 88142, 88598, 221234,
    90657, 212259, 147784, 213469, 82621, 82655, 152096, 351144, 154555, 352445, 354561, 158016, 485794, 356609, 229429, 495702,
    78578, 149837, 283490, 283536, 307263, 426049, 493582, 495691, 225395, 227405, 144694, 217410, 217807, 219543, 616621, 354515,
    301124, 366663, 147984, 211168, 151962, 346137, 280960, 348627, 348719, 283239, 76377, 94672, 143810, 80093, 146413, 147565,
    215051, 215369, 217423, 219335, 88821, 221366, 221623, 221782, 354304, 356566, 225663, 221502, 221689, 422199, 493581, 79657,
    79685, 82463, 145839, 414597, 482191, 418539, 418584, 215508, 76559, 81870, 81919, 144345, 227335, 76457, 78760, 147458,
    75814, 283567, 211649, 235560, 628839, 497776, 144894, 78451, 145877, 82914, 491610, 79267, 220031, 221600, 222179, 225763,
    225787, 432242, 432248, 78624, 351067, 78723, 283257, 86897, 416084, 352976, 90852, 90855, 90871, 223405, 84512, 78476,
    76785, 144977, 82394, 280601, 149659, 84620, 283091, 283183, 354355, 79803, 366659, 414562, 348733, 94668, 145259, 147125,
    153834, 481810, 347047, 358602, 145420, 148114, 149505, 154559, 158142, 415796, 76506, 145860, 309249, 218009, 221951, 75979,
    76473, 79985, 82262, 280738, 149864, 290963, 211541, 78995, 75931, 79997, 215182, 280864, 217182, 346827, 283399, 153698,
    281393, 86797, 90238, 144190, 80148, 145481, 80218, 283113, 78919, 145518, 82687, 84871, 150017, 151574, 151591, 155770,
    155835, 168159, 82569, 616929, 79606, 243742, 80301, 80991, 147566, 147869, 489502, 283354, 281529, 281574, 311324, 75813,
    77863, 78275, 80090, 80539, 81038, 81559, 213377, 82582, 215174, 346600, 217090, 217115, 217421, 219858, 220089, 352853,
    227329, 98306, 79852, 278643, 285627, 353254, 215218, 151775, 217873, 154245, 354359, 223344, 229397, 231452, 78742, 146980,
    348337, 491559, 493589, 624734, 495715, 76603, 77892, 80270, 80320, 81501, 81685, 147788, 82307, 82362, 84678, 347055,
    86298, 86738, 350323, 415880, 219302, 221709, 485564, 485811, 144606, 80089, 82060, 86626, 145238, 79871, 86985, 219589,
    76724, 626778, 213988, 349152, 280793, 147406, 80348, 86103, 346949, 80810, 285213, 146097, 212007, 146803, 278776, 88828,
    90627, 90728, 92571, 94268, 170001, 180235, 75805, 79919, 281146, 414455, 90750, 155884, 213167, 82117, 84015, 217256,
    217661, 88806, 155847, 287196, 156225, 299112, 76526, 84476, 215808, 285331, 88792, 155674, 352965, 353024, 92196, 94247,
    159922, 106526, 76779, 150262, 285342, 289013, 354597, 144522, 79291, 80473, 84642, 86194, 86273, 86434, 88595, 88608,
    88649, 90225, 90587, 92259, 94636, 150319, 499763, 503845, 356495, 217278, 350355, 83975, 481913, 489618, 112646, 79204,
    78734, 88946, 286914, 297067, 346449, 282892, 348677, 146137, 152552, 156628, 156661, 160225, 163956, 426110, 430325, 90217,
    156562, 147910, 170070, 156457, 156498, 353126, 156522, 156535, 156552, 353195, 414496, 225654, 438293, 145200, 144963, 76112,
    78302, 150345, 348624, 78286, 81331, 81341, 350560, 219692, 88969, 287049, 159796, 159837, 424041, 165938, 79169, 80191,
    81056, 150119, 152230, 152296, 154406, 290990, 229402, 229427, 366624, 78746, 150383, 155836, 618867, 76406, 143448, 146271,
    147085, 153784, 418418, 160043, 165950, 94488, 294960, 632842, 146233, 144266, 146286, 149989, 157960, 79447, 80260, 81343,
    88845, 221386, 225502, 231466, 144349, 144001, 145718, 152242, 81446, 217435, 217589, 415780, 418422, 77834, 82491, 84707,
    283554, 287414, 287417, 287498, 287508, 78856, 145153, 145552, 280919, 215642, 217333, 152400, 152427, 152447, 152451, 152454,
    219384, 418513, 156421, 144607, 84142, 86180, 88504, 88905, 96354, 76128, 215076, 215308, 215941, 282706, 217606, 217917,
    219324, 352525, 223306, 225326, 227361, 294947, 231464, 100416, 233508, 235540, 241665, 438285, 309261, 374805, 374807, 245762,
    311299, 145272, 150478, 76222, 145195, 148326, 148343, 282679, 418524, 144986, 145923, 148264, 149595, 150375, 150397, 350739,
    286994, 287067, 354403, 289194, 76570, 213286, 220123, 418341, 421955, 149594, 155828, 92601, 622700, 211228, 146659, 354556,
    414561, 157762, 172149, 147419, 150353, 152357, 84904, 96470, 102599, 305213, 114714, 81704, 86390, 217422, 484119, 237639,
    145740, 279355, 215422, 215689, 415818, 221189, 223557, 225567, 211857, 281088, 354809, 356612, 145869, 347109, 151987, 283287,
    286917, 358576, 147121, 90553, 165907, 156604, 156668, 278967, 489495, 144092, 155843, 88987, 223417, 290825, 421939, 290880,
    356695, 160096, 491605, 233486, 430324, 169995, 235589, 628805, 501769, 239641, 632857, 241706, 634922, 176182, 81547, 144545,
    145103, 213914, 94237, 75930, 81363, 86395, 86501, 483630, 495652, 503809, 147624, 79419, 156398, 156403, 287507, 291264,
    163944, 76580, 354327, 634924, 78609, 213873, 213903, 213904, 484100, 225720, 356805, 229472, 495791, 147656, 79590, 80016,
    143440, 145581, 150051, 152282, 485412, 225459, 292878, 161870, 294917, 241675, 176167, 79145, 350625, 144241, 147536, 157847,
    94642, 219779, 144856, 212757, 156093, 145521, 212785, 278578, 147798, 84190, 281200, 86059, 88964, 223286, 76739, 145425,
    82808, 154381, 286922, 291038, 297002, 495616, 144319, 81568, 92573, 158170, 143906, 217671, 356381, 167985, 170025, 75984,
    81400, 485462, 356366, 80470, 81998, 213374, 279084, 84134, 215350, 84608, 84653, 215761, 84741, 88279, 418120, 92225,
    288959, 354690, 94380, 94525, 100358, 416465, 80735, 88248, 352882, 420022, 94507, 100414, 78844, 143902, 217805, 284877,
    154043, 219874, 289233, 157714, 148296, 84870, 283478, 152444, 284718, 156367, 163945, 299176, 78701, 151755, 211255, 153928,
    210999, 213223, 147800, 217512, 219726, 227381, 624667, 231487, 76679, 144233, 145704, 145913, 148016, 148042, 155694, 157785,
    159856, 356645, 166010, 287288, 352884, 616563, 364623, 215845, 79116, 145309, 413974, 630859, 78614, 144201, 78591, 212917,
    213708, 84429, 146232, 285385, 618579, 80822, 291199, 80229, 287391, 358652, 282752, 151705, 219390, 350583, 155691, 618916,
    297039, 301136, 413794, 420220, 84624, 86480, 156627, 156652, 211203, 145951, 144749, 145098, 156453, 156565, 161993, 174134,
    79465, 82740, 215444, 84736, 86407, 217644, 217649, 283205, 350260, 154259, 285425, 352480, 616677, 354589, 354714, 223666,
    94239, 356654, 424044, 78657, 346539, 348907, 626693, 628745, 77911, 79892, 80928, 82888, 84374, 215718, 150330, 151743,
    283026, 152166, 86710, 156215, 84665, 84755, 88416, 285074, 89031, 287151, 90709, 96352, 622706, 94583, 145406, 278764,
    223619, 102539, 143419, 146293, 278598, 150107, 152067, 153888, 420120, 158093, 428111, 432208, 235634, 145551, 285212, 154357,
    290904, 291010, 90597, 156496, 143408, 144684, 147726, 151768, 153930, 416587, 215090, 88209, 289221, 169987, 79443, 283124,
    353208, 143569, 147679, 147924, 149714, 153636, 155676, 156278, 157717, 158045, 159751, 225376, 618840, 227401, 165901, 218105,
    353257, 484331, 80114, 81814, 78640, 81371, 212921, 82037, 84282, 86190, 78685, 283439, 350388, 221879, 352954, 221888,
    487866, 618947, 626860, 237681, 237692, 80138, 80951, 82724, 217373, 284752, 219533, 223407, 289055, 420321, 487445, 294973,
    231514, 301102, 301154, 303189, 78944, 245771, 218049, 221207, 370696, 144224, 156307, 279485, 283621, 418793, 78943, 145199,
    148369, 152468, 283564, 156346, 287427, 418500, 287451, 156380, 418546, 156418, 287495, 156427, 422339, 422345, 163946, 168099,
    168106, 168107, 180244, 483395, 616577, 485841, 624691, 211219, 145549, 80665, 81417, 82038, 82332, 213713, 88997, 89002,
    221378, 90411, 92296, 92330, 231531, 82638, 215860, 418630, 78865, 622645, 75977, 352469, 624718, 364695, 434184, 368709,
    372750, 86813, 213807, 148278, 620552, 626727, 630838, 632863, 634883, 636928, 442376, 144192, 347110, 630894, 88102, 219803,
    90224, 90329, 90355, 90674, 90775, 92301, 92357, 92387, 92566, 354729, 92647, 94411, 94464, 102458, 110592, 346924,
    348555, 352800, 94228, 227411, 80038, 86893, 86953, 86954, 90823, 622692, 360559, 350251, 78712, 80506, 417819, 420287,
    86072, 151872, 217840, 156182, 292895, 147278, 81744, 284723, 147889, 79481, 84008, 88200, 299046, 145046, 81573, 352984,
    422053, 618944, 491612, 346378, 152193, 434276, 278833, 348603, 485435, 223675, 360491, 622644, 235613, 147770, 76474, 146009,
    81178, 150178, 151730, 154144, 154185, 154404, 155733, 155761, 156123, 156164, 158137, 159821, 160172, 628753, 368662, 436232,
    243728, 213246, 84357, 358562, 303206, 145575, 279393, 149895, 151927, 153785, 147234, 98330, 235552, 212685, 143607, 144482,
    150059, 82703, 144891, 145035, 80451, 84212, 280984, 84477, 281179, 223538, 487811, 626768, 430219, 144306, 78190, 81256,
    81473, 279191, 215258, 151886, 294920, 172103, 77837, 81691, 281390, 348239, 212838, 78465, 79238, 291127, 144666, 162021,
    146184, 147601, 84868, 151849, 151984, 350306, 287180, 157996, 94501, 161967, 165959, 168054, 168151, 233718, 233719, 170040,
    170104, 78359, 353238, 616675, 170018, 634880, 86257, 283167, 287295, 147287, 80923, 215474, 84513, 416052, 432223, 143539,
    79669, 79677, 79682, 82782, 282925, 90894, 90900, 626864, 106613, 78100, 80154, 80751, 80796, 82379, 84567, 84722,
    88638, 90240, 90606, 92212, 92461, 96286, 96388, 78203, 80988, 82103, 213460, 215250, 84629, 90359, 620579, 231437,
    102525, 147150, 152184, 418324, 354339, 616863, 628777, 239627, 634901, 307245, 178188, 485639, 221285, 78534, 145790, 151607,
    151936, 153898, 154220, 158015, 158182, 161877, 80364, 220076, 96307, 145783, 212916, 152003, 217855, 235638, 225296, 165936,
    143873, 84659, 156199, 287315, 419875, 485853, 94348, 225444, 626777, 368654, 106537, 368741, 305163, 632847, 503834, 440331,
    243724, 440334, 374808, 309274, 374815, 114692, 442377, 507914, 618850, 78641, 80337, 80905, 146188, 82070, 150406, 90478,
    287702, 414406, 421914, 430230, 350805, 154468, 158174, 301061, 280955, 106533, 243716, 636956, 144351, 90758, 144587, 82877,
    212416, 213076, 79724, 90942, 90990, 91006, 96477, 96501, 146805, 149511, 178182, 146977, 147258, 147674, 147878, 150074,
    150230, 215939, 152108, 348745, 88391, 154240, 88768, 289053, 158130, 160020, 161944, 213879, 82818, 484060, 618954, 212695,
    82284, 221184, 212224, 223404, 144538, 86411, 153818, 286820, 632868, 283074, 217727, 223532, 618640, 291184, 231495, 233590,
    301159, 235640, 145967, 148096, 221329, 155831, 221608, 418317, 158184, 144281, 81181, 78280, 78565, 215400, 413982, 217866,
    219356, 219459, 420150, 225632, 362539, 493631, 299022, 628747, 634934, 81000, 354424, 356706, 358422, 368703, 80772, 149810,
    487591, 155689, 78491, 495632, 79134, 347033, 146586, 146268, 279321, 213824, 215926, 217888, 287223, 225284, 78547, 283117,
    78057, 80532, 80996, 147298, 147801, 147823, 217153, 151900, 217585, 152312, 219174, 350846, 221296, 352433, 221427, 221847,
    223459, 223638, 354767, 159823, 422029, 225536, 291175, 489501, 227473, 622603, 491558, 100383, 231458, 495626, 233530, 104470,
    235570, 301152, 499725, 630803, 303128, 368697, 368702, 305167, 241664, 112654, 243732, 346558, 348205, 628737, 628762, 634889,
    415786, 285062, 286752, 420078, 430218, 432206, 434253, 434268, 81131, 84093, 149888, 217285, 152167, 88129, 364716, 77900,
    82627, 215509, 219432, 88435, 352276, 352594, 221556, 221630, 221696, 287725, 354504, 290972, 225552, 151744, 624695, 79157,
    349018, 159801, 86141, 285454, 289235, 292983, 80132, 84600, 418415, 212339, 287073, 297049, 167999, 626798, 80165, 81944,
    82647, 82649, 84199, 84456, 86248, 86520, 86558, 88271, 221269, 501768, 622622, 78942, 81096, 215457, 153671, 499802,
    146885, 76547, 79456, 79468, 81835, 147645, 149859, 152005, 154016, 212931, 213735, 221568, 349180, 145475, 422304, 82369,
    346882, 218100, 222192, 420110, 422078, 225781, 347122, 146535, 81312, 215116, 88918, 88936, 88983, 88985, 221559, 157827,
    92632, 420328, 94575, 620600, 364661, 147554, 147729, 215149, 150067, 151657, 153954, 155698, 159866, 424069, 170056, 280804,
    84345, 356354, 618787, 80065, 80269, 81749, 82539, 84273, 86280, 348584, 88104, 90461, 92455, 94363, 358468, 362559,
    222174, 86600, 88756, 75861, 143869, 90631, 352789, 78217, 146022, 88242, 88306, 90717, 289124, 96282, 98348, 146476,
    415889, 153828, 482135, 168042, 172119, 353263, 144723, 145643, 146883, 278838, 149605, 152394, 350220, 154136, 352668, 356657,
    161860, 168056, 75960, 213913, 86772, 78827, 78030, 78187, 78189, 144668, 145062, 81827, 82645, 149688, 215717, 217250,
    282954, 217896, 287115, 287312, 292935, 622618, 426057, 624690, 366658, 75797, 77901, 81173, 81211, 81362, 81745, 84596,
    217339, 215343, 281334, 416414, 420085, 481483, 160164, 147230, 221237, 353070, 353088, 353158, 223273, 223724, 227421, 80553,
    81703, 281281, 235528, 86132, 483990, 484009, 279164, 280805, 80603, 287040, 92180, 159772, 290961, 291203, 96267, 80595,
    280594, 282982, 145082, 86058, 432214, 212864, 143841, 144494, 213478, 235551, 151758, 281442, 285653, 84757, 88590, 89003,
    92450, 420320, 82884, 346142, 162004, 162030, 223285, 231547, 144380, 348679, 77944, 81582, 84635, 354624, 428092, 497773,
    217931, 147487, 149672, 150268, 154469, 370707, 88659, 152437, 152443, 152452, 283532, 414624, 418537, 156401, 156408, 287499,
    422328, 160186, 160212, 426080, 426089, 82763, 155851, 90879, 145731, 79425, 151767, 217674, 358529, 147297, 215625, 150223,
    294978, 217954, 218000, 144779, 372773, 145969, 156114, 360468, 145738, 146371, 212217, 148219, 148231, 366652, 155958, 487813,
    78246, 79154, 79872, 81416, 626814, 170110, 350568, 618591, 618881, 358437, 491603, 626786, 149521, 88320, 146292, 81819,
    80238, 147932, 149742, 154116, 154175, 88995, 90668, 161901, 168036, 144922, 94349, 292947, 294923, 349064, 81302, 81889,
    278613, 82738, 84442, 88592, 90137, 90269, 90506, 90590, 287335, 90792, 92435, 92458, 223687, 487573, 356556, 291090,
    358469, 102493, 80234, 76519, 146914, 78873, 75925, 82153, 416309, 215872, 221909, 233633, 233634, 157841, 147078, 616662,
    283137, 143885, 146214, 146228, 213238, 147810, 152301, 159942, 160157, 167979, 78152, 78292, 143908, 81350, 81953, 82158,
    88531, 88945, 92377, 225450, 94476, 349054, 484067, 76610, 144881, 81074, 81934, 84848, 285179, 90223, 92400, 94290,
    104493, 78989, 147443, 147475, 147758, 148067, 149593, 217501, 348968, 152380, 152383, 156111, 420289, 421941, 423949, 426048,
    80976, 217740, 353181, 485614, 223501, 289109, 227390, 499789, 499804, 440336, 145409, 147474, 148134, 148331, 149665, 150175,
    150343, 295021, 291320, 144831, 143476, 213773, 215072, 84521, 217773, 225662, 416353, 287689, 287708, 291293, 291295, 104555,
    149911, 153615, 352654, 79383, 80492, 280786, 75892, 211703, 348538, 88885, 420058, 102441, 239650, 110611, 419989, 485626,
    485761, 414396, 156010, 157775, 418536, 291285, 213090, 280685, 88485, 484341, 225402, 489605, 235592, 632861, 145751, 82721,
    280913, 84308, 283141, 80046, 144200, 149653, 154041, 80718, 149685, 149759, 151581, 153911, 305180, 168017, 630877, 77924,
    80116, 80504, 81481, 82098, 82558, 215503, 217683, 92218, 77908, 414499, 291313, 76717, 149572, 86541, 152385, 88146,
    419905, 499823, 78500, 146089, 80687, 217926, 89028, 487567, 143752, 145473, 78366, 153647, 88507, 159917, 432235, 217908,
    220140, 143833, 146548, 165993, 78820, 220126, 225606, 211134, 213020, 213133, 219470, 221208, 221570, 221760, 172105, 106590,
    80738, 497748, 86208, 348508, 86756, 281473, 84847, 81527, 82225, 144974, 283547, 157846, 160026, 163938, 143901, 146844,
    152240, 219308, 154160, 622661, 154276, 145193, 287503, 291284, 161902, 295004, 638988, 354446, 421903, 487740, 146629, 348728,
    154507, 79713, 211012, 213080, 213574, 213673, 215413, 215904, 84926, 84939, 84944, 219472, 219979, 90908, 90911, 90920,
    90947, 90948, 91026, 91058, 96472, 96488, 102583, 102587, 102594, 102612, 102632, 237595, 108593, 76447, 144844, 283110,
    285053, 144308, 223315, 79090, 81679, 82479, 149656, 213171, 82490, 148141, 217271, 237678, 241721, 346993, 279309, 78497,
    80051, 145807, 80557, 82468, 148200, 148223, 84500, 346916, 150407, 86133, 282845, 86446, 88415, 88439, 352729, 354709,
    96279, 152029, 347064, 353172, 353207, 358604, 358650, 499762, 280915, 282726, 289184, 80081, 82581, 88098, 414215, 143905,
    145973, 81049, 147077, 213231, 82237, 149996, 154242, 156266, 158158, 159959, 88679, 291027, 423995, 221653, 80135, 147629,
    279425, 354755, 292979, 146396, 147347, 149568, 150006, 155655, 155682, 157708, 157755, 157930, 294964, 170061, 147485, 149776,
    154109, 154515, 364644, 144304, 143936, 79306, 145990, 146375, 81850, 147681, 82596, 90207, 290908, 78116, 283442, 285660,
    144112, 80641, 213009, 83986, 149691, 217305, 217321, 217383, 86374, 219212, 144608, 145596, 212063, 148047, 217134, 217907,
    153755, 223540, 225631, 225643, 233539, 144323, 211582, 161961, 162003, 428135, 168156, 376860, 616459, 211416, 217836, 154044,
    157815, 81189, 212742, 353235, 353260, 75850, 213253, 215184, 215506, 217458, 219955, 221654, 90731, 223376, 422070, 225628,
    622668, 434246, 489523, 143857, 146276, 348805, 485385, 428081, 495650, 215368, 215905, 219265, 285311, 285472, 221215, 221875,
    223418, 223596, 290887, 80402, 146069, 213187, 147883, 82512, 88117, 219325, 219909, 88843, 90279, 90523, 223395, 94222,
    94344, 96328, 98321, 491542, 98376, 428039, 237603, 90308, 360469, 374788, 144701, 212890, 223252, 290844, 620560, 358435,
    294941, 632839, 279154, 281137, 283085, 283383, 78411, 278906, 223713, 78863, 348449, 415967, 350606, 352335, 485463, 356589,
    81063, 146754, 219852, 155857, 155860, 157776, 165962, 80963, 215147, 88156, 285504, 352322, 223492, 616769, 160067, 423990,
    424048, 362600, 432186, 76250, 80379, 219140, 350708, 154556, 630805, 143794, 147455, 416656, 616589, 420267, 289256, 489528,
    358528, 106496, 78376, 143982, 145210, 146438, 81626, 148129, 213789, 215124, 217281, 348722, 217924, 219178, 221543, 156117,
    221691, 156226, 156361, 485553, 223470, 291116, 487846, 225716, 364640, 233610, 235598, 235631, 237645, 237660, 78713, 278538,
    278617, 213248, 281305, 283233, 285300, 285361, 354430, 292988, 628863, 278910, 281297, 483656, 420226, 76260, 278843, 281152,
    287081, 297070, 155649, 88614, 94561, 96284, 102536, 217217, 219254, 221248, 213249, 82797, 90843, 231534, 78642, 279424,
    283533, 291279, 291286, 78652, 86165, 283267, 88344, 354427, 92329, 225582, 94544, 79766, 79045, 144619, 78000, 144630,
    80267, 211017, 354784, 146525, 147995, 76200, 215156, 283166, 220058, 145249, 354656, 292890, 620630, 294956, 362593, 212365,
    418015, 221308, 147961, 151678, 153825, 157715, 485699, 159962, 160179, 215403, 221588, 76731, 80096, 82346, 88376, 149571,
    158169, 161874, 81405, 160200, 299182, 176188, 79567, 78921, 211021, 213118, 283574, 283583, 152553, 219454, 287683, 287688,
    287730, 295033, 143590, 146369, 147918, 90940, 91009, 91055, 354472, 489678, 489699, 102613, 144113, 156305, 143674, 165953,
    143431, 146253, 146716, 81791, 147616, 149822, 80232, 84246, 281476, 418660, 418672, 227544, 424184, 233691, 626907, 144042,
    144500, 212011, 280980, 215693, 153760, 289202, 80516, 84717, 88441, 78947, 79533, 87005, 91068, 215110, 88327, 223400,
    146498, 145826, 155766, 213834, 148306, 213843, 213847, 352989, 156382, 353022, 418617, 418744, 424124, 436264, 145265, 149964,
    150451, 416125, 417895, 76770, 79508, 145252, 81804, 81974, 156466, 156495, 156510, 156525, 156529, 156532, 156551, 353161,
    92651, 161974, 162011, 174121, 81073, 88970, 90639, 76280, 148240, 161808, 106511, 157751, 76782, 225713, 354342, 354384,
    362570, 297003, 489503, 299133, 80462, 217755, 154171, 155846, 221613, 221692, 157979, 290939, 159930, 291008, 225506, 161813,
    292891, 292939, 161895, 86861, 86878, 152486, 418562, 160218, 299186, 143378, 146129, 213683, 213723, 148213, 150348, 283179,
    152350, 153753, 284980, 155726, 156314, 418579, 159992, 301151, 149974, 616542, 618802, 347085, 358584, 150496, 280678, 288775,
    80411, 279037, 213934, 356842, 283499, 152445, 283519, 287493, 430240, 80577, 281004, 213542, 84887, 84890, 84895, 281514,
    84930, 84970, 84989, 217489, 415766, 219342, 287515, 484135, 287531, 287540, 287541, 484161, 484167, 91002, 91019, 484281,
    616599, 487751, 293039, 489666, 96463, 96484, 495797, 495798, 495809, 495817, 495826, 299223, 299225, 495834, 495851, 495852,
    299245, 501813, 501815, 501817, 501818, 501819, 108605, 507931, 165991, 144246, 414626, 418570, 287399, 144368, 145040, 285113,
    416466, 294998, 143382, 143530, 146311, 146488, 147851, 147943, 148125, 152129, 153725, 157843, 82178, 84176, 84327, 144506,
    215108, 215937, 76354, 145248, 426074, 168147, 86161, 221967, 76188, 414667, 144895, 80311, 86743, 88324, 88748, 144469,
    616723, 292899, 214006, 229449, 278600, 285271, 156407, 160203, 422357, 299095, 362579, 290912, 155707, 144596, 145044, 81621,
    213318, 82397, 364630, 146154, 422162, 428084, 364599, 235547, 152141, 154024, 358493, 493672, 213839, 213845, 213882, 218011,
    221926, 484074, 221968, 78898, 146364, 148461, 279533, 287103, 78930, 79056, 215544, 348438, 84842, 88508, 92279, 346751,
    146389, 213637, 90317, 144329, 82495, 86305, 414449, 88383, 168163, 82320, 84218, 98328, 163880, 149699, 485751, 235641,
    176160, 148358, 148366, 150503, 152431, 156369, 156374, 156455, 156473, 161982, 168118, 143932, 78483, 484301, 79645, 82796,
    82803, 82809, 82813, 82815, 349032, 86950, 88727, 90832, 92326, 157932, 94581, 487886, 168043, 364721, 104491, 507922,
    211708, 82100, 144016, 145980, 150096, 154280, 154442, 154524, 156004, 156198, 158013, 158066, 158165, 158167, 160141, 168005,
    168081, 348848, 288789, 76411, 81293, 81522, 82408, 84397, 219483, 219926, 90385, 90415, 483654, 288877, 354633, 223627,
    225678, 227391, 358494, 229464, 235625, 145886, 80679, 280587, 217663, 284871, 219638, 286902, 618513, 290995, 358414, 360453,
    307211, 241703, 144276, 145379, 280676, 79025, 353125, 161995, 162010, 168152, 168158, 168166, 174122, 174125, 145297, 80271,
    213378, 148394, 148453, 86240, 283601, 418797, 160229, 160240, 160244, 160253, 160255, 163955, 163963, 168190, 168191, 172158,
    172159, 346533, 81019, 221416, 92247, 80003, 82130, 84782, 90685, 80935, 76283, 76053, 144867, 215225, 86472, 86723,
    285211, 88655, 88675, 88738, 88962, 88966, 287289, 90703, 92407, 92563, 92574, 92616, 92630, 94289, 94474, 94534,
    94555, 96405, 231443, 102511, 143473, 145844, 81753, 215476, 221714, 144672, 76784, 88067, 416287, 92268, 94297, 487564,
    98326, 362607, 231538, 364590, 497703, 432192, 368672, 372744, 372746, 307230, 307233, 636940, 215317, 215376, 215922, 348928,
    219508, 145948, 82537, 86367, 84195, 84467, 86357, 88389, 88472, 219623, 484061, 484094, 616718, 94388, 96400, 102447,
    104497, 145599, 81330, 147478, 280852, 151863, 284727, 290854, 78332, 144817, 81589, 213857, 90848, 79208, 80450, 80720,
    81322, 146937, 279055, 280843, 346434, 84579, 86452, 92415, 98332, 78108, 81295, 84522, 285235, 154241, 144743, 485788,
    487449, 482215, 219964, 288950, 291107, 146115, 213316, 144927, 79127, 156500, 156588, 161962, 145134, 211993, 79051, 147947,
    144988, 146177, 147593, 148217, 84354, 150142, 282841, 217318, 282919, 152278, 348889, 153680, 88593, 155656, 155790, 288776,
    92459, 158009, 94283, 161905, 163901, 145487, 212505, 149597, 153860, 154074, 108574, 110645, 76079, 143698, 225414, 288798,
    354382, 223475, 293014, 364581, 145372, 299181, 243720, 505881, 79770, 79808, 79822, 222176, 222184, 233716, 144866, 79047,
    349030, 622690, 84267, 283644, 143724, 145924, 150022, 147177, 215243, 221239, 287147, 157927, 152534, 426102, 151896, 88956,
    79175, 76122, 75802, 143711, 80601, 80836, 147120, 81907, 81939, 84057, 86847, 284814, 221358, 90526, 82309, 92176,
    624737, 102532, 104573, 84450, 151970, 154131, 90459, 90561, 352771, 157795, 147744, 149690, 215539, 156243, 624757, 143844,
    146230, 231460, 432222, 78245, 79933, 81412, 86027, 86517, 352273, 212844, 143651, 146552, 285250, 223406, 233524, 503846,
    149883, 152154, 294913, 440321, 309263, 76588, 212320, 83993, 86483, 86575, 79375, 79310, 79358, 145574, 80648, 82065,
    104504, 360513, 428046, 430155, 144769, 147986, 79276, 146348, 155902, 78910, 144207, 152124, 81534, 84058, 90292, 90194,
    96292, 144847, 77865, 147930, 215195, 150156, 219791, 419891, 92320, 92383, 143392, 145793, 145892, 147550, 150045, 160052,
    413802, 212658, 145274, 88340, 156472, 145852, 213428, 215100, 346743, 215944, 217750, 416274, 223444, 223536, 223565, 231522,
    233527, 233582, 495731, 235624, 237628, 434285, 223235, 366657, 147456, 147627, 147969, 149624, 152025, 154256, 417878, 289197,
    424085, 622654, 297062, 167936, 167938, 170005, 243715, 75956, 78495, 86661, 293085, 233506, 278824, 352309, 354796, 76006,
    80392, 221344, 79174, 430122, 82187, 290974, 156594, 79284, 282681, 282852, 282901, 482244, 158205, 78470, 211702, 153798,
    155732, 157824, 144066, 143512, 149652, 215313, 215357, 149939, 150172, 217159, 217298, 151842, 217380, 217621, 153780, 153909,
    219691, 157915, 223495, 157963, 158019, 223691, 225551, 487771, 76333, 145434, 81329, 146944, 149534, 285476, 287236, 290893,
    239645, 82861, 489601, 143785, 349118, 79329, 145986, 212744, 223571, 217813, 352458, 221602, 352750, 616820, 225330, 422086,
    356574, 225575, 632854, 79224, 79860, 145138, 279193, 350938, 349053, 221894, 221902, 221932, 221949, 622701, 233644, 237689,
    414061, 616840, 82590, 146479, 151898, 419920, 356421, 160145, 146995, 213659, 143862, 146444, 147804, 149956, 151578, 151848,
    153824, 153987, 154372, 156277, 157891, 157903, 157937, 160004, 160014, 161932, 78669, 283032, 436259, 102559, 82042, 144785,
    75974, 79943, 86421, 94448, 76292, 81755, 292917, 281397, 422277, 493569, 362526, 432179, 145405, 145444, 147657, 148061,
    144421, 80307, 211399, 81555, 348371, 217679, 497764, 79342, 79422, 483785, 289067, 290891, 347017, 146747, 279470, 422378,
    149841, 346688, 282650, 152069, 285059, 352617, 288963, 362606, 285675, 76483, 76743, 150403, 145008, 146802, 146958, 215719,
    281346, 86728, 219574, 219737, 223425, 92635, 225580, 163881, 78674, 213467, 428123, 219705, 80080, 307237, 146924, 225380,
    491533, 493591, 233542, 299132, 628806, 632865, 634899, 81055, 213889, 218029, 286883, 221933, 618970, 215386, 153739, 88240,
    622659, 104480, 145111, 213017, 215196, 217730, 217914, 285103, 421926, 291040, 487698, 145036, 153743, 420065, 436248, 78494,
    283314, 414029, 217809, 76077, 212636, 81133, 279119, 213664, 362573, 366691, 237666, 81301, 217290, 370724, 76544, 81859,
    81912, 82663, 147514, 150228, 86820, 86848, 154409, 157809, 165960, 88549, 630841, 79571, 90186, 76674, 79604, 149587,
    84776, 78589, 279028, 79373, 145926, 147326, 84867, 213758, 221636, 78308, 80694, 82044, 92294, 96376, 102505, 77882,
    80617, 81841, 82160, 82595, 213801, 84447, 84699, 86609, 86833, 88197, 221491, 100420, 81307, 487453, 301097, 366677,
    145020, 353236, 353269, 626933, 78958, 78464, 144815, 211617, 147752, 76621, 81342, 81986, 213247, 213820, 84495, 223653,
    106580, 144571, 84272, 76442, 219553, 221407, 221449, 143922, 146468, 221379, 90321, 157733, 92374, 161807, 231510, 233489,
    432155, 145848, 82022, 147686, 86139, 86594, 86688, 90608, 90782, 92531, 92568, 94329, 94350, 100458, 102494, 79348,
    146088, 281480, 283454, 167965, 499809, 144689, 81111, 88195, 481528, 90482, 419894, 352941, 352951, 352960, 356785, 368753,
    78867, 80936, 346753, 217776, 144849, 146323, 147703, 626740, 169984, 144978, 213356, 150346, 348542, 86548, 420309, 145540,
    292946, 148431, 146577, 279054, 213572, 348689, 145459, 283243, 82066, 416440, 352885, 354476, 616755, 356622, 424015, 432236,
    144400, 145523, 149684, 149960, 84851, 151915, 147999, 281175, 286848, 289069, 98399, 76316, 76574, 79371, 78762, 487806,
    150218, 150253, 418794, 292923, 294958, 77914, 145418, 414470, 212245, 281448, 350517, 145514, 148059, 150097, 150138, 151725,
    154062, 154115, 157912, 157984, 160144, 624716, 364589, 358633, 370736, 376861, 146762, 157864, 84545, 89054, 90253, 90565,
    92207, 92426, 212847, 215454, 215590, 350482, 285295, 350981, 155915, 352702, 421994, 422044, 356624, 358492, 358526, 489635,
    493693, 299109, 366679, 79096, 81579, 219237, 284993, 221377, 223662, 356687, 620606, 301100, 628846, 295031, 146640, 150035,
    286816, 215323, 285409, 92162, 225428, 231516, 79250, 145690, 146332, 149965, 281111, 150103, 154174, 155776, 156142, 157997,
    161822, 161924, 360462, 80777, 147940, 77995, 79513, 81196, 81816, 149577, 84383, 420008, 159929, 94438, 166007, 102468,
    81029, 79046, 79209, 81392, 279545, 214011, 220112, 285679, 293029, 148170, 161985, 82870, 149870, 86999, 505869, 352910,
    146064, 146339, 149732, 282930, 485641, 485815, 622597, 426046, 503847, 281301, 151740, 158073, 168013, 364660, 146122, 279032,
    217578, 219332, 219802, 155810, 221446, 157771, 354514, 223625, 159903, 160064, 227443, 229421, 231506, 144499, 80558, 146424,
    84271, 86632, 92413, 79414, 218103, 222170, 225762, 225775, 225788, 217277, 282994, 419966, 288912, 291164, 499782, 76511,
    146011, 147562, 149737, 153657, 154059, 154358, 154599, 156238, 221849, 157794, 161930, 497730, 235621, 144837, 481322, 495754,
    364768, 497742, 77850, 279056, 350492, 352570, 354702, 297021, 104567, 217366, 350563, 143900, 145401, 147592, 84130, 151689,
    79630, 144647, 81040, 414677, 283633, 144121, 79088, 210948, 217654, 216000, 216007, 216041, 216063, 418675, 222096, 424108,
    424118, 424154, 620763, 430296, 632878, 638999, 278761, 145653, 211858, 151790, 157729, 79546, 145089, 80948, 90773, 354358,
    80922, 212693, 82697, 215230, 215519, 217466, 221375, 418815, 79228, 217195, 350377, 350976, 351143, 221620, 227379, 495628,
    233603, 432204, 81013, 146849, 82088, 215749, 86022, 414048, 86739, 352448, 485554, 292925, 78223, 78814, 84362, 616527,
    78891, 78972, 414681, 286741, 84242, 86333, 90246, 92309, 145953, 279052, 217720, 413908, 622710, 626937, 499837, 348610,
    283263, 289068, 356720, 297031, 299126, 366695, 301176, 76319, 77829, 80277, 146776, 81479, 81571, 151794, 86507, 152131,
    152294, 88681, 82834, 161929, 418454, 418473, 358442, 360471, 217912, 286845, 358520, 149615, 215291, 215990, 216057, 221524,
    156462, 353085, 156480, 353144, 156550, 353166, 222126, 223402, 227522, 180255, 144759, 283594, 153652, 215866, 219578, 420034,
    227418, 215571, 221505, 221727, 225736, 364718, 241724, 212357, 418212, 358423, 145121, 78736, 146010, 147635, 76023, 147422,
    81866, 79611, 82744, 616611, 86123, 90548, 104482, 630799, 215453, 157796, 163922, 221861, 79438, 279297, 364727, 364738,
    364756, 364776, 168174, 146567, 148415, 148420, 152512, 418781, 80989, 82459, 349128, 353091, 364731, 78205, 78592, 80321,
    80517, 80566, 84803, 144024, 78977, 79448, 79289, 90787, 80403, 82184, 424006, 283518, 418531, 422355, 438335, 153667,
    483574, 158192, 487840, 98325, 428143, 297074, 102419, 430126, 497728, 438280, 438282, 372766, 438303, 372769, 440328, 507921,
    481360, 420015, 428122, 497710, 84949, 283640, 76486, 94262, 227455, 153699, 223258, 420341, 280779, 217550, 86506, 88609,
    144631, 78802, 147124, 76667, 75910, 78232, 78319, 81465, 346661, 225427, 212427, 88127, 146061, 283299, 213305, 285247,
    299108, 211942, 81710, 218022, 219738, 484098, 364722, 146621, 278941, 84258, 84332, 86309, 217526, 152022, 88670, 155803,
    157920, 94253, 212319, 211791, 146574, 213522, 215185, 499719, 75909, 78393, 102404, 76311, 223332, 281335, 414441, 287375,
    92266, 223443, 158198, 159816, 225413, 225484, 160018, 227392, 227397, 227442, 161951, 229386, 294943, 294951, 163902, 163909,
    294987, 165918, 495631, 167978, 167989, 168020, 168076, 235555, 235571, 432185, 170052, 235591, 172037, 176148, 245775, 348330,
    348950, 78413, 217656, 285313, 484002, 430143, 235630, 616484, 356531, 423950, 425989, 372747, 307239, 147436, 213040, 151812,
    285602, 424098, 624686, 368742, 75961, 287337, 424046, 231450, 148434, 283573, 219352, 418760, 418796, 426105, 145450, 153758,
    159940, 165973, 212390, 229475, 507923, 212557, 81966, 149741, 215578, 147931, 79350, 280820, 150005, 79053, 148086, 151580,
    280910, 76441, 145998, 146284, 283392, 616511, 223594, 225431, 227368, 227387, 229422, 168082, 626834, 168088, 237587, 237634,
    239639, 485740, 428138, 82931, 79352, 350598, 352288, 143670, 81445, 147257, 147528, 147728, 213763, 149516, 149855, 152115,
    152182, 152362, 153967, 154199, 154286, 156320, 301182, 81381, 299105, 143453, 80580, 80889, 146830, 280765, 285150, 286737,
    483891, 212888, 211044, 350318, 285336, 422198, 620657, 360532, 628838, 432250, 213170, 213630, 350229, 354390, 414471, 285176,
    88569, 88626, 88731, 92262, 92528, 92634, 150430, 150444, 150447, 353071, 358601, 370742, 428073, 80385, 491571, 145834,
    156454, 354494, 82841, 222172, 281339, 78166, 78320, 483836, 420123, 82341, 83968, 152160, 219683, 221598, 487571, 628851,
    144328, 145542, 145713, 81961, 415811, 78799, 144526, 146900, 221816, 75811, 78111, 144511, 79108, 80734, 146555, 81758,
    86195, 286952, 487550, 360542, 156447, 168119, 168130, 168148, 168168, 144762, 144509, 282733, 303184, 76792, 78691, 81576,
    90669, 76126, 77883, 78180, 80757, 82095, 84540, 84571, 86184, 86531, 86589, 92557, 628803, 78147, 217327, 212952,
    84716, 483727, 80172, 152031, 414360, 283355, 90383, 90609, 92216, 100461, 80599, 219673, 285354, 422173, 102473, 81895,
    353086, 358617, 292885, 424011, 360505, 156590, 82873, 360562, 491647, 78085, 82102, 89007, 213957, 222206, 489480, 495655,
    495665, 497705, 499766, 501791, 503811, 505856, 290818, 143384, 145482, 146086, 149762, 292998, 80146, 88571, 90389, 354356,
    358430, 75812, 144220, 284941, 90635, 76710, 213074, 219386, 219507, 220068, 221604, 223416, 146248, 146623, 354584, 362575,
    366672, 84458, 356854, 501796, 78851, 279428, 283498, 283510, 291276, 426079, 434292, 287172, 86978, 78221, 148347, 152471,
    156376, 434300, 81438, 82189, 84180, 84686, 281394, 151913, 94223, 160060, 81364, 156280, 144297, 144931, 82741, 348162,
    100417, 153859, 146882, 211377, 146682, 144405, 152413, 79152, 87009, 144130, 76799, 82415, 219573, 358638, 144674, 78905,
    218018, 221962, 626742, 161869, 146687, 147378, 156046, 172071, 153635, 157767, 157873, 160166, 144378, 299094, 348712, 501775,
    309262, 143753, 75893, 78338, 78365, 81036, 81308, 82140, 82568, 217495, 86704, 153772, 88624, 225364, 497708, 241682,
    215080, 282682, 217200, 219272, 219525, 223518, 219190, 78804, 283459, 81232, 353163, 347044, 213343, 215362, 76274, 144248,
    79720, 281494, 90921, 484138, 484174, 90962, 90970, 287590, 90986, 90999, 91016, 484239, 91041, 287659, 91060, 91064,
    96454, 489675, 96462, 96476, 96483, 293101, 96503, 495795, 108592, 108594, 114713, 114717, 215852, 149999, 168040, 144300,
    215366, 151994, 354438, 364649, 283070, 79256, 79288, 147549, 149943, 217206, 217886, 154209, 348865, 351063, 156075, 167974,
    143694, 144993, 213255, 217461, 219636, 154237, 89034, 79151, 353271, 285109, 353102, 150328, 428100, 497745, 75816, 81510,
    212995, 213563, 348622, 366648, 220037, 354808, 88826, 353249, 301138, 79268, 350219, 352678, 354566, 356712, 144823, 616471,
    213953, 414468, 484303, 79200, 146554, 75851, 84485, 418025, 213390, 280750, 215721, 219768, 285513, 287091, 418472, 353040,
    289126, 297011, 430173, 147381, 148172, 281435, 221859, 151667, 349071, 154020, 155802, 158100, 160090, 161846, 161904, 376852,
    78608, 144493, 79223, 147026, 350241, 237605, 370689, 78004, 81009, 82671, 88869, 419949, 413864, 422092, 424005, 493592,
    79040, 211429, 80870, 80917, 81732, 84016, 84660, 88483, 88666, 146445, 79973, 215070, 351012, 356429, 78522, 78767,
    79010, 157865, 144000, 211115, 215439, 223582, 225430, 303175, 80774, 434265, 217160, 78560, 144563, 79594, 145164, 279213,
    213718, 148298, 148307, 148311, 280702, 346935, 282712, 282966, 152554, 152567, 285168, 351217, 287453, 287486, 156634, 156646,
    354331, 160226, 160231, 160233, 160239, 160241, 160252, 78515, 143542, 279105, 485383, 212961, 213810, 218088, 222164, 222197,
    495861, 212752, 153921, 418330, 420263, 422054, 422055, 231481, 84673, 86291, 88247, 285279, 90505, 77832, 145451, 145692,
    82011, 147833, 76376, 144440, 348392, 170043, 176130, 245766, 79773, 280836, 484102, 217408, 283376, 145226, 215864, 161971,
    144921, 79466, 79827, 80002, 148161, 281443, 283461, 286882, 353053, 353083, 353174, 288843, 92503, 291136, 79406, 149625,
    281218, 151964, 217565, 144044, 278736, 79378, 213716, 84041, 94393, 100471, 144120, 76212, 78695, 148352, 217660, 80494,
    82069, 84563, 86665, 86799, 219334, 88359, 88703, 88864, 89000, 90185, 90189, 221268, 223360, 92482, 92671, 94279,
    94465, 104506, 153750, 219370, 144060, 145847, 211492, 215098, 152279, 154284, 287004, 213119, 146394, 213389, 630840, 77853,
    279396, 350849, 76278, 146203, 285270, 144648, 212366, 362509, 79959, 86898, 86899, 86930, 86944, 484030, 353015, 94496,
    487872, 225738, 489505, 145063, 423945, 78364, 82056, 82258, 84645, 86824, 88572, 225283, 225511, 227369, 227494, 493618,
    235586, 80984, 217286, 79801, 82840, 82843, 82846, 82847, 82859, 82867, 213970, 82917, 281347, 349109, 86966, 86970,
    86972, 86975, 86980, 86989, 86991, 86997, 87000, 87004, 87011, 87015, 87018, 87019, 87021, 87022, 87024, 87028,
    87031, 87036, 91071, 91073, 91075, 91077, 91079, 91080, 91082, 91083, 91084, 91085, 91086, 91087, 91089, 91091,
    91095, 91096, 91097, 91098, 91099, 91101, 91105, 91107, 91110, 91111, 91113, 91114, 91116, 484332, 91117, 91118,
    91119, 91120, 91121, 91122, 91123, 91124, 91126, 91127, 91130, 91131, 91134, 91135, 616474, 288906, 94686, 94688,
    94690, 94691, 94692, 94693, 94695, 94696, 94697, 94698, 94700, 94701, 94702, 94703, 94704, 94705, 94706, 94707,
    94708, 94709, 94711, 94712, 94714, 94715, 94716, 94717, 94719, 292992, 98418, 98419, 98421, 98423, 98425, 98426,
    98427, 98428, 98429, 102648, 102653, 102654, 102655, 106622, 106623, 221430, 176159, 245777, 213724, 215951, 414671, 152548,
    284956, 418762, 289166, 418106, 616488, 84781, 144181, 149671, 149816, 158030, 80837, 81332, 281303, 281524, 281525, 281544,
    281567, 281585, 484122, 287566, 287600, 484265, 299196, 299235, 495850, 147324, 282895, 147747, 482289, 76067, 80637, 146198,
    146247, 285559, 90660, 146717, 148362, 152334, 283529, 152476, 154177, 156002, 90629, 156348, 156354, 156377, 287450, 156384,
    287509, 92467, 225460, 159973, 233519, 145018, 286791, 354335, 81818, 76421, 145981, 147853, 159788, 426106, 430329, 634904,
    503852, 154576, 76796, 82882, 79410, 161827, 282704, 221831, 221833, 291024, 279201, 348598, 414294, 348870, 75918, 78370,
    157738, 78687, 213999, 144033, 211267, 213199, 215286, 215802, 217391, 217631, 283402, 220065, 221408, 221410, 221516, 155985,
    221759, 419930, 223426, 362522, 428133, 235626, 145262, 213140, 213783, 432183, 370728, 432163, 432203, 145245, 82123, 219202,
    88887, 88910, 90594, 156603, 92293, 92437, 92462, 92624, 94413, 94429, 144381, 82339, 143797, 149922, 86324, 153781,
    157879, 147411, 278584, 279303, 285357, 288894, 628773, 78532, 79168, 145454, 213597, 285644, 170111, 423957, 491577, 145030,
    84346, 158003, 79164, 90859, 145001, 148276, 426070, 79890, 80571, 81838, 82672, 84241, 84380, 84671, 86087, 86524,
    154391, 356699, 102497, 80623, 84524, 354747, 82455, 215543, 153952, 352687, 156245, 287699, 287724, 76747, 79696, 279327,
    287519, 484136, 90988, 90992, 489682, 495796, 299191, 495800, 299202, 299214, 299220, 299240, 102638, 311320, 78062, 79026,
    145042, 80141, 80683, 279207, 213744, 82756, 281488, 152367, 278900, 280698, 282799, 86914, 80686, 75789, 77966, 79247,
    150122, 152034, 352850, 416623, 79841, 154126, 223506, 485856, 94409, 160062, 370717, 413724, 79897, 82122, 86552, 147310,
    215389, 88162, 88588, 289213, 225704, 428049, 78879, 82881, 148126, 217476, 283293, 88897, 155807, 287020, 90456, 221611,
    92391, 362611, 233510, 414714, 147500, 150388, 152132, 217733, 152299, 284867, 153999, 219645, 154494, 220051, 351228, 290827,
    291198, 620582, 161914, 364578, 430180, 168080, 432187, 170045, 434279, 438274, 507910, 150048, 278534, 620616, 104572, 172080,
    307229, 507917, 279332, 217389, 280991, 616616, 291046, 299076, 77997, 284742, 220158, 354500, 146165, 147493, 148167, 151714,
    152347, 154589, 154610, 155662, 79875, 285594, 364594, 75828, 88855, 147413, 147587, 147922, 157948, 172135, 84055, 86381,
    76195, 80736, 84438, 636932, 415836, 419942, 144226, 82384, 84269, 156145, 157752, 285466, 622678, 289276, 293031, 81686,
    145769, 81088, 147527, 82440, 84313, 84371, 156032, 94558, 218031, 618953, 78443, 80770, 80964, 82405, 346820, 356451,
    102428, 104564, 212947, 160044, 153732, 143557, 348525, 153621, 157782, 155945, 167961, 146434, 150216, 153620, 153965, 154305,
    154617, 156291, 144878, 78596, 358473, 79503, 79538, 79838, 80205, 81965, 346955, 86995, 487934, 80873, 152014, 293053,
    293076, 79159, 156580, 150522, 418245, 156549, 161972, 349067, 350929, 351202, 616862, 618998, 76276, 78501, 78868, 81574,
    84691, 88397, 422165, 144765, 76718, 288865, 499726, 144533, 80513, 286783, 288797, 290977, 290980, 292882, 434190, 212723,
    413885, 76107, 78242, 145742, 81636, 279538, 84339, 88878, 626731, 80262, 215993, 86156, 620729, 245788, 151803, 287070,
    288796, 288927, 420338, 292953, 299013, 301065, 303138, 307247, 372758, 374789, 374813, 76772, 78314, 215234, 217180, 229450,
    80999, 147626, 149565, 84577, 154103, 154277, 90716, 92291, 419978, 354792, 159784, 159946, 281403, 144657, 483809, 147222,
    212145, 146573, 287735, 144396, 81334, 144640, 82224, 354316, 290985, 366669, 481313, 481937, 501761, 143870, 279229, 149698,
    151644, 163914, 79099, 346149, 76246, 144595, 497665, 497690, 503817, 215498, 217163, 221662, 223289, 223507, 233565, 481316,
    483958, 145332, 147226, 174113, 305186, 143933, 84630, 152185, 285012, 92333, 77988, 147332, 149811, 84497, 364700, 241714,
    307255, 155745, 94381, 366699, 77906, 80010, 84641, 75886, 76152, 77993, 79526, 80272, 82458, 84152, 84492, 86681,
    90330, 279384, 84817, 86821, 176191, 80352, 346203, 149921, 149997, 420042, 78750, 213570, 482056, 223511, 145319, 211724,
    152562, 350691, 237683, 237686, 237690, 82076, 147526, 155819, 157968, 358649, 414291, 362513, 370718, 372789, 81253, 213961,
    218075, 360564, 622718, 626930, 144318, 144243, 81990, 90283, 420046, 92432, 211667, 416424, 416586, 422285, 212766, 482299,
    221339, 487451, 354716, 356377, 245778, 80098, 347072, 78720, 481296, 78878, 499772, 303191, 212039, 415885, 297025, 84117,
    212957, 81664, 420092, 76091, 88586, 211000, 213329, 294953, 628833, 145290, 148427, 76352, 212206, 215329, 481346, 285001,
    351031, 351054, 285637, 354437, 354581, 354606, 354768, 356557, 356573, 81483, 92174, 360467, 78555, 279150, 346273, 416484,
    285450, 288830, 354545, 358540, 80567, 81283, 154250, 624659, 79021, 147197, 213595, 215633, 348266, 217261, 219651, 416537,
    418202, 418484, 616482, 223520, 420214, 618597, 225680, 620704, 430125, 626817, 434231, 630884, 148123, 354613, 346917, 81459,
    282783, 154462, 154475, 351153, 351183, 221244, 146870, 219749, 487926, 145397, 299258, 299260, 221836, 79462, 219170, 484077,
    353037, 225728, 98408, 102577, 346174, 286844, 280824, 215684, 215739, 414027, 212553, 146159, 154471, 279482, 354652, 223669,
    225687, 628853, 507916, 151620, 154094, 349090, 352950, 353034, 353044, 147938, 485630, 211348, 148128, 346977, 145503, 213082,
    213407, 219850, 287190, 487606, 225561, 146372, 148101, 149726, 151595, 153685, 153777, 154500, 156102, 86084, 88558, 146613,
    152189, 483555, 420049, 428102, 144135, 219547, 144848, 78563, 352944, 493586, 364621, 221205, 147655, 149657, 151573, 154252,
    155840, 235562, 237611, 239616, 178193, 243739, 212956, 217566, 350748, 616721, 354703, 356440, 618586, 356546, 79832, 144951,
    147959, 212004, 157910, 294935, 150320, 86205, 428033, 430116, 438273, 282667, 285572, 352484, 279468, 287136, 290902, 81590,
    84063, 86295, 86436, 88367, 88872, 90479, 92507, 94557, 94614, 102422, 102489, 102522, 104495, 84628, 86363, 79887,
    213202, 215760, 215938, 217468, 217726, 283337, 348970, 219409, 154614, 221245, 352914, 211220, 493671, 211367, 354551, 493587,
    76551, 147093, 82673, 76182, 81028, 81059, 82193, 84377, 84531, 86121, 348702, 88158, 88418, 79688, 79690, 79693,
    346267, 281228, 84888, 84889, 84965, 84985, 482089, 155994, 90912, 90960, 91023, 91065, 157745, 354511, 96440, 96447,
    96455, 489697, 293094, 489713, 96499, 489724, 495807, 102603, 495830, 108584, 108596, 108601, 507930, 114716, 311327, 160234,
    144705, 77844, 212342, 219655, 416746, 279046, 81836, 220154, 487653, 434260, 90581, 90690, 147303, 211806, 78529, 78790,
    147014, 156200, 281402, 146355, 414701, 156626, 92187, 212430, 81655, 79317, 493574, 76409, 77852, 80091, 81195, 88163,
    493659, 152298, 156000, 159803, 144860, 348701, 144160, 145073, 81087, 81901, 78310, 80084, 80755, 147141, 82182, 82229,
    213308, 82266, 84595, 86034, 350267, 88201, 88278, 88468, 88521, 285324, 481950, 88844, 88913, 88977, 286912, 286939,
    354429, 485621, 92490, 92561, 356549, 94408, 98344, 414668, 283641, 287678, 299262, 278964, 283286, 152571, 156612, 156624,
    156640, 156648, 156665, 288980, 289072, 289101, 297040, 297058, 299063, 168180, 301160, 499821, 78611, 156543, 151975, 424073,
    151960, 286965, 213103, 143383, 143701, 279401, 285043, 285338, 426095, 78940, 233498, 79697, 84886, 489691, 144546, 219840,
    291092, 491565, 76496, 78192, 81952, 84405, 86413, 86645, 88657, 88827, 417986, 92486, 100387, 428139, 76480, 416101,
    626785, 362582, 364561, 78508, 81411, 147163, 145268, 219654, 172141, 84315, 616583, 485708, 499798, 79842, 213941, 75901,
    81997, 285210, 90442, 211504, 146204, 149834, 155870, 156332, 146656, 349028, 349104, 154302, 351105, 352983, 353026, 353229,
    82016, 84004, 346927, 154191, 154211, 154274, 154498, 156131, 156239, 158110, 158152, 158166, 160091, 223738, 624681, 288769,
    485818, 291180, 497747, 75856, 80275, 81374, 82376, 151874, 152271, 154007, 289218, 147713, 283209, 235536, 144914, 82234,
    434213, 503856, 281413, 283033, 284716, 79114, 215553, 217127, 415969, 221306, 221371, 419859, 422106, 78489, 223586, 280723,
    413847, 485488, 288992, 225602, 487780, 620645, 622628, 426075, 493596, 428079, 428119, 297077, 297078, 299035, 366715, 78053,
    146916, 279504, 84518, 156174, 287246, 156255, 287327, 420293, 146611, 279323, 149899, 151631, 348987, 220019, 287093, 419910,
    485447, 92318, 289094, 420190, 422038, 227334, 229383, 296995, 212967, 80962, 281024, 346598, 151808, 350405, 154104, 219777,
    352815, 90746, 170058, 78387, 82142, 82396, 84206, 88580, 88639, 96365, 102500, 148462, 148469, 76396, 76648, 77881,
    84415, 217089, 219362, 219387, 219837, 220110, 220118, 287104, 221874, 81156, 146763, 150027, 416498, 78194, 144616, 618636,
    94626, 145535, 148397, 148430, 86943, 94650, 291323, 79290, 145306, 80393, 217709, 283607, 219328, 483507, 287099, 288885,
    289180, 290841, 292916, 293025, 294959, 305154, 213842, 150340, 221918, 220142, 146996, 346922, 146815, 279011, 149892, 150208,
    151813, 217579, 217731, 219622, 154147, 154219, 221553, 156062, 223521, 485854, 161949, 360519, 364663, 368745, 79818, 349150,
    484297, 299165, 301119, 416727, 75807, 78438, 154159, 618628, 225682, 358575, 100413, 364759, 143786, 147584, 150020, 153779,
    219844, 221674, 223447, 223580, 223583, 78498, 146850, 279075, 283590, 418784, 350905, 143889, 147253, 151621, 416158, 354568,
    420148, 422239, 422251, 430147, 364654, 626831, 235608, 368700, 434269, 145467, 212787, 215593, 154123, 286876, 155872, 418232,
    289137, 161903, 170037, 283279, 346210, 436240, 86310, 79865, 152063, 223600, 160003, 225705, 227482, 163927, 100446, 362622,
    233507, 168063, 168090, 170003, 628761, 75882, 84156, 144436, 211192, 151988, 157951, 146674, 235521, 235546, 241673, 636935,
    143949, 495679, 301166, 80911, 81002, 84163, 149824, 86318, 152058, 144431, 150049, 172032, 80564, 82253, 86149, 86198,
    90409, 90564, 358480, 358484, 98385, 428063, 102425, 108558, 110604, 112658, 82657, 145236, 150498, 150510, 156477, 156536,
    156539, 174136, 76236, 79640, 86887, 90881, 90897, 78743, 143579, 146213, 280795, 84256, 284693, 288854, 210971, 346841,
    346817, 88562, 351052, 351171, 352360, 352881, 423942, 425991, 98359, 301064, 215651, 352301, 78593, 152178, 349022, 280994,
    213952, 82910, 88307, 144171, 149926, 76513, 283316, 284697, 158168, 485864, 493665, 150246, 154282, 348268, 221943, 225739,
    487893, 279005, 215529, 221557, 80700, 634905, 76637, 146034, 346904, 152084, 155888, 108580, 150117, 156122, 78908, 420062,
    351187, 145878, 149769, 143689, 280973, 151663, 280718, 346400, 282718, 418493, 418494, 287470, 287475, 287479, 422336, 160202,
    419893, 428155, 414658, 618672, 144681, 347134, 145988, 215547, 215740, 88466, 154127, 90221, 90543, 156177, 92339, 92366,
    223474, 158001, 94383, 94425, 620566, 98307, 106549, 630847, 150404, 104456, 79837, 82871, 81857, 147100, 283472, 618661,
    213058, 351205, 221783, 485386, 485425, 354456, 354725, 237652, 77933, 282839, 86398, 144198, 144677, 150329, 144264, 217551,
    284804, 483570, 484321, 618995, 144602, 76281, 144302, 79095, 153656, 153905, 89077, 147408, 416731, 628792, 145566, 213043,
    352557, 221665, 221754, 157854, 237674, 143736, 147993, 501778, 149850, 152207, 211181, 213656, 215082, 215306, 217265, 217364,
    225730, 237679, 76693, 79483, 285609, 287119, 485605, 622619, 430210, 630824, 79619, 79659, 79680, 82764, 82769, 82774,
    82790, 82807, 152144, 86868, 86958, 153662, 223262, 288846, 157939, 227478, 299045, 143447, 145428, 146464, 149910, 281254,
    217279, 348562, 217702, 152246, 154392, 156006, 156095, 221751, 231493, 76454, 217799, 144650, 82941, 214013, 144514, 146246,
    147446, 86035, 152387, 88975, 92669, 421938, 154267, 90344, 144600, 147153, 78915, 213985, 364761, 364781, 78507, 227336,
    233511, 237622, 239647, 241667, 243712, 86964, 154319, 82816, 279448, 349045, 283595, 143896, 212630, 215624, 158141, 422036,
    428124, 166014, 144531, 80790, 147647, 84468, 223599, 294988, 493685, 144414, 78610, 144223, 211764, 145175, 148364, 414174,
    152416, 283489, 283493, 88972, 287492, 287502, 299167, 287059, 147384, 77894, 81164, 86778, 88352, 145037, 90230, 223359,
    491594, 362557, 626774, 499783, 147392, 282989, 235579, 241666, 311302, 144184, 211695, 148092, 149733, 220007, 301183, 156304,
    76130, 350951, 94551, 160027, 285126, 290979, 291186, 292959, 296998, 624658, 80934, 151819, 349130, 153846, 154117, 358515,
    280735, 217530, 348976, 417830, 287323, 419974, 616741, 430185, 299122, 144375, 79631, 82791, 86936, 90873, 90876, 94662,
    98411, 102567, 102572, 145553, 146686, 217248, 350271, 88503, 88825, 94263, 94612, 154150, 618717, 213401, 215733, 152495,
    219213, 415859, 418034, 156294, 616934, 160213, 78936, 211478, 212959, 147367, 148139, 151593, 154070, 155765, 156189, 157909,
    158075, 159806, 159921, 160165, 161811, 168077, 143710, 154512, 158123, 79125, 146130, 622707, 76607, 221454, 77951, 215972,
    222021, 418655, 222138, 227499, 227509, 424120, 424131, 424167, 227579, 233662, 233695, 626911, 626923, 233711, 430321, 436270,
    239665, 436287, 79335, 147608, 149557, 149772, 150405, 217170, 152070, 152381, 284835, 153788, 157770, 159790, 161809, 161825,
    165928, 167972, 170004, 176129, 372749, 243722, 243725, 309269, 309271, 180226, 245763, 146711, 148074, 77984, 80503, 80978,
    84738, 283177, 86634, 88176, 90525, 153600, 284908, 156324, 291009, 163918, 432180, 368649, 434223, 374814, 79384, 88718,
    417882, 213782, 285674, 291019, 98403, 102564, 278532, 280615, 285590, 287355, 233479, 303114, 307227, 307243, 307252, 286731,
    80622, 147369, 147373, 148158, 148206, 84191, 86742, 285093, 88530, 482159, 90254, 90736, 92473, 94473, 96369, 102461,
    349165, 622711, 86581, 287460, 163932, 144076, 417837, 212909, 147503, 213742, 149522, 282857, 151910, 283419, 87026, 284748,
    285661, 285682, 354432, 423977, 424102, 152052, 218081, 159932, 79787, 79824, 96390, 79086, 219761, 286980, 231431, 364567,
    76612, 80153, 280924, 151734, 352744, 628751, 281092, 350916, 352746, 354519, 414603, 418492, 418498, 307261, 213089, 211309,
    217608, 151736, 366593, 366618, 372745, 76034, 483407, 616535, 78741, 145504, 80631, 147857, 150400, 76432, 348353, 157696,
    159958, 82282, 82330, 215099, 84482, 217379, 283146, 217718, 217782, 285239, 88713, 219823, 221739, 288956, 289203, 223726,
    290971, 94365, 424035, 299156, 628743, 237626, 80443, 212443, 86382, 416198, 422258, 145785, 295023, 147622, 284739, 348400,
    616476, 616507, 354396, 94589, 491563, 366685, 147926, 239642, 147604, 82666, 148247, 220116, 146528, 148088, 145077, 78621,
    146811, 346933, 154482, 154622, 81373, 347086, 285475, 154410, 154530, 154561, 353099, 213992, 432169, 497749, 80207, 81309,
    81540, 82473, 82572, 82576, 215158, 84526, 217793, 219327, 219494, 88481, 219596, 88807, 90335, 90377, 90489, 92469,
    291077, 108562, 81248, 481545, 432188, 75888, 77974, 84829, 90382, 78748, 79815, 79862, 213884, 356807, 356818, 619007,
    360560, 81902, 278714, 280838, 348469, 352686, 354469, 356663, 299138, 84009, 86021, 88925, 89015, 89044, 485480, 489640,
    499780, 77902, 144028, 79845, 79863, 213196, 82304, 86876, 86894, 619002, 78496, 352915, 211808, 303117, 108559, 636948,
    147199, 147497, 348801, 420305, 620712, 491579, 215548, 505866, 78904, 143954, 144782, 150333, 347049, 155772, 79115, 414456,
    297009, 346530, 423991, 176166, 243721, 279537, 144158, 147421, 279328, 348411, 81360, 90164, 104548, 144834, 215303, 217235,
    217580, 219224, 143836, 144221, 79552, 144295, 218106, 150485, 154567, 160109, 163934, 163948, 78829, 79867, 82021, 414137,
    489473, 87020, 145613, 78247, 80537, 81082, 81278, 84186, 215450, 92382, 163848, 279232, 291247, 497699, 497739, 147602,
    147643, 213750, 215935, 150401, 285002, 220148, 79100, 82338, 86361, 88405, 88739, 286831, 92200, 290898, 301101, 217300,
    145289, 348428, 485847, 79699, 413981, 96435, 286918, 149621, 350442, 157898, 172083, 145258, 79147, 146596, 147742, 347104,
    154192, 154330, 353150, 143819, 147159, 151666, 221241, 76228, 483996, 80787, 82167, 104448, 146696, 215673, 143864, 76550,
    144334, 79461, 79510, 145375, 143602, 80508, 147025, 157700, 215212, 217310, 217788, 217892, 217910, 285480, 351053, 354418,
    354596, 223560, 289115, 356399, 618611, 356553, 291165, 291222, 620621, 299030, 299097, 299130, 301103, 285268, 215372, 146087,
    147769, 215863, 289209, 432170, 434219, 368713, 436224, 438276, 374801, 440347, 81320, 100393, 497706, 499755, 434249, 501760,
    503812, 440337, 505883, 84828, 86841, 90779, 90781, 80175, 353097, 353149, 353164, 353197, 158138, 482282, 211263, 215310,
    150167, 88743, 221722, 90684, 92255, 157812, 223655, 485873, 225446, 225447, 102512, 235604, 146831, 145545, 80296, 147228,
    213322, 346339, 153724, 286761, 419975, 289100, 223711, 303155, 144700, 145838, 80791, 153646, 143845, 150086, 150098, 284885,
    287100, 289123, 154583, 213918, 110638, 414608, 82698, 78653, 144422, 75916, 76587, 145668, 80420, 81761, 81904, 84026,
    84687, 215889, 347038, 217893, 285329, 229442, 144097, 145509, 145603, 81237, 212883, 149730, 280839, 84390, 215542, 84638,
    152127, 86791, 284760, 350300, 88274, 154102, 154499, 285621, 221218, 92186, 354406, 223466, 92617, 92636, 289269, 159746,
    422179, 299058, 364656, 616799, 152345, 176174, 146576, 88841, 285547, 482255, 292932, 231491, 148188, 150415, 219913, 154444,
    155752, 156273, 94236, 144191, 150324, 76335, 151676, 158037, 280890, 229409, 213128, 279403, 280630, 282737, 282939, 283508,
    283511, 287112, 287416, 287437, 287443, 287458, 422180, 80950, 287555, 299195, 507929, 414601, 418522, 418581, 144928, 148196,
    143611, 145773, 146765, 146905, 147610, 157977, 364634, 90563, 92226, 79598, 284924, 145039, 212819, 349134, 618971, 626931,
    79133, 146140, 146163, 81344, 81426, 213569, 152332, 358505, 214004, 483678, 485404, 485535, 616946, 489561, 616854, 78550,
    149692, 489510, 301115, 307202, 376838, 79485, 81847, 81881, 279200, 148237, 144592, 211145, 213510, 146953, 147574, 147868,
    154533, 88446, 90139, 90210, 81175, 84260, 86497, 616633, 106509, 347061, 353136, 287722, 78266, 81454, 148090, 156120,
    288994, 372759, 156591, 353051, 353077, 147557, 146969, 151661, 153670, 88793, 223428, 172112, 146261, 80943, 76685, 79507,
    278567, 436241, 78230, 147045, 86425, 86463, 88108, 90212, 284899, 290997, 143576, 144098, 146309, 146751, 81289, 278872,
    147949, 430150, 628746, 305158, 80785, 149602, 217209, 98313, 239632, 153663, 75874, 78846, 291002, 356578, 143543, 80629,
    81957, 82631, 86178, 283198, 86811, 88137, 88440, 88497, 285605, 89053, 89074, 90126, 419877, 487797, 225695, 211158,
    487932, 303140, 636946, 282643, 285583, 78819, 282871, 219829, 143734, 420274, 358534, 79009, 78253, 80834, 215541, 88755,
    149717, 86637, 88256, 165979, 81067, 81310, 81420, 81707, 81776, 84725, 86620, 88583, 153969, 299018, 634902, 636933,
    80015, 89022, 349068, 484073, 221944, 353035, 487864, 225722, 225748, 491616, 491625, 86215, 350723, 168100, 79800, 148159,
    229391, 237572, 239637, 214009, 148475, 279548, 152183, 483840, 223382, 356430, 618727, 225562, 620641, 628784, 158173, 227372,
    108547, 176154, 112651, 148081, 213631, 219408, 92323, 92340, 360470, 102478, 106531, 144347, 303125, 303148, 150502, 213317,
    217441, 153742, 219451, 144688, 632850, 484038, 146016, 147048, 86544, 284869, 88568, 287279, 287691, 418778, 143601, 149846,
    154463, 148062, 279492, 350808, 154326, 146386, 81255, 280794, 481607, 351103, 225337, 79951, 221429, 88677, 88803, 90640,
    211820, 279432, 152035, 616815, 78635, 146799, 347077, 147590, 279369, 418520, 77869, 80305, 82049, 83989, 86408, 88143,
    90611, 92359, 92611, 213928, 218085, 353093, 156576, 222142, 484312, 487925, 487931, 358571, 358581, 161978, 161987, 491633,
    364734, 168160, 630909, 174140, 143865, 215260, 219532, 421992, 79206, 79359, 145329, 211978, 88350, 223516, 489560, 495718,
    80303, 82836, 79882, 86054, 88848, 301130, 156003, 156113, 483960, 145984, 211586, 213242, 223280, 151959, 88236, 416549,
    159828, 622623, 491589, 100453, 432172, 176146, 419851, 279081, 626708, 79264, 79941, 82628, 84331, 348372, 86644, 76214,
    144447, 150050, 84662, 86341, 152087, 88920, 90457, 92370, 96360, 434275, 144877, 80071, 82735, 84094, 86358, 416188,
    419959, 362576, 366696, 79055, 215256, 287046, 213739, 76509, 84104, 280956, 94311, 79805, 78812, 144385, 212152, 215914,
    301173, 180236, 281344, 414376, 154328, 620652, 76552, 82545, 86840, 347032, 76488, 213326, 82938, 353005, 624674, 104463,
    368653, 499731, 174095, 350422, 350665, 350988, 351121, 420102, 354634, 354705, 618553, 356552, 422248, 168165, 618982, 416546,
    356466, 291204, 100425, 303129, 305162, 307217, 241712, 146116, 146425, 82826, 90812, 90818, 90841, 145107, 75900, 78372,
    84597, 143499, 284996, 285501, 287331, 155683, 158078, 145762, 212304, 217355, 219516, 219653, 154415, 424051, 212248, 148027,
    283048, 348951, 94580, 297023, 79005, 215295, 219447, 218111, 290994, 292898, 305164, 215527, 215893, 215929, 217397, 219857,
    220130, 221614, 156655, 223397, 86301, 88461, 286785, 92335, 356373, 100442, 217771, 165892, 78138, 145144, 86580, 283545,
    283546, 287511, 79799, 484196, 484208, 96429, 293080, 299227, 84727, 217833, 98335, 98343, 98379, 299023, 102424, 364577,
    299064, 235577, 114689, 146472, 78209, 145748, 146275, 152100, 156096, 160033, 213942, 218071, 356513, 147695, 153839, 158038,
    413946, 78932, 84799, 221929, 223238, 213627, 146694, 279120, 414459, 211863, 81433, 213943, 215637, 78633, 145857, 81684,
    414275, 287332, 92516, 421953, 94450, 172120, 144096, 82390, 174106, 75858, 75854, 78110, 81234, 82370, 282874, 92336,
    144634, 212383, 147457, 213046, 278686, 149616, 215494, 151878, 284803, 284920, 350817, 287090, 354357, 223723, 362619, 79351,
    76735, 154577, 158203, 211814, 148402, 215421, 112648, 374809, 145821, 144589, 168117, 168129, 168137, 168138, 168171, 168172,
    368738, 174135, 174138, 174139, 616602, 80802, 86176, 86492, 92612, 102455, 299103, 104552, 147291, 354612, 356703, 356715,
    618861, 364611, 147820, 213812, 351223, 78171, 144610, 79282, 86177, 144230, 143743, 81941, 280737, 281247, 148175, 215945,
    104565, 368646, 499750, 374786, 79653, 210954, 353039, 356820, 360540, 145608, 213507, 489490, 154048, 88309, 628757, 76355,
    157818, 368688, 503837, 79639, 349013, 79694, 84905, 283168, 350228, 350573, 90421, 352899, 289088, 289106, 84444, 90183,
    287102, 485675, 487499, 618596, 376843, 217904, 281419, 212230, 285054, 286818, 150047, 152076, 153630, 89067, 156172, 160059,
    163858, 172058, 172097, 360457, 350266, 286960, 78097, 86204, 92537, 102477, 299124, 76786, 78783, 82727, 212056, 215102,
    280770, 418780, 79082, 223239, 76351, 78178, 80573, 76608, 110648, 279125, 283079, 285220, 78258, 212205, 622651, 75887,
    346745, 86415, 88533, 147812, 213939, 217700, 78455, 145989, 150148, 150203, 84806, 346609, 618535, 428054, 301089, 213011,
    278870, 215129, 217919, 284963, 495781, 151732, 80985, 213971, 87003, 218099, 487910, 212918, 418174, 144007, 80338, 82928,
    143605, 79187, 348193, 285491, 156267, 219147, 221606, 418717, 223494, 225640, 239677, 76795, 143365, 145813, 146024, 147667,
    281253, 282786, 86317, 151965, 152043, 283432, 154217, 362563, 432194, 145353, 350933, 217335, 354510, 211375, 227516, 227571,
    144030, 145371, 212098, 151979, 364551, 243727, 100419, 78288, 147108, 348925, 223440, 354574, 356542, 75935, 285366, 94457,
    424057, 487628, 489541, 79587, 146557, 149660, 84814, 86843, 146529, 215706, 291022, 423999, 227363, 84200, 626858, 106608,
    76619, 353139, 616914, 358618, 364760, 165952, 484326, 145885, 354404, 485762, 79173, 288878, 237665, 146606, 78146, 145066,
    84520, 88338, 92369, 92659, 94293, 96382, 100422, 77824, 79956, 148005, 148083, 153890, 154265, 88959, 100363, 497732,
    76494, 145764, 348540, 152283, 78800, 78922, 81198, 283448, 90209, 485882, 158133, 160151, 211653, 211927, 79059, 88275,
    286953, 160037, 215755, 416610, 225544, 225683, 146112, 147023, 497765, 146547, 80111, 145997, 149924, 151821, 152128, 88281,
    485769, 94373, 489638, 219782, 90192, 430118, 303193, 145893, 80645, 81207, 80723, 149854, 281229, 151985, 217806, 154345,
    220082, 92475, 485799, 159953, 160110, 487805, 86574, 76155, 150411, 219713, 156165, 626784, 143816, 145293, 287693, 215928,
    284815, 145364, 213848, 349160, 215341, 225785, 211557, 213496, 346948, 146965, 147092, 418497, 418504, 418506, 484046, 79797,
    213240, 296991, 144897, 348995, 291310, 90905, 147293, 78144, 81064, 81793, 84023, 149890, 86289, 86680, 86817, 153803,
    154129, 89060, 89061, 354312, 92317, 626795, 497713, 84780, 80974, 147496, 350544, 368667, 215435, 217167, 348705, 145642,
    356747, 148044, 414101, 287206, 630880, 144741, 280908, 213485, 152336, 350807, 616505, 288907, 157887, 489570, 297055, 364606,
    299074, 364631, 299096, 299131, 76641, 148080, 622684, 149738, 223256, 79554, 144296, 82874, 284969, 301132, 279259, 82653,
    213749, 215574, 215884, 348715, 217669, 284879, 154586, 221187, 156330, 225390, 489626, 626694, 235622, 157727, 145771, 151722,
    352721, 145674, 282909, 350288, 285069, 288943, 354591, 360509, 297050, 366638, 366690, 368725, 79874, 80030, 82243, 82399,
    622643, 76649, 79130, 353156, 146113, 78564, 84790, 86081, 86504, 86513, 219594, 350986, 483603, 354366, 420081, 616776,
    92569, 424076, 280603, 81139, 82544, 225539, 291241, 423948, 229463, 165982, 299043, 233599, 233626, 235539, 88522, 90127,
    90544, 281490, 281505, 281512, 281561, 281562, 287517, 287523, 287526, 287539, 287547, 484163, 287599, 484220, 287638, 287639,
    287651, 287672, 489659, 293057, 293062, 293069, 293070, 293078, 293084, 293087, 293091, 293097, 293099, 293100, 489709, 293111,
    293119, 299192, 299194, 495803, 299197, 299215, 299222, 495833, 495853, 305190, 305199, 305200, 305202, 305208, 305214, 305215,
    311321, 311322, 311325, 284724, 356360, 360475, 368680, 144724, 144923, 287026, 80143, 81672, 84668, 88551, 88888, 89016,
    286829, 354397, 420082, 92438, 92499, 92642, 94632, 106595, 143909, 146004, 81163, 212272, 82634, 150116, 153904, 501770,
    146855, 351186, 434296, 353078, 353121, 353194, 162022, 79650, 79595, 145711, 147431, 148381, 279474, 152507, 283587, 283603,
    283609, 152544, 283630, 422364, 291319, 78772, 80163, 90759, 76011, 487447, 212268, 287698, 299257, 288823, 348793, 288941,
    360573, 364792, 144155, 360539, 362583, 278614, 278953, 285560, 285576, 289016, 279002, 88258, 350863, 90174, 354425, 223392,
    80842, 76189, 279544, 221380, 419904, 92230, 422318, 75841, 78240, 81592, 81796, 281204, 86125, 88078, 88134, 88633,
    88799, 157892, 157933, 96409, 106576, 149510, 143984, 80937, 156031, 622652, 430161, 79755, 102646, 102647, 84294, 223366,
    420025, 487433, 618547, 487488, 422262, 364573, 168012, 233577, 299114, 366680, 76114, 81128, 82792, 213894, 213902, 217967,
    221905, 221910, 280789, 297051, 79233, 86225, 78442, 94563, 489629, 104488, 497777, 147519, 144858, 145220, 81027, 88797,
    90675, 493639, 616684, 487794, 81315, 78880, 160028, 280986, 414044, 154427, 155704, 159778, 161940, 376857, 147295, 212675,
    358410, 144108, 370716, 82240, 414016, 624754, 239629, 110598, 82048, 346125, 88243, 286937, 483615, 287282, 288909, 288965,
    289255, 78860, 82209, 82559, 279388, 283480, 283483, 283534, 287482, 211600, 413809, 143842, 213499, 145775, 289241, 499722,
    503835, 503851, 503860, 146350, 79848, 350208, 360526, 144429, 212024, 219261, 223379, 346829, 86722, 303201, 110636, 503858,
    79931, 104501, 145499, 81202, 281140, 154254, 422131, 76697, 221649, 76463, 283477, 78168, 75809, 78294, 88177, 78627,
    78969, 481405, 285516, 286824, 287345, 151640, 154054, 154096, 159907, 160114, 161887, 165926, 347063, 353065, 353106, 353160,
    358586, 370748, 145605, 217584, 284928, 219404, 88535, 221542, 287286, 223616, 291035, 422263, 160167, 145512, 278715, 147975,
    88402, 81305, 215393, 220129, 221812, 422051, 424031, 428070, 233551, 86863, 350724, 350944, 220048, 287276, 352873, 354393,
    223659, 356464, 94653, 358509, 489582, 296986, 362618, 150261, 76001, 146644, 144975, 76555, 79479, 76563, 78088, 215324,
    219471, 220022, 165905, 362540, 78331, 80323, 350274, 219959, 219982, 221666, 90602, 223365, 92375, 223509, 223534, 92511,
    223696, 289240, 225485, 225501, 290919, 227486, 79428, 414342, 152225, 356480, 86714, 282811, 414123, 283374, 221340, 223601,
    80390, 81790, 485503, 79401, 150481, 150487, 348943, 87039, 156566, 174118, 174142, 79767, 78537, 211397, 217207, 225625,
    154558, 76554, 78206, 281155, 84637, 487529, 283265, 144731, 213929, 78566, 144719, 349096, 287718, 144498, 144798, 144790,
    346264, 350649, 92639, 84085, 284906, 92362, 106547, 81467, 148216, 353056, 358599, 364753, 370740, 86867, 221947, 211665,
    350516, 366687, 417847, 290817, 84417, 144043, 81251, 352503, 78699, 82895, 348707, 76251, 81652, 84198, 88361, 88622,
    94530, 104524, 212946, 414617, 81114, 217559, 285511, 286995, 82467, 149576, 282805, 348857, 233517, 79430, 147854, 215214,
    150185, 154232, 219977, 354361, 354579, 420236, 618780, 618952, 227426, 364637, 634940, 376853, 81486, 147693, 151690, 154076,
    356443, 231471, 235635, 145679, 80848, 278798, 84114, 86222, 86612, 154087, 154552, 92378, 94361, 150008, 432129, 432154,
    438281, 144840, 211925, 299024, 299177, 78887, 620697, 144476, 150137, 154229, 80159, 82275, 84261, 84711, 287351, 217759,
    215265, 217165, 356667, 368705, 213910, 84483, 217997, 218023, 89049, 90264, 90555, 92552, 159855, 159969, 159997, 160048,
    160083, 160161, 167969, 151633, 144449, 153673, 153976, 154033, 144014, 280808, 283111, 86523, 295005, 295022, 295032, 426108,
    281220, 281275, 350443, 352824, 616456, 420107, 485647, 420163, 430159, 301109, 151843, 79737, 79750, 145712, 150150, 150366,
    84907, 84920, 84966, 151814, 154296, 155856, 90946, 91047, 91063, 161871, 489572, 96428, 96430, 489686, 489721, 495802,
    102598, 495818, 170092, 368650, 372763, 372779, 372788, 148370, 144598, 77925, 79487, 80004, 146005, 81026, 146584, 81287,
    81379, 146935, 483541, 90605, 96361, 495767, 499781, 503822, 112655, 145744, 215409, 358467, 144920, 157970, 430109, 364650,
    432216, 78869, 213291, 149767, 151699, 152044, 219311, 350914, 220026, 368736, 353148, 358587, 162027, 168120, 283008, 218048,
    484317, 80486, 279068, 420068, 426037, 630865, 92241, 487593, 144933, 145910, 147401, 148140, 82662, 82737, 84853, 288795,
    159955, 432240, 144179, 624701, 219674, 81169, 279176, 148053, 154148, 624692, 440333, 78549, 212859, 150499, 143997, 350948,
    211496, 149817, 283223, 76172, 78325, 79357, 223352, 281082, 285081, 288924, 487731, 297020, 145435, 82760, 233632, 233637,
    237688, 78828, 86264, 144336, 78859, 86613, 428045, 76031, 146324, 284991, 352877, 92445, 354787, 147190, 483561, 282862,
    288801, 84703, 86438, 88601, 144366, 150336, 144683, 82093, 215202, 144885, 485635, 495860, 79743, 79745, 84891, 84898,
    84909, 84955, 84956, 84960, 90914, 90927, 287544, 90997, 484217, 91005, 91010, 91031, 287646, 91053, 91059, 96439,
    96445, 96453, 96458, 96489, 102619, 495845, 102633, 501802, 78518, 481294, 489625, 352919, 616731, 154331, 485396, 489483,
    626696, 366709, 630790, 636930, 245772, 222140, 222204, 143647, 147294, 151731, 285465, 422170, 487778, 499775, 144751, 81030,
    430224, 94486, 144312, 144898, 281406, 154351, 483975, 483977, 352961, 352968, 352970, 90851, 221965, 487857, 94656, 94675,
    364704, 152573, 434302, 217130, 353067, 353076, 78622, 143507, 146294, 147760, 150257, 419852, 432205, 78243, 80006, 80177,
    81209, 149635, 154606, 80918, 81383, 146989, 82533, 88069, 88224, 154623, 90245, 90607, 90698, 90772, 90780, 94234,
    94593, 96374, 100372, 102498, 102550, 213813, 217283, 421889, 79652, 78568, 80524, 279406, 153654, 150099, 217259, 86251,
    88175, 284870, 88304, 153895, 90178, 286804, 90381, 90801, 288896, 158207, 170042, 146459, 215765, 348593, 219831, 485697,
    356561, 422201, 227416, 491587, 233554, 78594, 347045, 364766, 76120, 79486, 79920, 82691, 92264, 94313, 94392, 96424,
    104540, 106564, 144934, 219563, 362553, 147502, 77856, 80257, 82014, 84509, 94516, 78957, 79273, 151846, 352759, 76732,
    416754, 418342, 79416, 416349, 420201, 145151, 213288, 295006, 295020, 297043, 143766, 151961, 151999, 153644, 155748, 152456,
    414628, 156357, 418507, 156416, 418569, 156434, 422360, 426087, 438330, 76113, 80626, 84154, 150004, 86798, 88577, 88641,
    88674, 219880, 219939, 88953, 89025, 90466, 90474, 90573, 287272, 90707, 92239, 289135, 96377, 360524, 499779, 414059,
    415899, 78816, 217794, 147539, 149532, 149533, 151939, 350576, 154233, 154340, 354306, 223300, 157931, 157950, 225586, 161935,
    161938, 622601, 426033, 366620, 80939, 82124, 348851, 152395, 221536, 223292, 225339, 79336, 416401, 146403, 151923, 491535,
    499716, 501781, 90259, 76257, 143989, 213657, 150314, 151882, 156300, 215668, 351013, 78724, 225566, 144132, 84808, 86819,
    219519, 215915, 144586, 144639, 80807, 483953, 76291, 146856, 84099, 165929, 499785, 434270, 505873, 77926, 79944, 84101,
    84652, 153769, 154022, 154470, 155862, 158085, 96398, 104515, 76022, 82109, 82152, 82281, 84323, 84480, 86469, 86698,
    88480, 88923, 213844, 213886, 364714, 350475, 286746, 79992, 213924, 353248, 213641, 280826, 218052, 222167, 222186, 360488,
    144079, 144853, 170076, 215159, 484299, 353251, 484327, 356835, 618994, 487428, 80695, 151785, 153676, 88834, 90438, 143716,
    147813, 414399, 351199, 221503, 352917, 223301, 160170, 231425, 231445, 231448, 212279, 616493, 144058, 78953, 352467, 144208,
    279453, 283579, 152537, 79184, 152308, 88960, 144501, 221983, 418600, 418611, 222006, 222049, 222059, 222122, 222131, 424109,
    227511, 424146, 620786, 430260, 233655, 430264, 430272, 233666, 233667, 233678, 233684, 233704, 626922, 632871, 436265, 245784,
    76147, 80435, 147366, 160119, 622689, 630907, 144888, 283562, 76089, 278632, 147727, 82450, 281003, 289084, 303190, 76468,
    366633, 432213, 76390, 215988, 215989, 216008, 216049, 219259, 418586, 222030, 222064, 418729, 620762, 360516, 233660, 233699,
    626916, 430314, 632876, 639004, 144002, 78849, 76443, 426030, 77858, 80512, 80961, 82641, 215583, 84611, 217612, 219166,
    154603, 221708, 225595, 229394, 237594, 237633, 106594, 146076, 149901, 88333, 156287, 420101, 491644, 626943, 79076, 79296,
    279116, 88433, 245768, 144216, 82739, 78835, 353242, 145034, 283552, 154215, 418548, 489531, 102555, 150143, 84792, 156331,
    630848, 79806, 348786, 356503, 144995, 414073, 350841, 483535, 419908, 422194, 358546, 622641, 76329, 285553, 416748, 422058,
    423998, 491547, 499752, 636944, 147192, 147780, 483817, 485672, 491526, 75872, 82935, 86487, 79271, 145270, 150417, 150482,
    156460, 156479, 156481, 358651, 370737, 145228, 150450, 150474, 156487, 156573, 162038, 180247, 180254, 80128, 80940, 81260,
    151747, 86515, 88292, 416599, 102506, 106583, 144099, 79092, 81533, 278877, 285622, 220128, 286814, 287364, 294932, 295002,
    301111, 497783, 76665, 80992, 143450, 283346, 143775, 144491, 80025, 146457, 149946, 281172, 281450, 219762, 233586, 483835,
    497775, 303126, 178192, 217575, 307222, 309253, 309277, 82402, 75943, 78234, 84745, 151679, 279236, 215947, 144695, 487426,
    428101, 80666, 280905, 211797, 350813, 354665, 415758, 485572, 424089, 81854, 211446, 215927, 150410, 152560, 152564, 287395,
    156609, 156656, 225491, 160245, 217544, 156188, 215766, 217763, 219776, 154304, 225556, 225691, 426029, 143801, 90625, 227449,
    79819, 79825, 626938, 80480, 81512, 84137, 84525, 88840, 88926, 88939, 88974, 285617, 285647, 354643, 84695, 86091,
    156381, 156414, 92217, 211520, 102416, 78643, 221717, 147027, 282949, 356723, 292968, 215374, 150089, 145229, 150424, 493646,
    284683, 287142, 289030, 291176, 79089, 81140, 148220, 84936, 484158, 96448, 489701, 168021, 495805, 495823, 501807, 284985,
    144910, 352753, 354360, 79397, 212889, 225700, 285277, 289129, 212943, 143910, 147191, 420146, 292871, 144004, 145830, 219149,
    88761, 368760, 146836, 147275, 346704, 350888, 351050, 418271, 352806, 354621, 354674, 354773, 354775, 487597, 356749, 161879,
    364613, 364689, 212125, 149734, 84228, 151845, 283062, 283608, 153897, 154206, 223456, 159789, 160066, 428091, 364683, 170060,
    147545, 154228, 161916, 279511, 155919, 356558, 624700, 165997, 144025, 81770, 84837, 80331, 82526, 88790, 487440, 428080,
    144006, 144117, 418414, 353146, 353178, 485724, 628858, 79220, 358632, 77946, 146843, 217384, 352526, 84860, 221382, 493636,
    283414, 152500, 79455, 81839, 144452, 86623, 88487, 88506, 94407, 96346, 148405, 287696, 76413, 286935, 622632, 81853,
    150092, 151659, 349052, 156084, 352975, 221941, 438309, 212704, 215770, 351102, 352879, 487601, 358522, 628847, 150242, 364785,
    144273, 167962, 75993, 483491, 146197, 219587, 90807, 90816, 106609, 144176, 78213, 80664, 82067, 86837, 89085, 483355,
    81612, 102448, 147386, 278571, 148132, 630811, 280796, 213622, 217116, 146096, 147491, 150015, 152319, 170102, 215670, 221360,
    223457, 225572, 489500, 100467, 499790, 239640, 76695, 143679, 145810, 146662, 154532, 144041, 81268, 144406, 280741, 283099,
    482247, 78612, 84796, 78665, 145250, 287144, 162002, 86148, 88255, 88422, 88524, 286729, 221477, 160005, 78718, 79231,
    278786, 280784, 280935, 151627, 157753, 212878, 220039, 155905, 223302, 290871, 78356, 211333, 213506, 290899, 158028, 630893,
    158136, 149536, 152237, 160126, 618608, 143725, 215224, 348490, 352848, 358471, 491593, 285573, 98347, 279514, 430206, 497757,
    79067, 156220, 218069, 222171, 145667, 80333, 146124, 82695, 86451, 217751, 221381, 90404, 418173, 616557, 94308, 428110,
    430098, 102470, 432182, 497734, 436229, 283614, 418761, 287717, 144950, 153633, 284772, 174081, 150351, 152351, 168038, 104522,
    76783, 89073, 155769, 154111, 79120, 167946, 144069, 144350, 145416, 147662, 213267, 215246, 346415, 281168, 285352, 285514,
    287270, 289085, 289138, 289237, 289239, 422061, 291213, 299077, 299153, 211727, 145128, 145285, 347096, 353068, 353087, 353089,
    143813, 78450, 285215, 94639, 153746, 350477, 350575, 156171, 157935, 235647, 215946, 75941, 84314, 86671, 80104, 144012,
    78784, 78311, 144250, 80798, 82183, 346840, 88564, 219771, 148266, 146353, 346816, 283367, 154344, 154403, 90369, 221736,
    92556, 225335, 358482, 426050, 434243, 81072, 147830, 86476, 143540, 79576, 146545, 147876, 82625, 213740, 148207, 150036,
    151669, 151982, 153793, 153975, 154405, 418265, 418453, 420245, 144317, 155657, 90405, 78752, 79966, 88393, 89029, 90353,
    94562, 96278, 106559, 78661, 79318, 156076, 144806, 278947, 148476, 78401, 81406, 217440, 154371, 144803, 78709, 78833,
    84777, 148203, 75776, 78360, 80476, 92605, 362588, 100478, 626718, 219686, 143399, 145479, 145916, 280710, 280743, 280851,
    282626, 282651, 282957, 151957, 285394, 287307, 159984, 292865, 349044, 352952, 352973, 352979, 352994, 88711, 90595, 156254,
    158162, 160002, 346929, 143477, 346330, 86664, 96302, 96416, 221842, 497681, 213974, 416147, 430163, 212996, 220054, 221819,
    620628, 167943, 237578, 241691, 241707, 241716, 147867, 151937, 348564, 153754, 153822, 153924, 154429, 156259, 78874, 78011,
    79911, 82300, 82462, 215133, 219610, 417904, 420067, 420246, 94325, 422144, 430138, 108572, 632860, 174110, 438272, 176181,
    215663, 212067, 280997, 147488, 215322, 149941, 151949, 152181, 154193, 154363, 485593, 158022, 165923, 432237, 213679, 430335,
    283022, 284989, 287398, 144240, 149755, 150055, 151612, 155988, 156157, 356596, 78434, 144760, 346485, 351051, 151998, 285369,
    156052, 626726, 348649, 146338, 152028, 352647, 354436, 158148, 167991, 364639, 170088, 82172, 84219, 84329, 86076, 88112,
    88458, 286916, 90452, 94442, 152106, 155784, 170093, 432136, 91078, 91106, 353255, 211888, 213336, 151641, 285237, 90571,
    94224, 356452, 225563, 624663, 430088, 364614, 497680, 170101, 434182, 239622, 440322, 144369, 414122, 483712, 80926, 350478,
    278788, 221286, 352411, 287175, 215806, 287149, 287228, 223515, 356475, 225466, 356544, 291042, 227349, 358427, 358475, 294969,
    364713, 235580, 145260, 81629, 154243, 220120, 418221, 487547, 487616, 489499, 162037, 81376, 84460, 92622, 94423, 143886,
    283586, 291315, 227470, 235587, 143380, 79664, 81687, 495642, 78573, 84646, 418406, 215379, 217190, 217450, 219285, 223648,
    225490, 350222, 420077, 358553, 147756, 278880, 280825, 418460, 79467, 352379, 233645, 213158, 350275, 84082, 80529, 352957,
    94666, 491627, 149580, 281011, 283245, 284864, 154592, 156023, 288987, 145130, 75865, 147309, 81787, 82283, 148039, 84153,
    84464, 86080, 86286, 86419, 86545, 86606, 219204, 88204, 88924, 352786, 287713, 92161, 92582, 225404, 94549, 94572,
    100445, 102513, 301139, 104545, 104561, 144481, 211388, 146421, 278668, 280636, 153741, 299118, 303164, 152529, 414680, 146697,
    78200, 79771, 350431, 285070, 286799, 419927, 290823, 291053, 143422, 219551, 88816, 418148, 420157, 420210, 430149, 366634,
    368683, 303177, 370688, 372740, 309265, 374811, 144032, 94491, 144886, 148410, 283597, 283618, 353231, 356790, 356796, 487877,
    626853, 145015, 146490, 416038, 418229, 368729, 153791, 153958, 154060, 225541, 491528, 211641, 147409, 82499, 78528, 79338,
    353234, 364793, 279412, 283492, 283568, 287447, 287490, 212915, 145131, 212699, 489488, 501767, 145233, 632833, 212163, 76632,
    280674, 370704, 156463, 80077, 370723, 356396, 82894, 145025, 76653, 290930, 150030, 88707, 77985, 80244, 84565, 80781,
    90748, 618519, 78409, 79340, 81378, 77899, 82336, 84243, 151660, 217439, 94415, 96276, 96281, 81218, 231540, 417802,
    419916, 296990, 626703, 299050, 366643, 434199, 278688, 82678, 280894, 84863, 89076, 348695, 152191, 154370, 155974, 288978,
    616726, 485858, 289185, 219346, 82865, 218101, 353268, 495866, 76404, 148317, 148333, 152440, 418488, 418509, 156370, 418515,
    287452, 418530, 160181, 160183, 160187, 160190, 160196, 160215, 160217, 163942, 311315, 618939, 76217, 291288, 626688, 430237,
    432191, 307258, 212434, 148218, 148268, 150414, 77921, 79844, 79988, 82242, 82608, 82675, 82912, 84516, 86730, 88948,
    92484, 144776, 147920, 149902, 152121, 154573, 155837, 157953, 79704, 79711, 79742, 79748, 84935, 84963, 84983, 90938,
    90961, 90969, 90972, 90998, 91044, 102628, 102631, 102640, 495760, 346624, 84489, 156163, 156264, 158107, 159804, 499787,
    84449, 90477, 626836, 503854, 78453, 148121, 84040, 84778, 347046, 217337, 90764, 358580, 358595, 358631, 370734, 279299,
    153913, 221444, 221856, 418475, 172069, 78894, 148341, 152417, 616487, 420104, 485684, 487775, 487787, 624767, 495683, 430190,
    301144, 434236, 499805, 147312, 154119, 94522, 374785, 78900, 485476, 616834, 145685, 147921, 84049, 149790, 155737, 158008,
    168001, 170073, 76487, 81267, 82310, 82406, 215340, 149973, 150213, 286740, 155743, 287058, 156136, 156160, 288968, 159809,
    94327, 165900, 104465, 104479, 237668, 154039, 159799, 160148, 78580, 84488, 347051, 353073, 353111, 358572, 152459, 163951,
    217177, 90494, 157830, 487482, 421988, 168041, 143656, 219538, 219675, 219743, 219842, 88801, 220008, 220070, 220081, 220111,
    221495, 221525, 221615, 221655, 287199, 221675, 221753, 221806, 616470, 223410, 223438, 223529, 223615, 420313, 223717, 354797,
    225349, 225455, 225524, 422229, 227436, 424093, 229379, 491555, 491564, 428072, 428125, 493667, 624750, 493686, 233484, 432148,
    432225, 432241, 432249, 237621, 499795, 372768, 80545, 211636, 213269, 219860, 221581, 212162, 213295, 215161, 217500, 223624,
    290927, 291041, 291069, 291120, 291155, 291233, 423939, 146141, 215077, 150087, 151701, 153880, 160080, 144103, 75780, 78362,
    82616, 215447, 219151, 88220, 90237, 418190, 96265, 78787, 348938, 154339, 286944, 158201, 158202, 75896, 80310, 80499,
    81388, 84262, 86046, 86275, 89033, 89059, 102548, 278556, 215314, 217405, 285226, 221318, 287106, 287296, 223381, 292924,
    418404, 630807, 211943, 150000, 151822, 350397, 154460, 418322, 223324, 160085, 165981, 168049, 170081, 170097, 144249, 211804,
    350730, 148283, 155813, 159858, 282797, 285187, 286928, 289056, 145026, 416732, 212287, 485502, 81926, 82746, 78521, 283487,
    152442, 287476, 156406, 160185, 160206, 438333, 278698, 285349, 291018, 278575, 82001, 215055, 84157, 84169, 215460, 217102,
    217357, 217664, 88378, 88542, 88776, 287174, 287283, 92498, 223681, 225333, 227341, 229440, 145348, 79930, 86530, 94517,
    144313, 281409, 156335, 144141, 215096, 376840, 143874, 78437, 79339, 146572, 82143, 148104, 283031, 290900, 630817, 307218,
    76044, 146626, 147759, 349009, 352981, 94664, 364705, 364706, 368757, 110652, 110655, 147084, 418692, 233700, 156159, 147875,
    213919, 348565, 218108, 221670, 222173, 222185, 222198, 225760, 229490, 626935, 215085, 350513, 221256, 484092, 487878, 76673,
    81867, 284813, 90786, 419885, 144712, 146460, 483612, 145160, 217831, 360514, 147215, 143882, 210996, 149687, 281032, 350223,
    616795, 360466, 366683, 368666, 281044, 418741, 424149, 424175, 430284, 436267, 78219, 88839, 143733, 147264, 153991, 280942,
    287220, 80149, 233567, 152116, 144425, 80455, 149645, 84265, 152105, 221678, 144884, 211540, 144669, 624672, 144850, 147464,
    150505, 151552, 152370, 154588, 352949, 78257, 80465, 146503, 86329, 416763, 146759, 148046, 215242, 419912, 157885, 158128,
    158150, 356587, 360485, 493611, 362595, 368723, 281391, 225393, 366653, 144389, 303107, 422399, 281177, 283129, 283356, 217950,
    286859, 489660, 501800, 80628, 280649, 291001, 297079, 167940, 213704, 213693, 279300, 366702, 144664, 79269, 211450, 346923,
    80220, 86503, 86946, 90890, 79385, 80419, 81572, 81705, 88922, 80186, 213729, 280667, 483871, 485714, 76151, 354415,
    143855, 146661, 147001, 84505, 86181, 90331, 155890, 90673, 146221, 148049, 211751, 217331, 348735, 217849, 350532, 350710,
    219987, 352867, 354352, 485644, 485718, 291063, 291193, 358532, 430123, 80840, 282908, 415744, 154178, 92334, 421890, 422306,
    292958, 426062, 102452, 372774, 440329, 636958, 288920, 80133, 288891, 78473, 78173, 81124, 82032, 86335, 88901, 92312,
    94366, 145223, 145431, 147483, 148055, 217852, 152397, 152414, 283548, 219235, 221304, 223271, 231551, 352540, 144238, 81444,
    75947, 78488, 88587, 96367, 146795, 154355, 213764, 353104, 145225, 150419, 78913, 84102, 86018, 86043, 86349, 88786,
    96257, 147339, 78751, 90318, 90463, 484039, 353042, 98319, 98338, 102403, 167955, 104455, 628775, 106500, 108565, 241686,
    243717, 243741, 76515, 279446, 283559, 418585, 150288, 416553, 144475, 146959, 153961, 155867, 149844, 219467, 287169, 616754,
    487845, 495757, 290951, 144269, 278813, 279370, 215440, 152460, 287130, 223605, 487721, 291256, 620702, 295008, 295017, 297029,
    144136, 80204, 84787, 354496, 157791, 78131, 77917, 81294, 82593, 80233, 81991, 279262, 346460, 418463, 160030, 280812,
    282842, 80101, 81979, 84158, 84225, 86394, 90303, 351227, 420173, 78374, 160177, 144625, 76124, 78346, 82244, 213388,
    90154, 418281, 420136, 425990, 278934, 213968, 218077, 285027, 286895, 346520, 288953, 356361, 487475, 356416, 422231, 144685,
    347099, 76372, 280580, 283247, 287224, 419883, 485673, 422018, 79791, 211903, 217249, 153882, 225565, 225651, 229436, 495620,
    278965, 285586, 416092, 294944, 362537, 497666, 432132, 630878, 505878, 79564, 145906, 145985, 80971, 215157, 153705, 481718,
    416231, 155912, 223434, 297024, 237619, 415907, 374794, 219927, 352911, 143423, 144659, 146328, 150387, 348189, 148316, 152408,
    152411, 152462, 156410, 418566, 626707, 353168, 145724, 213702, 84119, 350285, 221869, 289150, 215526, 217546, 219509, 352836,
    350349, 144169, 81777, 414129, 416489, 422097, 487737, 94531, 77889, 145958, 80811, 81044, 81543, 147663, 149750, 215449,
    149988, 84508, 150266, 281417, 151855, 152095, 86712, 88329, 154075, 88796, 88992, 154529, 416680, 89001, 155874, 155980,
    90511, 156223, 157890, 92598, 356433, 422145, 294965, 170090, 108563, 79521, 80094, 82261, 283248, 145120, 280914, 219371,
    351197, 213165, 84451, 76462, 149879, 155921, 156231, 78447, 145571, 147775, 149906, 151576, 152233, 153800, 154167, 155901,
    156058, 159963, 77968, 211071, 80820, 80914, 81729, 82155, 82381, 215140, 84105, 149778, 84421, 84498, 86237, 151869,
    86370, 155782, 156121, 157845, 158101, 100429, 104547, 418308, 145449, 147861, 148303, 148309, 148346, 215756, 152441, 152475,
    283561, 153710, 153960, 350752, 287431, 156390, 418538, 156432, 422347, 422351, 161829, 163924, 430256, 144354, 79579, 82719,
    149977, 283444, 285676, 143472, 147251, 90336, 287032, 352696, 287408, 420188, 161981, 81273, 78562, 288936, 174091, 112652,
    287156, 90768, 144330, 75879, 80793, 82500, 82513, 82550, 86044, 225363, 279192, 280842, 282801, 282900, 283631, 418803,
    144159, 76208, 144225, 352588, 152142, 144682, 81747, 493628, 144540, 80050, 352585, 145338, 144163, 352257, 352479, 79674,
    213898, 349065, 218012, 221884, 221890, 90825, 221913, 352986, 221920, 353045, 78017, 144729, 80112, 81554, 86067, 217557,
    86656, 218082, 88474, 286764, 92242, 289110, 94455, 94539, 94585, 424014, 346427, 283504, 287513, 76522, 80743, 223673,
    143628, 347111, 358582, 76601, 79464, 215877, 79509, 84013, 77935, 79876, 80877, 82083, 86582, 146157, 147863, 213815,
    281357, 215921, 153744, 285057, 288929, 288996, 289036, 485778, 499793, 303211, 144337, 79037, 414559, 418439, 418441, 287425,
    287432, 287434, 156429, 422321, 299168, 144562, 215815, 309255, 152179, 283228, 353131, 147802, 350860, 215064, 215349, 217123,
    348252, 414512, 219671, 289099, 354790, 358525, 622607, 229406, 360522, 495721, 364658, 630788, 632853, 82503, 215638, 217554,
    223735, 360484, 362587, 78133, 80203, 86566, 86584, 153689, 287189, 624751, 626734, 628772, 108557, 634888, 78616, 78877,
    76597, 79511, 282635, 352626, 76231, 76729, 78711, 151818, 153622, 154598, 223449, 618514, 94257, 225548, 436252, 78857,
    217186, 219430, 155652, 221621, 223296, 223612, 225710, 147050, 415802, 76671, 82119, 82260, 148376, 84121, 84607, 86037,
    152514, 88716, 90304, 160228, 160232, 160236, 160238, 160242, 160243, 160250, 163964, 75792, 77910, 78427, 80164, 80433,
    147256, 82302, 148472, 84388, 84689, 84746, 86285, 86592, 88645, 287048, 90578, 156517, 215782, 78248, 79914, 80092,
    81738, 82278, 82359, 86399, 88222, 94404, 292969, 100437, 81022, 81035, 499776, 84446, 84616, 86214, 86643, 88849,
    89021, 89036, 90562, 287191, 90645, 287342, 90745, 90754, 352927, 92489, 92525, 92547, 289253, 420333, 94538, 94628,
    294915, 78493, 78585, 279197, 144763, 211064, 211988, 213106, 215239, 86098, 229447, 233591, 106587, 237673, 110620, 112653,
    178197, 114691, 362599, 145224, 156559, 168139, 145843, 350282, 483942, 79544, 348267, 358606, 358627, 418407, 79301, 159776,
    172034, 630821, 180240, 170091, 215848, 144196, 217942, 221889, 221896, 221898, 90857, 233640, 213373, 148018, 280938, 284997,
    287252, 616613, 92419, 485791, 160146, 491604, 624710, 231521, 104562, 143699, 78418, 84724, 145440, 279215, 148165, 82714,
    149713, 84368, 86062, 86314, 86835, 219168, 155979, 157884, 92468, 158131, 94436, 94559, 94571, 489630, 102467, 368730,
    278568, 86112, 288783, 487762, 347025, 168121, 364747, 76728, 77826, 79495, 80630, 81134, 81609, 219741, 90213, 92316,
    420073, 223593, 100412, 106560, 153879, 155827, 155906, 157940, 160102, 79702, 84916, 84941, 484139, 484200, 91039, 293072,
    293088, 489700, 489708, 96510, 299213, 495824, 495837, 144835, 358596, 283494, 628748, 147362, 147864, 349021, 352990, 155966,
    156153, 161886, 79050, 78313, 146562, 299015, 434262, 178191, 75867, 144247, 487726, 348804, 352888, 352912, 78924, 147473,
    413787, 151769, 151847, 215555, 634937, 347124, 162016, 79251, 145768, 149782, 217822, 172035, 82589, 211189, 213623, 215363,
    217443, 219627, 221327, 221586, 223305, 225688, 227435, 233543, 78631, 88114, 90133, 483710, 632838, 112647, 86940, 144177,
    152156, 100462, 417931, 143830, 82036, 76578, 79981, 80464, 80509, 82356, 82383, 82725, 84028, 84303, 84872, 86089,
    88498, 285202, 90697, 484087, 92491, 102510, 364659, 233638, 233649, 106556, 237682, 630908, 634943, 223737, 211574, 215607,
    352364, 354595, 172132, 144387, 146240, 76395, 144944, 153983, 154037, 156108, 156180, 289009, 291076, 293004, 368658, 634886,
    636947, 76510, 156109, 352808, 75817, 144693, 81731, 84517, 416313, 356621, 368690, 145294, 211254, 148423, 217778, 418810,
    88151, 416757, 237667, 217166, 289079, 223591, 303171, 76771, 213302, 146509, 92214, 157801, 78350, 145480, 146925, 213364,
    215162, 217263, 219219, 350371, 154197, 220155, 352472, 352656, 352774, 92184, 92270, 354533, 160079, 161934, 231520, 104453,
    104492, 170051, 309258, 213921, 358478, 76368, 151585, 78854, 79280, 145276, 82715, 150446, 150479, 150488, 150523, 156499,
    156564, 156571, 161996, 162042, 166004, 422216, 286780, 78583, 628739, 301098, 303147, 305152, 243729, 309275, 76696, 82031,
    283214, 78542, 147997, 219284, 278827, 220108, 84076, 284802, 78645, 213670, 215879, 278719, 283319, 86443, 156478, 78511,
    148184, 346344, 152059, 358589, 358612, 356863, 143758, 420099, 79851, 80381, 80401, 213958, 100448, 215600, 159883, 96419,
    351216, 144917, 79843, 145474, 219385, 153751, 156029, 98329, 165915, 165966, 167954, 170038, 235590, 174085, 414403, 157741,
    223604, 77843, 144524, 212173, 81285, 88492, 221433, 497729, 283192, 79460, 81908, 79759, 618942, 626847, 368751, 157719,
    148270, 82552, 147355, 347059, 144150, 145076, 414134, 430082, 144985, 212953, 94661, 75919, 213650, 220095, 215048, 281463,
    352931, 616883, 281382, 350897, 79006, 144966, 212498, 178184, 440345, 148151, 143610, 145079, 145635, 147908, 149724, 80884,
    280893, 219316, 287173, 372786, 438327, 76184, 212778, 84715, 152021, 221228, 223574, 146773, 158153, 158172, 80043, 147140,
    84367, 86274, 86750, 153916, 92510, 158116, 94358, 144725, 279212, 151810, 152286, 145675, 148272, 157990, 149832, 150369,
    153729, 219775, 219936, 155679, 156339, 157882, 158060, 225351, 79011, 434240, 217747, 157728, 221857, 88496, 159925, 143363,
    147094, 150088, 493600, 163895, 484116, 483647, 358559, 425994, 146378, 80953, 80373, 86157, 152126, 153995, 352451, 92589,
    94623, 100454, 76218, 81856, 350943, 235523, 78881, 150472, 77904, 80045, 84515, 86746, 419876, 94226, 159923, 227342,
    176139, 110631, 79458, 79927, 82134, 82151, 82214, 82679, 215918, 86440, 88123, 88299, 88654, 89027, 89051, 287269,
    92285, 92447, 94405, 96309, 100415, 104494, 104546, 106581, 81212, 81637, 82053, 217743, 158036, 80481, 80731, 146331,
    346164, 152072, 350554, 487522, 354810, 82430, 283645, 291308, 291322, 80431, 82487, 82560, 84428, 84655, 84710, 86702,
    88420, 88746, 155793, 90295, 90393, 92648, 147822, 146156, 154300, 153881, 155744, 149583, 346986, 78755, 79278, 143898,
    212359, 418449, 145380, 148448, 628835, 144340, 80901, 346116, 348783, 352760, 487554, 491638, 278607, 280874, 296978, 77948,
    81010, 92272, 100380, 428121, 299071, 104558, 82694, 145070, 144142, 211469, 219179, 78726, 212338, 278821, 84127, 281255,
    218096, 88092, 285110, 221222, 90715, 222145, 288961, 354687, 422274, 146928, 78371, 422047, 491601, 79354, 215892, 86388,
    152570, 618884, 630809, 79847, 218060, 225766, 146175, 348894, 84811, 483999, 144541, 81336, 149596, 170075, 81807, 225379,
    227517, 227540, 165924, 626716, 366686, 628852, 76266, 78439, 78444, 80555, 296996, 497758, 145511, 154362, 233598, 145101,
    79678, 162009, 75917, 76620, 80083, 80570, 81225, 84051, 84249, 280897, 86392, 86447, 152036, 88607, 154354, 352534,
    156105, 352793, 92440, 420232, 92588, 618550, 94269, 487535, 94336, 159915, 487649, 487677, 422147, 487728, 487763, 160133,
    487841, 426071, 100404, 100433, 495651, 430207, 432147, 215736, 153971, 417812, 418130, 287232, 420040, 618583, 147395, 147448,
    82116, 148199, 151739, 86299, 88470, 86243, 349038, 88561, 88773, 90527, 353036, 81107, 287235, 80543, 81090, 82223,
    88404, 289249, 144131, 79908, 82121, 82525, 144660, 144744, 282944, 348912, 79829, 215976, 285687, 221987, 221990, 222003,
    222011, 222063, 222102, 222103, 222115, 222136, 620718, 620720, 620722, 620724, 620730, 424123, 227521, 227526, 227533, 227534,
    620757, 227542, 227548, 227551, 227555, 227561, 227563, 227564, 424173, 227575, 227583, 233656, 626873, 233658, 430267, 626876,
    233661, 626892, 233679, 233686, 430297, 430317, 239654, 239663, 239664, 239666, 632882, 632883, 239672, 632889, 632892, 239678,
    239679, 632895, 245785, 245786, 245789, 358561, 370690, 76304, 78606, 145683, 77996, 84381, 353193, 145243, 146480, 150518,
    156514, 156582, 168144, 79855, 229462, 79541, 81425, 149925, 78763, 278913, 283473, 152439, 285625, 155686, 287445, 420169,
    163935, 295018, 299169, 299170, 303221, 79628, 81845, 82636, 82762, 82771, 82775, 215899, 348990, 278676, 144402, 144707,
    76516, 351036, 212073, 86049, 424065, 358616, 364763, 279142, 217183, 219739, 227332, 364733, 364751, 370735, 79402, 428062,
    497715, 77990, 78795, 80029, 80246, 81591, 82199, 219750, 212283, 79543, 84820, 143625, 145823, 145982, 146255, 149649,
    151637, 154164, 484343, 79782, 79856, 82854, 489514, 491543, 77896, 90227, 144023, 233476, 483687, 76647, 84255, 90311,
    90689, 349078, 349086, 223338, 290888, 225671, 366627, 366667, 75830, 77986, 81385, 81832, 82751, 280971, 282703, 86808,
    89042, 144788, 212222, 162034, 364784, 83999, 281171, 86260, 86715, 152391, 221250, 90261, 484048, 221960, 420259, 92657,
    289279, 421929, 225545, 301114, 237691, 80160, 80862, 151897, 153941, 154275, 219997, 88957, 220087, 352367, 157736, 223350,
    356434, 94420, 225658, 233615, 366637, 76560, 211868, 81742, 213829, 217597, 144537, 76298, 81681, 279042, 79252, 287706,
    616955, 495751, 145621, 147950, 215792, 282713, 156030, 487524, 233581, 481443, 440330, 102463, 353130, 146681, 213121, 150146,
    153836, 159937, 144645, 211647, 80444, 84560, 88744, 88906, 88988, 90468, 90662, 92477, 92530, 92629, 92631, 94605,
    102469, 102545, 497771, 356465, 497725, 88747, 90350, 360498, 144969, 154361, 287256, 78065, 84310, 88348, 88715, 88927,
    92558, 102499, 346992, 178206, 144935, 82045, 82289, 82444, 215919, 86502, 88077, 88445, 88800, 88830, 88909, 484308,
    419909, 92274, 92452, 94255, 227465, 434183, 279099, 154323, 217169, 495666, 76016, 78283, 79884, 79986, 145720, 80663,
    81623, 81625, 147296, 82028, 82578, 83969, 84091, 151771, 217625, 219231, 219706, 219935, 89023, 221807, 157706, 420184,
    92606, 225358, 220139, 346667, 215867, 217719, 483432, 352454, 354677, 144316, 79515, 145791, 213367, 215122, 215601, 217708,
    217858, 283428, 219299, 354651, 356758, 364633, 364666, 178186, 212839, 215524, 348174, 219611, 482216, 616687, 354753, 356405,
    358413, 360501, 360512, 211807, 213788, 483954, 231486, 78601, 147198, 75898, 213457, 149585, 215326, 286739, 221273, 354576,
    223544, 94421, 493649, 233537, 235609, 301158, 106534, 81803, 82043, 147989, 282674, 219145, 88318, 288958, 96262, 98311,
    150327, 94645, 94647, 94651, 94654, 94660, 94679, 94681, 98406, 77956, 80155, 80871, 81353, 81610, 82185, 84382,
    215758, 86068, 221719, 92252, 618836, 81284, 414520, 287267, 626764, 146008, 146296, 147454, 84366, 86200, 86246, 86585,
    89037, 90301, 92417, 481960, 487821, 144770, 81943, 495773, 497727, 283042, 220151, 616503, 420111, 143421, 279155, 287704,
    79078, 154032, 92665, 92666, 616954, 78872, 144059, 212263, 81766, 503830, 505861, 505885, 180232, 76318, 217556, 78798,
    81037, 150310, 150363, 353090, 358574, 358625, 628783, 80712, 221363, 221442, 354466, 223476, 225638, 291327, 143713, 147053,
    147342, 148380, 148396, 148404, 148436, 152505, 152519, 152530, 418800, 150516, 368763, 289272, 77979, 283114, 416450, 90346,
    287039, 288837, 289161, 225706, 296961, 296981, 296984, 218087, 218091, 225784, 229493, 626940, 78208, 145655, 86777, 88915,
    422191, 227460, 213825, 143927, 80853, 146491, 434280, 144063, 82907, 78060, 80786, 80980, 84532, 90677, 278852, 79033,
    79194, 144175, 76629, 76662, 79326, 219707, 76581, 147661, 219340, 221942, 225721, 94669, 225742, 364691, 503869, 75899,
    145664, 81291, 147731, 149710, 358510, 163906, 165914, 414457, 356634, 487820, 219422, 79482, 83971, 92361, 143433, 145565,
    145585, 147098, 147127, 147735, 154086, 285286, 156017, 287700, 287733, 211850, 215662, 484169, 484221, 489684, 299207, 501821,
    213086, 279345, 219886, 352923, 352925, 354343, 354344, 419958, 485540, 354539, 485642, 289039, 620610, 362623, 430223, 301137,
    440323, 483860, 79710, 146229, 146196, 168052, 364743, 151741, 151694, 75989, 146329, 148036, 144896, 416359, 285459, 352758,
    231449, 165941, 237584, 146858, 147066, 626826, 144852, 211109, 213509, 414634, 346673, 348621, 348919, 78270, 144210, 81689,
    90352, 354455, 144872, 81236, 81746, 81779, 84247, 84609, 86423, 285527, 90539, 94292, 102438, 366636, 110610, 80375,
    148471, 144613, 78385, 417959, 90501, 90738, 92327, 92621, 94347, 145275, 413906, 416299, 420103, 88155, 92289, 96357,
    100477, 90849, 364717, 630901, 80042, 82010, 82335, 283115, 88778, 94489, 76437, 213454, 82903, 281323, 81661, 278667,
    143454, 284896, 284970, 285444, 289087, 292949, 297022, 82414, 352638, 501766, 144822, 282707, 217882, 348986, 418525, 418558,
    76090, 78045, 80291, 282848, 86666, 88460, 90268, 92529, 94541, 100399, 354724, 280846, 154450, 287258, 289191, 290982,
    290983, 301140, 80972, 151569, 145256, 156577, 156596, 362512, 213353, 146379, 76518, 145528, 82736, 150282, 144924, 80578,
    147241, 154458, 352519, 354550, 144214, 219572, 493625, 283186, 299104, 211632, 84925, 84986, 90956, 91013, 96436, 102597,
    102600, 102614, 102626, 144800, 78033, 82388, 84178, 88100, 350412, 90140, 90742, 92181, 92509, 94215, 159840, 489507,
    620608, 161865, 100365, 484085, 76458, 144661, 79857, 81759, 212133, 278602, 281021, 211832, 78759, 78975, 279315, 86025,
    90462, 92188, 92319, 96345, 102405, 104457, 628771, 628811, 106530, 110639, 76026, 148409, 215727, 217774, 426098, 287372,
    76328, 80261, 80812, 82434, 84503, 348591, 414180, 88212, 90366, 620635, 294977, 362510, 364619, 223566, 76561, 143552,
    82012, 84189, 84301, 84347, 150297, 86618, 88505, 285284, 88787, 89040, 90469, 616694, 165943, 144829, 76259, 161897,
    78043, 346560, 415941, 219640, 418351, 348983, 352930, 281244, 76500, 80070, 88215, 90233, 90493, 100379, 100430, 102418,
    104502, 170054, 108549, 143861, 212132, 213278, 219728, 219866, 212440, 78175, 145638, 289190, 223682, 80396, 414533, 434197,
    434220, 86908, 221860, 90831, 98409, 146201, 149757, 154538, 223263, 92282, 432133, 235644, 303152, 438301, 638989, 146442,
    430182, 217155, 219904, 220071, 213091, 414529, 352826, 79101, 79362, 156350, 349154, 284947, 287184, 76009, 80817, 82215,
    88808, 213705, 143983, 483844, 148275, 484306, 213015, 144495, 151838, 616533, 420055, 420205, 210960, 217577, 79232, 82411,
    219380, 155858, 86478, 154128, 154180, 352312, 156328, 289153, 289257, 358548, 227328, 80536, 146869, 81897, 82368, 82748,
    220152, 155739, 231468, 76087, 218053, 350298, 76154, 143389, 80662, 81274, 81286, 417987, 90568, 221863, 158125, 293013,
    165990, 81855, 89062, 278717, 217301, 285088, 219930, 487507, 290938, 487626, 301128, 92202, 212291, 79775, 79864, 143487,
    287448, 418533, 418551, 426076, 168102, 168113, 172146, 84278, 84650, 211039, 80624, 147358, 148136, 151933, 352726, 291097,
    81306, 82094, 82701, 414619, 152484, 156363, 418559, 156425, 163943, 353180, 78584, 285297, 296967, 430103, 148449, 418766,
    418787, 287716, 422371, 422392, 426101, 78892, 147151, 149674, 151990, 285278, 221305, 485519, 290861, 76503, 81683, 82347,
    279108, 346588, 154071, 157859, 487496, 426006, 362503, 172067, 144094, 147832, 144864, 352583, 354631, 364584, 76323, 80240,
    215401, 279546, 349176, 618997, 76160, 82375, 283452, 90160, 81782, 143416, 150234, 485601, 501784, 84281, 217687, 217977,
    352967, 618941, 487883, 487887, 495792, 144734, 145983, 146082, 146832, 161875, 163851, 438318, 418489, 144173, 145586, 418121,
    86673, 94505, 94569, 213211, 213380, 219411, 154081, 285398, 221648, 212027, 278890, 350864, 354558, 420197, 485735, 291012,
    292874, 292901, 620598, 620656, 622653, 426067, 493612, 297045, 495764, 495772, 628834, 413741, 144965, 143768, 350872, 75803,
    76419, 78125, 78340, 78415, 79879, 82073, 86782, 88532, 154385, 90579, 352889, 92303, 354789, 160074, 235610, 145620,
    153737, 147690, 149984, 150270, 151854, 283202, 152198, 219808, 290937, 297066, 495711, 278649, 280931, 217401, 284938, 286770,
    90249, 287361, 94217, 94272, 160087, 424099, 233573, 77916, 78012, 81141, 81396, 81748, 86244, 89084, 213112, 213180,
    213651, 221526, 221772, 616656, 290922, 227420, 143401, 213466, 280731, 215692, 348228, 481574, 350702, 285327, 483765, 485427,
    157856, 144332, 80856, 82005, 84547, 84824, 88322, 89071, 90727, 288872, 94216, 290921, 291000, 96421, 293032, 98331,
    299111, 106536, 303172, 144283, 149556, 153982, 155675, 155746, 289090, 145186, 150188, 219305, 145728, 147507, 150372, 217483,
    283169, 287021, 156129, 156218, 172138, 144478, 82538, 81452, 440348, 211099, 231539, 366610, 215245, 215826, 283274, 618605,
    291149, 78890, 145423, 82177, 147992, 281497, 281502, 281516, 281519, 281549, 281573, 281598, 151840, 219787, 154605, 484143,
    287548, 287558, 287609, 287647, 293037, 293040, 293048, 489673, 293073, 293079, 293098, 293102, 293105, 233571, 495799, 299200,
    495810, 299218, 495828, 495848, 299246, 305205, 501814, 311318, 311323, 144054, 219701, 219818, 426004, 78029, 78422, 348326,
    86228, 88387, 485445, 354806, 424000, 491595, 432151, 84150, 84474, 285544, 90307, 90349, 92220, 100455, 299020, 102490,
    281475, 415756, 418204, 422193, 82890, 349008, 94573, 161820, 428140, 168072, 144271, 79344, 90863, 151630, 223543, 158055,
    223408, 79035, 162035, 79213, 428146, 503838, 503841, 79527, 76081, 80528, 80803, 84196, 84528, 347098, 217394, 153637,
    154014, 353117, 353135, 353187, 92368, 420105, 420279, 422062, 358593, 358614, 364730, 497676, 94229, 78964, 215594, 283341,
    219579, 285410, 354769, 620696, 75975, 80349, 420130, 495779, 495787, 231428, 144572, 79588, 280769, 217106, 219350, 219593,
    219916, 220049, 223562, 223633, 225480, 414040, 80505, 84735, 78824, 154374, 161968, 162026, 168146, 180246, 283083, 88620,
    485588, 618644, 422105, 80032, 86640, 94487, 356753, 79760, 82850, 82880, 82929, 147842, 213677, 148182, 150377, 211769,
    216027, 418598, 222014, 418651, 418673, 418712, 418720, 227531, 424145, 227546, 424162, 424171, 424189, 233657, 233663, 430283,
    233688, 233693, 233694, 233702, 239658, 632874, 239661, 442391, 80156, 82085, 78941, 79016, 222163, 222188, 212597, 94582,
    79202, 78683, 212504, 218056, 636949, 76103, 80122, 82427, 217176, 282813, 219590, 219632, 225443, 225650, 227423, 231462,
    84289, 218107, 222148, 222160, 222201, 362555, 217834, 219378, 287009, 487589, 86565, 80733, 215123, 288923, 352737, 368746,
    144590, 144713, 152098, 79146, 144217, 144919, 212574, 279461, 216032, 418635, 222078, 424153, 620765, 424165, 430269, 430287,
    436271, 144997, 211526, 149581, 221307, 213116, 227448, 145117, 81113, 219544, 81490, 284705, 305153, 79434, 213312, 144434,
    79899, 86470, 217821, 303141, 79907, 81931, 86818, 155848, 358542, 616572, 484071, 78721, 78577, 144628, 90769, 418633,
    418685, 418700, 418733, 424125, 424148, 233671, 626921, 436285, 144168, 81050, 215455, 483802, 225510, 233540, 144998, 213117,
    217574, 219517, 79977, 80230, 81509, 148033, 88547, 285155, 287041, 287263, 75980, 76606, 76756, 79578, 154487, 78535,
    144989, 146251, 148107, 420307, 487852, 356841, 424028, 489606, 360543, 428137, 364645, 152149, 146045, 148261, 144165, 211799,
    350556, 219897, 229408, 432130, 366596, 440342, 145021, 147417, 168031, 618753, 491590, 144903, 76630, 81863, 420346, 79039,
    215796, 215334, 346712, 217118, 352293, 354751, 422185, 233620, 76267, 80142, 82096, 278992, 213615, 84400, 90729, 145623,
    215201, 215711, 155659, 620686, 146314, 143781, 78459, 145469, 146948, 84081, 151563, 152053, 92364, 487583, 144073, 290865,
    483446, 104531, 144507, 78436, 144892, 80799, 88115, 80197, 144211, 76309, 80588, 483411, 352637, 491614, 362574, 364562,
    366646, 432198, 370693, 150024, 89030, 90399, 156274, 157863, 92502, 92562, 158157, 96366, 628793, 368645, 507905, 442383,
    78075, 79211, 80237, 81229, 81369, 82074, 88454, 90144, 92441, 299098, 487913, 143990, 144021, 213658, 150315, 150204,
    349085, 352365, 620634, 76361, 79926, 82232, 86207, 86630, 90679, 147717, 149648, 149970, 151922, 219618, 156110, 156118,
    157840, 356534, 160092, 628841, 368710, 144758, 146225, 616651, 360461, 362519, 165925, 168060, 364679, 366604, 301074, 501793,
    632866, 145110, 217650, 416126, 417890, 233568, 145255, 215859, 353107, 353189, 487693, 350244, 352713, 352886, 76256, 79333,
    82211, 281332, 296964, 78776, 157899, 172060, 174114, 147677, 149860, 217156, 348568, 219630, 616478, 620694, 78717, 211209,
    279252, 212902, 81272, 84414, 86061, 154179, 146985, 215708, 287077, 352794, 616753, 354727, 356518, 356519, 366676, 77915,
    81924, 84007, 280694, 283329, 88982, 90747, 106506, 110619, 110635, 110644, 80416, 86077, 152418, 152464, 150240, 292975,
    146960, 82081, 84252, 149998, 88950, 305171, 215462, 76418, 78249, 86218, 153867, 223642, 79940, 81968, 82008, 82502,
    82601, 84341, 84832, 88221, 88400, 88907, 106523, 146387, 82003, 147792, 149515, 83996, 83997, 150210, 151887, 151929,
    86403, 153799, 88697, 88804, 155830, 156087, 156246, 288770, 157764, 92395, 92414, 160050, 160127, 96399, 96402, 360497,
    497724, 78559, 221720, 225424, 227395, 147097, 160006, 161909, 424074, 632858, 211152, 282827, 218109, 499838, 146465, 152276,
    156092, 158050, 289245, 358444, 167968, 239619, 307226, 243723, 178200, 112666, 178207, 180229, 278898, 81197, 84543, 358615,
    215767, 215104, 221191, 223244, 235597, 84202, 149513, 155738, 286991, 297069, 501788, 632862, 634933, 79894, 156091, 233505,
    76303, 78013, 144569, 147472, 282942, 483522, 156216, 493675, 80127, 145139, 217611, 152288, 485607, 223535, 422398, 147112,
    156205, 79001, 163889, 82421, 98395, 100439, 88409, 90580, 291007, 94567, 424019, 425995, 294950, 360525, 360528, 104544,
    106520, 172094, 147344, 215144, 151575, 88266, 221492, 358643, 112643, 75820, 144435, 76424, 78055, 79967, 147546, 147871,
    82343, 154314, 219980, 221288, 221809, 160025, 106616, 78034, 145033, 215337, 215388, 89070, 287253, 146063, 352663, 483797,
    77945, 81167, 143915, 80693, 147987, 282918, 77835, 80059, 76579, 78994, 146099, 278660, 364732, 364771, 210946, 213315,
    213471, 156336, 212392, 231465, 628778, 630827, 632832, 634884, 636955, 638990, 80756, 147051, 220157, 352648, 616463, 487716,
    143632, 143817, 145610, 153962, 154548, 156036, 143849, 147774, 79451, 150395, 156301, 78160, 81367, 84253, 86115, 154083,
    75786, 215468, 145516, 147675, 147844, 150279, 153875, 156112, 358603, 416218, 213915, 218083, 222157, 215851, 346833, 620561,
    624680, 628756, 156309, 78290, 144236, 146906, 165976, 84300, 218078, 484315, 160162, 626939, 628800, 87016, 91069, 91092,
    91125, 94689, 98420, 364789, 484290, 144355, 211005, 213335, 348578, 285091, 289107, 221852, 221565, 354318, 163865, 231451,
    231502, 233490, 235574, 301126, 239621, 282641, 144845, 81146, 79138, 213533, 346182, 110605, 82301, 86199, 86804, 219468,
    416240, 154611, 287126, 223275, 225306, 225410, 233622, 106606, 110649, 279366, 225361, 487620, 143368, 143820, 146052, 147547,
    148422, 148428, 84758, 86097, 349016, 349019, 349070, 153977, 90159, 353018, 220078, 223643, 144461, 158002, 145109, 146049,
    151590, 217184, 154384, 419855, 618834, 366666, 144395, 160175, 147224, 154547, 285531, 283102, 358605, 79214, 78355, 218045,
    159842, 280847, 350473, 350764, 285490, 102565, 79868, 81888, 81892, 148153, 82757, 150358, 84912, 84937, 84990, 96433,
    489687, 489722, 495846, 81263, 156299, 174117, 156140, 366716, 176143, 149545, 151557, 154461, 154551, 287208, 354367, 420202,
    423976, 493593, 428085, 364690, 434242, 436247, 503819, 144519, 221193, 155941, 229376, 78758, 147856, 149754, 163864, 229416,
    88759, 419956, 94428, 100473, 347135, 145257, 144709, 156561, 156584, 430128, 432210, 78028, 146134, 215146, 215336, 215799,
    217255, 86336, 86599, 348905, 152304, 217865, 157907, 92662, 225309, 159786, 94280, 159877, 159948, 94482, 94520, 94552,
    161856, 161861, 161906, 96415, 163850, 229407, 98366, 98373, 100382, 428071, 430095, 495649, 102442, 102453, 102484, 102540,
    170019, 170035, 366649, 104516, 170055, 106501, 106597, 110599, 110612, 180233, 180239, 484011, 144471, 143405, 145841, 147585,
    149525, 84633, 86219, 151944, 153679, 156147, 157750, 223337, 157895, 158147, 144134, 146430, 348651, 80479, 82019, 82451,
    82561, 149749, 215629, 88452, 154466, 155751, 90692, 158191, 144309, 144690, 346316, 283639, 291324, 346942, 280632, 416733,
    146534, 212368, 213444, 76327, 88152, 347035, 353190, 148160, 168135, 370749, 221395, 420252, 421913, 354637, 292942, 414654,
    102480, 75806, 144320, 80367, 146037, 82054, 84756, 86555, 90550, 288813, 82587, 284765, 352281, 286884, 148406, 152535,
    295030, 144357, 215592, 352870, 223731, 225365, 426035, 80208, 217194, 88356, 290960, 292931, 146940, 154072, 481962, 279111,
    153823, 88462, 88698, 90113, 90191, 90656, 94445, 100362, 102420, 80732, 278915, 148360, 284764, 220085, 287441, 156435,
    223267, 288870, 223733, 299120, 78504, 217846, 297046, 299025, 497691, 150473, 161969, 144205, 360545, 364711, 638996, 86236,
    84945, 84951, 145318, 148390, 146013, 79062, 212352, 79445, 144037, 213446, 235606, 145116, 219163, 418467, 278759, 219320,
    225579, 143505, 80012, 152055, 88881, 158032, 104536, 279293, 416753, 636934, 211928, 215619, 215896, 419944, 421993, 422072,
    424104, 434244, 144833, 146014, 152502, 152508, 152511, 152527, 152533, 351087, 156611, 156616, 156618, 156625, 156635, 156658,
    156671, 163961, 636951, 507906, 145080, 281250, 287195, 291244, 231529, 146171, 165931, 233636, 79181, 223575, 81080, 212046,
    145105, 144905, 157994, 421991, 632851, 212466, 418448, 156528, 80759, 81946, 148447, 84326, 86806, 349121, 78519, 215878,
    144203, 76025, 143565, 147134, 152330, 153820, 153923, 354614, 428095, 106578, 219397, 78638, 79572, 84770, 290952, 294929,
    76727, 79600, 287727, 415788, 485659, 485725, 144564, 217535, 287680, 165937, 152524, 285552, 156638, 289175, 160230, 80422,
    81725, 88218, 146278, 174084, 294931, 213616, 346201, 78999, 284839, 493684, 147449, 348583, 279536, 280646, 622666, 152465,
    87037, 368766, 79381, 145493, 81607, 148041, 151715, 151836, 151868, 86906, 484115, 622699, 84431, 86611, 144670, 146783,
    225754, 144420, 212998, 143826, 145737, 82174, 215292, 88597, 159756, 146169, 81068, 146862, 483373, 225468, 225500, 297081,
    281237, 154412, 616567, 618854, 618912, 424034, 358564, 79901, 79938, 146796, 81852, 152051, 86825, 153849, 102409, 152008,
    90652, 160070, 144051, 148180, 88705, 211015, 84012, 217493, 86658, 90386, 90484, 225520, 279472, 283624, 145628, 146513,
    82322, 151935, 360464, 145247, 211932, 81980, 84311, 282771, 88833, 299128, 76645, 79450, 82654, 82704, 82892, 96331,
    364795, 78304, 144115, 350246, 352368, 352499, 354531, 354710, 356608, 364602, 372736, 146041, 76308, 223250, 81460, 219742,
    287080, 419983, 225325, 75860, 76569, 213123, 237671, 432146, 79387, 145241, 353050, 150421, 150455, 347095, 150517, 156464,
    156548, 162023, 174126, 353267, 144231, 213689, 215894, 281325, 223706, 297014, 212351, 215828, 616497, 430199, 151786, 80849,
    493627, 165935, 78058, 82452, 145993, 146110, 146290, 157807, 626758, 144139, 287430, 287438, 287468, 287485, 144153, 215458,
    156483, 168123, 211597, 147175, 153762, 154503, 223348, 358621, 278862, 82507, 348680, 481455, 350681, 434186, 438299, 438315,
    438324, 94618, 79648, 213860, 356794, 147442, 148254, 148269, 215127, 217453, 78484, 218054, 147072, 147934, 280783, 280966,
    78766, 143493, 149857, 154593, 156276, 168015, 81523, 100441, 75818, 77849, 80117, 80196, 80727, 146322, 146516, 81130,
    81404, 213474, 150068, 284981, 219484, 88527, 88536, 416426, 156213, 287291, 289035, 289091, 92523, 98336, 98356, 301058,
    235524, 309270, 145263, 217312, 281396, 168177, 346592, 144232, 225329, 218057, 222183, 225758, 213537, 213560, 213634, 215757,
    217629, 217851, 219585, 221673, 485483, 616569, 223385, 157954, 223528, 225418, 94565, 227378, 229382, 229388, 301108, 237577,
    303151, 217993, 221914, 221973, 487861, 487863, 487867, 215531, 80575, 212194, 285368, 616551, 291216, 292943, 620644, 495661,
    301164, 144061, 145464, 147397, 147793, 148194, 283516, 156349, 287439, 156405, 295019, 81954, 84133, 86491, 144706, 636957,
    311304, 152436, 156344, 156365, 156371, 156386, 346157, 215827, 416678, 422123, 497787, 148023, 281369, 151619, 153717, 154368,
    154535, 155929, 297015, 76664, 156563, 79517, 81171, 212533, 81604, 88380, 88989, 92580, 418722, 424113, 424126, 424133,
    424143, 424169, 424178, 430313, 626928, 436272, 442397, 213342, 217216, 283054, 219442, 285047, 225310, 231503, 235600, 221868,
    144053, 414476, 163913, 428128, 503840, 217707, 283501, 285204, 291156, 91133, 495871, 76096, 77964, 211989, 147939, 217126,
    219555, 219920, 356708, 424047, 297060, 432202, 145640, 358405, 368651, 144825, 420233, 227424, 497719, 168155, 79618, 487812,
    499737, 218006, 218014, 159931, 212989, 215959, 215962, 216052, 221978, 221989, 221999, 222026, 222047, 418658, 418679, 222074,
    222076, 222101, 222106, 424114, 227515, 424136, 227529, 626894, 233697, 239670, 639000, 147614, 150106, 151797, 153848, 81720,
    82456, 86304, 89069, 352434, 221512, 354434, 362580, 144710, 146727, 217601, 217970, 217971, 218002, 154251, 225345, 356810,
    168035, 278555, 159887, 211161, 145867, 213132, 217662, 217698, 220069, 418580, 354341, 356431, 422261, 160159, 436251, 82157,
    213696, 84459, 290907, 350278, 420036, 143374, 80997, 84111, 88125, 89086, 616927, 94613, 110613, 495618, 499829, 143459,
    147216, 165945, 81323, 282666, 348710, 418067, 419913, 149689, 151712, 152081, 76589, 78117, 80360, 155773, 483726, 161801,
    416402, 287402, 420349, 76123, 279049, 215639, 88824, 221312, 223533, 227358, 425998, 148058, 153847, 483544, 356738, 145822,
    146262, 481846, 159851, 145171, 146152, 146615, 82742, 148335, 152424, 418491, 426083, 231478, 78105, 146441, 82202, 82400,
    217088, 86159, 217906, 220124, 418485, 288895, 76373, 79651, 281018, 86947, 102566, 106610, 358619, 144020, 78251, 144126,
    80890, 86708, 289112, 143966, 90488, 143937, 146736, 624745, 146105, 146619, 81873, 90220, 92451, 79084, 213185, 148111,
    78974, 213200, 278790, 285521, 352998, 149658, 78590, 81959, 76164, 287110, 289183, 293000, 352419, 145013, 351091, 418165,
    616519, 420166, 428067, 366710, 145490, 79970, 80316, 145881, 152214, 157908, 158000, 158029, 165986, 368749, 76660, 146447,
    150512, 89055, 288928, 148313, 279408, 280688, 215754, 282950, 616501, 624763, 630842, 77876, 84072, 84527, 90420, 104535,
    146609, 154506, 156175, 211428, 81377, 88856, 90470, 90559, 484348, 149628, 282877, 219266, 219286, 80085, 145831, 143792,
    78268, 79074, 82228, 149651, 153922, 219877, 145710, 215181, 217641, 481740, 211739, 491541, 148382, 79002, 145231, 213197,
    278738, 281296, 282661, 283004, 284745, 285048, 285105, 286964, 290978, 299141, 168153, 364775, 168173, 76209, 144636, 148034,
    84184, 84601, 217538, 415869, 219725, 88693, 416520, 351142, 221417, 352669, 157975, 225642, 225648, 620611, 626710, 235623,
    368693, 630867, 499796, 174102, 76713, 77827, 145045, 79972, 81558, 84552, 143847, 146334, 147719, 154100, 146036, 147805,
    285120, 366711, 79823, 82889, 225761, 229492, 491646, 78040, 143620, 80773, 81215, 146892, 82413, 147470, 150384, 143811,
    80616, 81079, 279475, 148443, 618598, 215451, 98355, 358418, 219992, 144145, 81155, 147113, 281245, 152199, 154260, 482234,
    616833, 143815, 90744, 80581, 94306, 144441, 346822, 370751, 82112, 281070, 86571, 88372, 104543, 626733, 144567, 146942,
    282976, 287396, 155881, 78570, 211773, 213061, 217179, 75907, 81627, 81951, 84136, 84732, 88933, 215667, 291014, 77848,
    79946, 80550, 279097, 149677, 84226, 86126, 227462, 145485, 147582, 349062, 153918, 618984, 88867, 618908, 144591, 416050,
    279410, 152399, 283481, 283505, 152467, 481421, 350786, 287268, 287444, 618914, 160189, 160199, 163952, 144687, 77887, 80792,
    282653, 283009, 284894, 282732, 291234, 144818, 213540, 415970, 219487, 416445, 285420, 416718, 352627, 287140, 288952, 354632,
    354662, 616903, 487625, 422237, 227351, 622665, 362547, 430102, 495709, 432175, 82009, 82540, 84017, 86790, 88692, 96380,
    76775, 160209, 144968, 176163, 144484, 75842, 146081, 80711, 82208, 82216, 146949, 152113, 152227, 154093, 219889, 352518,
    221457, 155952, 157969, 159834, 160155, 358508, 360493, 495746, 76033, 146690, 484029, 353006, 353011, 148467, 413734, 416528,
    628810, 75902, 82217, 215063, 84334, 84413, 90165, 90486, 221804, 92652, 149524, 151876, 154088, 291095, 170057, 156553,
    81449, 86320, 76449, 152171, 349040, 353049, 353122, 353141, 353143, 358578, 358600, 213064, 219735, 279477, 152504, 152569,
    418769, 287721, 418795, 430333, 148354, 86932, 211437, 86473, 88185, 88233, 88486, 88934, 286850, 90326, 92549, 94395,
    96333, 78749, 144940, 79734, 84881, 84896, 84918, 84946, 90924, 90943, 90945, 90950, 287674, 485880, 96465, 96471,
    96497, 96505, 293115, 96509, 102580, 299231, 102627, 102629, 305201, 346859, 151708, 155969, 156191, 76277, 88584, 92409,
    143993, 145118, 78708, 79345, 149547, 151880, 222158, 222159, 222168, 279007, 80688, 86762, 283555, 416063, 219461, 416372,
    424060, 79777, 76320, 218097, 222153, 289193, 487617, 225755, 225757, 225759, 229489, 229494, 622713, 233715, 289133, 79809,
    145554, 148017, 148150, 148210, 149619, 150398, 87033, 153830, 154257, 154348, 286837, 155767, 353237, 288981, 290878, 290993,
    160017, 291237, 292883, 360575, 299149, 79328, 144140, 212387, 150435, 79478, 281107, 213462, 82680, 420345, 305178, 76600,
    144404, 481321, 491574, 495629, 154085, 153938, 143630, 86269, 75934, 79453, 86556, 155708, 78553, 630823, 143634, 149918,
    221390, 221535, 354800, 229410, 233475, 299027, 235527, 78169, 81119, 153963, 155879, 419870, 485454, 354547, 160130, 161894,
    424086, 495653, 172107, 145506, 150160, 86047, 152016, 75823, 149574, 282693, 86258, 86595, 79368, 152002, 154165, 154189,
    419917, 160106, 160112, 170087, 144932, 146050, 146601, 147706, 153723, 157744, 294980, 482015, 278797, 213489, 148244, 280788,
    281240, 283065, 223374, 290831, 356668, 145915, 79081, 81463, 154294, 292985, 143759, 146692, 149898, 76622, 79457, 78902,
    146114, 215164, 215176, 215390, 150201, 150291, 217521, 219881, 285618, 420086, 223725, 291015, 225489, 225656, 360536, 77864,
    78871, 81653, 148319, 148320, 148324, 148340, 152420, 152425, 152463, 152496, 88553, 156375, 287480, 291258, 167944, 430247,
    104485, 170046, 172038, 106569, 110630, 178178, 178185, 147154, 426092, 79730, 79731, 84902, 84924, 84928, 281564, 84964,
    90906, 484147, 287542, 287585, 287595, 91030, 287658, 287667, 489645, 293047, 96486, 495808, 299203, 102634, 108582, 501801,
    108606, 76680, 77954, 78031, 79974, 145705, 81598, 213768, 149529, 84506, 152019, 152111, 86794, 219277, 88284, 155954,
    289078, 147462, 223497, 223671, 491570, 149938, 283413, 487792, 86879, 213979, 81564, 278604, 217242, 217587, 153802, 223317,
    223578, 229426, 76071, 78634, 146208, 212282, 81552, 151564, 349106, 76226, 84039, 86475, 285596, 422035, 219194, 81987,
    84416, 215505, 217437, 86435, 353132, 362604, 626747, 144284, 90170, 92192, 78990, 80242, 81437, 284971, 78291, 145075,
    80206, 211622, 416000, 279289, 215120, 281456, 219214, 219670, 221729, 145466, 147006, 152066, 219575, 160053, 499753, 278583,
    348332, 350224, 78605, 219963, 155648, 221240, 227476, 430192, 301171, 144902, 80485, 82424, 84469, 88541, 278808, 283509,
    283515, 283524, 78628, 145007, 147418, 145269, 150468, 156502, 156558, 79582, 146706, 79629, 79638, 79686, 221830, 90886,
    225749, 145921, 350333, 352567, 90714, 92585, 159976, 79377, 84789, 217783, 158092, 78452, 212802, 278549, 88783, 96420,
    299085, 495732, 501797, 301056, 76125, 144964, 217368, 86621, 94550, 81613, 350389, 77841, 79888, 84854, 86505, 86816,
    88846, 417848, 90432, 354554, 354689, 92627, 354793, 94353, 94595, 96375, 424084, 150519, 92597, 495734, 168176, 422039,
    278723, 284946, 219759, 285445, 292990, 79995, 81075, 346826, 88473, 287176, 487633, 217510, 352908, 215103, 217922, 225285,
    423974, 213169, 286828, 289059, 145836, 152221, 154440, 155948, 419901, 159994, 620558, 79163, 282913, 287456, 156395, 76008,
    279060, 82120, 482126, 487645, 214008, 84706, 88874, 88994, 223354, 92613, 94437, 434224, 110615, 79553, 144899, 146132,
    80290, 145637, 145669, 147515, 149575, 84155, 149761, 281180, 152011, 217627, 221225, 288940, 485650, 289089, 225593, 225621,
    620655, 227485, 294979, 231517, 628812, 235617, 235633, 211268, 79758, 82887, 360570, 78797, 487558, 75885, 79608, 81767,
    90408, 90414, 90634, 76375, 81805, 147459, 349126, 352575, 484320, 354373, 291242, 362497, 362517, 362520, 143859, 144909,
    80129, 80214, 146106, 149785, 151928, 221681, 223288, 157976, 159805, 165969, 104571, 146950, 413800, 485598, 98312, 626692,
    144520, 81605, 279195, 213794, 213830, 219393, 501774, 75971, 81557, 147803, 211898, 174141, 145328, 152543, 278990, 221640,
    78118, 81430, 280854, 146226, 81321, 415979, 285535, 620624, 98369, 360529, 165902, 168011, 172050, 370702, 438278, 374802,
    79979, 80710, 82470, 86417, 217909, 223568, 81413, 352316, 376850, 143541, 80069, 145945, 82062, 82245, 213607, 148259,
    280593, 84118, 84564, 86366, 86369, 152048, 217610, 152320, 88138, 219392, 88379, 285287, 154387, 219924, 220074, 352283,
    90205, 221533, 90613, 221686, 221750, 90774, 90793, 287403, 92522, 158080, 94359, 225435, 94631, 96296, 96315, 620603,
    358499, 96397, 98350, 362505, 100377, 624665, 102414, 626822, 432139, 104517, 106512, 106515, 106562, 370697, 108567, 108569,
    501785, 110634, 503850, 144535, 79237, 81599, 280754, 217311, 283409, 219781, 419895, 288880, 289267, 290901, 296988, 624729,
    145513, 148037, 352397, 352577, 352799, 354351, 354450, 354570, 282997, 219773, 285571, 145502, 80150, 86755, 153929, 88772,
    351010, 154565, 155889, 483705, 92508, 160098, 225668, 161814, 237593, 172095, 239626, 241681, 176176, 76767, 79523, 79126,
    283345, 147242, 147328, 84203, 86485, 155692, 79176, 150418, 348663, 156475, 156493, 174143, 279073, 285121, 287209, 288921,
    223490, 289064, 290954, 160101, 618853, 292914, 294918, 294924, 366644, 303113, 368687, 146484, 79031, 82591, 213880, 217878,
    86873, 349146, 616455, 360552, 622709, 630829, 630905, 241709, 145441, 147945, 149823, 156027, 157813, 158108, 159769, 161844,
    161953, 163887, 630789, 174082, 634900, 144561, 245781, 145023, 82314, 281232, 283088, 150420, 150469, 150476, 78538, 147123,
    416153, 364632, 154360, 78575, 143899, 81673, 213008, 215465, 152293, 153861, 221752, 92618, 348249, 79400, 82709, 481588,
    155811, 485430, 497759, 79687, 76565, 84231, 86163, 86508, 624723, 154118, 78644, 144288, 413972, 288831, 354666, 225394,
    358440, 423978, 426007, 299154, 368706, 370711, 81136, 356635, 279551, 283109, 620669, 78297, 155673, 156042, 100457, 144167,
    78911, 75843, 77886, 78048, 78198, 80749, 215682, 84615, 219296, 219372, 90788, 354738, 225473, 287376, 287682, 144865,
    144056, 146712, 148222, 215268, 215283, 360456, 154021, 157800, 159849, 159928, 161960, 172100, 78745, 81647, 84744, 86583,
    86724, 416448, 79617, 86961, 90846, 221923, 225747, 279405, 417826, 420074, 96473, 212267, 76148, 148438, 81121, 79613,
    146295, 151805, 350435, 356533, 75991, 78093, 212411, 81764, 92641, 231507, 81102, 416300, 351026, 354573, 354713, 281059,
    151777, 489508, 150161, 79043, 485408, 81618, 75939, 84739, 218073, 419866, 92298, 92570, 422094, 96384, 229470, 229484,
    622608, 279535, 214014, 145818, 75895, 84171, 283623, 485405, 420189, 291047, 229431, 82642, 90177, 159765, 487587, 489567,
    360508, 493606, 79415, 144453, 145312, 160249, 144752, 76682, 78083, 146427, 219410, 154223, 227454, 94469, 145471, 211692,
    94291, 80022, 147796, 215153, 215348, 84809, 221439, 219195, 484309, 223357, 225477, 78094, 229501, 233720, 150448, 414305,
    150500, 211545, 146291, 82453, 215714, 153609, 155826, 221659, 157826, 225708, 165972, 76216, 86450, 88490, 145392, 80907,
    86753, 283591, 422391, 211362, 88169, 348278, 352903, 352905, 145884, 151639, 217789, 288918, 616698, 420191, 491583, 212907,
    81914, 149641, 149957, 150034, 151773, 151906, 153869, 220103, 221657, 223637, 165965, 170083, 77839, 79942, 86256, 354363,
    104541, 86281, 146787, 151594, 79201, 145784, 278864, 86187, 220035, 155809, 347027, 353176, 358653, 79692, 79698, 79701,
    79727, 79735, 79896, 84892, 84914, 84927, 84931, 84938, 84976, 90907, 90909, 90910, 90915, 90923, 287534, 90932,
    90933, 287552, 90951, 90953, 287622, 91033, 91034, 91037, 91049, 91056, 91057, 287666, 489646, 489670, 489683, 96493,
    96496, 96502, 495814, 495831, 495854, 114711, 114718, 146705, 348480, 634891, 212730, 215803, 152506, 152540, 152557, 156607,
    156653, 163962, 144444, 213006, 213447, 215272, 215920, 284751, 289219, 489546, 229432, 215215, 217618, 285555, 352629, 616481,
    354630, 618926, 292870, 294919, 362531, 301174, 114695, 145054, 80023, 211910, 147382, 213767, 84297, 217678, 223232, 485857,
    225494, 81138, 169985, 170010, 176137, 151799, 350559, 485376, 487638, 88955, 223325, 616630, 92541, 94396, 235534, 106548,
    176173, 78487, 217588, 84306, 86751, 349131, 622714, 78228, 78897, 82747, 94342, 424142, 424163, 620774, 626924, 219732,
    150281, 281399, 75840, 81104, 84499, 285151, 90693, 281401, 219388, 211396, 282932, 100468, 219910, 78186, 297033, 102451,
    503825, 438320, 143548, 78363, 146932, 82352, 90392, 78656, 279104, 75878, 145662, 215385, 215443, 419954, 420132, 421935,
    290959, 225630, 149804, 78561, 88559, 154445, 157810, 157988, 159791, 144594, 212763, 287371, 305189, 144085, 152510, 152528,
    152566, 152574, 156630, 146199, 217356, 284767, 223570, 82898, 78121, 149944, 413751, 283063, 144696, 152518, 79065, 144653,
    358452, 360495, 144442, 156306, 80292, 80681, 82717, 92248, 149808, 79054, 214002, 151904, 618512, 626732, 281444, 80815,
    225419, 160082, 294967, 75940, 147244, 147841, 149638, 151554, 151579, 348422, 151885, 151956, 285267, 154322, 154553, 287317,
    161793, 211300, 213688, 217819, 351214, 368696, 438296, 79433, 159847, 88866, 622705, 144708, 280583, 350577, 309254, 217103,
    362507, 211320, 151723, 288817, 289134, 79655, 213881, 349007, 418219, 288999, 154448, 213108, 348176, 351030, 351162, 282943,
    154269, 79000, 79020, 481762, 78883, 416341, 618892, 366597, 170108, 237616, 372765, 491550, 75890, 144560, 212949, 213458,
    215132, 352639, 223484, 235611, 346830, 80445, 82317, 416590, 94252, 422109, 438316, 211503, 213632, 286865, 221367, 223720,
    356353, 153872, 350952, 78049, 84179, 215404, 213179, 283128, 147516, 149847, 154369, 144104, 145358, 299252, 484056, 76168,
    78244, 84536, 154133, 90724, 78757, 145827, 348384, 155804, 158065, 143484, 144415, 146546, 283171, 628823, 286887, 288804,
    96270, 213174, 213817, 151860, 219317, 220079, 307250, 372791, 618688, 211142, 82845, 86971, 225303, 281089, 282663, 217685,
    144353, 81057, 147173, 81682, 88930, 90215, 92655, 215876, 154063, 213699, 79307, 146256, 84307, 150115, 86118, 86378,
    88213, 92576, 94418, 143687, 80635, 100395, 76764, 143718, 281447, 281469, 350353, 353124, 358648, 100371, 78329, 144450,
    150513, 145227, 156458, 156581, 162017, 98304, 153862, 219693, 154172, 221532, 419878, 223465, 92608, 290975, 159983, 160160,
    428106, 299093, 279360, 281462, 283424, 481847, 79017, 145468, 212198, 212951, 151916, 211240, 213235, 213552, 221230, 221359,
    223321, 225392, 422181, 231546, 163893, 76014, 76541, 81222, 487890, 495786, 144780, 151856, 288847, 84407, 88673, 351157,
    143790, 211274, 76262, 81221, 156587, 144151, 78089, 80719, 149728, 215621, 217635, 218084, 416522, 221785, 484298, 616539,
    158105, 289248, 489612, 624668, 497772, 213912, 217247, 218050, 219333, 219457, 156090, 221743, 225779, 628742, 148312, 148314,
    144411, 212349, 279451, 283619, 79670, 279301, 218102, 483709, 616462, 76591, 79512, 146164, 148417, 418767, 436226, 78619,
    630899, 630902, 630906, 144147, 481875, 285422, 616473, 419879, 419880, 485494, 420075, 618502, 620711, 297028, 428159, 495759,
    366673, 632864, 505859, 114699, 79222, 283361, 153811, 79796, 86963, 86990, 87034, 94684, 78792, 370701, 282798, 414614,
    76714, 283371, 153884, 158094, 213404, 418534, 618735, 80216, 213067, 82030, 423966, 76136, 76478, 79470, 81834, 81858,
    82693, 282753, 284790, 286784, 94260, 94301, 100402, 149865, 491537, 78587, 213348, 84305, 86533, 211252, 86982, 353212,
    144900, 354410, 421960, 143707, 151713, 84588, 80048, 81429, 81556, 489577, 144816, 78951, 218024, 618946, 98405, 106611,
    106614, 106618, 78284, 221506, 213495, 284797, 352489, 483995, 362592, 278536, 76639, 213379, 213434, 215750, 157731, 144229,
    356849, 149675, 155850, 159966, 165930, 630837, 145873, 82221, 346129, 84690, 90175, 221295, 92189, 419924, 225362, 94369,
    96274, 622602, 235565, 372751, 79386, 150466, 150506, 150525, 484302, 484323, 353252, 618972, 487907, 487928, 161999, 162020,
    491637, 77862, 146414, 81092, 82105, 82176, 83990, 84299, 86754, 414487, 88269, 284999, 88604, 88810, 92559, 92625,
    94296, 356442, 94402, 622657, 414600, 78870, 144883, 346318, 144876, 145307, 349051, 349060, 79316, 211884, 280878, 346706,
    221438, 288790, 356659, 422205, 281563, 484134, 287550, 484187, 484209, 484248, 484256, 293067, 489681, 293082, 489698, 489707,
    489725, 299193, 299199, 495819, 299224, 299229, 299230, 299238, 305194, 305197, 507927, 82220, 297076, 628837, 76661, 79909,
    82345, 82613, 84688, 86396, 86654, 88337, 88371, 219491, 88537, 90173, 356407, 96299, 80283, 146752, 147740, 82551,
    84291, 86371, 88555, 90255, 90436, 90514, 92233, 94616, 96363, 102471, 145271, 278989, 148022, 215789, 161792, 151946,
    143412, 150063, 352533, 170071, 79423, 413761, 424081, 428066, 79835, 353223, 211193, 78330, 213863, 86909, 280579, 418068,
    418381, 424033, 432176, 76242, 146724, 152219, 88635, 90310, 417998, 418143, 156077, 156156, 156578, 353188, 92443, 225403,
    94394, 225472, 159970, 96277, 227355, 227403, 96359, 358640, 426018, 430083, 495635, 432135, 155953, 352747, 156269, 158179,
    76571, 79565, 81634, 148168, 215772, 217561, 219477, 219792, 285533, 154493, 285623, 417903, 223272, 288886, 419996, 420080,
    354642, 354733, 159956, 291194, 489621, 362598, 233472, 233474, 299151, 235541, 432173, 309251, 76577, 80966, 282696, 233588,
    212155, 84519, 150105, 150123, 152057, 86684, 152284, 153955, 88775, 155787, 90621, 94265, 291060, 102515, 146610, 143675,
    146215, 148329, 148355, 151795, 348629, 283199, 283540, 152491, 285174, 285251, 417836, 156394, 156409, 156412, 288816, 420108,
    420182, 225527, 225657, 160198, 163933, 163950, 217902, 217413, 227432, 81110, 88563, 416648, 422343, 426096, 221537, 145900,
    370744, 211467, 144464, 362581, 84123, 78456, 281496, 281515, 281545, 281550, 287533, 287537, 287560, 287563, 287575, 287591,
    287592, 287619, 287650, 484260, 293036, 293041, 293074, 489712, 293106, 293108, 299206, 299219, 299237, 495856, 299249, 286766,
    213071, 215338, 219540, 291229, 231442, 299051, 217524, 481841, 223487, 86303, 88680, 90250, 92623, 94483, 227357, 147581,
    152038, 153981, 434194, 634892, 80900, 77851, 145529, 159933, 76296, 147149, 349148, 144263, 349079, 484078, 484083, 80276,
    148032, 149771, 358464, 426059, 626744, 366615, 147334, 149932, 150212, 152272, 153709, 153855, 155963, 157803, 616744, 159843,
    160132, 622598, 163925, 165897, 172057, 172061, 174089, 174090, 176145, 110640, 146622, 144574, 210973, 418553, 418556, 422342,
    301121, 144391, 287411, 75967, 78015, 80418, 84239, 144755, 346861, 144870, 213553, 213686, 213746, 215155, 149629, 153665,
    219793, 352373, 354517, 356365, 356414, 356529, 225553, 356773, 358419, 364685, 215793, 219778, 212160, 153691, 157825, 161893,
    166013, 78779, 79355, 145961, 356735, 80495, 88495, 94364, 94432, 146499, 350636, 353100, 92420, 225603, 161977, 168154,
    364770, 211461, 346311, 219374, 76142, 155861, 616483, 160135, 165999, 167942, 167982, 168087, 628774, 235584, 237576, 172064,
    172088, 172101, 172134, 632835, 501771, 176136, 176138, 176142, 241688, 110622, 110625, 636939, 440332, 636942, 505882, 418773,
    152482, 156426, 144424, 82926, 78686, 146603, 146956, 214005, 84801, 79889, 80219, 82101, 86727, 148271, 283380, 146352,
    212522, 86105, 217639, 356410, 290916, 624643, 626706, 299078, 364668, 432144, 628790, 632837, 174086, 237604, 79087, 79774,
    279510, 152509, 283634, 356861, 79360, 81051, 76567, 75859, 80887, 155759, 157936, 159826, 170029, 145897, 147577, 147600,
    147815, 149697, 153866, 154544, 156289, 79769, 79810, 349075, 221966, 360563, 360571, 233631, 626856, 144348, 219577, 282644,
    616625, 618918, 82078, 84570, 86261, 88312, 219526, 221216, 78937, 78468, 284817, 143939, 147207, 211975, 215255, 217147,
    414271, 219365, 285584, 485517, 223378, 422184, 618866, 227439, 424062, 432229, 147427, 144999, 145517, 146000, 147892, 147919,
    149564, 149839, 150408, 151625, 154034, 350738, 156233, 223401, 158027, 358585, 358591, 168046, 430195, 364746, 172092, 211151,
    213077, 82496, 84235, 220143, 221799, 225288, 422037, 227493, 229403, 430183, 237608, 79128, 112641, 148325, 215617, 221325,
    221637, 223279, 223498, 82833, 86919, 86931, 219221, 219313, 154618, 221638, 90760, 90761, 94663, 98416, 78647, 215180,
    217622, 420024, 618697, 364796, 145096, 418171, 419957, 86608, 90467, 90577, 90672, 92403, 219172, 481484, 221814, 223253,
    223581, 225287, 626835, 80576, 154457, 76721, 79480, 88714, 88891, 88965, 90168, 94242, 96404, 144588, 279381, 287462,
    287504, 81521, 79311, 79136, 428068, 628830, 279305, 86839, 350584, 286942, 81333, 151700, 154106, 90203, 286992, 485871,
    618610, 94532, 96336, 493624, 78675, 145999, 146359, 146406, 154202, 154467, 284715, 82252, 153766, 156173, 144219, 283092,
    144942, 82219, 279304, 88954, 237664, 350344, 616509, 213792, 311317, 212762, 84759, 86109, 98317, 100375, 102535, 104460,
    106497, 106508, 106510, 239649, 370722, 79744, 278642, 280775, 84967, 90919, 90937, 96446, 294983, 299127, 102582, 303209,
    501810, 144952, 217125, 219323, 221428, 225442, 233605, 76040, 78008, 281053, 231548, 144080, 78769, 78991, 81456, 413866,
    219364, 219657, 221589, 299037, 233578, 237655, 630881, 81005, 150364, 152377, 156315, 156317, 223458, 144048, 150338, 143403,
    79528, 79575, 145446, 147704, 152065, 351191, 86442, 622687, 491617, 622697, 146593, 86973, 80200, 213097, 84865, 213599,
    150085, 221617, 616485, 223712, 487546, 424012, 624730, 497736, 497751, 145975, 284809, 160034, 86230, 78744, 154013, 416344,
    76382, 223299, 102423, 145930, 346713, 282812, 348665, 348892, 352395, 289145, 231550, 219158, 220134, 288985, 291084, 86481,
    150256, 350593, 159747, 421940, 159975, 161833, 161958, 428082, 168045, 170050, 75881, 80534, 80591, 146827, 213099, 82729,
    82754, 149744, 215331, 413778, 151652, 414526, 219539, 154258, 154579, 155734, 158146, 159810, 163898, 428098, 626739, 233555,
    213528, 217376, 219276, 223654, 282730, 150461, 354811, 278548, 144437, 483372, 143556, 414543, 146015, 149640, 350310, 352880,
    221892, 616909, 225731, 413926, 418708, 418743, 424140, 424151, 424186, 430310, 78651, 144652, 84363, 86095, 278560, 348934,
    88353, 624644, 79570, 482141, 76108, 149881, 84472, 219485, 219810, 223646, 225676, 303104, 144528, 145801, 81936, 217406,
    152155, 350230, 153806, 90680, 420057, 422215, 366701, 78402, 149980, 155778, 221803, 94267, 94610, 489489, 96337, 493608,
    497684, 106584, 233646, 79437, 213674, 346295, 282781, 153796, 154050, 154266, 155910, 287022, 90540, 158089, 161907, 163885,
    165970, 414651, 78161, 80366, 213525, 284681, 219859, 77973, 84714, 283585, 283592, 149622, 152257, 416322, 225449, 434281,
    152245, 154334, 156197, 223572, 92615, 159811, 356511, 225519, 356597, 225696, 428114, 624733, 233579, 364657, 170027, 170044,
    434263, 440335, 616713, 143474, 146964, 77939, 145619, 84352, 86631, 94443, 102517, 155947, 157832, 283457, 81200, 215960,
    216009, 216014, 221997, 222001, 222024, 222027, 222039, 222055, 222056, 222083, 222114, 418724, 620713, 227500, 620716, 227505,
    227538, 424176, 227570, 227572, 233670, 233683, 233701, 430320, 233713, 632884, 145564, 215135, 149676, 151774, 217508, 152252,
    152356, 152374, 219944, 155775, 223322, 158024, 223579, 225629, 489478, 491527, 233494, 233561, 233594, 235567, 630866, 143445,
    78002, 82171, 84658, 86171, 86212, 86389, 88762, 92315, 106589, 356406, 150227, 284713, 92454, 294966, 299021, 153614,
    88610, 154335, 90371, 90407, 92206, 92314, 223469, 161945, 622626, 146336, 350793, 155897, 79525, 81829, 215194, 348558,
    350525, 352934, 146806, 285691, 368647, 76466, 81741, 213486, 279290, 281328, 281486, 284983, 350663, 354321, 289278, 626752,
    299117, 145403, 283599, 287690, 287697, 212428, 279057, 354721, 149807, 144671, 78010, 81727, 84394, 88081, 88099, 92231,
    92337, 92481, 94630, 98371, 79030, 78429, 78927, 78794, 279397, 78490, 144081, 81640, 78179, 145645, 147837, 148120,
    149546, 149770, 84445, 150272, 151729, 151735, 151828, 283185, 219148, 219795, 220159, 90138, 155781, 221596, 221624, 156143,
    156234, 156316, 94386, 422190, 225585, 160129, 489580, 161910, 165908, 168034, 172142, 176185, 499839, 144787, 289178, 618702,
    231489, 284939, 143706, 147583, 81032, 82027, 84259, 285318, 495654, 88235, 144573, 353082, 353105, 353113, 353116, 358611,
    349103, 157749, 166011, 213129, 278853, 215678, 282977, 217814, 284987, 90241, 221326, 157995, 223545, 290839, 159819, 227441,
    102487, 148305, 152401, 152453, 156402, 418554, 291253, 291255, 291259, 291262, 160201, 291277, 291287, 291289, 295014, 148105,
    213019, 278907, 213591, 282971, 283125, 219464, 285488, 624749, 215189, 152077, 153682, 350444, 356545, 143397, 143767, 79947,
    145707, 147651, 147767, 149682, 215774, 151941, 152337, 153610, 154373, 285462, 157701, 157808, 225332, 225373, 165916, 231474,
    493657, 364607, 170094, 81898, 81945, 82605, 348314, 86842, 354389, 354650, 349177, 364798, 150354, 212708, 81694, 84440,
    84734, 420215, 96414, 170105, 106585, 110624, 221886, 78296, 80440, 81328, 90354, 290984, 350811, 221716, 358404, 161965,
    90743, 376854, 215854, 82482, 217595, 350446, 349004, 484076, 628764, 146838, 149978, 278994, 279338, 289020, 618675, 620695,
    144551, 143427, 146797, 147038, 212999, 213054, 150179, 217785, 219497, 219592, 219892, 483623, 485530, 354801, 159824, 364638,
    78917, 145022, 84752, 94382, 94598, 148031, 151841, 153919, 80327, 86607, 352308, 366692, 213567, 217377, 219455, 76052,
    154017, 155871, 155913, 145903, 239652, 78602, 483984, 78461, 78747, 82765, 148201, 148289, 143856, 78920, 219171, 223303,
    223409, 225702, 147218, 217322, 485705, 440326, 287374, 212441, 213146, 418248, 146780, 152324, 84977, 281404, 80644, 147029,
    81519, 215732, 219161, 88603, 88671, 88691, 88770, 89039, 90360, 90423, 90546, 90603, 90632, 90712, 90734, 92457,
    92527, 92543, 92645, 94277, 94452, 94617, 96364, 102412, 145470, 147530, 150127, 155693, 286898, 484040, 288898, 297044,
    170096, 219833, 346471, 223576, 146888, 424009, 145607, 149630, 151894, 219613, 154141, 616898, 505860, 147416, 82856, 352966,
    352974, 353004, 353021, 353240, 145854, 279006, 148068, 149940, 156178, 414187, 147133, 151945, 221210, 223436, 143947, 82800,
    86885, 86920, 86929, 212803, 82936, 151601, 154434, 634893, 503836, 442370, 211493, 215480, 348599, 75777, 77905, 145316,
    86536, 152531, 225462, 497769, 237638, 212694, 79616, 86865, 349057, 346170, 143504, 146356, 146450, 147265, 147917, 148479,
    350902, 159993, 489593, 145495, 160032, 144726, 147529, 78775, 144324, 146167, 78852, 212937, 213676, 148273, 148383, 150380,
    152572, 156613, 156637, 156643, 156649, 156662, 160224, 160227, 160251, 163954, 161997, 213930, 213989, 225776, 618992, 225780,
    229491, 229499, 622717, 626936, 237695, 79477, 217212, 223573, 425984, 499818, 81783, 86769, 90458, 92209, 485786, 413973,
    79746, 280585, 84950, 286810, 418063, 96434, 96456, 489720, 428141, 102589, 102607, 108591, 80066, 81765, 79029, 86998,
    360566, 146785, 148249, 156151, 152159, 159943, 79586, 352625, 495728, 145700, 212792, 147838, 150225, 154181, 352584, 279148,
    280625, 283398, 148252, 143852, 364664, 88860, 78574, 76444, 76781, 145716, 76367, 80209, 346455, 279211, 144714, 81949,
    82570, 84240, 84263, 86459, 88573, 88979, 148456, 291325, 215971, 353198, 88627, 76751, 147564, 149627, 281430, 219226,
    156321, 215047, 346936, 243718, 229398, 84876, 348656, 284776, 292989, 153959, 144786, 78193, 78326, 80908, 82268, 84420,
    86042, 88288, 88451, 88836, 90741, 92355, 92367, 92401, 94468, 94478, 96396, 211892, 213453, 146231, 278750, 215890,
    86822, 285188, 221740, 292973, 79654, 212573, 146185, 80716, 144584, 149535, 346707, 151796, 152251, 217927, 286786, 155797,
    485795, 158193, 487465, 291081, 366650, 80713, 145299, 213162, 279130, 150025, 219813, 225482, 618883, 79421, 422014, 148468,
    76068, 84237, 213732, 351196, 78973, 349182, 491606, 148398, 285172, 291306, 146126, 147578, 153702, 418770, 289029, 364669,
    78625, 213726, 346956, 483992, 221858, 78899, 147193, 348876, 219433, 154317, 493580, 493610, 235596, 484037, 150039, 86213,
    283056, 233639, 241723, 241726, 212562, 415777, 485669, 436225, 78523, 215786, 221351, 221595, 616554, 145699, 148063, 86496,
    284892, 419929, 485686, 354673, 158176, 422010, 291051, 428075, 493687, 628782, 432200, 432215, 303187, 215227, 219206, 288964,
    364724, 144088, 143593, 146068, 146532, 212834, 213337, 213359, 82573, 217436, 283112, 283121, 217848, 284710, 416382, 286832,
    417969, 286963, 92253, 288995, 289174, 225359, 487565, 291072, 356711, 620609, 293009, 622596, 622669, 622672, 624645, 165919,
    296994, 299066, 104462, 170006, 301106, 366688, 630784, 368664, 434233, 434238, 370703, 307200, 178190, 309268, 80784, 146920,
    82249, 279348, 220144, 288874, 356424, 428064, 143667, 145497, 219690, 227388, 81654, 143570, 278710, 82195, 148408, 148418,
    279497, 84174, 280807, 281064, 150231, 282856, 219730, 285334, 285615, 156642, 422369, 292970, 426100, 630843, 80523, 146718,
    280945, 86050, 89072, 424003, 279198, 279283, 79022, 151717, 221426, 156209, 616709, 423989, 79091, 153775, 291170, 292886,
    303167, 145302, 82734, 210972, 79934, 81006, 278841, 217604, 414371, 90157, 299056, 168062, 104560, 212794, 217469, 80151,
    82521, 144262, 78165, 82306, 483426, 286975, 160047, 213975, 487874, 229485, 148174, 81665, 213212, 79389, 414401, 364774,
    82617, 84822, 144275, 145311, 79761, 484295, 79388, 86372, 76398, 348769, 153735, 281412, 79265, 415760, 76410, 279186,
    350716, 285631, 155697, 80689, 94266, 618615, 159908, 100389, 102524, 299143, 301068, 235538, 303118, 632845, 436257, 78740,
    79283, 145472, 213127, 147701, 231524, 78840, 286849, 223423, 634881, 221612, 212252, 147327, 80741, 84604, 279158, 213906,
    282652, 418081, 143456, 145541, 152045, 152122, 154264, 419914, 487486, 489491, 144991, 147375, 84379, 154138, 88791, 92331,
    360541, 360558, 144472, 149592, 153870, 350766, 352399, 418417, 420206, 618563, 487572, 78926, 144808, 81361, 78271, 278937,
    281269, 284749, 487470, 80147, 75932, 146035, 80531, 151811, 78978, 145500, 147909, 149708, 154550, 155742, 156292, 163860,
    163910, 163930, 170039, 213850, 82908, 78895, 280640, 346451, 283292, 354720, 628786, 144539, 84065, 86829, 100367, 630887,
    78130, 79624, 281318, 90763, 108581, 79183, 75976, 80201, 149590, 75985, 77854, 79971, 82435, 83991, 88120, 88369,
    90732, 215940, 278940, 212832, 217307, 285242, 485720, 290894, 420139, 421963, 287440, 150458, 81062, 145798, 145829, 79524,
    81899, 213710, 86348, 86412, 88406, 90518, 92221, 153683, 154396, 221882, 484036, 352987, 221916, 487875, 84511, 86540,
    88094, 417952, 90636, 94523, 98322, 106522, 106561, 77938, 146612, 81428, 81923, 82619, 428130, 148198, 282642, 283320,
    284886, 285129, 285452, 285585, 483846, 289098, 289169, 291016, 81830, 353225, 293028, 211733, 91115, 356856, 233619, 144278,
    145057, 145630, 147797, 149566, 145759, 153728, 80248, 90273, 353057, 353072, 170016, 289273, 481339, 413804, 144428, 82454,
    215411, 152023, 144766, 145377, 294933, 281418, 86486, 88602, 90267, 92384, 76681, 79646, 281016, 143414, 147308, 286853,
    287215, 287306, 291201, 292982, 622633, 163923, 296980, 299106, 285649, 143972, 146335, 282847, 153651, 285317, 287034, 485431,
    291172, 624657, 144256, 278791, 211316, 212068, 154327, 154572, 290895, 231455, 626698, 630797, 436239, 370715, 243726, 144976,
    79489, 144643, 217164, 414643, 217497, 219180, 143746, 143418, 146153, 147377, 147696, 148131, 149983, 150235, 152145, 152369,
    153733, 287027, 420032, 165956, 79700, 79719, 84962, 84974, 484129, 484144, 90941, 91000, 91003, 102585, 82835, 90888,
    360547, 638995, 215515, 76716, 150504, 353052, 353054, 353064, 353092, 489524, 358645, 143377, 145424, 213785, 150390, 347091,
    152041, 152352, 154382, 483384, 155968, 221687, 156449, 156507, 420090, 420225, 158163, 420329, 159889, 160131, 161911, 489620,
    162018, 174131, 354601, 160158, 294945, 165890, 165954, 167990, 151656, 147990, 152202, 88308, 88326, 88636, 88929, 88932,
    92399, 92638, 290885, 94447, 160077, 94609, 94624, 96372, 493661, 279391, 152415, 283497, 283525, 283535, 311316, 76284,
    80215, 147365, 215377, 281378, 86162, 348409, 217605, 418470, 81477, 348431, 89026, 221634, 221817, 419849, 420283, 78604,
    211530, 146691, 221404, 221800, 223415, 225340, 163882, 231536, 168028, 495717, 235599, 80878, 286962, 352788, 616464, 354454,
    288960, 485727, 289120, 289188, 487600, 487836, 227354, 489574, 229420, 622675, 624655, 428087, 297057, 626767, 364670, 626820,
    497717, 497720, 432221, 628861, 79420, 278661, 213045, 624642, 146452, 215320, 217222, 153626, 88295, 154307, 219895, 156028,
    157922, 354559, 158051, 424059, 78976, 147379, 147432, 80387, 81884, 147571, 147806, 215074, 84059, 149993, 215572, 215881,
    217205, 151680, 217518, 283315, 283368, 153906, 219511, 285203, 221458, 221531, 156021, 288867, 420187, 223658, 159774, 422253,
    165904, 165967, 170064, 212316, 213891, 90646, 143549, 352662, 80661, 346555, 483334, 288786, 80446, 147178, 84113, 347005,
    81752, 350749, 81011, 146828, 82667, 215183, 221695, 223740, 421921, 225685, 292930, 227495, 426026, 239648, 241685, 80809,
    212823, 86795, 217214, 151606, 157811, 81570, 219676, 156162, 225368, 487514, 225474, 483375, 221397, 233623, 303112, 237637,
    241678, 143761, 146123, 78368, 346748, 215978, 221188, 221979, 221982, 418599, 221995, 222004, 222005, 418620, 418625, 222042,
    222051, 222098, 222116, 222121, 418745, 424106, 620715, 227503, 620725, 620726, 424128, 424130, 424170, 227568, 227573, 424182,
    430261, 430262, 430273, 430276, 233669, 233672, 430281, 430290, 430293, 233687, 233689, 430298, 626910, 430311, 430315, 430316,
    233709, 430319, 436262, 632881, 436277, 436279, 436281, 436282, 436283, 436286, 442395, 442398, 639007, 145124, 81016, 94334,
    229428, 76501, 281134, 354655, 215955, 222032, 222040, 222104, 227557, 227581, 227582, 233652, 78618, 150347, 145755, 147637,
    152263, 146200, 212606, 147603, 346323, 152373, 284709, 285086, 154621, 223496, 279184, 82728, 84025, 84839, 84861, 151642,
    86441, 88183, 90296, 90483, 92518, 100403, 281380, 215200, 225549, 237618, 484043, 484346, 430130, 144254, 86418, 152412,
    152430, 417893, 156392, 442388, 106612, 212333, 147649, 82575, 215715, 285033, 350934, 420337, 225360, 422269, 362532, 495644,
    430174, 149968, 153795, 220156, 159755, 146267, 146847, 81575, 634912, 292902, 346205, 350682, 296977, 239644, 305182, 307253,
    81937, 86233, 86311, 634927, 80560, 81955, 84479, 86783, 287382, 287401, 104566, 156422, 150438, 347100, 353203, 358583,
    218104, 484318, 151560, 152368, 154161, 155715, 157877, 79568, 90661, 94275, 495684, 104508, 497778, 146222, 79260, 148321,
    430243, 430251, 80354, 84583, 86694, 86760, 88870, 163891, 79550, 146868, 291098, 620654, 428058, 80194, 82169, 215857,
    86883, 79691, 79749, 90922, 90980, 91001, 91045, 91054, 91061, 96469, 96481, 96495, 489716, 96504, 96508, 102604,
    102641, 108587, 108588, 501805, 108597, 507926, 114715, 76459, 78115, 81978, 84692, 219714, 88873, 92273, 157870, 487842,
    100424, 167988, 438310, 505865, 76532, 280967, 282886, 485750, 499767, 493572, 212614, 351047, 351122, 352531, 79003, 213085,
    215177, 84827, 217309, 287193, 289173, 235594, 616913, 215191, 159977, 282926, 285562, 157823, 421904, 362544, 144183, 279133,
    284774, 88544, 351180, 172114, 78069, 349095, 78066, 80099, 84188, 86255, 88643, 92164, 79113, 349098, 143992, 79374,
    146946, 346780, 348297, 414618, 418583, 422200, 76344, 80210, 82148, 84037, 215680, 215725, 215813, 283304, 88254, 88814,
    221194, 417811, 352378, 352443, 223297, 223308, 354663, 225590, 422273, 422287, 424054, 424061, 294995, 231430, 428052, 624678,
    430178, 432220, 432230, 237591, 434241, 436256, 309276, 147388, 153632, 145928, 147208, 213833, 86948, 88477, 90827, 353023,
    90889, 158035, 158178, 160168, 98407, 634941, 145952, 151613, 86488, 217817, 350769, 285655, 352323, 352712, 78334, 144298,
    217688, 624709, 145477, 149867, 152180, 287512, 146943, 213300, 347116, 353247, 356487, 358579, 231532, 434251, 144980, 225711,
    279096, 283165, 80554, 157983, 223678, 170030, 170082, 172117, 279173, 220145, 75787, 77941, 78309, 82755, 280856, 84550,
    84562, 282758, 88085, 219162, 219843, 285431, 88882, 221564, 90647, 92246, 354498, 420095, 616788, 223587, 489595, 144188,
    78960, 144836, 144784, 283565, 287481, 287484, 291270, 438331, 438334, 287378, 84882, 281534, 84947, 84958, 84971, 281584,
    84991, 90939, 90957, 287615, 91024, 489648, 489658, 489665, 495811, 495836, 501820, 108607, 170099, 147059, 219273, 292911,
    147732, 90214, 156103, 148334, 151733, 153720, 144462, 211373, 481380, 347034, 150484, 150509, 150514, 353098, 364769, 213951,
    213956, 350715, 364701, 366655, 144604, 79417, 145914, 143621, 147638, 154543, 82682, 82732, 84878, 419860, 487452, 491549,
    149707, 146748, 147589, 150037, 159987, 76502, 212619, 215106, 348304, 217732, 354532, 354572, 414103, 421972, 438290, 75946,
    81580, 213340, 84325, 282688, 285315, 219908, 90598, 90708, 420334, 159857, 624766, 299152, 301117, 434234, 145488, 211801,
    147572, 217753, 350375, 219780, 352552, 156066, 352778, 159794, 618680, 160039, 78600, 144720, 353138, 94515, 358608, 358624,
    364749, 144599, 212467, 80362, 413760, 88528, 92260, 485855, 159777, 98386, 282968, 422324, 145976, 80729, 279218, 281421,
    156341, 147046, 283154, 223542, 278739, 86956, 90819, 90887, 493635, 626865, 283549, 78480, 285301, 76028, 77976, 79506,
    149821, 84403, 84636, 151762, 86306, 151844, 152085, 86831, 88244, 154155, 88708, 92379, 157959, 158155, 160015, 81723,
    211778, 215773, 484310, 146624, 348463, 219747, 482064, 221775, 223688, 223702, 160105, 143677, 144095, 86700, 90515, 144413,
    145787, 147937, 79185, 145239, 156461, 156465, 156488, 156491, 156503, 156519, 156520, 156547, 487477, 489485, 76238, 78530,
    144797, 283468, 352555, 418502, 418540, 354440, 161837, 161878, 76346, 346196, 352615, 482073, 420347, 291139, 79987, 84148,
    84424, 282675, 86379, 283264, 285082, 485419, 288850, 291147, 227412, 81883, 217571, 143744, 152313, 285076, 154451, 487727,
    356765, 292996, 364587, 75957, 146745, 147688, 147817, 152234, 80124, 92183, 157866, 78970, 81595, 86197, 88184, 347121,
    296976, 150462, 79574, 215614, 157955, 170098, 145301, 148391, 148400, 148433, 283596, 287686, 90155, 144038, 81191, 152285,
    154051, 154313, 616526, 626725, 78441, 144407, 349181, 622715, 144013, 147169, 211681, 213290, 82472, 88632, 81538, 82236,
    84504, 416547, 81384, 84675, 221499, 90614, 161948, 622677, 231433, 624649, 301067, 239625, 370713, 372778, 76576, 76549,
    79459, 79612, 146330, 81326, 155978, 231545, 237620, 144374, 144071, 213483, 143799, 75846, 78123, 78158, 84139, 88701,
    90314, 94430, 100394, 489550, 90785, 215207, 215315, 346941, 350804, 221771, 354513, 362566, 215880, 215942, 217891, 144527,
    282902, 143760, 358500, 145634, 80399, 76566, 81031, 215846, 213643, 152119, 616546, 485843, 620707, 493673, 430181, 145185,
    148353, 152477, 152493, 156397, 145600, 483650, 287214, 356550, 146884, 144370, 146896, 144430, 146863, 148100, 148232, 213769,
    215370, 217928, 221406, 287019, 288904, 96301, 96342, 283082, 237571, 78098, 78185, 146867, 147058, 148169, 301161, 78074,
    78170, 82047, 78654, 147429, 144277, 149607, 152517, 146982, 418069, 79643, 213896, 82838, 86925, 86951, 90862, 90867,
    221971, 618956, 102573, 102575, 143624, 80797, 213397, 281292, 283388, 219246, 219496, 416288, 285558, 286840, 418113, 418335,
    352802, 288807, 419887, 419986, 420106, 227365, 229460, 231441, 428076, 297087, 347026, 84664, 414178, 144364, 81615, 84138,
    147425, 148274, 212628, 213831, 82777, 82822, 82830, 86895, 90838, 221956, 484103, 487871, 495789, 212118, 148432, 152541,
    147640, 213333, 414054, 221320, 156089, 221778, 158129, 618999, 489637, 235629, 280618, 153842, 221473, 157756, 291326, 303229,
    145666, 213055, 149858, 152382, 414535, 165968, 81487, 147032, 81608, 84704, 88835, 146720, 284762, 156440, 213260, 233614,
    212973, 76140, 280797, 81247, 81338, 161979, 78380, 346263, 414669, 285517, 288882, 289060, 290863, 79294, 146665, 215384,
    217767, 237669, 311305, 376842, 214015, 114698, 219871, 145515, 148006, 149786, 151953, 79110, 151801, 158064, 79403, 350221,
    144078, 360568, 76183, 82696, 86937, 92453, 78848, 148413, 283589, 283593, 147142, 75845, 76672, 79722, 79723, 156393,
    223522, 436228, 309256, 145670, 149743, 217111, 178179, 77932, 80875, 81053, 82263, 217260, 282974, 152097, 219152, 284746,
    155949, 233611, 94209, 423969, 507907, 212913, 213232, 213667, 215771, 217581, 217681, 217905, 219269, 231492, 80017, 81150,
    145321, 75863, 78235, 78345, 80185, 146448, 80975, 146792, 147009, 147273, 86099, 86128, 86598, 86737, 284763, 153761,
    88453, 285376, 88877, 483578, 418142, 90497, 419868, 288897, 419999, 92446, 94332, 356628, 424025, 292965, 163877, 165987,
    297085, 430085, 102508, 432137, 434210, 438319, 81024, 81025, 81587, 84342, 346618, 86494, 350617, 88712, 354460, 362556,
    217895, 90322, 80013, 215419, 348198, 350992, 211543, 145433, 147658, 152088, 149608, 155956, 422024, 426001, 143730, 145429,
    213230, 220017, 426011, 626735, 626791, 434216, 211812, 350462, 352261, 352435, 352514, 352920, 354392, 485538, 354548, 289040,
    289277, 356710, 428113, 283324, 79679, 82789, 82821, 495785, 368761, 145631, 368707, 78554, 78705, 146393, 219315, 144122,
    148395, 156622, 156623, 156632, 156654, 94299, 146668, 212648, 221741, 78466, 144426, 281420, 86113, 418456, 79996, 81958,
    145452, 145815, 146092, 147736, 148007, 152071, 155665, 225608, 161797, 172043, 86023, 82893, 349157, 619003, 88305, 90252,
    90487, 92190, 96406, 167973, 144116, 88710, 144394, 84141, 156303, 149952, 282973, 75836, 145672, 212993, 215152, 282636,
    217414, 350734, 156044, 420114, 143409, 147101, 148263, 221567, 354565, 620626, 76326, 211736, 351095, 156196, 159844, 426009,
    495634, 168006, 497718, 630860, 501765, 76605, 146313, 638993, 223242, 281568, 484171, 287614, 489689, 108600, 213443, 213463,
    213614, 213671, 346556, 215910, 215952, 221298, 78786, 144009, 143735, 145577, 148236, 78979, 79957, 80119, 82505, 86179,
    86300, 86332, 346426, 153613, 156286, 293001, 366629, 91090, 82250, 283170, 284929, 90589, 145394, 418756, 418786, 418788,
    299263, 217786, 90476, 281405, 221781, 82921, 144918, 146158, 213694, 279239, 149727, 151724, 86561, 283397, 153616, 90413,
    156272, 168075, 290929, 432189, 80794, 154491, 90868, 158077, 364709, 172084, 78551, 82940, 348724, 144857, 280873, 217234,
    418280, 76225, 76759, 80052, 82429, 82712, 211106, 217741, 225406, 80278, 76634, 79322, 76084, 212659, 144679, 80477,
    152215, 155845, 92170, 149696, 90694, 102531, 84349, 215568, 149845, 81698, 81810, 76386, 82377, 143822, 79297, 145949,
    81039, 81142, 278970, 281286, 350512, 88676, 88998, 420192, 92551, 358426, 620646, 360492, 100428, 428129, 211522, 79778,
    78492, 81743, 219337, 354387, 170103, 149745, 144011, 146565, 147786, 156125, 156232, 354439, 144458, 149764, 144855, 77937,
    79476, 80308, 148197, 81702, 150370, 151572, 154581, 618757, 436242, 503826, 78732, 144046, 150292, 152091, 217825, 156086,
    616911, 630833, 505876, 213756, 79094, 211967, 283189, 219240, 227453, 284730, 620604, 80024, 80585, 82630, 82637, 279257,
    90171, 90797, 418218, 354505, 420300, 78697, 145962, 212474, 149644, 86263, 152086, 285023, 350956, 418163, 352676, 485528,
    354488, 485797, 354737, 422117, 292887, 495638, 497774, 368724, 282897, 283425, 350411, 285668, 285669, 288925, 618732, 620661,
    144937, 147671, 84216, 86416, 284935, 79585, 144057, 144761, 81690, 88884, 290870, 102528, 76095, 80406, 482058, 156001,
    157883, 618618, 628808, 143366, 145489, 146179, 147139, 147363, 148181, 149909, 151973, 156227, 157737, 161885, 362552, 78984,
    420344, 151571, 154279, 154511, 160071, 218038, 218044, 218047, 218063, 90762, 222147, 222152, 222154, 222161, 222194, 222207,
    229497, 632849, 211886, 279465, 152204, 157965, 161854, 499734, 374800, 145702, 152262, 153686, 80489, 147477, 86073, 86293,
    217860, 285636, 90262, 160042, 227451, 626797, 497768, 78818, 79263, 622650, 217658, 352563, 278640, 217815, 219820, 354395,
    79118, 81407, 215487, 287168, 420286, 364643, 628845, 368743, 144209, 81762, 82265, 82287, 82710, 215053, 282628, 151617,
    86364, 152040, 152049, 86776, 283582, 153638, 285310, 154602, 155760, 286897, 155865, 155891, 156210, 156311, 157837, 157893,
    157923, 158102, 420265, 158183, 94287, 159835, 356493, 159947, 160000, 225639, 94615, 422377, 161937, 491524, 426022, 491597,
    491600, 165922, 430090, 167994, 170034, 628797, 235616, 499712, 434189, 237592, 303161, 303166, 239631, 436274, 176128, 178196,
    280993, 353199, 221292, 90228, 221900, 223523, 290944, 618755, 626815, 626857, 499833, 505857, 374799, 144691, 79274, 154386,
    282941, 414184, 219300, 221260, 286854, 352907, 287412, 485410, 288917, 289142, 487525, 618637, 618728, 618925, 489632, 495745,
    432178, 497760, 303159, 499812, 370725, 440340, 346761, 96285, 98342, 144508, 78188, 80388, 84486, 428117, 144344, 279306,
    144913, 150325, 434229, 239638, 148470, 219931, 221456, 420335, 225525, 294927, 303108, 305173, 90199, 145977, 215530, 79795,
    147122, 147143, 217372, 231456, 81616, 82522, 88311, 88890, 292948, 213094, 217666, 217760, 285292, 221680, 223640, 289254,
    225401, 231530, 145626, 86316, 86866, 349047, 90585, 90809, 92565, 144285, 147036, 84542, 350569, 155937, 289111, 290896,
    94588, 106568, 154036, 157799, 159999, 489542, 618973, 76556, 151685, 151703, 424016, 163921, 163931, 165975, 174094, 176140,
    178194, 212980, 75796, 78122, 213292, 84187, 84726, 219736, 219806, 418151, 223477, 289061, 353094, 82486, 152218, 152151,
    286727, 288780, 301133, 76484, 76789, 147534, 213561, 149670, 150144, 150189, 150277, 217768, 153815, 219500, 352275, 418459,
    418461, 157761, 223589, 165894, 301145, 366694, 80018, 82481, 82614, 82674, 84083, 84862, 88294, 88721, 88812, 221301,
    90231, 90312, 223445, 225342, 225457, 94481, 225701, 227347, 233613, 77942, 80087, 146587, 90123, 356477, 76110, 94331,
    94400, 96283, 144204, 75903, 146285, 150151, 144665, 150164, 151899, 374812, 146191, 88575, 154540, 155792, 92554, 356538,
    422114, 94506, 94526, 161915, 491596, 102503, 499815, 501779, 213571, 143417, 144550, 81245, 82911, 149951, 354750, 299107,
    497790, 149567, 152386, 153604, 89020, 159749, 76707, 79522, 212148, 80982, 151692, 484338, 622716, 90300, 485821, 159964,
    279183, 86968, 221195, 148442, 217496, 362550, 80009, 80394, 211992, 84317, 78982, 76116, 79885, 82146, 86553, 88634,
    88863, 90735, 94286, 241708, 634930, 147482, 149862, 86092, 221841, 346322, 347060, 417855, 79518, 81846, 237649, 145136,
    146202, 346441, 217591, 628804, 90345, 233557, 146404, 150278, 151942, 616626, 223652, 423997, 299134, 215151, 280827, 287060,
    288938, 78063, 144110, 79990, 215148, 86111, 217507, 217531, 88667, 92603, 94368, 96260, 278814, 285264, 285402, 82235,
    84544, 88374, 90473, 143938, 151750, 350404, 352878, 80830, 279021, 221242, 92198, 157961, 158135, 426034, 428088, 145367,
    283615, 79592, 211612, 147642, 149766, 215437, 159952, 168066, 157868, 143583, 149813, 102481, 144874, 350276, 80152, 84126,
    86427, 351098, 155962, 147174, 485817, 144728, 424077, 82353, 420311, 430225, 76072, 147188, 149868, 217558, 154158, 353030,
    225740, 487898, 283094, 350565, 144704, 213392, 215290, 88064, 229400, 294952, 98382, 309278, 150439, 156530, 79281, 145334,
    148407, 77943, 80167, 80471, 81726, 82344, 86164, 88570, 90599, 354615, 289127, 485827, 493617, 145933, 212510, 213261,
    147953, 148029, 215252, 215704, 217529, 153643, 284961, 157768, 157838, 225295, 291132, 348882, 79038, 145167, 148295, 148308,
    148350, 156420, 418567, 422335, 299178, 430253, 213770, 80696, 283224, 284766, 212933, 213730, 217145, 217365, 416708, 79216,
    80266, 231513, 348202, 76604, 80709, 81419, 81529, 82212, 147749, 213412, 82349, 213743, 86659, 154501, 221633, 92449,
    354782, 422309, 430221, 104549, 76749, 145279, 148460, 279532, 285241, 353080, 283071, 80057, 79763, 81046, 81843, 349135,
    87012, 353226, 353233, 150061, 154376, 154510, 420179, 622629, 624739, 217911, 144290, 145887, 147771, 150080, 153910, 156009,
    78737, 145727, 84392, 213234, 215491, 283035, 76540, 80213, 80329, 147486, 81996, 213111, 82643, 149506, 88091, 88283,
    219361, 219658, 154149, 154299, 90496, 287122, 223251, 225498, 225715, 292971, 170106, 75875, 146549, 146851, 82510, 84038,
    217157, 282735, 217670, 152275, 90499, 92349, 92592, 92614, 294949, 297059, 618498, 145876, 80501, 152269, 154338, 223277,
    223697, 489624, 80180, 347079, 168131, 146699, 213819, 211589, 82316, 86279, 225752, 241722, 481417, 145812, 144356, 152406,
    156343, 287418, 156352, 156353, 156360, 156362, 422330, 168104, 172145, 172156, 86934, 86942, 618933, 618935, 80302, 88110,
    418807, 291317, 421984, 92650, 293008, 301105, 311307, 174093, 634910, 634913, 77920, 80005, 217213, 86509, 86586, 88728,
    347066, 416692, 143714, 211358, 620642, 213041, 283030, 283531, 291093, 223255, 144593, 144101, 143704, 86265, 88273, 282921,
    219687, 148377, 283584, 283617, 104518, 75963, 278851, 346844, 348633, 484022, 364544, 364546, 366613, 84120, 86687, 154489,
    354671, 102507, 151806, 291099, 110617, 79221, 146502, 418111, 422067, 491525, 372775, 426078, 146781, 485429, 291005, 493691,
    78092, 84030, 215844, 360530, 79589, 223656, 309252, 278743, 278939, 90866, 79093, 78134, 78996, 81721, 81992, 82192,
    82351, 83980, 84319, 86579, 86646, 86812, 86826, 88431, 88663, 88750, 90784, 235646, 149709, 215788, 150290, 145919,
    143471, 145412, 147619, 152118, 147450, 213665, 82624, 279325, 215221, 215478, 215493, 215570, 217117, 151884, 217442, 86455,
    348789, 156054, 231501, 235619, 152354, 153693, 221209, 155812, 424078, 102549, 145212, 148330, 148338, 630895, 356857, 364790,
    364791, 80584, 146120, 211656, 81368, 146904, 82689, 348638, 86090, 102495, 155853, 172147, 172150, 172154, 80361, 81508,
    82041, 82064, 82279, 84161, 84209, 84304, 84850, 88330, 88436, 89008, 90753, 352341, 147852, 151815, 616548, 104449,
    104474, 110601, 505863, 217975, 158098, 94646, 94652, 229471, 360554, 172113, 145232, 350527, 350836, 156524, 157981, 420323,
    358524, 311314, 143676, 146354, 79779, 211028, 364788, 147595, 86719, 352757, 96305, 81281, 147268, 414335, 152306, 285100,
    416674, 484089, 283440, 223493, 75884, 144314, 81656, 86270, 176141, 112661, 79998, 80869, 81069, 82004, 88413, 94604,
    76545, 78689, 280898, 158039, 76255, 84270, 84337, 86628, 284948, 285386, 92182, 92237, 154246, 364582, 223268, 428115,
    80370, 80742, 423944, 430119, 434230, 108548, 436255, 438275, 440320, 84372, 219765, 287469, 78993, 346135, 352876, 75868,
    145537, 145589, 84019, 152075, 86752, 219267, 350383, 154416, 221554, 420071, 157999, 159879, 499808, 221912, 499836, 213640,
    616451, 356436, 618585, 628780, 372754, 144365, 279004, 416566, 416698, 80698, 146837, 241710, 347065, 349034, 349046, 356812,
    491615, 499828, 144626, 279454, 217933, 349084, 484153, 484280, 79949, 82046, 149664, 150081, 348667, 88382, 154590, 155789,
    156101, 157743, 157962, 160013, 172082, 143807, 78482, 151881, 283029, 156064, 291056, 366706, 174109, 78681, 78703, 76012,
    78396, 211561, 82412, 150162, 152146, 154166, 88682, 221334, 98360, 213795, 144789, 414599, 146642, 154173, 146630, 147553,
    217226, 151938, 350652, 219612, 156258, 354423, 288926, 158127, 163899, 233552, 368747, 348188, 153819, 212877, 347052, 217114,
    219523, 223427, 76644, 144489, 79519, 145590, 218067, 154077, 158088, 225366, 225391, 356520, 225505, 225533, 225584, 225619,
    225697, 493601, 430148, 143764, 90654, 289177, 78479, 79636, 350227, 352532, 352845, 628776, 366640, 143976, 146235, 81657,
    279386, 215187, 215448, 150029, 215643, 348311, 151931, 152037, 283334, 283340, 284758, 219424, 88781, 221544, 419952, 354447,
    223629, 290866, 422033, 422082, 291125, 422275, 489579, 426014, 428044, 428060, 428074, 362543, 624705, 362596, 231541, 231542,
    233499, 364652, 301179, 237635, 630873, 499816, 144156, 78109, 143851, 82574, 88813, 90302, 90439, 92280, 92281, 92487,
    485703, 96303, 102440, 368720, 307225, 145880, 152300, 279434, 348530, 287420, 287426, 418511, 287449, 354448, 356700, 145625,
    147596, 152162, 154333, 156211, 151698, 84494, 88991, 280865, 350537, 419973, 420142, 420304, 422093, 78525, 347031, 77931,
    145602, 80343, 80510, 81504, 213294, 84364, 84422, 414504, 217916, 487654, 145606, 86266, 628789, 212724, 147617, 149788,
    219694, 154486, 145055, 217396, 284853, 438294, 440325, 440349, 114696, 78156, 212421, 215744, 92169, 94312, 94519, 144457,
    281416, 281478, 283427, 346926, 144416, 80157, 80748, 283086, 212038, 76388, 79817, 80498, 213096, 149680, 215467, 287321,
    223548, 620629, 237654, 76353, 299180, 82664, 346320, 84877, 419990, 174129, 219849, 83985, 150299, 351011, 630851, 78850,
    147238, 217108, 80825, 81530, 215311, 416226, 156179, 354616, 163873, 362602, 79435, 368765, 278700, 414700, 284965, 420244,
    422234, 499835, 80378, 211642, 215321, 77861, 80171, 82115, 84146, 86387, 86801, 88837, 425999, 434180, 436245, 78875,
    146317, 81666, 217863, 156284, 75793, 78195, 80691, 83994, 84360, 84496, 150238, 348224, 86168, 86242, 86405, 88083,
    90388, 90538, 90701, 92620, 96353, 104496, 106600, 86153, 292934, 148277, 491645, 495864, 495868, 88111, 628821, 143520,
    146514, 150274, 348713, 152170, 153712, 156061, 78963, 78141, 144882, 156302, 80259, 149606, 419951, 145663, 282819, 422338,
    163949, 78581, 78014, 144107, 80640, 81042, 217178, 76706, 146569, 213314, 213680, 282835, 217802, 153789, 220020, 286795,
    223556, 289268, 290868, 290909, 297010, 144008, 81346, 147135, 281062, 418372, 147773, 84502, 215882, 144433, 350778, 487830,
    495738, 155808, 145170, 148345, 148349, 283471, 291261, 144446, 348586, 418176, 82230, 353084, 353183, 358634, 364754, 370741,
    376859, 145286, 156541, 156546, 156589, 280959, 221409, 76054, 84036, 415785, 426038, 430093, 350921, 110597, 82609, 280799,
    84831, 287344, 78597, 80702, 81252, 82810, 86939, 350217, 282731, 212668, 150010, 348673, 153871, 351080, 354699, 81820,
    149662, 151963, 221185, 79444, 146524, 147639, 82213, 150165, 281509, 281542, 281565, 281587, 217684, 221346, 287572, 287588,
    287603, 287660, 484282, 223307, 223450, 354569, 354743, 289246, 225433, 487655, 225600, 489493, 358457, 293034, 293081, 293086,
    293112, 293113, 299110, 168061, 630818, 288806, 80421, 212977, 144838, 168109, 143981, 158200, 163859, 77879, 81241, 150312,
    76106, 145196, 417886, 418436, 168145, 143643, 147819, 217140, 151616, 151955, 157697, 160108, 366675, 76010, 82033, 84845,
    145381, 346429, 284852, 352709, 215856, 80579, 219173, 219550, 154585, 487598, 282836, 423936, 81401, 79784, 279219, 278860,
    144869, 225770, 281322, 144655, 147941, 421987, 167964, 279187, 287062, 80223, 145879, 147510, 149958, 88192, 158187, 347102,
    81257, 154105, 89013, 157798, 92661, 102516, 434217, 146521, 143696, 145786, 149789, 151651, 217418, 221574, 221579, 221776,
    223282, 223647, 158117, 227399, 227464, 283426, 636938, 144341, 79199, 362560, 80867, 212099, 84224, 152158, 221813, 223439,
    225550, 221980, 221992, 222010, 222020, 222033, 222041, 222044, 418666, 227497, 620734, 227539, 424174, 424179, 626893, 233692,
    430302, 430308, 233703, 436280, 442396, 215126, 81874, 80295, 279031, 215973, 216006, 216029, 216051, 222036, 222052, 222067,
    222124, 418746, 227498, 227545, 227550, 620778, 227576, 227577, 620796, 353210, 162041, 212210, 150428, 282872, 156569, 156570,
    156593, 172121, 88267, 432177, 84972, 90964, 91052, 96426, 96432, 96444, 96490, 96500, 102581, 102593, 102601, 102602,
    102610, 495844, 102635, 102636, 108599, 501816, 108602, 108603, 114710, 507932, 81182, 626691, 350514, 76177, 144943, 217522,
    219562, 144279, 287346, 349063, 284708, 287350, 76476, 78390, 96295, 144793, 155854, 155999, 163874, 167939, 233491, 626741,
    169991, 80417, 80766, 283312, 144771, 346366, 414645, 352969, 225718, 225724, 491626, 143693, 76584, 81917, 279365, 81060,
    82206, 88656, 88794, 82612, 144745, 81188, 88875, 84335, 86629, 88996, 92410, 92532, 92649, 98367, 146898, 160097,
    76362, 78272, 144658, 81224, 82164, 88065, 92163, 78858, 90767, 213420, 213452, 213618, 149538, 149548, 215613, 217549,
    217847, 154084, 155922, 418384, 158122, 620583, 424007, 634885, 143988, 80285, 88343, 90291, 90370, 92404, 94566, 161956,
    78677, 76153, 84749, 86160, 348362, 88449, 92321, 92388, 92428, 303185, 106603, 84378, 217303, 88489, 286851, 161800,
    167975, 172086, 174111, 176131, 178176, 278942, 278748, 76582, 81848, 285068, 219656, 223481, 279420, 86624, 418424, 78659,
    356359, 94499, 481776, 79048, 82175, 353017, 353020, 356806, 148245, 144890, 484322, 484324, 144268, 414163, 219952, 282662,
    285456, 497738, 145767, 146947, 147699, 283637, 155823, 426104, 79708, 281511, 84906, 84923, 84980, 91007, 293060, 293066,
    78073, 211564, 81106, 150459, 88637, 76275, 80968, 79821, 212386, 82855, 82897, 414320, 491567, 146810, 146822, 278904,
    148152, 285678, 156104, 76633, 79504, 352666, 80562, 84185, 84555, 90633, 144388, 422390, 78254, 78801, 147776, 90510,
    106535, 108555, 305165, 176134, 110633, 76640, 92311, 278631, 217386, 152317, 154152, 154176, 155911, 157942, 157966, 159818,
    159920, 159934, 225508, 160156, 161824, 366718, 79668, 218020, 153611, 156070, 221899, 484095, 221961, 157958, 160104, 229479,
    78688, 90771, 353227, 147243, 283012, 221343, 352556, 426010, 636931, 78571, 78839, 286990, 79923, 219689, 159825, 624758,
    82586, 146800, 82180, 282775, 152000, 153808, 352306, 156008, 166005, 166006, 167963, 172099, 143929, 81514, 77923, 144398,
    81680, 286772, 100409, 301156, 146649, 90324, 145006, 147370, 147394, 148229, 420162, 78046, 146436, 81492, 82329, 84661,
    88141, 155750, 286875, 221639, 219698, 84899, 86350, 106567, 279194, 211381, 152348, 349094, 420342, 487608, 422232, 489536,
    497687, 81824, 146060, 417874, 352682, 616575, 616615, 354764, 423972, 143395, 146477, 147137, 147664, 280753, 284787, 420028,
    420275, 485862, 354798, 487829, 293011, 368698, 144087, 348935, 422032, 212726, 353112, 352356, 144186, 282742, 416059, 150436,
    156485, 156602, 168126, 168175, 616953, 147424, 147428, 148293, 485526, 231467, 215842, 86254, 92193, 217899, 352913, 146079,
    146908, 154438, 76315, 82431, 215776, 493663, 495739, 213645, 281309, 78016, 78901, 84761, 487557, 489586, 426053, 100407,
    79205, 144901, 149508, 152175, 153672, 154055, 156152, 484028, 157713, 159792, 161862, 372797, 79367, 150480, 351221, 146027,
    81115, 217092, 217101, 152099, 152239, 153722, 219774, 221361, 156249, 616617, 237625, 350754, 352515, 352551, 354350, 354458,
    146941, 149552, 215397, 150196, 153625, 351156, 221780, 79792, 213964, 87007, 78968, 219399, 78545, 231480, 157974, 172131,
    80690, 213628, 215269, 88253, 220067, 221328, 158090, 487704, 160143, 294993, 100366, 102475, 106514, 305166, 372742, 309266,
    374803, 79304, 79637, 213892, 82837, 217962, 86900, 217974, 217991, 349092, 90808, 352971, 90829, 90835, 90850, 90861,
    353033, 90892, 356802, 356808, 98401, 360551, 626851, 495780, 102569, 626859, 495790, 368756, 372796, 78524, 78826, 212090,
    150365, 420051, 366660, 432199, 212388, 283646, 489481, 146632, 347014, 217315, 219207, 219633, 221599, 352948, 420122, 354678,
    368695, 348386, 368744, 281080, 211891, 146111, 148477, 279549, 143469, 348375, 151934, 421971, 145494, 147559, 346874, 352482,
    165957, 80491, 86563, 217695, 86703, 88186, 90713, 225479, 77983, 145362, 484064, 146308, 215556, 215598, 413699, 152061,
    285380, 287210, 288983, 289116, 289119, 145135, 79218, 80079, 301146, 150174, 152327, 90696, 146051, 147692, 149720, 217385,
    154151, 219906, 352328, 221510, 82863, 84162, 86108, 618771, 98378, 630846, 616500, 620574, 346928, 88615, 485478, 616663,
    94367, 98354, 144047, 217121, 143617, 211646, 80919, 146551, 146930, 84165, 92250, 148233, 616710, 485806, 618856, 75779,
    82498, 84086, 84634, 86721, 154027, 350984, 90495, 90705, 92439, 297017, 364625, 77891, 80130, 81502, 147463, 147518,
    82326, 149747, 84322, 84643, 152249, 86846, 154056, 154356, 94288, 100432, 153827, 80037, 82437, 76669, 233574, 622627,
    346524, 348668, 295003, 297047, 145414, 147955, 149702, 151721, 151763, 217382, 153747, 88396, 155723, 156055, 156256, 92203,
    354741, 158196, 94338, 159995, 356759, 79393, 147845, 354314, 157712, 168068, 170109, 165998, 76002, 156124, 76527, 90315,
    616574, 149987, 146006, 278993, 346186, 280886, 286809, 221877, 354495, 289080, 159916, 160061, 299073, 76499, 144133, 212903,
    213675, 285671, 144686, 414532, 144338, 78956, 146237, 146493, 282778, 288853, 289114, 78252, 212223, 285238, 352406, 288779,
    81089, 348643, 80224, 348179, 351119, 146471, 152392, 211153, 213653, 279147, 215798, 350878, 354549, 78962, 151589, 153787,
    155892, 159906, 168069, 147364, 82557, 147403, 156322, 78255, 288774, 80253, 80678, 80692, 82274, 82357, 213706, 82707,
    84625, 86355, 86765, 88978, 90274, 92235, 94406, 94528, 94621, 102443, 144417, 79834, 82899, 82905, 86967, 218070,
    87027, 160134, 349083, 88723, 212091, 78773, 79034, 155972, 221840, 354315, 281543, 347103, 281599, 90995, 489644, 489690,
    495832, 221867, 634915, 143980, 76196, 212587, 285266, 350870, 351151, 418184, 358506, 145043, 288800, 434227, 143381, 146852,
    147680, 145750, 80846, 81737, 219227, 88225, 88403, 352606, 354332, 223361, 354463, 354562, 485874, 225471, 423986, 292929,
    358489, 227429, 294916, 360476, 98341, 294989, 229455, 294992, 493643, 100451, 231549, 364549, 366601, 368674, 106558, 372783,
    81903, 150152, 154557, 158025, 485869, 360451, 145929, 356623, 358622, 370733, 215586, 150198, 151877, 217623, 154456, 155993,
    157906, 161896, 499811, 84338, 86106, 168125, 168143, 174127, 150173, 481370, 483642, 356402, 143551, 82824, 152197, 349020,
    353000, 90899, 299042, 81145, 145083, 81775, 147404, 280663, 287250, 235607, 299102, 155844, 426005, 79186, 146384, 348229,
    152148, 157914, 159897, 80639, 221897, 630904, 78036, 79900, 81176, 86493, 76321, 145398, 147319, 82927, 86088, 221934,
    221939, 353043, 356800, 229480, 75800, 77967, 78264, 78676, 84011, 84728, 86344, 88223, 88317, 90643, 98337, 78101,
    81258, 81264, 219278, 146218, 147594, 149588, 151745, 217485, 217717, 219729, 154248, 219899, 154483, 156007, 483948, 223558,
    618803, 231459, 145561, 80779, 82752, 350521, 144401, 76342, 217665, 285249, 88742, 219993, 90419, 221701, 489529, 146107,
    168033, 626844, 414151, 416292, 102595, 144957, 178177, 212817, 76528, 80258, 86729, 79019, 80608, 81550, 88328, 89009,
    90453, 90583, 90681, 352857, 618785, 620620, 628844, 76596, 86120, 283220, 217880, 415845, 220114, 417985, 483989, 354378,
    485491, 288986, 420270, 421950, 423955, 213983, 80173, 212417, 81496, 217199, 303168, 150371, 152389, 217929, 352936, 79141,
    84793, 144767, 356853, 481921, 87001, 283628, 299259, 76383, 82410, 78317, 80288, 80368, 82131, 84106, 348468, 86352,
    88076, 90524, 90552, 354402, 94513, 297027, 299079, 628826, 79656, 88921, 90725, 92596, 92600, 149805, 281465, 92213,
    100475, 368681, 147525, 151643, 213065, 281459, 282787, 485611, 215823, 217945, 217969, 221924, 221927, 491618, 495775, 495782,
    499826, 147673, 150269, 217824, 284738, 154393, 154423, 154446, 155914, 156130, 157829, 157973, 157998, 158160, 223704, 159949,
    159965, 78668, 279182, 285421, 288889, 421915, 144783, 84772, 146526, 150126, 145789, 147893, 148243, 279528, 151891, 154514,
    287728, 78306, 211410, 281428, 285686, 286811, 489552, 622640, 229457, 239630, 243730, 309267, 144443, 286933, 299159, 368648,
    303173, 307214, 79198, 76529, 79472, 487914, 78567, 213766, 215830, 285049, 221231, 79135, 497695, 145214, 280939, 150440,
    287124, 484339, 162006, 180250, 212813, 94671, 94678, 147361, 148176, 145322, 278582, 217780, 418114, 289259, 495712, 223441,
    231494, 80034, 86669, 79148, 144678, 349159, 143506, 146756, 280656, 150402, 217801, 284750, 287265, 350664, 144432, 144614,
    287366, 79961, 88347, 352947, 78793, 80336, 212920, 82358, 88427, 90343, 354334, 419918, 289011, 94594, 96358, 358550,
    430117, 106571, 150166, 281398, 481858, 79529, 220100, 144162, 212898, 81831, 82603, 86057, 157757, 77838, 78477, 484106,
    146435, 347022, 221362, 223362, 231508, 78847, 86869, 349035, 217981, 349055, 86959, 416383, 353029, 94677, 98397, 98414,
    495776, 106617, 79558, 82652, 148389, 84879, 483488, 168016, 80427, 217479, 221232, 90591, 98380, 102430, 499727, 145901,
    215709, 291036, 144505, 144703, 211146, 219498, 220084, 221572, 76264, 78351, 284731, 219373, 219758, 219872, 221385, 90513,
    288893, 291013, 356824, 372794, 79609, 283438, 491591, 81004, 82743, 84849, 82423, 148195, 151553, 217094, 151681, 219199,
    153718, 153826, 153851, 154301, 154574, 154582, 155712, 418131, 156338, 147715, 149639, 149681, 215499, 153715, 153970, 350705,
    154204, 155658, 157772, 288984, 420268, 160054, 161939, 364677, 366684, 172055, 79104, 79330, 144768, 346158, 285672, 145601,
    80281, 80706, 148075, 84170, 84701, 86490, 283322, 88442, 90545, 78076, 219427, 144045, 346939, 98390, 295009, 299175,
    143441, 84627, 423967, 212833, 281161, 215869, 485822, 418804, 291301, 145287, 144994, 79584, 145538, 81864, 147670, 147750,
    213697, 82656, 148215, 348997, 153835, 154563, 154587, 352418, 352805, 354379, 356672, 213897, 217946, 364707, 364715, 284868,
    211517, 483807, 222155, 222175, 222196, 229495, 150011, 348351, 350996, 157938, 148336, 152421, 354785, 165977, 146722, 154419,
    146145, 147754, 159836, 155930, 156470, 156513, 156586, 168142, 180248, 364688, 84613, 288852, 628741, 432252, 241679, 80068,
    146075, 147095, 148118, 86781, 88616, 88640, 90375, 92406, 92430, 94282, 94384, 94398, 94620, 96288, 163842, 151658,
    282826, 144055, 222019, 620770, 233659, 430304, 442393, 80527, 287353, 349023, 217978, 353012, 90893, 346517, 354585, 79153,
    80430, 82273, 80805, 278903, 215481, 86647, 418304, 420053, 485777, 487566, 290969, 487624, 291089, 422294, 423993, 489543,
    358609, 428047, 430169, 364646, 430202, 233597, 145296, 418771, 295027, 295035, 90402, 484236, 484269, 489661, 624684, 100401,
    106605, 78673, 352680, 616712, 211971, 78753, 145191, 219192, 219441, 154613, 146518, 278944, 291023, 292889, 81577, 84433,
    86365, 86663, 217812, 88724, 423952, 436231, 76434, 283613, 76568, 144289, 358459, 81583, 82902, 82913, 150021, 90856,
    80020, 82204, 279070, 483874, 212584, 84107, 350865, 86690, 88153, 88479, 90441, 92396, 94360, 77994, 144513, 80952,
    82315, 82407, 88189, 92307, 634890, 634911, 636936, 86284, 283470, 287419, 418510, 495621, 497673, 499746, 503855, 278674,
    215534, 144265, 366704, 221843, 143894, 153852, 211307, 217258, 282990, 86763, 352280, 418257, 223270, 287405, 79499, 212138,
    144202, 484330, 144673, 76609, 212142, 147372, 86360, 156444, 156456, 156484, 79369, 283127, 78715, 212837, 81823, 82685,
    82692, 151974, 90783, 159859, 144492, 217198, 81314, 212900, 211170, 80998, 147439, 82277, 348863, 155967, 157765, 94634,
    165889, 165909, 165912, 217921, 362528, 77958, 80309, 80831, 84005, 92579, 94249, 94493, 94579, 98364, 235637, 499718,
    630822, 505858, 114700, 84296, 84833, 88193, 154239, 154400, 90143, 90803, 92346, 92524, 159815, 82644, 215883, 80685,
    147285, 215364, 217363, 364729, 76611, 78307, 144258, 213925, 223739, 618726, 150350, 152379, 286765, 146315, 148288, 346386,
    352390, 354453, 618580, 81977, 413735, 144960, 155650, 79254, 75959, 78406, 84312, 84788, 348454, 86695, 90547, 80789,
    212786, 279078, 284878, 88850, 418233, 487490, 360504, 491578, 144732, 79292, 145000, 145201, 80282, 148315, 147438, 150449,
    225383, 77833, 79925, 81614, 86648, 88892, 90258, 163939, 442387, 79063, 144912, 143645, 78864, 148110, 154349, 155838,
    155975, 157816, 157817, 158023, 161839, 167976, 434256, 372761, 78003, 81875, 82393, 213747, 217361, 217752, 217889, 284875,
    220132, 220133, 223389, 421893, 620688, 628785, 84014, 146028, 211974, 146978, 147888, 215209, 215597, 219912, 220046, 155928,
    426044, 233553, 143618, 146829, 283250, 86774, 481831, 156057, 223530, 348703, 352831, 106518, 144123, 217870, 88735, 299055,
    144128, 150339, 79853, 81624, 213946, 349133, 79242, 81238, 280877, 350605, 420127, 426045, 432226, 434261, 76156, 79298,
    353145, 78620, 347001, 76723, 80938, 146474, 79058, 146886, 284729, 219557, 88763, 285430, 288866, 293002, 79976, 82439,
    426031, 144773, 279061, 79905, 82409, 84287, 90491, 92277, 92572, 94233, 96308, 96417, 98351, 108546, 638991, 418486,
    144036, 145456, 148227, 144504, 145179, 152434, 152435, 152466, 152480, 291274, 168111, 215254, 415775, 416137, 144157, 283019,
    145069, 281471, 350538, 285684, 150457, 144106, 79014, 154182, 356401, 358434, 100374, 438307, 78955, 146420, 281489, 281518,
    281520, 281523, 281528, 281560, 281580, 352616, 287530, 287532, 287551, 287553, 287554, 287571, 287596, 287631, 287636, 287653,
    287663, 287671, 293038, 293043, 293068, 293089, 489717, 293114, 489723, 299211, 501809, 81266, 81422, 149774, 84274, 149885,
    84398, 150114, 84631, 150183, 151711, 86699, 152270, 153797, 88491, 154038, 154201, 88817, 154546, 90446, 90554, 156184,
    156186, 156207, 157889, 158119, 159888, 159910, 159911, 94466, 94574, 161859, 96387, 170068, 81123, 84682, 145198, 75914,
    78505, 146321, 82530, 215138, 163849, 305168, 79878, 82419, 84166, 86185, 86227, 86590, 88211, 88459, 90187, 90519,
    90720, 289205, 92660, 94459, 291223, 144031, 84794, 352381, 634916, 90663, 221779, 156526, 241701, 213787, 348251, 283451,
    78615, 281057, 416743, 348921, 422284, 145156, 148304, 148323, 148363, 152407, 156351, 299183, 98357, 213201, 218034, 147349,
    76093, 149915, 149982, 156098, 156181, 156281, 156290, 158083, 213715, 161950, 162029, 143646, 146770, 147906, 348410, 157872,
    423946, 79474, 145774, 211595, 213468, 219652, 219711, 156204, 161857, 227437, 96411, 163844, 229404, 163917, 98383, 362571,
    233572, 79667, 146301, 158120, 499822, 503865, 218039, 88122, 144039, 144050, 149614, 144439, 146277, 212657, 84713, 497714,
    80606, 76700, 79721, 281541, 489641, 489671, 96498, 299248, 501812, 79921, 80264, 147195, 81940, 82092, 82610, 84351,
    84813, 86064, 152470, 152478, 88357, 88896, 88984, 89048, 90799, 92381, 225626, 227440, 163965, 430215, 168184, 145874,
    346726, 220150, 81194, 153927, 223505, 225370, 227480, 76232, 213163, 279534, 311303, 80765, 82702, 282722, 284966, 287157,
    288832, 289148, 92584, 291246, 81300, 81739, 90430, 287061, 352735, 90617, 92219, 487765, 96343, 96350, 144667, 146730,
    148297, 283500, 283506, 283553, 283557, 414630, 287478, 287487, 291257, 422346, 160205, 291278, 303228, 81515, 147091, 144360,
    76343, 78884, 81370, 84092, 217341, 77963, 78789, 79323, 213403, 219460, 219965, 221795, 217652, 285689, 80312, 82150,
    88170, 213272, 143776, 82934, 483922, 279030, 350249, 360502, 364557, 212193, 283331, 153707, 156069, 159982, 79496, 211246,
    79912, 81475, 81563, 151780, 86732, 153807, 286805, 90391, 90560, 90570, 156325, 104550, 78637, 282640, 285494, 285626,
    144727, 79449, 281194, 283106, 346836, 225438, 79196, 237589, 237612, 145158, 287463, 426084, 231535, 100466, 626720, 233518,
    366631, 237600, 630885, 434299, 634887, 241672, 241674, 634896, 176158, 176161, 634925, 505868, 636952, 638981, 78001, 80339,
    237568, 76391, 211436, 145971, 151943, 353220, 353250, 420125, 420250, 618977, 622708, 102437, 213225, 147766, 79540, 79551,
    84856, 86709, 88798, 88853, 143919, 145994, 281455, 78923, 282807, 294921, 149612, 350338, 237677, 144676, 145157, 215415,
    152247, 352655, 418438, 418521, 160128, 291263, 493674, 75942, 79249, 82659, 90623, 211952, 77877, 78023, 81280, 82013,
    278806, 626697, 78472, 81008, 212288, 152297, 284819, 154480, 287127, 418261, 287328, 289113, 158103, 291067, 487831, 233487,
    233528, 170041, 146909, 220066, 78392, 145084, 81676, 147430, 82750, 215391, 150100, 217898, 219503, 152523, 76229, 78265,
    79993, 86070, 92275, 94397, 149947, 286726, 211237, 81893, 350747, 154247, 212750, 353048, 88408, 90150, 78229, 81498,
    212625, 84455, 84821, 84857, 86325, 88785, 89058, 90542, 92325, 76585, 350239, 350601, 618817, 168164, 145795, 616544,
    618776, 163905, 426065, 231438, 100391, 233547, 237586, 106532, 436238, 503814, 440338, 147625, 152498, 426060, 143871, 282639,
    86136, 348289, 350326, 352320, 493622, 211511, 147580, 350345, 225570, 161912, 622609, 78678, 148439, 80649, 80386, 422323,
    78361, 80660, 147711, 76704, 77934, 81553, 211609, 76586, 79514, 213068, 76748, 77922, 80457, 80574, 80754, 92271,
    213754, 213804, 215950, 80487, 215591, 221693, 148255, 150515, 162014, 150294, 215931, 283460, 284714, 219513, 155695, 287227,
    289006, 291252, 299146, 301134, 301167, 303181, 303196, 80835, 76663, 219446, 493609, 144907, 84558, 282927, 350262, 368704,
    144294, 215861, 80344, 213565, 150095, 86764, 350497, 223304, 222162, 79376, 287118, 303143, 279322, 354325, 80859, 84656,
    86538, 88852, 90678, 79179, 81756, 217795, 154359, 289023, 146272, 628832, 368659, 428103, 79170, 214010, 215360, 217594,
    75838, 78358, 80556, 80944, 81544, 82200, 82461, 215486, 217133, 86535, 88316, 154009, 485407, 421892, 618700, 160072,
    346365, 84316, 220136, 419925, 420186, 80293, 80036, 81443, 84075, 414443, 90337, 289058, 436261, 211455, 212690, 78938,
    144740, 211602, 215759, 284831, 483554, 495681, 299125, 346430, 81216, 413785, 75962, 79439, 146390, 151584, 161848, 215908,
    346482, 348250, 106620, 283404, 78210, 145339, 144242, 149787, 82626, 418772, 418805, 146018, 86670, 88131, 223685, 292991,
    503832, 372780, 77842, 80593, 80804, 147250, 82677, 84022, 84033, 84812, 86129, 86252, 86331, 217643, 86814, 88217,
    219444, 90190, 90263, 90504, 90778, 94456, 235615, 145251, 148133, 148193, 213809, 287387, 287389, 487798, 489639, 233548,
    352996, 76593, 213612, 215089, 217862, 282875, 79195, 213022, 215042, 88205, 219685, 352658, 235642, 86484, 356404, 356445,
    362546, 81670, 79343, 84705, 155983, 92261, 149853, 348932, 223677, 88135, 434266, 280725, 84079, 94212, 79370, 81817,
    416377, 616450, 485444, 485470, 487730, 424082, 156313, 358641, 364782, 143426, 416285, 213534, 416605, 416695, 352266, 418338,
    616460, 354380, 356662, 487755, 292950, 294996, 168083, 366714, 368663, 630869, 84774, 78765, 418299, 424101, 432239, 221480,
    221706, 92545, 495725, 145094, 348772, 219643, 219865, 352667, 81560, 149897, 78019, 213566, 231527, 150241, 144795, 212885,
    147402, 291190, 144105, 211638, 81265, 98352, 100360, 144715, 156192, 165898, 167956, 279550, 78177, 79182, 81806, 78658,
    356826, 144828, 80007, 81927, 81982, 84211, 86674, 86713, 88520, 88820, 144632, 215107, 223705, 75824, 78381, 79057,
    80330, 82517, 86471, 86622, 88113, 88612, 90588, 92178, 92610, 94682, 149736, 282624, 484021, 211281, 213376, 219755,
    419953, 485506, 423987, 428104, 493652, 151802, 219398, 153886, 351003, 220016, 485414, 223639, 493642, 144180, 146568, 147563,
    149795, 88669, 88868, 90197, 90628, 92521, 94285, 305187, 296971, 75990, 80475, 146245, 146955, 147065, 81999, 82026,
    82110, 213285, 82222, 82303, 147885, 82531, 82633, 84087, 84201, 84266, 84439, 84654, 86104, 86422, 414239, 152195,
    86844, 88121, 88149, 88241, 88518, 88523, 88560, 88822, 89063, 90243, 90357, 90424, 90566, 90575, 418367, 90702,
    92160, 92185, 92258, 157985, 420318, 92668, 94214, 94371, 94422, 94485, 94578, 620565, 96351, 96394, 96423, 100370,
    100390, 170085, 106592, 108576, 213594, 152264, 154542, 155868, 288976, 616703, 160065, 229417, 166000, 166008, 170063, 366697,
    76138, 212326, 278847, 78269, 86543, 219379, 156545, 168149, 416385, 143728, 154169, 88688, 212306, 144477, 79309, 145548,
    150396, 350245, 350622, 154417, 289032, 170072, 147237, 143560, 143360, 145492, 212513, 213210, 213414, 213541, 213619, 215610,
    219426, 219801, 154495, 165899, 628807, 78595, 144663, 81872, 79712, 79728, 212515, 84921, 84959, 90958, 90985, 287743,
    489649, 489679, 102609, 102611, 501808, 507933, 144293, 144566, 145266, 145447, 146983, 147325, 147684, 147998, 149573, 215131,
    280669, 149632, 215554, 281216, 281308, 348537, 283097, 348840, 285146, 285305, 285328, 285412, 154350, 351069, 351159, 286730,
    483347, 483440, 485378, 288808, 288833, 354372, 288844, 354398, 354422, 420004, 485603, 289003, 289022, 223503, 420178, 420199,
    420269, 159861, 422043, 356658, 291126, 356730, 487809, 227373, 227414, 620649, 489585, 489590, 489597, 293010, 622660, 360531,
    296966, 493588, 296982, 624689, 428134, 299008, 299010, 495714, 626793, 364687, 430228, 430236, 301077, 497692, 170017, 497709,
    235601, 301178, 303127, 499777, 630861, 174108, 239646, 501792, 503808, 241717, 374787, 418593, 418608, 81792, 147764, 82883,
    82900, 87023, 76133, 213625, 144651, 213580, 221835, 239653, 215817, 222150, 76725, 147461, 149609, 156262, 86020, 86029,
    350579, 88702, 90289, 106553, 145739, 152102, 219395, 285004, 220147, 352725, 288811, 159770, 290842, 290946, 168086, 299158,
    143970, 151603, 152192, 154010, 157778, 160075, 76021, 77872, 84622, 90751, 485637, 92535, 620733, 227524, 227530, 227536,
    620752, 620761, 227552, 424164, 424172, 620781, 626870, 626885, 233677, 430288, 430301, 79349, 76387, 79097, 145242, 84798,
    150524, 156471, 156490, 146764, 76258, 82018, 149613, 84140, 86238, 86716, 86838, 88366, 285198, 88842, 90162, 90239,
    90508, 92222, 354578, 92488, 94330, 104520, 501789, 80596, 278861, 151718, 282806, 352728, 489563, 432159, 634895, 180237,
    155756, 157987, 147570, 149703, 149838, 158007, 163911, 168055, 172137, 146344, 147944, 285019, 285314, 622595, 295000, 616722,
    283605, 483784, 148465, 422084, 281353, 78533, 81217, 145095, 144003, 211092, 81158, 144620, 146437, 146755, 215517, 414257,
    358647, 166012, 215674, 217132, 86527, 349175, 219321, 285169, 356860, 297086, 628809, 630874, 217438, 219210, 350496, 227469,
    79675, 79681, 144379, 152009, 416400, 618978, 144699, 79627, 114709, 418750, 287711, 91025, 484244, 91048, 484279, 96449,
    489676, 86241, 416634, 79395, 146058, 81920, 82091, 84088, 84700, 86489, 88720, 483523, 223661, 358549, 231526, 102400,
    102402, 104469, 86377, 94524, 215589, 282883, 86414, 88381, 90790, 362545, 624743, 303182, 280707, 288973, 301086, 84292,
    291101, 212486, 80460, 84143, 86546, 146672, 215649, 430106, 290964, 297052, 348977, 145399, 219449, 356726, 102476, 212157,
    217897, 348812, 358462, 146504, 155932, 79676, 82794, 82802, 416709, 161821, 428057, 362549, 493654, 495633, 104475, 147560,
    149931, 158012, 172118, 172128, 280922, 347029, 616877, 624742, 76269, 78352, 282762, 348383, 285148, 350853, 221794, 419896,
    354462, 223663, 229435, 426052, 299088, 434283, 78047, 414713, 426109, 430328, 430332, 79227, 150322, 288778, 622686, 415830,
    79893, 151737, 86282, 88086, 153631, 153993, 76049, 78218, 211549, 212476, 217867, 219608, 211694, 215263, 346776, 348601,
    221808, 288910, 356367, 422204, 76638, 82508, 100438, 102417, 301083, 283123, 356778, 428053, 278895, 348225, 283384, 418260,
    485839, 618516, 618687, 422247, 620573, 358545, 622630, 362530, 366642, 432224, 434200, 499774, 374804, 417809, 364739, 76356,
    80250, 80546, 81296, 90272, 96339, 98315, 372782, 416244, 145298, 148452, 283616, 155730, 161828, 144311, 80658, 144467,
    145833, 84843, 145140, 79830, 144363, 78837, 146174, 217980, 415806, 90813, 221903, 90869, 144109, 145060, 147435, 219279,
    487752, 149843, 151654, 151914, 153749, 158112, 159954, 75938, 81297, 81325, 81988, 151859, 212136, 79869, 81648, 148179,
    81535, 213516, 219902, 144470, 212621, 148162, 143751, 145644, 81584, 147203, 148384, 154005, 354698, 161868, 231525, 143479,
    146007, 415779, 350908, 419911, 420017, 354654, 356502, 422310, 426058, 434247, 80189, 80441, 149994, 80484, 82465, 82488,
    82562, 84685, 86557, 86779, 88513, 90601, 354411, 92313, 92456, 94346, 94600, 96306, 98310, 98316, 170036, 106505,
    172079, 213714, 77947, 80086, 84696, 86173, 88410, 88726, 225378, 96362, 145805, 149809, 151816, 153640, 286747, 155997,
    157991, 159899, 79873, 80249, 81938, 147505, 82598, 84129, 84639, 84807, 217494, 225557, 489614, 84144, 90465, 92347,
    626789, 420331, 144781, 81127, 146923, 149601, 84539, 84663, 215930, 152365, 88779, 220137, 90529, 165903, 94372, 161794,
    620551, 233607, 235532, 170002, 237582, 370721, 501794, 217511, 489609, 290830, 215835, 146370, 279136, 225415, 76572, 79484,
    145349, 148378, 148419, 149852, 152559, 152561, 284821, 156610, 156617, 156629, 618604, 291026, 160219, 160221, 160223, 160246,
    160254, 163953, 163958, 163966, 163967, 168178, 168179, 168181, 168186, 168187, 168188, 172157, 84277, 84873, 283003, 421911,
    618694, 291138, 620651, 362613, 362614, 364571, 628825, 432251, 143362, 146166, 146670, 147145, 155749, 90440, 157852, 485609,
    165948, 172096, 79752, 211684, 81505, 88341, 354420, 225587, 487788, 493579, 626807, 143904, 79776, 79794, 94713, 146002,
    147266, 352597, 223564, 227415, 78934, 78914, 82286, 79626, 212070, 217810, 81633, 418076, 76042, 213687, 162000, 168141,
    416227, 144456, 353061, 353173, 281173, 358453, 211867, 416115, 90121, 102408, 104510, 106502, 110596, 112642, 112649, 76263,
    79190, 146703, 81506, 493640, 76737, 146043, 147522, 148209, 217871, 220072, 221373, 156247, 419850, 419889, 223489, 354722,
    289189, 225537, 356716, 172116, 349091, 161955, 350913, 81210, 212842, 144542, 616644, 218042, 222143, 222189, 229498, 79166,
    81159, 86568, 88903, 90387, 92251, 94333, 630855, 414664, 303231, 78738, 150477, 156575, 420094, 358644, 146608, 82918,
    83995, 349164, 84213, 147555, 278673, 237650, 144984, 211697, 86431, 350633, 350966, 88894, 286863, 90400, 356486, 94414,
    358538, 150378, 156312, 77987, 215318, 84275, 283358, 299164, 78617, 279123, 223290, 278971, 348199, 348602, 620709, 110647,
    143387, 78446, 80902, 280782, 358654, 285680, 352958, 420167, 241689, 79895, 81947, 147579, 82519, 151862, 152388, 153642,
    154001, 154288, 156155, 156340, 418549, 157934, 158070, 356453, 160180, 161798, 358560, 163847, 364673, 168074, 170062, 170095,
    172087, 172109, 172124, 77903, 80893, 80895, 148424, 283611, 418763, 418768, 287715, 418791, 291299, 76741, 145745, 148388,
    414249, 287712, 149643, 143979, 78809, 146500, 144925, 144166, 212088, 213421, 213652, 215114, 215874, 348504, 352521, 144409,
    418264, 634903, 245773, 79305, 278533, 279440, 280681, 283474, 287334, 418505, 160210, 430248, 168114, 86151, 88133, 90229,
    92373, 94221, 94270, 94385, 94629, 96275, 102541, 162024, 148473, 82942, 155796, 158040, 82312, 106503, 174097, 307256,
    280659, 346374, 215580, 150239, 217784, 219401, 154137, 219868, 220064, 220073, 221583, 418300, 354587, 223670, 487505, 291202,
    292966, 364674, 75988, 279180, 152039, 415935, 416102, 416204, 285415, 90147, 92542, 487685, 305170, 282773, 284952, 291152,
    81886, 213855, 213856, 213876, 82814, 215097, 217513, 217956, 217961, 217999, 218032, 219255, 221368, 221555, 90820, 90837,
    221911, 353016, 223590, 94659, 231475, 495783, 110651, 110654, 76115, 145955, 80930, 86554, 94211, 356528, 356764, 358432,
    96297, 96422, 104514, 221846, 221865, 144868, 144554, 144015, 77909, 366609, 144764, 80325, 146030, 86135, 158018, 94553,
    278771, 288857, 290928, 487717, 297082, 78403, 81671, 90116, 80297, 84034, 86660, 149666, 416238, 285061, 289054, 485737,
    78946, 634898, 350414, 221518, 618626, 161908, 280830, 285551, 147318, 217958, 218028, 221891, 221915, 221959, 487870, 487876,
    356809, 487885, 487895, 94680, 487897, 491622, 165921, 499827, 499830, 499834, 110650, 285663, 287381, 76150, 149729, 151629,
    415757, 147239, 213311, 213811, 215442, 217112, 217769, 414389, 219336, 219703, 221437, 221594, 352939, 225499, 90172, 278818,
    281225, 217241, 284926, 286725, 286899, 286978, 287384, 288856, 420002, 289012, 223504, 223741, 291174, 362577, 280689, 346944,
    79083, 219456, 215862, 624696, 211509, 147452, 221802, 78021, 82125, 144880, 148285, 231440, 81365, 161866, 167986, 82089,
    221728, 231434, 233492, 212405, 144609, 151650, 153894, 156085, 157760, 158076, 160174, 144144, 150182, 217490, 148348, 414577,
    283539, 156430, 487603, 291271, 291282, 489623, 360510, 295024, 168095, 438311, 211210, 348445, 415824, 354479, 618517, 362586,
    432174, 358573, 364736, 227433, 211864, 283073, 223336, 225385, 225464, 358558, 227496, 233575, 237636, 76446, 283523, 283526,
    147302, 147739, 218016, 233647, 146466, 150078, 152090, 155885, 156253, 616600, 223424, 356356, 422064, 160124, 422300, 161818,
    423968, 424038, 163884, 172104, 217779, 626804, 84760, 219699, 227492, 501762, 632869, 212997, 278553, 280613, 215145, 217171,
    283450, 221798, 213635, 149950, 151597, 94536, 161926, 76602, 144904, 81880, 485421, 144187, 81970, 84592, 285600, 156208,
    430157, 628857, 148371, 152428, 152474, 152481, 152485, 156439, 426088, 168096, 168101, 172152, 144399, 144460, 78181, 212576,
    213360, 78862, 84411, 221190, 491631, 212473, 94302, 96321, 98308, 163868, 98381, 218065, 154162, 91103, 98398, 98412,
    150382, 281526, 281593, 221733, 287662, 291232, 293058, 299115, 636929, 505871, 147552, 215060, 149540, 149893, 485702, 432234,
    303202, 144635, 145654, 217107, 220047, 78935, 146012, 75832, 78044, 144516, 80502, 80973, 84535, 217729, 92265, 76617,
    78613, 79902, 82006, 82620, 84826, 88952, 88968, 92408, 211701, 82057, 213703, 82681, 84606, 217883, 88144, 219644,
    220125, 220146, 90120, 356355, 358441, 358566, 98365, 149803, 102502, 497786, 278873, 237658, 145556, 88335, 79615, 144697,
    79070, 484340, 356837, 368661, 368684, 215870, 349141, 80672, 146746, 86028, 86339, 414639, 485561, 618839, 144996, 147444,
    287101, 485485, 297038, 299026, 102472, 301110, 366662, 221904, 79398, 278871, 284844, 88346, 219696, 485525, 76709, 286748,
    289117, 79285, 102434, 481638, 76745, 78997, 414596, 76753, 144889, 358569, 215824, 152305, 154195, 219855, 144721, 80294,
    80369, 84355, 86221, 155713, 92431, 420131, 225301, 358513, 104529, 77846, 80775, 86593, 493637, 346934, 221917, 221950,
    156356, 160195, 442389, 79452, 81826, 78225, 80587, 217473, 219358, 213504, 215137, 150075, 150199, 151940, 217757, 217901,
    219583, 154315, 221484, 221500, 221575, 156065, 223364, 420006, 223430, 225675, 231439, 432171, 211512, 146265, 214000, 92205,
    80612, 146897, 81933, 152341, 225542, 227445, 358620, 370738, 213213, 215396, 81865, 82604, 223259, 79341, 81995, 143452,
    78068, 144427, 79492, 80072, 145701, 146655, 146677, 81641, 147284, 147480, 148183, 148185, 149735, 84432, 149992, 151709,
    151784, 152056, 86562, 88259, 153946, 219499, 154262, 88865, 154620, 221415, 485406, 420083, 94219, 618527, 290914, 225512,
    160099, 225659, 225666, 227430, 161898, 489622, 432134, 237643, 303186, 212241, 76690, 79593, 153843, 84648, 288871, 428108,
    167981, 366665, 94220, 84078, 81659, 82829, 349059, 618929, 78039, 81345, 81593, 485520, 497731, 211315, 146456, 144305,
    144463, 144474, 293023, 360458, 294974, 213535, 144257, 82518, 84632, 86457, 153889, 143531, 146732, 147352, 81976, 82484,
    149919, 86066, 88749, 92286, 485544, 92350, 159974, 168004, 432165, 80750, 84453, 92424, 211690, 144282, 283570, 144915,
    144466, 149814, 150186, 213780, 217539, 147839, 149975, 151958, 155779, 160021, 347073, 98402, 106615, 144010, 284692, 285037,
    285377, 287363, 295026, 426111, 418535, 144718, 144777, 79739, 79747, 145684, 281508, 84922, 84961, 90931, 287557, 484191,
    91021, 91040, 91043, 91062, 293035, 293045, 489656, 96442, 96443, 96451, 489667, 96452, 489703, 96492, 299198, 102621,
    102624, 299247, 495857, 501806, 108604, 501823, 213016, 213719, 213721, 215271, 215528, 217320, 217592, 632840, 148143, 84846,
    147182, 84020, 348443, 350614, 213931, 222190, 301112, 76537, 153603, 157804, 159833, 163862, 215267, 416182, 350695, 287496,
    364755, 213944, 213954, 222178, 487905, 491636, 149507, 79850, 213955, 218095, 222146, 222165, 225782, 225790, 229502, 229503,
    233714, 233717, 233722, 233723, 233724, 237693, 150149, 624641, 626818, 212410, 84501, 420285, 94451, 159745, 434181, 438292,
    507919, 143892, 78557, 94240, 159863, 422276, 231497, 102485, 106498, 434201, 438289, 372784, 634928, 114704, 350893, 81960,
    84024, 222025, 222077, 222092, 222125, 620784, 430265, 430268, 430291, 436275, 78980, 211766, 284910, 220063, 156167, 418325,
    96264, 102439, 106550, 108575, 110595, 112640, 283462, 364723, 370732, 79286, 278735, 280822, 281338, 286946, 287052, 288962,
    212743, 280813, 353276, 170021, 235582, 147929, 159927, 76078, 219954, 420273, 78196, 150013, 289074, 143912, 147162, 145861,
    215957, 215970, 215974, 215991, 216030, 216050, 216056, 216061, 151671, 221985, 221991, 221993, 222000, 222009, 418629, 222023,
    418634, 222031, 222034, 222037, 222043, 222046, 222050, 222058, 222062, 222065, 222068, 222069, 222071, 418687, 222084, 222085,
    222087, 222088, 222109, 222112, 222118, 418731, 418739, 160089, 424107, 620717, 620719, 227506, 227508, 424117, 227510, 424119,
    227514, 227518, 620737, 620738, 227523, 227525, 620742, 227528, 424138, 620747, 227535, 227547, 620764, 227554, 227556, 227559,
    424168, 227574, 626867, 233654, 430270, 626883, 233668, 233680, 626897, 626906, 233696, 430305, 626917, 233705, 239655, 239657,
    632875, 239662, 239667, 632890, 632891, 239676, 245783, 639001, 245790, 145202, 148234, 86424, 418571, 422348, 87014, 78344,
    81488, 81601, 215217, 219251, 420326, 422293, 227475, 501790, 503861, 76003, 146178, 231485, 88075, 90534, 92422, 94568,
    215923, 80620, 279110, 280949, 285008, 88764, 285515, 102558, 303131, 75986, 78037, 79932, 149995, 86380, 88199, 90394,
    157986, 80909, 81161, 81427, 81596, 86410, 285374, 100440, 82448, 80514, 81065, 82170, 213678, 88187, 92208, 94303,
    144089, 215552, 219995, 152568, 418782, 282794, 219668, 483793, 225620, 81061, 212914, 147808, 213765, 149753, 79177, 82904,
    143729, 86892, 86938, 86945, 86949, 218026, 349099, 90903, 283612, 283629, 349072, 79826, 223287, 75883, 80073, 82438,
    76360, 285227, 289031, 620671, 144549, 156327, 76369, 143415, 146777, 211645, 213381, 223248, 233604, 235645, 213263, 76365,
    102578, 278762, 79924, 94540, 174119, 418812, 92602, 217202, 294962, 215811, 223370, 618614, 227456, 630888, 75913, 86362,
    154157, 155982, 155996, 157929, 92538, 94271, 172129, 212158, 81588, 352416, 291106, 153816, 80341, 153618, 155876, 352595,
    145846, 149798, 151582, 348984, 154569, 154595, 168084, 76479, 81708, 147423, 86420, 219731, 285391, 150491, 212782, 414104,
    145278, 288905, 299041, 86216, 82127, 82749, 84214, 84452, 84730, 86319, 86559, 88539, 88993, 90338, 90444, 90687,
    92354, 104554, 219363, 143475, 88139, 352797, 159979, 351201, 152217, 491635, 145370, 148441, 152503, 283606, 484032, 90870,
    94649, 94670, 152220, 154311, 157797, 168051, 144448, 144558, 280981, 147724, 149718, 168078, 144102, 212253, 346873, 416132,
    153998, 154234, 88700, 155727, 156116, 287278, 356543, 159981, 160103, 424001, 96401, 491531, 360486, 426061, 426064, 624670,
    100386, 624715, 104498, 170080, 434176, 172056, 237630, 112660, 89004, 90256, 96379, 348214, 284773, 286913, 418462, 418464,
    354419, 289198, 493629, 78510, 156279, 485825, 491584, 79765, 82864, 76180, 346623, 151568, 154422, 154554, 289211, 426055,
    430211, 146544, 148019, 356603, 358446, 278692, 299070, 279238, 82333, 217275, 348619, 219169, 350308, 154156, 350777, 487641,
    291085, 303154, 239617, 81471, 147105, 147748, 147884, 147916, 148082, 148257, 84387, 150077, 150359, 152013, 152311, 152366,
    219230, 88360, 90558, 352872, 92586, 94480, 622593, 636943, 147080, 158052, 420235, 159986, 237656, 82422, 162013, 213610,
    501777, 634936, 77965, 79913, 80097, 80526, 82325, 82418, 82648, 84027, 84465, 84575, 150220, 282724, 86328, 86346,
    88174, 88332, 88424, 285216, 88751, 285651, 286806, 90667, 485501, 92533, 289236, 92654, 290882, 487621, 487701, 292864,
    96293, 294970, 98388, 628744, 106554, 161892, 75891, 78572, 154042, 161882, 491573, 145056, 211776, 163840, 78761, 79142,
    145665, 287217, 288824, 170107, 76670, 79596, 221293, 624683, 78386, 80932, 347123, 153863, 353108, 353196, 92166, 96271,
    96291, 358570, 165974, 167953, 366619, 110629, 352685, 422011, 422080, 423963, 78282, 80636, 150494, 347120, 353151, 79634,
    82806, 217939, 353019, 288861, 94648, 356811, 98400, 301070, 81311, 150251, 219355, 143533, 348920, 82188, 215049, 84182,
    221274, 352527, 362605, 102542, 84753, 484062, 283095, 88543, 157763, 493632, 213063, 82035, 82270, 84457, 215587, 86144,
    348973, 88370, 285387, 483666, 287086, 90485, 287137, 352768, 485576, 485682, 616904, 94238, 362511, 100431, 104528, 76481,
    622682, 364629, 145648, 217099, 283093, 219715, 352300, 354646, 159991, 160121, 79935, 76407, 81045, 152107, 153908, 170079,
    346805, 217547, 152030, 284812, 154099, 416339, 354443, 485640, 362591, 495726, 364667, 79408, 80342, 80929, 84723, 76401,
    81489, 626934, 278909, 147070, 148478, 154153, 291214, 75952, 78388, 79547, 81868, 354311, 154058, 80882, 154397, 92194,
    157844, 159831, 94309, 426003, 102529, 213916, 213932, 213940, 218041, 218055, 218066, 618990, 418038, 374792, 636953, 507908,
    442385, 146986, 86220, 155957, 366608, 108550, 148095, 84610, 88300, 94401, 346783, 286997, 485693, 485746, 618669, 495685,
    146639, 80671, 146539, 150443, 354681, 420319, 161964, 168134, 214003, 155805, 156126, 157971, 289223, 168029, 143697, 145902,
    217236, 219642, 151781, 79559, 213835, 233562, 147186, 79027, 215654, 217823, 287701, 295039, 80721, 82246, 150331, 217506,
    144206, 221629, 81254, 278626, 84279, 81478, 81491, 90664, 291187, 281429, 348996, 350250, 285393, 285666, 352763, 354542,
    356788, 364682, 366670, 366703, 368717, 368732, 150323, 285664, 78197, 80324, 81728, 88455, 489677, 76165, 82556, 84197,
    413990, 78714, 76740, 82086, 219203, 102426, 289102, 143614, 80439, 90141, 94498, 212568, 482103, 144267, 212137, 422091,
    144578, 212618, 90847, 618989, 630900, 145277, 370739, 76505, 416631, 221732, 354516, 76546, 213926, 144239, 346995, 79257,
    279515, 144617, 79454, 79545, 215907, 413771, 86154, 217925, 283465, 284791, 88540, 287397, 419897, 420115, 485772, 292962,
    79077, 75847, 81279, 213475, 84356, 346752, 84672, 86277, 86635, 88611, 88683, 354369, 96413, 362502, 143787, 145987,
    146426, 81507, 149748, 152244, 348933, 154200, 154270, 157941, 223525, 352290, 354538, 416721, 76227, 351107, 225529, 481918,
    499769, 364767, 145763, 213524, 148265, 148290, 215280, 84684, 415803, 285075, 219680, 219721, 219794, 220115, 221270, 221297,
    419965, 225346, 422085, 622625, 229434, 493634, 299091, 81638, 143774, 78981, 81876, 285119, 287036, 287111, 288900, 288966,
    94259, 291211, 620675, 497707, 241687, 279496, 80169, 80838, 213216, 213494, 90418, 225448, 147246, 219900, 616858, 80674,
    146302, 148235, 287047, 289095, 299048, 495788, 144525, 286821, 79404, 76687, 89047, 351195, 91109, 213589, 217543, 219684,
    213156, 233534, 143489, 346346, 215602, 219294, 225476, 231509, 152516, 152525, 156631, 156650, 414139, 76176, 153641, 163894,
    167949, 79270, 86880, 144093, 346932, 144403, 352545, 418175, 79044, 145723, 152163, 154110, 415881, 419967, 147437, 79332,
    78543, 296960, 143519, 149962, 352319, 346811, 76173, 80548, 82363, 82529, 83992, 84293, 150306, 86051, 219154, 88599,
    88736, 89082, 221412, 90749, 98334, 76774, 279077, 231435, 219185, 491611, 493655, 90891, 237590, 112656, 82872, 86094,
    92519, 151668, 76060, 77873, 78174, 79098, 80607, 81565, 147209, 84553, 217829, 88976, 221315, 92587, 354486, 213700,
    143791, 79324, 92197, 213394, 417995, 352277, 75953, 78440, 144146, 80699, 278919, 148164, 84493, 86672, 419984, 432195,
    235643, 630854, 82111, 148425, 215288, 217488, 152342, 414661, 283610, 152539, 416007, 287677, 287723, 291297, 295028, 145347,
    279101, 148412, 413786, 152513, 416033, 156664, 288840, 160235, 168185, 151556, 151565, 154238, 155825, 172089, 280644, 346492,
    150442, 489617, 75894, 78300, 80472, 80760, 79813, 82842, 279250, 144485, 78893, 78992, 80380, 81077, 86653, 86766,
    348910, 143996, 150341, 211167, 288969, 145430, 154609, 144722, 346483, 86499, 92308, 94295, 94390, 422128, 98346, 102492,
    106566, 144778, 143516, 146047, 147235, 149512, 346954, 78988, 145254, 353119, 156512, 156572, 161963, 161973, 162043, 287377,
    279347, 280792, 88411, 88854, 352582, 92555, 94606, 96319, 98392, 104553, 88132, 420281, 432193, 616592, 422022, 624704,
    358567, 147991, 151776, 86615, 154221, 289066, 233522, 628849, 144138, 346550, 78764, 147155, 346945, 626861, 82862, 618979,
    144603, 76451, 76754, 145284, 150471, 221255, 616747, 618571, 112668, 507915, 149517, 155718, 146021, 81585, 147960, 213885,
    284779, 287141, 291054, 356797, 618958, 143834, 86445, 155926, 156185, 413723, 146496, 81226, 346877, 351001, 154069, 78671,
    143757, 211374, 219182, 79672, 82780, 86872, 86875, 86941, 86957, 90874, 626866, 76699, 80425, 213425, 213584, 215623,
    616472, 354467, 354484, 618564, 227400, 622614, 229448, 368675, 419888, 495659, 346980, 285245, 366698, 145922, 282940, 416567,
    351173, 285659, 616495, 362564, 432209, 78391, 145695, 147811, 149797, 150247, 150511, 156511, 156540, 168161, 80533, 88876,
    163861, 299034, 76294, 147383, 151626, 90800, 156614, 356488, 360465, 144521, 82497, 618824, 287720, 147096, 215433, 352302,
    158189, 160120, 424083, 76575, 82660, 76414, 150120, 80855, 144954, 418752, 76646, 80594, 80776, 418386, 79789, 211994,
    156506, 153804, 418249, 299155, 144185, 213482, 75853, 81327, 86859, 104521, 79839, 148281, 144649, 235557, 301118, 159760,
    160035, 100400, 80507, 80668, 147492, 82251, 84298, 152335, 154536, 90334, 90711, 90796, 354353, 160001, 354582, 368739,
    280988, 283132, 78445, 78636, 92664, 98323, 81186, 84109, 84238, 84578, 84647, 86175, 86569, 86734, 88261, 88502,
    88665, 88883, 89010, 90648, 90650, 90671, 92353, 92583, 94352, 94374, 94375, 96323, 626754, 104532, 78215, 278931,
    347054, 285371, 145595, 278920, 281027, 303111, 370705, 503864, 78163, 80795, 77857, 81116, 152310, 211444, 283107, 82007,
    281164, 219514, 78949, 145996, 149679, 152082, 144118, 86368, 417955, 80659, 349183, 350774, 417942, 484319, 151800, 86679,
    90309, 88203, 144100, 150326, 78503, 348243, 217514, 86462, 88262, 90196, 90516, 92288, 145725, 221582, 237607, 148157,
    148190, 280610, 280719, 417981, 287231, 420097, 485723, 487457, 159798, 291221, 358466, 491562, 106516, 305184, 307221, 146871,
    149971, 154436, 620745, 632886, 144830, 225764, 225768, 225772, 225774, 225778, 225786, 229500, 172110, 154455, 213869, 484024,
    484051, 152177, 217756, 278590, 283321, 153764, 285428, 155724, 79661, 211413, 147206, 218046, 218064, 218110, 222166, 350456,
    618500, 144049, 348436, 484228, 299236, 79075, 78780, 280705, 353182, 622611, 76433, 416356, 221541, 285089, 286943, 80715,
    283172, 350351, 285021, 485656, 291105, 291212, 493647, 497744, 282678, 219281, 288883, 143808, 147700, 153601, 157699, 78552,
    217342, 94235, 176153, 147958, 159912, 290950, 279331, 283266, 350639, 356576, 626714, 80684, 90464, 418182, 223313, 420255,
    618665, 424058, 212033, 147323, 153740, 287467, 156404, 158118, 81918, 287379, 82034, 84167, 84302, 92471, 98375, 102519,
    106601, 213901, 144958, 79122, 79314, 145694, 148085, 80814, 82686, 428148, 146031, 154031, 159900, 159968, 422119, 499817,
    215676, 356783, 143692, 82906, 150169, 217452, 288873, 159848, 160055, 428034, 76017, 79597, 81840, 82162, 84419, 88662,
    420037, 144622, 144936, 215423, 416283, 155873, 165927, 303122, 172068, 505874, 219624, 616870, 299150, 78026, 217239, 86464,
    88272, 90472, 100469, 100470, 102427, 106563, 78954, 283558, 430258, 215067, 346166, 348175, 348273, 348475, 90236, 284962,
    285337, 227384, 156645, 434301, 88214, 77989, 147968, 86404, 154047, 155964, 156039, 157828, 157894, 160139, 624722, 366635,
    630871, 212284, 213125, 215573, 225523, 485760, 112665, 219599, 150464, 79331, 80677, 150463, 416307, 145534, 146405, 146582,
    146605, 153949, 353264, 160140, 285041, 616695, 75852, 78321, 146025, 80740, 213013, 84626, 151609, 151829, 86610, 283396,
    88630, 351172, 155798, 353272, 145632, 146346, 147132, 88280, 78692, 213986, 353274, 75954, 144393, 90659, 94294, 430156,
    172054, 484325, 146657, 419863, 75873, 78698, 212109, 84361, 155654, 418423, 282891, 283389, 414587, 286983, 289014, 289038,
    290992, 291006, 291228, 292896, 422031, 626839, 630853, 634894, 420149, 281143, 354617, 215223, 346405, 346938, 297064, 430189,
    76658, 79008, 145427, 146561, 149878, 153957, 285193, 351025, 287125, 428142, 364610, 84250, 212991, 280962, 285201, 485384,
    485792, 352357, 147466, 84619, 150233, 86567, 86577, 152161, 86691, 86711, 88557, 154289, 154337, 90706, 92305, 92433,
    94298, 94356, 94619, 292972, 100444, 497791, 279330, 78929, 282790, 413878, 143501, 144083, 79548, 145633, 81894, 147954,
    149563, 150094, 150111, 151864, 152402, 283517, 153868, 154287, 156203, 156342, 156345, 156358, 156366, 156396, 156413, 158069,
    158190, 163947, 168108, 172090, 172153, 145527, 215208, 215804, 220005, 281466, 285673, 78467, 422171, 428112, 432232, 211593,
    144875, 84375, 88079, 143963, 82705, 150441, 213175, 215701, 485433, 358498, 360474, 497696, 156567, 162025, 213936, 213969,
    353222, 79287, 79689, 79703, 79705, 147471, 279188, 213840, 82799, 149554, 84883, 281495, 84894, 84908, 84911, 84934,
    84957, 152094, 156250, 287514, 90925, 287535, 90929, 90952, 90955, 90967, 484194, 90983, 90984, 91011, 158121, 225614,
    293065, 170024, 305206, 78834, 148466, 144154, 88862, 354670, 487714, 495625, 296992, 157766, 420054, 624669, 624726, 626705,
    81043, 148299, 283203, 233589, 81151, 148356, 152426, 152438, 152455, 283556, 287435, 287497, 291266, 291272, 295015, 430244,
    430254, 303215, 303220, 307260, 217503, 88082, 90340, 225486, 213387, 213479, 145049, 86439, 358537, 80447, 79171, 146676,
    346982, 217323, 219247, 221453, 221873, 418523, 76083, 212045, 348867, 419871, 144274, 147733, 154324, 156045, 215514, 215516,
    79158, 146059, 215535, 154364, 297068, 80222, 80896, 92466, 147371, 86605, 88488, 428136, 288822, 497685, 172133, 176135,
    245769, 418774, 428096, 144091, 217996, 356792, 360544, 360553, 144129, 416080, 434203, 144255, 146566, 144164, 147353, 211813,
    82025, 286721, 219136, 229454, 440350, 144523, 78722, 79534, 81910, 82770, 86851, 90821, 90880, 90898, 288791, 152423,
    283503, 287465, 156417, 156433, 430252, 215118, 280975, 350243, 285372, 155703, 352389, 352751, 352842, 354375, 354481, 289118,
    290966, 356737, 356774, 358518, 229459, 362516, 364642, 368711, 624747, 497784, 144423, 279433, 283482, 212385, 84763, 86749,
    88515, 417984, 90361, 94323, 94494, 96356, 346507, 78774, 358447, 144259, 144579, 217861, 290828, 487681, 76678, 82879,
    153688, 353245, 487916, 75866, 217923, 144125, 218094, 222151, 222177, 222202, 225783, 81713, 215873, 285601, 156364, 301162,
    434297, 418183, 419972, 288937, 485549, 80633, 418273, 428093, 211741, 79866, 81871, 145150, 82896, 364794, 77840, 80001,
    80053, 82196, 86539, 284847, 88880, 90567, 92322, 92463, 94343, 434272, 430084, 145017, 346931, 211720, 417852, 235576,
    81983, 78513, 82024, 84395, 92476, 106582, 348827, 78791, 78912, 279418, 418501, 418560, 78423, 283050, 352640, 143872,
    146205, 213048, 217138, 219821, 223422, 628798, 81047, 147537, 148256, 149693, 149705, 150394, 282899, 153914, 154078, 154312,
    154318, 154601, 352819, 158034, 421937, 165910, 146453, 154019, 167952, 346730, 348642, 149781, 156219, 221845, 143958, 145510,
    146931, 150042, 221490, 88623, 616530, 618651, 423973, 489615, 76730, 81548, 146786, 284672, 294990, 505886, 413803, 86155,
    92581, 283484, 283502, 287464, 287500, 347101, 144005, 147389, 215628, 213761, 156282, 217844, 213161, 283387, 622679, 297012,
    628755, 80334, 356438, 143585, 149715, 80879, 146820, 86453, 86393, 155855, 92228, 92254, 94514, 79798, 82915, 483422,
    279450, 76521, 84582, 151765, 152015, 154394, 144949, 145326, 145587, 148004, 82646, 348381, 350434, 153951, 350909, 351182,
    351190, 418765, 161815, 78939, 79836, 82805, 86881, 86655, 420322, 147745, 349178, 92287, 483399, 84581, 144529, 281189,
    79429, 213698, 284897, 219835, 288787, 291034, 291251, 78328, 144544, 211409, 482170, 233566, 145904, 149642, 151888, 153612,
    156060, 156088, 160049, 144017, 76465, 286974, 418133, 165944, 82271, 84290, 156100, 155771, 76143, 78153, 78961, 86127,
    160024, 78222, 145240, 156474, 156497, 156505, 156508, 156534, 168167, 278721, 213647, 76744, 80778, 81071, 148246, 280589,
    84584, 414329, 88699, 220138, 221401, 221746, 223373, 354477, 223429, 485801, 223719, 225483, 144197, 79877, 79939, 80563,
    81569, 82348, 82380, 82546, 84541, 84766, 84823, 86477, 86775, 86830, 92168, 356768, 364651, 503813, 88323, 420217,
    213026, 215205, 217563, 88908, 90216, 90737, 79071, 81442, 82238, 150271, 217228, 212924, 148145, 213762, 150367, 348962,
    626783, 507925, 147248, 80979, 217267, 88591, 90641, 92465, 354781, 487706, 423980, 301094, 305155, 372762, 309259, 243736,
    178202, 243743, 245765, 144261, 84748, 624731, 146327, 80583, 81170, 80449, 84657, 90376, 483887, 94355, 76201, 79234,
    149598, 155817, 88313, 90425, 98370, 100378, 414711, 422396, 145024, 485399, 279353, 281002, 282864, 288860, 356517, 426027,
    301149, 213725, 285197, 219708, 154263, 154389, 483586, 287068, 616610, 223468, 485620, 487782, 160117, 96385, 98372, 143741,
    146053, 146102, 81620, 146270, 77975, 80255, 212984, 147652, 149849, 149907, 217119, 151835, 153763, 154006, 351164, 160094,
    112650, 147966, 84729, 153988, 220002, 221287, 156228, 223727, 143824, 212049, 281366, 415971, 350585, 88585, 286767, 90517,
    422069, 82378, 86117, 86334, 287404, 156542, 81190, 346868, 282627, 418314, 362500, 79757, 152499, 152526, 350927, 160220,
    291302, 426099, 426107, 78333, 211299, 80800, 88285, 285257, 79539, 147128, 79673, 215353, 348544, 348759, 350515, 483719,
    352829, 485508, 354473, 618551, 356789, 356791, 356795, 356798, 356804, 225737, 356813, 356823, 356825, 358443, 360550, 495695,
    626771, 366645, 76562, 79219, 81639, 88226, 90721, 356478, 418455, 418517, 430241, 430242, 144811, 279045, 280749, 420172,
    168000, 225372, 152432, 287421, 156441, 426091, 346324, 223335, 493676, 79277, 80252, 78803, 221314, 279088, 215257, 215627,
    221705, 370730, 211957, 634914, 144746, 79235, 223736, 229395, 283207, 421917, 624679, 235561, 301141, 144851, 146065, 147063,
    147761, 82708, 213411, 163940, 211830, 79714, 79726, 79740, 84910, 84943, 84952, 84979, 84987, 151837, 219216, 153997,
    90963, 91028, 91035, 157871, 223519, 421909, 96450, 96460, 96466, 96478, 96485, 96506, 229437, 165978, 102579, 102592,
    235566, 235618, 237653, 223479, 362515, 78517, 213107, 82305, 217632, 89017, 220091, 223552, 145650, 349110, 218089, 353219,
    353224, 353266, 157835, 360569, 165983, 168002, 168024, 168059, 149752, 151952, 350471, 213900, 217952, 86888, 364703, 114706,
    346444, 353214, 91132, 147354, 144018, 281109, 352473, 149967, 152147, 145273, 145281, 150456, 150492, 156482, 156583, 156595,
    156599, 161966, 161975, 420317, 489516, 366630, 370691, 438298, 374795, 374798, 309272, 243738, 309279, 311301, 151677, 620662,
    624660, 82264, 153713, 146790, 213745, 348824, 283407, 415947, 415980, 285608, 351204, 286909, 289025, 420258, 616923, 422081,
    291073, 159914, 307244, 213752, 485870, 499770, 147862, 79229, 145147, 145465, 147623, 147831, 148280, 279541, 149586, 150065,
    283077, 152172, 152322, 151558, 80145, 420260, 80235, 86186, 287185, 79072, 217772, 428089, 145295, 278950, 94243, 94399,
    144084, 143523, 284937, 285404, 291151, 235564, 286802, 292900, 79191, 280833, 420198, 616922, 356537, 428083, 362615, 211291,
    213173, 616952, 79308, 281291, 291080, 291219, 81086, 484184, 144252, 78140, 78279, 80818, 84427, 152289, 350337, 352287,
    421959, 80881, 146879, 144641, 147689, 346697, 484033, 484042, 76037, 79950, 281266, 303197, 78998, 493603, 626803, 235593,
    432246, 79321, 78509, 155719, 442379, 279513, 144530, 147415, 149954, 281133, 285448, 156235, 491580, 299089, 146801, 80561,
    79958, 82023, 100421, 165888, 75876, 86345, 90528, 157732, 301170, 108573, 88148, 82937, 279547, 76490, 213435, 279287,
    346214, 283066, 146469, 147538, 215735, 283437, 153850, 219851, 221550, 221601, 156068, 157880, 296975, 75835, 80544, 147453,
    484107, 366707, 78113, 80286, 146195, 80726, 80927, 146988, 81542, 416113, 416310, 352303, 90169, 94315, 366595, 112644,
    374790, 505884, 507912, 147618, 154255, 225282, 487715, 78933, 80783, 80903, 81015, 81335, 148379, 84183, 84435, 150275,
    86075, 86854, 152547, 88293, 88900, 156621, 157834, 92306, 158106, 487630, 620607, 161920, 150465, 150520, 414506, 348975,
    156501, 156544, 168132, 145657, 154120, 351096, 351112, 484333, 354552, 157945, 225765, 225789, 225791, 233726, 233727, 237694,
    82615, 281022, 282669, 150379, 630820, 305159, 76594, 81711, 283370, 76370, 79569, 213306, 289274, 78082, 90521, 346414,
    283255, 420213, 624652, 499832, 76535, 145956, 149562, 84248, 150038, 151670, 86150, 152133, 219343, 89050, 155651, 155667,
    90794, 159854, 424090, 170086, 499765, 76548, 78155, 82543, 347030, 283174, 157777, 76698, 78663, 79531, 156448, 156601,
    161983, 161991, 358630, 174120, 174132, 174137, 376863, 78842, 86143, 217633, 221485, 424141, 620760, 348760, 350302, 90257,
    221563, 223349, 223644, 225305, 227380, 227489, 229423, 362561, 239618, 79581, 106542, 636941, 86933, 221266, 227364, 213375,
    149842, 217890, 219229, 286745, 221348, 102410, 285500, 211425, 86290, 86641, 88594, 483757, 215373, 149873, 157773, 81402,
    291045, 76715, 346240, 215912, 86807, 76058, 288977, 297030, 83972, 86639, 88136, 88519, 90616, 92177, 94256, 96326,
    426054, 162036, 213661, 215977, 216023, 418094, 222053, 222072, 222075, 222094, 418738, 424158, 424180, 424191, 626882, 626886,
    626926, 436269, 442390, 80019, 81447, 81789, 84096, 149891, 150018, 151757, 282915, 283001, 283254, 219139, 219769, 154249,
    219876, 351040, 352284, 419842, 288836, 223467, 223486, 157967, 354653, 487473, 356507, 159901, 291122, 489526, 489584, 424049,
    227474, 491569, 231446, 364692, 432156, 104481, 628796, 170065, 90833, 147283, 218019, 484111, 624698, 626845, 628799, 145067,
    78684, 346536, 212471, 144775, 151706, 157781, 158042, 299261, 78305, 84220, 217519, 283108, 219679, 285426, 483606, 287177,
    483865, 616454, 618607, 225408, 618630, 290987, 618721, 618749, 618800, 618835, 291205, 618913, 231476, 493662, 626723, 79192,
    147071, 620545, 481678, 487661, 143375, 145478, 280684, 151647, 151792, 283043, 283067, 86718, 154203, 155938, 419899, 158139,
    161796, 294955, 170077, 78928, 79245, 278869, 215134, 221353, 618872, 217482, 152259, 90151, 92425, 92599, 288919, 346279,
    413711, 144575, 349151, 352902, 211129, 86649, 219476, 159767, 297048, 108561, 241720, 278980, 144967, 82939, 214012, 416474,
    484096, 485452, 499735, 145797, 147970, 153748, 76306, 147368, 213014, 148287, 280844, 348242, 283142, 152344, 219535, 88475,
    220030, 154578, 288842, 419955, 485638, 420170, 420174, 420241, 354734, 290862, 618593, 94321, 422088, 618710, 487784, 356734,
    292897, 489569, 227450, 360483, 297000, 299044, 495716, 233616, 301076, 497723, 235581, 632856, 307201, 503810, 503821, 374797,
    440341, 440343, 311298, 376835, 144861, 80808, 80872, 81675, 278958, 84343, 84471, 86705, 88414, 416295, 88617, 88789,
    90364, 90695, 288848, 92352, 92512, 297034, 374791, 78139, 86259, 88321, 616479, 144642, 288777, 422074, 487650, 79013,
    346490, 215686, 215902, 217350, 219832, 416694, 221392, 420147, 227407, 168115, 235628, 143780, 144346, 150072, 156260, 143492,
    145691, 146407, 146889, 147146, 147721, 281294, 151604, 156518, 157788, 632846, 423954, 79207, 78216, 78353, 79068, 79937,
    81717, 81891, 79346, 90766, 82943, 78558, 76592, 147350, 82720, 84116, 150215, 84840, 84858, 284822, 88373, 89065,
    92427, 92483, 146139, 157716, 225308, 161803, 143691, 148079, 501795, 144583, 414260, 352498, 86882, 86928, 79299, 145341,
    215171, 418759, 418790, 76109, 78061, 80483, 215906, 219167, 219529, 220117, 90285, 419988, 102501, 213477, 414177, 283327,
    90124, 90193, 90431, 92229, 618647, 94467, 160169, 98391, 100353, 100373, 100376, 102415, 167971, 102456, 102527, 102554,
    104467, 151588, 151648, 350626, 154616, 354319, 165932, 88644, 285499, 286776, 223617, 223721, 159893, 293012, 293022, 229439,
    493648, 168039, 213309, 215046, 155918, 618870, 358541, 364620, 364735, 84859, 285127, 288785, 223742, 290864, 81751, 347056,
    156516, 162040, 75995, 80826, 148025, 281019, 84775, 417798, 279170, 281293, 354786, 110642, 176183, 84100, 144451, 414193,
    487783, 499736, 76167, 280665, 283455, 100384, 151624, 81879, 82327, 84418, 215821, 86082, 88208, 219521, 90699, 223393,
    223460, 223500, 420242, 434257, 237675, 346976, 282747, 217722, 289139, 290958, 229496, 146749, 147509, 149912, 151895, 217798,
    217804, 219222, 221642, 212436, 287352, 76712, 82565, 215911, 215933, 81235, 147055, 148084, 219663, 158063, 229452, 428149,
    430107, 282906, 352423, 76726, 144376, 79555, 80108, 80228, 81667, 82141, 82179, 84103, 84145, 84193, 84324, 281159,
    86093, 217477, 86747, 88179, 88434, 88469, 88668, 90122, 92236, 94518, 96332, 96403, 626717, 301148, 106519, 438322,
    503863, 78471, 152006, 96412, 144438, 286880, 144182, 155706, 90883, 146458, 346966, 215061, 217480, 219215, 153697, 90435,
    221683, 223683, 96389, 428153, 144536, 215713, 161919, 144577, 215452, 84599, 217596, 154130, 155763, 233587, 153990, 155680,
    144408, 145032, 221815, 143712, 145809, 146818, 147082, 212751, 148116, 149578, 281221, 352369, 295037, 299256, 81606, 346724,
    76075, 80955, 157916, 354564, 217628, 350954, 147261, 366678, 75790, 76024, 78107, 80106, 145938, 213419, 82457, 148048,
    153653, 284861, 154379, 155815, 156059, 288931, 159758, 618568, 159880, 161864, 163857, 163912, 493575, 303139, 147346, 423988,
    149539, 152232, 153964, 158053, 146094, 279399, 346385, 86465, 152168, 156387, 418541, 287501, 157949, 160192, 160211, 168053,
    280655, 346725, 144412, 81101, 213160, 278829, 281298, 218059, 219567, 482082, 287676, 422002, 225436, 225504, 356740, 165961,
    368665, 370698, 372753, 307248, 222156, 222193, 222195, 80816, 79828, 215660, 638997, 279266, 282692, 86347, 285166, 79846,
    213923, 218068, 350257, 283149, 284754, 213905, 86026, 218004, 349100, 352963, 221954, 353031, 221963, 421942, 618948, 487881,
    618964, 618967, 618969, 622694, 229482, 233635, 233642, 233643, 245780, 76750, 86074, 86137, 94314, 108560, 481802, 91072,
    353244, 356829, 356831, 360518, 81989, 86107, 213215, 144906, 145572, 211336, 146979, 219429, 354594, 354708, 356698, 82919,
    150423, 150426, 150431, 347050, 86122, 156538, 356846, 76004, 78027, 80028, 82267, 149599, 151831, 151972, 153903, 154408,
    156015, 157786, 158043, 160093, 160150, 167958, 168025, 168058, 170031, 481821, 221747, 75810, 80044, 81962, 82259, 84670,
    348343, 86740, 88097, 219236, 154023, 88782, 90556, 92514, 223709, 292908, 102432, 108545, 174083, 241690, 178187, 112664,
    112671, 114693, 485604, 144638, 79262, 145092, 616957, 481354, 279008, 217636, 225569, 153948, 221755, 287367, 287369, 167937,
    364592, 169988, 366674, 243734, 144137, 223559, 233512, 626728, 499792, 346154, 211750, 213346, 217837, 225478, 225693, 233515,
    145280, 147540, 287259, 143987, 148450, 287687, 287738, 358421, 489547, 426041, 152223, 152226, 220025, 155977, 92470, 420207,
    90362, 618986, 630826, 78579, 146637, 279484, 287736, 487538, 436234, 78143, 80274, 80670, 81126, 82276, 82812, 618936,
    622688, 79041, 156013, 147702, 221236, 235620, 149731, 150003, 346612, 151893, 153925, 154008, 350646, 285159, 159924, 161936,
    167983, 170033, 76438, 413830, 147490, 82395, 149669, 152027, 154293, 279494, 418785, 418808, 215865, 212468, 503829, 84408,
    348215, 217527, 88265, 285606, 287211, 90733, 92643, 303157, 108566, 143731, 150032, 168103, 176187, 176190, 76427, 82090,
    84029, 84031, 86039, 86053, 86850, 88068, 88251, 88345, 88567, 88741, 90208, 90242, 90356, 94213, 94248, 94370,
    94410, 102533, 499733, 76733, 80121, 146903, 146482, 147613, 149961, 152194, 153896, 156020, 156094, 160016, 219796, 221530,
    223281, 76029, 79644, 146366, 82916, 349123, 218080, 349166, 90828, 157734, 618959, 618987, 356855, 622695, 282700, 212483,
    151883, 213668, 152343, 80744, 86532, 88231, 144962, 212128, 430096, 151561, 155998, 157724, 157855, 289266, 161881, 167941,
    626772, 169993, 172066, 176175, 213865, 349076, 218027, 221930, 221945, 221948, 225734, 229469, 229486, 213072, 211715, 81493,
    147214, 419862, 288826, 618578, 628781, 219989, 346983, 221626, 283188, 144605, 348442, 215472, 84173, 84754, 151832, 86675,
    622658, 143903, 80182, 215466, 152229, 86706, 284889, 351017, 354417, 223553, 290981, 159913, 160046, 160057, 161840, 358451,
    161876, 229443, 362568, 626700, 364675, 301168, 76056, 81098, 414473, 423937, 76179, 80828, 281392, 217143, 90596, 294972,
    82932, 145244, 279316, 144841, 211259, 212771, 217699, 219646, 426023, 237624, 307224, 636959, 83978, 84128, 217646, 94477,
    94564, 106546, 75778, 77913, 78224, 145743, 80857, 146845, 81441, 81495, 147076, 81722, 81993, 147620, 82138, 82342,
    148108, 148112, 150062, 84538, 86525, 86642, 284863, 285030, 285132, 154343, 158005, 618722, 356613, 428036, 493578, 167998,
    174098, 78541, 145410, 147779, 147935, 211950, 94351, 77970, 81220, 81825, 82640, 279361, 84866, 219243, 221605, 225518,
    150317, 86032, 88886, 89018, 145102, 88423, 144124, 106596, 616607, 281104, 213967, 156504, 156568, 211101, 81292, 147535,
    213203, 213271, 149647, 149802, 153661, 158204, 159750, 160149, 161959, 360490, 165906, 428099, 174112, 176149, 347053, 353058,
    353063, 353095, 353133, 358646, 364726, 364740, 376862, 146109, 346315, 152042, 154145, 156148, 430168, 217262, 213959, 217992,
    221893, 221952, 484105, 221970, 487896, 622685, 491623, 622702, 622703, 626862, 503866, 352992, 221931, 80843, 428050, 76175,
    219303, 143917, 153649, 618770, 76288, 144397, 283006, 217857, 153853, 219541, 420031, 420234, 354735, 356473, 225452, 225597,
    227404, 426002, 360507, 297061, 432219, 434202, 152078, 146333, 346797, 154398, 352450, 620578, 362603, 235553, 628801, 632844,
    78967, 79069, 358409, 84251, 86229, 86479, 88858, 223517, 485859, 283528, 283542, 283550, 287341, 289251, 147343, 152375,
    290915, 86689, 80704, 278724, 278886, 280921, 280979, 485490, 487471, 291166, 75869, 90379, 284676, 220092, 211615, 80371,
    146038, 82068, 82711, 84391, 150071, 90201, 79372, 144916, 147567, 80174, 79738, 146936, 84885, 84893, 84900, 281513,
    84919, 281559, 84975, 84981, 152211, 90918, 90926, 90928, 90936, 90944, 90949, 90975, 287589, 90982, 287608, 287611,
    91004, 91012, 91014, 91020, 287630, 484274, 91066, 96431, 96459, 96474, 489694, 96487, 489727, 102590, 102615, 102616,
    102622, 102625, 102630, 102639, 108586, 108589, 108590, 114719, 144062, 146044, 215235, 215500, 283422, 485678, 485840, 618541,
    487629, 301177, 241696, 82798, 145678, 147632, 149936, 156265, 495736, 79807, 82292, 84090, 86191, 624676, 78841, 237574,
    368678, 243714, 145078, 90864, 212032, 217237, 219416, 225616, 144629, 146975, 283620, 483699, 100408, 626781, 77982, 147451,
    82668, 279346, 217949, 90429, 76224, 146585, 484121, 79275, 215558, 146651, 153899, 80540, 84365, 77969, 287221, 92496,
    289130, 292904, 233618, 301081, 303123, 303170, 305175, 145221, 295011, 78150, 80549, 84393, 147068, 280591, 282638, 289217,
    290869, 292877, 294976, 76146, 77962, 79580, 81165, 217686, 219506, 221562, 156135, 294942, 430160, 366681, 211796, 148066,
    497767, 356510, 215669, 350762, 350980, 352642, 352832, 287310, 354394, 354623, 356474, 487819, 358485, 620677, 362558, 628802,
    366664, 366693, 76778, 82665, 346930, 75983, 145437, 81233, 213551, 84047, 280728, 151754, 152361, 88689, 155731, 418344,
    100359, 167945, 233495, 80176, 84614, 84830, 86278, 88760, 90320, 96335, 424036, 626701, 104556, 76271, 82811, 215065,
    280829, 151764, 217555, 217647, 349036, 86917, 86935, 349089, 349093, 90840, 353014, 618934, 356793, 618951, 94665, 487882,
    225741, 356814, 622696, 622704, 368764, 84603, 482034, 80986, 346764, 150226, 155711, 157725, 485460, 159905, 161810, 628860,
    438287, 80074, 88426, 89012, 90500, 285662, 76523, 215868, 358438, 144335, 147630, 481763, 154531, 495761, 78630, 278621,
    213032, 88363, 155671, 169990, 144887, 150057, 484286, 430179, 80118, 430080, 80183, 84276, 90367, 221513, 225681, 278754,
    284721, 285471, 287343, 346411, 149953, 221658, 80904, 149784, 151686, 88090, 88277, 88771, 90492, 92386, 92515, 144633,
    150334, 76683, 78861, 79109, 278850, 151954, 283338, 217854, 352331, 289092, 354804, 356780, 297065, 150245, 154132, 354620,
    160084, 165996, 368726, 80027, 147746, 149886, 150153, 155653, 158021, 160031, 217569, 217230, 414666, 616905, 356458, 292956,
    145828, 278647, 148253, 348308, 350714, 90789, 354374, 288851, 279235, 150316, 84818, 81018, 417970, 419970, 428116, 143840,
    79982, 80060, 145960, 82210, 82511, 84350, 84617, 151598, 152371, 88219, 90117, 92211, 158004, 92485, 94495, 160095,
    160107, 168003, 348615, 350756, 212527, 149740, 151770, 217338, 86170, 92245, 92506, 442380, 76099, 146537, 148147, 148187,
    84222, 88943, 78646, 148459, 279531, 104450, 112662, 80518, 146841, 81602, 350507, 88510, 92338, 493678, 430146, 79969,
    80724, 80851, 82256, 82371, 82391, 82542, 82599, 83979, 84674, 84838, 84880, 348177, 86351, 88263, 90226, 90294,
    90551, 90710, 94591, 146182, 219228, 154549, 223334, 158197, 233584, 211501, 362527, 82107, 92227, 211604, 79079, 282716,
    414047, 414165, 294986, 79647, 82759, 82772, 82823, 483974, 484057, 356799, 94439, 146361, 84810, 154080, 485562, 144465,
    152419, 287477, 145129, 150046, 152280, 78395, 81457, 80730, 81160, 278822, 483877, 225292, 487491, 80876, 147551, 147783,
    152150, 156026, 287182, 424052, 227410, 143934, 146658, 213145, 213796, 219604, 223375, 493679, 362610, 495662, 104451, 632836,
    503816, 503818, 438302, 438305, 505864, 638980, 285018, 421986, 79622, 79663, 82820, 86890, 86902, 90810, 352964, 221908,
    90844, 90882, 356803, 360548, 98410, 102563, 364708, 102570, 102571, 237684, 114708, 287089, 153876, 418283, 221805, 223715,
    281174, 283090, 289271, 426020, 77961, 148279, 84043, 150385, 86210, 86460, 79441, 278612, 153996, 634926, 212354, 147858,
    144532, 223641, 143480, 147118, 150171, 420030, 78811, 156476, 174116, 78458, 78664, 147110, 213826, 280816, 153933, 350611,
    285330, 286833, 364627, 78136, 80055, 80453, 84612, 84667, 156600, 293052, 161990, 161998, 162012, 162019, 358637, 293107,
    162039, 174128, 174130, 180249, 180253, 146733, 80429, 86987, 356844, 356858, 630910, 630911, 79991, 80195, 88194, 434293,
    144480, 79411, 217948, 217966, 221928, 102574, 102576, 507924, 217464, 223512, 225341, 231505, 144819, 144547, 215958, 418602,
    418638, 222054, 418703, 222123, 424139, 227565, 430259, 233706, 626925, 436268, 211634, 147881, 279033, 213526, 148149, 150224,
    151932, 152190, 153873, 153907, 154073, 154226, 155709, 421943, 161835, 168050, 76655, 78626, 86124, 151903, 426017, 495667,
    215361, 86500, 88818, 286998, 90569, 352764, 287257, 289051, 354696, 421999, 94379, 422113, 422141, 356611, 422192, 422227,
    94597, 422305, 622674, 360535, 430115, 364671, 366611, 418636, 418650, 418659, 418706, 222133, 222137, 620714, 227513, 620732,
    227519, 227520, 227527, 620751, 424144, 227541, 620758, 424152, 424157, 424160, 620769, 424166, 620780, 227567, 620785, 620788,
    227580, 233651, 233653, 626872, 626874, 626877, 626879, 233665, 626884, 430277, 430279, 430280, 233673, 233674, 233676, 430285,
    430286, 626895, 626896, 233681, 233685, 626902, 626904, 233690, 430299, 626909, 430303, 626912, 430306, 233707, 233708, 626927,
    632870, 239656, 239659, 239660, 632877, 632879, 632880, 239668, 239671, 632888, 239673, 239674, 239675, 632893, 632894, 638998,
    442392, 639002, 639005, 639006, 442399, 81788, 301072, 231488, 143573, 211346, 221372, 616893, 79189, 281381, 145889, 147031,
    213648, 616469, 223541, 626719, 76471, 143793, 146001, 147606, 82469, 82547, 151865, 88354, 88729, 156014, 297075, 432196,
    497735, 487901, 212684, 233520, 279419, 283543, 418542, 418547, 144496, 348333, 352464, 354592, 292894, 491534, 75944, 81994,
    84591, 90612, 90691, 92201, 96349, 297016, 82154, 213751, 84448, 348393, 86662, 350284, 155735, 419915, 422208, 489513,
    145671, 150260, 78785, 144468, 150335, 416659, 626788, 144953, 156368, 78054, 144871, 80833, 82382, 282683, 284901, 288914,
    144074, 278535, 285128, 484253, 291059, 88777, 80438, 79658, 79665, 148142, 82779, 82817, 86910, 86926, 352977, 98404,
    98413, 346410, 485685, 229385, 362514, 79849, 211959, 147290, 82885, 161994, 297001, 630857, 636945, 215304, 414013, 283343,
    284921, 285079, 305188, 86140, 88207, 92501, 154492, 145531, 483940, 213350, 213431, 217142, 223347, 279487, 86578, 102496,
    84437, 84677, 415919, 221204, 221522, 90624, 223432, 94273, 100364, 143998, 414169, 622683, 624727, 497667, 432149, 170026,
    172075, 174080, 112657, 178203, 148374, 152461, 152487, 149560, 146616, 215228, 484052, 484110, 221972, 638994, 75936, 78454,
    79783, 80461, 148122, 84779, 92215, 620633, 76514, 148099, 483997, 220094, 212178, 81137, 81259, 84548, 219709, 88819,
    76332, 75904, 80062, 81948, 82061, 82731, 84436, 88398, 90136, 90498, 90688, 497781, 376844, 146443, 346440, 281219,
    348784, 416544, 354490, 354668, 281188, 79049, 144863, 217254, 348342, 225423, 301123, 80625, 428109, 430165, 78379, 146809,
    212646, 219469, 78214, 79709, 81849, 84328, 153658, 484277, 91067, 489685, 489711, 495820, 501803, 147665, 78478, 79300,
    144772, 80181, 285477, 156038, 156043, 156240, 157746, 158111, 161863, 161928, 150012, 416711, 215871, 293005, 80906, 350687,
    156229, 421934, 76366, 79392, 81896, 84181, 100443, 79266, 144383, 145374, 149584, 153678, 154134, 156193, 80374, 215113,
    225465, 94419, 231543, 150308, 495742, 82607, 84815, 86815, 417921, 159793, 360463, 368644, 370709, 78383, 80458, 213159,
    213358, 282846, 283446, 289096, 422089, 356701, 364566, 366639, 301129, 79802, 145758, 82795, 353230, 222180, 618988, 80383,
    163855, 172036, 174101, 78808, 156620, 156633, 156657, 156659, 160237, 168189, 144068, 80949, 152349, 221866, 288792, 414092,
    414146, 348799, 285034, 219564, 88931, 351138, 285620, 287108, 354604, 422256, 362567, 364662, 432231, 366712, 368677, 75928,
    86283, 285378, 289065, 94622, 100354, 100418, 102454, 144821, 213733, 282821, 233592, 83981, 413753, 90182, 79532, 81135,
    346123, 350407, 352438, 150304, 219225, 154082, 227444, 285275, 219914, 287087, 354333, 223294, 292868, 628836, 83973, 90328,
    90406, 90410, 90512, 90630, 92389, 96395, 76050, 78731, 284685, 154297, 420088, 180252, 347097, 353059, 358655, 218040,
    484305, 495869, 143907, 154591, 147782, 219734, 278807, 416282, 213948, 222200, 225771, 233721, 351207, 144807, 348217, 348437,
    150508, 156295, 156297, 159748, 219975, 221459, 223323, 420060, 616708, 225405, 144362, 78815, 80095, 84783, 225439, 144301,
    214007, 215199, 215426, 219164, 219665, 352294, 221787, 419848, 76086, 78159, 147157, 147933, 213808, 430166, 157703, 80187,
    86627, 88574, 78475, 145146, 86516, 211535, 168127, 285135, 221309, 227337, 82925, 213997, 79502, 79812, 84598, 88072,
    89083, 90288, 92385, 94500, 368718, 108568, 428147, 79236, 144796, 80642, 217096, 219697, 221251, 223413, 76169, 78202,
    146085, 84000, 149929, 84559, 86701, 86757, 94590, 78754, 82104, 147814, 147895, 149542, 151555, 86231, 152138, 153623,
    153727, 92593, 147653, 82425, 149551, 86142, 88289, 154342, 221383, 92179, 288932, 94426, 94643, 364648, 79272, 80350,
    81972, 364773, 104563, 76705, 80188, 279469, 80251, 82401, 212360, 80463, 80823, 147116, 149861, 84402, 86770, 88089,
    350851, 156134, 156244, 80365, 144827, 223454, 278799, 428086, 281414, 82563, 84463, 154198, 102504, 215342, 281170, 283385,
    223254, 356669, 618869, 356748, 143672, 147274, 351137, 79137, 213487, 76624, 78298, 144359, 144565, 145656, 148093, 76677,
    78623, 287390, 79635, 81500, 81520, 213802, 84318, 215810, 86449, 217706, 414382, 88161, 219248, 88621, 88809, 90401,
    90447, 90460, 221597, 90718, 484072, 92276, 92393, 92626, 94389, 94417, 94440, 94444, 94508, 356821, 98345, 362617,
    102560, 106593, 503843, 80035, 82239, 84370, 281055, 86040, 86697, 88264, 88631, 90365, 90522, 94427, 76285, 146607,
    154235, 90765, 223554, 76525, 287042, 162008, 354697, 161888, 82137, 84733, 88857, 90378, 81269, 82730, 106517, 106540,
    80655, 285467, 291061, 628855, 148251, 217915, 211662, 147249, 219238, 84702, 215461, 82207, 145029, 79064, 144739, 144286,
    78788, 350433, 354323, 356570, 356787, 212728, 81798, 84396, 84676, 86312, 86736, 88173, 88319, 219938, 90427, 92267,
    92363, 94307, 94596, 98389, 100361, 100385, 106521, 106524, 106525, 108553, 108554, 108578, 110609, 79577, 144756, 283105,
    152255, 219998, 220011, 416689, 416719, 487575, 165955, 90817, 90824, 90826, 616636, 75880, 81503, 76491, 82853, 81516,
    84097, 81715, 278531, 221496, 221872, 622647, 294968, 213862, 622691, 75788, 80247, 84406, 503833, 616653, 157722, 354805,
    78120, 143866, 145253, 146054, 146678, 281015, 217936, 217963, 217983, 217987, 217990, 154002, 154139, 154207, 285281, 154212,
    154227, 154306, 154472, 154534, 154545, 351154, 154575, 155959, 155989, 156079, 352716, 156119, 221663, 156139, 156168, 156217,
    418393, 156270, 484049, 221957, 288821, 616510, 157874, 223421, 157902, 485622, 157993, 223537, 485728, 158079, 354777, 158181,
    159813, 159919, 159961, 159988, 356693, 160111, 291192, 160153, 423962, 161900, 358557, 163843, 426021, 426028, 165964, 362589,
    428131, 428150, 297083, 167948, 233502, 495682, 495704, 366689, 366705, 366713, 172085, 434259, 307232, 221723, 145261, 147144,
    154053, 484058, 484117, 156574, 79129, 353261, 278537, 213577, 217251, 217404, 147532, 219830, 155976, 92546, 358521, 80466,
    81730, 483866, 146385, 86802, 88456, 154079, 418191, 145939, 147720, 152253, 153768, 417896, 426047, 413814, 79364, 147407,
    84002, 219463, 88548, 290906, 624685, 626763, 84786, 76436, 81906, 82718, 82733, 421974, 76668, 150412, 225568, 493668,
    145104, 152545, 78882, 282818, 145682, 148221, 148228, 155925, 156319, 223377, 626769, 144832, 346362, 90327, 426024, 422049,
    79780, 79833, 211941, 346575, 348755, 98424, 146510, 212682, 279426, 426077, 426094, 79621, 218092, 352959, 484291, 225727,
    618974, 495793, 237641, 172126, 241725, 349069, 225467, 356815, 356822, 368727, 149920, 283047, 88239, 153935, 285688, 100369,
    297004, 151971, 159918, 626796, 414412, 493595, 626745, 144945, 364560, 215436, 150016, 154459, 143582, 147028, 147865, 148260,
    150197, 153677, 290897, 82372, 353115, 353137, 353184, 358626, 376855, 81499, 88935, 84003, 215226, 86696, 86809, 88428,
    92517, 143563, 79226, 154378, 221551, 157758, 75889, 362612, 78154, 80956, 84233, 217152, 223233, 420282, 225644, 432211,
    143975, 77897, 80049, 81562, 481311, 88550, 90481, 353218, 78548, 77893, 79989, 288772, 288784, 299140, 301181, 79121,
    145099, 81619, 349120, 618980, 79140, 416325, 419940, 145267, 80432, 348609, 280727, 286931, 88237, 286720, 90132, 90450,
    92360, 174115, 76300, 350540, 285683, 354347, 356482, 282720, 285630, 144193, 481302, 82114, 82676, 212891, 148230, 432128,
    79382, 147240, 150437, 350590, 352354, 157956, 158109, 347113, 151632, 76464, 287018, 293003, 282740, 78810, 79259, 436250,
    146227, 148302, 148361, 149530, 149896, 151704, 151778, 152405, 152410, 153619, 155924, 156074, 156237, 287489, 287505, 158156,
    161889, 299171, 299179, 170032, 172136, 88823, 90148, 92497, 212442, 143802, 146990, 81724, 213626, 221656, 354530, 438295,
    349144, 82852, 349127, 487927, 78369, 80353, 80437, 81541, 86130, 88706, 154316, 90730, 157952, 145230, 147306, 487523,
    360499, 100388, 301150, 145646, 147576, 147644, 148115, 150102, 152018, 155990, 156236, 158199, 225386, 161884, 294948, 79024,
    144948, 356548, 80860, 81811, 84032, 84470, 219435, 90119, 90146, 92172, 92394, 104525, 146652, 176184, 90390, 90649,
    143916, 211214, 213168, 221801, 75815, 77828, 81001, 146592, 81485, 81632, 81860, 149945, 149986, 215655, 151858, 217832,
    88101, 88478, 219667, 155995, 156097, 223331, 618594, 356526, 422249, 294938, 294963, 432160, 78838, 145457, 145800, 212731,
    147476, 147628, 148146, 149887, 150349, 151600, 414258, 153893, 154432, 154520, 154584, 157917, 291162, 292976, 430176, 211326,
    213222, 151583, 88227, 219434, 88582, 88780, 89011, 285628, 92416, 223551, 94492, 227413, 281063, 217168, 282933, 287150,
    288933, 303109, 630815, 636937, 442369, 376847, 81922, 346443, 88412, 102401, 104452, 178198, 145486, 485415, 485416, 493695,
    368733, 146819, 152422, 152457, 152492, 156355, 156378, 156379, 156423, 156437, 422325, 422327, 422331, 422334, 422340, 291273,
    422349, 422359, 422361, 426086, 434291, 434294, 434298, 245779, 281236, 150373, 289071, 221998, 222016, 222086, 222130, 424110,
    424134, 424147, 424161, 227558, 424177, 424188, 430271, 430278, 430318, 632873, 632885, 442394, 639003, 245791, 144801, 144959,
    80865, 346383, 416009, 155785, 92345, 159753, 159808, 225623, 352855, 616458, 420168, 487773, 497711, 78424, 81525, 219420,
    223630, 283415, 90904, 487891, 503871, 346450, 284907, 235573, 78730, 481845, 348248, 217420, 151991, 350704, 221590, 414642,
    213877, 82828, 217953, 147517, 149711, 346375, 149894, 281078, 215998, 222079, 424112, 424122, 620731, 424129, 424150, 620776,
    620789, 424185, 430266, 430275, 626887, 430282, 430300, 436284, 76174, 217245, 88465, 285416, 364742, 86397, 219412, 155780,
    144152, 79193, 79380, 211077, 219800, 150460, 156442, 76798, 81932, 84283, 88230, 90637, 146020, 148001, 148024, 148098,
    215294, 150221, 152093, 152315, 154049, 220015, 156137, 419947, 157849, 92418, 157992, 94319, 159882, 94433, 94461, 94512,
    94547, 94625, 161842, 163846, 163852, 102433, 235572, 628788, 172041, 237615, 630831, 286775, 354431, 149928, 143410, 150287,
    90836, 229476, 291305, 354324, 421916, 426013, 143498, 146701, 146966, 283222, 219493, 285042, 285140, 287098, 223247, 419890,
    288911, 422013, 356497, 422226, 356739, 424043, 360478, 432217, 630876, 350890, 413911, 81822, 82600, 413698, 79156, 82920,
    218074, 145355, 212159, 278982, 301142, 84557, 220131, 144742, 211460, 146788, 281000, 143891, 81929, 82706, 283560, 212309,
    88861, 223689, 223708, 151853, 154566, 155935, 223693, 213178, 215302, 225488, 233602, 213556, 217269, 219256, 346614, 281446,
    150376, 287694, 156644, 616757, 80372, 146489, 81041, 144548, 211487, 282873, 278923, 422021, 424050, 229378, 491551, 360517,
    145142, 80854, 279209, 281440, 86048, 90161, 96312, 278905, 154424, 81275, 620678, 147481, 152378, 81662, 281408, 84819,
    92670, 146842, 223433, 422046, 75999, 78256, 144970, 88386, 154341, 493633, 213262, 148214, 150399, 219466, 154612, 81712,
    346259, 217434, 354528, 158006, 618539, 364554, 626709, 79012, 76613, 79475, 79004, 144172, 483972, 285407, 413868, 147778,
    213368, 82564, 148144, 148211, 149559, 84336, 217299, 151825, 152216, 152266, 152353, 348994, 219283, 219339, 153932, 154484,
    154596, 154597, 89075, 221259, 221483, 221591, 221792, 157739, 223368, 157853, 158020, 420277, 223732, 356357, 159874, 225531,
    422295, 370714, 420156, 215849, 221774, 288858, 616614, 422283, 628779, 301157, 79623, 79649, 79968, 82321, 82623, 213883,
    84864, 86864, 86891, 86911, 86915, 86918, 218007, 90885, 80680, 143568, 80102, 280889, 414295, 78706, 145106, 81242,
    82723, 78201, 80897, 100456, 80109, 82584, 84010, 84234, 86193, 86292, 360567, 149514, 217211, 152186, 354340, 223603,
    225422, 94387, 160100, 161806, 110603, 75965, 77870, 79886, 80314, 80315, 80604, 81526, 147130, 82015, 82201, 82247,
    82536, 84062, 151823, 86614, 86785, 88198, 285154, 88811, 90281, 90286, 90490, 221646, 92243, 92421, 94305, 229458,
    81532, 215887, 283291, 217887, 151950, 153917, 156326, 80861, 88678, 144515, 155877, 352918, 352937, 84131, 215464, 86188,
    88080, 81760, 147682, 152089, 153695, 154170, 154399, 156271, 159822, 145598, 147484, 147597, 149972, 153934, 155672, 156034,
    156224, 280628, 215402, 348890, 285567, 221592, 356417, 495748, 497789, 481895, 350995, 418294, 162046, 296985, 231477, 303120,
    211672, 619006, 168133, 168136, 212185, 280759, 88260, 88514, 88730, 90374, 221486, 92553, 96371, 98349, 100434, 414485,
    294994, 75966, 78042, 79007, 79424, 76684, 349010, 484041, 225746, 622698, 495784, 626863, 233650, 145532, 147494, 348999,
    286869, 418051, 418087, 419886, 419994, 485823, 289265, 483880, 94641, 146142, 149663, 153628, 286758, 156251, 213270, 75791,
    77880, 81617, 81630, 82420, 215128, 84698, 219406, 88395, 92310, 94490, 146320, 84473, 84694, 86572, 88704, 285388,
    223312, 92594, 94484, 96408, 231498, 154225, 165895, 299031, 440327, 145074, 153857, 420117, 143406, 145422, 145851, 147062,
    278746, 147737, 151649, 217359, 217470, 152321, 350690, 154347, 155822, 159841, 213338, 221647, 146669, 278828, 211486, 213488,
    78005, 80409, 215057, 86512, 153856, 90622, 92544, 102464, 78918, 150041, 84221, 84530, 307246, 146723, 484351, 620586,
    622615, 301059, 145754, 147247, 148087, 149827, 151907, 154091, 155791, 156050, 157769, 160152, 292944, 161899, 168007, 84070,
    218098, 144148, 146310, 84115, 150332, 499801, 78096, 215917, 150475, 168022, 217758, 221235, 153821, 239635, 80478, 82020,
    84384, 86922, 88399, 80618, 419926, 145935, 81408, 356862, 145911, 352999, 618957, 491620, 626854, 630898, 147489, 147694,
    154067, 154481, 157913, 290986, 160012, 78907, 144488, 84795, 87032, 143850, 86519, 426008, 491560, 436254, 438325, 281491,
    287568, 287576, 287640, 618552, 293093, 293117, 293118, 299188, 346341, 150064, 157904, 364765, 211306, 176165, 144454, 493590,
    366625, 80459, 76489, 144292, 215656, 290820, 233482, 77866, 80615, 81288, 81462, 81877, 82051, 82294, 82386, 84053,
    84640, 86297, 152120, 88554, 417793, 155700, 90363, 92278, 92412, 94586, 167987, 102543, 170084, 106599, 78983, 348918,
    281453, 80535, 280616, 284808, 211965, 630897, 145173, 278588, 148373, 152446, 283527, 285441, 287459, 156428, 422344, 291283,
    163937, 168105, 438332, 281160, 354795, 144814, 78906, 144911, 90770, 157943, 296979, 150489, 283496, 156509, 156527, 422350,
    162047, 295010, 299174, 299185, 303218, 442386, 79015, 219188, 78725, 76061, 80397, 146974, 147725, 82417, 82493, 84035,
    149716, 150168, 150222, 151993, 88107, 219425, 92232, 92302, 92633, 159759, 225596, 96338, 144757, 215807, 153687, 303203,
    483711, 288975, 144086, 148337, 148367, 148368, 88652, 88928, 89032, 221538, 418516, 156389, 418564, 418574, 156436, 160184,
    163936, 430255, 144799, 80957, 280791, 145113, 237632, 505862, 144455, 79240, 90805, 278979, 84475, 215595, 215751, 286834,
    92402, 485704, 352623, 143532, 149917, 144930, 211010, 79955, 81862, 82050, 219403, 88784, 79279, 145954, 213341, 279018,
    215834, 225454, 432247, 241670, 176169, 75877, 77912, 78367, 81549, 147263, 213066, 84068, 149930, 215485, 86693, 283572,
    152501, 152565, 219353, 285481, 416679, 221229, 287732, 288881, 158017, 225445, 94377, 94510, 94521, 96304, 292915, 96340,
    424023, 163907, 297032, 299139, 430330, 497740, 235632, 86836, 89068, 90791, 215222, 84901, 84968, 284841, 285440, 285607,
    352691, 156230, 287518, 484130, 484156, 484159, 287578, 287587, 287634, 91042, 287652, 287657, 161843, 489642, 489664, 489706,
    96494, 293104, 293109, 489718, 430092, 168067, 495812, 299205, 299208, 495829, 495847, 495855, 366668, 501798, 501822, 507934,
    81355, 82385, 84254, 154580, 92472, 102465, 104537, 368740, 145588, 147965, 217614, 154107, 213791, 143748, 78531, 78948,
    147753, 149949, 144459, 82860, 143942, 149848, 86570, 152231, 88147, 286777, 90248, 156083, 493645, 104557, 80600, 279508,
    487923, 79107, 413704, 416305, 78836, 146650, 81692, 350291, 88188, 416185, 154426, 483728, 221619, 158175, 358502, 211731,
    213858, 217932, 217934, 221883, 634939, 634942, 160207, 160214, 78527, 78629, 144862, 79906, 82318, 82364, 82722, 82758,
    149678, 84649, 282699, 151716, 154040, 88696, 154441, 289043, 161890, 299101, 78950, 147772, 86433, 219150, 219788, 221376,
    289005, 227481, 78546, 347128, 144859, 81900, 213012, 147650, 279185, 213682, 149611, 215284, 215885, 86657, 219968, 358512,
    293006, 628815, 501786, 84585, 303194, 84071, 212971, 281052, 219619, 282715, 620595, 235520, 80357, 147951, 149509, 155864,
    155942, 155946, 156048, 156166, 157925, 161931, 418487, 418496, 434289, 146342, 148103, 287028, 287266, 213727, 281090, 346946,
    483515, 147187, 219682, 221443, 221479, 223278, 223386, 282657, 624651, 79490, 221632, 223597, 225469, 303207, 81125, 78781,
    144735, 146531, 215259, 156486, 162001, 162007, 162033, 353202, 78806, 79695, 146891, 84903, 84929, 84984, 287516, 90917,
    287528, 287546, 287556, 287569, 90965, 287577, 287580, 90981, 484202, 90994, 91008, 91018, 91029, 293033, 96457, 96461,
    293075, 96468, 489710, 489715, 102596, 102617, 299228, 495838, 299239, 102637, 108598, 79162, 79662, 94667, 360546, 364710,
    368754, 106619, 212026, 149991, 150357, 281458, 150393, 151861, 154321, 154594, 156078, 157861, 166003, 278579, 418093, 287201,
    287290, 303210, 78682, 76249, 81201, 81561, 81801, 217276, 86717, 88196, 88232, 88291, 88384, 88500, 88589, 88919,
    90153, 90398, 92263, 92444, 223609, 94463, 94470, 96373, 96383, 233549, 430196, 213909, 217982, 418365, 352995, 221964,
    487880, 356819, 229473, 233641, 503868, 76675, 75927, 82849, 284703, 285065, 88949, 352459, 485524, 426066, 150066, 79295,
    215800, 219295, 430145, 145385, 148421, 152521, 156647, 156670, 160222, 79418, 143832, 212829, 346408, 348187, 90557, 215141,
    217930, 219672, 283268, 285147, 618751, 143995, 217545, 147007, 149700, 88164, 153832, 351088, 156128, 156194, 221826, 354711,
    630875, 634908, 114694, 77836, 81091, 79361, 146451, 212083, 279050, 213711, 347043, 217607, 291144, 78965, 79112, 90166,
    149969, 151901, 76094, 146411, 146589, 147799, 279422, 282796, 284688, 418526, 299147, 79217, 79583, 219941, 84125, 86147,
    90184, 79684, 82825, 86860, 86862, 86874, 349101, 90811, 90858, 90884, 503867, 503870, 76325, 147317, 84340, 346606,
    348541, 416329, 96258, 622604, 622673, 165911, 233501, 168071, 104458, 169996, 104466, 172046, 305185, 436258, 80243, 80469,
    81399, 213398, 418005, 484292, 289234, 430231, 364799, 499720, 434245, 438286, 358416, 626775, 370695, 144503, 213254, 279210,
    219985, 80569, 81083, 279171, 418057, 76287, 285029, 495713, 75871, 84875, 285408, 156049, 436243, 618975, 368767, 82337,
    84750, 161980, 356485, 358514, 426015, 294981, 350684, 287038, 487735, 356757, 424002, 491588, 364624, 499819, 370720, 84061,
    88538, 225743, 225750, 84768, 88546, 76461, 143778, 146823, 147172, 215700, 154414, 156141, 352891, 485582, 624712, 104569,
    350234, 149591, 151917, 628795, 78952, 152103, 278837, 215696, 348378, 217552, 81187, 81967, 151782, 87008, 278652, 90795,
    225607, 292984, 82553, 76190, 80225, 147847, 215092, 215274, 219518, 221282, 147781, 151902, 151905, 153674, 153915, 156310,
    156329, 161933, 145340, 81120, 81716, 88685, 88766, 287709, 422380, 78670, 484314, 88365, 291048, 225635, 292941, 368722,
    79440, 356435, 144070, 148393, 80452, 281127, 84204, 86234, 88159, 215483, 352262, 76047, 77928, 80289, 150264, 153759,
    364609, 213209, 215805, 219929, 422242, 423958, 434239, 278703, 149780, 282792, 283197, 153655, 154205, 354537, 158057, 159782,
    353069, 353127, 358610, 146758, 76686, 350261, 154214, 146520, 279522, 146735, 150157, 76322, 77927, 149840, 150386, 153972,
    144064, 350294, 487569, 495794, 151951, 483707, 485493, 78167, 78599, 80067, 84194, 86591, 88566, 88963, 104574, 76039,
    78238, 149871, 290920, 144199, 358629, 78866, 76415, 286773, 289260, 280660, 281087, 287079, 150429, 417917, 80226, 146768,
    213307, 215616, 86426, 153980, 88879, 221545, 430200, 145559, 149833, 352267, 288768, 291030, 624648, 241702, 309257, 78813,
    221839, 219268, 144322, 76205, 78516, 350498, 350873, 156261, 158132, 78728, 484034, 80995, 79599, 430198, 280637, 285175,
    290856, 145568, 152176, 160023, 422289, 76719, 79603, 79952, 147460, 147633, 149543, 84232, 154518, 90530, 156283, 94258,
    94503, 624671, 102407, 172042, 176155, 176171, 176180, 211449, 215604, 158059, 163892, 215083, 217416, 79463, 80424, 80915,
    76439, 80722, 147983, 622600, 100427, 102537, 81132, 217738, 225613, 146645, 76512, 346180, 143629, 80134, 80806, 147300,
    82567, 84207, 219368, 221664, 221730, 158177, 620673, 624694, 221881, 229483, 626848, 626855, 283055, 348644, 144235, 145038,
    147531, 217486, 219453, 352311, 221862, 212868, 80107, 81347, 84069, 215251, 348828, 86858, 88600, 96324, 102520, 144027,
    149589, 151833, 155899, 157814, 160122, 424032, 168079, 79061, 80231, 82163, 90287, 144111, 217537, 628767, 90275, 76776,
    145122, 143404, 147824, 146144, 280652, 283057, 155757, 420161, 291025, 356665, 424026, 426051, 624759, 144291, 221313, 213867,
    360557, 144839, 84784, 156523, 79427, 237581, 216005, 418670, 418693, 418710, 418735, 424105, 424135, 424190, 430289, 626913,
    233712, 436276, 287308, 484166, 616924, 78719, 223264, 90112, 223343, 499717, 638977, 146260, 149758, 154479, 288890, 499760,
    176151, 634909, 215982, 215984, 215987, 215992, 216024, 216044, 151967, 416175, 221994, 221996, 222015, 222017, 222018, 222035,
    222060, 222095, 222100, 222117, 222135, 159950, 422112, 227502, 620721, 227507, 620736, 227532, 620750, 227553, 424181, 227578,
    424187, 626890, 233675, 626901, 626919, 632872, 436273, 356680, 284698, 147277, 350523, 346352, 350866, 78662, 75799, 88447,
    88501, 90572, 90644, 223473, 225540, 227468, 624725, 233558, 152364, 76643, 419979, 428127, 430203, 225371, 106544, 241693,
    442381, 218013, 147495, 416751, 422118, 495724, 157919, 94546, 76425, 149768, 348512, 420018, 616921, 422063, 227389, 283571,
    283598, 291292, 144654, 484288, 212667, 420133, 144149, 77867, 79910, 82168, 82340, 282625, 86529, 88257, 284898, 284923,
    285373, 285646, 285654, 287410, 78182, 78959, 76508, 418169, 79160, 81103, 88355, 78315, 217347, 289062, 278962, 418363,
    231473, 143888, 146072, 144194, 81981, 84175, 84358, 143435, 147872, 149779, 217196, 282975, 414695, 159951, 161812, 161817,
    144717, 82097, 86052, 89080, 483383, 100396, 143555, 229474, 79337, 146789, 81313, 616468, 92637, 620555, 161836, 168085,
    497672, 110618, 145126, 213243, 213712, 215730, 217243, 217284, 219834, 237661, 147220, 215546, 219545, 422195, 217390, 79715,
    79717, 84884, 281536, 84933, 84940, 281555, 281577, 90913, 90971, 484211, 287632, 489654, 96482, 108595, 143494, 144979,
    145845, 146367, 149541, 160029, 160115, 163900, 79963, 415798, 213938, 79172, 88096, 225571, 212213, 487935, 145957, 90683,
    349037, 210974, 213078, 213692, 215043, 215898, 220024, 220040, 223480, 144490, 481362, 285123, 79394, 156489, 79610, 82198,
    150486, 161970, 161992, 145570, 152205, 90845, 90878, 79366, 75781, 215640, 221221, 223679, 163866, 301088, 144299, 79964,
    82373, 84172, 89014, 90206, 90756, 98324, 98374, 98394, 104503, 79505, 81339, 82248, 86560, 89019, 92480, 100435,
    284832, 346937, 81076, 288868, 154329, 151926, 156554, 287710, 147714, 149712, 149863, 618533, 79178, 350254, 154244, 221871,
    143537, 78707, 145875, 346319, 285228, 147341, 149916, 152060, 426103, 487595, 356743, 290967, 352868, 346425, 283330, 78666,
    76194, 211466, 213872, 217957, 235554, 413958, 79436, 143657, 284953, 155970, 80265, 211386, 212300, 86619, 284694, 88270,
    285670, 354521, 356679, 301165, 144251, 279062, 348328, 86201, 88095, 88457, 145093, 80021, 81168, 84110, 149792, 86110,
    243740, 638987, 145709, 80868, 147558, 281285, 84742, 217472, 86406, 152130, 152224, 88331, 153877, 219756, 156144, 158104,
    223718, 159865, 291123, 358461, 165994, 434287, 227428, 229467, 231511, 76703, 156531, 638984, 80415, 83974, 415807, 620631,
    283236, 144367, 351175, 78825, 79102, 150060, 284726, 144161, 146854, 616578, 624724, 144090, 348993, 432131, 104490, 106539,
    108544, 112667, 211643, 215424, 217703, 221710, 221791, 485829, 233569, 143921, 146784, 218090, 222182, 225767, 225769, 225777,
    624688, 152359, 278751, 416266, 212753, 150453, 79573, 149623, 155893, 352384, 353074, 353109, 353118, 370729, 96334, 213109,
    82875, 215354, 349167, 223431, 79210, 217935, 349017, 349041, 218003, 352980, 225725, 225735, 229488, 110653, 279520, 221593,
    148282, 81066, 350881, 346289, 144879, 147003, 143385, 146263, 146666, 285020, 352827, 163872, 366594, 301060, 374806, 79936,
    147591, 82165, 425992, 165988, 78544, 144805, 279333, 156431, 352646, 354377, 354719, 358536, 78986, 168170, 146423, 81754,
    350820, 156005, 79042, 221534, 223260, 419882, 223391, 354802, 227417, 233477, 235529, 237602, 241711, 211113, 294982, 213131,
    144410, 148030, 154213, 222181, 414486, 92656, 79239, 348850, 289049, 163890, 211744, 146249, 217100, 219348, 219504, 231483,
    235548, 143449, 81597, 86409, 82186, 215729, 88642, 353205, 160040, 356770, 227422, 358613, 358639, 364748, 370731, 307238,
    374793, 278878, 284978, 86033, 283016, 147219, 279285, 219137, 211551, 84177, 84268, 151567, 148226, 86616, 364570, 76445,
    78231, 82231, 88074, 88740, 92165, 143659, 144227, 223387, 151628, 146725, 223371, 231519, 233538, 233560, 233595, 146238,
    481324, 102547, 144272, 145615, 213456, 144618, 80960, 79155, 350656, 78056, 78236, 80946, 82077, 84164, 84320, 84425,
    86353, 88296, 88941, 90592, 90658, 491544, 82113, 348240, 78925, 108579, 152114, 168032, 76558, 79562, 145366, 78576,
    282738, 290941, 221645, 278619, 279350, 280650, 279152, 146220, 76456, 77884, 80542, 82331, 82355, 84708, 86401, 88286,
    88388, 88893, 90723, 414213, 168110, 144384, 418754, 211933, 88154, 92167, 78539, 79149, 152542, 418779, 231424, 144813,
    358594, 219981, 223346, 223524, 225327, 78650, 213838, 217941, 353025, 353041, 146865, 212480, 356642, 366592, 279421, 152429,
    152489, 152490, 283563, 156359, 287466, 299184, 145593, 212372, 212811, 153940, 94231, 148106, 149799, 84426, 151995, 144611,
    144624, 151662, 346178, 280687, 349002, 350683, 350744, 352596, 354474, 279264, 354749, 352924, 76768, 77895, 80717, 81180,
    146774, 82080, 84286, 75829, 77918, 78323, 146825, 88385, 495686, 626760, 628752, 370694, 79928, 146874, 81861, 82257,
    82658, 281056, 84568, 283270, 90149, 92607, 348630, 79293, 84348, 78680, 81844, 78398, 79488, 80298, 81122, 81227,
    147609, 84491, 86326, 86852, 88106, 154068, 90204, 221660, 90619, 90804, 157839, 92398, 92534, 290917, 94644, 299137,
    102538, 104526, 104559, 106551, 106573, 106588, 280682, 280872, 346631, 282791, 283401, 143644, 215101, 150058, 219639, 225320,
    299012, 78694, 352350, 352900, 360538, 288990, 78586, 283576, 152556, 418755, 418802, 287741, 145719, 213383, 149812, 88392,
    155903, 80650, 147267, 280606, 416548, 421965, 79754, 156492, 168162, 143521, 145780, 212460, 150101, 84569, 84587, 86521,
    86748, 88419, 90251, 483531, 217336, 348949, 213189, 215087, 284691, 219878, 286919, 286996, 287309, 354468, 487495, 292961,
    491581, 430184, 301104, 497762, 499797, 78909, 213851, 88298, 79625, 217955, 353013, 295007, 81774, 84669, 86857, 487618,
    616545, 624677, 436230, 76676, 144222, 150337, 348172, 487508, 211991, 84443, 86313, 88336, 88752, 90437, 92460, 487568,
    100423, 102518, 104568, 150470, 150493, 78474, 279271, 78502, 219838, 77860, 77971, 80530, 80758, 82374, 149979, 84721,
    86169, 283485, 153854, 287454, 420223, 104534, 149723, 211734, 279255, 80178, 481981, 227376, 233585, 616573, 618693, 144143,
    144342, 149655, 215277, 282750, 146306, 212804, 146646, 278884, 213617, 217471, 348906, 221243, 426000, 146487, 430249, 303225,
    360556, 144215, 491552, 626689, 628740, 143439, 146429, 146431, 153666, 161946, 149963, 153891, 90895, 144711, 283447, 287394,
    146563, 483959, 90902, 76761, 148463, 147911, 76350, 79248, 212665, 84330, 487879, 491632, 79150, 215523, 81403, 84208,
    86116, 154003, 88722, 89043, 90198, 92628, 94274, 98362, 168019, 628818, 80408, 81203, 82063, 84244, 348457, 88119,
    285223, 94246, 233536, 219615, 78081, 81394, 211365, 346715, 354701, 495755, 212654, 215745, 219209, 219512, 219569, 169992,
    220113, 424095, 491530, 489559, 156597, 353279, 162005, 162031, 162044, 168140, 174123, 174124, 348806, 81192, 358425, 153844,
    158074, 212188, 487859, 75862, 86448, 78430, 75964, 81930, 84389, 84697, 86625, 88753, 219887, 483423, 432255, 281169,
    484127, 484172, 484186, 484195, 484210, 287617, 484242, 287669, 287673, 293049, 293055, 293056, 293063, 489680, 293077, 489688,
    489693, 489696, 489702, 293103, 293116, 299084, 299187, 299189, 299201, 495813, 495815, 495816, 299209, 299210, 299212, 495821,
    495822, 299217, 299221, 299226, 495835, 495839, 495842, 299243, 299244, 305192, 305195, 305196, 305204, 305207, 305209, 305210,
    305211, 507928, 507935, 76734, 91100, 78400, 145557, 80454, 80839, 146704, 82145, 149646, 215328, 151646, 78263, 144034,
    80652, 80912, 86977, 349169, 280599, 287340, 82474, 86903, 349088, 90865, 484084, 102561, 102562, 114707, 79165, 211074,
    215058, 217446, 79840, 143509, 80110, 81905, 82040, 82108, 82579, 84566, 150250, 151610, 86482, 90454, 90700, 92663,
    159850, 94454, 96348, 79197, 79854, 75821, 81033, 88512, 88944, 92299, 92567, 96272, 100447, 102466, 102488, 102523,
    483534, 483679, 616944, 491554, 495619, 497671, 145163, 180245, 212863, 148344, 152403, 287483, 291275, 295016, 88073, 81070,
    482293, 221644, 487425, 78316, 80412, 348365, 354575, 75857, 78469, 78945, 81214, 81318, 81973, 215582, 217129, 217348,
    217816, 219606, 221725, 223611, 285218, 286979, 221461, 287015, 288814, 288922, 291290, 348860, 147158, 90341, 150118, 152287,
    152483, 418575, 422356, 348923, 481754, 156521, 143653, 145896, 87002, 418528, 364783, 79591, 81651, 90306, 100459, 79729,
    79736, 84915, 84942, 413862, 290860, 102588, 299242, 170000, 501804, 84151, 78397, 154295, 485492, 166009, 618622, 212764,
    278858, 215213, 354506, 354636, 289247, 368691, 144245, 147447, 150313, 483800, 618633, 424008, 426056, 110621, 311309, 80014,
    88805, 428097, 283521, 147710, 220023, 86686, 90163, 143497, 146660, 147568, 213779, 149579, 151746, 151871, 217427, 151996,
    154437, 220050, 157848, 159902, 144287, 147533, 362594, 281507, 145411, 217104, 219958, 220090, 416738, 94218, 219413, 630883,
    81431, 76170, 80359, 82166, 145681, 417803, 159783, 225458, 172062, 217951, 349033, 349061, 349155, 424155, 180231, 84192,
    80120, 147988, 76520, 75998, 80586, 146938, 84108, 280853, 280912, 86550, 285044, 356581, 213977, 349142, 144555, 144358,
    146243, 80850, 215575, 146478, 81414, 215644, 157779, 157957, 222008, 222028, 418681, 222110, 620753, 626908, 233698, 430309,
    626920, 436266, 632887, 78777, 79706, 79707, 84913, 84982, 84988, 483985, 90935, 90954, 90978, 484206, 484229, 484246,
    91046, 484271, 489726, 102608, 495825, 213893, 86096, 425997, 294934, 428055, 231461, 233596, 213888, 283262, 284945, 286781,
    364764, 76204, 149830, 86458, 288902, 299113, 233559, 143470, 147089, 215169, 418354, 419981, 420327, 78704, 84054, 144582,
    84693, 86731, 487746, 96330, 96344, 102450, 102482, 102556, 78512, 145012, 84802, 88951, 287415, 287424, 291249, 303217,
    145966, 147809, 484003, 147676, 144557, 213852, 217944, 217947, 217998, 221946, 146455, 358431, 279543, 215734, 288903, 143843,
    483388, 150425, 150501, 221707, 618798, 161976, 348607, 348940, 297063, 143364, 79786, 147021, 147168, 347048, 285566, 420010,
    422065, 292986, 370726, 370750, 638982, 76422, 147441, 149544, 217146, 153736, 616534, 420003, 420020, 292936, 294984, 624647,
    434211, 143483, 78052, 145622, 81213, 84376, 150232, 86359, 281386, 144327, 145491, 147398, 147586, 154320, 154271, 221797,
    227477, 364591, 233583, 347012, 86338, 86735, 88471, 145359, 84135, 84280, 86172, 90433, 90655, 92494, 144479, 90277,
    94322, 150273, 151675, 86206, 86650, 156318, 158067, 225381, 159886, 422057, 168030, 233601, 78020, 152316, 155768, 157735,
    166015, 143595, 416071, 219882, 81249, 82159, 82713, 88303, 90615, 288942, 293026, 237670, 143589, 346255, 150054, 348298,
    416389, 352782, 352863, 287330, 289199, 487450, 294971, 495766, 372757, 146583, 147385, 81935, 213073, 84018, 215229, 215241,
    281482, 219614, 350775, 219848, 418355, 80894, 80954, 213445, 215094, 215244, 346742, 217201, 217324, 217403, 220086, 221278,
    221576, 221828, 354529, 620572, 229396, 229446, 229466, 626824, 235575, 630862, 370712, 146439, 149719, 151611, 152390, 153829,
    157842, 423941, 434187, 80983, 82753, 148225, 280940, 622646, 79246, 84797, 418466, 143604, 144315, 149966, 151749, 154401,
    172115, 79124, 217139, 217728, 219546, 223314, 225611, 78649, 78817, 79756, 79811, 79816, 212760, 82839, 82851, 82876,
    86969, 86974, 86986, 86988, 86992, 86996, 87006, 87025, 87030, 87035, 87038, 91074, 91076, 484296, 91081, 91088,
    91093, 91094, 91102, 91104, 91112, 91128, 91129, 92380, 289028, 94683, 94685, 94687, 94694, 94699, 94710, 94718,
    98417, 98422, 491641, 98430, 98431, 102642, 102643, 102644, 102645, 102649, 102650, 102651, 102652, 106621, 144212, 211560,
    146519, 81395, 212545, 215293, 149828, 221211, 356637, 288888, 145345, 147337, 147848, 172039, 239633, 372792, 146846, 348370,
    350763, 154433, 155992, 354567, 213176, 223665, 80426, 287684, 287714, 87013, 91070, 143814, 78588, 76206, 77955, 278996,
    84490, 217853, 219688, 219712, 221447, 223478, 223502, 92550, 225354, 225456, 225470, 225692, 227360, 296973, 215130, 221364,
    350466, 353096, 353128, 353155, 88660, 618525, 622639, 144343, 157897, 78164, 79319, 82296, 146875, 165971, 143945, 81663,
    82527, 82571, 149806, 84589, 282866, 152164, 283366, 350588, 88529, 285487, 88899, 90449, 157718, 485413, 354508, 92640,
    225515, 291231, 80497, 81622, 215142, 80496, 80788, 281224, 219991, 289097, 491523, 281119, 86239, 283148, 284960, 88709,
    287244, 419867, 291131, 294930, 303130, 303169, 146917, 279151, 420224, 364641, 145133, 145658, 147376, 147698, 147963, 149955,
    79123, 212425, 152165, 219189, 219233, 88401, 155701, 155971, 156022, 157946, 158068, 158185, 158188, 225294, 225416, 94511,
    161925, 229393, 163903, 493689, 211078, 211260, 79641, 481628, 360480, 495617, 168128, 497668, 77890, 144987, 82684, 215479,
    281415, 219745, 285652, 147194, 147426, 82726, 215166, 150267, 84844, 217430, 78807, 145448, 152268, 156106, 160078, 160123,
    79915, 82297, 420027, 225509, 285230, 235627, 430135, 76642, 146993, 352651, 419938, 356760, 364615, 493651, 143598, 145539,
    145677, 147666, 148286, 215636, 283434, 285039, 352487, 156221, 485541, 368715, 146136, 218001, 281410, 495747, 78457, 303219,
    303222, 303226, 416433, 151779, 154097, 154309, 156063, 76254, 143650, 146057, 146991, 212862, 149796, 152033, 154196, 417863,
    354376, 362521, 297013, 428118, 430097, 628763, 172045, 368656, 630808, 143490, 145776, 217509, 143535, 78250, 80131, 280677,
    149793, 84484, 217701, 283466, 415810, 219465, 285208, 154308, 285495, 285518, 220101, 221523, 156138, 288901, 157911, 289045,
    289070, 158044, 158047, 289232, 158186, 291021, 291037, 225634, 227350, 292981, 358544, 430186, 366641, 237631, 283153, 78267,
    144390, 211295, 279246, 84205, 219264, 78660, 414686, 434303, 75864, 78463, 213501, 84785, 347070, 150483, 150507, 150527,
    156560, 86780, 90232, 92199, 100479, 483457, 104464, 79683, 82793, 90822, 90830, 90860, 90877, 221650, 481507, 154121,
    156053, 487605, 77959, 81048, 491639, 364586, 80916, 150356, 79983, 212935, 213098, 82501, 215273, 282695, 282884, 217378,
    219193, 219894, 219921, 220135, 286861, 288815, 223330, 289021, 289034, 92578, 225564, 225610, 227466, 153810, 145003, 149914,
    154025, 89041, 92667, 92328, 229468, 152325, 79563, 418062, 420137, 225694, 360481, 231426, 231490, 233526, 497746, 413879,
    86682, 212419, 155944, 155950, 156170, 156195, 159830, 227396, 495692, 145544, 148077, 215564, 153627, 153812, 153968, 154018,
    285194, 154564, 157711, 354354, 358507, 165947, 170012, 144502, 280766, 283002, 624728, 154353, 154453, 90416, 94592, 495710,
    215496, 286771, 210976, 147709, 213701, 279318, 148250, 149904, 151850, 284704, 88377, 154030, 351210, 155908, 221515, 223420,
    223667, 159972, 215549, 78207, 78985, 145100, 281457, 152020, 219280, 82848, 86976, 91108, 215050, 225636, 211700, 168150,
    82785, 86602, 152346, 79442, 146501, 414143, 159878, 278657, 106577, 176178, 241719, 144339, 81021, 84124, 84719, 152209,
    154188, 219727, 154464, 154568, 291071, 160041, 225627, 301093, 366654, 82494, 146190, 148224, 221219, 156334, 223614, 82924,
    213996, 88444, 281212, 79542, 78239, 80898, 80921, 81642, 84410, 84680, 88084, 88429, 88769, 89081, 90235, 90755,
    147301, 145291, 168182, 168183, 152393, 287400, 104507, 110594, 638978, 180230, 148385, 75968, 79620, 82768, 82787, 82827,
    86871, 221424, 90815, 78655, 616498, 422183, 161799, 284800, 76623, 86677, 152281, 76736, 80551, 80609, 82218, 82233,
    155987, 158171, 78142, 145736, 80891, 152068, 153767, 88687, 212218, 79994, 84227, 86100, 88534, 90653, 92539, 150113,
    350751, 285681, 422028, 297071, 166002, 299054, 432167, 366656, 303136, 632843, 307208, 307210, 241694, 241697, 215698, 217682,
    235639, 143396, 146066, 88118, 79772, 211013, 80306, 80414, 80763, 82055, 82416, 217534, 88658, 100452, 76132, 213224,
    217770, 419844, 354388, 497788, 307215, 78102, 281283, 217856, 350823, 219923, 156149, 287222, 616466, 158058, 618634, 159895,
    94560, 161832, 620584, 161851, 163886, 426069, 428041, 165913, 100405, 624693, 167950, 102546, 102552, 104459, 497675, 170053,
    172048, 172051, 172098, 630850, 436233, 174103, 632855, 174105, 176170, 110646, 145722, 346563, 286966, 376836, 311313, 213998,
    79716, 79725, 82933, 84953, 84969, 152575, 90930, 90959, 90973, 90974, 90987, 90989, 90991, 90993, 90996, 91015,
    91027, 91038, 484258, 91051, 96438, 489662, 489669, 96475, 489705, 489714, 96511, 102591, 495849, 108585, 414155, 147357,
    148130, 84089, 86428, 432245, 311308, 75855, 80899, 81349, 213250, 215248, 215399, 88116, 353256, 81916, 618868, 76005,
    484059, 356378, 364694, 88981, 102557, 104511, 146921, 81813, 493602, 348349, 143575, 146881, 147129, 79948, 81964, 280679,
    215382, 84852, 86596, 219331, 284979, 88463, 285145, 219999, 88958, 285692, 287293, 225291, 487521, 292907, 96378, 102544,
    366651, 104509, 372738, 442374, 144234, 149650, 354688, 145617, 148117, 211710, 146186, 80598, 149610, 149800, 150263, 151719,
    283369, 86768, 152329, 92371, 159773, 94250, 94341, 94412, 96320, 96325, 96370, 98314, 163871, 362535, 364559, 430113,
    104483, 104499, 301113, 104519, 499730, 114697, 114703, 76041, 81524, 144601, 146326, 213183, 144386, 149561, 151977, 153719,
    155832, 156019, 158054, 165939, 79032, 430142, 78853, 281199, 76279, 217638, 143895, 285045, 348478, 219271, 352510, 289192,
    420291, 76543, 79396, 354413, 153731, 156018, 354535, 145708, 217567, 485589, 495705, 148455, 287707, 422382, 430331, 78051,
    81461, 212681, 352385, 146340, 149558, 149569, 151665, 151788, 151867, 155799, 156040, 288993, 96300, 305176, 75849, 76626,
    77847, 78162, 212235, 217972, 221880, 221901, 221907, 221922, 78526, 146508, 217105, 352956, 352962, 484047, 352985, 215445,
    352871, 221976, 360460, 212424, 80039, 84359, 86391, 88249, 143388, 145627, 146731, 145832, 82173, 88494, 90372, 299253,
    170011, 78716, 213228, 413827, 485440, 487854, 75827, 78426, 215850, 86211, 90271, 213192, 287016, 287242, 79334, 630794,
    632848, 634907, 634923, 634932, 213138, 81799, 83987, 221462, 221721, 289160, 487478, 356463, 356577, 356605, 291075, 356656,
    356691, 356769, 294999, 231518, 364579, 299135, 301075, 76234, 415834, 348357, 78831, 143425, 146347, 146580, 147079, 150044,
    152248, 153865, 154332, 154528, 154537, 156047, 158134, 421969, 174099, 148202, 418427, 80322, 358501, 168044, 144534, 79111,
    146667, 215934, 151793, 219884, 352953, 150391, 84874, 79820, 86952, 150318, 144580, 211424, 284976, 144842, 347042, 353103,
    353140, 353159, 358628, 364752, 279248, 154122, 150158, 158071, 432227, 346392, 348294, 348522, 144082, 81911, 279354, 76583,
    79500, 145330, 147148, 221823, 287719, 422056, 167984, 82447, 213871, 217960, 484027, 491619, 212983, 82716, 215787, 219836,
    144473, 86745, 88509, 350867, 146131, 75987, 145842, 145950, 146299, 147952, 219482, 154194, 219798, 356450, 227434, 144747,
    487734, 489486, 438283, 420014, 422016, 79363, 81969, 152236, 163960, 414689, 78888, 217690, 217838, 283144, 76758, 286915,
    223269, 346682, 352327, 213832, 349014, 86924, 94674, 229481, 364712, 280638, 148403, 152536, 618733, 148189, 280869, 281319,
    416023, 417971, 418050, 420084, 616940, 422246, 231528, 364653, 76307, 350773, 84160, 416111, 288967, 279092, 217617, 297084,
    143437, 146709, 147281, 213137, 150132, 212581, 279038, 149990, 150219, 215801, 152010, 153973, 285060, 154011, 154268, 154395,
    351074, 154497, 352359, 155920, 156072, 156187, 156206, 287300, 354799, 159989, 160007, 160008, 160137, 160147, 212103, 149531,
    215688, 416371, 220093, 223561, 425987, 79230, 212595, 634917, 77940, 78260, 84122, 88545, 213695, 150422, 287695, 283276,
    289037, 485881, 292926, 630806, 505872, 76301, 144774, 288859, 225461, 78032, 237657, 285685, 78679, 284816, 75911, 88325,
    100406, 143789, 146345, 149555, 152331, 154291, 497779, 76221, 303230, 148387, 282986, 152522, 152532, 421967, 501787, 413728,
    423992, 79781, 223236, 491602, 77859, 80941, 278545, 82128, 84834, 86036, 283318, 285180, 285359, 89045, 287275, 354492,
    354739, 289262, 303162, 215752, 348511, 219156, 219501, 219841, 221827, 144052, 281131, 283647, 418748, 418783, 289141, 618826,
    630792, 78830, 84800, 79498, 81646, 80436, 83983, 86030, 92609, 94261, 94376, 96269, 98368, 144675, 147131, 350374,
    144597, 79446, 153634, 346502, 76553, 144559, 145415, 145863, 152143, 152318, 417844, 353232, 432228, 150093, 81913, 281125,
    420134, 218036, 86965, 87029, 218076, 218093, 146734, 145047, 225745, 148138, 148416, 220141, 279269, 76102, 149875, 150452,
    80858, 289226, 282850, 303208, 146512, 88733, 220004, 289033, 289207, 223710, 624656, 144929, 217573, 156636, 287731, 146907,
    147573, 149818, 84572, 413775, 151726, 151830, 292869, 303115, 147512, 148020, 346504, 151602, 348312, 154285, 157822, 157886,
    497701, 81358, 144843, 145925, 153838, 215961, 215966, 215980, 215983, 216013, 216037, 216062, 418607, 222012, 222022, 222073,
    222111, 227501, 227504, 227512, 424137, 227537, 227543, 227562, 227566, 227569, 626868, 430263, 233664, 430274, 233682, 430292,
    430312, 233710, 239669, 436278, 245782, 245787, 168157, 78335, 144178, 213797, 215395, 414221, 88130, 415826, 90244, 414442,
    422059, 291207, 628862, 212110, 217330, 217481, 217830, 285052, 285272, 219951, 288972, 76243, 279499, 86467, 78382, 149866,
    217673, 217903, 156056, 290881, 430212, 432253, 346509, 79733, 84932, 90934, 90966, 90977, 91022, 91050, 96425, 96464,
    96467, 102605, 102606, 102620, 102623, 501799, 114712, 213776, 219505, 90426, 413796, 420308, 421954, 426042, 491642, 495865,
    77868, 82288, 281195, 143782, 146952, 146080, 211862, 432243, 353253, 289275, 144174, 79302, 76408, 86542, 80646, 211782,
    156555, 225463, 168048, 485448, 419982, 487439, 86307, 155669, 178183, 79413, 419869, 76656, 616502, 86805, 348988, 77898,
    79916, 80279, 82471, 90129, 96261, 491539, 106507, 86588, 154615, 213629, 291217, 144552, 214001, 79143, 346952, 348963,
    280957, 80158, 82549, 215115, 217532, 86810, 414709, 212936, 213728, 86134, 159891, 144067, 145091, 145965, 81921, 84080,
    86342, 150104, 155685, 158143, 86993, 626941, 78373, 80168, 80468, 82480, 82506, 84580, 88368, 90158, 92249, 94320,
    291109, 100474, 79785, 150108, 143999, 150321, 413715, 217613, 219218, 416655, 80410, 150191, 152238, 219804, 220033, 168124,
    364778, 78727, 215300, 82745, 78886, 147420, 148191, 346218, 348493, 348937, 350930, 358401, 81719, 626849, 626850, 81695,
    281190, 88757, 150298, 144280, 144656, 79999, 287423, 160191, 430257, 176189, 211008, 80136, 213237, 213344, 215289, 155668,
    155986, 157896, 84409, 86322, 418789, 353062, 358635, 78632, 145114, 212893, 213666, 417901, 161984, 414311, 110641, 485756,
    286985, 279284, 212671, 92356, 92397, 75926, 148003, 78499, 79537, 80377, 80395, 76101, 81925, 82147, 84073, 86601,
    90726, 422252, 628819, 420026, 420204, 160125, 426016, 628738, 636950, 212064, 155859, 80268, 147073, 82548, 149617, 84487,
    157900, 622670, 280670, 283258, 354638, 354675, 356494, 430245, 76207, 145389, 79028, 281407, 76347, 146391, 81714, 148284,
    143649, 481649, 417839, 155705, 178180, 79131, 79660, 78514, 346781, 152409, 152433, 414635, 418512, 156388, 156391, 156424,
    426082, 430239, 430246, 434290, 172155, 77874, 84751, 147410, 281187, 221403, 78782, 86683, 352462, 352607, 90541, 90620,
    159867, 159936, 94434, 620550, 161819, 161867, 622599, 360482, 364547, 430099, 366599, 211085, 88646, 76722, 215679, 158151,
    235563, 499799, 212958, 146481, 147790, 284942, 286926, 287071, 420336, 294946, 299011, 364563, 301063, 79161, 86984, 414531,
    351231, 212013, 81372, 86250, 88629, 88902, 358547, 82585, 217432, 146522, 213593, 215762, 281340, 217602, 285541, 221247,
    483820, 223261, 354416, 225441, 227346, 489573, 491556, 362508, 362524, 430177, 299116, 301087, 630852, 503823, 145011, 279540,
    217714, 156323, 352727, 157867, 158061, 223690, 145237, 156443, 156446, 156459, 156468, 156469, 156585, 162032, 618965, 356626,
    362548, 213052, 215383, 154015, 219905, 221284, 286923, 80590, 88166, 76062, 152137, 90520, 354320, 364676, 366717, 78007,
    80552, 80885, 212096, 81423, 88425, 90342, 92332, 102429, 434273, 212114, 100368, 278656, 284851, 481471, 370706, 143881,
    145908, 350336, 92191, 356583, 147231, 154411, 353114, 353123, 353170, 495735, 364741, 364744, 79132, 76345, 417929, 354489,
    421897, 421952, 487767, 90118, 241669, 149825, 618690, 620601, 493653, 86870, 86886, 86904, 86913, 90834, 90875, 94673,
    102568, 88815, 89006, 90332, 90451, 289151, 94529, 100464, 100472, 104527, 497761, 76420, 81097, 81781, 156373, 168097,
    168098, 180243, 75908, 143991, 213502, 84454, 84683, 150265, 150342, 86474, 88437, 219524, 88732, 88859, 285538, 88961,
    286823, 90384, 90536, 90651, 90670, 221764, 289263, 94453, 94471, 94472, 94601, 94611, 145426, 147374, 148208, 149777,
    150207, 151623, 88126, 422235, 414363, 350662, 356515, 356722, 358495, 362534, 414192, 349122, 356368, 356851, 297008, 76015,
    213393, 215936, 482102, 279010, 354571, 354627, 495699, 279380, 346940, 283486, 422354, 430250, 144371, 149620, 217630, 154522,
    145112, 147396, 213861, 414446, 78639, 78696, 144820, 146117, 221249, 485441, 290837, 483551, 346493, 143372, 146627, 215276,
    217306, 78700, 147636, 154527, 287422, 291163, 78448, 76384, 80063, 81942, 82190, 215316, 86232, 86327, 219191, 88394,
    288841, 225318, 414581, 77957, 80137, 80990, 82197, 84321, 84590, 215768, 86692, 88788, 89057, 90509, 90574, 90740,
    102479, 283000, 221848, 289048, 290877, 297041, 497716, 499759, 160216, 176186, 211188, 346920, 287388, 82000, 348725, 98396,
    78151, 144812, 283147, 350847, 215858, 153690, 80621, 153841, 285695, 155788, 157726, 161942, 233509, 213093, 219177, 229430,
    233485, 98318, 79066, 279161, 413999, 221493, 94304, 423971, 96329, 622667, 82699, 144517, 152333, 154523, 155774, 156515,
    354815, 489628, 165989, 235531, 301125, 305177, 307242, 241718, 353081, 288799, 354626, 485845, 79858, 279385, 148365, 152488,
    352346, 483599, 493677, 168093, 170047, 278966, 286951, 420005, 303179, 144568, 144644, 80140, 213384, 82489, 219338, 237575,
    305169, 438328, 78104, 86518, 75848, 79918, 81885, 82135, 82433, 284891, 76708, 79325, 146597, 78072, 84514, 283005,
    86551, 88745, 92295, 495668, 146739, 350759, 215420, 86323, 161913, 303227, 350779, 178204, 278655, 148392, 148426, 152549,
    156606, 418776, 159870, 422367, 422389, 422395, 426097, 78855, 212534, 215712, 418412, 79412, 81324, 147412, 82554, 150381,
    90584, 487860, 161831, 307223, 280780, 94208, 632841, 143612, 144576, 348980, 351212, 346388, 217835, 350263, 356390, 80064,
    147479, 348660, 79762, 217454, 286744, 626932, 79497, 352683, 223463, 144392, 215045, 221400, 221478, 221482, 221584, 221702,
    287383, 223461, 227467, 346401, 88664, 88734, 92405, 157989, 233648, 372795, 372798, 76139, 78349, 147229, 213051, 215297,
    348729, 90396, 287319, 419992, 420261, 626751, 432238, 303188, 434282, 78916, 79241, 146761, 151983, 217572, 154143, 219890,
    418070, 221641, 418326, 418329, 485768, 159872, 225451, 487683, 225669, 491607, 165940, 428126, 495743, 497683, 213814, 150200,
    80522, 78176, 80273, 81282, 81546, 82580, 84042, 215685, 420013, 81453, 213276, 86085, 89087, 283382, 79536, 144119,
    283131, 416407, 155965, 352818, 354445, 223388, 616873, 354791, 227356, 231484, 233608, 237646, 88228, 90188, 145611, 76564,
    288979, 235588, 301127, 155875, 147900, 147946, 413725, 219561, 285463, 78485, 144826, 356448, 81688, 82606, 284769, 153937,
    161849, 219257, 108577, 110632, 81537, 284725, 303110, 434214, 309250, 81418, 145065, 146579, 81821, 82594, 348739, 286843,
    418023, 144621, 213920, 84441, 150392, 626842, 285219, 287134, 145002, 79566, 145463, 147687, 86131, 151976, 153659, 88484,
    154190, 90266, 157821, 92564, 159941, 94554, 161845, 96310, 96368, 165951, 92434, 80945, 82161, 82254, 144623, 79261,
    88838, 79144, 82767, 86877, 217965, 86905, 218025, 221895, 353002, 90896, 364720, 349026, 372799, 143429, 145525, 80701,
    81152, 146750, 280849, 281290, 287261, 283458, 290821, 432140, 149836, 90135, 104454, 145031, 213084, 213152, 215261, 215358,
    76240, 80568, 354745, 211351, 213543, 81250, 147399, 154600, 211890, 624650, 82639, 211596, 221275, 145505, 147907, 147927,
    148078, 148135, 150374, 150416, 155762, 152235, 90417, 287133, 146395, 82311, 150192, 152074, 154388, 156214, 157980, 424024,
    493660, 356555, 144372, 79303, 354610, 143737, 146433, 215783, 348513, 152258, 350481, 352317, 620637, 165992, 84056, 88334,
    81470, 147380, 346380, 356398, 358433, 426019, 362536, 364580, 366612, 434267, 372737, 505877, 505879, 376834, 442371, 481849,
    483467, 146643, 215596, 145647, 145778, 212097, 213662, 215764, 217672, 88124, 153817, 219828, 219945, 89066, 221411, 352661,
    223345, 159909, 231496, 75905, 77885, 211046, 80819, 86196, 215071, 217332, 217787, 283463, 221333, 223729, 491585, 75924,
    144736, 215521, 86384, 145693, 82611, 82651, 218035, 218062, 225756, 356838, 491643, 143373, 147227, 148435, 152563, 422374,
    80987, 76711, 149908, 348803, 484023, 291118, 495840, 499825, 79258, 416311, 221876, 419874, 223606, 421989, 424096, 430209,
    237623, 147506, 150128, 353246, 76187, 77977, 80525, 81795, 143460, 145652, 146040, 147017, 147634, 148094, 281039, 283219,
    157754, 78582, 278784, 79898, 80254, 80959, 83998, 86071, 86400, 86668, 88362, 88684, 219940, 221700, 92429, 92479,
    92646, 225357, 96318, 96341, 434198, 309264, 356563, 212113, 215434, 215502, 219937, 237651, 212801, 215687, 94441, 215694,
    223607, 279488, 420351, 145794, 152265, 616887, 145086, 215198, 217499, 358400, 145573, 147973, 152136, 284686, 154292, 354541,
    293017, 157704, 430187, 81354, 86158, 292920, 144325, 144908, 212948, 82597, 217913, 219157, 221851, 221853, 223318, 424103,
    211831, 215608, 86249, 88140, 221796, 86923, 84773, 79432, 80389, 217326, 88302, 223265, 81678, 77907, 79320, 79557,
    81261, 152363, 213870, 152222, 155699, 225535, 352502, 442372, 376849, 82017, 415844, 79365, 86065, 88898, 352844, 76628,
    90776, 80993, 150170, 358463, 75948, 147311, 148163, 284968, 219949, 287166, 419969, 356508, 291088, 487700, 428157, 301143,
    148464, 86245, 78768, 150409, 622613, 481698, 485391, 287561, 287613, 287628, 287661, 495801, 495804, 495827, 108583, 501811,
    80054, 279228, 82661, 281434, 144750, 278930, 414319, 154163, 75801, 78281, 79244, 80725, 213083, 417974, 147183, 150280,
    152260, 79327, 213236, 149801, 501776, 78460, 145963, 81794, 215111, 84805, 283163, 156127, 354625, 291129, 291157, 293021,
    360515, 163916, 297053, 167966, 301153, 301169, 79764, 146090, 150124, 217973, 217979, 217988, 221937, 289214, 436263, 349143,
    145765, 146408, 82923, 213995, 149760, 217694, 152241, 283359, 350904, 154325, 155900, 287349, 157888, 158048, 289163, 291086,
    358479, 424030, 366700, 77930, 78261, 81658, 81693, 81890, 84386, 84731, 88245, 350595, 92343, 354499, 79493, 213369,
    213933, 213966, 346591, 356859, 78277, 354512, 289225, 289244, 159926, 98340, 432233, 172102, 79793, 213972, 84060, 481536,
    484336, 104539, 481835, 430132, 143986, 152376, 146374, 483453, 351158, 358598, 364758, 376858, 211722, 147504, 153757, 346527,
    356582, 364612, 78739, 215985, 215994, 418726, 418734, 290942, 424132, 227560, 626891, 626914, 287264, 296970, 299028, 213866,
    213874, 154146, 155907, 155943, 157742, 157850, 79105, 280668, 301147, 77919, 148057, 221498, 424037, 296972, 233580, 80515,
    82156, 280621, 416049, 286792, 82404, 350800, 100449, 77999, 84257, 76702, 158081, 94357, 94607, 96313, 305181, 151766,
    145187, 211534, 78821, 215482, 78278, 225334, 172052, 279286, 212341, 213962, 150497, 156467, 156557, 156579, 156598, 487903,
    162028, 87010, 618962, 147734, 279106, 289047, 76324, 145721, 81706, 81773, 153668, 154413, 90156, 354433, 159868, 422164,
    362621, 235603, 90167, 368737, 143413, 146816, 78603, 77953, 215818, 219559, 418018, 146121, 148166, 350363, 156333, 289104,
    241727, 83977, 88467, 90202, 221455, 231533, 102483, 481347, 90348, 286754, 289002, 79975, 280929, 285665, 287348, 299087,
    147840, 278709, 144419, 145563, 78770, 215119, 213978, 79139, 145148, 145732, 485539, 485556, 358472, 360520, 499747, 86789,
    75795, 78357, 102526, 75958, 84709, 88596, 94548, 100460, 76475, 144253, 151969, 428056, 150454, 78672, 79560, 618976,
    491640, 145320, 78274, 82029, 211785, 217959, 349039, 86901, 86907, 86916, 86955, 284884, 285040, 353001, 90872, 221953,
    221969, 94657, 618968, 297019, 301084, 80638, 278812, 360494, 78009, 81094, 81771, 213562, 149570, 84720, 86402, 152196,
    285116, 88737, 90722, 288887, 223390, 92591, 98363, 102509, 368750, 372793, 616562, 618522, 618543, 626838, 144702, 279085,
    352772, 78903, 77830, 79953, 80488, 80643, 81603, 84373, 86429, 86437, 348685, 152551, 152555, 350290, 156619, 156639,
    156660, 160248, 163957, 163959, 144846, 146894, 346950, 217634, 96263, 348641, 217791, 94633, 102435, 144382, 213681, 215903,
    415812, 414712, 152250, 75921, 213198, 76137, 84586, 86498, 287314, 217316, 78536, 346756, 218079, 80965, 147119, 76507,
    80627, 86796, 630864, 81204, 81270, 212616, 81701, 223588, 78702, 79804, 158154, 146633, 145507, 147971, 149527, 156268,
    78343, 81594, 86287, 75923, 348835, 147980, 153837, 154222, 154336, 154366, 89079, 155849, 419872, 436260, 143572, 145436,
    149706, 152026, 348858, 156081, 79080, 347129, 143652, 76791, 145979, 146475, 147660, 215635, 151787, 151918, 414387, 86855,
    153711, 153840, 155714, 155917, 156337, 354723, 223743, 160009, 167997, 143432, 211970, 220153, 76651, 78486, 81184, 84835,
    86853, 78124, 81387, 220127, 145779, 80920, 413947, 151908, 279241, 211853, 153892, 481503, 416142, 416378, 417871, 422125,
    282632, 285233, 220119, 286787, 288949, 147548, 147755, 149725, 154490, 90142, 94300, 303200, 346921, 144022, 221854, 148015,
    215192, 90195, 167959, 78756, 620710, 78184, 81536, 82398, 215247, 215430, 154502, 352825, 158099, 223698, 161941, 362572,
    168047, 144333, 213849, 213945, 491634, 622712, 78823, 212664, 148204, 78931, 481889, 78520, 221347, 146890, 282859, 284783,
    286989, 287409, 358456, 146161, 280968, 282776, 352338, 287146, 289228, 618830, 358436, 143994, 219537, 219824, 491566, 213018,
    215398, 78885, 79632, 81956, 144310, 283045, 284707, 288839, 288945, 291238, 483345, 279260, 281487, 76654, 144195, 218072,
    285083, 211117, 213757, 217818, 221855, 364772, 79929, 82087, 213181, 82295, 84050, 84529, 217541, 86636, 86786, 219552,
    353147, 212129, 76797, 348569, 350252, 487429, 92658, 170074, 172122, 143536, 145803, 80605, 84551, 147015, 279383, 156383,
    287494, 291260, 422341, 160204, 422362, 618520, 354603, 356427, 628831, 348552, 158011, 303205, 376841, 442378, 90298, 484070,
    81450, 212537, 279000, 213654, 76185, 80780, 86495, 146695, 82328, 152104, 154439, 155923, 223240, 157787, 159869, 495723,
    281445, 86083, 88181, 350589, 219666, 88832, 88999, 154539, 299123, 146176, 146578, 149942, 215551, 282714, 86167, 352935,
    281034, 78119, 292872, 299047, 303158, 305183, 307203, 309248, 78735, 220001, 356629, 108570, 76253, 81093, 88971, 157702,
    353066, 353167, 88210, 487592, 360500, 213104, 84288, 86522, 219973, 148474, 145917, 146325, 81389, 147828, 279190, 82781,
    149626, 150134, 151727, 86921, 88661, 154619, 155824, 221464, 90842, 90901, 157921, 160036, 94543, 423964, 426040, 165984,
    434254, 368755, 368758, 368762, 174104, 281400, 285340, 79391, 76538, 150193, 94504, 76043, 76517, 144698, 86294, 348732,
    90325, 155894, 94244, 94599, 100463, 102406, 102446, 102551, 104484, 170048, 172040, 106528, 106552, 630845, 106565, 499813,
    106598, 436235, 110600, 110602, 110606, 176152, 634906, 110623, 374796, 505880, 440346, 505887, 638976, 638986, 180241, 213002,
    279539, 215769, 217697, 219825, 76055, 78059, 80340, 147712, 82366, 149835, 84605, 86733, 350535, 154140, 88802, 154346,
    90626, 158161, 159832, 421978, 159938, 424088, 172053, 172076, 143402, 146824, 146998, 150176, 483329, 144680, 145714, 88268,
    484284, 356375, 82203, 76013, 481585, 221387, 420256, 616897, 622656, 144213, 145335, 281337, 350596, 221742, 225543, 225673,
    351120, 616728, 489598, 351185, 76530, 76129, 79535, 81833, 81837, 81950, 213436, 82622, 82670, 83970, 281423, 283423,
    285097, 88613, 104570, 147400, 148192, 348955, 86962, 351218, 352692, 78348, 346212, 144738, 221468, 144127, 149934, 88310,
    88581, 78077, 366600, 78796, 213853, 217976, 484045, 221906, 352988, 484066, 225717, 225719, 225723, 225726, 225732, 225751,
    225753, 229478, 376851, 146062, 78540, 278665, 281214, 283350, 155777, 286862, 223531, 289081, 225355, 292977, 168023, 76127,
    279098, 434286, 438329, 211137, 146242, 213611, 150237, 76635, 485596, 422387, 281387, 144040, 80667, 146269, 356595, 622624,
    76098, 81348, 82106, 215091, 84230, 86203, 417926, 221504, 485865, 225425, 94416, 225667, 102530, 82901, 281234, 283218,
    145942, 79978, 80634, 82523, 84561, 84602, 86060, 86189, 88526, 88579, 92376, 92448, 94608, 299053, 349006, 352955,
    78876, 147260, 287192, 287740, 616532, 491536, 366623, 503831, 114701, 79103, 144077, 80653, 86202, 153648, 153994, 94378,
    159978, 153701, 155841, 158126, 76752, 145438, 147542, 148156, 150362, 154488, 154504, 157944, 634918, 147161, 414454, 153939,
    233629, 104500, 235583, 106543, 145264, 150495, 156494, 358588, 165920, 145283, 75825, 78024, 78710, 79520, 279247, 219786,
    146098, 81757, 149721, 150091, 153700, 156169, 144581, 285269, 223342, 235525, 79399, 278705, 149694, 151930, 493656, 76790,
    80888, 152173, 153792, 211342, 146416, 146968, 215833, 217370, 352781, 231479, 144361, 144748, 80673, 82082, 215855, 88719,
    281176, 286757, 289215, 144331, 219677, 143876, 143951, 81808, 82483, 215809, 151742, 282922, 88795, 90180, 156115, 291044,
    432254, 370727, 144802, 347080, 80192, 81971, 84836, 90593, 90682, 106602, 146418, 82191, 84737, 90297, 159845, 356521,
    161838, 620614, 161952, 624675, 168065, 147912, 149913, 146813, 160194, 172143, 88758, 143749, 79167, 80287, 86101, 88628,
    618515, 618524, 622621, 81144, 146954, 285648, 156012, 78843, 211763, 215066, 215432, 217240, 217314, 219155, 221460, 221610,
    221773, 223692, 227425, 235568, 237672, 144794, 90339, 165949, 212036, 145770, 350380, 285232, 215659, 84855, 148109, 282972,
    483505, 157789, 169998, 110637, 79356, 146538, 82858, 279429, 223383, 148071, 86784, 219751, 88851, 156150, 487562, 489512,
    622638, 296969, 493621, 366603, 499778, 305161, 501783, 436249, 438314, 79904, 82629, 90443, 92348, 92595, 424094, 78006,
    144983, 147052, 149633, 419847, 618530, 620692, 81887, 147914, 151653, 151870, 90777, 215233, 153684, 144352, 78078, 350957,
    144510, 280972, 143806, 147225, 346845, 419991, 368640, 78112, 80400, 79831, 211981, 217692, 218051, 217317, 430167, 79426,
    79666, 78778, 144326, 80212, 80739, 84762, 86564, 86597, 86707, 88070, 88672, 88917, 90115, 90131, 287031, 90586,
    288946, 485849, 94318, 290991, 94497, 96317, 98377, 497785, 106574, 438304, 78733, 148327, 279441, 217173, 219700, 616628,
    489544, 491592, 630819, 176157, 376845, 143818, 146172, 221699, 82535, 233488, 281481, 495697, 76085, 79225, 215909, 90323,
    79188, 213081, 219764, 227452, 78889, 145235, 147069, 366647, 281384, 347041, 353075, 353175, 288955, 358623, 364728, 144570,
    150427, 150434, 150445, 350977, 155696, 156450, 156533, 161989, 168169, 213927, 82866, 349132, 144955, 292912, 299121, 79718,
    79732, 82776, 82778, 213859, 213899, 281498, 84948, 84973, 84978, 281588, 217943, 418099, 90814, 221887, 287525, 287562,
    287583, 90976, 484215, 287610, 287612, 91017, 287637, 287642, 91036, 94655, 96427, 489650, 96437, 293051, 489672, 96480,
    96491, 96507, 364719, 102584, 299233, 630849, 79255, 76631, 212886, 78336, 80669, 86017, 86145, 88182, 88290, 88315,
    88765, 89038, 89046, 90176, 90802, 79106, 354583, 80236, 80318, 213497, 82532, 84404, 88690, 90642, 102514, 152473,
    156415, 146600, 616797, 420052, 299060, 82129, 151947, 211089, 147083, 81586, 213624, 213408, 227348, 227353, 630886, 211505,
    213568, 215307, 616714, 81244, 152449, 156411, 145936, 83988, 280787, 86340, 86823, 88552, 223595, 76092, 77936, 81219,
    149981, 348721, 287160, 100476, 144035, 219873, 156242, 159892, 423959, 622636, 165980, 146258, 213519, 424100, 165963, 168073,
    144956, 147631, 151720, 152125, 154608, 233576, 144941, 211698, 281076, 219649, 219746, 283467, 79980, 82139, 82308, 84743,
    153692, 88339, 90576, 94241, 144612, 281385, 348664, 215044, 150355, 217711, 219208, 219591, 221688, 223249, 158206, 225328,
    227398, 283143, 143502, 147794, 215658, 150181, 217570, 152360, 350296, 154108, 290819, 292905, 358470, 293030, 301122, 80382,
    144483, 143488, 145028, 211871, 219187, 143513, 147331, 80326, 80654, 82688, 90798, 78987, 80103, 86741, 243719, 81928,
    213226, 215520, 86016, 217734, 86834, 89052, 287413, 413765, 146162, 215540, 347081, 154113, 154210, 353186, 227472, 358577,
    358642, 235569, 144926, 287429, 418503, 287488, 287506, 145508, 80202, 146397, 213297, 279132, 82886, 354336, 79253, 75822,
    81352, 364548, 146523, 215488, 86272, 88767, 144114, 143394, 146048, 146497, 148177, 281029, 150147, 282670, 219291, 289076,
    291167, 291179, 297037, 299075, 301155, 172130, 79405, 144029, 79203, 285243, 75920, 81645, 81842, 213417, 346481, 150076,
    86466, 219443, 219609, 88653, 221757, 92492, 92653, 487479, 94391, 94542, 94570, 94576, 94584, 94587, 227371, 489539,
    104538, 104551, 104575, 106586, 145741, 217721, 350548, 157869, 79390, 212827, 307206, 241705, 217877, 148091, 88605, 428038,
    440324, 81472, 80317, 82227, 213400, 151948, 88301, 88686, 90313, 221838, 353153, 353209, 222199, 222203, 358592, 364725,
    364737, 364745, 364757, 364762, 364779, 364780, 370743, 370745, 370746, 370747, 146602, 213366, 156556, 217985, 145215, 148357,
    145016, 347011, 82072, 84021, 84236, 84869, 151634, 151820, 86534, 86845, 88252, 92234, 94254, 96273, 96289, 100392,
    102436, 104468, 172123, 110593, 176156, 178189, 243733, 114690, 180227, 79790, 278889, 147360, 79036, 144810, 79409, 81466,
    215532, 223694, 225495, 144637, 79313, 78607, 82555, 352578, 288788, 356380, 360477, 346243, 350746, 227549, 626905, 145639,
    211699, 146880, 146962, 152277, 217868, 289015, 428051, 225670, 426081, 353055, 364750, 376856, 143770, 155955, 78071, 84523,
    157802, 616585, 223734, 225352, 290949, 291020, 225554, 292928, 292933, 292978, 227487, 294922, 360479, 360487, 229438, 229445,
    360523, 231454, 626721, 233514, 301079, 301091, 301095, 301107, 497721, 301131, 237573, 376833, 311311, 76428, 145609, 147974,
    160022, 147039, 82403, 78845, 143660, 81199, 84621, 212776, 168112, 307259, 307262, 347078, 219584, 82891, 86981, 416034,
    416409, 145087, 154496, 350962, 356523, 217879, 485456, 485743, 421957, 245767, 81528, 86354, 79788, 79814, 223340, 485883,
    225369, 144946, 79891, 80500, 80682, 82058, 82269, 84052, 86209, 350287, 88421, 88947, 90471, 301175, 219357, 219664,
    221864, 163926, 144982, 151791, 354308, 79880, 82126, 215188, 219445, 223499, 223555, 146392, 147541, 150083, 150360, 152372,
    153858, 154604, 154607, 156263, 354408, 159752, 356457, 356536, 161957, 358568, 163867, 364647, 172072, 368708, 288879, 80474,
    82099, 88517, 145579, 149903, 158046, 159894, 430170, 76317, 78771, 78481, 82683, 301116, 146217, 149519, 151566, 157784,
    158145, 159797, 161805, 163904, 212608, 213470, 144228, 213963, 86849, 213984, 81277, 215220, 215653, 217451, 483517, 616866,
    148089, 213028, 176172, 634935, 84074, 84264, 150023, 86183, 86793, 143545, 146507, 154273, 223419, 297007, 499739, 303160,
    372760, 636954, 79870, 148035, 81769, 92240, 100426, 221958, 350657, 352745, 616555, 354457, 289026, 354600, 356490, 225637,
    358450, 360454, 143538, 144662, 147707, 150194, 151707, 151748, 151925, 154298, 89005, 418263, 157851, 172125, 80239, 147313,
    150019, 78506, 213949, 82909, 349129, 618991, 152117, 88168, 88618, 96316, 96381, 76048, 145806, 145856, 80880, 81432,
    147037, 147221, 213324, 147843, 147898, 147948, 82761, 213907, 150214, 151834, 152274, 217964, 218010, 218017, 218021, 284838,
    287245, 221975, 92195, 491575, 163896, 491624, 287433, 160182, 160188, 426090, 482231, 212074, 291079, 291209, 147942, 170015,
    212873, 146673, 150084, 278836, 145850, 620580, 152101, 145325, 152538, 291321, 299254, 299255, 79215, 80300, 86685, 88904,
    90412, 94458, 84917, 217667, 287564, 90968, 91032, 96441, 102618, 299234, 84245, 497763, 79881, 81380, 84285, 86226,
    86308, 86549, 88619, 90638, 90719, 92423, 354757, 92619, 94479, 79212, 346855, 96355, 151978, 79085, 212323, 284895,
    219534, 221263, 354391, 225517, 144260, 82632, 353258, 79945, 76616, 79753, 145376, 219702, 286870, 176133, 143444, 79633,
    212615, 82766, 213841, 82786, 82819, 86454, 86800, 217937, 86912, 217989, 90806, 484090, 92351, 94676, 76485, 215394,
    217918, 231504, 81003, 354783, 86761, 155729, 167951, 167992, 104505, 279025, 152515, 160118, 628820, 301163, 78114, 80166,
    80405, 80521, 81108, 81455, 147356, 86726, 88150, 351215, 358565, 79954, 80227, 80345, 215632, 86678, 219816, 219978,
    220060, 90290, 221734, 92290, 92372, 92464, 223549, 92493, 223602, 223701, 223703, 356499, 356525, 225677, 100436, 100450,
    233541, 233617, 628843, 303213, 217395, 233628, 82181, 279292, 84112, 84434, 86386, 90582, 92304, 290998, 94556, 303174,
    76691, 79607, 145282, 149604, 212994, 79060, 211630, 152001, 223485, 356559, 630816, 78569, 79379, 75981, 78262, 84679,
    86114, 88358, 90549, 92224, 92540, 94638, 346731, 279051, 213644, 350399, 350566, 350668, 219879, 422149, 239634, 82930,
    482084, 487501, 176150, 178181, 178205, 144737, 78822, 81517, 278767, 82272, 84217, 211713, 281388, 84897, 281521, 281530,
    84954, 90916, 90979, 484262, 484270, 489668, 96479, 293096, 102586, 145460, 146210, 212016, 281108, 282741, 150229, 152267,
    350998, 618983, 620570, 161880, 168018, 168092, 278650, 217712, 225559, 487825, 82773, 82801, 82831, 82832, 349025, 349029,
    90853, 90854, 353028, 353038, 292927, 143564, 152135, 290845, 493607, 102449, 104489, 630802, 352342, 356418, 618910, 360506,
    624706, 278682, 79052, 79431, 84791, 493616, 78693, 152110, 624697, 219784, 154375, 616756, 626755, 416488, 348579, 145476,
    147544, 148038, 148137, 149549, 149877, 150368, 151683, 152302, 284735, 284977, 153936, 154443, 220149, 155720, 221271, 172059,
    80913, 153721, 634882, 112670, 170067, 76399, 81240, 217492, 86758, 350803, 616822, 362620, 630839, 353032, 219748, 145888,
    80448, 86788, 149686, 218061, 287155, 364556, 168026, 348734, 414648, 291318, 144377, 213960, 353003, 221940, 484304, 92500,
    98353, 76460, 279094, 147561, 354349, 144170, 289122, 346943, 78212, 212964, 84044, 346224, 215475, 151635, 151664, 217725,
    86856, 88165, 153989, 90305, 485585, 223451, 157982, 92590, 493638, 280875, 484075, 484120, 180242, 212301, 223513, 346379,
    150133, 346892, 221201, 227333, 237579, 278902, 286778, 82059, 350737, 89056, 358590, 358597, 358607, 364777, 221546, 223311,
    484025, 146778, 354608, 356631, 81439, 346369, 215622, 215634, 285379, 154418, 156183, 485631, 354659, 485810, 422073, 291104,
    487770, 620667, 424070, 493619, 428151, 79642, 153947, 154390, 418118, 158091, 160142, 161855, 163928, 170089, 153771, 485424,
    622610, 628816, 630810, 76272, 80075, 80139, 223626, 235612, 79180, 414678, 151618, 151920, 144754, 144072, 299173, 303224,
    76614, 79473, 148258, 148294, 280653, 284904, 80184, 86235, 155842, 354648, 428048, 165995, 75955, 145837, 80967, 146527,
    81085, 213102, 213405,
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
