/**
Brotli encoder (RFC 7932).

One-shot compression from memory to memory, levels 0 to 11.
Levels 0-9 use greedy and lazy matching, levels 10-11 optimal parsing.
Uses the decoder's tables and static dictionary, numem for allocations,
and godot-math for gm_log2.
Everything is `@safe pure nothrow @nogc`.

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
import godotmath : gm_log2;
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
// words[bucketStart[h] .. bucketStart[h + 1]], as (length << 10) | index,
// longest first.
struct DictionaryHashTable
{
    ushort[(1 << DICTIONARY_HASH_BITS) + 1] bucketStart;
    ushort[NUM_DICTIONARY_WORDS] words;
}

static immutable DictionaryHashTable kDictionaryHashTable = buildDictionaryHashTable();

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
            size_t len = entry >> 10;
            size_t idx = entry & 1023;
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
            t.words[fill[h]++] = cast(ushort)((len << 10) | idx);
        }
    }
    return t;
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
        size_t len = entry >> 10;
        size_t idx = entry & 1023;
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
enum LITERAL_BYTE_SCORE = 135;
enum DISTANCE_BIT_PENALTY = 30;
enum SCORE_BASE = DISTANCE_BIT_PENALTY * 8 * 8;
enum MIN_SCORE = SCORE_BASE + 100;

pragma(inline, true)
size_t backwardReferenceScore(size_t len, size_t distance)
{
    return SCORE_BASE + LITERAL_BYTE_SCORE * len - DISTANCE_BIT_PENALTY * bsr(distance);
}

pragma(inline, true)
size_t lastDistanceScore(size_t len, uint shortCode)
{
    size_t score = LITERAL_BYTE_SCORE * len + SCORE_BASE + 15;
    if (shortCode != 0)
        score -= 39 + ((0x1CA10 >> (shortCode & 0xE)) & 0xE);
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
}

static immutable LevelParams[BROTLI_MAX_LEVEL + 1] kLevelParams =
[
    LevelParams(14, 0,  4, false,  64, false, false, false, 0, 0, false, 0, 0, 0, false), // 0
    LevelParams(15, 0,  4, false,  64, false, false, false, 0, 0, false, 0, 0, 0, false), // 1
    LevelParams(15, 1,  4, true,   64, false, false, true, 0, 0, false, 0, 0, 0, false), // 2
    LevelParams(15, 2,  4, true,   64, false, false, true, 0, 0, false, 0, 0, 0, false), // 3
    LevelParams(15, 3,  4, true,   64, true, false, true, 0, 0, true, 0, 0, 0, false), // 4
    LevelParams(15, 4,  4, true,   64, true, false, true, 1, 1, true, 0, 0, 0, false), // 5
    LevelParams(15, 5, 10, true,   64, true, false, true, 1, 1, true, 0, 0, 0, false), // 6
    LevelParams(16, 5, 10, true,   64, true, false, true, 1, 1, true, 0, 0, 0, false), // 7
    LevelParams(16, 6, 16, true,   64, true, false, true, 1, 1, true, 0, 0, 0, false), // 8
    LevelParams(16, 6, 16, true,  512, true, true, true, 1, 2, true, 0, 0, 0, false), // 9
    LevelParams(16, 6, 16, true,  512, true, true, true, 2, 2, true, 2, 150, 1, false), // 10
    LevelParams(16, 6, 16, true,  512, true, true, true, 2, 2, true, 4, 325, 5, true), // 11
];

// Hash table of recent positions, by hash of their next 4 bytes.
// Each bucket is a ring of the most recent positions.
struct Hasher
{
pure nothrow @nogc @safe:
    MallocArray!uint positions; // bucket * blockSize + i, relative to `base`
    MallocArray!uint counts;    // insertions in each bucket
    uint bucketBits;
    uint blockBits;
    uint blockMask;
    size_t base;
    size_t dictionaryLookups;
    size_t dictionaryMatches;

    bool initialize(uint bucketBits, uint blockBits)
    {
        this.bucketBits = bucketBits;
        this.blockBits = blockBits;
        blockMask = (1u << blockBits) - 1;
        if (!positions.resize(size_t(1) << (bucketBits + blockBits))
            || !counts.resize(size_t(1) << bucketBits))
            return false;
        counts[] = 0; // positions are only read once written
        base = 0;
        return true;
    }

    void release()
    {
        positions.release();
        counts.release();
    }

    pragma(inline, true)
    uint hashAt(scope const(ubyte)[] data, size_t pos) const
    {
        return (load32(data, pos) * 0x1E35A7BDu) >> (32 - bucketBits);
    }

    // Remember position `pos`, which needs 4 bytes of data.
    pragma(inline, true)
    void store(scope const(ubyte)[] data, size_t pos)
    {
        uint key = hashAt(data, pos);
        uint n = counts[key];
        positions[(key << blockBits) + (n & blockMask)] = cast(uint)(pos - base);
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

// Find the best match at data[pos .. pos + maxLength], at distances <= maxDistance.
// Improves on `m`. Also remembers `pos` in the hash table.
void findLongestMatch(ref Hasher h, scope const(ubyte)[] data, size_t pos, size_t maxLength,
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

    uint key = h.hashAt(data, pos);
    size_t bucket = size_t(key) << h.blockBits;
    uint n = h.counts[key];
    uint count = n < (1u << h.blockBits) ? n : (1u << h.blockBits);
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
    h.positions[bucket + (n & h.blockMask)] = cast(uint)(pos - h.base);
    h.counts[key] = n + 1;

    // Static dictionary, if nothing was found. Stop looking when it rarely
    // finds anything.
    if (params.useDictionary && m.score == MIN_SCORE
        && h.dictionaryMatches >= (h.dictionaryLookups >> 7))
    {
        h.dictionaryLookups++;
        if (findDictionaryMatch(data, pos, maxLength, maxDistance, m))
            h.dictionaryMatches++;
    }
}

// LZ77 parse of data[start .. end]: appends commands to `cmds`, the last one
// is insert-only if the block doesn't end with a copy. Updates `last`.
bool createCommands(ref Hasher h, scope const(ubyte)[] data, size_t start, size_t end,
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
        findLongestMatch(h, data, pos, maxLength, maxDistance, shortDistances, params, m);
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
                    findLongestMatch(h, data, pos + 1, maxLength - 1, maxDistance2, shortDistances,
                                     params, m2);
                    stored = pos + 2;
                    if (m2.score < m.score + 175)
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
                h.store(data, q);
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
                if (pos > applyRandomHeuristics + 4 * params.randomWindow)
                {
                    size_t jump = (pos + 16 < end - margin) ? pos + 16 : end - margin;
                    for (; pos < jump; pos += 4)
                    {
                        h.store(data, pos);
                        insert += 4;
                    }
                }
                else
                {
                    size_t jump = (pos + 8 < end - margin) ? pos + 8 : end - margin;
                    for (; pos < jump; pos += 2)
                    {
                        h.store(data, pos);
                        insert += 2;
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

// log2 of integers, for cost estimates: exact below 512, interpolated above
// (error < 3e-6).
struct Log2Table
{
pure nothrow @nogc @safe:
    double[513] small;

    void initialize()
    {
        small[0] = 0;
        foreach (i; 1 .. small.length)
            small[i] = gm_log2(cast(double) i);
    }

    pragma(inline, true)
    double opCall(size_t v) const
    {
        if (v < 512)
            return small[v];
        uint e = bsr(v) - 8; // v >> e is in [256, 511]
        size_t m = v >> e;
        double frac = cast(double)(v & ((size_t(1) << e) - 1)) / (size_t(1) << e);
        return e + small[m] + frac * (small[m + 1] - small[m]);
    }
}

bool isEmpty(scope const(uint)[] h)
{
    foreach (c; h)
        if (c != 0)
            return false;
    return true;
}

// Shannon entropy of the symbols of `h`, in bits, at least 1 bit per symbol.
double bitsEntropy(scope const(uint)[] h, ref const(Log2Table) lg)
{
    ulong total = 0;
    double bits = 0;
    foreach (c; h)
    {
        total += c;
        bits -= c * lg(c);
    }
    if (total != 0)
        bits += total * lg(cast(size_t) total);
    return bits < total ? total : bits;
}

// Estimated size in bits of symbols coded with a prefix code, including the
// prefix code itself (same model as the reference encoder). The non-zero
// counts are given in increasing symbol order, with `add`.
struct PopulationCost
{
pure nothrow @nogc @safe:
    double log2Total;
    double bits = 0;
    uint maxDepth = 1;
    uint[18] depthHistogram;
    uint[4] firstCounts;
    size_t count;
    size_t next;    // symbol after the last one added

    this(ref const(Log2Table) lg, ulong total)
    {
        log2Total = lg(cast(size_t) total);
    }

    pragma(inline, true)
    void add(size_t symbol, uint c, ref const(Log2Table) lg)
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
                    bits += 3;
                    reps >>= 3;
                }
            }
        }

        // Entropy of the symbol, and its approximate code length.
        double log2p = log2Total - lg(c);
        uint depth = cast(uint)(log2p + 0.5);
        bits += c * log2p;
        if (depth > 15)
            depth = 15;
        if (depth > maxDepth)
            maxDepth = depth;
        depthHistogram[depth]++;
    }

    double result(ref const(Log2Table) lg)
    {
        enum double oneSymbolCost = 12;
        enum double twoSymbolsCost = 20;
        enum double threeSymbolsCost = 28;
        enum double fourSymbolsCost = 37;
        alias s = firstCounts;
        if (count <= 1)
            return oneSymbolCost;
        if (count == 2)
            return twoSymbolsCost + cast(double) s[0] + s[1];
        if (count == 3)
        {
            uint max = s[0] > s[1] ? s[0] : s[1];
            if (s[2] > max)
                max = s[2];
            return threeSymbolsCost + 2 * (cast(double) s[0] + s[1] + s[2]) - max;
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
            double h23 = cast(double) s[2] + s[3];
            double max = h23 > s[0] ? h23 : s[0];
            return fourSymbolsCost + 3 * h23 + 2 * (cast(double) s[0] + s[1]) - max;
        }
        return bits + 18 + 2 * maxDepth + bitsEntropy(depthHistogram[], lg);
    }
}

double populationCost(scope const(uint)[] h, ref const(Log2Table) lg)
{
    ulong total = 0;
    foreach (c; h)
        total += c;
    auto pc = PopulationCost(lg, total);
    foreach (i, c; h)
        if (c != 0)
            pc.add(i, c, lg);
    return pc.result(lg);
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
                       scope ushort[] scratch, scope double[] deltaScratch,
                       ref const(Log2Table) lg, out double totalCost)
{
    assert(n >= 1 && n <= MAX_CLUSTER_INPUTS && maxClusters >= 1);
    assert(alphabetSize <= MAX_ALPHABET_SIZE);
    assert(histograms.length >= n * alphabetSize && clusters.length >= n * alphabetSize);
    assert(scratch.length >= n * alphabetSize && deltaScratch.length >= n * n);

    double[MAX_CLUSTER_INPUTS] cost;
    ulong[MAX_CLUSTER_INPUTS] total;
    bool[MAX_CLUSTER_INPUTS] active;
    ubyte[MAX_CLUSTER_INPUTS] owner;
    size_t[MAX_CLUSTER_INPUTS] numSymbols;
    ushort[MAX_ALPHABET_SIZE] tmpSymbols;

    // Cost change if clusters i < j were merged.
    ref double delta(size_t i, size_t j) { return deltaScratch[i * n + j]; }

    const(uint)[] input(size_t i) { return histograms[i * alphabetSize .. (i + 1) * alphabetSize]; }
    uint[] cluster(size_t i) { return clusters[i * alphabetSize .. (i + 1) * alphabetSize]; }
    // Non-zero symbols of cluster i, in increasing order.
    ushort[] symbols(size_t i) { return scratch[i * alphabetSize .. i * alphabetSize + numSymbols[i]]; }

    // Cost of the sum of cluster `a` (or of `aCounts` with non-zero symbols
    // `aSymbols`) and cluster `b`.
    double mergedCost(scope const(uint)[] aCounts, scope const(ushort)[] aSymbols, ulong aTotal, size_t b)
    {
        const(uint)[] bCounts = cluster(b);
        const(ushort)[] bSymbols = symbols(b);
        auto pc = PopulationCost(lg, aTotal + total[b]);
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
            pc.add(s, aCounts[s] + bCounts[s], lg);
        }
        return pc.result(lg);
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
        cost[i] = active[i] ? populationCost(c, lg) : 0;
        owner[i] = cast(ubyte) i;
        if (active[i])
            numActive++;
    }

    if (numActive == 0)
    {
        map[0 .. n] = 0;
        cluster(0)[] = 0;
        totalCost = populationCost(cluster(0), lg);
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
        double best = double.max;
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
            double d = mergedCost(ci, symbols(bi), total[bi], j) - cost[bi] - cost[j];
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
            double bestCost = double.max;
            foreach (j; 0 .. n)
            {
                if (!active[j])
                    continue;
                double d = mergedCost(h, tmpSymbols[0 .. k], t, j) - cost[j];
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
        totalCost += populationCost(cluster(k), lg);
    return numClusters;
}

// ---------------------------------------------------------------------------
// Encoder: optimal parsing
// ---------------------------------------------------------------------------

// Binary tree match finder: a binary search tree of the window positions
// with the same 4-byte hash, sorted by the data that follows them. Finds
// every match length at a position, with the smallest distance found for it.
enum TREE_BUCKET_BITS = 17;
// The trees cover at most that much history: 8 bytes of memory per byte.
enum MAX_TREE_WINDOW = 1 << 22;
enum MAX_TREE_SEARCH_DEPTH = 64;
enum MAX_TREE_COMP_LENGTH = 128;
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
            size_t len = common + matchLength(data, prevPos + common, pos + common, limit - common);
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
    size_t stop = pos > 64 ? pos - 64 : 0;
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
        || t.dictionaryMatches < (t.dictionaryLookups >> 7))
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
    float[NUM_COMMAND_CODES] commandCost;
    float[16 + 48] distanceCost; // NPOSTFIX = 0, NDIRECT = 0
    float minCommandCost;
}

// Costs from a histogram: Shannon bits, at least 1, a bit more than the
// rarest for absent symbols.
void setCosts(scope const(uint)[] histogram, scope float[] cost, bool literals, ref const(Log2Table) lg)
{
    size_t sum = 0;
    size_t missing = 0;
    foreach (c; histogram)
    {
        sum += c;
        if (c == 0)
            missing++;
    }
    double log2Sum = lg(sum);
    float missingCost = cast(float)(lg(literals ? sum : sum + missing) + 2);
    foreach (i, c; histogram)
    {
        if (c == 0)
        {
            cost[i] = missingCost;
            continue;
        }
        float bits = cast(float)(log2Sum - lg(c));
        cost[i] = bits < 1 ? 1 : bits;
    }
}

struct ZopfliNode
{
    float cost = float.infinity;
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
    float costDiff;
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
                        scope const(double)[] literalCosts, ref const(LevelParams) params,
                        scope ZopfliNode[] nodes)
{
    enum LONG_COPY_QUICK_STEP = 16384;
    size_t n = end - start;
    size_t maxZopfliLen = params.optimalMaxLength;
    size_t maxStarts = params.optimalStarts;
    foreach (ref node; nodes[0 .. n + 1])
        node = ZopfliNode.init;
    nodes[0].cost = 0;
    foreach (k; 0 .. 4)
        nodes[0].cache[k] = cast(uint) last[k];

    StartPosQueue queue;
    size_t matchIndex = 0;
    size_t i = 0;
    while (i < n)
    {
        const(BackwardMatch)[] here = matches[matchIndex .. matchIndex + numMatches[i]];
        matchIndex += numMatches[i];

        // A command ends here: the next one may start its insert here.
        if (nodes[i].cost < float.infinity && (i == 0 || nodes[i].copyLength != 0))
        {
            if (i != 0)
                computeNodeCache(nodes, i, start, maxBackward);
            float costDiff = cast(float)(nodes[i].cost - literalCosts[i]);
            if (costDiff <= 0 || i == 0)
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
            float minCost = cast(float)(queue[0].costDiff + literalCosts[i]) + model.minCommandCost;
            size_t minLen = 2;
            {
                size_t bucket = 4, nextOffset = 10;
                while (i + minLen <= n && nodes[i + minLen].cost <= minCost)
                {
                    minLen++;
                    if (minLen == nextOffset)
                    {
                        minCost += 1;
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
                float baseCost = cast(float)(sp.costDiff + literalCosts[i]) + kInsertLengthBits[insertCode];
                const(uint)[] cache = nodes[sp.pos].cache[];

                void update(size_t len, size_t codedLength, size_t distance, float cost)
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
                    float distanceCost = baseCost + model.distanceCost[code];
                    size_t l = bestLen + 1;
                    for (; l <= len; ++l)
                    {
                        if (l > maxZopfliLen && l < len)
                            l = len; // long copies: only the full length
                        uint copyCode = copyLengthCode(l);
                        uint cmd = commandSymbol(insertCode, copyCode, code == 0);
                        float cost = (cmd < 128 ? baseCost : distanceCost) + kCopyLengthBits[copyCode]
                                   + model.commandCost[cmd];
                        update(l, l, cast(size_t) d, cost);
                    }
                    bestLen = len;
                }

                // Other matches, from the 2 best starts only.
                if (k >= 2)
                    continue;
                size_t len = minLen;
                foreach (ref const m; here)
                {
                    bool isDictionary = m.wordLength != 0;
                    uint dsym, dbits, dextra;
                    distanceSymbol(m.distance + 15, 0, 0, dsym, dbits, dextra);
                    float distanceCost = baseCost + dbits + model.distanceCost[dsym];
                    size_t maxMatchLen = m.length;
                    if (len < maxMatchLen && (isDictionary || maxMatchLen > maxZopfliLen))
                        len = maxMatchLen;
                    for (; len <= maxMatchLen; ++len)
                    {
                        size_t codedLength = isDictionary ? m.wordLength : len;
                        uint copyCode = copyLengthCode(codedLength);
                        uint cmd = commandSymbol(insertCode, copyCode, false);
                        float cost = distanceCost + kCopyLengthBits[copyCode] + model.commandCost[cmd];
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
                if (nodes[i].cost < float.infinity && nodes[i].copyLength != 0)
                {
                    computeNodeCache(nodes, i, start, maxBackward);
                    float costDiff = cast(float)(nodes[i].cost - literalCosts[i]);
                    if (costDiff <= 0)
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
                          scope const(double)[] literalCosts, ref const(ZopfliCostModel) model,
                          ref DistanceCache last, ref CommandBuffer cmds)
{
    size_t n = end - start;
    // End of the last command, the rest is inserted.
    size_t bestEnd = 0;
    double bestCost = double.infinity;
    foreach (e; 0 .. n + 1)
    {
        if (!(nodes[e].cost < float.infinity) || (e != 0 && nodes[e].copyLength == 0))
            continue;
        double cost = nodes[e].cost;
        if (e < n)
        {
            uint insertCode = insertLengthCode(n - e);
            cost += literalCosts[n] - literalCosts[e] + kInsertLengthBits[insertCode]
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

    void count(size_t from, size_t to)
    {
        ubyte p1 = from >= 1 ? data[from - 1] : 0;
        ubyte p2 = from >= 2 ? data[from - 2] : 0;
        foreach (b; data[from .. to])
        {
            foreach (mode; 0 .. 4)
            {
                uint context = kContextLookupTables[mode * 2][p1] | kContextLookupTables[mode * 2 + 1][p2];
                mh[(mode * 64 + context) * A + b]++;
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

    double best = double.max;
    foreach (mode; 0 .. 4)
    {
        double estimate = 0;
        foreach (context; 0 .. 64)
        {
            const(uint)[] h = mh[(mode * 64 + context) * A .. (mode * 64 + context + 1) * A];
            if (!isEmpty(h))
                estimate += populationCost(h, ws.log2);
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
                         scope const(Command)[] cmds, ref EncoderWorkspace ws, scope double[] costs)
{
    enum A = NUM_LITERAL_CODES;
    uint bestMode;
    if (!literalModeHistograms(data, start, n, cmds, ws, bestMode))
        return false;
    uint[] mh = ws.modeHistograms[0 .. 4 * 64 * A];

    // Costs of each context. Contexts without statistics use those of all contexts.
    if (!ws.literalContextCosts.ensure(64 * A))
        return false;
    float[] contextCosts = ws.literalContextCosts;
    uint[A] all;
    foreach (context; 0 .. 64)
        all[] += mh[(bestMode * 64 + context) * A .. (bestMode * 64 + context + 1) * A];
    float[A] allCosts;
    setCosts(all[], allCosts[], true, ws.log2);
    foreach (context; 0 .. 64)
    {
        const(uint)[] h = mh[(bestMode * 64 + context) * A .. (bestMode * 64 + context + 1) * A];
        float[] c = contextCosts[context * A .. (context + 1) * A];
        if (isEmpty(h) || cmds.length == 0)
            c[] = allCosts[];
        else
        {
            setCosts(h, c, true, ws.log2);
            foreach (k; 0 .. A)
                c[k] = 0.5f * (c[k] + allCosts[k]);
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
void computeCommandCosts(scope const(Command)[] cmds, ref const(Log2Table) lg, ref ZopfliCostModel model)
{
    if (cmds.length == 0)
    {
        foreach (i, ref c; model.commandCost)
            c = cast(float) lg(11 + i);
        foreach (i, ref c; model.distanceCost)
            c = cast(float) lg(20 + i);
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
        setCosts(commandHistogram[], model.commandCost[], false, lg);
        setCosts(distanceHistogram[], model.distanceCost[], false, lg);
    }
    model.minCommandCost = float.max;
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
    double[] literalCosts = ws.literalCosts[0 .. n + 1];
    ZopfliNode[] nodes = ws.nodes[0 .. n + 1];
    ZopfliCostModel model;
    computeCommandCosts(previous, ws.log2, model);
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
    float switchCost;           // bits
    uint iterations;
}

static immutable SplitParams[2][3] kSplitParams =
[
    // Literals, commands, distances: [fast, best]
    [SplitParams(16, 544, 70, 128, 28.1f, 2), SplitParams(100, 544, 70, 128, 28.1f, 10)],
    [SplitParams(16, 530, 40, 128, 13.5f, 2), SplitParams(50, 530, 40, 128, 13.5f, 10)],
    [SplitParams(16, 544, 40, 128, 14.6f, 2), SplitParams(50, 544, 40, 128, 14.6f, 10)],
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
                size_t numHistograms, float switchCost, scope ubyte[] blockIds,
                scope ubyte[] switchSignal, scope float[] insertCost, scope float[] cost,
                ref const(Log2Table) lg)
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
        double log2Total = lg(cast(size_t) total);
        foreach (s, c; h)
            insertCost[s * numHistograms + k] = cast(float)(log2Total - (c == 0 ? -2.0 : lg(c)));
    }

    // cost[k]: cost of the best path ending with histogram k, relative to
    // the best path. Capped at switchCost: then the path switches from the
    // best one, and switchSignal records it.
    size_t bitmapLength = (numHistograms + 7) >> 3;
    cost[0 .. numHistograms] = 0;
    switchSignal[0 .. n * bitmapLength] = 0;
    foreach (i, sym; symbols)
    {
        const(float)[] ic = insertCost[sym * numHistograms .. (sym + 1) * numHistograms];
        float minCost = float.max;
        uint best = 0;
        foreach (k; 0 .. numHistograms)
        {
            cost[k] += ic[k];
            if (cost[k] < minCost)
            {
                minCost = cost[k];
                best = cast(uint) k;
            }
        }
        blockIds[i] = cast(ubyte) best;
        // Switches are cheaper at the beginning, histograms are less reliable there.
        float sc = switchCost;
        if (i < 2000)
            sc *= 0.77f + 0.07f * i / 2000;
        ubyte[] signal = switchSignal[i * bitmapLength .. (i + 1) * bitmapLength];
        foreach (k; 0 .. numHistograms)
        {
            cost[k] -= minCost;
            if (cost[k] >= sc)
            {
                cost[k] = sc;
                signal[k >> 3] |= cast(ubyte)(1 << (k & 7));
            }
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
    float[256] cost;

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
    size_t refinements = 2 * n / p.stride + 100;
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
                   ws.splitSignal, ws.splitInsertCost, cost[], ws.log2);
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

    // Merge similar histograms into block types.
    ubyte[256] map;
    double clusterCost;
    clusterHistograms(histograms, numHistograms, alphabetSize, 256, map[], ws.clusters,
                      ws.clusterScratch, ws.clusterDelta, ws.log2, clusterCost);

    // Blocks, types numbered by first use.
    ushort[256] typeNumber = ushort.max;
    uint numTypes = 0;
    size_t numBlocks = 0;
    uint previous = uint.max;
    foreach (id; blockIds)
    {
        uint t = map[id];
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
    foreach (id; blockIds)
    {
        uint t = typeNumber[map[id]];
        if (t != previous)
        {
            split.types[b] = cast(ubyte) t;
            split.lengths[b] = 0;
            b++;
            previous = t;
        }
        split.lengths[b - 1]++;
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

// Like clusterHistograms, for any number of histograms (`map` has a uint per
// histogram): clusters batches of them first. The cluster histograms are
// written to ws.clusters.
BrotliStatus clusterMany(scope const(uint)[] histograms, size_t n, size_t alphabetSize,
                         uint maxClusters, scope uint[] map, ref EncoderWorkspace ws,
                         out uint numClusters, out double totalCost)
{
    assert(n >= 1 && maxClusters <= MAX_CLUSTER_INPUTS);
    enum batch = 64;
    if (!ws.reserveClustering(n < MAX_CLUSTER_INPUTS ? n : MAX_CLUSTER_INPUTS, alphabetSize))
        return BrotliStatus.outOfMemory;
    if (n <= MAX_CLUSTER_INPUTS)
    {
        ubyte[MAX_CLUSTER_INPUTS] m;
        numClusters = clusterHistograms(histograms, n, alphabetSize, maxClusters, m[], ws.clusters,
                                        ws.clusterScratch, ws.clusterDelta, ws.log2, totalCost);
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
            double cost;
            uint k = clusterHistograms(reduced[b * alphabetSize .. $], bn, alphabetSize,
                                       pass == 0 ? batch : batch / 2, m[], ws.clusters,
                                       ws.clusterScratch, ws.clusterDelta, ws.log2, cost);
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
                                    ws.clusterScratch, ws.clusterDelta, ws.log2, totalCost);
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
    enum size_t streakLimit = 1240;
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
double distanceCost(scope const(Command)[] cmds, uint npostfix, uint ndirect, ref const(Log2Table) lg)
{
    uint[NUM_DISTANCE_SYMBOLS_MAX] histogram;
    double extraBits = 0;
    foreach (ref const c; cmds)
    {
        if (!hasExplicitDistance(c))
            continue;
        uint dsym, dbits, dextra;
        distanceSymbol(c.distanceCode, npostfix, ndirect, dsym, dbits, dextra);
        histogram[dsym]++;
        extraBits += dbits;
    }
    return populationCost(histogram[0 .. 16 + ndirect + (48 << npostfix)], lg) + extraBits;
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
void chooseDistanceParams(scope const(Command)[] cmds, uint search, ref const(Log2Table) lg,
                          out uint npostfix, out uint ndirect)
{
    npostfix = 0;
    ndirect = 0;
    if (search == 0)
        return;
    double best = distanceCost(cmds, 0, 0, lg);
    foreach (p; 0 .. 4)
    {
        double previous = double.max;
        foreach (msb; 0 .. (search >= 2 ? 16 : 1))
        {
            if (p == 0 && msb == 0)
                continue;
            double cost = distanceCost(cmds, p, msb << p, lg);
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
    Log2Table log2;

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
    MallocArray!float splitInsertCost;

    // Histograms and prefix codes.
    MallocArray!uint modeHistograms;      // [mode][context][symbol], to choose the literal context mode
    MallocArray!uint histograms;
    MallocArray!uint treeHistograms;
    MallocArray!uint clusterMap;
    MallocArray!ubyte depths;
    MallocArray!ushort bits;

    // Clustering.
    MallocArray!uint clusters;
    MallocArray!ushort clusterScratch;
    MallocArray!double clusterDelta;
    size_t clusteringCapacity; // histograms of MAX_ALPHABET_SIZE

    MallocArray!uint scratch;

    // Optimal parsing.
    CommandBuffer previousCommands;
    CommandBuffer bestCommands;
    BitWriter trial;
    MatchBuffer matches;
    MallocArray!uint numMatches;
    MallocArray!double literalCosts;
    MallocArray!float literalContextCosts;
    MallocArray!ZopfliNode nodes;

    bool initialize()
    {
        log2.initialize();
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

// Write data[start .. start + length] as a compressed meta-block made of `cmds`.
BrotliStatus writeCompressedMetaBlock(ref BitWriter w, scope const(ubyte)[] data, size_t start,
                                      size_t length, scope const(Command)[] cmds, bool isLast,
                                      ref const(LevelParams) params, ref EncoderWorkspace ws)
{
    enum oom = BrotliStatus.outOfMemory;
    enum A = NUM_LITERAL_CODES;

    // Distance parameters.
    uint npostfix, ndirect;
    chooseDistanceParams(cmds, params.distanceParamSearch, ws.log2, npostfix, ndirect);
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
        || !ws.treeHistograms.ensure(literalSize + (distanceSize > commandSize ? distanceSize : commandSize))
        || !ws.clusterMap.ensure(numLiteralHistograms > numDistanceHistograms
                         ? numLiteralHistograms : numDistanceHistograms))
        return oom;

    ubyte[256 * 64] literalMap;
    uint numLiteralTrees;
    uint[] literalTrees = ws.treeHistograms;
    {
        uint bestTrees;
        double bestCost = double.max;
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
            double cost;
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
        double cost;
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
        if (params.optimalKeepBest)
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
                if (!copyCommands(ws.commands, ws.bestCommands))
                    return oom;
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
        if (!h.initialize(bucketBits, params.blockBits))
            return BrotliStatus.outOfMemory;
    }

    EncoderWorkspace ws;
    scope(exit) ws.release();
    if (!ws.initialize())
        return BrotliStatus.outOfMemory;

    DistanceCache last = kInitialDistanceCache;
    // Smaller meta-blocks adapt better, but cost more headers.
    size_t blockLength = optimal ? 1 << 19 : 1 << 20;
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
            if (!createCommands(h, input, pos, pos + length, maxBackward, params, newLast, ws.commands))
                return BrotliStatus.outOfMemory;
        }

        BitWriter.Mark mark = w.mark();
        BrotliStatus r = writeCompressedMetaBlock(w, input, pos, length, ws.commands[], isLast, params, ws);
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
    Log2Table lg;
    lg.initialize();
    enum A = 64;
    uint[MAX_CLUSTER_INPUTS * A] histograms;
    uint[MAX_CLUSTER_INPUTS * A] clusters;
    ushort[MAX_CLUSTER_INPUTS * A] scratch;
    MallocArray!double delta;
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
        double cost;
        uint k = clusterHistograms(histograms[], n, A, maxClusters, map[], clusters[], scratch[], delta, lg, cost);
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
        StartPosition p = StartPosition(i, cast(float) rng.below(20));
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
