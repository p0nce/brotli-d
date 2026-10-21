/**
Tiny, self-contained Brotli decoder (RFC 7932).

One-shot decompression from memory to memory. Only dependency is numem,
for its pure allocation functions `nu_malloc`, `nu_realloc` and `nu_free`
(which use libc by default).
Everything is `@safe pure nothrow @nogc`, except `brotliFree`.

This is a translation of the Google reference decoder (Java and C),
simplified to a non-streaming decoder.

Copyright:
    Copyright (c) 2009, 2010, 2013-2016 by the Brotli Authors.
    Copyright (c) 2026 Guillaume Piolat (D translation).

License:
    MIT, see bottom of file.

Example:
---
import brotli;

// 1. Decode into a buffer you own.
ubyte[1024] buf;
size_t len;
if (brotliDecompress(compressed, buf[], len) == BrotliStatus.success)
    use(buf[0..len]);

// 2. Decode into a malloc'ed buffer.
ubyte[] decoded;
if (brotliDecompressAlloc(compressed, decoded) == BrotliStatus.success)
{
    use(decoded);
    brotliFree(decoded); // @system
}
---
*/
module brotli;

import core.bitop : bsf, bsr;
import godotmath : gm_log2;
import numem.core.hooks : nu_free, nu_malloc, nu_realloc;

@safe pure nothrow @nogc:

/// Result of a decompression.
enum BrotliStatus
{
    success,        /// The whole stream was decoded.
    corrupted,      /// Invalid, truncated, or followed by trailing bytes.
    outputTooSmall, /// Output buffer too small, or `maxOutputSize` exceeded.
    outOfMemory,    /// An allocation failed.
}

/**
    Decompress a complete Brotli stream into a caller-provided buffer.

    Params:
        input       = The whole compressed stream. Trailing bytes are an error.
        output      = Where to decode. Its content is unspecified on error.
        decodedSize = On success, number of bytes written to `output`. 0 on error.

    Returns: `BrotliStatus.success`, or the error that occured.
*/
BrotliStatus brotliDecompress(const(ubyte)[] input, ubyte[] output, out size_t decodedSize)
{
    OutputBuffer o;
    o.buf = output;
    o.limit = output.length;
    o.growable = false;
    BrotliStatus r = decodeStream(input, o);
    if (r == BrotliStatus.success)
        decodedSize = o.len;
    return r;
}

/**
    Decompress a complete Brotli stream into a newly allocated buffer.

    Params:
        input         = The whole compressed stream. Trailing bytes are an error.
        output        = On success, the decoded data, to be released with `brotliFree`.
                        `null` on error, or if the decoded data is empty.
        maxOutputSize = Decoding fails with `BrotliStatus.outputTooSmall` if
                        the decoded data would be larger than this.

    Returns: `BrotliStatus.success`, or the error that occured.
*/
BrotliStatus brotliDecompressAlloc(const(ubyte)[] input, out ubyte[] output,
                                   size_t maxOutputSize = size_t.max)
{
    OutputBuffer o;
    o.limit = maxOutputSize;
    o.growable = true;
    BrotliStatus r = decodeStream(input, o);
    if (r != BrotliStatus.success || o.len == 0)
    {
        freeArray(o.buf);
        return r;
    }
    if (o.len < o.buf.length)
    {
        // Shrink to fit. If that fails, the larger buffer is still fine.
        ubyte[] shrunk = reallocArray(o.buf, o.len);
        if (shrunk !is null)
            o.buf = shrunk;
    }
    output = o.buf[0 .. o.len];
    return r;
}

/// Release a buffer returned by `brotliDecompressAlloc`. Accepts `null`.
void brotliFree(ubyte[] buffer) @system
{
    if (buffer.ptr !is null)
        nu_free(buffer.ptr);
}

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
    {
        freeArray(w.buf);
        return r;
    }
    assert(w.pos > 0);
    if (w.pos < w.buf.length)
    {
        // Shrink to fit. If that fails, the larger buffer is still fine.
        ubyte[] shrunk = reallocArray(w.buf, w.pos);
        if (shrunk !is null)
            w.buf = shrunk;
    }
    output = w.buf[0 .. w.pos];
    return r;
}

private:

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

// Resize a malloc'ed array (or allocate one if `arr` is null).
// On success, `arr` must not be used anymore. On failure, returns null and
// `arr` is untouched. `count` must be > 0.
T[] reallocArray(T)(T[] arr, size_t count) @trusted
{
    assert(count > 0);
    if (count > size_t.max / T.sizeof)
        return null;
    size_t bytes = count * T.sizeof;
    void* p = (arr.ptr is null) ? nu_malloc(bytes) : nu_realloc(arr.ptr, bytes);
    if (p is null)
        return null;
    return (cast(T*)p)[0 .. count];
}

void freeArray(T)(ref T[] arr) @trusted
{
    if (arr.ptr !is null)
        nu_free(arr.ptr);
    arr = null;
}

// Grow-only array, its content is not preserved.
bool ensureLength(T)(ref T[] arr, size_t count)
{
    if (arr.length >= count)
        return true;
    T[] bigger = reallocArray(arr, count);
    if (bigger is null)
        return false;
    arr = bigger;
    return true;
}

struct OutputBuffer
{
pure nothrow @nogc @safe:
    ubyte[] buf;    // Caller's buffer, or malloc'ed if `growable`.
    size_t len;     // Bytes decoded so far, always <= buf.length and <= limit.
    size_t limit;   // Maximum decoded size.
    bool growable;
    BrotliStatus error;

    // Make room for `n` more bytes.
    pragma(inline, true)
    bool reserve(size_t n)
    {
        if (n <= buf.length - len)
            return true;
        return grow(n);
    }

    bool grow(size_t n)
    {
        if (!growable || n > limit - len)
        {
            error = BrotliStatus.outputTooSmall;
            return false;
        }
        size_t need = len + n;
        size_t cap = buf.length < 4096 ? 4096 : buf.length;
        while (cap < need)
            cap = (cap > size_t.max / 2) ? need : cap * 2;
        if (cap > limit)
            cap = limit;
        ubyte[] bigger = reallocArray(buf, cap);
        if (bigger is null)
        {
            error = BrotliStatus.outOfMemory;
            return false;
        }
        buf = bigger;
        return true;
    }
}

// ---------------------------------------------------------------------------
// Bit reader
// ---------------------------------------------------------------------------

struct BitReader
{
pure nothrow @nogc @safe:
    const(ubyte)[] data;
    size_t pos;     // Next byte to load.
    ulong acc;      // Loaded bits not yet consumed, LSB first. Bits above nbits are zero.
    int nbits;      // Number of bits in acc, <= 63. Negative once bits past
                    // the end of input have been consumed.
    bool overrun;   // Whole bytes were read past the end of input.

    // Read past end of input?
    bool failed() const
    {
        return nbits < 0 || overrun;
    }

    // Load whole bytes into acc, until it has more than 55 bits, or input is exhausted.
    pragma(inline, true)
    void refill()
    {
        if (nbits < 0)
        {
            // Already failed, stay so.
            acc = 0;
            nbits = -1;
            return;
        }
        uint take = (63 - nbits) >> 3; // number of bytes that fit
        if (data.length - pos >= 8)
        {
            const(ubyte)[] p = data[pos .. pos + 8];
            ulong v = p[0]
                    | (cast(ulong)p[1] << 8)
                    | (cast(ulong)p[2] << 16)
                    | (cast(ulong)p[3] << 24)
                    | (cast(ulong)p[4] << 32)
                    | (cast(ulong)p[5] << 40)
                    | (cast(ulong)p[6] << 48)
                    | (cast(ulong)p[7] << 56);
            v &= (1UL << (8 * take)) - 1; // take <= 7
            acc |= v << nbits;
            pos += take;
            nbits += 8 * take;
        }
        else
        {
            for ( ; take > 0 && pos < data.length; --take)
            {
                acc |= cast(ulong)data[pos++] << nbits;
                nbits += 8;
            }
        }
    }

    // Next bits, without consuming them. At least `n` of them must be
    // asked with `ensure(n)` first, or they will read as zeroes.
    pragma(inline, true)
    uint peek() const
    {
        return cast(uint) acc;
    }

    pragma(inline, true)
    ulong peek64() const
    {
        return acc;
    }

    // Make sure n bits are buffered, if the input has them. n <= 56.
    pragma(inline, true)
    void ensure(uint n)
    {
        if (nbits < cast(int) n)
            refill();
    }

    // Consume n bits, n <= 48. Consuming bits past the end of input makes
    // nbits negative: `failed` becomes true, and the missing bits read as zeroes
    // (bits of acc above nbits are zero).
    pragma(inline, true)
    void drop(uint n)
    {
        acc >>= n;
        nbits -= n;
    }

    // Read n bits, n <= 24. Reads past the end of input return zeroes,
    // and set `failed`.
    pragma(inline, true)
    uint readBits(uint n)
    {
        assert(n <= 24);
        ensure(n);
        uint v = cast(uint)(acc & ((1UL << n) - 1));
        drop(n);
        return v;
    }

    // Skip to the next byte boundary, padding bits must be zero.
    // Then give back the buffered bytes, so that `pos` is the next byte to read.
    bool jumpToByteBoundary()
    {
        if (nbits < 0)
            return false;
        uint r = nbits & 7;
        bool ok = (acc & ((1UL << r) - 1)) == 0;
        pos -= nbits >> 3;
        acc = 0;
        nbits = 0;
        return ok;
    }

    // Read whole bytes, must be on a byte boundary.
    bool readBytes(scope ubyte[] dst)
    {
        assert(nbits == 0);
        if (dst.length > data.length - pos)
        {
            overrun = true;
            return false;
        }
        dst[] = data[pos .. pos + dst.length];
        pos += dst.length;
        return true;
    }

    // Skip whole bytes, must be on a byte boundary.
    bool skipBytes(size_t n)
    {
        assert(nbits == 0);
        if (n > data.length - pos)
        {
            overrun = true;
            return false;
        }
        pos += n;
        return true;
    }
}

// Decodes a number in the range [0..255], by reading 1 - 11 bits.
uint readVarLenUint8(ref BitReader br)
{
    if (br.readBits(1) == 0)
        return 0;
    uint n = br.readBits(3);
    if (n == 0)
        return 1;
    return br.readBits(n) + (1u << n);
}

// ---------------------------------------------------------------------------
// Prefix codes
// ---------------------------------------------------------------------------

// A prefix code is decoded with a lookup table of uint entries, each of them
// being (number of bits << 16 | symbol). The root table is indexed with the
// next HUFFMAN_TABLE_BITS bits of input. Longer codes go through a 2nd level
// table: the root entry then holds (2nd level table bits + HUFFMAN_TABLE_BITS) << 16
// and the offset of the 2nd level table relative to the root entry.
enum HUFFMAN_TABLE_BITS = 8;
enum HUFFMAN_TABLE_MASK = 0xFF;
enum MAX_CODE_LENGTH = 15;

enum MAX_ALPHABET_SIZE = 704; // insert-and-copy alphabet is the largest

// Maximum table size for an alphabet size of (index * 32), max code length 15
// and root table bits 8.
static immutable ushort[23] kMaxHuffmanTableSize =
[
    256, 402, 436, 468, 500, 534, 566, 598, 630, 662, 694, 726, 758, 790, 822,
    854, 886, 920, 952, 984, 1016, 1048, 1080
];

size_t maxHuffmanTableSize(uint alphabetSize)
{
    return kMaxHuffmanTableSize[(alphabetSize + 31) >> 5];
}

// Returns the binary reversed `key` incremented, for a code of `len` bits.
uint getNextKey(uint key, uint len)
{
    uint step = 1u << (len - 1);
    while ((key & step) != 0)
        step >>= 1;
    return (key & (step - 1)) + step;
}

// Stores `item` in table[0], table[step], table[2 * step] .., table[end - step].
void replicateValue(scope uint[] table, uint step, uint end, uint item)
{
    uint pos = end;
    while (pos > 0)
    {
        pos -= step;
        table[pos] = item;
    }
}

// Returns the bit width of the 2nd level table, for codes of length `len`.
uint nextTableBitSize(ref const(int)[16] count, uint len)
{
    uint bits = len;
    int left = 1 << (bits - HUFFMAN_TABLE_BITS);
    while (bits < MAX_CODE_LENGTH)
    {
        left -= count[bits];
        if (left <= 0)
            break;
        bits++;
        left <<= 1;
    }
    return bits - HUFFMAN_TABLE_BITS;
}

// Build a decoding table from code lengths (0 = unused symbol).
// Lengths must describe a complete code, or a single symbol.
// `table` must have maxHuffmanTableSize(lengths.length) entries.
void buildHuffmanTable(scope uint[] table, scope const(ubyte)[] lengths)
{
    assert(table.length >= maxHuffmanTableSize(cast(uint) lengths.length));
    ushort[MAX_ALPHABET_SIZE] sorted; // symbols sorted by code length
    int[16] count;  // number of codes of each length
    int[16] offset; // offsets in sorted table for each length

    foreach (l; lengths)
        count[l]++;
    offset[1] = 0;
    foreach (len; 1 .. MAX_CODE_LENGTH)
        offset[len + 1] = offset[len] + count[len];
    foreach (sym, l; lengths)
        if (l != 0)
            sorted[offset[l]++] = cast(ushort) sym;

    uint tableBits = HUFFMAN_TABLE_BITS;
    uint tableSize = 1u << tableBits;
    uint totalSize = tableSize;

    // Special case: code with a single symbol, it takes zero bits.
    if (offset[MAX_CODE_LENGTH] == 1)
    {
        table[0 .. totalSize] = sorted[0];
        return;
    }

    // Fill in root table.
    uint key = 0; // reversed prefix code
    uint symbol = 0;
    uint step = 1;
    foreach (len; 1 .. HUFFMAN_TABLE_BITS + 1)
    {
        step <<= 1;
        while (count[len] > 0)
        {
            replicateValue(table[key .. $], step, tableSize, (len << 16) | sorted[symbol++]);
            key = getNextKey(key, len);
            count[len]--;
        }
    }

    // Fill in 2nd level tables and add pointers to root table.
    uint mask = totalSize - 1;
    uint low = uint.max;
    uint currentOffset = 0;
    step = 1;
    foreach (len; HUFFMAN_TABLE_BITS + 1 .. MAX_CODE_LENGTH + 1)
    {
        step <<= 1;
        while (count[len] > 0)
        {
            if ((key & mask) != low)
            {
                currentOffset += tableSize;
                tableBits = nextTableBitSize(count, len);
                tableSize = 1u << tableBits;
                totalSize += tableSize;
                low = key & mask;
                table[low] = ((tableBits + HUFFMAN_TABLE_BITS) << 16) | (currentOffset - low);
            }
            replicateValue(table[currentOffset + (key >> HUFFMAN_TABLE_BITS) .. $], step, tableSize,
                           ((len - HUFFMAN_TABLE_BITS) << 16) | sorted[symbol++]);
            key = getNextKey(key, len);
            count[len]--;
        }
    }
}

// Decode one symbol.
pragma(inline, true)
uint readSymbol(ref BitReader br, scope const(uint)[] table)
{
    return readSymbolAt(br, table, 0);
}

// Decode one symbol, the table starting at `tables[base]`.
pragma(inline, true)
uint readSymbolAt(ref BitReader br, scope const(uint)[] tables, size_t base)
{
    br.ensure(MAX_CODE_LENGTH);
    uint bits = br.peek();
    size_t index = base + (bits & HUFFMAN_TABLE_MASK);
    uint entry = tables[index];
    uint n = entry >> 16;
    if (n <= HUFFMAN_TABLE_BITS)
    {
        br.drop(n);
        return entry & 0xFFFF;
    }
    index += (entry & 0xFFFF) + ((bits & ((1u << n) - 1)) >> HUFFMAN_TABLE_BITS);
    entry = tables[index];
    br.drop(HUFFMAN_TABLE_BITS + (entry >> 16));
    return entry & 0xFFFF;
}

static immutable ubyte[18] kCodeLengthCodeOrder =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

// Code lengths of the code length alphabet use a static code:
//   0: 00, 1: 0111, 2: 011, 3: 10, 4: 01, 5: 1111
// (bits parsed from right to left).
uint readCodeLengthCodeLength(ref BitReader br)
{
    switch (br.readBits(2))
    {
        case 0: return 0;
        case 1: return 4;
        case 2: return 3;
        default:
            if (br.readBits(1) == 0)
                return 2;
            return br.readBits(1) == 0 ? 1 : 5;
    }
}

// Read a prefix code for an alphabet of `alphabetSize` symbols, build its decoding table.
bool readPrefixCode(ref BitReader br, uint alphabetSize, scope uint[] table)
{
    assert(alphabetSize >= 2 && alphabetSize <= MAX_ALPHABET_SIZE);
    assert(table.length >= maxHuffmanTableSize(alphabetSize));

    ubyte[MAX_ALPHABET_SIZE] lengthsStorage = 0;
    ubyte[] lengths = lengthsStorage[0 .. alphabetSize];

    uint hskip = br.readBits(2);
    if (hskip == 1)
    {
        // Simple prefix code.
        uint nbits = 0;
        for (uint x = alphabetSize - 1; x != 0; x >>= 1)
            ++nbits;
        uint nsym = br.readBits(2) + 1;
        uint[4] sym;
        foreach (i; 0 .. nsym)
        {
            sym[i] = br.readBits(nbits);
            if (sym[i] >= alphabetSize)
                return false;
            foreach (j; 0 .. i)
                if (sym[j] == sym[i])
                    return false;
        }
        switch (nsym)
        {
            case 1:
                lengths[sym[0]] = 1; // single symbol: zero-length code
                break;
            case 2:
                lengths[sym[0]] = 1;
                lengths[sym[1]] = 1;
                break;
            case 3:
                lengths[sym[0]] = 1;
                lengths[sym[1]] = 2;
                lengths[sym[2]] = 2;
                break;
            default:
                if (br.readBits(1) == 0)
                {
                    lengths[sym[0]] = 2;
                    lengths[sym[1]] = 2;
                    lengths[sym[2]] = 2;
                    lengths[sym[3]] = 2;
                }
                else
                {
                    lengths[sym[0]] = 1;
                    lengths[sym[1]] = 2;
                    lengths[sym[2]] = 3;
                    lengths[sym[3]] = 3;
                }
                break;
        }
    }
    else
    {
        // Complex prefix code. First read the code length code lengths.
        ubyte[18] clLengths = 0;
        int space = 32;
        int numCodes = 0;
        foreach (i; hskip .. 18)
        {
            uint v = readCodeLengthCodeLength(br);
            clLengths[kCodeLengthCodeOrder[i]] = cast(ubyte) v;
            if (v != 0)
            {
                space -= 32 >> v;
                ++numCodes;
                if (space <= 0)
                    break;
            }
        }
        if (space != 0 && numCodes != 1)
            return false;

        uint[kMaxHuffmanTableSize[(18 + 31) >> 5]] clTable;
        buildHuffmanTable(clTable[], clLengths[]);

        // Then the symbol code lengths.
        uint symbol = 0;
        uint prevCodeLen = 8;
        uint repeat = 0;
        uint repeatCodeLen = 0;
        space = 32768;
        while (symbol < alphabetSize && space > 0)
        {
            uint codeLen = readSymbol(br, clTable[]);
            if (codeLen < 16)
            {
                repeat = 0;
                lengths[symbol++] = cast(ubyte) codeLen;
                if (codeLen != 0)
                {
                    prevCodeLen = codeLen;
                    space -= 32768 >> codeLen;
                }
            }
            else
            {
                uint extraBits = codeLen - 14; // 16: 2 bits, 17: 3 bits
                uint newLen = (codeLen == 16) ? prevCodeLen : 0;
                if (repeatCodeLen != newLen)
                {
                    repeat = 0;
                    repeatCodeLen = newLen;
                }
                uint oldRepeat = repeat;
                if (repeat > 0)
                {
                    repeat -= 2;
                    repeat <<= extraBits;
                }
                repeat += br.readBits(extraBits) + 3;
                uint repeatDelta = repeat - oldRepeat;
                if (repeatDelta > alphabetSize - symbol)
                    return false;
                lengths[symbol .. symbol + repeatDelta] = cast(ubyte) repeatCodeLen;
                symbol += repeatDelta;
                if (repeatCodeLen != 0)
                    space -= cast(int)(repeatDelta << (15 - repeatCodeLen));
            }
        }
        if (space != 0)
            return false;
    }
    buildHuffmanTable(table, lengths);
    return true;
}

// ---------------------------------------------------------------------------
// Context maps
// ---------------------------------------------------------------------------

void inverseMoveToFront(scope ubyte[] v)
{
    ubyte[256] mtf;
    foreach (i; 0 .. 256)
        mtf[i] = cast(ubyte) i;
    foreach (ref x; v)
    {
        uint index = x;
        ubyte value = mtf[index];
        x = value;
        for (uint k = index; k > 0; --k)
            mtf[k] = mtf[k - 1];
        mtf[0] = value;
    }
}

bool decodeContextMap(ref BitReader br, scope ubyte[] map, out uint numTrees)
{
    numTrees = readVarLenUint8(br) + 1;
    if (numTrees == 1)
    {
        map[] = 0;
        return true;
    }
    uint maxRunLengthPrefix = 0;
    if (br.readBits(1) != 0)
        maxRunLengthPrefix = br.readBits(4) + 1;
    uint alphabetSize = numTrees + maxRunLengthPrefix;
    uint[kMaxHuffmanTableSize[(256 + 16 + 31) >> 5]] tableStorage;
    uint[] table = tableStorage[0 .. maxHuffmanTableSize(alphabetSize)];
    if (!readPrefixCode(br, alphabetSize, table))
        return false;
    size_t i = 0;
    while (i < map.length)
    {
        uint code = readSymbol(br, table);
        if (code == 0)
        {
            map[i++] = 0;
        }
        else if (code <= maxRunLengthPrefix)
        {
            size_t reps = (1u << code) + br.readBits(code);
            if (reps > map.length - i)
                return false;
            map[i .. i + reps] = 0;
            i += reps;
        }
        else
        {
            map[i++] = cast(ubyte)(code - maxRunLengthPrefix);
        }
    }
    if (br.readBits(1) != 0)
        inverseMoveToFront(map);
    // Always true by construction, checked for defense in depth.
    foreach (x; map)
        if (x >= numTrees)
            return false;
    return true;
}

// ---------------------------------------------------------------------------
// Command and distance codes
// ---------------------------------------------------------------------------

static immutable uint[26] kBlockLengthOffset =
[
    1, 5, 9, 13, 17, 25, 33, 41, 49, 65, 81, 97, 113, 145, 177, 209, 241, 305, 369, 497,
    753, 1265, 2289, 4337, 8433, 16625
];

static immutable ubyte[26] kBlockLengthBits =
[
    2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 24
];

static immutable uint[24] kInsertLengthOffset =
[
    0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 26, 34, 50, 66, 98, 130, 194, 322, 578, 1090, 2114,
    6210, 22594
];

static immutable ubyte[24] kInsertLengthBits =
[
    0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 24
];

static immutable uint[24] kCopyLengthOffset =
[
    2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 22, 30, 38, 54, 70, 102, 134, 198, 326, 582,
    1094, 2118
];

static immutable ubyte[24] kCopyLengthBits =
[
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 24
];

// Insert-and-copy symbol: (symbol >> 6) selects a cell of 8x8 codes.
// Cells 0 and 1 use an implicit distance code 0.
static immutable ubyte[11] kInsertCodeBase = [0, 0, 0, 0, 8, 8, 0, 16, 8, 16, 16];
static immutable ubyte[11] kCopyCodeBase   = [0, 8, 0, 8, 0, 8, 16, 0, 16, 8, 16];

// Insert-and-copy symbol decoded: extra bits and offsets of both lengths.
struct CommandCode
{
    ubyte insertBits;
    ubyte copyBits;
    ushort copyOffset;
    uint insertOffset;
}

static immutable CommandCode[704] kCommandCodes = makeCommandCodes();

CommandCode[704] makeCommandCodes()
{
    CommandCode[704] r;
    foreach (cmd; 0 .. 704)
    {
        uint cell = cmd >> 6;
        uint insertCode = kInsertCodeBase[cell] + ((cmd >> 3) & 7);
        uint copyCode = kCopyCodeBase[cell] + (cmd & 7);
        r[cmd].insertBits = kInsertLengthBits[insertCode];
        r[cmd].copyBits = kCopyLengthBits[copyCode];
        r[cmd].insertOffset = kInsertLengthOffset[insertCode];
        r[cmd].copyOffset = cast(ushort) kCopyLengthOffset[copyCode];
    }
    return r;
}

// Distance short codes 0..15: index in the ring of last distances
// (relative to the last one), and value offset.
static immutable ubyte[16] kDistanceShortCodeIndexOffset =
    [0, 3, 2, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3];
static immutable byte[16] kDistanceShortCodeValueOffset =
    [0, 0, 0, 0, -1, 1, -2, 2, -3, 3, -1, 1, -2, 2, -3, 3];

// ---------------------------------------------------------------------------
// Static dictionary and transforms
// ---------------------------------------------------------------------------

enum MIN_DICTIONARY_WORD_LENGTH = 4;
enum MAX_DICTIONARY_WORD_LENGTH = 24;

static immutable ubyte[25] kDictionarySizeBits =
[
    0, 0, 0, 0, 10, 10, 11, 11, 10, 10, 10, 10, 10, 9, 9, 8, 7, 7, 8, 7, 7, 6, 6, 5, 5
];

static immutable uint[25] kDictionaryOffsets =
[
    0, 0, 0, 0, 0, 4096, 9216, 21504, 35840, 44032, 53248, 63488, 74752, 87040, 93696,
    100864, 104704, 106752, 108928, 113536, 115968, 118528, 119872, 121280, 122016
];

enum DICTIONARY_SIZE = 122784;

struct Transform
{
    string prefix;
    ubyte type;
    string suffix;
}

enum : ubyte
{
    TRANSFORM_IDENTITY        = 0,
    TRANSFORM_OMIT_LAST_1     = 1,
    TRANSFORM_OMIT_LAST_9     = 9,
    TRANSFORM_UPPERCASE_FIRST = 10,
    TRANSFORM_UPPERCASE_ALL   = 11,
    TRANSFORM_OMIT_FIRST_1    = 12,
    TRANSFORM_OMIT_FIRST_9    = 20,
}

enum MAX_TRANSFORMED_WORD_LENGTH = maxTransformedWordLength();

size_t maxTransformedWordLength()
{
    size_t r = 0;
    foreach (t; kTransforms)
    {
        size_t l = t.prefix.length + MAX_DICTIONARY_WORD_LENGTH + t.suffix.length;
        if (l > r)
            r = l;
    }
    return r;
}

// Uppercase in the peculiar Brotli way.
void toUpperCase(scope ubyte[] w, bool firstOnly)
{
    size_t i = 0;
    while (i < w.length)
    {
        if (w[i] < 0xC0)
        {
            if (w[i] >= 'a' && w[i] <= 'z')
                w[i] ^= 32;
            i += 1;
        }
        else if (w[i] < 0xE0)
        {
            if (i + 1 < w.length)
                w[i + 1] ^= 32;
            i += 2;
        }
        else
        {
            if (i + 2 < w.length)
                w[i + 2] ^= 5;
            i += 3;
        }
        if (firstOnly)
            break;
    }
}

// Write transformed `word` to `dst`, return its length.
size_t transformWord(scope ubyte[] dst, scope const(ubyte)[] word, ref const(Transform) t)
{
    size_t n = 0;
    foreach (c; t.prefix)
        dst[n++] = c;

    uint type = t.type;
    size_t start = 0;
    size_t len = word.length;
    if (type >= TRANSFORM_OMIT_LAST_1 && type <= TRANSFORM_OMIT_LAST_9)
    {
        size_t omit = type - TRANSFORM_OMIT_LAST_1 + 1;
        len = (omit > len) ? 0 : len - omit;
    }
    else if (type >= TRANSFORM_OMIT_FIRST_1 && type <= TRANSFORM_OMIT_FIRST_9)
    {
        size_t omit = type - TRANSFORM_OMIT_FIRST_1 + 1;
        if (omit > len)
            omit = len;
        start = omit;
        len -= omit;
    }
    size_t wordStart = n;
    foreach (c; word[start .. start + len])
        dst[n++] = c;
    if (type == TRANSFORM_UPPERCASE_FIRST || type == TRANSFORM_UPPERCASE_ALL)
        toUpperCase(dst[wordStart .. n], type == TRANSFORM_UPPERCASE_FIRST);

    foreach (c; t.suffix)
        dst[n++] = c;
    return n;
}

immutable(ubyte)[] dictionary()
{
    return cast(immutable(ubyte)[]) kDictionaryData;
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

enum NUM_LITERAL_CODES = 256;
enum NUM_COMMAND_CODES = 704;
enum NUM_BLOCK_LENGTH_CODES = 26;

// Block types and block counts of one category (literal, command, distance).
struct BlockCategory
{
pure nothrow @nogc @safe:
    uint numTypes;
    uint[2] ring;     // [second-to-last type, last type]
    uint blockLength; // remaining in current block
    uint[kMaxHuffmanTableSize[(256 + 2 + 31) >> 5]] typeTable;
    uint[kMaxHuffmanTableSize[(NUM_BLOCK_LENGTH_CODES + 31) >> 5]] lengthTable;

    uint type() const
    {
        return ring[1];
    }
}

uint readBlockLength(ref BitReader br, scope const(uint)[] table)
{
    uint code = readSymbol(br, table);
    return kBlockLengthOffset[code] + br.readBits(kBlockLengthBits[code]);
}

bool readBlockSwitch(ref BitReader br, ref BlockCategory cat)
{
    if (cat.numTypes < 2)
        return false; // only reachable with an exhausted 16M block
    uint blockType = readSymbol(br, cat.typeTable[]);
    cat.blockLength = readBlockLength(br, cat.lengthTable[]);
    if (blockType == 1)
        blockType = cat.ring[1] + 1;
    else if (blockType == 0)
        blockType = cat.ring[0];
    else
        blockType -= 2;
    if (blockType >= cat.numTypes)
        blockType -= cat.numTypes;
    cat.ring[0] = cat.ring[1];
    cat.ring[1] = blockType;
    return true;
}

// Memory reused across meta-blocks.
struct Workspace
{
pure nothrow @nogc @safe:
    uint[] literalTables;
    uint[] commandTables;
    uint[] distanceTables;
    ubyte[] literalContextMap;
    ubyte[] distanceContextMap;

    void release()
    {
        freeArray(literalTables);
        freeArray(commandTables);
        freeArray(distanceTables);
        freeArray(literalContextMap);
        freeArray(distanceContextMap);
    }
}

// State kept across meta-blocks.
struct StreamState
{
    size_t maxBackwardDistance;
    long[4] distRing = [16, 15, 11, 4];
    uint distRingIdx = 3; // index of the last distance
}

// Prefix codes of the same kind, each of them using `stride` table entries.
struct TableGroup
{
pure nothrow @nogc @safe:
    const(uint)[] tables;
    size_t stride;

    pragma(inline, true)
    const(uint)[] opIndex(size_t i) const
    {
        size_t start = i * stride;
        return tables[start .. start + stride];
    }
}

// Read `count` prefix codes into `storage`.
BrotliStatus readTableGroup(ref BitReader br, uint alphabetSize, uint count,
                            ref uint[] storage, out TableGroup group)
{
    size_t stride = maxHuffmanTableSize(alphabetSize);
    if (!ensureLength(storage, stride * count))
        return BrotliStatus.outOfMemory;
    foreach (i; 0 .. count)
        if (!readPrefixCode(br, alphabetSize, storage[i * stride .. (i + 1) * stride]))
            return BrotliStatus.corrupted;
    group.tables = storage[0 .. stride * count];
    group.stride = stride;
    return BrotliStatus.success;
}

BrotliStatus decodeCompressedMetaBlock(ref BitReader br, ref OutputBuffer o, ref Workspace ws,
                                       ref StreamState st, size_t metaBlockLength)
{
    enum corrupted = BrotliStatus.corrupted;
    enum oom = BrotliStatus.outOfMemory;

    // Block types and counts: 0 = literals, 1 = commands, 2 = distances.
    BlockCategory[3] cats;
    foreach (ref cat; cats)
    {
        cat.numTypes = readVarLenUint8(br) + 1;
        cat.ring = [1, 0];
        if (cat.numTypes >= 2)
        {
            if (!readPrefixCode(br, cat.numTypes + 2, cat.typeTable[]))
                return corrupted;
            if (!readPrefixCode(br, NUM_BLOCK_LENGTH_CODES, cat.lengthTable[]))
                return corrupted;
            cat.blockLength = readBlockLength(br, cat.lengthTable[]);
        }
        else
            cat.blockLength = 1u << 24;
    }

    uint npostfix = br.readBits(2);
    uint ndirect = br.readBits(4) << npostfix;

    ubyte[256] contextModes;
    foreach (i; 0 .. cats[0].numTypes)
        contextModes[i] = cast(ubyte) br.readBits(2);

    uint numLiteralTrees, numDistanceTrees;
    size_t literalMapSize = cats[0].numTypes * 64;
    if (!ensureLength(ws.literalContextMap, literalMapSize))
        return oom;
    ubyte[] literalMap = ws.literalContextMap[0 .. literalMapSize];
    if (!decodeContextMap(br, literalMap, numLiteralTrees))
        return corrupted;

    size_t distanceMapSize = cats[2].numTypes * 4;
    if (!ensureLength(ws.distanceContextMap, distanceMapSize))
        return oom;
    ubyte[] distanceMap = ws.distanceContextMap[0 .. distanceMapSize];
    if (!decodeContextMap(br, distanceMap, numDistanceTrees))
        return corrupted;

    TableGroup literalTables, commandTables, distanceTables;
    BrotliStatus r = readTableGroup(br, NUM_LITERAL_CODES, numLiteralTrees,
                                    ws.literalTables, literalTables);
    if (r != BrotliStatus.success)
        return r;
    r = readTableGroup(br, NUM_COMMAND_CODES, cats[1].numTypes, ws.commandTables, commandTables);
    if (r != BrotliStatus.success)
        return r;
    uint distanceAlphabetSize = 16 + ndirect + (48u << npostfix);
    r = readTableGroup(br, distanceAlphabetSize, numDistanceTrees, ws.distanceTables, distanceTables);
    if (r != BrotliStatus.success)
        return r;

    if (br.failed)
        return corrupted;

    // For each literal block type: is the same prefix code used in every context?
    bool[256] trivialLiteralContext;
    foreach (t; 0 .. cats[0].numTypes)
    {
        trivialLiteralContext[t] = true;
        foreach (x; literalMap[t * 64 .. t * 64 + 64])
            if (x != literalMap[t * 64])
                trivialLiteralContext[t] = false;
    }

    immutable(ubyte)[] dict = dictionary();
    size_t remaining = metaBlockLength;

    while (remaining > 0)
    {
        if (br.failed)
            return corrupted;

        // Insert-and-copy command.
        if (cats[1].blockLength == 0)
        {
            if (!readBlockSwitch(br, cats[1]))
                return corrupted;
        }
        cats[1].blockLength--;
        uint cmd = readSymbol(br, commandTables[cats[1].type]);
        CommandCode code = kCommandCodes[cmd];
        uint insertBits = code.insertBits;
        uint copyBits = code.copyBits;
        br.ensure(insertBits + copyBits); // <= 48 bits
        ulong extra = br.peek64();
        br.drop(insertBits + copyBits);
        size_t insertLength = code.insertOffset + cast(size_t)(extra & ((1UL << insertBits) - 1));
        size_t copyLength = code.copyOffset + cast(size_t)((extra >> insertBits) & ((1UL << copyBits) - 1));

        // Literals, decoded in runs that stay in the same literal block.
        if (insertLength > remaining)
            return corrupted;
        if (!o.reserve(insertLength))
            return o.error;
        ubyte[] dst = o.buf[o.len .. o.len + insertLength];
        size_t k = 0;
        while (k < insertLength)
        {
            if (cats[0].blockLength == 0)
            {
                if (!readBlockSwitch(br, cats[0]))
                    return corrupted;
            }
            size_t run = insertLength - k;
            if (run > cats[0].blockLength)
                run = cats[0].blockLength;
            cats[0].blockLength -= cast(uint) run;

            uint literalType = cats[0].type;
            size_t contextMapBase = literalType * 64;
            const(uint)[] tables = literalTables.tables;
            size_t stride = literalTables.stride;
            // Local copy of the bit reader, so that it can stay in registers.
            BitReader lbr = br;
            if (trivialLiteralContext[literalType])
            {
                size_t base = literalMap[contextMapBase] * stride;
                foreach (ref b; dst[k .. k + run])
                    b = cast(ubyte) readSymbolAt(lbr, tables, base);
            }
            else
            {
                uint mode = contextModes[literalType];
                const(ubyte[256])* lut1 = &kContextLookupTables[mode * 2];
                const(ubyte[256])* lut2 = &kContextLookupTables[mode * 2 + 1];
                size_t pos = o.len + k;
                ubyte p1 = pos >= 1 ? o.buf[pos - 1] : 0;
                ubyte p2 = pos >= 2 ? o.buf[pos - 2] : 0;
                foreach (ref b; dst[k .. k + run])
                {
                    uint context = (*lut1)[p1] | (*lut2)[p2];
                    size_t base = literalMap[contextMapBase + context] * stride;
                    p2 = p1;
                    p1 = cast(ubyte) readSymbolAt(lbr, tables, base);
                    b = p1;
                }
            }
            br = lbr;
            k += run;
        }
        o.len += insertLength;
        remaining -= insertLength;
        if (remaining == 0)
            break; // the copy part of the last command is ignored

        // Distance.
        long distance;
        bool pushDistance;
        if (cmd < 128)
        {
            // Implicit distance code 0: last distance, not pushed.
            distance = st.distRing[st.distRingIdx];
            pushDistance = false;
        }
        else
        {
            if (cats[2].blockLength == 0)
            {
                if (!readBlockSwitch(br, cats[2]))
                    return corrupted;
            }
            cats[2].blockLength--;
            uint distanceContext = copyLength > 4 ? 3 : cast(uint)(copyLength - 2);
            uint treeIndex = distanceMap[cats[2].type * 4 + distanceContext];
            uint dcode = readSymbol(br, distanceTables[treeIndex]);
            if (dcode < 16)
            {
                uint idx = (st.distRingIdx + kDistanceShortCodeIndexOffset[dcode]) & 3;
                distance = st.distRing[idx] + kDistanceShortCodeValueOffset[dcode];
                if (distance <= 0)
                    return corrupted;
            }
            else if (dcode < 16 + ndirect)
            {
                distance = dcode - 15;
            }
            else
            {
                uint x = dcode - ndirect - 16;
                uint ndistbits = 1 + (x >> (npostfix + 1));
                uint hcode = x >> npostfix;
                uint lcode = x & ((1u << npostfix) - 1);
                long offset = (cast(long)(2 + (hcode & 1)) << ndistbits) - 4;
                distance = ((offset + br.readBits(ndistbits)) << npostfix) + lcode + ndirect + 1;
            }
            pushDistance = (dcode != 0);
        }

        size_t maxDistance = o.len < st.maxBackwardDistance ? o.len : st.maxBackwardDistance;
        if (distance > maxDistance)
        {
            // Static dictionary reference, not pushed to the distance ring.
            if (copyLength < MIN_DICTIONARY_WORD_LENGTH || copyLength > MAX_DICTIONARY_WORD_LENGTH)
                return corrupted;
            uint shift = kDictionarySizeBits[copyLength];
            ulong address = cast(ulong)(distance - maxDistance - 1);
            ulong wordIndex = address & ((1u << shift) - 1);
            ulong transformIndex = address >> shift;
            if (transformIndex >= kTransforms.length)
                return corrupted;
            size_t wordOffset = kDictionaryOffsets[copyLength] + cast(size_t)(wordIndex * copyLength);
            const(ubyte)[] word = dict[wordOffset .. wordOffset + copyLength];

            ubyte[MAX_TRANSFORMED_WORD_LENGTH] transformed;
            size_t len = transformWord(transformed[], word, kTransforms[cast(size_t) transformIndex]);
            if (len > remaining)
                return corrupted;
            if (!o.reserve(len))
                return o.error;
            o.buf[o.len .. o.len + len] = transformed[0 .. len];
            o.len += len;
            remaining -= len;
        }
        else
        {
            // Backward reference.
            if (pushDistance)
            {
                st.distRingIdx = (st.distRingIdx + 1) & 3;
                st.distRing[st.distRingIdx] = distance;
            }
            if (copyLength > remaining)
                return corrupted;
            if (!o.reserve(copyLength))
                return o.error;
            size_t src = o.len - cast(size_t) distance;
            if (distance >= copyLength)
            {
                // No overlap.
                ubyte[] to = o.buf[o.len .. o.len + copyLength];
                const(ubyte)[] from = o.buf[src .. src + copyLength];
                foreach (j; 0 .. copyLength)
                    to[j] = from[j];
            }
            else
            {
                // Overlap, byte per byte.
                ubyte[] block = o.buf[src .. o.len + copyLength];
                size_t d = cast(size_t) distance;
                foreach (j; d .. d + copyLength)
                    block[j] = block[j - d];
            }
            o.len += copyLength;
            remaining -= copyLength;
        }
    }
    return BrotliStatus.success;
}

BrotliStatus decodeStream(const(ubyte)[] input, ref OutputBuffer o)
{
    enum corrupted = BrotliStatus.corrupted;

    BitReader br;
    br.data = input;

    Workspace ws;
    scope(exit) ws.release();

    // Stream header: window size.
    uint windowBits;
    if (br.readBits(1) == 0)
        windowBits = 16;
    else
    {
        uint n = br.readBits(3);
        if (n != 0)
            windowBits = 17 + n;
        else
        {
            n = br.readBits(3);
            if (n == 1)
                return corrupted; // large window brotli, not supported
            windowBits = (n != 0) ? 8 + n : 17;
        }
    }

    StreamState st;
    st.maxBackwardDistance = (size_t(1) << windowBits) - 16;

    for (;;)
    {
        if (br.failed)
            return corrupted;

        // Meta-block header.
        bool isLast = br.readBits(1) != 0;
        if (isLast)
        {
            if (br.readBits(1) != 0) // ISLASTEMPTY
                break;
        }
        uint nibbles = br.readBits(2) + 4;
        if (nibbles == 7)
        {
            // Metadata, skipped.
            if (br.readBits(1) != 0) // reserved
                return corrupted;
            uint skipBytes = br.readBits(2);
            size_t skipLength = 0;
            if (skipBytes > 0)
            {
                foreach (i; 0 .. skipBytes)
                {
                    uint b = br.readBits(8);
                    if (b == 0 && i + 1 == skipBytes && skipBytes > 1)
                        return corrupted;
                    skipLength |= cast(size_t)b << (8 * i);
                }
                skipLength += 1;
            }
            if (!br.jumpToByteBoundary())
                return corrupted;
            if (!br.skipBytes(skipLength))
                return corrupted;
            if (isLast)
                break;
            continue;
        }

        size_t metaBlockLength = 0;
        foreach (i; 0 .. nibbles)
        {
            uint b = br.readBits(4);
            if (b == 0 && i + 1 == nibbles && nibbles > 4)
                return corrupted;
            metaBlockLength |= cast(size_t)b << (4 * i);
        }
        metaBlockLength += 1;

        bool isUncompressed = false;
        if (!isLast)
            isUncompressed = br.readBits(1) != 0;

        if (isUncompressed)
        {
            if (!br.jumpToByteBoundary())
                return corrupted;
            if (metaBlockLength > input.length - br.pos)
                return corrupted; // truncated, don't even allocate
            if (!o.reserve(metaBlockLength))
                return o.error;
            if (!br.readBytes(o.buf[o.len .. o.len + metaBlockLength]))
                return corrupted;
            o.len += metaBlockLength;
            continue;
        }

        if (br.failed)
            return corrupted;
        BrotliStatus r = decodeCompressedMetaBlock(br, o, ws, st, metaBlockLength);
        if (r != BrotliStatus.success)
            return r;
        if (isLast)
            break;
    }

    // Padding bits must be zero, and nothing must follow the stream.
    if (!br.jumpToByteBoundary())
        return corrupted;
    if (br.failed)
        return corrupted;
    if (br.pos != input.length)
        return corrupted;
    return BrotliStatus.success;
}

unittest
{
    // Dictionary layout.
    assert(dictionary().length == DICTIONARY_SIZE);
    foreach (len; MIN_DICTIONARY_WORD_LENGTH .. MAX_DICTIONARY_WORD_LENGTH)
        assert(kDictionaryOffsets[len + 1] == kDictionaryOffsets[len] + (len << kDictionarySizeBits[len]));
    assert(kDictionaryOffsets[24] + (24 << kDictionarySizeBits[24]) == DICTIONARY_SIZE);

    // Length codes are contiguous.
    foreach (i; 0 .. 23)
    {
        assert(kInsertLengthOffset[i + 1] == kInsertLengthOffset[i] + (1u << kInsertLengthBits[i]));
        assert(kCopyLengthOffset[i + 1] == kCopyLengthOffset[i] + (1u << kCopyLengthBits[i]));
    }
    foreach (i; 0 .. 25)
        assert(kBlockLengthOffset[i + 1] == kBlockLengthOffset[i] + (1u << kBlockLengthBits[i]));
}

unittest
{
    // "hello" compressed with quality 11
    static immutable ubyte[] hello = [0x0b, 0x02, 0x80, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x03];
    ubyte[16] buf;
    size_t len;
    assert(brotliDecompress(hello, buf[], len) == BrotliStatus.success);
    assert(buf[0 .. len] == "hello");
    assert(brotliDecompress(hello, buf[0 .. 4], len) == BrotliStatus.outputTooSmall);
    assert(brotliDecompress(hello[0 .. $ - 1], buf[], len) == BrotliStatus.corrupted);
    static immutable ubyte[] helloTrailing = [0x0b, 0x02, 0x80, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x03, 0x00];
    assert(brotliDecompress(helloTrailing, buf[], len) == BrotliStatus.corrupted);
    assert(brotliDecompress([], buf[], len) == BrotliStatus.corrupted);

    // Empty stream.
    static immutable ubyte[] empty = [0x06];
    assert(brotliDecompress(empty, buf[], len) == BrotliStatus.success);
    assert(len == 0);
}

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
    ubyte[] buf;    // Caller's buffer, or malloc'ed if `growable`.
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
        ubyte[] bigger = reallocArray(buf, cap);
        if (bigger is null)
        {
            error = BrotliStatus.outOfMemory;
            return false;
        }
        buf = bigger;
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

// Growable array of commands.
struct CommandBuffer
{
pure nothrow @nogc @safe:
    Command[] storage;
    size_t length;

    bool push(Command c)
    {
        if (length == storage.length)
        {
            size_t cap = storage.length < 1024 ? 1024 : storage.length * 2;
            Command[] bigger = reallocArray(storage, cap);
            if (bigger is null)
                return false;
            storage = bigger;
        }
        storage[length++] = c;
        return true;
    }

    const(Command)[] opSlice() const
    {
        return storage[0 .. length];
    }

    void release()
    {
        freeArray(storage);
        length = 0;
    }
}

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
uint load32(scope const(ubyte)[] data, size_t pos) @trusted
{
    assert(pos + 4 <= data.length);
    const(ubyte)* p = data.ptr + pos;
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
pragma(inline, true)
size_t matchLength(scope const(ubyte)[] data, size_t a, size_t b, size_t limit) @trusted
{
    assert(a < b && b <= data.length && limit <= data.length - b);
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
    uint[] positions;   // bucket * blockSize + i, relative to `base`
    uint[] counts;      // insertions in each bucket
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
        positions = reallocArray(positions, size_t(1) << (bucketBits + blockBits));
        counts = reallocArray(counts, size_t(1) << bucketBits);
        if (positions is null || counts is null)
            return false;
        counts[] = 0; // positions are only read once written
        base = 0;
        return true;
    }

    void release()
    {
        freeArray(positions);
        freeArray(counts);
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
enum MAX_TREE_SEARCH_DEPTH = 64;
enum MAX_TREE_COMP_LENGTH = 128;
enum uint TREE_INVALID = uint.max;

struct TreeHasher
{
pure nothrow @nogc @safe:
    uint[] buckets;     // root of each tree, relative to `base`
    uint[] forest;      // children of position p: forest[2 * (p & windowMask)], [+ 1]
    size_t windowMask;
    size_t base;
    size_t dictionaryLookups;
    size_t dictionaryMatches;

    // windowSize: power of 2, larger than the largest distance.
    bool initialize(size_t windowSize)
    {
        windowMask = windowSize - 1;
        buckets = reallocArray(buckets, size_t(1) << TREE_BUCKET_BITS);
        forest = reallocArray(forest, 2 * windowSize);
        if (buckets is null || forest is null)
            return false;
        buckets[] = TREE_INVALID; // forest entries are written when a position is inserted
        base = 0;
        return true;
    }

    void release()
    {
        freeArray(buckets);
        freeArray(forest);
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

struct MatchBuffer
{
pure nothrow @nogc @safe:
    BackwardMatch[] storage;
    size_t length;
    bool failed;

    pragma(inline, true)
    void push(BackwardMatch m)
    {
        if (length == storage.length && !grow())
            return;
        storage[length++] = m;
    }

    bool grow()
    {
        size_t cap = storage.length < 4096 ? 4096 : storage.length * 2;
        BackwardMatch[] bigger = reallocArray(storage, cap);
        if (bigger is null)
        {
            failed = true;
            return false;
        }
        storage = bigger;
        return true;
    }

    void release()
    {
        freeArray(storage);
        length = 0;
    }
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
    if (!ensureLength(ws.modeHistograms, 4 * 64 * A))
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

    uint bestMode = 0;
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

    // Costs of each context. Contexts without statistics use those of all contexts.
    if (!ensureLength(ws.literalContextCosts, 64 * A))
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
    if (!ensureLength(ws.numMatches, n))
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
    if (!ensureLength(ws.literalCosts, n + 1) || !ensureLength(ws.nodes, n + 1))
        return false;
    double[] literalCosts = ws.literalCosts[0 .. n + 1];
    ZopfliNode[] nodes = ws.nodes[0 .. n + 1];
    ZopfliCostModel model;
    computeCommandCosts(previous, ws.log2, model);
    if (!computeLiteralCosts(data, start, n, previous, ws, literalCosts))
        return false;
    zopfliShortestPath(data, start, end, maxBackward, last, ws.numMatches[0 .. n],
                       ws.matches.storage[0 .. ws.matches.length], model, literalCosts, params, nodes);
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
    ubyte[] types;
    uint[] lengths;
    size_t numBlocks;
    uint numTypes = 1;

    // A single block of n symbols, n may be 0.
    bool setSingle(size_t n)
    {
        numTypes = 1;
        numBlocks = 0;
        if (n == 0)
            return true;
        if (!ensureLength(types, 1) || !ensureLength(lengths, 1))
            return false;
        types[0] = 0;
        lengths[0] = cast(uint) n;
        numBlocks = 1;
        return true;
    }

    void release()
    {
        freeArray(types);
        freeArray(lengths);
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
    if (!ensureLength(ws.splitHistograms, numHistograms * alphabetSize)
        || !ensureLength(ws.splitBlockIds, n)
        || !ensureLength(ws.splitSignal, n * bitmapLength)
        || !ensureLength(ws.splitInsertCost, numHistograms * alphabetSize)
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
    if (!ensureLength(split.types, numBlocks) || !ensureLength(split.lengths, numBlocks))
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
    uint[] reduced = reallocArray!uint(null, n * alphabetSize);
    scope(exit) freeArray(reduced);
    uint[] reducedMap = reallocArray!uint(null, n);
    scope(exit) freeArray(reducedMap);
    if (reduced is null || reducedMap is null)
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
    ushort[] literals;
    ushort[] commandSymbols;
    ushort[] distanceSymbols;

    // Block splits.
    BlockSplit[3] splits;
    ushort[] typeCodes;
    uint[] splitHistograms;
    ubyte[] splitBlockIds;
    ubyte[] splitSignal;
    float[] splitInsertCost;

    // Histograms and prefix codes.
    uint[] modeHistograms;      // [mode][context][symbol], to choose the literal context mode
    uint[] histograms;
    uint[] treeHistograms;
    uint[] clusterMap;
    ubyte[] depths;
    ushort[] bits;

    // Clustering.
    uint[] clusters;
    ushort[] clusterScratch;
    double[] clusterDelta;
    size_t clusteringCapacity; // histograms of MAX_ALPHABET_SIZE

    uint[] scratch;

    // Optimal parsing.
    CommandBuffer previousCommands;
    CommandBuffer bestCommands;
    ubyte[] trialOutput;
    MatchBuffer matches;
    uint[] numMatches;
    double[] literalCosts;
    float[] literalContextCosts;
    ZopfliNode[] nodes;

    bool initialize()
    {
        log2.initialize();
        scratch = reallocArray(scratch, 2 * MAX_CONTEXT_MAP_LENGTH);
        return scratch !is null;
    }

    // Room to cluster n histograms of alphabetSize symbols.
    bool reserveClustering(size_t n, size_t alphabetSize)
    {
        return ensureLength(clusters, n * alphabetSize)
            && ensureLength(clusterScratch, n * alphabetSize)
            && ensureLength(clusterDelta, n * n);
    }

    void release()
    {
        commands.release();
        freeArray(literals);
        freeArray(commandSymbols);
        freeArray(distanceSymbols);
        foreach (ref s; splits)
            s.release();
        freeArray(typeCodes);
        freeArray(splitHistograms);
        freeArray(splitBlockIds);
        freeArray(splitSignal);
        freeArray(splitInsertCost);
        freeArray(modeHistograms);
        freeArray(histograms);
        freeArray(treeHistograms);
        freeArray(clusterMap);
        freeArray(depths);
        freeArray(bits);
        freeArray(clusters);
        freeArray(clusterScratch);
        freeArray(clusterDelta);
        freeArray(scratch);
        previousCommands.release();
        bestCommands.release();
        freeArray(trialOutput);
        matches.release();
        freeArray(numMatches);
        freeArray(literalCosts);
        freeArray(literalContextCosts);
        freeArray(nodes);
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
    if (!ensureLength(ws.literals, numLiterals) || !ensureLength(ws.commandSymbols, cmds.length)
        || !ensureLength(ws.distanceSymbols, numDistances))
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
    if (params.contextModeling && !params.tryAllContextModes)
    {
        // Cheap estimate: cost of the literals of each context of each mode,
        // without clustering.
        if (!ensureLength(ws.modeHistograms, 4 * 64 * A))
            return oom;
        uint[] mh = ws.modeHistograms[0 .. 4 * 64 * A];
        mh[] = 0;
        size_t pos = start;
        foreach (ref const c; cmds)
        {
            ubyte p1 = pos >= 1 ? data[pos - 1] : 0;
            ubyte p2 = pos >= 2 ? data[pos - 2] : 0;
            foreach (b; data[pos .. pos + c.insertLength])
            {
                foreach (mode; 0 .. 4)
                {
                    uint context = kContextLookupTables[mode * 2][p1] | kContextLookupTables[mode * 2 + 1][p2];
                    mh[(mode * 64 + context) * A + b]++;
                }
                p2 = p1;
                p1 = b;
            }
            pos += c.insertLength + c.outputLength;
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
                contextMode = mode;
            }
        }
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
    if (!ensureLength(ws.histograms, literalSize > distanceSize ? literalSize : distanceSize)
        || !ensureLength(ws.treeHistograms, literalSize + (distanceSize > commandSize ? distanceSize : commandSize))
        || !ensureLength(ws.clusterMap, numLiteralHistograms > numDistanceHistograms
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
    if (!ensureLength(ws.typeCodes, numBlocks))
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
    if (!ensureLength(ws.depths, literalCodes + commandCodes + distanceCodes)
        || !ensureLength(ws.bits, literalCodes + commandCodes + distanceCodes))
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
            BitWriter trial;
            trial.growable = true;
            trial.buf = ws.trialOutput;
            BrotliStatus r = writeCompressedMetaBlock(trial, data, start, length, ws.commands[], isLast,
                                                      params, ws);
            ws.trialOutput = trial.buf;
            if (r != BrotliStatus.success)
                return r;
            if (trial.bitPosition() < bestBits)
            {
                bestBits = trial.bitPosition();
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
        while (windowSize < input.length && windowSize < (size_t(1) << lgwin))
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
        scope(exit) freeArray(w.buf);
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
    uint[] scratch = reallocArray!uint(null, 2 * MAX_CONTEXT_MAP_LENGTH);
    scope(exit) freeArray(scratch);
    ubyte[] map = reallocArray!ubyte(null, MAX_CONTEXT_MAP_LENGTH);
    scope(exit) freeArray(map);
    ubyte[] decodedStorage = reallocArray!ubyte(null, MAX_CONTEXT_MAP_LENGTH);
    scope(exit) freeArray(decodedStorage);
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
        scope(exit) freeArray(w.buf);
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
    double[] delta = reallocArray!double(null, MAX_CLUSTER_INPUTS * MAX_CLUSTER_INPUTS);
    scope(exit) freeArray(delta);
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
    ushort[] symbols = reallocArray!ushort(null, 50_000);
    scope(exit) freeArray(symbols);
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
    ubyte[] data = reallocArray!ubyte(null, 30_000);
    scope(exit) freeArray(data);
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
            foreach (ref m; matches.storage[0 .. matches.length])
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

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

// Context lookup table, [mode * 512 + p1] | [mode * 512 + 256 + p2]
// Same, as 8 tables: [mode * 2] for p1, [mode * 2 + 1] for p2.
static immutable ubyte[256][8] kContextLookupTables = splitContextLookup();

ubyte[256][8] splitContextLookup()
{
    ubyte[256][8] r;
    foreach (i; 0 .. 8)
        r[i][] = kContextLookup[i * 256 .. i * 256 + 256];
    return r;
}

static immutable ubyte[2048] kContextLookup =
[
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,
     4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,
     8,  8,  8,  8,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11,
    12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15,
    16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19,
    20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23,
    24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27,
    28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31,
    32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35,
    36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39,
    40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43,
    44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 46, 47, 47, 47, 47,
    48, 48, 48, 48, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51,
    52, 52, 52, 52, 53, 53, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55,
    56, 56, 56, 56, 57, 57, 57, 57, 58, 58, 58, 58, 59, 59, 59, 59,
    60, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 62, 63, 63, 63, 63,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  4,  0,  0,  4,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     8, 12, 16, 12, 12, 20, 12, 16, 24, 28, 12, 12, 32, 12, 36, 12,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 32, 32, 24, 40, 28, 12,
    12, 48, 52, 52, 52, 48, 52, 52, 52, 48, 52, 52, 52, 52, 52, 48,
    52, 52, 52, 52, 52, 48, 52, 52, 52, 52, 52, 24, 12, 28, 12, 12,
    12, 56, 60, 60, 60, 56, 60, 60, 60, 56, 60, 60, 60, 60, 60, 56,
    60, 60, 60, 60, 60, 56, 60, 60, 60, 60, 60, 24, 12, 28, 12,  0,
     0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
     0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
     0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
     0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
     2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
     2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
     2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
     2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,
     1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,
     1,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  1,  1,  1,  1,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
     0,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 56,
     0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
     3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
     4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
     4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
     4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
     5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,
];

// The 121 RFC 7932 transforms: prefix, operation, suffix.
static immutable Transform[121] kTransforms =
[
    Transform("", 0, ""),
    Transform("", 0, " "),
    Transform(" ", 0, " "),
    Transform("", 12, ""),
    Transform("", 10, " "),
    Transform("", 0, " the "),
    Transform(" ", 0, ""),
    Transform("s ", 0, " "),
    Transform("", 0, " of "),
    Transform("", 10, ""),
    Transform("", 0, " and "),
    Transform("", 13, ""),
    Transform("", 1, ""),
    Transform(", ", 0, " "),
    Transform("", 0, ", "),
    Transform(" ", 10, " "),
    Transform("", 0, " in "),
    Transform("", 0, " to "),
    Transform("e ", 0, " "),
    Transform("", 0, "\""),
    Transform("", 0, "."),
    Transform("", 0, "\">"),
    Transform("", 0, "\x0A"),
    Transform("", 3, ""),
    Transform("", 0, "]"),
    Transform("", 0, " for "),
    Transform("", 14, ""),
    Transform("", 2, ""),
    Transform("", 0, " a "),
    Transform("", 0, " that "),
    Transform(" ", 10, ""),
    Transform("", 0, ". "),
    Transform(".", 0, ""),
    Transform(" ", 0, ", "),
    Transform("", 15, ""),
    Transform("", 0, " with "),
    Transform("", 0, "'"),
    Transform("", 0, " from "),
    Transform("", 0, " by "),
    Transform("", 16, ""),
    Transform("", 17, ""),
    Transform(" the ", 0, ""),
    Transform("", 4, ""),
    Transform("", 0, ". The "),
    Transform("", 11, ""),
    Transform("", 0, " on "),
    Transform("", 0, " as "),
    Transform("", 0, " is "),
    Transform("", 7, ""),
    Transform("", 1, "ing "),
    Transform("", 0, "\x0A\x09"),
    Transform("", 0, ":"),
    Transform(" ", 0, ". "),
    Transform("", 0, "ed "),
    Transform("", 20, ""),
    Transform("", 18, ""),
    Transform("", 6, ""),
    Transform("", 0, "("),
    Transform("", 10, ", "),
    Transform("", 8, ""),
    Transform("", 0, " at "),
    Transform("", 0, "ly "),
    Transform(" the ", 0, " of "),
    Transform("", 5, ""),
    Transform("", 9, ""),
    Transform(" ", 10, ", "),
    Transform("", 10, "\""),
    Transform(".", 0, "("),
    Transform("", 11, " "),
    Transform("", 10, "\">"),
    Transform("", 0, "=\""),
    Transform(" ", 0, "."),
    Transform(".com/", 0, ""),
    Transform(" the ", 0, " of the "),
    Transform("", 10, "'"),
    Transform("", 0, ". This "),
    Transform("", 0, ","),
    Transform(".", 0, " "),
    Transform("", 10, "("),
    Transform("", 10, "."),
    Transform("", 0, " not "),
    Transform(" ", 0, "=\""),
    Transform("", 0, "er "),
    Transform(" ", 11, " "),
    Transform("", 0, "al "),
    Transform(" ", 11, ""),
    Transform("", 0, "='"),
    Transform("", 11, "\""),
    Transform("", 10, ". "),
    Transform(" ", 0, "("),
    Transform("", 0, "ful "),
    Transform(" ", 10, ". "),
    Transform("", 0, "ive "),
    Transform("", 0, "less "),
    Transform("", 11, "'"),
    Transform("", 0, "est "),
    Transform(" ", 10, "."),
    Transform("", 11, "\">"),
    Transform(" ", 0, "='"),
    Transform("", 10, ","),
    Transform("", 0, "ize "),
    Transform("", 11, "."),
    Transform("\xC2\xA0", 0, ""),
    Transform(" ", 0, ","),
    Transform("", 10, "=\""),
    Transform("", 11, "=\""),
    Transform("", 0, "ous "),
    Transform("", 11, ", "),
    Transform("", 10, "='"),
    Transform(" ", 10, ","),
    Transform(" ", 11, "=\""),
    Transform(" ", 11, ", "),
    Transform("", 11, ","),
    Transform("", 11, "("),
    Transform("", 11, ". "),
    Transform(" ", 11, "."),
    Transform("", 11, "='"),
    Transform(" ", 11, ". "),
    Transform(" ", 10, "=\""),
    Transform(" ", 11, "='"),
    Transform(" ", 10, "='"),
];

// RFC 7932 static dictionary, 122784 bytes.
// SHA-256: 20e42eb1b511c21806d4d227d07e5dd06877d8ce7b3a817f378f313653f35c70
static immutable string kDictionaryData =
    "timedownlifeleftbackcodedatashowonlysitecityopenjustlikefreework" ~
    "textyearoverbodyloveformbookplaylivelinehelphomesidemorewordlong" ~
    "themviewfindpagedaysfullheadtermeachareafromtruemarkableuponhigh" ~
    "datelandnewsevennextcasebothpostusedmadehandherewhatnameLinkblog" ~
    "sizebaseheldmakemainuser') +holdendswithNewsreadweresigntakehave" ~
    "gameseencallpathwellplusmenufilmpartjointhislistgoodneedwayswest" ~
    "jobsmindalsologorichuseslastteamarmyfoodkingwilleastwardbestfire" ~
    "Pageknowaway.pngmovethanloadgiveselfnotemuchfeedmanyrockicononce" ~
    "lookhidediedHomerulehostajaxinfoclublawslesshalfsomesuchzone100%" ~
    "onescareTimeracebluefourweekfacehopegavehardlostwhenparkkeptpass" ~
    "shiproomHTMLplanTypedonesavekeepflaglinksoldfivetookratetownjump" ~
    "thusdarkcardfilefearstaykillthatfallautoever.comtalkshopvotedeep" ~
    "moderestturnbornbandfellroseurl(skinrolecomeactsagesmeetgold.jpg" ~
    "itemvaryfeltthensenddropViewcopy1.0\"</a>stopelseliestourpack.gif" ~
    "pastcss?graymean&gt;rideshotlatesaidroadvar feeljohnrickportfast" ~
    "'UA-dead</b>poorbilltypeU.S.woodmust2px;Inforankwidewantwalllead" ~
    "[0];paulwavesure$('#waitmassarmsgoesgainlangpaid!-- lockunitroot" ~
    "walkfirmwifexml\"songtest20pxkindrowstoolfontmailsafestarmapscore" ~
    "rainflowbabyspansays4px;6px;artsfootrealwikiheatsteptriporg/lake" ~
    "weaktoldFormcastfansbankveryrunsjulytask1px;goalgrewslowedgeid=\"" ~
    "sets5px;.js?40pxif (soonseatnonetubezerosentreedfactintogiftharm" ~
    "18pxcamehillboldzoomvoideasyringfillpeakinitcost3px;jacktagsbits" ~
    "rolleditknewnear<!--growJSONdutyNamesaleyou lotspainjazzcoldeyes" ~
    "fishwww.risktabsprev10pxrise25pxBlueding300,ballfordearnwildbox." ~
    "fairlackverspairjunetechif(!pickevil$(\"#warmlorddoespull,000idea" ~
    "drawhugespotfundburnhrefcellkeystickhourlossfuel12pxsuitdealRSS\"" ~
    "agedgreyGET\"easeaimsgirlaids8px;navygridtips#999warsladycars); }" ~
    "php?helltallwhomzh:\xE5*/\x0D\x0A 100hall.\x0A\x0AA7px;pushchat0px;crew*/</hash" ~
    "75pxflatrare && tellcampontolaidmissskiptentfinemalegetsplot400," ~
    "\x0D\x0A\x0D\x0Acoolfeet.php<br>ericmostguidbelldeschairmathatom/img&#82luck" ~
    "cent000;tinygonehtmlselldrugFREEnodenick?id=losenullvastwindRSS " ~
    "wearrelybeensamedukenasacapewishgulfT23:hitsslotgatekickblurthey" ~
    "15px''););\">msiewinsbirdsortbetaseekT18:ordstreemall60pxfarm\xE2\x80\x99s" ~
    "boys[0].');\"POSTbearkids);}}marytend(UK)quadzh:\xE6-siz----prop');\x0D" ~
    "liftT19:viceandydebt>RSSpoolneckblowT16:doorevalT17:letsfailoral" ~
    "pollnovacolsgene \xE2\x80\x94softrometillross<h3>pourfadepink<tr>mini)|!(" ~
    "minezh:\xE8barshear00);milk -->ironfreddiskwentsoilputs/js/holyT22:" ~
    "ISBNT20:adamsees<h2>json', 'contT21: RSSloopasiamoon</p>soulLINE" ~
    "fortcartT14:<h1>80px!--<9px;T04:mike:46ZniceinchYorkricezh:\xE4'));" ~
    "puremageparatonebond:37Z_of_']);000,zh:\xE7tankyardbowlbush:56ZJava" ~
    "30px\x0A|}\x0A%C3%:34ZjeffEXPIcashvisagolfsnowzh:\xE9quer.csssickmeatmin." ~
    "binddellhirepicsrent:36ZHTTP-201fotowolfEND xbox:54ZBODYdick;\x0A}\x0A" ~
    "exit:35Zvarsbeat'});diet999;anne}}</[i].Langkm\xC2\xB2wiretoysaddsseal" ~
    "alex;\x0A\x09}echonine.org005)tonyjewssandlegsroof000) 200winegeardogs" ~
    "bootgarycutstyletemption.xmlcockgang$('.50pxPh.Dmiscalanloandesk" ~
    "mileryanunixdisc);}\x0Adustclip).\x0A\x0A70px-200DVDs7]><tapedemoi++)wage" ~
    "europhiloptsholeFAQsasin-26TlabspetsURL bulkcook;}\x0D\x0AHEAD[0])abbr" ~
    "juan(198leshtwin</i>sonyguysfuckpipe|-\x0A!002)ndow[1];[];\x0ALog salt" ~
    "\x0D\x0A\x09\x09bangtrimbath){\x0D\x0A00px\x0A});ko:\xECfeesad>\x0Ds:// [];tollplug(){\x0A{\x0D\x0A " ~
    ".js'200pdualboat.JPG);\x0A}quot);\x0A\x0A');\x0A\x0D\x0A}\x0D201420152016201720182019" ~
    "2020202120222023202420252026202720282029203020312032203320342035" ~
    "2036203720132012201120102009200820072006200520042003200220012000" ~
    "1999199819971996199519941993199219911990198919881987198619851984" ~
    "1983198219811980197919781977197619751974197319721971197019691968" ~
    "1967196619651964196319621961196019591958195719561955195419531952" ~
    "1951195010001024139400009999comom\xC3\xA1sesteestaperotodohacecadaa\xC3\xB1o" ~
    "biend\xC3\xADaas\xC3\xADvidacasootroforosolootracualdijosidograntipotemadebe" ~
    "algoqu\xC3\xA9estonadatrespococasabajotodasinoaguapuesunosantediceluis" ~
    "ellamayozonaamorpisoobraclicellodioshoracasi\xD0\xB7\xD0\xB0\xD0\xBD\xD0\xB0\xD0\xBE\xD0\xBC\xD1\x80\xD0\xB0\xD1\x80\xD1\x83" ~
    "\xD1\x82\xD0\xB0\xD0\xBD\xD0\xB5\xD0\xBF\xD0\xBE\xD0\xBE\xD1\x82\xD0\xB8\xD0\xB7\xD0\xBD\xD0\xBE\xD0\xB4\xD0\xBE\xD1\x82\xD0\xBE\xD0\xB6\xD0\xB5\xD0\xBE\xD0\xBD\xD0\xB8\xD1\x85\xD0\x9D\xD0\xB0\xD0\xB5\xD0\xB5\xD0\xB1\xD1\x8B\xD0\xBC\xD1\x8B\xD0\x92\xD1\x8B" ~
    "\xD1\x81\xD0\xBE\xD0\xB2\xD1\x8B\xD0\xB2\xD0\xBE\xD0\x9D\xD0\xBE\xD0\xBE\xD0\xB1\xD0\x9F\xD0\xBE\xD0\xBB\xD0\xB8\xD0\xBD\xD0\xB8\xD0\xA0\xD0\xA4\xD0\x9D\xD0\xB5\xD0\x9C\xD1\x8B\xD1\x82\xD1\x8B\xD0\x9E\xD0\xBD\xD0\xB8\xD0\xBC\xD0\xB4\xD0\xB0\xD0\x97\xD0\xB0" ~
    "\xD0\x94\xD0\xB0\xD0\x9D\xD1\x83\xD0\x9E\xD0\xB1\xD1\x82\xD0\xB5\xD0\x98\xD0\xB7\xD0\xB5\xD0\xB9\xD0\xBD\xD1\x83\xD0\xBC\xD0\xBC\xD0\xA2\xD1\x8B\xD1\x83\xD0\xB6\xD9\x81\xD9\x8A\xD8\xA3\xD9\x86\xD9\x85\xD8\xA7\xD9\x85\xD8\xB9\xD9\x83\xD9\x84\xD8\xA3\xD9\x88" ~
    "\xD8\xB1\xD8\xAF\xD9\x8A\xD8\xA7\xD9\x81\xD9\x89\xD9\x87\xD9\x88\xD9\x84\xD9\x85\xD9\x84\xD9\x83\xD8\xA7\xD9\x88\xD9\x84\xD9\x87\xD8\xA8\xD8\xB3\xD8\xA7\xD9\x84\xD8\xA5\xD9\x86\xD9\x87\xD9\x8A\xD8\xA3\xD9\x8A\xD9\x82\xD8\xAF\xD9\x87\xD9\x84\xD8\xAB\xD9\x85" ~
    "\xD8\xA8\xD9\x87\xD9\x84\xD9\x88\xD9\x84\xD9\x8A\xD8\xA8\xD9\x84\xD8\xA7\xD9\x8A\xD8\xA8\xD9\x83\xD8\xB4\xD9\x8A\xD8\xA7\xD9\x85\xD8\xA3\xD9\x85\xD9\x86\xD8\xAA\xD8\xA8\xD9\x8A\xD9\x84\xD9\x86\xD8\xAD\xD8\xA8\xD9\x87\xD9\x85\xD9\x85\xD8\xB4\xD9\x88\xD8\xB4" ~
    "firstvideolightworldmediawhitecloseblackrightsmallbooksplacemusi" ~
    "cfieldorderpointvalueleveltableboardhousegroupworksyearsstatetod" ~
    "aywaterstartstyledeathpowerphonenighterrorinputabouttermstitleto" ~
    "olseventlocaltimeslargewordsgamesshortspacefocusclearmodelblockg" ~
    "uideradiosharewomenagainmoneyimagenamesyounglineslatercolorgreen" ~
    "front&amp;watchforcepricerulesbeginaftervisitissueareasbelowinde" ~
    "xtotalhourslabelprintpressbuiltlinksspeedstudytradefoundsenseund" ~
    "ershownformsrangeaddedstillmovedtakenaboveflashfixedoftenothervi" ~
    "ewschecklegalriveritemsquickshapehumanexistgoingmoviethirdbasicp" ~
    "eacestagewidthloginideaswrotepagesusersdrivestorebreaksouthvoice" ~
    "sitesmonthwherebuildwhichearthforumthreesportpartyClicklowerlive" ~
    "sclasslayerentrystoryusagesoundcourtyour birthpopuptypesapplyIma" ~
    "gebeinguppernoteseveryshowsmeansextramatchtrackknownearlybegansu" ~
    "perpapernorthlearngivennamedendedTermspartsGroupbrandusingwomanf" ~
    "alsereadyaudiotakeswhile.com/livedcasesdailychildgreatjudgethose" ~
    "unitsneverbroadcoastcoverapplefilescyclesceneplansclickwritequee" ~
    "npieceemailframeolderphotolimitcachecivilscaleenterthemetheretou" ~
    "chboundroyalaskedwholesincestock namefaithheartemptyofferscopeow" ~
    "nedmightalbumthinkbloodarraymajortrustcanonunioncountvalidstoneS" ~
    "tyleLoginhappyoccurleft:freshquitefilmsgradeneedsurbanfightbasis" ~
    "hoverauto;route.htmlmixedfinalYour slidetopicbrownalonedrawnspli" ~
    "treachRightdatesmarchquotegoodsLinksdoubtasyncthumballowchiefyou" ~
    "thnovel10px;serveuntilhandsCheckSpacequeryjamesequaltwice0,000St" ~
    "artpanelsongsroundeightshiftworthpostsleadsweeksavoidthesemilesp" ~
    "lanesmartalphaplantmarksratesplaysclaimsalestextsstarswrong</h3>" ~
    "thing.org/multiheardPowerstandtokensolid(thisbringshipsstafftrie" ~
    "dcallsfullyfactsagentThis //-->adminegyptEvent15px;Emailtrue\"cro" ~
    "ssspentblogsbox\">notedleavechinasizesguest</h4>robotheavytrue,se" ~
    "vengrandcrimesignsawaredancephase><!--en_US&#39;200px_namelatine" ~
    "njoyajax.ationsmithU.S. holdspeterindianav\">chainscorecomesdoing" ~
    "priorShare1990sromanlistsjapanfallstrialowneragree</h2>abusealer" ~
    "topera\"-//WcardshillsteamsPhototruthclean.php?saintmetallouismea" ~
    "ntproofbriefrow\">genretrucklooksValueFrame.net/-->\x0A<try {\x0Avar ma" ~
    "kescostsplainadultquesttrainlaborhelpscausemagicmotortheir250pxl" ~
    "eaststepsCountcouldglasssidesfundshotelawardmouthmovesparisgives" ~
    "dutchtexasfruitnull,||[];top\">\x0A<!--POST\"ocean<br/>floorspeakdept" ~
    "h sizebankscatchchart20px;aligndealswould50px;url=\"parksmouseMos" ~
    "t ...</amongbrainbody none;basedcarrydraftreferpage_home.meterde" ~
    "laydreamprovejoint</tr>drugs<!-- aprilidealallenexactforthcodesl" ~
    "ogicView seemsblankports (200saved_linkgoalsgrantgreekhomesrings" ~
    "rated30px;whoseparse();\" Blocklinuxjonespixel');\">);if(-leftdavi" ~
    "dhorseFocusraiseboxesTrackement</em>bar\">.src=toweralt=\"cablehen" ~
    "ry24px;setupitalysharpminortastewantsthis.resetwheelgirls/css/10" ~
    "0%;clubsstuffbiblevotes 1000korea});\x0D\x0Abandsqueue= {};80px;cking{" ~
    "\x0D\x0A\x09\x09aheadclockirishlike ratiostatsForm\"yahoo)[0];Aboutfinds</h1>" ~
    "debugtasksURL =cells})();12px;primetellsturns0x600.jpg\"spainbeac" ~
    "htaxesmicroangel--></giftssteve-linkbody.});\x0A\x09mount (199FAQ</rog" ~
    "erfrankClass28px;feeds<h1><scotttests22px;drink) || lewisshall#0" ~
    "39; for lovedwaste00px;ja:\xE3\x82simon<fontreplymeetsuntercheaptightB" ~
    "rand) != dressclipsroomsonkeymobilmain.Name platefunnytreescom/\"" ~
    "1.jpgwmodeparamSTARTleft idden, 201);\x0A}\x0Aform.viruschairtranswors" ~
    "tPagesitionpatch<!--\x0Ao-cacfirmstours,000 asiani++){adobe')[0]id=" ~
    "10both;menu .2.mi.png\"kevincoachChildbruce2.jpgURL)+.jpg|suitesl" ~
    "iceharry120\" sweettr>\x0D\x0Aname=diegopage swiss-->\x0A\x0A#fff;\">Log.com\"t" ~
    "reatsheet) && 14px;sleepntentfiledja:\xE3\x83id=\"cName\"worseshots-box-" ~
    "delta\x0A&lt;bears:48Z<data-rural</a> spendbakershops= \"\";php\">ctio" ~
    "n13px;brianhellosize=o=%2F joinmaybe<img img\">, fjsimg\" \")[0]MTo" ~
    "pBType\"newlyDanskczechtrailknows</h5>faq\">zh-cn10);\x0A-1\");type=bl" ~
    "uestrulydavis.js';>\x0D\x0A<!steel you h2>\x0D\x0Aform jesus100% menu.\x0D\x0A\x09\x0D\x0Aw" ~
    "alesrisksumentddingb-likteachgif\" vegasdanskeestishqipsuomisobre" ~
    "desdeentretodospuedea\xC3\xB1osest\xC3\xA1tienehastaotrospartedondenuevohace" ~
    "rformamismomejormundoaqu\xC3\xADd\xC3\xADass\xC3\xB3loayudafechatodastantomenosdat" ~
    "osotrassitiomuchoahoralugarmayorestoshorastenerantesfotosestaspa" ~
    "\xC3\xADsnuevasaludforosmedioquienmesespoderchileser\xC3\xA1vecesdecirjos\xC3\xA9e" ~
    "starventagrupohechoellostengoamigocosasnivelgentemismaairesjulio" ~
    "temashaciafavorjuniolibrepuntobuenoautorabrilbuenatextomarzosabe" ~
    "rlistaluegoc\xC3\xB3moenerojuegoper\xC3\xBAhaberestoynuncamujervalorfueralib" ~
    "rogustaigualvotoscasosgu\xC3\xADapuedosomosavisousteddebennochebuscafa" ~
    "ltaeurosseriedichocursoclavecasasle\xC3\xB3nplazolargoobrasvistaapoyoj" ~
    "untotratavistocrearcampohemoscincocargopisosordenhacen\xC3\xA1readisco" ~
    "pedrocercapuedapapelmenor\xC3\xBAtilclarojorgecalleponertardenadiemarc" ~
    "asigueellassiglocochemotosmadreclaserestoni\xC3\xB1oquedapasarbancohij" ~
    "osviajepablo\xC3\xA9stevienereinodejarfondocanalnorteletracausatomarma" ~
    "noslunesautosvillavendopesartipostengamarcollevapadreunidovamosz" ~
    "onasambosbandamariaabusomuchasubirriojavivirgradochicaall\xC3\xADjoven" ~
    "dichaestantalessalirsuelopesosfinesllamabusco\xC3\xA9stalleganegroplaz" ~
    "ahumorpagarjuntadobleislasbolsaba\xC3\xB1ohablalucha\xC3\x81readicenjugarnot" ~
    "asvalleall\xC3\xA1cargadolorabajoest\xC3\xA9gustomentemariofirmacostofichapl" ~
    "atahogarartesleyesaquelmuseobasespocosmitadcielochicomiedoganars" ~
    "antoetapadebesplayaredessietecortecoreadudasdeseoviejodeseaaguas" ~
    "&quot;domaincommonstatuseventsmastersystemactionbannerremovescro" ~
    "llupdateglobalmediumfilternumberchangeresultpublicscreenchooseno" ~
    "rmaltravelissuessourcetargetspringmodulemobileswitchphotosborder" ~
    "regionitselfsocialactivecolumnrecordfollowtitle>eitherlengthfami" ~
    "lyfriendlayoutauthorcreatereviewsummerserverplayedplayerexpandpo" ~
    "licyformatdoublepointsseriespersonlivingdesignmonthsforcesunique" ~
    "weightpeopleenergynaturesearchfigurehavingcustomoffsetletterwind" ~
    "owsubmitrendergroupsuploadhealthmethodvideosschoolfutureshadowde" ~
    "batevaluesObjectothersrightsleaguechromesimplenoticesharedending" ~
    "seasonreportonlinesquarebuttonimagesenablemovinglatestwinterFran" ~
    "ceperiodstrongrepeatLondondetailformeddemandsecurepassedtogglepl" ~
    "acesdevicestaticcitiesstreamyellowattackstreetflighthiddeninfo\">" ~
    "openedusefulvalleycausesleadersecretseconddamagesportsexceptrati" ~
    "ngsignedthingseffectfieldsstatesofficevisualeditorvolumeReportmu" ~
    "seummoviesparentaccessmostlymother\" id=\"marketgroundchancesurvey" ~
    "beforesymbolmomentspeechmotioninsidematterCenterobjectexistsmidd" ~
    "leEuropegrowthlegacymannerenoughcareeransweroriginportalclientse" ~
    "lectrandomclosedtopicscomingfatheroptionsimplyraisedescapechosen" ~
    "churchdefinereasoncorneroutputmemoryiframepolicemodelsNumberduri" ~
    "ngoffersstyleskilledlistedcalledsilvermargindeletebetterbrowseli" ~
    "mitsGlobalsinglewidgetcenterbudgetnowrapcreditclaimsenginesafety" ~
    "choicespirit-stylespreadmakingneededrussiapleaseextentScriptbrok" ~
    "enallowschargedividefactormember-basedtheoryconfigaroundworkedhe" ~
    "lpedChurchimpactshouldalwayslogo\" bottomlist\">){var prefixorange" ~
    "Header.push(couplegardenbridgelaunchReviewtakingvisionlittledati" ~
    "ngButtonbeautythemesforgotSearchanchoralmostloadedChangereturnst" ~
    "ringreloadMobileincomesupplySourceordersviewed&nbsp;courseAbout " ~
    "island<html cookiename=\"amazonmodernadvicein</a>: The dialoghous" ~
    "esBEGIN MexicostartscentreheightaddingIslandassetsEmpireSchoolef" ~
    "fortdirectnearlymanualSelect.\x0A\x0AOnejoinedmenu\">Philipawardshandle" ~
    "importOfficeregardskillsnationSportsdegreeweekly (e.g.behinddoct" ~
    "orloggedunited</b></beginsplantsassistartistissued300px|canadaag" ~
    "encyschemeremainBrazilsamplelogo\">beyond-scaleacceptservedmarine" ~
    "Footercamera</h1>\x0A_form\"leavesstress\" />\x0D\x0A.gif\" onloadloaderOxfo" ~
    "rdsistersurvivlistenfemaleDesignsize=\"appealtext\">levelsthankshi" ~
    "gherforcedanimalanyoneAfricaagreedrecentPeople<br />wonderprices" ~
    "turned|| {};main\">inlinesundaywrap\">failedcensusminutebeaconquot" ~
    "es150px|estateremoteemail\"linkedright;signalformal1.htmlsignuppr" ~
    "incefloat:.png\" forum.AccesspaperssoundsextendHeightsliderUTF-8\"" ~
    "&amp; Before. WithstudioownersmanageprofitjQueryannualparamsboug" ~
    "htfamousgooglelongeri++) {israelsayingdecidehome\">headerensurebr" ~
    "anchpiecesblock;statedtop\"><racingresize--&gt;pacitysexualbureau" ~
    ".jpg\" 10,000obtaintitlesamount, Inc.comedymenu\" lyricstoday.inde" ~
    "edcounty_logo.FamilylookedMarketlse ifPlayerturkey);var forestgi" ~
    "vingerrorsDomain}else{insertBlog</footerlogin.fasteragents<body " ~
    "10px 0pragmafridayjuniordollarplacedcoversplugin5,000 page\">bost" ~
    "on.test(avatartested_countforumsschemaindex,filledsharesreaderal" ~
    "ert(appearSubmitline\">body\">\x0A* TheThoughseeingjerseyNews</verify" ~
    "expertinjurywidth=CookieSTART across_imagethreadnativepocketbox\"" ~
    ">\x0ASystem DavidcancertablesprovedApril reallydriveritem\">more\">bo" ~
    "ardscolorscampusfirst || [];media.guitarfinishwidth:showedOther " ~
    ".php\" assumelayerswilsonstoresreliefswedenCustomeasily your Stri" ~
    "ng\x0A\x0AWhiltaylorclear:resortfrenchthough\") + \"<body>buyingbrandsMe" ~
    "mbername\">oppingsector5px;\">vspacepostermajor coffeemartinmature" ~
    "happen</nav>kansaslink\">Images=falsewhile hspace0&amp; \x0A\x0AIn  pow" ~
    "erPolski-colorjordanBottomStart -count2.htmlnews\">01.jpgOnline-r" ~
    "ightmillerseniorISBN 00,000 guidesvalue)ectionrepair.xml\"  right" ~
    "s.html-blockregExp:hoverwithinvirginphones</tr>\x0Dusing \x0A\x09var >');" ~
    "\x0A\x09</td>\x0A</tr>\x0Abahasabrasilgalegomagyarpolskisrpski\xD8\xB1\xD8\xAF\xD9\x88\xE4\xB8\xAD\xE6\x96\x87\xE7\xAE" ~
    "\x80\xE4\xBD\x93\xE7\xB9\x81\xE9\xAB\x94\xE4\xBF\xA1\xE6\x81\xAF\xE4\xB8\xAD\xE5\x9B\xBD\xE6\x88\x91\xE4\xBB\xAC\xE4\xB8\x80\xE4\xB8\xAA\xE5\x85\xAC\xE5\x8F\xB8\xE7\xAE\xA1\xE7\x90\x86\xE8\xAE\xBA\xE5\x9D\x9B\xE5\x8F\xAF\xE4\xBB\xA5\xE6\x9C\x8D\xE5\x8A\xA1" ~
    "\xE6\x97\xB6\xE9\x97\xB4\xE4\xB8\xAA\xE4\xBA\xBA\xE4\xBA\xA7\xE5\x93\x81\xE8\x87\xAA\xE5\xB7\xB1\xE4\xBC\x81\xE4\xB8\x9A\xE6\x9F\xA5\xE7\x9C\x8B\xE5\xB7\xA5\xE4\xBD\x9C\xE8\x81\x94\xE7\xB3\xBB\xE6\xB2\xA1\xE6\x9C\x89\xE7\xBD\x91\xE7\xAB\x99\xE6\x89\x80\xE6" ~
    "\x9C\x89\xE8\xAF\x84\xE8\xAE\xBA\xE4\xB8\xAD\xE5\xBF\x83\xE6\x96\x87\xE7\xAB\xA0\xE7\x94\xA8\xE6\x88\xB7\xE9\xA6\x96\xE9\xA1\xB5\xE4\xBD\x9C\xE8\x80\x85\xE6\x8A\x80\xE6\x9C\xAF\xE9\x97\xAE\xE9\xA2\x98\xE7\x9B\xB8\xE5\x85\xB3\xE4\xB8\x8B\xE8\xBD\xBD\xE6\x90" ~
    "\x9C\xE7\xB4\xA2\xE4\xBD\xBF\xE7\x94\xA8\xE8\xBD\xAF\xE4\xBB\xB6\xE5\x9C\xA8\xE7\xBA\xBF\xE4\xB8\xBB\xE9\xA2\x98\xE8\xB5\x84\xE6\x96\x99\xE8\xA7\x86\xE9\xA2\x91\xE5\x9B\x9E\xE5\xA4\x8D\xE6\xB3\xA8\xE5\x86\x8C\xE7\xBD\x91\xE7\xBB\x9C\xE6\x94\xB6\xE8\x97\x8F" ~
    "\xE5\x86\x85\xE5\xAE\xB9\xE6\x8E\xA8\xE8\x8D\x90\xE5\xB8\x82\xE5\x9C\xBA\xE6\xB6\x88\xE6\x81\xAF\xE7\xA9\xBA\xE9\x97\xB4\xE5\x8F\x91\xE5\xB8\x83\xE4\xBB\x80\xE4\xB9\x88\xE5\xA5\xBD\xE5\x8F\x8B\xE7\x94\x9F\xE6\xB4\xBB\xE5\x9B\xBE\xE7\x89\x87\xE5\x8F\x91\xE5" ~
    "\xB1\x95\xE5\xA6\x82\xE6\x9E\x9C\xE6\x89\x8B\xE6\x9C\xBA\xE6\x96\xB0\xE9\x97\xBB\xE6\x9C\x80\xE6\x96\xB0\xE6\x96\xB9\xE5\xBC\x8F\xE5\x8C\x97\xE4\xBA\xAC\xE6\x8F\x90\xE4\xBE\x9B\xE5\x85\xB3\xE4\xBA\x8E\xE6\x9B\xB4\xE5\xA4\x9A\xE8\xBF\x99\xE4\xB8\xAA\xE7\xB3" ~
    "\xBB\xE7\xBB\x9F\xE7\x9F\xA5\xE9\x81\x93\xE6\xB8\xB8\xE6\x88\x8F\xE5\xB9\xBF\xE5\x91\x8A\xE5\x85\xB6\xE4\xBB\x96\xE5\x8F\x91\xE8\xA1\xA8\xE5\xAE\x89\xE5\x85\xA8\xE7\xAC\xAC\xE4\xB8\x80\xE4\xBC\x9A\xE5\x91\x98\xE8\xBF\x9B\xE8\xA1\x8C\xE7\x82\xB9\xE5\x87\xBB" ~
    "\xE7\x89\x88\xE6\x9D\x83\xE7\x94\xB5\xE5\xAD\x90\xE4\xB8\x96\xE7\x95\x8C\xE8\xAE\xBE\xE8\xAE\xA1\xE5\x85\x8D\xE8\xB4\xB9\xE6\x95\x99\xE8\x82\xB2\xE5\x8A\xA0\xE5\x85\xA5\xE6\xB4\xBB\xE5\x8A\xA8\xE4\xBB\x96\xE4\xBB\xAC\xE5\x95\x86\xE5\x93\x81\xE5\x8D\x9A\xE5" ~
    "\xAE\xA2\xE7\x8E\xB0\xE5\x9C\xA8\xE4\xB8\x8A\xE6\xB5\xB7\xE5\xA6\x82\xE4\xBD\x95\xE5\xB7\xB2\xE7\xBB\x8F\xE7\x95\x99\xE8\xA8\x80\xE8\xAF\xA6\xE7\xBB\x86\xE7\xA4\xBE\xE5\x8C\xBA\xE7\x99\xBB\xE5\xBD\x95\xE6\x9C\xAC\xE7\xAB\x99\xE9\x9C\x80\xE8\xA6\x81\xE4\xBB" ~
    "\xB7\xE6\xA0\xBC\xE6\x94\xAF\xE6\x8C\x81\xE5\x9B\xBD\xE9\x99\x85\xE9\x93\xBE\xE6\x8E\xA5\xE5\x9B\xBD\xE5\xAE\xB6\xE5\xBB\xBA\xE8\xAE\xBE\xE6\x9C\x8B\xE5\x8F\x8B\xE9\x98\x85\xE8\xAF\xBB\xE6\xB3\x95\xE5\xBE\x8B\xE4\xBD\x8D\xE7\xBD\xAE\xE7\xBB\x8F\xE6\xB5\x8E" ~
    "\xE9\x80\x89\xE6\x8B\xA9\xE8\xBF\x99\xE6\xA0\xB7\xE5\xBD\x93\xE5\x89\x8D\xE5\x88\x86\xE7\xB1\xBB\xE6\x8E\x92\xE8\xA1\x8C\xE5\x9B\xA0\xE4\xB8\xBA\xE4\xBA\xA4\xE6\x98\x93\xE6\x9C\x80\xE5\x90\x8E\xE9\x9F\xB3\xE4\xB9\x90\xE4\xB8\x8D\xE8\x83\xBD\xE9\x80\x9A\xE8" ~
    "\xBF\x87\xE8\xA1\x8C\xE4\xB8\x9A\xE7\xA7\x91\xE6\x8A\x80\xE5\x8F\xAF\xE8\x83\xBD\xE8\xAE\xBE\xE5\xA4\x87\xE5\x90\x88\xE4\xBD\x9C\xE5\xA4\xA7\xE5\xAE\xB6\xE7\xA4\xBE\xE4\xBC\x9A\xE7\xA0\x94\xE7\xA9\xB6\xE4\xB8\x93\xE4\xB8\x9A\xE5\x85\xA8\xE9\x83\xA8\xE9\xA1" ~
    "\xB9\xE7\x9B\xAE\xE8\xBF\x99\xE9\x87\x8C\xE8\xBF\x98\xE6\x98\xAF\xE5\xBC\x80\xE5\xA7\x8B\xE6\x83\x85\xE5\x86\xB5\xE7\x94\xB5\xE8\x84\x91\xE6\x96\x87\xE4\xBB\xB6\xE5\x93\x81\xE7\x89\x8C\xE5\xB8\xAE\xE5\x8A\xA9\xE6\x96\x87\xE5\x8C\x96\xE8\xB5\x84\xE6\xBA\x90" ~
    "\xE5\xA4\xA7\xE5\xAD\xA6\xE5\xAD\xA6\xE4\xB9\xA0\xE5\x9C\xB0\xE5\x9D\x80\xE6\xB5\x8F\xE8\xA7\x88\xE6\x8A\x95\xE8\xB5\x84\xE5\xB7\xA5\xE7\xA8\x8B\xE8\xA6\x81\xE6\xB1\x82\xE6\x80\x8E\xE4\xB9\x88\xE6\x97\xB6\xE5\x80\x99\xE5\x8A\x9F\xE8\x83\xBD\xE4\xB8\xBB\xE8" ~
    "\xA6\x81\xE7\x9B\xAE\xE5\x89\x8D\xE8\xB5\x84\xE8\xAE\xAF\xE5\x9F\x8E\xE5\xB8\x82\xE6\x96\xB9\xE6\xB3\x95\xE7\x94\xB5\xE5\xBD\xB1\xE6\x8B\x9B\xE8\x81\x98\xE5\xA3\xB0\xE6\x98\x8E\xE4\xBB\xBB\xE4\xBD\x95\xE5\x81\xA5\xE5\xBA\xB7\xE6\x95\xB0\xE6\x8D\xAE\xE7\xBE" ~
    "\x8E\xE5\x9B\xBD\xE6\xB1\xBD\xE8\xBD\xA6\xE4\xBB\x8B\xE7\xBB\x8D\xE4\xBD\x86\xE6\x98\xAF\xE4\xBA\xA4\xE6\xB5\x81\xE7\x94\x9F\xE4\xBA\xA7\xE6\x89\x80\xE4\xBB\xA5\xE7\x94\xB5\xE8\xAF\x9D\xE6\x98\xBE\xE7\xA4\xBA\xE4\xB8\x80\xE4\xBA\x9B\xE5\x8D\x95\xE4\xBD\x8D" ~
    "\xE4\xBA\xBA\xE5\x91\x98\xE5\x88\x86\xE6\x9E\x90\xE5\x9C\xB0\xE5\x9B\xBE\xE6\x97\x85\xE6\xB8\xB8\xE5\xB7\xA5\xE5\x85\xB7\xE5\xAD\xA6\xE7\x94\x9F\xE7\xB3\xBB\xE5\x88\x97\xE7\xBD\x91\xE5\x8F\x8B\xE5\xB8\x96\xE5\xAD\x90\xE5\xAF\x86\xE7\xA0\x81\xE9\xA2\x91\xE9" ~
    "\x81\x93\xE6\x8E\xA7\xE5\x88\xB6\xE5\x9C\xB0\xE5\x8C\xBA\xE5\x9F\xBA\xE6\x9C\xAC\xE5\x85\xA8\xE5\x9B\xBD\xE7\xBD\x91\xE4\xB8\x8A\xE9\x87\x8D\xE8\xA6\x81\xE7\xAC\xAC\xE4\xBA\x8C\xE5\x96\x9C\xE6\xAC\xA2\xE8\xBF\x9B\xE5\x85\xA5\xE5\x8F\x8B\xE6\x83\x85\xE8\xBF" ~
    "\x99\xE4\xBA\x9B\xE8\x80\x83\xE8\xAF\x95\xE5\x8F\x91\xE7\x8E\xB0\xE5\x9F\xB9\xE8\xAE\xAD\xE4\xBB\xA5\xE4\xB8\x8A\xE6\x94\xBF\xE5\xBA\x9C\xE6\x88\x90\xE4\xB8\xBA\xE7\x8E\xAF\xE5\xA2\x83\xE9\xA6\x99\xE6\xB8\xAF\xE5\x90\x8C\xE6\x97\xB6\xE5\xA8\xB1\xE4\xB9\x90" ~
    "\xE5\x8F\x91\xE9\x80\x81\xE4\xB8\x80\xE5\xAE\x9A\xE5\xBC\x80\xE5\x8F\x91\xE4\xBD\x9C\xE5\x93\x81\xE6\xA0\x87\xE5\x87\x86\xE6\xAC\xA2\xE8\xBF\x8E\xE8\xA7\xA3\xE5\x86\xB3\xE5\x9C\xB0\xE6\x96\xB9\xE4\xB8\x80\xE4\xB8\x8B\xE4\xBB\xA5\xE5\x8F\x8A\xE8\xB4\xA3\xE4" ~
    "\xBB\xBB\xE6\x88\x96\xE8\x80\x85\xE5\xAE\xA2\xE6\x88\xB7\xE4\xBB\xA3\xE8\xA1\xA8\xE7\xA7\xAF\xE5\x88\x86\xE5\xA5\xB3\xE4\xBA\xBA\xE6\x95\xB0\xE7\xA0\x81\xE9\x94\x80\xE5\x94\xAE\xE5\x87\xBA\xE7\x8E\xB0\xE7\xA6\xBB\xE7\xBA\xBF\xE5\xBA\x94\xE7\x94\xA8\xE5\x88" ~
    "\x97\xE8\xA1\xA8\xE4\xB8\x8D\xE5\x90\x8C\xE7\xBC\x96\xE8\xBE\x91\xE7\xBB\x9F\xE8\xAE\xA1\xE6\x9F\xA5\xE8\xAF\xA2\xE4\xB8\x8D\xE8\xA6\x81\xE6\x9C\x89\xE5\x85\xB3\xE6\x9C\xBA\xE6\x9E\x84\xE5\xBE\x88\xE5\xA4\x9A\xE6\x92\xAD\xE6\x94\xBE\xE7\xBB\x84\xE7\xBB\x87" ~
    "\xE6\x94\xBF\xE7\xAD\x96\xE7\x9B\xB4\xE6\x8E\xA5\xE8\x83\xBD\xE5\x8A\x9B\xE6\x9D\xA5\xE6\xBA\x90\xE6\x99\x82\xE9\x96\x93\xE7\x9C\x8B\xE5\x88\xB0\xE7\x83\xAD\xE9\x97\xA8\xE5\x85\xB3\xE9\x94\xAE\xE4\xB8\x93\xE5\x8C\xBA\xE9\x9D\x9E\xE5\xB8\xB8\xE8\x8B\xB1\xE8" ~
    "\xAF\xAD\xE7\x99\xBE\xE5\xBA\xA6\xE5\xB8\x8C\xE6\x9C\x9B\xE7\xBE\x8E\xE5\xA5\xB3\xE6\xAF\x94\xE8\xBE\x83\xE7\x9F\xA5\xE8\xAF\x86\xE8\xA7\x84\xE5\xAE\x9A\xE5\xBB\xBA\xE8\xAE\xAE\xE9\x83\xA8\xE9\x97\xA8\xE6\x84\x8F\xE8\xA7\x81\xE7\xB2\xBE\xE5\xBD\xA9\xE6\x97" ~
    "\xA5\xE6\x9C\xAC\xE6\x8F\x90\xE9\xAB\x98\xE5\x8F\x91\xE8\xA8\x80\xE6\x96\xB9\xE9\x9D\xA2\xE5\x9F\xBA\xE9\x87\x91\xE5\xA4\x84\xE7\x90\x86\xE6\x9D\x83\xE9\x99\x90\xE5\xBD\xB1\xE7\x89\x87\xE9\x93\xB6\xE8\xA1\x8C\xE8\xBF\x98\xE6\x9C\x89\xE5\x88\x86\xE4\xBA\xAB" ~
    "\xE7\x89\xA9\xE5\x93\x81\xE7\xBB\x8F\xE8\x90\xA5\xE6\xB7\xBB\xE5\x8A\xA0\xE4\xB8\x93\xE5\xAE\xB6\xE8\xBF\x99\xE7\xA7\x8D\xE8\xAF\x9D\xE9\xA2\x98\xE8\xB5\xB7\xE6\x9D\xA5\xE4\xB8\x9A\xE5\x8A\xA1\xE5\x85\xAC\xE5\x91\x8A\xE8\xAE\xB0\xE5\xBD\x95\xE7\xAE\x80\xE4" ~
    "\xBB\x8B\xE8\xB4\xA8\xE9\x87\x8F\xE7\x94\xB7\xE4\xBA\xBA\xE5\xBD\xB1\xE5\x93\x8D\xE5\xBC\x95\xE7\x94\xA8\xE6\x8A\xA5\xE5\x91\x8A\xE9\x83\xA8\xE5\x88\x86\xE5\xBF\xAB\xE9\x80\x9F\xE5\x92\xA8\xE8\xAF\xA2\xE6\x97\xB6\xE5\xB0\x9A\xE6\xB3\xA8\xE6\x84\x8F\xE7\x94" ~
    "\xB3\xE8\xAF\xB7\xE5\xAD\xA6\xE6\xA0\xA1\xE5\xBA\x94\xE8\xAF\xA5\xE5\x8E\x86\xE5\x8F\xB2\xE5\x8F\xAA\xE6\x98\xAF\xE8\xBF\x94\xE5\x9B\x9E\xE8\xB4\xAD\xE4\xB9\xB0\xE5\x90\x8D\xE7\xA7\xB0\xE4\xB8\xBA\xE4\xBA\x86\xE6\x88\x90\xE5\x8A\x9F\xE8\xAF\xB4\xE6\x98\x8E" ~
    "\xE4\xBE\x9B\xE5\xBA\x94\xE5\xAD\xA9\xE5\xAD\x90\xE4\xB8\x93\xE9\xA2\x98\xE7\xA8\x8B\xE5\xBA\x8F\xE4\xB8\x80\xE8\x88\xAC\xE6\x9C\x83\xE5\x93\xA1\xE5\x8F\xAA\xE6\x9C\x89\xE5\x85\xB6\xE5\xAE\x83\xE4\xBF\x9D\xE6\x8A\xA4\xE8\x80\x8C\xE4\xB8\x94\xE4\xBB\x8A\xE5" ~
    "\xA4\xA9\xE7\xAA\x97\xE5\x8F\xA3\xE5\x8A\xA8\xE6\x80\x81\xE7\x8A\xB6\xE6\x80\x81\xE7\x89\xB9\xE5\x88\xAB\xE8\xAE\xA4\xE4\xB8\xBA\xE5\xBF\x85\xE9\xA1\xBB\xE6\x9B\xB4\xE6\x96\xB0\xE5\xB0\x8F\xE8\xAF\xB4\xE6\x88\x91\xE5\x80\x91\xE4\xBD\x9C\xE4\xB8\xBA\xE5\xAA" ~
    "\x92\xE4\xBD\x93\xE5\x8C\x85\xE6\x8B\xAC\xE9\x82\xA3\xE4\xB9\x88\xE4\xB8\x80\xE6\xA0\xB7\xE5\x9B\xBD\xE5\x86\x85\xE6\x98\xAF\xE5\x90\xA6\xE6\xA0\xB9\xE6\x8D\xAE\xE7\x94\xB5\xE8\xA7\x86\xE5\xAD\xA6\xE9\x99\xA2\xE5\x85\xB7\xE6\x9C\x89\xE8\xBF\x87\xE7\xA8\x8B" ~
    "\xE7\x94\xB1\xE4\xBA\x8E\xE4\xBA\xBA\xE6\x89\x8D\xE5\x87\xBA\xE6\x9D\xA5\xE4\xB8\x8D\xE8\xBF\x87\xE6\xAD\xA3\xE5\x9C\xA8\xE6\x98\x8E\xE6\x98\x9F\xE6\x95\x85\xE4\xBA\x8B\xE5\x85\xB3\xE7\xB3\xBB\xE6\xA0\x87\xE9\xA2\x98\xE5\x95\x86\xE5\x8A\xA1\xE8\xBE\x93\xE5" ~
    "\x85\xA5\xE4\xB8\x80\xE7\x9B\xB4\xE5\x9F\xBA\xE7\xA1\x80\xE6\x95\x99\xE5\xAD\xA6\xE4\xBA\x86\xE8\xA7\xA3\xE5\xBB\xBA\xE7\xAD\x91\xE7\xBB\x93\xE6\x9E\x9C\xE5\x85\xA8\xE7\x90\x83\xE9\x80\x9A\xE7\x9F\xA5\xE8\xAE\xA1\xE5\x88\x92\xE5\xAF\xB9\xE4\xBA\x8E\xE8\x89" ~
    "\xBA\xE6\x9C\xAF\xE7\x9B\xB8\xE5\x86\x8C\xE5\x8F\x91\xE7\x94\x9F\xE7\x9C\x9F\xE7\x9A\x84\xE5\xBB\xBA\xE7\xAB\x8B\xE7\xAD\x89\xE7\xBA\xA7\xE7\xB1\xBB\xE5\x9E\x8B\xE7\xBB\x8F\xE9\xAA\x8C\xE5\xAE\x9E\xE7\x8E\xB0\xE5\x88\xB6\xE4\xBD\x9C\xE6\x9D\xA5\xE8\x87\xAA" ~
    "\xE6\xA0\x87\xE7\xAD\xBE\xE4\xBB\xA5\xE4\xB8\x8B\xE5\x8E\x9F\xE5\x88\x9B\xE6\x97\xA0\xE6\xB3\x95\xE5\x85\xB6\xE4\xB8\xAD\xE5\x80\x8B\xE4\xBA\xBA\xE4\xB8\x80\xE5\x88\x87\xE6\x8C\x87\xE5\x8D\x97\xE5\x85\xB3\xE9\x97\xAD\xE9\x9B\x86\xE5\x9B\xA2\xE7\xAC\xAC\xE4" ~
    "\xB8\x89\xE5\x85\xB3\xE6\xB3\xA8\xE5\x9B\xA0\xE6\xAD\xA4\xE7\x85\xA7\xE7\x89\x87\xE6\xB7\xB1\xE5\x9C\xB3\xE5\x95\x86\xE4\xB8\x9A\xE5\xB9\xBF\xE5\xB7\x9E\xE6\x97\xA5\xE6\x9C\x9F\xE9\xAB\x98\xE7\xBA\xA7\xE6\x9C\x80\xE8\xBF\x91\xE7\xBB\xBC\xE5\x90\x88\xE8\xA1" ~
    "\xA8\xE7\xA4\xBA\xE4\xB8\x93\xE8\xBE\x91\xE8\xA1\x8C\xE4\xB8\xBA\xE4\xBA\xA4\xE9\x80\x9A\xE8\xAF\x84\xE4\xBB\xB7\xE8\xA7\x89\xE5\xBE\x97\xE7\xB2\xBE\xE5\x8D\x8E\xE5\xAE\xB6\xE5\xBA\xAD\xE5\xAE\x8C\xE6\x88\x90\xE6\x84\x9F\xE8\xA7\x89\xE5\xAE\x89\xE8\xA3\x85" ~
    "\xE5\xBE\x97\xE5\x88\xB0\xE9\x82\xAE\xE4\xBB\xB6\xE5\x88\xB6\xE5\xBA\xA6\xE9\xA3\x9F\xE5\x93\x81\xE8\x99\xBD\xE7\x84\xB6\xE8\xBD\xAC\xE8\xBD\xBD\xE6\x8A\xA5\xE4\xBB\xB7\xE8\xAE\xB0\xE8\x80\x85\xE6\x96\xB9\xE6\xA1\x88\xE8\xA1\x8C\xE6\x94\xBF\xE4\xBA\xBA\xE6" ~
    "\xB0\x91\xE7\x94\xA8\xE5\x93\x81\xE4\xB8\x9C\xE8\xA5\xBF\xE6\x8F\x90\xE5\x87\xBA\xE9\x85\x92\xE5\xBA\x97\xE7\x84\xB6\xE5\x90\x8E\xE4\xBB\x98\xE6\xAC\xBE\xE7\x83\xAD\xE7\x82\xB9\xE4\xBB\xA5\xE5\x89\x8D\xE5\xAE\x8C\xE5\x85\xA8\xE5\x8F\x91\xE5\xB8\x96\xE8\xAE" ~
    "\xBE\xE7\xBD\xAE\xE9\xA2\x86\xE5\xAF\xBC\xE5\xB7\xA5\xE4\xB8\x9A\xE5\x8C\xBB\xE9\x99\xA2\xE7\x9C\x8B\xE7\x9C\x8B\xE7\xBB\x8F\xE5\x85\xB8\xE5\x8E\x9F\xE5\x9B\xA0\xE5\xB9\xB3\xE5\x8F\xB0\xE5\x90\x84\xE7\xA7\x8D\xE5\xA2\x9E\xE5\x8A\xA0\xE6\x9D\x90\xE6\x96\x99" ~
    "\xE6\x96\xB0\xE5\xA2\x9E\xE4\xB9\x8B\xE5\x90\x8E\xE8\x81\x8C\xE4\xB8\x9A\xE6\x95\x88\xE6\x9E\x9C\xE4\xBB\x8A\xE5\xB9\xB4\xE8\xAE\xBA\xE6\x96\x87\xE6\x88\x91\xE5\x9B\xBD\xE5\x91\x8A\xE8\xAF\x89\xE7\x89\x88\xE4\xB8\xBB\xE4\xBF\xAE\xE6\x94\xB9\xE5\x8F\x82\xE4" ~
    "\xB8\x8E\xE6\x89\x93\xE5\x8D\xB0\xE5\xBF\xAB\xE4\xB9\x90\xE6\x9C\xBA\xE6\xA2\xB0\xE8\xA7\x82\xE7\x82\xB9\xE5\xAD\x98\xE5\x9C\xA8\xE7\xB2\xBE\xE7\xA5\x9E\xE8\x8E\xB7\xE5\xBE\x97\xE5\x88\xA9\xE7\x94\xA8\xE7\xBB\xA7\xE7\xBB\xAD\xE4\xBD\xA0\xE4\xBB\xAC\xE8\xBF" ~
    "\x99\xE4\xB9\x88\xE6\xA8\xA1\xE5\xBC\x8F\xE8\xAF\xAD\xE8\xA8\x80\xE8\x83\xBD\xE5\xA4\x9F\xE9\x9B\x85\xE8\x99\x8E\xE6\x93\x8D\xE4\xBD\x9C\xE9\xA3\x8E\xE6\xA0\xBC\xE4\xB8\x80\xE8\xB5\xB7\xE7\xA7\x91\xE5\xAD\xA6\xE4\xBD\x93\xE8\x82\xB2\xE7\x9F\xAD\xE4\xBF\xA1" ~
    "\xE6\x9D\xA1\xE4\xBB\xB6\xE6\xB2\xBB\xE7\x96\x97\xE8\xBF\x90\xE5\x8A\xA8\xE4\xBA\xA7\xE4\xB8\x9A\xE4\xBC\x9A\xE8\xAE\xAE\xE5\xAF\xBC\xE8\x88\xAA\xE5\x85\x88\xE7\x94\x9F\xE8\x81\x94\xE7\x9B\x9F\xE5\x8F\xAF\xE6\x98\xAF\xE5\x95\x8F\xE9\xA1\x8C\xE7\xBB\x93\xE6" ~
    "\x9E\x84\xE4\xBD\x9C\xE7\x94\xA8\xE8\xB0\x83\xE6\x9F\xA5\xE8\xB3\x87\xE6\x96\x99\xE8\x87\xAA\xE5\x8A\xA8\xE8\xB4\x9F\xE8\xB4\xA3\xE5\x86\x9C\xE4\xB8\x9A\xE8\xAE\xBF\xE9\x97\xAE\xE5\xAE\x9E\xE6\x96\xBD\xE6\x8E\xA5\xE5\x8F\x97\xE8\xAE\xA8\xE8\xAE\xBA\xE9\x82" ~
    "\xA3\xE4\xB8\xAA\xE5\x8F\x8D\xE9\xA6\x88\xE5\x8A\xA0\xE5\xBC\xBA\xE5\xA5\xB3\xE6\x80\xA7\xE8\x8C\x83\xE5\x9B\xB4\xE6\x9C\x8D\xE5\x8B\x99\xE4\xBC\x91\xE9\x97\xB2\xE4\xBB\x8A\xE6\x97\xA5\xE5\xAE\xA2\xE6\x9C\x8D\xE8\xA7\x80\xE7\x9C\x8B\xE5\x8F\x82\xE5\x8A\xA0" ~
    "\xE7\x9A\x84\xE8\xAF\x9D\xE4\xB8\x80\xE7\x82\xB9\xE4\xBF\x9D\xE8\xAF\x81\xE5\x9B\xBE\xE4\xB9\xA6\xE6\x9C\x89\xE6\x95\x88\xE6\xB5\x8B\xE8\xAF\x95\xE7\xA7\xBB\xE5\x8A\xA8\xE6\x89\x8D\xE8\x83\xBD\xE5\x86\xB3\xE5\xAE\x9A\xE8\x82\xA1\xE7\xA5\xA8\xE4\xB8\x8D\xE6" ~
    "\x96\xAD\xE9\x9C\x80\xE6\xB1\x82\xE4\xB8\x8D\xE5\xBE\x97\xE5\x8A\x9E\xE6\xB3\x95\xE4\xB9\x8B\xE9\x97\xB4\xE9\x87\x87\xE7\x94\xA8\xE8\x90\xA5\xE9\x94\x80\xE6\x8A\x95\xE8\xAF\x89\xE7\x9B\xAE\xE6\xA0\x87\xE7\x88\xB1\xE6\x83\x85\xE6\x91\x84\xE5\xBD\xB1\xE6\x9C" ~
    "\x89\xE4\xBA\x9B\xE8\xA4\x87\xE8\xA3\xBD\xE6\x96\x87\xE5\xAD\xA6\xE6\x9C\xBA\xE4\xBC\x9A\xE6\x95\xB0\xE5\xAD\x97\xE8\xA3\x85\xE4\xBF\xAE\xE8\xB4\xAD\xE7\x89\xA9\xE5\x86\x9C\xE6\x9D\x91\xE5\x85\xA8\xE9\x9D\xA2\xE7\xB2\xBE\xE5\x93\x81\xE5\x85\xB6\xE5\xAE\x9E" ~
    "\xE4\xBA\x8B\xE6\x83\x85\xE6\xB0\xB4\xE5\xB9\xB3\xE6\x8F\x90\xE7\xA4\xBA\xE4\xB8\x8A\xE5\xB8\x82\xE8\xB0\xA2\xE8\xB0\xA2\xE6\x99\xAE\xE9\x80\x9A\xE6\x95\x99\xE5\xB8\x88\xE4\xB8\x8A\xE4\xBC\xA0\xE7\xB1\xBB\xE5\x88\xAB\xE6\xAD\x8C\xE6\x9B\xB2\xE6\x8B\xA5\xE6" ~
    "\x9C\x89\xE5\x88\x9B\xE6\x96\xB0\xE9\x85\x8D\xE4\xBB\xB6\xE5\x8F\xAA\xE8\xA6\x81\xE6\x97\xB6\xE4\xBB\xA3\xE8\xB3\x87\xE8\xA8\x8A\xE8\xBE\xBE\xE5\x88\xB0\xE4\xBA\xBA\xE7\x94\x9F\xE8\xAE\xA2\xE9\x98\x85\xE8\x80\x81\xE5\xB8\x88\xE5\xB1\x95\xE7\xA4\xBA\xE5\xBF" ~
    "\x83\xE7\x90\x86\xE8\xB4\xB4\xE5\xAD\x90\xE7\xB6\xB2\xE7\xAB\x99\xE4\xB8\xBB\xE9\xA1\x8C\xE8\x87\xAA\xE7\x84\xB6\xE7\xBA\xA7\xE5\x88\xAB\xE7\xAE\x80\xE5\x8D\x95\xE6\x94\xB9\xE9\x9D\xA9\xE9\x82\xA3\xE4\xBA\x9B\xE6\x9D\xA5\xE8\xAF\xB4\xE6\x89\x93\xE5\xBC\x80" ~
    "\xE4\xBB\xA3\xE7\xA0\x81\xE5\x88\xA0\xE9\x99\xA4\xE8\xAF\x81\xE5\x88\xB8\xE8\x8A\x82\xE7\x9B\xAE\xE9\x87\x8D\xE7\x82\xB9\xE6\xAC\xA1\xE6\x95\xB8\xE5\xA4\x9A\xE5\xB0\x91\xE8\xA7\x84\xE5\x88\x92\xE8\xB5\x84\xE9\x87\x91\xE6\x89\xBE\xE5\x88\xB0\xE4\xBB\xA5\xE5" ~
    "\x90\x8E\xE5\xA4\xA7\xE5\x85\xA8\xE4\xB8\xBB\xE9\xA1\xB5\xE6\x9C\x80\xE4\xBD\xB3\xE5\x9B\x9E\xE7\xAD\x94\xE5\xA4\xA9\xE4\xB8\x8B\xE4\xBF\x9D\xE9\x9A\x9C\xE7\x8E\xB0\xE4\xBB\xA3\xE6\xA3\x80\xE6\x9F\xA5\xE6\x8A\x95\xE7\xA5\xA8\xE5\xB0\x8F\xE6\x97\xB6\xE6\xB2" ~
    "\x92\xE6\x9C\x89\xE6\xAD\xA3\xE5\xB8\xB8\xE7\x94\x9A\xE8\x87\xB3\xE4\xBB\xA3\xE7\x90\x86\xE7\x9B\xAE\xE5\xBD\x95\xE5\x85\xAC\xE5\xBC\x80\xE5\xA4\x8D\xE5\x88\xB6\xE9\x87\x91\xE8\x9E\x8D\xE5\xB9\xB8\xE7\xA6\x8F\xE7\x89\x88\xE6\x9C\xAC\xE5\xBD\xA2\xE6\x88\x90" ~
    "\xE5\x87\x86\xE5\xA4\x87\xE8\xA1\x8C\xE6\x83\x85\xE5\x9B\x9E\xE5\x88\xB0\xE6\x80\x9D\xE6\x83\xB3\xE6\x80\x8E\xE6\xA0\xB7\xE5\x8D\x8F\xE8\xAE\xAE\xE8\xAE\xA4\xE8\xAF\x81\xE6\x9C\x80\xE5\xA5\xBD\xE4\xBA\xA7\xE7\x94\x9F\xE6\x8C\x89\xE7\x85\xA7\xE6\x9C\x8D\xE8" ~
    "\xA3\x85\xE5\xB9\xBF\xE4\xB8\x9C\xE5\x8A\xA8\xE6\xBC\xAB\xE9\x87\x87\xE8\xB4\xAD\xE6\x96\xB0\xE6\x89\x8B\xE7\xBB\x84\xE5\x9B\xBE\xE9\x9D\xA2\xE6\x9D\xBF\xE5\x8F\x82\xE8\x80\x83\xE6\x94\xBF\xE6\xB2\xBB\xE5\xAE\xB9\xE6\x98\x93\xE5\xA4\xA9\xE5\x9C\xB0\xE5\x8A" ~
    "\xAA\xE5\x8A\x9B\xE4\xBA\xBA\xE4\xBB\xAC\xE5\x8D\x87\xE7\xBA\xA7\xE9\x80\x9F\xE5\xBA\xA6\xE4\xBA\xBA\xE7\x89\xA9\xE8\xB0\x83\xE6\x95\xB4\xE6\xB5\x81\xE8\xA1\x8C\xE9\x80\xA0\xE6\x88\x90\xE6\x96\x87\xE5\xAD\x97\xE9\x9F\xA9\xE5\x9B\xBD\xE8\xB4\xB8\xE6\x98\x93" ~
    "\xE5\xBC\x80\xE5\xB1\x95\xE7\x9B\xB8\xE9\x97\x9C\xE8\xA1\xA8\xE7\x8E\xB0\xE5\xBD\xB1\xE8\xA7\x86\xE5\xA6\x82\xE6\xAD\xA4\xE7\xBE\x8E\xE5\xAE\xB9\xE5\xA4\xA7\xE5\xB0\x8F\xE6\x8A\xA5\xE9\x81\x93\xE6\x9D\xA1\xE6\xAC\xBE\xE5\xBF\x83\xE6\x83\x85\xE8\xAE\xB8\xE5" ~
    "\xA4\x9A\xE6\xB3\x95\xE8\xA7\x84\xE5\xAE\xB6\xE5\xB1\x85\xE4\xB9\xA6\xE5\xBA\x97\xE8\xBF\x9E\xE6\x8E\xA5\xE7\xAB\x8B\xE5\x8D\xB3\xE4\xB8\xBE\xE6\x8A\xA5\xE6\x8A\x80\xE5\xB7\xA7\xE5\xA5\xA5\xE8\xBF\x90\xE7\x99\xBB\xE5\x85\xA5\xE4\xBB\xA5\xE6\x9D\xA5\xE7\x90" ~
    "\x86\xE8\xAE\xBA\xE4\xBA\x8B\xE4\xBB\xB6\xE8\x87\xAA\xE7\x94\xB1\xE4\xB8\xAD\xE5\x8D\x8E\xE5\x8A\x9E\xE5\x85\xAC\xE5\xA6\x88\xE5\xA6\x88\xE7\x9C\x9F\xE6\xAD\xA3\xE4\xB8\x8D\xE9\x94\x99\xE5\x85\xA8\xE6\x96\x87\xE5\x90\x88\xE5\x90\x8C\xE4\xBB\xB7\xE5\x80\xBC" ~
    "\xE5\x88\xAB\xE4\xBA\xBA\xE7\x9B\x91\xE7\x9D\xA3\xE5\x85\xB7\xE4\xBD\x93\xE4\xB8\x96\xE7\xBA\xAA\xE5\x9B\xA2\xE9\x98\x9F\xE5\x88\x9B\xE4\xB8\x9A\xE6\x89\xBF\xE6\x8B\x85\xE5\xA2\x9E\xE9\x95\xBF\xE6\x9C\x89\xE4\xBA\xBA\xE4\xBF\x9D\xE6\x8C\x81\xE5\x95\x86\xE5" ~
    "\xAE\xB6\xE7\xBB\xB4\xE4\xBF\xAE\xE5\x8F\xB0\xE6\xB9\xBE\xE5\xB7\xA6\xE5\x8F\xB3\xE8\x82\xA1\xE4\xBB\xBD\xE7\xAD\x94\xE6\xA1\x88\xE5\xAE\x9E\xE9\x99\x85\xE7\x94\xB5\xE4\xBF\xA1\xE7\xBB\x8F\xE7\x90\x86\xE7\x94\x9F\xE5\x91\xBD\xE5\xAE\xA3\xE4\xBC\xA0\xE4\xBB" ~
    "\xBB\xE5\x8A\xA1\xE6\xAD\xA3\xE5\xBC\x8F\xE7\x89\xB9\xE8\x89\xB2\xE4\xB8\x8B\xE6\x9D\xA5\xE5\x8D\x8F\xE4\xBC\x9A\xE5\x8F\xAA\xE8\x83\xBD\xE5\xBD\x93\xE7\x84\xB6\xE9\x87\x8D\xE6\x96\xB0\xE5\x85\xA7\xE5\xAE\xB9\xE6\x8C\x87\xE5\xAF\xBC\xE8\xBF\x90\xE8\xA1\x8C" ~
    "\xE6\x97\xA5\xE5\xBF\x97\xE8\xB3\xA3\xE5\xAE\xB6\xE8\xB6\x85\xE8\xBF\x87\xE5\x9C\x9F\xE5\x9C\xB0\xE6\xB5\x99\xE6\xB1\x9F\xE6\x94\xAF\xE4\xBB\x98\xE6\x8E\xA8\xE5\x87\xBA\xE7\xAB\x99\xE9\x95\xBF\xE6\x9D\xAD\xE5\xB7\x9E\xE6\x89\xA7\xE8\xA1\x8C\xE5\x88\xB6\xE9" ~
    "\x80\xA0\xE4\xB9\x8B\xE4\xB8\x80\xE6\x8E\xA8\xE5\xB9\xBF\xE7\x8E\xB0\xE5\x9C\xBA\xE6\x8F\x8F\xE8\xBF\xB0\xE5\x8F\x98\xE5\x8C\x96\xE4\xBC\xA0\xE7\xBB\x9F\xE6\xAD\x8C\xE6\x89\x8B\xE4\xBF\x9D\xE9\x99\xA9\xE8\xAF\xBE\xE7\xA8\x8B\xE5\x8C\xBB\xE7\x96\x97\xE7\xBB" ~
    "\x8F\xE8\xBF\x87\xE8\xBF\x87\xE5\x8E\xBB\xE4\xB9\x8B\xE5\x89\x8D\xE6\x94\xB6\xE5\x85\xA5\xE5\xB9\xB4\xE5\xBA\xA6\xE6\x9D\x82\xE5\xBF\x97\xE7\xBE\x8E\xE4\xB8\xBD\xE6\x9C\x80\xE9\xAB\x98\xE7\x99\xBB\xE9\x99\x86\xE6\x9C\xAA\xE6\x9D\xA5\xE5\x8A\xA0\xE5\xB7\xA5" ~
    "\xE5\x85\x8D\xE8\xB4\xA3\xE6\x95\x99\xE7\xA8\x8B\xE7\x89\x88\xE5\x9D\x97\xE8\xBA\xAB\xE4\xBD\x93\xE9\x87\x8D\xE5\xBA\x86\xE5\x87\xBA\xE5\x94\xAE\xE6\x88\x90\xE6\x9C\xAC\xE5\xBD\xA2\xE5\xBC\x8F\xE5\x9C\x9F\xE8\xB1\x86\xE5\x87\xBA\xE5\x83\xB9\xE4\xB8\x9C\xE6" ~
    "\x96\xB9\xE9\x82\xAE\xE7\xAE\xB1\xE5\x8D\x97\xE4\xBA\xAC\xE6\xB1\x82\xE8\x81\x8C\xE5\x8F\x96\xE5\xBE\x97\xE8\x81\x8C\xE4\xBD\x8D\xE7\x9B\xB8\xE4\xBF\xA1\xE9\xA1\xB5\xE9\x9D\xA2\xE5\x88\x86\xE9\x92\x9F\xE7\xBD\x91\xE9\xA1\xB5\xE7\xA1\xAE\xE5\xAE\x9A\xE5\x9B" ~
    "\xBE\xE4\xBE\x8B\xE7\xBD\x91\xE5\x9D\x80\xE7\xA7\xAF\xE6\x9E\x81\xE9\x94\x99\xE8\xAF\xAF\xE7\x9B\xAE\xE7\x9A\x84\xE5\xAE\x9D\xE8\xB4\x9D\xE6\x9C\xBA\xE5\x85\xB3\xE9\xA3\x8E\xE9\x99\xA9\xE6\x8E\x88\xE6\x9D\x83\xE7\x97\x85\xE6\xAF\x92\xE5\xAE\xA0\xE7\x89\xA9" ~
    "\xE9\x99\xA4\xE4\xBA\x86\xE8\xA9\x95\xE8\xAB\x96\xE7\x96\xBE\xE7\x97\x85\xE5\x8F\x8A\xE6\x97\xB6\xE6\xB1\x82\xE8\xB4\xAD\xE7\xAB\x99\xE7\x82\xB9\xE5\x84\xBF\xE7\xAB\xA5\xE6\xAF\x8F\xE5\xA4\xA9\xE4\xB8\xAD\xE5\xA4\xAE\xE8\xAE\xA4\xE8\xAF\x86\xE6\xAF\x8F\xE4" ~
    "\xB8\xAA\xE5\xA4\xA9\xE6\xB4\xA5\xE5\xAD\x97\xE4\xBD\x93\xE5\x8F\xB0\xE7\x81\xA3\xE7\xBB\xB4\xE6\x8A\xA4\xE6\x9C\xAC\xE9\xA1\xB5\xE4\xB8\xAA\xE6\x80\xA7\xE5\xAE\x98\xE6\x96\xB9\xE5\xB8\xB8\xE8\xA7\x81\xE7\x9B\xB8\xE6\x9C\xBA\xE6\x88\x98\xE7\x95\xA5\xE5\xBA" ~
    "\x94\xE5\xBD\x93\xE5\xBE\x8B\xE5\xB8\x88\xE6\x96\xB9\xE4\xBE\xBF\xE6\xA0\xA1\xE5\x9B\xAD\xE8\x82\xA1\xE5\xB8\x82\xE6\x88\xBF\xE5\xB1\x8B\xE6\xA0\x8F\xE7\x9B\xAE\xE5\x91\x98\xE5\xB7\xA5\xE5\xAF\xBC\xE8\x87\xB4\xE7\xAA\x81\xE7\x84\xB6\xE9\x81\x93\xE5\x85\xB7" ~
    "\xE6\x9C\xAC\xE7\xBD\x91\xE7\xBB\x93\xE5\x90\x88\xE6\xA1\xA3\xE6\xA1\x88\xE5\x8A\xB3\xE5\x8A\xA8\xE5\x8F\xA6\xE5\xA4\x96\xE7\xBE\x8E\xE5\x85\x83\xE5\xBC\x95\xE8\xB5\xB7\xE6\x94\xB9\xE5\x8F\x98\xE7\xAC\xAC\xE5\x9B\x9B\xE4\xBC\x9A\xE8\xAE\xA1\xE8\xAA\xAA\xE6" ~
    "\x98\x8E\xE9\x9A\x90\xE7\xA7\x81\xE5\xAE\x9D\xE5\xAE\x9D\xE8\xA7\x84\xE8\x8C\x83\xE6\xB6\x88\xE8\xB4\xB9\xE5\x85\xB1\xE5\x90\x8C\xE5\xBF\x98\xE8\xAE\xB0\xE4\xBD\x93\xE7\xB3\xBB\xE5\xB8\xA6\xE6\x9D\xA5\xE5\x90\x8D\xE5\xAD\x97\xE7\x99\xBC\xE8\xA1\xA8\xE5\xBC" ~
    "\x80\xE6\x94\xBE\xE5\x8A\xA0\xE7\x9B\x9F\xE5\x8F\x97\xE5\x88\xB0\xE4\xBA\x8C\xE6\x89\x8B\xE5\xA4\xA7\xE9\x87\x8F\xE6\x88\x90\xE4\xBA\xBA\xE6\x95\xB0\xE9\x87\x8F\xE5\x85\xB1\xE4\xBA\xAB\xE5\x8C\xBA\xE5\x9F\x9F\xE5\xA5\xB3\xE5\xAD\xA9\xE5\x8E\x9F\xE5\x88\x99" ~
    "\xE6\x89\x80\xE5\x9C\xA8\xE7\xBB\x93\xE6\x9D\x9F\xE9\x80\x9A\xE4\xBF\xA1\xE8\xB6\x85\xE7\xBA\xA7\xE9\x85\x8D\xE7\xBD\xAE\xE5\xBD\x93\xE6\x97\xB6\xE4\xBC\x98\xE7\xA7\x80\xE6\x80\xA7\xE6\x84\x9F\xE6\x88\xBF\xE4\xBA\xA7\xE9\x81\x8A\xE6\x88\xB2\xE5\x87\xBA\xE5" ~
    "\x8F\xA3\xE6\x8F\x90\xE4\xBA\xA4\xE5\xB0\xB1\xE4\xB8\x9A\xE4\xBF\x9D\xE5\x81\xA5\xE7\xA8\x8B\xE5\xBA\xA6\xE5\x8F\x82\xE6\x95\xB0\xE4\xBA\x8B\xE4\xB8\x9A\xE6\x95\xB4\xE4\xB8\xAA\xE5\xB1\xB1\xE4\xB8\x9C\xE6\x83\x85\xE6\x84\x9F\xE7\x89\xB9\xE6\xAE\x8A\xE5\x88" ~
    "\x86\xE9\xA1\x9E\xE6\x90\x9C\xE5\xB0\x8B\xE5\xB1\x9E\xE4\xBA\x8E\xE9\x97\xA8\xE6\x88\xB7\xE8\xB4\xA2\xE5\x8A\xA1\xE5\xA3\xB0\xE9\x9F\xB3\xE5\x8F\x8A\xE5\x85\xB6\xE8\xB4\xA2\xE7\xBB\x8F\xE5\x9D\x9A\xE6\x8C\x81\xE5\xB9\xB2\xE9\x83\xA8\xE6\x88\x90\xE7\xAB\x8B" ~
    "\xE5\x88\xA9\xE7\x9B\x8A\xE8\x80\x83\xE8\x99\x91\xE6\x88\x90\xE9\x83\xBD\xE5\x8C\x85\xE8\xA3\x85\xE7\x94\xA8\xE6\x88\xB6\xE6\xAF\x94\xE8\xB5\x9B\xE6\x96\x87\xE6\x98\x8E\xE6\x8B\x9B\xE5\x95\x86\xE5\xAE\x8C\xE6\x95\xB4\xE7\x9C\x9F\xE6\x98\xAF\xE7\x9C\xBC\xE7" ~
    "\x9D\x9B\xE4\xBC\x99\xE4\xBC\xB4\xE5\xA8\x81\xE6\x9C\x9B\xE9\xA2\x86\xE5\x9F\x9F\xE5\x8D\xAB\xE7\x94\x9F\xE4\xBC\x98\xE6\x83\xA0\xE8\xAB\x96\xE5\xA3\x87\xE5\x85\xAC\xE5\x85\xB1\xE8\x89\xAF\xE5\xA5\xBD\xE5\x85\x85\xE5\x88\x86\xE7\xAC\xA6\xE5\x90\x88\xE9\x99" ~
    "\x84\xE4\xBB\xB6\xE7\x89\xB9\xE7\x82\xB9\xE4\xB8\x8D\xE5\x8F\xAF\xE8\x8B\xB1\xE6\x96\x87\xE8\xB5\x84\xE4\xBA\xA7\xE6\xA0\xB9\xE6\x9C\xAC\xE6\x98\x8E\xE6\x98\xBE\xE5\xAF\x86\xE7\xA2\xBC\xE5\x85\xAC\xE4\xBC\x97\xE6\xB0\x91\xE6\x97\x8F\xE6\x9B\xB4\xE5\x8A\xA0" ~
    "\xE4\xBA\xAB\xE5\x8F\x97\xE5\x90\x8C\xE5\xAD\xA6\xE5\x90\xAF\xE5\x8A\xA8\xE9\x80\x82\xE5\x90\x88\xE5\x8E\x9F\xE6\x9D\xA5\xE9\x97\xAE\xE7\xAD\x94\xE6\x9C\xAC\xE6\x96\x87\xE7\xBE\x8E\xE9\xA3\x9F\xE7\xBB\xBF\xE8\x89\xB2\xE7\xA8\xB3\xE5\xAE\x9A\xE7\xBB\x88\xE4" ~
    "\xBA\x8E\xE7\x94\x9F\xE7\x89\xA9\xE4\xBE\x9B\xE6\xB1\x82\xE6\x90\x9C\xE7\x8B\x90\xE5\x8A\x9B\xE9\x87\x8F\xE4\xB8\xA5\xE9\x87\x8D\xE6\xB0\xB8\xE8\xBF\x9C\xE5\x86\x99\xE7\x9C\x9F\xE6\x9C\x89\xE9\x99\x90\xE7\xAB\x9E\xE4\xBA\x89\xE5\xAF\xB9\xE8\xB1\xA1\xE8\xB4" ~
    "\xB9\xE7\x94\xA8\xE4\xB8\x8D\xE5\xA5\xBD\xE7\xBB\x9D\xE5\xAF\xB9\xE5\x8D\x81\xE5\x88\x86\xE4\xBF\x83\xE8\xBF\x9B\xE7\x82\xB9\xE8\xAF\x84\xE5\xBD\xB1\xE9\x9F\xB3\xE4\xBC\x98\xE5\x8A\xBF\xE4\xB8\x8D\xE5\xB0\x91\xE6\xAC\xA3\xE8\xB5\x8F\xE5\xB9\xB6\xE4\xB8\x94" ~
    "\xE6\x9C\x89\xE7\x82\xB9\xE6\x96\xB9\xE5\x90\x91\xE5\x85\xA8\xE6\x96\xB0\xE4\xBF\xA1\xE7\x94\xA8\xE8\xAE\xBE\xE6\x96\xBD\xE5\xBD\xA2\xE8\xB1\xA1\xE8\xB5\x84\xE6\xA0\xBC\xE7\xAA\x81\xE7\xA0\xB4\xE9\x9A\x8F\xE7\x9D\x80\xE9\x87\x8D\xE5\xA4\xA7\xE4\xBA\x8E\xE6" ~
    "\x98\xAF\xE6\xAF\x95\xE4\xB8\x9A\xE6\x99\xBA\xE8\x83\xBD\xE5\x8C\x96\xE5\xB7\xA5\xE5\xAE\x8C\xE7\xBE\x8E\xE5\x95\x86\xE5\x9F\x8E\xE7\xBB\x9F\xE4\xB8\x80\xE5\x87\xBA\xE7\x89\x88\xE6\x89\x93\xE9\x80\xA0\xE7\x94\xA2\xE5\x93\x81\xE6\xA6\x82\xE5\x86\xB5\xE7\x94" ~
    "\xA8\xE4\xBA\x8E\xE4\xBF\x9D\xE7\x95\x99\xE5\x9B\xA0\xE7\xB4\xA0\xE4\xB8\xAD\xE5\x9C\x8B\xE5\xAD\x98\xE5\x82\xA8\xE8\xB4\xB4\xE5\x9B\xBE\xE6\x9C\x80\xE6\x84\x9B\xE9\x95\xBF\xE6\x9C\x9F\xE5\x8F\xA3\xE4\xBB\xB7\xE7\x90\x86\xE8\xB4\xA2\xE5\x9F\xBA\xE5\x9C\xB0" ~
    "\xE5\xAE\x89\xE6\x8E\x92\xE6\xAD\xA6\xE6\xB1\x89\xE9\x87\x8C\xE9\x9D\xA2\xE5\x88\x9B\xE5\xBB\xBA\xE5\xA4\xA9\xE7\xA9\xBA\xE9\xA6\x96\xE5\x85\x88\xE5\xAE\x8C\xE5\x96\x84\xE9\xA9\xB1\xE5\x8A\xA8\xE4\xB8\x8B\xE9\x9D\xA2\xE4\xB8\x8D\xE5\x86\x8D\xE8\xAF\x9A\xE4" ~
    "\xBF\xA1\xE6\x84\x8F\xE4\xB9\x89\xE9\x98\xB3\xE5\x85\x89\xE8\x8B\xB1\xE5\x9B\xBD\xE6\xBC\x82\xE4\xBA\xAE\xE5\x86\x9B\xE4\xBA\x8B\xE7\x8E\xA9\xE5\xAE\xB6\xE7\xBE\xA4\xE4\xBC\x97\xE5\x86\x9C\xE6\xB0\x91\xE5\x8D\xB3\xE5\x8F\xAF\xE5\x90\x8D\xE7\xA8\xB1\xE5\xAE" ~
    "\xB6\xE5\x85\xB7\xE5\x8A\xA8\xE7\x94\xBB\xE6\x83\xB3\xE5\x88\xB0\xE6\xB3\xA8\xE6\x98\x8E\xE5\xB0\x8F\xE5\xAD\xA6\xE6\x80\xA7\xE8\x83\xBD\xE8\x80\x83\xE7\xA0\x94\xE7\xA1\xAC\xE4\xBB\xB6\xE8\xA7\x82\xE7\x9C\x8B\xE6\xB8\x85\xE6\xA5\x9A\xE6\x90\x9E\xE7\xAC\x91" ~
    "\xE9\xA6\x96\xE9\xA0\x81\xE9\xBB\x84\xE9\x87\x91\xE9\x80\x82\xE7\x94\xA8\xE6\xB1\x9F\xE8\x8B\x8F\xE7\x9C\x9F\xE5\xAE\x9E\xE4\xB8\xBB\xE7\xAE\xA1\xE9\x98\xB6\xE6\xAE\xB5\xE8\xA8\xBB\xE5\x86\x8A\xE7\xBF\xBB\xE8\xAF\x91\xE6\x9D\x83\xE5\x88\xA9\xE5\x81\x9A\xE5" ~
    "\xA5\xBD\xE4\xBC\xBC\xE4\xB9\x8E\xE9\x80\x9A\xE8\xAE\xAF\xE6\x96\xBD\xE5\xB7\xA5\xE7\x8B\x80\xE6\x85\x8B\xE4\xB9\x9F\xE8\xAE\xB8\xE7\x8E\xAF\xE4\xBF\x9D\xE5\x9F\xB9\xE5\x85\xBB\xE6\xA6\x82\xE5\xBF\xB5\xE5\xA4\xA7\xE5\x9E\x8B\xE6\x9C\xBA\xE7\xA5\xA8\xE7\x90" ~
    "\x86\xE8\xA7\xA3\xE5\x8C\xBF\xE5\x90\x8Dcuandoenviarmadridbuscariniciotiempoporquecuentaestado" ~
    "puedenjuegoscontraest\xC3\xA1nnombretienenperfilmaneraamigosciudadcent" ~
    "roaunquepuedesdentroprimerprecioseg\xC3\xBAnbuenosvolverpuntossemanaha" ~
    "b\xC3\xADaagostonuevosunidoscarlosequiponi\xC3\xB1osmuchosalgunacorreoimagen" ~
    "partirarribamar\xC3\xADahombreempleoverdadcambiomuchasfueronpasadol\xC3\xADn" ~
    "eaparecenuevascursosestabaquierolibroscuantoaccesomiguelvarioscu" ~
    "atrotienesgruposser\xC3\xA1neuropamediosfrenteacercadem\xC3\xA1sofertacoches" ~
    "modeloitalialetrasalg\xC3\xBAncompracualesexistecuerposiendoprensalleg" ~
    "arviajesdineromurciapodr\xC3\xA1puestodiariopuebloquieremanuelpropiocr" ~
    "isisciertoseguromuertefuentecerrargrandeefectopartesmedidapropia" ~
    "ofrecetierrae-mailvariasformasfuturoobjetoseguirriesgonormasmism" ~
    "os\xC3\xBAnicocaminositiosraz\xC3\xB3ndebidopruebatoledoten\xC3\xADajes\xC3\xBAsesperoco" ~
    "cinaorigentiendacientoc\xC3\xA1dizhablarser\xC3\xADalatinafuerzaestiloguerra" ~
    "entrar\xC3\xA9xitol\xC3\xB3pezagendav\xC3\xADdeoevitarpaginametrosjavierpadresf\xC3\xA1c" ~
    "ilcabeza\xC3\xA1reassalidaenv\xC3\xADojap\xC3\xB3nabusosbienestextosllevarpuedanfu" ~
    "ertecom\xC3\xBAnclaseshumanotenidobilbaounidadest\xC3\xA1seditarcreado\xD0\xB4\xD0\xBB\xD1\x8F" ~
    "\xD1\x87\xD1\x82\xD0\xBE\xD0\xBA\xD0\xB0\xD0\xBA\xD0\xB8\xD0\xBB\xD0\xB8\xD1\x8D\xD1\x82\xD0\xBE\xD0\xB2\xD1\x81\xD0\xB5\xD0\xB5\xD0\xB3\xD0\xBE\xD0\xBF\xD1\x80\xD0\xB8\xD1\x82\xD0\xB0\xD0\xBA\xD0\xB5\xD1\x89\xD0\xB5\xD1\x83\xD0\xB6\xD0\xB5\xD0\x9A\xD0\xB0" ~
    "\xD0\xBA\xD0\xB1\xD0\xB5\xD0\xB7\xD0\xB1\xD1\x8B\xD0\xBB\xD0\xBE\xD0\xBD\xD0\xB8\xD0\x92\xD1\x81\xD0\xB5\xD0\xBF\xD0\xBE\xD0\xB4\xD0\xAD\xD1\x82\xD0\xBE\xD1\x82\xD0\xBE\xD0\xBC\xD1\x87\xD0\xB5\xD0\xBC\xD0\xBD\xD0\xB5\xD1\x82\xD0\xBB\xD0\xB5\xD1\x82\xD1\x80" ~
    "\xD0\xB0\xD0\xB7\xD0\xBE\xD0\xBD\xD0\xB0\xD0\xB3\xD0\xB4\xD0\xB5\xD0\xBC\xD0\xBD\xD0\xB5\xD0\x94\xD0\xBB\xD1\x8F\xD0\x9F\xD1\x80\xD0\xB8\xD0\xBD\xD0\xB0\xD1\x81\xD0\xBD\xD0\xB8\xD1\x85\xD1\x82\xD0\xB5\xD0\xBC\xD0\xBA\xD1\x82\xD0\xBE\xD0\xB3\xD0\xBE\xD0\xB4" ~
    "\xD0\xB2\xD0\xBE\xD1\x82\xD1\x82\xD0\xB0\xD0\xBC\xD0\xA1\xD0\xA8\xD0\x90\xD0\xBC\xD0\xB0\xD1\x8F\xD0\xA7\xD1\x82\xD0\xBE\xD0\xB2\xD0\xB0\xD1\x81\xD0\xB2\xD0\xB0\xD0\xBC\xD0\xB5\xD0\xBC\xD1\x83\xD0\xA2\xD0\xB0\xD0\xBA\xD0\xB4\xD0\xB2\xD0\xB0\xD0\xBD\xD0\xB0" ~
    "\xD0\xBC\xD1\x8D\xD1\x82\xD0\xB8\xD1\x8D\xD1\x82\xD1\x83\xD0\x92\xD0\xB0\xD0\xBC\xD1\x82\xD0\xB5\xD1\x85\xD0\xBF\xD1\x80\xD0\xBE\xD1\x82\xD1\x83\xD1\x82\xD0\xBD\xD0\xB0\xD0\xB4\xD0\xB4\xD0\xBD\xD1\x8F\xD0\x92\xD0\xBE\xD1\x82\xD1\x82\xD1\x80\xD0\xB8\xD0\xBD" ~
    "\xD0\xB5\xD0\xB9\xD0\x92\xD0\xB0\xD1\x81\xD0\xBD\xD0\xB8\xD0\xBC\xD1\x81\xD0\xB0\xD0\xBC\xD1\x82\xD0\xBE\xD1\x82\xD1\x80\xD1\x83\xD0\xB1\xD0\x9E\xD0\xBD\xD0\xB8\xD0\xBC\xD0\xB8\xD1\x80\xD0\xBD\xD0\xB5\xD0\xB5\xD0\x9E\xD0\x9E\xD0\x9E\xD0\xBB\xD0\xB8\xD1\x86" ~
    "\xD1\x8D\xD1\x82\xD0\xB0\xD0\x9E\xD0\xBD\xD0\xB0\xD0\xBD\xD0\xB5\xD0\xBC\xD0\xB4\xD0\xBE\xD0\xBC\xD0\xBC\xD0\xBE\xD0\xB9\xD0\xB4\xD0\xB2\xD0\xB5\xD0\xBE\xD0\xBD\xD0\xBE\xD1\x81\xD1\x83\xD0\xB4\xE0\xA4\x95\xE0\xA5\x87\xE0\xA4\xB9\xE0\xA5\x88\xE0\xA4\x95\xE0" ~
    "\xA5\x80\xE0\xA4\xB8\xE0\xA5\x87\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\x95\xE0\xA5\x8B\xE0\xA4\x94\xE0\xA4\xB0\xE0\xA4\xAA\xE0\xA4\xB0\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\x8F\xE0\xA4\x95\xE0\xA4\x95\xE0\xA4\xBF\xE0\xA4\xAD\xE0\xA5\x80\xE0\xA4\x87\xE0\xA4\xB8\xE0\xA4" ~
    "\x95\xE0\xA4\xB0\xE0\xA4\xA4\xE0\xA5\x8B\xE0\xA4\xB9\xE0\xA5\x8B\xE0\xA4\x86\xE0\xA4\xAA\xE0\xA4\xB9\xE0\xA5\x80\xE0\xA4\xAF\xE0\xA4\xB9\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xA4\xE0\xA4\x95\xE0\xA4\xA5\xE0\xA4\xBEjagran\xE0\xA4\x86\xE0\xA4\x9C" ~
    "\xE0\xA4\x9C\xE0\xA5\x8B\xE0\xA4\x85\xE0\xA4\xAC\xE0\xA4\xA6\xE0\xA5\x8B\xE0\xA4\x97\xE0\xA4\x88\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\x97\xE0\xA4\x8F\xE0\xA4\xB9\xE0\xA4\xAE\xE0\xA4\x87\xE0\xA4\xA8\xE0\xA4\xB5\xE0\xA4\xB9\xE0\xA4\xAF\xE0\xA5\x87\xE0\xA4\xA5\xE0" ~
    "\xA5\x87\xE0\xA4\xA5\xE0\xA5\x80\xE0\xA4\x98\xE0\xA4\xB0\xE0\xA4\x9C\xE0\xA4\xAC\xE0\xA4\xA6\xE0\xA5\x80\xE0\xA4\x95\xE0\xA4\x88\xE0\xA4\x9C\xE0\xA5\x80\xE0\xA4\xB5\xE0\xA5\x87\xE0\xA4\xA8\xE0\xA4\x88\xE0\xA4\xA8\xE0\xA4\x8F\xE0\xA4\xB9\xE0\xA4\xB0\xE0\xA4" ~
    "\x89\xE0\xA4\xB8\xE0\xA4\xAE\xE0\xA5\x87\xE0\xA4\x95\xE0\xA4\xAE\xE0\xA4\xB5\xE0\xA5\x8B\xE0\xA4\xB2\xE0\xA5\x87\xE0\xA4\xB8\xE0\xA4\xAC\xE0\xA4\xAE\xE0\xA4\x88\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\x93\xE0\xA4\xB0\xE0\xA4\x86\xE0\xA4\xAE\xE0\xA4\xAC\xE0\xA4\xB8" ~
    "\xE0\xA4\xAD\xE0\xA4\xB0\xE0\xA4\xAC\xE0\xA4\xA8\xE0\xA4\x9A\xE0\xA4\xB2\xE0\xA4\xAE\xE0\xA4\xA8\xE0\xA4\x86\xE0\xA4\x97\xE0\xA4\xB8\xE0\xA5\x80\xE0\xA4\xB2\xE0\xA5\x80\xD8\xB9\xD9\x84\xD9\x89\xD8\xA5\xD9\x84\xD9\x89\xD9\x87\xD8\xB0\xD8\xA7\xD8\xA2\xD8\xAE" ~
    "\xD8\xB1\xD8\xB9\xD8\xAF\xD8\xAF\xD8\xA7\xD9\x84\xD9\x89\xD9\x87\xD8\xB0\xD9\x87\xD8\xB5\xD9\x88\xD8\xB1\xD8\xBA\xD9\x8A\xD8\xB1\xD9\x83\xD8\xA7\xD9\x86\xD9\x88\xD9\x84\xD8\xA7\xD8\xA8\xD9\x8A\xD9\x86\xD8\xB9\xD8\xB1\xD8\xB6\xD8\xB0\xD9\x84\xD9\x83\xD9\x87" ~
    "\xD9\x86\xD8\xA7\xD9\x8A\xD9\x88\xD9\x85\xD9\x82\xD8\xA7\xD9\x84\xD8\xB9\xD9\x84\xD9\x8A\xD8\xA7\xD9\x86\xD8\xA7\xD9\x84\xD9\x83\xD9\x86\xD8\xAD\xD8\xAA\xD9\x89\xD9\x82\xD8\xA8\xD9\x84\xD9\x88\xD8\xAD\xD8\xA9\xD8\xA7\xD8\xAE\xD8\xB1\xD9\x81\xD9\x82\xD8\xB7" ~
    "\xD8\xB9\xD8\xA8\xD8\xAF\xD8\xB1\xD9\x83\xD9\x86\xD8\xA5\xD8\xB0\xD8\xA7\xD9\x83\xD9\x85\xD8\xA7\xD8\xA7\xD8\xAD\xD8\xAF\xD8\xA5\xD9\x84\xD8\xA7\xD9\x81\xD9\x8A\xD9\x87\xD8\xA8\xD8\xB9\xD8\xB6\xD9\x83\xD9\x8A\xD9\x81\xD8\xA8\xD8\xAD\xD8\xAB\xD9\x88\xD9\x85" ~
    "\xD9\x86\xD9\x88\xD9\x87\xD9\x88\xD8\xA3\xD9\x86\xD8\xA7\xD8\xAC\xD8\xAF\xD8\xA7\xD9\x84\xD9\x87\xD8\xA7\xD8\xB3\xD9\x84\xD9\x85\xD8\xB9\xD9\x86\xD8\xAF\xD9\x84\xD9\x8A\xD8\xB3\xD8\xB9\xD8\xA8\xD8\xB1\xD8\xB5\xD9\x84\xD9\x89\xD9\x85\xD9\x86\xD8\xB0\xD8\xA8" ~
    "\xD9\x87\xD8\xA7\xD8\xA3\xD9\x86\xD9\x87\xD9\x85\xD8\xAB\xD9\x84\xD9\x83\xD9\x86\xD8\xAA\xD8\xA7\xD9\x84\xD8\xA7\xD8\xAD\xD9\x8A\xD8\xAB\xD9\x85\xD8\xB5\xD8\xB1\xD8\xB4\xD8\xB1\xD8\xAD\xD8\xAD\xD9\x88\xD9\x84\xD9\x88\xD9\x81\xD9\x8A\xD8\xA7\xD8\xB0\xD8\xA7" ~
    "\xD9\x84\xD9\x83\xD9\x84\xD9\x85\xD8\xB1\xD8\xA9\xD8\xA7\xD9\x86\xD8\xAA\xD8\xA7\xD9\x84\xD9\x81\xD8\xA3\xD8\xA8\xD9\x88\xD8\xAE\xD8\xA7\xD8\xB5\xD8\xA3\xD9\x86\xD8\xAA\xD8\xA7\xD9\x86\xD9\x87\xD8\xA7\xD9\x84\xD9\x8A\xD8\xB9\xD8\xB6\xD9\x88\xD9\x88\xD9\x82" ~
    "\xD8\xAF\xD8\xA7\xD8\xA8\xD9\x86\xD8\xAE\xD9\x8A\xD8\xB1\xD8\xA8\xD9\x86\xD8\xAA\xD9\x84\xD9\x83\xD9\x85\xD8\xB4\xD8\xA7\xD8\xA1\xD9\x88\xD9\x87\xD9\x8A\xD8\xA7\xD8\xA8\xD9\x88\xD9\x82\xD8\xB5\xD8\xB5\xD9\x88\xD9\x85\xD8\xA7\xD8\xB1\xD9\x82\xD9\x85\xD8\xA3" ~
    "\xD8\xAD\xD8\xAF\xD9\x86\xD8\xAD\xD9\x86\xD8\xB9\xD8\xAF\xD9\x85\xD8\xB1\xD8\xA3\xD9\x8A\xD8\xA7\xD8\xAD\xD8\xA9\xD9\x83\xD8\xAA\xD8\xA8\xD8\xAF\xD9\x88\xD9\x86\xD9\x8A\xD8\xAC\xD8\xA8\xD9\x85\xD9\x86\xD9\x87\xD8\xAA\xD8\xAD\xD8\xAA\xD8\xAC\xD9\x87\xD8\xA9" ~
    "\xD8\xB3\xD9\x86\xD8\xA9\xD9\x8A\xD8\xAA\xD9\x85\xD9\x83\xD8\xB1\xD8\xA9\xD8\xBA\xD8\xB2\xD8\xA9\xD9\x86\xD9\x81\xD8\xB3\xD8\xA8\xD9\x8A\xD8\xAA\xD9\x84\xD9\x84\xD9\x87\xD9\x84\xD9\x86\xD8\xA7\xD8\xAA\xD9\x84\xD9\x83\xD9\x82\xD9\x84\xD8\xA8\xD9\x84\xD9\x85" ~
    "\xD8\xA7\xD8\xB9\xD9\x86\xD9\x87\xD8\xA3\xD9\x88\xD9\x84\xD8\xB4\xD9\x8A\xD8\xA1\xD9\x86\xD9\x88\xD8\xB1\xD8\xA3\xD9\x85\xD8\xA7\xD9\x81\xD9\x8A\xD9\x83\xD8\xA8\xD9\x83\xD9\x84\xD8\xB0\xD8\xA7\xD8\xAA\xD8\xB1\xD8\xAA\xD8\xA8\xD8\xA8\xD8\xA3\xD9\x86\xD9\x87" ~
    "\xD9\x85\xD8\xB3\xD8\xA7\xD9\x86\xD9\x83\xD8\xA8\xD9\x8A\xD8\xB9\xD9\x81\xD9\x82\xD8\xAF\xD8\xAD\xD8\xB3\xD9\x86\xD9\x84\xD9\x87\xD9\x85\xD8\xB4\xD8\xB9\xD8\xB1\xD8\xA3\xD9\x87\xD9\x84\xD8\xB4\xD9\x87\xD8\xB1\xD9\x82\xD8\xB7\xD8\xB1\xD8\xB7\xD9\x84\xD8\xA8" ~
    "profileservicedefaulthimselfdetailscontentsupportstartedmessages" ~
    "uccessfashion<title>countryaccountcreatedstoriesresultsrunningpr" ~
    "ocesswritingobjectsvisiblewelcomearticleunknownnetworkcompanydyn" ~
    "amicbrowserprivacyproblemServicerespectdisplayrequestreservewebs" ~
    "itehistoryfriendsoptionsworkingversionmillionchannelwindow.addre" ~
    "ssvisitedweathercorrectproductedirectforwardyou canremovedsubjec" ~
    "tcontrolarchivecurrentreadinglibrarylimitedmanagerfurthersummary" ~
    "machineminutesprivatecontextprogramsocietynumberswrittenenabledt" ~
    "riggersourcesloadingelementpartnerfinallyperfectmeaningsystemske" ~
    "epingculture&quot;,journalprojectsurfaces&quot;expiresreviewsbal" ~
    "anceEnglishContentthroughPlease opinioncontactaverageprimaryvill" ~
    "ageSpanishgallerydeclinemeetingmissionpopularqualitymeasuregener" ~
    "alspeciessessionsectionwriterscounterinitialreportsfiguresmember" ~
    "sholdingdisputeearlierexpressdigitalpictureAnothermarriedtraffic" ~
    "leadingchangedcentralvictoryimages/reasonsstudiesfeaturelistingm" ~
    "ust beschoolsVersionusuallyepisodeplayinggrowingobviousoverlaypr" ~
    "esentactions</ul>\x0D\x0Awrapperalreadycertainrealitystorageanotherdes" ~
    "ktopofferedpatternunusualDigitalcapitalWebsitefailureconnectredu" ~
    "cedAndroiddecadesregular &amp; animalsreleaseAutomatgettingmetho" ~
    "dsnothingPopularcaptionletterscapturesciencelicensechangesEnglan" ~
    "d=1&amp;History = new CentralupdatedSpecialNetworkrequirecomment" ~
    "warningCollegetoolbarremainsbecauseelectedDeutschfinanceworkersq" ~
    "uicklybetweenexactlysettingdiseaseSocietyweaponsexhibit&lt;!--Co" ~
    "ntrolclassescoveredoutlineattacksdevices(windowpurposetitle=\"Mob" ~
    "ile killingshowingItaliandroppedheavilyeffects-1']);\x0AconfirmCurr" ~
    "entadvancesharingopeningdrawingbillionorderedGermanyrelated</for" ~
    "m>includewhetherdefinedSciencecatalogArticlebuttonslargestunifor" ~
    "mjourneysidebarChicagoholidayGeneralpassage,&quot;animatefeeling" ~
    "arrivedpassingnaturalroughly.\x0A\x0AThe but notdensityBritainChinesel" ~
    "ack oftributeIreland\" data-factorsreceivethat isLibraryhusbandin" ~
    " factaffairsCharlesradicalbroughtfindinglanding:lang=\"return lea" ~
    "dersplannedpremiumpackageAmericaEdition]&quot;Messageneed tovalu" ~
    "e=\"complexlookingstationbelievesmaller-mobilerecordswant tokind " ~
    "ofFirefoxyou aresimilarstudiedmaximumheadingrapidlyclimatekingdo" ~
    "memergedamountsfoundedpioneerformuladynastyhow to Supportrevenue" ~
    "economyResultsbrothersoldierlargelycalling.&quot;AccountEdward s" ~
    "egmentRobert effortsPacificlearnedup withheight:we haveAngelesna" ~
    "tions_searchappliedacquiremassivegranted: falsetreatedbiggestben" ~
    "efitdrivingStudiesminimumperhapsmorningsellingis usedreversevari" ~
    "ant role=\"missingachievepromotestudentsomeoneextremerestorebotto" ~
    "m:evolvedall thesitemapenglishway to  AugustsymbolsCompanymatter" ~
    "smusicalagainstserving})();\x0D\x0Apaymenttroubleconceptcompareparents" ~
    "playersregionsmonitor ''The winningexploreadaptedGalleryproducea" ~
    "bilityenhancecareers). The collectSearch ancientexistedfooter ha" ~
    "ndlerprintedconsoleEasternexportswindowsChannelillegalneutralsug" ~
    "gest_headersigning.html\">settledwesterncausing-webkitclaimedJust" ~
    "icechaptervictimsThomas mozillapromisepartieseditionoutside:fals" ~
    "e,hundredOlympic_buttonauthorsreachedchronicdemandssecondsprotec" ~
    "tadoptedprepareneithergreatlygreateroverallimprovecommandspecial" ~
    "search.worshipfundingthoughthighestinsteadutilityquarterCulturet" ~
    "estingclearlyexposedBrowserliberal} catchProjectexamplehide();Fl" ~
    "oridaanswersallowedEmperordefenseseriousfreedomSeveral-buttonFur" ~
    "therout of != nulltrainedDenmarkvoid(0)/all.jspreventRequestStep" ~
    "hen\x0A\x0AWhen observe</h2>\x0D\x0AModern provide\" alt=\"borders.\x0A\x0AFor \x0A\x0AMan" ~
    "y artistspoweredperformfictiontype ofmedicalticketsopposedCounci" ~
    "lwitnessjusticeGeorge Belgium...</a>twitternotablywaitingwarfare" ~
    " Other rankingphrasesmentionsurvivescholar</p>\x0D\x0A Countryignoredl" ~
    "oss ofjust asGeorgiastrange<head><stopped1']);\x0D\x0Aislandsnotablebo" ~
    "rder:list ofcarried100,000</h3>\x0A severalbecomesselect wedding00." ~
    "htmlmonarchoff theteacherhighly biologylife ofor evenrise of&raq" ~
    "uo;plusonehunting(thoughDouglasjoiningcirclesFor theAncientVietn" ~
    "amvehiclesuch ascrystalvalue =Windowsenjoyeda smallassumed<a id=" ~
    "\"foreign All rihow theDisplayretiredhoweverhidden;battlesseeking" ~
    "cabinetwas notlook atconductget theJanuaryhappensturninga:hoverO" ~
    "nline French lackingtypicalextractenemieseven ifgeneratdecidedar" ~
    "e not/searchbeliefs-image:locatedstatic.login\">convertviolentent" ~
    "eredfirst\">circuitFinlandchemistshe was10px;\">as suchdivided</sp" ~
    "an>will beline ofa greatmystery/index.fallingdue to railwaycolle" ~
    "gemonsterdescentit withnuclearJewish protestBritishflowerspredic" ~
    "treformsbutton who waslectureinstantsuicidegenericperiodsmarkets" ~
    "Social fishingcombinegraphicwinners<br /><by the NaturalPrivacyc" ~
    "ookiesoutcomeresolveSwedishbrieflyPersianso muchCenturydepictsco" ~
    "lumnshousingscriptsnext tobearingmappingrevisedjQuery(-width:tit" ~
    "le\">tooltipSectiondesignsTurkishyounger.match(})();\x0A\x0Aburningoper" ~
    "atedegreessource=Richardcloselyplasticentries</tr>\x0D\x0Acolor:#ul id" ~
    "=\"possessrollingphysicsfailingexecutecontestlink toDefault<br />" ~
    "\x0A: true,chartertourismclassicproceedexplain</h1>\x0D\x0Aonline.?xml ve" ~
    "helpingdiamonduse theairlineend -->).attr(readershosting#ffffffr" ~
    "ealizeVincentsignals src=\"/ProductdespitediversetellingPublic he" ~
    "ld inJoseph theatreaffects<style>a largedoesn'tlater, Elementfav" ~
    "iconcreatorHungaryAirportsee theso thatMichaelSystemsPrograms, a" ~
    "nd  width=e&quot;tradingleft\">\x0ApersonsGolden Affairsgrammarformi" ~
    "ngdestroyidea ofcase ofoldest this is.src = cartoonregistrCommon" ~
    "sMuslimsWhat isin manymarkingrevealsIndeed,equally/show_aoutdoor" ~
    "escape(Austriageneticsystem,In the sittingHe alsoIslandsAcademy\x0A" ~
    "\x09\x09<!--Daniel bindingblock\">imposedutilizeAbraham(except{width:pu" ~
    "tting).html(|| [];\x0ADATA[ *kitchenmountedactual dialectmainly _bl" ~
    "ank'installexpertsif(typeIt also&copy; \">Termsborn inOptionseast" ~
    "erntalkingconcerngained ongoingjustifycriticsfactoryits ownassau" ~
    "ltinvitedlastinghis ownhref=\"/\" rel=\"developconcertdiagramdollar" ~
    "sclusterphp?id=alcohol);})();using a><span>vesselsrevivalAddress" ~
    "amateurandroidallegedillnesswalkingcentersqualifymatchesunifiede" ~
    "xtinctDefensedied in\x0A\x09<!-- customslinkingLittle Book ofeveningmi" ~
    "n.js?are thekontakttoday's.html\" target=wearingAll Rig;\x0A})();rai" ~
    "sing Also, crucialabout\">declare-->\x0A<scfirefoxas muchappliesinde" ~
    "x, s, but type = \x0A\x0D\x0A<!--towardsRecordsPrivateForeignPremierchoic" ~
    "esVirtualreturnsCommentPoweredinline;povertychamberLiving volume" ~
    "sAnthonylogin\" RelatedEconomyreachescuttinggravitylife inChapter" ~
    "-shadowNotable</td>\x0D\x0A returnstadiumwidgetsvaryingtravelsheld byw" ~
    "ho arework infacultyangularwho hadairporttown of\x0A\x0ASome 'click'ch" ~
    "argeskeywordit willcity of(this);Andrew unique checkedor more300" ~
    "px; return;rsion=\"pluginswithin herselfStationFederalventurepubl" ~
    "ishsent totensionactresscome tofingersDuke ofpeople,exploitwhat " ~
    "isharmonya major\":\"httpin his menu\">\x0Amonthlyofficercouncilgainin" ~
    "geven inSummarydate ofloyaltyfitnessand wasemperorsupremeSecond " ~
    "hearingRussianlongestAlbertalateralset of small\">.appenddo withf" ~
    "ederalbank ofbeneathDespiteCapitalgrounds), and percentit fromcl" ~
    "osingcontainInsteadfifteenas well.yahoo.respondfighterobscureref" ~
    "lectorganic= Math.editingonline paddinga wholeonerroryear ofend " ~
    "of barrierwhen itheader home ofresumedrenamedstrong>heatingretai" ~
    "nscloudfrway of March 1knowingin partBetweenlessonsclosestvirtua" ~
    "llinks\">crossedEND -->famous awardedLicenseHealth fairly wealthy" ~
    "minimalAfricancompetelabel\">singingfarmersBrasil)discussreplaceG" ~
    "regoryfont copursuedappearsmake uproundedboth ofblockedsaw theof" ~
    "ficescoloursif(docuwhen heenforcepush(fuAugust UTF-8\">Fantasyin " ~
    "mostinjuredUsuallyfarmingclosureobject defenceuse of Medical<bod" ~
    "y>\x0Aevidentbe usedkeyCodesixteenIslamic#000000entire widely activ" ~
    "e (typeofone cancolor =speakerextendsPhysicsterrain<tbody>funera" ~
    "lviewingmiddle cricketprophetshifteddoctorsRussell targetcompact" ~
    "algebrasocial-bulk ofman and</td>\x0A he left).val()false);logicalb" ~
    "ankinghome tonaming Arizonacredits);\x0A});\x0Afounderin turnCollinsbe" ~
    "fore But thechargedTitle\">CaptainspelledgoddessTag -->Adding:but" ~
    " wasRecent patientback in=false&Lincolnwe knowCounterJudaismscri" ~
    "pt altered']);\x0A  has theunclearEvent',both innot all\x0A\x0A<!-- placi" ~
    "nghard to centersort ofclientsstreetsBernardassertstend tofantas" ~
    "ydown inharbourFreedomjewelry/about..searchlegendsis mademodern " ~
    "only ononly toimage\" linear painterand notrarely acronymdelivers" ~
    "horter00&amp;as manywidth=\"/* <![Ctitle =of the lowest picked es" ~
    "capeduses ofpeoples PublicMatthewtacticsdamagedway forlaws ofeas" ~
    "y to windowstrong  simple}catch(seventhinfoboxwent topaintedciti" ~
    "zenI don'tretreat. Some ww.\");\x0Abombingmailto:made in. Many carri" ~
    "es||{};wiwork ofsynonymdefeatsfavoredopticalpageTraunless sendin" ~
    "gleft\"><comScorAll thejQuery.touristClassicfalse\" Wilhelmsuburbs" ~
    "genuinebishops.split(global followsbody ofnominalContactsecularl" ~
    "eft tochiefly-hidden-banner</li>\x0A\x0A. When in bothdismissExploreal" ~
    "ways via thespa\xC3\xB1olwelfareruling arrangecaptainhis sonrule ofhe " ~
    "tookitself,=0&amp;(calledsamplesto makecom/pagMartin Kennedyacce" ~
    "ptsfull ofhandledBesides//--></able totargetsessencehim to its b" ~
    "y common.mineralto takeways tos.org/ladvisedpenaltysimple:if the" ~
    "yLettersa shortHerbertstrikes groups.lengthflightsoverlapslowly " ~
    "lesser social </p>\x0A\x09\x09it intoranked rate oful>\x0D\x0A  attemptpair ofm" ~
    "ake itKontaktAntoniohaving ratings activestreamstrapped\").css(ho" ~
    "stilelead tolittle groups,Picture-->\x0D\x0A\x0D\x0A rows=\" objectinverse<fo" ~
    "oterCustomV><\\/scrsolvingChamberslaverywoundedwhereas!= 'undfor " ~
    "allpartly -right:Arabianbacked centuryunit ofmobile-Europe,is ho" ~
    "merisk ofdesiredClintoncost ofage of become none ofp&quot;Middle" ~
    " ead')[0Criticsstudios>&copy;group\">assemblmaking pressedwidget." ~
    "ps:\" ? rebuiltby someFormer editorsdelayedCanonichad thepushingc" ~
    "lass=\"but arepartialBabylonbottom carrierCommandits useAs withco" ~
    "ursesa thirddenotesalso inHouston20px;\">accuseddouble goal ofFam" ~
    "ous ).bind(priests Onlinein Julyst + \"gconsultdecimalhelpfulrevi" ~
    "vedis veryr'+'iptlosing femalesis alsostringsdays ofarrivalfutur" ~
    "e <objectforcingString(\" />\x0A\x09\x09here isencoded.  The balloondone b" ~
    "y/commonbgcolorlaw of Indianaavoidedbut the2px 3pxjquery.after a" ~
    "policy.men andfooter-= true;for usescreen.Indian image =family,h" ~
    "ttp:// &nbsp;driverseternalsame asnoticedviewers})();\x0A is morese" ~
    "asonsformer the newis justconsent Searchwas thewhy theshippedbr>" ~
    "<br>width: height=made ofcuisineis thata very Admiral fixed;norm" ~
    "al MissionPress, ontariocharsettry to invaded=\"true\"spacingis mo" ~
    "sta more totallyfall of});\x0D\x0A  immensetime inset outsatisfyto fin" ~
    "ddown tolot of Playersin Junequantumnot thetime todistantFinnish" ~
    "src = (single help ofGerman law andlabeledforestscookingspace\">h" ~
    "eader-well asStanleybridges/globalCroatia About [0];\x0A  it, andgr" ~
    "oupedbeing a){throwhe madelighterethicalFFFFFF\"bottom\"like a emp" ~
    "loyslive inas seenprintermost ofub-linkrejectsand useimage\">succ" ~
    "eedfeedingNuclearinformato helpWomen'sNeitherMexicanprotein<tabl" ~
    "e by manyhealthylawsuitdevised.push({sellerssimply Through.cooki" ~
    "e Image(older\">us.js\"> Since universlarger open to!-- endlies in" ~
    "']);\x0D\x0A  marketwho is (\"DOMComanagedone fortypeof Kingdomprofitsp" ~
    "roposeto showcenter;made itdressedwere inmixtureprecisearisingsr" ~
    "c = 'make a securedBaptistvoting \x0A\x09\x09var March 2grew upClimate.re" ~
    "moveskilledway the</head>face ofacting right\">to workreduceshas " ~
    "haderectedshow();action=book ofan area== \"htt<header\x0A<html>confo" ~
    "rmfacing cookie.rely onhosted .customhe wentbut forspread Family" ~
    " a meansout theforums.footage\">MobilClements\" id=\"as highintense" ~
    "--><!--female is seenimpliedset thea stateand hisfastestbesidesb" ~
    "utton_bounded\"><img Infoboxevents,a youngand areNative cheaperTi" ~
    "meoutand hasengineswon the(mostlyright: find a -bottomPrince are" ~
    "a ofmore ofsearch_nature,legallyperiod,land ofor withinducedprov" ~
    "ingmissilelocallyAgainstthe wayk&quot;px;\">\x0D\x0Apushed abandonnumer" ~
    "alCertainIn thismore inor somename isand, incrownedISBN 0-create" ~
    "sOctobermay notcenter late inDefenceenactedwish tobroadlycooling" ~
    "onload=it. TherecoverMembersheight assumes<html>\x0Apeople.in one =" ~
    "windowfooter_a good reklamaothers,to this_cookiepanel\">London,de" ~
    "finescrushedbaptismcoastalstatus title\" move tolost inbetter imp" ~
    "liesrivalryservers SystemPerhapses and contendflowinglasted rise" ~
    " inGenesisview ofrising seem tobut in backinghe willgiven agivin" ~
    "g cities.flow of Later all butHighwayonly bysign ofhe doesdiffer" ~
    "sbattery&amp;lasinglesthreatsintegertake onrefusedcalled =US&amp" ~
    "See thenativesby thissystem.head of:hover,lesbiansurnameand allc" ~
    "ommon/header__paramsHarvard/pixel.removalso longrole ofjointlysk" ~
    "yscraUnicodebr />\x0D\x0AAtlantanucleusCounty,purely count\">easily bui" ~
    "ld aonclicka givenpointerh&quot;events else {\x0Aditionsnow the, wi" ~
    "th man whoorg/Webone andcavalryHe diedseattle00,000 {windowhave " ~
    "toif(windand itssolely m&quot;renewedDetroitamongsteither them i" ~
    "nSenatorUs</a><King ofFrancis-produche usedart andhim andused by" ~
    "scoringat hometo haverelatesibilityfactionBuffalolink\"><what hef" ~
    "ree toCity ofcome insectorscountedone daynervoussquare };if(goin" ~
    " whatimg\" alis onlysearch/tuesdaylooselySolomonsexual - <a hrmed" ~
    "ium\"DO NOT France,with a war andsecond take a >\x0D\x0A\x0D\x0A\x0D\x0Amarket.high" ~
    "waydone inctivity\"last\">obligedrise to\"undefimade to Early prais" ~
    "edin its for hisathleteJupiterYahoo! termed so manyreally s. The" ~
    " a woman?value=direct right\" bicycleacing=\"day andstatingRather," ~
    "higher Office are nowtimes, when a pay foron this-link\">;bordera" ~
    "round annual the Newput the.com\" takin toa brief(in thegroups.; " ~
    "widthenzymessimple in late{returntherapya pointbanninginks\">\x0A();" ~
    "\" rea place\\u003Caabout atr>\x0D\x0A\x09\x09ccount gives a<SCRIPTRailwaythem" ~
    "es/toolboxById(\"xhumans,watchesin some if (wicoming formats Unde" ~
    "r but hashanded made bythan infear ofdenoted/iframeleft involtag" ~
    "ein eacha&quot;base ofIn manyundergoregimesaction </p>\x0D\x0A<ustomVa" ~
    ";&gt;</importsor thatmostly &amp;re size=\"</a></ha classpassiveH" ~
    "ost = WhetherfertileVarious=[];(fucameras/></td>acts asIn some>\x0D" ~
    "\x0A\x0D\x0A<!organis <br />Beijingcatal\xC3\xA0deutscheuropeueuskaragaeilgesve" ~
    "nskaespa\xC3\xB1amensajeusuariotrabajom\xC3\xA9xicop\xC3\xA1ginasiempresistemaoctu" ~
    "bredurantea\xC3\xB1adirempresamomentonuestroprimeratrav\xC3\xA9sgraciasnuest" ~
    "raprocesoestadoscalidadpersonan\xC3\xBAmeroacuerdom\xC3\xBAsicamiembrooferta" ~
    "salgunospa\xC3\xADsesejemploderechoadem\xC3\xA1sprivadoagregarenlacesposible" ~
    "hotelessevillaprimero\xC3\xBAltimoeventosarchivoculturamujeresentradaa" ~
    "nuncioembargomercadograndesestudiomejoresfebrerodise\xC3\xB1oturismoc\xC3" ~
    "\xB3digoportadaespaciofamiliaantoniopermiteguardaralgunaspreciosalg" ~
    "uiensentidovisitast\xC3\xADtuloconocersegundoconsejofranciaminutossegu" ~
    "ndatenemosefectosm\xC3\xA1lagasesi\xC3\xB3nrevistagranadacompraringresogarc\xC3" ~
    "\xADaacci\xC3\xB3necuadorquienesinclusodeber\xC3\xA1materiahombresmuestrapodr\xC3\xAD" ~
    "ama\xC3\xB1ana\xC3\xBAltimaestamosoficialtambienning\xC3\xBAnsaludospodemosmejorar" ~
    "positionbusinesshomepagesecuritylanguagestandardcampaignfeatures" ~
    "categoryexternalchildrenreservedresearchexchangefavoritetemplate" ~
    "militaryindustryservicesmaterialproductsz-index:commentssoftware" ~
    "completecalendarplatformarticlesrequiredmovementquestionbuilding" ~
    "politicspossiblereligionphysicalfeedbackregisterpicturesdisabled" ~
    "protocolaudiencesettingsactivityelementslearninganythingabstract" ~
    "progressoverviewmagazineeconomictrainingpressurevarious <strong>" ~
    "propertyshoppingtogetheradvancedbehaviordownloadfeaturedfootball" ~
    "selectedLanguagedistanceremembertrackingpasswordmodifiedstudents" ~
    "directlyfightingnortherndatabasefestivalbreakinglocationinternet" ~
    "dropdownpracticeevidencefunctionmarriageresponseproblemsnegative" ~
    "programsanalysisreleasedbanner\">purchasepoliciesregionalcreative" ~
    "argumentbookmarkreferrerchemicaldivisioncallbackseparateprojects" ~
    "conflicthardwareinterestdeliverymountainobtained= false;for(var " ~
    "acceptedcapacitycomputeridentityaircraftemployedproposeddomestic" ~
    "includesprovidedhospitalverticalcollapseapproachpartnerslogo\"><a" ~
    "daughterauthor\" culturalfamilies/images/assemblypowerfulteaching" ~
    "finisheddistrictcriticalcgi-bin/purposesrequireselectionbecoming" ~
    "providesacademicexerciseactuallymedicineconstantaccidentMagazine" ~
    "documentstartingbottom\">observed: &quot;extendedpreviousSoftware" ~
    "customerdecisionstrengthdetailedslightlyplanningtextareacurrency" ~
    "everyonestraighttransferpositiveproducedheritageshippingabsolute" ~
    "receivedrelevantbutton\" violenceanywherebenefitslaunchedrecently" ~
    "alliancefollowedmultiplebulletinincludedoccurredinternal$(this)." ~
    "republic><tr><tdcongressrecordedultimatesolution<ul id=\"discover" ~
    "Home</a>websitesnetworksalthoughentirelymemorialmessagescontinue" ~
    "active\">somewhatvictoriaWestern  title=\"Locationcontractvisitors" ~
    "Downloadwithout right\">\x0Ameasureswidth = variableinvolvedvirginia" ~
    "normallyhappenedaccountsstandingnationalRegisterpreparedcontrols" ~
    "accuratebirthdaystrategyofficialgraphicscriminalpossiblyconsumer" ~
    "Personalspeakingvalidateachieved.jpg\" />machines</h2>\x0A  keywords" ~
    "friendlybrotherscombinedoriginalcomposedexpectedadequatepakistan" ~
    "follow\" valuable</label>relativebringingincreasegovernorplugins/" ~
    "List of Header\">\" name=\" (&quot;graduate</head>\x0Acommercemalaysia" ~
    "directormaintain;height:schedulechangingback to catholicpatterns" ~
    "color: #greatestsuppliesreliable</ul>\x0A\x09\x09<select citizensclothing" ~
    "watching<li id=\"specificcarryingsentence<center>contrastthinking" ~
    "catch(e)southernMichael merchantcarouselpadding:interior.split(\"" ~
    "lizationOctober ){returnimproved--&gt;\x0A\x0Acoveragechairman.png\" />" ~
    "subjectsRichard whateverprobablyrecoverybaseballjudgmentconnect." ~
    ".css\" /> websitereporteddefault\"/></a>\x0D\x0Aelectricscotlandcreation" ~
    "quantity. ISBN 0did not instance-search-\" lang=\"speakersComputer" ~
    "containsarchivesministerreactiondiscountItalianocriteriastrongly" ~
    ": 'http:'script'coveringofferingappearedBritish identifyFacebook" ~
    "numerousvehiclesconcernsAmericanhandlingdiv id=\"William provider" ~
    "_contentaccuracysection andersonflexibleCategorylawrence<script>" ~
    "layout=\"approved maximumheader\"></table>Serviceshamiltoncurrent " ~
    "canadianchannels/themes//articleoptionalportugalvalue=\"\"interval" ~
    "wirelessentitledagenciesSearch\" measuredthousandspending&hellip;" ~
    "new Date\" size=\"pageNamemiddle\" \" /></a>hidden\">sequencepersonal" ~
    "overflowopinionsillinoislinks\">\x0A\x09<title>versionssaturdayterminal" ~
    "itempropengineersectionsdesignerproposal=\"false\"Espa\xC3\xB1olreleases" ~
    "submit\" er&quot;additionsymptomsorientedresourceright\"><pleasure" ~
    "stationshistory.leaving  border=contentscenter\">.\x0A\x0ASome directed" ~
    "suitablebulgaria.show();designedGeneral conceptsExampleswilliams" ~
    "Original\"><span>search\">operatorrequestsa &quot;allowingDocument" ~
    "revision. \x0A\x0AThe yourselfContact michiganEnglish columbiapriority" ~
    "printingdrinkingfacilityreturnedContent officersRussian generate" ~
    "-8859-1\"indicatefamiliar qualitymargin:0 contentviewportcontacts" ~
    "-title\">portable.length eligibleinvolvesatlanticonload=\"default." ~
    "suppliedpaymentsglossary\x0A\x0AAfter guidance</td><tdencodingmiddle\">" ~
    "came to displaysscottishjonathanmajoritywidgets.clinicalthailand" ~
    "teachers<head>\x0A\x09affectedsupportspointer;toString</small>oklahoma" ~
    "will be investor0\" alt=\"holidaysResourcelicensed (which . After " ~
    "considervisitingexplorerprimary search\" android\"quickly meetings" ~
    "estimate;return ;color:# height=approval, &quot; checked.min.js\"" ~
    "magnetic></a></hforecast. While thursdaydvertise&eacute;hasClass" ~
    "evaluateorderingexistingpatients Online coloradoOptions\"campbell" ~
    "<!-- end</span><<br />\x0D\x0A_popups|sciences,&quot; quality Windows " ~
    "assignedheight: <b classle&quot; value=\" Companyexamples<iframe " ~
    "believespresentsmarshallpart of properly).\x0A\x0AThe taxonomymuch of " ~
    "</span>\x0A\" data-srtugu\xC3\xAAsscrollTo project<head>\x0D\x0Aattorneyemphasis" ~
    "sponsorsfancyboxworld's wildlifechecked=sessionsprogrammpx;font-" ~
    " Projectjournalsbelievedvacationthompsonlightingand the special " ~
    "border=0checking</tbody><button Completeclearfix\x0A<head>\x0Aarticle " ~
    "<sectionfindingsrole in popular  Octoberwebsite exposureused to " ~
    " changesoperatedclickingenteringcommandsinformed numbers  </div>" ~
    "creatingonSubmitmarylandcollegesanalyticlistingscontact.loggedIn" ~
    "advisorysiblingscontent\"s&quot;)s. This packagescheckboxsuggests" ~
    "pregnanttomorrowspacing=icon.pngjapanesecodebasebutton\">gambling" ~
    "such as , while </span> missourisportingtop:1px .</span>tensions" ~
    "width=\"2lazyloadnovemberused in height=\"cript\">\x0A&nbsp;</<tr><td " ~
    "height:2/productcountry include footer\" &lt;!-- title\"></jquery." ~
    "</form>\x0A(\xE7\xAE\x80\xE4\xBD\x93)(\xE7\xB9\x81\xE9\xAB\x94)hrvatskiitalianorom\xC3\xA2n\xC4\x83t\xC3\xBCrk\xC3\xA7e\xD8\xA7\xD8\xB1\xD8\xAF\xD9\x88" ~
    "tambi\xC3\xA9nnoticiasmensajespersonasderechosnacionalserviciocontacto" ~
    "usuariosprogramagobiernoempresasanunciosvalenciacolombiadespu\xC3\xA9s" ~
    "deportesproyectoproductop\xC3\xBAbliconosotroshistoriapresentemillones" ~
    "mediantepreguntaanteriorrecursosproblemasantiagonuestrosopini\xC3\xB3n" ~
    "imprimirmientrasam\xC3\xA9ricavendedorsociedadrespectorealizarregistro" ~
    "palabrasinter\xC3\xA9sentoncesespecialmiembrosrealidadc\xC3\xB3rdobazaragoza" ~
    "p\xC3\xA1ginassocialesbloqueargesti\xC3\xB3nalquilersistemascienciascompleto" ~
    "versi\xC3\xB3ncompletaestudiosp\xC3\xBAblicaobjetivoalicantebuscadorcantidad" ~
    "entradasaccionesarchivossuperiormayor\xC3\xADaalemaniafunci\xC3\xB3n\xC3\xBAltimos" ~
    "haciendoaquellosedici\xC3\xB3nfernandoambientefacebooknuestrasclientes" ~
    "procesosbastantepresentareportarcongresopublicarcomerciocontrato" ~
    "j\xC3\xB3venesdistritot\xC3\xA9cnicaconjuntoenerg\xC3\xADatrabajarasturiasreciente" ~
    "utilizarbolet\xC3\xADnsalvadorcorrectatrabajosprimerosnegocioslibertad" ~
    "detallespantallapr\xC3\xB3ximoalmer\xC3\xADaanimalesqui\xC3\xA9nescoraz\xC3\xB3nsecci\xC3\xB3n" ~
    "buscandoopcionesexteriorconceptotodav\xC3\xADagaler\xC3\xADaescribirmedicina" ~
    "licenciaconsultaaspectoscr\xC3\xADticad\xC3\xB3laresjusticiadeber\xC3\xA1nper\xC3\xADodo" ~
    "necesitamantenerpeque\xC3\xB1orecibidatribunaltenerifecanci\xC3\xB3ncanarias" ~
    "descargadiversosmallorcarequieret\xC3\xA9cnicodeber\xC3\xADaviviendafinanzas" ~
    "adelantefuncionaconsejosdif\xC3\xADcilciudadesantiguasavanzadat\xC3\xA9rmino" ~
    "unidadess\xC3\xA1nchezcampa\xC3\xB1asoftonicrevistascontienesectoresmomentos" ~
    "facultadcr\xC3\xA9ditodiversassupuestofactoressegundospeque\xC3\xB1a\xD0\xB3\xD0\xBE\xD0\xB4\xD0\xB0" ~
    "\xD0\xB5\xD1\x81\xD0\xBB\xD0\xB8\xD0\xB5\xD1\x81\xD1\x82\xD1\x8C\xD0\xB1\xD1\x8B\xD0\xBB\xD0\xBE\xD0\xB1\xD1\x8B\xD1\x82\xD1\x8C\xD1\x8D\xD1\x82\xD0\xBE\xD0\xBC\xD0\x95\xD1\x81\xD0\xBB\xD0\xB8\xD1\x82\xD0\xBE\xD0\xB3\xD0\xBE\xD0\xBC\xD0\xB5\xD0\xBD\xD1\x8F" ~
    "\xD0\xB2\xD1\x81\xD0\xB5\xD1\x85\xD1\x8D\xD1\x82\xD0\xBE\xD0\xB9\xD0\xB4\xD0\xB0\xD0\xB6\xD0\xB5\xD0\xB1\xD1\x8B\xD0\xBB\xD0\xB8\xD0\xB3\xD0\xBE\xD0\xB4\xD1\x83\xD0\xB4\xD0\xB5\xD0\xBD\xD1\x8C\xD1\x8D\xD1\x82\xD0\xBE\xD1\x82\xD0\xB1\xD1\x8B\xD0\xBB\xD0\xB0" ~
    "\xD1\x81\xD0\xB5\xD0\xB1\xD1\x8F\xD0\xBE\xD0\xB4\xD0\xB8\xD0\xBD\xD1\x81\xD0\xB5\xD0\xB1\xD0\xB5\xD0\xBD\xD0\xB0\xD0\xB4\xD0\xBE\xD1\x81\xD0\xB0\xD0\xB9\xD1\x82\xD1\x84\xD0\xBE\xD1\x82\xD0\xBE\xD0\xBD\xD0\xB5\xD0\xB3\xD0\xBE\xD1\x81\xD0\xB2\xD0\xBE\xD0\xB8" ~
    "\xD1\x81\xD0\xB2\xD0\xBE\xD0\xB9\xD0\xB8\xD0\xB3\xD1\x80\xD1\x8B\xD1\x82\xD0\xBE\xD0\xB6\xD0\xB5\xD0\xB2\xD1\x81\xD0\xB5\xD0\xBC\xD1\x81\xD0\xB2\xD0\xBE\xD1\x8E\xD0\xBB\xD0\xB8\xD1\x88\xD1\x8C\xD1\x8D\xD1\x82\xD0\xB8\xD1\x85\xD0\xBF\xD0\xBE\xD0\xBA\xD0\xB0" ~
    "\xD0\xB4\xD0\xBD\xD0\xB5\xD0\xB9\xD0\xB4\xD0\xBE\xD0\xBC\xD0\xB0\xD0\xBC\xD0\xB8\xD1\x80\xD0\xB0\xD0\xBB\xD0\xB8\xD0\xB1\xD0\xBE\xD1\x82\xD0\xB5\xD0\xBC\xD1\x83\xD1\x85\xD0\xBE\xD1\x82\xD1\x8F\xD0\xB4\xD0\xB2\xD1\x83\xD1\x85\xD1\x81\xD0\xB5\xD1\x82\xD0\xB8" ~
    "\xD0\xBB\xD1\x8E\xD0\xB4\xD0\xB8\xD0\xB4\xD0\xB5\xD0\xBB\xD0\xBE\xD0\xBC\xD0\xB8\xD1\x80\xD0\xB5\xD1\x82\xD0\xB5\xD0\xB1\xD1\x8F\xD1\x81\xD0\xB2\xD0\xBE\xD0\xB5\xD0\xB2\xD0\xB8\xD0\xB4\xD0\xB5\xD1\x87\xD0\xB5\xD0\xB3\xD0\xBE\xD1\x8D\xD1\x82\xD0\xB8\xD0\xBC" ~
    "\xD1\x81\xD1\x87\xD0\xB5\xD1\x82\xD1\x82\xD0\xB5\xD0\xBC\xD1\x8B\xD1\x86\xD0\xB5\xD0\xBD\xD1\x8B\xD1\x81\xD1\x82\xD0\xB0\xD0\xBB\xD0\xB2\xD0\xB5\xD0\xB4\xD1\x8C\xD1\x82\xD0\xB5\xD0\xBC\xD0\xB5\xD0\xB2\xD0\xBE\xD0\xB4\xD1\x8B\xD1\x82\xD0\xB5\xD0\xB1\xD0\xB5" ~
    "\xD0\xB2\xD1\x8B\xD1\x88\xD0\xB5\xD0\xBD\xD0\xB0\xD0\xBC\xD0\xB8\xD1\x82\xD0\xB8\xD0\xBF\xD0\xB0\xD1\x82\xD0\xBE\xD0\xBC\xD1\x83\xD0\xBF\xD1\x80\xD0\xB0\xD0\xB2\xD0\xBB\xD0\xB8\xD1\x86\xD0\xB0\xD0\xBE\xD0\xB4\xD0\xBD\xD0\xB0\xD0\xB3\xD0\xBE\xD0\xB4\xD1\x8B" ~
    "\xD0\xB7\xD0\xBD\xD0\xB0\xD1\x8E\xD0\xBC\xD0\xBE\xD0\xB3\xD1\x83\xD0\xB4\xD1\x80\xD1\x83\xD0\xB3\xD0\xB2\xD1\x81\xD0\xB5\xD0\xB9\xD0\xB8\xD0\xB4\xD0\xB5\xD1\x82\xD0\xBA\xD0\xB8\xD0\xBD\xD0\xBE\xD0\xBE\xD0\xB4\xD0\xBD\xD0\xBE\xD0\xB4\xD0\xB5\xD0\xBB\xD0\xB0" ~
    "\xD0\xB4\xD0\xB5\xD0\xBB\xD0\xB5\xD1\x81\xD1\x80\xD0\xBE\xD0\xBA\xD0\xB8\xD1\x8E\xD0\xBD\xD1\x8F\xD0\xB2\xD0\xB5\xD1\x81\xD1\x8C\xD0\x95\xD1\x81\xD1\x82\xD1\x8C\xD1\x80\xD0\xB0\xD0\xB7\xD0\xB0\xD0\xBD\xD0\xB0\xD1\x88\xD0\xB8\xD8\xA7\xD9\x84\xD9\x84\xD9\x87" ~
    "\xD8\xA7\xD9\x84\xD8\xAA\xD9\x8A\xD8\xAC\xD9\x85\xD9\x8A\xD8\xB9\xD8\xAE\xD8\xA7\xD8\xB5\xD8\xA9\xD8\xA7\xD9\x84\xD8\xB0\xD9\x8A\xD8\xB9\xD9\x84\xD9\x8A\xD9\x87\xD8\xAC\xD8\xAF\xD9\x8A\xD8\xAF\xD8\xA7\xD9\x84\xD8\xA2\xD9\x86\xD8\xA7\xD9\x84\xD8\xB1\xD8\xAF" ~
    "\xD8\xAA\xD8\xAD\xD9\x83\xD9\x85\xD8\xB5\xD9\x81\xD8\xAD\xD8\xA9\xD9\x83\xD8\xA7\xD9\x86\xD8\xAA\xD8\xA7\xD9\x84\xD9\x84\xD9\x8A\xD9\x8A\xD9\x83\xD9\x88\xD9\x86\xD8\xB4\xD8\xA8\xD9\x83\xD8\xA9\xD9\x81\xD9\x8A\xD9\x87\xD8\xA7\xD8\xA8\xD9\x86\xD8\xA7\xD8\xAA" ~
    "\xD8\xAD\xD9\x88\xD8\xA7\xD8\xA1\xD8\xA3\xD9\x83\xD8\xAB\xD8\xB1\xD8\xAE\xD9\x84\xD8\xA7\xD9\x84\xD8\xA7\xD9\x84\xD8\xAD\xD8\xA8\xD8\xAF\xD9\x84\xD9\x8A\xD9\x84\xD8\xAF\xD8\xB1\xD9\x88\xD8\xB3\xD8\xA7\xD8\xB6\xD8\xBA\xD8\xB7\xD8\xAA\xD9\x83\xD9\x88\xD9\x86" ~
    "\xD9\x87\xD9\x86\xD8\xA7\xD9\x83\xD8\xB3\xD8\xA7\xD8\xAD\xD8\xA9\xD9\x86\xD8\xA7\xD8\xAF\xD9\x8A\xD8\xA7\xD9\x84\xD8\xB7\xD8\xA8\xD8\xB9\xD9\x84\xD9\x8A\xD9\x83\xD8\xB4\xD9\x83\xD8\xB1\xD8\xA7\xD9\x8A\xD9\x85\xD9\x83\xD9\x86\xD9\x85\xD9\x86\xD9\x87\xD8\xA7" ~
    "\xD8\xB4\xD8\xB1\xD9\x83\xD8\xA9\xD8\xB1\xD8\xA6\xD9\x8A\xD8\xB3\xD9\x86\xD8\xB4\xD9\x8A\xD8\xB7\xD9\x85\xD8\xA7\xD8\xB0\xD8\xA7\xD8\xA7\xD9\x84\xD9\x81\xD9\x86\xD8\xB4\xD8\xA8\xD8\xA7\xD8\xA8\xD8\xAA\xD8\xB9\xD8\xA8\xD8\xB1\xD8\xB1\xD8\xAD\xD9\x85\xD8\xA9" ~
    "\xD9\x83\xD8\xA7\xD9\x81\xD8\xA9\xD9\x8A\xD9\x82\xD9\x88\xD9\x84\xD9\x85\xD8\xB1\xD9\x83\xD8\xB2\xD9\x83\xD9\x84\xD9\x85\xD8\xA9\xD8\xA3\xD8\xAD\xD9\x85\xD8\xAF\xD9\x82\xD9\x84\xD8\xA8\xD9\x8A\xD9\x8A\xD8\xB9\xD9\x86\xD9\x8A\xD8\xB5\xD9\x88\xD8\xB1\xD8\xA9" ~
    "\xD8\xB7\xD8\xB1\xD9\x8A\xD9\x82\xD8\xB4\xD8\xA7\xD8\xB1\xD9\x83\xD8\xAC\xD9\x88\xD8\xA7\xD9\x84\xD8\xA3\xD8\xAE\xD8\xB1\xD9\x89\xD9\x85\xD8\xB9\xD9\x86\xD8\xA7\xD8\xA7\xD8\xA8\xD8\xAD\xD8\xAB\xD8\xB9\xD8\xB1\xD9\x88\xD8\xB6\xD8\xA8\xD8\xB4\xD9\x83\xD9\x84" ~
    "\xD9\x85\xD8\xB3\xD8\xAC\xD9\x84\xD8\xA8\xD9\x86\xD8\xA7\xD9\x86\xD8\xAE\xD8\xA7\xD9\x84\xD8\xAF\xD9\x83\xD8\xAA\xD8\xA7\xD8\xA8\xD9\x83\xD9\x84\xD9\x8A\xD8\xA9\xD8\xA8\xD8\xAF\xD9\x88\xD9\x86\xD8\xA3\xD9\x8A\xD8\xB6\xD8\xA7\xD9\x8A\xD9\x88\xD8\xAC\xD8\xAF" ~
    "\xD9\x81\xD8\xB1\xD9\x8A\xD9\x82\xD9\x83\xD8\xAA\xD8\xA8\xD8\xAA\xD8\xA3\xD9\x81\xD8\xB6\xD9\x84\xD9\x85\xD8\xB7\xD8\xA8\xD8\xAE\xD8\xA7\xD9\x83\xD8\xAB\xD8\xB1\xD8\xA8\xD8\xA7\xD8\xB1\xD9\x83\xD8\xA7\xD9\x81\xD8\xB6\xD9\x84\xD8\xA7\xD8\xAD\xD9\x84\xD9\x89" ~
    "\xD9\x86\xD9\x81\xD8\xB3\xD9\x87\xD8\xA3\xD9\x8A\xD8\xA7\xD9\x85\xD8\xB1\xD8\xAF\xD9\x88\xD8\xAF\xD8\xA3\xD9\x86\xD9\x87\xD8\xA7\xD8\xAF\xD9\x8A\xD9\x86\xD8\xA7\xD8\xA7\xD9\x84\xD8\xA7\xD9\x86\xD9\x85\xD8\xB9\xD8\xB1\xD8\xB6\xD8\xAA\xD8\xB9\xD9\x84\xD9\x85" ~
    "\xD8\xAF\xD8\xA7\xD8\xAE\xD9\x84\xD9\x85\xD9\x85\xD9\x83\xD9\x86\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\x01\x00\x01\x00\x02\x00\x02\x00\x02\x00\x02\x00\x04\x00\x04\x00\x04\x00\x04\x00\x00\x01\x02\x03\x04\x05\x06\x07\x07\x06\x05\x04\x03\x02\x01\x00" ~
    "\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F\x0F\x0E\x0D\x0C\x0B\x0A\x09\x08\x10\x11\x12\x13\x14\x15\x16\x17\x17\x16\x15\x14\x13\x12\x11\x10\x18\x19\x1A\x1B\x1C\x1D\x1E\x1F\x1F\x1E\x1D\x1C\x1B\x1A\x19\x18\xFF\xFF\xFF\xFF\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF\xFF\xFF" ~
    "\x01\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00\xFF\xFF\x00\x01\x00\x00\x00\x01\x00\x00\xFF\xFF\x00\x01\x00\x00\x00\x08\x00\x08\x00\x08\x00\x08\x00\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07" ~
    "resourcescountriesquestionsequipmentcommunityavailablehighlightD" ~
    "TD/xhtmlmarketingknowledgesomethingcontainerdirectionsubscribead" ~
    "vertisecharacter\" value=\"</select>Australia\" class=\"situationaut" ~
    "horityfollowingprimarilyoperationchallengedevelopedanonymousfunc" ~
    "tion functionscompaniesstructureagreement\" title=\"potentialeduca" ~
    "tionargumentssecondarycopyrightlanguagesexclusivecondition</form" ~
    ">\x0D\x0AstatementattentionBiography} else {\x0Asolutionswhen the Analyti" ~
    "cstemplatesdangeroussatellitedocumentspublisherimportantprototyp" ~
    "einfluence&raquo;</effectivegenerallytransformbeautifultransport" ~
    "organizedpublishedprominentuntil thethumbnailNational .focus();o" ~
    "ver the migrationannouncedfooter\">\x0Aexceptionless thanexpensivefo" ~
    "rmationframeworkterritoryndicationcurrentlyclassNamecriticismtra" ~
    "ditionelsewhereAlexanderappointedmaterialsbroadcastmentionedaffi" ~
    "liate</option>treatmentdifferent/default.Presidentonclick=\"biogr" ~
    "aphyotherwisepermanentFran\xC3\xA7aisHollywoodexpansionstandards</styl" ~
    "e>\x0AreductionDecember preferredCambridgeopponentsBusiness confusi" ~
    "on>\x0A<title>presentedexplaineddoes not worldwideinterfaceposition" ~
    "snewspaper</table>\x0Amountainslike the essentialfinancialselection" ~
    "action=\"/abandonedEducationparseInt(stabilityunable to</title>\x0Ar" ~
    "elationsNote thatefficientperformedtwo yearsSince thethereforewr" ~
    "apper\">alternateincreasedBattle ofperceivedtrying tonecessarypor" ~
    "trayedelectionsElizabeth</iframe>discoveryinsurances.length;lege" ~
    "ndaryGeographycandidatecorporatesometimesservices.inherited</str" ~
    "ong>CommunityreligiouslocationsCommitteebuildingsthe worldno lon" ~
    "gerbeginningreferencecannot befrequencytypicallyinto the relativ" ~
    "e;recordingpresidentinitiallytechniquethe otherit can beexistenc" ~
    "eunderlinethis timetelephoneitemscopepracticesadvantage);return " ~
    "For otherprovidingdemocracyboth the extensivesufferingsupportedc" ~
    "omputers functionpracticalsaid thatit may beEnglish</from the sc" ~
    "heduleddownloads</label>\x0Asuspectedmargin: 0spiritual</head>\x0A\x0Amic" ~
    "rosoftgraduallydiscussedhe becameexecutivejquery.jshouseholdconf" ~
    "irmedpurchasedliterallydestroyedup to thevariationremainingit is" ~
    " notcenturiesJapanese among thecompletedalgorithminterestsrebell" ~
    "ionundefinedencourageresizableinvolvingsensitiveuniversalprovisi" ~
    "on(althoughfeaturingconducted), which continued-header\">February" ~
    " numerous overflow:componentfragmentsexcellentcolspan=\"technical" ~
    "near the Advanced source ofexpressedHong Kong Facebookmultiple m" ~
    "echanismelevationoffensive</form>\x0A\x09sponsoreddocument.or &quot;th" ~
    "ere arethose whomovementsprocessesdifficultsubmittedrecommendcon" ~
    "vincedpromoting\" width=\".replace(classicalcoalitionhis firstdeci" ~
    "sionsassistantindicatedevolution-wrapper\"enough toalong thedeliv" ~
    "ered-->\x0D\x0A<!--American protectedNovember </style><furnitureIntern" ~
    "et  onblur=\"suspendedrecipientbased on Moreover,abolishedcollect" ~
    "edwere madeemotionalemergencynarrativeadvocatespx;bordercommitte" ~
    "ddir=\"ltr\"employeesresearch. selectedsuccessorcustomersdisplayed" ~
    "SeptemberaddClass(Facebook suggestedand lateroperatingelaborateS" ~
    "ometimesInstitutecertainlyinstalledfollowersJerusalemthey haveco" ~
    "mputinggeneratedprovincesguaranteearbitraryrecognizewanted topx;" ~
    "width:theory ofbehaviourWhile theestimatedbegan to it becamemagn" ~
    "itudemust havemore thanDirectoryextensionsecretarynaturallyoccur" ~
    "ringvariablesgiven theplatform.</label><failed tocompoundskinds " ~
    "of societiesalongside --&gt;\x0A\x0Asouthwestthe rightradiationmay hav" ~
    "e unescape(spoken in\" href=\"/programmeonly the come fromdirector" ~
    "yburied ina similarthey were</font></Norwegianspecifiedproducing" ~
    "passenger(new DatetemporaryfictionalAfter theequationsdownload.r" ~
    "egularlydeveloperabove thelinked tophenomenaperiod oftooltip\">su" ~
    "bstanceautomaticaspect ofAmong theconnectedestimatesAir Forcesys" ~
    "tem ofobjectiveimmediatemaking itpaintingsconqueredare stillproc" ~
    "eduregrowth ofheaded byEuropean divisionsmoleculesfranchiseinten" ~
    "tionattractedchildhoodalso useddedicatedsingaporedegree offather" ~
    " ofconflicts</a></p>\x0Acame fromwere usednote thatreceivingExecuti" ~
    "veeven moreaccess tocommanderPoliticalmusiciansdeliciousprisoner" ~
    "sadvent ofUTF-8\" /><![CDATA[\">ContactSouthern bgcolor=\"series of" ~
    ". It was in Europepermittedvalidate.appearingofficialsseriously-" ~
    "languageinitiatedextendinglong-terminflationsuch thatgetCookiema" ~
    "rked by</button>implementbut it isincreasesdown the requiringdep" ~
    "endent-->\x0A<!-- interviewWith the copies ofconsensuswas builtVene" ~
    "zuela(formerlythe statepersonnelstrategicfavour ofinventionWikip" ~
    "ediacontinentvirtuallywhich wasprincipleComplete identicalshow t" ~
    "hatprimitiveaway frommolecularpreciselydissolvedUnder theversion" ~
    "=\">&nbsp;</It is the This is will haveorganismssome timeFriedric" ~
    "hwas firstthe only fact thatform id=\"precedingTechnicalphysicist" ~
    "occurs innavigatorsection\">span id=\"sought tobelow thesurviving}" ~
    "</style>his deathas in thecaused bypartiallyexisting using thewa" ~
    "s givena list oflevels ofnotion ofOfficial dismissedscientistres" ~
    "emblesduplicateexplosiverecoveredall othergalleries{padding:peop" ~
    "le ofregion ofaddressesassociateimg alt=\"in modernshould bemetho" ~
    "d ofreportingtimestampneeded tothe Greatregardingseemed toviewed" ~
    " asimpact onidea thatthe Worldheight ofexpandingThese arecurrent" ~
    "\">carefullymaintainscharge ofClassicaladdressedpredictedownershi" ~
    "p<div id=\"right\">\x0D\x0Aresidenceleave thecontent\">are often  })();\x0D\x0A" ~
    "probably Professor-button\" respondedsays thathad to beplaced inH" ~
    "ungarianstatus ofserves asUniversalexecutionaggregatefor whichin" ~
    "fectionagreed tohowever, popular\">placed onconstructelectoralsym" ~
    "bol ofincludingreturn toarchitectChristianprevious living ineasi" ~
    "er toprofessor\x0A&lt;!-- effect ofanalyticswas takenwhere thetook " ~
    "overbelief inAfrikaansas far aspreventedwork witha special<field" ~
    "setChristmasRetrieved\x0A\x0AIn the back intonortheastmagazines><stron" ~
    "g>committeegoverninggroups ofstored inestablisha generalits firs" ~
    "ttheir ownpopulatedan objectCaribbeanallow thedistrictswisconsin" ~
    "location.; width: inhabitedSocialistJanuary 1</footer>similarlyc" ~
    "hoice ofthe same specific business The first.length; desire tode" ~
    "al withsince theuserAgentconceivedindex.phpas &quot;engage inrec" ~
    "ently,few yearswere also\x0A<head>\x0A<edited byare knowncities inacce" ~
    "sskeycondemnedalso haveservices,family ofSchool ofconvertednatur" ~
    "e of languageministers</object>there is a popularsequencesadvoca" ~
    "tedThey wereany otherlocation=enter themuch morereflectedwas nam" ~
    "edoriginal a typicalwhen theyengineerscould notresidentswednesda" ~
    "ythe third productsJanuary 2what theya certainreactionsprocessor" ~
    "after histhe last contained\"></div>\x0A</a></td>depend onsearch\">\x0Ap" ~
    "ieces ofcompetingReferencetennesseewhich has version=</span> <</" ~
    "header>gives thehistorianvalue=\"\">padding:0view thattogether,the" ~
    " most was foundsubset ofattack onchildren,points ofpersonal posi" ~
    "tion:allegedlyClevelandwas laterand afterare givenwas stillscrol" ~
    "lingdesign ofmakes themuch lessAmericans.\x0A\x0AAfter , but theMuseum" ~
    " oflouisiana(from theminnesotaparticlesa processDominicanvolume " ~
    "ofreturningdefensive00px|righmade frommouseover\" style=\"states o" ~
    "f(which iscontinuesFranciscobuilding without awith somewho would" ~
    "a form ofa part ofbefore itknown as  Serviceslocation and oftenm" ~
    "easuringand it ispaperbackvalues of\x0D\x0A<title>= window.determineer" ~
    "&quot; played byand early</center>from thisthe threepower andof " ~
    "&quot;innerHTML<a href=\"y:inline;Church ofthe eventvery highoffi" ~
    "cial -height: content=\"/cgi-bin/to createafrikaansesperantofran\xC3" ~
    "\xA7aislatvie\xC5\xA1ulietuvi\xC5\xB3\xC4\x8Ce\xC5\xA1tina\xC4\x8De\xC5\xA1tina\xE0\xB9\x84\xE0\xB8\x97\xE0\xB8\xA2\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E\xE7\xAE\x80\xE4\xBD\x93" ~
    "\xE5\xAD\x97\xE7\xB9\x81\xE9\xAB\x94\xE5\xAD\x97\xED\x95\x9C\xEA\xB5\xAD\xEC\x96\xB4\xE4\xB8\xBA\xE4\xBB\x80\xE4\xB9\x88\xE8\xAE\xA1\xE7\xAE\x97\xE6\x9C\xBA\xE7\xAC\x94\xE8\xAE\xB0\xE6\x9C\xAC\xE8\xA8\x8E\xE8\xAB\x96\xE5\x8D\x80\xE6\x9C\x8D\xE5\x8A\xA1\xE5" ~
    "\x99\xA8\xE4\xBA\x92\xE8\x81\x94\xE7\xBD\x91\xE6\x88\xBF\xE5\x9C\xB0\xE4\xBA\xA7\xE4\xBF\xB1\xE4\xB9\x90\xE9\x83\xA8\xE5\x87\xBA\xE7\x89\x88\xE7\xA4\xBE\xE6\x8E\x92\xE8\xA1\x8C\xE6\xA6\x9C\xE9\x83\xA8\xE8\x90\xBD\xE6\xA0\xBC\xE8\xBF\x9B\xE4\xB8\x80\xE6\xAD" ~
    "\xA5\xE6\x94\xAF\xE4\xBB\x98\xE5\xAE\x9D\xE9\xAA\x8C\xE8\xAF\x81\xE7\xA0\x81\xE5\xA7\x94\xE5\x91\x98\xE4\xBC\x9A\xE6\x95\xB0\xE6\x8D\xAE\xE5\xBA\x93\xE6\xB6\x88\xE8\xB4\xB9\xE8\x80\x85\xE5\x8A\x9E\xE5\x85\xAC\xE5\xAE\xA4\xE8\xAE\xA8\xE8\xAE\xBA\xE5\x8C\xBA" ~
    "\xE6\xB7\xB1\xE5\x9C\xB3\xE5\xB8\x82\xE6\x92\xAD\xE6\x94\xBE\xE5\x99\xA8\xE5\x8C\x97\xE4\xBA\xAC\xE5\xB8\x82\xE5\xA4\xA7\xE5\xAD\xA6\xE7\x94\x9F\xE8\xB6\x8A\xE6\x9D\xA5\xE8\xB6\x8A\xE7\xAE\xA1\xE7\x90\x86\xE5\x91\x98\xE4\xBF\xA1\xE6\x81\xAF\xE7\xBD\x91s" ~
    "erviciosart\xC3\xADculoargentinabarcelonacualquierpublicadoproductospo" ~
    "l\xC3\xADticarespuestawikipediasiguienteb\xC3\xBAsquedacomunidadseguridadpri" ~
    "ncipalpreguntascontenidorespondervenezuelaproblemasdiciembrerela" ~
    "ci\xC3\xB3nnoviembresimilaresproyectosprogramasinstitutoactividadencue" ~
    "ntraeconom\xC3\xADaim\xC3\xA1genescontactardescargarnecesarioatenci\xC3\xB3ntel\xC3\xA9f" ~
    "onocomisi\xC3\xB3ncancionescapacidadencontraran\xC3\xA1lisisfavoritost\xC3\xA9rmin" ~
    "osprovinciaetiquetaselementosfuncionesresultadocar\xC3\xA1cterpropieda" ~
    "dprincipionecesidadmunicipalcreaci\xC3\xB3ndescargaspresenciacomercial" ~
    "opinionesejercicioeditorialsalamancagonz\xC3\xA1lezdocumentopel\xC3\xADcular" ~
    "ecientesgeneralestarragonapr\xC3\xA1cticanovedadespropuestapacientest\xC3" ~
    "\xA9cnicasobjetivoscontactos\xE0\xA4\xAE\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\xB2\xE0\xA4\xBF\xE0\xA4\x8F\xE0\xA4\xB9\xE0\xA5\x88\xE0\xA4\x82\xE0\xA4\x97\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xB8" ~
    "\xE0\xA4\xBE\xE0\xA4\xA5\xE0\xA4\x8F\xE0\xA4\xB5\xE0\xA4\x82\xE0\xA4\xB0\xE0\xA4\xB9\xE0\xA5\x87\xE0\xA4\x95\xE0\xA5\x8B\xE0\xA4\x88\xE0\xA4\x95\xE0\xA5\x81\xE0\xA4\x9B\xE0\xA4\xB0\xE0\xA4\xB9\xE0\xA4\xBE\xE0\xA4\xAC\xE0\xA4\xBE\xE0\xA4\xA6\xE0\xA4\x95\xE0" ~
    "\xA4\xB9\xE0\xA4\xBE\xE0\xA4\xB8\xE0\xA4\xAD\xE0\xA5\x80\xE0\xA4\xB9\xE0\xA5\x81\xE0\xA4\x8F\xE0\xA4\xB0\xE0\xA4\xB9\xE0\xA5\x80\xE0\xA4\xAE\xE0\xA5\x88\xE0\xA4\x82\xE0\xA4\xA6\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA4\xAC\xE0\xA4\xBE\xE0\xA4\xA4diplo" ~
    "docs\xE0\xA4\xB8\xE0\xA4\xAE\xE0\xA4\xAF\xE0\xA4\xB0\xE0\xA5\x82\xE0\xA4\xAA\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\xAE\xE0\xA4\xAA\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\xAB\xE0\xA4\xBF\xE0\xA4\xB0\xE0\xA4\x94\xE0\xA4\xB8\xE0\xA4\xA4\xE0\xA4\xA4\xE0\xA4\xB0" ~
    "\xE0\xA4\xB9\xE0\xA4\xB2\xE0\xA5\x8B\xE0\xA4\x97\xE0\xA4\xB9\xE0\xA5\x81\xE0\xA4\x86\xE0\xA4\xAC\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\xB6\xE0\xA4\xB9\xE0\xA5\x81\xE0\xA4\x88\xE0\xA4\x96\xE0\xA5\x87\xE0\xA4\xB2\xE0\xA4\xAF\xE0\xA4\xA6\xE0" ~
    "\xA4\xBF\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xAE\xE0\xA4\xB5\xE0\xA5\x87\xE0\xA4\xAC\xE0\xA4\xA4\xE0\xA5\x80\xE0\xA4\xA8\xE0\xA4\xAC\xE0\xA5\x80\xE0\xA4\x9A\xE0\xA4\xAE\xE0\xA5\x8C\xE0\xA4\xA4\xE0\xA4\xB8\xE0\xA4\xBE\xE0\xA4\xB2\xE0\xA4\xB2\xE0\xA5\x87\xE0\xA4" ~
    "\x96\xE0\xA4\x9C\xE0\xA5\x89\xE0\xA4\xAC\xE0\xA4\xAE\xE0\xA4\xA6\xE0\xA4\xA6\xE0\xA4\xA4\xE0\xA4\xA5\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA4\xB9\xE0\xA5\x80\xE0\xA4\xB6\xE0\xA4\xB9\xE0\xA4\xB0\xE0\xA4\x85\xE0\xA4\xB2\xE0\xA4\x97\xE0\xA4\x95\xE0\xA4\xAD\xE0\xA5\x80" ~
    "\xE0\xA4\xA8\xE0\xA4\x97\xE0\xA4\xB0\xE0\xA4\xAA\xE0\xA4\xBE\xE0\xA4\xB8\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\xA4\xE0\xA4\x95\xE0\xA4\xBF\xE0\xA4\x8F\xE0\xA4\x89\xE0\xA4\xB8\xE0\xA5\x87\xE0\xA4\x97\xE0\xA4\xAF\xE0\xA5\x80\xE0\xA4\xB9\xE0\xA5\x82\xE0\xA4\x81\xE0" ~
    "\xA4\x86\xE0\xA4\x97\xE0\xA5\x87\xE0\xA4\x9F\xE0\xA5\x80\xE0\xA4\xAE\xE0\xA4\x96\xE0\xA5\x8B\xE0\xA4\x9C\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\x85\xE0\xA4\xAD\xE0\xA5\x80\xE0\xA4\x97\xE0\xA4\xAF\xE0\xA5\x87\xE0\xA4\xA4\xE0\xA5\x81\xE0\xA4\xAE\xE0\xA4" ~
    "\xB5\xE0\xA5\x8B\xE0\xA4\x9F\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\x85\xE0\xA4\x97\xE0\xA4\xB0\xE0\xA4\x90\xE0\xA4\xB8\xE0\xA5\x87\xE0\xA4\xAE\xE0\xA5\x87\xE0\xA4\xB2\xE0\xA4\xB2\xE0\xA4\x97\xE0\xA4\xBE\xE0\xA4\xB9\xE0\xA4\xBE\xE0\xA4\xB2\xE0\xA4\x8A" ~
    "\xE0\xA4\xAA\xE0\xA4\xB0\xE0\xA4\x9A\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\x90\xE0\xA4\xB8\xE0\xA4\xBE\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\xB0\xE0\xA4\x9C\xE0\xA4\xBF\xE0\xA4\xB8\xE0\xA4\xA6\xE0\xA4\xBF\xE0\xA4\xB2\xE0\xA4\xAC\xE0\xA4\x82\xE0\xA4\xA6\xE0\xA4\xAC\xE0" ~
    "\xA4\xA8\xE0\xA4\xBE\xE0\xA4\xB9\xE0\xA5\x82\xE0\xA4\x82\xE0\xA4\xB2\xE0\xA4\xBE\xE0\xA4\x96\xE0\xA4\x9C\xE0\xA5\x80\xE0\xA4\xA4\xE0\xA4\xAC\xE0\xA4\x9F\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xBF\xE0\xA4\xB2\xE0\xA4\x87\xE0\xA4\xB8\xE0\xA5\x87\xE0\xA4\x86\xE0\xA4" ~
    "\xA8\xE0\xA5\x87\xE0\xA4\xA8\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\x95\xE0\xA5\x81\xE0\xA4\xB2\xE0\xA4\xB2\xE0\xA5\x89\xE0\xA4\x97\xE0\xA4\xAD\xE0\xA4\xBE\xE0\xA4\x97\xE0\xA4\xB0\xE0\xA5\x87\xE0\xA4\xB2\xE0\xA4\x9C\xE0\xA4\x97\xE0\xA4\xB9\xE0\xA4\xB0\xE0\xA4\xBE" ~
    "\xE0\xA4\xAE\xE0\xA4\xB2\xE0\xA4\x97\xE0\xA5\x87\xE0\xA4\xAA\xE0\xA5\x87\xE0\xA4\x9C\xE0\xA4\xB9\xE0\xA4\xBE\xE0\xA4\xA5\xE0\xA4\x87\xE0\xA4\xB8\xE0\xA5\x80\xE0\xA4\xB8\xE0\xA4\xB9\xE0\xA5\x80\xE0\xA4\x95\xE0\xA4\xB2\xE0\xA4\xBE\xE0\xA4\xA0\xE0\xA5\x80\xE0" ~
    "\xA4\x95\xE0\xA4\xB9\xE0\xA4\xBE\xE0\xA4\x81\xE0\xA4\xA6\xE0\xA5\x82\xE0\xA4\xB0\xE0\xA4\xA4\xE0\xA4\xB9\xE0\xA4\xA4\xE0\xA4\xB8\xE0\xA4\xBE\xE0\xA4\xA4\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xA6\xE0\xA4\x86\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xAA\xE0\xA4\xBE\xE0\xA4" ~
    "\x95\xE0\xA4\x95\xE0\xA5\x8C\xE0\xA4\xA8\xE0\xA4\xB6\xE0\xA4\xBE\xE0\xA4\xAE\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\x96\xE0\xA4\xAF\xE0\xA4\xB9\xE0\xA5\x80\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\xAF\xE0\xA4\x96\xE0\xA5\x81\xE0\xA4\xA6\xE0\xA4\xB2\xE0\xA4\x97\xE0\xA5\x80" ~
    "categoriesexperience</title>\x0D\x0ACopyright javascriptconditionsever" ~
    "ything<p class=\"technologybackground<a class=\"management&copy; 2" ~
    "01javaScriptcharactersbreadcrumbthemselveshorizontalgovernmentCa" ~
    "liforniaactivitiesdiscoveredNavigationtransitionconnectionnaviga" ~
    "tionappearance</title><mcheckbox\" techniquesprotectionapparently" ~
    "as well asunt', 'UA-resolutionoperationstelevisiontranslatedWash" ~
    "ingtonnavigator. = window.impression&lt;br&gt;literaturepopulati" ~
    "onbgcolor=\"#especially content=\"productionnewsletterpropertiesde" ~
    "finitionleadershipTechnologyParliamentcomparisonul class=\".index" ~
    "Of(\"conclusiondiscussioncomponentsbiologicalRevolution_container" ~
    "understoodnoscript><permissioneach otheratmosphere onfocus=\"<for" ~
    "m id=\"processingthis.valuegenerationConferencesubsequentwell-kno" ~
    "wnvariationsreputationphenomenondisciplinelogo.png\" (document,bo" ~
    "undariesexpressionsettlementBackgroundout of theenterprise(\"http" ~
    "s:\" unescape(\"password\" democratic<a href=\"/wrapper\">\x0Amembership" ~
    "linguisticpx;paddingphilosophyassistanceuniversityfacilitiesreco" ~
    "gnizedpreferenceif (typeofmaintainedvocabularyhypothesis.submit(" ~
    ");&amp;nbsp;annotationbehind theFoundationpublisher\"assumptionin" ~
    "troducedcorruptionscientistsexplicitlyinstead ofdimensions onCli" ~
    "ck=\"considereddepartmentoccupationsoon afterinvestmentpronounced" ~
    "identifiedexperimentManagementgeographic\" height=\"link rel=\".rep" ~
    "lace(/depressionconferencepunishmenteliminatedresistanceadaptati" ~
    "onoppositionwell knownsupplementdeterminedh1 class=\"0px;marginme" ~
    "chanicalstatisticscelebratedGovernment\x0A\x0ADuring tdevelopersartifi" ~
    "cialequivalentoriginatedCommissionattachment<span id=\"there were" ~
    "Nederlandsbeyond theregisteredjournalistfrequentlyall of thelang" ~
    "=\"en\" </style>\x0D\x0Aabsolute; supportingextremely mainstream</strong" ~
    "> popularityemployment</table>\x0D\x0A colspan=\"</form>\x0A  conversionab" ~
    "out the </p></div>integrated\" lang=\"enPortuguesesubstituteindivi" ~
    "dualimpossiblemultimediaalmost allpx solid #apart fromsubject to" ~
    "in Englishcriticizedexcept forguidelinesoriginallyremarkablethe " ~
    "secondh2 class=\"<a title=\"(includingparametersprohibited= \"http:" ~
    "//dictionaryperceptionrevolutionfoundationpx;height:successfulsu" ~
    "pportersmillenniumhis fatherthe &quot;no-repeat;commercialindust" ~
    "rialencouragedamount of unofficialefficiencyReferencescoordinate" ~
    "disclaimerexpeditiondevelopingcalculatedsimplifiedlegitimatesubs" ~
    "tring(0\" class=\"completelyillustratefive yearsinstrumentPublishi" ~
    "ng1\" class=\"psychologyconfidencenumber of absence offocused onjo" ~
    "ined thestructurespreviously></iframe>once againbut ratherimmigr" ~
    "antsof course,a group ofLiteratureUnlike the</a>&nbsp;\x0Afunction " ~
    "it was theConventionautomobileProtestantaggressiveafter the Simi" ~
    "larly,\" /></div>collection\x0D\x0Afunctionvisibilitythe use ofvoluntee" ~
    "rsattractionunder the threatened*<![CDATA[importancein generalth" ~
    "e latter</form>\x0A</.indexOf('i = 0; i <differencedevoted totradit" ~
    "ionssearch forultimatelytournamentattributesso-called }\x0A</style>" ~
    "evaluationemphasizedaccessible</section>successionalong withMean" ~
    "while,industries</a><br />has becomeaspects ofTelevisionsufficie" ~
    "ntbasketballboth sidescontinuingan article<img alt=\"adventureshi" ~
    "s mothermanchesterprinciplesparticularcommentaryeffects ofdecide" ~
    "d to\"><strong>publishersJournal ofdifficultyfacilitateacceptable" ~
    "style.css\"\x09function innovation>Copyrightsituationswould havebusi" ~
    "nessesDictionarystatementsoften usedpersistentin Januarycomprisi" ~
    "ng</title>\x0A\x09diplomaticcontainingperformingextensionsmay not beco" ~
    "ncept of onclick=\"It is alsofinancial making theLuxembourgadditi" ~
    "onalare calledengaged in\"script\");but it waselectroniconsubmit=\"" ~
    "\x0A<!-- End electricalofficiallysuggestiontop of theunlike theAust" ~
    "ralianOriginallyreferences\x0A</head>\x0D\x0Arecognisedinitializelimited " ~
    "toAlexandriaretirementAdventuresfour years\x0A\x0A&lt;!-- increasingde" ~
    "corationh3 class=\"origins ofobligationregulationclassified(funct" ~
    "ion(advantagesbeing the historians<base hrefrepeatedlywilling to" ~
    "comparabledesignatednominationfunctionalinside therevelationend " ~
    "of thes for the authorizedrefused totake placeautonomouscompromi" ~
    "sepolitical restauranttwo of theFebruary 2quality ofswfobject.un" ~
    "derstandnearly allwritten byinterviews\" width=\"1withdrawalfloat:" ~
    "leftis usuallycandidatesnewspapersmysteriousDepartmentbest known" ~
    "parliamentsuppressedconvenientremembereddifferent systematichas " ~
    "led topropagandacontrolledinfluencesceremonialproclaimedProtecti" ~
    "onli class=\"Scientificclass=\"no-trademarksmore than widespreadLi" ~
    "berationtook placeday of theas long asimprisonedAdditional\x0A<head" ~
    ">\x0A<mLaboratoryNovember 2exceptionsIndustrialvariety offloat: lef" ~
    "During theassessmenthave been deals withStatisticsoccurrence/ul>" ~
    "</div>clearfix\">the publicmany yearswhich wereover time,synonymo" ~
    "uscontent\">\x0Apresumablyhis familyuserAgent.unexpectedincluding ch" ~
    "allengeda minorityundefined\"belongs totaken fromin Octoberpositi" ~
    "on: said to bereligious Federation rowspan=\"only a fewmeant that" ~
    "led to the-->\x0D\x0A<div <fieldset>Archbishop class=\"nobeing usedappr" ~
    "oachesprivilegesnoscript>\x0Aresults inmay be theEaster eggmechanis" ~
    "msreasonablePopulationCollectionselected\">noscript>\x0D/index.phpar" ~
    "rival of-jssdk'));managed toincompletecasualtiescompletionChrist" ~
    "iansSeptember arithmeticproceduresmight haveProductionit appears" ~
    "Philosophyfriendshipleading togiving thetoward theguaranteeddocu" ~
    "mentedcolor:#000video gamecommissionreflectingchange theassociat" ~
    "edsans-serifonkeypress; padding:He was theunderlyingtypically , " ~
    "and the srcElementsuccessivesince the should be networkingaccoun" ~
    "tinguse of thelower thanshows that</span>\x0A\x09\x09complaintscontinuous" ~
    "quantitiesastronomerhe did notdue to itsapplied toan averageeffo" ~
    "rts tothe futureattempt toTherefore,capabilityRepublicanwas form" ~
    "edElectronickilometerschallengespublishingthe formerindigenousdi" ~
    "rectionssubsidiaryconspiracydetails ofand in theaffordablesubsta" ~
    "ncesreason forconventionitemtype=\"absolutelysupposedlyremained a" ~
    "attractivetravellingseparatelyfocuses onelementaryapplicablefoun" ~
    "d thatstylesheetmanuscriptstands for no-repeat(sometimesCommerci" ~
    "alin Americaundertakenquarter ofan examplepersonallyindex.php?</" ~
    "button>\x0Apercentagebest-knowncreating a\" dir=\"ltrLieutenant\x0A<div " ~
    "id=\"they wouldability ofmade up ofnoted thatclear thatargue that" ~
    "to anotherchildren'spurpose offormulatedbased uponthe regionsubj" ~
    "ect ofpassengerspossession.\x0A\x0AIn the Before theafterwardscurrentl" ~
    "y across thescientificcommunity.capitalismin Germanyright-wingth" ~
    "e systemSociety ofpoliticiandirection:went on toremoval of New Y" ~
    "ork apartmentsindicationduring theunless thehistoricalhad been a" ~
    "definitiveingredientattendanceCenter forprominencereadyStatestra" ~
    "tegiesbut in theas part ofconstituteclaim thatlaboratorycompatib" ~
    "lefailure of, such as began withusing the to providefeature offr" ~
    "om which/\" class=\"geologicalseveral ofdeliberateimportant holds " ~
    "thating&quot; valign=topthe Germanoutside ofnegotiatedhis career" ~
    "separationid=\"searchwas calledthe fourthrecreationother thanprev" ~
    "entionwhile the education,connectingaccuratelywere builtwas kill" ~
    "edagreementsmuch more Due to thewidth: 100some otherKingdom ofth" ~
    "e entirefamous forto connectobjectivesthe Frenchpeople andfeatur" ~
    "ed\">is said tostructuralreferendummost oftena separate->\x0A<div id" ~
    " Official worldwide.aria-labelthe planetand it wasd\" value=\"look" ~
    "ing atbeneficialare in themonitoringreportedlythe modernworking " ~
    "onallowed towhere the innovative</a></div>soundtracksearchFormte" ~
    "nd to beinput id=\"opening ofrestrictedadopted byaddressingtheolo" ~
    "gianmethods ofvariant ofChristian very largeautomotiveby far the" ~
    "range frompursuit offollow thebrought toin Englandagree thataccu" ~
    "sed ofcomes frompreventingdiv style=his or hertremendousfreedom " ~
    "ofconcerning0 1em 1em;Basketball/style.cssan earliereven after/\"" ~
    " title=\".com/indextaking thepittsburghcontent\">\x0D<script>(fturned" ~
    " outhaving the</span>\x0D\x0A occasionalbecause itstarted tophysically" ~
    "></div>\x0A  created byCurrently, bgcolor=\"tabindex=\"disastrousAnal" ~
    "ytics also has a><div id=\"</style>\x0A<called forsinger and.src = \"" ~
    "//violationsthis pointconstantlyis locatedrecordingsd from thene" ~
    "derlandsportugu\xC3\xAAs\xD7\xA2\xD7\x91\xD7\xA8\xD7\x99\xD7\xAA\xD9\x81\xD8\xA7\xD8\xB1\xD8\xB3\xDB\x8Cdesarrollocomentarioeducac" ~
    "i\xC3\xB3nseptiembreregistradodirecci\xC3\xB3nubicaci\xC3\xB3npublicidadrespuestas" ~
    "resultadosimportantereservadosart\xC3\xADculosdiferentessiguientesrep\xC3" ~
    "\xBAblicasituaci\xC3\xB3nministerioprivacidaddirectorioformaci\xC3\xB3npoblaci\xC3" ~
    "\xB3npresidentecontenidosaccesoriostechnoratipersonalescategor\xC3\xADaes" ~
    "pecialesdisponibleactualidadreferenciavalladolidbibliotecarelaci" ~
    "onescalendariopol\xC3\xADticasanterioresdocumentosnaturalezamateriales" ~
    "diferenciaecon\xC3\xB3micatransporterodr\xC3\xADguezparticiparencuentrandisc" ~
    "usi\xC3\xB3nestructurafundaci\xC3\xB3nfrecuentespermanentetotalmente\xD0\xBC\xD0\xBE\xD0\xB6\xD0\xBD" ~
    "\xD0\xBE\xD0\xB1\xD1\x83\xD0\xB4\xD0\xB5\xD1\x82\xD0\xBC\xD0\xBE\xD0\xB6\xD0\xB5\xD1\x82\xD0\xB2\xD1\x80\xD0\xB5\xD0\xBC\xD1\x8F\xD1\x82\xD0\xB0\xD0\xBA\xD0\xB6\xD0\xB5\xD1\x87\xD1\x82\xD0\xBE\xD0\xB1\xD1\x8B\xD0\xB1\xD0\xBE\xD0\xBB\xD0\xB5\xD0\xB5\xD0\xBE" ~
    "\xD1\x87\xD0\xB5\xD0\xBD\xD1\x8C\xD1\x8D\xD1\x82\xD0\xBE\xD0\xB3\xD0\xBE\xD0\xBA\xD0\xBE\xD0\xB3\xD0\xB4\xD0\xB0\xD0\xBF\xD0\xBE\xD1\x81\xD0\xBB\xD0\xB5\xD0\xB2\xD1\x81\xD0\xB5\xD0\xB3\xD0\xBE\xD1\x81\xD0\xB0\xD0\xB9\xD1\x82\xD0\xB5\xD1\x87\xD0\xB5\xD1\x80" ~
    "\xD0\xB5\xD0\xB7\xD0\xBC\xD0\xBE\xD0\xB3\xD1\x83\xD1\x82\xD1\x81\xD0\xB0\xD0\xB9\xD1\x82\xD0\xB0\xD0\xB6\xD0\xB8\xD0\xB7\xD0\xBD\xD0\xB8\xD0\xBC\xD0\xB5\xD0\xB6\xD0\xB4\xD1\x83\xD0\xB1\xD1\x83\xD0\xB4\xD1\x83\xD1\x82\xD0\x9F\xD0\xBE\xD0\xB8\xD1\x81\xD0\xBA" ~
    "\xD0\xB7\xD0\xB4\xD0\xB5\xD1\x81\xD1\x8C\xD0\xB2\xD0\xB8\xD0\xB4\xD0\xB5\xD0\xBE\xD1\x81\xD0\xB2\xD1\x8F\xD0\xB7\xD0\xB8\xD0\xBD\xD1\x83\xD0\xB6\xD0\xBD\xD0\xBE\xD1\x81\xD0\xB2\xD0\xBE\xD0\xB5\xD0\xB9\xD0\xBB\xD1\x8E\xD0\xB4\xD0\xB5\xD0\xB9\xD0\xBF\xD0\xBE" ~
    "\xD1\x80\xD0\xBD\xD0\xBE\xD0\xBC\xD0\xBD\xD0\xBE\xD0\xB3\xD0\xBE\xD0\xB4\xD0\xB5\xD1\x82\xD0\xB5\xD0\xB9\xD1\x81\xD0\xB2\xD0\xBE\xD0\xB8\xD1\x85\xD0\xBF\xD1\x80\xD0\xB0\xD0\xB2\xD0\xB0\xD1\x82\xD0\xB0\xD0\xBA\xD0\xBE\xD0\xB9\xD0\xBC\xD0\xB5\xD1\x81\xD1\x82" ~
    "\xD0\xBE\xD0\xB8\xD0\xBC\xD0\xB5\xD0\xB5\xD1\x82\xD0\xB6\xD0\xB8\xD0\xB7\xD0\xBD\xD1\x8C\xD0\xBE\xD0\xB4\xD0\xBD\xD0\xBE\xD0\xB9\xD0\xBB\xD1\x83\xD1\x87\xD1\x88\xD0\xB5\xD0\xBF\xD0\xB5\xD1\x80\xD0\xB5\xD0\xB4\xD1\x87\xD0\xB0\xD1\x81\xD1\x82\xD0\xB8\xD1\x87" ~
    "\xD0\xB0\xD1\x81\xD1\x82\xD1\x8C\xD1\x80\xD0\xB0\xD0\xB1\xD0\xBE\xD1\x82\xD0\xBD\xD0\xBE\xD0\xB2\xD1\x8B\xD1\x85\xD0\xBF\xD1\x80\xD0\xB0\xD0\xB2\xD0\xBE\xD1\x81\xD0\xBE\xD0\xB1\xD0\xBE\xD0\xB9\xD0\xBF\xD0\xBE\xD1\x82\xD0\xBE\xD0\xBC\xD0\xBC\xD0\xB5\xD0\xBD" ~
    "\xD0\xB5\xD0\xB5\xD1\x87\xD0\xB8\xD1\x81\xD0\xBB\xD0\xB5\xD0\xBD\xD0\xBE\xD0\xB2\xD1\x8B\xD0\xB5\xD1\x83\xD1\x81\xD0\xBB\xD1\x83\xD0\xB3\xD0\xBE\xD0\xBA\xD0\xBE\xD0\xBB\xD0\xBE\xD0\xBD\xD0\xB0\xD0\xB7\xD0\xB0\xD0\xB4\xD1\x82\xD0\xB0\xD0\xBA\xD0\xBE\xD0\xB5" ~
    "\xD1\x82\xD0\xBE\xD0\xB3\xD0\xB4\xD0\xB0\xD0\xBF\xD0\xBE\xD1\x87\xD1\x82\xD0\xB8\xD0\x9F\xD0\xBE\xD1\x81\xD0\xBB\xD0\xB5\xD1\x82\xD0\xB0\xD0\xBA\xD0\xB8\xD0\xB5\xD0\xBD\xD0\xBE\xD0\xB2\xD1\x8B\xD0\xB9\xD1\x81\xD1\x82\xD0\xBE\xD0\xB8\xD1\x82\xD1\x82\xD0\xB0" ~
    "\xD0\xBA\xD0\xB8\xD1\x85\xD1\x81\xD1\x80\xD0\xB0\xD0\xB7\xD1\x83\xD0\xA1\xD0\xB0\xD0\xBD\xD0\xBA\xD1\x82\xD1\x84\xD0\xBE\xD1\x80\xD1\x83\xD0\xBC\xD0\x9A\xD0\xBE\xD0\xB3\xD0\xB4\xD0\xB0\xD0\xBA\xD0\xBD\xD0\xB8\xD0\xB3\xD0\xB8\xD1\x81\xD0\xBB\xD0\xBE\xD0\xB2" ~
    "\xD0\xB0\xD0\xBD\xD0\xB0\xD1\x88\xD0\xB5\xD0\xB9\xD0\xBD\xD0\xB0\xD0\xB9\xD1\x82\xD0\xB8\xD1\x81\xD0\xB2\xD0\xBE\xD0\xB8\xD0\xBC\xD1\x81\xD0\xB2\xD1\x8F\xD0\xB7\xD1\x8C\xD0\xBB\xD1\x8E\xD0\xB1\xD0\xBE\xD0\xB9\xD1\x87\xD0\xB0\xD1\x81\xD1\x82\xD0\xBE\xD1\x81" ~
    "\xD1\x80\xD0\xB5\xD0\xB4\xD0\xB8\xD0\x9A\xD1\x80\xD0\xBE\xD0\xBC\xD0\xB5\xD0\xA4\xD0\xBE\xD1\x80\xD1\x83\xD0\xBC\xD1\x80\xD1\x8B\xD0\xBD\xD0\xBA\xD0\xB5\xD1\x81\xD1\x82\xD0\xB0\xD0\xBB\xD0\xB8\xD0\xBF\xD0\xBE\xD0\xB8\xD1\x81\xD0\xBA\xD1\x82\xD1\x8B\xD1\x81" ~
    "\xD1\x8F\xD1\x87\xD0\xBC\xD0\xB5\xD1\x81\xD1\x8F\xD1\x86\xD1\x86\xD0\xB5\xD0\xBD\xD1\x82\xD1\x80\xD1\x82\xD1\x80\xD1\x83\xD0\xB4\xD0\xB0\xD1\x81\xD0\xB0\xD0\xBC\xD1\x8B\xD1\x85\xD1\x80\xD1\x8B\xD0\xBD\xD0\xBA\xD0\xB0\xD0\x9D\xD0\xBE\xD0\xB2\xD1\x8B\xD0\xB9" ~
    "\xD1\x87\xD0\xB0\xD1\x81\xD0\xBE\xD0\xB2\xD0\xBC\xD0\xB5\xD1\x81\xD1\x82\xD0\xB0\xD1\x84\xD0\xB8\xD0\xBB\xD1\x8C\xD0\xBC\xD0\xBC\xD0\xB0\xD1\x80\xD1\x82\xD0\xB0\xD1\x81\xD1\x82\xD1\x80\xD0\xB0\xD0\xBD\xD0\xBC\xD0\xB5\xD1\x81\xD1\x82\xD0\xB5\xD1\x82\xD0\xB5" ~
    "\xD0\xBA\xD1\x81\xD1\x82\xD0\xBD\xD0\xB0\xD1\x88\xD0\xB8\xD1\x85\xD0\xBC\xD0\xB8\xD0\xBD\xD1\x83\xD1\x82\xD0\xB8\xD0\xBC\xD0\xB5\xD0\xBD\xD0\xB8\xD0\xB8\xD0\xBC\xD0\xB5\xD1\x8E\xD1\x82\xD0\xBD\xD0\xBE\xD0\xBC\xD0\xB5\xD1\x80\xD0\xB3\xD0\xBE\xD1\x80\xD0\xBE" ~
    "\xD0\xB4\xD1\x81\xD0\xB0\xD0\xBC\xD0\xBE\xD0\xBC\xD1\x8D\xD1\x82\xD0\xBE\xD0\xBC\xD1\x83\xD0\xBA\xD0\xBE\xD0\xBD\xD1\x86\xD0\xB5\xD1\x81\xD0\xB2\xD0\xBE\xD0\xB5\xD0\xBC\xD0\xBA\xD0\xB0\xD0\xBA\xD0\xBE\xD0\xB9\xD0\x90\xD1\x80\xD1\x85\xD0\xB8\xD0\xB2\xD9\x85" ~
    "\xD9\x86\xD8\xAA\xD8\xAF\xD9\x89\xD8\xA5\xD8\xB1\xD8\xB3\xD8\xA7\xD9\x84\xD8\xB1\xD8\xB3\xD8\xA7\xD9\x84\xD8\xA9\xD8\xA7\xD9\x84\xD8\xB9\xD8\xA7\xD9\x85\xD9\x83\xD8\xAA\xD8\xA8\xD9\x87\xD8\xA7\xD8\xA8\xD8\xB1\xD8\xA7\xD9\x85\xD8\xAC\xD8\xA7\xD9\x84\xD9\x8A" ~
    "\xD9\x88\xD9\x85\xD8\xA7\xD9\x84\xD8\xB5\xD9\x88\xD8\xB1\xD8\xAC\xD8\xAF\xD9\x8A\xD8\xAF\xD8\xA9\xD8\xA7\xD9\x84\xD8\xB9\xD8\xB6\xD9\x88\xD8\xA5\xD8\xB6\xD8\xA7\xD9\x81\xD8\xA9\xD8\xA7\xD9\x84\xD9\x82\xD8\xB3\xD9\x85\xD8\xA7\xD9\x84\xD8\xB9\xD8\xA7\xD8\xA8" ~
    "\xD8\xAA\xD8\xAD\xD9\x85\xD9\x8A\xD9\x84\xD9\x85\xD9\x84\xD9\x81\xD8\xA7\xD8\xAA\xD9\x85\xD9\x84\xD8\xAA\xD9\x82\xD9\x89\xD8\xAA\xD8\xB9\xD8\xAF\xD9\x8A\xD9\x84\xD8\xA7\xD9\x84\xD8\xB4\xD8\xB9\xD8\xB1\xD8\xA3\xD8\xAE\xD8\xA8\xD8\xA7\xD8\xB1\xD8\xAA\xD8\xB7" ~
    "\xD9\x88\xD9\x8A\xD8\xB1\xD8\xB9\xD9\x84\xD9\x8A\xD9\x83\xD9\x85\xD8\xA5\xD8\xB1\xD9\x81\xD8\xA7\xD9\x82\xD8\xB7\xD9\x84\xD8\xA8\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD9\x84\xD8\xBA\xD8\xA9\xD8\xAA\xD8\xB1\xD8\xAA\xD9\x8A\xD8\xA8\xD8\xA7\xD9\x84\xD9\x86\xD8\xA7" ~
    "\xD8\xB3\xD8\xA7\xD9\x84\xD8\xB4\xD9\x8A\xD8\xAE\xD9\x85\xD9\x86\xD8\xAA\xD8\xAF\xD9\x8A\xD8\xA7\xD9\x84\xD8\xB9\xD8\xB1\xD8\xA8\xD8\xA7\xD9\x84\xD9\x82\xD8\xB5\xD8\xB5\xD8\xA7\xD9\x81\xD9\x84\xD8\xA7\xD9\x85\xD8\xB9\xD9\x84\xD9\x8A\xD9\x87\xD8\xA7\xD8\xAA" ~
    "\xD8\xAD\xD8\xAF\xD9\x8A\xD8\xAB\xD8\xA7\xD9\x84\xD9\x84\xD9\x87\xD9\x85\xD8\xA7\xD9\x84\xD8\xB9\xD9\x85\xD9\x84\xD9\x85\xD9\x83\xD8\xAA\xD8\xA8\xD8\xA9\xD9\x8A\xD9\x85\xD9\x83\xD9\x86\xD9\x83\xD8\xA7\xD9\x84\xD8\xB7\xD9\x81\xD9\x84\xD9\x81\xD9\x8A\xD8\xAF" ~
    "\xD9\x8A\xD9\x88\xD8\xA5\xD8\xAF\xD8\xA7\xD8\xB1\xD8\xA9\xD8\xAA\xD8\xA7\xD8\xB1\xD9\x8A\xD8\xAE\xD8\xA7\xD9\x84\xD8\xB5\xD8\xAD\xD8\xA9\xD8\xAA\xD8\xB3\xD8\xAC\xD9\x8A\xD9\x84\xD8\xA7\xD9\x84\xD9\x88\xD9\x82\xD8\xAA\xD8\xB9\xD9\x86\xD8\xAF\xD9\x85\xD8\xA7" ~
    "\xD9\x85\xD8\xAF\xD9\x8A\xD9\x86\xD8\xA9\xD8\xAA\xD8\xB5\xD9\x85\xD9\x8A\xD9\x85\xD8\xA3\xD8\xB1\xD8\xB4\xD9\x8A\xD9\x81\xD8\xA7\xD9\x84\xD8\xB0\xD9\x8A\xD9\x86\xD8\xB9\xD8\xB1\xD8\xA8\xD9\x8A\xD8\xA9\xD8\xA8\xD9\x88\xD8\xA7\xD8\xA8\xD8\xA9\xD8\xA3\xD9\x84" ~
    "\xD8\xB9\xD8\xA7\xD8\xA8\xD8\xA7\xD9\x84\xD8\xB3\xD9\x81\xD8\xB1\xD9\x85\xD8\xB4\xD8\xA7\xD9\x83\xD9\x84\xD8\xAA\xD8\xB9\xD8\xA7\xD9\x84\xD9\x89\xD8\xA7\xD9\x84\xD8\xA3\xD9\x88\xD9\x84\xD8\xA7\xD9\x84\xD8\xB3\xD9\x86\xD8\xA9\xD8\xAC\xD8\xA7\xD9\x85\xD8\xB9" ~
    "\xD8\xA9\xD8\xA7\xD9\x84\xD8\xB5\xD8\xAD\xD9\x81\xD8\xA7\xD9\x84\xD8\xAF\xD9\x8A\xD9\x86\xD9\x83\xD9\x84\xD9\x85\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD8\xAE\xD8\xA7\xD8\xB5\xD8\xA7\xD9\x84\xD9\x85\xD9\x84\xD9\x81\xD8\xA3\xD8\xB9\xD8\xB6\xD8\xA7\xD8\xA1\xD9\x83" ~
    "\xD8\xAA\xD8\xA7\xD8\xA8\xD8\xA9\xD8\xA7\xD9\x84\xD8\xAE\xD9\x8A\xD8\xB1\xD8\xB1\xD8\xB3\xD8\xA7\xD8\xA6\xD9\x84\xD8\xA7\xD9\x84\xD9\x82\xD9\x84\xD8\xA8\xD8\xA7\xD9\x84\xD8\xA3\xD8\xAF\xD8\xA8\xD9\x85\xD9\x82\xD8\xA7\xD8\xB7\xD8\xB9\xD9\x85\xD8\xB1\xD8\xA7" ~
    "\xD8\xB3\xD9\x84\xD9\x85\xD9\x86\xD8\xB7\xD9\x82\xD8\xA9\xD8\xA7\xD9\x84\xD9\x83\xD8\xAA\xD8\xA8\xD8\xA7\xD9\x84\xD8\xB1\xD8\xAC\xD9\x84\xD8\xA7\xD8\xB4\xD8\xAA\xD8\xB1\xD9\x83\xD8\xA7\xD9\x84\xD9\x82\xD8\xAF\xD9\x85\xD9\x8A\xD8\xB9\xD8\xB7\xD9\x8A\xD9\x83" ~
    "sByTagName(.jpg\" alt=\"1px solid #.gif\" alt=\"transparentinformati" ~
    "onapplication\" onclick=\"establishedadvertising.png\" alt=\"environ" ~
    "mentperformanceappropriate&amp;mdash;immediately</strong></rathe" ~
    "r thantemperaturedevelopmentcompetitionplaceholdervisibility:cop" ~
    "yright\">0\" height=\"even thoughreplacementdestinationCorporation<" ~
    "ul class=\"AssociationindividualsperspectivesetTimeout(url(http:/" ~
    "/mathematicsmargin-top:eventually description) no-repeatcollecti" ~
    "ons.JPG|thumb|participate/head><bodyfloat:left;<li class=\"hundre" ~
    "ds of\x0A\x0AHowever, compositionclear:both;cooperationwithin the labe" ~
    "l for=\"border-top:New Zealandrecommendedphotographyinteresting&l" ~
    "t;sup&gt;controversyNetherlandsalternativemaxlength=\"switzerland" ~
    "Developmentessentially\x0A\x0AAlthough </textarea>thunderbirdrepresent" ~
    "ed&amp;ndash;speculationcommunitieslegislationelectronics\x0A\x09<div " ~
    "id=\"illustratedengineeringterritoriesauthoritiesdistributed6\" he" ~
    "ight=\"sans-serif;capable of disappearedinteractivelooking forit " ~
    "would beAfghanistanwas createdMath.floor(surroundingcan also beo" ~
    "bservationmaintenanceencountered<h2 class=\"more recentit has bee" ~
    "ninvasion of).getTime()fundamentalDespite the\"><div id=\"inspirat" ~
    "ionexaminationpreparationexplanation<input id=\"</a></span>versio" ~
    "ns ofinstrumentsbefore the  = 'http://Descriptionrelatively .sub" ~
    "string(each of theexperimentsinfluentialintegrationmany peopledu" ~
    "e to the combinationdo not haveMiddle East<noscript><copyright\" " ~
    "perhaps theinstitutionin Decemberarrangementmost famouspersonali" ~
    "tycreation oflimitationsexclusivelysovereignty-content\">\x0A<td cla" ~
    "ss=\"undergroundparallel todoctrine ofoccupied byterminologyRenai" ~
    "ssancea number ofsupport forexplorationrecognitionpredecessor<im" ~
    "g src=\"/<h1 class=\"publicationmay also bespecialized</fieldset>p" ~
    "rogressivemillions ofstates thatenforcementaround the one anothe" ~
    "r.parentNodeagricultureAlternativeresearcherstowards theMost of " ~
    "themany other (especially<td width=\";width:100%independent<h3 cl" ~
    "ass=\" onchange=\").addClass(interactionOne of the daughter ofacce" ~
    "ssoriesbranches of\x0D\x0A<div id=\"the largestdeclarationregulationsIn" ~
    "formationtranslationdocumentaryin order to\">\x0A<head>\x0A<\" height=\"1" ~
    "across the orientation);</script>implementedcan be seenthere was" ~
    " ademonstratecontainer\">connectionsthe Britishwas written!import" ~
    "ant;px; margin-followed byability to complicatedduring the immig" ~
    "rationalso called<h4 class=\"distinctionreplaced bygovernmentsloc" ~
    "ation ofin Novemberwhether the</p>\x0A</div>acquisitioncalled the p" ~
    "ersecutiondesignation{font-size:appeared ininvestigateexperience" ~
    "dmost likelywidely useddiscussionspresence of (document.extensiv" ~
    "elyIt has beenit does notcontrary toinhabitantsimprovementschola" ~
    "rshipconsumptioninstructionfor exampleone or morepx; paddingthe " ~
    "currenta series ofare usuallyrole in thepreviously derivativesev" ~
    "idence ofexperiencescolorschemestated thatcertificate</a></div>\x0A" ~
    " selected=\"high schoolresponse tocomfortableadoption ofthree yea" ~
    "rsthe countryin Februaryso that thepeople who provided by<param " ~
    "nameaffected byin terms ofappointmentISO-8859-1\"was born inhisto" ~
    "rical regarded asmeasurementis based on and other : function(sig" ~
    "nificantcelebrationtransmitted/js/jquery.is known astheoretical " ~
    "tabindex=\"it could be<noscript>\x0Ahaving been\x0D\x0A<head>\x0D\x0A< &quot;The" ~
    " compilationhe had beenproduced byphilosopherconstructedintended" ~
    " toamong othercompared toto say thatEngineeringa differentreferr" ~
    "ed todifferencesbelief thatphotographsidentifyingHistory of Repu" ~
    "blic ofnecessarilyprobabilitytechnicallyleaving thespectacularfr" ~
    "action ofelectricityhead of therestaurantspartnershipemphasis on" ~
    "most recentshare with saying thatfilled withdesigned toit is oft" ~
    "en\"></iframe>as follows:merged withthrough thecommercial pointed" ~
    " outopportunityview of therequirementdivision ofprogramminghe re" ~
    "ceivedsetInterval\"></span></in New Yorkadditional compression\x0A\x0A<" ~
    "div id=\"incorporate;</script><attachEventbecame the \" target=\"_c" ~
    "arried outSome of thescience andthe time ofContainer\">maintainin" ~
    "gChristopherMuch of thewritings of\" height=\"2size of theversion " ~
    "of mixture of between theExamples ofeducationalcompetitive onsub" ~
    "mit=\"director ofdistinctive/DTD XHTML relating totendency toprov" ~
    "ince ofwhich woulddespite thescientific legislature.innerHTML al" ~
    "legationsAgriculturewas used inapproach tointelligentyears later" ~
    ",sans-serifdeterminingPerformanceappearances, which is foundatio" ~
    "nsabbreviatedhigher thans from the individual composed ofsuppose" ~
    "d toclaims thatattributionfont-size:1elements ofHistorical his b" ~
    "rotherat the timeanniversarygoverned byrelated to ultimately inn" ~
    "ovationsit is stillcan only bedefinitionstoGMTStringA number ofi" ~
    "mg class=\"Eventually,was changedoccurred inneighboringdistinguis" ~
    "hwhen he wasintroducingterrestrialMany of theargues thatan Ameri" ~
    "canconquest ofwidespread were killedscreen and In order toexpect" ~
    "ed todescendantsare locatedlegislativegenerations backgroundmost" ~
    " peopleyears afterthere is nothe highestfrequently they do notar" ~
    "gued thatshowed thatpredominanttheologicalby the timeconsidering" ~
    "short-lived</span></a>can be usedvery littleone of the had alrea" ~
    "dyinterpretedcommunicatefeatures ofgovernment,</noscript>entered" ~
    " the\" height=\"3Independentpopulationslarge-scale. Although used " ~
    "in thedestructionpossibilitystarting intwo or moreexpressionssub" ~
    "ordinatelarger thanhistory and</option>\x0D\x0AContinentaleliminatingw" ~
    "ill not bepractice ofin front ofsite of theensure thatto create " ~
    "amississippipotentiallyoutstandingbetter thanwhat is nowsituated" ~
    " inmeta name=\"TraditionalsuggestionsTranslationthe form ofatmosp" ~
    "hericideologicalenterprisescalculatingeast of theremnants ofplug" ~
    "inspage/index.php?remained intransformedHe was alsowas alreadyst" ~
    "atisticalin favor ofMinistry ofmovement offormulationis required" ~
    "<link rel=\"This is the <a href=\"/popularizedinvolved inare used " ~
    "toand severalmade by theseems to belikely thatPalestiniannamed a" ~
    "fterit had beenmost commonto refer tobut this isconsecutivetempo" ~
    "rarilyIn general,conventionstakes placesubdivisionterritorialope" ~
    "rationalpermanentlywas largelyoutbreak ofin the pastfollowing a " ~
    "xmlns:og=\"><a class=\"class=\"textConversion may be usedmanufactur" ~
    "eafter beingclearfix\">\x0Aquestion ofwas electedto become abecause " ~
    "of some peopleinspired bysuccessful a time whenmore commonamongs" ~
    "t thean officialwidth:100%;technology,was adoptedto keep thesett" ~
    "lementslive birthsindex.html\"Connecticutassigned to&amp;times;ac" ~
    "count foralign=rightthe companyalways beenreturned toinvolvement" ~
    "Because thethis period\" name=\"q\" confined toa result ofvalue=\"\" " ~
    "/>is actuallyEnvironment\x0D\x0A</head>\x0D\x0AConversely,>\x0A<div id=\"0\" widt" ~
    "h=\"1is probablyhave becomecontrollingthe problemcitizens ofpolit" ~
    "iciansreached theas early as:none; over<table cellvalidity ofdir" ~
    "ectly toonmousedownwhere it iswhen it wasmembers of relation toa" ~
    "ccommodatealong with In the latethe Englishdelicious\">this is no" ~
    "tthe presentif they areand finallya matter of\x0D\x0A\x09</div>\x0D\x0A\x0D\x0A</scri" ~
    "pt>faster thanmajority ofafter whichcomparativeto maintainimprov" ~
    "e theawarded theer\" class=\"frameborderrestorationin the sameanal" ~
    "ysis oftheir firstDuring the continentalsequence offunction(){fo" ~
    "nt-size: work on the</script>\x0A<begins withjavascript:constituent" ~
    "was foundedequilibriumassume thatis given byneeds to becoordinat" ~
    "esthe variousare part ofonly in thesections ofis a commontheorie" ~
    "s ofdiscoveriesassociationedge of thestrength ofposition inprese" ~
    "nt-dayuniversallyto form thebut insteadcorporationattached tois " ~
    "commonlyreasons for &quot;the can be madewas able towhich meansb" ~
    "ut did notonMouseOveras possibleoperated bycoming fromthe primar" ~
    "yaddition offor severaltransferreda period ofare able tohowever," ~
    " itshould havemuch larger\x0A\x09</script>adopted theproperty ofdirect" ~
    "ed byeffectivelywas broughtchildren ofProgramminglonger thanmanu" ~
    "scriptswar againstby means ofand most ofsimilar to proprietaryor" ~
    "iginatingprestigiousgrammaticalexperience.to make theIt was also" ~
    "is found incompetitorsin the U.S.replace thebrought thecalculati" ~
    "onfall of thethe generalpracticallyin honor ofreleased inresiden" ~
    "tialand some ofking of thereaction to1st Earl ofculture andprinc" ~
    "ipally</title>\x0A  they can beback to thesome of hisexposure toare" ~
    " similarform of theaddFavoritecitizenshippart in thepeople withi" ~
    "n practiceto continue&amp;minus;approved by the first allowed th" ~
    "eand for thefunctioningplaying thesolution toheight=\"0\" in his b" ~
    "ookmore than afollows thecreated thepresence in&nbsp;</td>nation" ~
    "alistthe idea ofa characterwere forced class=\"btndays of thefeat" ~
    "ured inshowing theinterest inin place ofturn of thethe head ofLo" ~
    "rd of thepoliticallyhas its ownEducationalapproval ofsome of the" ~
    "each other,behavior ofand becauseand anotherappeared onrecorded " ~
    "inblack&quot;may includethe world'scan lead torefers to aborder=" ~
    "\"0\" government winning theresulted in while the Washington,the s" ~
    "ubjectcity in the></div>\x0D\x0A\x09\x09reflect theto completebecame morerad" ~
    "ioactiverejected bywithout anyhis father,which couldcopy of thet" ~
    "o indicatea politicalaccounts ofconstitutesworked wither</a></li" ~
    ">of his lifeaccompaniedclientWidthprevent theLegislativedifferen" ~
    "tlytogether inhas severalfor anothertext of thefounded thee with" ~
    " the is used forchanged theusually theplace wherewhereas the> <a" ~
    " href=\"\"><a href=\"themselves,although hethat can betraditionalro" ~
    "le of theas a resultremoveChilddesigned bywest of theSome people" ~
    "production,side of thenewslettersused by thedown to theaccepted " ~
    "bylive in theattempts tooutside thefrequenciesHowever, inprogram" ~
    "mersat least inapproximatealthough itwas part ofand variousGover" ~
    "nor ofthe articleturned into><a href=\"/the economyis the mostmos" ~
    "t widelywould laterand perhapsrise to theoccurs whenunder whichc" ~
    "onditions.the westerntheory thatis producedthe city ofin which h" ~
    "eseen in thethe centralbuilding ofmany of hisarea of theis the o" ~
    "nlymost of themany of thethe WesternThere is noextended toStatis" ~
    "ticalcolspan=2 |short storypossible totopologicalcritical ofrepo" ~
    "rted toa Christiandecision tois equal toproblems ofThis can beme" ~
    "rchandisefor most ofno evidenceeditions ofelements in&quot;. The" ~
    "com/images/which makesthe processremains theliterature,is a memb" ~
    "erthe popularthe ancientproblems intime of thedefeated bybody of" ~
    " thea few yearsmuch of thethe work ofCalifornia,served as agover" ~
    "nment.concepts ofmovement in\x09\x09<div id=\"it\" value=\"language ofas " ~
    "they areproduced inis that theexplain thediv></div>\x0AHowever thel" ~
    "ead to the\x09<a href=\"/was grantedpeople havecontinuallywas seen a" ~
    "sand relatedthe role ofproposed byof the besteach other.Constant" ~
    "inepeople fromdialects ofto revisionwas renameda source ofthe in" ~
    "itiallaunched inprovide theto the westwhere thereand similarbetw" ~
    "een twois also theEnglish andconditions,that it wasentitled toth" ~
    "emselves.quantity ofransparencythe same asto join thecountry and" ~
    "this is theThis led toa statementcontrast tolastIndexOfthrough h" ~
    "isis designedthe term isis providedprotect theng</a></li>The cur" ~
    "rentthe site ofsubstantialexperience,in the Westthey shouldslove" ~
    "n\xC4\x8Dinacomentariosuniversidadcondicionesactividadesexperienciatec" ~
    "nolog\xC3\xADaproducci\xC3\xB3npuntuaci\xC3\xB3naplicaci\xC3\xB3ncontrase\xC3\xB1acategor\xC3\xADasr" ~
    "egistrarseprofesionaltratamientoreg\xC3\xADstratesecretar\xC3\xADaprincipale" ~
    "sprotecci\xC3\xB3nimportantesimportanciaposibilidadinteresantecrecimie" ~
    "ntonecesidadessuscribirseasociaci\xC3\xB3ndisponiblesevaluaci\xC3\xB3nestudi" ~
    "antesresponsableresoluci\xC3\xB3nguadalajararegistradosoportunidadcome" ~
    "rcialesfotograf\xC3\xADaautoridadesingenier\xC3\xADatelevisi\xC3\xB3ncompetenciaop" ~
    "eracionesestablecidosimplementeactualmentenavegaci\xC3\xB3nconformidad" ~
    "line-height:font-family:\" : \"http://applicationslink\" href=\"spec" ~
    "ifically//<![CDATA[\x0AOrganizationdistribution0px; height:relation" ~
    "shipdevice-width<div class=\"<label for=\"registration</noscript>\x0A" ~
    "/index.html\"window.open( !important;application/independence//ww" ~
    "w.googleorganizationautocompleterequirementsconservative<form na" ~
    "me=\"intellectualmargin-left:18th centuryan importantinstitutions" ~
    "abbreviation<img class=\"organisationcivilization19th centuryarch" ~
    "itectureincorporated20th century-container\">most notably/></a></" ~
    "div>notification'undefined')Furthermore,believe thatinnerHTML = " ~
    "prior to thedramaticallyreferring tonegotiationsheadquartersSout" ~
    "h AfricaunsuccessfulPennsylvaniaAs a result,<html lang=\"&lt;/sup" ~
    "&gt;dealing withphiladelphiahistorically);</script>\x0Apadding-top:" ~
    "experimentalgetAttributeinstructionstechnologiespart of the =fun" ~
    "ction(){subscriptionl.dtd\">\x0D\x0A<htgeographicalConstitution', funct" ~
    "ion(supported byagriculturalconstructionpublicationsfont-size: 1" ~
    "a variety of<div style=\"Encyclopediaiframe src=\"demonstratedacco" ~
    "mplisheduniversitiesDemographics);</script><dedicated toknowledg" ~
    "e ofsatisfactionparticularly</div></div>English (US)appendChild(" ~
    "transmissions. However, intelligence\" tabindex=\"float:right;Comm" ~
    "onwealthranging fromin which theat least onereproductionencyclop" ~
    "edia;font-size:1jurisdictionat that time\"><a class=\"In addition," ~
    "description+conversationcontact withis generallyr\" content=\"repr" ~
    "esenting&lt;math&gt;presentationoccasionally<img width=\"navigati" ~
    "on\">compensationchampionshipmedia=\"all\" violation ofreference to" ~
    "return true;Strict//EN\" transactionsinterventionverificationInfo" ~
    "rmation difficultiesChampionshipcapabilities<![endif]-->}\x0A</scri" ~
    "pt>\x0AChristianityfor example,Professionalrestrictionssuggest that" ~
    "was released(such as theremoveClass(unemploymentthe Americanstru" ~
    "cture of/index.html published inspan class=\"\"><a href=\"/introduc" ~
    "tionbelonging toclaimed thatconsequences<meta name=\"Guide to the" ~
    "overwhelmingagainst the concentrated,\x0A.nontouch observations</a>" ~
    "\x0A</div>\x0Af (document.border: 1px {font-size:1treatment of0\" heigh" ~
    "t=\"1modificationIndependencedivided intogreater thanachievements" ~
    "establishingJavaScript\" neverthelesssignificanceBroadcasting>&nb" ~
    "sp;</td>container\">\x0Asuch as the influence ofa particularsrc='htt" ~
    "p://navigation\" half of the substantial &nbsp;</div>advantage of" ~
    "discovery offundamental metropolitanthe opposite\" xml:lang=\"deli" ~
    "beratelyalign=centerevolution ofpreservationimprovementsbeginnin" ~
    "g inJesus ChristPublicationsdisagreementtext-align:r, function()" ~
    "similaritiesbody></html>is currentlyalphabeticalis sometimestype" ~
    "=\"image/many of the flow:hidden;available indescribe theexistenc" ~
    "e ofall over thethe Internet\x09<ul class=\"installationneighborhood" ~
    "armed forcesreducing thecontinues toNonetheless,temperatures\x0A\x09\x09<" ~
    "a href=\"close to theexamples of is about the(see below).\" id=\"se" ~
    "archprofessionalis availablethe official\x09\x09</script>\x0A\x0A\x09\x09<div id=\"" ~
    "accelerationthrough the Hall of Famedescriptionstranslationsinte" ~
    "rference type='text/recent yearsin the worldvery popular{backgro" ~
    "und:traditional some of the connected toexploitationemergence of" ~
    "constitutionA History ofsignificant manufacturedexpectations><no" ~
    "script><can be foundbecause the has not beenneighbouringwithout " ~
    "the added to the\x09<li class=\"instrumentalSoviet Unionacknowledged" ~
    "which can bename for theattention toattempts to developmentsIn f" ~
    "act, the<li class=\"aimplicationssuitable formuch of the coloniza" ~
    "tionpresidentialcancelBubble Informationmost of the is described" ~
    "rest of the more or lessin SeptemberIntelligencesrc=\"http://px; " ~
    "height: available tomanufacturerhuman rightslink href=\"/availabi" ~
    "lityproportionaloutside the astronomicalhuman beingsname of the " ~
    "are found inare based onsmaller thana person whoexpansion ofargu" ~
    "ing thatnow known asIn the earlyintermediatederived fromScandina" ~
    "vian</a></div>\x0D\x0Aconsider thean estimatedthe National<div id=\"pag" ~
    "resulting incommissionedanalogous toare required/ul>\x0A</div>\x0Awas " ~
    "based onand became a&nbsp;&nbsp;t\" value=\"\" was capturedno more " ~
    "thanrespectivelycontinue to >\x0D\x0A<head>\x0D\x0A<were createdmore general" ~
    "information used for theindependent the Imperialcomponent ofto t" ~
    "he northinclude the Constructionside of the would not befor inst" ~
    "anceinvention ofmore complexcollectivelybackground: text-align: " ~
    "its originalinto accountthis processan extensivehowever, thethey" ~
    " are notrejected thecriticism ofduring whichprobably thethis art" ~
    "icle(function(){It should bean agreementaccidentallydiffers from" ~
    "Architecturebetter knownarrangementsinfluence onattended theiden" ~
    "tical tosouth of thepass throughxml\" title=\"weight:bold;creating" ~
    " thedisplay:nonereplaced the<img src=\"/ihttps://www.World War II" ~
    "testimonialsfound in therequired to and that thebetween the was " ~
    "designedconsists of considerablypublished bythe languageConserva" ~
    "tionconsisted ofrefer to theback to the css\" media=\"People from " ~
    "available onproved to besuggestions\"was known asvarieties oflike" ~
    "ly to becomprised ofsupport the hands of thecoupled withconnect " ~
    "and border:none;performancesbefore beinglater becamecalculations" ~
    "often calledresidents ofmeaning that><li class=\"evidence forexpl" ~
    "anationsenvironments\"></a></div>which allowsIntroductiondevelope" ~
    "d bya wide rangeon behalf ofvalign=\"top\"principle ofat the time," ~
    "</noscript>\x0Dsaid to havein the firstwhile othershypotheticalphil" ~
    "osopherspower of thecontained inperformed byinability towere wri" ~
    "ttenspan style=\"input name=\"the questionintended forrejection of" ~
    "implies thatinvented thethe standardwas probablylink betweenprof" ~
    "essor ofinteractionschanging theIndian Ocean class=\"lastworking " ~
    "with'http://www.years beforeThis was therecreationalentering the" ~
    "measurementsan extremelyvalue of thestart of the\x0A</script>\x0A\x0Aan e" ~
    "ffort toincrease theto the southspacing=\"0\">sufficientlythe Euro" ~
    "peanconverted toclearTimeoutdid not haveconsequentlyfor the next" ~
    "extension ofeconomic andalthough theare producedand with theinsu" ~
    "fficientgiven by thestating thatexpenditures</span></a>\x0Athought " ~
    "thaton the basiscellpadding=image of thereturning toinformation," ~
    "separated byassassinateds\" content=\"authority ofnorthwestern</di" ~
    "v>\x0A<div \"></div>\x0D\x0A  consultationcommunity ofthe nationalit shoul" ~
    "d beparticipants align=\"leftthe greatestselection ofsupernatural" ~
    "dependent onis mentionedallowing thewas inventedaccompanyinghis " ~
    "personalavailable atstudy of theon the otherexecution ofHuman Ri" ~
    "ghtsterms of theassociationsresearch andsucceeded bydefeated the" ~
    "and from thebut they arecommander ofstate of theyears of agethe " ~
    "study of<ul class=\"splace in thewhere he was<li class=\"fthere ar" ~
    "e nowhich becamehe publishedexpressed into which thecommissioner" ~
    "font-weight:territory ofextensions\">Roman Empireequal to theIn c" ~
    "ontrast,however, andis typicallyand his wife(also called><ul cla" ~
    "ss=\"effectively evolved intoseem to havewhich is thethere was no" ~
    "an excellentall of thesedescribed byIn practice,broadcastingchar" ~
    "ged withreflected insubjected tomilitary andto the pointeconomic" ~
    "allysetTargetingare actuallyvictory over();</script>continuously" ~
    "required forevolutionaryan effectivenorth of the, which was fron" ~
    "t of theor otherwisesome form ofhad not beengenerated byinformat" ~
    "ion.permitted toincludes thedevelopment,entered intothe previous" ~
    "consistentlyare known asthe field ofthis type ofgiven to thethe " ~
    "title ofcontains theinstances ofin the northdue to theirare desi" ~
    "gnedcorporationswas that theone of thesemore popularsucceeded in" ~
    "support fromin differentdominated bydesigned forownership ofand " ~
    "possiblystandardizedresponseTextwas intendedreceived theassumed " ~
    "thatareas of theprimarily inthe basis ofin the senseaccounts for" ~
    "destroyed byat least twowas declaredcould not beSecretary ofappe" ~
    "ar to bemargin-top:1/^\\s+|\\s+$/ge){throw e};the start oftwo sepa" ~
    "ratelanguage andwho had beenoperation ofdeath of thereal numbers" ~
    "\x09<link rel=\"provided thethe story ofcompetitionsenglish (UK)engl" ~
    "ish (US)\xD0\x9C\xD0\xBE\xD0\xBD\xD0\xB3\xD0\xBE\xD0\xBB\xD0\xA1\xD1\x80\xD0\xBF\xD1\x81\xD0\xBA\xD0\xB8\xD1\x81\xD1\x80\xD0\xBF\xD1\x81\xD0\xBA\xD0\xB8\xD1\x81\xD1\x80\xD0\xBF\xD1\x81\xD0\xBA\xD0\xBE\xD9\x84\xD8\xB9\xD8\xB1\xD8\xA8" ~
    "\xD9\x8A\xD8\xA9\xE6\xAD\xA3\xE9\xAB\x94\xE4\xB8\xAD\xE6\x96\x87\xE7\xAE\x80\xE4\xBD\x93\xE4\xB8\xAD\xE6\x96\x87\xE7\xB9\x81\xE4\xBD\x93\xE4\xB8\xAD\xE6\x96\x87\xE6\x9C\x89\xE9\x99\x90\xE5\x85\xAC\xE5\x8F\xB8\xE4\xBA\xBA\xE6\xB0\x91\xE6\x94\xBF\xE5\xBA\x9C" ~
    "\xE9\x98\xBF\xE9\x87\x8C\xE5\xB7\xB4\xE5\xB7\xB4\xE7\xA4\xBE\xE4\xBC\x9A\xE4\xB8\xBB\xE4\xB9\x89\xE6\x93\x8D\xE4\xBD\x9C\xE7\xB3\xBB\xE7\xBB\x9F\xE6\x94\xBF\xE7\xAD\x96\xE6\xB3\x95\xE8\xA7\x84informaci\xC3\xB3nherr" ~
    "amientaselectr\xC3\xB3nicodescripci\xC3\xB3nclasificadosconocimientopublicac" ~
    "i\xC3\xB3nrelacionadasinform\xC3\xA1ticarelacionadosdepartamentotrabajadores" ~
    "directamenteayuntamientomercadoLibrecont\xC3\xA1ctenoshabitacionescump" ~
    "limientorestaurantesdisposici\xC3\xB3nconsecuenciaelectr\xC3\xB3nicaaplicaci" ~
    "onesdesconectadoinstalaci\xC3\xB3nrealizaci\xC3\xB3nutilizaci\xC3\xB3nenciclopedia" ~
    "enfermedadesinstrumentosexperienciasinstituci\xC3\xB3nparticularessubc" ~
    "ategoria\xD1\x82\xD0\xBE\xD0\xBB\xD1\x8C\xD0\xBA\xD0\xBE\xD0\xA0\xD0\xBE\xD1\x81\xD1\x81\xD0\xB8\xD0\xB8\xD1\x80\xD0\xB0\xD0\xB1\xD0\xBE\xD1\x82\xD1\x8B\xD0\xB1\xD0\xBE\xD0\xBB\xD1\x8C\xD1\x88\xD0\xB5\xD0\xBF\xD1\x80\xD0\xBE\xD1\x81" ~
    "\xD1\x82\xD0\xBE\xD0\xBC\xD0\xBE\xD0\xB6\xD0\xB5\xD1\x82\xD0\xB5\xD0\xB4\xD1\x80\xD1\x83\xD0\xB3\xD0\xB8\xD1\x85\xD1\x81\xD0\xBB\xD1\x83\xD1\x87\xD0\xB0\xD0\xB5\xD1\x81\xD0\xB5\xD0\xB9\xD1\x87\xD0\xB0\xD1\x81\xD0\xB2\xD1\x81\xD0\xB5\xD0\xB3\xD0\xB4\xD0\xB0" ~
    "\xD0\xA0\xD0\xBE\xD1\x81\xD1\x81\xD0\xB8\xD1\x8F\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB5\xD0\xB4\xD1\x80\xD1\x83\xD0\xB3\xD0\xB8\xD0\xB5\xD0\xB3\xD0\xBE\xD1\x80\xD0\xBE\xD0\xB4\xD0\xB0\xD0\xB2\xD0\xBE\xD0\xBF\xD1\x80\xD0\xBE\xD1\x81\xD0\xB4\xD0\xB0" ~
    "\xD0\xBD\xD0\xBD\xD1\x8B\xD1\x85\xD0\xB4\xD0\xBE\xD0\xBB\xD0\xB6\xD0\xBD\xD1\x8B\xD0\xB8\xD0\xBC\xD0\xB5\xD0\xBD\xD0\xBD\xD0\xBE\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD1\x8B\xD1\x80\xD1\x83\xD0\xB1\xD0\xBB\xD0\xB5\xD0\xB9\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA" ~
    "\xD0\xB2\xD0\xB0\xD1\x81\xD1\x82\xD1\x80\xD0\xB0\xD0\xBD\xD1\x8B\xD0\xBD\xD0\xB8\xD1\x87\xD0\xB5\xD0\xB3\xD0\xBE\xD1\x80\xD0\xB0\xD0\xB1\xD0\xBE\xD1\x82\xD0\xB5\xD0\xB4\xD0\xBE\xD0\xBB\xD0\xB6\xD0\xB5\xD0\xBD\xD1\x83\xD1\x81\xD0\xBB\xD1\x83\xD0\xB3\xD0\xB8" ~
    "\xD1\x82\xD0\xB5\xD0\xBF\xD0\xB5\xD1\x80\xD1\x8C\xD0\x9E\xD0\xB4\xD0\xBD\xD0\xB0\xD0\xBA\xD0\xBE\xD0\xBF\xD0\xBE\xD1\x82\xD0\xBE\xD0\xBC\xD1\x83\xD1\x80\xD0\xB0\xD0\xB1\xD0\xBE\xD1\x82\xD1\x83\xD0\xB0\xD0\xBF\xD1\x80\xD0\xB5\xD0\xBB\xD1\x8F\xD0\xB2\xD0\xBE" ~
    "\xD0\xBE\xD0\xB1\xD1\x89\xD0\xB5\xD0\xBE\xD0\xB4\xD0\xBD\xD0\xBE\xD0\xB3\xD0\xBE\xD1\x81\xD0\xB2\xD0\xBE\xD0\xB5\xD0\xB3\xD0\xBE\xD1\x81\xD1\x82\xD0\xB0\xD1\x82\xD1\x8C\xD0\xB8\xD0\xB4\xD1\x80\xD1\x83\xD0\xB3\xD0\xBE\xD0\xB9\xD1\x84\xD0\xBE\xD1\x80\xD1\x83" ~
    "\xD0\xBC\xD0\xB5\xD1\x85\xD0\xBE\xD1\x80\xD0\xBE\xD1\x88\xD0\xBE\xD0\xBF\xD1\x80\xD0\xBE\xD1\x82\xD0\xB8\xD0\xB2\xD1\x81\xD1\x81\xD1\x8B\xD0\xBB\xD0\xBA\xD0\xB0\xD0\xBA\xD0\xB0\xD0\xB6\xD0\xB4\xD1\x8B\xD0\xB9\xD0\xB2\xD0\xBB\xD0\xB0\xD1\x81\xD1\x82\xD0\xB8" ~
    "\xD0\xB3\xD1\x80\xD1\x83\xD0\xBF\xD0\xBF\xD1\x8B\xD0\xB2\xD0\xBC\xD0\xB5\xD1\x81\xD1\x82\xD0\xB5\xD1\x80\xD0\xB0\xD0\xB1\xD0\xBE\xD1\x82\xD0\xB0\xD1\x81\xD0\xBA\xD0\xB0\xD0\xB7\xD0\xB0\xD0\xBB\xD0\xBF\xD0\xB5\xD1\x80\xD0\xB2\xD1\x8B\xD0\xB9\xD0\xB4\xD0\xB5" ~
    "\xD0\xBB\xD0\xB0\xD1\x82\xD1\x8C\xD0\xB4\xD0\xB5\xD0\xBD\xD1\x8C\xD0\xB3\xD0\xB8\xD0\xBF\xD0\xB5\xD1\x80\xD0\xB8\xD0\xBE\xD0\xB4\xD0\xB1\xD0\xB8\xD0\xB7\xD0\xBD\xD0\xB5\xD1\x81\xD0\xBE\xD1\x81\xD0\xBD\xD0\xBE\xD0\xB2\xD0\xB5\xD0\xBC\xD0\xBE\xD0\xBC\xD0\xB5" ~
    "\xD0\xBD\xD1\x82\xD0\xBA\xD1\x83\xD0\xBF\xD0\xB8\xD1\x82\xD1\x8C\xD0\xB4\xD0\xBE\xD0\xBB\xD0\xB6\xD0\xBD\xD0\xB0\xD1\x80\xD0\xB0\xD0\xBC\xD0\xBA\xD0\xB0\xD1\x85\xD0\xBD\xD0\xB0\xD1\x87\xD0\xB0\xD0\xBB\xD0\xBE\xD0\xA0\xD0\xB0\xD0\xB1\xD0\xBE\xD1\x82\xD0\xB0" ~
    "\xD0\xA2\xD0\xBE\xD0\xBB\xD1\x8C\xD0\xBA\xD0\xBE\xD1\x81\xD0\xBE\xD0\xB2\xD1\x81\xD0\xB5\xD0\xBC\xD0\xB2\xD1\x82\xD0\xBE\xD1\x80\xD0\xBE\xD0\xB9\xD0\xBD\xD0\xB0\xD1\x87\xD0\xB0\xD0\xBB\xD0\xB0\xD1\x81\xD0\xBF\xD0\xB8\xD1\x81\xD0\xBE\xD0\xBA\xD1\x81\xD0\xBB" ~
    "\xD1\x83\xD0\xB6\xD0\xB1\xD1\x8B\xD1\x81\xD0\xB8\xD1\x81\xD1\x82\xD0\xB5\xD0\xBC\xD0\xBF\xD0\xB5\xD1\x87\xD0\xB0\xD1\x82\xD0\xB8\xD0\xBD\xD0\xBE\xD0\xB2\xD0\xBE\xD0\xB3\xD0\xBE\xD0\xBF\xD0\xBE\xD0\xBC\xD0\xBE\xD1\x89\xD0\xB8\xD1\x81\xD0\xB0\xD0\xB9\xD1\x82" ~
    "\xD0\xBE\xD0\xB2\xD0\xBF\xD0\xBE\xD1\x87\xD0\xB5\xD0\xBC\xD1\x83\xD0\xBF\xD0\xBE\xD0\xBC\xD0\xBE\xD1\x89\xD1\x8C\xD0\xB4\xD0\xBE\xD0\xBB\xD0\xB6\xD0\xBD\xD0\xBE\xD1\x81\xD1\x81\xD1\x8B\xD0\xBB\xD0\xBA\xD0\xB8\xD0\xB1\xD1\x8B\xD1\x81\xD1\x82\xD1\x80\xD0\xBE" ~
    "\xD0\xB4\xD0\xB0\xD0\xBD\xD0\xBD\xD1\x8B\xD0\xB5\xD0\xBC\xD0\xBD\xD0\xBE\xD0\xB3\xD0\xB8\xD0\xB5\xD0\xBF\xD1\x80\xD0\xBE\xD0\xB5\xD0\xBA\xD1\x82\xD0\xA1\xD0\xB5\xD0\xB9\xD1\x87\xD0\xB0\xD1\x81\xD0\xBC\xD0\xBE\xD0\xB4\xD0\xB5\xD0\xBB\xD0\xB8\xD1\x82\xD0\xB0" ~
    "\xD0\xBA\xD0\xBE\xD0\xB3\xD0\xBE\xD0\xBE\xD0\xBD\xD0\xBB\xD0\xB0\xD0\xB9\xD0\xBD\xD0\xB3\xD0\xBE\xD1\x80\xD0\xBE\xD0\xB4\xD0\xB5\xD0\xB2\xD0\xB5\xD1\x80\xD1\x81\xD0\xB8\xD1\x8F\xD1\x81\xD1\x82\xD1\x80\xD0\xB0\xD0\xBD\xD0\xB5\xD1\x84\xD0\xB8\xD0\xBB\xD1\x8C" ~
    "\xD0\xBC\xD1\x8B\xD1\x83\xD1\x80\xD0\xBE\xD0\xB2\xD0\xBD\xD1\x8F\xD1\x80\xD0\xB0\xD0\xB7\xD0\xBD\xD1\x8B\xD1\x85\xD0\xB8\xD1\x81\xD0\xBA\xD0\xB0\xD1\x82\xD1\x8C\xD0\xBD\xD0\xB5\xD0\xB4\xD0\xB5\xD0\xBB\xD1\x8E\xD1\x8F\xD0\xBD\xD0\xB2\xD0\xB0\xD1\x80\xD1\x8F" ~
    "\xD0\xBC\xD0\xB5\xD0\xBD\xD1\x8C\xD1\x88\xD0\xB5\xD0\xBC\xD0\xBD\xD0\xBE\xD0\xB3\xD0\xB8\xD1\x85\xD0\xB4\xD0\xB0\xD0\xBD\xD0\xBD\xD0\xBE\xD0\xB9\xD0\xB7\xD0\xBD\xD0\xB0\xD1\x87\xD0\xB8\xD1\x82\xD0\xBD\xD0\xB5\xD0\xBB\xD1\x8C\xD0\xB7\xD1\x8F\xD1\x84\xD0\xBE" ~
    "\xD1\x80\xD1\x83\xD0\xBC\xD0\xB0\xD0\xA2\xD0\xB5\xD0\xBF\xD0\xB5\xD1\x80\xD1\x8C\xD0\xBC\xD0\xB5\xD1\x81\xD1\x8F\xD1\x86\xD0\xB0\xD0\xB7\xD0\xB0\xD1\x89\xD0\xB8\xD1\x82\xD1\x8B\xD0\x9B\xD1\x83\xD1\x87\xD1\x88\xD0\xB8\xD0\xB5\xE0\xA4\xA8\xE0\xA4\xB9\xE0\xA5" ~
    "\x80\xE0\xA4\x82\xE0\xA4\x95\xE0\xA4\xB0\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\x85\xE0\xA4\xAA\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\x95\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\x95\xE0\xA4\xB0\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\x85\xE0\xA4\xA8\xE0\xA5\x8D\xE0\xA4\xAF" ~
    "\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\x97\xE0\xA4\xBE\xE0\xA4\x87\xE0\xA4\xA1\xE0\xA4\xAC\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA5\x87\xE0\xA4\x95\xE0\xA4\xBF\xE0\xA4\xB8\xE0\xA5\x80\xE0\xA4\xA6\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xAA\xE0" ~
    "\xA4\xB9\xE0\xA4\xB2\xE0\xA5\x87\xE0\xA4\xB8\xE0\xA4\xBF\xE0\xA4\x82\xE0\xA4\xB9\xE0\xA4\xAD\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xA4\xE0\xA4\x85\xE0\xA4\xAA\xE0\xA4\xA8\xE0\xA5\x80\xE0\xA4\xB5\xE0\xA4\xBE\xE0\xA4\xB2\xE0\xA5\x87\xE0\xA4\xB8\xE0\xA5\x87\xE0\xA4" ~
    "\xB5\xE0\xA4\xBE\xE0\xA4\x95\xE0\xA4\xB0\xE0\xA4\xA4\xE0\xA5\x87\xE0\xA4\xAE\xE0\xA5\x87\xE0\xA4\xB0\xE0\xA5\x87\xE0\xA4\xB9\xE0\xA5\x8B\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\xB8\xE0\xA4\x95\xE0\xA4\xA4\xE0\xA5\x87\xE0\xA4\xAC\xE0\xA4\xB9\xE0\xA5\x81\xE0\xA4\xA4" ~
    "\xE0\xA4\xB8\xE0\xA4\xBE\xE0\xA4\x87\xE0\xA4\x9F\xE0\xA4\xB9\xE0\xA5\x8B\xE0\xA4\x97\xE0\xA4\xBE\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\xAE\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA4\x9F\xE0\xA4\x95\xE0\xA4\xB0\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\x95\xE0" ~
    "\xA4\xB0\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\x89\xE0\xA4\xA8\xE0\xA4\x95\xE0\xA5\x87\xE0\xA4\xAF\xE0\xA4\xB9\xE0\xA4\xBE\xE0\xA4\x81\xE0\xA4\xB8\xE0\xA4\xAC\xE0\xA4\xB8\xE0\xA5\x87\xE0\xA4\xAD\xE0\xA4\xBE\xE0\xA4\xB7\xE0\xA4\xBE\xE0\xA4\x86\xE0\xA4\xAA\xE0\xA4" ~
    "\x95\xE0\xA5\x87\xE0\xA4\xB2\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA5\x87\xE0\xA4\xB6\xE0\xA5\x81\xE0\xA4\xB0\xE0\xA5\x82\xE0\xA4\x87\xE0\xA4\xB8\xE0\xA4\x95\xE0\xA5\x87\xE0\xA4\x98\xE0\xA4\x82\xE0\xA4\x9F\xE0\xA5\x87\xE0\xA4\xAE\xE0\xA5\x87\xE0\xA4\xB0\xE0\xA5\x80" ~
    "\xE0\xA4\xB8\xE0\xA4\x95\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\xAE\xE0\xA5\x87\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\xB2\xE0\xA5\x87\xE0\xA4\x95\xE0\xA4\xB0\xE0\xA4\x85\xE0\xA4\xA7\xE0\xA4\xBF\xE0\xA4\x95\xE0\xA4\x85\xE0\xA4\xAA\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\xB8\xE0" ~
    "\xA4\xAE\xE0\xA4\xBE\xE0\xA4\x9C\xE0\xA4\xAE\xE0\xA5\x81\xE0\xA4\x9D\xE0\xA5\x87\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xA3\xE0\xA4\xB9\xE0\xA5\x8B\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\x95\xE0\xA4\xA1\xE0\xA4\xBC\xE0\xA5\x80\xE0\xA4\xAF\xE0\xA4\xB9\xE0\xA4" ~
    "\xBE\xE0\xA4\x82\xE0\xA4\xB9\xE0\xA5\x8B\xE0\xA4\x9F\xE0\xA4\xB2\xE0\xA4\xB6\xE0\xA4\xAC\xE0\xA5\x8D\xE0\xA4\xA6\xE0\xA4\xB2\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\x9C\xE0\xA5\x80\xE0\xA4\xB5\xE0\xA4\xA8\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\xA4\xE0\xA4\xBE" ~
    "\xE0\xA4\x95\xE0\xA5\x88\xE0\xA4\xB8\xE0\xA5\x87\xE0\xA4\x86\xE0\xA4\xAA\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB5\xE0\xA4\xBE\xE0\xA4\xB2\xE0\xA5\x80\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\xAA\xE0\xA5\x82\xE0\xA4\xB0\xE0\xA5\x80\xE0\xA4\xAA\xE0" ~
    "\xA4\xBE\xE0\xA4\xA8\xE0\xA5\x80\xE0\xA4\x89\xE0\xA4\xB8\xE0\xA4\x95\xE0\xA5\x87\xE0\xA4\xB9\xE0\xA5\x8B\xE0\xA4\x97\xE0\xA5\x80\xE0\xA4\xAC\xE0\xA5\x88\xE0\xA4\xA0\xE0\xA4\x95\xE0\xA4\x86\xE0\xA4\xAA\xE0\xA4\x95\xE0\xA5\x80\xE0\xA4\xB5\xE0\xA4\xB0\xE0\xA5" ~
    "\x8D\xE0\xA4\xB7\xE0\xA4\x97\xE0\xA4\xBE\xE0\xA4\x82\xE0\xA4\xB5\xE0\xA4\x86\xE0\xA4\xAA\xE0\xA4\x95\xE0\xA5\x8B\xE0\xA4\x9C\xE0\xA4\xBF\xE0\xA4\xB2\xE0\xA4\xBE\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\xB8\xE0\xA4\xB9\xE0\xA4\xAE\xE0\xA4\xA4" ~
    "\xE0\xA4\xB9\xE0\xA4\xAE\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\x89\xE0\xA4\xA8\xE0\xA4\x95\xE0\xA5\x80\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xB9\xE0\xA5\x82\xE0\xA4\xA6\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\x9C\xE0\xA4\xB8\xE0\xA5\x82\xE0\xA4\x9A\xE0\xA5\x80\xE0\xA4\xAA\xE0" ~
    "\xA4\xB8\xE0\xA4\x82\xE0\xA4\xA6\xE0\xA4\xB8\xE0\xA4\xB5\xE0\xA4\xBE\xE0\xA4\xB2\xE0\xA4\xB9\xE0\xA5\x8B\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\xB9\xE0\xA5\x8B\xE0\xA4\xA4\xE0\xA5\x80\xE0\xA4\x9C\xE0\xA5\x88\xE0\xA4\xB8\xE0\xA5\x87\xE0\xA4\xB5\xE0\xA4\xBE\xE0\xA4" ~
    "\xAA\xE0\xA4\xB8\xE0\xA4\x9C\xE0\xA4\xA8\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA5\x80\xE0\xA4\x98\xE0\xA4\xBE\xE0\xA4\xAF\xE0\xA4\xB2\xE0\xA4\x9C\xE0\xA4\xBF\xE0\xA4\xB2\xE0\xA5\x87" ~
    "\xE0\xA4\xA8\xE0\xA5\x80\xE0\xA4\x9A\xE0\xA5\x87\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\x82\xE0\xA4\x9A\xE0\xA4\xAA\xE0\xA4\xA4\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\x97\xE0\xA5\x82\xE0\xA4\x97\xE0\xA4\xB2\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\xA4\xE0\xA5\x87\xE0\xA4\xAC\xE0" ~
    "\xA4\xBE\xE0\xA4\xB9\xE0\xA4\xB0\xE0\xA4\x86\xE0\xA4\xAA\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\xB5\xE0\xA4\xBE\xE0\xA4\xB9\xE0\xA4\xA8\xE0\xA4\x87\xE0\xA4\xB8\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB8\xE0\xA5\x81\xE0\xA4\xAC\xE0\xA4\xB9\xE0\xA4\xB0\xE0\xA4\xB9\xE0\xA4" ~
    "\xA8\xE0\xA5\x87\xE0\xA4\x87\xE0\xA4\xB8\xE0\xA4\xB8\xE0\xA5\x87\xE0\xA4\xB8\xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\xA4\xE0\xA4\xAC\xE0\xA4\xA1\xE0\xA4\xBC\xE0\xA5\x87\xE0\xA4\x98\xE0\xA4\x9F\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\xA4\xE0\xA4\xB2\xE0\xA4\xBE\xE0\xA4\xB6" ~
    "\xE0\xA4\xAA\xE0\xA4\xBE\xE0\xA4\x82\xE0\xA4\x9A\xE0\xA4\xB6\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA5\x80\xE0\xA4\xAC\xE0\xA4\xA1\xE0\xA4\xBC\xE0\xA5\x80\xE0\xA4\xB9\xE0\xA5\x8B\xE0\xA4\xA4\xE0\xA5\x87\xE0\xA4\xB8\xE0\xA4\xBE\xE0\xA4\x88\xE0\xA4\x9F\xE0\xA4\xB6\xE0" ~
    "\xA4\xBE\xE0\xA4\xAF\xE0\xA4\xA6\xE0\xA4\xB8\xE0\xA4\x95\xE0\xA4\xA4\xE0\xA5\x80\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\xA4\xE0\xA5\x80\xE0\xA4\xB5\xE0\xA4\xBE\xE0\xA4\xB2\xE0\xA4\xBE\xE0\xA4\xB9\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xAA\xE0\xA4\x9F\xE0\xA4" ~
    "\xA8\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\x96\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\xB8\xE0\xA4\xA1\xE0\xA4\xBC\xE0\xA4\x95\xE0\xA4\xAE\xE0\xA4\xBF\xE0\xA4\xB2\xE0\xA4\xBE\xE0\xA4\x89\xE0\xA4\xB8\xE0\xA4\x95\xE0\xA5\x80\xE0\xA4\x95\xE0\xA5\x87\xE0\xA4\xB5\xE0\xA4\xB2" ~
    "\xE0\xA4\xB2\xE0\xA4\x97\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\x96\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\x85\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xA5\xE0\xA4\x9C\xE0\xA4\xB9\xE0\xA4\xBE\xE0\xA4\x82\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\x96\xE0\xA4\xBE\xE0\xA4\xAA\xE0" ~
    "\xA4\xB9\xE0\xA4\xB2\xE0\xA5\x80\xE0\xA4\xA8\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA4\xAE\xE0\xA4\xAC\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\xAC\xE0\xA5\x88\xE0\xA4\x82\xE0\xA4\x95\xE0\xA4\x95\xE0\xA4\xB9\xE0\xA5\x80\xE0\xA4\x82\xE0\xA4\x95\xE0\xA4\xB9\xE0\xA4" ~
    "\xA8\xE0\xA4\xBE\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\xB9\xE0\xA4\xAE\xE0\xA4\xB2\xE0\xA5\x87\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xAB\xE0\xA5\x80\xE0\xA4\x9C\xE0\xA4\xAC\xE0\xA4\x95\xE0\xA4\xBF\xE0\xA4\xA4\xE0\xA5\x81\xE0\xA4\xB0\xE0\xA4\xA4" ~
    "\xE0\xA4\xAE\xE0\xA4\xBE\xE0\xA4\x82\xE0\xA4\x97\xE0\xA4\xB5\xE0\xA4\xB9\xE0\xA5\x80\xE0\xA4\x82\xE0\xA4\xB0\xE0\xA5\x8B\xE0\xA4\x9C\xE0\xA4\xBC\xE0\xA4\xAE\xE0\xA4\xBF\xE0\xA4\xB2\xE0\xA5\x80\xE0\xA4\x86\xE0\xA4\xB0\xE0\xA5\x8B\xE0\xA4\xAA\xE0\xA4\xB8\xE0" ~
    "\xA5\x87\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xA6\xE0\xA4\xB5\xE0\xA4\xB2\xE0\xA5\x87\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\x96\xE0\xA4\xBE\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\x95\xE0\xA4\xB0\xE0\xA5\x80\xE0\xA4\xAC\xE0\xA4\x89\xE0\xA4\xA8\xE0\xA4" ~
    "\x95\xE0\xA4\xBE\xE0\xA4\x9C\xE0\xA4\xB5\xE0\xA4\xBE\xE0\xA4\xAC\xE0\xA4\xAA\xE0\xA5\x82\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\xAC\xE0\xA4\xA1\xE0\xA4\xBC\xE0\xA4\xBE\xE0\xA4\xB8\xE0\xA5\x8C\xE0\xA4\xA6\xE0\xA4\xBE\xE0\xA4\xB6\xE0\xA5\x87\xE0\xA4\xAF\xE0\xA4\xB0" ~
    "\xE0\xA4\x95\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA5\x87\xE0\xA4\x95\xE0\xA4\xB9\xE0\xA4\xBE\xE0\xA4\x82\xE0\xA4\x85\xE0\xA4\x95\xE0\xA4\xB8\xE0\xA4\xB0\xE0\xA4\xAC\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\x8F\xE0\xA4\xB5\xE0\xA4\xB9\xE0\xA4\xBE\xE0\xA4\x82\xE0\xA4\xB8\xE0" ~
    "\xA5\x8D\xE0\xA4\xA5\xE0\xA4\xB2\xE0\xA4\xAE\xE0\xA4\xBF\xE0\xA4\xB2\xE0\xA5\x87\xE0\xA4\xB2\xE0\xA5\x87\xE0\xA4\x96\xE0\xA4\x95\xE0\xA4\xB5\xE0\xA4\xBF\xE0\xA4\xB7\xE0\xA4\xAF\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\x82\xE0\xA4\xB8\xE0\xA4\xAE\xE0\xA5" ~
    "\x82\xE0\xA4\xB9\xE0\xA4\xA5\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA4\xBE\xD8\xAA\xD8\xB3\xD8\xAA\xD8\xB7\xD9\x8A\xD8\xB9\xD9\x85\xD8\xB4\xD8\xA7\xD8\xB1\xD9\x83\xD8\xA9\xD8\xA8\xD9\x88\xD8\xA7\xD8\xB3\xD8\xB7\xD8\xA9\xD8\xA7\xD9\x84\xD8\xB5\xD9\x81\xD8\xAD\xD8\xA9" ~
    "\xD9\x85\xD9\x88\xD8\xA7\xD8\xB6\xD9\x8A\xD8\xB9\xD8\xA7\xD9\x84\xD8\xAE\xD8\xA7\xD8\xB5\xD8\xA9\xD8\xA7\xD9\x84\xD9\x85\xD8\xB2\xD9\x8A\xD8\xAF\xD8\xA7\xD9\x84\xD8\xB9\xD8\xA7\xD9\x85\xD8\xA9\xD8\xA7\xD9\x84\xD9\x83\xD8\xA7\xD8\xAA\xD8\xA8\xD8\xA7\xD9\x84" ~
    "\xD8\xB1\xD8\xAF\xD9\x88\xD8\xAF\xD8\xA8\xD8\xB1\xD9\x86\xD8\xA7\xD9\x85\xD8\xAC\xD8\xA7\xD9\x84\xD8\xAF\xD9\x88\xD9\x84\xD8\xA9\xD8\xA7\xD9\x84\xD8\xB9\xD8\xA7\xD9\x84\xD9\x85\xD8\xA7\xD9\x84\xD9\x85\xD9\x88\xD9\x82\xD8\xB9\xD8\xA7\xD9\x84\xD8\xB9\xD8\xB1" ~
    "\xD8\xA8\xD9\x8A\xD8\xA7\xD9\x84\xD8\xB3\xD8\xB1\xD9\x8A\xD8\xB9\xD8\xA7\xD9\x84\xD8\xAC\xD9\x88\xD8\xA7\xD9\x84\xD8\xA7\xD9\x84\xD8\xB0\xD9\x87\xD8\xA7\xD8\xA8\xD8\xA7\xD9\x84\xD8\xAD\xD9\x8A\xD8\xA7\xD8\xA9\xD8\xA7\xD9\x84\xD8\xAD\xD9\x82\xD9\x88\xD9\x82" ~
    "\xD8\xA7\xD9\x84\xD9\x83\xD8\xB1\xD9\x8A\xD9\x85\xD8\xA7\xD9\x84\xD8\xB9\xD8\xB1\xD8\xA7\xD9\x82\xD9\x85\xD8\xAD\xD9\x81\xD9\x88\xD8\xB8\xD8\xA9\xD8\xA7\xD9\x84\xD8\xAB\xD8\xA7\xD9\x86\xD9\x8A\xD9\x85\xD8\xB4\xD8\xA7\xD9\x87\xD8\xAF\xD8\xA9\xD8\xA7\xD9\x84" ~
    "\xD9\x85\xD8\xB1\xD8\xA3\xD8\xA9\xD8\xA7\xD9\x84\xD9\x82\xD8\xB1\xD8\xA2\xD9\x86\xD8\xA7\xD9\x84\xD8\xB4\xD8\xA8\xD8\xA7\xD8\xA8\xD8\xA7\xD9\x84\xD8\xAD\xD9\x88\xD8\xA7\xD8\xB1\xD8\xA7\xD9\x84\xD8\xAC\xD8\xAF\xD9\x8A\xD8\xAF\xD8\xA7\xD9\x84\xD8\xA3\xD8\xB3" ~
    "\xD8\xB1\xD8\xA9\xD8\xA7\xD9\x84\xD8\xB9\xD9\x84\xD9\x88\xD9\x85\xD9\x85\xD8\xAC\xD9\x85\xD9\x88\xD8\xB9\xD8\xA9\xD8\xA7\xD9\x84\xD8\xB1\xD8\xAD\xD9\x85\xD9\x86\xD8\xA7\xD9\x84\xD9\x86\xD9\x82\xD8\xA7\xD8\xB7\xD9\x81\xD9\x84\xD8\xB3\xD8\xB7\xD9\x8A\xD9\x86" ~
    "\xD8\xA7\xD9\x84\xD9\x83\xD9\x88\xD9\x8A\xD8\xAA\xD8\xA7\xD9\x84\xD8\xAF\xD9\x86\xD9\x8A\xD8\xA7\xD8\xA8\xD8\xB1\xD9\x83\xD8\xA7\xD8\xAA\xD9\x87\xD8\xA7\xD9\x84\xD8\xB1\xD9\x8A\xD8\xA7\xD8\xB6\xD8\xAA\xD8\xAD\xD9\x8A\xD8\xA7\xD8\xAA\xD9\x8A\xD8\xA8\xD8\xAA" ~
    "\xD9\x88\xD9\x82\xD9\x8A\xD8\xAA\xD8\xA7\xD9\x84\xD8\xA3\xD9\x88\xD9\x84\xD9\x89\xD8\xA7\xD9\x84\xD8\xA8\xD8\xB1\xD9\x8A\xD8\xAF\xD8\xA7\xD9\x84\xD9\x83\xD9\x84\xD8\xA7\xD9\x85\xD8\xA7\xD9\x84\xD8\xB1\xD8\xA7\xD8\xA8\xD8\xB7\xD8\xA7\xD9\x84\xD8\xB4\xD8\xAE" ~
    "\xD8\xB5\xD9\x8A\xD8\xB3\xD9\x8A\xD8\xA7\xD8\xB1\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD8\xAB\xD8\xA7\xD9\x84\xD8\xAB\xD8\xA7\xD9\x84\xD8\xB5\xD9\x84\xD8\xA7\xD8\xA9\xD8\xA7\xD9\x84\xD8\xAD\xD8\xAF\xD9\x8A\xD8\xAB\xD8\xA7\xD9\x84\xD8\xB2\xD9\x88\xD8\xA7\xD8\xB1" ~
    "\xD8\xA7\xD9\x84\xD8\xAE\xD9\x84\xD9\x8A\xD8\xAC\xD8\xA7\xD9\x84\xD8\xAC\xD9\x85\xD9\x8A\xD8\xB9\xD8\xA7\xD9\x84\xD8\xB9\xD8\xA7\xD9\x85\xD9\x87\xD8\xA7\xD9\x84\xD8\xAC\xD9\x85\xD8\xA7\xD9\x84\xD8\xA7\xD9\x84\xD8\xB3\xD8\xA7\xD8\xB9\xD8\xA9\xD9\x85\xD8\xB4" ~
    "\xD8\xA7\xD9\x87\xD8\xAF\xD9\x87\xD8\xA7\xD9\x84\xD8\xB1\xD8\xA6\xD9\x8A\xD8\xB3\xD8\xA7\xD9\x84\xD8\xAF\xD8\xAE\xD9\x88\xD9\x84\xD8\xA7\xD9\x84\xD9\x81\xD9\x86\xD9\x8A\xD8\xA9\xD8\xA7\xD9\x84\xD9\x83\xD8\xAA\xD8\xA7\xD8\xA8\xD8\xA7\xD9\x84\xD8\xAF\xD9\x88" ~
    "\xD8\xB1\xD9\x8A\xD8\xA7\xD9\x84\xD8\xAF\xD8\xB1\xD9\x88\xD8\xB3\xD8\xA7\xD8\xB3\xD8\xAA\xD8\xBA\xD8\xB1\xD9\x82\xD8\xAA\xD8\xB5\xD8\xA7\xD9\x85\xD9\x8A\xD9\x85\xD8\xA7\xD9\x84\xD8\xA8\xD9\x86\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD8\xB9\xD8\xB8\xD9\x8A\xD9\x85" ~
    "entertainmentunderstanding = function().jpg\" width=\"configuratio" ~
    "n.png\" width=\"<body class=\"Math.random()contemporary United Stat" ~
    "escircumstances.appendChild(organizations<span class=\"\"><img src" ~
    "=\"/distinguishedthousands of communicationclear\"></div>investiga" ~
    "tionfavicon.ico\" margin-right:based on the Massachusettstable bo" ~
    "rder=internationalalso known aspronunciationbackground:#fpadding" ~
    "-left:For example, miscellaneous&lt;/math&gt;psychologicalin par" ~
    "ticularearch\" type=\"form method=\"as opposed toSupreme Courtoccas" ~
    "ionally Additionally,North Americapx;backgroundopportunitiesEnte" ~
    "rtainment.toLowerCase(manufacturingprofessional combined withFor" ~
    " instance,consisting of\" maxlength=\"return false;consciousnessMe" ~
    "diterraneanextraordinaryassassinationsubsequently button type=\"t" ~
    "he number ofthe original comprehensiverefers to the</ul>\x0A</div>\x0A" ~
    "philosophicallocation.hrefwas publishedSan Francisco(function(){" ~
    "\x0A<div id=\"mainsophisticatedmathematical /head>\x0D\x0A<bodysuggests th" ~
    "atdocumentationconcentrationrelationshipsmay have been(for examp" ~
    "le,This article in some casesparts of the definition ofGreat Bri" ~
    "tain cellpadding=equivalent toplaceholder=\"; font-size: justific" ~
    "ationbelieved thatsuffered fromattempted to leader of thecript\" " ~
    "src=\"/(function() {are available\x0A\x09<link rel=\" src='http://intere" ~
    "sted inconventional \" alt=\"\" /></are generallyhas also beenmost " ~
    "popular correspondingcredited withtyle=\"border:</a></span></.gif" ~
    "\" width=\"<iframe src=\"table class=\"inline-block;according to tog" ~
    "ether withapproximatelyparliamentarymore and moredisplay:none;tr" ~
    "aditionallypredominantly&nbsp;|&nbsp;&nbsp;</span> cellspacing=<" ~
    "input name=\"or\" content=\"controversialproperty=\"og:/x-shockwave-" ~
    "demonstrationsurrounded byNevertheless,was the firstconsiderable" ~
    " Although the collaborationshould not beproportion of<span style" ~
    "=\"known as the shortly afterfor instance,described as /head>\x0A<bo" ~
    "dy starting withincreasingly the fact thatdiscussion ofmiddle of" ~
    " thean individualdifficult to point of viewhomosexualityacceptan" ~
    "ce of</span></div>manufacturersorigin of thecommonly usedimporta" ~
    "nce ofdenominationsbackground: #length of thedeterminationa sign" ~
    "ificant\" border=\"0\">revolutionaryprinciples ofis consideredwas d" ~
    "evelopedIndo-Europeanvulnerable toproponents ofare sometimesclos" ~
    "er to theNew York City name=\"searchattributed tocourse of themat" ~
    "hematicianby the end ofat the end of\" border=\"0\" technological.r" ~
    "emoveClass(branch of theevidence that![endif]-->\x0D\x0AInstitute of i" ~
    "nto a singlerespectively.and thereforeproperties ofis located in" ~
    "some of whichThere is alsocontinued to appearance of &amp;ndash;" ~
    " describes theconsiderationauthor of theindependentlyequipped wi" ~
    "thdoes not have</a><a href=\"confused with<link href=\"/at the age" ~
    " ofappear in theThese includeregardless ofcould be used style=&q" ~
    "uot;several timesrepresent thebody>\x0A</html>thought to bepopulati" ~
    "on ofpossibilitiespercentage ofaccess to thean attempt toproduct" ~
    "ion ofjquery/jquerytwo differentbelong to theestablishmentreplac" ~
    "ing thedescription\" determine theavailable forAccording to wide " ~
    "range of\x09<div class=\"more commonlyorganisationsfunctionalitywas " ~
    "completed &amp;mdash; participationthe characteran additionalapp" ~
    "ears to befact that thean example ofsignificantlyonmouseover=\"be" ~
    "cause they async = true;problems withseems to havethe result of " ~
    "src=\"http://familiar withpossession offunction () {took place in" ~
    "and sometimessubstantially<span></span>is often usedin an attemp" ~
    "tgreat deal ofEnvironmentalsuccessfully virtually all20th centur" ~
    "y,professionalsnecessary to determined bycompatibilitybecause it" ~
    " isDictionary ofmodificationsThe followingmay refer to:Consequen" ~
    "tly,Internationalalthough somethat would beworld's firstclassifi" ~
    "ed asbottom of the(particularlyalign=\"left\" most commonlybasis f" ~
    "or thefoundation ofcontributionspopularity ofcenter of theto red" ~
    "uce thejurisdictionsapproximation onmouseout=\"New Testamentcolle" ~
    "ction of</span></a></in the Unitedfilm director-strict.dtd\">has " ~
    "been usedreturn to thealthough thischange in theseveral otherbut" ~
    " there areunprecedentedis similar toespecially inweight: bold;is" ~
    " called thecomputationalindicate thatrestricted to\x09<meta name=\"a" ~
    "re typicallyconflict withHowever, the An example ofcompared with" ~
    "quantities ofrather than aconstellationnecessary forreported tha" ~
    "tspecificationpolitical and&nbsp;&nbsp;<references tothe same ye" ~
    "arGovernment ofgeneration ofhave not beenseveral yearscommitment" ~
    " to\x09\x09<ul class=\"visualization19th century,practitionersthat he w" ~
    "ouldand continuedoccupation ofis defined ascentre of thethe amou" ~
    "nt of><div style=\"equivalent ofdifferentiatebrought aboutmargin-" ~
    "left: automaticallythought of asSome of these\x0A<div class=\"input " ~
    "class=\"replaced withis one of theeducation andinfluenced byreput" ~
    "ation as\x0A<meta name=\"accommodation</div>\x0A</div>large part ofInst" ~
    "itute forthe so-called against the In this case,was appointedcla" ~
    "imed to beHowever, thisDepartment ofthe remainingeffect on thepa" ~
    "rticularly deal with the\x0A<div style=\"almost alwaysare currentlye" ~
    "xpression ofphilosophy offor more thancivilizationson the island" ~
    "selectedIndexcan result in\" value=\"\" />the structure /></a></div" ~
    ">Many of thesecaused by theof the Unitedspan class=\"mcan be trac" ~
    "edis related tobecame one ofis frequentlyliving in thetheoretica" ~
    "llyFollowing theRevolutionarygovernment inis determinedthe polit" ~
    "icalintroduced insufficient todescription\">short storiesseparati" ~
    "on ofas to whetherknown for itswas initiallydisplay:blockis an e" ~
    "xamplethe principalconsists of arecognized as/body></html>a subs" ~
    "tantialreconstructedhead of stateresistance toundergraduateThere" ~
    " are twogravitationalare describedintentionallyserved as theclas" ~
    "s=\"headeropposition tofundamentallydominated theand the otherall" ~
    "iance withwas forced torespectively,and politicalin support ofpe" ~
    "ople in the20th century.and publishedloadChartbeatto understandm" ~
    "ember statesenvironmentalfirst half ofcountries andarchitectural" ~
    "be consideredcharacterizedclearIntervalauthoritativeFederation o" ~
    "fwas succeededand there area consequencethe Presidentalso includ" ~
    "edfree softwaresuccession ofdeveloped thewas destroyedaway from " ~
    "the;\x0A</script>\x0A<although theyfollowed by amore powerfulresulted " ~
    "in aUniversity ofHowever, manythe presidentHowever, someis thoug" ~
    "ht tountil the endwas announcedare importantalso includes><input" ~
    " type=the center of DO NOT ALTERused to referthemes/?sort=that h" ~
    "ad beenthe basis forhas developedin the summercomparativelydescr" ~
    "ibed thesuch as thosethe resultingis impossiblevarious otherSout" ~
    "h Africanhave the sameeffectivenessin which case; text-align:str" ~
    "ucture and; background:regarding thesupported theis also knownst" ~
    "yle=\"marginincluding thebahasa Melayunorsk bokm\xC3\xA5lnorsk nynorsks" ~
    "loven\xC5\xA1\xC4\x8Dinainternacionalcalificaci\xC3\xB3ncomunicaci\xC3\xB3nconstrucci\xC3\xB3n" ~
    "\"><div class=\"disambiguationDomainName', 'administrationsimultan" ~
    "eouslytransportationInternational margin-bottom:responsibility<!" ~
    "[endif]-->\x0A</><meta name=\"implementationinfrastructurerepresenta" ~
    "tionborder-bottom:</head>\x0A<body>=http%3A%2F%2F<form method=\"meth" ~
    "od=\"post\" /favicon.ico\" });\x0A</script>\x0A.setAttribute(Administrati" ~
    "on= new Array();<![endif]-->\x0D\x0Adisplay:block;Unfortunately,\">&nbs" ~
    "p;</div>/favicon.ico\">='stylesheet' identification, for example," ~
    "<li><a href=\"/an alternativeas a result ofpt\"></script>\x0Atype=\"su" ~
    "bmit\" \x0A(function() {recommendationform action=\"/transformationre" ~
    "construction.style.display According to hidden\" name=\"along with" ~
    " thedocument.body.approximately Communicationspost\" action=\"mean" ~
    "ing &quot;--<![endif]-->Prime Ministercharacteristic</a> <a clas" ~
    "s=the history of onmouseover=\"the governmenthref=\"https://was or" ~
    "iginallywas introducedclassificationrepresentativeare considered" ~
    "<![endif]-->\x0A\x0Adepends on theUniversity of in contrast to placeho" ~
    "lder=\"in the case ofinternational constitutionalstyle=\"border-: " ~
    "function() {Because of the-strict.dtd\">\x0A<table class=\"accompanie" ~
    "d byaccount of the<script src=\"/nature of the the people in in a" ~
    "ddition tos); js.id = id\" width=\"100%\"regarding the Roman Cathol" ~
    "ican independentfollowing the .gif\" width=\"1the following discri" ~
    "minationarchaeologicalprime minister.js\"></script>combination of" ~
    " marginwidth=\"createElement(w.attachEvent(</a></td></tr>src=\"htt" ~
    "ps://aIn particular, align=\"left\" Czech RepublicUnited Kingdomco" ~
    "rrespondenceconcluded that.html\" title=\"(function () {comes from" ~
    " theapplication of<span class=\"sbelieved to beement('script'</a>" ~
    "\x0A</li>\x0A<livery different><span class=\"option value=\"(also known " ~
    "as\x09<li><a href=\"><input name=\"separated fromreferred to as valig" ~
    "n=\"top\">founder of theattempting to carbon dioxide\x0A\x0A<div class=\"" ~
    "class=\"search-/body>\x0A</html>opportunity tocommunications</head>\x0D" ~
    "\x0A<body style=\"width:Ti\xE1\xBA\xBFng Vi\xE1\xBB\x87tchanges in theborder-color:#0\"" ~
    " border=\"0\" </span></div><was discovered\" type=\"text\" );\x0A</scrip" ~
    "t>\x0A\x0ADepartment of ecclesiasticalthere has beenresulting from</bo" ~
    "dy></html>has never beenthe first timein response toautomaticall" ~
    "y </div>\x0A\x0A<div iwas consideredpercent of the\" /></a></div>collec" ~
    "tion of descended fromsection of theaccept-charsetto be confused" ~
    "member of the padding-right:translation ofinterpretation href='h" ~
    "ttp://whether or notThere are alsothere are manya small numberot" ~
    "her parts ofimpossible to  class=\"buttonlocated in the. However," ~
    " theand eventuallyAt the end of because of itsrepresents the<for" ~
    "m action=\" method=\"post\"it is possiblemore likely toan increase " ~
    "inhave also beencorresponds toannounced thatalign=\"right\">many c" ~
    "ountriesfor many yearsearliest knownbecause it waspt\"></script>\x0D" ~
    " valign=\"top\" inhabitants offollowing year\x0D\x0A<div class=\"million " ~
    "peoplecontroversial concerning theargue that thegovernment anda " ~
    "reference totransferred todescribing the style=\"color:although t" ~
    "herebest known forsubmit\" name=\"multiplicationmore than one reco" ~
    "gnition ofCouncil of theedition of the  <meta name=\"Entertainmen" ~
    "t away from the ;margin-right:at the time ofinvestigationsconnec" ~
    "ted withand many otheralthough it isbeginning with <span class=\"" ~
    "descendants of<span class=\"i align=\"right\"</head>\x0A<body aspects " ~
    "of thehas since beenEuropean Unionreminiscent ofmore difficultVi" ~
    "ce Presidentcomposition ofpassed throughmore importantfont-size:" ~
    "11pxexplanation ofthe concept ofwritten in the\x09<span class=\"is o" ~
    "ne of the resemblance toon the groundswhich containsincluding th" ~
    "e defined by thepublication ofmeans that theoutside of thesuppor" ~
    "t of the<input class=\"<span class=\"t(Math.random()most prominent" ~
    "description ofConstantinoplewere published<div class=\"seappears " ~
    "in the1\" height=\"1\" most importantwhich includeswhich had beende" ~
    "struction ofthe population\x0A\x09<div class=\"possibility ofsometimes " ~
    "usedappear to havesuccess of theintended to bepresent in thestyl" ~
    "e=\"clear:b\x0D\x0A</script>\x0D\x0A<was founded ininterview with_id\" content" ~
    "=\"capital of the\x0D\x0A<link rel=\"srelease of thepoint out thatxMLHtt" ~
    "pRequestand subsequentsecond largestvery importantspecifications" ~
    "surface of theapplied to theforeign policy_setDomainNameestablis" ~
    "hed inis believed toIn addition tomeaning of theis named afterto" ~
    " protect theis representedDeclaration ofmore efficientClassifica" ~
    "tionother forms ofhe returned to<span class=\"cperformance of(fun" ~
    "ction() {\x0Dif and only ifregions of theleading to therelations wi" ~
    "thUnited Nationsstyle=\"height:other than theype\" content=\"Associ" ~
    "ation of\x0A</head>\x0A<bodylocated on theis referred to(including the" ~
    "concentrationsthe individualamong the mostthan any other/>\x0A<link" ~
    " rel=\" return false;the purpose ofthe ability to;color:#fff}\x0A.\x0A<" ~
    "span class=\"the subject ofdefinitions of>\x0D\x0A<link rel=\"claim that" ~
    " thehave developed<table width=\"celebration ofFollowing the to d" ~
    "istinguish<span class=\"btakes place inunder the namenoted that t" ~
    "he><![endif]-->\x0Astyle=\"margin-instead of theintroduced thethe pr" ~
    "ocess ofincreasing thedifferences inestimated thatespecially the" ~
    "/div><div id=\"was eventuallythroughout histhe differencesomethin" ~
    "g thatspan></span></significantly ></script>\x0D\x0A\x0D\x0Aenvironmental to" ~
    " prevent thehave been usedespecially forunderstand theis essenti" ~
    "allywere the firstis the largesthave been made\" src=\"http://inte" ~
    "rpreted assecond half ofcrolling=\"no\" is composed ofII, Holy Rom" ~
    "anis expected tohave their owndefined as thetraditionally have d" ~
    "ifferentare often usedto ensure thatagreement withcontaining the" ~
    "are frequentlyinformation onexample is theresulting in a</a></li" ~
    "></ul> class=\"footerand especiallytype=\"button\" </span></span>wh" ~
    "ich included>\x0A<meta name=\"considered thecarried out byHowever, i" ~
    "t isbecame part ofin relation topopular in thethe capital ofwas " ~
    "officiallywhich has beenthe History ofalternative todifferent fr" ~
    "omto support thesuggested thatin the process  <div class=\"the fo" ~
    "undationbecause of hisconcerned withthe universityopposed to the" ~
    "the context of<span class=\"ptext\" name=\"q\"\x09\x09<div class=\"the scie" ~
    "ntificrepresented bymathematicianselected by thethat have been><" ~
    "div class=\"cdiv id=\"headerin particular,converted into);\x0A</scrip" ~
    "t>\x0A<philosophical srpskohrvatskiti\xE1\xBA\xBFng Vi\xE1\xBB\x87t\xD0\xA0\xD1\x83\xD1\x81\xD1\x81\xD0\xBA\xD0\xB8\xD0\xB9\xD1\x80\xD1\x83" ~
    "\xD1\x81\xD1\x81\xD0\xBA\xD0\xB8\xD0\xB9investigaci\xC3\xB3nparticipaci\xC3\xB3n\xD0\xBA\xD0\xBE\xD1\x82\xD0\xBE\xD1\x80\xD1\x8B\xD0\xB5\xD0\xBE\xD0\xB1\xD0\xBB\xD0\xB0\xD1\x81\xD1\x82" ~
    "\xD0\xB8\xD0\xBA\xD0\xBE\xD1\x82\xD0\xBE\xD1\x80\xD1\x8B\xD0\xB9\xD1\x87\xD0\xB5\xD0\xBB\xD0\xBE\xD0\xB2\xD0\xB5\xD0\xBA\xD1\x81\xD0\xB8\xD1\x81\xD1\x82\xD0\xB5\xD0\xBC\xD1\x8B\xD0\x9D\xD0\xBE\xD0\xB2\xD0\xBE\xD1\x81\xD1\x82\xD0\xB8\xD0\xBA\xD0\xBE\xD1\x82" ~
    "\xD0\xBE\xD1\x80\xD1\x8B\xD1\x85\xD0\xBE\xD0\xB1\xD0\xBB\xD0\xB0\xD1\x81\xD1\x82\xD1\x8C\xD0\xB2\xD1\x80\xD0\xB5\xD0\xBC\xD0\xB5\xD0\xBD\xD0\xB8\xD0\xBA\xD0\xBE\xD1\x82\xD0\xBE\xD1\x80\xD0\xB0\xD1\x8F\xD1\x81\xD0\xB5\xD0\xB3\xD0\xBE\xD0\xB4\xD0\xBD\xD1\x8F" ~
    "\xD1\x81\xD0\xBA\xD0\xB0\xD1\x87\xD0\xB0\xD1\x82\xD1\x8C\xD0\xBD\xD0\xBE\xD0\xB2\xD0\xBE\xD1\x81\xD1\x82\xD0\xB8\xD0\xA3\xD0\xBA\xD1\x80\xD0\xB0\xD0\xB8\xD0\xBD\xD1\x8B\xD0\xB2\xD0\xBE\xD0\xBF\xD1\x80\xD0\xBE\xD1\x81\xD1\x8B\xD0\xBA\xD0\xBE\xD1\x82\xD0\xBE" ~
    "\xD1\x80\xD0\xBE\xD0\xB9\xD1\x81\xD0\xB4\xD0\xB5\xD0\xBB\xD0\xB0\xD1\x82\xD1\x8C\xD0\xBF\xD0\xBE\xD0\xBC\xD0\xBE\xD1\x89\xD1\x8C\xD1\x8E\xD1\x81\xD1\x80\xD0\xB5\xD0\xB4\xD1\x81\xD1\x82\xD0\xB2\xD0\xBE\xD0\xB1\xD1\x80\xD0\xB0\xD0\xB7\xD0\xBE\xD0\xBC\xD1\x81" ~
    "\xD1\x82\xD0\xBE\xD1\x80\xD0\xBE\xD0\xBD\xD1\x8B\xD1\x83\xD1\x87\xD0\xB0\xD1\x81\xD1\x82\xD0\xB8\xD0\xB5\xD1\x82\xD0\xB5\xD1\x87\xD0\xB5\xD0\xBD\xD0\xB8\xD0\xB5\xD0\x93\xD0\xBB\xD0\xB0\xD0\xB2\xD0\xBD\xD0\xB0\xD1\x8F\xD0\xB8\xD1\x81\xD1\x82\xD0\xBE\xD1\x80" ~
    "\xD0\xB8\xD0\xB8\xD1\x81\xD0\xB8\xD1\x81\xD1\x82\xD0\xB5\xD0\xBC\xD0\xB0\xD1\x80\xD0\xB5\xD1\x88\xD0\xB5\xD0\xBD\xD0\xB8\xD1\x8F\xD0\xA1\xD0\xBA\xD0\xB0\xD1\x87\xD0\xB0\xD1\x82\xD1\x8C\xD0\xBF\xD0\xBE\xD1\x8D\xD1\x82\xD0\xBE\xD0\xBC\xD1\x83\xD1\x81\xD0\xBB" ~
    "\xD0\xB5\xD0\xB4\xD1\x83\xD0\xB5\xD1\x82\xD1\x81\xD0\xBA\xD0\xB0\xD0\xB7\xD0\xB0\xD1\x82\xD1\x8C\xD1\x82\xD0\xBE\xD0\xB2\xD0\xB0\xD1\x80\xD0\xBE\xD0\xB2\xD0\xBA\xD0\xBE\xD0\xBD\xD0\xB5\xD1\x87\xD0\xBD\xD0\xBE\xD1\x80\xD0\xB5\xD1\x88\xD0\xB5\xD0\xBD\xD0\xB8" ~
    "\xD0\xB5\xD0\xBA\xD0\xBE\xD1\x82\xD0\xBE\xD1\x80\xD0\xBE\xD0\xB5\xD0\xBE\xD1\x80\xD0\xB3\xD0\xB0\xD0\xBD\xD0\xBE\xD0\xB2\xD0\xBA\xD0\xBE\xD1\x82\xD0\xBE\xD1\x80\xD0\xBE\xD0\xBC\xD0\xA0\xD0\xB5\xD0\xBA\xD0\xBB\xD0\xB0\xD0\xBC\xD0\xB0\xD8\xA7\xD9\x84\xD9\x85" ~
    "\xD9\x86\xD8\xAA\xD8\xAF\xD9\x89\xD9\x85\xD9\x86\xD8\xAA\xD8\xAF\xD9\x8A\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD9\x85\xD9\x88\xD8\xB6\xD9\x88\xD8\xB9\xD8\xA7\xD9\x84\xD8\xA8\xD8\xB1\xD8\xA7\xD9\x85\xD8\xAC\xD8\xA7\xD9\x84\xD9\x85\xD9\x88\xD8\xA7\xD9\x82\xD8\xB9" ~
    "\xD8\xA7\xD9\x84\xD8\xB1\xD8\xB3\xD8\xA7\xD8\xA6\xD9\x84\xD9\x85\xD8\xB4\xD8\xA7\xD8\xB1\xD9\x83\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD8\xA3\xD8\xB9\xD8\xB6\xD8\xA7\xD8\xA1\xD8\xA7\xD9\x84\xD8\xB1\xD9\x8A\xD8\xA7\xD8\xB6\xD8\xA9\xD8\xA7\xD9\x84\xD8\xAA\xD8\xB5" ~
    "\xD9\x85\xD9\x8A\xD9\x85\xD8\xA7\xD9\x84\xD8\xA7\xD8\xB9\xD8\xB6\xD8\xA7\xD8\xA1\xD8\xA7\xD9\x84\xD9\x86\xD8\xAA\xD8\xA7\xD8\xA6\xD8\xAC\xD8\xA7\xD9\x84\xD8\xA3\xD9\x84\xD8\xB9\xD8\xA7\xD8\xA8\xD8\xA7\xD9\x84\xD8\xAA\xD8\xB3\xD8\xAC\xD9\x8A\xD9\x84\xD8\xA7" ~
    "\xD9\x84\xD8\xA3\xD9\x82\xD8\xB3\xD8\xA7\xD9\x85\xD8\xA7\xD9\x84\xD8\xB6\xD8\xBA\xD8\xB7\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD9\x81\xD9\x8A\xD8\xAF\xD9\x8A\xD9\x88\xD8\xA7\xD9\x84\xD8\xAA\xD8\xB1\xD8\xAD\xD9\x8A\xD8\xA8\xD8\xA7\xD9\x84\xD8\xAC\xD8\xAF\xD9\x8A" ~
    "\xD8\xAF\xD8\xA9\xD8\xA7\xD9\x84\xD8\xAA\xD8\xB9\xD9\x84\xD9\x8A\xD9\x85\xD8\xA7\xD9\x84\xD8\xA3\xD8\xAE\xD8\xA8\xD8\xA7\xD8\xB1\xD8\xA7\xD9\x84\xD8\xA7\xD9\x81\xD9\x84\xD8\xA7\xD9\x85\xD8\xA7\xD9\x84\xD8\xA3\xD9\x81\xD9\x84\xD8\xA7\xD9\x85\xD8\xA7\xD9\x84" ~
    "\xD8\xAA\xD8\xA7\xD8\xB1\xD9\x8A\xD8\xAE\xD8\xA7\xD9\x84\xD8\xAA\xD9\x82\xD9\x86\xD9\x8A\xD8\xA9\xD8\xA7\xD9\x84\xD8\xA7\xD9\x84\xD8\xB9\xD8\xA7\xD8\xA8\xD8\xA7\xD9\x84\xD8\xAE\xD9\x88\xD8\xA7\xD8\xB7\xD8\xB1\xD8\xA7\xD9\x84\xD9\x85\xD8\xAC\xD8\xAA\xD9\x85" ~
    "\xD8\xB9\xD8\xA7\xD9\x84\xD8\xAF\xD9\x8A\xD9\x83\xD9\x88\xD8\xB1\xD8\xA7\xD9\x84\xD8\xB3\xD9\x8A\xD8\xA7\xD8\xAD\xD8\xA9\xD8\xB9\xD8\xA8\xD8\xAF\xD8\xA7\xD9\x84\xD9\x84\xD9\x87\xD8\xA7\xD9\x84\xD8\xAA\xD8\xB1\xD8\xA8\xD9\x8A\xD8\xA9\xD8\xA7\xD9\x84\xD8\xB1" ~
    "\xD9\x88\xD8\xA7\xD8\xA8\xD8\xB7\xD8\xA7\xD9\x84\xD8\xA3\xD8\xAF\xD8\xA8\xD9\x8A\xD8\xA9\xD8\xA7\xD9\x84\xD8\xA7\xD8\xAE\xD8\xA8\xD8\xA7\xD8\xB1\xD8\xA7\xD9\x84\xD9\x85\xD8\xAA\xD8\xAD\xD8\xAF\xD8\xA9\xD8\xA7\xD9\x84\xD8\xA7\xD8\xBA\xD8\xA7\xD9\x86\xD9\x8A" ~
    "cursor:pointer;</title>\x0A<meta \" href=\"http://\"><span class=\"memb" ~
    "ers of the window.locationvertical-align:/a> | <a href=\"<!doctyp" ~
    "e html>media=\"screen\" <option value=\"favicon.ico\" />\x0A\x09\x09<div clas" ~
    "s=\"characteristics\" method=\"get\" /body>\x0A</html>\x0Ashortcut icon\" d" ~
    "ocument.write(padding-bottom:representativessubmit\" value=\"align" ~
    "=\"center\" throughout the science fiction\x0A  <div class=\"submit\" c" ~
    "lass=\"one of the most valign=\"top\"><was established);\x0D\x0A</script>" ~
    "\x0D\x0Areturn false;\">).style.displaybecause of the document.cookie<f" ~
    "orm action=\"/}body{margin:0;Encyclopedia ofversion of the .creat" ~
    "eElement(name\" content=\"</div>\x0A</div>\x0A\x0Aadministrative </body>\x0A</" ~
    "html>history of the \"><input type=\"portion of the as part of the" ~
    " &nbsp;<a href=\"other countries\">\x0A<div class=\"</span></span><In " ~
    "other words,display: block;control of the introduction of/>\x0A<met" ~
    "a name=\"as well as the in recent years\x0D\x0A\x09<div class=\"</div>\x0A\x09</d" ~
    "iv>\x0Ainspired by thethe end of the compatible withbecame known as" ~
    " style=\"margin:.js\"></script>< International there have beenGerm" ~
    "an language style=\"color:#Communist Partyconsistent withborder=\"" ~
    "0\" cell marginheight=\"the majority of\" align=\"centerrelated to t" ~
    "he many different Orthodox Churchsimilar to the />\x0A<link rel=\"sw" ~
    "as one of the until his death})();\x0A</script>other languagescompa" ~
    "red to theportions of thethe Netherlandsthe most commonbackgroun" ~
    "d:url(argued that thescrolling=\"no\" included in theNorth America" ~
    "n the name of theinterpretationsthe traditionaldevelopment of fr" ~
    "equently useda collection ofvery similar tosurrounding theexampl" ~
    "e of thisalign=\"center\">would have beenimage_caption =attached t" ~
    "o thesuggesting thatin the form of involved in theis derived fro" ~
    "mnamed after theIntroduction torestrictions on style=\"width: can" ~
    " be used to the creation ofmost important information andresulte" ~
    "d in thecollapse of theThis means thatelements of thewas replace" ~
    "d byanalysis of theinspiration forregarded as themost successful" ~
    "known as &quot;a comprehensiveHistory of the were consideredretu" ~
    "rned to theare referred toUnsourced image>\x0A\x09<div class=\"consists" ~
    " of thestopPropagationinterest in theavailability ofappears to h" ~
    "aveelectromagneticenableServices(function of theIt is important<" ~
    "/script></div>function(){var relative to theas a result of the p" ~
    "osition ofFor example, in method=\"post\" was followed by&amp;mdas" ~
    "h; thethe applicationjs\"></script>\x0D\x0Aul></div></div>after the dea" ~
    "thwith respect tostyle=\"padding:is particularlydisplay:inline; t" ~
    "ype=\"submit\" is divided into\xE4\xB8\xAD\xE6\x96\x87 (\xE7\xAE\x80\xE4\xBD\x93)responsabilidadadmini" ~
    "straci\xC3\xB3ninternacionalescorrespondiente\xE0\xA4\x89\xE0\xA4\xAA\xE0\xA4\xAF\xE0\xA5\x8B\xE0\xA4\x97\xE0\xA4\xAA\xE0\xA5\x82\xE0\xA4\xB0\xE0" ~
    "\xA5\x8D\xE0\xA4\xB5\xE0\xA4\xB9\xE0\xA4\xAE\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA5\x87\xE0\xA4\xB2\xE0\xA5\x8B\xE0\xA4\x97\xE0\xA5\x8B\xE0\xA4\x82\xE0\xA4\x9A\xE0\xA5\x81\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\xB5\xE0\xA4\xB2\xE0\xA5\x87\xE0\xA4\x95\xE0\xA4\xBF\xE0\xA4" ~
    "\xA8\xE0\xA4\xB8\xE0\xA4\xB0\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xAA\xE0\xA5\x81\xE0\xA4\xB2\xE0\xA4\xBF\xE0\xA4\xB8\xE0\xA4\x96\xE0\xA5\x8B\xE0\xA4\x9C\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\x9A\xE0\xA4\xBE\xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\x8F\xE0\xA4\xAD" ~
    "\xE0\xA5\x87\xE0\xA4\x9C\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\xB6\xE0\xA4\xBE\xE0\xA4\xAE\xE0\xA4\xBF\xE0\xA4\xB2\xE0\xA4\xB9\xE0\xA4\xAE\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA5\x80\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\x97\xE0\xA4\xB0\xE0\xA4\xA3\xE0\xA4\xAC\xE0\xA4\xA8\xE0" ~
    "\xA4\xBE\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\x95\xE0\xA5\x81\xE0\xA4\xAE\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xAC\xE0\xA5\x8D\xE0\xA4\xB2\xE0\xA5\x89\xE0\xA4\x97\xE0\xA4\xAE\xE0\xA4\xBE\xE0\xA4\xB2\xE0\xA4\xBF\xE0\xA4\x95\xE0\xA4\xAE\xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4" ~
    "\xB2\xE0\xA4\xBE\xE0\xA4\xAA\xE0\xA5\x83\xE0\xA4\xB7\xE0\xA5\x8D\xE0\xA4\xA0\xE0\xA4\xAC\xE0\xA4\xA2\xE0\xA4\xBC\xE0\xA4\xA4\xE0\xA5\x87\xE0\xA4\xAD\xE0\xA4\xBE\xE0\xA4\x9C\xE0\xA4\xAA\xE0\xA4\xBE\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB2\xE0\xA4\xBF\xE0\xA4\x95" ~
    "\xE0\xA4\x9F\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA5\x87\xE0\xA4\xA8\xE0\xA4\x96\xE0\xA4\xBF\xE0\xA4\xB2\xE0\xA4\xBE\xE0\xA4\xAB\xE0\xA4\xA6\xE0\xA5\x8C\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xBE\xE0\xA4\xAE\xE0\xA4\xB2\xE0\xA5\x87\xE0\xA4\xAE\xE0" ~
    "\xA4\xA4\xE0\xA4\xA6\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA4\xAC\xE0\xA4\xBE\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xB5\xE0\xA4\xBF\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB8\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xAF\xE0\xA5\x8B\xE0\xA4\x82\xE0\xA4\x9A\xE0\xA4\xBE\xE0\xA4" ~
    "\xB9\xE0\xA4\xA4\xE0\xA5\x87\xE0\xA4\xAA\xE0\xA4\xB9\xE0\xA5\x81\xE0\xA4\x81\xE0\xA4\x9A\xE0\xA4\xAC\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xB8\xE0\xA4\x82\xE0\xA4\xB5\xE0\xA4\xBE\xE0\xA4\xA6\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\x96\xE0\xA4\xA8" ~
    "\xE0\xA5\x87\xE0\xA4\xAA\xE0\xA4\xBF\xE0\xA4\x9B\xE0\xA4\xB2\xE0\xA5\x87\xE0\xA4\xB5\xE0\xA4\xBF\xE0\xA4\xB6\xE0\xA5\x87\xE0\xA4\xB7\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\x9C\xE0\xA5\x8D\xE0\xA4\xAF\xE0\xA4\x89\xE0\xA4\xA4\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA4\xB0\xE0" ~
    "\xA4\xAE\xE0\xA5\x81\xE0\xA4\x82\xE0\xA4\xAC\xE0\xA4\x88\xE0\xA4\xA6\xE0\xA5\x8B\xE0\xA4\xA8\xE0\xA5\x8B\xE0\xA4\x82\xE0\xA4\x89\xE0\xA4\xAA\xE0\xA4\x95\xE0\xA4\xB0\xE0\xA4\xA3\xE0\xA4\xAA\xE0\xA4\xA2\xE0\xA4\xBC\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\xB8\xE0\xA5" ~
    "\x8D\xE0\xA4\xA5\xE0\xA4\xBF\xE0\xA4\xA4\xE0\xA4\xAB\xE0\xA4\xBF\xE0\xA4\xB2\xE0\xA5\x8D\xE0\xA4\xAE\xE0\xA4\xAE\xE0\xA5\x81\xE0\xA4\x96\xE0\xA5\x8D\xE0\xA4\xAF\xE0\xA4\x85\xE0\xA4\x9A\xE0\xA5\x8D\xE0\xA4\x9B\xE0\xA4\xBE\xE0\xA4\x9B\xE0\xA5\x82\xE0\xA4\x9F" ~
    "\xE0\xA4\xA4\xE0\xA5\x80\xE0\xA4\xB8\xE0\xA4\x82\xE0\xA4\x97\xE0\xA5\x80\xE0\xA4\xA4\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\x8F\xE0\xA4\x97\xE0\xA4\xBE\xE0\xA4\xB5\xE0\xA4\xBF\xE0\xA4\xAD\xE0\xA4\xBE\xE0\xA4\x97\xE0\xA4\x98\xE0\xA4\xA3\xE0\xA5\x8D\xE0\xA4\x9F\xE0" ~
    "\xA5\x87\xE0\xA4\xA6\xE0\xA5\x82\xE0\xA4\xB8\xE0\xA4\xB0\xE0\xA5\x87\xE0\xA4\xA6\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA5\x8B\xE0\xA4\x82\xE0\xA4\xB9\xE0\xA4\xA4\xE0\xA5\x8D\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xB8\xE0\xA5\x87\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB8\xE0\xA4" ~
    "\x97\xE0\xA4\xBE\xE0\xA4\x82\xE0\xA4\xA7\xE0\xA5\x80\xE0\xA4\xB5\xE0\xA4\xBF\xE0\xA4\xB6\xE0\xA5\x8D\xE0\xA4\xB5\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\xA4\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\xA6\xE0\xA5\x88\xE0\xA4\x9F\xE0\xA5\x8D\xE0\xA4\xB8\xE0\xA4\xA8\xE0\xA4\x95" ~
    "\xE0\xA5\x8D\xE0\xA4\xB6\xE0\xA4\xBE\xE0\xA4\xB8\xE0\xA4\xBE\xE0\xA4\xAE\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\x85\xE0\xA4\xA6\xE0\xA4\xBE\xE0\xA4\xB2\xE0\xA4\xA4\xE0\xA4\xAC\xE0\xA4\xBF\xE0\xA4\x9C\xE0\xA4\xB2\xE0\xA5\x80\xE0\xA4\xAA\xE0\xA5\x81\xE0\xA4\xB0\xE0" ~
    "\xA5\x82\xE0\xA4\xB7\xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\x82\xE0\xA4\xA6\xE0\xA5\x80\xE0\xA4\xAE\xE0\xA4\xBF\xE0\xA4\xA4\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\x95\xE0\xA4\xB5\xE0\xA4\xBF\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA5\x81\xE0\xA4\xAA\xE0\xA4\xAF\xE0\xA5" ~
    "\x87\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA5\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA4\x95\xE0\xA4\xB0\xE0\xA5\x8B\xE0\xA4\xA1\xE0\xA4\xBC\xE0\xA4\xAE\xE0\xA5\x81\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA4\xAF\xE0\xA5\x8B\xE0\xA4\x9C\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\x95" ~
    "\xE0\xA5\x83\xE0\xA4\xAA\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xAA\xE0\xA5\x8B\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\x9F\xE0\xA4\x98\xE0\xA4\xB0\xE0\xA5\x87\xE0\xA4\xB2\xE0\xA5\x82\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xAF\xE0\xA4\xB5\xE0\xA4\xBF\xE0" ~
    "\xA4\x9A\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xB8\xE0\xA5\x82\xE0\xA4\x9A\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\xAE\xE0\xA5\x82\xE0\xA4\xB2\xE0\xA5\x8D\xE0\xA4\xAF\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\x96\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\xB9\xE0\xA4\xAE\xE0\xA5\x87\xE0\xA4" ~
    "\xB6\xE0\xA4\xBE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\x95\xE0\xA5\x82\xE0\xA4\xB2\xE0\xA4\xAE\xE0\xA5\x88\xE0\xA4\x82\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\xA4\xE0\xA5\x88\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\x9C\xE0\xA4\xBF\xE0\xA4\xB8\xE0\xA4\x95\xE0\xA5\x87" ~
    "rss+xml\" title=\"-type\" content=\"title\" content=\"at the same time" ~
    ".js\"></script>\x0A<\" method=\"post\" </span></a></li>vertical-align:t" ~
    "/jquery.min.js\">.click(function( style=\"padding-})();\x0A</script>\x0A" ~
    "</span><a href=\"<a href=\"http://); return false;text-decoration:" ~
    " scrolling=\"no\" border-collapse:associated with Bahasa Indonesia" ~
    "English language<text xml:space=.gif\" border=\"0\"</body>\x0A</html>\x0A" ~
    "overflow:hidden;img src=\"http://addEventListenerresponsible for " ~
    "s.js\"></script>\x0A/favicon.ico\" />operating system\" style=\"width:1" ~
    "target=\"_blank\">State Universitytext-align:left;\x0Adocument.write(" ~
    ", including the around the world);\x0D\x0A</script>\x0D\x0A<\" style=\"height:" ~
    ";overflow:hiddenmore informationan internationala member of the " ~
    "one of the firstcan be found in </div>\x0A\x09\x09</div>\x0Adisplay: none;\">" ~
    "\" />\x0A<link rel=\"\x0A  (function() {the 15th century.preventDefault(" ~
    "large number of Byzantine Empire.jpg|thumb|left|vast majority of" ~
    "majority of the  align=\"center\">University Pressdominated by the" ~
    "Second World Wardistribution of style=\"position:the rest of the " ~
    "characterized by rel=\"nofollow\">derives from therather than the " ~
    "a combination ofstyle=\"width:100English-speakingcomputer science" ~
    "border=\"0\" alt=\"the existence ofDemocratic Party\" style=\"margin-" ~
    "For this reason,.js\"></script>\x0A\x09sByTagName(s)[0]js\"></script>\x0D\x0A<" ~
    ".js\"></script>\x0D\x0Alink rel=\"icon\" ' alt='' class='formation of the" ~
    "versions of the </a></div></div>/page>\x0A  <page>\x0A<div class=\"cont" ~
    "became the firstbahasa Indonesiaenglish (simple)\xCE\x95\xCE\xBB\xCE\xBB\xCE\xB7\xCE\xBD\xCE\xB9\xCE\xBA\xCE\xAC" ~
    "\xD1\x85\xD1\x80\xD0\xB2\xD0\xB0\xD1\x82\xD1\x81\xD0\xBA\xD0\xB8\xD0\xBA\xD0\xBE\xD0\xBC\xD0\xBF\xD0\xB0\xD0\xBD\xD0\xB8\xD0\xB8\xD1\x8F\xD0\xB2\xD0\xBB\xD1\x8F\xD0\xB5\xD1\x82\xD1\x81\xD1\x8F\xD0\x94\xD0\xBE\xD0\xB1\xD0\xB0\xD0\xB2\xD0\xB8\xD1\x82\xD1\x8C" ~
    "\xD1\x87\xD0\xB5\xD0\xBB\xD0\xBE\xD0\xB2\xD0\xB5\xD0\xBA\xD0\xB0\xD1\x80\xD0\xB0\xD0\xB7\xD0\xB2\xD0\xB8\xD1\x82\xD0\xB8\xD1\x8F\xD0\x98\xD0\xBD\xD1\x82\xD0\xB5\xD1\x80\xD0\xBD\xD0\xB5\xD1\x82\xD0\x9E\xD1\x82\xD0\xB2\xD0\xB5\xD1\x82\xD0\xB8\xD1\x82\xD1\x8C" ~
    "\xD0\xBD\xD0\xB0\xD0\xBF\xD1\x80\xD0\xB8\xD0\xBC\xD0\xB5\xD1\x80\xD0\xB8\xD0\xBD\xD1\x82\xD0\xB5\xD1\x80\xD0\xBD\xD0\xB5\xD1\x82\xD0\xBA\xD0\xBE\xD1\x82\xD0\xBE\xD1\x80\xD0\xBE\xD0\xB3\xD0\xBE\xD1\x81\xD1\x82\xD1\x80\xD0\xB0\xD0\xBD\xD0\xB8\xD1\x86\xD1\x8B" ~
    "\xD0\xBA\xD0\xB0\xD1\x87\xD0\xB5\xD1\x81\xD1\x82\xD0\xB2\xD0\xB5\xD1\x83\xD1\x81\xD0\xBB\xD0\xBE\xD0\xB2\xD0\xB8\xD1\x8F\xD1\x85\xD0\xBF\xD1\x80\xD0\xBE\xD0\xB1\xD0\xBB\xD0\xB5\xD0\xBC\xD1\x8B\xD0\xBF\xD0\xBE\xD0\xBB\xD1\x83\xD1\x87\xD0\xB8\xD1\x82\xD1\x8C" ~
    "\xD1\x8F\xD0\xB2\xD0\xBB\xD1\x8F\xD1\x8E\xD1\x82\xD1\x81\xD1\x8F\xD0\xBD\xD0\xB0\xD0\xB8\xD0\xB1\xD0\xBE\xD0\xBB\xD0\xB5\xD0\xB5\xD0\xBA\xD0\xBE\xD0\xBC\xD0\xBF\xD0\xB0\xD0\xBD\xD0\xB8\xD1\x8F\xD0\xB2\xD0\xBD\xD0\xB8\xD0\xBC\xD0\xB0\xD0\xBD\xD0\xB8\xD0\xB5" ~
    "\xD1\x81\xD1\x80\xD0\xB5\xD0\xB4\xD1\x81\xD1\x82\xD0\xB2\xD0\xB0\xD8\xA7\xD9\x84\xD9\x85\xD9\x88\xD8\xA7\xD8\xB6\xD9\x8A\xD8\xB9\xD8\xA7\xD9\x84\xD8\xB1\xD8\xA6\xD9\x8A\xD8\xB3\xD9\x8A\xD8\xA9\xD8\xA7\xD9\x84\xD8\xA7\xD9\x86\xD8\xAA\xD9\x82\xD8\xA7\xD9\x84" ~
    "\xD9\x85\xD8\xB4\xD8\xA7\xD8\xB1\xD9\x83\xD8\xA7\xD8\xAA\xD9\x83\xD8\xA7\xD9\x84\xD8\xB3\xD9\x8A\xD8\xA7\xD8\xB1\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD9\x85\xD9\x83\xD8\xAA\xD9\x88\xD8\xA8\xD8\xA9\xD8\xA7\xD9\x84\xD8\xB3\xD8\xB9\xD9\x88\xD8\xAF\xD9\x8A\xD8\xA9" ~
    "\xD8\xA7\xD8\xAD\xD8\xB5\xD8\xA7\xD8\xA6\xD9\x8A\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD8\xB9\xD8\xA7\xD9\x84\xD9\x85\xD9\x8A\xD8\xA9\xD8\xA7\xD9\x84\xD8\xB5\xD9\x88\xD8\xAA\xD9\x8A\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD8\xA7\xD9\x86\xD8\xAA\xD8\xB1\xD9\x86\xD8\xAA" ~
    "\xD8\xA7\xD9\x84\xD8\xAA\xD8\xB5\xD8\xA7\xD9\x85\xD9\x8A\xD9\x85\xD8\xA7\xD9\x84\xD8\xA5\xD8\xB3\xD9\x84\xD8\xA7\xD9\x85\xD9\x8A\xD8\xA7\xD9\x84\xD9\x85\xD8\xB4\xD8\xA7\xD8\xB1\xD9\x83\xD8\xA9\xD8\xA7\xD9\x84\xD9\x85\xD8\xB1\xD8\xA6\xD9\x8A\xD8\xA7\xD8\xAA" ~
    "robots\" content=\"<div id=\"footer\">the United States<img src=\"htt" ~
    "p://.jpg|right|thumb|.js\"></script>\x0D\x0A<location.protocolframebord" ~
    "er=\"0\" s\" />\x0A<meta name=\"</a></div></div><font-weight:bold;&quot" ~
    "; and &quot;depending on the margin:0;padding:\" rel=\"nofollow\" P" ~
    "resident of the twentieth centuryevision>\x0A  </pageInternet Explo" ~
    "rera.async = true;\x0D\x0Ainformation about<div id=\"header\">\" action=\"" ~
    "http://<a href=\"https://<div id=\"content\"</div>\x0D\x0A</div>\x0D\x0A<derive" ~
    "d from the <img src='http://according to the \x0A</body>\x0A</html>\x0Ast" ~
    "yle=\"font-size:script language=\"Arial, Helvetica,</a><span class" ~
    "=\"</script><script political partiestd></tr></table><href=\"http:" ~
    "//www.interpretation ofrel=\"stylesheet\" document.write('<charset" ~
    "=\"utf-8\">\x0Abeginning of the revealed that thetelevision series\" r" ~
    "el=\"nofollow\"> target=\"_blank\">claiming that thehttp%3A%2F%2Fwww" ~
    ".manifestations ofPrime Minister ofinfluenced by theclass=\"clear" ~
    "fix\">/div>\x0D\x0A</div>\x0D\x0A\x0D\x0Athree-dimensionalChurch of Englandof North" ~
    " Carolinasquare kilometres.addEventListenerdistinct from thecomm" ~
    "only known asPhonetic Alphabetdeclared that thecontrolled by the" ~
    "Benjamin Franklinrole-playing gamethe University ofin Western Eu" ~
    "ropepersonal computerProject Gutenbergregardless of thehas been " ~
    "proposedtogether with the></li><li class=\"in some countriesmin.j" ~
    "s\"></script>of the populationofficial language<img src=\"images/i" ~
    "dentified by thenatural resourcesclassification ofcan be conside" ~
    "redquantum mechanicsNevertheless, themillion years ago</body>\x0D\x0A<" ~
    "/html>\x0D\xCE\x95\xCE\xBB\xCE\xBB\xCE\xB7\xCE\xBD\xCE\xB9\xCE\xBA\xCE\xAC\x0Atake advantage ofand, according toattrib" ~
    "uted to theMicrosoft Windowsthe first centuryunder the controldi" ~
    "v class=\"headershortly after thenotable exceptiontens of thousan" ~
    "dsseveral differentaround the world.reaching militaryisolated fr" ~
    "om theopposition to thethe Old TestamentAfrican Americansinserte" ~
    "d into theseparate from themetropolitan areamakes it possibleack" ~
    "nowledged thatarguably the mosttype=\"text/css\">\x0Athe Internationa" ~
    "lAccording to the pe=\"text/css\" />\x0Acoincide with thetwo-thirds o" ~
    "f theDuring this time,during the periodannounced that hethe inte" ~
    "rnationaland more recentlybelieved that theconsciousness andform" ~
    "erly known assurrounded by thefirst appeared inoccasionally used" ~
    "position:absolute;\" target=\"_blank\" position:relative;text-align" ~
    ":center;jax/libs/jquery/1.background-color:#type=\"application/an" ~
    "guage\" content=\"<meta http-equiv=\"Privacy Policy</a>e(\"%3Cscript" ~
    " src='\" target=\"_blank\">On the other hand,.jpg|thumb|right|2</di" ~
    "v><div class=\"<div style=\"float:nineteenth century</body>\x0D\x0A</htm" ~
    "l>\x0D\x0A<img src=\"http://s;text-align:centerfont-weight: bold; Accor" ~
    "ding to the difference between\" frameborder=\"0\" \" style=\"positio" ~
    "n:link href=\"http://html4/loose.dtd\">\x0Aduring this period</td></t" ~
    "r></table>closely related tofor the first time;font-weight:bold;" ~
    "input type=\"text\" <span style=\"font-onreadystatechange\x09<div clas" ~
    "s=\"cleardocument.location. For example, the a wide variety of <!" ~
    "DOCTYPE html>\x0D\x0A<&nbsp;&nbsp;&nbsp;\"><a href=\"http://style=\"float" ~
    ":left;concerned with the=http%3A%2F%2Fwww.in popular culturetype" ~
    "=\"text/css\" />it is possible to Harvard Universitytylesheet\" hre" ~
    "f=\"/the main characterOxford University  name=\"keywords\" cstyle=" ~
    "\"text-align:the United Kingdomfederal government<div style=\"marg" ~
    "in depending on the description of the<div class=\"header.min.js\"" ~
    "></script>destruction of theslightly differentin accordance with" ~
    "telecommunicationsindicates that theshortly thereafterespecially" ~
    " in the European countriesHowever, there aresrc=\"http://staticsu" ~
    "ggested that the\" src=\"http://www.a large number of Telecommunic" ~
    "ations\" rel=\"nofollow\" tHoly Roman Emperoralmost exclusively\" bo" ~
    "rder=\"0\" alt=\"Secretary of Stateculminating in theCIA World Fact" ~
    "bookthe most importantanniversary of thestyle=\"background-<li><e" ~
    "m><a href=\"/the Atlantic Oceanstrictly speaking,shortly before t" ~
    "hedifferent types ofthe Ottoman Empire><img src=\"http://An Intro" ~
    "duction toconsequence of thedeparture from theConfederate States" ~
    "indigenous peoplesProceedings of theinformation on thetheories h" ~
    "ave beeninvolvement in thedivided into threeadjacent countriesis" ~
    " responsible fordissolution of thecollaboration withwidely regar" ~
    "ded ashis contemporariesfounding member ofDominican Republicgene" ~
    "rally acceptedthe possibility ofare also availableunder construc" ~
    "tionrestoration of thethe general publicis almost entirelypasses" ~
    " through thehas been suggestedcomputer and videoGermanic languag" ~
    "es according to the different from theshortly afterwardshref=\"ht" ~
    "tps://www.recent developmentBoard of Directors<div class=\"search" ~
    "| <a href=\"http://In particular, theMultiple footnotesor other s" ~
    "ubstancethousands of yearstranslation of the</div>\x0D\x0A</div>\x0D\x0A\x0D\x0A<a" ~
    " href=\"index.phpwas established inmin.js\"></script>\x0Aparticipate " ~
    "in thea strong influencestyle=\"margin-top:represented by thegrad" ~
    "uated from theTraditionally, theElement(\"script\");However, since" ~
    " the/div>\x0A</div>\x0A<div left; margin-left:protection against0; ver" ~
    "tical-align:Unfortunately, thetype=\"image/x-icon/div>\x0A<div class" ~
    "=\" class=\"clearfix\"><div class=\"footer\x09\x09</div>\x0A\x09\x09</div>\x0Athe moti" ~
    "on picture\xD0\x91\xD1\x8A\xD0\xBB\xD0\xB3\xD0\xB0\xD1\x80\xD1\x81\xD0\xBA\xD0\xB8\xD0\xB1\xD1\x8A\xD0\xBB\xD0\xB3\xD0\xB0\xD1\x80\xD1\x81\xD0\xBA\xD0\xB8\xD0\xA4\xD0\xB5\xD0\xB4\xD0\xB5\xD1\x80\xD0\xB0\xD1\x86\xD0\xB8\xD0\xB8" ~
    "\xD0\xBD\xD0\xB5\xD1\x81\xD0\xBA\xD0\xBE\xD0\xBB\xD1\x8C\xD0\xBA\xD0\xBE\xD1\x81\xD0\xBE\xD0\xBE\xD0\xB1\xD1\x89\xD0\xB5\xD0\xBD\xD0\xB8\xD0\xB5\xD1\x81\xD0\xBE\xD0\xBE\xD0\xB1\xD1\x89\xD0\xB5\xD0\xBD\xD0\xB8\xD1\x8F\xD0\xBF\xD1\x80\xD0\xBE\xD0\xB3\xD1\x80" ~
    "\xD0\xB0\xD0\xBC\xD0\xBC\xD1\x8B\xD0\x9E\xD1\x82\xD0\xBF\xD1\x80\xD0\xB0\xD0\xB2\xD0\xB8\xD1\x82\xD1\x8C\xD0\xB1\xD0\xB5\xD1\x81\xD0\xBF\xD0\xBB\xD0\xB0\xD1\x82\xD0\xBD\xD0\xBE\xD0\xBC\xD0\xB0\xD1\x82\xD0\xB5\xD1\x80\xD0\xB8\xD0\xB0\xD0\xBB\xD1\x8B\xD0\xBF" ~
    "\xD0\xBE\xD0\xB7\xD0\xB2\xD0\xBE\xD0\xBB\xD1\x8F\xD0\xB5\xD1\x82\xD0\xBF\xD0\xBE\xD1\x81\xD0\xBB\xD0\xB5\xD0\xB4\xD0\xBD\xD0\xB8\xD0\xB5\xD1\x80\xD0\xB0\xD0\xB7\xD0\xBB\xD0\xB8\xD1\x87\xD0\xBD\xD1\x8B\xD1\x85\xD0\xBF\xD1\x80\xD0\xBE\xD0\xB4\xD1\x83\xD0\xBA" ~
    "\xD1\x86\xD0\xB8\xD0\xB8\xD0\xBF\xD1\x80\xD0\xBE\xD0\xB3\xD1\x80\xD0\xB0\xD0\xBC\xD0\xBC\xD0\xB0\xD0\xBF\xD0\xBE\xD0\xBB\xD0\xBD\xD0\xBE\xD1\x81\xD1\x82\xD1\x8C\xD1\x8E\xD0\xBD\xD0\xB0\xD1\x85\xD0\xBE\xD0\xB4\xD0\xB8\xD1\x82\xD1\x81\xD1\x8F\xD0\xB8\xD0\xB7" ~
    "\xD0\xB1\xD1\x80\xD0\xB0\xD0\xBD\xD0\xBD\xD0\xBE\xD0\xB5\xD0\xBD\xD0\xB0\xD1\x81\xD0\xB5\xD0\xBB\xD0\xB5\xD0\xBD\xD0\xB8\xD1\x8F\xD0\xB8\xD0\xB7\xD0\xBC\xD0\xB5\xD0\xBD\xD0\xB5\xD0\xBD\xD0\xB8\xD1\x8F\xD0\xBA\xD0\xB0\xD1\x82\xD0\xB5\xD0\xB3\xD0\xBE\xD1\x80" ~
    "\xD0\xB8\xD0\xB8\xD0\x90\xD0\xBB\xD0\xB5\xD0\xBA\xD1\x81\xD0\xB0\xD0\xBD\xD0\xB4\xD1\x80\xE0\xA4\xA6\xE0\xA5\x8D\xE0\xA4\xB5\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\xAE\xE0\xA5\x88\xE0\xA4\xA8\xE0\xA5\x81\xE0\xA4\x85\xE0\xA4\xB2\xE0\xA4\xAA\xE0\xA5\x8D" ~
    "\xE0\xA4\xB0\xE0\xA4\xA6\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA4\xAD\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xA4\xE0\xA5\x80\xE0\xA4\xAF\xE0\xA4\x85\xE0\xA4\xA8\xE0\xA5\x81\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\xB6\xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA5\x8D\xE0\xA4\xA6\xE0" ~
    "\xA5\x80\xE0\xA4\x87\xE0\xA4\x82\xE0\xA4\xA1\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xA6\xE0\xA4\xBF\xE0\xA4\xB2\xE0\xA5\x8D\xE0\xA4\xB2\xE0\xA5\x80\xE0\xA4\x85\xE0\xA4\xA7\xE0\xA4\xBF\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xB5\xE0\xA5\x80\xE0\xA4" ~
    "\xA1\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA5\x8B\xE0\xA4\x9A\xE0\xA4\xBF\xE0\xA4\x9F\xE0\xA5\x8D\xE0\xA4\xA0\xE0\xA5\x87\xE0\xA4\xB8\xE0\xA4\xAE\xE0\xA4\xBE\xE0\xA4\x9A\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\x9C\xE0\xA4\x82\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB6\xE0\xA4\xA8" ~
    "\xE0\xA4\xA6\xE0\xA5\x81\xE0\xA4\xA8\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xAA\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xAF\xE0\xA5\x8B\xE0\xA4\x97\xE0\xA4\x85\xE0\xA4\xA8\xE0\xA5\x81\xE0\xA4\xB8\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\x91\xE0\xA4\xA8\xE0\xA4\xB2\xE0" ~
    "\xA4\xBE\xE0\xA4\x87\xE0\xA4\xA8\xE0\xA4\xAA\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\x9F\xE0\xA5\x80\xE0\xA4\xB6\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x8B\xE0\xA4\x82\xE0\xA4\xB2\xE0\xA5\x8B\xE0\xA4\x95\xE0\xA4\xB8\xE0\xA4\xAD\xE0\xA4\xBE\xE0\xA4" ~
    "\xAB\xE0\xA4\xBC\xE0\xA5\x8D\xE0\xA4\xB2\xE0\xA5\x88\xE0\xA4\xB6\xE0\xA4\xB6\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\xAA\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\xB6\xE0\xA4\xAA\xE0\xA5\x8D\xE0\xA4\xB2\xE0\xA5\x87" ~
    "\xE0\xA4\xAF\xE0\xA4\xB0\xE0\xA4\x95\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\xA6\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA5\xE0\xA4\xBF\xE0\xA4\xA4\xE0\xA4\xBF\xE0\xA4\x89\xE0\xA4\xA4\xE0\xA5\x8D\xE0\xA4\xAA\xE0\xA4\xBE\xE0\xA4\xA6\xE0\xA4\x89\xE0" ~
    "\xA4\xA8\xE0\xA5\x8D\xE0\xA4\xB9\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\x9A\xE0\xA4\xBF\xE0\xA4\x9F\xE0\xA5\x8D\xE0\xA4\xA0\xE0\xA4\xBE\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\xA4\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\x9C\xE0\xA5\x8D\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4" ~
    "\xA6\xE0\xA4\xBE\xE0\xA4\xAA\xE0\xA5\x81\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\x9C\xE0\xA5\x8B\xE0\xA4\xA1\xE0\xA4\xBC\xE0\xA5\x87\xE0\xA4\x82\xE0\xA4\x85\xE0\xA4\xA8\xE0\xA5\x81\xE0\xA4\xB5\xE0\xA4\xBE\xE0\xA4\xA6\xE0\xA4\xB6\xE0\xA5\x8D" ~
    "\xE0\xA4\xB0\xE0\xA5\x87\xE0\xA4\xA3\xE0\xA5\x80\xE0\xA4\xB6\xE0\xA4\xBF\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7\xE0\xA4\xBE\xE0\xA4\xB8\xE0\xA4\xB0\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA5\x80\xE0\xA4\xB8\xE0\xA4\x82\xE0\xA4\x97\xE0\xA5\x8D\xE0\xA4\xB0\xE0" ~
    "\xA4\xB9\xE0\xA4\xAA\xE0\xA4\xB0\xE0\xA4\xBF\xE0\xA4\xA3\xE0\xA4\xBE\xE0\xA4\xAE\xE0\xA4\xAC\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\x82\xE0\xA4\xA1\xE0\xA4\xAC\xE0\xA4\x9A\xE0\xA5\x8D\xE0\xA4\x9A\xE0\xA5\x8B\xE0\xA4\x82\xE0\xA4\x89\xE0\xA4\xAA\xE0\xA4" ~
    "\xB2\xE0\xA4\xAC\xE0\xA5\x8D\xE0\xA4\xA7\xE0\xA4\xAE\xE0\xA4\x82\xE0\xA4\xA4\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA5\x80\xE0\xA4\xB8\xE0\xA4\x82\xE0\xA4\xAA\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\x95\xE0\xA4\x89\xE0\xA4\xAE\xE0\xA5\x8D\xE0\xA4\xAE\xE0\xA5\x80\xE0\xA4\xA6" ~
    "\xE0\xA4\xAE\xE0\xA4\xBE\xE0\xA4\xA7\xE0\xA5\x8D\xE0\xA4\xAF\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA4\xB9\xE0\xA4\xBE\xE0\xA4\xAF\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\xB6\xE0\xA4\xAC\xE0\xA5\x8D\xE0\xA4\xA6\xE0\xA5\x8B\xE0\xA4\x82\xE0\xA4\xAE\xE0\xA5\x80\xE0\xA4\xA1\xE0" ~
    "\xA4\xBF\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4\x86\xE0\xA4\x88\xE0\xA4\xAA\xE0\xA5\x80\xE0\xA4\x8F\xE0\xA4\xB2\xE0\xA4\xAE\xE0\xA5\x8B\xE0\xA4\xAC\xE0\xA4\xBE\xE0\xA4\x87\xE0\xA4\xB2\xE0\xA4\xB8\xE0\xA4\x82\xE0\xA4\x96\xE0\xA5\x8D\xE0\xA4\xAF\xE0\xA4\xBE\xE0\xA4" ~
    "\x86\xE0\xA4\xAA\xE0\xA4\xB0\xE0\xA5\x87\xE0\xA4\xB6\xE0\xA4\xA8\xE0\xA4\x85\xE0\xA4\xA8\xE0\xA5\x81\xE0\xA4\xAC\xE0\xA4\x82\xE0\xA4\xA7\xE0\xA4\xAC\xE0\xA4\xBE\xE0\xA4\x9C\xE0\xA4\xBC\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xA8\xE0\xA4\xB5\xE0\xA5\x80\xE0\xA4\xA8" ~
    "\xE0\xA4\xA4\xE0\xA4\xAE\xE0\xA4\xAA\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xAE\xE0\xA5\x81\xE0\xA4\x96\xE0\xA4\xAA\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xB6\xE0\xA5\x8D\xE0\xA4\xA8\xE0\xA4\xAA\xE0\xA4\xB0\xE0\xA4\xBF\xE0\xA4\xB5\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xA8\xE0" ~
    "\xA5\x81\xE0\xA4\x95\xE0\xA4\xB8\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA4\xB8\xE0\xA4\xAE\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xA5\xE0\xA4\xA8\xE0\xA4\x86\xE0\xA4\xAF\xE0\xA5\x8B\xE0\xA4\x9C\xE0\xA4\xBF\xE0\xA4\xA4\xE0\xA4\xB8\xE0\xA5\x8B\xE0\xA4\xAE\xE0\xA4\xB5\xE0\xA4" ~
    "\xBE\xE0\xA4\xB0\xD8\xA7\xD9\x84\xD9\x85\xD8\xB4\xD8\xA7\xD8\xB1\xD9\x83\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD9\x85\xD9\x86\xD8\xAA\xD8\xAF\xD9\x8A\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD9\x83\xD9\x85\xD8\xA8\xD9\x8A\xD9\x88\xD8\xAA\xD8\xB1\xD8\xA7\xD9\x84\xD9\x85" ~
    "\xD8\xB4\xD8\xA7\xD9\x87\xD8\xAF\xD8\xA7\xD8\xAA\xD8\xB9\xD8\xAF\xD8\xAF\xD8\xA7\xD9\x84\xD8\xB2\xD9\x88\xD8\xA7\xD8\xB1\xD8\xB9\xD8\xAF\xD8\xAF\xD8\xA7\xD9\x84\xD8\xB1\xD8\xAF\xD9\x88\xD8\xAF\xD8\xA7\xD9\x84\xD8\xA5\xD8\xB3\xD9\x84\xD8\xA7\xD9\x85\xD9\x8A" ~
    "\xD8\xA9\xD8\xA7\xD9\x84\xD9\x81\xD9\x88\xD8\xAA\xD9\x88\xD8\xB4\xD9\x88\xD8\xA8\xD8\xA7\xD9\x84\xD9\x85\xD8\xB3\xD8\xA7\xD8\xA8\xD9\x82\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD9\x85\xD8\xB9\xD9\x84\xD9\x88\xD9\x85\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD9\x85\xD8\xB3" ~
    "\xD9\x84\xD8\xB3\xD9\x84\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD8\xAC\xD8\xB1\xD8\xA7\xD9\x81\xD9\x8A\xD9\x83\xD8\xB3\xD8\xA7\xD9\x84\xD8\xA7\xD8\xB3\xD9\x84\xD8\xA7\xD9\x85\xD9\x8A\xD8\xA9\xD8\xA7\xD9\x84\xD8\xA7\xD8\xAA\xD8\xB5\xD8\xA7\xD9\x84\xD8\xA7\xD8\xAA" ~
    "keywords\" content=\"w3.org/1999/xhtml\"><a target=\"_blank\" text/ht" ~
    "ml; charset=\" target=\"_blank\"><table cellpadding=\"autocomplete=\"" ~
    "off\" text-align: center;to last version by background-color: #\" " ~
    "href=\"http://www./div></div><div id=<a href=\"#\" class=\"\"><img sr" ~
    "c=\"http://cript\" src=\"http://\x0A<script language=\"//EN\" \"http://ww" ~
    "w.wencodeURIComponent(\" href=\"javascript:<div class=\"contentdocu" ~
    "ment.write('<scposition: absolute;script src=\"http:// style=\"mar" ~
    "gin-top:.min.js\"></script>\x0A</div>\x0A<div class=\"w3.org/1999/xhtml\"" ~
    " \x0A\x0D\x0A</body>\x0D\x0A</html>distinction between/\" target=\"_blank\"><link " ~
    "href=\"http://encoding=\"utf-8\"?>\x0Aw.addEventListener?action=\"http:" ~
    "//www.icon\" href=\"http:// style=\"background:type=\"text/css\" />\x0Am" ~
    "eta property=\"og:t<input type=\"text\"  style=\"text-align:the deve" ~
    "lopment of tylesheet\" type=\"tehtml; charset=utf-8is considered t" ~
    "o betable width=\"100%\" In addition to the contributed to the dif" ~
    "ferences betweendevelopment of the It is important to </script>\x0A" ~
    "\x0A<script  style=\"font-size:1></span><span id=gbLibrary of Congre" ~
    "ss<img src=\"http://imEnglish translationAcademy of Sciencesdiv s" ~
    "tyle=\"display:construction of the.getElementById(id)in conjuncti" ~
    "on withElement('script'); <meta property=\"og:\xD0\x91\xD1\x8A\xD0\xBB\xD0\xB3\xD0\xB0\xD1\x80\xD1\x81\xD0\xBA\xD0\xB8\x0A" ~
    " type=\"text\" name=\">Privacy Policy</a>administered by theenableS" ~
    "ingleRequeststyle=&quot;margin:</div></div></div><><img src=\"htt" ~
    "p://i style=&quot;float:referred to as the total population ofin" ~
    " Washington, D.C. style=\"background-among other things,organizat" ~
    "ion of theparticipated in thethe introduction ofidentified with " ~
    "thefictional character Oxford University misunderstanding ofTher" ~
    "e are, however,stylesheet\" href=\"/Columbia Universityexpanded to" ~
    " includeusually referred toindicating that thehave suggested tha" ~
    "taffiliated with thecorrelation betweennumber of different></td>" ~
    "</tr></table>Republic of Ireland\x0A</script>\x0A<script under the inf" ~
    "luencecontribution to theOfficial website ofheadquarters of thec" ~
    "entered around theimplications of thehave been developedFederal " ~
    "Republic ofbecame increasinglycontinuation of theNote, however, " ~
    "thatsimilar to that of capabilities of theaccordance with thepar" ~
    "ticipants in thefurther developmentunder the directionis often c" ~
    "onsideredhis younger brother</td></tr></table><a http-equiv=\"X-U" ~
    "A-physical propertiesof British Columbiahas been criticized(with" ~
    " the exceptionquestions about thepassing through the0\" cellpaddi" ~
    "ng=\"0\" thousands of peopleredirects here. Forhave children under" ~
    "%3E%3C/script%3E\"));<a href=\"http://www.<li><a href=\"http://site" ~
    "_name\" content=\"text-decoration:nonestyle=\"display: none<meta ht" ~
    "tp-equiv=\"X-new Date().getTime() type=\"image/x-icon\"</span><span" ~
    " class=\"language=\"javascriptwindow.location.href<a href=\"javascr" ~
    "ipt:-->\x0D\x0A<script type=\"t<a href='http://www.hortcut icon\" href=\"" ~
    "</div>\x0D\x0A<div class=\"<script src=\"http://\" rel=\"stylesheet\" t</di" ~
    "v>\x0A<script type=/a> <a href=\"http:// allowTransparency=\"X-UA-Com" ~
    "patible\" conrelationship between\x0A</script>\x0D\x0A<script </a></li></u" ~
    "l></div>associated with the programming language</a><a href=\"htt" ~
    "p://</a></li><li class=\"form action=\"http://<div style=\"display:" ~
    "type=\"text\" name=\"q\"<table width=\"100%\" background-position:\" bo" ~
    "rder=\"0\" width=\"rel=\"shortcut icon\" h6><ul><li><a href=\"  <meta " ~
    "http-equiv=\"css\" media=\"screen\" responsible for the \" type=\"appl" ~
    "ication/\" style=\"background-html; charset=utf-8\" allowtransparen" ~
    "cy=\"stylesheet\" type=\"te\x0D\x0A<meta http-equiv=\"></span><span class=" ~
    "\"0\" cellspacing=\"0\">;\x0A</script>\x0A<script sometimes called thedoes" ~
    " not necessarilyFor more informationat the beginning of <!DOCTYP" ~
    "E html><htmlparticularly in the type=\"hidden\" name=\"javascript:v" ~
    "oid(0);\"effectiveness of the autocomplete=\"off\" generally consid" ~
    "ered><input type=\"text\" \"></script>\x0D\x0A<scriptthroughout the world" ~
    "common misconceptionassociation with the</div>\x0A</div>\x0A<div cduri" ~
    "ng his lifetime,corresponding to thetype=\"image/x-icon\" an incre" ~
    "asing numberdiplomatic relationsare often consideredmeta charset" ~
    "=\"utf-8\" <input type=\"text\" examples include the\"><img src=\"http" ~
    "://iparticipation in thethe establishment of\x0A</div>\x0A<div class=\"" ~
    "&amp;nbsp;&amp;nbsp;to determine whetherquite different frommark" ~
    "ed the beginningdistance between thecontributions to theconflict" ~
    " between thewidely considered towas one of the firstwith varying" ~
    " degreeshave speculated that(document.getElementparticipating in" ~
    " theoriginally developedeta charset=\"utf-8\"> type=\"text/css\" />\x0A" ~
    "interchangeably withmore closely relatedsocial and politicalthat" ~
    " would otherwiseperpendicular to thestyle type=\"text/csstype=\"su" ~
    "bmit\" name=\"families residing indeveloping countriescomputer pro" ~
    "grammingeconomic developmentdetermination of thefor more informa" ~
    "tionon several occasionsportugu\xC3\xAAs (Europeu)\xD0\xA3\xD0\xBA\xD1\x80\xD0\xB0\xD1\x97\xD0\xBD\xD1\x81\xD1\x8C\xD0\xBA\xD0\xB0" ~
    "\xD1\x83\xD0\xBA\xD1\x80\xD0\xB0\xD1\x97\xD0\xBD\xD1\x81\xD1\x8C\xD0\xBA\xD0\xB0\xD0\xA0\xD0\xBE\xD1\x81\xD1\x81\xD0\xB8\xD0\xB9\xD1\x81\xD0\xBA\xD0\xBE\xD0\xB9\xD0\xBC\xD0\xB0\xD1\x82\xD0\xB5\xD1\x80\xD0\xB8\xD0\xB0\xD0\xBB\xD0\xBE\xD0\xB2\xD0\xB8\xD0\xBD" ~
    "\xD1\x84\xD0\xBE\xD1\x80\xD0\xBC\xD0\xB0\xD1\x86\xD0\xB8\xD0\xB8\xD1\x83\xD0\xBF\xD1\x80\xD0\xB0\xD0\xB2\xD0\xBB\xD0\xB5\xD0\xBD\xD0\xB8\xD1\x8F\xD0\xBD\xD0\xB5\xD0\xBE\xD0\xB1\xD1\x85\xD0\xBE\xD0\xB4\xD0\xB8\xD0\xBC\xD0\xBE\xD0\xB8\xD0\xBD\xD1\x84\xD0\xBE" ~
    "\xD1\x80\xD0\xBC\xD0\xB0\xD1\x86\xD0\xB8\xD1\x8F\xD0\x98\xD0\xBD\xD1\x84\xD0\xBE\xD1\x80\xD0\xBC\xD0\xB0\xD1\x86\xD0\xB8\xD1\x8F\xD0\xA0\xD0\xB5\xD1\x81\xD0\xBF\xD1\x83\xD0\xB1\xD0\xBB\xD0\xB8\xD0\xBA\xD0\xB8\xD0\xBA\xD0\xBE\xD0\xBB\xD0\xB8\xD1\x87\xD0\xB5" ~
    "\xD1\x81\xD1\x82\xD0\xB2\xD0\xBE\xD0\xB8\xD0\xBD\xD1\x84\xD0\xBE\xD1\x80\xD0\xBC\xD0\xB0\xD1\x86\xD0\xB8\xD1\x8E\xD1\x82\xD0\xB5\xD1\x80\xD1\x80\xD0\xB8\xD1\x82\xD0\xBE\xD1\x80\xD0\xB8\xD0\xB8\xD0\xB4\xD0\xBE\xD1\x81\xD1\x82\xD0\xB0\xD1\x82\xD0\xBE\xD1\x87" ~
    "\xD0\xBD\xD0\xBE\xD8\xA7\xD9\x84\xD9\x85\xD8\xAA\xD9\x88\xD8\xA7\xD8\xAC\xD8\xAF\xD9\x88\xD9\x86\xD8\xA7\xD9\x84\xD8\xA7\xD8\xB4\xD8\xAA\xD8\xB1\xD8\xA7\xD9\x83\xD8\xA7\xD8\xAA\xD8\xA7\xD9\x84\xD8\xA7\xD9\x82\xD8\xAA\xD8\xB1\xD8\xA7\xD8\xAD\xD8\xA7\xD8\xAA" ~
    "html; charset=UTF-8\" setTimeout(function()display:inline-block;<" ~
    "input type=\"submit\" type = 'text/javascri<img src=\"http://www.\" " ~
    "\"http://www.w3.org/shortcut icon\" href=\"\" autocomplete=\"off\" </a" ~
    "></div><div class=</a></li>\x0A<li class=\"css\" type=\"text/css\" <for" ~
    "m action=\"http://xt/css\" href=\"http://link rel=\"alternate\" \x0D\x0A<sc" ~
    "ript type=\"text/ onclick=\"javascript:(new Date).getTime()}height" ~
    "=\"1\" width=\"1\" People's Republic of  <a href=\"http://www.text-de" ~
    "coration:underthe beginning of the </div>\x0A</div>\x0A</div>\x0Aestablis" ~
    "hment of the </div></div></div></d#viewport{min-height:\x0A<script " ~
    "src=\"http://option><option value=often referred to as /option>\x0A<" ~
    "option valu<!DOCTYPE html>\x0A<!--[International Airport>\x0A<a href=\"" ~
    "http://www</a><a href=\"http://w\xE0\xB8\xA0\xE0\xB8\xB2\xE0\xB8\xA9\xE0\xB8\xB2\xE0\xB9\x84\xE0\xB8\x97\xE0\xB8\xA2\xE1\x83\xA5\xE1\x83\x90\xE1\x83\xA0\xE1\x83\x97" ~
    "\xE1\x83\xA3\xE1\x83\x9A\xE1\x83\x98\xE6\xAD\xA3\xE9\xAB\x94\xE4\xB8\xAD\xE6\x96\x87 (\xE7\xB9\x81\xE9\xAB\x94)\xE0\xA4\xA8\xE0\xA4\xBF\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xA6\xE0\xA5\x87\xE0\xA4\xB6\xE0\xA4\xA1\xE0\xA4\xBE\xE0\xA4\x89\xE0\xA4\xA8\xE0" ~
    "\xA4\xB2\xE0\xA5\x8B\xE0\xA4\xA1\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7\xE0\xA5\x87\xE0\xA4\xA4\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\x9C\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA5\x80\xE0\xA4\xB8\xE0\xA4\x82\xE0\xA4\xAC\xE0\xA4\x82\xE0\xA4" ~
    "\xA7\xE0\xA4\xBF\xE0\xA4\xA4\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA5\xE0\xA4\xBE\xE0\xA4\xAA\xE0\xA4\xA8\xE0\xA4\xBE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xB5\xE0\xA5\x80\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xB8\xE0\xA4\x82\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\x95" ~
    "\xE0\xA4\xB0\xE0\xA4\xA3\xE0\xA4\xB8\xE0\xA4\xBE\xE0\xA4\xAE\xE0\xA4\x97\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA5\x80\xE0\xA4\x9A\xE0\xA4\xBF\xE0\xA4\x9F\xE0\xA5\x8D\xE0\xA4\xA0\xE0\xA5\x8B\xE0\xA4\x82\xE0\xA4\xB5\xE0\xA4\xBF\xE0\xA4\x9C\xE0\xA5\x8D\xE0\xA4\x9E\xE0" ~
    "\xA4\xBE\xE0\xA4\xA8\xE0\xA4\x85\xE0\xA4\xAE\xE0\xA5\x87\xE0\xA4\xB0\xE0\xA4\xBF\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB5\xE0\xA4\xBF\xE0\xA4\xAD\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA5\x8D\xE0\xA4\xA8\xE0\xA4\x97\xE0\xA4\xBE\xE0\xA4\xA1\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA4" ~
    "\xBE\xE0\xA4\x81\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xAF\xE0\xA5\x8B\xE0\xA4\x82\xE0\xA4\x95\xE0\xA4\xBF\xE0\xA4\xB8\xE0\xA5\x81\xE0\xA4\xB0\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7\xE0\xA4\xBE\xE0\xA4\xAA\xE0\xA4\xB9\xE0\xA5\x81\xE0\xA4\x81\xE0\xA4\x9A\xE0\xA4\xA4" ~
    "\xE0\xA5\x80\xE0\xA4\xAA\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xAC\xE0\xA4\x82\xE0\xA4\xA7\xE0\xA4\xA8\xE0\xA4\x9F\xE0\xA4\xBF\xE0\xA4\xAA\xE0\xA5\x8D\xE0\xA4\xAA\xE0\xA4\xA3\xE0\xA5\x80\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xBF\xE0\xA4\x95\xE0\xA5\x87\xE0" ~
    "\xA4\x9F\xE0\xA4\xAA\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\x82\xE0\xA4\xAD\xE0\xA4\xAA\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\xAA\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA4\xAE\xE0\xA4\xBE\xE0\xA4\xB2\xE0\xA4\xBF\xE0\xA4\x95\xE0\xA5\x8B\xE0\xA4" ~
    "\x82\xE0\xA4\xB0\xE0\xA4\xAB\xE0\xA4\xBC\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA4\xA8\xE0\xA4\xBF\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xAE\xE0\xA4\xBE\xE0\xA4\xA3\xE0\xA4\xB2\xE0\xA4\xBF\xE0\xA4\xAE\xE0\xA4\xBF\xE0\xA4\x9F\xE0\xA5\x87\xE0\xA4\xA1" ~
    "description\" content=\"document.location.prot.getElementsByTagNam" ~
    "e(<!DOCTYPE html>\x0A<html <meta charset=\"utf-8\">:url\" content=\"htt" ~
    "p://.css\" rel=\"stylesheet\"style type=\"text/css\">type=\"text/css\" " ~
    "href=\"w3.org/1999/xhtml\" xmltype=\"text/javascript\" method=\"get\" " ~
    "action=\"link rel=\"stylesheet\"  = document.getElementtype=\"image/" ~
    "x-icon\" />cellpadding=\"0\" cellsp.css\" type=\"text/css\" </a></li><" ~
    "li><a href=\"\" width=\"1\" height=\"1\"\"><a href=\"http://www.style=\"d" ~
    "isplay:none;\">alternate\" type=\"appli-//W3C//DTD XHTML 1.0 ellspa" ~
    "cing=\"0\" cellpad type=\"hidden\" value=\"/a>&nbsp;<span role=\"s\x0A<in" ~
    "put type=\"hidden\" language=\"JavaScript\"  document.getElementsBg=" ~
    "\"0\" cellspacing=\"0\" ype=\"text/css\" media=\"type='text/javascript'" ~
    "with the exception of ype=\"text/css\" rel=\"st height=\"1\" width=\"1" ~
    "\" ='+encodeURIComponent(<link rel=\"alternate\" \x0Abody, tr, input, " ~
    "textmeta name=\"robots\" conmethod=\"post\" action=\">\x0A<a href=\"http:" ~
    "//www.css\" rel=\"stylesheet\" </div></div><div classlanguage=\"java" ~
    "script\">aria-hidden=\"true\">\xC2\xB7<ript\" type=\"text/javasl=0;})();\x0A(f" ~
    "unction(){background-image: url(/a></li><li><a href=\"h\x09\x09<li><a h" ~
    "ref=\"http://ator\" aria-hidden=\"tru> <a href=\"http://www.language" ~
    "=\"javascript\" /option>\x0A<option value/div></div><div class=rator\"" ~
    " aria-hidden=\"tre=(new Date).getTime()portugu\xC3\xAAs (do Brasil)\xD0\xBE\xD1\x80" ~
    "\xD0\xB3\xD0\xB0\xD0\xBD\xD0\xB8\xD0\xB7\xD0\xB0\xD1\x86\xD0\xB8\xD0\xB8\xD0\xB2\xD0\xBE\xD0\xB7\xD0\xBC\xD0\xBE\xD0\xB6\xD0\xBD\xD0\xBE\xD1\x81\xD1\x82\xD1\x8C\xD0\xBE\xD0\xB1\xD1\x80\xD0\xB0\xD0\xB7\xD0\xBE\xD0\xB2\xD0\xB0\xD0\xBD\xD0\xB8\xD1\x8F\xD1\x80" ~
    "\xD0\xB5\xD0\xB3\xD0\xB8\xD1\x81\xD1\x82\xD1\x80\xD0\xB0\xD1\x86\xD0\xB8\xD0\xB8\xD0\xB2\xD0\xBE\xD0\xB7\xD0\xBC\xD0\xBE\xD0\xB6\xD0\xBD\xD0\xBE\xD1\x81\xD1\x82\xD0\xB8\xD0\xBE\xD0\xB1\xD1\x8F\xD0\xB7\xD0\xB0\xD1\x82\xD0\xB5\xD0\xBB\xD1\x8C\xD0\xBD\xD0\xB0" ~
    "<!DOCTYPE html PUBLIC \"nt-Type\" content=\"text/<meta http-equiv=\"" ~
    "Conteransitional//EN\" \"http:<html xmlns=\"http://www-//W3C//DTD X" ~
    "HTML 1.0 TDTD/xhtml1-transitional//www.w3.org/TR/xhtml1/pe = 'te" ~
    "xt/javascript';<meta name=\"descriptionparentNode.insertBefore<in" ~
    "put type=\"hidden\" najs\" type=\"text/javascri(document).ready(func" ~
    "tiscript type=\"text/javasimage\" content=\"http://UA-Compatible\" c" ~
    "ontent=tml; charset=utf-8\" />\x0Alink rel=\"shortcut icon<link rel=\"" ~
    "stylesheet\" </script>\x0A<script type== document.createElemen<a tar" ~
    "get=\"_blank\" href= document.getElementsBinput type=\"text\" name=a" ~
    ".type = 'text/javascrinput type=\"hidden\" namehtml; charset=utf-8" ~
    "\" />dtd\">\x0A<html xmlns=\"http-//W3C//DTD HTML 4.01 TentsByTagName(" ~
    "'script')input type=\"hidden\" nam<script type=\"text/javas\" style=" ~
    "\"display:none;\">document.getElementById(=document.createElement(" ~
    "' type='text/javascript'input type=\"text\" name=\"d.getElementsByT" ~
    "agName(snical\" href=\"http://www.C//DTD HTML 4.01 Transit<style t" ~
    "ype=\"text/css\">\x0A\x0A<style type=\"text/css\">ional.dtd\">\x0A<html xmlns=" ~
    "http-equiv=\"Content-Typeding=\"0\" cellspacing=\"0\"html; charset=ut" ~
    "f-8\" />\x0A style=\"display:none;\"><<li><a href=\"http://www. type='t" ~
    "ext/javascript'>\xD0\xB4\xD0\xB5\xD1\x8F\xD1\x82\xD0\xB5\xD0\xBB\xD1\x8C\xD0\xBD\xD0\xBE\xD1\x81\xD1\x82\xD0\xB8\xD1\x81\xD0\xBE\xD0\xBE\xD1\x82\xD0\xB2\xD0\xB5\xD1\x82\xD1\x81\xD1\x82\xD0\xB2\xD0\xB8\xD0\xB8" ~
    "\xD0\xBF\xD1\x80\xD0\xBE\xD0\xB8\xD0\xB7\xD0\xB2\xD0\xBE\xD0\xB4\xD1\x81\xD1\x82\xD0\xB2\xD0\xB0\xD0\xB1\xD0\xB5\xD0\xB7\xD0\xBE\xD0\xBF\xD0\xB0\xD1\x81\xD0\xBD\xD0\xBE\xD1\x81\xD1\x82\xD0\xB8\xE0\xA4\xAA\xE0\xA5\x81\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0" ~
    "\xA4\xBF\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\x82\xE0\xA4\x97\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA5\x87\xE0\xA4\xB8\xE0\xA4\x89\xE0\xA4\xA8\xE0\xA5\x8D\xE0\xA4\xB9\xE0\xA5\x8B\xE0\xA4\x82\xE0\xA4\xA8\xE0\xA5\x87\xE0\xA4\xB5\xE0\xA4\xBF\xE0\xA4" ~
    "\xA7\xE0\xA4\xBE\xE0\xA4\xA8\xE0\xA4\xB8\xE0\xA4\xAD\xE0\xA4\xBE\xE0\xA4\xAB\xE0\xA4\xBF\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB8\xE0\xA4\xBF\xE0\xA4\x82\xE0\xA4\x97\xE0\xA4\xB8\xE0\xA5\x81\xE0\xA4\xB0\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7\xE0\xA4\xBF\xE0\xA4\xA4" ~
    "\xE0\xA4\x95\xE0\xA5\x89\xE0\xA4\xAA\xE0\xA5\x80\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\x87\xE0\xA4\x9F\xE0\xA4\xB5\xE0\xA4\xBF\xE0\xA4\x9C\xE0\xA5\x8D\xE0\xA4\x9E\xE0\xA4\xBE\xE0\xA4\xAA\xE0\xA4\xA8\xE0\xA4\x95\xE0\xA4\xBE\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xB0\xE0" ~
    "\xA4\xB5\xE0\xA4\xBE\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA4\xA4\xE0\xA4\xBE";

/*
Copyright (c) 2009, 2010, 2013-2016 by the Brotli Authors.
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
