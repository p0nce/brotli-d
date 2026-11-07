/**
    Recompress a WOFF2 font into a smaller one.

    Usage: woff2-optimize [--crunch] <input.woff2> <output.woff2>

    A WOFF2 file is a header, a table directory (plus a collection directory
    for .ttc fonts), one Brotli stream holding all the font tables, then an
    optional Brotli-compressed metadata block and an optional private block.
    This program decodes both Brotli streams and re-encodes them with
    BROTLI_MAX_LEVEL and the default window size (with --crunch: every window
    size), keeping the smallest result (the original stream if nothing beats
    it). The directories and the
    decoded font data are untouched, so the font is bit-identical once
    decoded; only the header's sizes and offsets are rewritten.

    See https://www.w3.org/TR/WOFF2/
*/
module app;

import std.file;
import std.stdio;

import brotli;

enum uint WOFF2_SIGNATURE = 0x774F4632; // 'wOF2'
enum uint TTCF_FLAVOR     = 0x74746366; // 'ttcf'
enum WOFF2_HEADER_SIZE    = 48;

int main(string[] args)
{
    bool crunch = false;
    string[] files;
    foreach (arg; args[1 .. $])
    {
        if (arg == "--crunch")
            crunch = true;
        else
            files ~= arg;
    }

    if (files.length != 2)
    {
        stderr.writefln("Usage: woff2-optimize [--crunch] <input.woff2> <output.woff2>");
        stderr.writefln("  --crunch  try every window size instead of the default one only");
        return 1;
    }

    const(ubyte)[] input = cast(const(ubyte)[]) read(files[0]);
    try
    {
        ubyte[] output = optimize(input, crunch);
        std.file.write(files[1], output);
        double ratio = 100.0 * output.length / input.length;
        writefln("%s: %d -> %d bytes (%.2f%%)", files[0], input.length, output.length, ratio);
        return 0;
    }
    catch (Exception e)
    {
        stderr.writefln("woff2-optimize: %s", e.msg);
        return 1;
    }
}

ubyte[] optimize(const(ubyte)[] f, bool crunch)
{
    if (f.length < WOFF2_HEADER_SIZE || be32(f, 0) != WOFF2_SIGNATURE)
        throw new Exception("not a WOFF2 file");

    uint flavor          = be32(f, 4);
    uint numTables       = be16(f, 12);
    uint compressedSize  = be32(f, 20);
    uint metaOffset      = be32(f, 28);
    uint metaLength      = be32(f, 32);
    uint metaOrigLength  = be32(f, 36);
    uint privOffset      = be32(f, 40);
    uint privLength      = be32(f, 44);

    // The font data stream starts right after the directories, which have
    // variable-length entries: walk them to find where they end.
    size_t pos = WOFF2_HEADER_SIZE;
    skipTableDirectory(f, pos, numTables);
    if (flavor == TTCF_FLAVOR)
        skipCollectionDirectory(f, pos);

    const(ubyte)[] directories = f[WOFF2_HEADER_SIZE .. pos];
    const(ubyte)[] fontStream  = slice(f, pos, compressedSize);
    const(ubyte)[] metaStream  = metaOffset ? slice(f, metaOffset, metaLength) : null;
    const(ubyte)[] privBlock   = privOffset ? slice(f, privOffset, privLength) : null;

    const(ubyte)[] newFont = recompress(fontStream, "font data", crunch);
    const(ubyte)[] newMeta = metaStream ? recompress(metaStream, "metadata", crunch, metaOrigLength) : null;

    // Layout: header, directories, font stream, then each of metadata and
    // private data starting on a 4-byte boundary.
    ubyte[] o = new ubyte[WOFF2_HEADER_SIZE];
    o[] = f[0 .. WOFF2_HEADER_SIZE];
    o ~= directories;
    o ~= newFont;
    uint newMetaOffset = 0, newPrivOffset = 0;
    if (newMeta)
    {
        pad4(o);
        newMetaOffset = cast(uint) o.length;
        o ~= newMeta;
    }
    if (privBlock)
    {
        pad4(o);
        newPrivOffset = cast(uint) o.length;
        o ~= privBlock;
    }

    setBe32(o, 8,  cast(uint) o.length);
    setBe32(o, 20, cast(uint) newFont.length);
    setBe32(o, 28, newMetaOffset);
    setBe32(o, 32, cast(uint) newMeta.length);
    setBe32(o, 40, newPrivOffset);
    return o;
}

/// Decode a Brotli stream, re-encode it at the highest level, return the
/// smaller of the two. With `crunch`, try every window size, else only the
/// default one.
const(ubyte)[] recompress(const(ubyte)[] stream, string what, bool crunch,
                          size_t expectedLength = size_t.max)
{
    ubyte[] decoded;
    BrotliStatus st = brotliDecompressAlloc(stream, decoded);
    if (st != BrotliStatus.success)
        throw new Exception(what ~ ": Brotli decoding failed");
    scope (exit) () @trusted { brotliFree(decoded); }();
    if (expectedLength != size_t.max && decoded.length != expectedLength)
        throw new Exception(what ~ ": unexpected decoded size");

    writefln("  %s: %d bytes decoded, %d bytes compressed", what, decoded.length, stream.length);

    // With crunch, try every window size. A window covering the whole input is
    // as large as useful (the encoder uses no larger one), so stop there.
    int minBits = crunch ? BROTLI_MIN_WINDOW_BITS : BROTLI_DEFAULT_WINDOW_BITS;
    int maxBits = crunch ? BROTLI_MAX_WINDOW_BITS : BROTLI_DEFAULT_WINDOW_BITS;
    const(ubyte)[] best = stream;
    foreach (windowBits; minBits .. maxBits + 1)
    {
        ubyte[] encoded;
        st = brotliCompressAlloc(decoded, encoded, BROTLI_MAX_LEVEL, windowBits);
        if (st != BrotliStatus.success)
            throw new Exception(what ~ ": Brotli encoding failed");
        scope (exit) () @trusted { brotliFree(encoded); }();

        writefln("    window %2d: %8d bytes", windowBits, encoded.length);
        if (encoded.length < best.length)
            best = encoded.dup;

        if ((size_t(1) << windowBits) - 16 >= decoded.length)
            break;
    }
    writefln("    kept %d bytes", best.length);
    return best;
}

void skipTableDirectory(const(ubyte)[] f, ref size_t pos, uint numTables)
{
    enum GLYF_INDEX = 10, LOCA_INDEX = 11; // in the WOFF2 known-tag table
    foreach (i; 0 .. numTables)
    {
        ubyte flags = byteAt(f, pos++);
        uint tagIndex = flags & 0x3F;
        uint transform = flags >> 6;
        if (tagIndex == 0x3F)
            pos += 4; // arbitrary tag
        readBase128(f, pos); // origLength
        // glyf/loca are transformed with version 0; others with version != 0.
        bool glyfOrLoca = tagIndex == GLYF_INDEX || tagIndex == LOCA_INDEX;
        if (glyfOrLoca ? transform == 0 : transform != 0)
            readBase128(f, pos); // transformLength
    }
}

void skipCollectionDirectory(const(ubyte)[] f, ref size_t pos)
{
    pos += 4; // version
    uint numFonts = read255UShort(f, pos);
    foreach (i; 0 .. numFonts)
    {
        uint numTables = read255UShort(f, pos);
        pos += 4; // flavor
        foreach (j; 0 .. numTables)
            read255UShort(f, pos);
    }
}

uint readBase128(const(ubyte)[] f, ref size_t pos)
{
    uint r = 0;
    foreach (i; 0 .. 5)
    {
        ubyte b = byteAt(f, pos++);
        r = (r << 7) | (b & 0x7F);
        if (!(b & 0x80))
            return r;
    }
    throw new Exception("bad UIntBase128");
}

uint read255UShort(const(ubyte)[] f, ref size_t pos)
{
    ubyte code = byteAt(f, pos++);
    switch (code)
    {
        case 253:
            uint v = be16(f, pos);
            pos += 2;
            return v;
        case 254: return 253 * 2 + byteAt(f, pos++);
        case 255: return 253 + byteAt(f, pos++);
        default:  return code;
    }
}

const(ubyte)[] slice(const(ubyte)[] f, size_t offset, size_t length)
{
    if (offset > f.length || length > f.length - offset)
        throw new Exception("block out of file bounds");
    return f[offset .. offset + length];
}

ubyte byteAt(const(ubyte)[] f, size_t pos)
{
    if (pos >= f.length)
        throw new Exception("truncated file");
    return f[pos];
}

uint be16(const(ubyte)[] f, size_t pos)
{
    return (byteAt(f, pos) << 8) | byteAt(f, pos + 1);
}

uint be32(const(ubyte)[] f, size_t pos)
{
    return (be16(f, pos) << 16) | be16(f, pos + 2);
}

void setBe32(ubyte[] o, size_t pos, uint v)
{
    o[pos .. pos + 4] = [cast(ubyte)(v >> 24), cast(ubyte)(v >> 16), cast(ubyte)(v >> 8), cast(ubyte) v];
}

void pad4(ref ubyte[] o)
{
    while (o.length % 4)
        o ~= 0;
}
