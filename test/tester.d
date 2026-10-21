/**
Test driver for the brotli decoder and encoder.

    tester batch <in> <out>
        <in> is a sequence of records: uint32 little-endian length, then bytes.
        For each record, writes a line "<status> <length> <fnv1a64 hex>" to <out>.

    tester bench <file.br> <iterations>

    tester encbatch <in> <out> <level> <windowBits>
        Compresses each record of <in> (same format as batch), checks that it
        decodes back, and that the fixed-size API agrees. Writes the compressed
        records to <out>, in the same format.

    tester enc <file> <level> <windowBits> <iterations>
        Compresses a file, checks that it decodes back, prints
        "<compressed size> <MB/s>" (best of the iterations).
*/
import std.stdio, std.file, std.conv, std.format, std.datetime.stopwatch;
import brotli;

ulong fnv1a(const(ubyte)[] d)
{
    ulong h = 0xcbf29ce484222325UL;
    foreach (b; d)
    {
        h ^= b;
        h *= 0x100000001b3UL;
    }
    return h;
}

// Decode with both APIs, check they agree.
string decodeAndCheck(const(ubyte)[] input)
{
    ubyte[] decoded;
    BrotliStatus s = brotliDecompressAlloc(input, decoded, 64 * 1024 * 1024);
    scope(exit) () @trusted { brotliFree(decoded); }();

    if (s == BrotliStatus.success)
    {
        // Exact-size buffer must work.
        ubyte[] buf = new ubyte[decoded.length + 1];
        size_t len;
        BrotliStatus s2 = brotliDecompress(input, buf[0 .. decoded.length], len);
        if (s2 != BrotliStatus.success || len != decoded.length || buf[0 .. len] != decoded)
            throw new Exception("fixed-size API disagrees");
        // One byte too short must fail with outputTooSmall.
        if (decoded.length > 0)
        {
            s2 = brotliDecompress(input, buf[0 .. decoded.length - 1], len);
            if (s2 != BrotliStatus.outputTooSmall)
                throw new Exception("expected outputTooSmall");
            ubyte[] limited;
            s2 = brotliDecompressAlloc(input, limited, decoded.length - 1);
            if (s2 != BrotliStatus.outputTooSmall || limited !is null)
                throw new Exception("expected outputTooSmall with maxOutputSize");
        }
        return format("%d %d %016x", 0, decoded.length, fnv1a(decoded));
    }
    else
    {
        // Fixed-size API must fail too (corrupted, or outputTooSmall).
        ubyte[] buf = new ubyte[1 << 16];
        size_t len;
        BrotliStatus s2 = brotliDecompress(input, buf, len);
        if (s2 == BrotliStatus.success)
            throw new Exception("fixed-size API succeeds but alloc API fails");
        if (s == BrotliStatus.corrupted && s2 != BrotliStatus.corrupted && s2 != BrotliStatus.outputTooSmall)
            throw new Exception("unexpected status");
        return format("%d 0 0", cast(int) s);
    }
}

// Compress with both APIs, check they agree, that it decodes back, and the
// fixed-size API limits.
ubyte[] compressAndCheck(const(ubyte)[] input, int level, int windowBits)
{
    ubyte[] compressed;
    BrotliStatus s = brotliCompressAlloc(input, compressed, level, windowBits);
    if (s != BrotliStatus.success)
        throw new Exception(format("brotliCompressAlloc failed: %s", s));
    scope(exit) () @trusted { brotliFree(compressed); }();
    ubyte[] result = compressed.dup;

    size_t bound = brotliCompressBound(input.length);
    if (result.length > bound)
        throw new Exception(format("compressed size %s > bound %s", result.length, bound));

    // Fixed-size API: same output with an exact size buffer. For small inputs
    // (cheap to compress again), also with the bound, and fails one byte short.
    // That doesn't depend on the level: not for the slow levels.
    ubyte[] buf = new ubyte[bound];
    size_t len;
    s = brotliCompress(input, buf[0 .. result.length], len, level, windowBits);
    if (s != BrotliStatus.success || buf[0 .. len] != result)
        throw new Exception("brotliCompress fails with an exact size buffer");
    if (input.length <= 65536 && level < 10)
    {
        s = brotliCompress(input, buf, len, level, windowBits);
        if (s != BrotliStatus.success || buf[0 .. len] != result)
            throw new Exception("brotliCompress disagrees with brotliCompressAlloc");
        s = brotliCompress(input, buf[0 .. result.length - 1], len, level, windowBits);
        if (s != BrotliStatus.outputTooSmall)
            throw new Exception("brotliCompress doesn't fail one byte short");
    }

    // Decodes back.
    ubyte[] decoded = new ubyte[input.length];
    s = brotliDecompress(result, decoded, len);
    if (s != BrotliStatus.success || len != input.length || decoded != input)
        throw new Exception(format("round trip failed: %s", s));
    return result;
}

int main(string[] args)
{
    if (args.length >= 6 && args[1] == "encbatch")
    {
        auto data = cast(ubyte[]) read(args[2]);
        int level = to!int(args[4]);
        int windowBits = to!int(args[5]);
        auto f = File(args[3], "wb");
        size_t p = 0, i = 0;
        while (p < data.length)
        {
            uint n = data[p] | (data[p + 1] << 8) | (data[p + 2] << 16) | (data[p + 3] << 24);
            p += 4;
            ubyte[] c;
            try
                c = compressAndCheck(data[p .. p + n], level, windowBits);
            catch (Exception e)
            {
                std.file.write("encfail.bin", data[p .. p + n]);
                stderr.writefln("record %s (level %s, window %s): %s, input saved to encfail.bin",
                                i, level, windowBits, e.msg);
                return 1;
            }
            uint cn = cast(uint) c.length;
            ubyte[4] hdr = [cast(ubyte) cn, cast(ubyte)(cn >> 8), cast(ubyte)(cn >> 16), cast(ubyte)(cn >> 24)];
            f.rawWrite(hdr[]);
            f.rawWrite(c);
            p += n;
            i++;
        }
        return 0;
    }
    if (args.length >= 6 && args[1] == "enc")
    {
        auto input = cast(ubyte[]) read(args[2]);
        int level = to!int(args[3]);
        int windowBits = to!int(args[4]);
        int iters = to!int(args[5]);
        ubyte[] c = compressAndCheck(input, level, windowBits);
        ubyte[] buf = new ubyte[brotliCompressBound(input.length)];
        double best = 0;
        foreach (it; 0 .. iters)
        {
            size_t len;
            auto sw = StopWatch(AutoStart.yes);
            if (brotliCompress(input, buf, len, level, windowBits) != BrotliStatus.success)
                return 1;
            double secs = sw.peek.total!"usecs" / 1e6;
            double speed = secs > 0 ? input.length / secs / 1e6 : 0;
            if (speed > best)
                best = speed;
        }
        writefln("%s %.2f", c.length, best);
        return 0;
    }
    if (args.length >= 4 && args[1] == "batch")
    {
        auto data = cast(ubyte[]) read(args[2]);
        auto f = File(args[3], "w");
        size_t p = 0;
        while (p < data.length)
        {
            uint n = data[p] | (data[p + 1] << 8) | (data[p + 2] << 16) | (data[p + 3] << 24);
            p += 4;
            f.writeln(decodeAndCheck(data[p .. p + n]));
            p += n;
        }
        return 0;
    }
    if (args.length >= 4 && args[1] == "check")
    {
        import std.string : splitLines;
        auto data = cast(ubyte[]) read(args[2]);
        string[] expected = (cast(string) read(args[3])).splitLines;
        size_t p = 0, i = 0, bad = 0;
        while (p < data.length)
        {
            uint n = data[p] | (data[p + 1] << 8) | (data[p + 2] << 16) | (data[p + 3] << 24);
            p += 4;
            string got = decodeAndCheck(data[p .. p + n]);
            if (i >= expected.length || got != expected[i])
            {
                writefln("MISMATCH record %s: got \"%s\", expected \"%s\"", i, got,
                         i < expected.length ? expected[i] : "nothing");
                bad++;
            }
            p += n;
            i++;
        }
        if (i != expected.length)
        {
            writefln("record count %s, expected %s", i, expected.length);
            bad++;
        }
        writefln("%s records, %s mismatches", i, bad);
        return bad ? 1 : 0;
    }
    if (args.length >= 4 && args[1] == "bench")
    {
        auto input = cast(ubyte[]) read(args[2]);
        int iters = to!int(args[3]);
        ubyte[] decoded;
        brotliDecompressAlloc(input, decoded);
        ubyte[] buf = new ubyte[decoded.length];
        auto sw = StopWatch(AutoStart.yes);
        foreach (i; 0 .. iters)
        {
            size_t len;
            if (brotliDecompress(input, buf, len) != BrotliStatus.success)
                return 1;
        }
        double secs = sw.peek.total!"usecs" / 1e6;
        writefln("%.1f MB/s", decoded.length * cast(double) iters / secs / 1e6);
        return 0;
    }
    stderr.writeln("usage: tester batch <in> <out> | tester check <suite.bin> <suite.txt> | tester bench <file> <iters>\n"
                 ~ "       tester encbatch <in> <out> <level> <windowBits> | tester enc <file> <level> <windowBits> <iters>");
    return 1;
}
