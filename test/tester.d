/**
Test driver for the brotli decoder.

    tester batch <in> <out>
        <in> is a sequence of records: uint32 little-endian length, then bytes.
        For each record, writes a line "<status> <length> <fnv1a64 hex>" to <out>.

    tester bench <file.br> <iterations>
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

int main(string[] args)
{
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
    stderr.writeln("usage: tester batch <in> <out> | tester bench <file> <iters>");
    return 1;
}
