/**
    Compress a file to Brotli, trying every level.

    Usage: brotli-encode <input> <output.br>

    Encodes the input with each level (BROTLI_MIN_LEVEL..BROTLI_MAX_LEVEL), with
    the default window size (BROTLI_DEFAULT_WINDOW_BITS, the largest: the
    encoder uses no larger window than the input needs), prints the compressed
    size, ratio (compressed / original) and encoding time of each, then writes
    the smallest output to <output.br>. Ties go to the faster encode.
*/
module app;

import std.datetime.stopwatch : StopWatch, AutoStart;
import std.file : read, write;
import std.stdio : stderr, writefln;

import brotli;

int main(string[] args)
{
    if (args.length != 3)
    {
        stderr.writefln("Usage: brotli-encode <input> <output.br>");
        return 1;
    }

    const(ubyte)[] input = cast(const(ubyte)[]) read(args[1]);

    enum windowBits = BROTLI_DEFAULT_WINDOW_BITS;
    writefln("%s: %d bytes, window %d", args[1], input.length, windowBits);
    writefln("%5s %12s %8s %12s", "level", "bytes", "ratio", "time");

    ubyte[] best;
    int bestLevel;
    double bestMs;

    foreach (level; BROTLI_MIN_LEVEL .. BROTLI_MAX_LEVEL + 1)
    {
        ubyte[] output;

        StopWatch sw = StopWatch(AutoStart.yes);
        BrotliStatus status = brotliCompressAlloc(input, output, level, windowBits);
        sw.stop();
        double ms = sw.peek.total!"usecs" / 1000.0;

        if (status != BrotliStatus.success)
        {
            stderr.writefln("brotli-encode: compression failed at level %d (%s)", level, status);
            () @trusted { brotliFree(best); }();
            return 1;
        }

        double ratio = input.length ? 100.0 * output.length / input.length : 0.0;
        writefln("%5d %12d %7.2f%% %9.1f ms", level, output.length, ratio, ms);

        bool better = best.length == 0 || output.length < best.length
                   || (output.length == best.length && ms < bestMs);
        if (better)
        {
            () @trusted { brotliFree(best); }();
            best = output;
            bestLevel = level;
            bestMs = ms;
        }
        else
            () @trusted { brotliFree(output); }();
    }

    scope (exit) () @trusted { brotliFree(best); }();

    double bestRatio = input.length ? 100.0 * best.length / input.length : 0.0;
    writefln("best: level %d, %d bytes, %.2f%%, %.1f ms", bestLevel, best.length, bestRatio, bestMs);

    write(args[2], best);
    writefln("wrote %s", args[2]);
    return 0;
}
