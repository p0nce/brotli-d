/**
    Decompress a Brotli file.

    Usage: brotli-decode <input.br> <output>

    The whole input is read into memory, and so is the whole output.
    Prints the decoding time in microseconds (reading and writing the files
    are not counted).
*/
module app;

import std.datetime.stopwatch : AutoStart, StopWatch;
import std.file : read, write;
import std.stdio : stderr, writefln;

import brotli;

int main(string[] args)
{
    if (args.length != 3)
    {
        stderr.writefln("Usage: brotli-decode <input.br> <output>");
        return 1;
    }

    const(ubyte)[] input = cast(const(ubyte)[]) read(args[1]);

    ubyte[] output;
    StopWatch sw = StopWatch(AutoStart.yes);
    BrotliStatus status = brotliDecompressAlloc(input, output);
    sw.stop();
    long us = sw.peek.total!"usecs";
    if (status != BrotliStatus.success)
    {
        stderr.writefln("brotli-decode: decompression failed (%s)", status);
        return 1;
    }
    scope (exit) () @trusted { brotliFree(output); }();

    write(args[2], output);
    writefln("%s: %d -> %d bytes, %d µs", args[1], input.length, output.length, us);
    return 0;
}
