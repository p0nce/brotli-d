/**
    Decompress a Brotli file.

    Usage: brotli-decode <input.br> <output>

    The whole input is read into memory, and so is the whole output.
*/
module app;

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
    BrotliStatus status = brotliDecompressAlloc(input, output);
    if (status != BrotliStatus.success)
    {
        stderr.writefln("brotli-decode: decompression failed (%s)", status);
        return 1;
    }
    scope (exit) () @trusted { brotliFree(output); }();

    write(args[2], output);
    writefln("%s: %d -> %d bytes", args[1], input.length, output.length);
    return 0;
}
