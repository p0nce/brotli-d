/**
    Compress a file to Brotli.

    Usage: brotli-encode [-q level] [-w windowBits] <input> <output.br>

    Level is 0-11 (default 6, see BROTLI_DEFAULT_LEVEL). Window bits are 10-24.
*/
module app;

import std.file : read, write;
import std.getopt : getopt, defaultGetoptPrinter;
import std.stdio : stderr, writefln;

import brotli;

int main(string[] args)
{
    int level = BROTLI_DEFAULT_LEVEL;
    int windowBits = BROTLI_DEFAULT_WINDOW_BITS;

    auto opt = getopt(args,
        "q|quality", "Compression level, 0-11 (default 6).", &level,
        "w|window", "Window size in bits, 10-24 (default 24).", &windowBits);

    if (opt.helpWanted || args.length != 3)
    {
        defaultGetoptPrinter("Usage: brotli-encode [options] <input> <output.br>",
                             opt.options);
        return opt.helpWanted ? 0 : 1;
    }

    const(ubyte)[] input = cast(const(ubyte)[]) read(args[1]);

    ubyte[] output;
    BrotliStatus status = brotliCompressAlloc(input, output, level, windowBits);
    if (status != BrotliStatus.success)
    {
        stderr.writefln("brotli-encode: compression failed (%s)", status);
        return 1;
    }
    scope (exit) () @trusted { brotliFree(output); }();

    write(args[2], output);
    writefln("%s: %d -> %d bytes (level %d)", args[1], input.length, output.length, level);
    return 0;
}
