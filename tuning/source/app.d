/**
    Compress every file in inputs/ with the D encoder, at a range of levels.

    Usage: test-encoder <level> [--iters <num>]
           test-encoder <min level> <max level> [--iters <num>]

    inputs/ is the directory next to the executable. For each level and file,
    prints the compression ratio (compressed / original), the bytes before and
    after, and the speed in MB/s (10^6 bytes per second, of input), the best of
    --iters runs (default 1). Every output is decoded again and compared to its
    input (not timed), then written to outputs/<file>.br (outputs/ is created
    next to the executable). With several levels, each level overwrites the
    previous one: the file holds the last level.

    Small inputs (CHUNK_GROUPS): 64 chunks of 100 B, 32 of 1 KB and 16 of
    5 KB are cut from the inputs, spread over the files, and compressed one by
    one; outputs in outputs/<group>/. Each group is one line, with the bytes
    and the best times summed over its chunks, and one ratio.

    After each level, prints the min, max, median and mean of its ratios.
*/
module app;

import std.algorithm;
import std.conv;
import std.datetime.stopwatch;
import std.file;
import std.format;
import std.path;
import std.stdio;

import brotli;
import consolecolors;

struct ChunkGroup
{
    string name;
    size_t size;  // bytes per chunk
    size_t count; // chunks in all
}

immutable ChunkGroup[] CHUNK_GROUPS = [
    ChunkGroup("small-100", 100, 64),
    ChunkGroup("small-1k", 1024, 32),
    ChunkGroup("small-5k", 5120, 16),
];

struct Input
{
    string name;
    const(ubyte)[] data;
}

int main(string[] args)
{
    enableConsoleUTF8(); // file names with accents

    int minLevel, maxLevel;
    int iters = 1;
    try
    {
        string[] levels;
        for (size_t i = 1; i < args.length; i++)
        {
            if (args[i] == "--iters")
            {
                if (++i == args.length)
                    return usage();
                iters = args[i].to!int;
            }
            else
                levels ~= args[i];
        }
        if (levels.length == 1)
            minLevel = maxLevel = levels[0].to!int;
        else if (levels.length == 2)
        {
            minLevel = levels[0].to!int;
            maxLevel = levels[1].to!int;
        }
        else
            return usage();
    }
    catch (ConvException)
        return usage();

    if (minLevel < BROTLI_MIN_LEVEL || maxLevel > BROTLI_MAX_LEVEL || minLevel > maxLevel)
    {
        stderr.writefln("test-encoder: levels must be in %d..%d, min <= max",
                        BROTLI_MIN_LEVEL, BROTLI_MAX_LEVEL);
        return 1;
    }
    if (iters < 1)
    {
        stderr.writefln("test-encoder: --iters must be at least 1");
        return 1;
    }

    string dir = buildPath(dirName(thisExePath), "inputs");
    if (!exists(dir) || !isDir(dir))
    {
        stderr.writefln("test-encoder: no directory %s", dir);
        return 1;
    }

    string[] files = dirEntries(dir, SpanMode.shallow).filterFiles;
    files.sort();
    if (files.length == 0)
    {
        stderr.writefln("test-encoder: %s is empty", dir);
        return 1;
    }

    Input[] inputs;
    foreach (path; files)
        inputs ~= Input(baseName(path), cast(const(ubyte)[]) read(path));

    // Small inputs, cut from the files above.
    Input[][] groupChunks;
    foreach (group; CHUNK_GROUPS)
        groupChunks ~= cutChunks(inputs, group);

    string outDir = buildPath(dirName(thisExePath), "outputs");
    mkdirRecurse(outDir);
    foreach (group; CHUNK_GROUPS)
        mkdirRecurse(buildPath(outDir, group.name));

    foreach (level; minLevel .. maxLevel + 1)
    {
        if (level > minLevel)
            writefln("");
        writefln("%-32s %5s %8s %12s %12s %10s", "file", "level", "ratio", "before", "after", "MB/s");

        double[] ratios;

        foreach (input; inputs)
        {
            size_t after;
            long ns;
            ubyte[] output;
            if (!encode(input.data, level, iters, input.name, after, ns, output))
                return 1;
            std.file.write(buildPath(outDir, input.name ~ ".br"), output);
            ratios ~= report(input.name, level, input.data.length, after, ns);
        }

        foreach (g, group; CHUNK_GROUPS)
        {
            Input[] chunks = groupChunks[g];
            if (chunks.length == 0)
                continue;

            size_t before, after;
            long ns;
            foreach (chunk; chunks)
            {
                size_t chunkAfter;
                long chunkNs;
                ubyte[] output;
                if (!encode(chunk.data, level, iters, group.name ~ "/" ~ chunk.name,
                            chunkAfter, chunkNs, output))
                    return 1;
                std.file.write(buildPath(outDir, group.name, chunk.name ~ ".br"), output);
                before += chunk.data.length;
                after += chunkAfter;
                ns += chunkNs;
            }
            ratios ~= report(format("%s (%d chunks)", group.name, chunks.length),
                             level, before, after, ns);
        }

        summary(level, ratios);
    }
    return 0;
}

// The min, max, median and mean of one level's ratios.
void summary(int level, double[] ratios)
{
    double[] s = ratios.dup;
    s.sort();
    size_t n = s.length;
    double median = n % 2 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
    writefln("");
    writefln("level %d: %d ratios (compressed / original)", level, n);
    say("  min    " ~ yellow(format("%7.2f%%", s[0])));
    say("  max    " ~ yellow(format("%7.2f%%", s[$ - 1])));
    say("  median " ~ yellow(format("%7.2f%%", median)));
    say("  mean   " ~ yellow(format("%7.2f%%", sum(s) / n)));
}

// group.count chunks of group.size bytes, dealt to the inputs in turn (files
// shorter than a chunk are skipped), at evenly spaced offsets in each file.
Input[] cutChunks(const(Input)[] inputs, ChunkGroup group)
{
    const(Input)[] eligible;
    foreach (input; inputs)
        if (input.data.length >= group.size)
            eligible ~= input;
    if (eligible.length == 0)
        return null;

    Input[] chunks;
    foreach (f, input; eligible)
    {
        // Chunks k with k % eligible.length == f go to this file.
        size_t n = group.count / eligible.length + (f < group.count % eligible.length ? 1 : 0);
        foreach (i; 0 .. n)
        {
            // Centered in n equal slices of the file.
            size_t offset = (input.data.length - group.size) * (2 * i + 1) / (2 * n);
            chunks ~= Input(format("%s_%d", input.name, i), input.data[offset .. offset + group.size]);
        }
    }
    return chunks;
}

// Compresses `input` at `level` `iters` times; `ns` is the best time. Then
// checks that the output decodes back to `input` (not timed). `output` is the
// compressed stream. Prints an error naming `what` and returns false on failure.
// One encoder for every call: files, levels and chunks.
BrotliEncoder encoder;

bool encode(const(ubyte)[] input, int level, int iters, string what,
            out size_t encodedSize, out long ns, out ubyte[] output)
{
    ubyte[] buf = new ubyte[brotliCompressBound(input.length)];
    ns = long.max;
    foreach (iter; 0 .. iters)
    {
        StopWatch sw = StopWatch(AutoStart.yes);
        BrotliStatus status = brotliCompress(encoder, input, buf, encodedSize, level);
        sw.stop();
        ns = min(ns, sw.peek.total!"nsecs");

        if (status != BrotliStatus.success)
        {
            stderr.writefln("test-encoder: %s: compression failed at level %d (%s)", what, level, status);
            return false;
        }
    }

    output = buf[0 .. encodedSize];

    ubyte[] decoded = new ubyte[input.length];
    size_t decodedSize;
    if (brotliDecompress(output, decoded, decodedSize) != BrotliStatus.success
        || decodedSize != input.length || decoded != input)
    {
        stderr.writefln("test-encoder: %s: level %d does not decode to the input", what, level);
        return false;
    }
    return true;
}

// Prints one line and returns its ratio.
double report(string name, int level, size_t before, size_t after, long ns)
{
    double ratio = before ? 100.0 * after / before : 0.0;
    double mbs = ns > 0 ? before * 1e3 / ns : 0.0; // bytes per ns * 1000 = MB/s
    // Padded before escaping, so that "&amp;" counts as one column.
    say(format("%s %5d %s %12d %12d %s", escaped(format("%-32s", name)),
               level, yellow(format("%7.2f%%", ratio)), before, after,
               lcyan(format("%10.2f", mbs))));
    return ratio;
}

// console-colors 1.3.1 prints nothing at all, not even the text, when stdout
// is not a console (redirected to a file, piped). Then write plain text.
version (Windows) extern (C) int _isatty(int fd);
version (Posix) import core.sys.posix.unistd;

bool colors()
{
    static bool done, tty;
    if (!done)
    {
        version (Windows) tty = _isatty(stdout.fileno) != 0;
        version (Posix) tty = isatty(stdout.fileno) != 0;
        done = true;
    }
    return tty;
}

string yellow(string s) { return colors ? "<yellow>" ~ s ~ "</yellow>" : s; }
string lcyan(string s) { return colors ? "<lcyan>" ~ s ~ "</lcyan>" : s; }
string escaped(string s) { return colors ? s.escapeCCL.idup : s; } // plain text inside a CCL line

void say(string line)
{
    if (colors)
        cwritefln("%s", line);
    else
        writefln("%s", line);
}

int usage()
{
    stderr.writefln("Usage: test-encoder <level> [--iters <num>]");
    stderr.writefln("       test-encoder <min level> <max level> [--iters <num>]");
    return 1;
}

string[] filterFiles(R)(R entries)
{
    string[] r;
    foreach (e; entries)
        if (e.isFile)
            r ~= e.name;
    return r;
}
