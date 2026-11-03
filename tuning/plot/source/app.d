/**
    Runs test-encoder builds (eg. baseline.exe and test-encoder.exe) and
    plots their speed against their mean ratio, one point per level, in an
    SVG.

    Usage: plot [--levels <min> <max>] [--iters <num>] [--category <list>]
                [--out <file.svg>] [--assume-same-speed] <encoder.exe> [<encoder.exe> ...]

    Each encoder runs as `<encoder.exe> <level> [--iters <num>]
    [--category=<list>]`, one at a time (no parallel jobs: the machine
    heats), and its summary line "level N   median X%   mean Y%   Z MB/s" is
    read from its output. At each level the encoders run ABBA: in order, then
    in reverse order, so that the machine heating up slows them alike. An
    encoder's speed at a level is the best of its two runs; --iters is per
    run.
    Levels default to 1..11. The SVG goes to plot.svg next to this executable
    unless --out says otherwise. The numbers are printed as a table too.

    --category (also -category=<list>): which inputs the encoders compress,
    passed down as is: 100, 1k, 5k, large, all, comma-separated. Without it
    they compress everything. Every encoder must know the option. Shown in
    the SVG title.

    --assume-same-speed: in the SVG, every encoder gets the same speed at a
    level, the mean of their measured speeds, so that only the ratios differ
    (for changes that don't touch speed, where 1 iteration is noise). The
    table still shows the measured speeds.

    X axis: mean ratio (compressed / original, lower is better). Y axis:
    speed in MB/s, log scale. Better is up and left. The time of the run is
    shown top right, so that browsers don't show a cached older plot.
*/
module app;

import std.algorithm;
import std.array;
import std.conv;
import std.datetime;
import std.file;
import std.format;
import std.math;
import std.path;
import std.process;
import std.range;
import std.stdio;
import std.string;

struct Point
{
    int level;
    double mean;  // mean ratio, %
    double speed; // MB/s
}

struct Series
{
    string name;
    Point[] points;
}

int main(string[] args)
{
    int minLevel = 1, maxLevel = 11;
    string iters;
    string category; // --category, passed to the encoders; empty: their default (all)
    bool sameSpeed;
    string outPath = buildPath(dirName(thisExePath), "plot.svg");
    string[] exes;
    try
    {
        for (size_t i = 1; i < args.length; i++)
        {
            if (args[i] == "--levels")
            {
                if (i + 2 >= args.length)
                    return usage();
                minLevel = args[++i].to!int;
                maxLevel = args[++i].to!int;
            }
            else if (args[i] == "--assume-same-speed")
                sameSpeed = true;
            else if (args[i] == "--iters")
            {
                if (++i == args.length)
                    return usage();
                iters = args[i];
            }
            else if (args[i] == "--category" || args[i] == "-category")
            {
                if (++i == args.length)
                    return usage();
                category = args[i];
            }
            else if (args[i].startsWith("--category="))
                category = args[i]["--category=".length .. $];
            else if (args[i].startsWith("-category="))
                category = args[i]["-category=".length .. $];
            else if (args[i] == "--out")
            {
                if (++i == args.length)
                    return usage();
                outPath = args[i];
            }
            else
                exes ~= args[i];
        }
    }
    catch (ConvException)
        return usage();
    if (exes.length == 0 || minLevel > maxLevel)
        return usage();

    // ABBA: at each level, the encoders run in order, then in reverse order,
    // so that a drift (the machine heating) slows them alike. An encoder's
    // speed at a level is its best of its runs there.
    Series[] series;
    foreach (exe; exes)
        series ~= Series(stripExtension(baseName(exe)));
    size_t[] order = iota(exes.length).array ~ iota(exes.length).retro.array;
    foreach (level; minLevel .. maxLevel + 1)
    {
        foreach (e; order)
        {
            string[] cmd = [absolutePath(exes[e]), level.to!string];
            if (iters.length)
                cmd ~= ["--iters", iters];
            if (category.length)
                cmd ~= "--category=" ~ category;
            stderr.writefln("running %s", cmd.join(" "));
            auto r = execute(cmd);
            if (r.status != 0)
            {
                stderr.writefln("plot: %s failed (exit %d):\n%s", exes[e], r.status, r.output);
                return 1;
            }
            Point[] points = parse(r.output);
            if (points.length != 1 || points[0].level != level)
            {
                stderr.writefln("plot: no summary line for level %d in the output of %s:\n%s",
                                level, exes[e], r.output);
                return 1;
            }
            Point p = points[0];
            Point[] mine = series[e].points; // shares its elements
            if (mine.length && mine[$ - 1].level == level)
                mine[$ - 1].speed = max(mine[$ - 1].speed, p.speed);
            else
                series[e].points ~= p;
        }
    }

    printTable(series);
    if (sameSpeed)
        averageSpeeds(series);
    std.file.write(outPath, svg(series, category));
    writefln("wrote %s", outPath);
    return 0;
}

// "level  5   median  22.10%   mean  25.31%     45.67 MB/s"
Point[] parse(string output)
{
    Point[] points;
    foreach (line; output.lineSplitter)
    {
        // console-colors may put an escape sequence before the first line.
        ptrdiff_t start = line.indexOf("level");
        if (start < 0)
            continue;
        string[] t = line[start .. $].split;
        if (t.length < 2 || t[0] != "level")
            continue;
        ptrdiff_t m = t.countUntil("mean");
        ptrdiff_t mb = t.countUntil("MB/s");
        if (m < 0 || m + 1 >= t.length || mb < 1)
            continue;
        try
            points ~= Point(t[1].to!int, t[m + 1].chomp("%").to!double, t[mb - 1].to!double);
        catch (ConvException)
            continue;
    }
    return points;
}

// One row per level: each encoder's mean and speed, and for the others their
// change against the first one.
void printTable(Series[] series)
{
    write("level");
    foreach (i, s; series)
    {
        writef("  %22s", s.name);
        if (i > 0)
            writef("  %16s", "vs " ~ series[0].name);
    }
    writeln();
    foreach (p0; series[0].points)
    {
        writef("%5d", p0.level);
        foreach (i, s; series)
        {
            auto f = s.points.find!(p => p.level == p0.level);
            if (f.empty)
            {
                writef("  %22s", "-");
                continue;
            }
            Point p = f.front;
            writef("  %7.3f%% %8.2f MB/s", p.mean, p.speed);
            if (i > 0)
                writef("  %+6.3f %+7.1f%%", p.mean - p0.mean, 100 * (p.speed / p0.speed - 1));
        }
        writeln();
    }
}

// --assume-same-speed: at each level, every encoder's speed becomes the mean
// of the speeds measured at that level.
void averageSpeeds(Series[] series)
{
    double[int] total;
    int[int] count;
    foreach (s; series)
        foreach (p; s.points)
        {
            total[p.level] = total.get(p.level, 0.0) + p.speed;
            count[p.level] = count.get(p.level, 0) + 1;
        }
    foreach (ref s; series)
        foreach (ref p; s.points)
            p.speed = total[p.level] / count[p.level];
}

// Categorical slots, light and dark steps, in fixed order.
immutable string[] LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                            "#e87ba4", "#008300", "#6250d6", "#e34948"];
immutable string[] DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                           "#d55181", "#008300", "#9085e9", "#e66767"];

enum W = 900, H = 600;
enum L = 80, R = 30, T = 70, B = 60; // margins

string svg(Series[] series, string category)
{
    double xmin = double.max, xmax = -double.max, ymin = double.max, ymax = -double.max;
    foreach (s; series)
        foreach (p; s.points)
        {
            xmin = min(xmin, p.mean);
            xmax = max(xmax, p.mean);
            ymin = min(ymin, max(p.speed, 1e-3));
            ymax = max(ymax, max(p.speed, 1e-3));
        }
    // Room around the points; log padding on speed.
    double xpad = max((xmax - xmin) * 0.06, 0.05);
    xmin -= xpad;
    xmax += xpad;
    double ly0 = log10(ymin) - 0.08, ly1 = log10(ymax) + 0.08;

    double px(double x) { return L + (x - xmin) / (xmax - xmin) * (W - L - R); }
    double py(double y) { return H - B - (log10(max(y, 1e-3)) - ly0) / (ly1 - ly0) * (H - T - B); }

    auto o = appender!string;
    o ~= format(`<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="system-ui, Segoe UI, sans-serif">`,
                W, H, W, H);
    o ~= "\n<style>\n";
    o ~= ".bg{fill:#fcfcfb}.ink{fill:#0b0b0b}.ink2{fill:#52514e}.grid{stroke:#e4e3df;stroke-width:1}.axis{stroke:#9b9a95;stroke-width:1}circle.ring{stroke:#fcfcfb}\n";
    foreach (i; 0 .. LIGHT.length)
        o ~= format(".s%d{stroke:%s;fill:%s}\n", i, LIGHT[i], LIGHT[i]);
    o ~= "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.ink{fill:#f0efec}.ink2{fill:#c3c2b7}.grid{stroke:#2f2f2d}.axis{stroke:#6b6a66}circle.ring{stroke:#1a1a19}\n";
    foreach (i; 0 .. DARK.length)
        o ~= format(".s%d{stroke:%s;fill:%s}\n", i, DARK[i], DARK[i]);
    o ~= "}\n</style>\n";
    o ~= format(`<rect class="bg" width="%d" height="%d"/>`, W, H) ~ "\n";

    o ~= format(`<text class="ink" x="%d" y="28" font-size="18" font-weight="600">Speed vs mean ratio, levels %d to %d%s</text>`,
                L, series[0].points[0].level, series[0].points[$ - 1].level,
                category.length ? xmlEscape(", category " ~ category) : "") ~ "\n";
    // Time of the run: each SVG differs, so browsers don't show a cached one.
    SysTime now = Clock.currTime;
    o ~= format(`<text class="ink2" x="%d" y="28" font-size="12" text-anchor="end">%04d-%02d-%02d %02d:%02d:%02d</text>`,
                W - R, now.year, cast(int) now.month, now.day, now.hour, now.minute, now.second) ~ "\n";

    // Legend, one row above the plot.
    double lx = L;
    foreach (i, s; series)
    {
        size_t c = i % LIGHT.length;
        o ~= format(`<circle class="s%d" cx="%.1f" cy="50" r="5"/>`, c, lx + 5);
        o ~= format(`<text class="ink2" x="%.1f" y="54.5" font-size="13">%s</text>`, lx + 15, xmlEscape(s.name)) ~ "\n";
        lx += 30 + 7.5 * s.name.length;
    }

    // Y grid: 1, 2, 5 x 10^k.
    for (int k = cast(int) floor(ly0); k <= cast(int) ceil(ly1); k++)
        foreach (m; [1, 2, 5])
        {
            double v = m * 10.0 ^^ k;
            if (log10(v) < ly0 || log10(v) > ly1)
                continue;
            double y = py(v);
            o ~= format(`<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>`, L, W - R, y, y);
            o ~= format(`<text class="ink2" x="%d" y="%.1f" font-size="12" text-anchor="end">%s</text>`,
                        L - 8, y + 4, v >= 1 ? format("%g", v) : format("%.2g", v)) ~ "\n";
        }
    // X grid: a nice step.
    double step = niceStep((xmax - xmin) / 8);
    for (double v = ceil(xmin / step) * step; v <= xmax; v += step)
    {
        double x = px(v);
        o ~= format(`<line class="grid" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>`, x, x, T, H - B);
        o ~= format(`<text class="ink2" x="%.1f" y="%d" font-size="12" text-anchor="middle">%s%%</text>`,
                    x, H - B + 18, format("%g", round(v / step) * step)) ~ "\n";
    }
    o ~= format(`<line class="axis" x1="%d" x2="%d" y1="%d" y2="%d"/>`, L, W - R, H - B, H - B) ~ "\n";
    o ~= format(`<line class="axis" x1="%d" x2="%d" y1="%d" y2="%d"/>`, L, L, T, H - B) ~ "\n";
    o ~= format(`<text class="ink2" x="%d" y="%d" font-size="13" text-anchor="middle">mean ratio (compressed / original, lower is better)</text>`,
                (L + W - R) / 2, H - 18) ~ "\n";
    o ~= format(`<text class="ink2" x="20" y="%d" font-size="13" text-anchor="middle" transform="rotate(-90 20 %d)">speed, MB/s (log)</text>`,
                (T + H - B) / 2, (T + H - B) / 2) ~ "\n";

    // Lines, then points with a level label and a tooltip.
    foreach (i, s; series)
    {
        size_t c = i % LIGHT.length;
        string pts = s.points.map!(p => format("%.1f,%.1f", px(p.mean), py(p.speed))).join(" ");
        o ~= format(`<polyline class="s%d" fill="none" stroke-width="1" stroke-linejoin="round" opacity="0.8" points="%s" style="fill:none"/>`, c, pts) ~ "\n";
    }
    foreach (i, s; series)
    {
        size_t c = i % LIGHT.length;
        // Labels of the first encoder to the right, the second to the left...
        bool right = i % 2 == 0;
        foreach (p; s.points)
        {
            double x = px(p.mean), y = py(p.speed);
            o ~= format(`<g><title>%s, level %d: mean %.3f%%, %.2f MB/s</title>`, xmlEscape(s.name), p.level, p.mean, p.speed);
            o ~= format(`<circle cx="%.1f" cy="%.1f" r="12" fill="transparent"/>`, x, y);
            o ~= format(`<circle class="s%d ring" cx="%.1f" cy="%.1f" r="5" stroke-width="2" opacity="0.5"/>`,
                        c, x, y);
            o ~= format(`<text class="ink2" x="%.1f" y="%.1f" font-size="11" text-anchor="%s">%d</text></g>`,
                        right ? x + 9 : x - 9, y - 6, right ? "start" : "end", p.level) ~ "\n";
        }
    }
    o ~= "</svg>\n";
    return o[];
}

double niceStep(double raw)
{
    double p = 10.0 ^^ floor(log10(raw));
    foreach (m; [1.0, 2.0, 5.0])
        if (m * p >= raw)
            return m * p;
    return 10 * p;
}

string xmlEscape(string s)
{
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
}

int usage()
{
    stderr.writefln("Usage: plot [--levels <min> <max>] [--iters <num>] [--category <list>] [--out <file.svg>]");
    stderr.writefln("            [--assume-same-speed] <encoder.exe> [<encoder.exe> ...]");
    return 1;
}
