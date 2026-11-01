# brotli

Self-contained [Brotli](https://www.rfc-editor.org/rfc/rfc7932) decoder/encoder in the D programming language.

- **WARNING** Very much vibe-coded code (Opus 5.5).
- `@safe pure nothrow @nogc`, D source.
- One-shot decoding, memory to memory.
- Translation of Google's reference decoder (MIT), but non-streaming decoder.
- 


## Usage

```d
import brotli;

// Decode into your own buffer.
ubyte[4096] buf;
size_t len;
BrotliStatus s = brotliDecompress(compressed, buf[], len);
if (s == BrotliStatus.success)
    use(buf[0..len]);

// Decode into a malloc'ed buffer, with an optional size limit.
ubyte[] decoded;
if (brotliDecompressAlloc(compressed, decoded, 100_000_000) == BrotliStatus.success)
{
    use(decoded);
    () @trusted { brotliFree(decoded); }();
}
```

`BrotliStatus` is either:
- `BrotliStatus.success`, 
- `BrotliStatus.corrupted` (invalid, truncated, or followed
by trailing bytes), 
- `BrotliStatus.outputTooSmall` (give larger buffer), 
- `BrotliStatus.outOfMemory`.

## Limitations

- No streaming API: the whole input must be in memory, and the whole output too.
- Large-window Brotli (non-RFC extension) and shared dictionaries are not supported.


## Testing

```
dmd -of=tester -I<numem>/source -Isource test/tester.d source/brotli/*.d <numem>/source/numem/core/hooks.d
./tester check test/suite.bin test/suite.txt
```

`test/suite.*` holds ~18000 streams (valid and corrupted) with the reference
decoder's results. `test/make_suite.py`, `test/compare.py` and `test/bench.py`
regenerate it, fuzz, and benchmark against the reference (`pip install brotli`).

## License

MIT. Copyright (c) 2009, 2010, 2013-2016 by the Brotli Authors, (c) 2026 Guillaume Piolat.
