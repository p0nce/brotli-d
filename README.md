# brotli

Tiny, self-contained [Brotli](https://www.rfc-editor.org/rfc/rfc7932) decoder for D.

- `source/brotli/decoder.d` (the 122 KB static dictionary is embedded), `source/brotli/encoder.d`
  (work in progress, see `TODO.txt`); `import brotli;` imports both.
- `@safe pure nothrow @nogc`. Only dependency is [numem](https://github.com/Inochi2D/numem),
  for its pure `nu_malloc`/`nu_realloc`/`nu_free` (libc by default). Works with `-betterC`.
- One-shot decoding, memory to memory.
- Translation of Google's reference decoder (MIT), simplified to a non-streaming decoder.
- Differentially tested against the reference C decoder: official test vectors,
  every quality/window/mode of the reference encoder, and hundreds of thousands of
  mutated streams (identical output, identical accept/reject decision).

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

`BrotliStatus` is one of `success`, `corrupted` (invalid, truncated, or followed
by trailing bytes), `outputTooSmall`, `outOfMemory`.

## Limitations

- No streaming API: the whole input must be in memory, and the whole output too.
- Large-window Brotli (non-RFC extension) and shared dictionaries are not supported.
- Speed is about the same as the reference C decoder with LDC (~200-340 MB/s on text
  and source code), slightly faster with `-boundscheck=off`. DMD is ~35% slower.

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
