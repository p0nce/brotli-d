# brotli

Tiny, self-contained [Brotli](https://www.rfc-editor.org/rfc/rfc7932) decoder for D.

- Single file: `source/brotli.d` (the 122 KB static dictionary is embedded).
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
- Speed is ~75-165 MB/s (LDC, depends on data), about 2x slower than the reference
  decoder: this one aims to be small.

## Testing

```
pip install brotli
git clone https://github.com/google/brotli
dmd -of=tester -I<numem>/source test/tester.d source/brotli.d <numem>/source/numem/core/hooks.d
python test/compare.py ./tester ./brotli [seed] [fuzz iterations]
python test/large.py ./tester
```

## License

MIT. Copyright (c) 2009, 2010, 2013-2016 by the Brotli Authors, (c) 2026 Guillaume Piolat.
