# brotli

[Brotli](https://www.rfc-editor.org/rfc/rfc7932) decoder/encoder in the D programming language.

- **WARNING** Very much vibe-coded code (Opus 5.5). A few constants were hand-tuned to improve efficiency a bit.
- `@safe pure nothrow @nogc`, D source. However **memory-safety is not guaranteed, only agents checked that `@safe` is respected.**
- Decoder is a translation of Google's reference decoder (MIT), but non-streaming.
- Encoder largly inspired from Google's reference too.


## Usage

```d

//
// DECODING API
//

// Decompress a complete Brotli stream into a caller-provided buffer.
BrotliStatus brotliDecompress(const(ubyte)[] input, ubyte[] output, out size_t decodedSize);

// Decompress a complete Brotli stream into a newly allocated buffer.
BrotliStatus brotliDecompressAlloc(const(ubyte)[] input, out ubyte[] output);

// Decompress a complete Brotli stream into a buffer owned by `decoder`.
BrotliStatus brotliDecompressOwned(ref BrotliDecoder decoder, const(ubyte)[] input, out const(ubyte)[] output);


//
// ENCODING API
//

// Compress `input` into a caller-provided buffer. Use a new encoder each time.
BrotliStatus brotliCompress(const(ubyte)[] input, ubyte[] output, out size_t encodedSize,
                            int level = BROTLI_DEFAULT_LEVEL);

// Compress `input` into a caller-provided buffer, reusing an encoder.
BrotliStatus brotliCompress(ref BrotliEncoder encoder, const(ubyte)[] input, ubyte[] output,
                            out size_t encodedSize,
                            int level = BROTLI_DEFAULT_LEVEL);

// Compress `input` into a newly allocated buffer, reusing an encoder.
BrotliStatus brotliCompressAlloc(ref BrotliEncoder encoder, const(ubyte)[] input,
                                 out ubyte[] output,
                                 int level = BROTLI_DEFAULT_LEVEL)

// Maximum compressed size of an input for any level and window.
size_t brotliCompressBound(size_t inputLength);
```


## Limitations

- No streaming API: the whole input must be in memory, and the whole output too.
- Large-window Brotli (non-RFC extension) and shared dictionaries are not supported.



## License

MIT. Copyright (c) 2009, 2010, 2013-2016 by the Brotli Authors, (c) 2026 Guillaume Piolat.
