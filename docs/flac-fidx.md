# External FLAC frame indexes (`.fidx`)

## Status

`.fidx` is an experimental, preservation-safe source-access layer for native
FLAC captures that have no seek table or declared total-sample count. It is a
sidecar only: the source FLAC is opened read-only and is never rewritten.

The index addresses frame discovery and bounded source access. It does not
change RF arithmetic, field decoding, or preservation authority. CPU remains
the only backend involved.

## Version 1 binary format

All integers are little-endian. The file consists of a fixed 192-byte header
followed by fixed-width records.

| Header offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `[u8; 8]` | `FLACFIDX` magic |
| 8 | `u16` | format version (`1`) |
| 10 | `u16` | header bytes (`192`) |
| 12 | `u32` | source-hash, fixed-block, and complete flags |
| 16 | `u16` | record kind (`1` fixed offset, `2` explicit) |
| 18 | `u16` | record bytes (`8` or `24`) |
| 20 | `u32` | retained-frame stride (`1` is dense) |
| 24 | `u64` | source byte length |
| 32 | `u64` | byte offset immediately after FLAC metadata |
| 40 | `u64` | source modification time in Unix nanoseconds, when available |
| 48 | `u32` | sample rate |
| 52 | `u16` | channels |
| 54 | `u16` | bits per sample |
| 56 | `u32` | STREAMINFO minimum block size |
| 60 | `u32` | STREAMINFO maximum block size |
| 64 | `u32` | fixed block size, or zero for variable-block streams |
| 68 | `u32` | reserved; must be zero |
| 72 | `u64` | first frame number (version 1 writes zero) |
| 80 | `u64` | observed total samples from validated frames |
| 88 | `u64` | observed frame count |
| 96 | `u64` | retained record count |
| 104 | `u64` | record-array byte offset (`192`) |
| 112 | `[u8; 32]` | source SHA-256 |
| 144 | `[u8; 16]` | STREAMINFO audio MD5 |
| 160 | `[u8; 32]` | record-array SHA-256 |

Fixed-block records are one absolute source-byte offset (`u64`). Their sample
positions derive from the record index, stride, and fixed block size. A sparse
anchor therefore bounds forward parsing to fewer than `stride` FLAC frames.

Variable-block records are 24 bytes: sample start (`u64`), source-byte offset
(`u64`), block size (`u32`), and reserved flags (`u32`).

## Building and inspecting

The builder makes one sequential frame-validation pass through the source using
Symphonia's FLAC parser. Frame header CRC-8 and frame CRC-16 are validated before
an anchor is retained. It then hashes the complete source and atomically moves
the completed sidecar into place. A sparse index is the default; use stride 1
only when a dense offset array is worth its size.

```text
tape-decode fidx build --stride 4096 capture.flac
tape-decode fidx inspect --source capture.flac --verify-source-hash capture.flac.fidx
```

Inspection is a JSON text export. It is not the primary random-access store.

## Exact bounded extraction

An indexed extraction reopens a logical read-only FLAC consisting of the
original metadata followed by source bytes beginning at the nearest prior
anchor. It decodes forward from that anchor and writes exactly the requested
sample count. For an 8-bit preservation FLAC, `u8` output is bit-exact.

```text
tape-decode fidx extract \
  --index capture.flac.fidx \
  --sample-offset 63123025920 \
  --sample-count 343636356 \
  --out sample.u8 \
  capture.flac
```

Independent extraction commands can run concurrently against the same
read-only source. Each worker starts from a validated sparse anchor instead of
decoding from byte zero. The source and sidecar should reside on storage that
can sustain the resulting parallel reads.

`decode --input-format flac --offset N --flac-index capture.flac.fidx` uses the
same indexed reopening path. Without `.fidx`, an unknown-length FLAC reaches a
forward offset by sequential decoding. Known-length ordinary FLACs retain
Symphonia's standard seek path. Stale, malformed, truncated, or incompatible
sidecars fail closed; they never trigger source rewriting.

## Measured preservation check (2026-08-26)

Cosmo built a stride-4096 sidecar from the read-only 3,623,483,068-byte
VHS-0002 r02 preservation FLAC. The build and complete source hash took 17.221
seconds and observed 1,922,608 fixed 4096-sample frames (7,874,999,825 samples).
The resulting sidecar has 470 records, is 3,952 bytes, and has SHA-256
`036edf1a61f0c2da105613eb6fd996a9392d862915d80c629d6d84f94861b480`.

Indexed extraction of samples `[1,861,363,595, 2,204,999,951)` took 7.932
seconds and produced 343,636,356 U8 bytes with SHA-256
`17879672963e3e5c8b6c15b66ec7ce648772432c1c42047534fca0ac939afde0`.
That is an exact match to the independently retained VHS-0002 `[120,132)`
fixture oracle. Two concurrent one-second extractions also matched exact slices
of that fixture. The regenerated derivatives were removed after verification;
the preservation FLAC was not modified.
