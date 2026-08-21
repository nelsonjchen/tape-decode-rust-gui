# tape-decode-dist POC

`tape-decode-dist` is a localhost-first proof of concept for distributing bounded
`tape-decode` jobs. A coordinator owns a versioned manifest and a persistent
leased queue; heterogeneous runners pull one job at a time, cache the
content-addressed FLAC input, launch a low-priority decoder child, and stream
verified results back.

Internal content identity uses BLAKE3. Input downloads are hashed as they stream
to a `.partial` cache entry (including a resumed prefix), result uploads are
hashed by the coordinator while they stream to `.partial` files, and assembled
luma/chroma outputs are hashed while they are written. This avoids a second
read of the multi-gigabyte happy path. The manifest also records SHA-256 for the
original input and decoder in the same initial scan for preservation and
interchange; SHA-256 is not used for runner cache keys or result identity.

The coordinator deliberately refuses non-loopback bind addresses. This POC has
no authentication or TLS and is not intended for LAN exposure.

## Commands

- `manifest` hashes the input and decoder once with BLAKE3 and SHA-256 and plans fixed canonical shards with
  explicit internal and outer guards. Guarded starts are extended downward to
  the shared absolute 60-field decoder grid so every shard reproduces the same
  deterministic internal-worker boundaries as a whole-input baseline.
- `coordinator` serves the resumable input, lease, heartbeat, failure, upload,
  completion, and status endpoints. Its state is atomically replaced on every
  transition and can be resumed after restart.
- `runner` processes one lease at a time. `--decode-threads` is an administrator
  limit for that runner; decoder children use nice level 19 and macOS background
  scheduling while the daemon retains normal priority.
- `assemble` requires completed artifacts for every shard. It keeps the earlier
  shard authoritative until two consecutive overlap fields match by absolute
  `fileLoc`, parity, sync confidence, luma, and chroma. An unmatched seam is a
  hard error.
- `cleanup` removes only an exact, marked child of `/Users/nelson/NoSync` after
  all external retained artifacts match a retention manifest and no recorded
  process is alive. Retention manifests remain SHA-256-based because they are
  external preservation evidence rather than an internal scheduling hot path.

Run `tape-decode-dist <command> --help` for the complete arguments. The default
manifest geometry describes the VHS-0005 `[542,862)` guarded fixture and its
canonical `[552,852)` interval.
