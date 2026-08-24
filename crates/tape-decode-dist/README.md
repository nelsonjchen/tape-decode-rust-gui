# tape-decode-dist POC

`tape-decode-dist` is a localhost-first proof of concept for distributing bounded
`tape-decode` jobs. A coordinator owns a versioned manifest and a persistent
leased queue; heterogeneous runners pull one job at a time, launch a
low-priority decoder child, and stream verified results back. A runner can
either cache the complete content-addressed FLAC or let the decoder read bounded
byte ranges directly from the coordinator.

Internal content identity uses BLAKE3. Input downloads are hashed as they stream
to a `.partial` cache entry (including a resumed prefix), result uploads are
hashed by the coordinator while they stream to `.partial` files, and assembled
luma/chroma outputs are hashed while they are written. This avoids a second
read of the multi-gigabyte happy path. The manifest also records SHA-256 for the
original input and decoder in the same initial scan for preservation and
interchange; SHA-256 is not used for runner cache keys or result identity.

The coordinator refuses non-loopback bind addresses by default. `--allow-lan`
is an explicit escape hatch for a trusted LAN; this POC still has no
authentication or TLS, so it must not be exposed to an untrusted network.

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
  scheduling while the daemon retains normal priority. The default
  `--input-mode full-cache` preserves resumable, BLAKE3-verified LRU behavior.
  `--input-mode http-range` skips that cache and gives the decoder a seekable
  HTTP source backed by one bounded read-ahead window (8 MiB by default). The
  latter validates the coordinator's BLAKE3 ETag and records every requested
  range plus aggregate input bytes in each attempt directory.
  Heterogeneous hosts can supply a native binary with `--decoder`; if its hash
  necessarily differs from the manifest's platform build,
  `--allow-platform-decoder` is also required. Each runner registers its actual
  local decoder BLAKE3 for audit evidence.
- `assemble` requires completed artifacts for every shard. It keeps the earlier
  shard authoritative until two consecutive overlap fields match by absolute
  `fileLoc`, parity, sync confidence, luma, and chroma. An unmatched seam is a
  hard error.
- `repair-plan` turns zero-based failed comparison fields into guarded referee
  decode ranges. Nearby failures are clustered, and ranges whose RF guards
  overlap are merged so the referee never decodes the same guarded interval
  twice.
- `repair-apply` replaces one failed cluster from a guarded referee decode. It
  uses the last valid matching run immediately before the failure and the first
  valid run after it, minimizing the replacement while requiring independently
  proven handoffs on both sides. When applying multiple disjoint repair ranges,
  apply them from the highest field index downward so a field-count change in a
  later range cannot invalidate an earlier index.
- `cleanup` removes only an exact, marked child of `/Users/nelson/NoSync` after
  all external retained artifacts match a retention manifest and no recorded
  process is alive. Retention manifests remain SHA-256-based because they are
  external preservation evidence rather than an internal scheduling hot path.

Run `tape-decode-dist <command> --help` for the complete arguments. The default
manifest geometry describes the VHS-0005 `[542,862)` guarded fixture and its
canonical `[552,852)` interval.

Native heterogeneous decoders can be deterministic on each host yet differ on
a few threshold-sensitive fields across architectures. A matched VHS-0006
stress test produced byte-identical repeat runs on both macOS arm64 and Windows
x86-64, while the two platforms disagreed on a handful of fields. Adaptive
repair is therefore an explicit referee policy, not a claim that averaging or
blindly selecting one platform's samples restores ground truth.
