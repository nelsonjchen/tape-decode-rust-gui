# Experimental CUDA backend

## Status

The CUDA backend is explicit, experimental, and non-authoritative. CPU is the
default and the quality oracle. A CUDA request fails rather than silently
falling back when the feature, driver, device, format, profile, or option graph
is unsupported.

The implementation is usable end to end on an NVIDIA RTX 2080, but it is now
mothballed: it did not satisfy the 1.5x performance gate, and a deeper GPU-sync
experiment exposed a rare quality regression. Do not substitute CUDA output
for a preservation decode.

## Design

The backend applies the usual high-throughput GPU DSP structure to the existing
32,768-sample overlap-save graph:

- a complete field's overlapping blocks are uploaded as one batch;
- cuFFT `planMany` handles batched real/complex transforms;
- device kernels apply RF gains, construct the one-sided analytic spectrum,
  extract the envelope, perform phase and differential demodulation, repair
  spikes, apply luma filters, and extract the burst channel;
- the first-order zero-phase envelope filter uses a device-wide affine scan,
  while spike detection and burst-DC removal use parallel reductions rather
  than one serial CUDA thread per RF block;
- packed usable block interiors are copied back to the existing CPU field,
  sync, chroma, scaling, dropout, and metadata pipeline;
- device buffers and cuFFT plans are retained by batch shape and reused; and
- NVRTC compiles a device-specific CUBIN with contraction disabled for closer
  agreement with the scalar Rust phase path. The CUBIN cache key includes the
  kernel source, crate version, architecture, and compiler options.

The existing bounded multi-worker path can give each worker an independent CUDA
stream while retaining one forward-only shared input producer. Workers do not
reopen or seek the source. This improves overlap between GPU block work and CPU
field processing, but does not transfer decoder history between speculative
segments.

Primary implementation references:

- [NVIDIA cuFFT documentation](https://docs.nvidia.com/cuda/cufft/)
- [NVIDIA NVRTC documentation](https://docs.nvidia.com/cuda/nvrtc/)
- [`cudarc` 0.19.9](https://github.com/coreylowman/cudarc/tree/0.19.9)

## Build and runtime requirements

```bash
cargo build --release --features cuda -p tape-decode-cli
```

The binary dynamically loads the installed CUDA driver, NVRTC, and cuFFT. The
tested host has CUDA Toolkit 13.3.1 and an `sm_75` RTX 2080. CUDA is not linked
into default builds, so ordinary CPU-only builds and tests retain their previous
requirements.

Supported invocation:

```bash
tape-decode decode \
  --profile NTSC_VHS \
  --frequency 28.636363M \
  --backend cuda \
  --cuda-device 0 \
  --mt-threads 12 \
  --mt-distance-size 108 \
  --luma-out decoded.tbc \
  --chroma-out decoded_chroma.tbc \
  --metadata-out decoded.tbc.json \
  capture.u8
```

Unsupported requests fail without a CPU fallback. Argument-level graph errors
are rejected before output files are opened; driver, library, and device errors
are reported when the decoder backend is constructed. In particular, CUDA v1
rejects custom profiles, non-NTSC systems, nonstandard sample rates, raw-TBC
export, notch/high-boost/sharpness/chroma-trap/nonlinear block-graph changes,
and disabled differential demodulation.

Multi-worker CUDA is accepted but remains experimental. Each segment starts
with fresh sync, level, and chroma state. A short overlap match is sufficient on
the clean fixtures, but it does not prove future state equivalence on unstable
material. Use the serial CUDA path when isolating GPU numerical behavior from
the multi-worker stitcher, and use CPU output for preservation work.

## Validation gates

The acceptance corpus compares CUDA against a single-thread CPU decode and
requires:

- identical field count, order, and identifiers;
- trimmed per-field MSRE below 64 for both luma and chroma;
- metadata within 0.11 absolute or `1e-9` relative tolerance;
- matching no-lock behavior without crashes;
- three warm end-to-end runs at least 1.5 times faster than the 12-thread CPU
  path; and
- bounded device memory below 8 GB with no per-block allocation growth.

Device tests cover unavailable ordinals, exact-architecture CUBIN cache keys,
partial and multi-block cuFFT plans, inverse-transform normalization, the
parallel envelope scan, burst reduction, and spike repair. The CLI tests cover
default CPU selection and fail-closed CUDA option validation.

## Measured result on Reverie (2026-08-25)

Reverie used an RTX 2080 (`sm_75`), NVIDIA driver 610.88, and CUDA Toolkit
13.3.1. Eight exact 12-second U8 fixtures were compared against the serial CPU
oracle. Field counts matched in every locking case. The no-lock fixture emitted
zero fields from both backends without a crash; its comparison command exits 1
because there is no field metadata to compare.

| Fixture source interval | Fields CPU/CUDA | Acceptance result |
| --- | ---: | --- |
| VHS-0001 `[2206,2218)` | 718/718 | pass |
| VHS-0002 `[120,132)` | 717/717 | pass |
| VHS-0004 `[140,152)` | 718/718 | pass |
| VHS-0006 `[120,132)` | 717/717 | fail: one luma field at 83.85 MSRE |
| VHS-0006 `[1400,1412)` | 0/0 | matching no-lock behavior |
| VHS-0006 `[7200,7212)` | 717/717 | fail: luma and chroma outliers |
| VHS-0008 `[132,144)` | 716/716 | fail: luma and chroma outliers |
| VHS-0008 `[7524,7536)` | 718/718 | fail: luma/chroma outliers and dropout metadata divergence |

The pre-existing 20-second VHS-0005 fixture likewise produced 1,196 fields on
both paths and passed metadata tolerance, but had 25 luma and 18 chroma field
outliers. Visual one-field luma checks found no obvious CPU/CUDA picture
divergence at ordinary motion and high-contrast frames; the visible difference
maps clustered around unstable sync and endpoint-collapse material. These are
technical, luma-only checks and are not watchable color-output validation.

Three warm end-to-end runs on the VHS-0005 fixture produced a 16.824-second
median for the 12-thread CPU path and a 98.638-second median for CUDA. The
CPU-to-CUDA ratio is 0.171, far below the required 1.5 speedup. `nvidia-smi dmon`
recorded 384 samples from 785 to 1,061 MiB of framebuffer use, with a flat
1,060-1,061 MiB plateau during CUDA work and no per-block growth. Memory
acceptance passes; quality and performance acceptance fail. No performance
improvement is claimed, and CUDA output remains non-authoritative.

Artifacts are retained on Reverie under
`D:\VHS-Decode\tape-decode-cuda-dev\corpus-final-v2` and
`D:\VHS-Decode\tape-decode-cuda-dev\benchmark-final`.

## Multi-stream optimization pass (2026-08-25)

A second pass connected CUDA to the bounded shared-`Tape` worker architecture.
Twelve workers use independent streams on the same primary context; the input
still has one forward-only producer. Nsight Compute identified two accidentally
serial kernels: spike detection and burst mean removal each scanned an entire
32,768-sample block from one thread. Parallel reductions, plus the device
envelope scan, reduced observed block-backend service from roughly 12.5 to
7.5-8.2 ms per field under load. Registered/pinned host-buffer experiments and
cached phase arrays did not improve end-to-end time and were not retained.

The final three-run VHS-0005 timing set used `--mt-threads 12` and
`--mt-distance-size 108` for both backends:

| Backend | Runs (seconds) | Median | CPU/CUDA ratio |
| --- | --- | ---: | ---: |
| CPU | 16.595, 16.289, 16.303 | 16.303 | - |
| CUDA | 13.990, 14.131, 13.640 | 13.990 | 1.165x |

This is substantially faster than the serial CUDA prototype, but it remains
below the required 1.5x threshold, so no performance-improvement claim is made.
Worker-count and distance sweeps found twelve workers and roughly 100-108 fields
per segment best for this 20-second fixture. Observed framebuffer use during
the scaling runs stayed between 819 and 4,028 MiB, below the 8 GB gate, with no
per-field allocation growth.

The final multi-worker eight-window sweep preserved the original three clean
passes, the known one-field/outlier failures, and matching zero-field no-lock
behavior. The endpoint-collapse fixture also exposed a multi-worker-only failure
cluster at fields 586-611: a speculative decoder matched at its stitch, then
diverged when later unstable material exercised its independently initialized
state. A serial CUDA rerun did not contain that extra cluster, confirming that
it is a state-stitching limitation rather than a new CUDA-kernel error. The
no-lock case completed safely but took 79.8 seconds because all speculative
workers scanned to EOF before proving there was no field.

Optimized artifacts are retained under
`D:\VHS-Decode\tape-decode-cuda-dev\corpus-opt-v3-mt`,
`D:\VHS-Decode\tape-decode-cuda-dev\quality-opt-v3-serial`, and
`D:\VHS-Decode\tape-decode-cuda-dev\benchmark-opt-v3.json`.

## Mothballed optimization state (2026-08-25)

A final profiling pass found that CPU sync-level estimation dominated the
remaining runtime. Moving its three zero-phase SOS filters to the existing GPU
affine-scan implementation reduced one CUDA12 run to about 14.54 seconds, but
the parallel scan's different floating-point association changed a rare
serration minimum. On VHS-0005 field 948 this altered line geometry and raised
trimmed luma MSRE to 761.033, despite matching field count, chroma, and metadata
tolerance. A serial CUDA run reproduced the same field failure, ruling out the
multi-worker stitcher. Running the CPU sync filters restored the field's CPU
luma statistics and line geometry.

Three additional experiments were evaluated:

- cross-worker GPU field batching averaged only 1.5-1.9 fields per launch and
  regressed wall time to 25-30 seconds;
- CUDA Graph capture could not capture the cuFFT path (`CUFFT_EXEC_FAILED`);
- field-wide GPU chroma analytic rotation increased contention and was slower,
  so it was removed.

The parked branch retains the useful low-risk work: independent worker streams,
fused luma/delayed-luma/burst output spectra with one larger batched inverse
cuFFT, reusable buffers/plans, GPU chroma final filtering, and selective CPU
oracle replay of spike-sensitive overlap-save blocks. GPU sync filtering is
disabled and the CPU reference remains authoritative for sync decisions.

The clean three-run baseline immediately before the final experiments was
16.592 seconds for CPU12 and 17.128 seconds for CUDA12 (0.969x), with peak VRAM
2,951 MiB. A later single, non-accepted run with fused outputs and selective
block replay reached 12.869 seconds, about 1.29x the CPU12 baseline, but still
failed field 948 while GPU sync was enabled. It was neither a three-run median
nor quality-valid, so it is not a performance claim. Telemetry during the deep
GPU pass averaged 77.2% utilization, peaked at 95%, and stayed below 3,529 MiB
VRAM; the RTX 2080 was active and clocked, but the workload remained dominated
by FFT/memory/launch overhead and CPU downstream work rather than power limits.

The experiment is stopped here. The 1.5x acceptance target was not achieved,
no upstream pull request is planned, and the branch remains a research artifact
on the fork. A final release-build check of the parked CPU-sync configuration
decoded the 20-second VHS-0005 fixture in 14.933 seconds and passed the preserved
CPU oracle for luma, chroma, and metadata at the documented acceptance
tolerances. This is a single verification run, not a speed claim.
