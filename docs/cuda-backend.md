# Experimental CUDA backend

## Status

The CUDA backend is explicit, experimental, and non-authoritative. CPU is the
default and the quality oracle. A CUDA request fails rather than silently
falling back when the feature, driver, device, format, profile, or option graph
is unsupported.

The implementation is usable end to end on an NVIDIA RTX 2080, but the initial
bounded acceptance run does not yet satisfy the project's cross-backend MSRE or
performance gates. Do not substitute CUDA output for a preservation decode.

## Design

The backend applies the usual high-throughput GPU DSP structure to the existing
32,768-sample overlap-save graph:

- a complete field's overlapping blocks are uploaded as one batch;
- cuFFT `planMany` handles batched real/complex transforms;
- device kernels apply RF gains, construct the one-sided analytic spectrum,
  extract the envelope, perform phase and differential demodulation, repair
  spikes, apply luma filters, and extract the burst channel;
- packed usable block interiors are copied back to the existing CPU field,
  sync, chroma, scaling, dropout, and metadata pipeline;
- device buffers and cuFFT plans are retained by batch shape and reused; and
- NVRTC compiles a device-specific CUBIN with contraction disabled for closer
  agreement with the scalar Rust phase path. The CUBIN cache key includes the
  kernel source, crate version, architecture, and compiler options.

This is a deliberately bounded first cut. It does not change the sequential
input producer, duplicate the source reader, or seek/reopen compressed input.

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
disabled differential demodulation, and multi-worker decoding.

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
partial and multi-block cuFFT plans, and inverse-transform normalization. The
CLI tests cover default CPU selection and fail-closed CUDA option validation.

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
