//! Experimental NTSC VHS CUDA block demodulator.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fs;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context as _, Result};
use cudarc::cufft::{sys as cufft_sys, CudaFft, FftDirection};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{result as nvrtc_result, sys as nvrtc_sys, Ptx};

use super::{iretohz, ColorSystem, DecoderSpec, VideoChannels, BLOCKCUT, BLOCKSIZE};
use crate::optimized::sosfiltfilt_f32;

const KERNEL_SOURCE: &str = include_str!("cuda_kernels.cu");
const NVRTC_OPTIONS_TAG: &str =
    "sm-specific;c++14;fmad=false;ftz=false;prec-div=true;prec-sqrt=true";
const HALF_BINS: usize = BLOCKSIZE / 2 + 1;

fn catch_cuda_initialization<T>(operation: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|_| {
        anyhow!(
            "failed to initialize CUDA backend: a required CUDA driver, NVRTC, or cuFFT shared library could not be loaded"
        )
    })?
}

struct Kernels {
    filter_real: CudaFunction,
    filter_complex: CudaFunction,
    analytic_expand: CudaFunction,
    demod_envelope: CudaFunction,
    demod_diffed: CudaFunction,
    mark_candidates: CudaFunction,
    repair_spikes: CudaFunction,
    pack_luma: CudaFunction,
    burst_means: CudaFunction,
    pack_burst: CudaFunction,
}

impl Kernels {
    fn load(context: &Arc<CudaContext>, image: Ptx) -> Result<Self> {
        let module = context
            .load_module(image)
            .context("failed to load CUDA demodulation CUBIN")?;
        Ok(Self {
            filter_real: module.load_function("filter_real")?,
            filter_complex: module.load_function("filter_complex")?,
            analytic_expand: module.load_function("analytic_expand")?,
            demod_envelope: module.load_function("demod_envelope")?,
            demod_diffed: module.load_function("demod_diffed")?,
            mark_candidates: module.load_function("mark_candidates")?,
            repair_spikes: module.load_function("repair_spikes")?,
            pack_luma: module.load_function("pack_luma")?,
            burst_means: module.load_function("burst_means")?,
            pack_burst: module.load_function("pack_burst")?,
        })
    }
}

struct CudaBatch {
    input: CudaSlice<f32>,
    raw_fft: CudaSlice<cufft_sys::float2>,
    rf_fft: CudaSlice<cufft_sys::float2>,
    analytic: CudaSlice<cufft_sys::float2>,
    hilbert: CudaSlice<cufft_sys::float2>,
    demod: CudaSlice<f32>,
    diffed_demod: CudaSlice<f32>,
    candidates: CudaSlice<u8>,
    raw_envelope: CudaSlice<f32>,
    demod_fft: CudaSlice<cufft_sys::float2>,
    video_fft: CudaSlice<cufft_sys::float2>,
    video05_fft: CudaSlice<cufft_sys::float2>,
    burst_fft: CudaSlice<cufft_sys::float2>,
    video: CudaSlice<f32>,
    video05: CudaSlice<f32>,
    burst: CudaSlice<f32>,
    burst_means: CudaSlice<f32>,
    packed_video: CudaSlice<f32>,
    packed_video05: CudaSlice<f32>,
    packed_burst: CudaSlice<f32>,
    r2c_raw: CudaFft,
    r2c_blocks: CudaFft,
    c2r: CudaFft,
    c2c_inverse: CudaFft,
    host_envelope: Vec<f32>,
    host_video: Vec<f32>,
    host_video05: Vec<f32>,
    host_burst: Vec<f32>,
}

impl CudaBatch {
    fn new(stream: &Arc<CudaStream>, blocks: usize, usable: usize) -> Result<Self> {
        let real_len = blocks * BLOCKSIZE;
        let input_len = (blocks - 1) * usable + BLOCKSIZE;
        let spectrum_len = blocks * HALF_BINS;
        let packed_len = blocks * usable;
        let n = [BLOCKSIZE as i32];
        let real_embed = [BLOCKSIZE as i32];
        let complex_embed = [HALF_BINS as i32];
        let r2c_raw = CudaFft::plan_many(
            &n,
            Some(&real_embed),
            1,
            usable as i32,
            Some(&complex_embed),
            1,
            HALF_BINS as i32,
            cufft_sys::cufftType::CUFFT_R2C,
            blocks as i32,
            Arc::clone(stream),
        )?;
        let r2c_blocks = CudaFft::plan_many(
            &n,
            Some(&real_embed),
            1,
            BLOCKSIZE as i32,
            Some(&complex_embed),
            1,
            HALF_BINS as i32,
            cufft_sys::cufftType::CUFFT_R2C,
            blocks as i32,
            Arc::clone(stream),
        )?;
        let c2r = CudaFft::plan_many(
            &n,
            Some(&complex_embed),
            1,
            HALF_BINS as i32,
            Some(&real_embed),
            1,
            BLOCKSIZE as i32,
            cufft_sys::cufftType::CUFFT_C2R,
            blocks as i32,
            Arc::clone(stream),
        )?;
        let c2c_inverse = CudaFft::plan_many(
            &n,
            Some(&real_embed),
            1,
            BLOCKSIZE as i32,
            Some(&real_embed),
            1,
            BLOCKSIZE as i32,
            cufft_sys::cufftType::CUFFT_C2C,
            blocks as i32,
            Arc::clone(stream),
        )?;
        Ok(Self {
            input: stream.alloc_zeros(input_len)?,
            raw_fft: stream.alloc_zeros(spectrum_len)?,
            rf_fft: stream.alloc_zeros(spectrum_len)?,
            analytic: stream.alloc_zeros(real_len)?,
            hilbert: stream.alloc_zeros(real_len)?,
            demod: stream.alloc_zeros(real_len)?,
            diffed_demod: stream.alloc_zeros(real_len)?,
            candidates: stream.alloc_zeros(real_len)?,
            raw_envelope: stream.alloc_zeros(real_len)?,
            demod_fft: stream.alloc_zeros(spectrum_len)?,
            video_fft: stream.alloc_zeros(spectrum_len)?,
            video05_fft: stream.alloc_zeros(spectrum_len)?,
            burst_fft: stream.alloc_zeros(spectrum_len)?,
            video: stream.alloc_zeros(real_len)?,
            video05: stream.alloc_zeros(real_len)?,
            burst: stream.alloc_zeros(real_len)?,
            burst_means: stream.alloc_zeros(blocks)?,
            packed_video: stream.alloc_zeros(packed_len)?,
            packed_video05: stream.alloc_zeros(packed_len)?,
            packed_burst: stream.alloc_zeros(packed_len)?,
            r2c_raw,
            r2c_blocks,
            c2r,
            c2c_inverse,
            host_envelope: vec![0.0; real_len],
            host_video: vec![0.0; packed_len],
            host_video05: vec![0.0; packed_len],
            host_burst: vec![0.0; packed_len],
        })
    }
}

pub(super) struct CudaBlockDecoder {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    rf_filter: CudaSlice<f32>,
    video_filter: CudaSlice<cufft_sys::float2>,
    video05_filter: CudaSlice<cufft_sys::float2>,
    burst_filter: CudaSlice<f32>,
    batches: HashMap<usize, CudaBatch>,
}

impl CudaBlockDecoder {
    pub(super) fn new(device_ordinal: usize, spec: &DecoderSpec) -> Result<Self> {
        validate_spec(spec)?;
        catch_cuda_initialization(|| Self::new_inner(device_ordinal, spec))
    }

    fn new_inner(device_ordinal: usize, spec: &DecoderSpec) -> Result<Self> {
        let context = CudaContext::new(device_ordinal)
            .with_context(|| format!("failed to initialize CUDA device {device_ordinal}"))?;
        let (major, minor) = context.compute_capability()?;
        if (major, minor) < (7, 5) {
            bail!("CUDA backend requires compute capability 7.5 or newer; device reports {major}.{minor}");
        }
        let name = context.name()?;
        let total_mem = context.total_mem()?;
        let cubin = load_or_compile_cubin(major, minor)?;
        let kernels = Kernels::load(&context, cubin)?;
        let stream = context.default_stream();
        let to_complex = |values: &[rustfft::num_complex::Complex32]| {
            values
                .iter()
                .map(|value| cufft_sys::float2 {
                    x: value.re,
                    y: value.im,
                })
                .collect::<Vec<_>>()
        };
        let rf_filter = stream.clone_htod(&spec.video_rf_filter[..HALF_BINS])?;
        let video_filter = stream.clone_htod(&to_complex(&spec.video_filter))?;
        let video05_filter = stream.clone_htod(&to_complex(&spec.video05_filter))?;
        let burst_filter = stream.clone_htod(&spec.chroma_burst_block_fft_gain)?;
        tracing::info!(
            device = device_ordinal,
            gpu = %name,
            compute_capability = %format!("{major}.{minor}"),
            memory_mib = total_mem / (1024 * 1024),
            kernel = "nvrtc-cubin",
            "initialized experimental CUDA block backend"
        );
        Ok(Self {
            stream,
            kernels,
            rf_filter,
            video_filter,
            video05_filter,
            burst_filter,
            batches: HashMap::new(),
        })
    }

    pub(super) fn decode_blocks(
        &mut self,
        rawdata: &[f32],
        blocks: usize,
        spec: &DecoderSpec,
        out: &mut VideoChannels,
    ) -> Result<()> {
        if blocks == 0 {
            return Ok(());
        }
        let usable = spec.usable_blocksize();
        let expected = (blocks - 1) * usable + BLOCKSIZE;
        if rawdata.len() != expected {
            bail!(
                "CUDA field batch has {} samples; expected {expected}",
                rawdata.len()
            );
        }
        if !self.batches.contains_key(&blocks) {
            self.batches
                .insert(blocks, CudaBatch::new(&self.stream, blocks, usable)?);
        }
        let batch = self
            .batches
            .get_mut(&blocks)
            .expect("CUDA batch initialized");
        let spectrum_total = (blocks * HALF_BINS) as i32;
        let real_total = (blocks * BLOCKSIZE) as i32;
        let packed_total = (blocks * usable) as i32;
        let block_count = blocks as i32;
        let n = BLOCKSIZE as i32;
        let half = HALF_BINS as i32;
        let usable_i32 = usable as i32;
        let cut = BLOCKCUT as i32;
        let video05_shift = DecoderSpec::VIDEO05_FILTER_OFFSET as i32;
        let inv_n = 1.0f32 / BLOCKSIZE as f32;
        let ire0 = spec.sys_ire0;
        let freq = spec.freq_hz() as f32;
        let spike_threshold = iretohz(ire0, spec.sys_hz_ire, 100.0) * 2.0 - ire0;

        self.stream.memcpy_htod(rawdata, &mut batch.input)?;
        batch.r2c_raw.exec_r2c(&batch.input, &mut batch.raw_fft)?;
        self.stream.memcpy_dtod(&batch.raw_fft, &mut batch.rf_fft)?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.filter_real)
                .arg(&mut batch.rf_fft)
                .arg(&self.rf_filter)
                .arg(&half)
                .arg(&spectrum_total)
                .launch(LaunchConfig::for_num_elems(spectrum_total as u32))?;
            self.stream
                .launch_builder(&self.kernels.analytic_expand)
                .arg(&batch.rf_fft)
                .arg(&mut batch.analytic)
                .arg(&n)
                .arg(&half)
                .arg(&real_total)
                .launch(LaunchConfig::for_num_elems(real_total as u32))?;
        }
        batch.c2c_inverse.exec_c2c(
            &mut batch.analytic,
            &mut batch.hilbert,
            FftDirection::Inverse,
        )?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.demod_envelope)
                .arg(&batch.hilbert)
                .arg(&mut batch.demod)
                .arg(&mut batch.raw_envelope)
                .arg(&n)
                .arg(&real_total)
                .arg(&freq)
                .arg(&ire0)
                .launch(LaunchConfig::for_num_elems(real_total as u32))?;
            self.stream
                .launch_builder(&self.kernels.demod_diffed)
                .arg(&batch.hilbert)
                .arg(&mut batch.diffed_demod)
                .arg(&n)
                .arg(&real_total)
                .arg(&freq)
                .arg(&ire0)
                .launch(LaunchConfig::for_num_elems(real_total as u32))?;
            self.stream
                .launch_builder(&self.kernels.mark_candidates)
                .arg(&batch.demod)
                .arg(&mut batch.candidates)
                .arg(&real_total)
                .arg(&spike_threshold)
                .launch(LaunchConfig::for_num_elems(real_total as u32))?;
            self.stream
                .launch_builder(&self.kernels.repair_spikes)
                .arg(&mut batch.demod)
                .arg(&batch.diffed_demod)
                .arg(&batch.candidates)
                .arg(&n)
                .arg(&block_count)
                .arg(&spike_threshold)
                .launch(LaunchConfig::for_num_elems(blocks as u32))?;
        }

        batch
            .r2c_blocks
            .exec_r2c(&batch.demod, &mut batch.demod_fft)?;
        self.stream
            .memcpy_dtod(&batch.demod_fft, &mut batch.video_fft)?;
        self.stream
            .memcpy_dtod(&batch.demod_fft, &mut batch.video05_fft)?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.filter_complex)
                .arg(&mut batch.video_fft)
                .arg(&self.video_filter)
                .arg(&half)
                .arg(&spectrum_total)
                .launch(LaunchConfig::for_num_elems(spectrum_total as u32))?;
            self.stream
                .launch_builder(&self.kernels.filter_complex)
                .arg(&mut batch.video05_fft)
                .arg(&self.video05_filter)
                .arg(&half)
                .arg(&spectrum_total)
                .launch(LaunchConfig::for_num_elems(spectrum_total as u32))?;
        }
        batch.c2r.exec_c2r(&mut batch.video_fft, &mut batch.video)?;
        batch
            .c2r
            .exec_c2r(&mut batch.video05_fft, &mut batch.video05)?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.pack_luma)
                .arg(&batch.video)
                .arg(&batch.video05)
                .arg(&mut batch.packed_video)
                .arg(&mut batch.packed_video05)
                .arg(&n)
                .arg(&usable_i32)
                .arg(&cut)
                .arg(&video05_shift)
                .arg(&packed_total)
                .arg(&inv_n)
                .arg(&ire0)
                .launch(LaunchConfig::for_num_elems(packed_total as u32))?;
        }

        if !spec.chroma_afc_enabled() {
            self.stream
                .memcpy_dtod(&batch.raw_fft, &mut batch.burst_fft)?;
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.filter_real)
                    .arg(&mut batch.burst_fft)
                    .arg(&self.burst_filter)
                    .arg(&half)
                    .arg(&spectrum_total)
                    .launch(LaunchConfig::for_num_elems(spectrum_total as u32))?;
            }
            batch.c2r.exec_c2r(&mut batch.burst_fft, &mut batch.burst)?;
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.burst_means)
                    .arg(&batch.burst)
                    .arg(&mut batch.burst_means)
                    .arg(&n)
                    .arg(&block_count)
                    .arg(&inv_n)
                    .launch(LaunchConfig::for_num_elems(blocks as u32))?;
                let chroma_shift = spec.chroma_offset() as i32;
                self.stream
                    .launch_builder(&self.kernels.pack_burst)
                    .arg(&batch.burst)
                    .arg(&batch.burst_means)
                    .arg(&mut batch.packed_burst)
                    .arg(&n)
                    .arg(&usable_i32)
                    .arg(&cut)
                    .arg(&chroma_shift)
                    .arg(&packed_total)
                    .arg(&inv_n)
                    .launch(LaunchConfig::for_num_elems(packed_total as u32))?;
            }
        }

        self.stream
            .memcpy_dtoh(&batch.raw_envelope, &mut batch.host_envelope)?;
        self.stream
            .memcpy_dtoh(&batch.packed_video, &mut batch.host_video)?;
        self.stream
            .memcpy_dtoh(&batch.packed_video05, &mut batch.host_video05)?;
        if !spec.chroma_afc_enabled() {
            self.stream
                .memcpy_dtoh(&batch.packed_burst, &mut batch.host_burst)?;
        }
        self.stream.synchronize()?;

        out.demod.extend_from_slice(&batch.host_video);
        out.demod_05.extend_from_slice(&batch.host_video05);
        for block in 0..blocks {
            let start = block * BLOCKSIZE;
            let env = sosfiltfilt_f32(
                &spec.video_env_post_filter,
                &batch.host_envelope[start..start + BLOCKSIZE],
            );
            out.envelope
                .extend_from_slice(&env[BLOCKCUT..BLOCKSIZE - BLOCKCUT]);
            if spec.chroma_afc_enabled() {
                let raw_start = block * usable + BLOCKCUT;
                out.demod_burst
                    .extend_from_slice(&rawdata[raw_start..raw_start + usable]);
            }
        }
        if !spec.chroma_afc_enabled() {
            out.demod_burst.extend_from_slice(&batch.host_burst);
        }
        Ok(())
    }
}

fn validate_spec(spec: &DecoderSpec) -> Result<()> {
    if spec.color_system != ColorSystem::Ntsc || (spec.freq_hz() - 28_636_363.0).abs() > 1.0 {
        bail!("CUDA v1 supports only NTSC VHS at 28,636,363 Hz");
    }
    if spec.video_notch_filter.is_some()
        || spec.video_high_boost_value.is_some()
        || spec.video_eq_fft_gain.is_some()
        || spec.video_chroma_trap.is_some()
        || spec.video_nldeemp_enabled
        || spec.video_subdeemp_enabled
        || spec.video_fsc_notch.is_some()
        || spec.rf_export_raw_tbc
    {
        bail!("CUDA v1 does not support modified/custom NTSC VHS block filters");
    }
    Ok(())
}

fn cache_path(major: i32, minor: i32) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    KERNEL_SOURCE.hash(&mut hasher);
    major.hash(&mut hasher);
    minor.hash(&mut hasher);
    env!("CARGO_PKG_VERSION").hash(&mut hasher);
    NVRTC_OPTIONS_TAG.hash(&mut hasher);
    let root = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("tape-decode")
        .join("cuda-cache");
    root.join(format!(
        "demod-sm_{major}{minor}-{:016x}.cubin",
        hasher.finish()
    ))
}

fn load_or_compile_cubin(major: i32, minor: i32) -> Result<Ptx> {
    let path = cache_path(major, minor);
    if let Ok(bytes) = fs::read(&path) {
        tracing::debug!(path = %path.display(), "using cached CUDA CUBIN");
        return Ok(Ptx::from_binary(bytes));
    }

    let source = CString::new(KERNEL_SOURCE).expect("CUDA kernel source contains no NUL");
    let name = CString::new("tape_decode_demod.cu").unwrap();
    let program = nvrtc_result::create_program(&source, Some(&name))?;
    let architecture = format!("--gpu-architecture=sm_{major}{minor}");
    let options = [
        architecture.as_str(),
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    ];
    if let Err(error) = unsafe { nvrtc_result::compile_program(program, &options) } {
        let log = unsafe { nvrtc_result::get_program_log(program) }
            .ok()
            .and_then(|bytes| {
                unsafe { CStr::from_ptr(bytes.as_ptr()) }
                    .to_str()
                    .ok()
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "NVRTC returned no compile log".to_string());
        let _ = unsafe { nvrtc_result::destroy_program(program) };
        bail!("NVRTC CUBIN compilation failed: {error}: {log}");
    }
    let mut size = 0usize;
    unsafe { nvrtc_sys::nvrtcGetCUBINSize(program, &mut size) }.result()?;
    let mut cubin = vec![0u8; size];
    unsafe { nvrtc_sys::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()) }.result()?;
    unsafe { nvrtc_result::destroy_program(program) }?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create CUDA cache {}", parent.display()))?;
    }
    fs::write(&path, &cubin)
        .with_context(|| format!("failed to cache CUDA CUBIN at {}", path.display()))?;
    tracing::info!(path = %path.display(), bytes = cubin.len(), "compiled device-specific CUDA CUBIN");
    Ok(Ptx::from_binary(cubin))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubin_cache_key_changes_with_architecture() {
        assert_ne!(cache_path(7, 5), cache_path(8, 6));
    }

    #[test]
    fn invalid_device_is_reported() {
        let result =
            catch_cuda_initialization(|| CudaContext::new(usize::MAX).map_err(anyhow::Error::from));
        assert!(result.is_err());
    }

    #[test]
    fn cufft_round_trip_and_partial_batches() -> Result<()> {
        let Ok(context) =
            catch_cuda_initialization(|| CudaContext::new(0).map_err(anyhow::Error::from))
        else {
            // CUDA-feature builds remain testable on hosts without a CUDA device.
            return Ok(());
        };
        let stream = context.default_stream();

        // One block exercises the partial/final-batch shape. Three blocks
        // exercises the batched plan and its contiguous block stride.
        let Ok(_partial) =
            catch_cuda_initialization(|| CudaBatch::new(&stream, 1, BLOCKSIZE - 2 * BLOCKCUT))
        else {
            return Ok(());
        };
        let Ok(mut batch) =
            catch_cuda_initialization(|| CudaBatch::new(&stream, 3, BLOCKSIZE - 2 * BLOCKCUT))
        else {
            return Ok(());
        };
        let input = (0..3 * BLOCKSIZE)
            .map(|index| {
                let phase = index as f32 * 0.013_579;
                phase.sin() + 0.25 * (phase * 0.37).cos()
            })
            .collect::<Vec<_>>();
        stream.memcpy_htod(&input, &mut batch.demod)?;
        batch
            .r2c_blocks
            .exec_r2c(&batch.demod, &mut batch.demod_fft)?;
        batch.c2r.exec_c2r(&mut batch.demod_fft, &mut batch.video)?;
        let mut output = vec![0.0f32; input.len()];
        stream.memcpy_dtoh(&batch.video, &mut output)?;
        stream.synchronize()?;

        let inv_n = 1.0 / BLOCKSIZE as f32;
        let max_error = input
            .iter()
            .zip(output)
            .map(|(&expected, actual)| (expected - actual * inv_n).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 2.0e-5, "cuFFT round-trip error {max_error}");
        Ok(())
    }
}
