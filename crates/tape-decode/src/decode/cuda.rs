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
use sci_rs::signal::filter::design::Sos;

use super::{
    decode_video_block, iretohz, ColorSystem, DecoderSpec, VideoChannels, BLOCKCUT, BLOCKSIZE,
};

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

#[allow(dead_code)] // A few reference primitives are retained for CUDA tests.
struct Kernels {
    filter_real: CudaFunction,
    filter_complex: CudaFunction,
    prepare_output_spectra: CudaFunction,
    analytic_expand: CudaFunction,
    demod_envelope: CudaFunction,
    filter_envelope_scan: CudaFunction,
    demod_diffed: CudaFunction,
    mark_candidates: CudaFunction,
    repair_spikes: CudaFunction,
    pack_luma: CudaFunction,
    pack_luma_combined: CudaFunction,
    burst_means: CudaFunction,
    burst_means_combined: CudaFunction,
    pack_burst: CudaFunction,
    pack_burst_combined: CudaFunction,
    sos_section_scan: CudaFunction,
    capture_last: CudaFunction,
}

impl Kernels {
    fn load(context: &Arc<CudaContext>, image: Ptx) -> Result<Self> {
        let module = context
            .load_module(image)
            .context("failed to load CUDA demodulation CUBIN")?;
        Ok(Self {
            filter_real: module.load_function("filter_real")?,
            filter_complex: module.load_function("filter_complex")?,
            prepare_output_spectra: module.load_function("prepare_output_spectra")?,
            analytic_expand: module.load_function("analytic_expand")?,
            demod_envelope: module.load_function("demod_envelope")?,
            filter_envelope_scan: module.load_function("filter_envelope_scan")?,
            demod_diffed: module.load_function("demod_diffed")?,
            mark_candidates: module.load_function("mark_candidates")?,
            repair_spikes: module.load_function("repair_spikes")?,
            pack_luma: module.load_function("pack_luma")?,
            pack_luma_combined: module.load_function("pack_luma_combined")?,
            burst_means: module.load_function("burst_means")?,
            burst_means_combined: module.load_function("burst_means_combined")?,
            pack_burst: module.load_function("pack_burst")?,
            pack_burst_combined: module.load_function("pack_burst_combined")?,
            sos_section_scan: module.load_function("sos_section_scan")?,
            capture_last: module.load_function("capture_last")?,
        })
    }
}

#[derive(Clone, Copy)]
struct GpuBiquad {
    b0: f32,
    neg_a1: f32,
    neg_a2: f32,
    bff1: f32,
    bff2: f32,
    zi0: f32,
    zi1: f32,
}

struct CudaChroma {
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    initial: CudaSlice<f32>,
    sections: Vec<GpuBiquad>,
    edge: usize,
    host_extended: Vec<f32>,
    host_output: Vec<f32>,
}

impl CudaChroma {
    fn new(stream: &Arc<CudaStream>, len: usize, filter: &[Sos<f32>]) -> Result<Self> {
        if filter.is_empty() {
            bail!("CUDA chroma filter has no SOS sections");
        }
        let edge = sosfiltfilt_edge(filter);
        if len <= edge {
            bail!(
                "CUDA chroma field has {len} samples; zero-phase filter requires more than {edge}"
            );
        }
        let sections = gpu_biquads(filter);
        let extended_len = len + 2 * edge;
        Ok(Self {
            a: stream.alloc_zeros(extended_len)?,
            b: stream.alloc_zeros(extended_len)?,
            initial: stream.alloc_zeros(1)?,
            sections,
            edge,
            host_extended: Vec::with_capacity(extended_len),
            host_output: vec![0.0; extended_len],
        })
    }

    fn filter(
        &mut self,
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        chroma: &mut [f32],
    ) -> Result<()> {
        let left = chroma[0];
        let right = chroma[chroma.len() - 1];
        self.host_extended.clear();
        self.host_extended.extend(
            (1..=self.edge)
                .rev()
                .map(|index| 2.0f32.mul_add(left, -chroma[index])),
        );
        self.host_extended.extend_from_slice(chroma);
        self.host_extended.extend(
            (1..=self.edge).map(|index| 2.0f32.mul_add(right, -chroma[chroma.len() - 1 - index])),
        );
        debug_assert_eq!(self.host_extended.len(), self.host_output.len());

        stream.memcpy_htod(&self.host_extended, &mut self.a)?;
        stream.memcpy_htod(&self.host_extended[..1], &mut self.initial)?;
        let len = self.host_extended.len() as i32;
        let chunk = self.host_extended.len().div_ceil(1024) as i32;
        let launch = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut source_is_a = true;
        for &section in &self.sections {
            if source_is_a {
                launch_sos_section(
                    stream,
                    &kernels.sos_section_scan,
                    &self.a,
                    &mut self.b,
                    &self.initial,
                    len,
                    chunk,
                    false,
                    section,
                    launch,
                )?;
            } else {
                launch_sos_section(
                    stream,
                    &kernels.sos_section_scan,
                    &self.b,
                    &mut self.a,
                    &self.initial,
                    len,
                    chunk,
                    false,
                    section,
                    launch,
                )?;
            }
            source_is_a = !source_is_a;
        }

        let forward = if source_is_a { &self.a } else { &self.b };
        unsafe {
            stream
                .launch_builder(&kernels.capture_last)
                .arg(forward)
                .arg(&mut self.initial)
                .arg(&len)
                .launch(LaunchConfig::for_num_elems(1))?;
        }

        for &section in &self.sections {
            if source_is_a {
                launch_sos_section(
                    stream,
                    &kernels.sos_section_scan,
                    &self.a,
                    &mut self.b,
                    &self.initial,
                    len,
                    chunk,
                    true,
                    section,
                    launch,
                )?;
            } else {
                launch_sos_section(
                    stream,
                    &kernels.sos_section_scan,
                    &self.b,
                    &mut self.a,
                    &self.initial,
                    len,
                    chunk,
                    true,
                    section,
                    launch,
                )?;
            }
            source_is_a = !source_is_a;
        }
        debug_assert!(source_is_a, "two equal SOS passes must end in buffer a");
        stream.memcpy_dtoh(&self.a, &mut self.host_output)?;
        stream.synchronize()?;
        chroma.copy_from_slice(&self.host_output[self.edge..self.edge + chroma.len()]);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_sos_section(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    initial: &CudaSlice<f32>,
    len: i32,
    chunk: i32,
    reverse: bool,
    section: GpuBiquad,
    launch: LaunchConfig,
) -> Result<()> {
    let reverse = i32::from(reverse);
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(input)
            .arg(output)
            .arg(initial)
            .arg(&len)
            .arg(&chunk)
            .arg(&reverse)
            .arg(&section.b0)
            .arg(&section.neg_a1)
            .arg(&section.neg_a2)
            .arg(&section.bff1)
            .arg(&section.bff2)
            .arg(&section.zi0)
            .arg(&section.zi1)
            .launch(launch)?;
    }
    Ok(())
}

fn sosfiltfilt_edge(filter: &[Sos<f32>]) -> usize {
    let bzeros = filter.iter().filter(|section| section.b[2] == 0.0).count();
    let azeros = filter.iter().filter(|section| section.a[2] == 0.0).count();
    ((2 * filter.len() + 1) - bzeros.min(azeros)) * 3
}

fn gpu_biquads(filter: &[Sos<f32>]) -> Vec<GpuBiquad> {
    let mut scale = 1.0f64;
    filter
        .iter()
        .map(|section| {
            let a = section.a.map(f64::from);
            let b = section.b.map(f64::from);
            let a_start = a
                .iter()
                .position(|&value| value != 0.0)
                .expect("SOS denominator must contain a nonzero coefficient");
            let a0 = a[a_start];
            let normalized_b = [b[0] / a0, b[1] / a0, b[2] / a0];
            let mut normalized_a = [1.0, 0.0, 0.0];
            for (dst, &src) in normalized_a[1..].iter_mut().zip(&a[a_start + 1..]) {
                *dst = src / a0;
            }
            let b1_term = normalized_b[1] - normalized_a[1] * normalized_b[0];
            let zi0 = (b1_term + (normalized_b[2] - normalized_a[2] * normalized_b[0]))
                / normalized_a.iter().sum::<f64>();
            let zi1 = (1.0 + normalized_a[1]) * zi0 - b1_term;
            let result = GpuBiquad {
                b0: section.b[0],
                neg_a1: -section.a[1],
                neg_a2: -section.a[2],
                bff1: section.b[1] - section.a[1] * section.b[0],
                bff2: section.b[2] - section.a[2] * section.b[0],
                zi0: (scale * zi0) as f32,
                zi1: (scale * zi1) as f32,
            };
            scale *= b.iter().sum::<f64>() / a.iter().sum::<f64>();
            result
        })
        .collect()
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
    spike_blocks: CudaSlice<u8>,
    raw_envelope: CudaSlice<f32>,
    envelope_work: CudaSlice<f32>,
    packed_envelope: CudaSlice<f32>,
    demod_fft: CudaSlice<cufft_sys::float2>,
    output_fft: CudaSlice<cufft_sys::float2>,
    output_real: CudaSlice<f32>,
    burst_means: CudaSlice<f32>,
    packed_video: CudaSlice<f32>,
    packed_video05: CudaSlice<f32>,
    packed_burst: CudaSlice<f32>,
    r2c_raw: CudaFft,
    r2c_blocks: CudaFft,
    c2r_outputs: CudaFft,
    c2c_inverse: CudaFft,
    host_envelope: Vec<f32>,
    host_video: Vec<f32>,
    host_video05: Vec<f32>,
    host_burst: Vec<f32>,
    host_spike_blocks: Vec<u8>,
}

impl CudaBatch {
    fn new(stream: &Arc<CudaStream>, blocks: usize, usable: usize) -> Result<Self> {
        let input_len = (blocks - 1) * usable + BLOCKSIZE;
        let real_len = blocks * BLOCKSIZE;
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
        let c2r_outputs = CudaFft::plan_many(
            &n,
            Some(&complex_embed),
            1,
            HALF_BINS as i32,
            Some(&real_embed),
            1,
            BLOCKSIZE as i32,
            cufft_sys::cufftType::CUFFT_C2R,
            (3 * blocks) as i32,
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
            spike_blocks: stream.alloc_zeros(blocks)?,
            raw_envelope: stream.alloc_zeros(real_len)?,
            envelope_work: stream.alloc_zeros(blocks * (BLOCKSIZE + 6))?,
            packed_envelope: stream.alloc_zeros(packed_len)?,
            demod_fft: stream.alloc_zeros(spectrum_len)?,
            output_fft: stream.alloc_zeros(3 * spectrum_len)?,
            output_real: stream.alloc_zeros(3 * real_len)?,
            burst_means: stream.alloc_zeros(blocks)?,
            packed_video: stream.alloc_zeros(packed_len)?,
            packed_video05: stream.alloc_zeros(packed_len)?,
            packed_burst: stream.alloc_zeros(packed_len)?,
            r2c_raw,
            r2c_blocks,
            c2r_outputs,
            c2c_inverse,
            host_envelope: vec![0.0; packed_len],
            host_video: vec![0.0; packed_len],
            host_video05: vec![0.0; packed_len],
            host_burst: vec![0.0; packed_len],
            host_spike_blocks: vec![0; blocks],
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
    chroma_fields: HashMap<usize, CudaChroma>,
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
        // Every multithreaded decoder retains the same CUDA primary context.
        // The legacy default stream is shared by all of those context handles,
        // which silently serializes otherwise independent field pipelines.
        // Give each decoder a non-blocking stream so copies, FFTs, and kernels
        // from different workers can overlap on the device.
        let stream = context
            .new_stream()
            .context("failed to create CUDA worker stream")?;
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
            chroma_fields: HashMap::new(),
        })
    }

    pub(super) fn process_chroma_spectrum(
        &mut self,
        chroma: &mut [f32],
        spec: &DecoderSpec,
        allow_gpu: bool,
    ) -> Result<bool> {
        if !allow_gpu {
            return Ok(false);
        }
        if chroma.is_empty() {
            return Ok(true);
        }
        let len = chroma.len();
        if !self.chroma_fields.contains_key(&len) {
            self.chroma_fields.insert(
                len,
                CudaChroma::new(&self.stream, len, &spec.chroma_filter_final)?,
            );
        }
        let field = self
            .chroma_fields
            .get_mut(&len)
            .expect("CUDA chroma field initialized");
        field.filter(&self.stream, &self.kernels, chroma)?;
        Ok(true)
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
        let env_section = spec
            .video_env_post_filter
            .first()
            .context("missing CUDA envelope post-filter")?;
        if spec.video_env_post_filter.len() != 1
            || env_section.b[2] != 0.0
            || env_section.a[2] != 0.0
        {
            bail!("CUDA envelope post-filter requires one first-order SOS section");
        }
        let env_b0 = env_section.b[0];
        let env_recurrence = -env_section.a[1];
        let env_feed_forward = env_section.b[1] - env_section.a[1] * env_section.b[0];
        let env_zi0 = {
            let a0 = f64::from(env_section.a[0]);
            let b0 = f64::from(env_section.b[0]) / a0;
            let b1 = f64::from(env_section.b[1]) / a0;
            let b2 = f64::from(env_section.b[2]) / a0;
            let a1 = f64::from(env_section.a[1]) / a0;
            let a2 = f64::from(env_section.a[2]) / a0;
            let b1_term = b1 - a1 * b0;
            ((b1_term + (b2 - a2 * b0)) / (1.0 + a1 + a2)) as f32
        };
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
                .launch_builder(&self.kernels.filter_envelope_scan)
                .arg(&batch.raw_envelope)
                .arg(&mut batch.envelope_work)
                .arg(&mut batch.packed_envelope)
                .arg(&n)
                .arg(&usable_i32)
                .arg(&cut)
                .arg(&block_count)
                .arg(&env_b0)
                .arg(&env_recurrence)
                .arg(&env_feed_forward)
                .arg(&env_zi0)
                .launch(LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                })?;
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
                .arg(&mut batch.spike_blocks)
                .arg(&n)
                .arg(&block_count)
                .arg(&spike_threshold)
                .launch(LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        batch
            .r2c_blocks
            .exec_r2c(&batch.demod, &mut batch.demod_fft)?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.prepare_output_spectra)
                .arg(&batch.demod_fft)
                .arg(&batch.raw_fft)
                .arg(&self.video_filter)
                .arg(&self.video05_filter)
                .arg(&self.burst_filter)
                .arg(&mut batch.output_fft)
                .arg(&half)
                .arg(&spectrum_total)
                .launch(LaunchConfig::for_num_elems(3 * spectrum_total as u32))?;
        }
        batch
            .c2r_outputs
            .exec_c2r(&mut batch.output_fft, &mut batch.output_real)?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.pack_luma_combined)
                .arg(&batch.output_real)
                .arg(&mut batch.packed_video)
                .arg(&mut batch.packed_video05)
                .arg(&real_total)
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
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.burst_means_combined)
                    .arg(&batch.output_real)
                    .arg(&mut batch.burst_means)
                    .arg(&real_total)
                    .arg(&n)
                    .arg(&block_count)
                    .arg(&inv_n)
                    .launch(LaunchConfig {
                        grid_dim: (blocks as u32, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
                let chroma_shift = spec.chroma_offset() as i32;
                self.stream
                    .launch_builder(&self.kernels.pack_burst_combined)
                    .arg(&batch.output_real)
                    .arg(&batch.burst_means)
                    .arg(&mut batch.packed_burst)
                    .arg(&real_total)
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
            .memcpy_dtoh(&batch.packed_envelope, &mut batch.host_envelope)?;
        self.stream
            .memcpy_dtoh(&batch.packed_video, &mut batch.host_video)?;
        self.stream
            .memcpy_dtoh(&batch.packed_video05, &mut batch.host_video05)?;
        if !spec.chroma_afc_enabled() {
            self.stream
                .memcpy_dtoh(&batch.packed_burst, &mut batch.host_burst)?;
        }
        self.stream
            .memcpy_dtoh(&batch.spike_blocks, &mut batch.host_spike_blocks)?;
        self.stream.synchronize()?;

        // Phase unwrap is ill-conditioned exactly where the analytic RF signal
        // approaches zero. Replay only the affected independent overlap-save
        // blocks through the CPU oracle, then replace their complete usable
        // spans. This keeps the decision field-local and avoids rerunning the
        // other 22-23 healthy blocks merely because one block crossed the spike
        // threshold.
        let mut replayed_blocks = 0usize;
        for (block, &spike_flag) in batch.host_spike_blocks.iter().enumerate() {
            if spike_flag == 0 {
                continue;
            }
            let mut cpu = VideoChannels {
                demod: Vec::with_capacity(usable),
                demod_05: Vec::with_capacity(usable),
                demod_burst: Vec::with_capacity(usable),
                envelope: Vec::with_capacity(usable),
                oracle_replayed: false,
            };
            let raw_start = block * usable;
            decode_video_block(&rawdata[raw_start..raw_start + BLOCKSIZE], spec, &mut cpu)?;
            let packed = raw_start..raw_start + usable;
            batch.host_video[packed.clone()].copy_from_slice(&cpu.demod);
            batch.host_video05[packed.clone()].copy_from_slice(&cpu.demod_05);
            batch.host_envelope[packed.clone()].copy_from_slice(&cpu.envelope);
            batch.host_burst[packed].copy_from_slice(&cpu.demod_burst);
            replayed_blocks += 1;
        }
        if replayed_blocks != 0 {
            tracing::debug!(replayed_blocks, blocks, "replayed CUDA spike blocks on CPU");
        }

        out.demod.extend_from_slice(&batch.host_video);
        out.demod_05.extend_from_slice(&batch.host_video05);
        out.envelope.extend_from_slice(&batch.host_envelope);
        if spec.chroma_afc_enabled() {
            for block in 0..blocks {
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
    use crate::optimized::{sosfiltfilt_f32, sum_algebraic};
    use sci_rs::signal::filter::design::Sos;

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
    fn parallel_sos_filtfilt_matches_cpu_for_partial_chunks() -> Result<()> {
        let Ok(context) =
            catch_cuda_initialization(|| CudaContext::new(0).map_err(anyhow::Error::from))
        else {
            return Ok(());
        };
        let (major, minor) = context.compute_capability()?;
        let Ok(image) = catch_cuda_initialization(|| load_or_compile_cubin(major, minor)) else {
            return Ok(());
        };
        let kernels = Kernels::load(&context, image)?;
        let stream = context.default_stream();
        let filter = [
            Sos::new(
                [0.067_455_27f32, 0.134_910_54, 0.067_455_27],
                [1.0, -1.142_980_5, 0.412_801_6],
            ),
            Sos::new(
                [0.206_572_09f32, 0.413_144_17, 0.206_572_09],
                [1.0, -0.369_527_37, 0.195_815_71],
            ),
        ];
        // Deliberately not divisible by the 1,024-lane scan width.
        let mut actual = (0..239_317)
            .map(|index| {
                let x = index as f32;
                0.72 * (x * 0.002_71).sin()
                    + 0.21 * (x * 0.017_3).cos()
                    + (index % 29) as f32 * 0.000_2
            })
            .collect::<Vec<_>>();
        let expected = sosfiltfilt_f32(&filter, &actual);
        let mut field = CudaChroma::new(&stream, actual.len(), &filter)?;
        field.filter(&stream, &kernels, &mut actual)?;
        let max_error = expected
            .iter()
            .zip(&actual)
            .map(|(&cpu, &gpu)| (cpu - gpu).abs())
            .fold(0.0f32, f32::max);
        let msre = expected
            .iter()
            .zip(&actual)
            .map(|(&cpu, &gpu)| {
                let denominator = cpu.abs().max(1.0e-3);
                let relative = (cpu - gpu) / denominator;
                relative * relative
            })
            .sum::<f32>()
            / expected.len() as f32;
        assert!(max_error < 2.0e-3, "parallel SOS max error {max_error}");
        assert!(msre < 2.0e-7, "parallel SOS MSRE {msre}");
        Ok(())
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
        let n = [BLOCKSIZE as i32];
        let real_embed = [BLOCKSIZE as i32];
        let complex_embed = [HALF_BINS as i32];
        let mut c2r = CudaFft::plan_many(
            &n,
            Some(&complex_embed),
            1,
            HALF_BINS as i32,
            Some(&real_embed),
            1,
            BLOCKSIZE as i32,
            cufft_sys::cufftType::CUFFT_C2R,
            3,
            Arc::clone(&stream),
        )?;
        let mut output_device: CudaSlice<f32> = stream.alloc_zeros(input.len())?;
        c2r.exec_c2r(&mut batch.demod_fft, &mut output_device)?;
        let mut output = vec![0.0f32; input.len()];
        stream.memcpy_dtoh(&output_device, &mut output)?;
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

    #[test]
    fn envelope_reductions_and_spike_repair_match_cpu_primitives() -> Result<()> {
        let Ok(context) =
            catch_cuda_initialization(|| CudaContext::new(0).map_err(anyhow::Error::from))
        else {
            return Ok(());
        };
        let (major, minor) = context.compute_capability()?;
        let Ok(image) = catch_cuda_initialization(|| load_or_compile_cubin(major, minor)) else {
            return Ok(());
        };
        let kernels = Kernels::load(&context, image)?;
        let stream = context.default_stream();
        let blocks = 3usize;
        let usable = BLOCKSIZE - 2 * BLOCKCUT;
        let real_len = blocks * BLOCKSIZE;
        let packed_len = blocks * usable;
        let n = BLOCKSIZE as i32;
        let usable_i32 = usable as i32;
        let cut = BLOCKCUT as i32;
        let block_count = blocks as i32;

        let section = Sos::new([0.12f32, 0.12, 0.0], [1.0, -0.76, 0.0]);
        let raw = (0..real_len)
            .map(|index| {
                let x = index as f32;
                0.65 * (x * 0.003_17).sin() + 0.2 * (x * 0.021_3).cos()
            })
            .collect::<Vec<_>>();
        let mut device_raw = stream.clone_htod(&raw)?;
        let mut work: CudaSlice<f32> = stream.alloc_zeros(blocks * (BLOCKSIZE + 6))?;
        let mut packed: CudaSlice<f32> = stream.alloc_zeros(packed_len)?;
        let b0 = section.b[0];
        let recurrence = -section.a[1];
        let feed_forward = section.b[1] - section.a[1] * section.b[0];
        let zi0 = ((section.b[1] - section.a[1] * section.b[0])
            + (section.b[2] - section.a[2] * section.b[0]))
            / (1.0 + section.a[1] + section.a[2]);
        unsafe {
            stream
                .launch_builder(&kernels.filter_envelope_scan)
                .arg(&mut device_raw)
                .arg(&mut work)
                .arg(&mut packed)
                .arg(&n)
                .arg(&usable_i32)
                .arg(&cut)
                .arg(&block_count)
                .arg(&b0)
                .arg(&recurrence)
                .arg(&feed_forward)
                .arg(&zi0)
                .launch(LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let mut actual_envelope = vec![0.0; packed_len];
        stream.memcpy_dtoh(&packed, &mut actual_envelope)?;
        stream.synchronize()?;
        let mut expected_envelope = Vec::with_capacity(packed_len);
        for block in raw.chunks_exact(BLOCKSIZE) {
            let filtered = sosfiltfilt_f32(&[section], block);
            expected_envelope.extend_from_slice(&filtered[BLOCKCUT..BLOCKSIZE - BLOCKCUT]);
        }
        let envelope_error = expected_envelope
            .iter()
            .zip(&actual_envelope)
            .map(|(&expected, &actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(
            envelope_error < 2.0e-4,
            "parallel envelope scan error {envelope_error}"
        );

        let burst = (0..real_len)
            .map(|index| {
                let x = index as f32;
                1200.0 * (x * 0.007_31).sin() + (index % 37) as f32
            })
            .collect::<Vec<_>>();
        let device_burst = stream.clone_htod(&burst)?;
        let mut device_means: CudaSlice<f32> = stream.alloc_zeros(blocks)?;
        let inv_n = 1.0 / BLOCKSIZE as f32;
        unsafe {
            stream
                .launch_builder(&kernels.burst_means)
                .arg(&device_burst)
                .arg(&mut device_means)
                .arg(&n)
                .arg(&block_count)
                .arg(&inv_n)
                .launch(LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let mut actual_means = vec![0.0; blocks];
        stream.memcpy_dtoh(&device_means, &mut actual_means)?;
        stream.synchronize()?;
        for (block, &actual) in burst.chunks_exact(BLOCKSIZE).zip(&actual_means) {
            let normalized = block.iter().map(|&value| value * inv_n).collect::<Vec<_>>();
            let expected = sum_algebraic(&normalized) * inv_n;
            assert!(
                (expected - actual).abs() < 2.0e-7,
                "parallel burst mean: expected {expected}, got {actual}"
            );
        }

        let threshold = 100.0f32;
        let spike_index = BLOCKSIZE + 100;
        let mut demod = vec![1.0f32; real_len];
        demod[spike_index] = 200.0;
        let diffed = vec![0.0f32; real_len];
        let mut candidates = vec![0u8; real_len];
        candidates[spike_index] = 1;
        let mut device_demod = stream.clone_htod(&demod)?;
        let device_diffed = stream.clone_htod(&diffed)?;
        let device_candidates = stream.clone_htod(&candidates)?;
        let mut device_block_flags: CudaSlice<u8> = stream.alloc_zeros(blocks)?;
        unsafe {
            stream
                .launch_builder(&kernels.repair_spikes)
                .arg(&mut device_demod)
                .arg(&device_diffed)
                .arg(&device_candidates)
                .arg(&mut device_block_flags)
                .arg(&n)
                .arg(&block_count)
                .arg(&threshold)
                .launch(LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let mut repaired = vec![0.0f32; real_len];
        stream.memcpy_dtoh(&device_demod, &mut repaired)?;
        stream.synchronize()?;
        assert_eq!(repaired[spike_index - 9], 1.0);
        assert!(repaired[spike_index - 8..spike_index + 30]
            .iter()
            .all(|&value| value == 0.0));
        assert_eq!(repaired[spike_index + 30], 1.0);
        Ok(())
    }
}
