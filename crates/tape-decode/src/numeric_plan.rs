//! Versioned, bit-exact numeric plans for deterministic decoding.
//!
//! A plan contains every architecture-sensitive numeric vector currently held
//! by [`DecoderSpec`]. Values are encoded as their IEEE-754 `f32` bits in a
//! fixed little-endian binary format. The header binds the payload to the
//! fully resolved [`DecodeRequest`], its profile, its exact input-rate bits,
//! and the decoder's block/field geometry.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, ensure, Context as _, Result};
use rustfft::num_complex::Complex32;
use sci_rs::signal::filter::design::Sos;
use serde::Serialize;

use crate::decode::BLOCKSIZE;
use crate::request::{ColorSystem, DecodeRequest, FieldOrderAction, WowInterpolation};
use crate::spec::DecoderSpec;

const MAGIC: &[u8; 8] = b"TDNPLAN\0";
const FORMAT_VERSION: u32 = 1;
const HEADER_LEN: u32 = 212;
const MAX_PLAN_BYTES: u64 = 256 * 1024 * 1024;
const MAX_SECTIONS: u32 = 64;
const MAX_SECTION_NAME: usize = 96;
const MAX_RANK: usize = 32;
const MAX_VALUES: u64 = (MAX_PLAN_BYTES / 4) - 1;

const REQUEST_DOMAIN: &[u8] = b"tape-decode canonical numeric request v1\0";
const PROFILE_DOMAIN: &[u8] = b"tape-decode canonical numeric profile v1\0";
const SCALAR_WITNESS_DOMAIN: &[u8] = b"tape-decode DecoderSpec scalar witness v1\0";
const FFT_WITNESS_DOMAIN: &[u8] = b"tape-decode DecoderSpec FFT witness v1\0";

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// BLAKE3 identity of the complete canonical numeric-plan file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanonicalNumericPlanIdentity([u8; 32]);

impl CanonicalNumericPlanIdentity {
    fn for_bytes(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for CanonicalNumericPlanIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", blake3::Hash::from_bytes(self.0).to_hex())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum Encoding {
    F32 = 1,
    ComplexF32 = 2,
    SosF32 = 3,
    NestedF32 = 4,
    NestedSosF32 = 5,
    OptionalF32 = 6,
    OptionalComplexF32 = 7,
    OptionalSosF32 = 8,
}

impl TryFrom<u8> for Encoding {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::F32),
            2 => Ok(Self::ComplexF32),
            3 => Ok(Self::SosF32),
            4 => Ok(Self::NestedF32),
            5 => Ok(Self::NestedSosF32),
            6 => Ok(Self::OptionalF32),
            7 => Ok(Self::OptionalComplexF32),
            8 => Ok(Self::OptionalSosF32),
            _ => bail!("unknown numeric-plan section encoding {value}"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Section {
    name: String,
    encoding: Encoding,
    dimensions: Vec<u64>,
    values: Vec<u32>,
}

impl Section {
    fn new(
        name: impl Into<String>,
        encoding: Encoding,
        dimensions: Vec<u64>,
        values: Vec<u32>,
    ) -> Self {
        Self {
            name: name.into(),
            encoding,
            dimensions,
            values,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Binding {
    request_digest: [u8; 32],
    profile_digest: [u8; 32],
    scalar_witness_digest: [u8; 32],
    fft_witness_digest: [u8; 32],
    input_frequency_bits: u64,
    block_size: u64,
    field_length: u64,
}

struct PlanFile {
    binding: Binding,
    sections: Vec<Section>,
}

impl PlanFile {
    fn encode(&self) -> Result<Vec<u8>> {
        ensure!(
            self.sections.len() <= MAX_SECTIONS as usize,
            "numeric plan has too many sections"
        );

        let mut payload = Vec::new();
        for section in &self.sections {
            ensure!(
                !section.name.is_empty() && section.name.len() <= MAX_SECTION_NAME,
                "invalid numeric-plan section name length for {}",
                section.name
            );
            ensure!(
                section
                    .name
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'),
                "invalid numeric-plan section name {}",
                section.name
            );
            ensure!(
                section.dimensions.len() <= MAX_RANK,
                "numeric-plan section {} has too many dimensions",
                section.name
            );
            ensure!(
                section.values.len() as u64 <= MAX_VALUES,
                "numeric-plan section {} is too large",
                section.name
            );
            put_u16(&mut payload, section.name.len() as u16);
            payload.extend_from_slice(section.name.as_bytes());
            payload.push(section.encoding as u8);
            payload.push(section.dimensions.len() as u8);
            for &dimension in &section.dimensions {
                put_u64(&mut payload, dimension);
            }
            put_u64(&mut payload, section.values.len() as u64);
            for &value in &section.values {
                put_u32(&mut payload, value);
            }
        }
        ensure!(
            payload.len() as u64 + u64::from(HEADER_LEN) <= MAX_PLAN_BYTES,
            "numeric plan exceeds the {}-byte safety limit",
            MAX_PLAN_BYTES
        );

        let payload_digest = blake3::hash(&payload);
        let mut output = Vec::with_capacity(HEADER_LEN as usize + payload.len());
        output.extend_from_slice(MAGIC);
        put_u32(&mut output, FORMAT_VERSION);
        put_u32(&mut output, HEADER_LEN);
        output.extend_from_slice(&self.binding.request_digest);
        output.extend_from_slice(&self.binding.profile_digest);
        output.extend_from_slice(&self.binding.scalar_witness_digest);
        output.extend_from_slice(&self.binding.fft_witness_digest);
        put_u64(&mut output, self.binding.input_frequency_bits);
        put_u64(&mut output, self.binding.block_size);
        put_u64(&mut output, self.binding.field_length);
        put_u32(&mut output, self.sections.len() as u32);
        put_u64(&mut output, payload.len() as u64);
        output.extend_from_slice(payload_digest.as_bytes());
        debug_assert_eq!(output.len(), HEADER_LEN as usize);
        output.extend_from_slice(&payload);
        Ok(output)
    }

    fn decode(input: &[u8]) -> Result<Self> {
        ensure!(
            input.len() as u64 <= MAX_PLAN_BYTES,
            "numeric plan exceeds the {}-byte safety limit",
            MAX_PLAN_BYTES
        );
        let mut cursor = Cursor::new(input);
        ensure!(cursor.take(8)? == MAGIC, "not a tape-decode numeric plan");
        let version = cursor.u32()?;
        ensure!(
            version == FORMAT_VERSION,
            "unsupported numeric-plan format version {version}; expected {FORMAT_VERSION}"
        );
        let header_len = cursor.u32()?;
        ensure!(
            header_len == HEADER_LEN,
            "unsupported numeric-plan header length {header_len}; expected {HEADER_LEN}"
        );
        let request_digest = cursor.array_32()?;
        let profile_digest = cursor.array_32()?;
        let scalar_witness_digest = cursor.array_32()?;
        let fft_witness_digest = cursor.array_32()?;
        let input_frequency_bits = cursor.u64()?;
        let block_size = cursor.u64()?;
        let field_length = cursor.u64()?;
        let section_count = cursor.u32()?;
        ensure!(
            section_count <= MAX_SECTIONS,
            "numeric plan has too many sections: {section_count}"
        );
        let payload_len = cursor.u64()?;
        let payload_digest = cursor.array_32()?;
        ensure!(
            cursor.position() == HEADER_LEN as usize,
            "invalid numeric-plan header layout"
        );
        ensure!(
            payload_len == cursor.remaining() as u64,
            "numeric-plan payload length mismatch: header says {payload_len}, file contains {}",
            cursor.remaining()
        );
        let payload = cursor.take(cursor.remaining())?;
        ensure!(
            blake3::hash(payload).as_bytes() == &payload_digest,
            "numeric-plan payload BLAKE3 mismatch"
        );

        let mut payload_cursor = Cursor::new(payload);
        let mut sections = Vec::with_capacity(section_count as usize);
        for _ in 0..section_count {
            let name_len = payload_cursor.u16()? as usize;
            ensure!(
                (1..=MAX_SECTION_NAME).contains(&name_len),
                "invalid numeric-plan section name length {name_len}"
            );
            let name_bytes = payload_cursor.take(name_len)?;
            ensure!(
                name_bytes.iter().all(|byte| byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || *byte == b'_'),
                "numeric-plan section name is not canonical ASCII"
            );
            let name = std::str::from_utf8(name_bytes)
                .context("numeric-plan section name is not UTF-8")?
                .to_owned();
            let encoding = Encoding::try_from(payload_cursor.u8()?)?;
            let rank = payload_cursor.u8()? as usize;
            ensure!(
                rank <= MAX_RANK,
                "numeric-plan section {name} rank is too large"
            );
            let mut dimensions = Vec::with_capacity(rank);
            for _ in 0..rank {
                dimensions.push(payload_cursor.u64()?);
            }
            let value_count = payload_cursor.u64()?;
            ensure!(
                value_count <= MAX_VALUES,
                "numeric-plan section {name} is too large"
            );
            let value_bytes = value_count
                .checked_mul(4)
                .context("numeric-plan section byte length overflow")?;
            ensure!(
                value_bytes <= payload_cursor.remaining() as u64,
                "numeric-plan section {name} value data is truncated"
            );
            let mut values = Vec::with_capacity(
                usize::try_from(value_count)
                    .context("numeric-plan value count does not fit usize")?,
            );
            for _ in 0..value_count {
                values.push(payload_cursor.u32()?);
            }
            sections.push(Section {
                name,
                encoding,
                dimensions,
                values,
            });
        }
        ensure!(
            payload_cursor.remaining() == 0,
            "numeric-plan payload has {} trailing bytes",
            payload_cursor.remaining()
        );

        Ok(Self {
            binding: Binding {
                request_digest,
                profile_digest,
                scalar_witness_digest,
                fft_witness_digest,
                input_frequency_bits,
                block_size,
                field_length,
            },
            sections,
        })
    }
}

impl DecoderSpec {
    /// Construct a spec and replace all supported numeric vectors from a
    /// validated canonical plan.
    pub fn new_with_canonical_numeric_plan(
        request: &DecodeRequest,
        path: &Path,
    ) -> Result<(Self, CanonicalNumericPlanIdentity)> {
        validate_supported_request(request)?;
        let input = read_plan_bytes(path)?;
        let identity = CanonicalNumericPlanIdentity::for_bytes(&input);
        let plan = PlanFile::decode(&input)
            .with_context(|| format!("failed to parse numeric plan {}", path.display()))?;

        let mut spec = Self::new(request)?;
        let binding = binding_for(request, &spec)?;
        validate_binding(plan.binding, binding)?;
        spec.apply_numeric_sections(plan.sections)?;
        Ok((spec, identity))
    }

    /// Write the supported numeric vectors in the canonical little-endian
    /// format and return the BLAKE3 identity of the complete file.
    pub fn write_canonical_numeric_plan(
        &self,
        request: &DecodeRequest,
        path: &Path,
        overwrite: bool,
    ) -> Result<CanonicalNumericPlanIdentity> {
        validate_supported_request(request)?;
        let plan = PlanFile {
            binding: binding_for(request, self)?,
            sections: self.numeric_sections()?,
        };
        let output = plan.encode()?;
        let identity = CanonicalNumericPlanIdentity::for_bytes(&output);
        publish_atomically(path, &output, overwrite)?;
        Ok(identity)
    }

    fn numeric_sections(&self) -> Result<Vec<Section>> {
        ensure!(
            self.video_chroma_trap.is_none(),
            "canonical numeric plans do not yet support --chroma-trap"
        );
        Ok(vec![
            optional_f32("video_eq_fft_gain", self.video_eq_fft_gain.as_deref()),
            nested_sos("chroma_afc_narrowband", &self.chroma_afc_narrowband),
            nested_f32("rf_chroma_heterodyne", &self.rf_chroma_heterodyne),
            pairs_f32("rf_fsc_wave", &self.rf_fsc_wave),
            f32_section(
                "chroma_burst_block_fft_gain",
                &self.chroma_burst_block_fft_gain,
            ),
            optional_sos(
                "chroma_filter_video_notch",
                self.chroma_filter_video_notch.as_deref(),
            ),
            optional_sos(
                "chroma_filter_deemphasis",
                self.chroma_filter_deemphasis.as_deref(),
            ),
            optional_sos(
                "chroma_filter_audio_notch",
                self.chroma_filter_audio_notch.as_deref(),
            ),
            sos_section("chroma_filter_final", &self.chroma_filter_final),
            optional_sos(
                "chroma_filter_secam_under",
                self.chroma_filter_secam_under.as_deref(),
            ),
            f32_section("video_rf_filter", &self.video_rf_filter),
            optional_f32("video_notch_filter", self.video_notch_filter.as_deref()),
            sos_section("video_env_post_filter", &self.video_env_post_filter),
            optional_f32(
                "video_rf_top_fft_gain",
                self.video_rf_top_fft_gain.as_deref(),
            ),
            complex_f32("video_filter", &self.video_filter),
            sos_section("video_nl_amplitude_lpf", &self.video_nl_amplitude_lpf),
            optional_complex_f32("video_nl_high_pass_f", self.video_nl_high_pass_f.as_deref()),
            optional_sos("video_fsc_notch", self.video_fsc_notch.as_deref()),
            complex_f32("video05_filter", &self.video05_filter),
            sos_section("resync_vsync_env_filter", &self.resync_vsync_env_filter),
            sos_section(
                "resync_serration_filter_base_0",
                &self.resync_serration_filter_base[0],
            ),
            sos_section(
                "resync_serration_filter_base_1",
                &self.resync_serration_filter_base[1],
            ),
            sos_section(
                "resync_serration_filter_envelope",
                &self.resync_serration_filter_envelope,
            ),
        ])
    }

    fn apply_numeric_sections(&mut self, sections: Vec<Section>) -> Result<()> {
        ensure!(
            self.video_chroma_trap.is_none(),
            "canonical numeric plans do not yet support --chroma-trap"
        );
        let mut by_name = BTreeMap::new();
        for section in sections {
            let name = section.name.clone();
            ensure!(
                by_name.insert(name.clone(), section).is_none(),
                "duplicate numeric-plan section {name}"
            );
        }

        let section = take_matching(
            &mut by_name,
            optional_f32("video_eq_fft_gain", self.video_eq_fft_gain.as_deref()),
        )?;
        assign_optional_f32(&mut self.video_eq_fft_gain, &section);

        let section = take_matching(
            &mut by_name,
            nested_sos("chroma_afc_narrowband", &self.chroma_afc_narrowband),
        )?;
        assign_nested_sos(&mut self.chroma_afc_narrowband, &section);

        let section = take_matching(
            &mut by_name,
            nested_f32("rf_chroma_heterodyne", &self.rf_chroma_heterodyne),
        )?;
        assign_nested_f32(&mut self.rf_chroma_heterodyne, &section);

        let section = take_matching(&mut by_name, pairs_f32("rf_fsc_wave", &self.rf_fsc_wave))?;
        assign_pairs(&mut self.rf_fsc_wave, &section);

        apply_f32(
            &mut by_name,
            "chroma_burst_block_fft_gain",
            &mut self.chroma_burst_block_fft_gain,
        )?;
        apply_optional_sos(
            &mut by_name,
            "chroma_filter_video_notch",
            &mut self.chroma_filter_video_notch,
        )?;
        apply_optional_sos(
            &mut by_name,
            "chroma_filter_deemphasis",
            &mut self.chroma_filter_deemphasis,
        )?;
        apply_optional_sos(
            &mut by_name,
            "chroma_filter_audio_notch",
            &mut self.chroma_filter_audio_notch,
        )?;
        apply_sos(
            &mut by_name,
            "chroma_filter_final",
            &mut self.chroma_filter_final,
        )?;
        apply_optional_sos(
            &mut by_name,
            "chroma_filter_secam_under",
            &mut self.chroma_filter_secam_under,
        )?;
        apply_f32(&mut by_name, "video_rf_filter", &mut self.video_rf_filter)?;
        apply_optional_f32(
            &mut by_name,
            "video_notch_filter",
            &mut self.video_notch_filter,
        )?;
        apply_sos(
            &mut by_name,
            "video_env_post_filter",
            &mut self.video_env_post_filter,
        )?;
        apply_optional_f32(
            &mut by_name,
            "video_rf_top_fft_gain",
            &mut self.video_rf_top_fft_gain,
        )?;
        apply_complex(&mut by_name, "video_filter", &mut self.video_filter)?;
        apply_sos(
            &mut by_name,
            "video_nl_amplitude_lpf",
            &mut self.video_nl_amplitude_lpf,
        )?;
        apply_optional_complex(
            &mut by_name,
            "video_nl_high_pass_f",
            &mut self.video_nl_high_pass_f,
        )?;
        apply_optional_sos(&mut by_name, "video_fsc_notch", &mut self.video_fsc_notch)?;
        apply_complex(&mut by_name, "video05_filter", &mut self.video05_filter)?;
        apply_sos(
            &mut by_name,
            "resync_vsync_env_filter",
            &mut self.resync_vsync_env_filter,
        )?;
        apply_sos(
            &mut by_name,
            "resync_serration_filter_base_0",
            &mut self.resync_serration_filter_base[0],
        )?;
        apply_sos(
            &mut by_name,
            "resync_serration_filter_base_1",
            &mut self.resync_serration_filter_base[1],
        )?;
        apply_sos(
            &mut by_name,
            "resync_serration_filter_envelope",
            &mut self.resync_serration_filter_envelope,
        )?;

        if let Some(name) = by_name.keys().next() {
            bail!("unknown numeric-plan section {name}");
        }
        Ok(())
    }
}

fn validate_supported_request(request: &DecodeRequest) -> Result<()> {
    ensure!(
        cfg!(feature = "deterministic"),
        "canonical numeric plans require a deterministic build; rebuild with --no-default-features --features deterministic"
    );
    ensure!(
        !request.cafc,
        "canonical numeric plans do not yet support --cafc because it constructs architecture-sensitive filters at decode time"
    );
    ensure!(
        !request.chroma_trap,
        "canonical numeric plans do not yet support --chroma-trap"
    );
    ensure!(
        request
            .decode_profile
            .decoder_params
            .chroma_carrier_mult
            .is_none(),
        "canonical numeric plans do not yet support SECAM method 1 because its data-dependent phase restoration uses runtime platform transcendentals"
    );
    Ok(())
}

fn read_plan_bytes(path: &Path) -> Result<Vec<u8>> {
    // Reject ordinary FIFOs/devices before opening them; opening a FIFO for
    // reading can block before descriptor metadata is available.
    let path_metadata = std::fs::metadata(path)
        .with_context(|| format!("failed to stat numeric plan {}", path.display()))?;
    ensure!(
        path_metadata.is_file(),
        "numeric plan {} is not a regular file",
        path.display()
    );

    // Open once so a path replacement cannot change the file between the size
    // check and the read. The bounded reader separately catches a file that
    // grows after metadata was sampled without allocating beyond the limit.
    let mut file = File::open(path)
        .with_context(|| format!("failed to open numeric plan {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to stat numeric plan {}", path.display()))?;
    ensure!(
        metadata.is_file(),
        "numeric plan {} is not a regular file",
        path.display()
    );
    ensure!(
        metadata.len() <= MAX_PLAN_BYTES,
        "numeric plan {} exceeds the {}-byte safety limit",
        path.display(),
        MAX_PLAN_BYTES
    );

    read_bounded(&mut file, metadata.len(), MAX_PLAN_BYTES)
        .with_context(|| format!("failed to read numeric plan {}", path.display()))
}

fn read_bounded(reader: impl Read, advertised_len: u64, maximum_len: u64) -> Result<Vec<u8>> {
    let read_limit = maximum_len
        .checked_add(1)
        .context("numeric-plan read limit overflow")?;
    let capacity = usize::try_from(advertised_len.min(maximum_len))
        .context("numeric-plan allocation size does not fit this platform")?;
    let mut input = Vec::with_capacity(capacity);
    reader
        .take(read_limit)
        .read_to_end(&mut input)
        .context("numeric-plan read failed")?;
    ensure!(
        input.len() as u64 <= maximum_len,
        "numeric plan exceeds the {maximum_len}-byte safety limit"
    );
    Ok(input)
}

fn binding_for(request: &DecodeRequest, spec: &DecoderSpec) -> Result<Binding> {
    let field_lines = spec
        .sys_field_lines
        .iter()
        .copied()
        .max()
        .context("decoder profile has no field lengths")?;
    ensure!(
        field_lines >= 0,
        "decoder profile has a negative field length"
    );
    let field_length = u64::try_from(spec.sys_outlinelen)?
        .checked_mul(field_lines as u64)
        .context("decoder field length overflow")?;
    Ok(Binding {
        request_digest: json_digest(REQUEST_DOMAIN, request)?,
        profile_digest: json_digest(PROFILE_DOMAIN, &request.decode_profile)?,
        scalar_witness_digest: scalar_witness(spec)?,
        fft_witness_digest: fft_witness(spec)?,
        input_frequency_bits: request.inputfreq.to_bits(),
        block_size: BLOCKSIZE as u64,
        field_length,
    })
}

fn json_digest(domain: &[u8], value: &impl Serialize) -> Result<[u8; 32]> {
    let json = serde_json::to_vec(value).context("failed to serialize numeric-plan binding")?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&json);
    Ok(*hasher.finalize().as_bytes())
}

fn validate_binding(actual: Binding, expected: Binding) -> Result<()> {
    ensure!(
        actual.profile_digest == expected.profile_digest,
        "numeric plan was emitted for a different resolved profile"
    );
    ensure!(
        actual.input_frequency_bits == expected.input_frequency_bits,
        "numeric-plan input rate mismatch: plan has f64 bits {:016x}, request has {:016x}",
        actual.input_frequency_bits,
        expected.input_frequency_bits
    );
    ensure!(
        actual.request_digest == expected.request_digest,
        "numeric plan was emitted for different resolved decode options"
    );
    ensure!(
        actual.scalar_witness_digest == expected.scalar_witness_digest,
        "numeric-plan scalar witness mismatch; this build derived different output-critical scalar state"
    );
    ensure!(
        actual.fft_witness_digest == expected.fft_witness_digest,
        "numeric-plan FFT witness mismatch; this build's FFT implementation is not bit-conformant"
    );
    ensure!(
        actual.block_size == expected.block_size,
        "numeric-plan block size mismatch: plan has {}, decoder requires {}",
        actual.block_size,
        expected.block_size
    );
    ensure!(
        actual.field_length == expected.field_length,
        "numeric-plan field length mismatch: plan has {}, decoder requires {}",
        actual.field_length,
        expected.field_length
    );
    Ok(())
}

fn publish_atomically(path: &Path, contents: &[u8], overwrite: bool) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    ensure!(
        path.file_name().is_some(),
        "numeric-plan output path must name a file"
    );

    let (temporary_path, mut temporary_file) = (0..100)
        .find_map(|_| {
            let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let temporary_path = parent.join(format!(
                ".tape-decode-numeric-plan-{}-{sequence}.tmp",
                std::process::id()
            ));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary_path)
            {
                Ok(file) => Some(Ok((temporary_path, file))),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => Some(Err(error)),
            }
        })
        .context("unable to allocate a unique numeric-plan temporary filename")?
        .with_context(|| {
            format!(
                "failed to create a numeric-plan temporary file in {}",
                parent.display()
            )
        })?;
    let mut cleanup = TemporaryFileCleanup(Some(temporary_path.clone()));

    temporary_file.write_all(contents).with_context(|| {
        format!(
            "failed to write temporary numeric plan in {}",
            parent.display()
        )
    })?;
    temporary_file.sync_all().with_context(|| {
        format!(
            "failed to sync temporary numeric plan in {}",
            parent.display()
        )
    })?;
    drop(temporary_file);

    if overwrite {
        #[cfg(unix)]
        std::fs::rename(&temporary_path, path).with_context(|| {
            format!(
                "failed to atomically replace numeric plan {}",
                path.display()
            )
        })?;

        // The standard library does not expose an atomic replace operation on
        // every non-Unix platform. Refuse to remove an existing approved plan;
        // callers can publish to a fresh filename instead.
        #[cfg(not(unix))]
        {
            ensure!(
                !path.exists(),
                "atomic numeric-plan replacement is unavailable on this platform; choose a new output filename"
            );
            std::fs::hard_link(&temporary_path, path)
                .with_context(|| format!("failed to publish numeric plan {}", path.display()))?;
            std::fs::remove_file(&temporary_path).with_context(|| {
                format!(
                    "failed to remove temporary numeric plan {}",
                    temporary_path.display()
                )
            })?;
        }
    } else {
        // A same-directory hard link is an atomic, no-clobber publication: it
        // fails if another process won the destination name race.
        std::fs::hard_link(&temporary_path, path).with_context(|| {
            format!(
                "failed to publish numeric plan {} without overwriting an existing file",
                path.display()
            )
        })?;
        std::fs::remove_file(&temporary_path).with_context(|| {
            format!(
                "failed to remove temporary numeric plan {}",
                temporary_path.display()
            )
        })?;
    }
    cleanup.0 = None;

    // Persist the directory entry as well as the already-synced file contents.
    #[cfg(unix)]
    File::open(parent)
        .with_context(|| format!("failed to open numeric-plan directory {}", parent.display()))?
        .sync_all()
        .with_context(|| format!("failed to sync numeric-plan directory {}", parent.display()))?;

    Ok(())
}

struct TemporaryFileCleanup(Option<PathBuf>);

impl Drop for TemporaryFileCleanup {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Exact witness for every non-vector value retained by `DecoderSpec`.
///
/// The plan overrides vector/SOS state only. This witness makes the remaining
/// constructor output part of the conformance boundary instead of assuming it
/// is architecture-independent.
fn scalar_witness(spec: &DecoderSpec) -> Result<[u8; 32]> {
    ensure!(
        spec.video_chroma_trap.is_none(),
        "canonical numeric plans do not yet support --chroma-trap"
    );
    let mut witness = Witness::new(SCALAR_WITNESS_DOMAIN);

    witness.f64("freq", spec.freq);
    witness.u8("color_system", color_system_tag(spec.color_system));
    witness.f64("sys_fsc_mhz", spec.sys_fsc_mhz);
    witness.u64("sys_frame_lines", spec.sys_frame_lines.line_count() as u64);
    witness.i64_slice("sys_field_lines", &spec.sys_field_lines);
    witness.f64("sys_line_period", spec.sys_line_period);
    witness.f64_slice("sys_active_video_us", &spec.sys_active_video_us);
    witness.f64("sys_fps", spec.sys_fps);
    witness.f32("sys_ire0", spec.sys_ire0);
    witness.f32("sys_hz_ire", spec.sys_hz_ire);
    witness.f32("sys_vsync_ire", spec.sys_vsync_ire);
    witness.f64_slice("sys_color_burst_us", &spec.sys_color_burst_us);
    witness.usize_slice("sys_blacksnr_slice", &spec.sys_blacksnr_slice);
    witness.usize("sys_num_pulses", spec.sys_num_pulses);
    witness.f64("sys_hsync_pulse_us", spec.sys_hsync_pulse_us);
    witness.f64("sys_eq_pulse_us", spec.sys_eq_pulse_us);
    witness.f64("sys_vsync_pulse_us", spec.sys_vsync_pulse_us);
    witness.i64("sys_output_zero", spec.sys_output_zero);
    witness.usize("sys_outlinelen", spec.sys_outlinelen);
    witness.f64("sys_outfreq", spec.sys_outfreq);
    witness.label("sys_ld_vits_whitelocs");
    witness.u64_raw(spec.sys_ld_vits_whitelocs.len() as u64);
    for location in &spec.sys_ld_vits_whitelocs {
        for &value in location {
            witness.u64_raw(value as u64);
        }
    }
    witness.option_f32("sys_burst_abs_ref", spec.sys_burst_abs_ref);
    witness.f64_slice("sys_track_ire0_offset", &spec.sys_track_ire0_offset);
    witness.option_f32("sys_nonlinear_deviation", spec.sys_nonlinear_deviation);

    witness.f64(
        "decoder_color_under_carrier",
        spec.decoder_color_under_carrier,
    );
    witness.f64("decoder_chroma_bpf_upper", spec.decoder_chroma_bpf_upper);
    witness.usize("decoder_chroma_bpf_order", spec.decoder_chroma_bpf_order);
    witness.f64("decoder_chroma_bpf_lower", spec.decoder_chroma_bpf_lower);
    witness.label("decoder_chroma_rotation");
    match spec.decoder_chroma_rotation {
        Some(values) => {
            witness.u8_raw(1);
            witness.i64_raw(values[0]);
            witness.i64_raw(values[1]);
        }
        None => witness.u8_raw(0),
    }
    witness.option_f64(
        "decoder_chroma_carrier_mult",
        spec.decoder_chroma_carrier_mult,
    );
    witness.f64("decoder_chroma_offset", spec.decoder_chroma_offset);
    witness.f32(
        "decoder_nonlinear_highpass_limit_l",
        spec.decoder_nonlinear_highpass_limit_l,
    );
    witness.f32(
        "decoder_nonlinear_highpass_limit_h",
        spec.decoder_nonlinear_highpass_limit_h,
    );
    witness.f32(
        "decoder_nonlinear_exp_scaling",
        spec.decoder_nonlinear_exp_scaling,
    );
    witness.option_f32(
        "decoder_nonlinear_scaling_1",
        spec.decoder_nonlinear_scaling_1,
    );
    witness.option_f32(
        "decoder_nonlinear_scaling_2",
        spec.decoder_nonlinear_scaling_2,
    );
    witness.label("decoder_nonlinear_logistic");
    match spec.decoder_nonlinear_logistic {
        Some((mid, rate)) => {
            witness.u8_raw(1);
            witness.f32_raw(mid);
            witness.f32_raw(rate);
        }
        None => witness.u8_raw(0),
    }
    witness.option_f32(
        "decoder_nonlinear_static_factor",
        spec.decoder_nonlinear_static_factor,
    );

    witness.u8(
        "field_order_action",
        field_order_action_tag(spec.field_order_action),
    );
    witness.f64(
        "chroma_afc_fine_tune_fh_ratio",
        spec.chroma_afc_fine_tune_fh_ratio,
    );
    witness.f32("dod_threshold_p", spec.dod_threshold_p);
    witness.option_f32("dod_threshold_a", spec.dod_threshold_a);
    witness.f32("dod_hysteresis", spec.dod_hysteresis);

    witness.bool("rf_disable_comb", spec.rf_disable_comb);
    witness.bool("rf_disable_right_hsync", spec.rf_disable_right_hsync);
    witness.bool("rf_disable_dc_offset", spec.rf_disable_dc_offset);
    witness.bool("rf_fallback_vsync", spec.rf_fallback_vsync);
    witness.i64("rf_field_order_confidence", spec.rf_field_order_confidence);
    witness.bool("rf_saved_levels", spec.rf_saved_levels);
    witness.f32("rf_y_comb", spec.rf_y_comb);
    witness.bool("rf_write_chroma", spec.rf_write_chroma);
    witness.bool("rf_skip_hsync_refine", spec.rf_skip_hsync_refine);
    witness.bool("rf_export_raw_tbc", spec.rf_export_raw_tbc);
    witness.bool("rf_ire0_adjust", spec.rf_ire0_adjust);
    witness.bool("rf_relaxed_line0", spec.rf_relaxed_line0);
    witness.bool(
        "rf_detect_chroma_track_phase",
        spec.rf_detect_chroma_track_phase,
    );
    witness.bool("rf_disable_burst_hsync", spec.rf_disable_burst_hsync);
    witness.bool(
        "rf_disable_phase_correction",
        spec.rf_disable_phase_correction,
    );

    witness.option_f32("video_high_boost_value", spec.video_high_boost_value);
    witness.bool("video_disable_diff_demod", spec.video_disable_diff_demod);
    witness.bool("video_chroma_trap_present", false);
    witness.bool("video_nldeemp_enabled", spec.video_nldeemp_enabled);
    witness.bool("video_subdeemp_enabled", spec.video_subdeemp_enabled);

    witness.usize("resync_divisor", spec.resync_divisor);
    witness.usize(
        "resync_vsync_env_decimation",
        spec.resync_vsync_env_decimation,
    );
    witness.bool("ntscj", spec.ntscj);
    witness.label("track_phase");
    match spec.track_phase {
        Some(value) => {
            witness.u8_raw(1);
            witness.i64_raw(value);
        }
        None => witness.u8_raw(0),
    }
    witness.f32(
        "wow_level_adjust_smoothing",
        spec.wow_level_adjust_smoothing,
    );
    witness.u8(
        "wow_interpolation_method",
        wow_interpolation_tag(spec.wow_interpolation_method),
    );
    witness.bool("do_dod", spec.do_dod);
    witness.f32("level_adjust", spec.level_adjust);

    Ok(witness.finish())
}

/// Execute each FFT plan used by `DecoderSpec` on an exact, generated witness.
/// This binds a plan to the actual FFT arithmetic/twiddle behavior without
/// serializing implementation-private FFT state.
fn fft_witness(spec: &DecoderSpec) -> Result<[u8; 32]> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(FFT_WITNESS_DOMAIN);
    witness_complex_fft(
        &mut hasher,
        b"block_inverse",
        spec.fft_block_inverse_f32.as_ref(),
    );
    witness_real_forward_fft(
        &mut hasher,
        b"block_real_forward",
        spec.fft_block_r2c_f32.as_ref(),
    )?;
    witness_real_inverse_fft(
        &mut hasher,
        b"block_real_inverse",
        spec.fft_block_c2r_f32.as_ref(),
    )?;
    witness_complex_fft(
        &mut hasher,
        b"field_forward",
        spec.fft_field_forward_f32.as_ref(),
    );
    witness_complex_fft(
        &mut hasher,
        b"field_inverse",
        spec.fft_field_inverse_f32.as_ref(),
    );
    Ok(*hasher.finalize().as_bytes())
}

fn witness_sample(index: usize, salt: usize) -> f32 {
    let integer = ((index.wrapping_mul(73).wrapping_add(salt * 29)) % 257) as i32 - 128;
    integer as f32 * (1.0 / 128.0)
}

fn witness_complex_fft(hasher: &mut blake3::Hasher, label: &[u8], fft: &dyn rustfft::Fft<f32>) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(fft.len() as u64).to_le_bytes());
    let mut values = (0..fft.len())
        .map(|index| Complex32::new(witness_sample(index, 1), witness_sample(index, 2)))
        .collect::<Vec<_>>();
    fft.process(&mut values);
    for value in values {
        hasher.update(&value.re.to_bits().to_le_bytes());
        hasher.update(&value.im.to_bits().to_le_bytes());
    }
}

fn witness_real_forward_fft(
    hasher: &mut blake3::Hasher,
    label: &[u8],
    fft: &dyn realfft::RealToComplex<f32>,
) -> Result<()> {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(fft.len() as u64).to_le_bytes());
    let mut input = fft.make_input_vec();
    for (index, value) in input.iter_mut().enumerate() {
        *value = witness_sample(index, 3);
    }
    let mut output = fft.make_output_vec();
    fft.process(&mut input, &mut output)
        .context("failed to execute real-forward FFT witness")?;
    for value in output {
        hasher.update(&value.re.to_bits().to_le_bytes());
        hasher.update(&value.im.to_bits().to_le_bytes());
    }
    Ok(())
}

fn witness_real_inverse_fft(
    hasher: &mut blake3::Hasher,
    label: &[u8],
    fft: &dyn realfft::ComplexToReal<f32>,
) -> Result<()> {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(fft.len() as u64).to_le_bytes());
    let mut input = fft.make_input_vec();
    for (index, value) in input.iter_mut().enumerate() {
        *value = Complex32::new(witness_sample(index, 4), witness_sample(index, 5));
    }
    if let Some(first) = input.first_mut() {
        first.im = 0.0;
    }
    if fft.len().is_multiple_of(2) {
        if let Some(last) = input.last_mut() {
            last.im = 0.0;
        }
    }
    let mut output = fft.make_output_vec();
    fft.process(&mut input, &mut output)
        .context("failed to execute real-inverse FFT witness")?;
    for value in output {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    Ok(())
}

struct Witness(blake3::Hasher);

impl Witness {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(domain);
        Self(hasher)
    }

    fn label(&mut self, label: &str) {
        self.u64_raw(label.len() as u64);
        self.0.update(label.as_bytes());
    }

    fn u8_raw(&mut self, value: u8) {
        self.0.update(&[value]);
    }

    fn u64_raw(&mut self, value: u64) {
        self.0.update(&value.to_le_bytes());
    }

    fn i64_raw(&mut self, value: i64) {
        self.0.update(&value.to_le_bytes());
    }

    fn f32_raw(&mut self, value: f32) {
        self.0.update(&value.to_bits().to_le_bytes());
    }

    fn u8(&mut self, label: &str, value: u8) {
        self.label(label);
        self.u8_raw(value);
    }

    fn bool(&mut self, label: &str, value: bool) {
        self.u8(label, u8::from(value));
    }

    fn usize(&mut self, label: &str, value: usize) {
        self.label(label);
        self.u64_raw(value as u64);
    }

    fn u64(&mut self, label: &str, value: u64) {
        self.label(label);
        self.u64_raw(value);
    }

    fn i64(&mut self, label: &str, value: i64) {
        self.label(label);
        self.i64_raw(value);
    }

    fn f32(&mut self, label: &str, value: f32) {
        self.label(label);
        self.f32_raw(value);
    }

    fn f64(&mut self, label: &str, value: f64) {
        self.label(label);
        self.0.update(&value.to_bits().to_le_bytes());
    }

    fn option_f32(&mut self, label: &str, value: Option<f32>) {
        self.label(label);
        match value {
            Some(value) => {
                self.u8_raw(1);
                self.f32_raw(value);
            }
            None => self.u8_raw(0),
        }
    }

    fn option_f64(&mut self, label: &str, value: Option<f64>) {
        self.label(label);
        match value {
            Some(value) => {
                self.u8_raw(1);
                self.0.update(&value.to_bits().to_le_bytes());
            }
            None => self.u8_raw(0),
        }
    }

    fn i64_slice(&mut self, label: &str, values: &[i64]) {
        self.label(label);
        self.u64_raw(values.len() as u64);
        for &value in values {
            self.i64_raw(value);
        }
    }

    fn usize_slice(&mut self, label: &str, values: &[usize]) {
        self.label(label);
        self.u64_raw(values.len() as u64);
        for &value in values {
            self.u64_raw(value as u64);
        }
    }

    fn f64_slice(&mut self, label: &str, values: &[f64]) {
        self.label(label);
        self.u64_raw(values.len() as u64);
        for &value in values {
            self.0.update(&value.to_bits().to_le_bytes());
        }
    }

    fn finish(self) -> [u8; 32] {
        *self.0.finalize().as_bytes()
    }
}

fn color_system_tag(value: ColorSystem) -> u8 {
    match value {
        ColorSystem::Ntsc => 0,
        ColorSystem::Pal => 1,
        ColorSystem::Secam => 2,
        ColorSystem::Monochrome => 3,
    }
}

fn field_order_action_tag(value: FieldOrderAction) -> u8 {
    match value {
        FieldOrderAction::Detect => 0,
        FieldOrderAction::Duplicate => 1,
        FieldOrderAction::Drop => 2,
        FieldOrderAction::None => 3,
    }
}

fn wow_interpolation_tag(value: WowInterpolation) -> u8 {
    match value {
        WowInterpolation::Linear => 0,
        WowInterpolation::Quadratic => 1,
        WowInterpolation::Cubic => 2,
    }
}

fn f32_section(name: &str, values: &[f32]) -> Section {
    Section::new(
        name,
        Encoding::F32,
        vec![values.len() as u64],
        values.iter().map(|value| value.to_bits()).collect(),
    )
}

fn complex_f32(name: &str, values: &[Complex32]) -> Section {
    Section::new(
        name,
        Encoding::ComplexF32,
        vec![values.len() as u64, 2],
        values
            .iter()
            .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
            .collect(),
    )
}

fn pairs_f32(name: &str, values: &[(f32, f32)]) -> Section {
    Section::new(
        name,
        Encoding::ComplexF32,
        vec![values.len() as u64, 2],
        values
            .iter()
            .flat_map(|&(first, second)| [first.to_bits(), second.to_bits()])
            .collect(),
    )
}

fn sos_section(name: &str, sections: &[Sos<f32>]) -> Section {
    Section::new(
        name,
        Encoding::SosF32,
        vec![sections.len() as u64, 8],
        sos_bits(sections),
    )
}

fn sos_bits(sections: &[Sos<f32>]) -> Vec<u32> {
    sections
        .iter()
        .flat_map(|section| {
            section
                .b
                .into_iter()
                .chain(section.a)
                .chain([section.zi0, section.zi1])
                .map(f32::to_bits)
        })
        .collect()
}

fn nested_f32(name: &str, values: &[Vec<f32>]) -> Section {
    let mut dimensions = Vec::with_capacity(values.len() + 1);
    dimensions.push(values.len() as u64);
    dimensions.extend(values.iter().map(|values| values.len() as u64));
    Section::new(
        name,
        Encoding::NestedF32,
        dimensions,
        values
            .iter()
            .flatten()
            .map(|value| value.to_bits())
            .collect(),
    )
}

fn nested_sos(name: &str, values: &[Vec<Sos<f32>>]) -> Section {
    let mut dimensions = Vec::with_capacity(values.len() + 2);
    dimensions.push(values.len() as u64);
    dimensions.push(8);
    dimensions.extend(values.iter().map(|values| values.len() as u64));
    Section::new(
        name,
        Encoding::NestedSosF32,
        dimensions,
        values
            .iter()
            .flat_map(|sections| sos_bits(sections))
            .collect(),
    )
}

fn optional_f32(name: &str, values: Option<&[f32]>) -> Section {
    match values {
        Some(values) => Section::new(
            name,
            Encoding::OptionalF32,
            vec![1, values.len() as u64],
            values.iter().map(|value| value.to_bits()).collect(),
        ),
        None => Section::new(name, Encoding::OptionalF32, vec![0, 0], Vec::new()),
    }
}

fn optional_complex_f32(name: &str, values: Option<&[Complex32]>) -> Section {
    match values {
        Some(values) => Section::new(
            name,
            Encoding::OptionalComplexF32,
            vec![1, values.len() as u64, 2],
            values
                .iter()
                .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
                .collect(),
        ),
        None => Section::new(
            name,
            Encoding::OptionalComplexF32,
            vec![0, 0, 2],
            Vec::new(),
        ),
    }
}

fn optional_sos(name: &str, sections: Option<&[Sos<f32>]>) -> Section {
    match sections {
        Some(sections) => Section::new(
            name,
            Encoding::OptionalSosF32,
            vec![1, sections.len() as u64, 8],
            sos_bits(sections),
        ),
        None => Section::new(name, Encoding::OptionalSosF32, vec![0, 0, 8], Vec::new()),
    }
}

fn take_matching(sections: &mut BTreeMap<String, Section>, expected: Section) -> Result<Section> {
    let actual = sections
        .remove(&expected.name)
        .with_context(|| format!("numeric plan is missing section {}", expected.name))?;
    ensure!(
        actual.encoding == expected.encoding,
        "numeric-plan section {} encoding mismatch: plan has {:?}, decoder requires {:?}",
        actual.name,
        actual.encoding,
        expected.encoding
    );
    ensure!(
        actual.dimensions == expected.dimensions,
        "numeric-plan section {} shape mismatch: plan has {:?}, decoder requires {:?}",
        actual.name,
        actual.dimensions,
        expected.dimensions
    );
    ensure!(
        actual.values.len() == expected.values.len(),
        "numeric-plan section {} value-count mismatch: plan has {}, decoder requires {}",
        actual.name,
        actual.values.len(),
        expected.values.len()
    );
    Ok(actual)
}

fn apply_f32(
    sections: &mut BTreeMap<String, Section>,
    name: &str,
    target: &mut Vec<f32>,
) -> Result<()> {
    let section = take_matching(sections, f32_section(name, target))?;
    assign_f32(target, &section);
    Ok(())
}

fn apply_complex(
    sections: &mut BTreeMap<String, Section>,
    name: &str,
    target: &mut Vec<Complex32>,
) -> Result<()> {
    let section = take_matching(sections, complex_f32(name, target))?;
    assign_complex(target, &section);
    Ok(())
}

fn apply_sos(
    sections: &mut BTreeMap<String, Section>,
    name: &str,
    target: &mut Vec<Sos<f32>>,
) -> Result<()> {
    let section = take_matching(sections, sos_section(name, target))?;
    assign_sos(target, &section);
    Ok(())
}

fn apply_optional_f32(
    sections: &mut BTreeMap<String, Section>,
    name: &str,
    target: &mut Option<Vec<f32>>,
) -> Result<()> {
    let section = take_matching(sections, optional_f32(name, target.as_deref()))?;
    assign_optional_f32(target, &section);
    Ok(())
}

fn apply_optional_complex(
    sections: &mut BTreeMap<String, Section>,
    name: &str,
    target: &mut Option<Vec<Complex32>>,
) -> Result<()> {
    let section = take_matching(sections, optional_complex_f32(name, target.as_deref()))?;
    if let Some(target) = target {
        assign_complex(target, &section);
    }
    Ok(())
}

fn apply_optional_sos(
    sections: &mut BTreeMap<String, Section>,
    name: &str,
    target: &mut Option<Vec<Sos<f32>>>,
) -> Result<()> {
    let section = take_matching(sections, optional_sos(name, target.as_deref()))?;
    if let Some(target) = target {
        assign_sos(target, &section);
    }
    Ok(())
}

fn assign_f32(target: &mut [f32], section: &Section) {
    for (target, &bits) in target.iter_mut().zip(&section.values) {
        *target = f32::from_bits(bits);
    }
}

fn assign_optional_f32(target: &mut Option<Vec<f32>>, section: &Section) {
    if let Some(target) = target {
        assign_f32(target, section);
    }
}

fn assign_complex(target: &mut [Complex32], section: &Section) {
    for (target, bits) in target.iter_mut().zip(section.values.chunks_exact(2)) {
        target.re = f32::from_bits(bits[0]);
        target.im = f32::from_bits(bits[1]);
    }
}

fn assign_pairs(target: &mut [(f32, f32)], section: &Section) {
    for (target, bits) in target.iter_mut().zip(section.values.chunks_exact(2)) {
        target.0 = f32::from_bits(bits[0]);
        target.1 = f32::from_bits(bits[1]);
    }
}

fn assign_sos(target: &mut [Sos<f32>], section: &Section) {
    for (target, bits) in target.iter_mut().zip(section.values.chunks_exact(8)) {
        target.b = [
            f32::from_bits(bits[0]),
            f32::from_bits(bits[1]),
            f32::from_bits(bits[2]),
        ];
        target.a = [
            f32::from_bits(bits[3]),
            f32::from_bits(bits[4]),
            f32::from_bits(bits[5]),
        ];
        target.zi0 = f32::from_bits(bits[6]);
        target.zi1 = f32::from_bits(bits[7]);
    }
}

fn assign_nested_f32(target: &mut [Vec<f32>], section: &Section) {
    let mut values = section.values.iter().copied();
    for inner in target {
        for target in inner {
            *target = f32::from_bits(values.next().expect("shape was validated"));
        }
    }
    debug_assert!(values.next().is_none());
}

fn assign_nested_sos(target: &mut [Vec<Sos<f32>>], section: &Section) {
    let mut chunks = section.values.chunks_exact(8);
    for inner in target {
        for target in inner {
            let bits = chunks.next().expect("shape was validated");
            target.b = [
                f32::from_bits(bits[0]),
                f32::from_bits(bits[1]),
                f32::from_bits(bits[2]),
            ];
            target.a = [
                f32::from_bits(bits[3]),
                f32::from_bits(bits[4]),
                f32::from_bits(bits[5]),
            ];
            target.zi0 = f32::from_bits(bits[6]);
            target.zi1 = f32::from_bits(bits[7]);
        }
    }
    debug_assert!(chunks.next().is_none());
}

fn put_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

struct Cursor<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self { input, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn remaining(&self) -> usize {
        self.input.len() - self.position
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .context("numeric-plan offset overflow")?;
        ensure!(end <= self.input.len(), "truncated numeric plan");
        let output = &self.input[self.position..end];
        self.position = end;
        Ok(output)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn array_32(&mut self) -> Result<[u8; 32]> {
        Ok(self.take(32)?.try_into().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_binding() -> Binding {
        Binding {
            request_digest: [0x11; 32],
            profile_digest: [0x22; 32],
            scalar_witness_digest: [0x33; 32],
            fft_witness_digest: [0x44; 32],
            input_frequency_bits: 28.636_363_f64.to_bits(),
            block_size: 32768,
            field_length: 478_660,
        }
    }

    fn test_plan() -> PlanFile {
        PlanFile {
            binding: test_binding(),
            sections: vec![
                f32_section(
                    "signed_values",
                    &[f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234)],
                ),
                complex_f32(
                    "complex_values",
                    &[Complex32::new(
                        f32::from_bits(0x3f80_0000),
                        f32::from_bits(0xbf80_0000),
                    )],
                ),
            ],
        }
    }

    #[test]
    fn binary_round_trip_preserves_exact_float_bits() {
        let encoded = test_plan().encode().unwrap();
        let decoded = PlanFile::decode(&encoded).unwrap();
        assert_eq!(decoded.binding, test_binding());
        assert_eq!(decoded.sections, test_plan().sections);
        assert_eq!(decoded.sections[0].values, vec![0x8000_0000, 0x7fc0_1234]);
        assert_eq!(
            CanonicalNumericPlanIdentity::for_bytes(&encoded)
                .to_string()
                .len(),
            64
        );
    }

    #[test]
    fn payload_corruption_is_rejected() {
        let mut encoded = test_plan().encode().unwrap();
        *encoded.last_mut().unwrap() ^= 1;
        let error = PlanFile::decode(&encoded).err().unwrap().to_string();
        assert!(error.contains("payload BLAKE3 mismatch"), "{error}");
    }

    #[test]
    fn binding_reports_profile_rate_and_options_separately() {
        let expected = test_binding();

        let mut actual = expected;
        actual.profile_digest[0] ^= 1;
        assert!(validate_binding(actual, expected)
            .err()
            .unwrap()
            .to_string()
            .contains("different resolved profile"));

        let mut actual = expected;
        actual.input_frequency_bits ^= 1;
        assert!(validate_binding(actual, expected)
            .err()
            .unwrap()
            .to_string()
            .contains("input rate mismatch"));

        let mut actual = expected;
        actual.request_digest[0] ^= 1;
        assert!(validate_binding(actual, expected)
            .err()
            .unwrap()
            .to_string()
            .contains("different resolved decode options"));

        let mut actual = expected;
        actual.scalar_witness_digest[0] ^= 1;
        assert!(validate_binding(actual, expected)
            .err()
            .unwrap()
            .to_string()
            .contains("scalar witness mismatch"));

        let mut actual = expected;
        actual.fft_witness_digest[0] ^= 1;
        assert!(validate_binding(actual, expected)
            .err()
            .unwrap()
            .to_string()
            .contains("FFT witness mismatch"));
    }

    #[test]
    fn section_shape_mismatch_is_rejected() {
        let mut sections =
            BTreeMap::from([("values".to_owned(), f32_section("values", &[1.0, 2.0]))]);
        let error = take_matching(&mut sections, f32_section("values", &[0.0]))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("shape mismatch"), "{error}");
    }

    #[test]
    fn trailing_payload_bytes_are_rejected_even_with_valid_digest() {
        let plan = test_plan();
        let mut encoded = plan.encode().unwrap();
        encoded.push(0);
        let payload_len = encoded.len() - HEADER_LEN as usize;
        let payload_len_offset = 172;
        encoded[payload_len_offset..payload_len_offset + 8]
            .copy_from_slice(&(payload_len as u64).to_le_bytes());
        let payload_digest_offset = 180;
        let digest = blake3::hash(&encoded[HEADER_LEN as usize..]);
        encoded[payload_digest_offset..payload_digest_offset + 32]
            .copy_from_slice(digest.as_bytes());
        let error = PlanFile::decode(&encoded).err().unwrap().to_string();
        assert!(error.contains("trailing bytes"), "{error}");
    }

    #[test]
    fn bounded_read_accepts_the_limit_and_rejects_growth_past_it() {
        let mut exact = std::io::Cursor::new(vec![0u8; 8]);
        assert_eq!(read_bounded(&mut exact, 8, 8).unwrap().len(), 8);
        assert_eq!(exact.position(), 8);

        // Simulate a file that advertised one byte, then grew before the read.
        let mut grown = std::io::Cursor::new(vec![0u8; 32]);
        let error = read_bounded(&mut grown, 1, 8).err().unwrap().to_string();
        assert!(error.contains("8-byte safety limit"), "{error}");
        assert_eq!(grown.position(), 9, "reader must stop at limit plus one");
    }

    #[cfg(unix)]
    #[test]
    fn non_regular_plan_file_is_rejected_before_reading() {
        let error = read_plan_bytes(Path::new("/dev/null"))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("not a regular file"), "{error}");
    }

    #[test]
    fn atomic_publication_does_not_clobber_without_permission() {
        let test_directory = std::env::temp_dir().join(format!(
            "tape-decode-numeric-plan-test-{}-{}",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&test_directory).unwrap();
        let output = test_directory.join("plan.tdnp");

        publish_atomically(&output, b"first", false).unwrap();
        let error = publish_atomically(&output, b"second", false)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("without overwriting"), "{error}");
        assert_eq!(std::fs::read(&output).unwrap(), b"first");

        #[cfg(unix)]
        {
            publish_atomically(&output, b"replacement", true).unwrap();
            assert_eq!(std::fs::read(&output).unwrap(), b"replacement");
        }

        assert_eq!(std::fs::read_dir(&test_directory).unwrap().count(), 1);
        std::fs::remove_file(output).unwrap();
        std::fs::remove_dir(test_directory).unwrap();
    }
}
