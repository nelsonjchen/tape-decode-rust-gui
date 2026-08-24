//! Strict, architecture-parity comparison for complete decoder outputs.
//!
//! Unlike the tolerant `compare` command, this module does not perform signal
//! quality comparisons or normalize metadata. Raster outputs must have exactly
//! the same bytes and metadata must have exactly the same schema and values.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _, Result};
use serde::Deserialize;
use serde_json::Value;

const COPY_BUFFER_BYTES: usize = 1024 * 1024;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExactMetadata {
    fields: Vec<ExactField>,
    pcm_audio_parameters: ExactPcmAudioParameters,
    video_parameters: ExactVideoParameters,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExactPcmAudioParameters {
    bits: usize,
    is_little_endian: bool,
    is_signed: bool,
    sample_rate: usize,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExactVideoParameters {
    number_of_sequential_fields: usize,
    os_info: String,
    git_branch: String,
    git_commit: String,
    system: String,
    field_width: usize,
    sample_rate: f64,
    black_16b_ire: f64,
    white_16b_ire: f64,
    field_height: usize,
    colour_burst_start: i64,
    colour_burst_end: i64,
    active_video_start: i64,
    active_video_end: i64,
    tape_format: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExactField {
    is_first_field: bool,
    detected_first_field: bool,
    is_duplicate_field: bool,
    sync_conf: i64,
    seq_no: usize,
    disk_loc: f32,
    file_loc: u64,
    #[serde(rename = "fieldPhaseID")]
    field_phase_id: i64,
    vits_metrics: ExactVitsMetrics,
    drop_outs: Option<ExactDropOuts>,
    decode_faults: Option<i64>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactVitsMetrics {
    #[serde(rename = "wSNR")]
    w_snr: Option<f64>,
    #[serde(rename = "bPSNR")]
    b_psnr: Option<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExactDropOuts {
    field_line: Vec<usize>,
    startx: Vec<usize>,
    endx: Vec<usize>,
}

struct ParsedMetadata {
    raw: Value,
    typed: ExactMetadata,
}

#[derive(Default)]
struct MetadataFloats {
    f32_values: BTreeMap<String, f32>,
    f64_values: BTreeMap<String, f64>,
}

#[derive(Debug, PartialEq, Eq)]
struct JsonMismatch {
    path: String,
    expected: String,
    actual: String,
}

impl JsonMismatch {
    fn new(
        path: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ByteMismatch {
    byte_offset: u64,
    reference_byte: Option<u8>,
    candidate_byte: Option<u8>,
}

struct BinaryComparison {
    reference_len: u64,
    candidate_len: u64,
    reference_hash: blake3::Hash,
    candidate_hash: blake3::Hash,
    first_difference: Option<ByteMismatch>,
}

pub(crate) fn run(metadata: [PathBuf; 2], luma: [PathBuf; 2], chroma: [PathBuf; 2]) -> Result<()> {
    let reference_metadata = read_metadata(&metadata[0])?;
    let candidate_metadata = read_metadata(&metadata[1])?;
    validate_metadata(&metadata[0], &reference_metadata.typed)?;
    validate_metadata(&metadata[1], &candidate_metadata.typed)?;

    let video = &reference_metadata.typed.video_parameters;
    let field_samples = video
        .field_width
        .checked_mul(video.field_height)
        .context("reference field geometry overflows usize")?;
    if field_samples == 0 || video.number_of_sequential_fields == 0 {
        bail!(
            "invalid reference geometry: {} fields of {}x{} samples",
            video.number_of_sequential_fields,
            video.field_width,
            video.field_height
        );
    }

    let luma_field_bytes =
        infer_field_bytes(&luma[0], field_samples, video.number_of_sequential_fields)?;
    let chroma_field_bytes =
        infer_field_bytes(&chroma[0], field_samples, video.number_of_sequential_fields)?;

    let luma_result = compare_binary(&luma[0], &luma[1])?;
    let chroma_result = compare_binary(&chroma[0], &chroma[1])?;
    let metadata_difference = compare_metadata(
        &reference_metadata.raw,
        &candidate_metadata.raw,
        &reference_metadata.typed,
        &candidate_metadata.typed,
    );

    let mut failed = false;
    report_binary("luma", &luma_result, luma_field_bytes, &mut failed);
    report_binary("chroma", &chroma_result, chroma_field_bytes, &mut failed);
    match metadata_difference {
        None => println!("metadata ({}): OK", metadata[1].display()),
        Some(difference) => {
            println!("metadata ({}): FAILED", metadata[1].display());
            println!(
                "  first difference at {}: expected {}, got {}",
                difference.path, difference.expected, difference.actual
            );
            failed = true;
        }
    }

    if failed {
        bail!("exact verification failed");
    }
    Ok(())
}

fn read_metadata(path: &Path) -> Result<ParsedMetadata> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read metadata {}", path.display()))?;
    let raw = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse metadata JSON {}", path.display()))?;
    let typed = serde_json::from_str(&data).with_context(|| {
        format!(
            "metadata {} does not match the exact known schema",
            path.display()
        )
    })?;
    Ok(ParsedMetadata { raw, typed })
}

fn validate_metadata(path: &Path, metadata: &ExactMetadata) -> Result<()> {
    let declared = metadata.video_parameters.number_of_sequential_fields;
    if metadata.fields.len() != declared {
        bail!(
            "metadata {} has {} field entries but declares {declared}",
            path.display(),
            metadata.fields.len()
        );
    }
    Ok(())
}

fn infer_field_bytes(path: &Path, field_samples: usize, field_count: usize) -> Result<u64> {
    let file_len = std::fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();
    let field_samples = u64::try_from(field_samples).context("field size does not fit u64")?;
    let field_count = u64::try_from(field_count).context("field count does not fit u64")?;
    for bytes_per_sample in [2_u64, 4_u64] {
        let field_bytes = field_samples
            .checked_mul(bytes_per_sample)
            .context("field byte size overflows u64")?;
        let expected_len = field_bytes
            .checked_mul(field_count)
            .context("raster byte size overflows u64")?;
        if file_len == expected_len {
            return Ok(field_bytes);
        }
    }
    bail!(
        "reference raster {} has {file_len} bytes, inconsistent with {field_count} fields of {field_samples} samples in u16 or f32 format",
        path.display()
    )
}

fn compare_binary(reference: &Path, candidate: &Path) -> Result<BinaryComparison> {
    let reference_len = std::fs::metadata(reference)
        .with_context(|| format!("failed to stat {}", reference.display()))?
        .len();
    let candidate_len = std::fs::metadata(candidate)
        .with_context(|| format!("failed to stat {}", candidate.display()))?
        .len();
    let common_len = reference_len.min(candidate_len);
    let mut reference_reader = BufReader::new(
        File::open(reference).with_context(|| format!("failed to open {}", reference.display()))?,
    );
    let mut candidate_reader = BufReader::new(
        File::open(candidate).with_context(|| format!("failed to open {}", candidate.display()))?,
    );
    let mut reference_buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let mut candidate_buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let mut reference_hasher = blake3::Hasher::new();
    let mut candidate_hasher = blake3::Hasher::new();
    let mut first_difference = None;
    let mut offset = 0_u64;

    while offset < common_len {
        let remaining = usize::try_from((common_len - offset).min(COPY_BUFFER_BYTES as u64))
            .context("comparison chunk size does not fit usize")?;
        reference_reader
            .read_exact(&mut reference_buffer[..remaining])
            .with_context(|| format!("failed to read {}", reference.display()))?;
        candidate_reader
            .read_exact(&mut candidate_buffer[..remaining])
            .with_context(|| format!("failed to read {}", candidate.display()))?;
        reference_hasher.update(&reference_buffer[..remaining]);
        candidate_hasher.update(&candidate_buffer[..remaining]);
        if first_difference.is_none() {
            if let Some(index) = reference_buffer[..remaining]
                .iter()
                .zip(&candidate_buffer[..remaining])
                .position(|(expected, actual)| expected != actual)
            {
                first_difference = Some(ByteMismatch {
                    byte_offset: offset + index as u64,
                    reference_byte: Some(reference_buffer[index]),
                    candidate_byte: Some(candidate_buffer[index]),
                });
            }
        }
        offset += remaining as u64;
    }

    if reference_len < candidate_len {
        let extra = hash_remaining(
            &mut candidate_reader,
            candidate_len - common_len,
            &mut candidate_buffer,
            &mut candidate_hasher,
            candidate,
        )?;
        if first_difference.is_none() {
            first_difference = Some(ByteMismatch {
                byte_offset: common_len,
                reference_byte: None,
                candidate_byte: Some(extra),
            });
        }
    } else if reference_len > candidate_len {
        let extra = hash_remaining(
            &mut reference_reader,
            reference_len - common_len,
            &mut reference_buffer,
            &mut reference_hasher,
            reference,
        )?;
        if first_difference.is_none() {
            first_difference = Some(ByteMismatch {
                byte_offset: common_len,
                reference_byte: Some(extra),
                candidate_byte: None,
            });
        }
    }

    Ok(BinaryComparison {
        reference_len,
        candidate_len,
        reference_hash: reference_hasher.finalize(),
        candidate_hash: candidate_hasher.finalize(),
        first_difference,
    })
}

fn hash_remaining(
    reader: &mut BufReader<File>,
    mut remaining: u64,
    buffer: &mut [u8],
    hasher: &mut blake3::Hasher,
    path: &Path,
) -> Result<u8> {
    let mut first_byte = None;
    while remaining > 0 {
        let chunk_len = usize::try_from(remaining.min(buffer.len() as u64))
            .context("hash chunk size does not fit usize")?;
        reader
            .read_exact(&mut buffer[..chunk_len])
            .with_context(|| format!("failed to read {}", path.display()))?;
        first_byte.get_or_insert(buffer[0]);
        hasher.update(&buffer[..chunk_len]);
        remaining -= chunk_len as u64;
    }
    first_byte.context("length mismatch had no extra byte")
}

fn report_binary(kind: &str, result: &BinaryComparison, field_bytes: u64, failed: &mut bool) {
    match result.first_difference.as_ref() {
        None => println!("{kind}: OK"),
        Some(difference) => {
            println!("{kind}: FAILED");
            if result.reference_len != result.candidate_len {
                println!(
                    "  length: expected {} bytes, got {} bytes",
                    result.reference_len, result.candidate_len
                );
            }
            let field_index = difference.byte_offset / field_bytes;
            let field_offset = difference.byte_offset % field_bytes;
            println!(
                "  first differing byte: field[{field_index}] offset {field_offset} (global offset {}): expected {}, got {}",
                difference.byte_offset,
                format_byte(difference.reference_byte),
                format_byte(difference.candidate_byte)
            );
            *failed = true;
        }
    }
    println!(
        "  reference: {} bytes, BLAKE3 {}",
        result.reference_len, result.reference_hash
    );
    println!(
        "  candidate: {} bytes, BLAKE3 {}",
        result.candidate_len, result.candidate_hash
    );
}

fn format_byte(byte: Option<u8>) -> String {
    match byte {
        Some(value) => format!("0x{value:02x}"),
        None => "<EOF>".to_string(),
    }
}

fn compare_metadata(
    expected_raw: &Value,
    actual_raw: &Value,
    expected: &ExactMetadata,
    actual: &ExactMetadata,
) -> Option<JsonMismatch> {
    let expected_floats = collect_metadata_floats(expected);
    let actual_floats = collect_metadata_floats(actual);
    compare_json_values(
        expected_raw,
        actual_raw,
        "$",
        &expected_floats,
        &actual_floats,
    )
}

fn collect_metadata_floats(metadata: &ExactMetadata) -> MetadataFloats {
    let mut floats = MetadataFloats::default();
    for (index, field) in metadata.fields.iter().enumerate() {
        let path = format!("$.fields[{index}]");
        floats
            .f32_values
            .insert(format!("{path}.diskLoc"), field.disk_loc);
        if let Some(value) = field.vits_metrics.w_snr {
            floats
                .f64_values
                .insert(format!("{path}.vitsMetrics.wSNR"), value);
        }
        if let Some(value) = field.vits_metrics.b_psnr {
            floats
                .f64_values
                .insert(format!("{path}.vitsMetrics.bPSNR"), value);
        }
    }
    let video = &metadata.video_parameters;
    floats.f64_values.insert(
        "$.videoParameters.sampleRate".to_string(),
        video.sample_rate,
    );
    floats.f64_values.insert(
        "$.videoParameters.black16bIre".to_string(),
        video.black_16b_ire,
    );
    floats.f64_values.insert(
        "$.videoParameters.white16bIre".to_string(),
        video.white_16b_ire,
    );
    floats
}

/// Recursively compare JSON values in a stable key order. Numbers are compared
/// according to the declared metadata schema so alternate JSON spellings of
/// the same float are accepted while distinct IEEE-754 values are not.
fn compare_json_values(
    expected: &Value,
    actual: &Value,
    path: &str,
    expected_floats: &MetadataFloats,
    actual_floats: &MetadataFloats,
) -> Option<JsonMismatch> {
    match (expected, actual) {
        (Value::Object(expected), Value::Object(actual)) => {
            let expected_keys: BTreeSet<_> = expected.keys().collect();
            let actual_keys: BTreeSet<_> = actual.keys().collect();
            if let Some(key) = expected_keys.difference(&actual_keys).next() {
                return Some(JsonMismatch::new(
                    child_path(path, key),
                    "present",
                    "missing",
                ));
            }
            if let Some(key) = actual_keys.difference(&expected_keys).next() {
                return Some(JsonMismatch::new(
                    child_path(path, key),
                    "missing",
                    "present",
                ));
            }
            for key in expected_keys {
                let child = child_path(path, key);
                if let Some(difference) = compare_json_values(
                    &expected[key],
                    &actual[key],
                    &child,
                    expected_floats,
                    actual_floats,
                ) {
                    return Some(difference);
                }
            }
            None
        }
        (Value::Array(expected), Value::Array(actual)) => {
            if expected.len() != actual.len() {
                return Some(JsonMismatch::new(
                    path,
                    format!("array length {}", expected.len()),
                    format!("array length {}", actual.len()),
                ));
            }
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                if let Some(difference) = compare_json_values(
                    expected,
                    actual,
                    &format!("{path}[{index}]"),
                    expected_floats,
                    actual_floats,
                ) {
                    return Some(difference);
                }
            }
            None
        }
        (Value::Null, Value::Null) => None,
        (Value::Bool(expected), Value::Bool(actual)) => compare_eq(path, expected, actual),
        (Value::String(expected), Value::String(actual)) => compare_eq(path, expected, actual),
        (Value::Number(expected), Value::Number(actual)) => {
            compare_json_number(path, expected, actual, expected_floats, actual_floats)
        }
        _ => Some(JsonMismatch::new(
            path,
            json_type(expected),
            json_type(actual),
        )),
    }
}

fn compare_json_number(
    path: &str,
    expected: &serde_json::Number,
    actual: &serde_json::Number,
    expected_floats: &MetadataFloats,
    actual_floats: &MetadataFloats,
) -> Option<JsonMismatch> {
    if let Some(expected) = expected_floats.f32_values.get(path) {
        let actual = actual_floats
            .f32_values
            .get(path)
            .expect("strict schemas disagree about an f32 path");
        compare_f32(path, *expected, *actual)
    } else if let Some(expected) = expected_floats.f64_values.get(path) {
        let actual = actual_floats
            .f64_values
            .get(path)
            .expect("strict schemas disagree about an f64 path");
        compare_f64(path, *expected, *actual)
    } else {
        (expected != actual)
            .then(|| JsonMismatch::new(path, expected.to_string(), actual.to_string()))
    }
}

fn child_path(parent: &str, key: &str) -> String {
    if key.chars().all(|c| c == '_' || c.is_ascii_alphanumeric()) {
        format!("{parent}.{key}")
    } else {
        format!("{parent}[{key:?}]")
    }
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn compare_eq<T: PartialEq + Debug>(path: &str, expected: T, actual: T) -> Option<JsonMismatch> {
    (expected != actual)
        .then(|| JsonMismatch::new(path, format!("{expected:?}"), format!("{actual:?}")))
}

fn compare_f32(path: &str, expected: f32, actual: f32) -> Option<JsonMismatch> {
    (expected.to_bits() != actual.to_bits()).then(|| {
        JsonMismatch::new(
            path,
            format!("{expected:?} (f32 bits 0x{:08x})", expected.to_bits()),
            format!("{actual:?} (f32 bits 0x{:08x})", actual.to_bits()),
        )
    })
}

fn compare_f64(path: &str, expected: f64, actual: f64) -> Option<JsonMismatch> {
    (expected.to_bits() != actual.to_bits()).then(|| {
        JsonMismatch::new(
            path,
            format!("{expected:?} (f64 bits 0x{:016x})", expected.to_bits()),
            format!("{actual:?} (f64 bits 0x{:016x})", actual.to_bits()),
        )
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    fn temporary_file(label: &str, bytes: &[u8]) -> PathBuf {
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "tape-decode-exact-verify-{}-{id}-{label}",
            std::process::id()
        ));
        fs::write(&path, bytes).unwrap();
        path
    }

    fn metadata(disk_loc: &str, extra_field_key: &str) -> String {
        format!(
            r#"{{
              "fields": [{{
                "isFirstField": true,
                "detectedFirstField": true,
                "isDuplicateField": false,
                "syncConf": 45,
                "seqNo": 1,
                "diskLoc": {disk_loc},
                "fileLoc": 100,
                "fieldPhaseID": 1,
                "vitsMetrics": {{"wSNR": 2.5, "bPSNR": 3.5}}{extra_field_key}
              }}],
              "pcmAudioParameters": {{
                "bits": 16, "isLittleEndian": true, "isSigned": true, "sampleRate": 0
              }},
              "videoParameters": {{
                "numberOfSequentialFields": 1,
                "osInfo": "",
                "gitBranch": "UNKNOWN",
                "gitCommit": "UNKNOWN",
                "system": "NTSC",
                "fieldWidth": 2,
                "sampleRate": 4.0,
                "black16bIre": 5.0,
                "white16bIre": 6.0,
                "fieldHeight": 1,
                "colourBurstStart": 0,
                "colourBurstEnd": 1,
                "activeVideoStart": 0,
                "activeVideoEnd": 2,
                "tapeFormat": "TAPE"
              }}
            }}"#
        )
    }

    fn parse(data: &str) -> ParsedMetadata {
        ParsedMetadata {
            raw: serde_json::from_str(data).unwrap(),
            typed: serde_json::from_str(data).unwrap(),
        }
    }

    #[test]
    fn binary_comparison_reports_first_byte_and_eof() {
        let reference = temporary_file("reference.tbc", &[0, 1, 2, 3, 4, 5]);
        let changed = temporary_file("changed.tbc", &[0, 1, 2, 9, 4, 5]);
        let short = temporary_file("short.tbc", &[0, 1, 2, 3]);

        let changed_result = compare_binary(&reference, &changed).unwrap();
        assert_eq!(
            changed_result.first_difference,
            Some(ByteMismatch {
                byte_offset: 3,
                reference_byte: Some(3),
                candidate_byte: Some(9),
            })
        );
        assert_eq!(
            changed_result.reference_hash,
            blake3::hash(&[0, 1, 2, 3, 4, 5])
        );
        assert_eq!(
            changed_result.candidate_hash,
            blake3::hash(&[0, 1, 2, 9, 4, 5])
        );
        let short_result = compare_binary(&reference, &short).unwrap();
        assert_eq!(
            short_result.first_difference,
            Some(ByteMismatch {
                byte_offset: 4,
                reference_byte: Some(4),
                candidate_byte: None,
            })
        );
        assert_eq!(
            short_result.reference_hash,
            blake3::hash(&[0, 1, 2, 3, 4, 5])
        );
        assert_eq!(short_result.candidate_hash, blake3::hash(&[0, 1, 2, 3]));

        fs::remove_file(reference).unwrap();
        fs::remove_file(changed).unwrap();
        fs::remove_file(short).unwrap();
    }

    #[test]
    fn metadata_float_comparison_uses_declared_f32_bits() {
        let expected = parse(&metadata("0.0", ""));
        let actual = parse(&metadata("-0.0", ""));
        let difference =
            compare_metadata(&expected.raw, &actual.raw, &expected.typed, &actual.typed).unwrap();

        assert_eq!(difference.path, "$.fields[0].diskLoc");
        assert!(difference.expected.contains("0x00000000"));
        assert!(difference.actual.contains("0x80000000"));
    }

    #[test]
    fn metadata_object_order_and_whitespace_do_not_matter() {
        let expected = parse(&metadata("1.25", ""));
        let reordered = metadata("1.25", "")
            .replace("\"isFirstField\": true,", "\"isFirstField\":true,")
            .replace(
                "\"detectedFirstField\": true,\n                \"isDuplicateField\": false,",
                "\"isDuplicateField\": false,\n                \"detectedFirstField\": true,",
            );
        let actual = parse(&reordered);

        assert_eq!(
            compare_metadata(&expected.raw, &actual.raw, &expected.typed, &actual.typed),
            None
        );
    }

    #[test]
    fn metadata_distinguishes_missing_from_null() {
        let without_wsnr = metadata("1.25", "").replace("\"wSNR\": 2.5, ", "");
        let null_wsnr = metadata("1.25", "").replace("\"wSNR\": 2.5", "\"wSNR\": null");
        let expected = parse(&without_wsnr);
        let actual = parse(&null_wsnr);

        let difference =
            compare_metadata(&expected.raw, &actual.raw, &expected.typed, &actual.typed).unwrap();
        assert_eq!(difference.path, "$.fields[0].vitsMetrics.wSNR");
        assert_eq!(difference.expected, "missing");
        assert_eq!(difference.actual, "present");
    }

    #[test]
    fn exact_schema_rejects_unknown_fields() {
        let data = metadata("1.25", ", \"newDecoderField\": 1");
        let error = serde_json::from_str::<ExactMetadata>(&data).unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown field `newDecoderField`"));
    }

    #[test]
    fn complete_verification_returns_error_for_a_raster_mismatch() {
        let metadata_path = temporary_file("metadata.json", metadata("1.25", "").as_bytes());
        let reference_luma = temporary_file("reference-luma.tbc", &[0, 1, 2, 3]);
        let candidate_luma = temporary_file("candidate-luma.tbc", &[0, 1, 9, 3]);
        let chroma = temporary_file("chroma.tbc", &[4, 5, 6, 7]);

        let result = run(
            [metadata_path.clone(), metadata_path.clone()],
            [reference_luma.clone(), candidate_luma.clone()],
            [chroma.clone(), chroma.clone()],
        );
        assert!(result.is_err());

        for path in [metadata_path, reference_luma, candidate_luma, chroma] {
            fs::remove_file(path).unwrap();
        }
    }
}
