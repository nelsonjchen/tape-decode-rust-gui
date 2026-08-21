use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::hash::blake3_file;
use crate::model::{
    read_json, write_json_atomic, CoordinatorState, JobSpec, JobState, RunManifest,
};

#[derive(Debug)]
struct Shard {
    job: JobSpec,
    runner_id: String,
    attempt: u32,
    luma: PathBuf,
    chroma: PathBuf,
    metadata: Value,
    fields: Vec<Value>,
    field_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SeamEvidence {
    pub left_job: String,
    pub right_job: String,
    pub boundary_sample: u64,
    pub left_match_index: usize,
    pub right_match_index: usize,
    pub handoff_file_loc: u64,
    pub consecutive_matches: usize,
    pub last_luma_msre: f64,
    pub last_chroma_msre: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectionEvidence {
    pub job_id: String,
    pub runner_id: String,
    pub attempt: u32,
    pub start_field: usize,
    pub end_field_exclusive: usize,
    pub fields: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssemblyReport {
    pub run_id: String,
    pub output_prefix: PathBuf,
    pub fields: usize,
    pub seams: Vec<SeamEvidence>,
    pub selections: Vec<SelectionEvidence>,
    pub luma_blake3: String,
    pub luma_length: u64,
    pub chroma_blake3: String,
    pub chroma_length: u64,
    pub metadata_blake3: String,
    pub metadata_length: u64,
}

fn artifact_path(root: &Path, job_id: &str, lease_id: &str, kind: &str) -> PathBuf {
    root.join("coordinator/uploads")
        .join(job_id)
        .join(lease_id)
        .join(kind)
}

fn value_u64(field: &Value, key: &str) -> Result<u64> {
    field
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("field is missing integer {key}"))
}

fn value_i64(field: &Value, key: &str) -> Result<i64> {
    field
        .get(key)
        .and_then(Value::as_i64)
        .with_context(|| format!("field is missing integer {key}"))
}

fn value_bool(field: &Value, key: &str) -> Result<bool> {
    field
        .get(key)
        .and_then(Value::as_bool)
        .with_context(|| format!("field is missing boolean {key}"))
}

fn load_shards(manifest: &RunManifest, state: &CoordinatorState) -> Result<Vec<Shard>> {
    let mut shards = Vec::new();
    for record in &state.jobs {
        let JobState::Complete {
            runner_id,
            lease_id,
            attempt,
            ..
        } = &record.state
        else {
            bail!("job {} is not complete", record.spec.id);
        };
        let metadata_path =
            artifact_path(&manifest.work_root, &record.spec.id, lease_id, "metadata");
        let metadata: Value = read_json(&metadata_path)?;
        let fields = metadata
            .get("fields")
            .and_then(Value::as_array)
            .context("metadata has no fields array")?
            .clone();
        let video = metadata
            .get("videoParameters")
            .context("metadata has no videoParameters")?;
        let width = value_u64(video, "fieldWidth")?;
        let height = value_u64(video, "fieldHeight")?;
        let field_bytes = width
            .checked_mul(height)
            .and_then(|samples| samples.checked_mul(2))
            .context("field byte size overflow")?;
        anyhow::ensure!(
            width == manifest.output_geometry.field_width
                && height == manifest.output_geometry.field_height
                && manifest.output_geometry.bytes_per_sample == 2,
            "decoded output geometry does not match manifest"
        );
        let luma = artifact_path(&manifest.work_root, &record.spec.id, lease_id, "luma");
        let chroma = artifact_path(&manifest.work_root, &record.spec.id, lease_id, "chroma");
        let expected = field_bytes * fields.len() as u64;
        anyhow::ensure!(
            fs::metadata(&luma)?.len() == expected,
            "luma geometry mismatch"
        );
        anyhow::ensure!(
            fs::metadata(&chroma)?.len() == expected,
            "chroma geometry mismatch"
        );
        shards.push(Shard {
            job: record.spec.clone(),
            runner_id: runner_id.clone(),
            attempt: *attempt,
            luma,
            chroma,
            metadata,
            fields,
            field_bytes,
        });
    }
    shards.sort_by_key(|shard| shard.job.ordinal);
    for (expected, shard) in shards.iter().enumerate() {
        anyhow::ensure!(
            shard.job.ordinal == expected,
            "job ordinals are not contiguous"
        );
    }
    Ok(shards)
}

fn lower_bound(fields: &[Value], target: u64) -> Result<usize> {
    let mut low = 0usize;
    let mut high = fields.len();
    while low < high {
        let mid = (low + high) / 2;
        if value_u64(&fields[mid], "fileLoc")? < target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    Ok(low)
}

fn first_frame_at_or_after(fields: &[Value], target: u64) -> Result<usize> {
    let mut index = lower_bound(fields, target)?;
    while index < fields.len() && !value_bool(&fields[index], "isFirstField")? {
        index += 1;
    }
    if index == fields.len() {
        bail!("no first field at or after sample {target}");
    }
    Ok(index)
}

fn frame_end_at_or_after(fields: &[Value], target: u64) -> Result<usize> {
    let mut index = lower_bound(fields, target)?;
    while index < fields.len() && !value_bool(&fields[index], "isFirstField")? {
        index += 1;
    }
    Ok(index)
}

fn read_field(path: &Path, index: usize, field_bytes: u64) -> Result<Vec<u16>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(index as u64 * field_bytes))?;
    let mut bytes = vec![0u8; field_bytes as usize];
    file.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn wrapped_msre(reference: &[u16], candidate: &[u16], trim_fraction: f64) -> f64 {
    let mut squared = reference
        .iter()
        .zip(candidate)
        .map(|(&left, &right)| {
            let difference = (i32::from(left) - i32::from(right)).abs();
            let wrapped = difference.min(65_536 - difference) as u32;
            wrapped * wrapped
        })
        .collect::<Vec<_>>();
    if squared.is_empty() {
        return 0.0;
    }
    let keep = (squared.len() - (squared.len() as f64 * trim_fraction) as usize).max(1);
    if keep < squared.len() {
        squared.select_nth_unstable(keep);
    }
    let sum: u64 = squared[..keep].iter().map(|value| u64::from(*value)).sum();
    (sum as f64 / keep as f64).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn find_handoff(
    left: &Shard,
    right: &Shard,
    boundary: u64,
    search_end: u64,
    samples_per_field: u64,
    consecutive: usize,
    threshold: f64,
    trim_fraction: f64,
) -> Result<SeamEvidence> {
    let tolerance = samples_per_field / 2;
    let mut left_index = lower_bound(&left.fields, boundary)?;
    let mut right_index = lower_bound(&right.fields, boundary.saturating_sub(samples_per_field))?;
    let mut run = 0usize;
    let mut last_luma = f64::INFINITY;
    let mut last_chroma = f64::INFINITY;
    while left_index < left.fields.len() {
        let left_loc = value_u64(&left.fields[left_index], "fileLoc")?;
        if left_loc >= search_end {
            break;
        }
        while right_index < right.fields.len()
            && value_u64(&right.fields[right_index], "fileLoc")?.saturating_add(tolerance)
                < left_loc
        {
            right_index += 1;
        }
        if right_index >= right.fields.len() {
            break;
        }
        let right_loc = value_u64(&right.fields[right_index], "fileLoc")?;
        let structural = left_loc.abs_diff(right_loc) <= tolerance
            && value_bool(&left.fields[left_index], "isFirstField")?
                == value_bool(&right.fields[right_index], "isFirstField")?
            && value_i64(&left.fields[left_index], "syncConf")?
                == value_i64(&right.fields[right_index], "syncConf")?;
        let matched = if structural {
            let left_luma = read_field(&left.luma, left_index, left.field_bytes)?;
            let right_luma = read_field(&right.luma, right_index, right.field_bytes)?;
            last_luma = wrapped_msre(&left_luma, &right_luma, trim_fraction);
            let left_chroma = read_field(&left.chroma, left_index, left.field_bytes)?;
            let right_chroma = read_field(&right.chroma, right_index, right.field_bytes)?;
            last_chroma = wrapped_msre(&left_chroma, &right_chroma, trim_fraction);
            last_luma < threshold && last_chroma < threshold
        } else {
            false
        };
        if matched {
            run += 1;
            if run >= consecutive {
                return Ok(SeamEvidence {
                    left_job: left.job.id.clone(),
                    right_job: right.job.id.clone(),
                    boundary_sample: boundary,
                    left_match_index: left_index,
                    right_match_index: right_index,
                    handoff_file_loc: left_loc,
                    consecutive_matches: run,
                    last_luma_msre: last_luma,
                    last_chroma_msre: last_chroma,
                });
            }
        } else {
            run = 0;
        }
        left_index += 1;
        if right_loc <= left_loc.saturating_add(tolerance) {
            right_index += 1;
        }
    }
    bail!(
        "no {}-field match between {} and {} in [{boundary},{search_end})",
        consecutive,
        left.job.id,
        right.job.id
    )
}

struct Blake3Writer<W> {
    inner: W,
    hasher: blake3::Hasher,
    length: u64,
}

impl<W: Write> Blake3Writer<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: blake3::Hasher::new(),
            length: 0,
        }
    }

    fn finish(mut self) -> Result<(String, u64)> {
        self.flush()?;
        Ok((self.hasher.finalize().to_hex().to_string(), self.length))
    }
}

impl<W: Write> Write for Blake3Writer<W> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buffer)?;
        self.hasher.update(&buffer[..written]);
        self.length += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn copy_fields<W: Write>(
    input: &Path,
    output: &mut W,
    start: usize,
    end: usize,
    field_bytes: u64,
) -> Result<()> {
    if start > end {
        bail!("invalid field selection [{start},{end})");
    }
    let mut input = BufReader::new(File::open(input)?);
    input.seek(SeekFrom::Start(start as u64 * field_bytes))?;
    let mut limited = input.take((end - start) as u64 * field_bytes);
    let copied = std::io::copy(&mut limited, output)?;
    anyhow::ensure!(
        copied == (end - start) as u64 * field_bytes,
        "short TBC copy"
    );
    Ok(())
}

fn phase_id(first_field: bool, global_seq: usize) -> i64 {
    let second_phase = (global_seq / 2).is_multiple_of(2);
    match (first_field, second_phase) {
        (true, true) => 1,
        (false, false) => 2,
        (true, false) => 3,
        (false, true) => 4,
    }
}

pub fn run(
    manifest_path: PathBuf,
    state_path: PathBuf,
    output_prefix: PathBuf,
) -> Result<AssemblyReport> {
    let manifest: RunManifest = read_json(&manifest_path)?;
    let state: CoordinatorState = read_json(&state_path)?;
    anyhow::ensure!(
        manifest.run_id == state.run_id,
        "manifest/state run id mismatch"
    );
    let shards = load_shards(&manifest, &state)?;
    if shards.is_empty() {
        bail!("no shards to assemble");
    }
    let mut starts = vec![0usize; shards.len()];
    let mut ends = vec![0usize; shards.len()];
    starts[0] = first_frame_at_or_after(&shards[0].fields, manifest.ranges.canonical_start_sample)?;
    let mut seams = Vec::new();
    for index in 0..shards.len() - 1 {
        let boundary = shards[index + 1].job.canonical_start_sample;
        let seam = find_handoff(
            &shards[index],
            &shards[index + 1],
            boundary,
            boundary.saturating_add(manifest.ranges.guard_samples),
            manifest.ranges.samples_per_field,
            manifest.decode.mt_overlap_count,
            manifest.decode.mt_threshold,
            manifest.decode.mt_trim_fraction,
        )?;
        ends[index] = seam.left_match_index + 1;
        starts[index + 1] = seam.right_match_index + 1;
        seams.push(seam);
    }
    ends[shards.len() - 1] = frame_end_at_or_after(
        &shards.last().unwrap().fields,
        manifest.ranges.canonical_end_sample,
    )?;

    let luma_path = output_prefix.with_extension("tbc");
    let chroma_path = output_prefix.with_file_name(format!(
        "{}_chroma.tbc",
        output_prefix.file_name().unwrap().to_string_lossy()
    ));
    let metadata_path = output_prefix.with_extension("tbc.json");
    for path in [&luma_path, &chroma_path, &metadata_path] {
        if path.exists() {
            bail!("refusing to overwrite {}", path.display());
        }
    }
    fs::create_dir_all(
        output_prefix
            .parent()
            .context("output prefix has no parent")?,
    )?;
    let mut luma_output = Blake3Writer::new(BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&luma_path)?,
    ));
    let mut chroma_output = Blake3Writer::new(BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&chroma_path)?,
    ));
    let mut output_fields = Vec::new();
    let mut selections = Vec::new();
    for (index, shard) in shards.iter().enumerate() {
        let start = starts[index];
        let end = ends[index];
        anyhow::ensure!(
            start <= end && end <= shard.fields.len(),
            "invalid selection for {}",
            shard.job.id
        );
        copy_fields(&shard.luma, &mut luma_output, start, end, shard.field_bytes)?;
        copy_fields(
            &shard.chroma,
            &mut chroma_output,
            start,
            end,
            shard.field_bytes,
        )?;
        for field in &shard.fields[start..end] {
            let mut field = field.clone();
            let seq = output_fields.len() + 1;
            field["seqNo"] = Value::from(seq as u64);
            field["fieldPhaseID"] = Value::from(phase_id(value_bool(&field, "isFirstField")?, seq));
            output_fields.push(field);
        }
        selections.push(SelectionEvidence {
            job_id: shard.job.id.clone(),
            runner_id: shard.runner_id.clone(),
            attempt: shard.attempt,
            start_field: start,
            end_field_exclusive: end,
            fields: end - start,
        });
    }
    let (luma_blake3, luma_length) = luma_output.finish()?;
    let (chroma_blake3, chroma_length) = chroma_output.finish()?;
    let mut metadata = shards[0].metadata.clone();
    metadata["fields"] = Value::Array(output_fields);
    metadata["videoParameters"]["numberOfSequentialFields"] =
        Value::from(metadata["fields"].as_array().unwrap().len() as u64);
    write_json_atomic(&metadata_path, &metadata)?;

    let (metadata_blake3, metadata_length) = blake3_file(&metadata_path)?;
    let report = AssemblyReport {
        run_id: manifest.run_id,
        output_prefix,
        fields: metadata["fields"].as_array().unwrap().len(),
        seams,
        selections,
        luma_blake3,
        luma_length,
        chroma_blake3,
        chroma_length,
        metadata_blake3,
        metadata_length,
    };
    let report_path = report.output_prefix.with_extension("assembly.json");
    write_json_atomic(&report_path, &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_shard(root: &Path, id: &str, values: &[u16]) -> Shard {
        let luma = root.join(format!("{id}.tbc"));
        let chroma = root.join(format!("{id}_chroma.tbc"));
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(&luma, &bytes).unwrap();
        fs::write(&chroma, &bytes).unwrap();
        let fields = (0..values.len())
            .map(|index| {
                serde_json::json!({
                    "fileLoc": 1_000 + index as u64 * 100,
                    "isFirstField": index.is_multiple_of(2),
                    "syncConf": 45
                })
            })
            .collect();
        Shard {
            job: JobSpec {
                id: id.into(),
                ordinal: 0,
                canonical_start_sample: 1_000,
                canonical_end_sample: 1_400,
                decode_start_sample: 900,
                decode_end_sample: 1_500,
            },
            runner_id: "test".into(),
            attempt: 1,
            luma,
            chroma,
            metadata: serde_json::json!({}),
            fields,
            field_bytes: 2,
        }
    }

    #[test]
    fn wrapped_metric_accepts_identical_and_rejects_large_difference() {
        assert_eq!(wrapped_msre(&[1, 2, 3], &[1, 2, 3], 0.1), 0.0);
        assert!(wrapped_msre(&[0; 100], &[1000; 100], 0.1) > 64.0);
    }

    #[test]
    fn synthetic_stitch_requires_two_matching_fields() {
        let root = std::env::temp_dir().join(format!("dist-stitch-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let left = synthetic_shard(&root, "left", &[10, 20, 30, 40]);
        let mut right = synthetic_shard(&root, "right", &[9_000, 20, 30, 40]);
        right.job.ordinal = 1;
        let seam = find_handoff(&left, &right, 1_000, 1_400, 100, 2, 64.0, 0.0).unwrap();
        assert_eq!(seam.left_match_index, 2);
        assert_eq!(seam.right_match_index, 2);

        fs::write(
            &right.luma,
            [9_000u16, 8_000, 7_000, 6_000]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(find_handoff(&left, &right, 1_000, 1_400, 100, 2, 64.0, 0.0).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
