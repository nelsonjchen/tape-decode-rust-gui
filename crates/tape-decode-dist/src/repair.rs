use std::fs::{self, OpenOptions};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::assemble::{
    copy_fields, lower_bound, phase_id, read_field, value_bool, value_i64, value_u64, wrapped_msre,
    Blake3Writer,
};
use crate::hash::blake3_file;
use crate::model::{read_json, write_json_atomic};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RepairRange {
    pub id: String,
    pub first_failed_field: usize,
    pub last_failed_field: usize,
    pub target_start_file_loc: u64,
    pub target_end_file_loc: u64,
    pub canonical_start_sample: u64,
    pub canonical_end_sample: u64,
    pub decode_start_sample: u64,
    pub decode_end_sample: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RepairPlan {
    pub schema_version: u32,
    pub metadata: PathBuf,
    pub failed_fields: Vec<usize>,
    pub ranges: Vec<RepairRange>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchEvidence {
    pub left_index: usize,
    pub right_index: usize,
    pub file_loc: u64,
    pub consecutive_matches: usize,
    pub last_luma_msre: f64,
    pub last_chroma_msre: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RepairReport {
    pub base_prefix: PathBuf,
    pub repair_prefix: PathBuf,
    pub output_prefix: PathBuf,
    pub first_failed_field: usize,
    pub last_failed_field: usize,
    pub left_handoff: MatchEvidence,
    pub right_handoff: MatchEvidence,
    pub base_fields_removed: usize,
    pub repair_fields_inserted: usize,
    pub output_fields: usize,
    pub luma_blake3: String,
    pub luma_length: u64,
    pub chroma_blake3: String,
    pub chroma_length: u64,
    pub metadata_blake3: String,
    pub metadata_length: u64,
}

struct DecodedOutput {
    prefix: PathBuf,
    luma: PathBuf,
    chroma: PathBuf,
    metadata: Value,
    fields: Vec<Value>,
    field_bytes: u64,
}

#[derive(Clone, Copy)]
enum MatchSelection {
    First,
    Last,
}

fn output_paths(prefix: &Path) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let name = prefix
        .file_name()
        .context("output prefix has no file name")?
        .to_string_lossy();
    Ok((
        prefix.with_extension("tbc"),
        prefix.with_file_name(format!("{name}_chroma.tbc")),
        prefix.with_extension("tbc.json"),
    ))
}

impl DecodedOutput {
    fn load(prefix: PathBuf) -> Result<Self> {
        let (luma, chroma, metadata_path) = output_paths(&prefix)?;
        let metadata: Value = read_json(&metadata_path)?;
        let fields = metadata
            .get("fields")
            .and_then(Value::as_array)
            .context("metadata has no fields array")?
            .clone();
        let video = metadata
            .get("videoParameters")
            .context("metadata has no videoParameters")?;
        let field_bytes = value_u64(video, "fieldWidth")?
            .checked_mul(value_u64(video, "fieldHeight")?)
            .and_then(|samples| samples.checked_mul(2))
            .context("field byte size overflow")?;
        let expected = field_bytes
            .checked_mul(fields.len() as u64)
            .context("decoded output size overflow")?;
        anyhow::ensure!(
            fs::metadata(&luma)?.len() == expected,
            "luma geometry mismatch"
        );
        anyhow::ensure!(
            fs::metadata(&chroma)?.len() == expected,
            "chroma geometry mismatch"
        );
        Ok(Self {
            prefix,
            luma,
            chroma,
            metadata,
            fields,
            field_bytes,
        })
    }
}

fn cluster_failures(failed_fields: &[usize], max_gap: usize) -> Vec<(usize, usize)> {
    let mut clusters = Vec::new();
    let Some(&first) = failed_fields.first() else {
        return clusters;
    };
    let mut start = first;
    let mut end = first;
    for &field in &failed_fields[1..] {
        if field.saturating_sub(end) > max_gap {
            clusters.push((start, end));
            start = field;
        }
        end = field;
    }
    clusters.push((start, end));
    clusters
}

fn merge_overlapping_ranges(ranges: Vec<RepairRange>) -> Vec<RepairRange> {
    let mut merged: Vec<RepairRange> = Vec::new();
    for range in ranges {
        if let Some(current) = merged.last_mut() {
            if range.decode_start_sample <= current.decode_end_sample {
                current.last_failed_field = range.last_failed_field;
                current.target_end_file_loc = range.target_end_file_loc;
                current.canonical_end_sample =
                    current.canonical_end_sample.max(range.canonical_end_sample);
                current.decode_end_sample = current.decode_end_sample.max(range.decode_end_sample);
                continue;
            }
        }
        merged.push(range);
    }
    for (ordinal, range) in merged.iter_mut().enumerate() {
        range.id = format!("repair-{ordinal:04}");
    }
    merged
}

#[allow(clippy::too_many_arguments)]
pub fn plan(
    metadata_path: PathBuf,
    mut failed_fields: Vec<usize>,
    out: PathBuf,
    cluster_gap_fields: usize,
    canonical_pad_fields: u64,
    guard_seconds: u64,
    sample_clock: u64,
    samples_per_field: u64,
    fixture_samples: Option<u64>,
) -> Result<RepairPlan> {
    if out.exists() {
        bail!("refusing to overwrite {}", out.display());
    }
    anyhow::ensure!(
        sample_clock > 0 && samples_per_field > 0,
        "sample rates must be positive"
    );
    let metadata: Value = read_json(&metadata_path)?;
    let fields = metadata
        .get("fields")
        .and_then(Value::as_array)
        .context("metadata has no fields array")?;
    failed_fields.sort_unstable();
    failed_fields.dedup();
    anyhow::ensure!(!failed_fields.is_empty(), "no failed fields supplied");
    anyhow::ensure!(
        failed_fields.last().copied().unwrap() < fields.len(),
        "failed field index exceeds metadata length"
    );
    let pad_samples = canonical_pad_fields.saturating_mul(samples_per_field);
    let guard_samples = guard_seconds.saturating_mul(sample_clock);
    let ranges = cluster_failures(&failed_fields, cluster_gap_fields)
        .into_iter()
        .enumerate()
        .map(|(ordinal, (first, last))| -> Result<RepairRange> {
            let target_start = value_u64(&fields[first], "fileLoc")?;
            let target_end = value_u64(&fields[last], "fileLoc")?;
            let canonical_start = target_start.saturating_sub(pad_samples);
            let canonical_end = target_end
                .saturating_add(samples_per_field)
                .saturating_add(pad_samples);
            let decode_start = canonical_start.saturating_sub(guard_samples);
            let mut decode_end = canonical_end.saturating_add(guard_samples);
            if let Some(limit) = fixture_samples {
                decode_end = decode_end.min(limit);
            }
            Ok(RepairRange {
                id: format!("repair-{ordinal:04}"),
                first_failed_field: first,
                last_failed_field: last,
                target_start_file_loc: target_start,
                target_end_file_loc: target_end,
                canonical_start_sample: canonical_start,
                canonical_end_sample: canonical_end,
                decode_start_sample: decode_start,
                decode_end_sample: decode_end,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let ranges = merge_overlapping_ranges(ranges);
    let plan = RepairPlan {
        schema_version: 1,
        metadata: metadata_path,
        failed_fields,
        ranges,
    };
    write_json_atomic(&out, &plan)?;
    Ok(plan)
}

#[allow(clippy::too_many_arguments)]
fn find_match_run(
    left: &DecodedOutput,
    right: &DecodedOutput,
    start_loc: u64,
    end_loc: u64,
    samples_per_field: u64,
    consecutive: usize,
    threshold: f64,
    trim_fraction: f64,
    selection: MatchSelection,
) -> Result<MatchEvidence> {
    anyhow::ensure!(consecutive > 0, "consecutive match count must be positive");
    let tolerance = samples_per_field / 2;
    let mut left_index = lower_bound(&left.fields, start_loc)?;
    let mut right_index = lower_bound(&right.fields, start_loc.saturating_sub(tolerance))?;
    let mut run = 0usize;
    let mut last_luma = f64::INFINITY;
    let mut last_chroma = f64::INFINITY;
    let mut last_evidence = None;
    while left_index < left.fields.len() {
        let left_loc = value_u64(&left.fields[left_index], "fileLoc")?;
        if left_loc >= end_loc {
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
            last_luma = wrapped_msre(
                &read_field(&left.luma, left_index, left.field_bytes)?,
                &read_field(&right.luma, right_index, right.field_bytes)?,
                trim_fraction,
            );
            last_chroma = wrapped_msre(
                &read_field(&left.chroma, left_index, left.field_bytes)?,
                &read_field(&right.chroma, right_index, right.field_bytes)?,
                trim_fraction,
            );
            last_luma < threshold && last_chroma < threshold
        } else {
            false
        };
        if matched {
            run += 1;
            if run >= consecutive {
                let evidence = MatchEvidence {
                    left_index,
                    right_index,
                    file_loc: left_loc,
                    consecutive_matches: run,
                    last_luma_msre: last_luma,
                    last_chroma_msre: last_chroma,
                };
                if matches!(selection, MatchSelection::First) {
                    return Ok(evidence);
                }
                last_evidence = Some(evidence);
            }
        } else {
            run = 0;
        }
        left_index += 1;
        if right_loc <= left_loc.saturating_add(tolerance) {
            right_index += 1;
        }
    }
    last_evidence.with_context(|| {
        format!(
            "no {consecutive}-field match between {} and {} in [{start_loc},{end_loc})",
            left.prefix.display(),
            right.prefix.display()
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn apply(
    base_prefix: PathBuf,
    repair_prefix: PathBuf,
    output_prefix: PathBuf,
    first_failed_field: usize,
    last_failed_field: usize,
    search_fields: u64,
    samples_per_field: u64,
    consecutive_matches: usize,
    threshold: f64,
    trim_fraction: f64,
) -> Result<RepairReport> {
    let base = DecodedOutput::load(base_prefix.clone())?;
    let repair = DecodedOutput::load(repair_prefix.clone())?;
    anyhow::ensure!(
        first_failed_field <= last_failed_field && last_failed_field < base.fields.len(),
        "invalid failed field range"
    );
    anyhow::ensure!(
        base.field_bytes == repair.field_bytes,
        "output geometry differs"
    );
    let failed_start_loc = value_u64(&base.fields[first_failed_field], "fileLoc")?;
    let failed_end_loc = value_u64(&base.fields[last_failed_field], "fileLoc")?;
    let search_samples = search_fields.saturating_mul(samples_per_field);
    let left = find_match_run(
        &base,
        &repair,
        failed_start_loc.saturating_sub(search_samples),
        failed_start_loc,
        samples_per_field,
        consecutive_matches,
        threshold,
        trim_fraction,
        MatchSelection::Last,
    )?;
    let right = find_match_run(
        &repair,
        &base,
        failed_end_loc.saturating_add(1),
        failed_end_loc.saturating_add(search_samples),
        samples_per_field,
        consecutive_matches,
        threshold,
        trim_fraction,
        MatchSelection::First,
    )?;
    let base_left_end = left.left_index + 1;
    let repair_start = left.right_index + 1;
    let repair_end = right.left_index + 1;
    let base_right_start = right.right_index + 1;
    anyhow::ensure!(repair_start <= repair_end, "repair handoffs are reversed");
    anyhow::ensure!(
        base_left_end <= first_failed_field,
        "left handoff did not precede failure"
    );
    anyhow::ensure!(
        base_right_start > last_failed_field,
        "right handoff did not follow failure"
    );

    let (luma_path, chroma_path, metadata_path) = output_paths(&output_prefix)?;
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
    let mut luma_out = Blake3Writer::new(BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&luma_path)?,
    ));
    let mut chroma_out = Blake3Writer::new(BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&chroma_path)?,
    ));
    for (decoded, start, end) in [
        (&base, 0, base_left_end),
        (&repair, repair_start, repair_end),
        (&base, base_right_start, base.fields.len()),
    ] {
        copy_fields(
            &decoded.luma,
            &mut luma_out,
            start,
            end,
            decoded.field_bytes,
        )?;
        copy_fields(
            &decoded.chroma,
            &mut chroma_out,
            start,
            end,
            decoded.field_bytes,
        )?;
    }
    let mut fields = base.fields[..base_left_end].to_vec();
    fields.extend_from_slice(&repair.fields[repair_start..repair_end]);
    fields.extend_from_slice(&base.fields[base_right_start..]);
    for (index, field) in fields.iter_mut().enumerate() {
        let sequence = index + 1;
        field["seqNo"] = Value::from(sequence as u64);
        field["fieldPhaseID"] = Value::from(phase_id(value_bool(field, "isFirstField")?, sequence));
    }
    let (luma_blake3, luma_length) = luma_out.finish()?;
    let (chroma_blake3, chroma_length) = chroma_out.finish()?;
    let mut metadata = base.metadata.clone();
    metadata["fields"] = Value::Array(fields);
    metadata["videoParameters"]["numberOfSequentialFields"] =
        Value::from(metadata["fields"].as_array().unwrap().len() as u64);
    write_json_atomic(&metadata_path, &metadata)?;
    let (metadata_blake3, metadata_length) = blake3_file(&metadata_path)?;
    let report = RepairReport {
        base_prefix,
        repair_prefix,
        output_prefix,
        first_failed_field,
        last_failed_field,
        left_handoff: left,
        right_handoff: right,
        base_fields_removed: base_right_start - base_left_end,
        repair_fields_inserted: repair_end - repair_start,
        output_fields: metadata["fields"].as_array().unwrap().len(),
        luma_blake3,
        luma_length,
        chroma_blake3,
        chroma_length,
        metadata_blake3,
        metadata_length,
    };
    write_json_atomic(&report.output_prefix.with_extension("repair.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_output(root: &Path, name: &str, start: u64, values: &[u16]) -> PathBuf {
        let prefix = root.join(name);
        let (luma, chroma, metadata) = output_paths(&prefix).unwrap();
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(luma, &bytes).unwrap();
        fs::write(chroma, &bytes).unwrap();
        let fields = values
            .iter()
            .enumerate()
            .map(|(index, _)| {
                serde_json::json!({
                    "seqNo": index + 1,
                    "fileLoc": start + index as u64 * 100,
                    "isFirstField": index.is_multiple_of(2),
                    "syncConf": 45,
                    "fieldPhaseID": 1
                })
            })
            .collect::<Vec<_>>();
        write_json_atomic(
            &metadata,
            &serde_json::json!({
                "videoParameters": {
                    "fieldWidth": 1,
                    "fieldHeight": 1,
                    "numberOfSequentialFields": values.len()
                },
                "fields": fields
            }),
        )
        .unwrap();
        prefix
    }

    #[test]
    fn failures_are_coalesced_by_gap() {
        assert_eq!(
            cluster_failures(&[3, 4, 10, 30, 31], 6),
            vec![(3, 10), (30, 31)]
        );
        assert_eq!(cluster_failures(&[], 6), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn repair_plan_adds_context_and_guards() {
        let root = std::env::temp_dir().join(format!("dist-repair-plan-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let metadata = root.join("input.tbc.json");
        write_json_atomic(
            &metadata,
            &serde_json::json!({
                "fields": (0..40).map(|index| serde_json::json!({"fileLoc": 1_000 + index * 100})).collect::<Vec<_>>()
            }),
        )
        .unwrap();
        let out = root.join("plan.json");
        let report = plan(
            metadata,
            vec![3, 4, 10, 30],
            out,
            6,
            2,
            1,
            100,
            100,
            Some(10_000),
        )
        .unwrap();
        assert_eq!(report.ranges.len(), 2);
        assert_eq!(report.ranges[0].canonical_start_sample, 1_100);
        assert_eq!(report.ranges[0].canonical_end_sample, 2_300);
        assert_eq!(report.ranges[0].decode_start_sample, 1_000);
        assert_eq!(report.ranges[0].decode_end_sample, 2_400);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn overlapping_guarded_ranges_are_merged() {
        let make = |id: &str, first, last, start, end| RepairRange {
            id: id.into(),
            first_failed_field: first,
            last_failed_field: last,
            target_start_file_loc: start,
            target_end_file_loc: end,
            canonical_start_sample: start,
            canonical_end_sample: end,
            decode_start_sample: start.saturating_sub(100),
            decode_end_sample: end + 100,
        };
        let ranges = merge_overlapping_ranges(vec![
            make("old-0", 3, 4, 1_000, 1_200),
            make("old-1", 8, 9, 1_250, 1_400),
            make("old-2", 20, 20, 2_000, 2_100),
        ]);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].id, "repair-0000");
        assert_eq!(ranges[0].first_failed_field, 3);
        assert_eq!(ranges[0].last_failed_field, 9);
        assert_eq!(ranges[0].decode_end_sample, 1_500);
        assert_eq!(ranges[1].id, "repair-0001");
    }

    #[test]
    fn repair_apply_requires_two_sided_matches_and_replaces_cluster() {
        let root = std::env::temp_dir().join(format!("dist-repair-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let base = write_output(
            &root,
            "base",
            1_000,
            &[10, 20, 30, 40, 9_000, 8_000, 70, 80, 90, 100],
        );
        let repair = write_output(&root, "repair", 1_200, &[30, 40, 50, 60, 70, 80]);
        let output = root.join("output");
        let report = apply(base, repair, output.clone(), 4, 5, 10, 100, 2, 64.0, 0.0).unwrap();
        assert_eq!(report.base_fields_removed, 4);
        assert_eq!(report.repair_fields_inserted, 4);
        assert_eq!(report.output_fields, 10);
        let bytes = fs::read(output.with_extension("tbc")).unwrap();
        let values = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn repair_uses_the_last_match_before_a_failure() {
        let root = std::env::temp_dir().join(format!("dist-repair-last-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let base = DecodedOutput::load(write_output(
            &root,
            "base",
            1_000,
            &[10, 20, 30, 40, 50, 60, 70, 80],
        ))
        .unwrap();
        let repair = DecodedOutput::load(write_output(
            &root,
            "repair",
            1_000,
            &[10, 20, 9_000, 8_000, 50, 60, 70, 80],
        ))
        .unwrap();
        let matched = find_match_run(
            &base,
            &repair,
            1_000,
            1_600,
            100,
            2,
            64.0,
            0.0,
            MatchSelection::Last,
        )
        .unwrap();
        assert_eq!(matched.left_index, 5);
        assert_eq!(matched.right_index, 5);
        fs::remove_dir_all(root).unwrap();
    }
}
