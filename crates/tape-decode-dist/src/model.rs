use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 2;
pub const ARTIFACT_KINDS: [&str; 3] = ["luma", "chroma", "metadata"];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct InputArtifact {
    pub blake3: String,
    /// Preservation/interchange digest, calculated in the same initial pass.
    pub sha256: String,
    pub length: u64,
    pub format: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DecodeConfig {
    pub decoder_path: PathBuf,
    pub decoder_blake3: String,
    /// Preservation/interchange digest, calculated in the same initial pass.
    pub decoder_sha256: String,
    pub profile: String,
    pub frequency_mhz: f64,
    #[serde(default)]
    pub extra_args: Vec<String>,
    pub mt_distance_fields: u64,
    pub mt_overlap_count: usize,
    pub mt_threshold: f64,
    pub mt_trim_fraction: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RangeConfig {
    pub fixture_source_start_second: u64,
    pub fixture_samples: u64,
    pub canonical_start_sample: u64,
    pub canonical_end_sample: u64,
    pub shard_samples: u64,
    pub guard_samples: u64,
    pub outer_guard_samples: u64,
    pub samples_per_field: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct OutputGeometry {
    pub field_width: u64,
    pub field_height: u64,
    pub bytes_per_sample: u64,
    pub has_luma: bool,
    pub has_chroma: bool,
    pub has_metadata: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RunManifest {
    pub schema_version: u32,
    pub run_id: String,
    pub created_at_unix_ms: u64,
    pub work_root: PathBuf,
    pub input: InputArtifact,
    pub decode: DecodeConfig,
    pub output_geometry: OutputGeometry,
    pub ranges: RangeConfig,
    pub jobs: Vec<JobSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct JobSpec {
    pub id: String,
    pub ordinal: usize,
    pub canonical_start_sample: u64,
    pub canonical_end_sample: u64,
    pub decode_start_sample: u64,
    pub decode_end_sample: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactInfo {
    pub kind: String,
    pub blake3: String,
    pub length: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "camelCase")]
pub enum JobState {
    Pending,
    Leased {
        runner_id: String,
        lease_id: String,
        expires_at_unix_ms: u64,
        attempt: u32,
    },
    Complete {
        runner_id: String,
        lease_id: String,
        attempt: u32,
        artifacts: BTreeMap<String, ArtifactInfo>,
        completed_at_unix_ms: u64,
    },
    Failed {
        attempts: u32,
        message: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct JobRecord {
    pub spec: JobSpec,
    pub attempts: u32,
    pub state: JobState,
    #[serde(default)]
    pub history: Vec<JobEvent>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct JobEvent {
    pub at_unix_ms: u64,
    pub event: String,
    pub runner_id: Option<String>,
    pub lease_id: Option<String>,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CoordinatorState {
    pub schema_version: u32,
    pub run_id: String,
    pub jobs: Vec<JobRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunnerRegistration {
    pub runner_id: String,
    pub decode_threads: usize,
    pub platform: String,
    pub decoder_blake3: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaseRequest {
    pub runner_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaseGrant {
    pub lease_id: String,
    pub expires_at_unix_ms: u64,
    pub attempt: u32,
    pub job: JobSpec,
    pub input: InputArtifact,
    pub decode: DecodeConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaseResponse {
    pub lease: Option<LeaseGrant>,
    pub run_complete: bool,
    pub run_failed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaseAction {
    pub runner_id: String,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompleteRequest {
    pub runner_id: String,
    pub artifacts: Vec<ArtifactInfo>,
}

pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is after Unix epoch")
        .as_millis() as u64
}

pub fn plan_jobs(ranges: &RangeConfig) -> Result<Vec<JobSpec>> {
    if ranges.canonical_start_sample >= ranges.canonical_end_sample {
        bail!("canonical range is empty or reversed");
    }
    if ranges.canonical_end_sample > ranges.fixture_samples {
        bail!("canonical end exceeds fixture length");
    }
    if ranges.shard_samples == 0 || ranges.guard_samples == 0 {
        bail!("shard and guard sample counts must be positive");
    }
    let mut jobs = Vec::new();
    let mut start = ranges.canonical_start_sample;
    while start < ranges.canonical_end_sample {
        let end = start
            .saturating_add(ranges.shard_samples)
            .min(ranges.canonical_end_sample);
        let ordinal = jobs.len();
        let decode_start = if ordinal == 0 {
            start.saturating_sub(ranges.outer_guard_samples)
        } else {
            start.saturating_sub(ranges.guard_samples)
        };
        let decode_end = if end == ranges.canonical_end_sample {
            end.saturating_add(ranges.outer_guard_samples)
                .min(ranges.fixture_samples)
        } else {
            end.saturating_add(ranges.guard_samples)
                .min(ranges.fixture_samples)
        };
        jobs.push(JobSpec {
            id: format!("shard-{ordinal:04}"),
            ordinal,
            canonical_start_sample: start,
            canonical_end_sample: end,
            decode_start_sample: decode_start,
            decode_end_sample: decode_end,
        });
        start = end;
    }
    Ok(jobs)
}

pub fn align_job_starts(jobs: &mut [JobSpec], segment_samples: u64) -> Result<()> {
    if segment_samples == 0 {
        bail!("multithread segment size must be positive");
    }
    for job in jobs {
        job.decode_start_sample -= job.decode_start_sample % segment_samples;
    }
    Ok(())
}

pub fn initial_state(manifest: &RunManifest) -> CoordinatorState {
    CoordinatorState {
        schema_version: SCHEMA_VERSION,
        run_id: manifest.run_id.clone(),
        jobs: manifest
            .jobs
            .iter()
            .cloned()
            .map(|spec| JobRecord {
                spec,
                attempts: 0,
                state: JobState::Pending,
                history: Vec::new(),
            })
            .collect(),
    }
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("invalid JSON in {}", path.display()))
}

pub fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let parent = path.parent().context("JSON path has no parent")?;
    fs::create_dir_all(parent)?;
    let partial = path.with_extension("json.partial");
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(&partial, bytes)?;
    fs::rename(&partial, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_minute_plan_has_exact_coverage_and_outer_guards() {
        let second = 28_636_363;
        let ranges = RangeConfig {
            fixture_source_start_second: 542,
            fixture_samples: 320 * second,
            canonical_start_sample: 10 * second,
            canonical_end_sample: 310 * second,
            shard_samples: 30 * second,
            guard_samples: 5 * second,
            outer_guard_samples: 10 * second,
            samples_per_field: 477_750,
        };
        let jobs = plan_jobs(&ranges).unwrap();
        assert_eq!(jobs.len(), 10);
        assert_eq!(jobs.first().unwrap().decode_start_sample, 0);
        assert_eq!(jobs.last().unwrap().decode_end_sample, 320 * second);
        assert_eq!(jobs.first().unwrap().canonical_start_sample, 10 * second);
        assert_eq!(jobs.last().unwrap().canonical_end_sample, 310 * second);
        for pair in jobs.windows(2) {
            assert_eq!(pair[0].canonical_end_sample, pair[1].canonical_start_sample);
            assert_eq!(
                pair[0].decode_end_sample - pair[1].decode_start_sample,
                10 * second
            );
        }
    }

    #[test]
    fn interior_plan_uses_bounded_outer_guards() {
        let second = 28_636_363;
        let ranges = RangeConfig {
            fixture_source_start_second: 542,
            fixture_samples: 320 * second,
            canonical_start_sample: 84 * second,
            canonical_end_sample: 144 * second,
            shard_samples: 30 * second,
            guard_samples: 5 * second,
            outer_guard_samples: 10 * second,
            samples_per_field: 477_750,
        };
        let jobs = plan_jobs(&ranges).unwrap();
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].decode_start_sample, 74 * second);
        assert_eq!(jobs[0].decode_end_sample, 119 * second);
        assert_eq!(jobs[1].decode_start_sample, 109 * second);
        assert_eq!(jobs[1].decode_end_sample, 154 * second);
    }

    #[test]
    fn aligned_starts_share_one_absolute_worker_grid() {
        let mut jobs = vec![JobSpec {
            id: "shard-0000".into(),
            ordinal: 0,
            canonical_start_sample: 100,
            canonical_end_sample: 200,
            decode_start_sample: 83,
            decode_end_sample: 220,
        }];
        align_job_starts(&mut jobs, 20).unwrap();
        assert_eq!(jobs[0].decode_start_sample, 80);
        assert_eq!(jobs[0].decode_start_sample % 20, 0);
    }
}
