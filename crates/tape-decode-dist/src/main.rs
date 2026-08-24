mod assemble;
mod cleanup;
mod coordinator;
mod hash;
mod model;
mod repair;
mod runner;

use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::hash::file_hashes;
use crate::model::{
    align_job_starts, now_unix_ms, plan_jobs, write_json_atomic, DecodeConfig, InputArtifact,
    OutputGeometry, RangeConfig, RunManifest, SCHEMA_VERSION,
};

#[derive(Parser)]
#[command(
    name = "tape-decode-dist",
    about = "Local-first distributed tape-decode POC"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Create a versioned, content-addressed run manifest.
    Manifest(ManifestArgs),
    /// Run the leased-work coordinator (loopback-only unless explicitly allowed).
    Coordinator(CoordinatorArgs),
    /// Run one pull-based decoder worker.
    Runner(RunnerArgs),
    /// Stitch all completed shard artifacts into one canonical TBC.
    Assemble(AssembleArgs),
    /// Plan small guarded repair decodes around failed output fields.
    RepairPlan(RepairPlanArgs),
    /// Replace one failed field cluster using a guarded referee decode.
    RepairApply(RepairApplyArgs),
    /// Remove an explicitly marked scratch root after retained evidence verifies.
    Cleanup(CleanupArgs),
}

#[derive(Args)]
struct ManifestArgs {
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    run_id: String,
    #[arg(long)]
    work_root: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long, default_value = "flac")]
    input_format: String,
    #[arg(long)]
    decoder: PathBuf,
    #[arg(long, default_value = "NTSC_VHS")]
    profile: String,
    #[arg(long, default_value_t = 28.636363)]
    frequency_mhz: f64,
    #[arg(long = "decode-arg", default_value = "--ire0-adjust")]
    decode_args: Vec<String>,
    #[arg(long, default_value_t = 542)]
    fixture_source_start_second: u64,
    #[arg(long, default_value_t = 320)]
    fixture_seconds: u64,
    #[arg(long, default_value_t = 10)]
    canonical_start_second: u64,
    #[arg(long, default_value_t = 310)]
    canonical_end_second: u64,
    #[arg(long, default_value_t = 30)]
    shard_seconds: u64,
    #[arg(long, default_value_t = 5)]
    guard_seconds: u64,
    #[arg(long, default_value_t = 10)]
    outer_guard_seconds: u64,
    #[arg(long, default_value_t = 28_636_363)]
    sample_clock: u64,
    #[arg(long, default_value_t = 477_750)]
    samples_per_field: u64,
    #[arg(long, default_value_t = 910)]
    field_width: u64,
    #[arg(long, default_value_t = 263)]
    field_height: u64,
    #[arg(long, default_value_t = 60)]
    mt_distance_fields: u64,
    #[arg(long, default_value_t = 2)]
    mt_overlap_count: usize,
    #[arg(long, default_value_t = 64.0)]
    mt_threshold: f64,
    #[arg(long, default_value_t = 0.10)]
    mt_trim_fraction: f64,
}

#[derive(Args)]
struct CoordinatorArgs {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8787")]
    bind: SocketAddr,
    /// Permit an unauthenticated coordinator bind outside loopback.
    #[arg(long)]
    allow_lan: bool,
    #[arg(long, default_value_t = 60)]
    lease_seconds: u64,
    #[arg(long, default_value_t = 3)]
    max_attempts: u32,
}

#[derive(Args)]
struct RunnerArgs {
    #[arg(long, default_value = "http://127.0.0.1:8787")]
    coordinator: String,
    #[arg(long)]
    runner_id: String,
    #[arg(long)]
    runner_root: PathBuf,
    #[arg(long)]
    decode_threads: usize,
    /// Runner-local decoder binary (needed when manifest paths are not shared).
    #[arg(long)]
    decoder: Option<PathBuf>,
    /// Accept a runner-local platform build whose binary hash differs from the manifest build.
    #[arg(long, requires = "decoder")]
    allow_platform_decoder: bool,
    #[arg(long, default_value_t = 16 * 1024 * 1024 * 1024)]
    cache_bytes: u64,
    /// How the decoder obtains RF input for each leased job.
    #[arg(long, value_enum, default_value_t = RunnerInputMode::FullCache)]
    input_mode: RunnerInputMode,
    /// Maximum read-ahead retained by the decoder in HTTP-range mode.
    #[arg(long, default_value_t = 8 * 1024 * 1024)]
    http_range_buffer_bytes: usize,
    #[arg(long, default_value_t = 10)]
    heartbeat_seconds: u64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RunnerInputMode {
    FullCache,
    HttpRange,
}

#[derive(Args)]
struct AssembleArgs {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    state: PathBuf,
    #[arg(long)]
    output_prefix: PathBuf,
}

#[derive(Args)]
struct RepairPlanArgs {
    /// Metadata sidecar for the assembled output that failed comparison.
    #[arg(long)]
    metadata: PathBuf,
    /// Zero-based failed output field indices; may be repeated or comma-separated.
    #[arg(long = "failed-field", required = true, value_delimiter = ',')]
    failed_fields: Vec<usize>,
    #[arg(long)]
    out: PathBuf,
    /// Failures separated by no more than this many fields share one repair job.
    #[arg(long, default_value_t = 180)]
    cluster_gap_fields: usize,
    /// Canonical context retained around each failed cluster before RF guards.
    #[arg(long, default_value_t = 4)]
    canonical_pad_fields: u64,
    #[arg(long, default_value_t = 5)]
    guard_seconds: u64,
    #[arg(long, default_value_t = 28_636_363)]
    sample_clock: u64,
    #[arg(long, default_value_t = 477_750)]
    samples_per_field: u64,
    /// Optional upper bound for guarded decode ranges, relative to the fixture.
    #[arg(long)]
    fixture_samples: Option<u64>,
}

#[derive(Args)]
struct RepairApplyArgs {
    #[arg(long)]
    base_prefix: PathBuf,
    #[arg(long)]
    repair_prefix: PathBuf,
    #[arg(long)]
    output_prefix: PathBuf,
    #[arg(long)]
    first_failed_field: usize,
    #[arg(long)]
    last_failed_field: usize,
    #[arg(long, default_value_t = 300)]
    search_fields: u64,
    #[arg(long, default_value_t = 477_750)]
    samples_per_field: u64,
    #[arg(long, default_value_t = 2)]
    consecutive_matches: usize,
    #[arg(long, default_value_t = 64.0)]
    threshold: f64,
    #[arg(long, default_value_t = 0.10)]
    trim_fraction: f64,
}

#[derive(Args)]
struct CleanupArgs {
    #[arg(long)]
    scratch_root: PathBuf,
    #[arg(long)]
    retention_manifest: PathBuf,
}

struct PidGuard(PathBuf);

impl PidGuard {
    fn create(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, format!("{}\n", std::process::id()))?;
        Ok(Self(path))
    }
}

impl Drop for PidGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn exact_child(root: &Path, child: &Path) -> Result<()> {
    let root = root.canonicalize()?;
    let child_parent = child.parent().unwrap_or(Path::new(".")).canonicalize()?;
    if !child_parent.starts_with(&root) {
        bail!("{} must be inside {}", child.display(), root.display());
    }
    Ok(())
}

fn create_manifest(args: ManifestArgs) -> Result<()> {
    if args.out.exists() {
        bail!("refusing to overwrite {}", args.out.display());
    }
    let work_root = args.work_root.canonicalize()?;
    exact_child(&work_root, &args.out)?;
    let input = args.input.canonicalize()?;
    let decoder = args.decoder.canonicalize()?;
    let input_hashes = file_hashes(&input)?;
    let decoder_hashes = file_hashes(&decoder)?;
    let ranges = RangeConfig {
        fixture_source_start_second: args.fixture_source_start_second,
        fixture_samples: args.fixture_seconds.saturating_mul(args.sample_clock),
        canonical_start_sample: args
            .canonical_start_second
            .saturating_mul(args.sample_clock),
        canonical_end_sample: args.canonical_end_second.saturating_mul(args.sample_clock),
        shard_samples: args.shard_seconds.saturating_mul(args.sample_clock),
        guard_samples: args.guard_seconds.saturating_mul(args.sample_clock),
        outer_guard_samples: args.outer_guard_seconds.saturating_mul(args.sample_clock),
        samples_per_field: args.samples_per_field,
    };
    let mut jobs = plan_jobs(&ranges)?;
    let segment_samples = args
        .mt_distance_fields
        .checked_mul(args.samples_per_field)
        .context("multithread segment sample count overflow")?;
    align_job_starts(&mut jobs, segment_samples)?;
    let manifest = RunManifest {
        schema_version: SCHEMA_VERSION,
        run_id: args.run_id,
        created_at_unix_ms: now_unix_ms(),
        work_root,
        input: InputArtifact {
            blake3: input_hashes.blake3,
            sha256: input_hashes.sha256,
            length: input_hashes.length,
            format: args.input_format,
            path: input,
        },
        decode: DecodeConfig {
            decoder_path: decoder,
            decoder_blake3: decoder_hashes.blake3,
            decoder_sha256: decoder_hashes.sha256,
            profile: args.profile,
            frequency_mhz: args.frequency_mhz,
            extra_args: args.decode_args,
            mt_distance_fields: args.mt_distance_fields,
            mt_overlap_count: args.mt_overlap_count,
            mt_threshold: args.mt_threshold,
            mt_trim_fraction: args.mt_trim_fraction,
        },
        output_geometry: OutputGeometry {
            field_width: args.field_width,
            field_height: args.field_height,
            bytes_per_sample: 2,
            has_luma: true,
            has_chroma: true,
            has_metadata: true,
        },
        ranges,
        jobs,
    };
    write_json_atomic(&args.out, &manifest)?;
    println!(
        "wrote {} with {} jobs",
        args.out.display(),
        manifest.jobs.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Manifest(args) => create_manifest(args),
        Command::Coordinator(args) => {
            let manifest: RunManifest = model::read_json(&args.manifest)?;
            let _pid = PidGuard::create(manifest.work_root.join("coordinator/coordinator.pid"))?;
            tokio::runtime::Runtime::new()?.block_on(coordinator::run(
                args.manifest,
                args.bind,
                args.allow_lan,
                args.lease_seconds,
                args.max_attempts,
            ))
        }
        Command::Runner(args) => {
            let _pid = PidGuard::create(args.runner_root.join("runner.pid"))?;
            runner::run(
                args.coordinator,
                args.runner_id,
                args.runner_root,
                args.decode_threads,
                args.decoder,
                args.allow_platform_decoder,
                args.cache_bytes,
                args.input_mode,
                args.http_range_buffer_bytes,
                args.heartbeat_seconds,
            )
        }
        Command::Assemble(args) => {
            let report = assemble::run(args.manifest, args.state, args.output_prefix)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::RepairPlan(args) => {
            let report = repair::plan(
                args.metadata,
                args.failed_fields,
                args.out,
                args.cluster_gap_fields,
                args.canonical_pad_fields,
                args.guard_seconds,
                args.sample_clock,
                args.samples_per_field,
                args.fixture_samples,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::RepairApply(args) => {
            let report = repair::apply(
                args.base_prefix,
                args.repair_prefix,
                args.output_prefix,
                args.first_failed_field,
                args.last_failed_field,
                args.search_fields,
                args.samples_per_field,
                args.consecutive_matches,
                args.threshold,
                args.trim_fraction,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::Cleanup(args) => {
            let bytes = cleanup::run(args.scratch_root, args.retention_manifest)?;
            println!("removed {bytes} verified scratch bytes");
            Ok(())
        }
    }
}
