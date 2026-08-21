use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::{Duration, SystemTime};

use anyhow::{bail, Context, Result};
use reqwest::blocking::{Body, Client};
use reqwest::header::{CONTENT_LENGTH, RANGE};

use crate::hash::blake3_file;
use crate::model::{
    ArtifactInfo, CompleteRequest, LeaseAction, LeaseGrant, LeaseRequest, LeaseResponse,
    RunnerRegistration, ARTIFACT_KINDS,
};
use crate::RunnerInputMode;

fn cache_size(cache: &Path) -> Result<u64> {
    if !cache.exists() {
        return Ok(0);
    }
    Ok(fs::read_dir(cache)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.metadata().ok())
        .filter(|metadata| metadata.is_file())
        .map(|metadata| metadata.len())
        .sum())
}

fn evict_lru(cache: &Path, needed: u64, max_bytes: u64, protected: &Path) -> Result<()> {
    fs::create_dir_all(cache)?;
    let mut entries = fs::read_dir(cache)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path() != protected && entry.path().extension().is_none())
        .filter_map(|entry| {
            let metadata = entry.metadata().ok()?;
            Some((
                metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                metadata.len(),
                entry.path(),
            ))
        })
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.0);
    let mut total = cache_size(cache)?;
    for (_, length, path) in entries {
        if total.saturating_add(needed) <= max_bytes {
            break;
        }
        fs::remove_file(&path)?;
        total = total.saturating_sub(length);
    }
    if total.saturating_add(needed) > max_bytes {
        bail!("cache limit {max_bytes} cannot accommodate {needed} input bytes");
    }
    Ok(())
}

fn ensure_input(
    client: &Client,
    coordinator: &str,
    cache_dir: &Path,
    grant: &LeaseGrant,
    max_cache_bytes: u64,
) -> Result<(PathBuf, bool)> {
    fs::create_dir_all(cache_dir)?;
    let final_path = cache_dir.join(&grant.input.blake3);
    if final_path.exists() {
        let (hash, length) = blake3_file(&final_path)?;
        if hash == grant.input.blake3 && length == grant.input.length {
            let now = SystemTime::now();
            let _ = filetime_compat_touch(&final_path, now);
            return Ok((final_path, true));
        }
        bail!(
            "existing cache entry failed verification: {}",
            final_path.display()
        );
    }
    evict_lru(cache_dir, grant.input.length, max_cache_bytes, &final_path)?;
    let partial = final_path.with_extension("partial");
    let mut offset = fs::metadata(&partial)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    if offset > grant.input.length {
        fs::remove_file(&partial)?;
        offset = 0;
    }
    let url = format!("{coordinator}/v1/inputs/{}", grant.input.blake3);
    let mut request = client.get(&url);
    if offset != 0 {
        request = request.header(RANGE, format!("bytes={offset}-"));
    }
    let mut response = request.send()?.error_for_status()?;
    if offset != 0 && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        offset = 0;
    }
    let mut hasher = blake3::Hasher::new();
    if offset != 0 {
        let mut existing = File::open(&partial)?;
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let read = existing.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }
    let mut output = OpenOptions::new()
        .create(true)
        .write(true)
        .append(offset != 0)
        .truncate(offset == 0)
        .open(&partial)?;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = response.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        hasher.update(&buffer[..read]);
    }
    output.flush()?;
    drop(output);
    let length = fs::metadata(&partial)?.len();
    let hash = hasher.finalize().to_hex().to_string();
    if hash != grant.input.blake3 || length != grant.input.length {
        bail!("download verification failed: got {length} bytes {hash}");
    }
    fs::rename(&partial, &final_path)?;
    Ok((final_path, false))
}

// Touching is only an LRU hint. Opening and rewriting content would be unsafe;
// use the platform's `touch` utility without a shell so the cache bytes remain
// unchanged. Failure is deliberately non-fatal.
fn filetime_compat_touch(path: &Path, _now: SystemTime) -> Result<()> {
    let status = Command::new("/usr/bin/touch").arg(path).status()?;
    anyhow::ensure!(status.success(), "touch failed");
    Ok(())
}

enum DecoderInput<'a> {
    File(&'a Path),
    HttpRange {
        url: String,
        etag: &'a str,
        buffer_bytes: usize,
        metrics_path: &'a Path,
    },
}

fn child_command(
    grant: &LeaseGrant,
    input: DecoderInput<'_>,
    attempt_dir: &Path,
    threads: usize,
) -> Command {
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = Command::new("/usr/sbin/taskpolicy");
        command.args(["-b", "/usr/bin/nice", "-n", "19"]);
        command.arg(&grant.decode.decoder_path);
        command
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = {
        let mut command = Command::new("/usr/bin/nice");
        command.args(["-n", "19"]);
        command.arg(&grant.decode.decoder_path);
        command
    };
    #[cfg(windows)]
    let mut command = Command::new(&grant.decode.decoder_path);

    let prefix = attempt_dir.join("output");
    command
        .arg("decode")
        .args(["--luma-out", prefix.with_extension("tbc").to_str().unwrap()])
        .args([
            "--chroma-out",
            attempt_dir.join("output_chroma.tbc").to_str().unwrap(),
        ])
        .args([
            "--metadata-out",
            attempt_dir.join("output.tbc.json").to_str().unwrap(),
        ])
        .args(["--profile", &grant.decode.profile])
        .args(["--frequency", &format!("{:.6}", grant.decode.frequency_mhz)])
        .args(["--input-format", &grant.input.format])
        .args(["--offset", &grant.job.decode_start_sample.to_string()])
        .args(["--end-offset", &grant.job.decode_end_sample.to_string()])
        .args(["--mt-threads", &threads.to_string()])
        .args([
            "--mt-distance-size",
            &grant.decode.mt_distance_fields.to_string(),
        ])
        .args([
            "--mt-overlap-count",
            &grant.decode.mt_overlap_count.to_string(),
        ])
        .args(["--mt-threshold", &grant.decode.mt_threshold.to_string()])
        .args([
            "--mt-trim-fraction",
            &grant.decode.mt_trim_fraction.to_string(),
        ])
        .args(&grant.decode.extra_args);
    match input {
        DecoderInput::File(path) => {
            command.arg(path);
        }
        DecoderInput::HttpRange {
            url,
            etag,
            buffer_bytes,
            metrics_path,
        } => {
            command
                .args(["--input-url", &url])
                .args(["--input-http-etag", etag])
                .args(["--input-http-buffer-bytes", &buffer_bytes.to_string()])
                .args(["--input-http-metrics-out", metrics_path.to_str().unwrap()]);
        }
    }
    command
}

fn heartbeat_thread(
    client: Client,
    coordinator: String,
    runner_id: String,
    lease_id: String,
    interval: Duration,
) -> (Sender<()>, thread::JoinHandle<()>) {
    let (stop_tx, stop_rx) = mpsc::channel();
    let handle = thread::spawn(move || loop {
        match stop_rx.recv_timeout(interval) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = client
                    .post(format!("{coordinator}/v1/leases/{lease_id}/heartbeat"))
                    .json(&LeaseAction {
                        runner_id: runner_id.clone(),
                        message: None,
                    })
                    .send();
            }
        }
    });
    (stop_tx, handle)
}

fn upload_artifact(
    client: &Client,
    coordinator: &str,
    runner_id: &str,
    lease_id: &str,
    kind: &str,
    path: &Path,
) -> Result<ArtifactInfo> {
    let length = fs::metadata(path)?.len();
    let file = File::open(path)?;
    let response = client
        .put(format!(
            "{coordinator}/v1/leases/{lease_id}/artifacts/{kind}"
        ))
        .header("x-runner-id", runner_id)
        .header("x-content-length", length)
        .header(CONTENT_LENGTH, length)
        .body(Body::new(file))
        .send()?
        .error_for_status()?;
    let verified: ArtifactInfo = response.json()?;
    anyhow::ensure!(verified.length == length);
    Ok(verified)
}

fn report_failure(
    client: &Client,
    coordinator: &str,
    runner_id: &str,
    lease_id: &str,
    error: &anyhow::Error,
) {
    let _ = client
        .post(format!("{coordinator}/v1/leases/{lease_id}/fail"))
        .json(&LeaseAction {
            runner_id: runner_id.to_string(),
            message: Some(format!("{error:#}")),
        })
        .send();
}

#[allow(clippy::too_many_arguments)]
fn execute_lease(
    client: &Client,
    coordinator: &str,
    runner_id: &str,
    runner_root: &Path,
    threads: usize,
    cache_bytes: u64,
    input_mode: RunnerInputMode,
    http_range_buffer_bytes: usize,
    heartbeat_seconds: u64,
    grant: &LeaseGrant,
) -> Result<()> {
    let (decoder_hash, _) = blake3_file(&grant.decode.decoder_path)?;
    anyhow::ensure!(
        decoder_hash == grant.decode.decoder_blake3,
        "decoder binary does not match manifest hash"
    );
    if http_range_buffer_bytes == 0 {
        bail!("HTTP range buffer must be positive");
    }
    let attempt_parent = runner_root.join("attempts").join(&grant.job.id);
    fs::create_dir_all(&attempt_parent)?;
    let attempt_dir = attempt_parent.join(&grant.lease_id);
    let partial_attempt = attempt_parent.join(format!("{}.partial", grant.lease_id));
    if attempt_dir.exists() || partial_attempt.exists() {
        bail!("attempt path already exists for lease {}", grant.lease_id);
    }
    fs::create_dir(&partial_attempt)?;
    let metrics_path = partial_attempt.join("input-http-metrics.json");
    let cached_input;
    let decoder_input = match input_mode {
        RunnerInputMode::FullCache => {
            let cache_dir = runner_root.join("cache");
            let (input, cache_hit) =
                ensure_input(client, coordinator, &cache_dir, grant, cache_bytes)?;
            println!(
                "runner {runner_id} leased {} attempt {} (cache {})",
                grant.job.id,
                grant.attempt,
                if cache_hit { "hit" } else { "miss" }
            );
            cached_input = input;
            DecoderInput::File(&cached_input)
        }
        RunnerInputMode::HttpRange => {
            println!(
                "runner {runner_id} leased {} attempt {} (HTTP range)",
                grant.job.id, grant.attempt
            );
            DecoderInput::HttpRange {
                url: format!("{coordinator}/v1/inputs/{}", grant.input.blake3),
                etag: &grant.input.blake3,
                buffer_bytes: http_range_buffer_bytes,
                metrics_path: &metrics_path,
            }
        }
    };
    let stdout = File::create(partial_attempt.join("decoder.stdout.log"))?;
    let stderr = File::create(partial_attempt.join("decoder.stderr.log"))?;
    let (stop_heartbeat, heartbeat) = heartbeat_thread(
        client.clone(),
        coordinator.to_string(),
        runner_id.to_string(),
        grant.lease_id.clone(),
        Duration::from_secs(heartbeat_seconds),
    );
    let status = child_command(grant, decoder_input, &partial_attempt, threads)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .status()
        .context("failed to launch decoder child")?;
    let _ = stop_heartbeat.send(());
    let _ = heartbeat.join();
    if !status.success() {
        bail!("decoder exited with {status}");
    }
    if metrics_path.exists() {
        let metrics: serde_json::Value = serde_json::from_reader(File::open(&metrics_path)?)?;
        println!(
            "runner {runner_id} {} HTTP input: {} requests, {} bytes",
            grant.job.id, metrics["requests"], metrics["bytesReceived"]
        );
    }
    fs::rename(&partial_attempt, &attempt_dir)?;
    let paths = [
        attempt_dir.join("output.tbc"),
        attempt_dir.join("output_chroma.tbc"),
        attempt_dir.join("output.tbc.json"),
    ];
    let mut artifacts = Vec::new();
    for (kind, path) in ARTIFACT_KINDS.iter().zip(paths.iter()) {
        artifacts.push(upload_artifact(
            client,
            coordinator,
            runner_id,
            &grant.lease_id,
            kind,
            path,
        )?);
    }
    client
        .post(format!(
            "{coordinator}/v1/leases/{}/complete",
            grant.lease_id
        ))
        .json(&CompleteRequest {
            runner_id: runner_id.to_string(),
            artifacts,
        })
        .send()?
        .error_for_status()?;
    println!("runner {runner_id} completed {}", grant.job.id);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    coordinator: String,
    runner_id: String,
    runner_root: PathBuf,
    threads: usize,
    cache_bytes: u64,
    input_mode: RunnerInputMode,
    http_range_buffer_bytes: usize,
    heartbeat_seconds: u64,
) -> Result<()> {
    if threads == 0 || heartbeat_seconds == 0 {
        bail!("threads and heartbeat interval must be positive");
    }
    #[cfg(unix)]
    {
        // Give the daemon and every decoder child one private process group so
        // an administrator can terminate a failed runner without orphaning its
        // low-priority attempt process.
        let result = unsafe { libc::setpgid(0, 0) };
        if result != 0 {
            let error = std::io::Error::last_os_error();
            if !matches!(error.raw_os_error(), Some(libc::EACCES | libc::EPERM)) {
                return Err(error).context("failed to isolate runner process group");
            }
        }
    }
    fs::create_dir_all(&runner_root)?;
    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    client
        .post(format!("{coordinator}/v1/runners/register"))
        .json(&RunnerRegistration {
            runner_id: runner_id.clone(),
            decode_threads: threads,
            platform: std::env::consts::OS.to_string(),
        })
        .send()?
        .error_for_status()?;
    loop {
        let response: LeaseResponse = client
            .post(format!("{coordinator}/v1/jobs/lease"))
            .json(&LeaseRequest {
                runner_id: runner_id.clone(),
            })
            .send()?
            .error_for_status()?
            .json()?;
        if response.run_failed {
            bail!("coordinator reports a failed run");
        }
        if response.run_complete {
            println!("runner {runner_id}: run complete");
            return Ok(());
        }
        let Some(grant) = response.lease else {
            thread::sleep(Duration::from_secs(1));
            continue;
        };
        if let Err(error) = execute_lease(
            &client,
            &coordinator,
            &runner_id,
            &runner_root,
            threads,
            cache_bytes,
            input_mode,
            http_range_buffer_bytes,
            heartbeat_seconds,
            &grant,
        ) {
            report_failure(&client, &coordinator, &runner_id, &grant.lease_id, &error);
            eprintln!("runner {runner_id}: {} failed: {error:#}", grant.job.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_limit_rejects_an_input_larger_than_the_limit() {
        let root =
            std::env::temp_dir().join(format!("tape-decode-cache-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let protected = root.join("protected");
        assert!(evict_lru(&root, 11, 10, &protected).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cache_eviction_makes_room_without_removing_protected_path() {
        let root =
            std::env::temp_dir().join(format!("tape-decode-cache-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("first"), [0u8; 8]).unwrap();
        fs::write(root.join("second"), [0u8; 8]).unwrap();
        let protected = root.join("protected");
        fs::write(&protected, [0u8; 8]).unwrap();
        evict_lru(&root, 4, 20, &protected).unwrap();
        assert!(protected.exists());
        assert!(cache_size(&root).unwrap() + 4 <= 20);
        fs::remove_dir_all(root).unwrap();
    }
}
