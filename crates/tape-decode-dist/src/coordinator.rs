use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{header, HeaderMap, HeaderValue, Response, StatusCode};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post, put};
use axum::Router;
use futures_util::StreamExt;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio_util::io::ReaderStream;
use uuid::Uuid;

use crate::model::{
    initial_state, now_unix_ms, read_json, write_json_atomic, ArtifactInfo, CompleteRequest,
    CoordinatorState, JobEvent, JobState, LeaseAction, LeaseGrant, LeaseRequest, LeaseResponse,
    RunManifest, RunnerRegistration, ARTIFACT_KINDS, SCHEMA_VERSION,
};

#[derive(Clone)]
struct AppState {
    manifest: Arc<RunManifest>,
    state: Arc<Mutex<CoordinatorState>>,
    state_path: PathBuf,
    work_root: PathBuf,
    lease_ms: u64,
    max_attempts: u32,
}

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (self.0, Json(serde_json::json!({"error": self.1}))).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
    }
}

impl From<std::io::Error> for ApiError {
    fn from(error: std::io::Error) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
    }
}

fn bad_request(message: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, message.into())
}

fn conflict(message: impl Into<String>) -> ApiError {
    ApiError(StatusCode::CONFLICT, message.into())
}

fn persist(app: &AppState, state: &CoordinatorState) -> Result<()> {
    write_json_atomic(&app.state_path, state)
}

fn requeue_expired(state: &mut CoordinatorState, now: u64, max_attempts: u32) {
    for job in &mut state.jobs {
        let expired = match &job.state {
            JobState::Leased {
                expires_at_unix_ms,
                runner_id,
                lease_id,
                ..
            } if *expires_at_unix_ms <= now => Some((runner_id.clone(), lease_id.clone())),
            _ => None,
        };
        if let Some((runner_id, lease_id)) = expired {
            job.history.push(JobEvent {
                at_unix_ms: now,
                event: "leaseExpired".into(),
                runner_id: Some(runner_id),
                lease_id: Some(lease_id),
                detail: None,
            });
            job.state = if job.attempts >= max_attempts {
                JobState::Failed {
                    attempts: job.attempts,
                    message: "lease expired at maximum attempt count".into(),
                }
            } else {
                JobState::Pending
            };
        }
    }
}

async fn register(
    State(app): State<AppState>,
    Json(registration): Json<RunnerRegistration>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if registration.runner_id.trim().is_empty()
        || !registration
            .runner_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
        || registration.decode_threads == 0
    {
        return Err(bad_request(
            "runner id must use ASCII letters, digits, '-' or '_', and threads must be positive",
        ));
    }
    let path = app
        .work_root
        .join("coordinator/runners")
        .join(format!("{}.json", registration.runner_id));
    write_json_atomic(&path, &registration)?;
    Ok(Json(
        serde_json::json!({"ok": true, "runId": app.manifest.run_id}),
    ))
}

async fn lease(
    State(app): State<AppState>,
    Json(request): Json<LeaseRequest>,
) -> Result<Json<LeaseResponse>, ApiError> {
    let now = now_unix_ms();
    let mut state = app.state.lock().unwrap();
    requeue_expired(&mut state, now, app.max_attempts);
    let run_failed = state
        .jobs
        .iter()
        .any(|job| matches!(job.state, JobState::Failed { .. }));
    let run_complete = state
        .jobs
        .iter()
        .all(|job| matches!(job.state, JobState::Complete { .. }));
    if run_failed || run_complete {
        persist(&app, &state)?;
        return Ok(Json(LeaseResponse {
            lease: None,
            run_complete,
            run_failed,
        }));
    }
    let Some(job) = state
        .jobs
        .iter_mut()
        .find(|job| matches!(job.state, JobState::Pending))
    else {
        persist(&app, &state)?;
        return Ok(Json(LeaseResponse {
            lease: None,
            run_complete: false,
            run_failed: false,
        }));
    };
    job.attempts += 1;
    let attempt = job.attempts;
    let lease_id = Uuid::new_v4().to_string();
    let expires_at_unix_ms = now + app.lease_ms;
    job.state = JobState::Leased {
        runner_id: request.runner_id.clone(),
        lease_id: lease_id.clone(),
        expires_at_unix_ms,
        attempt,
    };
    job.history.push(JobEvent {
        at_unix_ms: now,
        event: "leased".into(),
        runner_id: Some(request.runner_id),
        lease_id: Some(lease_id.clone()),
        detail: Some(format!("attempt {attempt}")),
    });
    let spec = job.spec.clone();
    persist(&app, &state)?;
    Ok(Json(LeaseResponse {
        lease: Some(LeaseGrant {
            lease_id,
            expires_at_unix_ms,
            attempt,
            job: spec,
            input: app.manifest.input.clone(),
            decode: app.manifest.decode.clone(),
        }),
        run_complete: false,
        run_failed: false,
    }))
}

fn find_current_lease<'a>(
    state: &'a mut CoordinatorState,
    lease_id: &str,
    runner_id: &str,
) -> Result<&'a mut crate::model::JobRecord, ApiError> {
    state
        .jobs
        .iter_mut()
        .find(|job| {
            matches!(
                &job.state,
                JobState::Leased { runner_id: active_runner, lease_id: active_lease, .. }
                    if active_runner == runner_id && active_lease == lease_id
            )
        })
        .ok_or_else(|| conflict("lease is stale, unknown, or owned by another runner"))
}

async fn heartbeat(
    State(app): State<AppState>,
    AxumPath(lease_id): AxumPath<String>,
    Json(action): Json<LeaseAction>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let now = now_unix_ms();
    let mut state = app.state.lock().unwrap();
    requeue_expired(&mut state, now, app.max_attempts);
    let job = find_current_lease(&mut state, &lease_id, &action.runner_id)?;
    let JobState::Leased {
        expires_at_unix_ms, ..
    } = &mut job.state
    else {
        unreachable!()
    };
    *expires_at_unix_ms = now + app.lease_ms;
    let expires = *expires_at_unix_ms;
    persist(&app, &state)?;
    Ok(Json(
        serde_json::json!({"ok": true, "expiresAtUnixMs": expires}),
    ))
}

async fn fail_lease(
    State(app): State<AppState>,
    AxumPath(lease_id): AxumPath<String>,
    Json(action): Json<LeaseAction>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let now = now_unix_ms();
    let mut state = app.state.lock().unwrap();
    requeue_expired(&mut state, now, app.max_attempts);
    let max_attempts = app.max_attempts;
    let job = find_current_lease(&mut state, &lease_id, &action.runner_id)?;
    job.history.push(JobEvent {
        at_unix_ms: now,
        event: "failed".into(),
        runner_id: Some(action.runner_id),
        lease_id: Some(lease_id),
        detail: action.message.clone(),
    });
    job.state = if job.attempts >= max_attempts {
        JobState::Failed {
            attempts: job.attempts,
            message: action
                .message
                .unwrap_or_else(|| "runner reported failure".into()),
        }
    } else {
        JobState::Pending
    };
    persist(&app, &state)?;
    Ok(Json(serde_json::json!({"ok": true})))
}

fn artifact_path(root: &Path, job_id: &str, lease_id: &str, kind: &str) -> PathBuf {
    root.join("coordinator/uploads")
        .join(job_id)
        .join(lease_id)
        .join(kind)
}

async fn upload_artifact(
    State(app): State<AppState>,
    AxumPath((lease_id, kind)): AxumPath<(String, String)>,
    headers: HeaderMap,
    body: Body,
) -> Result<Json<ArtifactInfo>, ApiError> {
    if !ARTIFACT_KINDS.contains(&kind.as_str()) {
        return Err(bad_request("unknown artifact kind"));
    }
    let runner_id = headers
        .get("x-runner-id")
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| bad_request("missing x-runner-id"))?;
    let expected_length: u64 = headers
        .get("x-content-length")
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| bad_request("missing x-content-length"))?
        .parse()
        .map_err(|_| bad_request("invalid x-content-length"))?;
    let job_id = {
        let mut state = app.state.lock().unwrap();
        requeue_expired(&mut state, now_unix_ms(), app.max_attempts);
        let job_id = find_current_lease(&mut state, &lease_id, runner_id)?
            .spec
            .id
            .clone();
        persist(&app, &state)?;
        job_id
    };
    let final_path = artifact_path(&app.work_root, &job_id, &lease_id, &kind);
    let partial = final_path.with_extension("partial");
    tokio::fs::create_dir_all(final_path.parent().unwrap()).await?;
    let mut output = tokio::fs::File::create(&partial).await?;
    let mut stream = body.into_data_stream();
    let mut hasher = blake3::Hasher::new();
    let mut length = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| ApiError(StatusCode::BAD_REQUEST, error.to_string()))?;
        output.write_all(&chunk).await?;
        hasher.update(&chunk);
        length += chunk.len() as u64;
    }
    output.flush().await?;
    drop(output);
    let hash = hasher.finalize().to_hex().to_string();
    if length != expected_length {
        let _ = tokio::fs::remove_file(&partial).await;
        return Err(bad_request(format!(
            "artifact length verification failed: expected {expected_length}, got {length}"
        )));
    }
    tokio::fs::rename(&partial, &final_path).await?;
    Ok(Json(ArtifactInfo {
        kind,
        blake3: hash,
        length,
    }))
}

async fn complete_lease(
    State(app): State<AppState>,
    AxumPath(lease_id): AxumPath<String>,
    Json(request): Json<CompleteRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let now = now_unix_ms();
    let mut state = app.state.lock().unwrap();
    requeue_expired(&mut state, now, app.max_attempts);
    let job = find_current_lease(&mut state, &lease_id, &request.runner_id)?;
    let (attempt, job_id) = match &job.state {
        JobState::Leased { attempt, .. } => (*attempt, job.spec.id.clone()),
        _ => unreachable!(),
    };
    let artifacts: BTreeMap<_, _> = request
        .artifacts
        .into_iter()
        .map(|artifact| (artifact.kind.clone(), artifact))
        .collect();
    for kind in ARTIFACT_KINDS {
        let info = artifacts
            .get(kind)
            .ok_or_else(|| bad_request(format!("missing {kind} artifact")))?;
        let path = artifact_path(&app.work_root, &job_id, &lease_id, kind);
        let metadata = std::fs::metadata(&path)
            .with_context(|| format!("uploaded {kind} artifact does not exist"))?;
        if metadata.len() != info.length {
            return Err(bad_request(format!("uploaded {kind} length changed")));
        }
    }
    job.history.push(JobEvent {
        at_unix_ms: now,
        event: "completed".into(),
        runner_id: Some(request.runner_id.clone()),
        lease_id: Some(lease_id.clone()),
        detail: Some(format!("attempt {attempt}")),
    });
    job.state = JobState::Complete {
        runner_id: request.runner_id,
        lease_id,
        attempt,
        artifacts,
        completed_at_unix_ms: now,
    };
    persist(&app, &state)?;
    Ok(Json(serde_json::json!({"ok": true})))
}

fn parse_range(headers: &HeaderMap, length: u64) -> Result<(u64, u64, StatusCode)> {
    let Some(value) = headers.get(header::RANGE) else {
        return Ok((0, length, StatusCode::OK));
    };
    let value = value
        .to_str()?
        .strip_prefix("bytes=")
        .context("unsupported range")?;
    let (start, end) = value.split_once('-').context("invalid byte range")?;
    let start: u64 = start.parse()?;
    let end_exclusive = if end.is_empty() {
        length
    } else {
        end.parse::<u64>()?.saturating_add(1).min(length)
    };
    if start >= end_exclusive || start >= length {
        bail!("range is outside input");
    }
    Ok((start, end_exclusive, StatusCode::PARTIAL_CONTENT))
}

async fn get_input(
    State(app): State<AppState>,
    AxumPath(hash): AxumPath<String>,
    headers: HeaderMap,
) -> Result<Response<Body>, ApiError> {
    if hash != app.manifest.input.blake3 {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "input hash not found".into(),
        ));
    }
    let path = &app.manifest.input.path;
    let length = tokio::fs::metadata(path).await?.len();
    let (start, end, status) = parse_range(&headers, length)
        .map_err(|error| ApiError(StatusCode::RANGE_NOT_SATISFIABLE, error.to_string()))?;
    let mut file = tokio::fs::File::open(path).await?;
    file.seek(std::io::SeekFrom::Start(start)).await?;
    let content_length = end - start;
    let stream = ReaderStream::new(file.take(content_length));
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&content_length.to_string()).unwrap(),
    );
    response.headers_mut().insert(
        header::ETAG,
        HeaderValue::from_str(&format!("\"{}\"", app.manifest.input.blake3)).unwrap(),
    );
    response
        .headers_mut()
        .insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    if status == StatusCode::PARTIAL_CONTENT {
        response.headers_mut().insert(
            header::CONTENT_RANGE,
            HeaderValue::from_str(&format!("bytes {start}-{}/{length}", end - 1)).unwrap(),
        );
    }
    Ok(response)
}

async fn status(State(app): State<AppState>) -> Json<CoordinatorState> {
    Json(app.state.lock().unwrap().clone())
}

fn validate_bind(bind: SocketAddr, allow_lan: bool) -> Result<()> {
    let loopback = matches!(bind.ip(), IpAddr::V4(ip) if ip.is_loopback())
        || matches!(bind.ip(), IpAddr::V6(ip) if ip.is_loopback());
    if !loopback && !allow_lan {
        bail!("POC coordinator refuses non-loopback bind address {bind} without --allow-lan");
    }
    Ok(())
}

pub async fn run(
    manifest_path: PathBuf,
    bind: SocketAddr,
    allow_lan: bool,
    lease_seconds: u64,
    max_attempts: u32,
) -> Result<()> {
    validate_bind(bind, allow_lan)?;
    let manifest: RunManifest = read_json(&manifest_path)?;
    if manifest.schema_version != SCHEMA_VERSION {
        bail!("unsupported manifest schema {}", manifest.schema_version);
    }
    anyhow::ensure!(
        std::fs::metadata(&manifest.input.path)?.len() == manifest.input.length,
        "manifest input length has changed"
    );
    let state_path = manifest.work_root.join("coordinator/state.json");
    let state = if state_path.exists() {
        let loaded: CoordinatorState = read_json(&state_path)?;
        if loaded.run_id != manifest.run_id {
            bail!("coordinator state belongs to a different run");
        }
        loaded
    } else {
        let initial = initial_state(&manifest);
        write_json_atomic(&state_path, &initial)?;
        initial
    };
    let app_state = AppState {
        manifest: Arc::new(manifest.clone()),
        state: Arc::new(Mutex::new(state)),
        state_path,
        work_root: manifest.work_root,
        lease_ms: lease_seconds.saturating_mul(1000),
        max_attempts,
    };
    let router = Router::new()
        .route("/v1/runners/register", post(register))
        .route("/v1/jobs/lease", post(lease))
        .route("/v1/leases/{lease_id}/heartbeat", post(heartbeat))
        .route("/v1/leases/{lease_id}/fail", post(fail_lease))
        .route(
            "/v1/leases/{lease_id}/artifacts/{kind}",
            put(upload_artifact),
        )
        .route("/v1/leases/{lease_id}/complete", post(complete_lease))
        .route("/v1/inputs/{hash}", get(get_input))
        .route("/v1/status", get(status))
        .with_state(app_state);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    println!(
        "coordinator listening on http://{bind} for run {}",
        manifest.run_id
    );
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{JobRecord, JobSpec};

    #[test]
    fn lan_bind_requires_explicit_opt_in() {
        let lan = "10.168.1.16:8787".parse().unwrap();
        let loopback = "127.0.0.1:8787".parse().unwrap();
        assert!(validate_bind(loopback, false).is_ok());
        assert!(validate_bind(lan, false).is_err());
        assert!(validate_bind(lan, true).is_ok());
    }

    fn leased(expires: u64, attempts: u32) -> CoordinatorState {
        CoordinatorState {
            schema_version: 1,
            run_id: "test".into(),
            jobs: vec![JobRecord {
                spec: JobSpec {
                    id: "job".into(),
                    ordinal: 0,
                    canonical_start_sample: 0,
                    canonical_end_sample: 10,
                    decode_start_sample: 0,
                    decode_end_sample: 12,
                },
                attempts,
                state: JobState::Leased {
                    runner_id: "runner".into(),
                    lease_id: "lease".into(),
                    expires_at_unix_ms: expires,
                    attempt: attempts,
                },
                history: Vec::new(),
            }],
        }
    }

    #[test]
    fn expired_lease_is_requeued_below_attempt_cap() {
        let mut state = leased(10, 1);
        requeue_expired(&mut state, 11, 3);
        assert!(matches!(state.jobs[0].state, JobState::Pending));
    }

    #[test]
    fn expired_lease_fails_at_attempt_cap() {
        let mut state = leased(10, 3);
        requeue_expired(&mut state, 11, 3);
        assert!(matches!(state.jobs[0].state, JobState::Failed { .. }));
    }

    #[test]
    fn expired_or_reassigned_lease_is_stale() {
        let mut state = leased(10, 1);
        requeue_expired(&mut state, 11, 3);
        assert!(find_current_lease(&mut state, "lease", "runner").is_err());
    }

    #[test]
    fn byte_ranges_are_resumable_and_bounded() {
        let mut headers = HeaderMap::new();
        headers.insert(header::RANGE, HeaderValue::from_static("bytes=40-"));
        assert_eq!(
            parse_range(&headers, 100).unwrap(),
            (40, 100, StatusCode::PARTIAL_CONTENT)
        );
        headers.insert(header::RANGE, HeaderValue::from_static("bytes=40-59"));
        assert_eq!(
            parse_range(&headers, 100).unwrap(),
            (40, 60, StatusCode::PARTIAL_CONTENT)
        );
    }
}
