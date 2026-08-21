use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::hash::sha256_file;
use crate::model::read_json;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetainedArtifact {
    pub path: PathBuf,
    pub sha256: String,
    pub length: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetentionManifest {
    pub schema_version: u32,
    pub run_id: String,
    pub scratch_root: PathBuf,
    pub artifacts: Vec<RetainedArtifact>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScratchMarker {
    schema_version: u32,
    purpose: String,
    run_id: String,
    root: PathBuf,
}

fn visit_files(root: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.file_type().is_symlink() {
            bail!(
                "scratch tree contains a symlink: {}",
                entry.path().display()
            );
        }
        if metadata.is_dir() {
            visit_files(&entry.path(), output)?;
        } else if metadata.is_file() {
            output.push(entry.path());
        }
    }
    Ok(())
}

fn process_alive(pid: i32) -> bool {
    #[cfg(unix)]
    {
        let result = unsafe { libc::kill(pid, 0) };
        result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

fn tree_size(root: &Path) -> Result<u64> {
    let mut files = Vec::new();
    visit_files(root, &mut files)?;
    Ok(files
        .into_iter()
        .map(|path| fs::metadata(path).map(|metadata| metadata.len()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .sum())
}

pub fn run(scratch_root: PathBuf, retention_manifest: PathBuf) -> Result<u64> {
    let nosync = Path::new("/Users/nelson/NoSync").canonicalize()?;
    let root = scratch_root
        .canonicalize()
        .with_context(|| format!("scratch root does not exist: {}", scratch_root.display()))?;
    if root == nosync || !root.starts_with(&nosync) {
        bail!("refusing cleanup outside a child of {}", nosync.display());
    }
    if fs::symlink_metadata(&root)?.file_type().is_symlink() {
        bail!("scratch root must not be a symlink");
    }
    let marker: ScratchMarker = read_json(&root.join(".tape-decode-dist-scratch.json"))?;
    if marker.schema_version != 1
        || marker.purpose != "tape-decode distributed POC scratch"
        || marker.root != root
    {
        bail!("scratch marker does not authorize this exact root");
    }
    let retention: RetentionManifest = read_json(&retention_manifest)?;
    if retention.schema_version != 1
        || retention.run_id != marker.run_id
        || retention.scratch_root != root
    {
        bail!("retention manifest does not match scratch marker");
    }
    if retention_manifest.canonicalize()?.starts_with(&root) {
        bail!("retention manifest must live outside scratch root");
    }
    if retention.artifacts.is_empty() {
        bail!("retention manifest has no verified artifacts");
    }
    for artifact in &retention.artifacts {
        let artifact_path = artifact.path.canonicalize()?;
        if artifact_path.starts_with(&root) {
            bail!("retained artifact is still inside scratch root");
        }
        let (hash, length) = sha256_file(&artifact_path)?;
        if hash != artifact.sha256 || length != artifact.length {
            bail!(
                "retained artifact failed verification: {}",
                artifact.path.display()
            );
        }
    }
    let mut files = Vec::new();
    visit_files(&root, &mut files)?;
    for pid_path in files
        .iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "pid"))
    {
        let pid: i32 = fs::read_to_string(pid_path)?.trim().parse()?;
        if process_alive(pid) {
            bail!("active process {pid} recorded in {}", pid_path.display());
        }
    }
    let bytes = tree_size(&root)?;
    fs::remove_dir_all(&root)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_refuses_broad_nosync_root() {
        let result = run(
            PathBuf::from("/Users/nelson/NoSync"),
            PathBuf::from("/nonexistent/retention.json"),
        );
        assert!(result.is_err());
    }
}
