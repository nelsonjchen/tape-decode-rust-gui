use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileHashes {
    pub blake3: String,
    pub sha256: String,
    pub length: u64,
}

/// Hash a file once with both the fast internal digest and the preservation
/// digest. Manifest creation is the only place that normally needs both.
pub fn file_hashes(path: &Path) -> Result<FileHashes> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut blake3 = blake3::Hasher::new();
    let mut sha256 = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut length = 0u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        blake3.update(&buffer[..read]);
        sha256.update(&buffer[..read]);
        length += read as u64;
    }
    Ok(FileHashes {
        blake3: blake3.finalize().to_hex().to_string(),
        sha256: hex::encode(sha256.finalize()),
        length,
    })
}

pub fn blake3_file(path: &Path) -> Result<(String, u64)> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut length = 0u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        length += read as u64;
    }
    Ok((hasher.finalize().to_hex().to_string(), length))
}

/// SHA-256 remains supported for external retention manifests and preservation
/// provenance. It is not used for distributed cache or result identity.
pub fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut length = 0u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        length += read as u64;
    }
    Ok((hex::encode(hasher.finalize()), length))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined_hashing_matches_individual_hashes() {
        let path = std::env::temp_dir().join(format!("dist-hash-{}", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"distributed tape decode").unwrap();
        let both = file_hashes(&path).unwrap();
        assert_eq!(blake3_file(&path).unwrap(), (both.blake3, both.length));
        assert_eq!(sha256_file(&path).unwrap(), (both.sha256, both.length));
        std::fs::remove_file(path).unwrap();
    }
}
