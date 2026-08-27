//! Preservation-safe external FLAC frame indexes.
//!
//! `.fidx` never modifies the FLAC source. Version 1 uses a fixed 192-byte
//! little-endian header followed by fixed-width records. Fixed-block streams
//! store one `u64` source-byte offset per retained frame anchor; variable-block
//! streams store `(sample_start: u64, byte_offset: u64, block_size: u32,
//! flags: u32)`. `stride == 1` is dense; larger strides are sparse and bound
//! the forward frame parsing required after an indexed seek.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{bail, Context as _, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use symphonia_bundle_flac::FlacReader;
use symphonia_core::codecs::CodecParameters;
use symphonia_core::formats::{FormatOptions, FormatReader};
use symphonia_core::io::MediaSourceStream;

const MAGIC: &[u8; 8] = b"FLACFIDX";
const VERSION: u16 = 1;
const HEADER_SIZE: u16 = 192;
const RECORD_FIXED_OFFSET: u16 = 1;
const RECORD_EXPLICIT: u16 = 2;
const FIXED_RECORD_SIZE: u16 = 8;
const EXPLICIT_RECORD_SIZE: u16 = 24;
const FLAG_SOURCE_SHA256: u32 = 1 << 0;
const FLAG_FIXED_BLOCK: u32 = 1 << 1;
const FLAG_COMPLETE: u32 = 1 << 2;

#[derive(Clone, Debug)]
pub struct FlacIndex {
    header: Header,
    records: Records,
}

#[derive(Clone, Debug)]
enum Records {
    Fixed(Vec<u64>),
    Explicit(Vec<ExplicitRecord>),
}

#[derive(Clone, Copy, Debug)]
struct ExplicitRecord {
    sample_start: u64,
    byte_offset: u64,
    block_size: u32,
    flags: u32,
}

#[derive(Clone, Debug)]
struct Header {
    flags: u32,
    record_kind: u16,
    record_size: u16,
    stride: u32,
    source_size: u64,
    metadata_end: u64,
    source_mtime_ns: u64,
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    min_block_size: u32,
    max_block_size: u32,
    fixed_block_size: u32,
    first_frame_number: u64,
    observed_total_samples: u64,
    frame_count: u64,
    record_count: u64,
    records_offset: u64,
    source_sha256: [u8; 32],
    stream_md5: [u8; 16],
    records_sha256: [u8; 32],
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexSummary {
    pub version: u16,
    pub source_bytes: u64,
    pub metadata_end_byte: u64,
    pub source_mtime_ns: u64,
    pub source_sha256: String,
    pub stream_md5: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub min_block_size: u32,
    pub max_block_size: u32,
    pub fixed_block_size: Option<u32>,
    pub observed_total_samples: u64,
    pub frame_count: u64,
    pub record_count: u64,
    pub record_bytes: u16,
    pub stride: u32,
    pub dense: bool,
    pub complete: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameAnchor {
    pub sample_start: u64,
    pub byte_offset: u64,
}

#[derive(Debug)]
struct StreamInfo {
    min_block_size: u32,
    max_block_size: u32,
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    stream_md5: [u8; 16],
}

impl FlacIndex {
    pub fn read(path: &Path) -> Result<Self> {
        let mut file = BufReader::new(
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
        );
        let mut bytes = [0u8; HEADER_SIZE as usize];
        file.read_exact(&mut bytes)
            .with_context(|| format!("truncated .fidx header in {}", path.display()))?;
        let header = Header::decode(&bytes)?;
        header.validate()?;

        let expected_len = header
            .records_offset
            .checked_add(
                header
                    .record_count
                    .checked_mul(u64::from(header.record_size))
                    .context(".fidx record region length overflow")?,
            )
            .context(".fidx file length overflow")?;
        let actual_len = file
            .get_ref()
            .metadata()
            .context("failed to stat .fidx")?
            .len();
        if actual_len != expected_len {
            bail!(
                ".fidx length mismatch: header requires {expected_len} bytes, file has {actual_len}"
            );
        }
        file.seek(SeekFrom::Start(header.records_offset))?;

        let mut record_hash = Sha256::new();
        let records = match header.record_kind {
            RECORD_FIXED_OFFSET => {
                let capacity = usize::try_from(header.record_count)
                    .context(".fidx record count does not fit memory")?;
                let mut records = Vec::with_capacity(capacity);
                for _ in 0..header.record_count {
                    let mut record = [0u8; FIXED_RECORD_SIZE as usize];
                    file.read_exact(&mut record)?;
                    record_hash.update(record);
                    records.push(u64::from_le_bytes(record));
                }
                Records::Fixed(records)
            }
            RECORD_EXPLICIT => {
                let capacity = usize::try_from(header.record_count)
                    .context(".fidx record count does not fit memory")?;
                let mut records = Vec::with_capacity(capacity);
                for _ in 0..header.record_count {
                    let mut record = [0u8; EXPLICIT_RECORD_SIZE as usize];
                    file.read_exact(&mut record)?;
                    record_hash.update(record);
                    records.push(ExplicitRecord {
                        sample_start: u64::from_le_bytes(record[0..8].try_into().unwrap()),
                        byte_offset: u64::from_le_bytes(record[8..16].try_into().unwrap()),
                        block_size: u32::from_le_bytes(record[16..20].try_into().unwrap()),
                        flags: u32::from_le_bytes(record[20..24].try_into().unwrap()),
                    });
                }
                Records::Explicit(records)
            }
            other => bail!("unsupported .fidx record kind {other}"),
        };
        let digest: [u8; 32] = record_hash.finalize().into();
        if digest != header.records_sha256 {
            bail!(".fidx record SHA-256 mismatch");
        }
        let index = Self { header, records };
        index.validate_records()?;
        Ok(index)
    }

    pub fn summary(&self) -> IndexSummary {
        IndexSummary {
            version: VERSION,
            source_bytes: self.header.source_size,
            metadata_end_byte: self.header.metadata_end,
            source_mtime_ns: self.header.source_mtime_ns,
            source_sha256: hex(&self.header.source_sha256),
            stream_md5: hex(&self.header.stream_md5),
            sample_rate: self.header.sample_rate,
            channels: self.header.channels,
            bits_per_sample: self.header.bits_per_sample,
            min_block_size: self.header.min_block_size,
            max_block_size: self.header.max_block_size,
            fixed_block_size: (self.header.fixed_block_size != 0)
                .then_some(self.header.fixed_block_size),
            observed_total_samples: self.header.observed_total_samples,
            frame_count: self.header.frame_count,
            record_count: self.header.record_count,
            record_bytes: self.header.record_size,
            stride: self.header.stride,
            dense: self.header.stride == 1,
            complete: self.header.flags & FLAG_COMPLETE != 0,
        }
    }

    pub fn metadata_end(&self) -> u64 {
        self.header.metadata_end
    }

    pub fn source_size(&self) -> u64 {
        self.header.source_size
    }

    pub fn observed_total_samples(&self) -> u64 {
        self.header.observed_total_samples
    }

    pub fn validate_source(&self, source: &File, verify_hash: bool) -> Result<()> {
        let metadata = source.metadata().context("failed to stat indexed FLAC")?;
        if metadata.len() != self.header.source_size {
            bail!(
                ".fidx source-size mismatch: index has {}, source has {}",
                self.header.source_size,
                metadata.len()
            );
        }
        if verify_hash {
            if self.header.flags & FLAG_SOURCE_SHA256 == 0 {
                bail!(".fidx does not contain a source SHA-256");
            }
            let mut clone = source.try_clone().context("failed to clone indexed FLAC")?;
            clone.seek(SeekFrom::Start(0))?;
            let actual = sha256_reader(&mut clone)?;
            if actual != self.header.source_sha256 {
                bail!(".fidx source SHA-256 mismatch");
            }
        }
        Ok(())
    }

    pub fn anchor_for_sample(&self, sample: u64) -> Result<FrameAnchor> {
        if sample >= self.header.observed_total_samples {
            bail!(
                "sample offset {sample} is outside indexed range 0..{}",
                self.header.observed_total_samples
            );
        }
        match &self.records {
            Records::Fixed(offsets) => {
                let block = u64::from(self.header.fixed_block_size);
                let frame = sample / block;
                let anchor_index = frame / u64::from(self.header.stride);
                let index = usize::try_from(anchor_index).context("anchor index overflow")?;
                let &byte_offset = offsets
                    .get(index)
                    .context("fixed .fidx anchor is missing")?;
                let anchor_frame = anchor_index * u64::from(self.header.stride);
                Ok(FrameAnchor {
                    sample_start: anchor_frame * block,
                    byte_offset,
                })
            }
            Records::Explicit(records) => {
                let index = records.partition_point(|record| record.sample_start <= sample);
                let record = records
                    .get(index.saturating_sub(1))
                    .context("variable-block .fidx has no anchor before sample")?;
                Ok(FrameAnchor {
                    sample_start: record.sample_start,
                    byte_offset: record.byte_offset,
                })
            }
        }
    }

    fn validate_records(&self) -> Result<()> {
        if self.header.record_count == 0 || self.header.frame_count == 0 {
            bail!(".fidx contains no frames");
        }
        match &self.records {
            Records::Fixed(offsets) => {
                if offsets.first().copied() != Some(self.header.metadata_end) {
                    bail!("first fixed .fidx record is not the first FLAC frame");
                }
                if !strictly_increasing(offsets.iter().copied()) {
                    bail!("fixed .fidx offsets are not strictly increasing");
                }
            }
            Records::Explicit(records) => {
                if records.first().map(|record| record.byte_offset)
                    != Some(self.header.metadata_end)
                {
                    bail!("first variable .fidx record is not the first FLAC frame");
                }
                if !strictly_increasing(records.iter().map(|record| record.byte_offset))
                    || !strictly_increasing(records.iter().map(|record| record.sample_start))
                {
                    bail!("variable .fidx records are not strictly increasing");
                }
                if records
                    .iter()
                    .any(|record| record.block_size == 0 || record.flags != 0)
                {
                    bail!("variable .fidx record contains invalid fields");
                }
            }
        }
        Ok(())
    }
}

impl Header {
    fn validate(&self) -> Result<()> {
        if self.flags & FLAG_COMPLETE == 0 {
            bail!(".fidx is not marked complete");
        }
        if self.stride == 0 {
            bail!(".fidx stride must be at least 1");
        }
        if self.records_offset != u64::from(HEADER_SIZE) {
            bail!("unsupported .fidx records offset {}", self.records_offset);
        }
        match self.record_kind {
            RECORD_FIXED_OFFSET
                if self.record_size == FIXED_RECORD_SIZE
                    && self.flags & FLAG_FIXED_BLOCK != 0
                    && self.fixed_block_size != 0 => {}
            RECORD_EXPLICIT
                if self.record_size == EXPLICIT_RECORD_SIZE
                    && self.flags & FLAG_FIXED_BLOCK == 0
                    && self.fixed_block_size == 0 => {}
            _ => bail!("inconsistent .fidx record layout"),
        }
        let expected_records = self.frame_count.div_ceil(u64::from(self.stride));
        if self.record_count != expected_records {
            bail!(
                ".fidx record count mismatch: expected {expected_records}, got {}",
                self.record_count
            );
        }
        if self.metadata_end >= self.source_size || self.observed_total_samples == 0 {
            bail!("invalid .fidx source/sample bounds");
        }
        Ok(())
    }

    fn encode(&self) -> [u8; HEADER_SIZE as usize] {
        let mut out = [0u8; HEADER_SIZE as usize];
        out[0..8].copy_from_slice(MAGIC);
        put_u16(&mut out, 8, VERSION);
        put_u16(&mut out, 10, HEADER_SIZE);
        put_u32(&mut out, 12, self.flags);
        put_u16(&mut out, 16, self.record_kind);
        put_u16(&mut out, 18, self.record_size);
        put_u32(&mut out, 20, self.stride);
        put_u64(&mut out, 24, self.source_size);
        put_u64(&mut out, 32, self.metadata_end);
        put_u64(&mut out, 40, self.source_mtime_ns);
        put_u32(&mut out, 48, self.sample_rate);
        put_u16(&mut out, 52, self.channels);
        put_u16(&mut out, 54, self.bits_per_sample);
        put_u32(&mut out, 56, self.min_block_size);
        put_u32(&mut out, 60, self.max_block_size);
        put_u32(&mut out, 64, self.fixed_block_size);
        put_u64(&mut out, 72, self.first_frame_number);
        put_u64(&mut out, 80, self.observed_total_samples);
        put_u64(&mut out, 88, self.frame_count);
        put_u64(&mut out, 96, self.record_count);
        put_u64(&mut out, 104, self.records_offset);
        out[112..144].copy_from_slice(&self.source_sha256);
        out[144..160].copy_from_slice(&self.stream_md5);
        out[160..192].copy_from_slice(&self.records_sha256);
        out
    }

    fn decode(bytes: &[u8; HEADER_SIZE as usize]) -> Result<Self> {
        if &bytes[0..8] != MAGIC {
            bail!("not a FLAC .fidx sidecar");
        }
        let version = get_u16(bytes, 8);
        if version != VERSION {
            bail!("unsupported .fidx version {version}");
        }
        let header_size = get_u16(bytes, 10);
        if header_size != HEADER_SIZE {
            bail!("unsupported .fidx header size {header_size}");
        }
        if bytes[68..72] != [0; 4] {
            bail!(".fidx reserved header bytes are nonzero");
        }
        let mut source_sha256 = [0u8; 32];
        source_sha256.copy_from_slice(&bytes[112..144]);
        let mut stream_md5 = [0u8; 16];
        stream_md5.copy_from_slice(&bytes[144..160]);
        let mut records_sha256 = [0u8; 32];
        records_sha256.copy_from_slice(&bytes[160..192]);
        Ok(Self {
            flags: get_u32(bytes, 12),
            record_kind: get_u16(bytes, 16),
            record_size: get_u16(bytes, 18),
            stride: get_u32(bytes, 20),
            source_size: get_u64(bytes, 24),
            metadata_end: get_u64(bytes, 32),
            source_mtime_ns: get_u64(bytes, 40),
            sample_rate: get_u32(bytes, 48),
            channels: get_u16(bytes, 52),
            bits_per_sample: get_u16(bytes, 54),
            min_block_size: get_u32(bytes, 56),
            max_block_size: get_u32(bytes, 60),
            fixed_block_size: get_u32(bytes, 64),
            first_frame_number: get_u64(bytes, 72),
            observed_total_samples: get_u64(bytes, 80),
            frame_count: get_u64(bytes, 88),
            record_count: get_u64(bytes, 96),
            records_offset: get_u64(bytes, 104),
            source_sha256,
            stream_md5,
            records_sha256,
        })
    }
}

pub fn build(
    source_path: &Path,
    output_path: &Path,
    stride: u32,
    overwrite: bool,
) -> Result<IndexSummary> {
    if stride == 0 {
        bail!(".fidx stride must be at least 1");
    }
    if output_path.exists() && !overwrite {
        bail!("output already exists: {}", output_path.display());
    }

    let mut metadata_file = File::open(source_path)
        .with_context(|| format!("failed to open {}", source_path.display()))?;
    let (metadata_end, stream_info) = read_stream_info(&mut metadata_file)?;
    let source_metadata = metadata_file.metadata()?;
    let source_size = source_metadata.len();
    let source_mtime_ns = source_metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok())
        .unwrap_or(0);
    let fixed_block =
        stream_info.min_block_size != 0 && stream_info.min_block_size == stream_info.max_block_size;
    let (record_kind, record_size) = if fixed_block {
        (RECORD_FIXED_OFFSET, FIXED_RECORD_SIZE)
    } else {
        (RECORD_EXPLICIT, EXPLICIT_RECORD_SIZE)
    };

    let temporary_path = temporary_path(output_path);
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
            .with_context(|| format!("failed to create {}", temporary_path.display()))?,
    );
    output.write_all(&[0u8; HEADER_SIZE as usize])?;
    let result = (|| -> Result<Header> {
        let source = File::open(source_path)?;
        let mss = MediaSourceStream::new(Box::new(source), Default::default());
        let mut reader = FlacReader::try_new(mss, FormatOptions::default())
            .context("failed to parse FLAC while building .fidx")?;
        let params = reader
            .tracks()
            .first()
            .and_then(|track| track.codec_params.as_ref())
            .and_then(CodecParameters::audio)
            .context("FLAC track is not audio")?;
        if params.sample_rate != Some(stream_info.sample_rate)
            || params.bits_per_sample != Some(u32::from(stream_info.bits_per_sample))
            || params.channels.as_ref().map(|channels| channels.count())
                != Some(usize::from(stream_info.channels))
        {
            bail!("FLAC parser parameters disagree with STREAMINFO");
        }

        let mut byte_offset = metadata_end;
        let mut expected_sample = 0u64;
        let mut frame_count = 0u64;
        let mut record_count = 0u64;
        let mut records_hash = Sha256::new();
        while let Some(packet) = reader
            .next_packet()
            .context("FLAC frame parse/CRC failure")?
        {
            let sample_start = packet.pts.get() as u64;
            let block_size = packet.dur.get();
            if sample_start != expected_sample {
                bail!(
                    "non-contiguous FLAC samples at frame {frame_count}: expected {expected_sample}, got {sample_start}"
                );
            }
            if block_size == 0 || block_size > u64::from(u32::MAX) {
                bail!("invalid FLAC block size {block_size} at frame {frame_count}");
            }
            if frame_count.checked_rem(u64::from(stride)) == Some(0) {
                if fixed_block {
                    let record = byte_offset.to_le_bytes();
                    output.write_all(&record)?;
                    records_hash.update(record);
                } else {
                    let mut record = [0u8; EXPLICIT_RECORD_SIZE as usize];
                    record[0..8].copy_from_slice(&sample_start.to_le_bytes());
                    record[8..16].copy_from_slice(&byte_offset.to_le_bytes());
                    record[16..20].copy_from_slice(&(block_size as u32).to_le_bytes());
                    output.write_all(&record)?;
                    records_hash.update(record);
                }
                record_count += 1;
            }
            byte_offset = byte_offset
                .checked_add(u64::try_from(packet.data.len()).context("FLAC frame is too large")?)
                .context("FLAC byte offset overflow")?;
            expected_sample = sample_start
                .checked_add(block_size)
                .context("FLAC sample count overflow")?;
            frame_count += 1;
        }
        if frame_count == 0 {
            bail!("FLAC contains no audio frames");
        }
        if byte_offset != source_size {
            bail!(
                "validated FLAC frames end at byte {byte_offset}, but source length is {source_size}"
            );
        }

        output.flush()?;
        let mut source_for_hash = File::open(source_path)?;
        let source_sha256 = sha256_reader(&mut source_for_hash)?;
        let records_sha256: [u8; 32] = records_hash.finalize().into();
        let mut flags = FLAG_SOURCE_SHA256 | FLAG_COMPLETE;
        if fixed_block {
            flags |= FLAG_FIXED_BLOCK;
        }
        Ok(Header {
            flags,
            record_kind,
            record_size,
            stride,
            source_size,
            metadata_end,
            source_mtime_ns,
            sample_rate: stream_info.sample_rate,
            channels: stream_info.channels,
            bits_per_sample: stream_info.bits_per_sample,
            min_block_size: stream_info.min_block_size,
            max_block_size: stream_info.max_block_size,
            fixed_block_size: if fixed_block {
                stream_info.min_block_size
            } else {
                0
            },
            first_frame_number: 0,
            observed_total_samples: expected_sample,
            frame_count,
            record_count,
            records_offset: u64::from(HEADER_SIZE),
            source_sha256,
            stream_md5: stream_info.stream_md5,
            records_sha256,
        })
    })();

    let header = match result {
        Ok(header) => header,
        Err(error) => {
            drop(output);
            let _ = std::fs::remove_file(&temporary_path);
            return Err(error);
        }
    };
    output.seek(SeekFrom::Start(0))?;
    output.write_all(&header.encode())?;
    output.flush()?;
    output.get_ref().sync_all()?;
    drop(output);
    if let Err(first_error) = std::fs::rename(&temporary_path, output_path) {
        if overwrite && output_path.exists() {
            std::fs::remove_file(output_path)
                .with_context(|| format!("failed to replace {}", output_path.display()))?;
            std::fs::rename(&temporary_path, output_path).with_context(|| {
                format!(
                    "failed to move completed index {} to {}",
                    temporary_path.display(),
                    output_path.display()
                )
            })?;
        } else {
            return Err(first_error).with_context(|| {
                format!(
                    "failed to move completed index {} to {}",
                    temporary_path.display(),
                    output_path.display()
                )
            });
        }
    }
    let index = FlacIndex::read(output_path)?;
    Ok(index.summary())
}

fn read_stream_info(file: &mut File) -> Result<(u64, StreamInfo)> {
    file.seek(SeekFrom::Start(0))?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"fLaC" {
        bail!("source is not a native FLAC stream");
    }
    let mut stream_info = None;
    loop {
        let mut header = [0u8; 4];
        file.read_exact(&mut header)
            .context("truncated FLAC metadata header")?;
        let last = header[0] & 0x80 != 0;
        let block_type = header[0] & 0x7f;
        let length = u32::from_be_bytes([0, header[1], header[2], header[3]]) as usize;
        let mut body = vec![0u8; length];
        file.read_exact(&mut body)
            .context("truncated FLAC metadata block")?;
        if block_type == 0 {
            if length != 34 || stream_info.is_some() {
                bail!("invalid FLAC STREAMINFO block");
            }
            let packed = u64::from_be_bytes(body[10..18].try_into().unwrap());
            let mut stream_md5 = [0u8; 16];
            stream_md5.copy_from_slice(&body[18..34]);
            stream_info = Some(StreamInfo {
                min_block_size: u32::from(u16::from_be_bytes(body[0..2].try_into().unwrap())),
                max_block_size: u32::from(u16::from_be_bytes(body[2..4].try_into().unwrap())),
                sample_rate: ((packed >> 44) & 0x000f_ffff) as u32,
                channels: (((packed >> 41) & 0x7) + 1) as u16,
                bits_per_sample: (((packed >> 36) & 0x1f) + 1) as u16,
                stream_md5,
            });
        }
        if last {
            break;
        }
    }
    Ok((
        file.stream_position()?,
        stream_info.context("FLAC has no STREAMINFO block")?,
    ))
}

fn sha256_reader(reader: &mut impl Read) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(hash.finalize().into())
}

fn temporary_path(output: &Path) -> PathBuf {
    let mut name = output.as_os_str().to_owned();
    name.push(format!(".tmp-{}", std::process::id()));
    PathBuf::from(name)
}

fn strictly_increasing(values: impl IntoIterator<Item = u64>) -> bool {
    let mut previous = None;
    for value in values {
        if previous.is_some_and(|previous| value <= previous) {
            return false;
        }
        previous = Some(value);
    }
    true
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut out, "{byte:02x}").unwrap();
    }
    out
}

fn put_u16(out: &mut [u8], offset: usize, value: u16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_u32(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_u64(out: &mut [u8], offset: usize, value: u64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
fn get_u16(input: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(input[offset..offset + 2].try_into().unwrap())
}
fn get_u32(input: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap())
}
fn get_u64(input: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(input[offset..offset + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip_is_exactly_192_bytes() {
        let header = Header {
            flags: FLAG_SOURCE_SHA256 | FLAG_FIXED_BLOCK | FLAG_COMPLETE,
            record_kind: RECORD_FIXED_OFFSET,
            record_size: FIXED_RECORD_SIZE,
            stride: 4096,
            source_size: 99_000,
            metadata_end: 42,
            source_mtime_ns: 123,
            sample_rate: 28_636_363,
            channels: 1,
            bits_per_sample: 8,
            min_block_size: 4096,
            max_block_size: 4096,
            fixed_block_size: 4096,
            first_frame_number: 0,
            observed_total_samples: 1_000_000,
            frame_count: 245,
            record_count: 1,
            records_offset: 192,
            source_sha256: [0x11; 32],
            stream_md5: [0x22; 16],
            records_sha256: [0x33; 32],
        };
        let encoded = header.encode();
        assert_eq!(encoded.len(), 192);
        let decoded = Header::decode(&encoded).unwrap();
        assert_eq!(decoded.source_size, header.source_size);
        assert_eq!(decoded.records_sha256, header.records_sha256);
        decoded.validate().unwrap();
    }

    #[test]
    fn fixed_anchor_bounds_forward_scan() {
        let index = FlacIndex {
            header: Header {
                flags: FLAG_SOURCE_SHA256 | FLAG_FIXED_BLOCK | FLAG_COMPLETE,
                record_kind: RECORD_FIXED_OFFSET,
                record_size: FIXED_RECORD_SIZE,
                stride: 4,
                source_size: 10_000,
                metadata_end: 100,
                source_mtime_ns: 0,
                sample_rate: 48_000,
                channels: 1,
                bits_per_sample: 16,
                min_block_size: 4096,
                max_block_size: 4096,
                fixed_block_size: 4096,
                first_frame_number: 0,
                observed_total_samples: 40 * 4096,
                frame_count: 40,
                record_count: 10,
                records_offset: 192,
                source_sha256: [0; 32],
                stream_md5: [0; 16],
                records_sha256: [0; 32],
            },
            records: Records::Fixed((0..10).map(|i| 100 + i * 800).collect()),
        };
        let anchor = index.anchor_for_sample(23 * 4096 + 17).unwrap();
        assert_eq!(anchor.sample_start, 20 * 4096);
        assert_eq!(anchor.byte_offset, 4_100);
    }
}
