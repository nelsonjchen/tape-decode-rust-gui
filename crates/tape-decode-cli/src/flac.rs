//! FLAC input via symphonia. Decodes a mono FLAC stream to `f32` samples behind
//! the same [`SampleSource`] interface as the raw formats. Signed integer FLAC
//! samples are normalized to ~[-1, 1], so a 16-bit FLAC reads identically to the
//! equivalent raw `s16` capture. A validated `.fidx` sidecar can reopen a
//! seek-table-free source at a sparse frame anchor; otherwise unknown-length
//! streams retain a forward-only sequential fallback.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

use anyhow::{bail, Context as _, Result};
use symphonia_bundle_flac::{FlacDecoder, FlacReader};
use symphonia_core::audio::{Audio, GenericAudioBufferRef};
use symphonia_core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia_core::codecs::CodecParameters;
use symphonia_core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia_core::io::{MediaSource, MediaSourceStream};
use symphonia_core::units::Timestamp;

use crate::fidx::{FlacIndex, FrameAnchor};
use crate::reader::SampleSource;

/// Decode `source` (a seekable file or a forward-only pipe) as a FLAC sample
/// source.
pub fn open(source: Box<dyn MediaSource>) -> Result<Box<dyn SampleSource>> {
    Ok(Box::new(FlacSource::new(source, None)?))
}

/// Decode a regular FLAC file with frame seeks bounded by `index.stride`.
pub fn open_indexed(source: File, index: FlacIndex) -> Result<Box<dyn SampleSource>> {
    let initial = FrameAnchor {
        sample_start: 0,
        byte_offset: index.metadata_end(),
    };
    let view = SegmentMediaSource::new(
        source.try_clone().context("failed to clone indexed FLAC")?,
        index.metadata_end(),
        initial.byte_offset,
        index.source_size(),
    )?;
    Ok(Box::new(FlacSource::new(
        Box::new(view),
        Some(IndexedAccess { source, index }),
    )?))
}

struct IndexedAccess {
    source: File,
    index: FlacIndex,
}

struct FlacSource {
    reader: FlacReader<'static>,
    decoder: FlacDecoder,
    track_id: u32,
    /// Right shift undoing symphonia's normalization of samples to the full i32
    /// range (`32 - bits_per_sample`), then `scale` normalizes the native signed
    /// integer amplitude to match raw signed PCM input.
    shift: u32,
    scale: f32,
    /// Whether the source supports real seeking (a regular file); pipes can only
    /// skip forward by decoding.
    seekable: bool,
    /// Streams with a declared length can use symphonia's ordinary binary seek.
    /// Unknown-length preservation FLACs stay sequential unless `.fidx` exists.
    known_length: bool,
    indexed: Option<IndexedAccess>,
    /// Decoded samples of the current packet not yet returned.
    pending: Vec<f32>,
    pending_pos: usize,
    /// Absolute index of the next sample `read` will return.
    position: u64,
    eof: bool,
}

impl FlacSource {
    fn new(source: Box<dyn MediaSource>, indexed: Option<IndexedAccess>) -> Result<Self> {
        let seekable = source.is_seekable();
        let mss = MediaSourceStream::new(source, Default::default());
        let reader =
            FlacReader::try_new(mss, FormatOptions::default()).context("failed to read FLAC")?;

        let (track_id, known_length, params) = {
            let track = reader.tracks().first().context("FLAC has no tracks")?;
            let params = track
                .codec_params
                .as_ref()
                .and_then(CodecParameters::audio)
                .context("FLAC track is not audio")?
                .clone();
            (track.id, track.num_frames.is_some(), params)
        };

        let bits = params
            .bits_per_sample
            .context("FLAC stream info missing bits per sample")?;
        if !(1..=32).contains(&bits) {
            bail!("unsupported FLAC bit depth: {bits}");
        }
        let channels = params.channels.as_ref().map_or(1, |c| c.count());
        if channels != 1 {
            bail!("only mono FLAC input is supported, found {channels} channels");
        }

        let decoder = FlacDecoder::try_new(&params, &AudioDecoderOptions::default())
            .context("failed to initialize FLAC decoder")?;

        let scale = 1.0 / ((1u64 << (bits - 1)) as f32);

        Ok(Self {
            reader,
            decoder,
            track_id,
            shift: 32 - bits,
            scale,
            seekable,
            known_length,
            indexed,
            pending: Vec::new(),
            pending_pos: 0,
            position: 0,
            eof: false,
        })
    }

    /// Decode the next packet into `pending`. Returns false at end of input.
    fn decode_next(&mut self) -> Result<bool> {
        let shift = self.shift;
        let scale = self.scale;
        loop {
            let Some(packet) = self.reader.next_packet().context("FLAC read error")? else {
                self.eof = true;
                return Ok(false);
            };
            if packet.track_id != self.track_id {
                continue;
            }
            let decoded = self.decoder.decode(&packet).context("FLAC decode error")?;
            if decoded.frames() == 0 {
                continue;
            }
            let GenericAudioBufferRef::S32(buf) = decoded else {
                bail!("unexpected FLAC sample format (expected signed 32-bit)");
            };
            let plane = buf.plane(0).expect("mono FLAC has one plane");
            self.pending.clear();
            self.pending
                .extend(plane.iter().map(|&s| ((s >> shift) as f32) * scale));
            self.pending_pos = 0;
            return Ok(true);
        }
    }

    /// Ensure `pending` has unconsumed samples, decoding if needed. False at EOF.
    fn fill(&mut self) -> Result<bool> {
        if self.pending_pos < self.pending.len() {
            return Ok(true);
        }
        if self.eof {
            return Ok(false);
        }
        self.decode_next()
    }

    /// Drop samples until `position` reaches `target` (forward only).
    fn skip_to(&mut self, target: u64) -> Result<()> {
        while self.position < target {
            if !self.fill()? {
                bail!("FLAC input ended before reaching sample offset {target}");
            }
            let avail = self.pending.len() - self.pending_pos;
            let n = avail.min((target - self.position) as usize);
            self.pending_pos += n;
            self.position += n as u64;
        }
        Ok(())
    }

    fn indexed_seek(&mut self, sample: u64) -> Result<()> {
        let access = self.indexed.as_ref().expect("indexed seek requires index");
        let anchor = access.index.anchor_for_sample(sample)?;
        let view = SegmentMediaSource::new(
            access
                .source
                .try_clone()
                .context("failed to clone indexed FLAC")?,
            access.index.metadata_end(),
            anchor.byte_offset,
            access.index.source_size(),
        )?;
        let mss = MediaSourceStream::new(Box::new(view), Default::default());
        let reader = FlacReader::try_new(mss, FormatOptions::default())
            .context("failed to reopen FLAC at .fidx anchor")?;
        self.reader = reader;
        self.decoder.reset();
        self.pending.clear();
        self.pending_pos = 0;
        self.eof = false;
        self.position = anchor.sample_start;
        self.skip_to(sample)
    }

    fn symphonia_seek(&mut self, sample: u64) -> Result<()> {
        let ts = Timestamp::try_from(sample).context("seek offset too large")?;
        let seeked = self
            .reader
            .seek(
                SeekMode::Coarse,
                SeekTo::Timestamp {
                    ts,
                    track_id: self.track_id,
                },
            )
            .context("FLAC seek failed")?;
        self.decoder.reset();
        self.pending.clear();
        self.pending_pos = 0;
        self.eof = false;
        self.position = seeked.actual_ts.get() as u64;
        self.skip_to(sample)
    }
}

impl SampleSource for FlacSource {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let mut written = 0;
        while written < out.len() {
            if !self.fill()? {
                break;
            }
            let avail = &self.pending[self.pending_pos..];
            let n = avail.len().min(out.len() - written);
            out[written..written + n].copy_from_slice(&avail[..n]);
            self.pending_pos += n;
            written += n;
        }
        self.position += written as u64;
        Ok(written)
    }

    fn seek_samples(&mut self, sample: u64) -> Result<()> {
        if sample == self.position {
            return Ok(());
        }
        if self.indexed.is_some() {
            return self.indexed_seek(sample);
        }
        if self.seekable && (self.known_length || sample < self.position) {
            return self.symphonia_seek(sample);
        } else if sample < self.position {
            bail!("cannot seek backward on a non-seekable FLAC input (stdin)");
        }
        self.skip_to(sample)
    }
}

/// A read-only logical FLAC made from the original metadata followed by source
/// frames beginning at one indexed anchor. No source bytes are rewritten.
struct SegmentMediaSource {
    file: File,
    metadata_end: u64,
    frame_start: u64,
    source_size: u64,
    position: u64,
    logical_size: u64,
}

impl SegmentMediaSource {
    fn new(file: File, metadata_end: u64, frame_start: u64, source_size: u64) -> Result<Self> {
        if metadata_end > frame_start || frame_start >= source_size {
            bail!("invalid indexed FLAC segment bounds");
        }
        let logical_size = metadata_end
            .checked_add(source_size - frame_start)
            .context("indexed FLAC segment length overflow")?;
        Ok(Self {
            file,
            metadata_end,
            frame_start,
            source_size,
            position: 0,
            logical_size,
        })
    }

    fn source_position(&self) -> u64 {
        if self.position < self.metadata_end {
            self.position
        } else {
            self.frame_start + (self.position - self.metadata_end)
        }
    }
}

impl Read for SegmentMediaSource {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() || self.position >= self.logical_size {
            return Ok(0);
        }
        let remaining_region = if self.position < self.metadata_end {
            self.metadata_end - self.position
        } else {
            self.source_size - self.source_position()
        };
        let limit = usize::try_from(remaining_region.min(out.len() as u64)).unwrap_or(out.len());
        self.file.seek(SeekFrom::Start(self.source_position()))?;
        let read = self.file.read(&mut out[..limit])?;
        self.position += read as u64;
        Ok(read)
    }
}

impl Seek for SegmentMediaSource {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        let target = match position {
            SeekFrom::Start(value) => i128::from(value),
            SeekFrom::Current(delta) => i128::from(self.position) + i128::from(delta),
            SeekFrom::End(delta) => i128::from(self.logical_size) + i128::from(delta),
        };
        if target < 0 || target > i128::from(self.logical_size) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek outside indexed FLAC segment",
            ));
        }
        self.position = target as u64;
        Ok(self.position)
    }
}

impl MediaSource for SegmentMediaSource {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.logical_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn segment_source_splices_metadata_to_indexed_frame_without_writes() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("tape-decode-fidx-{nonce}.bin"));
        std::fs::write(&path, b"METAxxxxFRAMEPAYLOAD").unwrap();
        let file = File::open(&path).unwrap();
        let mut source = SegmentMediaSource::new(file, 4, 8, 20).unwrap();
        let mut bytes = Vec::new();
        source.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"METAFRAMEPAYLOAD");
        source.seek(SeekFrom::Start(2)).unwrap();
        let mut middle = [0u8; 6];
        source.read_exact(&mut middle).unwrap();
        assert_eq!(&middle, b"TAFRAM");
        assert_eq!(std::fs::read(&path).unwrap(), b"METAxxxxFRAMEPAYLOAD");
        std::fs::remove_file(path).unwrap();
    }
}
