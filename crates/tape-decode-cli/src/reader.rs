use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::marker::PhantomData;

use anyhow::{bail, Context as _, Result};
use symphonia_core::io::{MediaSource, MediaSourceStream, ReadBytes};
use tracing::error;

/// Input encoding. The raw formats widen straight to `f32` with no rescaling
/// (so `S16LE` is little-endian `i16`, `F32LE` is passed through verbatim);
/// `Flac` is decoded by [`crate::flac`].
#[derive(Clone, Copy, Debug)]
pub enum SampleFormat {
    U8,
    S8,
    S16LE,
    U16LE,
    F32LE,
    Flac,
}

/// How one on-disk sample is sized and widened to `f32`. Implemented by a
/// zero-sized marker per [`SampleFormat`] so the per-sample widening loop is
/// monomorphized into [`Source`] with no runtime branch on the format.
trait SampleEncoding: Send {
    /// Bytes occupied by one sample on disk.
    const BYTES: usize;
    /// Widen `bytes` (exactly `out.len() * BYTES` long) into `out`.
    fn widen(bytes: &[u8], out: &mut [f32]);
}

/// One unsigned byte per sample, centered and normalized to ~[-1, 1].
struct U8Sample;
impl SampleEncoding for U8Sample {
    const BYTES: usize = 1;
    fn widen(bytes: &[u8], out: &mut [f32]) {
        const MID: f32 = 127.5;
        const INV: f32 = 1.0 / 127.5;
        for (dst, &byte) in out.iter_mut().zip(bytes) {
            *dst = (f32::from(byte) - MID) * INV;
        }
    }
}

/// One signed byte per sample, normalized to ~[-1, 1].
struct S8Sample;
impl SampleEncoding for S8Sample {
    const BYTES: usize = 1;
    fn widen(bytes: &[u8], out: &mut [f32]) {
        const INV: f32 = 1.0 / 128.0;
        for (dst, &byte) in out.iter_mut().zip(bytes) {
            *dst = f32::from(byte as i8) * INV;
        }
    }
}

/// One little-endian `i16` per sample, normalized to ~[-1, 1].
struct S16Sample;
impl SampleEncoding for S16Sample {
    const BYTES: usize = 2;
    fn widen(bytes: &[u8], out: &mut [f32]) {
        const INV: f32 = 1.0 / 32768.0;
        for (dst, word) in out.iter_mut().zip(bytes.chunks_exact(2)) {
            *dst = f32::from(i16::from_le_bytes([word[0], word[1]])) * INV;
        }
    }
}

/// One little-endian `u16` per sample, centered and normalized to ~[-1, 1].
struct U16Sample;
impl SampleEncoding for U16Sample {
    const BYTES: usize = 2;
    fn widen(bytes: &[u8], out: &mut [f32]) {
        const MID: f32 = 32767.5;
        const INV: f32 = 1.0 / 32767.5;
        for (dst, word) in out.iter_mut().zip(bytes.chunks_exact(2)) {
            *dst = (f32::from(u16::from_le_bytes([word[0], word[1]])) - MID) * INV;
        }
    }
}

/// One little-endian `f32` per sample, passed through verbatim.
struct F32Sample;
impl SampleEncoding for F32Sample {
    const BYTES: usize = 4;
    fn widen(bytes: &[u8], out: &mut [f32]) {
        for (dst, word) in out.iter_mut().zip(bytes.chunks_exact(4)) {
            *dst = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        }
    }
}

/// Streams input as `f32` samples. Templated over the [`SampleEncoding`] (width
/// and widening) so the widening loop is monomorphized; the seek strategy is
/// chosen at runtime from the [`MediaSource`]'s seekability. The underlying file
/// vs pipe distinction is hidden by symphonia's [`MediaSourceStream`], whose sole
/// dynamic dispatch is the boxed [`MediaSource`] it wraps.
struct Source<F: SampleEncoding> {
    stream: MediaSourceStream<'static>,
    // Reusable byte scratch so a read does not allocate per call.
    scratch: Vec<u8>,
    _format: PhantomData<F>,
}

impl<F: SampleEncoding> Source<F> {
    fn new(stream: MediaSourceStream<'static>) -> Self {
        Self {
            stream,
            scratch: Vec::new(),
            _format: PhantomData,
        }
    }
}

impl<F: SampleEncoding> SampleSource for Source<F> {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let want = out.len() * F::BYTES;
        self.scratch.resize(want, 0);
        let mut filled = 0;
        while filled < want {
            let read = self.stream.read(&mut self.scratch[filled..])?;
            if read == 0 {
                break;
            }
            filled += read;
        }
        let samples = filled / F::BYTES;
        F::widen(&self.scratch[..samples * F::BYTES], &mut out[..samples]);
        Ok(samples)
    }

    fn seek_samples(&mut self, sample: u64) -> Result<()> {
        let target = sample
            .checked_mul(F::BYTES as u64)
            .context("seek offset is too large for input sample width")?;
        if self.stream.is_seekable() {
            // Seekable files reposition directly, in either direction.
            self.stream
                .seek(SeekFrom::Start(target))
                .context("failed to seek input")?;
        } else {
            // Forward-only pipes reach a forward target by reading and
            // discarding; a backward target is an error.
            let pos = self.stream.pos();
            if target < pos {
                bail!("cannot seek backward on a non-seekable input (stdin)");
            }
            self.stream
                .ignore_bytes(target - pos)
                .context("unexpected end of input while skipping to sample offset on stdin")?;
        }
        Ok(())
    }
}

/// Streams the input as `f32` samples and seeks by sample index, hiding the
/// on-disk [`SampleFormat`] and whether the source is a seekable file or a
/// forward-only pipe.
pub trait SampleSource: Send {
    /// Read up to `out.len()` samples; a count below `out.len()` means EOF. A
    /// trailing partial sample at EOF is dropped. Sequential-only, valid on pipes.
    fn read(&mut self, out: &mut [f32]) -> Result<usize>;
    /// Seek to absolute sample `sample`.
    fn seek_samples(&mut self, sample: u64) -> Result<()>;
}

/// Open `input` as a sample source in the given `format`. The path `-` selects
/// standard input, read forward-only (no real seek); anything else is a regular,
/// seekable file. Raw formats stream bytes directly; `Flac` is decoded by
/// [`crate::flac`].
pub fn open_source(file: File, format: SampleFormat) -> Result<Box<dyn SampleSource>> {
    let source: Box<dyn MediaSource> = Box::new(file);
    match format {
        SampleFormat::U8 => raw::<U8Sample>(source),
        SampleFormat::S8 => raw::<S8Sample>(source),
        SampleFormat::S16LE => raw::<S16Sample>(source),
        SampleFormat::U16LE => raw::<U16Sample>(source),
        SampleFormat::F32LE => raw::<F32Sample>(source),
        SampleFormat::Flac => crate::flac::open(source),
    }
}

/// Wrap a [`MediaSource`] as a raw stream of fixed-width samples.
fn raw<F: SampleEncoding + 'static>(source: Box<dyn MediaSource>) -> Result<Box<dyn SampleSource>> {
    let stream = MediaSourceStream::new(source, Default::default());
    Ok(Box::new(Source::<F>::new(stream)))
}

/// Wraps a boxed [`SampleSource`], forwarding reads and seeks to it. The concrete
/// stream and sample format are baked into the boxed source by [`open_source`],
/// so this layer is format- and transport-agnostic.
pub struct DecodeReader {
    source: Box<dyn SampleSource>,
    eof: bool,
    position: u64,
    end_offset: Option<u64>,
}

impl DecodeReader {
    pub fn new(source: Box<dyn SampleSource>) -> Self {
        Self {
            source,
            eof: false,
            position: 0,
            end_offset: None,
        }
    }

    /// Treat `end_offset` as an exclusive synthetic EOF in the decoded sample
    /// stream. The bound is expressed in absolute input samples, just like
    /// [`Self::seek_samples`].
    pub fn with_end_offset(mut self, end_offset: Option<u64>) -> Self {
        self.end_offset = end_offset;
        self
    }

    pub fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        if self.eof {
            return Ok(0);
        }
        let permitted = match self.end_offset {
            Some(end) => end.saturating_sub(self.position).min(out.len() as u64) as usize,
            None => out.len(),
        };
        if permitted == 0 {
            self.eof = true;
            return Ok(0);
        }
        match self.source.read(&mut out[..permitted]) {
            Ok(n) => {
                self.position += n as u64;
                if n < permitted || self.end_offset == Some(self.position) {
                    self.eof = true;
                }
                Ok(n)
            }
            Err(e) => {
                error!("{e:#}");
                self.eof = true;
                Err(e)
            }
        }
    }

    pub fn seek_samples(&mut self, sample: u64) -> Result<()> {
        if self.end_offset.is_some_and(|end| sample > end) {
            bail!(
                "sample offset {sample} is past exclusive end offset {}",
                self.end_offset.unwrap()
            );
        }
        self.source.seek_samples(sample)?;
        self.position = sample;
        self.eof = self.end_offset == Some(sample);
        Ok(())
    }

    pub fn position(&self) -> u64 {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingSource {
        samples: Vec<f32>,
        position: usize,
    }

    impl SampleSource for CountingSource {
        fn read(&mut self, out: &mut [f32]) -> Result<usize> {
            let available = self.samples.len().saturating_sub(self.position);
            let count = available.min(out.len());
            out[..count].copy_from_slice(&self.samples[self.position..self.position + count]);
            self.position += count;
            Ok(count)
        }

        fn seek_samples(&mut self, sample: u64) -> Result<()> {
            self.position = usize::try_from(sample)?;
            Ok(())
        }
    }

    #[test]
    fn exclusive_end_offset_limits_reads_after_seek() {
        let source = CountingSource {
            samples: (0..100).map(|value| value as f32).collect(),
            position: 0,
        };
        let mut reader = DecodeReader::new(Box::new(source)).with_end_offset(Some(37));
        reader.seek_samples(20).unwrap();

        let mut output = [0.0; 32];
        assert_eq!(reader.read(&mut output).unwrap(), 17);
        assert_eq!(
            &output[..17],
            &(20..37).map(|value| value as f32).collect::<Vec<_>>()
        );
        assert_eq!(reader.position(), 37);
        assert_eq!(reader.read(&mut output).unwrap(), 0);
    }

    #[test]
    fn seeking_past_end_is_rejected() {
        let source = CountingSource {
            samples: vec![0.0; 100],
            position: 0,
        };
        let mut reader = DecodeReader::new(Box::new(source)).with_end_offset(Some(37));
        assert!(reader.seek_samples(38).is_err());
    }
}
