use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::time::Instant;

use anyhow::{Context as _, Result};
use tape_decode::{DecoderMetadata, LumaOutput, WriteableField};

use crate::metadata::metadata_to_tbc;

pub struct DecodeWriter {
    outfile_video: File,
    outfile_chroma: Option<File>,
    /// JSON sidecar, kept open and rewritten incrementally: after every field the
    /// file on disk is a complete, valid document (see [`Self::write_writeable`]).
    json_file: Option<File>,
    /// Byte offset of the array-closing `]` in the JSON file — i.e. just past the
    /// last field object written. The next field, or the final metadata, is
    /// written starting here, overwriting the previous `]` + trailing metadata.
    json_field_end: u64,
    field_count: usize,
    first_field_write: Option<Instant>,
    last_field_write: Option<Instant>,
}

fn fps(field_count: usize, elapsed: f64) -> f64 {
    if elapsed > 0.0 {
        field_count as f64 / (elapsed * 2.0)
    } else {
        0.0
    }
}

impl DecodeWriter {
    pub fn new(luma: File, chroma: Option<File>, mut metadata: Option<File>) -> Result<Self> {
        const FIELDS_OPEN: &[u8] = b"{\"fields\":[";
        if let Some(metadata) = metadata.as_mut() {
            let mut chunk = FIELDS_OPEN.to_vec();
            append_tail(&mut chunk, None, 0)?;
            metadata.write_all(&chunk)?;
        }
        Ok(Self {
            outfile_video: luma,
            outfile_chroma: chroma,
            json_file: metadata,
            json_field_end: FIELDS_OPEN.len() as u64,
            field_count: 0,
            first_field_write: None,
            last_field_write: None,
        })
    }

    pub(crate) fn write_writeable(
        &mut self,
        field: &WriteableField,
        metadata: Option<&DecoderMetadata>,
    ) -> Result<()> {
        match field.luma() {
            LumaOutput::Encoded(values) => write_u16_le_slice(&mut self.outfile_video, values)?,
            LumaOutput::Raw(values) => write_f32_le_slice(&mut self.outfile_video, values)?,
        }
        if let Some(outfile) = &mut self.outfile_chroma {
            let chroma = field
                .chroma()
                .context("missing chroma output for chroma file")?;
            write_u16_le_slice(outfile, chroma)?;
        }
        let now = Instant::now();
        let start = self.first_field_write.get_or_insert(now);
        self.last_field_write = Some(now);

        self.field_count += 1;

        if let Some(json_file) = self.json_file.as_mut() {
            // Starting at the array-closing `]`, append this field's object followed
            // by a fresh `]` and the metadata "as if decoding had finished here", so
            // the file is a complete, valid document the instant it hits disk. Track
            // where this field's object ends and seek back there: the next field
            // overwrites the tail with `,<field>]...`, and the final pass overwrites
            // it with the authoritative metadata. Field + tail go out as one write.

            let mut chunk = Vec::new();
            if self.field_count > 1 {
                chunk.push(b',');
            }
            serde_json::to_writer(&mut chunk, &field.info)?;
            let field_end = self.json_field_end + chunk.len() as u64;

            json_file.seek(SeekFrom::Start(self.json_field_end))?;
            append_tail(&mut chunk, metadata, self.field_count)?;
            json_file.write_all(&chunk)?;

            self.json_field_end = field_end;
        }

        const LOG_INTERVAL: usize = 500;

        let field_num = self.field_count;
        tracing::debug!("Written field {field_num}");
        if field_num.is_multiple_of(LOG_INTERVAL) {
            let elapsed = now.duration_since(*start).as_secs_f64();
            let fps = fps(field_num, elapsed);
            tracing::info!(
                "Decoded {} fields so far in {:.3}s ({:.2} FPS)",
                field_num,
                elapsed,
                fps
            );
        }

        Ok(())
    }

    pub(crate) fn close(&mut self, metadata: Option<DecoderMetadata>) -> Result<()> {
        let field_count = self.field_count;
        let elapsed = match (self.first_field_write, self.last_field_write) {
            (Some(first), Some(last)) => last.duration_since(first).as_secs_f64(),
            _ => 0.0,
        };
        let fps = fps(field_count, elapsed);
        tracing::info!(
            "Decode finished: {} fields in {:.3}s ({:.2} FPS)",
            field_count,
            elapsed,
            fps,
        );

        if let Some(json_file) = self.json_file.as_mut() {
            // Overwrite the trailing `]` + metadata with the final, authoritative
            // metadata and truncate, so the file matches a one-shot serialization
            // exactly (the array was opened up front by `open`).
            let mut chunk = Vec::new();
            append_tail(&mut chunk, metadata.as_ref(), field_count)?;
            json_file.seek(SeekFrom::Start(self.json_field_end))?;
            json_file.write_all(&chunk)?;
            json_file.set_len(self.json_field_end + chunk.len() as u64)?;
        }

        Ok(())
    }
}

/// Append the array-closing `]`, the flattened metadata (when present), and the
/// document-closing `}` plus newline to `chunk`. With `metadata` set this yields
/// `],"pcmAudioParameters":{…},"videoParameters":{…}}\n`; without it, `]}\n`.
fn append_tail(
    chunk: &mut Vec<u8>,
    metadata: Option<&DecoderMetadata>,
    field_count: usize,
) -> Result<()> {
    chunk.push(b']');
    if let Some(metadata) = metadata {
        // Serialize the metadata object on its own, then splice its body into the
        // root by replacing its leading `{` with `,` and dropping its trailing `}`.
        let bytes = serde_json::to_vec(&metadata_to_tbc(metadata, field_count))?;
        chunk.push(b',');
        chunk.extend_from_slice(&bytes[1..bytes.len() - 1]);
    }
    chunk.push(b'}');
    chunk.push(b'\n');
    Ok(())
}

/// Write the canonical little-endian representation of a `u16` raster.
///
/// All supported production targets are currently little-endian, so retain the
/// zero-copy path there. `u16` has no padding bytes and every bit pattern is
/// valid, which makes viewing this initialized slice as bytes sound.
#[cfg(target_endian = "little")]
fn write_u16_le_slice(file: &mut dyn Write, values: &[u16]) -> Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    file.write_all(bytes)?;
    Ok(())
}

#[cfg(target_endian = "big")]
fn write_u16_le_slice(file: &mut dyn Write, values: &[u16]) -> Result<()> {
    write_u16_le_converted(file, values)
}

/// Write the canonical little-endian representation of a raw `f32` luma
/// raster. This is a bit-preserving operation: signed zero and NaN payloads are
/// output exactly as stored in the source slice.
#[cfg(target_endian = "little")]
fn write_f32_le_slice(file: &mut dyn Write, values: &[f32]) -> Result<()> {
    // `f32` has no padding bytes and every bit pattern has a defined object
    // representation, so this initialized slice can safely be viewed as bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    file.write_all(bytes)?;
    Ok(())
}

#[cfg(target_endian = "big")]
fn write_f32_le_slice(file: &mut dyn Write, values: &[f32]) -> Result<()> {
    write_f32_le_converted(file, values)
}

// Keep the big-endian fallback bounded: a full field should not require a
// field-sized allocation merely to swap byte order.
#[cfg(any(target_endian = "big", test))]
const ENDIAN_CONVERSION_BUFFER_BYTES: usize = 16 * 1024;

#[cfg(any(target_endian = "big", test))]
fn write_u16_le_converted(file: &mut dyn Write, values: &[u16]) -> Result<()> {
    let mut bytes = [0_u8; ENDIAN_CONVERSION_BUFFER_BYTES];
    for values_chunk in values.chunks(bytes.len() / std::mem::size_of::<u16>()) {
        let output = &mut bytes[..std::mem::size_of_val(values_chunk)];
        for (value, destination) in values_chunk.iter().zip(output.chunks_exact_mut(2)) {
            destination.copy_from_slice(&value.to_le_bytes());
        }
        file.write_all(output)?;
    }
    Ok(())
}

#[cfg(any(target_endian = "big", test))]
fn write_f32_le_converted(file: &mut dyn Write, values: &[f32]) -> Result<()> {
    let mut bytes = [0_u8; ENDIAN_CONVERSION_BUFFER_BYTES];
    for values_chunk in values.chunks(bytes.len() / std::mem::size_of::<f32>()) {
        let output = &mut bytes[..std::mem::size_of_val(values_chunk)];
        for (value, destination) in values_chunk.iter().zip(output.chunks_exact_mut(4)) {
            // Work from the stored IEEE-754 bits so this conversion cannot lose
            // a sign bit or normalize a NaN payload.
            destination.copy_from_slice(&value.to_bits().to_le_bytes());
        }
        file.write_all(output)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        write_f32_le_converted, write_f32_le_slice, write_u16_le_converted, write_u16_le_slice,
        ENDIAN_CONVERSION_BUFFER_BYTES,
    };

    #[test]
    fn encoded_luma_and_chroma_are_little_endian() {
        let values = [0x0000_u16, 0x0001, 0x1234, 0x8000, 0xffff];
        let expected = [0x00, 0x00, 0x01, 0x00, 0x34, 0x12, 0x00, 0x80, 0xff, 0xff];

        let mut actual = Vec::new();
        write_u16_le_slice(&mut actual, &values).unwrap();
        assert_eq!(actual, expected);

        let mut converted = Vec::new();
        write_u16_le_converted(&mut converted, &values).unwrap();
        assert_eq!(converted, expected);
    }

    #[test]
    fn raw_luma_preserves_ieee_754_bits_in_little_endian_order() {
        let values = [
            f32::from_bits(0x0000_0000), // +0.0
            f32::from_bits(0x8000_0000), // -0.0
            f32::from_bits(0x3f80_0000), // 1.0
            f32::from_bits(0x7fc0_1234), // NaN with a payload
            f32::from_bits(0xff80_0000), // -infinity
        ];
        let expected = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x80, 0x3f, 0x34, 0x12,
            0xc0, 0x7f, 0x00, 0x00, 0x80, 0xff,
        ];

        let mut actual = Vec::new();
        write_f32_le_slice(&mut actual, &values).unwrap();
        assert_eq!(actual, expected);

        let mut converted = Vec::new();
        write_f32_le_converted(&mut converted, &values).unwrap();
        assert_eq!(converted, expected);
    }

    #[test]
    fn conversion_is_exact_across_internal_buffer_boundaries() {
        let value_count = ENDIAN_CONVERSION_BUFFER_BYTES / std::mem::size_of::<u16>() + 3;
        let values = (0..value_count)
            .map(|index| index as u16 ^ 0xa55a)
            .collect::<Vec<_>>();
        let expected = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();

        let mut actual = Vec::new();
        write_u16_le_converted(&mut actual, &values).unwrap();
        assert_eq!(actual, expected);
    }
}
