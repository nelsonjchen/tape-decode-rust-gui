//! Bounded-memory, seekable HTTP input for Symphonia.
//!
//! The source issues explicit byte-range requests and retains only one bounded
//! read-ahead window. It deliberately rejects a server that ignores `Range`,
//! since accepting a `200 OK` response could silently download a complete RF
//! capture into memory.

use std::io::{self, Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::{ACCEPT_ENCODING, CONTENT_RANGE, ETAG, RANGE};
use reqwest::StatusCode;
use serde::Serialize;
use symphonia_core::io::MediaSource;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HttpRangeRecord {
    pub start: u64,
    pub end_exclusive: u64,
}

#[derive(Debug)]
pub struct HttpRangeMetrics {
    source_length: AtomicU64,
    requests: AtomicU64,
    bytes_received: AtomicU64,
    buffer_bytes: usize,
    ranges: Mutex<Vec<HttpRangeRecord>>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HttpRangeMetricsSnapshot {
    pub source_length: u64,
    pub requests: u64,
    pub bytes_received: u64,
    pub buffer_bytes: usize,
    pub ranges: Vec<HttpRangeRecord>,
}

impl HttpRangeMetrics {
    pub fn snapshot(&self) -> HttpRangeMetricsSnapshot {
        HttpRangeMetricsSnapshot {
            source_length: self.source_length.load(Ordering::Relaxed),
            requests: self.requests.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            buffer_bytes: self.buffer_bytes,
            ranges: self.ranges.lock().unwrap().clone(),
        }
    }
}

pub struct HttpRangeSource {
    client: Client,
    url: String,
    expected_etag: Option<String>,
    source_length: u64,
    position: u64,
    buffer_start: u64,
    buffer: Vec<u8>,
    buffer_bytes: usize,
    metrics: Arc<HttpRangeMetrics>,
}

impl HttpRangeSource {
    pub fn open(
        url: String,
        expected_etag: Option<String>,
        buffer_bytes: usize,
    ) -> io::Result<(Self, Arc<HttpRangeMetrics>)> {
        if buffer_bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "HTTP range buffer must be positive",
            ));
        }
        let metrics = Arc::new(HttpRangeMetrics {
            source_length: AtomicU64::new(0),
            requests: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            buffer_bytes,
            ranges: Mutex::new(Vec::new()),
        });
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(600))
            .build()
            .map_err(io_other)?;
        let mut source = Self {
            client,
            url,
            expected_etag,
            source_length: 0,
            position: 0,
            buffer_start: 0,
            buffer: Vec::new(),
            buffer_bytes,
            metrics: Arc::clone(&metrics),
        };
        source.refill(0)?;
        Ok((source, metrics))
    }

    fn refill(&mut self, start: u64) -> io::Result<()> {
        if self.source_length != 0 && start >= self.source_length {
            self.buffer_start = start;
            self.buffer.clear();
            return Ok(());
        }
        let requested_end = start
            .saturating_add(self.buffer_bytes as u64)
            .saturating_sub(1);
        let response = self
            .client
            .get(&self.url)
            .header(RANGE, format!("bytes={start}-{requested_end}"))
            .header(ACCEPT_ENCODING, "identity")
            .send()
            .map_err(io_other)?;
        if response.status() != StatusCode::PARTIAL_CONTENT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "HTTP source ignored byte range (expected 206, got {})",
                    response.status()
                ),
            ));
        }
        if let Some(expected) = &self.expected_etag {
            let received = response
                .headers()
                .get(ETAG)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing HTTP ETag"))?;
            if normalize_etag(received) != normalize_etag(expected) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("HTTP ETag mismatch: expected {expected}, got {received}"),
                ));
            }
        }
        let content_range = response
            .headers()
            .get(CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing Content-Range"))?;
        let (actual_start, actual_end_exclusive, total) = parse_content_range(content_range)?;
        if actual_start != start {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("HTTP range began at {actual_start}, requested {start}"),
            ));
        }
        if self.source_length != 0 && total != self.source_length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HTTP source length changed between range requests",
            ));
        }
        self.source_length = total;
        self.metrics.source_length.store(total, Ordering::Relaxed);

        let expected_length = actual_end_exclusive - actual_start;
        if expected_length > self.buffer_bytes as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HTTP range response exceeded bounded buffer",
            ));
        }
        self.buffer.clear();
        response
            .take(self.buffer_bytes as u64 + 1)
            .read_to_end(&mut self.buffer)?;
        if self.buffer.len() as u64 != expected_length {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "HTTP range returned {} bytes, expected {expected_length}",
                    self.buffer.len()
                ),
            ));
        }
        self.buffer_start = actual_start;
        self.metrics.requests.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .bytes_received
            .fetch_add(expected_length, Ordering::Relaxed);
        self.metrics.ranges.lock().unwrap().push(HttpRangeRecord {
            start: actual_start,
            end_exclusive: actual_end_exclusive,
        });
        Ok(())
    }

    fn buffer_offset(&self) -> Option<usize> {
        let offset = self.position.checked_sub(self.buffer_start)?;
        (offset < self.buffer.len() as u64).then_some(offset as usize)
    }
}

impl Read for HttpRangeSource {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() || self.position >= self.source_length {
            return Ok(0);
        }
        if self.buffer_offset().is_none() {
            self.refill(self.position)?;
        }
        let offset = self.buffer_offset().ok_or_else(|| {
            io::Error::new(io::ErrorKind::UnexpectedEof, "HTTP range buffer is empty")
        })?;
        let count = output
            .len()
            .min(self.buffer.len() - offset)
            .min((self.source_length - self.position) as usize);
        output[..count].copy_from_slice(&self.buffer[offset..offset + count]);
        self.position += count as u64;
        Ok(count)
    }
}

impl Seek for HttpRangeSource {
    fn seek(&mut self, target: SeekFrom) -> io::Result<u64> {
        let position = match target {
            SeekFrom::Start(position) => position as i128,
            SeekFrom::Current(delta) => self.position as i128 + delta as i128,
            SeekFrom::End(delta) => self.source_length as i128 + delta as i128,
        };
        if !(0..=u64::MAX as i128).contains(&position) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid HTTP source seek",
            ));
        }
        self.position = position as u64;
        Ok(self.position)
    }
}

impl MediaSource for HttpRangeSource {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.source_length)
    }
}

fn normalize_etag(value: &str) -> &str {
    value
        .trim()
        .strip_prefix("W/")
        .unwrap_or(value.trim())
        .trim_matches('"')
}

fn parse_content_range(value: &str) -> io::Result<(u64, u64, u64)> {
    let value = value
        .strip_prefix("bytes ")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid Content-Range unit"))?;
    let (range, total) = value
        .split_once('/')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid Content-Range"))?;
    let (start, end) = range.split_once('-').ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "invalid Content-Range bounds")
    })?;
    let start = start.parse::<u64>().map_err(io_other)?;
    let end = end.parse::<u64>().map_err(io_other)?;
    let total = total.parse::<u64>().map_err(io_other)?;
    let end_exclusive = end
        .checked_add(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "range end overflow"))?;
    if start >= end_exclusive || end_exclusive > total {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid Content-Range extent",
        ));
    }
    Ok((start, end_exclusive, total))
}

fn io_other(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    const BODY: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

    fn read_range(stream: &mut TcpStream) -> (u64, u64) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut line = String::new();
        let mut range = None;
        loop {
            line.clear();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.trim().split_once(':') {
                if name.eq_ignore_ascii_case("range") {
                    let value = value.trim().strip_prefix("bytes=").unwrap();
                    let (start, end) = value.split_once('-').unwrap();
                    range = Some((start.parse().unwrap(), end.parse().unwrap()));
                }
            }
        }
        range.unwrap()
    }

    fn range_server(request_count: usize, etag: &'static str) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            for _ in 0..request_count {
                let (mut stream, _) = listener.accept().unwrap();
                let (start, requested_end) = read_range(&mut stream);
                let end = requested_end.min(BODY.len() as u64 - 1);
                let part = &BODY[start as usize..=end as usize];
                write!(
                    stream,
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{}\r\nETag: \"{etag}\"\r\nConnection: close\r\n\r\n",
                    part.len(),
                    BODY.len()
                )
                .unwrap();
                stream.write_all(part).unwrap();
            }
        });
        (format!("http://{address}/input"), handle)
    }

    fn non_range_server() -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_range(&mut stream);
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                BODY.len()
            )
            .unwrap();
            stream.write_all(BODY).unwrap();
        });
        (format!("http://{address}/input"), handle)
    }

    #[test]
    fn range_source_reads_and_seeks_with_bounded_windows() {
        let (url, server) = range_server(3, "fixture");
        let (mut source, metrics) = HttpRangeSource::open(url, Some("fixture".into()), 8).unwrap();
        let mut first = [0; 4];
        source.read_exact(&mut first).unwrap();
        assert_eq!(&first, b"abcd");
        source.seek(SeekFrom::Start(16)).unwrap();
        let mut middle = [0; 3];
        source.read_exact(&mut middle).unwrap();
        assert_eq!(&middle, b"qrs");
        source.seek(SeekFrom::End(-2)).unwrap();
        let mut last = Vec::new();
        source.read_to_end(&mut last).unwrap();
        assert_eq!(&last, b"yz");
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.requests, 3);
        assert_eq!(snapshot.bytes_received, 18);
        assert_eq!(snapshot.ranges[0].end_exclusive, 8);
        server.join().unwrap();
    }

    #[test]
    fn range_source_rejects_an_etag_mismatch() {
        let (url, server) = range_server(1, "actual");
        let error = HttpRangeSource::open(url, Some("expected".into()), 8)
            .err()
            .unwrap();
        assert!(error.to_string().contains("ETag mismatch"));
        server.join().unwrap();
    }

    #[test]
    fn range_source_rejects_a_server_that_ignores_range() {
        let (url, server) = non_range_server();
        let error = HttpRangeSource::open(url, None, 8).err().unwrap();
        assert!(error.to_string().contains("expected 206"));
        server.join().unwrap();
    }

    #[test]
    fn content_range_parser_rejects_invalid_extents() {
        assert_eq!(parse_content_range("bytes 4-7/10").unwrap(), (4, 8, 10));
        assert!(parse_content_range("items 4-7/10").is_err());
        assert!(parse_content_range("bytes 7-4/10").is_err());
        assert!(parse_content_range("bytes 4-10/10").is_err());
    }
}
