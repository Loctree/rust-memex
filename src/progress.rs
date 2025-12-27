//! Smart progress tracking for batch indexing operations.
//!
//! Provides three-phase progress display:
//! 1. Pre-scan: Count files and estimate chunks
//! 2. Calibration: Measure embedding speed on first file
//! 3. Indexing: Progress bar with ETA based on calibration

use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;

/// Format bytes into human-readable string
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Progress tracker for batch indexing operations.
///
/// Implements a three-phase progress display:
/// 1. Pre-scan phase: counts files and estimates total chunks
/// 2. Calibration phase: measures embedding speed on first file
/// 3. Indexing phase: displays progress bar with ETA
pub struct IndexProgressTracker {
    // Pre-scan phase
    /// Total number of files to process
    pub total_files: usize,
    /// Total size of all files in bytes
    pub total_bytes: u64,
    /// Average chunk size in characters (used for estimation)
    pub chunk_size: usize,
    /// Estimated total chunks based on file sizes
    pub estimated_chunks: usize,

    // Calibration phase
    /// When calibration started
    calibration_start: Option<Instant>,
    /// Measured chunks per second after calibration
    pub chunks_per_sec: Option<f64>,
    /// Name of the embedder model (for display)
    pub embedder_model: Option<String>,
    /// Whether calibration is complete
    calibration_done: bool,

    // Progress tracking
    /// Number of chunks processed so far
    pub processed_chunks: usize,
    /// Number of files processed (indexed + skipped + failed)
    pub processed_files: usize,
    /// Number of files skipped (duplicates)
    pub skipped_files: usize,
    /// Number of files that failed to index
    pub failed_files: usize,
    /// Number of files successfully indexed
    pub indexed_files: usize,

    // Progress bar
    progress_bar: Option<ProgressBar>,
}

impl IndexProgressTracker {
    /// Create a new progress tracker by pre-scanning the given paths.
    ///
    /// This performs Phase 1: Pre-scan to count files and estimate chunks.
    pub fn pre_scan(paths: &[PathBuf]) -> Self {
        let total_bytes: u64 = paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();

        // Rough estimate: 1 chunk per 500 chars, avg 1 byte = 1 char for text files
        // This is conservative - actual chunk count depends on slice mode
        let chunk_size = 500;
        let estimated_chunks = (total_bytes as usize) / chunk_size;

        Self {
            total_files: paths.len(),
            total_bytes,
            chunk_size,
            estimated_chunks,
            calibration_start: None,
            chunks_per_sec: None,
            embedder_model: None,
            calibration_done: false,
            processed_chunks: 0,
            processed_files: 0,
            skipped_files: 0,
            failed_files: 0,
            indexed_files: 0,
            progress_bar: None,
        }
    }

    /// Display Phase 1 pre-scan summary to stderr.
    pub fn display_pre_scan(&self) {
        eprintln!();
        eprintln!("Phase 1: Pre-scan");
        eprintln!("  |-- Files: {}", self.total_files);
        eprintln!("  |-- Total size: {}", format_bytes(self.total_bytes));
        eprintln!(
            "  `-- Est. chunks: ~{} (@ {} chars/chunk)",
            self.estimated_chunks, self.chunk_size
        );
    }

    /// Start the calibration phase (Phase 2).
    ///
    /// Call this before processing the first file.
    pub fn start_calibration(&mut self) {
        self.calibration_start = Some(Instant::now());
        eprintln!();
        eprintln!("Phase 2: Calibration (first file)...");
    }

    /// Finish calibration with results from first file.
    ///
    /// # Arguments
    /// * `chunks_processed` - Number of chunks indexed in first file
    /// * `model` - Name of the embedder model used
    pub fn finish_calibration(&mut self, chunks_processed: usize, model: &str) {
        if let Some(start) = self.calibration_start {
            let elapsed = start.elapsed();
            if elapsed.as_secs_f64() > 0.0 && chunks_processed > 0 {
                self.chunks_per_sec = Some(chunks_processed as f64 / elapsed.as_secs_f64());
            }
            self.embedder_model = Some(model.to_string());
            self.calibration_done = true;

            eprintln!(
                "  `-- Speed: {:.1} chunks/sec ({})",
                self.chunks_per_sec.unwrap_or(0.0),
                model
            );
        }
    }

    /// Check if calibration has been completed.
    pub fn is_calibrated(&self) -> bool {
        self.calibration_done
    }

    /// Start the progress bar for Phase 3: Indexing.
    ///
    /// Creates a progress bar based on estimated chunks.
    pub fn start_progress_bar(&mut self) {
        let pb = ProgressBar::new(self.estimated_chunks as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} chunks | ETA: {eta} | {msg}",
                )
                .expect("Invalid progress bar template")
                .progress_chars("#>-"),
        );

        eprintln!();
        eprintln!("Phase 3: Indexing...");
        self.progress_bar = Some(pb);
    }

    /// Increment the chunk counter and update progress bar.
    pub fn inc_chunks(&mut self, count: usize) {
        self.processed_chunks += count;
        if let Some(ref pb) = self.progress_bar {
            pb.set_position(self.processed_chunks as u64);
        }
    }

    /// Record a successfully indexed file.
    ///
    /// # Arguments
    /// * `chunks` - Number of chunks created from this file
    pub fn file_indexed(&mut self, chunks: usize) {
        self.indexed_files += 1;
        self.processed_files += 1;
        self.inc_chunks(chunks);
    }

    /// Record a skipped file (duplicate).
    pub fn file_skipped(&mut self) {
        self.skipped_files += 1;
        self.processed_files += 1;
    }

    /// Record a failed file.
    pub fn file_failed(&mut self) {
        self.failed_files += 1;
        self.processed_files += 1;
    }

    /// Set the current message on the progress bar.
    pub fn set_message(&mut self, msg: &str) {
        if let Some(ref pb) = self.progress_bar {
            pb.set_message(msg.to_string());
        }
    }

    /// Finish the progress bar with completion message.
    pub fn finish(&mut self) {
        if let Some(ref pb) = self.progress_bar {
            pb.finish_with_message("Complete");
        }
    }

    /// Display final summary after indexing is complete.
    pub fn display_summary(&self) {
        eprintln!();
        eprintln!("Indexing complete:");
        eprintln!("  |-- Chunks indexed: {}", self.processed_chunks);
        eprintln!("  |-- Files processed: {}", self.processed_files);
        eprintln!("  |   |-- Indexed: {}", self.indexed_files);
        if self.skipped_files > 0 {
            eprintln!("  |   |-- Skipped (duplicate): {}", self.skipped_files);
        }
        if self.failed_files > 0 {
            eprintln!("  |   `-- Failed: {}", self.failed_files);
        }
        if let Some(speed) = self.chunks_per_sec {
            eprintln!("  `-- Avg speed: {:.1} chunks/sec", speed);
        }
    }

    /// Adjust estimated chunks based on actual chunk count from calibration file.
    ///
    /// This improves ETA accuracy after calibration by using the actual
    /// bytes-to-chunks ratio observed in the first file.
    pub fn adjust_estimate(&mut self, file_bytes: u64, actual_chunks: usize) {
        if file_bytes > 0 && actual_chunks > 0 {
            // Calculate actual bytes per chunk ratio
            let bytes_per_chunk = file_bytes as f64 / actual_chunks as f64;
            // Remaining bytes to process
            let remaining_bytes = self.total_bytes.saturating_sub(file_bytes);
            // Estimate remaining chunks based on actual ratio
            let remaining_chunks = (remaining_bytes as f64 / bytes_per_chunk) as usize;
            // Update total estimate
            self.estimated_chunks = actual_chunks + remaining_chunks;

            // Update progress bar length if active
            if let Some(ref pb) = self.progress_bar {
                pb.set_length(self.estimated_chunks as u64);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_pre_scan_empty() {
        let tracker = IndexProgressTracker::pre_scan(&[]);
        assert_eq!(tracker.total_files, 0);
        assert_eq!(tracker.total_bytes, 0);
        assert_eq!(tracker.estimated_chunks, 0);
    }

    #[test]
    fn test_pre_scan_with_files() {
        let temp = TempDir::new().unwrap();
        let file1 = temp.path().join("file1.txt");
        let file2 = temp.path().join("file2.txt");

        // Create test files with known sizes
        let mut f1 = std::fs::File::create(&file1).unwrap();
        f1.write_all(&[b'a'; 1000]).unwrap();

        let mut f2 = std::fs::File::create(&file2).unwrap();
        f2.write_all(&[b'b'; 500]).unwrap();

        let paths = vec![file1, file2];
        let tracker = IndexProgressTracker::pre_scan(&paths);

        assert_eq!(tracker.total_files, 2);
        assert_eq!(tracker.total_bytes, 1500);
        // 1500 bytes / 500 chars per chunk = 3 estimated chunks
        assert_eq!(tracker.estimated_chunks, 3);
    }

    #[test]
    fn test_file_tracking() {
        let tracker_paths: Vec<PathBuf> = vec![];
        let mut tracker = IndexProgressTracker::pre_scan(&tracker_paths);

        // Simulate indexing some files
        tracker.file_indexed(10);
        tracker.file_indexed(5);
        tracker.file_skipped();
        tracker.file_failed();

        assert_eq!(tracker.processed_files, 4);
        assert_eq!(tracker.indexed_files, 2);
        assert_eq!(tracker.skipped_files, 1);
        assert_eq!(tracker.failed_files, 1);
        assert_eq!(tracker.processed_chunks, 15);
    }

    #[test]
    fn test_calibration_flow() {
        let tracker_paths: Vec<PathBuf> = vec![];
        let mut tracker = IndexProgressTracker::pre_scan(&tracker_paths);

        assert!(!tracker.is_calibrated());

        tracker.start_calibration();
        // Simulate some processing time
        std::thread::sleep(std::time::Duration::from_millis(10));
        tracker.finish_calibration(100, "test-model");

        assert!(tracker.is_calibrated());
        assert!(tracker.chunks_per_sec.is_some());
        assert_eq!(tracker.embedder_model, Some("test-model".to_string()));
    }

    #[test]
    fn test_adjust_estimate() {
        let tracker_paths: Vec<PathBuf> = vec![];
        let mut tracker = IndexProgressTracker::pre_scan(&tracker_paths);
        tracker.total_bytes = 10000;
        tracker.estimated_chunks = 20;

        // First file: 1000 bytes produced 5 chunks
        // Ratio: 200 bytes per chunk
        // Remaining: 9000 bytes -> 45 chunks
        // Total: 5 + 45 = 50 chunks
        tracker.adjust_estimate(1000, 5);

        assert_eq!(tracker.estimated_chunks, 50);
    }
}
