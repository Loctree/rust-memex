//! Smart progress tracking for batch indexing operations.
//!
//! Provides a compact interactive viewport for TTY runs and a quieter
//! summary-oriented fallback for non-interactive stderr.
//!
//! This module is only available when the `cli` feature is enabled.

use crossterm::cursor::MoveUp;
use crossterm::execute;
use crossterm::terminal::{Clear, ClearType, size};
use std::collections::VecDeque;
use std::io::{IsTerminal, Write, stderr};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const VIEWPORT_LINES: usize = 16;
const ACTIVE_FILE_LINES: usize = 11;
const RENDER_THROTTLE: Duration = Duration::from_millis(80);
const SPINNER_FRAMES: [&str; 4] = ["|", "/", "-", "\\"];

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
    /// Measured chunks per second (EMA - updated dynamically)
    pub chunks_per_sec: Option<f64>,
    /// Name of the embedder model (for display)
    pub embedder_model: Option<String>,
    /// Whether calibration is complete
    calibration_done: bool,

    // Dynamic speed tracking (rolling average)
    /// Last time we updated the speed measurement
    last_speed_update: Option<Instant>,
    /// Chunks processed since last speed update
    chunks_since_update: usize,
    /// EMA smoothing factor (0.3 = 30% new, 70% old)
    speed_ema_alpha: f64,

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

    // Compact TTY viewport state
    active_files: VecDeque<String>,
    last_event: Option<String>,
    mode_label: Option<String>,
    dedup_enabled: bool,
    preprocess_enabled: bool,
    parallel_workers: Option<u8>,
    interactive: bool,
    rendered_lines: usize,
    spinner_frame: usize,
    last_render: Option<Instant>,
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
            last_speed_update: None,
            chunks_since_update: 0,
            speed_ema_alpha: 0.3, // 30% weight for new measurements
            processed_chunks: 0,
            processed_files: 0,
            skipped_files: 0,
            failed_files: 0,
            indexed_files: 0,
            active_files: VecDeque::new(),
            last_event: None,
            mode_label: None,
            dedup_enabled: false,
            preprocess_enabled: false,
            parallel_workers: None,
            interactive: stderr().is_terminal(),
            rendered_lines: 0,
            spinner_frame: 0,
            last_render: None,
        }
    }

    /// Attach run context that should appear in the compact viewport.
    pub fn set_run_context(
        &mut self,
        mode_label: &str,
        dedup_enabled: bool,
        preprocess_enabled: bool,
        parallel_workers: u8,
    ) {
        self.mode_label = Some(mode_label.to_string());
        self.dedup_enabled = dedup_enabled;
        self.preprocess_enabled = preprocess_enabled;
        self.parallel_workers = Some(parallel_workers);
        self.render(true);
    }

    /// Display Phase 1 pre-scan summary to stderr.
    pub fn display_pre_scan(&mut self) {
        if self.interactive {
            self.render(true);
            return;
        }

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
        self.render(true);
        if self.interactive {
            return;
        }
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
            // Initialize dynamic speed tracking
            self.last_speed_update = Some(Instant::now());
            self.chunks_since_update = 0;

            self.render(true);
            if self.interactive {
                return;
            }

            eprintln!(
                "  `-- Speed: {:.1} chunks/sec ({}) [dynamic]",
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
        self.render(true);
        if self.interactive {
            return;
        }
        eprintln!();
        eprintln!("Phase 3: Indexing...");
    }

    /// Increment the chunk counter and update progress bar.
    ///
    /// Also updates the rolling average speed (EMA) every 2 seconds
    /// to reflect actual embedding performance after GPU warm-up.
    pub fn inc_chunks(&mut self, count: usize) {
        self.processed_chunks += count;
        self.chunks_since_update += count;

        // Update speed every 2 seconds using EMA
        if let Some(last_update) = self.last_speed_update {
            let elapsed = last_update.elapsed().as_secs_f64();
            if elapsed >= 2.0 && self.chunks_since_update > 0 {
                let current_speed = self.chunks_since_update as f64 / elapsed;

                // Exponential Moving Average: new = alpha * current + (1-alpha) * old
                self.chunks_per_sec = Some(match self.chunks_per_sec {
                    Some(old_speed) => {
                        self.speed_ema_alpha * current_speed
                            + (1.0 - self.speed_ema_alpha) * old_speed
                    }
                    None => current_speed,
                });

                // Reset for next measurement window
                self.last_speed_update = Some(Instant::now());
                self.chunks_since_update = 0;
            }
        }
    }

    /// Record a successfully indexed file.
    ///
    /// # Arguments
    /// * `chunks` - Number of chunks created from this file
    pub fn file_indexed(&mut self, chunks: usize) {
        self.last_event = Some(format!("Indexed {} chunks", chunks));
        self.record_indexed(chunks, false);
    }

    /// Record a successfully indexed file and remove it from the active viewport.
    pub fn file_indexed_path(&mut self, path: &str, chunks: usize) {
        self.remove_active_file(path);
        self.last_event = Some(format!(
            "Indexed {} ({} chunks)",
            short_tail(path, 48),
            chunks
        ));
        self.record_indexed(chunks, false);
    }

    /// Record a skipped file (duplicate).
    pub fn file_skipped(&mut self) {
        self.last_event = Some("Skipped duplicate".to_string());
        self.record_skipped(false);
    }

    /// Record a skipped file and remove it from the active viewport.
    pub fn file_skipped_path(&mut self, path: &str, reason: &str) {
        self.remove_active_file(path);
        self.last_event = Some(format!(
            "Skipped {} ({})",
            short_tail(path, 48),
            truncate_end(reason, 28)
        ));
        self.record_skipped(false);
    }

    /// Record a failed file.
    pub fn file_failed(&mut self) {
        self.last_event = Some("Failed".to_string());
        self.record_failed(true);
    }

    /// Record a failed file and remove it from the active viewport.
    pub fn file_failed_path(&mut self, path: &str, error: &str) {
        self.remove_active_file(path);
        self.last_event = Some(format!(
            "FAILED {} ({})",
            short_tail(path, 48),
            truncate_end(error, 28)
        ));
        self.record_failed(true);
    }

    /// Mark a file as actively processing in the compact viewport.
    pub fn set_message(&mut self, msg: &str) {
        self.touch_active_file(msg);
        self.render(false);
    }

    /// Finish the progress bar with completion message.
    pub fn finish(&mut self) {
        self.render(true);
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
        }
    }

    fn touch_active_file(&mut self, path: &str) {
        if let Some(existing) = self.active_files.iter().position(|current| current == path) {
            let entry = self
                .active_files
                .remove(existing)
                .expect("active file exists");
            self.active_files.push_back(entry);
        } else {
            self.active_files.push_back(path.to_string());
        }
    }

    fn remove_active_file(&mut self, path: &str) {
        if let Some(existing) = self.active_files.iter().position(|current| current == path) {
            self.active_files.remove(existing);
        }
    }

    fn record_indexed(&mut self, chunks: usize, force_render: bool) {
        self.indexed_files += 1;
        self.processed_files += 1;
        self.inc_chunks(chunks);
        self.render(force_render);
    }

    fn record_skipped(&mut self, force_render: bool) {
        self.skipped_files += 1;
        self.processed_files += 1;
        self.render(force_render);
    }

    fn record_failed(&mut self, force_render: bool) {
        self.failed_files += 1;
        self.processed_files += 1;
        self.render(force_render);
    }

    fn render(&mut self, force: bool) {
        if !self.interactive {
            return;
        }

        let now = Instant::now();
        if !force
            && self
                .last_render
                .is_some_and(|last| now.duration_since(last) < RENDER_THROTTLE)
        {
            return;
        }

        self.last_render = Some(now);
        let width = size().map(|(cols, _)| cols as usize).unwrap_or(100).max(60);
        let lines = self.render_lines(width);
        let mut err = stderr();

        if self.rendered_lines > 0 {
            let _ = execute!(err, MoveUp(self.rendered_lines as u16));
        }

        for _ in 0..self.rendered_lines {
            let _ = execute!(err, Clear(ClearType::CurrentLine));
            let _ = writeln!(err);
        }

        if self.rendered_lines > 0 {
            let _ = execute!(err, MoveUp(self.rendered_lines as u16));
        }

        for line in &lines {
            let _ = execute!(err, Clear(ClearType::CurrentLine));
            let _ = writeln!(err, "{line}");
        }

        let _ = err.flush();
        self.rendered_lines = lines.len();
    }

    fn render_lines(&mut self, width: usize) -> Vec<String> {
        let progress_ratio = if self.total_files == 0 {
            0.0
        } else {
            self.processed_files as f64 / self.total_files as f64
        };
        let progress_pct = progress_ratio * 100.0;
        let spinner = SPINNER_FRAMES[self.spinner_frame % SPINNER_FRAMES.len()];
        self.spinner_frame = self.spinner_frame.wrapping_add(1);

        let bar_width = width.saturating_sub(34).clamp(12, 40);
        let filled = (progress_ratio * bar_width as f64).round() as usize;
        let bar = make_progress_bar(bar_width, filled.min(bar_width));

        let phase = if self.processed_files >= self.total_files && self.total_files > 0 {
            "complete"
        } else if self.calibration_done {
            "indexing"
        } else if self.calibration_start.is_some() {
            "calibrating"
        } else {
            "preparing"
        };

        let line1 = truncate_end(
            &format!(
                "{} [{}] {}/{} files ({:.1}%) {}",
                spinner, bar, self.processed_files, self.total_files, progress_pct, phase
            ),
            width,
        );

        let line2 = truncate_end(
            &format!(
                "Stats: indexed {} | skipped {} | failed {} | active {} | workers {}",
                self.indexed_files,
                self.skipped_files,
                self.failed_files,
                self.active_files.len(),
                self.parallel_workers.unwrap_or(0)
            ),
            width,
        );

        let line3 = truncate_end(
            &format!(
                "Chunks: {}/~{} | speed {} | ETA {} | mode {} | dedup {}",
                self.processed_chunks,
                self.estimated_chunks.max(self.processed_chunks),
                self.chunks_per_sec
                    .map(|speed| format!("{speed:.1}/s"))
                    .unwrap_or_else(|| "calibrating".to_string()),
                self.eta_label(),
                self.mode_label.as_deref().unwrap_or("n/a"),
                if self.dedup_enabled { "on" } else { "off" }
            ),
            width,
        );

        let line4 = truncate_end(
            &format!(
                "Model: {}{}{}",
                self.embedder_model.as_deref().unwrap_or("warming"),
                if self.preprocess_enabled {
                    " | preprocess on"
                } else {
                    ""
                },
                self.last_event
                    .as_ref()
                    .map(|event| format!(" | last: {event}"))
                    .unwrap_or_default()
            ),
            width,
        );

        let visible_active: Vec<String> = self
            .active_files
            .iter()
            .rev()
            .take(ACTIVE_FILE_LINES)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let mut lines = vec![line1, line2, line3, line4, String::new()];
        for path in visible_active {
            lines.push(short_tail(&format!("  * {path}"), width));
        }
        while lines.len() < VIEWPORT_LINES {
            lines.push(String::new());
        }
        lines.truncate(VIEWPORT_LINES);
        lines
    }

    fn eta_label(&self) -> String {
        let Some(speed) = self.chunks_per_sec else {
            return "--".to_string();
        };
        if speed <= 0.0 {
            return "--".to_string();
        }
        let remaining_chunks = self.estimated_chunks.saturating_sub(self.processed_chunks);
        let remaining_seconds = (remaining_chunks as f64 / speed).round() as u64;
        format_duration(remaining_seconds)
    }
}

fn make_progress_bar(width: usize, filled: usize) -> String {
    if width == 0 {
        return String::new();
    }
    if filled >= width {
        return "=".repeat(width);
    }

    let complete = "=".repeat(filled);
    let head = ">";
    let remaining = "-".repeat(width.saturating_sub(filled + 1));
    format!("{complete}{head}{remaining}")
}

fn truncate_end(value: &str, max_chars: usize) -> String {
    let len = value.chars().count();
    if len <= max_chars {
        return value.to_string();
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let kept: String = value.chars().take(max_chars - 3).collect();
    format!("{kept}...")
}

fn short_tail(value: &str, max_chars: usize) -> String {
    let len = value.chars().count();
    if len <= max_chars {
        return value.to_string();
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let tail: String = value
        .chars()
        .rev()
        .take(max_chars - 3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("...{tail}")
}

fn format_duration(seconds: u64) -> String {
    match seconds {
        0..=59 => format!("{seconds}s"),
        60..=3599 => format!("{}m {:02}s", seconds / 60, seconds % 60),
        _ => format!("{}h {:02}m", seconds / 3600, (seconds % 3600) / 60),
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

    #[test]
    fn test_render_lines_use_fixed_viewport_and_latest_active_files() {
        let tracker_paths: Vec<PathBuf> = vec![];
        let mut tracker = IndexProgressTracker::pre_scan(&tracker_paths);
        tracker.total_files = 42;
        tracker.mode_label = Some("onion".to_string());
        tracker.dedup_enabled = true;
        tracker.parallel_workers = Some(8);

        for index in 0..13 {
            tracker.set_message(&format!("workspace/project/deep/path/file-{index}.md"));
        }

        let lines = tracker.render_lines(96);

        assert_eq!(lines.len(), VIEWPORT_LINES);
        assert!(lines[0].contains("files"));
        assert!(lines[1].contains("active"));
        assert_eq!(lines[4], "");
        assert!(!lines.iter().any(|line| line.contains("file-0.md")));
        assert!(!lines.iter().any(|line| line.contains("file-1.md")));
        assert!(lines.iter().any(|line| line.contains("file-2.md")));
        assert!(lines.iter().any(|line| line.contains("file-12.md")));
    }

    #[test]
    fn test_completed_file_leaves_active_viewport_and_records_last_event() {
        let tracker_paths: Vec<PathBuf> = vec![];
        let mut tracker = IndexProgressTracker::pre_scan(&tracker_paths);
        tracker.total_files = 2;
        tracker.mode_label = Some("flat".to_string());
        tracker.preprocess_enabled = true;
        tracker.parallel_workers = Some(2);
        tracker.set_message("repo/alpha.md");
        tracker.set_message("repo/beta.md");

        tracker.file_failed_path("repo/alpha.md", "invalid utf-8");

        let lines = tracker.render_lines(100);
        let active_lines = &lines[5..];

        assert!(!active_lines.iter().any(|line| line.contains("alpha.md")));
        assert!(active_lines.iter().any(|line| line.contains("beta.md")));
        assert!(lines[3].contains("FAILED"));
        assert_eq!(tracker.failed_files, 1);
    }
}
