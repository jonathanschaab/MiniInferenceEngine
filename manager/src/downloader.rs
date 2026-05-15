use crate::AppState;
use manager::lock_status;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tracing::{error, info, warn};

pub struct DownloadCleanupGuard {
    state: Arc<AppState>,
    id: String,
}

impl Drop for DownloadCleanupGuard {
    fn drop(&mut self) {
        self.state
            .active_downloads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.id);
        self.state
            .download_tasks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.id);
    }
}

pub async fn perform_model_download(
    state: Arc<AppState>,
    id: String,
    repo: String,
    filename: String,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) {
    let _guard = DownloadCleanupGuard {
        state: state.clone(),
        id: id.clone(),
    };

    let url = format!(
        "{}/{}/resolve/main/{}",
        state.config.hf_base_url.trim_end_matches('/'),
        repo,
        filename
    );

    let downloads_dir = &state.config.downloads_directory;
    if let Err(e) = tokio::fs::create_dir_all(downloads_dir).await {
        error!("Failed to create downloads directory for {}: {}", id, e);
        return;
    }
    let file_path = std::path::Path::new(downloads_dir).join(&filename);
    let tmp_file_path = {
        let mut p = file_path.clone().into_os_string();
        p.push(".tmp");
        std::path::PathBuf::from(p)
    };
    let meta_file_path = {
        let mut p = file_path.clone().into_os_string();
        p.push(".meta");
        std::path::PathBuf::from(p)
    };

    let (mut existing_size, mut expected_hash) =
        check_existing_metadata(&state, &id, &tmp_file_path, &meta_file_path).await;

    // Wait in the queue for an available download slot
    let _permit = match state.download_semaphore.acquire().await {
        Ok(p) => p,
        Err(_) => return, // Semaphore closed, safely shutting down
    };

    let mut hasher = Sha256::new();
    if existing_size > 0 {
        {
            let mut dl = state
                .active_downloads
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(status) = dl.get_mut(&id) {
                status.state = "Verifying existing data...".to_string();
            }
        }

        hasher = restore_hasher_state(&tmp_file_path, existing_size).await;
    }

    update_status_connecting(&state, &id, existing_size);

    let mut res = match initiate_request(&state, &url, &id, &mut existing_size, &mut hasher).await {
        Some(r) => r,
        None => return,
    };

    if verify_and_update_etag(
        &state,
        &url,
        &id,
        &mut res,
        &mut existing_size,
        &mut expected_hash,
        &mut hasher,
    )
    .await
    .is_err()
    {
        return;
    }

    let is_partial = res.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    if !is_partial && existing_size > 0 {
        info!(
            "Server did not return partial content, restarting download for {}",
            id
        );
        existing_size = 0;
        hasher = Sha256::new();
    }

    let total_size = if is_partial {
        existing_size + res.content_length().unwrap_or(0)
    } else {
        res.content_length().unwrap_or(0)
    };

    save_metadata(&meta_file_path, existing_size, &expected_hash).await;
    update_status_downloading(&state, &id, existing_size, total_size);

    let mut file = match open_temp_file(&tmp_file_path, is_partial, existing_size).await {
        Ok(f) => f,
        Err(e) => {
            error!("Failed to open/create file for {}: {}", id, e);
            return;
        }
    };

    let (stream_error, downloaded) = process_download_stream(
        &state,
        &id,
        &mut res,
        &mut file,
        &mut hasher,
        existing_size,
        &meta_file_path,
        &expected_hash,
        &mut shutdown_rx,
    )
    .await;

    if stream_error {
        save_metadata(&meta_file_path, downloaded, &expected_hash).await;
        info!(
            "Network interrupted. Kept partial temp file for {} to resume later.",
            id
        );
        return;
    }

    drop(file);

    let final_hash = hex::encode(hasher.finalize());

    let hash_mismatch = if let Some(ref expected) = expected_hash {
        if expected.len() == 64 && expected.chars().all(|c| c.is_ascii_hexdigit()) {
            if &final_hash != expected {
                error!(
                    "Checksum mismatch for {}. Expected {}, got {}. Marking as corrupted.",
                    id, expected, final_hash
                );
                true
            } else {
                info!("Checksum validated successfully for {}.", id);
                false
            }
        } else {
            info!(
                "No SHA-256 ETag found for {}. Saving. Computed: {}",
                id, final_hash
            );
            false
        }
    } else {
        info!(
            "No SHA-256 ETag found for {}. Saving. Computed: {}",
            id, final_hash
        );
        false
    };

    finalize_download(
        &state,
        &id,
        &filename,
        &file_path,
        &tmp_file_path,
        &meta_file_path,
        hash_mismatch,
    )
    .await;
}

async fn restore_hasher_state(tmp_file_path: &Path, existing_size: u64) -> Sha256 {
    let path = tmp_file_path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let mut h = Sha256::new();
        if let Ok(mut f) = std::fs::File::open(&path) {
            use std::io::Read;
            let mut buf = vec![0u8; 4 * 1024 * 1024];
            let mut remaining = existing_size;
            while remaining > 0 {
                let to_read = (buf.len() as u64).min(remaining) as usize;
                if let Ok(n) = f.read(&mut buf[..to_read]) {
                    if n == 0 {
                        break;
                    }
                    h.update(&buf[..n]);
                    remaining -= n as u64;
                } else {
                    break;
                }
            }
        }
        h
    })
    .await
    .unwrap_or_else(|_| Sha256::new())
}

async fn check_existing_metadata(
    state: &Arc<AppState>,
    id: &str,
    tmp_file_path: &Path,
    meta_file_path: &Path,
) -> (u64, Option<String>) {
    let mut existing_size = 0;
    let mut expected_hash = None;

    if let Ok(meta_str) = tokio::fs::read_to_string(meta_file_path).await {
        if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta_str)
            && let Some(bytes) = meta.get("downloaded_bytes").and_then(|v| v.as_u64())
        {
            existing_size = bytes;
            if let Some(hash) = meta.get("expected_hash").and_then(|v| v.as_str()) {
                expected_hash = Some(hash.to_string());
            }
        } else {
            warn!(
                "Corrupted or invalid metadata file found for {}. Discarding and restarting download.",
                id
            );
            {
                let mut dl = state
                    .active_downloads
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if let Some(status) = dl.get_mut(id) {
                    status.state = "Corrupted metadata. Restarting...".to_string();
                }
            }
            let _ = tokio::fs::remove_file(meta_file_path).await;
            let _ = tokio::fs::remove_file(tmp_file_path).await;
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }
    }

    if existing_size > 0 {
        if let Ok(file) = tokio::fs::OpenOptions::new()
            .write(true)
            .open(tmp_file_path)
            .await
            && let Ok(metadata) = file.metadata().await
        {
            let actual_size = metadata.len();
            if actual_size < existing_size {
                existing_size = actual_size;
            } else if actual_size > existing_size {
                let _ = file.set_len(existing_size).await;
            }
        } else {
            existing_size = 0;
        }
    }

    (existing_size, expected_hash)
}

fn update_status_connecting(state: &Arc<AppState>, id: &str, existing_size: u64) {
    let mut dl = state
        .active_downloads
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(status) = dl.get_mut(id) {
        status.bytes_transferred = existing_size;
        status.state = "Connecting...".to_string();
    }
}

async fn initiate_request(
    state: &Arc<AppState>,
    url: &str,
    id: &str,
    existing_size: &mut u64,
    hasher: &mut Sha256,
) -> Option<reqwest::Response> {
    let client = &state.reqwest_client;
    let mut req = client.get(url);
    if *existing_size > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={}-", existing_size));
    }

    match req.send().await {
        Ok(r) => {
            if r.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                *existing_size = 0;
                *hasher = Sha256::new();
                match client.get(url).send().await {
                    Ok(r2) => {
                        if !r2.status().is_success() {
                            error!("Download failed for {} with status: {}", id, r2.status());
                            None
                        } else {
                            Some(r2)
                        }
                    }
                    Err(e) => {
                        error!("Download failed for {}: {}", id, e);
                        None
                    }
                }
            } else if !r.status().is_success() {
                error!("Download failed for {} with status: {}", id, r.status());
                None
            } else {
                Some(r)
            }
        }
        Err(e) => {
            error!("Download failed for {}: {}", id, e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_restore_hasher_state() {
        let test_dir = std::env::temp_dir().join("minio_test_hasher");
        let _ = tokio::fs::create_dir_all(&test_dir).await;
        let test_file = test_dir.join("test_file.bin");

        // Write a "partially downloaded" file
        let part1: &[u8] = b"Hello, ";
        let part2: &[u8] = b"World!";
        tokio::fs::write(&test_file, [part1, part2].concat())
            .await
            .unwrap();

        // 1. Hash the entire combined data directly to get the expected baseline
        let mut expected_hasher = Sha256::new();
        expected_hasher.update([part1, part2].concat());
        let expected_hash = hex::encode(expected_hasher.finalize());

        // 2. Restore the hasher state from the disk up to the size of part1
        let mut resumed_hasher = restore_hasher_state(&test_file, part1.len() as u64).await;
        // 3. Manually push in the incoming stream chunk (part2)
        resumed_hasher.update(part2);
        let resumed_hash = hex::encode(resumed_hasher.finalize());

        assert_eq!(
            expected_hash, resumed_hash,
            "Resumed hash should match the full continuous hash"
        );
        let _ = tokio::fs::remove_dir_all(&test_dir).await;
    }
}

async fn verify_and_update_etag(
    state: &Arc<AppState>,
    url: &str,
    id: &str,
    res: &mut reqwest::Response,
    existing_size: &mut u64,
    expected_hash: &mut Option<String>,
    hasher: &mut Sha256,
) -> Result<(), ()> {
    let mut new_etag = None;
    if let Some(etag) = res
        .headers()
        .get("X-Linked-Etag")
        .or_else(|| res.headers().get("ETag"))
    {
        let etag_str = etag.to_str().unwrap_or("").trim_matches('"');
        let clean_etag = etag_str
            .strip_prefix("W/")
            .unwrap_or(etag_str)
            .trim_matches('"');
        if !clean_etag.is_empty() {
            new_etag = Some(clean_etag.to_string());
        }
    }

    if *existing_size > 0
        && expected_hash.is_some()
        && new_etag.is_some()
        && *expected_hash != new_etag
    {
        warn!(
            "ETag mismatch for {} (expected {:?}, got {:?}). Restarting download.",
            id, expected_hash, new_etag
        );
        *existing_size = 0;
        *hasher = Sha256::new();
        *expected_hash = new_etag.clone();

        if let Ok(r) = state.reqwest_client.get(url).send().await {
            if r.status().is_success() {
                *res = r;
            } else {
                error!(
                    "Download failed for {} on restart with status: {}",
                    id,
                    r.status()
                );
                return Err(());
            }
        } else {
            error!("Download failed for {} on restart", id);
            return Err(());
        }
    } else if new_etag.is_some() {
        *expected_hash = new_etag;
    }
    Ok(())
}

async fn save_metadata(meta_file_path: &Path, downloaded: u64, expected_hash: &Option<String>) {
    let _ = tokio::fs::write(
        meta_file_path,
        serde_json::json!({
            "downloaded_bytes": downloaded,
            "expected_hash": expected_hash
        })
        .to_string(),
    )
    .await;
}

fn update_status_downloading(state: &Arc<AppState>, id: &str, existing_size: u64, total_size: u64) {
    let mut dl = state
        .active_downloads
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(status) = dl.get_mut(id) {
        status.total_bytes = total_size;
        status.bytes_transferred = existing_size;
        status.state = "Downloading...".to_string();
    }
}

async fn open_temp_file(
    tmp_file_path: &Path,
    is_partial: bool,
    existing_size: u64,
) -> std::io::Result<tokio::fs::File> {
    if is_partial && existing_size > 0 {
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(tmp_file_path)
            .await
    } else {
        tokio::fs::File::create(tmp_file_path).await
    }
}

#[allow(clippy::too_many_arguments)]
async fn process_download_stream(
    state: &Arc<AppState>,
    id: &str,
    res: &mut reqwest::Response,
    file: &mut tokio::fs::File,
    hasher: &mut Sha256,
    existing_size: u64,
    meta_file_path: &Path,
    expected_hash: &Option<String>,
    shutdown_rx: &mut tokio::sync::broadcast::Receiver<()>,
) -> (bool, u64) {
    let start_time = std::time::Instant::now();
    let mut downloaded: u64 = existing_size;
    let mut stream_error = false;
    let mut last_meta_save = std::time::Instant::now();
    let mut last_ui_update = std::time::Instant::now();
    let mut bytes_since_last_ui_update = 0;

    let update_interval_ms = 500;
    let update_bytes_threshold = 1024 * 1024;

    loop {
        tokio::select! {
            chunk_res = res.chunk() => {
                let bytes = match chunk_res {
                    Ok(Some(c)) => c,
                    Ok(None) => break,
                    Err(e) => {
                        error!("Error while streaming {}: {}", id, e);
                        stream_error = true;
                        break;
                    }
                };

                hasher.update(&bytes);
                if let Err(e) = file.write_all(&bytes).await {
                    error!("Failed to write to file for {}: {}", id, e);
                    stream_error = true;
                    break;
                }
                downloaded += bytes.len() as u64;
                bytes_since_last_ui_update += bytes.len() as u64;

                if last_ui_update.elapsed().as_millis() >= update_interval_ms
                    || bytes_since_last_ui_update >= update_bytes_threshold
                {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let speed = if elapsed > 0.0 {
                        (downloaded.saturating_sub(existing_size)) as f64 / elapsed
                    } else {
                        0.0
                    };

                    {
                        let mut dl = state.active_downloads.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(status) = dl.get_mut(id) {
                            status.bytes_transferred = downloaded;
                            status.current_speed_bps = speed;
                            status.state = "Downloading...".to_string();
                        }
                    }

                    last_ui_update = std::time::Instant::now();
                    bytes_since_last_ui_update = 0;
                }

                if last_meta_save.elapsed().as_secs() > 5 {
                    save_metadata(meta_file_path, downloaded, expected_hash).await;
                    last_meta_save = std::time::Instant::now();
                }
            }
            _ = shutdown_rx.recv() => {
                info!("Shutdown signal received. Flushing metadata for {}...", id);
                save_metadata(meta_file_path, downloaded, expected_hash).await;
                return (true, downloaded);
            }
        }
    }

    (stream_error, downloaded)
}

async fn finalize_download(
    state: &Arc<AppState>,
    id: &str,
    filename: &str,
    file_path: &Path,
    tmp_file_path: &Path,
    meta_file_path: &Path,
    hash_mismatch: bool,
) {
    let target_path = if hash_mismatch {
        let mut p = file_path.to_path_buf().into_os_string();
        p.push(".corrupted");
        PathBuf::from(p)
    } else {
        file_path.to_path_buf()
    };

    let mut success = false;
    if let Err(e) = tokio::fs::rename(tmp_file_path, &target_path).await {
        warn!(
            "Failed to rename temp file for {} ({}). Falling back to copy...",
            id, e
        );

        let mut copy_tmp_path = target_path.to_path_buf().into_os_string();
        copy_tmp_path.push(".copy_tmp");
        let copy_tmp_path = PathBuf::from(copy_tmp_path);

        match tokio::fs::copy(tmp_file_path, &copy_tmp_path).await {
            Ok(_) => {
                // Ensure data is physically on the destination disk before the atomic rename
                if let Ok(file) = tokio::fs::OpenOptions::new()
                    .write(true)
                    .open(&copy_tmp_path)
                    .await
                {
                    let _ = file.sync_all().await;
                }

                if let Err(rename_err) = tokio::fs::rename(&copy_tmp_path, &target_path).await {
                    error!(
                        "Failed to rename copied temp file for {}: {}",
                        id, rename_err
                    );
                    let _ = tokio::fs::remove_file(&copy_tmp_path).await;
                } else {
                    let _ = tokio::fs::remove_file(tmp_file_path).await;
                    success = true;
                }
            }
            Err(copy_err) => {
                error!(
                    "Failed to copy temp file to final file for {}: {}",
                    id, copy_err
                );
                let _ = tokio::fs::remove_file(&copy_tmp_path).await;
            }
        }
    } else {
        success = true;
    }

    if success {
        let _ = tokio::fs::remove_file(meta_file_path).await;
        if hash_mismatch {
            error!(
                "Download finished but checksum failed for {}. Saved as {}. Please verify the file and rename it manually to {} if it is valid.",
                id,
                target_path.display(),
                filename
            );
            {
                let mut status = lock_status(&state.engine_status);
                status.corrupted_models.insert(id.to_string());
            }
        } else {
            info!("Finished downloading {}", id);
            {
                let mut status = lock_status(&state.engine_status);
                status.downloaded_models.insert(id.to_string());
            }
        }
    }
}
