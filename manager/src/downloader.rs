use crate::AppState;
use crypto_common::hazmat::SerializableState;
use manager::{lock_mutex, lock_status};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tracing::{error, info, warn};

const UPDATE_INTERVAL_MS: u128 = 500;
const UPDATE_BYTES_THRESHOLD: u64 = 1024 * 1024;
const META_SAVE_INTERVAL_SECS: u64 = 5;
const CORRUPT_RESTART_DELAY_SECS: u64 = 3;
const HASHER_STATE_VERSION: u32 = 1;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct Checkpoint {
    pub downloaded_bytes: u64,
    pub hasher_state: String,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct DownloadMetadata {
    pub expected_hash: Option<String>,
    pub is_sha256: bool,
    #[serde(default)]
    pub checkpoints: Vec<Checkpoint>,
    #[serde(default)]
    pub hasher_version: u32,
}

// Safely extract the internal state of the Sha256 struct into a hex string
fn serialize_hasher(hasher: &Sha256) -> String {
    let state = hasher.serialize();
    hex::encode(state.as_slice())
}

// Safely reconstruct the Sha256 struct from a hex string
fn deserialize_hasher(hex_str: &str) -> Option<Sha256> {
    let decoded = hex::decode(hex_str).ok()?;
    let state: crypto_common::hazmat::SerializedState<Sha256> =
        decoded.as_slice().try_into().ok()?;
    Sha256::deserialize(&state).ok()
}

struct ActiveStreamGuard {
    counter: Arc<std::sync::atomic::AtomicUsize>,
}
impl ActiveStreamGuard {
    fn new(counter: Arc<std::sync::atomic::AtomicUsize>) -> Self {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self { counter }
    }
}
impl Drop for ActiveStreamGuard {
    fn drop(&mut self) {
        self.counter
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

async fn wait_for_backoff(
    backoff: u64,
    shutdown_rx: &mut tokio::sync::broadcast::Receiver<()>,
    cancel_rx: &mut tokio::sync::broadcast::Receiver<()>,
) -> Option<&'static str> {
    tokio::select! {
        _ = tokio::time::sleep(std::time::Duration::from_secs(backoff)) => None,
        _ = shutdown_rx.recv() => Some("Shutdown"),
        _ = cancel_rx.recv() => Some("Cancel"),
    }
}

async fn save_download_checkpoint(
    hasher: &Sha256,
    downloaded: u64,
    checkpoints: &mut Vec<Checkpoint>,
    meta_file_path: &Path,
    expected_hash: &Option<String>,
    is_sha256: bool,
) {
    let state_hex = serialize_hasher(hasher);
    checkpoints.push(Checkpoint {
        downloaded_bytes: downloaded,
        hasher_state: state_hex,
    });
    if checkpoints.len() > 5 {
        checkpoints.remove(0);
    }
    save_metadata(meta_file_path, expected_hash, is_sha256, checkpoints).await;
}

pub struct DownloadCleanupGuard {
    state: Arc<AppState>,
    id: String,
}

impl Drop for DownloadCleanupGuard {
    fn drop(&mut self) {
        lock_mutex(&self.state.active_downloads).remove(&self.id);
        lock_mutex(&self.state.download_tasks).remove(&self.id);
    }
}

pub async fn perform_model_download(
    state: Arc<AppState>,
    id: String,
    repo: String,
    filename: String,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
    mut cancel_rx: tokio::sync::broadcast::Receiver<()>,
) {
    let _guard = DownloadCleanupGuard {
        state: state.clone(),
        id: id.clone(),
    };

    let downloads_dir = manager::types::resolve_absolute_path(&state.config.downloads_directory);

    if let Err(e) = tokio::fs::create_dir_all(&downloads_dir).await {
        error!("Failed to create downloads directory for {}: {}", id, e);
        return;
    }

    let files = manager::get_split_filenames(&filename);
    let shared_downloaded = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let grand_total = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let active_streams = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let session_downloaded = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let download_client = match reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(std::time::Duration::from_secs(10))
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .pool_idle_timeout(std::time::Duration::from_secs(55))
        .build()
    {
        Ok(client) => Arc::new(client),
        Err(e) => {
            error!(
                "Failed to build reqwest client for downloader (model {}): {}",
                id, e
            );
            return;
        }
    };

    let hf_token = std::env::var("HF_TOKEN").ok();

    let mut chunk_infos = Vec::new();

    // Sequential HEAD pre-check to establish totals immediately
    for fname in &files {
        let file_path = downloads_dir.join(fname);
        let url = format!(
            "{}/{}/resolve/main/{}",
            state.config.hf_base_url.trim_end_matches('/'),
            repo,
            fname
        );

        if let Ok(meta) = tokio::fs::metadata(&file_path).await
            && meta.len() > 0
        {
            shared_downloaded.fetch_add(meta.len(), std::sync::atomic::Ordering::Relaxed);
            grand_total.fetch_add(meta.len(), std::sync::atomic::Ordering::Relaxed);
            chunk_infos.push((fname.clone(), url, None, true, meta.len()));
            continue;
        }

        let mut remote_sha256 = None;
        let mut chunk_size = 0;
        let mut backoff = 2;
        let mut retries = 0;
        const MAX_RETRIES: usize = 5;
        loop {
            let mut req = download_client.head(&url);
            if let Some(token) = &hf_token {
                req = req.bearer_auth(token);
            }
            match req.send().await {
                Ok(head_res) => {
                    if head_res.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                        retries += 1;
                        if retries > MAX_RETRIES {
                            error!(
                                "Max retries reached for 429 Too Many Requests during metadata fetch for {}. Aborting.",
                                fname
                            );
                            return;
                        }
                        warn!(
                            "429 Too Many Requests during metadata fetch for {}. Retrying ({}/{}) in {} seconds...",
                            fname, retries, MAX_RETRIES, backoff
                        );
                        if let Some(reason) =
                            wait_for_backoff(backoff, &mut shutdown_rx, &mut cancel_rx).await
                        {
                            info!(
                                "{} signal received during metadata fetch backoff. Aborting.",
                                reason
                            );
                            return;
                        }
                        backoff = (backoff * 2).min(30);
                        continue;
                    }

                    if !head_res.status().is_success() && !head_res.status().is_redirection() {
                        warn!(
                            "Unexpected status code {} during metadata fetch for {}",
                            head_res.status(),
                            fname
                        );
                        break;
                    }

                    let size_header = head_res
                        .headers()
                        .get("X-Linked-Size")
                        .or_else(|| head_res.headers().get(reqwest::header::CONTENT_LENGTH));
                    if let Some(cl) = size_header
                        && let Ok(s) = cl.to_str().unwrap_or("0").parse::<u64>()
                    {
                        // Ignore small Content-Length values that represent the 302 redirect body
                        if !head_res.status().is_redirection()
                            || s > 10000
                            || head_res.headers().contains_key("X-Linked-Size")
                        {
                            chunk_size = s;
                            grand_total.fetch_add(s, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    if let Some(etag) = head_res
                        .headers()
                        .get("X-Linked-Etag")
                        .or_else(|| head_res.headers().get("ETag"))
                        && let Ok(etag_str) = etag.to_str()
                    {
                        let clean_etag = etag_str
                            .strip_prefix("W/")
                            .unwrap_or(etag_str)
                            .trim_matches('"');
                        if clean_etag.len() == 64
                            && clean_etag.chars().all(|c| c.is_ascii_hexdigit())
                        {
                            remote_sha256 = Some(clean_etag.to_string());
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to fetch metadata for {}: {}", fname, e);
                }
            }
            break;
        }
        chunk_infos.push((fname.clone(), url, remote_sha256, false, chunk_size));
    }

    // Check if fully completed already
    if chunk_infos.iter().all(|(_, _, _, done, _)| *done) {
        let mut status = lock_status(&state.engine_status);
        status.downloaded_models.insert(id.clone());
        return;
    }

    // Wait in the queue for an available download slot
    let _permit = tokio::select! {
        permit_res = state.download_semaphore.acquire() => {
            match permit_res {
                Ok(p) => p,
                Err(_) => return, // Semaphore closed, safely shutting down
            }
        }
        _ = shutdown_rx.recv() => {
            info!("Shutdown signal received while waiting in queue for {}. Aborting.", id);
            return;
        }
        _ = cancel_rx.recv() => {
            info!("Cancel signal received while waiting in queue for {}. Aborting.", id);
            return;
        }
    };

    update_status_connecting(
        &state,
        &id,
        shared_downloaded.load(std::sync::atomic::Ordering::Relaxed),
    );

    let start_time = std::time::Instant::now();
    let mut join_set = tokio::task::JoinSet::new();

    // Limit concurrent chunk downloads to prevent blasting HF and getting 429s
    let chunk_semaphore = Arc::new(tokio::sync::Semaphore::new(
        state.config.max_concurrent_chunk_downloads,
    ));

    for (chunk_filename, url, remote_sha256, is_already_done, chunk_size) in chunk_infos {
        if is_already_done {
            continue;
        }
        let state = state.clone();
        let client = download_client.clone();
        let id = id.clone();
        let downloads_dir = downloads_dir.clone();
        let mut shutdown_rx = shutdown_rx.resubscribe();
        let mut cancel_rx = cancel_rx.resubscribe();
        let shared_downloaded = shared_downloaded.clone();
        let grand_total = grand_total.clone();
        let sem = chunk_semaphore.clone();
        let active_streams = active_streams.clone();
        let hf_token = hf_token.clone();
        let hf_base_url = state.config.hf_base_url.clone();
        let session_downloaded = session_downloaded.clone();

        join_set.spawn(async move {
            let _permit = tokio::select! {
                res = sem.acquire_owned() => {
                    match res {
                        Ok(p) => p,
                        Err(_) => return false,
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Shutdown signal received while waiting for chunk permit. Aborting.");
                    return false;
                }
                _ = cancel_rx.recv() => {
                    info!("Cancel signal received while waiting for chunk permit. Aborting.");
                    return false;
                }
            };
            download_chunk(
                state,
                client,
                id,
                url,
                chunk_filename,
                downloads_dir,
                remote_sha256,
                shared_downloaded,
                grand_total,
                shutdown_rx,
                cancel_rx,
                start_time,
                session_downloaded,
                chunk_size,
                active_streams,
                hf_token,
                hf_base_url,
            )
            .await
        });
    }

    // We intentionally drain the entire JoinSet to completion here instead of
    // aborting early if one chunk fails. Because the downloader supports partial
    // HTTP byte-resumes, letting sibling chunk tasks finish (even if one fails
    // due to a network hiccup) saves the user significant bandwidth when they retry.
    let mut hash_mismatch_occurred = false;
    let mut task_failed = false;
    while let Some(res) = join_set.join_next().await {
        match res {
            Ok(chunk_mismatch) => {
                if chunk_mismatch {
                    hash_mismatch_occurred = true;
                }
            }
            Err(e) => {
                error!("A chunk download task failed or panicked: {}", e);
                task_failed = true;
            }
        }
    }

    // Finalize state
    let mut status = lock_status(&state.engine_status);
    if hash_mismatch_occurred {
        status.corrupted_models.insert(id.clone());
    } else if !task_failed {
        let mut all_exist = true;
        for f in &files {
            if !downloads_dir.join(f).exists() {
                all_exist = false;
                break;
            }
        }
        if all_exist {
            info!("Finished downloading all parts for {}", id);
            status.downloaded_models.insert(id.clone());
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn download_chunk(
    state: Arc<AppState>,
    client: Arc<reqwest::Client>,
    id: String,
    url: String,
    filename: String,
    downloads_dir: std::path::PathBuf,
    remote_sha256: Option<String>,
    shared_downloaded: Arc<std::sync::atomic::AtomicU64>,
    grand_total: Arc<std::sync::atomic::AtomicU64>,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
    mut cancel_rx: tokio::sync::broadcast::Receiver<()>,
    start_time: std::time::Instant,
    session_downloaded: Arc<std::sync::atomic::AtomicU64>,
    known_chunk_size: u64,
    active_streams: Arc<std::sync::atomic::AtomicUsize>,
    hf_token: Option<String>,
    hf_base_url: String,
) -> bool {
    let tmp_file_path = downloads_dir.join(format!("{}.tmp", filename));
    let meta_file_path = downloads_dir.join(format!("{}.meta", filename));

    if let Some(parent) = tmp_file_path.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }

    let mut retries = 0;
    let max_retries = state.config.download_retry_max_attempts;
    let mut backoff = state.config.download_retry_backoff_seconds;

    loop {
        let (
            mut existing_size,
            mut expected_hash,
            mut is_sha256,
            mut initial_hasher,
            mut checkpoints,
        ) = check_existing_metadata(&state, &id, &tmp_file_path, &meta_file_path, &remote_sha256)
            .await;

        let original_existing_size = existing_size;
        shared_downloaded.fetch_add(existing_size, std::sync::atomic::Ordering::Relaxed);

        let (mut res, _final_url) = loop {
            let (response, final_url) = match initiate_request(
                &client,
                &url,
                &id,
                &mut existing_size,
                &active_streams,
                &mut shutdown_rx,
                &mut cancel_rx,
                hf_token.as_deref(),
                &hf_base_url,
                &state.config,
            )
            .await
            {
                Some(result) => result,
                None => {
                    balance_chunk_counters_before_stream(
                        &shared_downloaded,
                        &session_downloaded,
                        original_existing_size,
                    );
                    return false;
                }
            };

            if verify_and_update_etag(
                &response,
                &id,
                &mut existing_size,
                &mut expected_hash,
                &mut is_sha256,
            )
            .await
            {
                // ETag mismatch detected! The upstream file has changed.
                // We mutate `existing_size` to 0 in memory and `continue` this inner loop.
                // This forces `initiate_request` to ask for the full file from the beginning
                // on the next pass, without ever re-reading the outdated `.meta` file.
                continue;
            }
            break (response, final_url);
        };

        let is_partial = res.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        if !is_partial && existing_size > 0 {
            existing_size = 0;
        }

        // Balance the atomic counters after ETag/headers adjustments.
        // `original_existing_size` was blindly added at the top of the outer loop.
        // If an ETag mismatch occurred, `existing_size` was mutated to 0 during the inner loop.
        // This block completely subtracts the old `original_existing_size` to perfectly
        // cancel out the earlier addition, preventing the UI from double-counting the bytes.
        let mut contribution_from_existing = original_existing_size;
        if existing_size < original_existing_size {
            let delta = original_existing_size - existing_size;
            shared_downloaded.fetch_sub(delta, std::sync::atomic::Ordering::Relaxed);
            contribution_from_existing -= delta;
        }

        if known_chunk_size == 0
            && let Some(content_length) = res.content_length()
        {
            let actual_chunk_size = existing_size + content_length;
            grand_total.fetch_add(actual_chunk_size, std::sync::atomic::Ordering::Relaxed);
        }

        if existing_size == 0 {
            initial_hasher = Sha256::new();
            checkpoints.clear();
        }

        save_metadata(&meta_file_path, &expected_hash, is_sha256, &checkpoints).await;

        // Truncates the .tmp file to 0 bytes using File::create if existing_size is 0
        let mut file = match open_temp_file(&tmp_file_path, is_partial, existing_size).await {
            Ok(f) => f,
            Err(e) => {
                error!("Failed to open file for {}: {}", id, e);
                balance_chunk_counters_before_stream(
                    &shared_downloaded,
                    &session_downloaded,
                    contribution_from_existing,
                );
                return false;
            }
        };

        let stream_result = process_download_stream(
            &state,
            &client,
            &id,
            &mut res,
            &mut file,
            &mut initial_hasher,
            existing_size,
            &meta_file_path,
            &expected_hash,
            is_sha256,
            checkpoints,
            &mut shutdown_rx,
            &mut cancel_rx,
            shared_downloaded.clone(),
            grand_total.clone(),
            start_time,
            session_downloaded.clone(),
            active_streams.clone(),
        )
        .await;

        if stream_result.is_aborted {
            balance_chunk_counters(
                &shared_downloaded,
                &session_downloaded,
                contribution_from_existing,
                stream_result.streamed_bytes,
            );
            return false;
        }

        if stream_result.stream_error {
            balance_chunk_counters(
                &shared_downloaded,
                &session_downloaded,
                contribution_from_existing,
                stream_result.streamed_bytes,
            );
            retries += 1;
            if retries > max_retries {
                error!(
                    "Max retries reached for download stream of chunk {}. Aborting.",
                    filename
                );
                return false;
            }
            warn!(
                "Download stream for {} failed. Retrying ({}/{}) in {} seconds...",
                filename, retries, max_retries, backoff
            );
            if let Some(reason) = wait_for_backoff(backoff, &mut shutdown_rx, &mut cancel_rx).await
            {
                info!(
                    "{} signal received during stream retry backoff. Aborting.",
                    reason
                );
                return false;
            }
            backoff = (backoff * 2).min(60);
            continue;
        }

        let final_hash = hex::encode(initial_hasher.finalize());
        let hash_mismatch = if let Some(ref expected) = expected_hash {
            is_sha256
                && expected.len() == 64
                && expected.chars().all(|c| c.is_ascii_hexdigit())
                && &final_hash != expected
        } else {
            false
        };

        // Explicitly flush all data to physical storage to prevent corruption on power loss
        if let Err(e) = file.sync_all().await {
            error!("Failed to sync downloaded data to disk for {}: {}", id, e);
        }

        // Explicitly drop the file handle to release the OS lock before renaming,
        // which is required to prevent sharing violations on Windows.
        drop(file);

        let file_path = downloads_dir.join(&filename);
        finalize_download(
            &id,
            &filename,
            &file_path,
            &tmp_file_path,
            &meta_file_path,
            hash_mismatch,
        )
        .await;

        // CRITICAL: Do NOT call `balance_chunk_counters` on this success path!
        // Balancing is only for error/retry paths to prevent double-counting.
        // Once a chunk succeeds, its bytes are permanently written to disk and must
        // remain in the global totals, otherwise the UI progress bar will regress.

        return hash_mismatch;
    }
}

/// Balances download counters after streaming completes.
///
/// Called by `download_chunk` to subtract the total contribution (existing + streamed)
/// from `shared_downloaded` and `session_downloaded` before returning.
///
/// This function is intentionally separate so the invariant "every fetch_add has a
/// matching fetch_sub on every exit path" can be unit tested independently of the
/// async download machinery.
///
/// WARNING: Do NOT call this on a successful download path! It is only intended for
/// `Aborted` and `StreamError` exit paths so that the retry loop can safely re-add
/// the bytes without double-counting them.
pub(crate) fn balance_chunk_counters(
    shared_downloaded: &std::sync::atomic::AtomicU64,
    session_downloaded: &std::sync::atomic::AtomicU64,
    contribution_from_existing: u64,
    streamed_bytes: u64,
) {
    let total = contribution_from_existing + streamed_bytes;
    shared_downloaded.fetch_sub(total, std::sync::atomic::Ordering::Relaxed);
    session_downloaded.fetch_sub(streamed_bytes, std::sync::atomic::Ordering::Relaxed);
}

/// Balances download counters on early exit before streaming begins.
///
/// Called by `download_chunk` when the file cannot be opened or initiate_request
/// fails — only the `contribution_from_existing` is subtracted (no streamed bytes
/// were added because the stream was never entered).
pub(crate) fn balance_chunk_counters_before_stream(
    shared_downloaded: &std::sync::atomic::AtomicU64,
    _session_downloaded: &std::sync::atomic::AtomicU64,
    contribution_from_existing: u64,
) {
    shared_downloaded.fetch_sub(
        contribution_from_existing,
        std::sync::atomic::Ordering::Relaxed,
    );
}

async fn check_existing_metadata(
    state: &Arc<AppState>,
    id: &str,
    tmp_file_path: &Path,
    meta_file_path: &Path,
    remote_sha256: &Option<String>,
) -> (u64, Option<String>, bool, Sha256, Vec<Checkpoint>) {
    let mut existing_size = 0;
    let mut expected_hash = remote_sha256.clone();
    let mut is_sha256 = remote_sha256.is_some();
    let mut initial_hasher = Sha256::new();
    let mut checkpoints = Vec::new();

    if let Ok(meta_str) = tokio::fs::read_to_string(meta_file_path).await {
        let mut parsed_meta: Option<DownloadMetadata> = None;

        if let Ok(meta) = serde_json::from_str::<DownloadMetadata>(&meta_str) {
            parsed_meta = Some(meta);
        } else if let Ok(old_meta) = serde_json::from_str::<serde_json::Value>(&meta_str) {
            // Fallback for migrating old metadata
            if let Some(bytes) = old_meta.get("downloaded_bytes").and_then(|v| v.as_u64()) {
                let mut cps = Vec::new();
                if let Some(state_hex) = old_meta.get("hasher_state").and_then(|v| v.as_str()) {
                    cps.push(Checkpoint {
                        downloaded_bytes: bytes,
                        hasher_state: state_hex.to_string(),
                    });
                }
                parsed_meta = Some(DownloadMetadata {
                    expected_hash: old_meta
                        .get("expected_hash")
                        .and_then(|v| v.as_str().map(|s| s.to_string())),
                    is_sha256: old_meta
                        .get("is_sha256")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    checkpoints: cps,
                    hasher_version: 0,
                });
            }
        }

        if let Some(meta) = parsed_meta {
            let local_hash = meta.expected_hash.clone();
            let local_is_sha256 = meta.is_sha256;

            // If we successfully fetched a remote SHA-256, and it DIFFERS from the local tracked SHA-256,
            // the file has been updated upstream. We must restart the download!
            if let Some(remote) = remote_sha256 {
                if local_is_sha256 && local_hash != Some(remote.to_string()) {
                    warn!(
                        "Upstream file for {} has changed (SHA-256 mismatch). Discarding local partial download.",
                        id
                    );
                    return (0, expected_hash, is_sha256, initial_hasher, checkpoints);
                }
            } else if let Some(hash) = local_hash {
                // Fallback to whatever was tracked if we couldn't fetch remote_sha256
                expected_hash = Some(hash.to_string());
                is_sha256 = local_is_sha256;
            }

            if meta.hasher_version != HASHER_STATE_VERSION {
                warn!(
                    "Metadata for {} has an incompatible hasher state version. Discarding local partial download.",
                    id
                );
                return (0, expected_hash, is_sha256, initial_hasher, Vec::new());
            }

            // Find the most recent valid checkpoint
            if let Ok(metadata) = tokio::fs::metadata(tmp_file_path).await {
                let actual_size = metadata.len();
                // Iterate backwards (newest first)
                for cp in meta.checkpoints.iter().rev() {
                    if cp.downloaded_bytes <= actual_size
                        && let Some(restored) = deserialize_hasher(&cp.hasher_state)
                    {
                        initial_hasher = restored;
                        existing_size = cp.downloaded_bytes;
                        checkpoints = meta.checkpoints.clone();

                        // If the physical file is larger than the checkpoint, it means the OS crashed
                        // before syncing the newest metadata to disk. We truncate the physical file back to the checkpoint.
                        if actual_size > existing_size {
                            info!(
                                "Dropping back to checkpoint at {} bytes for {}",
                                existing_size, id
                            );
                            if let Ok(f) = tokio::fs::OpenOptions::new()
                                .write(true)
                                .open(tmp_file_path)
                                .await
                            {
                                if let Err(e) = f.set_len(existing_size).await {
                                    error!(
                                        "Failed to truncate {} to checkpoint size {}: {}",
                                        id, existing_size, e
                                    );
                                    return (
                                        0,
                                        expected_hash,
                                        is_sha256,
                                        Sha256::new(),
                                        Vec::new(),
                                    );
                                }
                            } else {
                                error!("Failed to open {} for truncation", id);
                                return (0, expected_hash, is_sha256, Sha256::new(), Vec::new());
                            }
                        }
                        break;
                    }
                }
            }
        } else {
            warn!(
                "Corrupted or invalid metadata file found for {}. Discarding and restarting download.",
                id
            );
            {
                let mut dl = lock_mutex(&state.active_downloads);
                if let Some(status) = dl.get_mut(id) {
                    status.state = "Corrupted metadata. Restarting...".to_string();
                }
            }
            let _ = tokio::fs::remove_file(meta_file_path).await;
            let _ = tokio::fs::remove_file(tmp_file_path).await;
            tokio::time::sleep(std::time::Duration::from_secs(CORRUPT_RESTART_DELAY_SECS)).await;
        }
    }

    (
        existing_size,
        expected_hash,
        is_sha256,
        initial_hasher,
        checkpoints,
    )
}

fn update_status_connecting(state: &Arc<AppState>, id: &str, existing_size: u64) {
    let mut dl = lock_mutex(&state.active_downloads);
    if let Some(status) = dl.get_mut(id) {
        status.bytes_transferred = existing_size;
        status.state = "Connecting...".to_string();
    }
}

#[allow(clippy::too_many_arguments)]
async fn initiate_request(
    client: &reqwest::Client,
    url: &str,
    id: &str,
    existing_size: &mut u64,
    active_streams: &Arc<std::sync::atomic::AtomicUsize>,
    shutdown_rx: &mut tokio::sync::broadcast::Receiver<()>,
    cancel_rx: &mut tokio::sync::broadcast::Receiver<()>,
    hf_token: Option<&str>,
    hf_base_url: &str,
    config: &crate::AppConfig,
) -> Option<(reqwest::Response, String)> {
    let mut current_url = url.to_string();
    let mut backoff = config.download_retry_backoff_seconds;
    let mut retries = 0;
    let mut redirects = 0;
    const MAX_REDIRECTS: usize = 10;

    loop {
        let mut req = client.get(&current_url);
        if *existing_size > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={}-", existing_size));
        }
        if current_url.starts_with(hf_base_url.trim_end_matches('/'))
            && let Some(token) = hf_token
        {
            req = req.bearer_auth(token);
        }

        // We cannot use reqwest's built-in global timeout on the Client because it would abort long,
        // successful streaming downloads. Instead, we wrap the initial `send()` connection phase in a
        // tokio timeout to ensure we don't hang indefinitely if the server accepts the connection but
        // never sends the HTTP headers.
        let timeout_duration =
            std::time::Duration::from_secs(config.download_stream_chunk_timeout_seconds);
        match tokio::time::timeout(timeout_duration, req.send()).await {
            Ok(Ok(r)) => {
                if r.status().is_redirection()
                    && let Some(location) = r.headers().get(reqwest::header::LOCATION)
                    && let Ok(new_url) = location.to_str()
                {
                    info!("Following redirect for {} to {}", id, new_url);
                    current_url = new_url.to_string();
                    redirects += 1;
                    if redirects > MAX_REDIRECTS {
                        break;
                    }
                    // Don't reset existing_size, let the server decide with RANGE_NOT_SATISFIABLE
                    continue;
                }
                if r.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                    if active_streams.load(std::sync::atomic::Ordering::Relaxed) > 0 {
                        warn!(
                            "Hugging Face rate limit (429) hit for {}. Other chunks are actively downloading, waiting {} seconds...",
                            id, backoff
                        );
                        if let Some(reason) =
                            wait_for_backoff(backoff, shutdown_rx, cancel_rx).await
                        {
                            info!(
                                "{} signal received during request backoff for {}. Aborting.",
                                reason, id
                            );
                            return None;
                        }
                        backoff = (backoff * 2).min(60);
                        continue;
                    } else {
                        retries += 1;
                        if retries > config.download_retry_max_attempts {
                            error!(
                                "Max retries reached for 429 Too Many Requests on {}. Aborting.",
                                id
                            );
                            return None;
                        }
                        warn!(
                            "Hugging Face rate limit (429) hit for {}. No active chunks. Retrying ({}/{}) in {} seconds...",
                            id, retries, config.download_retry_max_attempts, backoff
                        );
                        if let Some(reason) =
                            wait_for_backoff(backoff, shutdown_rx, cancel_rx).await
                        {
                            info!(
                                "{} signal received during request backoff for {}. Aborting.",
                                reason, id
                            );
                            return None;
                        }
                        backoff = (backoff * 2).min(60);
                        continue;
                    }
                } else if r.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                    *existing_size = 0;
                    continue; // Loop will retry without the RANGE header
                } else if !r.status().is_success() {
                    error!("Download failed for {} with status: {}", id, r.status());
                    return None;
                } else {
                    return Some((r, current_url));
                }
            }
            Ok(Err(e)) => {
                retries += 1;
                if retries > config.download_retry_max_attempts {
                    error!(
                        "Max retries reached for network error on {}: {}. Aborting.",
                        id, e
                    );
                    return None;
                }
                warn!(
                    "Network error for {}: {}. Retrying ({}/{}) in {} seconds...",
                    id, e, retries, config.download_retry_max_attempts, backoff
                );
                if let Some(reason) = wait_for_backoff(backoff, shutdown_rx, cancel_rx).await {
                    info!(
                        "{} signal received during network error backoff for {}. Aborting.",
                        reason, id
                    );
                    return None;
                }
                backoff = (backoff * 2).min(60);
                continue;
            }
            Err(_) => {
                retries += 1;
                if retries > config.download_retry_max_attempts {
                    error!(
                        "Max retries reached for connection timeout on {}. Aborting.",
                        id
                    );
                    return None;
                }
                warn!(
                    "Connection timeout for {}. Retrying ({}/{}) in {} seconds...",
                    id, retries, config.download_retry_max_attempts, backoff
                );
                if let Some(reason) = wait_for_backoff(backoff, shutdown_rx, cancel_rx).await {
                    info!(
                        "{} signal received during connection timeout backoff for {}. Aborting.",
                        reason, id
                    );
                    return None;
                }
                backoff = (backoff * 2).min(60);
                continue;
            }
        }
    }
    error!("Too many redirects for {}", id);
    None
}

/// Verifies the ETag of the response. Returns `true` if a restart is needed.
async fn verify_and_update_etag(
    res: &reqwest::Response,
    id: &str,
    existing_size: &mut u64,
    expected_hash: &mut Option<String>,
    is_sha256: &mut bool,
) -> bool {
    info!("Inspecting HTTP Headers for {}:", id);
    for (k, v) in res.headers() {
        info!("  {}: {:?}", k.as_str(), v.to_str().unwrap_or("<binary>"));
    }

    if *is_sha256 {
        info!(
            "  Already tracking a verified upstream SHA-256 for {}. Ignoring CDN ETags.",
            id
        );
        return false;
    }

    let mut new_etag = None;
    let mut x_linked_found = false;

    let x_linked = res
        .headers()
        .get("X-Linked-Etag")
        .and_then(|v| v.to_str().ok());
    let std_etag = res.headers().get("ETag").and_then(|v| v.to_str().ok());

    info!("  Parsed X-Linked-Etag: {:?}", x_linked);
    info!("  Parsed ETag: {:?}", std_etag);

    if let Some(etag_str) = x_linked {
        x_linked_found = true;
        let clean_etag = etag_str
            .strip_prefix("W/")
            .unwrap_or(etag_str)
            .trim_matches('"');
        if !clean_etag.is_empty() {
            new_etag = Some(clean_etag.to_string());
            info!("  Selected X-Linked-Etag for validation: {}", clean_etag);
        }
    } else if let Some(etag_str) = std_etag {
        let clean_etag = etag_str
            .strip_prefix("W/")
            .unwrap_or(etag_str)
            .trim_matches('"');
        if !clean_etag.is_empty() {
            new_etag = Some(clean_etag.to_string());
            info!(
                "  Selected standard ETag for resume tracking: {}",
                clean_etag
            );
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
        *expected_hash = new_etag.clone();
        *is_sha256 = x_linked_found;
        true // Restart needed
    } else {
        if new_etag.is_some() {
            *expected_hash = new_etag;
            *is_sha256 = x_linked_found;
        }
        false // OK to proceed
    }
}

async fn save_metadata(
    meta_file_path: &Path,
    expected_hash: &Option<String>,
    is_sha256: bool,
    checkpoints: &[Checkpoint],
) {
    let meta = DownloadMetadata {
        expected_hash: expected_hash.clone(),
        is_sha256,
        checkpoints: checkpoints.to_vec(),
        hasher_version: HASHER_STATE_VERSION,
    };
    if let Ok(data) = serde_json::to_string(&meta) {
        let tmp_path = meta_file_path.with_extension("meta.tmp");
        if tokio::fs::write(&tmp_path, data).await.is_ok()
            && let Err(e) = tokio::fs::rename(&tmp_path, meta_file_path).await
        {
            error!("Failed to atomically rename metadata file: {}", e);
        }
    }
}

async fn open_temp_file(
    tmp_file_path: &Path,
    is_partial: bool,
    existing_size: u64,
) -> std::io::Result<tokio::fs::File> {
    if is_partial && existing_size > 0 {
        tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(tmp_file_path)
            .await
    } else {
        tokio::fs::File::create(tmp_file_path).await
    }
}

struct ProcessStreamResult {
    stream_error: bool,
    is_aborted: bool,
    streamed_bytes: u64,
}

#[allow(clippy::too_many_arguments)]
async fn process_download_stream(
    state: &Arc<AppState>,
    _client: &Arc<reqwest::Client>,
    id: &str,
    res: &mut reqwest::Response,
    file: &mut tokio::fs::File,
    hasher: &mut Sha256,
    existing_size: u64,
    meta_file_path: &Path,
    expected_hash: &Option<String>,
    is_sha256: bool,
    mut checkpoints: Vec<Checkpoint>,
    shutdown_rx: &mut tokio::sync::broadcast::Receiver<()>,
    cancel_rx: &mut tokio::sync::broadcast::Receiver<()>,
    shared_downloaded: Arc<std::sync::atomic::AtomicU64>,
    grand_total: Arc<std::sync::atomic::AtomicU64>,
    start_time: std::time::Instant,
    session_downloaded: Arc<std::sync::atomic::AtomicU64>,
    active_streams: Arc<std::sync::atomic::AtomicUsize>,
) -> ProcessStreamResult {
    let _active_guard = ActiveStreamGuard::new(active_streams);

    let starting_existing_size = existing_size;
    let mut downloaded: u64 = existing_size;
    let mut stream_error = false;
    let mut last_meta_save = std::time::Instant::now();
    let mut last_ui_update = std::time::Instant::now();
    let mut bytes_since_last_ui_update = 0;

    let timeout_secs = state.config.download_stream_chunk_timeout_seconds;

    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.recv() => {
                info!("Shutdown signal received. Flushing metadata for {}...", id);
                save_download_checkpoint(hasher, downloaded, &mut checkpoints, meta_file_path, expected_hash, is_sha256).await;
                return ProcessStreamResult {
                    stream_error: false,
                    is_aborted: true,
                    streamed_bytes: downloaded - starting_existing_size,
                };
            }
            _ = cancel_rx.recv() => {
                info!("Cancel signal received. Flushing metadata for {}...", id);
                save_download_checkpoint(hasher, downloaded, &mut checkpoints, meta_file_path, expected_hash, is_sha256).await;
                return ProcessStreamResult {
                    stream_error: false,
                    is_aborted: true,
                    streamed_bytes: downloaded - starting_existing_size,
                };
            }
            chunk_res = tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), res.chunk()) => {
                match chunk_res {
                    Ok(Ok(Some(bytes))) => {
                        hasher.update(&bytes);

                        if let Err(e) = file.write_all(&bytes).await {
                            error!("Failed to write to file for {}: {}", id, e);
                            stream_error = true;
                            break;
                        }
                        let chunk_len = bytes.len() as u64;
                        downloaded += chunk_len;
                        bytes_since_last_ui_update += chunk_len;

                        let total_transferred = shared_downloaded.fetch_add(chunk_len, std::sync::atomic::Ordering::Relaxed) + chunk_len;
                        let current_session_bytes = session_downloaded.fetch_add(chunk_len, std::sync::atomic::Ordering::Relaxed) + chunk_len;

                        if last_ui_update.elapsed().as_millis() >= UPDATE_INTERVAL_MS
                            || bytes_since_last_ui_update >= UPDATE_BYTES_THRESHOLD
                        {
                            let elapsed = start_time.elapsed().as_secs_f64();
                            let speed = if elapsed > 0.0 {
                                current_session_bytes as f64 / elapsed
                            } else {
                                0.0
                            };

                            let current_grand_total = grand_total.load(std::sync::atomic::Ordering::Relaxed);
                            {
                                let mut dl = lock_mutex(&state.active_downloads);
                                if let Some(status) = dl.get_mut(id) {
                                    status.bytes_transferred = total_transferred;
                                    status.total_bytes = current_grand_total;
                                    status.current_speed_bps = speed;
                                    status.state = "Downloading...".to_string();
                                }
                            }

                            last_ui_update = std::time::Instant::now();
                            bytes_since_last_ui_update = 0;
                        }

                        if last_meta_save.elapsed().as_secs() >= META_SAVE_INTERVAL_SECS {
                            save_download_checkpoint(hasher, downloaded, &mut checkpoints, meta_file_path, expected_hash, is_sha256).await;
                            last_meta_save = std::time::Instant::now();
                        }
                    }
                    Ok(Ok(None)) => {
                        break; // Stream finished
                    }
                    Ok(Err(e)) => {
                        error!("Error while streaming {}: {}", id, e);
                        stream_error = true;
                        break;
                    }
                    Err(_) => {
                        error!("Stream for {} timed out after {} seconds.", id, timeout_secs);
                        stream_error = true;
                        break;
                    }
                }
            }
        }
    }

    ProcessStreamResult {
        stream_error,
        is_aborted: false,
        streamed_bytes: downloaded - starting_existing_size,
    }
}

async fn finalize_download(
    id: &str,
    filename: &str,
    file_path: &Path,
    tmp_file_path: &Path,
    meta_file_path: &Path,
    hash_mismatch: bool,
) {
    let target_filename = if hash_mismatch {
        format!("{}.corrupted", filename)
    } else {
        filename.to_string()
    };

    let target_path = file_path.with_file_name(&target_filename);

    let mut success = false;
    if let Err(e) = tokio::fs::rename(tmp_file_path, &target_path).await {
        warn!(
            "Failed to rename temp file for {} ({}). Falling back to copy...",
            id, e
        );

        let copy_tmp_path = target_path.with_file_name(format!("{}.copy_tmp", target_filename));

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
                    // CRITICAL: Do not remove the original `tmp_file_path` here!
                    // If the fallback rename fails, we want to preserve the downloaded data
                    // so the user doesn't have to restart the download from scratch.
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
                // CRITICAL: Do not remove the original `tmp_file_path` here!
                // If the fallback copy fails, we want to preserve the downloaded data
                // so the user doesn't have to restart the download from scratch.
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn test_serialize_deserialize_hasher() {
        let mut hasher1 = Sha256::new();
        hasher1.update(b"hello");
        let state_hex = serialize_hasher(&hasher1);

        let mut hasher2 = deserialize_hasher(&state_hex).expect("Failed to deserialize");
        hasher2.update(b" world");

        let result = hex::encode(hasher2.finalize());
        assert_eq!(
            result,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_deserialize_hasher_invalid_hex() {
        assert!(deserialize_hasher("invalidhex").is_none());
        assert!(deserialize_hasher("deadbeef").is_none()); // Valid hex characters, but incomplete byte array
    }

    #[tokio::test]
    async fn test_finalize_download_fallback_failure_retains_tmp_file() {
        let downloads_dir = std::env::temp_dir().join("test_finalize_failure");
        let _ = tokio::fs::create_dir_all(&downloads_dir).await;

        let id = "test-model";
        let filename = "model.safetensors";
        let file_path = downloads_dir.join(filename);
        let tmp_file_path = downloads_dir.join(format!("{}.tmp", filename));
        let meta_file_path = downloads_dir.join(format!("{}.meta", filename));

        // Create dummy tmp and meta files
        let _ = tokio::fs::write(&tmp_file_path, "dummy data").await;
        let _ = tokio::fs::write(&meta_file_path, "{}").await;

        // Force both `rename` and `copy` to fail by creating a directory at `file_path`.
        let _ = tokio::fs::create_dir(&file_path).await;

        let (queue_tx, _) = tokio::sync::mpsc::channel(1);
        let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);
        let db = surrealdb::engine::any::connect("mem://").await.unwrap();
        db.use_ns("test").use_db("test").await.unwrap();

        let _state = Arc::new(crate::AppState {
            queue_tx,
            engine_status: Arc::new(std::sync::Mutex::new(crate::EngineStatus::default())),
            telemetry: Arc::new(std::sync::Mutex::new(crate::TelemetryStore::default())),
            auth_store: Arc::new(std::sync::Mutex::new(crate::auth::AuthStore::default())),
            reqwest_client: reqwest::Client::new(),
            oauth_client: oauth2::basic::BasicClient::new(
                oauth2::ClientId::new("dummy".to_string()),
                None,
                oauth2::AuthUrl::new("http://localhost".to_string()).unwrap(),
                None,
            ),
            config: Arc::new(crate::AppConfig::default()),
            log_buffer: crate::SharedLogBuffer(Arc::new(std::sync::Mutex::new((
                0,
                std::collections::VecDeque::new(),
            )))),
            log_reload_handle: tracing_subscriber::reload::Layer::new(
                tracing_subscriber::EnvFilter::new("info"),
            )
            .1,
            current_log_level: Arc::new(std::sync::Mutex::new("info".to_string())),
            active_downloads: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            download_tasks: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            download_semaphore: Arc::new(tokio::sync::Semaphore::new(2)),
            db,
            shutdown_tx,
        });

        super::finalize_download(
            id,
            filename,
            &file_path,
            &tmp_file_path,
            &meta_file_path,
            false,
        )
        .await;

        // The tmp file should still exist to prevent data loss!
        assert!(
            tmp_file_path.exists(),
            "The .tmp file should have been preserved after fallback failure."
        );
        assert!(
            meta_file_path.exists(),
            "The .meta file should have been preserved after fallback failure."
        );

        let _ = tokio::fs::remove_dir_all(&downloads_dir).await;
    }

    #[tokio::test]
    async fn test_check_existing_metadata_hasher_version_mismatch() {
        let temp_dir = std::env::temp_dir().join("test_metadata_version_mismatch");
        let _ = tokio::fs::create_dir_all(&temp_dir).await;

        let tmp_file_path = temp_dir.join("model.tmp");
        let meta_file_path = temp_dir.join("model.meta");

        let _ = tokio::fs::write(&tmp_file_path, "dummy data").await;

        // Write metadata with an incompatible hasher_version (e.g., 999)
        let meta_json = serde_json::json!({
            "expected_hash": "abcdef",
            "is_sha256": true,
            "checkpoints": [{
                "downloaded_bytes": 10,
                "hasher_state": "dummyhex"
            }],
            "hasher_version": 999
        });
        let _ = tokio::fs::write(&meta_file_path, meta_json.to_string()).await;

        // Create dummy AppState
        let (queue_tx, _) = tokio::sync::mpsc::channel(1);
        let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);
        let db = surrealdb::engine::any::connect("mem://").await.unwrap();
        db.use_ns("test").use_db("test").await.unwrap();

        let state = Arc::new(crate::AppState {
            queue_tx,
            engine_status: Arc::new(std::sync::Mutex::new(crate::EngineStatus::default())),
            telemetry: Arc::new(std::sync::Mutex::new(crate::TelemetryStore::default())),
            auth_store: Arc::new(std::sync::Mutex::new(crate::auth::AuthStore::default())),
            reqwest_client: reqwest::Client::new(),
            oauth_client: oauth2::basic::BasicClient::new(
                oauth2::ClientId::new("dummy".to_string()),
                None,
                oauth2::AuthUrl::new("http://localhost".to_string()).unwrap(),
                None,
            ),
            config: Arc::new(crate::AppConfig::default()),
            log_buffer: crate::SharedLogBuffer(Arc::new(std::sync::Mutex::new((
                0,
                std::collections::VecDeque::new(),
            )))),
            log_reload_handle: tracing_subscriber::reload::Layer::new(
                tracing_subscriber::EnvFilter::new("info"),
            )
            .1,
            current_log_level: Arc::new(std::sync::Mutex::new("info".to_string())),
            active_downloads: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            download_tasks: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            download_semaphore: Arc::new(tokio::sync::Semaphore::new(2)),
            db,
            shutdown_tx,
        });

        let remote_sha256 = None;
        let (existing_size, expected_hash, is_sha256, _hasher, checkpoints) =
            super::check_existing_metadata(
                &state,
                "test-model",
                &tmp_file_path,
                &meta_file_path,
                &remote_sha256,
            )
            .await;

        // The mismatch should cause existing_size to reset to 0 and checkpoints to be cleared
        assert_eq!(
            existing_size, 0,
            "Existing size should be reset to 0 due to version mismatch"
        );
        assert_eq!(
            expected_hash,
            Some("abcdef".to_string()),
            "Expected hash should be preserved"
        );
        assert!(is_sha256, "is_sha256 flag should be preserved");
        assert!(
            checkpoints.is_empty(),
            "Checkpoints should be cleared due to version mismatch"
        );

        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    // ── Atomic counter balancing tests ────
    // These tests exercise the real production helpers (balance_chunk_counters,
    // balance_chunk_counters_before_stream) that download_chunk calls on every
    // exit path. If someone changes download_chunk without keeping the helpers
    // in sync, these tests will catch it.

    #[derive(Clone, Copy, Debug)]
    enum CounterExit {
        InitRequestFail,
        OpenFileFail,
        Aborted,
        StreamError,
        Success,
    }

    /// Simulates the full lifecycle of a single download-chunk loop iteration:
    /// fetch_add(existing), optional ETag delta, then one exit path's subtraction.
    /// Calls the real production helpers on every path.
    fn simulate_counter_balance(
        init_existing: u64,
        etag_delta: Option<u64>,
        streamed_bytes: u64,
        exit_path: CounterExit,
    ) -> (u64, u64) {
        let shared = Arc::new(AtomicU64::new(0));
        let session = Arc::new(AtomicU64::new(0));

        // Step 1: initial contribution (as in download_chunk loop top)
        shared.fetch_add(init_existing, Ordering::Relaxed);

        // Step 2: ETag/headers adjustment (as in download_chunk after verify_and_update_etag)
        let mut contribution_from_existing = init_existing;
        if let Some(delta) = etag_delta {
            shared.fetch_sub(delta, Ordering::Relaxed);
            contribution_from_existing -= delta;
        }

        // Step 3: exit path subtraction - calls the real production helpers
        match exit_path {
            CounterExit::InitRequestFail | CounterExit::OpenFileFail => {
                super::balance_chunk_counters_before_stream(
                    &shared,
                    &session,
                    contribution_from_existing,
                );
            }
            CounterExit::Aborted | CounterExit::StreamError => {
                // Simulate process_download_stream adding streamed bytes to the counters
                shared.fetch_add(streamed_bytes, Ordering::Relaxed);
                session.fetch_add(streamed_bytes, Ordering::Relaxed);
                super::balance_chunk_counters(
                    &shared,
                    &session,
                    contribution_from_existing,
                    streamed_bytes,
                );
            }
            CounterExit::Success => {
                // Simulate process_download_stream adding streamed bytes to the counters
                shared.fetch_add(streamed_bytes, Ordering::Relaxed);
                session.fetch_add(streamed_bytes, Ordering::Relaxed);
                // On success, we do NOT balance the counters. The bytes are permanently downloaded.
            }
        }

        (
            shared.load(Ordering::Relaxed),
            session.load(Ordering::Relaxed),
        )
    }

    #[test]
    fn test_balance_counters_before_stream_no_etag() {
        let shared = Arc::new(AtomicU64::new(0));
        let session = Arc::new(AtomicU64::new(0));
        shared.fetch_add(100, Ordering::Relaxed);
        super::balance_chunk_counters_before_stream(&shared, &session, 100);
        assert_eq!(shared.load(Ordering::Relaxed), 0);
        assert_eq!(session.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_balance_counters_before_stream_with_etag_delta() {
        let shared = Arc::new(AtomicU64::new(0));
        let session = Arc::new(AtomicU64::new(0));
        shared.fetch_add(100, Ordering::Relaxed);
        shared.fetch_sub(50, Ordering::Relaxed);
        super::balance_chunk_counters_before_stream(&shared, &session, 50);
        assert_eq!(shared.load(Ordering::Relaxed), 0);
        assert_eq!(session.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_balance_counters_after_stream_no_etag() {
        let shared = Arc::new(AtomicU64::new(0));
        let session = Arc::new(AtomicU64::new(0));
        shared.fetch_add(100, Ordering::Relaxed);
        shared.fetch_add(50, Ordering::Relaxed);
        session.fetch_add(50, Ordering::Relaxed);
        super::balance_chunk_counters(&shared, &session, 100, 50);
        assert_eq!(shared.load(Ordering::Relaxed), 0);
        assert_eq!(session.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_balance_counters_after_stream_with_etag_delta() {
        let shared = Arc::new(AtomicU64::new(0));
        let session = Arc::new(AtomicU64::new(0));
        shared.fetch_add(100, Ordering::Relaxed);
        shared.fetch_sub(50, Ordering::Relaxed);
        shared.fetch_add(25, Ordering::Relaxed);
        session.fetch_add(25, Ordering::Relaxed);
        super::balance_chunk_counters(&shared, &session, 50, 25);
        assert_eq!(shared.load(Ordering::Relaxed), 0);
        assert_eq!(session.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_balance_counters_zero_existing() {
        let shared = Arc::new(AtomicU64::new(0));
        let session = Arc::new(AtomicU64::new(0));
        shared.fetch_add(200, Ordering::Relaxed);
        session.fetch_add(200, Ordering::Relaxed);
        super::balance_chunk_counters(&shared, &session, 0, 200);
        assert_eq!(shared.load(Ordering::Relaxed), 0);
        assert_eq!(session.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_balance_counters_large_transfer() {
        let shared = Arc::new(AtomicU64::new(0));
        let session = Arc::new(AtomicU64::new(0));
        shared.fetch_add(10_000_000, Ordering::Relaxed);
        shared.fetch_add(50_000_000, Ordering::Relaxed);
        session.fetch_add(50_000_000, Ordering::Relaxed);
        super::balance_chunk_counters(&shared, &session, 10_000_000, 50_000_000);
        assert_eq!(shared.load(Ordering::Relaxed), 0);
        assert_eq!(session.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_session_downloaded_excludes_existing_size() {
        let shared = Arc::new(AtomicU64::new(0));
        let session = Arc::new(AtomicU64::new(0));

        let existing = 5000;
        let streamed = 1000;

        shared.fetch_add(existing, Ordering::Relaxed);
        // Only streamed bytes are added to the session tracker
        shared.fetch_add(streamed, Ordering::Relaxed);
        session.fetch_add(streamed, Ordering::Relaxed);

        assert_eq!(
            session.load(Ordering::Relaxed),
            1000,
            "Session should only contain streamed bytes"
        );
        assert_eq!(
            shared.load(Ordering::Relaxed),
            6000,
            "Shared should contain both"
        );

        super::balance_chunk_counters(&shared, &session, existing, streamed);

        assert_eq!(
            session.load(Ordering::Relaxed),
            0,
            "Session counter should balance to 0"
        );
        assert_eq!(
            shared.load(Ordering::Relaxed),
            0,
            "Shared counter should balance to 0"
        );
    }

    #[test]
    fn test_balance_counters_all_exit_paths() {
        let paths = vec![
            (CounterExit::InitRequestFail, 0, true),
            (CounterExit::OpenFileFail, 0, true),
            (CounterExit::Aborted, 50, true),
            (CounterExit::StreamError, 25, true),
            (CounterExit::Success, 100, false),
        ];
        for (exit, streamed, should_be_zero) in paths {
            let (shared, session) = simulate_counter_balance(100, None, streamed, exit);
            if should_be_zero {
                assert_eq!(
                    shared, 0,
                    "shared_downloaded not zero for path {:?} with {} bytes",
                    exit, streamed
                );
                assert_eq!(
                    session, 0,
                    "session_downloaded not zero for path {:?} with {} bytes",
                    exit, streamed
                );
            } else {
                assert_eq!(
                    shared,
                    100 + streamed,
                    "shared_downloaded incorrect for success path"
                );
                assert_eq!(
                    session, streamed,
                    "session_downloaded incorrect for success path"
                );
            }
        }
    }

    #[test]
    fn test_balance_counters_with_etag_and_streaming_combinations() {
        let etag_deltas = vec![Some(100), Some(50), None];
        let stream_sizes = vec![0u64, 10, 50, 100, 1000];
        let exits = vec![
            CounterExit::Success,
            CounterExit::Aborted,
            CounterExit::StreamError,
            CounterExit::InitRequestFail,
            CounterExit::OpenFileFail,
        ];
        for delta in etag_deltas {
            for streamed in &stream_sizes {
                for exit in &exits {
                    let (shared, session) = simulate_counter_balance(100, delta, *streamed, *exit);
                    if matches!(exit, CounterExit::Success) {
                        let expected_shared = 100 - delta.unwrap_or(0) + streamed;
                        assert_eq!(
                            shared, expected_shared,
                            "shared incorrect for delta={:?} streamed={} exit={:?}",
                            delta, streamed, exit
                        );
                        assert_eq!(
                            session, *streamed,
                            "session incorrect for delta={:?} streamed={} exit={:?}",
                            delta, streamed, exit
                        );
                    } else {
                        assert_eq!(
                            shared, 0,
                            "shared not zero for delta={:?} streamed={} exit={:?}",
                            delta, streamed, exit
                        );
                        assert_eq!(
                            session, 0,
                            "session not zero for delta={:?} streamed={} exit={:?}",
                            delta, streamed, exit
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_finalize_download_success_after_file_drop() {
        let downloads_dir = std::env::temp_dir().join("test_finalize_success");
        let _ = tokio::fs::create_dir_all(&downloads_dir).await;

        let id = "test-model-success";
        let filename = "model-success.safetensors";
        let file_path = downloads_dir.join(filename);
        let tmp_file_path = downloads_dir.join(format!("{}.tmp", filename));
        let meta_file_path = downloads_dir.join(format!("{}.meta", filename));

        // 1. Create and open the file, explicitly mimicking the stream behavior
        let mut file = tokio::fs::File::create(&tmp_file_path).await.unwrap();
        file.write_all(b"test data").await.unwrap();
        file.sync_all().await.unwrap();

        // 2. EXPLICITLY DROP THE FILE HANDLE.
        // This acts as a regression check ensuring the file lock is released on Windows
        // before we enter the rename logic inside finalize_download.
        drop(file);

        // Create dummy meta file
        let _ = tokio::fs::write(&meta_file_path, "{}").await;

        // 3. Run finalize_download
        super::finalize_download(
            id,
            filename,
            &file_path,
            &tmp_file_path,
            &meta_file_path,
            false,
        )
        .await;

        // 4. Assert success: tmp and meta should be gone, target file should exist
        assert!(
            !tmp_file_path.exists(),
            "Temp file should be deleted upon success."
        );
        assert!(
            !meta_file_path.exists(),
            "Meta file should be deleted upon success."
        );
        assert!(file_path.exists(), "Target file should exist.");
        assert_eq!(
            tokio::fs::read_to_string(&file_path).await.unwrap(),
            "test data"
        );

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&downloads_dir).await;
    }

    #[tokio::test]
    async fn test_etag_mismatch_inner_loop_restarts_and_truncates() {
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
        use axum::Router;

        // 1. Mock server that returns a NEW ETag and content
        let mock_app = Router::new().route(
            "/{*path}",
            axum::routing::get(|req: axum::extract::Request| async move {
                let has_range = req.headers().contains_key(axum::http::header::RANGE);
                let mut res = axum::response::Response::builder().header("ETag", "\"new_etag\""); // Notice this differs from "old_etag" in our seed

                if has_range {
                    res = res.status(axum::http::StatusCode::PARTIAL_CONTENT);
                    res.body(axum::body::Body::from("partial")).unwrap()
                } else {
                    res = res.status(axum::http::StatusCode::OK);
                    res.body(axum::body::Body::from("full new content"))
                        .unwrap()
                }
            })
            .head(|_req: axum::extract::Request| async move {
                axum::response::Response::builder()
                    .header("ETag", "\"new_etag\"")
                    .header(axum::http::header::CONTENT_LENGTH, "16")
                    .body(axum::body::Body::empty())
                    .unwrap()
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, mock_app).await;
        });

        let temp_dir = std::env::temp_dir().join("test_etag_mismatch");
        let _ = tokio::fs::create_dir_all(&temp_dir).await;

        let config = crate::AppConfig {
            hf_base_url: format!("http://127.0.0.1:{}", port),
            downloads_directory: temp_dir.to_string_lossy().to_string(),
            ..Default::default()
        };

        // 2. Seed the directory with an outdated, partial download
        let tmp_file_path = temp_dir.join("model.safetensors.tmp");
        let meta_file_path = temp_dir.join("model.safetensors.meta");
        let final_file_path = temp_dir.join("model.safetensors");

        let old_content = b"old dummy content";
        tokio::fs::write(&tmp_file_path, old_content).await.unwrap();

        let mut hasher = Sha256::new();
        hasher.update(old_content);
        let valid_hasher_hex = super::serialize_hasher(&hasher);

        // Create a metadata file explicitly expecting "old_etag"
        let meta_json = serde_json::json!({
            "expected_hash": "old_etag",
            "is_sha256": false,
            "checkpoints": [{
                "downloaded_bytes": old_content.len(),
                "hasher_state": valid_hasher_hex
            }],
            "hasher_version": super::HASHER_STATE_VERSION
        });
        tokio::fs::write(&meta_file_path, meta_json.to_string())
            .await
            .unwrap();

        // 3. Initialize AppState
        let (queue_tx, _) = tokio::sync::mpsc::channel(1);
        let db = surrealdb::engine::any::connect("mem://").await.unwrap();
        db.use_ns("test").use_db("test").await.unwrap();
        let (_, log_reload_handle) =
            tracing_subscriber::reload::Layer::new(tracing_subscriber::EnvFilter::new("info"));
        let (shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel(16);

        let oauth_client = oauth2::basic::BasicClient::new(
            oauth2::ClientId::new("dummy".to_string()),
            None,
            oauth2::AuthUrl::new("http://localhost".to_string()).unwrap(),
            None,
        );
        let state = Arc::new(crate::AppState {
            queue_tx,
            engine_status: Arc::new(std::sync::Mutex::new(crate::EngineStatus::default())),
            telemetry: Arc::new(std::sync::Mutex::new(crate::TelemetryStore::default())),
            auth_store: Arc::new(std::sync::Mutex::new(crate::auth::AuthStore::default())),
            reqwest_client: reqwest::Client::new(),
            oauth_client,
            config: Arc::new(config),
            log_buffer: crate::SharedLogBuffer(Arc::new(std::sync::Mutex::new((
                0,
                std::collections::VecDeque::new(),
            )))),
            log_reload_handle,
            current_log_level: Arc::new(std::sync::Mutex::new("info".to_string())),
            active_downloads: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            download_tasks: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            download_semaphore: Arc::new(tokio::sync::Semaphore::new(2)),
            db,
            shutdown_tx: shutdown_tx.clone(),
        });

        {
            let mut dl = state.active_downloads.lock().unwrap();
            dl.insert("test-model".to_string(), crate::DownloadStatus::default());
        }

        let (_cancel_tx, cancel_rx) = tokio::sync::broadcast::channel(1);

        // 4. Run the downloader
        super::perform_model_download(
            state,
            "test-model".to_string(),
            "test/repo".to_string(),
            "model.safetensors".to_string(),
            shutdown_rx,
            cancel_rx,
        )
        .await;

        // 5. Verify the ETag mismatch correctly restarted the stream from 0 and completely overwrote the old dummy content
        let downloaded_content = tokio::fs::read_to_string(&final_file_path)
            .await
            .expect("Final file should exist");
        assert_eq!(
            downloaded_content, "full new content",
            "The file should contain ONLY the new content. If it says 'old dummy contentpartial', the file was not properly truncated!"
        );

        // Verify cleanup
        assert!(!tmp_file_path.exists());
        assert!(!meta_file_path.exists());
        let _ = tokio::fs::remove_dir_all(temp_dir).await;
    }

    #[tokio::test]
    async fn test_metadata_fetch_ignores_error_body_size() {
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
        use axum::Router;
        use tokio_stream::wrappers::ReceiverStream;

        // 1. Mock server that returns a 404 error page for the second chunk
        let mock_app = Router::new().route(
            "/{*path}",
            axum::routing::get(|req: axum::extract::Request| async move {
                let uri = req.uri().to_string();
                if uri.contains("00001") {
                    let (tx, rx) = tokio::sync::mpsc::channel(1);
                    tokio::spawn(async move {
                        let _ = tx
                            .send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                                vec![0u8; 500],
                            )))
                            .await;
                        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
                        let _ = tx
                            .send(Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                                vec![0u8; 500],
                            )))
                            .await;
                        // Keep the stream open a bit longer so the test polling loop can intercept the active state
                        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
                    });
                    axum::response::Response::builder()
                        .header(axum::http::header::CONTENT_LENGTH, "1000")
                        .body(axum::body::Body::from_stream(ReceiverStream::new(rx)))
                        .unwrap()
                } else {
                    axum::response::Response::builder()
                        .status(axum::http::StatusCode::NOT_FOUND)
                        .header(axum::http::header::CONTENT_LENGTH, "500") // Error page size
                        .body(axum::body::Body::from("404 Not Found error page body..."))
                        .unwrap()
                }
            })
            .head(|req: axum::extract::Request| async move {
                let uri = req.uri().to_string();
                if uri.contains("00001") {
                    axum::response::Response::builder()
                        .header(axum::http::header::CONTENT_LENGTH, "1000")
                        .body(axum::body::Body::empty())
                        .unwrap()
                } else {
                    axum::response::Response::builder()
                        .status(axum::http::StatusCode::NOT_FOUND)
                        .header(axum::http::header::CONTENT_LENGTH, "500") // Error page size
                        .body(axum::body::Body::empty())
                        .unwrap()
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, mock_app).await;
        });

        let temp_dir = std::env::temp_dir().join("test_metadata_error_size");
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
        let _ = tokio::fs::create_dir_all(&temp_dir).await;

        let config = crate::AppConfig {
            hf_base_url: format!("http://127.0.0.1:{}", port),
            downloads_directory: temp_dir.to_string_lossy().to_string(),
            ..Default::default()
        };

        // 2. Setup isolated AppState
        let (queue_tx, _) = tokio::sync::mpsc::channel(1);
        let db = surrealdb::engine::any::connect("mem://").await.unwrap();
        db.use_ns("test").use_db("test").await.unwrap();
        let (_, log_reload_handle) =
            tracing_subscriber::reload::Layer::new(tracing_subscriber::EnvFilter::new("info"));
        let (shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel(16);

        let state = Arc::new(crate::AppState {
            queue_tx,
            engine_status: Arc::new(std::sync::Mutex::new(crate::EngineStatus::default())),
            telemetry: Arc::new(std::sync::Mutex::new(crate::TelemetryStore::default())),
            auth_store: Arc::new(std::sync::Mutex::new(crate::auth::AuthStore::default())),
            reqwest_client: reqwest::Client::new(),
            oauth_client: oauth2::basic::BasicClient::new(
                oauth2::ClientId::new("dummy".to_string()),
                None,
                oauth2::AuthUrl::new("http://localhost".to_string()).unwrap(),
                None,
            ),
            config: Arc::new(config),
            log_buffer: crate::SharedLogBuffer(Arc::new(std::sync::Mutex::new((
                0,
                std::collections::VecDeque::new(),
            )))),
            log_reload_handle,
            current_log_level: Arc::new(std::sync::Mutex::new("info".to_string())),
            active_downloads: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            download_tasks: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            download_semaphore: Arc::new(tokio::sync::Semaphore::new(2)),
            db,
            shutdown_tx,
        });

        {
            let mut dl = state.active_downloads.lock().unwrap();
            dl.insert("test-model".to_string(), crate::DownloadStatus::default());
        }

        let (_cancel_tx, cancel_rx) = tokio::sync::broadcast::channel(1);

        let state_clone = state.clone();
        let task = tokio::spawn(async move {
            super::perform_model_download(
                state_clone,
                "test-model".to_string(),
                "test/repo".to_string(),
                "model-00001-of-00002.safetensors".to_string(), // Triggers split file logic
                shutdown_rx,
                cancel_rx,
            )
            .await;
        });

        // 3. Poll until bytes_transferred > 0 to intercept the active state
        let mut total_bytes_observed = 0;
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let active = {
                let dl = state.active_downloads.lock().unwrap();
                dl.get("test-model").cloned()
            };
            if let Some(dl) = active
                && dl.total_bytes > 0
            {
                total_bytes_observed = dl.total_bytes;
                break;
            }
        }

        let _ = task.await;

        // 4. Assert that the total bytes size is exactly the size of the valid 1st chunk, completely ignoring the 404 page's 500-byte length
        assert_eq!(
            total_bytes_observed, 1000,
            "Total bytes should be exactly 1000, ignoring the 500 byte error page body."
        );

        let _ = tokio::fs::remove_dir_all(temp_dir).await;
    }

    #[tokio::test]
    async fn test_initiate_request_timeout() {
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
        use axum::Router;

        // Mock server that hangs (accepts connection but doesn't send headers)
        let mock_app = Router::new().route(
            "/{*path}",
            axum::routing::get(|_req: axum::extract::Request| async move {
                // Sleep longer than the timeout to trigger it
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                axum::response::Response::builder()
                    .status(axum::http::StatusCode::OK)
                    .body(axum::body::Body::from("too late"))
                    .unwrap()
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, mock_app).await;
        });

        // Use a short timeout
        let config = crate::AppConfig {
            download_stream_chunk_timeout_seconds: 1, // 1 second timeout
            download_retry_max_attempts: 1,           // Retry once to keep the test fast
            download_retry_backoff_seconds: 1,
            ..Default::default()
        };

        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/test", port);
        let id = "test_timeout";
        let mut existing_size = 0;
        let active_streams = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (_shutdown_tx, mut shutdown_rx) = tokio::sync::broadcast::channel(1);
        let (_cancel_tx, mut cancel_rx) = tokio::sync::broadcast::channel(1);

        let result = super::initiate_request(
            &client,
            &url,
            id,
            &mut existing_size,
            &active_streams,
            &mut shutdown_rx,
            &mut cancel_rx,
            None,
            "http://127.0.0.1",
            &config,
        )
        .await;

        assert!(
            result.is_none(),
            "Request should have timed out and returned None after exhausting retries."
        );
    }

    #[tokio::test]
    async fn test_downloader_ignores_zero_byte_final_files() {
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
        use axum::Router;

        // 1. Mock server that returns valid content
        let mock_app = Router::new().route(
            "/{*path}",
            axum::routing::get(|_req: axum::extract::Request| async move {
                axum::response::Response::builder()
                    .status(axum::http::StatusCode::OK)
                    .body(axum::body::Body::from("actual file content"))
                    .unwrap()
            })
            .head(|_req: axum::extract::Request| async move {
                axum::response::Response::builder()
                    .header(axum::http::header::CONTENT_LENGTH, "19")
                    .body(axum::body::Body::empty())
                    .unwrap()
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, mock_app).await;
        });

        let temp_dir = std::env::temp_dir().join("test_zero_byte_files");
        let _ = tokio::fs::create_dir_all(&temp_dir).await;

        let config = crate::AppConfig {
            hf_base_url: format!("http://127.0.0.1:{}", port),
            downloads_directory: temp_dir.to_string_lossy().to_string(),
            ..Default::default()
        };

        // 2. Pre-create the 0-byte final file to simulate an aborted setup / touched file
        let final_file_path = temp_dir.join("model.safetensors");
        tokio::fs::write(&final_file_path, "").await.unwrap();

        // 3. Initialize AppState
        let (queue_tx, _) = tokio::sync::mpsc::channel(1);
        let db = surrealdb::engine::any::connect("mem://").await.unwrap();
        db.use_ns("test").use_db("test").await.unwrap();
        let (_, log_reload_handle) =
            tracing_subscriber::reload::Layer::new(tracing_subscriber::EnvFilter::new("info"));
        let (shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel(16);

        let oauth_client = oauth2::basic::BasicClient::new(
            oauth2::ClientId::new("dummy".to_string()),
            None,
            oauth2::AuthUrl::new("http://localhost".to_string()).unwrap(),
            None,
        );

        let state = Arc::new(crate::AppState {
            queue_tx,
            engine_status: Arc::new(std::sync::Mutex::new(crate::EngineStatus::default())),
            telemetry: Arc::new(std::sync::Mutex::new(crate::TelemetryStore::default())),
            auth_store: Arc::new(std::sync::Mutex::new(crate::auth::AuthStore::default())),
            reqwest_client: reqwest::Client::new(),
            oauth_client,
            config: Arc::new(config),
            log_buffer: crate::SharedLogBuffer(Arc::new(std::sync::Mutex::new((
                0,
                std::collections::VecDeque::new(),
            )))),
            log_reload_handle,
            current_log_level: Arc::new(std::sync::Mutex::new("info".to_string())),
            active_downloads: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            download_tasks: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            download_semaphore: Arc::new(tokio::sync::Semaphore::new(2)),
            db,
            shutdown_tx,
        });

        {
            let mut dl = state.active_downloads.lock().unwrap();
            dl.insert(
                "test-model-zero-byte".to_string(),
                crate::DownloadStatus::default(),
            );
        }

        let (_cancel_tx, cancel_rx) = tokio::sync::broadcast::channel(1);

        // 4. Run the downloader
        super::perform_model_download(
            state,
            "test-model-zero-byte".to_string(),
            "test/repo".to_string(),
            "model.safetensors".to_string(),
            shutdown_rx,
            cancel_rx,
        )
        .await;

        // 5. Verify the file was actually downloaded and the 0-byte file was successfully overwritten
        let downloaded_content = tokio::fs::read_to_string(&final_file_path)
            .await
            .expect("Final file should exist");

        assert_eq!(
            downloaded_content, "actual file content",
            "The 0-byte file should have been ignored and overwritten by the actual download."
        );

        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }
}
