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

        if let Ok(meta) = tokio::fs::metadata(&file_path).await {
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

    let (mut existing_size, mut expected_hash, mut is_sha256, mut initial_hasher, mut checkpoints) =
        check_existing_metadata(&state, &id, &tmp_file_path, &meta_file_path, &remote_sha256).await;

    let original_existing_size = existing_size;
    shared_downloaded.fetch_add(existing_size, std::sync::atomic::Ordering::Relaxed);

    let (mut res, final_url) = match initiate_request(
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
        None => return false,
    };

    if verify_and_update_etag(
        &client,
        &final_url,
        &id,
        &mut res,
        &mut existing_size,
        &mut expected_hash,
        &mut is_sha256,
        hf_token.as_deref(),
        &hf_base_url,
    )
    .await
    .is_err()
    {
        return false;
    }

    let is_partial = res.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    if !is_partial && existing_size > 0 {
        existing_size = 0;
    }
    if existing_size < original_existing_size {
        shared_downloaded.fetch_sub(
            original_existing_size - existing_size,
            std::sync::atomic::Ordering::Relaxed,
        );
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

    let mut file = match open_temp_file(&tmp_file_path, is_partial, existing_size).await {
        Ok(f) => f,
        Err(e) => {
            error!("Failed to open file for {}: {}", id, e);
            return false;
        }
    };

    let (stream_error, _) = process_download_stream(
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
        shared_downloaded,
        grand_total,
        start_time,
        session_downloaded,
        active_streams,
    )
    .await;

    if stream_error {
        return false;
    }

    drop(file);

    let final_hash = hex::encode(initial_hasher.finalize());
    let hash_mismatch = if let Some(ref expected) = expected_hash {
        is_sha256
            && expected.len() == 64
            && expected.chars().all(|c| c.is_ascii_hexdigit())
            && &final_hash != expected
    } else {
        false
    };

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

    hash_mismatch
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
                                    error!("Failed to truncate {} to checkpoint size {}: {}", id, existing_size, e);
                                    return (0, expected_hash, is_sha256, Sha256::new(), Vec::new());
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

        match req.send().await {
            Ok(r) => {
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
            Err(e) => {
                retries += 1;
                if retries > config.download_retry_max_attempts {
                    error!("Max retries reached for network error on {}: {}. Aborting.", id, e);
                    return None;
                }
                warn!(
                    "Network error for {}: {}. Retrying ({}/{}) in {} seconds...",
                    id, e, retries, config.download_retry_max_attempts, backoff
                );
                if let Some(reason) = wait_for_backoff(backoff, shutdown_rx, cancel_rx).await {
                    info!("{} signal received during network error backoff for {}. Aborting.", reason, id);
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

#[allow(clippy::too_many_arguments)]
async fn verify_and_update_etag(
    client: &reqwest::Client,
    url: &str, // This is now the final URL
    id: &str,
    res: &mut reqwest::Response,
    existing_size: &mut u64,
    expected_hash: &mut Option<String>,
    is_sha256: &mut bool,
    hf_token: Option<&str>,
    hf_base_url: &str,
) -> Result<(), ()> {
    info!("Inspecting HTTP Headers for {}:", id);
    for (k, v) in res.headers() {
        info!("  {}: {:?}", k.as_str(), v.to_str().unwrap_or("<binary>"));
    }

    if *is_sha256 {
        info!(
            "  Already tracking a verified upstream SHA-256 for {}. Ignoring CDN ETags.",
            id
        );
        return Ok(());
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

        let mut req = client.get(url);
        if url.starts_with(hf_base_url.trim_end_matches('/'))
            && let Some(token) = hf_token
        {
            req = req.bearer_auth(token);
        }
        if let Ok(r) = req.send().await {
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
        *is_sha256 = x_linked_found;
    }
    Ok(())
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
    let _ = tokio::fs::write(
        meta_file_path,
        serde_json::to_string(&meta).unwrap_or_default(),
    )
    .await;
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
) -> (bool, u64) {
    let _active_guard = ActiveStreamGuard::new(active_streams);

    let mut downloaded: u64 = existing_size;
    let mut stream_error = false;
    let mut last_meta_save = std::time::Instant::now();
    let mut last_ui_update = std::time::Instant::now();
    let mut bytes_since_last_ui_update = 0;

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
            _ = shutdown_rx.recv() => {
                info!("Shutdown signal received. Flushing metadata for {}...", id);
                save_download_checkpoint(hasher, downloaded, &mut checkpoints, meta_file_path, expected_hash, is_sha256).await;
                return (true, downloaded);
            }
            _ = cancel_rx.recv() => {
                info!("Cancel signal received. Flushing metadata for {}...", id);
                save_download_checkpoint(hasher, downloaded, &mut checkpoints, meta_file_path, expected_hash, is_sha256).await;
                return (true, downloaded);
            }
        }
    }

    (stream_error, downloaded)
}

async fn finalize_download(
    id: &str,
    filename: &str,
    file_path: &Path,
    tmp_file_path: &Path,
    meta_file_path: &Path,
    hash_mismatch: bool,
) {
    let target_path = if hash_mismatch {
        file_path.with_file_name(format!(
            "{}.corrupted",
            file_path.file_name().unwrap_or_default().to_string_lossy()
        ))
    } else {
        file_path.to_path_buf()
    };

    let mut success = false;
    if let Err(e) = tokio::fs::rename(tmp_file_path, &target_path).await {
        warn!(
            "Failed to rename temp file for {} ({}). Falling back to copy...",
            id, e
        );

        let copy_tmp_path = target_path.with_file_name(format!(
            "{}.copy_tmp",
            target_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        ));

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
}
