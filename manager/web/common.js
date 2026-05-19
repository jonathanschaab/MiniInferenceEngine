// Called by: chat.js, settings.js, memory.js, stats.js, models.js, console.js (internal fetch wrapper)
async function fetchWithAuth(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
        if (response.status === 401) {
            window.location.href = '/auth/login';
            const err = new Error('Unauthorized');
            err.status = 401;
            throw err; // Stop further execution in the caller
        }
        const err = new Error(`HTTP error ${response.status}: ${response.statusText}`);
        err.status = response.status;
        throw err;
    }
    return response;
}

/* eslint-disable-next-line no-unused-vars -- Called by: stats.js submitBenchmark() and stats.html generation parameter UI */
function getGenerationParameters() {
    const params = {
        temperature: parseFloat(document.getElementById('param-temp').value),
        top_p: parseFloat(document.getElementById('param-top-p').value),
        top_k: parseInt(document.getElementById('param-top-k').value),
        max_tokens: parseInt(document.getElementById('param-max-tokens').value),
        context_buffer: parseInt(document.getElementById('param-context-buffer').value) || 0
    };
    const seedVal = document.getElementById('param-seed').value;
    params.seed = seedVal !== "" ? parseInt(seedVal) : null;
    const memStrategyEl = document.getElementById('param-memory-strategy');
    if (memStrategyEl) params.memory_strategy = memStrategyEl.value;
    const yarnEnabledEl = document.getElementById('param-yarn-enabled');
    if (yarnEnabledEl) params.yarn_enabled = yarnEnabledEl.checked;
    return params;
}

/**
 * Shared download progress poller to prevent redundant API calls
 * when multiple models are downloading or multiple pages are tracking.
 */
const SharedDownloadProgress = {
    lastFetchTime: 0,
    cache: null,
    fetchPromise: null,
    async get() {
        const now = Date.now();
        if (this.fetchPromise) return this.fetchPromise;
        if (now - this.lastFetchTime < 500 && this.cache) {
            return this.cache;
        }
        this.fetchPromise = (async () => {
            try {
                const res = await fetchWithAuth('/api/downloads');
                this.cache = await res.json();
                this.lastFetchTime = Date.now();
                return this.cache;
            } finally {
                this.fetchPromise = null;
            }
        })();
        return this.fetchPromise;
    },
    clearCache() {
        this.cache = null;
        this.lastFetchTime = 0;
    }
};

/**
 * Shared utility to handle model downloads with progress polling and cancellation.
 * @param {string} modelId - The ID of the model to download.
 * @param {Object} callbacks - Functions to handle UI updates.
 * @param {Function} [callbacks.onProgress] - Called with (status, pct, speedMB, transMB, totalMB, etaStr).
 * @param {Function} [callbacks.onStatusText] - Called with string status updates.
 * @param {Function} [callbacks.onComplete] - Called when the download successfully completes.
 * @returns {Object} An object containing a `promise` that resolves when complete, and `cancel`/`pause` methods.
 */
/* eslint-disable-next-line no-unused-vars -- Called by: chat.js and models.js */
function downloadModel(modelId, callbacks = {}) {
    let isStopped = false;
    let stopReason = null;
    let retryCount = 0;
    const MAX_RETRIES = 5;

    const cancel = async () => {
        isStopped = true;
        stopReason = "Canceled";
        try {
            await fetchWithAuth(`/api/downloads/${modelId}`, { method: 'DELETE' });
            SharedDownloadProgress.clearCache();
        } catch (e) {
            console.error("Failed to send cancel request to server:", e);
        }
    };

    const pause = async () => {
        isStopped = true;
        stopReason = "Paused";
        try {
            await fetchWithAuth(`/api/downloads/${modelId}/pause`, { method: 'POST' });
            SharedDownloadProgress.clearCache();
        } catch (e) {
            console.error("Failed to send pause request to server:", e);
        }
    };

    async function sleep(ms) {
        for (let i = 0; i < ms; i += 100) {
            if (isStopped) return;
            await new Promise(r => setTimeout(r, 100));
        }
    }

    const promise = (async () => {
        while (!isStopped) {
            try {
                const activeDls = await SharedDownloadProgress.get();
                
                if (!activeDls[modelId]) {
                    try {
                        await fetchWithAuth('/api/downloads', { 
                            method: 'POST', 
                            headers: { 'Content-Type': 'application/json' }, 
                            body: JSON.stringify({ model_id: modelId }) 
                        });
                        SharedDownloadProgress.clearCache();
                        retryCount = 0;
                    } catch (e) {
                        if (e.status === 401 || e.message === 'Unauthorized') throw e;
                        if (e.status !== 409) {
                            // Immediately abort on permanent 4xx errors (excluding 409 Conflict and 429 Too Many Requests)
                            const isPermanent = e.status >= 400 && e.status < 500 && e.status !== 429;
                            if (isPermanent) {
                                if (callbacks.onStatusText) callbacks.onStatusText('Download Failed (Permanent Error).');
                                throw e;
                            }
                            retryCount++;
                            if (retryCount > MAX_RETRIES) {
                                if (callbacks.onStatusText) callbacks.onStatusText('Download Failed (Max Retries).');
                                throw new Error(`Max retries reached: ${e.message}`);
                            }
                            if (callbacks.onStatusText) callbacks.onStatusText(`Failed to start. Retrying in 5s... (${retryCount}/${MAX_RETRIES})`);
                            await sleep(5000);
                            continue;
                        }
                    }
                }

                while (!isStopped) {
                    const downloads = await SharedDownloadProgress.get();
                    const status = downloads[modelId];
                    
                    if (status) {
                        retryCount = 0;
                        const pct = status.total_bytes > 0 ? (status.bytes_transferred / status.total_bytes) * 100 : 0;
                        const speedMB = (status.current_speed_bps / 1024 / 1024).toFixed(1);
                        const transMB = (status.bytes_transferred / 1024 / 1024).toFixed(1);
                        const totalMB = (status.total_bytes / 1024 / 1024).toFixed(1);
                        
                        let etaStr = "Calculating...";
                        if (status.current_speed_bps > 0 && status.total_bytes > 0) {
                            const bytesLeft = status.total_bytes - status.bytes_transferred;
                            const secsLeft = bytesLeft / status.current_speed_bps;
                            if (secsLeft >= 3600) {
                                etaStr = `${Math.floor(secsLeft / 3600)}h ${Math.floor((secsLeft % 3600) / 60)}m ${Math.round(secsLeft % 60)}s`;
                            } else if (secsLeft >= 60) {
                                etaStr = `${Math.floor(secsLeft / 60)}m ${Math.round(secsLeft % 60)}s`;
                            } else {
                                etaStr = `${Math.round(secsLeft)}s`;
                            }
                        }
                        
                        if (callbacks.onProgress) {
                            callbacks.onProgress(status, pct, speedMB, transMB, totalMB, etaStr);
                        }
                        
                        await sleep(1000);
                    } else {
                        const verifyRes = await fetchWithAuth(`/api/models/${modelId}`);
                        const verifyModel = await verifyRes.json();
                        
                        if (verifyModel && verifyModel.is_downloaded) {
                            if (callbacks.onStatusText) callbacks.onStatusText('Download Complete!');
                            if (callbacks.onComplete) callbacks.onComplete();
                            await sleep(500);
                            return; // Download succeeded!
                        } else {
                            throw new Error("Interrupted");
                        }
                    }
                }

                if (isStopped) {
                    break;
                }

            } catch (e) {
                if (isStopped) {
                    break;
                }
                if (e.status === 401 || e.message === 'Unauthorized') throw e;
                
                const isPermanent = e.status >= 400 && e.status < 500 && e.status !== 409 && e.status !== 429;
                if (isPermanent) {
                    if (callbacks.onStatusText) callbacks.onStatusText('Download Failed (Permanent Error).');
                    throw e;
                }
                retryCount++;
                if (retryCount > MAX_RETRIES) {
                    if (callbacks.onStatusText) callbacks.onStatusText('Download Failed (Max Retries).');
                    console.error(`Download ${modelId} failed after max retries:`, e);
                    throw new Error(`Max retries reached: ${e.message}`);
                }
                if (callbacks.onStatusText) callbacks.onStatusText(`Download Interrupted. Retrying in 5s... (${retryCount}/${MAX_RETRIES})`);
                console.error(`Download ${modelId} interrupted, retrying in 5s... (${retryCount}/${MAX_RETRIES})`, e);
                await sleep(5000);
            }
        }
        
        if (stopReason === "Paused") {
            if (callbacks.onStatusText) callbacks.onStatusText('Download Paused.');
            throw new Error("Download paused by user.");
        } else {
            if (callbacks.onStatusText) callbacks.onStatusText('Download Canceled.');
            throw new Error("Download canceled by user.");
        }
    })();

    return { promise, cancel, pause };
}

/**
 * Dynamically injects the main navigation bar into the header and 
 * securely checks if the user is an admin to display the Console button.
 */
function injectNavbar() {
    const header = document.querySelector('header');
    if (!header) return;

    const navDiv = document.createElement('div');
    navDiv.style.display = 'flex';
    navDiv.style.gap = '15px';
    navDiv.style.alignItems = 'center';

    navDiv.innerHTML = `
        <button onclick="window.location.href='/'" style="background: #89b4fa; height: 30px; padding: 0 10px;">💬 Chat</button>
        <button onclick="window.location.href='/models'" style="background: #fab387; height: 30px; padding: 0 10px;">🤖 Models</button>
        <button onclick="window.location.href='/queue'" style="background: #94e2d5; height: 30px; padding: 0 10px;">⬇️ Queue</button>
        <button onclick="window.location.href='/memory'" style="background: #f9e2af; height: 30px; padding: 0 10px;">💾 Memory</button>
        <button onclick="window.location.href='/stats'" style="background: #cba6f7; height: 30px; padding: 0 10px;">📊 Stats</button>
        <button id="nav-console-btn" onclick="window.location.href='/console'" style="background: #89dceb; height: 30px; padding: 0 10px; display: none;">🖥️ Console</button>
        <button onclick="window.location.href='/settings'" style="background: #a6e3a1; height: 30px; padding: 0 10px;">⚙️ Settings</button>
        <button onclick="window.location.href='/auth/logout'" style="background: #45475a; color: white; height: 30px; padding: 0 10px;">Logout</button>
    `;

    header.appendChild(navDiv);

    fetch('/api/console/loglevel')
        .then(res => { if (res.ok) document.getElementById('nav-console-btn').style.display = 'inline-flex'; })
        .catch(e => console.debug("Console authorization check failed", e));
}

if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', injectNavbar); } else { injectNavbar(); }