// Called by: chat.js, settings.js, memory.js, stats.js, models.js, console.js (internal fetch wrapper)
async function fetchWithAuth(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
        if (response.status === 401) {
            window.location.href = '/auth/login';
            throw new Error('Unauthorized'); // Stop further execution in the caller
        }
        throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
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
 * Shared utility to handle model downloads with progress polling and cancellation.
 * @param {string} modelId - The ID of the model to download.
 * @param {Object} callbacks - Functions to handle UI updates.
 * @param {Function} [callbacks.onProgress] - Called with (status, pct, speedMB, transMB, totalMB, etaStr).
 * @param {Function} [callbacks.onStatusText] - Called with string status updates.
 * @param {Function} [callbacks.onComplete] - Called when the download successfully completes.
 * @returns {Object} An object containing a `promise` that resolves when complete, and a `cancel` method.
 */
/* eslint-disable-next-line no-unused-vars -- Called by: chat.js and models.js */
function downloadModel(modelId, callbacks = {}) {
    let isCanceled = false;

    const cancel = async () => {
        isCanceled = true;
        try {
            await fetchWithAuth(`/api/models/${modelId}/download`, { method: 'DELETE' });
        } catch (e) {
            console.error("Failed to send cancel request to server:", e);
        }
    };

    async function sleep(ms) {
        for (let i = 0; i < ms; i += 100) {
            if (isCanceled) return;
            await new Promise(r => setTimeout(r, 100));
        }
    }

    const promise = (async () => {
        while (!isCanceled) {
            try {
                const progCheck = await fetchWithAuth('/api/models/download/progress');
                const activeDls = await progCheck.json();
                
                if (!activeDls[modelId]) {
                    try {
                        await fetchWithAuth(`/api/models/${modelId}/download`, { method: 'POST' });
                    } catch (e) {
                        if (e.message === 'Unauthorized') return;
                        if (!e.message.includes('409')) {
                            if (callbacks.onStatusText) callbacks.onStatusText(`Failed to start. Retrying in 5s...`);
                            await sleep(5000);
                            continue;
                        }
                    }
                }

                await new Promise((resolve, reject) => {
                    const interval = setInterval(async () => {
                        if (isCanceled) {
                            clearInterval(interval);
                            reject(new Error("Canceled"));
                            return;
                        }
                        
                        try {
                            const progRes = await fetchWithAuth('/api/models/download/progress');
                            const downloads = await progRes.json();
                            const status = downloads[modelId];
                            
                            if (status) {
                                const pct = status.total_bytes > 0 ? (status.bytes_transferred / status.total_bytes) * 100 : 0;
                                const speedMB = (status.current_speed_bps / 1024 / 1024).toFixed(1);
                                const transMB = (status.bytes_transferred / 1024 / 1024).toFixed(1);
                                const totalMB = (status.total_bytes / 1024 / 1024).toFixed(1);
                                
                                let etaStr = "Calculating...";
                                if (status.current_speed_bps > 0 && status.total_bytes > 0) {
                                    const bytesLeft = status.total_bytes - status.bytes_transferred;
                                    const secsLeft = bytesLeft / status.current_speed_bps;
                                    etaStr = secsLeft > 60 ? `${Math.floor(secsLeft/60)}m ${Math.round(secsLeft%60)}s` : `${Math.round(secsLeft)}s`;
                                }
                                
                                if (callbacks.onProgress) {
                                    callbacks.onProgress(status, pct, speedMB, transMB, totalMB, etaStr);
                                }
                            } else {
                                clearInterval(interval);
                                
                                const verifyRes = await fetchWithAuth('/api/models');
                                const verifyModels = await verifyRes.json();
                                const verifyModel = verifyModels.find(m => m.id === modelId);
                                
                                if (verifyModel && verifyModel.is_downloaded) {
                                    if (callbacks.onStatusText) callbacks.onStatusText('Download Complete!');
                                    if (callbacks.onComplete) callbacks.onComplete();
                                    setTimeout(() => resolve(), 500);
                                } else {
                                    reject(new Error("Interrupted"));
                                }
                            }
                        } catch (e) {
                            clearInterval(interval);
                            reject(e);
                        }
                    }, 1000);
                });

                return; // Download succeeded!

            } catch (e) {
                if (isCanceled) {
                    if (callbacks.onStatusText) callbacks.onStatusText('Download Canceled.');
                    throw new Error("Download canceled by user.");
                }
                if (callbacks.onStatusText) callbacks.onStatusText('Download Interrupted. Retrying in 5s...');
                console.error(`Download ${modelId} interrupted, retrying in 5s...`, e);
                await sleep(5000);
            }
        }
        
        if (callbacks.onStatusText) callbacks.onStatusText('Download Canceled.');
        throw new Error("Download canceled by user.");
    })();

    return { promise, cancel };
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