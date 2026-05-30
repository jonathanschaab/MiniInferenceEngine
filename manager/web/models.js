let isAdmin = false;

async function checkAdmin() {
    try {
        const res = await fetchWithAuth('/api/console/loglevel');
        if (res.ok) isAdmin = true;
    } catch (e) {
        console.debug("Could not verify admin status, assuming non-admin.", e);
        isAdmin = false;
    }
}

async function loadModels() {
    try {
        const [modelsRes, statusRes] = await Promise.all([
            fetchWithAuth('/api/models'),
            fetchWithAuth('/api/status')
        ]);
        const models = await modelsRes.json();
        const engineStatus = statusRes.ok ? await statusRes.json() : { model_health: {} };
        const healthMap = engineStatus.model_health || {};
        const container = document.getElementById('models-container');
        container.innerHTML = '';

        models.forEach(model => {
            const getBadge = (source) => {
                if (source === 'override') return '<span class="badge badge-override">Override</span>';
                if (source === 'config.json') return '<span class="badge badge-json">config.json</span>';
                if (source === 'fallback') return '<span class="badge badge-fallback">Fallback</span>';
                if (source === 'disk') return '<span class="badge badge-disk">Disk Search</span>';
                return `<span class="badge bg-surface1">${source || 'Unknown'}</span>`;
            };

            const card = document.createElement('div');
            card.className = `model-dir-card ${model.is_downloaded ? '' : 'model-undownloaded'}`;
            card.id = `model-card-${model.id}`;

            const rolesStr = model.roles.join(', ');
            const backendsStr = model.supported_backends.join(', ');

            let healthBadge = '';
            if (model.id === engineStatus.loading_model_id) {
                healthBadge = `<span class="badge bg-yellow text-base ml-10" title="Currently loading into VRAM">⏳ Loading</span>`;
            } else if (healthMap[model.id] === true) {
                healthBadge = `<span class="badge bg-green text-base ml-10" title="Last run succeeded">✅ Passed</span>`;
            } else if (healthMap[model.id] === false) {
                healthBadge = `<span class="badge bg-red text-base ml-10" title="Last run failed">❌ Failed</span>`;
            }

            const corruptedBadge = model.is_corrupted ? `<span class="badge bg-red text-base ml-10" title="A corrupted download file was detected. Please delete to clear it.">⚠️ Corrupted</span>` : '';

            let adminDeleteBtn = '';
            if (isAdmin) {
                adminDeleteBtn = `<button class="btn-delete btn-cancel p-5-10 text-085">🗑️ Delete</button>`;
            }

            const cacheBadge = model.is_in_hf_cache ? `<span class="badge bg-mauve text-base" title="This model is also present in the global Hugging Face cache.">HF Cache</span>` : '';

            const downloadBtnHtml = model.is_downloaded
                ? `<div class="flex-gap-8-center">${adminDeleteBtn}<span class="badge badge-json p-5-10">Ready</span>${cacheBadge}</div>`
                : `<div class="flex-gap-8-center">${adminDeleteBtn}<button class="btn-download">Download Model</button></div>`;

            const moeHtml = (model.num_local_experts != null && model.num_experts_per_tok != null)
                ? `<div class="setting-item setting-moe"><span>MoE Routing</span> <span>${model.num_experts_per_tok} / ${model.num_local_experts} Active ${getBadge(model.provenance.num_local_experts)}</span></div>`
                : '';

            const cardHtml = `
                <div class="model-header">
                    <div>
                        <h2 class="model-title ${model.is_downloaded ? 'text-peach' : 'text-overlay0'}">${model.name} ${healthBadge}${corruptedBadge}</h2>
                        <p class="model-subtitle">ID: ${model.id} | Repo: <a href="https://huggingface.co/${model.repo}" target="_blank" class="text-blue">${model.repo}</a></p>
                    </div>
                    <div class="model-actions-col">
                        ${downloadBtnHtml}
                        <div class="text-085 text-mauve mb-4"><strong>Roles:</strong> ${rolesStr}</div>
                        <div class="text-085 text-green"><strong>Backends:</strong> ${backendsStr}</div>
                    </div>
                </div>
                <div class="model-settings">
                    <div class="setting-item"><span>Architecture</span> <span>${model.arch} ${getBadge(model.provenance.arch)}</span></div>
                    ${moeHtml}
                    <div class="setting-item"><span>KV Cache DType</span> <span>${model.kv_cache_dtype} ${getBadge(model.provenance.kv_cache_dtype)}</span></div>
                    <div class="setting-item"><span>Max Context Len</span> <span>${model.max_context_len} ${getBadge(model.provenance.max_context_len)}</span></div>
                    <div class="setting-item"><span>Sliding Window</span> <span>${model.sliding_window || 'None'} ${getBadge(model.provenance.sliding_window)}</span></div>
                    <div class="setting-item"><span>Num Layers</span> <span>${model.num_layers} ${getBadge(model.provenance.num_layers)}</span></div>
                    <div class="setting-item"><span>Embed Dim (n_embd)</span> <span>${model.n_embd} ${getBadge(model.provenance.n_embd)}</span></div>
                    <div class="setting-item"><span>Attention Heads (n_head)</span> <span>${model.n_head} ${getBadge(model.provenance.n_head)}</span></div>
                    <div class="setting-item"><span>KV Heads (n_head_kv)</span> <span>${model.n_head_kv} ${getBadge(model.provenance.n_head_kv)}</span></div>
                    <div class="setting-item"><span>Head Dim</span> <span>${model.head_dim} ${getBadge(model.provenance.head_dim)}</span></div>
                    <div class="setting-item"><span>Size on Disk</span> <span>${model.size_on_disk_gb.toFixed(2)} GB ${getBadge(model.provenance.size_on_disk_gb)}</span></div>
                </div>
            `;
            card.innerHTML = DOMPurify.sanitize(cardHtml, { ADD_ATTR: ['target'] });

            const downloadBtn = card.querySelector('.btn-download');
            if (downloadBtn) {
                downloadBtn.addEventListener('click', () => startDownload(model.id));
            }

            const deleteBtn = card.querySelector('.btn-delete');
            if (deleteBtn) {
                deleteBtn.addEventListener('click', () => deleteModel(model.id));
            }

            container.appendChild(card);
        });
    } catch (e) {
        if (e.name === 'TypeError' && e.message === 'Failed to fetch') return; // Ignore browser teardown aborts
        console.error('Failed to load models directory:', e);
    }
}

const activeDownloads = new Map();

/**
 * Discovers downloads that might be active on the server (e.g., from a previous session or another tab)
 * and initializes the UI and polling for them on the current page. This runs periodically.
 */
async function discoverActiveDownloads() {
    try {
        const downloads = await SharedDownloadProgress.get();
        
        const activeIds = Object.keys(downloads);

        activeIds.forEach(id => {
            // If a download is active on the server but not tracked on this page, start tracking it.
            if (!activeDownloads.has(id)) {
                const card = document.getElementById(`model-card-${id}`);
                if (card) { // Check if the model card is rendered on the page
                    // Prevent duplicate progress containers if one already exists
                    const existingProgress = card.querySelector('.download-progress-container');
                    if (existingProgress) {
                        existingProgress.remove();
                    }
                    const dlBtn = card.querySelector('.btn-download');
                    if (dlBtn) dlBtn.style.display = 'none';
                    startDownload(id); // This will create the UI and start the poller via downloadModel
                }
            }
        });
    } catch (e) {
        if (e.name === 'TypeError' && e.message === 'Failed to fetch') return; // Ignore browser teardown aborts
        console.error("Failed to discover active downloads:", e);
    }
}

async function startDownload(modelId) {
    if (activeDownloads.has(modelId)) return;
    
    let card = document.getElementById(`model-card-${modelId}`);
    if (!card) return;
    
    const rightCol = card.querySelector('.model-header > div:nth-child(2)');
    if (rightCol) {
        const btn = rightCol.querySelector('.btn-download');
        if (btn) btn.style.display = 'none';

        let progressDiv = card.querySelector('.download-progress-container');
        if (!progressDiv) {
            progressDiv = document.createElement('div');
            progressDiv.className = 'download-progress-container';
            progressDiv.innerHTML = `
                <div class="dl-bar-wrapper-sm">
                    <div class="download-progress-bar dl-bar"></div>
                </div>
                <div class="flex-between-center">
                    <div class="download-stats dl-stats-text text-left whitespace-nowrap">Starting...</div>
                    <div class="dl-btn-row">
                        <button class="dl-pause-btn dl-pause-btn-sm">Pause</button>
                        <button class="dl-cancel-btn dl-cancel-btn-sm">Cancel</button>
                    </div>
                </div>
            `;
            rightCol.prepend(progressDiv);
        } else {
            const cancelBtn = progressDiv.querySelector('.dl-cancel-btn');
            if (cancelBtn) cancelBtn.style.display = 'inline-block';
            const pauseBtn = progressDiv.querySelector('.dl-pause-btn');
            if (pauseBtn) pauseBtn.style.display = 'inline-block';
        }
    }

    const dl = downloadModel(modelId, {
        onProgress: (status, pct, speedMB, transMB, totalMB, etaStr) => {
            const card = document.getElementById(`model-card-${modelId}`);
            const bar = card ? card.querySelector('.download-progress-bar') : null;
            const stats = card ? card.querySelector('.download-stats') : null;
            
            if (bar) bar.style.width = `${pct}%`;
            if (stats) {
                if (status.state === 'Queued...') {
                    stats.innerHTML = `<span class="text-yellow">Queued...</span> <a href="/queue" class="text-blue no-underline ml-5">(View Queue)</a>`;
                } else if (status.state !== 'Downloading...') {
                    stats.innerHTML = `${pct.toFixed(1)}% (${transMB} / ${totalMB} MB) | ${DOMPurify.sanitize(status.state)}`;
                } else {
                    stats.innerHTML = `${pct.toFixed(1)}% (${transMB} / ${totalMB} MB) @ ${speedMB} MB/s | ETA: ${etaStr}`;
                }
            }
        },
        onStatusText: (text) => {
            const card = document.getElementById(`model-card-${modelId}`);
            if (!card) return;
            const stats = card.querySelector('.download-stats');
            if (stats) {
                if (text === 'Queued...') {
                    stats.innerHTML = `<span class="text-yellow">Queued...</span> <a href="/queue" class="text-blue no-underline ml-5">(View Queue)</a>`;
                } else {
                    stats.innerText = text;
                }
            }
        },
        onComplete: () => {
            const card = document.getElementById(`model-card-${modelId}`);
            const bar = card ? card.querySelector('.download-progress-bar') : null;
            if (bar) bar.style.width = '100%';
            
            const cancelBtn = card ? card.querySelector('.dl-cancel-btn') : null;
            if (cancelBtn) cancelBtn.style.display = 'none';

            const pauseBtn = card ? card.querySelector('.dl-pause-btn') : null;
            if (pauseBtn) pauseBtn.style.display = 'none';
        }
    });

    const cancelBtn = card.querySelector('.dl-cancel-btn');
    if (cancelBtn) {
        const newCancelBtn = cancelBtn.cloneNode(true);
        cancelBtn.parentNode.replaceChild(newCancelBtn, cancelBtn);
        newCancelBtn.addEventListener('click', () => {
            dl.cancel();
        });
    }

    const pauseBtn = card.querySelector('.dl-pause-btn');
    if (pauseBtn) {
        const newPauseBtn = pauseBtn.cloneNode(true);
        pauseBtn.parentNode.replaceChild(newPauseBtn, pauseBtn);
        newPauseBtn.addEventListener('click', () => {
            dl.pause();
        });
    }

    activeDownloads.set(modelId, dl);

    try {
        await dl.promise;
    } catch (e) {
        if (e.status === 409) {
            // Gracefully handle conflicts without polluting the console
        } else if (e.message !== "Canceled" && e.message !== "Download canceled by user." && e.message !== "Download paused by user.") {
            console.error(`Download failed for ${modelId}:`, e);
        }
    } finally {
        activeDownloads.delete(modelId);
        setTimeout(() => loadModels(), 1500);
    }
}

async function deleteModel(modelId) {
    let warning = `Are you sure you want to permanently delete the weights for ${modelId} from disk?`;

    if (!confirm(warning)) return;

    try {
        await fetchWithAuth(`/api/models/${modelId}`, { method: 'DELETE' });
        SharedDownloadProgress.clearCache();
        loadModels();
    } catch (e) {
        if (e.status === 409) {
            alert("Cannot delete a model while it is downloading or loaded in VRAM.");
        } else {
            console.error("Error deleting model:", e);
            alert("Error deleting model. Check console.");
        }
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    await checkAdmin();
    loadModels();
    discoverActiveDownloads();
    setInterval(discoverActiveDownloads, 5000);
});