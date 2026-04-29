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
                return `<span class="badge" style="background: #45475a;">${source || 'Unknown'}</span>`;
            };

            const card = document.createElement('div');
            card.className = `model-card ${model.is_downloaded ? '' : 'model-undownloaded'}`;
            card.id = `model-card-${model.id}`;

            const rolesStr = model.roles.join(', ');
            const backendsStr = model.supported_backends.join(', ');

            let healthBadge = '';
            if (healthMap[model.id] === true) {
                healthBadge = `<span class="badge" style="background: #a6e3a1; color: #11111b; margin-left: 10px;" title="Last run succeeded">✅ Passed</span>`;
            } else if (healthMap[model.id] === false) {
                healthBadge = `<span class="badge" style="background: #f38ba8; color: #11111b; margin-left: 10px;" title="Last run failed">❌ Failed</span>`;
            }

            let adminDeleteBtn = '';
            if (isAdmin && model.is_downloaded) {
                adminDeleteBtn = `<button class="btn-cancel" style="padding: 5px 10px; font-size: 0.85rem;" onclick="deleteModel('${model.id}')">🗑️ Delete</button>`;
            }

            const downloadBtnHtml = model.is_downloaded
                ? `<div style="display: flex; gap: 8px; align-items: center;">${adminDeleteBtn}<span class="badge badge-json" style="padding: 5px 10px;">Ready</span></div>`
                : `<button class="btn-download">Download Model</button>`;

            const moeHtml = (model.num_local_experts != null && model.num_experts_per_tok != null)
                ? `<div class="setting-item setting-moe"><span>MoE Routing</span> <span>${model.num_experts_per_tok} / ${model.num_local_experts} Active ${getBadge(model.provenance.num_local_experts)}</span></div>`
                : '';

            const cardHtml = `
                <div class="model-header">
                    <div>
                        <h2 style="margin:0; color: ${model.is_downloaded ? '#fab387' : '#6c7086'}; display: flex; align-items: center;">${model.name} ${healthBadge}</h2>
                        <p style="margin: 5px 0 0 0; color: #a6adc8; font-size: 0.9rem;">ID: ${model.id} | Repo: <a href="https://huggingface.co/${model.repo}" target="_blank" style="color: #89b4fa;">${model.repo}</a></p>
                    </div>
                    <div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: 8px;">
                        ${downloadBtnHtml}
                        <div style="font-size: 0.85rem; color: #cba6f7; margin-bottom: 4px;"><strong>Roles:</strong> ${rolesStr}</div>
                        <div style="font-size: 0.85rem; color: #a6e3a1;"><strong>Backends:</strong> ${backendsStr}</div>
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

            container.appendChild(card);
        });
    } catch (e) {
        console.error('Failed to load models directory:', e);
    }
}

const cancelDownloadFlags = new Set();
let activePollInterval = null;

async function pollDownloads() {
    try {
        const res = await fetchWithAuth('/api/models/download/progress');
        if (!res.ok) return;
        const downloads = await res.json();
        
        const activeIds = Object.keys(downloads);
        
        if (activeIds.length > 0) {
            activeIds.forEach(id => {
                const card = document.getElementById(`model-card-${id}`);
                if (card && !card.querySelector('.download-progress-container')) {
                    startDownload(id);
                }
            });
            
            if (!activePollInterval) {
                activePollInterval = setInterval(pollDownloads, 5000);
            }
        } else {
            if (activePollInterval) {
                clearInterval(activePollInterval);
                activePollInterval = null;
            }
        }
    } catch (e) {
        console.error("Polling failed", e);
    }
}

function updateDownloadUI(modelId, status) {
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

    const card = document.getElementById(`model-card-${modelId}`);
    const bar = card ? card.querySelector('.download-progress-bar') : null;
    const stats = card ? card.querySelector('.download-stats') : null;
    
    if (bar) bar.style.width = `${pct}%`;
    if (stats) {
        if (status.state === 'Verifying...') {
            stats.innerText = `${pct.toFixed(1)}% (${transMB} / ${totalMB} MB) | Verifying...`;
        } else {
            stats.innerText = `${pct.toFixed(1)}% (${transMB} / ${totalMB} MB) @ ${speedMB} MB/s | ETA: ${etaStr}`;
        }
    }
}

window.startDownload = async function(modelId) {
    cancelDownloadFlags.delete(modelId);
    
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
                <div style="width: 250px; background: #313244; border-radius: 4px; overflow: hidden; margin-bottom: 4px; border: 1px solid #45475a;">
                    <div class="download-progress-bar" style="width: 0%; height: 8px; background: #a6e3a1; transition: width 0.5s ease-out;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="download-stats" style="font-size: 0.75rem; color: #a6adc8; text-align: left; white-space: nowrap;">Starting...</div>
                    <button class="dl-cancel-btn" style="padding: 3px 8px; background: #f38ba8; color: #11111b; border: none; border-radius: 4px; cursor: pointer; font-size: 0.75rem; font-weight: bold;">Cancel</button>
                </div>
            `;
            rightCol.prepend(progressDiv);
            
            progressDiv.querySelector('.dl-cancel-btn').addEventListener('click', () => {
                cancelDownloadFlags.add(modelId);
            });
        } else {
            const cancelBtn = progressDiv.querySelector('.dl-cancel-btn');
            if (cancelBtn) cancelBtn.style.display = 'block';
        }
    }

    while (!cancelDownloadFlags.has(modelId)) {
        try {
            const progCheck = await fetchWithAuth('/api/models/download/progress');
            const activeDls = await progCheck.json();
            
            if (!activeDls[modelId]) {
                try {
                    await fetchWithAuth(`/api/models/${modelId}/download`, { method: 'POST' });
                } catch (e) {
                    if (e.message === 'Unauthorized') return;
                    if (!e.message.includes('409')) {
                        updateDownloadStatusText(modelId, `Failed to start. Retrying in 5s...`);
                        await sleep(5000, modelId);
                        continue;
                    }
                }
            }

            await new Promise((resolve, reject) => {
                const interval = setInterval(async () => {
                    if (cancelDownloadFlags.has(modelId)) {
                        clearInterval(interval);
                        reject(new Error("Canceled"));
                        return;
                    }
                    
                    try {
                        const progRes = await fetchWithAuth('/api/models/download/progress');
                        const downloads = await progRes.json();
                        const status = downloads[modelId];
                        
                        if (status) {
                            updateDownloadUI(modelId, status);
                        } else {
                            clearInterval(interval);
                            
                            const verifyRes = await fetchWithAuth('/api/models');
                            const verifyModels = await verifyRes.json();
                            const verifyModel = verifyModels.find(m => m.id === modelId);
                            
                            if (verifyModel && verifyModel.is_downloaded) {
                                updateDownloadStatusText(modelId, 'Download Complete!');
                                const card = document.getElementById(`model-card-${modelId}`);
                                const bar = card ? card.querySelector('.download-progress-bar') : null;
                                if (bar) bar.style.width = '100%';
                                
                                const cancelBtn = card ? card.querySelector('.dl-cancel-btn') : null;
                                if (cancelBtn) cancelBtn.style.display = 'none';
                                
                                setTimeout(() => loadModels(), 1500); 
                                resolve();
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
            
            return; 

        } catch (e) {
            if (cancelDownloadFlags.has(modelId)) {
                updateDownloadStatusText(modelId, 'Download Canceled.');
                setTimeout(() => loadModels(), 1500);
                return;
            }
            updateDownloadStatusText(modelId, 'Interrupted. Retrying in 5s...');
            console.error(`Download ${modelId} interrupted, retrying in 5s...`, e);
            await sleep(5000, modelId);
        }
    }
    
    updateDownloadStatusText(modelId, 'Download Canceled.');
    setTimeout(() => loadModels(), 1500);
};

function updateDownloadStatusText(modelId, text) {
    const card = document.getElementById(`model-card-${modelId}`);
    if (!card) return;
    const stats = card.querySelector('.download-stats');
    if (stats) stats.innerText = text;
}

async function sleep(ms, modelId) {
    for (let i = 0; i < ms; i += 100) {
        if (cancelDownloadFlags.has(modelId)) return;
        await new Promise(r => setTimeout(r, 100));
    }
}

window.deleteModel = async function(modelId) {
    if (!confirm(`Are you sure you want to permanently delete the weights for ${modelId} from disk?`)) return;
    try {
        const res = await fetchWithAuth(`/api/models/${modelId}/download`, { method: 'DELETE' });
        if (res.ok) {
            loadModels();
        } else {
            const text = await res.text();
            alert(`Failed to delete model: ${text}`);
        }
    } catch (e) {
        console.error("Error deleting model:", e);
        alert("Error deleting model. Check console.");
    }
};

document.addEventListener('DOMContentLoaded', async () => {
    await checkAdmin();
    loadModels();
    pollDownloads();
});