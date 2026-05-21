let isAdmin = false;

async function checkAdmin() {
    try {
        const res = await fetchWithAuth('/api/console/loglevel');
        if (res.ok) {
            isAdmin = true;
            document.getElementById('clear-all-btn').classList.remove('hidden');
        }
    } catch (e) {
        console.debug("Could not verify admin status.", e);
    }
}

async function loadQueue() {
    try {
        const downloads = await SharedDownloadProgress.get();
        
        const tbody = document.getElementById('queue-tbody');
        const keys = Object.keys(downloads);
        
        if (keys.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="text-center text-subtext0">No active or queued downloads.</td></tr>`;
            return;
        }

        // Clear the "No active downloads" message if it exists
        if (tbody.children.length === 1 && !tbody.firstElementChild.dataset.id) {
            tbody.innerHTML = '';
        }

        const currentIds = new Set(keys);

        // 1. Remove rows that are no longer in the queue
        Array.from(tbody.children).forEach(row => {
            if (row.dataset.id && !currentIds.has(row.dataset.id)) {
                row.remove();
            }
        });

        keys.forEach(id => {
            const status = downloads[id];
            const pct = status.total_bytes > 0 ? ((status.bytes_transferred / status.total_bytes) * 100).toFixed(1) : 0;
            const speed = (status.current_speed_bps / 1024 / 1024).toFixed(1);
            const transMB = (status.bytes_transferred / 1024 / 1024).toFixed(1);
            const totalMB = (status.total_bytes / 1024 / 1024).toFixed(1);
            
            // Strip ALL HTML tags from the backend status to be absolutely safe
            let safeState = DOMPurify.sanitize(status.state, { ALLOWED_TAGS: [] });
            let statusStr = safeState;
            if (safeState === 'Queued...') {
                statusStr = `<span class="text-yellow">⏳ ${safeState}</span>`;
            } else if (safeState === 'Downloading...') {
                statusStr = `<span class="text-green">⬇️ ${safeState}</span>`;
            }
            
            // 2. Find existing row or create a new one
            let row = Array.from(tbody.children).find(r => r.dataset.id === id);
            
            if (!row) {
                row = document.createElement('tr');
                row.dataset.id = id;
                row.innerHTML = `
                    <td class="col-id"></td>
                    <td class="col-status"></td>
                    <td class="col-progress"></td>
                    <td class="col-speed"></td>
                    <td class="col-actions"></td>
                `;

                // Build buttons dynamically to avoid string literal injection (single quotes breaking onclick)
                if (isAdmin) {
                    const actionsTd = row.querySelector('.col-actions');
                    
                    const pauseBtn = document.createElement('button');
                    pauseBtn.textContent = 'Pause';
                    pauseBtn.className = 'btn-pause';
                    pauseBtn.className = 'btn-run bg-yellow text-base mr-5';
                    pauseBtn.onclick = () => pauseDownload(id);
                    
                    const cancelBtn = document.createElement('button');
                    cancelBtn.textContent = 'Cancel';
                    cancelBtn.className = 'btn-run bg-red text-base';
                    cancelBtn.onclick = () => cancelDownload(id);
                    
                    actionsTd.appendChild(pauseBtn);
                    actionsTd.appendChild(cancelBtn);
                }

                tbody.appendChild(row);
            }

            // Strip ALL HTML tags from the ID
            const idHtml = `<strong>${DOMPurify.sanitize(id, { ALLOWED_TAGS: [] })}</strong>`;
            const progressHtml = `${pct}% (${transMB} / ${totalMB} MB)`;
            const speedHtml = `${speed} MB/s`;

            // 3. Only update the inner HTML if it has actually changed.
            // This preserves button hover states and keyboard focus if the state hasn't shifted!
            if (row.querySelector('.col-id').innerHTML !== idHtml) row.querySelector('.col-id').innerHTML = idHtml;
            if (row.querySelector('.col-status').innerHTML !== statusStr) row.querySelector('.col-status').innerHTML = statusStr;
            if (row.querySelector('.col-progress').innerHTML !== progressHtml) row.querySelector('.col-progress').innerHTML = progressHtml;
            if (row.querySelector('.col-speed').innerHTML !== speedHtml) row.querySelector('.col-speed').innerHTML = speedHtml;
            
            // Update pause button visibility dynamically
            if (isAdmin) {
                const pauseBtn = row.querySelector('.btn-pause');
                if (pauseBtn) {
                    pauseBtn.style.display = (status.state === 'Downloading...') ? 'inline-block' : 'none';
                }
            }
        });
    } catch (e) {
        console.error("Failed to load queue:", e);
    }
}

async function cancelDownload(id) {
    if (!confirm(`Are you sure you want to cancel the download for ${id}?`)) return;
    try { await fetchWithAuth(`/api/downloads/${id}`, { method: 'DELETE' }); SharedDownloadProgress.clearCache(); loadQueue(); } catch (e) { console.error("Failed to cancel download:", e); alert("Failed to cancel download."); }
};

async function pauseDownload(id) {
    try { await fetchWithAuth(`/api/downloads/${id}/pause`, { method: 'POST' }); SharedDownloadProgress.clearCache(); loadQueue(); } catch (e) { console.error("Failed to pause download:", e); alert("Failed to pause download."); }
};

async function clearAllDownloads() {
    if (!confirm("Are you sure you want to cancel ALL active and queued downloads?")) return;
    try { await fetchWithAuth('/api/downloads', { method: 'DELETE' }); SharedDownloadProgress.clearCache(); loadQueue(); } catch (e) { console.error("Failed to clear downloads:", e); alert("Failed to clear downloads."); }
};

document.addEventListener('DOMContentLoaded', async () => {
    await checkAdmin();
    loadQueue();
    setInterval(loadQueue, 1500);
    document.getElementById('clear-all-btn')?.addEventListener('click', clearAllDownloads);
});