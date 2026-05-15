let isAdmin = false;

async function checkAdmin() {
    try {
        const res = await fetchWithAuth('/api/console/loglevel');
        if (res.ok) {
            isAdmin = true;
            document.getElementById('clear-all-btn').style.display = 'inline-flex';
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
            tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #a6adc8;">No active or queued downloads.</td></tr>`;
            return;
        }
        
        let html = '';
        keys.forEach(id => {
            const status = downloads[id];
            const pct = status.total_bytes > 0 ? ((status.bytes_transferred / status.total_bytes) * 100).toFixed(1) : 0;
            const speed = (status.current_speed_bps / 1024 / 1024).toFixed(1);
            const transMB = (status.bytes_transferred / 1024 / 1024).toFixed(1);
            const totalMB = (status.total_bytes / 1024 / 1024).toFixed(1);
            
            let actionHtml = '';
            if (isAdmin) {
                actionHtml = `<button onclick="cancelDownload('${DOMPurify.sanitize(id)}')" style="background: #f38ba8; padding: 4px 8px; font-size: 0.8rem;">Cancel</button>`;
                if (status.state === 'Downloading...') {
                    actionHtml = `<button onclick="pauseDownload('${DOMPurify.sanitize(id)}')" style="background: #f9e2af; padding: 4px 8px; font-size: 0.8rem; margin-right: 5px;">Pause</button>` + actionHtml;
                }
            }

            let statusStr = DOMPurify.sanitize(status.state);
            if (status.state === 'Queued...') {
                statusStr = `<span style="color:#f9e2af;">⏳ ${statusStr}</span>`;
            } else if (status.state === 'Downloading...') {
                statusStr = `<span style="color:#a6e3a1;">⬇️ ${statusStr}</span>`;
            }
            
            html += `<tr>
                <td><strong>${DOMPurify.sanitize(id)}</strong></td>
                <td>${statusStr}</td>
                <td>${pct}% (${transMB} / ${totalMB} MB)</td>
                <td>${speed} MB/s</td>
                <td>${actionHtml}</td>
            </tr>`;
        });
        
        tbody.innerHTML = html;
    } catch (e) {
        console.error("Failed to load queue:", e);
    }
}

window.cancelDownload = async function(id) {
    if (!confirm(`Are you sure you want to cancel the download for ${id}?`)) return;
    try { await fetchWithAuth(`/api/models/${id}/download`, { method: 'DELETE' }); SharedDownloadProgress.clearCache(); loadQueue(); } catch (e) { console.error("Failed to cancel download:", e); alert("Failed to cancel download."); }
};

window.pauseDownload = async function(id) {
    try { await fetchWithAuth(`/api/models/${id}/pause`, { method: 'POST' }); SharedDownloadProgress.clearCache(); loadQueue(); } catch (e) { console.error("Failed to pause download:", e); alert("Failed to pause download."); }
};

window.clearAllDownloads = async function() {
    if (!confirm("Are you sure you want to cancel ALL active and queued downloads?")) return;
    try { await fetchWithAuth('/api/downloads', { method: 'DELETE' }); SharedDownloadProgress.clearCache(); loadQueue(); } catch (e) { console.error("Failed to clear downloads:", e); alert("Failed to clear downloads."); }
};

document.addEventListener('DOMContentLoaded', async () => {
    await checkAdmin();
    loadQueue();
    setInterval(loadQueue, 1500);
});