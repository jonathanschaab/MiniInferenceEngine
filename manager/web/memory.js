function formatMB(bytes) {
    if (bytes === 0) return "-";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}
function formatGB(bytes) {
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
}
function formatTime(millis) {
    const d = new Date(millis);
    return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}`;
}

/* eslint-disable-next-line no-unused-vars -- Called by: memory.html tab buttons onclick="switchTab('tabName')" */
function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    const activeTab = document.querySelector(`.tab[onclick="switchTab('${tabName}')"]`);
    const activeContent = document.getElementById(`${tabName}-view`);
    if (activeTab) activeTab.classList.add('active');
    if (activeContent) activeContent.classList.add('active');
}

async function updateDashboard() {
    let res;
    try {
        res = await fetchWithAuth('/api/status');
    } catch (e) {
        console.error("Failed to update dashboard", e);
        return;
    }
    if (!res.ok) return;
    
    const status = await res.json();
    
    // VRAM Update
    if (status.vram_total > 0) {
        let totalWeights = 0, totalKv = 0, totalCompute = 0;
        
        const modelsContainer = document.getElementById('loaded-models-container');
        let modelsHtml = '';

        status.models_vram.forEach(m => {
            totalWeights += m.weights;
            totalKv += m.kv_cache;
            totalCompute += m.compute;

            const statusClass = m.status === 'Active' ? 'status-active' : 'status-idle';
            modelsHtml += `
                <div class="model-card">
                    <h3>${DOMPurify.sanitize(m.id)} <span style="color:#a6adc8; font-size:0.8rem;">(${DOMPurify.sanitize(m.backend)})</span></h3>
                    <div class="model-stat"><span>Weights:</span> <span style="color:#89b4fa; font-weight:bold;">${formatMB(m.weights)}</span></div>
                    <div class="model-stat"><span>KV Cache Context:</span> <span style="color:#a6e3a1; font-weight:bold;">${formatMB(m.kv_cache)}</span></div>
                    <div class="model-stat"><span>Compute Buffer:</span> <span style="color:#f9e2af; font-weight:bold;">${formatMB(m.compute)}</span></div>
                    <div class="model-stat"><span>Status:</span> <span class="${statusClass}">${DOMPurify.sanitize(m.status)}</span></div>
                    <div class="model-stat" style="margin-top:10px; border-top:1px solid #45475a; padding-top:8px;"><span>Total Impact:</span> <span style="color:#cdd6f4; font-weight:bold;">${formatMB(m.weights + m.kv_cache + m.compute)}</span></div>
                </div>
            `;
        });
        modelsContainer.innerHTML = modelsHtml;

        const otherPct = (status.vram_other_processes / status.vram_total) * 100;
        const freePct = (status.vram_free / status.vram_total) * 100;

        document.getElementById('bar-weights').style.width = `${(totalWeights / status.vram_total) * 100}%`;
        document.getElementById('bar-kv').style.width = `${(totalKv / status.vram_total) * 100}%`;
        document.getElementById('bar-compute').style.width = `${(totalCompute / status.vram_total) * 100}%`;
        document.getElementById('bar-other').style.width = `${otherPct}%`;
        document.getElementById('bar-free').style.width = `${freePct}%`;

        document.getElementById('txt-weights').innerText = `Weights: ${formatGB(totalWeights)}`;
        document.getElementById('txt-kv').innerText = `KV Cache: ${formatGB(totalKv)}`;
        document.getElementById('txt-compute').innerText = `Compute: ${formatGB(totalCompute)}`;
        document.getElementById('txt-other').innerText = `OS/Other: ${formatGB(status.vram_other_processes)}`;
        document.getElementById('txt-free').innerText = `Free: ${formatGB(status.vram_free)}`;
    }

    const vramTbody = document.getElementById('vram-log-body');
    let vramTbodyHtml = '';
    
    status.vram_events.slice().reverse().forEach(ev => { // Newest on top
        let colorClass = ev.action === "Allocate" ? "log-allocate" : (ev.action === "Free" ? "log-free" : (ev.action === "Fail" ? "log-fail" : "log-measure"));
        vramTbodyHtml += `
            <tr>
                <td style="color: #6c7086;">${formatTime(ev.timestamp)}</td>
                <td class="${colorClass}">${DOMPurify.sanitize(ev.action)}</td>
                <td style="color: #cba6f7;">${DOMPurify.sanitize(ev.subsystem)}</td>
                <td>${DOMPurify.sanitize(ev.description)}</td>
                <td>${formatMB(ev.bytes)}</td>
            </tr>`;
    });
    vramTbody.innerHTML = vramTbodyHtml;

    // RAM Update
    if (status.ram_total > 0) {
        const processPct = (status.ram_process_used / status.ram_total) * 100;
        const otherPct = (status.ram_other_processes / status.ram_total) * 100;
        const freePct = (status.ram_free / status.ram_total) * 100;

        document.getElementById('bar-ram-process').style.width = `${processPct}%`;
        document.getElementById('bar-ram-other').style.width = `${otherPct}%`;
        document.getElementById('bar-ram-free').style.width = `${freePct}%`;

        document.getElementById('txt-ram-process').innerText = `Engine Process: ${formatGB(status.ram_process_used)}`;
        document.getElementById('txt-ram-other').innerText = `OS/Other: ${formatGB(status.ram_other_processes)}`;
        document.getElementById('txt-ram-free').innerText = `Free: ${formatGB(status.ram_free)}`;
    }

    const ramTbody = document.getElementById('ram-log-body');
    let ramTbodyHtml = '';
    status.ram_events.slice().reverse().forEach(ev => { // Newest on top
        let colorClass = ev.action === "Allocate" ? "log-allocate" : (ev.action === "Free" ? "log-free" : (ev.action === "Fail" ? "log-fail" : "log-measure"));
        ramTbodyHtml += `
            <tr>
                <td style="color: #6c7086;">${formatTime(ev.timestamp)}</td>
                <td class="${colorClass}">${DOMPurify.sanitize(ev.action)}</td>
                <td style="color: #cba6f7;">${DOMPurify.sanitize(ev.subsystem)}</td>
                <td>${DOMPurify.sanitize(ev.description)}</td>
                <td>${formatMB(ev.bytes)}</td>
            </tr>`;
    });
    ramTbody.innerHTML = ramTbodyHtml;
}
setInterval(updateDashboard, 1000);
window.onload = updateDashboard;