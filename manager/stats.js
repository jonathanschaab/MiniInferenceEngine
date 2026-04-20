const colorPalette = ['#89b4fa', '#f38ba8', '#a6e3a1', '#f9e2af', '#cba6f7', '#94e2d5'];

async function openBenchmarkModal() {
    if (isBenchmarking) {
        alert("A benchmark is already running!");
        return;
    }

    try {
        // Fetch the current roster
        const res = await fetchWithAuth('/api/models');
        const models = await res.json();
        
        const listDiv = document.getElementById('model-checkbox-list');
        
        const allBackends = new Set();
        let modelsHtml = '';
        models.forEach(m => {
            const backendsStr = DOMPurify.sanitize(m.supported_backends.map(b => b.toLowerCase()).join(','));
            modelsHtml += `
                <label class="model-item">
                    <input type="checkbox" class="model-cb" value="${DOMPurify.sanitize(m.id)}" data-backends="${backendsStr}" checked>
                    ${DOMPurify.sanitize(m.name)} <span style="color: #6c7086; margin-left: 5px;">(${DOMPurify.sanitize(m.arch)})</span>
                    <span class="incompatible-warning" style="display: none; color: #f38ba8; margin-left: auto; font-size: 0.85em; font-style: italic;">(Incompatible)</span>
                </label>
            `;
            m.supported_backends.forEach(b => allBackends.add(b));
        });
        listDiv.innerHTML = modelsHtml;
        
        const backendDiv = document.getElementById('backend-checkbox-list');
        let backendsHtml = '';
        allBackends.forEach(b => {
            backendsHtml += `
                <label class="model-item"><input type="checkbox" class="backend-cb" value="${DOMPurify.sanitize(b.toLowerCase())}" checked> ${DOMPurify.sanitize(b)}</label>
            `;
        });
        backendDiv.innerHTML = backendsHtml;
        
        // Add listeners to auto-grey out incompatible models
        document.querySelectorAll('.backend-cb').forEach(cb => {
            cb.addEventListener('change', updateBenchmarkCompatibility);
        });

        updateBenchmarkCompatibility();

        // Show the modal
        document.getElementById('benchmark-modal').style.display = 'flex';
    } catch (e) {
        console.error("Failed to load models for modal", e);
        alert("Failed to fetch model registry.");
    }
}

function updateBenchmarkCompatibility() {
    const checkedBackends = Array.from(document.querySelectorAll('.backend-cb:checked')).map(cb => cb.value);
    
    document.querySelectorAll('.model-cb').forEach(cb => {
        const supportedBackends = cb.dataset.backends ? cb.dataset.backends.split(',') : [];
        const supported = supportedBackends.some(b => checkedBackends.includes(b));
        
        cb.disabled = !supported;
        cb.parentElement.style.opacity = supported ? "1" : "0.5";
        
        const warningSpan = cb.parentElement.querySelector('.incompatible-warning');
        if (warningSpan) {
            warningSpan.style.display = supported ? 'none' : 'inline';
        }
    });
}

function closeModal() {
    document.getElementById('benchmark-modal').style.display = 'none';
}

async function submitBenchmark() {
    // Gather all checked boxes that are currently compatible (not disabled)
    const checkboxes = document.querySelectorAll('#model-checkbox-list input:checked:not(:disabled)');
    const selectedModels = Array.from(checkboxes).map(cb => cb.value);

    if (selectedModels.length === 0) {
        alert("Please select at least one model to benchmark.");
        return;
    }
    
    const backendCheckboxes = document.querySelectorAll('#backend-checkbox-list input:checked');
    const selectedBackends = Array.from(backendCheckboxes).map(cb => cb.value);
    
    if (selectedBackends.length === 0) {
        alert("Please select at least one backend to benchmark.");
        return;
    }
    
    const parameters = getGenerationParameters();

    closeModal();

    // Fire the updated POST request
    const res = await fetchWithAuth('/api/stats/collect', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
        models: selectedModels,
        target_backends: selectedBackends,
        parameters: parameters
    })
    });

    if (res.status === 409) {
        alert("A benchmark is already running in the background!");
    } else {
        checkStatus(); // Instantly lock the UI
    }
}

async function loadDashboard() {
    let modelsRes, statsRes;
    try {
        [modelsRes, statsRes] = await Promise.all([
            fetchWithAuth('/api/models'),
            fetchWithAuth('/api/stats/data')
        ]);
    } catch (e) {
        return; // Redirected by fetchWithAuth
    }
    
    if (!modelsRes.ok || !statsRes.ok) return;
    
    const models = await modelsRes.json();
    const stats = await statsRes.json();
    
    populateTable(models, stats.loads);
    renderSpeedChart(models, stats.generations);
    renderTokenChart(models, stats.generations);
    renderTokTimeChart(models, stats.generations);
}

function populateTable(models, loads) {
    const tbody = document.getElementById('model-table-body');
    const thead = document.querySelector('table thead');
    tbody.innerHTML = '';
    
    const allBackends = Array.from(new Set(models.flatMap(m => m.supported_backends))).sort();
    
    let theadHTML = `<tr>
        <th>Model Name</th>
        <th>Parameters</th>
        <th>Size on Disk</th>
        <th>Max Context</th>`;
    allBackends.forEach(b => {
        theadHTML += `<th>Avg Load (${DOMPurify.sanitize(b)})</th>`;
    });
    theadHTML += `</tr>`;
    thead.innerHTML = theadHTML;

    const loadAverages = {};
    loads.forEach(l => {
        if (!loadAverages[l.model_id]) loadAverages[l.model_id] = {};
        const b = l.backend || 'Candle'; // Fallback for old data points
        if (!loadAverages[l.model_id][b]) loadAverages[l.model_id][b] = { sum: 0, count: 0 };
        loadAverages[l.model_id][b].sum += l.load_time_ms;
        loadAverages[l.model_id][b].count++;
    });

    let tbodyHTML = '';
    models.forEach(m => {
        let rowHTML = `<tr>
            <td>${DOMPurify.sanitize(m.name)}</td>
            <td>${DOMPurify.sanitize(m.parameters_billions.toString())}B</td>
            <td>${DOMPurify.sanitize(m.size_on_disk_gb.toString())} GB</td>
            <td>${DOMPurify.sanitize(m.max_context_len.toLocaleString())}</td>`;
            
        allBackends.forEach(b => {
            if (loadAverages[m.id] && loadAverages[m.id][b]) {
                rowHTML += `<td>${Math.round(loadAverages[m.id][b].sum / loadAverages[m.id][b].count)} ms</td>`;
            } else {
                rowHTML += `<td><span style="color:#6c7086;">-</span></td>`;
            }
        });
        rowHTML += `</tr>`;
        tbodyHTML += rowHTML;
    });
    tbody.innerHTML = tbodyHTML;
}

function renderSpeedChart(models, generations) {
    const uniqueCombos = [];
    models.forEach(m => {
        m.supported_backends.forEach(b => {
            const hasData = generations.some(g => g.model_id === m.id && g.backend === b);
            if (hasData) uniqueCombos.push({ model: m, backend: b });
        });
    });

    const datasets = uniqueCombos.map((combo, i) => {
        const modelData = generations.filter(g => g.model_id === combo.model.id && g.backend === combo.backend);
        // Group by prompt size to calculate averages
        const grouped = {};
        modelData.forEach(g => {
            const bucket = g.prompt_tokens;
            if (!grouped[bucket]) grouped[bucket] = { sum: 0, count: 0, offload_sum: 0 };
            grouped[bucket].sum += g.generation_time_ms;
            grouped[bucket].offload_sum += g.offload_pct || 0;
            grouped[bucket].count++;
        });

        const dataPoints = Object.keys(grouped).map(tokens => ({
            x: parseInt(tokens),
            y: grouped[tokens].sum / grouped[tokens].count,
            offload: grouped[tokens].offload_sum / grouped[tokens].count
        })).sort((a, b) => a.x - b.x);

        return {
            label: `${combo.model.name} (${combo.backend})`,
            data: dataPoints,
            borderColor: colorPalette[i % colorPalette.length],
            backgroundColor: colorPalette[i % colorPalette.length],
            tension: 0.3,
            showLine: true
        };
    }).filter(d => d.data.length > 0);

    new Chart(document.getElementById('speedChart'), {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { 
                title: { display: true, text: 'Average Generation Time vs. Prompt Tokens', color: '#cdd6f4' }, 
                legend: { labels: { color: '#cdd6f4' } },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) { label += ': '; }
                            label += `${Math.round(context.parsed.y)} ms`;
                            if (context.raw.offload > 0) { label += ` (${(context.raw.offload * 100).toFixed(1)}% offload)`; }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: { type: 'logarithmic', title: { display: true, text: 'Prompt Size (Tokens) - Log Scale', color: '#cdd6f4' }, grid: { color: '#313244' }, ticks: { color: '#a6adc8' } },
                y: { title: { display: true, text: 'Generation Time (ms)', color: '#cdd6f4' }, grid: { color: '#313244' }, ticks: { color: '#a6adc8' } }
            }
        }
    });
}

function renderTokenChart(models, generations) {
    const uniqueCombos = [];
    models.forEach(m => {
        m.supported_backends.forEach(b => {
            const hasData = generations.some(g => g.model_id === m.id && g.backend === b);
            if (hasData) uniqueCombos.push({ model: m, backend: b });
        });
    });

    const datasets = uniqueCombos.map((combo, i) => {
        const modelData = generations.filter(g => g.model_id === combo.model.id && g.backend === combo.backend);
        // Compare Raw Characters to Parsed Tokens
        const dataPoints = modelData.map(g => ({
            x: g.prompt_chars,
            y: g.prompt_tokens
        })).sort((a, b) => a.x - b.x);

        return {
            label: `${combo.model.name} (${combo.backend})`,
            data: dataPoints,
            borderColor: colorPalette[i % colorPalette.length],
            backgroundColor: colorPalette[i % colorPalette.length],
            tension: 0.3,
            showLine: true
        };
    }).filter(d => d.data.length > 0);

    new Chart(document.getElementById('tokenChart'), {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { title: { display: true, text: 'Tokenizer Efficiency (Tokens vs. Raw Characters)', color: '#cdd6f4' }, legend: { labels: { color: '#cdd6f4' } } },
            scales: {
                x: { type: 'logarithmic', title: { display: true, text: 'File Size (Raw Characters) - Log Scale', color: '#cdd6f4' }, grid: { color: '#313244' }, ticks: { color: '#a6adc8' } },
                y: { type: 'logarithmic', title: { display: true, text: 'Required Tokens', color: '#cdd6f4' }, grid: { color: '#313244' }, ticks: { color: '#a6adc8' } }
            }
        }
    });
}

function renderTokTimeChart(models, generations) {
    const uniqueCombos = [];
    models.forEach(m => {
        m.supported_backends.forEach(b => {
            const hasData = generations.some(g => g.model_id === m.id && g.backend === b);
            if (hasData) uniqueCombos.push({ model: m, backend: b });
        });
    });

    const datasets = uniqueCombos.map((combo, i) => {
        const modelData = generations.filter(g => g.model_id === combo.model.id && g.backend === combo.backend);
        const grouped = {};
        modelData.forEach(g => {
            const bucket = g.prompt_tokens;
            if (!grouped[bucket]) grouped[bucket] = { sum: 0, count: 0 };
            grouped[bucket].sum += g.tokenization_time_ms || 0;
            grouped[bucket].count++;
        });

        const dataPoints = Object.keys(grouped).map(tokens => ({
            x: parseInt(tokens),
            y: grouped[tokens].sum / grouped[tokens].count
        })).sort((a, b) => a.x - b.x);

        return {
            label: `${combo.model.name} (${combo.backend})`,
            data: dataPoints,
            borderColor: colorPalette[(i + 2) % colorPalette.length],
            backgroundColor: colorPalette[(i + 2) % colorPalette.length],
            tension: 0.3,
            showLine: true
        };
    }).filter(d => d.data.length > 0);

    new Chart(document.getElementById('tokTimeChart'), {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { 
                title: { display: true, text: 'Average Tokenization Time vs. Prompt Tokens', color: '#cdd6f4' }, 
                legend: { labels: { color: '#cdd6f4' } }
            },
            scales: {
                x: { type: 'logarithmic', title: { display: true, text: 'Prompt Size (Tokens) - Log Scale', color: '#cdd6f4' }, grid: { color: '#313244' }, ticks: { color: '#a6adc8' } },
                y: { title: { display: true, text: 'Tokenization Time (ms)', color: '#cdd6f4' }, grid: { color: '#313244' }, ticks: { color: '#a6adc8' } }
            }
        }
    });
}

window.onload = loadDashboard;

let isBenchmarking = false;

async function checkStatus() {
    try {
        const res = await fetchWithAuth('/api/status');
        if (!res.ok) return;
        
        const status = await res.json();
        
        const btn = document.getElementById('btn-run-benchmark');
        const banner = document.getElementById('benchmark-banner');
        
        if (status.benchmark_running) {
            if (!isBenchmarking) {
                isBenchmarking = true;
                btn.disabled = true;
                btn.innerText = "⏳ Benchmark Running...";
                btn.style.background = "#f38ba8"; 
                btn.style.cursor = "not-allowed";
                if (banner) banner.style.display = "block";
            }
        } else {
            if (isBenchmarking) {
                // The benchmark just finished! Reset UI and reload the graphs.
                isBenchmarking = false;
                btn.disabled = false;
                btn.innerText = "▶ Run Benchmark Suite";
                btn.style.background = "#89b4fa"; 
                btn.style.cursor = "pointer";
                if (banner) banner.style.display = "none";
                loadDashboard(); 
            }
        }
    } catch (e) {
        console.error("Failed to fetch engine status", e);
    }
}

// Poll the server every 2 seconds
setInterval(checkStatus, 2000);

// Call checkStatus immediately on load alongside the dashboard
window.onload = () => {
    loadDashboard();
    checkStatus();
};