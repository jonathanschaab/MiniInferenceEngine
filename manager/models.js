async function loadModels() {
    try {
        const res = await fetchWithAuth('/api/models');
        const models = await res.json();
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
            card.className = 'model-card';

            const rolesStr = model.roles.join(', ');
            const backendsStr = model.supported_backends.join(', ');

            card.innerHTML = `
                <div class="model-header">
                    <div>
                        <h2 style="margin:0; color: #fab387;">${model.name}</h2>
                        <p style="margin: 5px 0 0 0; color: #a6adc8; font-size: 0.9rem;">ID: ${model.id} | Repo: <a href="https://huggingface.co/${model.repo}" target="_blank" style="color: #89b4fa;">${model.repo}</a></p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.85rem; color: #cba6f7; margin-bottom: 4px;"><strong>Roles:</strong> ${rolesStr}</div>
                        <div style="font-size: 0.85rem; color: #a6e3a1;"><strong>Backends:</strong> ${backendsStr}</div>
                    </div>
                </div>
                <div class="model-settings">
                    <div class="setting-item"><span>Architecture</span> <span>${model.arch} ${getBadge(model.provenance.arch)}</span></div>
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
            container.appendChild(card);
        });
    } catch (e) {
        console.error('Failed to load models directory:', e);
    }
}

document.addEventListener('DOMContentLoaded', loadModels);