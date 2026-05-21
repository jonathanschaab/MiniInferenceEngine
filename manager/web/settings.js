async function loadKeys() {
    let res;
    try {
        res = await fetchWithAuth('/api/settings/keys');
    } catch (e) {
        console.error("Failed to load keys", e);
        return;
    }
    if (!res.ok) return;
    const keys = await res.json();
    
    const tbody = document.getElementById('keys-tbody');
    tbody.innerHTML = '';

    keys.forEach(record => {
        const tr = document.createElement('tr');
        
        const tdName = document.createElement('td');
        tdName.style.fontWeight = 'bold';
        tdName.textContent = record.name; // Natively escapes HTML and quotes!
        
        const tdDesc = document.createElement('td');
        if (record.description) {
            tdDesc.textContent = record.description;
        } else {
            tdDesc.innerHTML = '<span class="text-overlay0 font-italic">None</span>';
        }
        
        const tdHash = document.createElement('td');
        tdHash.className = 'hash-text';
        tdHash.textContent = record.hash.substring(0, 16) + '...';
        
        const tdAction = document.createElement('td');
        const btn = document.createElement('button');
        btn.className = 'btn-danger';
        btn.textContent = 'Revoke';
        btn.onclick = () => deleteKey(record.hash); // Safely keeps the hash inside a JS closure
        tdAction.appendChild(btn);
        
        tr.appendChild(tdName);
        tr.appendChild(tdDesc);
        tr.appendChild(tdHash);
        tr.appendChild(tdAction);
        
        tbody.appendChild(tr);
    });
}

function openKeyModal() {
    document.getElementById('new-key-name').value = '';
    document.getElementById('new-key-desc').value = '';
    document.getElementById('new-key-modal').style.display = 'flex';
};

function closeKeyModal() {
    document.getElementById('new-key-modal').style.display = 'none';
};

async function submitNewKey() {
    const name = document.getElementById('new-key-name').value.trim();
    const desc = document.getElementById('new-key-desc').value.trim();
    
    if (!name) { alert("A name is required for the API Key."); return; }
    
    try {
        const res = await fetchWithAuth('/api/settings/keys', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name, description: desc ? desc : null })
        });
        const plaintextKey = await res.json();
        closeKeyModal();
        // Show the plaintext key to the user EXACTLY once
        window.prompt("Keep this safe! You will never see it again. Copy it now:", plaintextKey);
        loadKeys();
    } catch (e) {
        console.error("Failed to create API key:", e);
        alert("Failed to create API key. Check console for details.");
    }
}

function showDeleteKeyModal() {
    return new Promise((resolve) => {
        const modal = document.getElementById('delete-key-modal');
        const confirmBtn = document.getElementById('delete-key-confirm-btn');
        const cancelBtn = document.getElementById('delete-key-cancel-btn');
        
        modal.style.display = 'flex';

        const cleanup = () => {
            modal.style.display = 'none';
            confirmBtn.onclick = null;
            cancelBtn.onclick = null;
        };

        confirmBtn.onclick = () => {
            cleanup();
            resolve(true);
        };

        cancelBtn.onclick = () => {
            cleanup();
            resolve(false);
        };
    });
}

async function deleteKey(hash) {
    const confirmed = await showDeleteKeyModal();
    if (confirmed) {
        try {
            await fetchWithAuth(`/api/settings/keys/${hash}`, { method: 'DELETE' });
            loadKeys();
        } catch (e) {
            console.error("Failed to revoke API key:", e);
            alert("Failed to revoke API key. Check console for details.");
        }
    }
}

window.onload = () => {
    loadKeys();
    
    document.getElementById('open-key-modal-btn')?.addEventListener('click', openKeyModal);
    document.getElementById('new-key-cancel-btn')?.addEventListener('click', closeKeyModal);
    document.getElementById('create-key-btn')?.addEventListener('click', submitNewKey);
};