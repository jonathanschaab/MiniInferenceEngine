async function loadKeys() {
    let res;
    try {
        res = await fetchWithAuth('/api/settings/keys');
    } catch (e) {
        return;
    }
    if (!res.ok) return;
    const keys = await res.json();
    
    const tbody = document.getElementById('keys-tbody');
    tbody.innerHTML = '';
    keys.forEach(record => {
        const shortHash = record.hash.substring(0, 16) + '...';
        const desc = record.description ? record.description : '<span style="color: #6c7086; font-style: italic;">None</span>';
        tbody.innerHTML += `
            <tr>
                <td style="font-weight: bold;">${record.name}</td>
                <td>${desc}</td>
                <td class="hash-text">${shortHash}</td>
                <td><button class="btn-danger" onclick="deleteKey('${record.hash}')">Revoke</button></td>
            </tr>
        `;
    });
}

function openKeyModal() {
    document.getElementById('new-key-name').value = '';
    document.getElementById('new-key-desc').value = '';
    document.getElementById('new-key-modal').style.display = 'flex';
}

function closeKeyModal() {
    document.getElementById('new-key-modal').style.display = 'none';
}

async function submitNewKey() {
    const name = document.getElementById('new-key-name').value.trim();
    const desc = document.getElementById('new-key-desc').value.trim();
    
    if (!name) { alert("A name is required for the API Key."); return; }
    
    const res = await fetchWithAuth('/api/settings/keys', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, description: desc ? desc : null })
    });

    if (res.ok) {
        const plaintextKey = await res.json();
        closeKeyModal();
        // Show the plaintext key to the user EXACTLY once
        window.prompt("Keep this safe! You will never see it again. Copy it now:", plaintextKey);
        loadKeys();
    } else {
        alert("Failed to create API key.");
    }
}

async function deleteKey(hash) {
    if(!confirm("Are you sure you want to permanently revoke this key? External apps using it will instantly fail.")) return;
    await fetchWithAuth(`/api/settings/keys/${hash}`, { method: 'DELETE' });
    loadKeys();
}

window.onload = loadKeys;