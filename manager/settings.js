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
    let tbodyHtml = '';
    keys.forEach(record => {
        const shortHash = record.hash.substring(0, 16) + '...';
        const desc = record.description ? DOMPurify.sanitize(record.description) : '<span style="color: #6c7086; font-style: italic;">None</span>';
        tbodyHtml += `
            <tr>
                <td style="font-weight: bold;">${DOMPurify.sanitize(record.name)}</td>
                <td>${desc}</td>
                <td class="hash-text">${shortHash}</td>
                <td><button class="btn-danger" onclick="deleteKey('${DOMPurify.sanitize(record.hash)}')">Revoke</button></td>
            </tr>
        `;
    });
    tbody.innerHTML = tbodyHtml;
}

/* eslint-disable-next-line no-unused-vars -- Called by: settings.html button onclick="openKeyModal()" */
function openKeyModal() {
    document.getElementById('new-key-name').value = '';
    document.getElementById('new-key-desc').value = '';
    document.getElementById('new-key-modal').style.display = 'flex';
}

function closeKeyModal() {
    document.getElementById('new-key-modal').style.display = 'none';
}

/* eslint-disable-next-line no-unused-vars -- Called by: settings.html modal button onclick="submitNewKey()" */
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

/* eslint-disable-next-line no-unused-vars -- Called by: settings.html button onclick="deleteKey('${hash}')" */
async function deleteKey(hash) {
    if(!confirm("Are you sure you want to permanently revoke this key? External apps using it will instantly fail.")) return;
    await fetchWithAuth(`/api/settings/keys/${hash}`, { method: 'DELETE' });
    loadKeys();
}

window.onload = loadKeys;