/**
 * Wrapper around standard fetch that automatically handles 401 Unauthorized redirects.
 */
async function fetchWithAuth(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok && response.status === 401) {
        window.location.href = '/auth/login';
        throw new Error('Unauthorized'); // Stop further execution in the caller
    }
    return response;
}

/**
 * Extracts generation parameters from the shared parameter UI panel.
 */
function getGenerationParameters() {
    const params = {
        temperature: parseFloat(document.getElementById('param-temp').value),
        top_p: parseFloat(document.getElementById('param-top-p').value),
        top_k: parseInt(document.getElementById('param-top-k').value),
        max_tokens: parseInt(document.getElementById('param-max-tokens').value),
        context_buffer: parseInt(document.getElementById('param-context-buffer').value) || 0
    };
    const seedVal = document.getElementById('param-seed').value;
    params.seed = seedVal !== "" ? parseInt(seedVal) : null;
    const memStrategyEl = document.getElementById('param-memory-strategy');
    if (memStrategyEl) params.memory_strategy = memStrategyEl.value;
    const yarnEnabledEl = document.getElementById('param-yarn-enabled');
    if (yarnEnabledEl) params.yarn_enabled = yarnEnabledEl.checked;
    return params;
}

/**
 * Dynamically injects the main navigation bar into the header and 
 * securely checks if the user is an admin to display the Console button.
 */
function injectNavbar() {
    const header = document.querySelector('header');
    if (!header) return;

    const navDiv = document.createElement('div');
    navDiv.style.display = 'flex';
    navDiv.style.gap = '15px';
    navDiv.style.alignItems = 'center';

    navDiv.innerHTML = `
        <button onclick="window.location.href='/'" style="background: #89b4fa; height: 30px; padding: 0 10px;">💬 Chat</button>
        <button onclick="window.location.href='/models'" style="background: #fab387; height: 30px; padding: 0 10px;">🤖 Models</button>
        <button onclick="window.location.href='/memory'" style="background: #f9e2af; height: 30px; padding: 0 10px;">💾 Memory</button>
        <button onclick="window.location.href='/stats'" style="background: #cba6f7; height: 30px; padding: 0 10px;">📊 Stats</button>
        <button id="nav-console-btn" onclick="window.location.href='/console'" style="background: #89dceb; height: 30px; padding: 0 10px; display: none;">🖥️ Console</button>
        <button onclick="window.location.href='/settings'" style="background: #a6e3a1; height: 30px; padding: 0 10px;">⚙️ Settings</button>
        <button onclick="window.location.href='/auth/logout'" style="background: #45475a; color: white; height: 30px; padding: 0 10px;">Logout</button>
    `;

    header.appendChild(navDiv);

    fetch('/api/console/loglevel')
        .then(res => { if (res.ok) document.getElementById('nav-console-btn').style.display = 'inline-flex'; })
        .catch(e => console.debug("Console authorization check failed", e));
}

if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', injectNavbar); } else { injectNavbar(); }