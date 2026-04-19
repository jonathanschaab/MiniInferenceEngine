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
    return params;
}