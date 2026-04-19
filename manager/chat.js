const chatContainer = document.getElementById('chat-container');
const inputField = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const regenBtn = document.getElementById('regen-btn');
const typingIndicator = document.getElementById('typing-indicator');

let chatHistory = [];
let currentAbortController = null;

window.onload = initializeUI;

async function updateStatus() {
    try {
        const res = await fetchWithAuth('/api/status');
        const status = await res.json();
        if (status.active_backend) {
            document.getElementById('engine-status-indicator').textContent = `(${status.active_backend})`;
        }
    } catch (e) {
        console.warn("Failed to update engine status", e);
    }
}

async function initializeUI() {
    try {
        const results = await Promise.allSettled([
            fetchWithAuth('/api/status').then(res => res.json()),
            fetchWithAuth('/api/models').then(res => res.json())
        ]);

        const statusResult = results[0];
        const modelsResult = results[1];

        let models = [];
        if (modelsResult.status === 'fulfilled') {
            models = modelsResult.value;
        } else {
            console.error("Critical: Failed to fetch models.", modelsResult.reason);
            appendMessage("System Error: Could not connect to the model registry.", false);
            return; 
        }

        let status = {};
        if (statusResult.status === 'fulfilled') {
            status = statusResult.value;
        } else {
            console.warn("Engine status unavailable. Falling back to defaults.");
        }

        const chatSelect = document.getElementById('chat-model-select');
        const compSelect = document.getElementById('compressor-model-select');
        const backendSelect = document.getElementById('backend-select');

        chatSelect.innerHTML = '';
        compSelect.innerHTML = '';
        backendSelect.innerHTML = '';
        backendSelect.add(new Option('Backend: Auto', ''));

        // Collect all unique backends from the model registry
        const allBackends = new Set();
        models.forEach(m => {
            m.supported_backends.forEach(b => allBackends.add(b));
        });

        // Populate the dropdown
        allBackends.forEach(b => backendSelect.add(new Option(b, b.toLowerCase())));

        models.forEach(m => {
            if (m.roles.includes("GeneralChat") || m.roles.includes("CodeSpecialist")) {
                let opt = new Option(`Chat: ${m.name}`, m.id);
                opt.dataset.backends = m.supported_backends.map(b => b.toLowerCase()).join(',');
                chatSelect.add(opt);
            }
            if (m.roles.includes("ContextCompressor")) {
                let opt = new Option(`Compressor: ${m.name}`, m.id);
                opt.dataset.backends = m.supported_backends.map(b => b.toLowerCase()).join(',');
                compSelect.add(opt);
            }
        });

        if (status.active_chat_model_id) {
            chatSelect.value = status.active_chat_model_id;
        } else {
            const defChat = models.find(m => m.is_default_chat);
            if (defChat) chatSelect.value = defChat.id;
        }
        if (status.last_compressor_model_id) {
            compSelect.value = status.last_compressor_model_id;
        } else {
            const defComp = models.find(m => m.is_default_compressor);
            if (defComp) compSelect.value = defComp.id;
        }

        chatSelect.addEventListener('change', updateDropdownCompatibility);
        backendSelect.addEventListener('change', updateDropdownCompatibility);

        updateDropdownCompatibility();

        updateStatus();
    } catch (err) {
        console.error("Failed to execute UI initialization:", err);
    }
}

function updateDropdownCompatibility() {
    const chatSelect = document.getElementById('chat-model-select');
    const backendSelect = document.getElementById('backend-select');

    const selectedBackend = backendSelect.value;
    
    // 1. Filter models based on selected backend (Only Chat models, Compressor fallbacks are handled by the orchestrator)
    if (selectedBackend) {
        Array.from(chatSelect.options).forEach(opt => {
            const supported = opt.dataset.backends && opt.dataset.backends.split(',').includes(selectedBackend);
            opt.disabled = !supported;
        });
        if (chatSelect.options[chatSelect.selectedIndex]?.disabled) {
            chatSelect.value = Array.from(chatSelect.options).find(o => !o.disabled)?.value || '';
        }
    } else {
        Array.from(chatSelect.options).forEach(opt => opt.disabled = false);
    }

    // 2. Filter backends based on selected Chat model
    const chatOpt = chatSelect.options[chatSelect.selectedIndex];
    
    const chatBackends = chatOpt && chatOpt.dataset.backends ? chatOpt.dataset.backends.split(',') : [];

    Array.from(backendSelect.options).forEach(opt => {
        if (opt.value === '') { opt.disabled = false; return; } // Auto is always allowed
        
        let supported = chatOpt && chatBackends.includes(opt.value);
        
        opt.disabled = !supported;
    });
    if (backendSelect.options[backendSelect.selectedIndex]?.disabled) {
        backendSelect.value = ''; // Fallback to Auto
    }
}

inputField.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

function clearChat() {
    chatHistory = [];
    // Remove all messages except the first AI greeting
    regenBtn.style.display = 'none';
    const messages = chatContainer.querySelectorAll('.message');
    for (let i = 1; i < messages.length; i++) {
        chatContainer.removeChild(messages[i]);
    }
}

function appendMessage(text, isUser) {
    const div = document.createElement('div');
    div.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
    div.textContent = text;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function sendMessage() {
    const text = inputField.value.trim();
    if (!text) return;

    // Grab the IDs from the UI dropdowns
    const chatModelId = document.getElementById('chat-model-select').value;
    const compModelId = document.getElementById('compressor-model-select').value;
    const targetBackend = document.getElementById('backend-select').value;

    if (!chatModelId || !compModelId) {
        alert("Please select a valid chat model and compressor model. Check your backend compatibility if options are disabled.");
        return;
    }

    appendMessage(text, true);
    chatHistory.push({ role: "user", content: text });
    
    inputField.value = '';
    inputField.style.height = 'auto'; 
    sendBtn.disabled = true;
    sendBtn.style.display = 'none';
    stopBtn.style.display = 'inline-flex';
    regenBtn.style.display = 'none';
    typingIndicator.style.display = 'block';
    
    currentAbortController = new AbortController();

    let aiMessageDiv = document.createElement('div');
    aiMessageDiv.className = 'message ai-message';
    chatContainer.appendChild(aiMessageDiv);

    try {
        // Grab parameters
        const parameters = getGenerationParameters();

        const response = await fetchWithAuth('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                chat_model_id: chatModelId,
                compressor_model_id: compModelId,
                messages: chatHistory,
                parameters: parameters,
                target_backend: targetBackend !== "" ? targetBackend : null
            }),
            signal: currentAbortController.signal
        });
        
        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }

        typingIndicator.style.display = 'none';
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullAnswer = "";
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                fullAnswer += decoder.decode();
                aiMessageDiv.textContent = fullAnswer;
                break;
            }
            fullAnswer += decoder.decode(value, { stream: true });
            aiMessageDiv.textContent = fullAnswer;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        chatHistory.push({ role: "assistant", content: fullAnswer });
        updateStatus();

    } catch (err) {
        if (err.name === 'AbortError') {
            aiMessageDiv.textContent += " [Stopped]";
            chatHistory.push({ role: "assistant", content: aiMessageDiv.textContent });
            updateStatus();
        } else {
            typingIndicator.style.display = 'none';
            aiMessageDiv.textContent = "Error: Failed to connect to engine.";
            chatHistory.pop(); 
        }
    }
    
    sendBtn.style.display = 'inline-flex';
    stopBtn.style.display = 'none';
    if (chatHistory.length > 0) regenBtn.style.display = 'inline-flex';
    sendBtn.disabled = false;
    inputField.focus();
    currentAbortController = null;
}

function regenerateLast() {
    if (chatHistory.length === 0) return;
    
    // Remove the assistant's last message
    if (chatHistory[chatHistory.length - 1].role === 'assistant') {
        chatHistory.pop();
        chatContainer.removeChild(chatContainer.lastChild);
    }
    
    // Remove the user's last message, place it in the input box, and immediately re-send
    if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'user') {
        const lastUserMsg = chatHistory.pop();
        chatContainer.removeChild(chatContainer.lastChild);
        inputField.value = lastUserMsg.content;
        sendMessage();
    }
}

sendBtn.addEventListener('click', sendMessage);
stopBtn.addEventListener('click', () => {
    if (currentAbortController) currentAbortController.abort();
});
regenBtn.addEventListener('click', regenerateLast);

inputField.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});