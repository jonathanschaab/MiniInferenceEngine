const chatContainer = document.getElementById('chat-container');
const inputField = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const regenBtn = document.getElementById('regen-btn');
const typingIndicator = document.getElementById('typing-indicator');

let chatHistory = [];
let currentAbortController = null;
let currentSessionId = "";
let currentSessionTitle = "";

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
        
        await loadSessions();
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
    const chatOpt = chatSelect.selectedIndex >= 0 ? chatSelect.options[chatSelect.selectedIndex] : null;
    
    const chatBackends = chatOpt && chatOpt.dataset.backends ? chatOpt.dataset.backends.split(',') : [];

    Array.from(backendSelect.options).forEach(opt => {
        if (opt.value === '') { opt.disabled = false; return; } // Auto is always allowed
        
        let supported = chatOpt && chatBackends.includes(opt.value);
        
        opt.disabled = !supported;
    });
    if (backendSelect.options[backendSelect.selectedIndex]?.disabled) {
        backendSelect.value = ''; // Fallback to Auto
        Array.from(chatSelect.options).forEach(opt => opt.disabled = false);
    }
}

async function loadSessions() {
    try {
        const res = await fetchWithAuth('/api/chat/sessions');
        const sessions = await res.json();
        
        if (!currentSessionId) {
            const lastId = localStorage.getItem('mini_inference_last_chat_id');
            if (lastId && sessions.some(s => s.id === lastId)) {
                await loadSession(lastId);
                return; 
            } else if (lastId) {
                localStorage.removeItem('mini_inference_last_chat_id');
            }
        }

        const list = document.getElementById('session-list');
        list.innerHTML = '';
        sessions.forEach(s => {
            const div = document.createElement('div');
            div.className = `session-item ${s.id === currentSessionId ? 'active' : ''}`;
            div.onclick = () => loadSession(s.id);
            
            const infoDiv = document.createElement('div');
            infoDiv.className = 'session-info';

            const title = document.createElement('div');
            title.className = 'session-title';
            title.textContent = s.title || "Untitled Chat";
            
            const dateStr = s.updated_at ? new Date(s.updated_at * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' }) : '';
            const dateDiv = document.createElement('div');
            dateDiv.className = 'session-date';
            dateDiv.textContent = dateStr;

            infoDiv.appendChild(title);
            infoDiv.appendChild(dateDiv);

            const actionsDiv = document.createElement('div');
            actionsDiv.style.whiteSpace = 'nowrap';

            const editBtn = document.createElement('button');
            editBtn.className = 'session-action-btn';
            editBtn.textContent = '✎';
            editBtn.title = "Rename Chat";
            editBtn.onclick = async (e) => {
                e.stopPropagation();
                const newTitle = prompt("Enter new chat name:", s.title);
                if (newTitle && newTitle.trim() !== "" && newTitle !== s.title) {
                    await renameSession(s.id, newTitle.trim());
                }
            };

            const delBtn = document.createElement('button');
            delBtn.className = 'session-action-btn delete-btn';
            delBtn.textContent = '×';
            delBtn.title = "Delete Chat";
            delBtn.onclick = async (e) => {
                e.stopPropagation();
                if (confirm("Delete this chat?")) {
                    await fetchWithAuth(`/api/chat/sessions/${s.id}`, { method: 'DELETE' });
                    if (currentSessionId === s.id) startNewSession();
                    else loadSessions();
                }
            };

            actionsDiv.appendChild(editBtn);
            actionsDiv.appendChild(delBtn);
            div.appendChild(infoDiv);
            div.appendChild(actionsDiv);
            list.appendChild(div);
        });
    } catch(e) { console.error("Failed to load sessions:", e); }
}

async function renameSession(id, newTitle) {
    try {
        const res = await fetchWithAuth('/api/chat/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                id: id,
                title: newTitle
            })
        });
        if (res.ok) {
            if (id === currentSessionId) {
                currentSessionTitle = newTitle;
            }
            loadSessions();
        } else {
            console.error("Failed to rename session");
        }
    } catch(e) { console.error("Error renaming session:", e); }
}

async function loadSession(id) {
    try {
        const res = await fetchWithAuth(`/api/chat/sessions/${id}`);
        const session = await res.json();
        currentSessionId = session.id;
        currentSessionTitle = session.title;
        localStorage.setItem('mini_inference_last_chat_id', currentSessionId);
        chatHistory = session.messages || [];
        
        chatContainer.innerHTML = '';
        regenBtn.style.display = 'none';
        
        chatHistory.forEach(msg => {
            appendMessage(msg.content, msg.role === 'user');
        });
        
        // Ensure we scroll to the bottom after the browser renders the messages
        requestAnimationFrame(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        });
        
        if (chatHistory.length > 0) regenBtn.style.display = 'inline-flex';
        
        loadSessions(); // update active class
    } catch(e) { console.error("Failed to load session:", e); }
}

window.startNewSession = function() {
    currentSessionId = "";
    currentSessionTitle = "";
    localStorage.removeItem('mini_inference_last_chat_id');
    chatHistory = [];
    chatContainer.innerHTML = '<div class="message ai-message">System: New chat session started. How can I help you?</div>';
    regenBtn.style.display = 'none';
    loadSessions();
}

inputField.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

window.clearChat = function() {
    startNewSession();
}

function appendMessage(text, isUser) {
    const div = document.createElement('div');
    div.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
    div.textContent = text;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function ensureSession(firstMessageText) {
    if (currentSessionId) return;
    currentSessionTitle = firstMessageText ? firstMessageText.substring(0, 30) : "New Chat";
    try {
        const res = await fetchWithAuth('/api/chat/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: currentSessionTitle
            })
        });
        const saved = await res.json();
        currentSessionId = saved.id;
        localStorage.setItem('mini_inference_last_chat_id', currentSessionId);
        loadSessions();
    } catch(e) { console.error("Failed to create session", e); }
}

async function appendMessageToDB(role, content, index) {
    if (!currentSessionId) return;
    try {
        await fetchWithAuth(`/api/chat/sessions/${currentSessionId}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentSessionId,
                message_index: index,
                role: role,
                content: content
            })
        });
        loadSessions(); // Updates the updated_at timestamp in sidebar
    } catch(e) { console.error("Failed to append message", e); }
}

async function truncateMessagesInDB(fromIndex) {
    if (!currentSessionId) return;
    try {
        await fetchWithAuth(`/api/chat/sessions/${currentSessionId}/messages/${fromIndex}`, {
            method: 'DELETE'
        });
    } catch(e) { console.error("Failed to truncate messages", e); }
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

    await ensureSession(text);

    const userIndex = chatHistory.length;
    appendMessage(text, true);
    chatHistory.push({ role: "user", content: text });
    await appendMessageToDB("user", text, userIndex);
    
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
        
        const aiIndex = chatHistory.length;
        chatHistory.push({ role: "assistant", content: fullAnswer });
        updateStatus();
        await appendMessageToDB("assistant", fullAnswer, aiIndex);

    } catch (err) {
        if (err.name === 'AbortError') {
            aiMessageDiv.textContent += " [Stopped]";
            const aiIndex = chatHistory.length;
            chatHistory.push({ role: "assistant", content: aiMessageDiv.textContent });
            updateStatus();
            await appendMessageToDB("assistant", aiMessageDiv.textContent, aiIndex);
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

async function regenerateLast() {
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
        await truncateMessagesInDB(chatHistory.length);
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