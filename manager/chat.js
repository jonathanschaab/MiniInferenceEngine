/* global downloadModel */

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
const MESSAGE_LIMIT = 50;
let hasMoreMessages = false;
let isLoadingMessages = false;
const SESSION_LIMIT = 20;
let sessionOffset = 0;
let hasMoreSessions = false;
let isLoadingSessions = false;
let allModels = [];

const chatScrollObserver = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && hasMoreMessages && !isLoadingMessages && currentSessionId) {
        fetchMoreMessages(currentSessionId, false);
    }
}, {
    root: chatContainer,
    rootMargin: '150px', // Trigger the fetch slightly before the user hits the very top
    threshold: 0
});

const sessionScrollObserver = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && hasMoreSessions && !isLoadingSessions) {
        fetchMoreSessions(false);
    }
}, {
    root: document.getElementById('session-list'),
    rootMargin: '100px',
    threshold: 0
});

window.onload = initializeUI;

function showRenameModal(currentTitle) {
    return new Promise((resolve) => {
        const modal = document.getElementById('rename-modal');
        const input = document.getElementById('rename-input');
        const confirmBtn = document.getElementById('rename-confirm-btn');
        const cancelBtn = document.getElementById('rename-cancel-btn');
        
        input.value = currentTitle || "";
        modal.style.display = 'flex';
        setTimeout(() => input.focus(), 10);

        const cleanup = () => {
            modal.style.display = 'none';
            confirmBtn.onclick = null;
            cancelBtn.onclick = null;
            input.onkeydown = null;
        };

        confirmBtn.onclick = () => {
            const val = input.value;
            cleanup();
            resolve(val);
        };

        cancelBtn.onclick = () => {
            cleanup();
            resolve(null);
        };

        input.onkeydown = (e) => {
            if (e.key === 'Enter') confirmBtn.onclick();
            if (e.key === 'Escape') cancelBtn.onclick();
        };
    });
}

function showDeleteModal() {
    return new Promise((resolve) => {
        const modal = document.getElementById('delete-modal');
        const confirmBtn = document.getElementById('delete-confirm-btn');
        const cancelBtn = document.getElementById('delete-cancel-btn');
        
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
            allModels = models;
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
            opt.title = supported ? '' : 'Incompatible with selected backend';
        });
    } else {
        Array.from(chatSelect.options).forEach(opt => {
            opt.disabled = false;
            opt.title = '';
        });
    }

    // 2. Filter backends based on selected Chat model
    const chatOpt = chatSelect.selectedIndex >= 0 ? chatSelect.options[chatSelect.selectedIndex] : null;
    
    const chatBackends = chatOpt && chatOpt.dataset.backends ? chatOpt.dataset.backends.split(',') : [];

    Array.from(backendSelect.options).forEach(opt => {
        if (opt.value === '') { 
            opt.disabled = false; 
            opt.title = '';
            return; 
        } // Auto is always allowed
        
        let supported = chatOpt && chatBackends.includes(opt.value);
        
        opt.disabled = !supported;
        opt.title = supported ? '' : 'Incompatible with selected chat model';
    });
}

async function loadSessions() {
    sessionOffset = 0;
    hasMoreSessions = true;
    document.getElementById('session-list').innerHTML = '';
    await fetchMoreSessions(true);
}

function updateActiveSessionClass() {
    document.querySelectorAll('.session-item').forEach(el => {
        if (el.dataset.id === currentSessionId) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });
}

async function fetchMoreSessions(isInitialLoad = false) {
    if (isLoadingSessions || !hasMoreSessions) return;
    isLoadingSessions = true;

    try {
        const res = await fetchWithAuth(`/api/chat/sessions?limit=${SESSION_LIMIT}&offset=${sessionOffset}`);
        const sessions = await res.json();
        
        if (isInitialLoad && !currentSessionId) {
            const lastId = localStorage.getItem('mini_inference_last_chat_id');
            if (lastId) {
                const success = await loadSession(lastId, true);
                if (!success) {
                    localStorage.removeItem('mini_inference_last_chat_id');
                    startNewSession();
                }
            }
        }

        hasMoreSessions = sessions.length === SESSION_LIMIT;
        sessionOffset += sessions.length;

        renderSessionList(sessions);
    } catch(e) { console.error("Failed to load sessions:", e); } finally {
        isLoadingSessions = false;
    }
}

function renderSessionList(sessions) {
    sessionScrollObserver.disconnect();
    const list = document.getElementById('session-list');
    
    const existingSentinel = document.getElementById('session-sentinel');
    if (existingSentinel) existingSentinel.remove();

        sessions.forEach(s => {
            list.appendChild(createSessionElement(s));
        });

    if (hasMoreSessions) {
        const sentinel = document.createElement('div');
        sentinel.id = 'session-sentinel';
        sentinel.style.padding = '10px';
        sentinel.style.textAlign = 'center';
        sentinel.style.color = '#6c7086';
        sentinel.style.fontSize = '0.85rem';
        sentinel.textContent = 'Loading more...';
        list.appendChild(sentinel);
        sessionScrollObserver.observe(sentinel);
    }
}

function createSessionElement(s) {
    const div = document.createElement('div');
    div.className = `session-item ${s.id === currentSessionId ? 'active' : ''}`;
    div.dataset.id = s.id;
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
        const newTitle = await showRenameModal(s.title);
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
        const confirmed = await showDeleteModal();
        if (confirmed) {
            await fetchWithAuth(`/api/chat/sessions/${s.id}`, { method: 'DELETE' });
            if (currentSessionId === s.id) startNewSession();
            loadSessions();
        }
    };

    actionsDiv.appendChild(editBtn);
    actionsDiv.appendChild(delBtn);
    div.appendChild(infoDiv);
    div.appendChild(actionsDiv);
    return div;
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
            const sessionEl = document.querySelector(`.session-item[data-id="${id}"]`);
            if (sessionEl) {
                const titleDiv = sessionEl.querySelector('.session-title');
                if (titleDiv) titleDiv.textContent = newTitle;
            }
        } else {
            console.error("Failed to rename session");
        }
    } catch(e) { console.error("Error renaming session:", e); }
}

async function loadSession(id, skipSessionListUpdate = false) {
    chatScrollObserver.disconnect();
    chatHistory = [];
    hasMoreMessages = true;
    chatContainer.innerHTML = '';
    regenBtn.style.display = 'none';
    
    const success = await fetchMoreMessages(id, true);
    
    if (success && !skipSessionListUpdate) {
        updateActiveSessionClass();
    }
    return success;
}

async function fetchMoreMessages(id, isInitialLoad = false) {
    if (isLoadingMessages || !hasMoreMessages) return false;
    isLoadingMessages = true;
    
    try {
        const offset = chatHistory.length;
        const res = await fetchWithAuth(`/api/chat/sessions/${id}?limit=${MESSAGE_LIMIT}&offset=${offset}`);
        if (!res.ok) throw new Error("Failed to fetch session messages");
        const session = await res.json();
        
        if (isInitialLoad) {
            currentSessionId = session.id;
            currentSessionTitle = session.title;
            localStorage.setItem('mini_inference_last_chat_id', currentSessionId);
        }
        
        const fetchedMessages = session.messages || [];
        hasMoreMessages = fetchedMessages.length === MESSAGE_LIMIT;
        
        // Prepend older messages
        chatHistory = [...fetchedMessages, ...chatHistory];
        
        const oldScrollHeight = chatContainer.scrollHeight;
        
        prependMessagesToUI(fetchedMessages);
        
        if (isInitialLoad) {
            requestAnimationFrame(() => {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            });
        } else {
            // Maintain smooth scrolling point when older messages are injected
            requestAnimationFrame(() => {
                chatContainer.scrollTop = chatContainer.scrollHeight - oldScrollHeight;
            });
        }
        
        if (chatHistory.length > 0) regenBtn.style.display = 'inline-flex';
        return true;
        
    } catch(e) { 
        console.error("Failed to fetch messages:", e); 
        hasMoreMessages = false;
        return false;
    } finally {
        isLoadingMessages = false;
    }
}

function prependMessagesToUI(messages) {
    chatScrollObserver.disconnect();
    
    const existingSentinel = document.getElementById('chat-sentinel');
    if (existingSentinel) existingSentinel.remove();
    
    const fragment = document.createDocumentFragment();
    
    if (hasMoreMessages) {
        const sentinel = document.createElement('div');
        sentinel.id = 'chat-sentinel';
        sentinel.style.padding = '10px';
        sentinel.style.textAlign = 'center';
        sentinel.style.color = '#6c7086';
        sentinel.style.fontSize = '0.9rem';
        sentinel.textContent = 'Loading older messages...';
        fragment.appendChild(sentinel);
        chatScrollObserver.observe(sentinel);
    }
    
    messages.forEach(msg => {
        const div = document.createElement('div');
        div.className = `message ${msg.role === 'user' ? 'user-message' : 'ai-message'}`;
        div.textContent = msg.content;
        fragment.appendChild(div);
    });
    
    chatContainer.prepend(fragment);
}

window.startNewSession = function() {
    chatScrollObserver.disconnect();
    currentSessionId = "";
    currentSessionTitle = "";
    localStorage.removeItem('mini_inference_last_chat_id');
    chatHistory = [];
    hasMoreMessages = false;
    chatContainer.innerHTML = '<div class="message ai-message">System: New chat session started. How can I help you?</div>';
    regenBtn.style.display = 'none';
    updateActiveSessionClass();
}

inputField.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

window.clearChat = async function() {
    if (chatHistory.length === 0) {
        startNewSession();
        return;
    }
    
    const confirmed = await showDeleteModal();
    if (confirmed) {
        if (currentSessionId) {
            await fetchWithAuth(`/api/chat/sessions/${currentSessionId}`, { method: 'DELETE' });
        }
        startNewSession();
        loadSessions();
    }
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
        const newSessionEl = createSessionElement(saved);
        document.getElementById('session-list').prepend(newSessionEl);
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

async function startChatDownload(modelId, modelName) {
    const div = document.createElement('div');
    div.className = 'message ai-message';
    div.innerHTML = `
        <div>Downloading <strong>${DOMPurify.sanitize(modelName)}</strong>...</div>
        <div class="download-progress-container" style="margin-top: 10px;">
            <div style="width: 100%; max-width: 300px; background: #313244; border-radius: 4px; overflow: hidden; border: 1px solid #45475a;">
                <div id="dl-bar-${modelId}" style="width: 0%; height: 8px; background: #a6e3a1; transition: width 0.5s ease-out;"></div>
            </div>
                <div style="display: flex; justify-content: space-between; max-width: 300px; align-items: center; margin-top: 5px;">
                    <div id="dl-stats-${modelId}" style="font-size: 0.75rem; color: #a6adc8;">Starting...</div>
                    <button id="dl-cancel-${modelId}" style="padding: 4px 8px; background: #f38ba8; color: #11111b; border: none; border-radius: 4px; cursor: pointer; font-size: 0.75rem; font-weight: bold;">Cancel</button>
                </div>
        </div>
    `;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    const cancelBtn = document.getElementById(`dl-cancel-${modelId}`);
    const dl = downloadModel(modelId, {
        onProgress: (status, pct, speedMB, transMB, totalMB, etaStr) => {
            const bar = document.getElementById(`dl-bar-${modelId}`);
            const stats = document.getElementById(`dl-stats-${modelId}`);
            if (bar) bar.style.width = `${pct}%`;
            if (stats) {
                if (status.state === 'Verifying...') {
                    stats.innerText = `${pct.toFixed(1)}% (${transMB} / ${totalMB} MB) | Verifying...`;
                } else {
                    stats.innerText = `${pct.toFixed(1)}% (${transMB} / ${totalMB} MB) @ ${speedMB} MB/s | ETA: ${etaStr}`;
                }
            }
        },
        onStatusText: (text) => {
            const stats = document.getElementById(`dl-stats-${modelId}`);
            if (stats) stats.innerText = text;
        },
        onComplete: () => {
            const bar = document.getElementById(`dl-bar-${modelId}`);
            const stats = document.getElementById(`dl-stats-${modelId}`);
            if (bar) bar.style.width = '100%';
            if (stats) stats.innerText = 'Download Complete!';
            if (cancelBtn) cancelBtn.style.display = 'none';
        }
    });

    cancelBtn.addEventListener('click', () => {
        dl.cancel();
    });

    await dl.promise;
}

/**
 * Resets the chat UI state after generation or upon failure.
 */
function resetChatUI() {
    typingIndicator.style.display = 'none';
    sendBtn.style.display = 'inline-flex';
    stopBtn.style.display = 'none';
    if (chatHistory.length > 0) regenBtn.style.display = 'inline-flex';
    sendBtn.disabled = false;
}

async function sendMessage() {
    const text = inputField.value.trim();
    if (!text) return;

    await ensureSession(text);

    const userIndex = chatHistory.length;
    appendMessage(text, true);
    chatHistory.push({ role: "user", content: text });
    await appendMessageToDB("user", text, userIndex);
    
    inputField.value = '';
    inputField.style.height = 'auto'; 

    await requestAiResponse();
}

async function requestAiResponse() {
    // Grab the IDs from the UI dropdowns
    const chatSelect = document.getElementById('chat-model-select');
    const backendSelect = document.getElementById('backend-select');
    const chatModelId = chatSelect.value;
    const compModelId = document.getElementById('compressor-model-select').value;

    const targetBackend = backendSelect.value;

    if (!chatModelId || !compModelId || 
        chatSelect.options[chatSelect.selectedIndex]?.disabled || 
        backendSelect.options[backendSelect.selectedIndex]?.disabled) {
        alert("Please select a valid chat model and compressor model. Check your backend compatibility if options are disabled.");
        return;
    }

    const chatModel = allModels.find(m => m.id === chatModelId);
    const compModel = allModels.find(m => m.id === compModelId);

    sendBtn.disabled = true;
    sendBtn.style.display = 'none';
    stopBtn.style.display = 'inline-flex';
    regenBtn.style.display = 'none';
    typingIndicator.style.display = 'block';
    
    if (chatModel && !chatModel.is_downloaded) {
        try { await startChatDownload(chatModel.id, chatModel.name); chatModel.is_downloaded = true; } 
        catch (e) {
            console.error("Chat model download failed:", e);
            resetChatUI();
            return;
        }
    }
    if (compModel && !compModel.is_downloaded) {
        try { await startChatDownload(compModel.id, compModel.name); compModel.is_downloaded = true; } 
        catch (e) {
            console.error("Compressor model download failed:", e);
            resetChatUI();
            return;
        }
    }
    
    currentAbortController = new AbortController();

    const generatingSessionId = currentSessionId;
    const generatingSessionEl = document.querySelector(`.session-item[data-id="${generatingSessionId}"]`);
    if (generatingSessionEl) generatingSessionEl.classList.add('generating');

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
            aiMessageDiv.textContent = "Error: Failed to connect to engine.";
            chatHistory.pop(); 
        }
    }
    
    resetChatUI();
    inputField.focus();
    currentAbortController = null;
    
    // Move the active session to the top of the list and update its timestamp
    const generatingSessionElAfter = document.querySelector(`.session-item[data-id="${generatingSessionId}"]`);
    if (generatingSessionElAfter) {
        document.getElementById('session-list').prepend(generatingSessionElAfter);
        const dateDiv = generatingSessionElAfter.querySelector('.session-date');
        if (dateDiv) {
            dateDiv.textContent = new Date().toLocaleString([], { dateStyle: 'short', timeStyle: 'short' });
        }
        generatingSessionElAfter.classList.remove('generating');
    }
}

async function regenerateLast() {
    if (chatHistory.length === 0) return;
    
    // Remove the assistant's last message
    if (chatHistory[chatHistory.length - 1].role === 'assistant') {
        chatHistory.pop();
        chatContainer.removeChild(chatContainer.lastChild);
        await truncateMessagesInDB(chatHistory.length);
    }
    
    // If the last remaining message is from the user, request a new response
    if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'user') {
        await requestAiResponse();
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