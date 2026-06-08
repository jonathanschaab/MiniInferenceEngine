const chatContainer = document.getElementById('chat-container');
const inputField = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const regenBtn = document.getElementById('regen-btn');
const typingIndicator = document.getElementById('typing-indicator');

let chatHistory = [];
let messagesMap = new Map();
let currentLeafId = null;
let splitViewParentId = null;
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
    rootMargin: '150px', // Trigger slightly before the user hits the very top
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

function generateUUID() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

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
        const indicator = document.getElementById('engine-status-indicator');
        if (status.loading_model_id) {
            indicator.textContent = `(Loading...)`;
            indicator.className = 'status text-yellow';
        } else if (status.active_backend) {
            indicator.textContent = `(${status.active_backend})`;
            indicator.className = 'status text-green';
        } else {
            indicator.textContent = '';
            indicator.className = 'status';
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
        sentinel.className = 'session-sentinel';
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
    actionsDiv.className = 'whitespace-nowrap';

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
    messagesMap.clear();
    currentLeafId = null;
    splitViewParentId = null;
    window.branchingParentId = undefined;
    hasMoreMessages = true;
    isLoadingMessages = false;
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
        const offset = messagesMap.size;
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
        
        const oldScrollHeight = chatContainer.scrollHeight;

        buildTree(fetchedMessages, isInitialLoad);
        renderActivePath();

        if (isInitialLoad) {
            requestAnimationFrame(() => {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            });
        } else {
            requestAnimationFrame(() => {
                chatContainer.scrollTop = chatContainer.scrollHeight - oldScrollHeight;
            });
        }

        return true;
        
    } catch(e) { 
        console.error("Failed to fetch messages:", e); 
        return false;
    } finally {
        isLoadingMessages = false;
    }
}

function buildTree(fetchedMessages, isInitialLoad) {
    let rootNodes = [];
    
    fetchedMessages.forEach(msg => {
        if (!msg.metadata) msg.metadata = {};
        if (!msg.metadata.id) msg.metadata.id = generateUUID();
    });

    // Legacy linear history stitching (Legacy IDs contained underscores: sessionID_index)
    let lastLegacyId = null;
    fetchedMessages.forEach((msg) => {
        const isLegacy = msg.metadata.id.includes('_');
        if (isLegacy) {
            if (msg.metadata.parent_id == null && lastLegacyId !== null) {
                msg.metadata.parent_id = lastLegacyId;
            }
            lastLegacyId = msg.metadata.id;
        }
    });
    
    if (!isInitialLoad && fetchedMessages.length > 0 && messagesMap.size > 0) {
        let earliestExisting = Array.from(messagesMap.values()).reduce((oldest, m) => 
            (!oldest || m.metadata.timestamp < oldest.metadata.timestamp) ? m : oldest
        , null);
        
        if (earliestExisting && earliestExisting.metadata.parent_id == null && earliestExisting.metadata.id.includes('_')) {
            let latestFetchedLegacy = null;
            for (let i = fetchedMessages.length - 1; i >= 0; i--) {
                if (fetchedMessages[i].metadata.id.includes('_')) {
                    latestFetchedLegacy = fetchedMessages[i];
                    break;
                }
            }
            if (latestFetchedLegacy) {
                earliestExisting.metadata.parent_id = latestFetchedLegacy.metadata.id;
            }
        }
    }

    fetchedMessages.forEach(msg => {
        if (!messagesMap.has(msg.metadata.id)) msg.children = [];
        messagesMap.set(msg.metadata.id, msg);
    });

    // Re-link all children recursively for safety as older chunks arrive
    Array.from(messagesMap.values()).forEach(m => m.children = []);
    
    Array.from(messagesMap.values()).forEach(msg => {
        if (msg.metadata.parent_id && messagesMap.has(msg.metadata.parent_id)) {
            messagesMap.get(msg.metadata.parent_id).children.push(msg.metadata.id);
        } else {
            rootNodes.push(msg.metadata.id);
        }
    });

    if (isInitialLoad || !currentLeafId) {
        currentLeafId = getDefaultLeaf(rootNodes);
    }
}

function getDefaultLeaf(nodeIds) {
    if (!nodeIds || nodeIds.length === 0) return null;

    function evaluateSubtree(id) {
        const node = messagesMap.get(id);
        if (!node) return { maxScore: -1, maxTime: -1, leafId: null };

        let maxScore = node.metadata.score || 0;
        let maxTime = node.metadata.timestamp || 0;
        let bestLeafId = null;

        if (node.children.length === 0) {
            bestLeafId = id;
        } else {
            let bestChildScore = -1;
            let bestChildTime = -1;
            
            node.children.forEach(childId => {
                const childEval = evaluateSubtree(childId);
                
                if (childEval.maxScore > maxScore) {
                    maxScore = childEval.maxScore;
                }
                if (childEval.maxTime > maxTime) {
                    maxTime = childEval.maxTime;
                }

                if (childEval.maxScore > bestChildScore) {
                    bestChildScore = childEval.maxScore;
                    bestChildTime = childEval.maxTime;
                    bestLeafId = childEval.leafId;
                } else if (childEval.maxScore === bestChildScore && childEval.maxTime >= bestChildTime) {
                    bestChildTime = childEval.maxTime;
                    bestLeafId = childEval.leafId;
                }
            });
        }
        return { maxScore, maxTime, leafId: bestLeafId };
    }

    let globalBestScore = -1;
    let globalBestTime = -1;
    let globalBestLeafId = null;

    nodeIds.forEach(id => {
        const ev = evaluateSubtree(id);
        if (ev.maxScore > globalBestScore) {
            globalBestScore = ev.maxScore;
            globalBestTime = ev.maxTime;
            globalBestLeafId = ev.leafId;
        } else if (ev.maxScore === globalBestScore && ev.maxTime >= globalBestTime) {
            globalBestTime = ev.maxTime;
            globalBestLeafId = ev.leafId;
        }
    });

    return globalBestLeafId;
}

function getActivePath(leafId) {
    let path = [];
    let curr = leafId;
    while (curr && messagesMap.has(curr)) {
        const node = messagesMap.get(curr);
        if (!node.metadata) node.metadata = { id: curr };
        path.unshift(node);
        curr = node.metadata.parent_id;
    }
    return path;
}

function toggleSplitView(siblings, parentId) {
    const key = parentId || 'root';
    if (splitViewParentId === key) {
        splitViewParentId = null;
    } else {
        splitViewParentId = key;
    }
    renderActivePath();
}

async function pruneBranch(msgId) {
    if (!confirm("Are you sure you want to delete this branch and all its descendants? This cannot be undone.")) return;

    const idsToDelete = [];
    function traverse(id) {
        idsToDelete.push(id);
        const node = messagesMap.get(id);
        if (node && node.children) {
            node.children.forEach(traverse);
        }
    }
    traverse(msgId);

    // Send to backend first to confirm deletion is allowed
    try {
        const res = await fetchWithAuth(`/api/chat/sessions/${currentSessionId}/messages`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message_ids: idsToDelete })
        });
        if (!res.ok) throw new Error("Failed to delete branch in DB");

        // Remove from local map only on success
        const targetNode = messagesMap.get(msgId);
        const parentId = targetNode ? targetNode.metadata.parent_id : null;
        
        if (parentId && messagesMap.has(parentId)) {
            const pNode = messagesMap.get(parentId);
            pNode.children = pNode.children.filter(id => id !== msgId);
        }

        idsToDelete.forEach(id => messagesMap.delete(id));

        // Select a new leaf
        let rootNodes = [];
        Array.from(messagesMap.values()).forEach(msg => {
            if (!msg.metadata.parent_id || !messagesMap.has(msg.metadata.parent_id)) {
                rootNodes.push(msg.metadata.id);
            }
        });

        currentLeafId = getDefaultLeaf(rootNodes);
        renderActivePath();
    } catch(e) {
        console.error("Failed to delete branch in DB:", e);
        alert("Failed to delete branch. Please check your connection or try refreshing the page.");
    }
}

function renderActivePath() {
    chatContainer.innerHTML = '';
    chatScrollObserver.disconnect();

    if (hasMoreMessages) {
        const sentinel = document.createElement('div');
        sentinel.id = 'chat-sentinel';
        sentinel.className = 'chat-sentinel';
        sentinel.textContent = 'Loading older messages...';
        chatContainer.appendChild(sentinel);
        chatScrollObserver.observe(sentinel);
    }

    const path = getActivePath(currentLeafId);
    chatHistory = path;
    
    path.forEach((msg) => {
        const wrapper = document.createElement('div');
        wrapper.id = `msg-wrapper-${msg.metadata.id}`;
        wrapper.style.display = 'flex';
        wrapper.style.flexDirection = 'column';
        wrapper.style.alignItems = msg.role === 'user' ? 'flex-end' : 'flex-start';
        wrapper.style.width = '100%';
        
        let siblings = [];
        if (msg.metadata.parent_id && messagesMap.has(msg.metadata.parent_id)) {
            siblings = messagesMap.get(msg.metadata.parent_id).children;
        } else {
            siblings = Array.from(messagesMap.values()).filter(m => !m.metadata.parent_id || !messagesMap.has(m.metadata.parent_id)).map(m => m.metadata.id);
        }

        const parentKey = msg.metadata.parent_id || 'root';

        if (siblings.length > 1) {
            const currentIndex = siblings.indexOf(msg.metadata.id);
            const branchCtrl = document.createElement('div');
            branchCtrl.className = 'branch-controls text-subtext0 text-085 mb-4';
            branchCtrl.style.alignSelf = msg.role === 'user' ? 'flex-end' : 'flex-start';
            
            const prevBtn = document.createElement('span');
            prevBtn.textContent = '◀';
            prevBtn.onclick = () => {
                const prevId = siblings[(currentIndex - 1 + siblings.length) % siblings.length];
                currentLeafId = getDefaultLeaf([prevId]);
                renderActivePath();
            };
            
            const nextBtn = document.createElement('span');
            nextBtn.textContent = '▶';
            nextBtn.onclick = () => {
                const nextId = siblings[(currentIndex + 1) % siblings.length];
                currentLeafId = getDefaultLeaf([nextId]);
                renderActivePath();
            };

            const expandBtn = document.createElement('span');
            expandBtn.textContent = splitViewParentId === parentKey ? ' ⊟ Collapse' : ' ◫ Split View';
            expandBtn.onclick = () => toggleSplitView(siblings, msg.metadata.parent_id);

            const pruneBtn = document.createElement('span');
            pruneBtn.innerHTML = '&nbsp; 🗑️ Prune';
            pruneBtn.title = 'Delete this branch and all its descendants';
            pruneBtn.onclick = () => pruneBranch(msg.metadata.id);

            branchCtrl.appendChild(prevBtn);
            branchCtrl.appendChild(document.createTextNode(` ${currentIndex + 1} of ${siblings.length} `));
            branchCtrl.appendChild(nextBtn);
            branchCtrl.appendChild(expandBtn);
            branchCtrl.appendChild(pruneBtn);
            wrapper.appendChild(branchCtrl);
        }

        if (splitViewParentId === parentKey && siblings.length > 1) {
            const splitContainer = document.createElement('div');
            splitContainer.className = 'split-container';
            
            siblings.forEach(sibId => {
                const sibMsg = messagesMap.get(sibId);
                const sibWrapper = document.createElement('div');
                sibWrapper.className = `split-item ${sibId === msg.metadata.id ? 'active-split' : ''}`;
                sibWrapper.onclick = () => {
                    splitViewParentId = null;
                    currentLeafId = getDefaultLeaf([sibId]);
                    renderActivePath();
                };

                const div = document.createElement('div');
                div.className = `message ${sibMsg.role === 'user' ? 'user-message' : 'ai-message'}`;
                div.style.maxWidth = '100%';
                div.textContent = sibMsg.content;
                sibWrapper.appendChild(div);
                
                const metaDiv = renderMetadataDiv(sibMsg);
                if (metaDiv) sibWrapper.appendChild(metaDiv);
                splitContainer.appendChild(sibWrapper);
            });
            wrapper.appendChild(splitContainer);
        } else {
            if (splitViewParentId === parentKey) splitViewParentId = null;
            
            const div = document.createElement('div');
            div.className = `message ${msg.role === 'user' ? 'user-message' : 'ai-message'} mb-15`;
            
            if (msg.role === 'user') {
                const textSpan = document.createElement('span');
                textSpan.textContent = msg.content;
                div.appendChild(textSpan);
                
                const editBtn = document.createElement('button');
                editBtn.className = 'session-action-btn ml-10';
                editBtn.style.display = 'inline-block';
                editBtn.textContent = '✎';
                editBtn.title = 'Edit Prompt (Creates New Branch)';
                editBtn.onclick = () => {
                    inputField.value = msg.content;
                    inputField.focus();
                    window.branchingParentId = msg.metadata.parent_id;
                };
                div.appendChild(editBtn);
            } else {
                div.textContent = msg.content;
            }
            wrapper.appendChild(div);
            
            const metaDiv = renderMetadataDiv(msg);
            if (metaDiv) wrapper.appendChild(metaDiv);
        }
        chatContainer.appendChild(wrapper);
    });

    if (path.length > 0) {
        regenBtn.style.display = 'inline-flex';
    } else {
        regenBtn.style.display = 'none';
    }
}

function renderMetadataDiv(msg) {
    if (!msg.metadata) return null;
    
    const metaDiv = document.createElement('div');
    metaDiv.className = 'msg-metadata text-subtext0 text-085 mt-10 mb-15';
    metaDiv.style.alignSelf = msg.role === 'user' ? 'flex-end' : 'flex-start';
    
    let tokenCountStr = Object.entries(msg.metadata.token_counts || {}).map(([m, c]) => `${m}: ${c}`).join(', ');
    
    if (msg.role === 'user') {
        metaDiv.innerHTML = `<strong>Tokens:</strong> ${tokenCountStr}`;
        const scoreSelect = createScoreSelect(msg);
        metaDiv.appendChild(scoreSelect);
    } else {
        let timeStr = msg.metadata.generation_time_ms ? (msg.metadata.generation_time_ms / 1000).toFixed(2) + 's' : 'N/A';
        let modelStr = msg.metadata.model || 'Unknown';
        metaDiv.innerHTML = `
            <strong>Model:</strong> ${modelStr} | 
            <strong>Time:</strong> ${timeStr} | 
            <strong>Tokens:</strong> ${tokenCountStr}
        `;
        
        const scoreSelect = createScoreSelect(msg);
        metaDiv.appendChild(scoreSelect);
    }
    return metaDiv;
}

function createScoreSelect(msg) {
    const scoreSelect = document.createElement('select');
    scoreSelect.className = 'control-select-bordered ml-10';
    scoreSelect.style.fontSize = '0.75rem';
    scoreSelect.style.padding = '2px';
    scoreSelect.innerHTML = `
        <option value="">Rate...</option>
        <option value="1">1 - Poor</option>
        <option value="2">2 - Fair</option>
        <option value="3">3 - Good</option>
        <option value="4">4 - Very Good</option>
        <option value="5">5 - Excellent</option>
    `;
    if (msg.metadata.score) {
        scoreSelect.value = msg.metadata.score;
    }
    scoreSelect.onchange = async () => {
        const previousScore = msg.metadata.score;
        msg.metadata.score = parseInt(scoreSelect.value) || null;
        const success = await appendMessageToDB(msg.role, msg.content, msg.metadata.id, msg.metadata);
        if (!success) {
            msg.metadata.score = previousScore;
            scoreSelect.value = previousScore || "";
            alert("Failed to save score. Please check your connection.");
        }
    };
    return scoreSelect;
}

function startNewSession() {
    currentSessionId = "";
    currentSessionTitle = "";
    localStorage.removeItem('mini_inference_last_chat_id');
    chatHistory = [];
    messagesMap.clear();
    currentLeafId = null;
    splitViewParentId = null;
    window.branchingParentId = undefined;
    chatContainer.innerHTML = '<div style="display: flex; flex-direction: column; align-items: flex-start;"><div class="message ai-message mb-15">System: New chat session started. How can I help you?</div></div>';
    regenBtn.style.display = 'none';
    updateActiveSessionClass();
}

inputField.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

async function clearChat() {
    if (messagesMap.size === 0) {
        startNewSession();
        return;
    }
    
    const confirmed = await showDeleteModal();
    if (confirmed) {
        if (currentSessionId) {
            try {
                const res = await fetchWithAuth(`/api/chat/sessions/${currentSessionId}`, { method: 'DELETE' });
                if (!res.ok) throw new Error("Failed to clear chat in DB");
            } catch (e) {
                console.error("Failed to clear chat in DB:", e);
                alert("Failed to delete chat session. Please check your connection or try again.");
                return;
            }
        }
        startNewSession();
        loadSessions();
    }
}

function appendMessage(text, isUser, index = null) {
    const wrapper = document.createElement('div');
    if (index !== null) wrapper.id = `msg-wrapper-${index}`;
    wrapper.style.display = 'flex';
    wrapper.style.flexDirection = 'column';
    wrapper.style.alignItems = isUser ? 'flex-end' : 'flex-start';
    
    const div = document.createElement('div');
    div.className = `message ${isUser ? 'user-message' : 'ai-message'} mb-15`;
    div.textContent = text;
    wrapper.appendChild(div);
    
    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return wrapper;
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

async function appendMessageToDB(role, content, id, metadata = null) {
    if (!currentSessionId) return false;
    try {
        const payload = {
            session_id: currentSessionId,
            id: id,
            parent_id: metadata ? metadata.parent_id : null,
            role: role,
            content: content
        };
        
        if (metadata) {
            payload.timestamp = metadata.timestamp;
            payload.model = metadata.model;
            payload.generation_time_ms = metadata.generation_time_ms;
            payload.token_counts = metadata.token_counts;
            payload.score = metadata.score;
            payload.parameters = metadata.parameters;
        }
        const res = await fetchWithAuth(`/api/chat/sessions/${currentSessionId}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return res.ok;
    } catch(e) { 
        console.error("Failed to append message", e); 
        return false;
    }
}

async function startChatDownload(modelId, modelName) {
    const div = document.createElement('div');
    div.className = 'message ai-message';

    const titleDiv = document.createElement('div');
    titleDiv.textContent = 'Downloading ';
    const strong = document.createElement('strong');
    strong.textContent = modelName;
    titleDiv.appendChild(strong);
    titleDiv.appendChild(document.createTextNode('...'));

    const containerDiv = document.createElement('div');
    containerDiv.className = 'download-progress-container';
    containerDiv.classList.add('mt-10');

    const barWrapper = document.createElement('div');
    barWrapper.className = 'dl-bar-wrapper';
    
    const bar = document.createElement('div');
    bar.id = `dl-bar-${modelId}`; // Direct property assignment is inherently safe from HTML breakouts
    bar.className = 'dl-bar';
    barWrapper.appendChild(bar);

    const statsRow = document.createElement('div');
    statsRow.className = 'dl-stats-row';

    const statsText = document.createElement('div');
    statsText.id = `dl-stats-${modelId}`;
    statsText.className = 'dl-stats-text';
    statsText.textContent = 'Starting...';

    const btnRow = document.createElement('div');
    btnRow.className = 'dl-btn-row';

    const pauseBtn = document.createElement('button');
    pauseBtn.id = `dl-pause-${modelId}`;
    pauseBtn.className = 'dl-pause-btn';
    pauseBtn.textContent = 'Pause';

    const cancelBtn = document.createElement('button');
    cancelBtn.id = `dl-cancel-${modelId}`;
    cancelBtn.className = 'dl-cancel-btn';
    cancelBtn.textContent = 'Cancel';

    btnRow.appendChild(pauseBtn);
    btnRow.appendChild(cancelBtn);
    statsRow.appendChild(statsText);
    statsRow.appendChild(btnRow);
    containerDiv.appendChild(barWrapper);
    containerDiv.appendChild(statsRow);
    
    div.appendChild(titleDiv);
    div.appendChild(containerDiv);

    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    const dl = downloadModel(modelId, {
        onProgress: (status, pct, speedMB, transMB, totalMB, etaStr) => {
            const bar = document.getElementById(`dl-bar-${modelId}`);
            const stats = document.getElementById(`dl-stats-${modelId}`);
            if (bar) bar.style.width = `${pct}%`;
            if (stats) {
                if (status.state === 'Queued...') {
                    stats.innerHTML = `<span class="text-yellow">Queued...</span> <a href="/queue" class="text-blue no-underline ml-5">(View Queue)</a>`;
                } else if (status.state !== 'Downloading...') {
                    stats.innerText = `${pct.toFixed(1)}% (${transMB} / ${totalMB} MB) | ${status.state}`;
                } else {
                    stats.innerText = `${pct.toFixed(1)}% (${transMB} / ${totalMB} MB) @ ${speedMB} MB/s | ETA: ${etaStr}`;
                }
            }
        },
        onStatusText: (text) => {
            const stats = document.getElementById(`dl-stats-${modelId}`);
            if (stats) {
                if (text === 'Queued...') {
                    stats.innerHTML = `<span class="text-yellow">Queued...</span> <a href="/queue" class="text-blue no-underline ml-5">(View Queue)</a>`;
                } else {
                    stats.innerText = text;
                }
            }
        },
        onComplete: () => {
            const bar = document.getElementById(`dl-bar-${modelId}`);
            const stats = document.getElementById(`dl-stats-${modelId}`);
            if (bar) bar.style.width = '100%';
            if (stats) stats.innerText = 'Download Complete!';
            if (cancelBtn) cancelBtn.style.display = 'none';
            if (pauseBtn) pauseBtn.style.display = 'none';
        }
    });

    cancelBtn.addEventListener('click', () => {
        dl.cancel();
    });
    pauseBtn.addEventListener('click', () => {
        dl.pause();
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

    sendBtn.disabled = true;

    await ensureSession(text);

    const newId = generateUUID();
    const parentId = window.branchingParentId !== undefined ? window.branchingParentId : currentLeafId;
    window.branchingParentId = undefined; // Reset state

    const msgObj = {
        role: "user",
        content: text,
        metadata: {
            id: newId,
            parent_id: parentId,
            timestamp: Math.floor(Date.now() / 1000)
        },
        children: []
    };
    messagesMap.set(newId, msgObj);
    if (parentId && messagesMap.has(parentId)) {
        messagesMap.get(parentId).children.push(newId);
    }
    currentLeafId = newId;
    
    renderActivePath();

    const success = await appendMessageToDB("user", text, newId, msgObj.metadata);
    if (!success) {
        messagesMap.delete(newId);
        if (parentId && messagesMap.has(parentId)) {
            const pNode = messagesMap.get(parentId);
            pNode.children = pNode.children.filter(childId => childId !== newId);
        }
        currentLeafId = parentId;
        renderActivePath();
        sendBtn.disabled = false;
        alert("Failed to send message to server. Please check your connection.");
        return;
    }
    
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
        resetChatUI();
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

    const aiMessageId = generateUUID();
    const parentId = currentLeafId;

    let aiMetadata = { id: aiMessageId, parent_id: parentId, timestamp: Math.floor(Date.now() / 1000) };
    const aiMsgObj = {
        role: "assistant",
        content: "",
        metadata: aiMetadata,
        children: []
    };
    
    messagesMap.set(aiMessageId, aiMsgObj);
    if (parentId && messagesMap.has(parentId)) {
        messagesMap.get(parentId).children.push(aiMessageId);
    }
    currentLeafId = aiMessageId;
    
    renderActivePath();
    
    const wrapper = document.getElementById(`msg-wrapper-${aiMessageId}`);
    const aiMessageDiv = wrapper ? wrapper.querySelector('.ai-message') : null;

    try {
        // Grab parameters
        const parameters = getGenerationParameters();
        const requestMessages = chatHistory.slice(0, -1); // Exclude the blank msg we just created

        const response = await fetchWithAuth('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                chat_model_id: chatModelId,
                compressor_model_id: compModelId,
                messages: requestMessages,
                parent_message_id: parentId,
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
        let buffer = "";
        
        const processStreamObj = (obj) => {
            if (obj.token !== undefined) {
                fullAnswer += obj.token;
                if (aiMessageDiv) aiMessageDiv.textContent = fullAnswer;
                aiMsgObj.content = fullAnswer;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            } else if (obj.metadata) {
                if (obj.metadata.generation_time_ms !== undefined || obj.metadata.model) {
                    Object.assign(aiMetadata, obj.metadata);
                    aiMetadata.id = aiMessageId;
                    aiMetadata.parent_id = parentId;
                } else {
                    const pMsg = messagesMap.get(parentId);
                    if (pMsg) {
                        pMsg.metadata = pMsg.metadata || {};
                        Object.assign(pMsg.metadata, obj.metadata);
                        pMsg.metadata.id = parentId;
                    }
                }
            } else if (obj.error !== undefined) {
                fullAnswer += "\nError: " + obj.error;
                if (aiMessageDiv) aiMessageDiv.textContent = fullAnswer;
                aiMsgObj.content = fullAnswer;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        };

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                if (buffer.length > 0) {
                    const lines = buffer.split('\n');
                    for (const line of lines) {
                        if (line.trim()) {
                            try {
                                processStreamObj(JSON.parse(line.trim()));
                            } catch (e) {
                                console.error("Failed to parse stream line:", line, e);
                            }
                        }
                    }
                }
                if (aiMessageDiv) aiMessageDiv.textContent = fullAnswer;
                aiMsgObj.content = fullAnswer;
                break;
            }
            buffer += decoder.decode(value, { stream: true });
            
            let newlineIdx;
            while ((newlineIdx = buffer.indexOf('\n')) !== -1) {
                const line = buffer.slice(0, newlineIdx).trim();
                buffer = buffer.slice(newlineIdx + 1);
                if (line) {
                    try {
                        processStreamObj(JSON.parse(line));
                    } catch (e) {
                        console.error("Failed to parse stream line:", line, e);
                    }
                }
            }
        }
        
        aiMsgObj.metadata = aiMetadata;
        renderActivePath();
        updateStatus();
        const dbSuccess = await appendMessageToDB("assistant", fullAnswer, aiMessageId, aiMetadata);
        if (!dbSuccess) {
            throw new Error("DB_SAVE_FAILED");
        }

    } catch (err) {
        if (err.name === 'AbortError') {
            if (aiMessageDiv) aiMessageDiv.textContent += " [Stopped]";
            aiMsgObj.content += " [Stopped]";
            renderActivePath();
            updateStatus();
            const dbSuccess = await appendMessageToDB("assistant", aiMsgObj.content, aiMessageId, aiMetadata);
            if (!dbSuccess) {
                messagesMap.delete(aiMessageId);
                if (parentId && messagesMap.has(parentId)) {
                    const pNode = messagesMap.get(parentId);
                    pNode.children = pNode.children.filter(childId => childId !== aiMessageId);
                }
                currentLeafId = parentId;
                renderActivePath();
                alert("Error: Failed to save the stopped response to the database.");
            }
        } else {
            messagesMap.delete(aiMessageId);
            if (parentId && messagesMap.has(parentId)) {
                const pNode = messagesMap.get(parentId);
                pNode.children = pNode.children.filter(childId => childId !== aiMessageId);
            }
            currentLeafId = parentId;
            renderActivePath();
            if (err.message === "DB_SAVE_FAILED") {
                alert("Error: Failed to save AI response to the database.");
            } else {
                alert("Error: Failed to connect to engine or generate response.");
            }
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
    if (!currentLeafId) return;
    const leafNode = messagesMap.get(currentLeafId);
    if (!leafNode) return;
    
    if (leafNode.role === 'assistant') {
        currentLeafId = leafNode.metadata ? leafNode.metadata.parent_id : null;
        renderActivePath(); 
        await requestAiResponse();
    } else {
        await requestAiResponse();
    }
}

function showExportModal() {
    if (chatHistory.length === 0) {
        alert("No messages to export.");
        return;
    }
    document.getElementById('export-modal').style.display = 'flex';
}

function hideExportModal() {
    document.getElementById('export-modal').style.display = 'none';
}

function exportChat(format) {
    const title = (currentSessionTitle || "chat").replace(/[^a-z0-9]/gi, '_').toLowerCase();
    const filename = `${title}_${new Date().toISOString().slice(0,10)}.${format}`;
    let content = "";
    let mimeType = "";
    
    if (format === 'json') {
        content = JSON.stringify(chatHistory, null, 2);
        mimeType = 'application/json';
    } else if (format === 'md') {
        content = `# ${currentSessionTitle || "Chat Export"}\n\n`;
        chatHistory.forEach(msg => {
            const role = msg.role === 'user' ? 'User' : 'Assistant';
            content += `### ${role}\n${msg.content}\n\n`;
        });
        mimeType = 'text/markdown';
    }
    
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    hideExportModal();
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

setInterval(updateStatus, 2000);

// Bind top-level buttons
document.getElementById('toggle-parameters-btn')?.addEventListener('click', () => {
    const panel = document.getElementById('parameters-panel');
    panel.style.display = panel.style.display === 'none' ? 'flex' : 'none';
});
document.getElementById('clear-chat-btn')?.addEventListener('click', clearChat);
document.getElementById('export-chat-btn')?.addEventListener('click', showExportModal);
document.getElementById('export-cancel-btn')?.addEventListener('click', hideExportModal);
document.getElementById('export-md-btn')?.addEventListener('click', () => exportChat('md'));
document.getElementById('export-json-btn')?.addEventListener('click', () => exportChat('json'));
document.getElementById('param-temp')?.addEventListener('input', function() {
    document.getElementById('val-temp').innerText = this.value;
});
document.getElementById('param-top-p')?.addEventListener('input', function() {
    document.getElementById('val-top-p').innerText = this.value;
});
document.getElementById('new-chat-btn')?.addEventListener('click', startNewSession);