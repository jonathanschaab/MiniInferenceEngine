document.addEventListener("DOMContentLoaded", () => {
    const logContainer = document.getElementById("logContainer");
    const applyBtn = document.getElementById("applyLogLevel");
    const clearBtn = document.getElementById("clearLogs");
    const levelSelect = document.getElementById("logLevel");

    let isAtBottom = true;
    let currentCursor = 0;

    logContainer.addEventListener("scroll", () => {
        const threshold = 10; // Pixel threshold to snap to bottom
        const position = logContainer.scrollHeight - logContainer.scrollTop - logContainer.clientHeight;
        isAtBottom = position <= threshold;
    });

    function updateLevelIndicatorColor() {
        const colors = {
            'trace': '#cba6f7',
            'debug': '#a6adc8',
            'info': '#89b4fa',
            'warn': '#f9e2af',
            'error': '#f38ba8'
        };
        levelSelect.style.color = colors[levelSelect.value] || 'white';
        levelSelect.style.borderColor = colors[levelSelect.value] || '#45475a';
    }
    
    levelSelect.addEventListener('change', updateLevelIndicatorColor);

    async function fetchLogLevel() {
        try {
            const res = await fetch("/api/console/loglevel");
            if (res.ok) {
                const data = await res.json();
                levelSelect.value = data.level;
                updateLevelIndicatorColor();
            }
        } catch (e) {
            console.error("Failed to fetch log level", e);
        }
    }

    async function fetchLogs() {
        try {
            const res = await fetch(`/api/console/logs?since=${currentCursor}`);
            if (res.ok) {
                const data = await res.json();
                
                if (data.logs && data.logs.length > 0) {
                    if (data.next_cursor - currentCursor >= 1000 || currentCursor === 0) {
                        logContainer.innerHTML = ""; // Hard reset if we fell entirely behind
                    }

                    const fragment = document.createDocumentFragment();
                    
                    data.logs.forEach(line => {
                        let safe = line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                        safe = safe.replace(/\bERROR\b/, '<span style="color: #f38ba8; font-weight: bold;">ERROR</span>');
                        safe = safe.replace(/\bWARN\b/, '<span style="color: #f9e2af; font-weight: bold;">WARN</span>');
                        safe = safe.replace(/\bINFO\b/, '<span style="color: #89b4fa; font-weight: bold;">INFO</span>');
                        safe = safe.replace(/\bDEBUG\b/, '<span style="color: #a6adc8; font-weight: bold;">DEBUG</span>');
                        safe = safe.replace(/\bTRACE\b/, '<span style="color: #cba6f7; font-weight: bold;">TRACE</span>');
                        
                        const div = document.createElement("div");
                        div.innerHTML = safe;
                        fragment.appendChild(div);
                    });
                    
                    logContainer.appendChild(fragment);
                    
                    while (logContainer.childElementCount > 1000) {
                        logContainer.removeChild(logContainer.firstElementChild);
                    }
                    
                    if (isAtBottom) {
                        logContainer.scrollTop = logContainer.scrollHeight;
                    }
                }
                currentCursor = data.next_cursor;
            }
        } catch (e) {
            console.error("Failed to fetch engine logs", e);
        }
    }

    applyBtn.addEventListener("click", async () => {
        const level = levelSelect.value;
        await fetch("/api/console/loglevel", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ level })
        });
        alert(`Memory buffer log level updated to ${level.toUpperCase()}`);
    });

    clearBtn.addEventListener("click", async () => {
        await fetch("/api/console/logs", { method: "DELETE" });
        logContainer.innerHTML = "";
        currentCursor = 0;
        isAtBottom = true;
    });

    setInterval(fetchLogs, 1000);
    fetchLogs(); // Initial fetch
    fetchLogLevel(); // Fetch initial log level to set the dropdown
});