document.addEventListener("DOMContentLoaded", () => {
    const logContainer = document.getElementById("logContainer");
    const applyBtn = document.getElementById("applyLogLevel");
    const clearBtn = document.getElementById("clearLogs");
    const levelSelect = document.getElementById("logLevel");

    let isAtBottom = true;

    logContainer.addEventListener("scroll", () => {
        const threshold = 10; // Pixel threshold to snap to bottom
        const position = logContainer.scrollHeight - logContainer.scrollTop - logContainer.clientHeight;
        isAtBottom = position <= threshold;
    });

    async function fetchLogLevel() {
        try {
            const res = await fetch("/api/console/loglevel");
            if (res.ok) {
                const data = await res.json();
                levelSelect.value = data.level;
            }
        } catch (e) {
            console.error("Failed to fetch log level", e);
        }
    }

    async function fetchLogs() {
        try {
            const res = await fetch("/api/console/logs");
            if (res.ok) {
                const logs = await res.json();
                logContainer.textContent = logs.join("\n");
                
                // Only auto-scroll if the user hasn't manually scrolled up
                if (isAtBottom) {
                    logContainer.scrollTop = logContainer.scrollHeight;
                }
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
        logContainer.textContent = "";
        isAtBottom = true;
    });

    setInterval(fetchLogs, 1000);
    fetchLogs(); // Initial fetch
    fetchLogLevel(); // Fetch initial log level to set the dropdown
});