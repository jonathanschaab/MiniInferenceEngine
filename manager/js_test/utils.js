import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Bypasses the Axum backend entirely.
 * Intercepts browser requests and serves the raw HTML/JS/CSS files from the disk.
 */
export async function mockStaticAssets(page) {
    const basePath = path.join(__dirname, '../web');

    // Mock HTML routes
    const routes = {
        '/': 'index.html',
        '/memory': 'memory.html',
        '/models': 'models.html',
        '/settings': 'settings.html',
        '/stats': 'stats.html',
        '/console': 'console.html',
        '/queue': 'queue.html',
    };

    for (const [routePath, file] of Object.entries(routes)) {
        await page.route(routePath, async route => {
            route.fulfill({
                contentType: 'text/html',
                body: await fs.promises.readFile(path.join(basePath, file))
            });
        });
    }

    // Mock CSS & JS assets
    await page.route('**/css/*.css', async route => {
        const urlObj = new URL(route.request().url());
        const file = urlObj.pathname.split('/').pop();
        route.fulfill({ contentType: 'text/css', body: await fs.promises.readFile(path.join(basePath, file)) });
    });

    await page.route('**/js/*.js', async route => {
        const urlObj = new URL(route.request().url());
        const file = urlObj.pathname.split('/').pop();
        route.fulfill({ contentType: 'application/javascript', body: await fs.promises.readFile(path.join(basePath, file)) });
    });
}

/**
 * Mocks the Engine's JSON API responses to simulate a healthy, fully loaded GPU system.
 */
export async function mockEngineApis(page) {
    const defaultModels = [
        { id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: true },
        { id: 'mock-comp-1', name: 'Mock Compressor', roles: ['ContextCompressor'], supported_backends: ['Candle'], arch: 'XLMRoberta', parameters_billions: 0.5, size_on_disk_gb: 1.0, max_context_len: 1024, provenance: {}, is_downloaded: true }
    ];

    await page.route('**/api/status', route => {
        route.fulfill({
            status: 200,
            json: {
                active_chat_model_id: 'mock-model-1',
                last_compressor_model_id: 'mock-comp-1',
                active_backend: 'Candle',
                benchmark_running: false,
                vram_total: 16000000000,
                vram_used: 8000000000,
                models_vram: [],
                vram_events: [],
                ram_events: []
            }
        });
    });

    await page.route('**/api/models', route => {
        if (route.request().method() === 'GET') {
            route.fulfill({ status: 200, json: defaultModels });
        } else {
            route.fallback();
        }
    });

    await page.route('**/api/models/*', route => {
        if (route.request().method() === 'GET') {
            const id = route.request().url().split('/').pop();
            const model = defaultModels.find(m => m.id === id);
            if (model) route.fulfill({ status: 200, json: model });
            else route.fulfill({ status: 404 });
        } else if (route.request().method() === 'DELETE') {
            route.fulfill({ status: 200 });
        } else {
            route.fallback();
        }
    });

    const handleSessionsRoute = route => {
        if (route.request().method() === 'GET') {
            route.fulfill({
                status: 200,
                json: [
                    { id: 'session-1', title: 'First Chat Session', updated_at: 1678886400, email: 'mock@example.com' },
                    { id: 'session-2', title: 'Second Chat Session', updated_at: 1678886500, email: 'mock@example.com' }
                ]
            });
        } else {
            const postData = JSON.parse(route.request().postData());
            route.fulfill({ status: 200, json: {
                id: postData.id || 'new-mock-session',
                title: postData.title || 'New Session',
                updated_at: Math.floor(Date.now() / 1000),
                email: 'mock@example.com', messages: []
            } });
        }
    };
    await page.route('**/api/chat/sessions', handleSessionsRoute);
    await page.route('**/api/chat/sessions?*', handleSessionsRoute);

    await page.route('**/api/chat/sessions/**', route => {
        if (route.request().url().includes('/messages')) {
            route.fulfill({ status: 200 });
        } else if (route.request().method() === 'GET') {
            const id = route.request().url().split('?')[0].split('/').pop();
            route.fulfill({
                status: 200,
                json: { id, title: 'Mocked Session', updated_at: Math.floor(Date.now() / 1000), email: 'mock@example.com', messages: [] }
            });
        } else if (route.request().method() === 'DELETE') {
            route.fulfill({ status: 200 });
        } else {
            route.fallback();
        }
    });

    await page.route('**/api/console/loglevel', route => {
        route.fulfill({ status: 200, json: { level: 'info' } });
    });

    await page.route('**/api/downloads', async route => {
        if (route.request().method() === 'GET') {
            await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
        } else if (route.request().method() === 'POST') {
            await route.fulfill({ status: 202, body: '' });
        } else if (route.request().method() === 'DELETE') {
            await route.fulfill({ status: 200 });
        } else {
            route.fallback();
        }
    });

    await page.route('**/api/downloads/*', async route => {
        if (route.request().method() === 'DELETE' || route.request().method() === 'POST') {
            await route.fulfill({ status: 200 });
        } else {
            route.fallback();
        }
    });

    await page.route('**/api/settings/keys', route => {
        if (route.request().method() === 'GET') {
            route.fulfill({ status: 200, json: [{ name: 'Test Key', hash: 'abcdef1234567890', description: 'Used for Playwright' }] });
        } else if (route.request().method() === 'POST') {
            route.fulfill({ status: 200, json: "sk-mocked-api-key" });
        } else {
            route.fallback();
        }
    });

    await page.route('**/api/settings/keys/*', route => {
        if (route.request().method() === 'DELETE') {
            route.fulfill({ status: 200 });
        } else {
            route.fallback();
        }
    });

    await page.route('**/api/generate', route => {
        if (route.request().method() === 'POST') {
            route.fulfill({ status: 200, body: '{"token":"Mocked response token"}\n', contentType: 'text/plain' });
        } else {
            route.fallback();
        }
    });
}

/**
 * Sets up the standard page error and console listeners for the test suite.
 */
export function setupPageErrorHandlers(page, testInfo) {
    page.on('pageerror', err => {
        if (err.message.includes("Content Security Policy")) {
            throw new Error(`CSP VIOLATION: ${err.message}`);
        }
        throw err;
    });

    page.on('console', msg => {
        if (msg.type() === 'error' || msg.type() === 'warning') {
            const text = msg.text();

            if (testInfo.title.includes('network interruptions') || testInfo.title.includes('network errors')) {
                if (text.includes('interrupted, retrying in 5s') || text.includes('502 (Bad Gateway)')) return;
            }

            if (testInfo.title.includes('ServerDropped') && text.includes('Download was stopped on the server.')) return;

            if (testInfo.title.includes('401 Unauthorized')) {
                if (text.includes('401 (Unauthorized)') || text.includes('Unauthorized') || text.includes('Engine status unavailable')) return;
            }

            if (
                text.includes('net::ERR_NO_BUFFER_SPACE') || 
                text.includes('net::ERR_CONNECTION_REFUSED') || 
                text.includes('Failed to append message') || 
                text.includes('Failed to fetch') ||
                text.includes('Failed to delete branch in DB') ||
                text.includes('Failed to clear chat in DB') ||
                text.includes('500 (Internal Server Error)')
            ) return;

            console.log(`[Browser Console]: ${text}`);
        }
    });
}