import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Bypasses the Axum backend entirely.
 * Intercepts browser requests and serves the raw HTML/JS/CSS files from the disk.
 */
async function mockStaticAssets(page) {
    const basePath = __dirname;

    // Mock HTML routes
    const routes = {
        '/': 'index.html',
        '/memory': 'memory.html',
        '/models': 'models.html',
        '/settings': 'settings.html',
        '/stats': 'stats.html',
        '/console': 'console.html',
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
async function mockEngineApis(page) {
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
        route.fulfill({
            status: 200,
            json: [
                { id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {} },
                { id: 'mock-comp-1', name: 'Mock Compressor', roles: ['ContextCompressor'], supported_backends: ['Candle'], arch: 'XLMRoberta', parameters_billions: 0.5, size_on_disk_gb: 1.0, max_context_len: 1024, provenance: {} }
            ]
        });
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
            // Mock for POST: assume it creates a new session
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

    await page.route('**/api/chat/sessions/*/messages*', route => {
        route.fulfill({ status: 200 });
    });

    await page.route('**/api/console/loglevel', route => {
        // Grant console access to the UI by mocking a successful 200 OK
        route.fulfill({ status: 200, json: { level: 'info' } });
    });
}

test.describe('Mini Inference Engine - UI Functionality', () => {
    test.beforeEach(async ({ page }) => {
        // Route browser console logs and uncaught errors directly to the terminal
        page.on('pageerror', err => console.log(`[Browser Exception]: ${err.message}`));
        page.on('console', msg => {
            if (msg.type() === 'error' || msg.type() === 'warning') {
                console.log(`[Browser Console]: ${msg.text()}`);
            }
        });

        await mockStaticAssets(page);
        await mockEngineApis(page);
    });

    test('Navbar dynamically injects and renders', async ({ page }) => {
        await page.goto('/');
        await expect(page.locator('h1')).toContainText('Mini Inference Engine');
        
        // Check if the navbar links generated by common.js exist
        const chatBtn = page.locator('button:has-text("💬 Chat")');
        await expect(chatBtn).toBeVisible();
        
        const consoleBtn = page.locator('#nav-console-btn');
        await expect(consoleBtn).toBeVisible();
    });

    test('Chat UI streams a mock generation', async ({ page }) => {
        await page.route('**/api/generate', route => {
            // Mocks the chunked HTTP stream by instantly fulfilling a single text payload
            route.fulfill({ status: 200, body: 'Hello! I am a simulated AI response.', contentType: 'text/plain' });
        });

        await page.goto('/');

        // Wait for the async initializeUI() function to finish populating the dropdowns
        await expect(page.locator('#chat-model-select option')).not.toHaveCount(0);
        await expect(page.locator('#compressor-model-select option')).not.toHaveCount(0);

        const input = page.locator('#prompt-input');
        const sendBtn = page.locator('#send-btn');
        
        await input.fill('Can you hear me?');
        await sendBtn.click();

        // Verify the DOM inserts the correct elements
        await expect(page.locator('.user-message').last()).toContainText('Can you hear me?');
        await expect(page.locator('.ai-message').last()).toContainText('Hello! I am a simulated AI response.');
        await expect(page.locator('#regen-btn')).toBeVisible();
    });

    test('Memory view tabs switch content correctly', async ({ page }) => {
        await page.goto('/memory');

        const vramTab = page.locator('.tab:has-text("GPU VRAM")');
        const ramTab = page.locator('.tab:has-text("System RAM")');
        const vramView = page.locator('#vram-view');
        const ramView = page.locator('#ram-view');

        await expect(vramTab).toHaveClass(/active/);
        await expect(vramView).toHaveClass(/active/);

        await ramTab.click();

        await expect(ramTab).toHaveClass(/active/);
        await expect(ramView).toHaveClass(/active/);
        await expect(vramView).not.toHaveClass(/active/);
    });

    test('Settings UI loads API Keys and opens generation modal', async ({ page }) => {
        await page.route('**/api/settings/keys', route => {
            if (route.request().method() === 'GET') {
                route.fulfill({ status: 200, json: [{ name: 'Test Key', hash: 'abcdef1234567890', description: 'Used for Playwright' }] });
            } else {
                route.fallback();
            }
        });

        await page.goto('/settings');

        await expect(page.locator('#keys-tbody')).toContainText('Test Key');
        await expect(page.locator('#keys-tbody')).toContainText('Used for Playwright');

        await page.locator('button:has-text("+ Create New Key")').click();
        await expect(page.locator('#new-key-modal')).toBeVisible();

        await page.locator('#new-key-modal .btn-cancel').click();
        await expect(page.locator('#new-key-modal')).not.toBeVisible();
    });

    test('Models Directory renders the model configuration cards', async ({ page }) => {
        await page.goto('/models');
        
        const modelCards = page.locator('.model-card');
        await expect(modelCards).toHaveCount(2); // Based on our mock Apis output
        await expect(modelCards.first()).toContainText('Mock Chat Model');
    });

    test('Chat UI loads existing sessions and allows switching', async ({ page }) => {
        // Mock responses for loading specific sessions
        await page.route('**/api/chat/sessions/session-1*', route => {
            route.fulfill({
                status: 200,
                json: { id: 'session-1', title: 'First Chat Session', updated_at: 1678886400, email: 'mock@example.com', messages: [{ role: 'user', content: 'Hi session 1' }, { role: 'assistant', content: 'Hello from session 1' }] }
            });
        });
        await page.route('**/api/chat/sessions/session-2*', route => {
            route.fulfill({
                status: 200,
                json: { id: 'session-2', title: 'Second Chat Session', updated_at: 1678886500, email: 'mock@example.com', messages: [{ role: 'user', content: 'Hi session 2' }, { role: 'assistant', content: 'Hello from session 2' }] }
            });
        });

        await page.goto('/');
        await expect(page.locator('.session-item')).toHaveCount(2);
        await expect(page.locator('.session-item').first()).toContainText('First Chat Session');
        await expect(page.locator('.session-item').last()).toContainText('Second Chat Session');

        await page.locator('.session-item').last().click(); // Click on 'Second Chat Session'
        await expect(page.locator('.ai-message').last()).toContainText('Hello from session 2');
        await expect(page.locator('.session-item').last()).toHaveClass(/active/);
    });

    test('Chat UI remembers active session across reloads and navigation', async ({ page }) => {
        await page.route('**/api/chat/sessions/session-2*', route => {
            route.fulfill({
                status: 200,
                json: { id: 'session-2', title: 'Second Chat Session', updated_at: 1678886500, email: 'mock@example.com', messages: [{ role: 'assistant', content: 'Persistent message' }] }
            });
        });

        await page.goto('/');
        
        // Click the second session
        const session2 = page.locator('.session-item', { hasText: 'Second Chat Session' });
        await session2.click();
        
        // Wait for it to become active and load messages
        await expect(session2).toHaveClass(/active/);
        await expect(page.locator('.ai-message').last()).toContainText('Persistent message');

        // Reload the page
        await page.reload();
        
        // Verify it automatically loads session-2
        const reloadedSession2 = page.locator('.session-item', { hasText: 'Second Chat Session' });
        await expect(reloadedSession2).toHaveClass(/active/);
        await expect(page.locator('.ai-message').last()).toContainText('Persistent message');

        // Navigate away and back
        await page.goto('/models');
        await page.goto('/');

        // Verify it automatically loads session-2 again
        const navigatedSession2 = page.locator('.session-item', { hasText: 'Second Chat Session' });
        await expect(navigatedSession2).toHaveClass(/active/);
        await expect(page.locator('.ai-message').last()).toContainText('Persistent message');
    });

    test('Chat UI restores last session from localStorage even if not in initial API results', async ({ page }) => {
        // Inject the last chat ID into localStorage before the page scripts run
        await page.addInitScript(() => {
            window.localStorage.setItem('mini_inference_last_chat_id', 'off-page-session');
        });

        // Mock the specific fetch for the off-page session that the UI will request
        await page.route('**/api/chat/sessions/off-page-session*', route => {
            route.fulfill({
                status: 200,
                json: { id: 'off-page-session', title: 'Off Page Chat Session', updated_at: 1678886600, email: 'mock@example.com', messages: [{ role: 'assistant', content: 'Message from the off-page session' }] }
            });
        });

        await page.goto('/');

        // Wait for the UI to resolve and verify the off-page session messages loaded directly
        // Because it was not in the initial page of results, the active session will not be highlighted in the sidebar, 
        // but the messages will be correctly loaded into the chat container.
        await expect(page.locator('.ai-message').last()).toContainText('Message from the off-page session');
    });

    test('Chat UI can rename a session', async ({ page }) => {
        let fetchCount = 0;
        const handleSessionRenameRoute = async route => {
            if (route.request().method() === 'GET') {
                const title = fetchCount === 0 ? 'First Chat Session' : 'Renamed Chat Session';
                fetchCount++;
                await route.fulfill({ status: 200, json: [{ id: 'session-1', title: title, updated_at: 1678886400, email: 'mock@example.com' }] });
            } else if (route.request().method() === 'POST') {
                await route.fulfill({ status: 200, json: { id: 'session-1', title: 'Renamed Chat Session' } });
            } else {
                await route.fallback();
            }
        };
        await page.route('**/api/chat/sessions', handleSessionRenameRoute);
        await page.route('**/api/chat/sessions?*', handleSessionRenameRoute);

        await page.goto('/');
        const sessionItem = page.locator('.session-item').first();
        await expect(sessionItem).toContainText('First Chat Session');
        
        await sessionItem.hover();
        await sessionItem.locator('button[title="Rename Chat"]').click();
        
        await page.locator('#rename-input').fill('Renamed Chat Session');
        await page.locator('#rename-confirm-btn').click();
        
        // Verify the DOM listing has updated to the new name based on the mocked second API call
        await expect(sessionItem).toContainText('Renamed Chat Session');
    });

    test('Chat UI can delete a session via custom modal', async ({ page }) => {
        let fetchCount = 0;
        let deleteCalled = false;
        
        // Override the default mock to provide a specific sequence for deletion
        const handleSessionDeleteRoute = async route => {
            if (route.request().method() === 'GET') {
                if (fetchCount === 0) {
                    fetchCount++;
                    await route.fulfill({ status: 200, json: [{ id: 'session-to-delete', title: 'Delete Me', updated_at: 1678886400, email: 'mock@example.com' }] });
                } else {
                    await route.fulfill({ status: 200, json: [] }); // Return empty after deletion
                }
            } else {
                await route.fallback();
            }
        };
        await page.route('**/api/chat/sessions', handleSessionDeleteRoute);
        await page.route('**/api/chat/sessions?*', handleSessionDeleteRoute);

        await page.route('**/api/chat/sessions/session-to-delete*', async route => {
            if (route.request().method() === 'DELETE') {
                deleteCalled = true;
                await route.fulfill({ status: 200 });
            } else {
                await route.fallback();
            }
        });

        await page.goto('/');
        const sessionItem = page.locator('.session-item').first();
        await expect(sessionItem).toContainText('Delete Me');
        
        await sessionItem.hover();
        await sessionItem.locator('button[title="Delete Chat"]').click();
        
        // Verify the custom modal appears
        const deleteModal = page.locator('#delete-modal');
        await expect(deleteModal).toBeVisible();
        
        // Confirm deletion
        await page.locator('#delete-confirm-btn').click();
        
        // Verify the modal closes, the request was sent, and the DOM updates
        await expect(deleteModal).not.toBeVisible();
        await expect(page.locator('.session-item')).toHaveCount(0);
        expect(deleteCalled).toBe(true);
    });

    test('Chat UI auto-scrolls to the newest message in a long session', async ({ page }) => {
        const longMessages = Array.from({ length: 50 }, (_, i) => ({
            role: i % 2 === 0 ? 'user' : 'assistant',
            content: `Message number ${i}\nThis is a bit longer to take up vertical space.\nLine 3.`
        }));

        await page.route('**/api/chat/sessions/session-long*', route => {
            route.fulfill({
                status: 200,
                json: { id: 'session-long', title: 'Long Chat Session', updated_at: 1678886600, email: 'mock@example.com', messages: longMessages }
            });
        });

        const handleLongSessionRoute = route => {
            if (route.request().method() === 'GET') {
                route.fulfill({
                    status: 200,
                    json: [
                        { id: 'session-long', title: 'Long Chat Session', updated_at: 1678886600, email: 'mock@example.com' }
                    ]
                });
            } else {
                route.fallback();
            }
        };
        await page.route('**/api/chat/sessions', handleLongSessionRoute);
        await page.route('**/api/chat/sessions?*', handleLongSessionRoute);

        await page.goto('/');

        const longSessionItem = page.locator('.session-item', { hasText: 'Long Chat Session' });
        await longSessionItem.click();

        await expect(longSessionItem).toHaveClass(/active/);
        await expect(page.locator('.message')).toHaveCount(50);

        // Wait until the container has completed its asynchronous auto-scroll
        await page.waitForFunction(() => {
            const container = document.getElementById('chat-container');
            return Math.abs(container.scrollHeight - container.scrollTop - container.clientHeight) <= 2;
        });
    });
});