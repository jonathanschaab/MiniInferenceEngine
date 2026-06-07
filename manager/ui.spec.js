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
    const basePath = path.join(__dirname, 'web');

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
async function mockEngineApis(page) {
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
            route.fulfill({
                status: 200,
                json: defaultModels
            });
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
        // Grant console access to the UI by mocking a successful 200 OK
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

test.describe('Mini Inference Engine - UI Functionality', () => {
    test.beforeEach(async ({ page }, testInfo) => {
        // Fail tests on any Content Security Policy violation or uncaught browser error
        page.on('pageerror', err => {
            if (err.message.includes("Content Security Policy")) {
                throw new Error(`CSP VIOLATION: ${err.message}`);
            }
            throw err; // Re-throw other errors to fail the test
        });

        page.on('console', msg => {
            if (msg.type() === 'error' || msg.type() === 'warning') {
                const text = msg.text();

                // Ignore expected intentional errors from network interruption tests
                if (testInfo.title.includes('network interruptions') || testInfo.title.includes('network errors')) {
                    if (text.includes('interrupted, retrying in 5s') || text.includes('502 (Bad Gateway)')) {
                        return;
                    }
                }

                // Ignore expected intentional errors from server drop tests
                if (testInfo.title.includes('ServerDropped') && text.includes('Download was stopped on the server.')) {
                    return;
                }

                // Ignore expected intentional errors from 401 redirect test
                if (testInfo.title.includes('401 Unauthorized')) {
                    if (text.includes('401 (Unauthorized)') || text.includes('Unauthorized') || text.includes('Engine status unavailable')) {
                        return;
                    }
                }

                // Ignore Chromium internal network errors for unmocked endpoints hitting dead localhost ports
                if (text.includes('net::ERR_NO_BUFFER_SPACE') || text.includes('net::ERR_CONNECTION_REFUSED') || text.includes('Failed to append message') || text.includes('Failed to fetch')) {
                    return;
                }

                console.log(`[Browser Console]: ${text}`);
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
            // Mocks the chunked HTTP stream
            const streamResponse = 
                '{"metadata":{"id":"mock-prompt-id","timestamp":1000,"token_counts":{"mock-model-1":5}}}\n' +
                '{"token":"Hello! I am a simulated AI response."}\n' +
                '{"metadata":{"id":"mock-ai-id","parent_id":"mock-prompt-id","timestamp":1002,"model":"mock-model-1","generation_time_ms":100,"token_counts":{"mock-model-1":15},"parameters":{}}}\n';
            route.fulfill({ status: 200, body: streamResponse, contentType: 'text/plain' });
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
        await expect(page.locator('.msg-metadata').first()).toContainText('Tokens: mock-model-1: 5');
        await expect(page.locator('.msg-metadata').last()).toContainText('Tokens: mock-model-1: 15');
        await expect(page.locator('#regen-btn')).toBeVisible();
    });

    test('Chat UI generates fallback UUID in non-secure contexts', async ({ page }) => {
        // Undefine crypto.randomUUID to simulate HTTP access on a non-localhost IP
        await page.addInitScript(() => {
            if (window.crypto) {
                window.crypto.randomUUID = undefined;
            }
        });

        await page.route('**/api/generate', route => {
            route.fulfill({ status: 200, body: '{"token":"Fallback UUID worked!"}\n', contentType: 'text/plain' });
        });

        await page.goto('/');

        // Wait for UI to initialize
        await expect(page.locator('#chat-model-select option')).not.toHaveCount(0);

        await page.locator('#prompt-input').fill('Test fallback UUID');
        await page.locator('#send-btn').click();

        await expect(page.locator('.user-message').last()).toContainText('Test fallback UUID');
        await expect(page.locator('.ai-message').last()).toContainText('Fallback UUID worked!');
    });

    test('Chat UI renders message metadata correctly and handles scoring updates', async ({ page }) => {
        // Setup initial state with a session having metadata
        const handleMetaSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: [{ id: 'session-meta', title: 'Meta Chat', updated_at: 1678886400, email: 'mock@example.com' }]
                });
            } else {
                route.fallback();
            }
        };
        await page.route('**/api/chat/sessions', handleMetaSessionRoute);
        await page.route('**/api/chat/sessions?*', handleMetaSessionRoute);

        await page.route('**/api/chat/sessions/session-meta*', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-meta', title: 'Meta Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'What is Rust?',
                            metadata: { id: 'mock-msg-1', parent_id: null, timestamp: 1234, token_counts: { 'mock-model-1': 10 } }
                        }, {
                            role: 'assistant', content: 'Rust is a systems programming language.',
                            metadata: { id: 'mock-msg-2', parent_id: 'mock-msg-1', timestamp: 1235, model: 'mock-model-1', generation_time_ms: 1500, token_counts: { 'mock-model-1': 25 } }
                        }]
                    }
                });
            } else {
                route.fallback();
            }
        });

        await page.goto('/');
        
        await page.locator('.session-item', { hasText: 'Meta Chat' }).click();
        
        // Wait for UI to populate
        await expect(page.locator('.user-message').last()).toContainText('What is Rust?');
        
        // Verify User Metadata
        const userMeta = page.locator('.msg-metadata').first();
        await expect(userMeta).toContainText('Tokens: mock-model-1: 10');
        
        // Verify AI Metadata
        const aiMeta = page.locator('.msg-metadata').last();
        await expect(aiMeta).toContainText('Model: mock-model-1');
        await expect(aiMeta).toContainText('Time: 1.50s');
        await expect(aiMeta).toContainText('Tokens: mock-model-1: 25');
        
        // Test Scoring
        const scoreSelect = aiMeta.locator('select');
        
        const [request] = await Promise.all([
            page.waitForRequest(req => req.url().includes('/api/chat/sessions/session-meta/messages') && req.method() === 'POST'),
            scoreSelect.selectOption('5')
        ]);
        
        const postData = JSON.parse(request.postData());
        expect(postData.score).toBe(5);
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
        await page.goto('/settings');

        await expect(page.locator('#keys-tbody')).toContainText('Test Key');
        await expect(page.locator('#keys-tbody')).toContainText('Used for Playwright');

        await page.locator('button:has-text("+ Create New Key")').click();
        await expect(page.locator('#new-key-modal')).toBeVisible();

        await page.locator('#new-key-modal .btn-cancel').click();
        await expect(page.locator('#new-key-modal')).not.toBeVisible();
    });

    test('Settings UI resists XSS payloads in API Key names and descriptions', async ({ page }) => {
        await page.route('**/api/settings/keys', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: [{
                        name: '<script>alert("XSS-Name")</script>MaliciousName',
                        hash: 'abcdef1234567890',
                        description: '<img src="x" onerror="alert(\'XSS-Desc\')">MaliciousDesc'
                    }]
                });
            } else {
                route.fallback();
            }
        });

        let alertFired = false;
        page.on('dialog', dialog => {
            alertFired = true;
            dialog.dismiss();
        });

        await page.goto('/settings');

        const tbody = page.locator('#keys-tbody');
        await expect(tbody).toBeVisible();

        // Because settings.js uses .textContent, the literal HTML tags should be rendered to the screen
        // safely, rather than being stripped out or executed.
        await expect(tbody).toContainText('<script>');
        await expect(tbody).toContainText('alert("XSS-Name")');
        await expect(tbody).toContainText('<img');

        // Confirm no actual code execution occurred
        expect(alertFired).toBe(false);
    });

    test('Models Directory renders the model configuration cards', async ({ page }) => {
        await page.goto('/models');
        
        const modelCards = page.locator('.model-dir-card');
        await expect(modelCards).toHaveCount(2); // Based on our mock Apis output
        await expect(modelCards.first()).toContainText('Mock Chat Model');
    });

    test('Models Directory renders even if status API fails or is slow', async ({ page }) => {
        // Mock status API to hang indefinitely to simulate extreme slowness/failure
        await page.route('**/api/status', () => {
            // Do not fulfill the route, leaving it hanging
        });

        await page.goto('/models');
        
        const modelCards = page.locator('.model-dir-card');
        // The models should load immediately without waiting for the status API
        await expect(modelCards).toHaveCount(2);
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

    test('Chat UI can create a branch, navigate branches, and use split view', async ({ page }) => {
        const handleSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-branch', title: 'Branch Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Original Prompt',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'Original Response',
                            metadata: { id: 'msg-2', parent_id: 'msg-1', timestamp: 1001, model: 'mock-model-1' }
                        }]
                    }
                });
            } else {
                route.fallback();
            }
        };
        await page.route('**/api/chat/sessions/session-branch*', handleSessionRoute);
        const handleBranchSessionsRoute = route => {
            route.fulfill({ status: 200, json: [{ id: 'session-branch', title: 'Branch Chat', updated_at: 1678886400, email: 'mock@example.com' }] });
        };
        await page.route('**/api/chat/sessions', handleBranchSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleBranchSessionsRoute);
        await page.route('**/api/generate', route => {
            const streamResponse = 
                '{"metadata":{"id":"msg-4","parent_id":"msg-3","timestamp":1003,"model":"mock-model-1"}}\n' +
                '{"token":"New Response"}\n';
            route.fulfill({ status: 200, body: streamResponse, contentType: 'text/plain' });
        });

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Branch Chat' }).click();

        // 1. Edit the prompt to create a branch
        await expect(page.locator('.user-message').first()).toContainText('Original Prompt');
        const editBtn = page.locator('button[title="Edit Prompt (Creates New Branch)"]');
        await expect(editBtn).toBeVisible();
        await editBtn.click();

        const input = page.locator('#prompt-input');
        await expect(input).toHaveValue('Original Prompt');
        await input.fill('New Prompt');
        await page.locator('#send-btn').click();

        // Wait for generation
        await expect(page.locator('.user-message').last()).toContainText('New Prompt');
        await expect(page.locator('.ai-message').last()).toContainText('New Response');

        // Verify Branch Controls appear
        const branchControls = page.locator('.branch-controls');
        await expect(branchControls).toBeVisible();
        await expect(branchControls).toContainText('2 of 2');

        // 2. Navigate back to the original branch
        await branchControls.locator('span', { hasText: '◀' }).click();
        await expect(page.locator('.user-message').last()).toContainText('Original Prompt');
        await expect(page.locator('.ai-message').last()).toContainText('Original Response');
        await expect(branchControls).toContainText('1 of 2');

        // 3. Toggle Split View
        await branchControls.locator('span', { hasText: '◫ Split View' }).click();
        const splitContainer = page.locator('.split-container');
        await expect(splitContainer).toBeVisible();
        await expect(splitContainer.locator('.split-item').first()).toContainText('Original Prompt');
        await expect(splitContainer.locator('.split-item').last()).toContainText('New Prompt');

        // Click a split item to select it
        await splitContainer.locator('.split-item').last().click();
        await expect(splitContainer).not.toBeVisible();
        await expect(branchControls).toContainText('2 of 2');
    });

    test('Chat UI clears branchingParentId when switching or starting a new session', async ({ page }) => {
        const handleSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-branch-state', title: 'Branch State Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Prompt 1',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'Response 1',
                            metadata: { id: 'msg-2', parent_id: 'msg-1', timestamp: 1001, model: 'mock-model-1' }
                        }, {
                            role: 'user', content: 'Prompt 2',
                            metadata: { id: 'msg-3', parent_id: 'msg-2', timestamp: 1002 }
                        }]
                    }
                });
            } else {
                route.fallback();
            }
        };
        await page.route('**/api/chat/sessions/session-branch-state*', handleSessionRoute);
        const handleBranchSessionsRoute = route => {
            if (route.request().method() === 'GET') {
                route.fulfill({ status: 200, json: [{ id: 'session-branch-state', title: 'Branch State Chat', updated_at: 1678886400, email: 'mock@example.com' }] });
            } else {
                route.fallback();
            }
        };
        await page.route('**/api/chat/sessions', handleBranchSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleBranchSessionsRoute);

        let capturedParentId = "not_captured";
        await page.route('**/api/chat/sessions/*/messages', async route => {
            if (route.request().method() === 'POST') {
                const body = JSON.parse(route.request().postData());
                if (body.role === 'user') {
                    capturedParentId = body.parent_id;
                }
                await route.fulfill({ status: 200 });
            } else {
                route.fallback();
            }
        });

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Branch State Chat' }).click();

        await expect(page.locator('.user-message').last()).toContainText('Prompt 2');
        const editBtn = page.locator('button[title="Edit Prompt (Creates New Branch)"]').last();
        await expect(editBtn).toBeVisible();
        await editBtn.click();
        
        // branchingParentId is now 'msg-2', start a new session
        await page.locator('#new-chat-btn').click();

        await page.locator('#prompt-input').fill('Hello from new session');
        await page.locator('#send-btn').click();

        await page.waitForResponse(res => res.url().includes('/api/chat/sessions/') && res.url().includes('/messages') && res.request().method() === 'POST');
        expect(capturedParentId).toBeNull();
    });

    test('Chat UI creates a sibling branch when regenerating a response', async ({ page }) => {
        const handleRegenSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-regen', title: 'Regenerate Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Original Prompt',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'Original Response',
                            metadata: { id: 'msg-2', parent_id: 'msg-1', timestamp: 1001, model: 'mock-model-1' }
                        }]
                    }
                });
            } else {
                route.fallback();
            }
        };

        await page.route('**/api/chat/sessions/session-regen*', handleRegenSessionRoute);
        const handleRegenSessionsRoute = route => {
            route.fulfill({ status: 200, json: [{ id: 'session-regen', title: 'Regenerate Chat', updated_at: 1678886400, email: 'mock@example.com' }] });
        };
        await page.route('**/api/chat/sessions', handleRegenSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleRegenSessionsRoute);
        await page.route('**/api/generate', route => {
            const streamResponse = 
                '{"metadata":{"id":"msg-3","parent_id":"msg-1","timestamp":1002,"model":"mock-model-1"}}\n' +
                '{"token":"Regenerated Response"}\n';
            route.fulfill({ status: 200, body: streamResponse, contentType: 'text/plain' });
        });

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Regenerate Chat' }).click();

        await expect(page.locator('.ai-message').last()).toContainText('Original Response');
        
        // Click Regenerate
        await page.locator('#regen-btn').click();

        // Verify the new response is rendered
        await expect(page.locator('.ai-message').last()).toContainText('Regenerated Response');
        
        // Verify branch controls appear on the new AI message (since it's a sibling of msg-2)
        const branchControls = page.locator('.branch-controls').last();
        await expect(branchControls).toBeVisible();
        await expect(branchControls).toContainText('2 of 2');
    });

    test('Chat UI gracefully handles regeneration when current leaf node is missing', async ({ page }) => {
        const handleRegenMissingSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-regen-missing', title: 'Regenerate Missing Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Original Prompt',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'Original Response',
                            metadata: { id: 'msg-2', parent_id: 'msg-1', timestamp: 1001, model: 'mock-model-1' }
                        }]
                    }
                });
            } else {
                route.fallback();
            }
        };

        await page.route('**/api/chat/sessions/session-regen-missing*', handleRegenMissingSessionRoute);
        const handleRegenMissingSessionsRoute = route => {
            route.fulfill({ status: 200, json: [{ id: 'session-regen-missing', title: 'Regenerate Missing Chat', updated_at: 1678886400, email: 'mock@example.com' }] });
        };
        await page.route('**/api/chat/sessions', handleRegenMissingSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleRegenMissingSessionsRoute);

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Regenerate Missing Chat' }).click();

        await expect(page.locator('.ai-message').last()).toContainText('Original Response');
        
        // Force the mismatch state
        await page.evaluate(() => {
            // eslint-disable-next-line no-undef
            messagesMap.delete(currentLeafId);
        });

        // Click Regenerate
        await page.locator('#regen-btn').click();

        // Verify the app did not crash (our pageerror listener in test.beforeEach would catch it) and is still responsive
        await expect(page.locator('#prompt-input')).toBeVisible();
    });

    test('Chat UI correctly applies metadata to a parent message missing metadata during streaming', async ({ page }) => {
        const handleMissingMetaSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-missing-meta', title: 'Missing Meta Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Original Prompt',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'Original Response',
                            metadata: { id: 'msg-2', parent_id: 'msg-1', timestamp: 1001, model: 'mock-model-1' }
                        }]
                    }
                });
            } else {
                route.fallback();
            }
        };

        await page.route('**/api/chat/sessions/session-missing-meta*', handleMissingMetaSessionRoute);
        const handleMissingMetaSessionsRoute = route => {
            route.fulfill({ status: 200, json: [{ id: 'session-missing-meta', title: 'Missing Meta Chat', updated_at: 1678886400, email: 'mock@example.com' }] });
        };
        await page.route('**/api/chat/sessions', handleMissingMetaSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleMissingMetaSessionsRoute);

        await page.route('**/api/generate', route => {
            const streamResponse = 
                '{"metadata":{"token_counts":{"mock-model-1":42}}}\n' +
                '{"token":"Regenerated Response"}\n';
            route.fulfill({ status: 200, body: streamResponse, contentType: 'text/plain' });
        });

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Missing Meta Chat' }).click();
        await expect(page.locator('.ai-message').last()).toContainText('Original Response');
        
        await page.evaluate(() => {
            // eslint-disable-next-line no-undef
            messagesMap.get('msg-1').metadata = undefined;
        });

        await page.locator('#regen-btn').click();
        await expect(page.locator('.ai-message').last()).toContainText('Regenerated Response');
        await expect(page.locator('.msg-metadata').first()).toContainText('Tokens: mock-model-1: 42');
    });

    test('Chat UI correctly stitches legacy linear chat histories missing parent_ids', async ({ page }) => {
        const handleLegacySessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-legacy', title: 'Legacy Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Message 1',
                            metadata: { id: 'session-legacy_0', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'Message 2',
                            metadata: { id: 'session-legacy_1', parent_id: null, timestamp: 1001 }
                        }, {
                            role: 'user', content: 'Message 3',
                            metadata: { id: 'session-legacy_2', parent_id: null, timestamp: 1002 }
                        }]
                    }
                });
            } else {
                route.fallback();
            }
        };

        await page.route('**/api/chat/sessions/session-legacy*', handleLegacySessionRoute);
        const handleLegacySessionsList = route => {
            route.fulfill({ status: 200, json: [{ id: 'session-legacy', title: 'Legacy Chat', updated_at: 1678886400, email: 'mock@example.com' }] });
        };
        await page.route('**/api/chat/sessions', handleLegacySessionsList);
        await page.route('**/api/chat/sessions?*', handleLegacySessionsList);

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Legacy Chat' }).click();

        // If stitching fails, only Message 3 will render. If successful, all 3 render in sequence!
        await expect(page.locator('.message')).toHaveCount(3);
        await expect(page.locator('.message').nth(0)).toContainText('Message 1');
        await expect(page.locator('.message').nth(1)).toContainText('Message 2');
        await expect(page.locator('.message').nth(2)).toContainText('Message 3');
    });

    test('Chat UI correctly stitches legacy linear chat histories with undefined parent_ids', async ({ page }) => {
        const handleLegacyUndefinedRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-legacy-undefined', title: 'Legacy Chat Undefined', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Message A',
                            metadata: { id: 'session-legacy-undefined_0', timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'Message B',
                            metadata: { id: 'session-legacy-undefined_1', timestamp: 1001 }
                        }, {
                            role: 'user', content: 'Message C',
                            metadata: { id: 'session-legacy-undefined_2', timestamp: 1002 }
                        }]
                    }
                });
            } else {
                route.fallback();
            }
        };

        await page.route('**/api/chat/sessions/session-legacy-undefined*', handleLegacyUndefinedRoute);
        const handleLegacyUndefinedList = route => {
            route.fulfill({ status: 200, json: [{ id: 'session-legacy-undefined', title: 'Legacy Chat Undefined', updated_at: 1678886400, email: 'mock@example.com' }] });
        };
        await page.route('**/api/chat/sessions', handleLegacyUndefinedList);
        await page.route('**/api/chat/sessions?*', handleLegacyUndefinedList);

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Legacy Chat Undefined' }).click();

        await expect(page.locator('.message')).toHaveCount(3);
        await expect(page.locator('.message').nth(2)).toContainText('Message C');
    });

    test('Chat UI can prune a branch', async ({ page }) => {
        await page.route('**/api/chat/sessions/session-prune*', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-prune', title: 'Prune Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Prompt 1',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'user', content: 'Prompt 2',
                            metadata: { id: 'msg-2', parent_id: null, timestamp: 1001 }
                        }]
                    }
                });
            } else {
                route.fallback();
            }
        });

        const handlePruneSessionsRoute = route => {
            route.fulfill({ status: 200, json: [{ id: 'session-prune', title: 'Prune Chat', updated_at: 1678886400, email: 'mock@example.com' }] });
        };
        await page.route('**/api/chat/sessions', handlePruneSessionsRoute);
        await page.route('**/api/chat/sessions?*', handlePruneSessionsRoute);

        let deleteRequestReceived = false;
        await page.route('**/api/chat/sessions/session-prune/messages', async route => {
            if (route.request().method() === 'DELETE') {
                deleteRequestReceived = true;
                const body = JSON.parse(route.request().postData());
                expect(body.message_ids).toContain('msg-2');
                await route.fulfill({ status: 200 });
            } else {
                route.fallback();
            }
        });

        page.on('dialog', dialog => dialog.accept()); // Accept the pruning confirmation dialog

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Prune Chat' }).click();

        const branchControls = page.locator('.branch-controls');
        await expect(branchControls).toBeVisible();
        await expect(page.locator('.user-message').last()).toContainText('Prompt 2');

        await branchControls.locator('span', { hasText: '🗑️ Prune' }).click();

        await expect(page.locator('.user-message').last()).toContainText('Prompt 1');
        expect(deleteRequestReceived).toBe(true);
        await expect(branchControls).not.toBeVisible();
    });

    test('Chat UI selects the highest scored branch as the default path', async ({ page }) => {
        const handleScoreSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-score', title: 'Score Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Root Prompt',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'High Score Branch',
                            // Even though this branch is older, the score of 5 should make it the default leaf
                            metadata: { id: 'msg-2', parent_id: 'msg-1', timestamp: 1001, score: 5 }
                        }, {
                            role: 'assistant', content: 'Recent Low Score Branch',
                            metadata: { id: 'msg-3', parent_id: 'msg-1', timestamp: 1002, score: 1 }
                        }]
                    }
                });
            } else {
                route.fallback();
            }
        };

        await page.route('**/api/chat/sessions/session-score*', handleScoreSessionRoute);
        const handleScoreSessionsList = route => {
            route.fulfill({ status: 200, json: [{ id: 'session-score', title: 'Score Chat', updated_at: 1678886400, email: 'mock@example.com' }] });
        };
        await page.route('**/api/chat/sessions', handleScoreSessionsList);
        await page.route('**/api/chat/sessions?*', handleScoreSessionsList);

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Score Chat' }).click();

        // Should automatically render the High Score Branch
        await expect(page.locator('.ai-message').last()).toContainText('High Score Branch');
        const branchControls = page.locator('.branch-controls');
        await expect(branchControls).toBeVisible();
        await expect(branchControls).toContainText('1 of 2');
    });

    test('Chat UI recursively gathers descendants when pruning a branch', async ({ page }) => {
        const handleDeepPruneSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-deep-prune', title: 'Deep Prune Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Root',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'Branch A',
                            metadata: { id: 'msg-2', parent_id: 'msg-1', timestamp: 1001 }
                        }, {
                            role: 'assistant', content: 'Branch B',
                            metadata: { id: 'msg-3', parent_id: 'msg-1', timestamp: 1002 }
                        }, {
                            role: 'user', content: 'Child of Branch A',
                            metadata: { id: 'msg-4', parent_id: 'msg-2', timestamp: 1003 }
                        }]
                    }
                });
            } else {
                route.fallback();
            }
        };

        await page.route('**/api/chat/sessions/session-deep-prune*', handleDeepPruneSessionRoute);
        const handleDeepPruneSessionsList = route => {
            route.fulfill({ status: 200, json: [{ id: 'session-deep-prune', title: 'Deep Prune Chat', updated_at: 1678886400, email: 'mock@example.com' }] });
        };
        await page.route('**/api/chat/sessions', handleDeepPruneSessionsList);
        await page.route('**/api/chat/sessions?*', handleDeepPruneSessionsList);

        let deletePayload = null;
        await page.route('**/api/chat/sessions/session-deep-prune/messages', async route => {
            if (route.request().method() === 'DELETE') {
                deletePayload = JSON.parse(route.request().postData());
                await route.fulfill({ status: 200 });
            } else {
                route.fallback();
            }
        });

        page.on('dialog', dialog => dialog.accept());

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Deep Prune Chat' }).click();

        // Path defaults to Branch A because msg-4 has the latest timestamp
        await expect(page.locator('.user-message').last()).toContainText('Child of Branch A');
        
        const branchControls = page.locator('#msg-wrapper-msg-2 .branch-controls');
        await expect(branchControls).toBeVisible();
        
        // Prune the parent of the current leaf
        await branchControls.locator('span', { hasText: '🗑️ Prune' }).click();

        expect(deletePayload).not.toBeNull();
        expect(deletePayload.message_ids).toContain('msg-2');
        expect(deletePayload.message_ids).toContain('msg-4'); // Child was successfully gathered!
        expect(deletePayload.message_ids).not.toContain('msg-1');
        expect(deletePayload.message_ids).not.toContain('msg-3');

        // Validates that after dropping the branch, the UI safely falls back to the remaining sibling branch
        await expect(page.locator('.ai-message').last()).toContainText('Branch B');
    });

    test('Chat UI can export chat history', async ({ page }) => {
        await page.route('**/api/chat/sessions/session-export*', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-export', title: 'Export Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Test Prompt',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'Test Response',
                            metadata: { id: 'msg-2', parent_id: 'msg-1', timestamp: 1001 }
                        }]
                    }
                });
            } else {
                route.fallback();
            }
        });

        const handleExportSessionsRoute = route => {
            route.fulfill({ status: 200, json: [{ id: 'session-export', title: 'Export Chat', updated_at: 1678886400, email: 'mock@example.com' }] });
        };
        await page.route('**/api/chat/sessions', handleExportSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleExportSessionsRoute);

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Export Chat' }).click();

        await page.locator('#export-chat-btn').click();
        const exportModal = page.locator('#export-modal');
        await expect(exportModal).toBeVisible();

        const downloadPromise = page.waitForEvent('download');
        await page.locator('#export-md-btn').click();
        const download = await downloadPromise;
        
        expect(download.suggestedFilename()).toContain('.md');
        await expect(exportModal).not.toBeVisible();
    });

    test('Chat UI resists XSS payloads in session titles and messages', async ({ page }) => {
        // Intercept session list to inject XSS in the title
        await page.route('**/api/chat/sessions?*', async route => {
            await route.fulfill({
                status: 200,
                json: [{ 
                    id: 'xss-session', 
                    title: '<script>alert("XSS-Title")</script>', 
                    updated_at: 1678886400, 
                    email: 'mock@example.com' 
                }]
            });
        });

        // Intercept session messages to inject XSS in the chat history
        await page.route('**/api/chat/sessions/xss-session*', async route => {
            await route.fulfill({
                status: 200,
                json: { 
                    id: 'xss-session', 
                    title: '<script>alert("XSS-Title")</script>', 
                    updated_at: 1678886400, 
                    email: 'mock@example.com', 
                    messages: [{ 
                        role: 'assistant', 
                        content: '<img src="x" onerror="alert(\'XSS-Message\')">Malicious Message' 
                    }] 
                }
            });
        });

        let alertFired = false;
        page.on('dialog', dialog => {
            alertFired = true;
            dialog.dismiss();
        });

        await page.goto('/');

        // Wait for the session to load in the sidebar
        const sessionItem = page.locator('.session-item[data-id="xss-session"]');
        await expect(sessionItem).toBeVisible();
        
        // Check that the literal text is visible (proving it wasn't parsed as HTML tags)
        await expect(sessionItem.locator('.session-title')).toContainText('<script>');
        
        // Click the session to load the messages
        await sessionItem.click();
        
        const aiMessage = page.locator('.ai-message').last();
        await expect(aiMessage).toBeVisible();
        await expect(aiMessage).toContainText('<img src="x" onerror="alert(\'XSS-Message\')">Malicious Message');

        // Confirm no alerts fired
        expect(alertFired).toBe(false);
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
        const longMessages = [];
        let lastId = null;
        for (let i = 0; i < 50; i++) {
            const id = `msg-${i}`;
            longMessages.push({
                role: i % 2 === 0 ? 'user' : 'assistant',
                content: `Message number ${i}\nThis is a bit longer to take up vertical space.\nLine 3.`,
                metadata: { id, parent_id: lastId, timestamp: 1000 + i }
            });
            lastId = id;
        }

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

    test('Models Directory handles streaming download progress', async ({ page }) => {
        const downloadState = {};
        const modelState = [
            { id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: true },
            { id: 'mock-comp-1', name: 'Mock Compressor', roles: ['ContextCompressor'], supported_backends: ['Candle'], arch: 'XLMRoberta', parameters_billions: 0.5, size_on_disk_gb: 1.0, max_context_len: 1024, provenance: {}, is_downloaded: false }
        ];

        await page.route('**/api/models', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(modelState) });
            } else {
                route.fallback();
            }
        });
        
        await page.route('**/api/models/*', async route => {
            if (route.request().method() === 'GET') {
                const id = route.request().url().split('/').pop();
                const model = modelState.find(m => m.id === id);
                if (model) await route.fulfill({ status: 200, json: model });
                else await route.fulfill({ status: 404 });
            } else {
                route.fallback();
            }
        });

        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(downloadState) });
            } else if (route.request().method() === 'POST') {
                const postData = JSON.parse(route.request().postData());
                if (postData.model_id === 'mock-comp-1') {
                    downloadState['mock-comp-1'] = {
                        bytes_transferred: 52428800, // 50 MB
                        total_bytes: 104857600,      // 100 MB
                        current_speed_bps: 10485760, // 10 MB/s
                        start_time: Math.floor(Date.now() / 1000) - 5,
                        state: 'Downloading...'
                    };
                }
                await route.fulfill({ status: 202, body: '' });
            } else {
                route.fallback();
            }
        });

        await page.goto('/models');

        const compCard = page.locator('#model-card-mock-comp-1');
        await expect(compCard).toHaveClass(/model-undownloaded/);
        
        const downloadBtn = compCard.locator('.btn-download');
        await downloadBtn.click();

        const progressContainer = compCard.locator('.download-progress-container');
        await expect(progressContainer).toBeVisible();
        await expect(progressContainer.locator('.download-stats')).toContainText('50.0%');
        await expect(progressContainer.locator('.download-stats')).toContainText('10.0 MB/s');

        // Simulate completion
        delete downloadState['mock-comp-1'];
        modelState[1].is_downloaded = true;

        // Polling will detect activeIds is empty and trigger loadModels. We use expect's
        // polling to wait for the UI to update after the app's own setInterval fires.
        await expect(async () => {
            await expect(compCard.locator('.badge-json', { hasText: 'Ready' })).toBeVisible();
        }).toPass();
        await expect(compCard).not.toHaveClass(/model-undownloaded/);
    });

    test('Chat UI initiates model download before generation if missing', async ({ page }) => {
        const downloadState = {};
        const modelState = [
            { id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: false },
            { id: 'mock-comp-1', name: 'Mock Compressor', roles: ['ContextCompressor'], supported_backends: ['Candle'], arch: 'XLMRoberta', parameters_billions: 0.5, size_on_disk_gb: 1.0, max_context_len: 1024, provenance: {}, is_downloaded: true }
        ];

        // Mock models so the selected chat model is NOT downloaded
        await page.route('**/api/models', route => {
            route.fulfill({
                status: 200,
                json: modelState
            });
        });

        await page.route('**/api/models/*', async route => {
            if (route.request().method() === 'GET') {
                const id = route.request().url().split('/').pop();
                const model = modelState.find(m => m.id === id);
                if (model) await route.fulfill({ status: 200, json: model });
                else await route.fulfill({ status: 404 });
            } else {
                route.fallback();
            }
        });

        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({ status: 200, json: downloadState });
            } else if (route.request().method() === 'POST') {
                const postData = JSON.parse(route.request().postData());
                if (postData.model_id === 'mock-model-1') {
                    downloadState['mock-model-1'] = { bytes_transferred: 50, total_bytes: 100, current_speed_bps: 1000000, start_time: 0, state: 'Downloading...' };
                }
                await route.fulfill({ status: 202, body: '' });
            } else {
                route.fallback();
            }
        });

        // Mock generate
        await page.route('**/api/generate', route => {
            route.fulfill({ status: 200, body: '{"token":"Generation after download."}\n', contentType: 'text/plain' });
        });

        await page.goto('/');

        // Wait for UI to initialize
        await expect(page.locator('#chat-model-select option')).not.toHaveCount(0);

        await page.locator('#prompt-input').fill('Test download');
        await page.locator('#send-btn').click();

        // The download bar should appear dynamically in the chat
        const progressContainer = page.locator('.download-progress-container');
        await expect(progressContainer).toBeVisible();
        await expect(progressContainer.locator('#dl-stats-mock-model-1')).toContainText('50.0%');

        // Simulate completion
        delete downloadState['mock-model-1'];
        modelState[0].is_downloaded = true;

        // Wait for generation to complete (after the simulated download finishes)
        await expect(page.locator('.ai-message').last()).toContainText('Generation after download.');
    });

    test('Models Directory handles download cancellation', async ({ page }) => {
        const downloadState = {};
        const modelState = [
            { id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: false }
        ];

        await page.route('**/api/models', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(modelState) });
            } else {
                route.fallback();
            }
        });

        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(downloadState) });
            } else if (route.request().method() === 'POST') {
                const postData = JSON.parse(route.request().postData());
                if (postData.model_id === 'mock-model-1') {
                    downloadState['mock-model-1'] = { bytes_transferred: 50, total_bytes: 100, current_speed_bps: 10, start_time: Math.floor(Date.now() / 1000), state: 'Downloading...' };
                }
                await route.fulfill({ status: 202, body: '' });
            } else {
                route.fallback();
            }
        });

        await page.goto('/models');
        const card = page.locator('#model-card-mock-model-1');
        await card.locator('.btn-download').click();

        const progressContainer = card.locator('.download-progress-container');
        await expect(progressContainer).toBeVisible();

        // Click cancel and verify it escapes the retry loop
        await progressContainer.locator('.dl-cancel-btn').click();
        await expect(progressContainer.locator('.download-stats')).toContainText('Download Canceled.');
    });

    test('Models Directory recovers from network interruptions during download', async ({ page }) => {
        const downloadState = {};
        let isDownloaded = false;
        
        await page.route('**/api/models', async route => {
            route.fulfill({
                status: 200,
                json: [{ id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: isDownloaded }]
            });
        });

        await page.route('**/api/models/*', async route => {
            if (route.request().method() === 'GET') {
                const id = route.request().url().split('/').pop();
                if (id === 'mock-model-1') {
                    await route.fulfill({ status: 200, json: { id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: isDownloaded } });
                } else {
                    await route.fulfill({ status: 404 });
                }
            } else {
                route.fallback();
            }
        });

        let simulateDrop = false;
        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') {
                if (simulateDrop) {
                    await route.fulfill({ status: 502, body: 'Bad Gateway' });
                } else {
                    await route.fulfill({ status: 200, json: downloadState });
                }
            } else if (route.request().method() === 'POST') {
                downloadState['mock-model-1'] = { bytes_transferred: 50, total_bytes: 100, current_speed_bps: 10, start_time: 0, state: "Downloading..." };
                await route.fulfill({ status: 202, body: '' });
            } else {
                route.fallback();
            }
        });

        await page.goto('/models');
        const card = page.locator('#model-card-mock-model-1');
        await card.locator('.btn-download').click();

        const stats = card.locator('.download-stats');
        await expect(stats).toContainText('50.0%');
        simulateDrop = true; // Simulate network drop
        await expect(stats).toContainText('Retrying in 5s...');
        
        simulateDrop = false; // Recover network
        downloadState['mock-model-1'].bytes_transferred = 100;
        await expect(stats).toContainText('100.0%', { timeout: 15000 }); // Wait for the 5s loop to recover
        
        delete downloadState['mock-model-1']; // Simulate Finish
        isDownloaded = true;
        await expect(card.locator('.badge-json', { hasText: 'Ready' })).toBeVisible();
    });

    test('Queue UI displays downloads and supports admin actions', async ({ page }) => {
        let deleteCalled = false;
        let pauseCalled = false;
        let clearAllCalled = false;

        await page.route('**/api/downloads/mock-model-2', async route => {
            if (route.request().method() === 'DELETE') {
                deleteCalled = true;
                await route.fulfill({ status: 200 });
            } else { route.fallback(); }
        });

        await page.route('**/api/downloads/mock-model-1/pause', async route => {
            if (route.request().method() === 'POST') {
                pauseCalled = true;
                await route.fulfill({ status: 200 });
            } else { route.fallback(); }
        });

        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        'mock-model-1': { bytes_transferred: 5242880, total_bytes: 10485760, current_speed_bps: 1048576, start_time: 0, state: 'Downloading...' },
                        'mock-model-2': { bytes_transferred: 0, total_bytes: 0, current_speed_bps: 0, start_time: 0, state: 'Queued...' }
                    }
                });
            } else if (route.request().method() === 'DELETE') {
                clearAllCalled = true;
                await route.fulfill({ status: 200 });
            } else { route.fallback(); }
        });

        // Automatically accept any confirm() dialogs (Clear All / Cancel)
        page.on('dialog', dialog => dialog.accept());

        await page.goto('/queue');

        const tbody = page.locator('#queue-tbody');
        await expect(tbody).toContainText('mock-model-1');
        await expect(tbody).toContainText('mock-model-2');
        await expect(tbody).toContainText('Downloading...');
        await expect(tbody).toContainText('Queued...');

        // Test individual Cancel
        await page.locator('tr', { hasText: 'mock-model-2' }).locator('button', { hasText: 'Cancel' }).click();
        expect(deleteCalled).toBe(true);

        // Test individual Pause
        await page.locator('tr', { hasText: 'mock-model-1' }).locator('button', { hasText: 'Pause' }).click();
        expect(pauseCalled).toBe(true);

        // Test Clear All
        await page.locator('#clear-all-btn').click();
        expect(clearAllCalled).toBe(true);
    });

    test('Queue UI sanitizes malicious XSS payloads in model IDs and status', async ({ page }) => {
        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        // Inject XSS payload into the Model ID
                        '<img src="x" onerror="alert(\'XSS-ID\')">malicious-model': { 
                            bytes_transferred: 50, 
                            total_bytes: 100, 
                            current_speed_bps: 10, 
                            start_time: 0, 
                            // Inject XSS payload into the Status string
                            state: '<script>alert("XSS-STATUS")</script>Downloading...' 
                        }
                    }
                });
            } else { 
                route.fallback(); 
            }
        });

        // Listen for browser dialogs. If an alert fires, the XSS attack succeeded!
        let alertFired = false;
        page.on('dialog', dialog => {
            alertFired = true;
            dialog.dismiss();
        });

        await page.goto('/queue');

        const tbody = page.locator('#queue-tbody');
        
        // Wait for the table to render the stripped text
        await expect(tbody).toContainText('malicious-model');
        await expect(tbody).toContainText('Downloading...');

        // Explicitly verify the malicious tags were completely purged from the inner HTML
        const rowHtml = await tbody.innerHTML();
        expect(rowHtml).not.toContain('<img');
        expect(rowHtml).not.toContain('<script');
        expect(alertFired).toBe(false);
    });

    test('SharedDownloadProgress prevents redundant API requests', async ({ page }) => {
        let apiCallCount = 0;
        
        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') {
                apiCallCount++;
                await new Promise(resolve => setTimeout(resolve, 50));
                await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
            } else {
                route.fallback();
            }
        });

        // Use the chat page, which does not aggressively poll on load by default
        await page.goto('/');
        apiCallCount = 0; // Reset in case any immediate load checks occurred

        // Fire 5 simultaneous requests from the UI
        await page.evaluate(async () => {
            const promises = Array.from({ length: 5 }).map(() => SharedDownloadProgress.get());
            await Promise.all(promises);
        });

        // The poller should have coalesced all 5 requests into exactly 1 network call
        expect(apiCallCount).toBe(1);

        // Wait for the 500ms cache TTL to expire
        await page.waitForTimeout(600);

        // Fire another request; it should bypass the expired cache
        await page.evaluate(async () => await SharedDownloadProgress.get());
        expect(apiCallCount).toBe(2);
    });

    test('Models Directory handles download dropped by server (ServerDropped)', async ({ page }) => {
        await page.route('**/api/models', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: [{ id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: false }]
                });
            } else {
                route.fallback();
            }
        });

        await page.route('**/api/models/mock-model-1', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({ status: 200, json: { is_downloaded: false } });
            } else {
                route.fallback();
            }
        });

        let postCount = 0;
        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') {
                if (postCount === 1) {
                    await route.fulfill({ status: 200, json: { 'mock-model-1': { bytes_transferred: 10, total_bytes: 100, current_speed_bps: 10, start_time: 0, state: 'Downloading...' } } });
                    postCount++; // Increment so next poll gets empty object
                } else {
                    await route.fulfill({ status: 200, json: {} });
                }
            } else if (route.request().method() === 'POST') {
                postCount++;
                await route.fulfill({ status: 202, body: '' });
            } else {
                route.fallback();
            }
        });

        await page.goto('/models');
        
        const card = page.locator('#model-card-mock-model-1');
        await card.locator('.btn-download').click();

        await expect(card.locator('.download-stats')).toContainText('Download Stopped.', { timeout: 10000 });
        expect(postCount).toBe(2);
    });

    test('Models Directory retries on network errors or 429 Too Many Requests', async ({ page }) => {
        await page.route('**/api/models', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: [{ id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: false }]
                });
            } else {
                route.fallback();
            }
        });

        let postCalled = false;
        let errorServed = false;
        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'POST') {
                postCalled = true;
                await route.fulfill({ status: 202, body: '' });
            } else if (route.request().method() === 'GET') {
                if (postCalled && !errorServed) {
                    errorServed = true;
                    await route.fulfill({ status: 502, body: 'Bad Gateway' });
                } else if (postCalled && errorServed) {
                    await route.fulfill({ status: 200, json: { 'mock-model-1': { bytes_transferred: 50, total_bytes: 100, current_speed_bps: 1000, start_time: 0, state: 'Downloading...' } } });
                } else {
                    // Before POST, just return empty so discoverActiveDownloads does nothing
                    await route.fulfill({ status: 200, json: {} });
                }
            } else {
                route.fallback();
            }
        });

        await page.goto('/models');
        
        const card = page.locator('#model-card-mock-model-1');
        await card.locator('.btn-download').click();

        await expect(card.locator('.download-stats')).toContainText('Retrying', { timeout: 6000 });
        await expect(card.locator('.download-stats')).toContainText('50.0%', { timeout: 10000 });
    });

    test('UI redirects to login on 401 Unauthorized API response', async ({ page }) => {
        // Mock a 401 response for the initial status check
        await page.route('**/api/status', async route => {
            await route.fulfill({ status: 401, body: 'Unauthorized' });
        });

        // Mock the destination login page so Playwright doesn't fail on an unhandled navigation
        await page.route('**/auth/login', async route => {
            await route.fulfill({ status: 200, contentType: 'text/html', body: '<html><body>Mock Login Page</body></html>' });
        });

        await page.goto('/');

        // Verify the global fetchWithAuth helper caught the 401 and redirected the window
        await expect(page).toHaveURL(/.*\/auth\/login/);
    });
});