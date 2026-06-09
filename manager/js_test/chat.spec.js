import { test, expect } from '@playwright/test';
import { AxeBuilder } from '@axe-core/playwright';
import { mockStaticAssets, mockEngineApis, setupPageErrorHandlers } from './utils.js';

test.describe('Mini Inference Engine - Chat UI', () => {
    test.beforeEach(async ({ page }, testInfo) => {
        setupPageErrorHandlers(page, testInfo);
        await mockStaticAssets(page);
        await mockEngineApis(page);
    });

    test('Navbar dynamically injects and renders', async ({ page }) => {
        await page.goto('/');
        await expect(page.locator('h1')).toContainText('Mini Inference Engine');
        
        const chatBtn = page.locator('button:has-text("💬 Chat")');
        await expect(chatBtn).toBeVisible();
    });
    
    test('Chat UI should not have any automatically detectable accessibility issues', async ({ page }) => {
        await page.goto('/');
        await expect(page.locator('#chat-model-select option')).not.toHaveCount(0);
        const accessibilityScanResults = await new AxeBuilder({ page }).analyze();
        expect(accessibilityScanResults.violations).toEqual([]);
    });

    test('Chat UI streams a mock generation', async ({ page }) => {
        await page.route('**/api/generate', route => {
            const streamResponse = 
                '{"metadata":{"id":"mock-prompt-id","timestamp":1000,"token_counts":{"mock-model-1":5}}}\n' +
                '{"token":"Hello! I am a simulated AI response."}\n' +
                '{"metadata":{"id":"mock-ai-id","parent_id":"mock-prompt-id","timestamp":1002,"model":"mock-model-1","generation_time_ms":100,"token_counts":{"mock-model-1":15},"parameters":{}}}\n';
            route.fulfill({ status: 200, body: streamResponse, contentType: 'text/plain' });
        });

        await page.goto('/');
        await expect(page.locator('#chat-model-select option')).not.toHaveCount(0);
        await expect(page.locator('#compressor-model-select option')).not.toHaveCount(0);

        const input = page.locator('#prompt-input');
        const sendBtn = page.locator('#send-btn');
        
        await input.fill('Can you hear me?');
        await sendBtn.click();

        await expect(page.locator('.user-message').last()).toContainText('Can you hear me?');
        await expect(page.locator('.ai-message').last()).toContainText('Hello! I am a simulated AI response.');
        await expect(page.locator('.msg-metadata').first()).toContainText('Tokens: mock-model-1: 5');
        await expect(page.locator('.msg-metadata').last()).toContainText('Tokens: mock-model-1: 15');
        await expect(page.locator('#regen-btn')).toBeVisible();
    });

    test('Chat UI safely ignores null metadata in streaming response', async ({ page }) => {
        await page.route('**/api/generate', route => {
            const streamResponse = 
                '{"metadata":null}\n' +
                '{"token":"Response with null metadata"}\n';
            route.fulfill({ status: 200, body: streamResponse, contentType: 'text/plain' });
        });

        await page.goto('/');
        await page.locator('#prompt-input').fill('Test null metadata');
        await page.locator('#send-btn').click();
        await expect(page.locator('.ai-message').last()).toContainText('Response with null metadata');
    });

    test('Chat UI handles error messages emitted mid-stream', async ({ page }) => {
        await page.route('**/api/generate', route => {
            const streamResponse = 
                '{"token":"Partial response"}\n' +
                '{"error":"Context length exceeded"}\n';
            route.fulfill({ status: 200, body: streamResponse, contentType: 'text/plain' });
        });

        await page.goto('/');
        await page.locator('#prompt-input').fill('Test stream error');
        await page.locator('#send-btn').click();
        await expect(page.locator('.ai-message').last()).toContainText('Partial response\nError: Context length exceeded');
    });

    test('Chat UI generates fallback UUID in non-secure contexts', async ({ page }) => {
        await page.addInitScript(() => {
            if (window.crypto) window.crypto.randomUUID = undefined;
        });
        await page.route('**/api/generate', route => {
            route.fulfill({ status: 200, body: '{"token":"Fallback UUID worked!"}\n', contentType: 'text/plain' });
        });

        await page.goto('/');
        await expect(page.locator('#chat-model-select option')).not.toHaveCount(0);
        await page.locator('#prompt-input').fill('Test fallback UUID');
        await page.locator('#send-btn').click();
        await expect(page.locator('.user-message').last()).toContainText('Test fallback UUID');
        await expect(page.locator('.ai-message').last()).toContainText('Fallback UUID worked!');
    });

    test('Chat UI renders message metadata correctly and handles scoring updates', async ({ page }) => {
        const handleMetaSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: [{ id: 'session-meta', title: 'Meta Chat', updated_at: 1678886400, email: 'mock@example.com' }]
                });
            } else { route.fallback(); }
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
            } else { route.fallback(); }
        });

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Meta Chat' }).click();
        await expect(page.locator('.user-message').last()).toContainText('What is Rust?');
        
        const userMeta = page.locator('.msg-metadata').first();
        await expect(userMeta).toContainText('Tokens: mock-model-1: 10');
        
        const aiMeta = page.locator('.msg-metadata').last();
        await expect(aiMeta).toContainText('Model: mock-model-1');
        await expect(aiMeta).toContainText('Time: 1.50s');
        await expect(aiMeta).toContainText('Tokens: mock-model-1: 25');
        
        const scoreSelect = aiMeta.locator('select');
        const [request] = await Promise.all([
            page.waitForRequest(req => req.url().includes('/api/chat/sessions/session-meta/messages') && req.method() === 'POST'),
            scoreSelect.selectOption('5')
        ]);
        
        const postData = JSON.parse(request.postData());
        expect(postData.score).toBe(5);
    });

    test('Chat UI rolls back AI message score if saving to DB fails', async ({ page }) => {
        const handleScoreDbFailSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-score-db-fail', title: 'Score DB Fail Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Prompt 1',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'Response 1',
                            metadata: { id: 'msg-2', parent_id: 'msg-1', timestamp: 1001, model: 'mock-model-1', score: 3 }
                        }]
                    }
                });
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-score-db-fail*', handleScoreDbFailSessionRoute);
        const handleScoreDbFailSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-score-db-fail', title: 'Score DB Fail Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleScoreDbFailSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleScoreDbFailSessionsRoute);

        await page.route('**/api/chat/sessions/session-score-db-fail/messages', async route => {
            if (route.request().method() === 'POST') {
                await route.fulfill({ status: 500, body: 'Internal Server Error' });
            } else { route.fallback(); }
        });

        let alertText = null;
        page.on('dialog', dialog => { if (dialog.type() === 'alert') { alertText = dialog.message(); dialog.accept(); } });

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Score DB Fail Chat' }).click();

        const aiMeta = page.locator('.msg-metadata').last();
        const scoreSelect = aiMeta.locator('select');
        await expect(scoreSelect).toHaveValue('3');

        await scoreSelect.selectOption('5');

        await expect.poll(() => alertText).toContain('Failed to save score. Please check your connection.');
        
        // Assert the DOM select element rolled back to the previous score
        await expect(scoreSelect).toHaveValue('3');
    });

    test('Chat UI safely handles corrupted trees with orphaned nodes from the backend', async ({ page }) => {
        const handleCorruptTreeSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-corrupt-tree', title: 'Corrupt Tree Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Valid Root Prompt',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'Valid Response',
                            metadata: { id: 'msg-2', parent_id: 'msg-1', timestamp: 1001, model: 'mock-model-1' }
                        }, {
                            role: 'user', content: 'Orphaned Prompt',
                            metadata: { id: 'msg-4', parent_id: 'msg-missing', timestamp: 1002 } // Missing parent
                        }, {
                            role: 'assistant', content: 'Orphaned Response',
                            metadata: { id: 'msg-5', parent_id: 'msg-4', timestamp: 1003, model: 'mock-model-1' }
                        }]
                    }
                });
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-corrupt-tree*', handleCorruptTreeSessionRoute);
        const handleCorruptSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-corrupt-tree', title: 'Corrupt Tree Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleCorruptSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleCorruptSessionsRoute);

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Corrupt Tree Chat' }).click();

        // The orphaned node should be treated as a second root branch.
        // Because its timestamp (1003) is newer, getDefaultLeaf should select it as the active path.
        await expect(page.locator('.user-message').first()).toContainText('Orphaned Prompt');
        await expect(page.locator('.ai-message').last()).toContainText('Orphaned Response');
        
        // Since it's treated as a root node and there is another root node (msg-1), 
        // the branch controls should show 2 of 2 roots.
        const branchControls = page.locator('.branch-controls').first();
        await expect(branchControls).toBeVisible();
        await expect(branchControls).toContainText('2 of 2');
        
        // Switch to the first root branch
        await branchControls.locator('span', { hasText: '◀' }).click();
        await expect(page.locator('.user-message').first()).toContainText('Valid Root Prompt');
        await expect(page.locator('.ai-message').last()).toContainText('Valid Response');
        await expect(branchControls).toContainText('1 of 2');
    });

    test('Chat UI loads existing sessions and allows switching', async ({ page }) => {
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

        await page.locator('.session-item').last().click();
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
        const session2 = page.locator('.session-item', { hasText: 'Second Chat Session' });
        await session2.click();
        await expect(session2).toHaveClass(/active/);
        await expect(page.locator('.ai-message').last()).toContainText('Persistent message');

        await page.reload();
        const reloadedSession2 = page.locator('.session-item', { hasText: 'Second Chat Session' });
        await expect(reloadedSession2).toHaveClass(/active/);
        await expect(page.locator('.ai-message').last()).toContainText('Persistent message');

        await page.goto('/models');
        await page.goto('/');
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
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-branch*', handleSessionRoute);
        const handleBranchSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-branch', title: 'Branch Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
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

        await expect(page.locator('.user-message').first()).toContainText('Original Prompt');
        const editBtn = page.locator('button[title="Edit Prompt (Creates New Branch)"]');
        await expect(editBtn).toBeVisible();
        await editBtn.click();

        const input = page.locator('#prompt-input');
        await expect(input).toHaveValue('Original Prompt');
        await input.fill('New Prompt');
        await page.locator('#send-btn').click();

        await expect(page.locator('.user-message').last()).toContainText('New Prompt');
        await expect(page.locator('.ai-message').last()).toContainText('New Response');

        const branchControls = page.locator('.branch-controls');
        await expect(branchControls).toBeVisible();
        await expect(branchControls).toContainText('2 of 2');

        await branchControls.locator('span', { hasText: '◀' }).click();
        await expect(page.locator('.user-message').last()).toContainText('Original Prompt');
        await expect(page.locator('.ai-message').last()).toContainText('Original Response');
        await expect(branchControls).toContainText('1 of 2');

        await branchControls.locator('span', { hasText: '◫ Split View' }).click();
        const splitContainer = page.locator('.split-container');
        await expect(splitContainer).toBeVisible();
        await expect(splitContainer.locator('.split-item').first()).toContainText('Original Prompt');
        await expect(splitContainer.locator('.split-item').last()).toContainText('New Prompt');

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
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-branch-state*', handleSessionRoute);
        const handleBranchSessionsRoute = route => {
            if (route.request().method() === 'GET') { route.fulfill({ status: 200, json: [{ id: 'session-branch-state', title: 'Branch State Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions', handleBranchSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleBranchSessionsRoute);

        let capturedParentId = "not_captured";
        await page.route('**/api/chat/sessions/*/messages', async route => {
            if (route.request().method() === 'POST') {
                const body = JSON.parse(route.request().postData());
                if (body.role === 'user') { capturedParentId = body.parent_id; }
                await route.fulfill({ status: 200 });
            } else { route.fallback(); }
        });

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Branch State Chat' }).click();

        await expect(page.locator('.user-message').last()).toContainText('Prompt 2');
        const editBtn = page.locator('button[title="Edit Prompt (Creates New Branch)"]').last();
        await expect(editBtn).toBeVisible();
        await editBtn.click();
        
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
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-regen*', handleRegenSessionRoute);
        const handleRegenSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-regen', title: 'Regenerate Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
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
        await page.locator('#regen-btn').click();
        await expect(page.locator('.ai-message').last()).toContainText('Regenerated Response');
        
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
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-regen-missing*', handleRegenMissingSessionRoute);
        const handleRegenMissingSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-regen-missing', title: 'Regenerate Missing Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleRegenMissingSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleRegenMissingSessionsRoute);

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Regenerate Missing Chat' }).click();
        await expect(page.locator('.ai-message').last()).toContainText('Original Response');
        
        await page.evaluate(() => {
            // eslint-disable-next-line no-undef
            messagesMap.delete(currentLeafId);
        });

        await page.locator('#regen-btn').click();
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
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-missing-meta*', handleMissingMetaSessionRoute);
        const handleMissingMetaSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-missing-meta', title: 'Missing Meta Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
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
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-legacy*', handleLegacySessionRoute);
        const handleLegacySessionsList = route => { route.fulfill({ status: 200, json: [{ id: 'session-legacy', title: 'Legacy Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleLegacySessionsList);
        await page.route('**/api/chat/sessions?*', handleLegacySessionsList);

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Legacy Chat' }).click();

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
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-legacy-undefined*', handleLegacyUndefinedRoute);
        const handleLegacyUndefinedList = route => { route.fulfill({ status: 200, json: [{ id: 'session-legacy-undefined', title: 'Legacy Chat Undefined', updated_at: 1678886400, email: 'mock@example.com' }] }); };
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
            } else { route.fallback(); }
        });
        const handlePruneSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-prune', title: 'Prune Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handlePruneSessionsRoute);
        await page.route('**/api/chat/sessions?*', handlePruneSessionsRoute);

        let deleteRequestReceived = false;
        await page.route('**/api/chat/sessions/session-prune/messages', async route => {
            if (route.request().method() === 'DELETE') {
                deleteRequestReceived = true;
                const body = JSON.parse(route.request().postData());
                expect(body.message_ids).toContain('msg-2');
                await route.fulfill({ status: 200 });
            } else { route.fallback(); }
        });

        page.on('dialog', dialog => dialog.accept());

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
                            metadata: { id: 'msg-2', parent_id: 'msg-1', timestamp: 1001, score: 5 }
                        }, {
                            role: 'assistant', content: 'Recent Low Score Branch',
                            metadata: { id: 'msg-3', parent_id: 'msg-1', timestamp: 1002, score: 1 }
                        }]
                    }
                });
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-score*', handleScoreSessionRoute);
        const handleScoreSessionsList = route => { route.fulfill({ status: 200, json: [{ id: 'session-score', title: 'Score Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleScoreSessionsList);
        await page.route('**/api/chat/sessions?*', handleScoreSessionsList);

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Score Chat' }).click();

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
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-deep-prune*', handleDeepPruneSessionRoute);
        const handleDeepPruneSessionsList = route => { route.fulfill({ status: 200, json: [{ id: 'session-deep-prune', title: 'Deep Prune Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleDeepPruneSessionsList);
        await page.route('**/api/chat/sessions?*', handleDeepPruneSessionsList);

        let deletePayload = null;
        await page.route('**/api/chat/sessions/session-deep-prune/messages', async route => {
            if (route.request().method() === 'DELETE') {
                deletePayload = JSON.parse(route.request().postData());
                await route.fulfill({ status: 200 });
            } else { route.fallback(); }
        });

        page.on('dialog', dialog => dialog.accept());

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Deep Prune Chat' }).click();

        await expect(page.locator('.user-message').last()).toContainText('Child of Branch A');
        const branchControls = page.locator('#msg-wrapper-msg-2 .branch-controls');
        await expect(branchControls).toBeVisible();
        
        await branchControls.locator('span', { hasText: '🗑️ Prune' }).click();

        expect(deletePayload).not.toBeNull();
        expect(deletePayload.message_ids).toContain('msg-2');
        expect(deletePayload.message_ids).toContain('msg-4');
        expect(deletePayload.message_ids).not.toContain('msg-1');
        expect(deletePayload.message_ids).not.toContain('msg-3');

        await expect(page.locator('.ai-message').last()).toContainText('Branch B');
    });

    test('Chat UI can prune a root branch and correctly falls back to a sibling root', async ({ page }) => {
        const handleRootPruneSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-root-prune', title: 'Root Prune Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Root Prompt A',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'user', content: 'Root Prompt B',
                            metadata: { id: 'msg-2', parent_id: null, timestamp: 1001 }
                        }]
                    }
                });
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-root-prune*', handleRootPruneSessionRoute);
        const handleRootPruneSessionsList = route => { route.fulfill({ status: 200, json: [{ id: 'session-root-prune', title: 'Root Prune Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleRootPruneSessionsList);
        await page.route('**/api/chat/sessions?*', handleRootPruneSessionsList);

        let deletePayload = null;
        await page.route('**/api/chat/sessions/session-root-prune/messages', async route => {
            if (route.request().method() === 'DELETE') {
                deletePayload = JSON.parse(route.request().postData());
                await route.fulfill({ status: 200 });
            } else { route.fallback(); }
        });

        page.on('dialog', dialog => dialog.accept());

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Root Prune Chat' }).click();

        // Active path should default to Root Prompt B (newer timestamp)
        await expect(page.locator('.user-message').first()).toContainText('Root Prompt B');
        const branchControls = page.locator('.branch-controls').first();
        await expect(branchControls).toBeVisible();
        await expect(branchControls).toContainText('2 of 2');

        // Prune Root Prompt B
        await branchControls.locator('span', { hasText: '🗑️ Prune' }).click();

        // Verify Delete API payload
        expect(deletePayload).not.toBeNull();
        expect(deletePayload.message_ids).toContain('msg-2');
        expect(deletePayload.message_ids).not.toContain('msg-1');

        // UI should fall back to Root Prompt A and remove branch controls since no siblings remain
        await expect(page.locator('.user-message').first()).toContainText('Root Prompt A');
        await expect(page.locator('.branch-controls')).not.toBeVisible();
    });

    test('Chat UI automatically collapses Split View when pruning leaves only one sibling', async ({ page }) => {
        const handleSplitPruneSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-split-prune', title: 'Split Prune Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Root Prompt',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'assistant', content: 'Response A',
                            metadata: { id: 'msg-2', parent_id: 'msg-1', timestamp: 1001, model: 'mock-model-1' }
                        }, {
                            role: 'assistant', content: 'Response B',
                            metadata: { id: 'msg-3', parent_id: 'msg-1', timestamp: 1002, model: 'mock-model-1' }
                        }]
                    }
                });
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-split-prune*', handleSplitPruneSessionRoute);
        const handleSplitPruneSessionsList = route => { route.fulfill({ status: 200, json: [{ id: 'session-split-prune', title: 'Split Prune Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleSplitPruneSessionsList);
        await page.route('**/api/chat/sessions?*', handleSplitPruneSessionsList);

        await page.route('**/api/chat/sessions/session-split-prune/messages', async route => {
            if (route.request().method() === 'DELETE') { await route.fulfill({ status: 200 }); } else { route.fallback(); }
        });

        page.on('dialog', dialog => dialog.accept());

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Split Prune Chat' }).click();

        // Active path should default to Response B (newer timestamp)
        await expect(page.locator('.ai-message').last()).toContainText('Response B');
        const branchControls = page.locator('.branch-controls').last();
        await expect(branchControls).toBeVisible();

        // Open Split View
        await branchControls.locator('span', { hasText: '◫ Split View' }).click();
        const splitContainer = page.locator('.split-container');
        await expect(splitContainer).toBeVisible();
        await expect(splitContainer.locator('.split-item')).toHaveCount(2);

        // Prune Response B
        await branchControls.locator('span', { hasText: '🗑️ Prune' }).click();

        // UI should fall back to Response A and Split View should be removed
        await expect(page.locator('.ai-message').last()).toContainText('Response A');
        await expect(page.locator('.split-container')).toHaveCount(0); 
        await expect(page.locator('.branch-controls')).toHaveCount(0); 
    });

    test('Chat UI resists UI state corruption when branch pruning fails on the backend', async ({ page }) => {
        await page.route('**/api/chat/sessions/session-prune-fail*', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-prune-fail', title: 'Prune Fail Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Prompt 1',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }, {
                            role: 'user', content: 'Prompt 2',
                            metadata: { id: 'msg-2', parent_id: null, timestamp: 1001 }
                        }]
                    }
                });
            } else { route.fallback(); }
        });
        const handlePruneFailSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-prune-fail', title: 'Prune Fail Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handlePruneFailSessionsRoute);
        await page.route('**/api/chat/sessions?*', handlePruneFailSessionsRoute);
        await page.route('**/api/chat/sessions/session-prune-fail/messages', async route => {
            if (route.request().method() === 'DELETE') { await route.fulfill({ status: 500, body: 'Internal Server Error' }); } else { route.fallback(); }
        });

        let alertText = null;
        page.on('dialog', dialog => {
            if (dialog.type() === 'confirm') dialog.accept();
            else if (dialog.type() === 'alert') { alertText = dialog.message(); dialog.accept(); }
        });

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Prune Fail Chat' }).click();

        const branchControls = page.locator('.branch-controls');
        await expect(branchControls).toBeVisible();
        await expect(page.locator('.user-message').last()).toContainText('Prompt 2');

        await branchControls.locator('span', { hasText: '🗑️ Prune' }).click();
        await expect.poll(() => alertText).toContain('Failed to delete branch');
        await expect(page.locator('.user-message').last()).toContainText('Prompt 2');
        await expect(branchControls).toBeVisible();
    });

    test('Chat UI resists UI state corruption when clearChat fails on the backend', async ({ page }) => {
        const handleClearFailSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-clear-fail', title: 'Clear Fail Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Prompt 1',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }]
                    }
                });
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-clear-fail*', handleClearFailSessionRoute);
        const handleClearFailSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-clear-fail', title: 'Clear Fail Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleClearFailSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleClearFailSessionsRoute);
        await page.route('**/api/chat/sessions/session-clear-fail', async route => {
            if (route.request().method() === 'DELETE') { await route.fulfill({ status: 500, body: 'Internal Server Error' }); } else { route.fallback(); }
        });

        let alertText = null;
        page.on('dialog', dialog => { if (dialog.type() === 'alert') { alertText = dialog.message(); dialog.accept(); } });

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Clear Fail Chat' }).click();
        await expect(page.locator('.user-message').last()).toContainText('Prompt 1');

        await page.locator('#clear-chat-btn').click();
        await page.locator('#delete-confirm-btn').click();

        await expect.poll(() => alertText).toContain('Failed to delete chat session');
        await expect(page.locator('.user-message').last()).toContainText('Prompt 1');
        await expect(page.locator('.session-item', { hasText: 'Clear Fail Chat' })).toBeVisible();
    });

    test('Chat UI rolls back optimistic message updates if backend save fails', async ({ page }) => {
        const handleSendFailSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-send-fail', title: 'Send Fail Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Prompt 1',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }]
                    }
                });
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-send-fail*', handleSendFailSessionRoute);
        const handleSendFailSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-send-fail', title: 'Send Fail Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleSendFailSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleSendFailSessionsRoute);

        let resolvePost;
        const postPromise = new Promise(res => resolvePost = res);
        await page.route('**/api/chat/sessions/session-send-fail/messages', async route => {
            if (route.request().method() === 'POST') {
                await postPromise;
                await route.fulfill({ status: 500, body: 'Internal Server Error' });
            } else { route.fallback(); }
        });

        let alertText = null;
        page.on('dialog', dialog => { if (dialog.type() === 'alert') { alertText = dialog.message(); dialog.accept(); } });

        await page.goto('/');
        await expect(page.locator('#chat-model-select option')).not.toHaveCount(0);
        await page.locator('.session-item', { hasText: 'Send Fail Chat' }).click();
        await expect(page.locator('.user-message').last()).toContainText('Prompt 1');

        const input = page.locator('#prompt-input');
        await input.fill('This message will fail to save');
        await page.locator('#send-btn').click();

        await expect(page.locator('#send-btn')).toBeDisabled();
        resolvePost();
        await expect.poll(() => alertText).toContain('Failed to send message to server');

        await expect(page.locator('.user-message').last()).not.toContainText('This message will fail to save');
        await expect(page.locator('.user-message').last()).toContainText('Prompt 1');
        await expect(input).toHaveValue('This message will fail to save');
        await expect(page.locator('#send-btn')).toBeEnabled();
    });

    test('Chat UI rolls back optimistic AI message node if generation fails', async ({ page }) => {
        const handleGenFailSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-gen-fail', title: 'Gen Fail Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Prompt 1',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }]
                    }
                });
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-gen-fail*', handleGenFailSessionRoute);
        const handleGenFailSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-gen-fail', title: 'Gen Fail Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleGenFailSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleGenFailSessionsRoute);

        await page.route('**/api/chat/sessions/session-gen-fail/messages', async route => {
            if (route.request().method() === 'POST') {
                await route.fulfill({ status: 200 }); // User message saves successfully
            } else { route.fallback(); }
        });

        let resolveGenerate;
        const generatePromise = new Promise(res => resolveGenerate = res);
        await page.route('**/api/generate', async route => {
            await generatePromise;
            await route.fulfill({ status: 500, body: 'Internal Server Error' });
        });

        let alertText = null;
        page.on('dialog', dialog => { if (dialog.type() === 'alert') { alertText = dialog.message(); dialog.accept(); } });

        await page.goto('/');
        await expect(page.locator('#chat-model-select option')).not.toHaveCount(0);
        await page.locator('.session-item', { hasText: 'Gen Fail Chat' }).click();
        await expect(page.locator('.user-message').last()).toContainText('Prompt 1');

        const input = page.locator('#prompt-input');
        await input.fill('Trigger generation failure');
        await page.locator('#send-btn').click();

        await expect(page.locator('.user-message').last()).toContainText('Trigger generation failure');
        
        resolveGenerate();
        await expect.poll(() => alertText).toContain('Failed to connect to engine or generate response');

        await expect(page.locator('.ai-message')).toHaveCount(0); // Ensure the AI node was purged
        await expect(page.locator('.user-message')).toHaveCount(2); // 'Prompt 1' + new prompt
        await expect(page.locator('#send-btn')).toBeEnabled();
    });

    test('Chat UI rolls back AI message if saving to DB fails after generation', async ({ page }) => {
        const handleDbFailSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-ai-db-fail', title: 'AI DB Fail Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Prompt 1',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }]
                    }
                });
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-ai-db-fail*', handleDbFailSessionRoute);
        const handleDbFailSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-ai-db-fail', title: 'AI DB Fail Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleDbFailSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleDbFailSessionsRoute);

        await page.route('**/api/generate', route => {
            const streamResponse = '{"token":"Successfully generated but database save will fail"}\n';
            route.fulfill({ status: 200, body: streamResponse, contentType: 'text/plain' });
        });

        let postCount = 0;
        await page.route('**/api/chat/sessions/session-ai-db-fail/messages', async route => {
            if (route.request().method() === 'POST') {
                postCount++;
                if (postCount === 1) {
                    await route.fulfill({ status: 200 }); // User message saves successfully
                } else {
                    await route.fulfill({ status: 500, body: 'Internal Server Error' }); // AI message DB save fails
                }
            } else { route.fallback(); }
        });

        let alertText = null;
        page.on('dialog', dialog => { if (dialog.type() === 'alert') { alertText = dialog.message(); dialog.accept(); } });

        await page.goto('/');
        await expect(page.locator('#chat-model-select option')).not.toHaveCount(0);
        await page.locator('.session-item', { hasText: 'AI DB Fail Chat' }).click();
        
        await expect(page.locator('.user-message').last()).toContainText('Prompt 1');

        await page.locator('#prompt-input').fill('Trigger AI DB fail');
        await page.locator('#send-btn').click();

        await expect.poll(() => alertText).toContain('Failed to save AI response to the database');
        
        await expect(page.locator('.ai-message')).toHaveCount(0); // Ensure the AI node was purged
        await expect(page.locator('.user-message')).toHaveCount(2); // 'Prompt 1' + new prompt
        await expect(page.locator('#send-btn')).toBeEnabled();
    });

    test('Chat UI rolls back AI message if saving to DB fails after Stop is clicked', async ({ page }) => {
        const handleAbortDbFailSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-abort-db-fail', title: 'Abort DB Fail Chat', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'user', content: 'Prompt 1',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }]
                    }
                });
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-abort-db-fail*', handleAbortDbFailSessionRoute);
        const handleAbortDbFailSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-abort-db-fail', title: 'Abort DB Fail Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
        await page.route('**/api/chat/sessions', handleAbortDbFailSessionsRoute);
        await page.route('**/api/chat/sessions?*', handleAbortDbFailSessionsRoute);

        let continueStream;
        const streamPromise = new Promise(res => continueStream = res);
        await page.route('**/api/generate', async route => {
            await streamPromise;
            route.fulfill({ status: 200, body: '{"token":"Partial"}\n', contentType: 'text/plain' });
        });

        let postCount = 0;
        await page.route('**/api/chat/sessions/session-abort-db-fail/messages', async route => {
            if (route.request().method() === 'POST') {
                postCount++;
                if (postCount === 1) {
                    await route.fulfill({ status: 200 }); // User message
                } else {
                    await route.fulfill({ status: 500, body: 'Internal Server Error' }); // AI message DB save fails
                }
            } else { route.fallback(); }
        });

        let alertText = null;
        page.on('dialog', dialog => { if (dialog.type() === 'alert') { alertText = dialog.message(); dialog.accept(); } });

        await page.goto('/');
        await expect(page.locator('#chat-model-select option')).not.toHaveCount(0);
        await page.locator('.session-item', { hasText: 'Abort DB Fail Chat' }).click();
        
        await expect(page.locator('.user-message').last()).toContainText('Prompt 1');

        await page.locator('#prompt-input').fill('Trigger abort DB fail');
        await page.locator('#send-btn').click();

        await expect(page.locator('#stop-btn')).toBeVisible();
        await page.locator('#stop-btn').click();
        continueStream(); // Unblock the route so the fetch aborts naturally

        await expect.poll(() => alertText).toContain('Failed to save the stopped response to the database');
        await expect(page.locator('.ai-message')).toHaveCount(0);
        await expect(page.locator('.user-message')).toHaveCount(2);
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
            } else { route.fallback(); }
        });
        const handleExportSessionsRoute = route => { route.fulfill({ status: 200, json: [{ id: 'session-export', title: 'Export Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); };
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
        await page.route('**/api/chat/sessions?*', async route => {
            await route.fulfill({ status: 200, json: [{ id: 'xss-session', title: '<script>alert("XSS-Title")</script>', updated_at: 1678886400, email: 'mock@example.com' }] });
        });
        await page.route('**/api/chat/sessions/xss-session*', async route => {
            await route.fulfill({
                status: 200,
                json: { 
                    id: 'xss-session', title: '<script>alert("XSS-Title")</script>', updated_at: 1678886400, email: 'mock@example.com', 
                    messages: [{ role: 'assistant', content: '<img src="x" onerror="alert(\'XSS-Message\')">Malicious Message' }] 
                }
            });
        });

        let alertFired = false;
        page.on('dialog', dialog => { alertFired = true; dialog.dismiss(); });

        await page.goto('/');
        const sessionItem = page.locator('.session-item[data-id="xss-session"]');
        await expect(sessionItem).toBeVisible();
        await expect(sessionItem.locator('.session-title')).toContainText('<script>');
        
        await sessionItem.click();
        const aiMessage = page.locator('.ai-message').last();
        await expect(aiMessage).toBeVisible();
        await expect(aiMessage).toContainText('<img src="x" onerror="alert(\'XSS-Message\')">Malicious Message');
        expect(alertFired).toBe(false);
    });

    test('Chat UI restores last session from localStorage even if not in initial API results', async ({ page }) => {
        await page.addInitScript(() => { window.localStorage.setItem('mini_inference_last_chat_id', 'off-page-session'); });
        await page.route('**/api/chat/sessions/off-page-session*', route => {
            route.fulfill({
                status: 200,
                json: { id: 'off-page-session', title: 'Off Page Chat Session', updated_at: 1678886600, email: 'mock@example.com', messages: [{ role: 'assistant', content: 'Message from the off-page session' }] }
            });
        });

        await page.goto('/');
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
            } else { await route.fallback(); }
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
        
        await expect(sessionItem).toContainText('Renamed Chat Session');
    });

    test('Chat UI alerts user if renaming a session fails', async ({ page }) => {
        const handleSessionRenameFailRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({ status: 200, json: [{ id: 'session-rename-fail', title: 'Original Title', updated_at: 1678886400, email: 'mock@example.com' }] });
            } else if (route.request().method() === 'POST') {
                await route.fulfill({ status: 500, body: 'Internal Server Error' });
            } else { await route.fallback(); }
        };
        await page.route('**/api/chat/sessions', handleSessionRenameFailRoute);
        await page.route('**/api/chat/sessions?*', handleSessionRenameFailRoute);

        let alertText = null;
        page.on('dialog', dialog => { if (dialog.type() === 'alert') { alertText = dialog.message(); dialog.accept(); } });

        await page.goto('/');
        const sessionItem = page.locator('.session-item').first();
        await expect(sessionItem).toContainText('Original Title');
        
        await sessionItem.hover();
        await sessionItem.locator('button[title="Rename Chat"]').click();
        
        await page.locator('#rename-input').fill('New Title');
        await page.locator('#rename-confirm-btn').click();
        
        await expect.poll(() => alertText).toContain('Failed to rename chat session');
        await expect(sessionItem).toContainText('Original Title');
    });

    test('Chat UI can delete a session via custom modal', async ({ page }) => {
        let fetchCount = 0;
        let deleteCalled = false;
        const handleSessionDeleteRoute = async route => {
            if (route.request().method() === 'GET') {
                if (fetchCount === 0) {
                    fetchCount++;
                    await route.fulfill({ status: 200, json: [{ id: 'session-to-delete', title: 'Delete Me', updated_at: 1678886400, email: 'mock@example.com' }] });
                } else { await route.fulfill({ status: 200, json: [] }); }
            } else { await route.fallback(); }
        };
        await page.route('**/api/chat/sessions', handleSessionDeleteRoute);
        await page.route('**/api/chat/sessions?*', handleSessionDeleteRoute);
        await page.route('**/api/chat/sessions/session-to-delete*', async route => {
            if (route.request().method() === 'DELETE') { deleteCalled = true; await route.fulfill({ status: 200 }); } else { await route.fallback(); }
        });

        await page.goto('/');
        const sessionItem = page.locator('.session-item').first();
        await expect(sessionItem).toContainText('Delete Me');
        
        await sessionItem.hover();
        await sessionItem.locator('button[title="Delete Chat"]').click();
        
        const deleteModal = page.locator('#delete-modal');
        await expect(deleteModal).toBeVisible();
        
        await page.locator('#delete-confirm-btn').click();
        
        await expect(deleteModal).not.toBeVisible();
        await expect(page.locator('.session-item')).toHaveCount(0);
        expect(deleteCalled).toBe(true);
    });

    test('Chat UI auto-scrolls to the newest message in a long session', async ({ page }) => {
        const longMessages = [];
        let lastId = null;
        for (let i = 0; i < 50; i++) {
            const id = `msg-${i}`;
            longMessages.push({ role: i % 2 === 0 ? 'user' : 'assistant', content: `Message number ${i}\nThis is a bit longer to take up vertical space.\nLine 3.`, metadata: { id, parent_id: lastId, timestamp: 1000 + i } });
            lastId = id;
        }
        await page.route('**/api/chat/sessions/session-long*', route => { route.fulfill({ status: 200, json: { id: 'session-long', title: 'Long Chat Session', updated_at: 1678886600, email: 'mock@example.com', messages: longMessages } }); });
        const handleLongSessionRoute = route => {
            if (route.request().method() === 'GET') { route.fulfill({ status: 200, json: [ { id: 'session-long', title: 'Long Chat Session', updated_at: 1678886600, email: 'mock@example.com' } ] }); } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions', handleLongSessionRoute);
        await page.route('**/api/chat/sessions?*', handleLongSessionRoute);

        await page.goto('/');
        const longSessionItem = page.locator('.session-item', { hasText: 'Long Chat Session' });
        await longSessionItem.click();

        await expect(longSessionItem).toHaveClass(/active/);
        await expect(page.locator('.message')).toHaveCount(50);

        await page.waitForFunction(() => {
            const container = document.getElementById('chat-container');
            return Math.abs(container.scrollHeight - container.scrollTop - container.clientHeight) <= 2;
        });
    });

    test('Chat UI initiates model download before generation if missing', async ({ page }) => {
        const downloadState = {};
        const modelState = [
            { id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: false },
            { id: 'mock-comp-1', name: 'Mock Compressor', roles: ['ContextCompressor'], supported_backends: ['Candle'], arch: 'XLMRoberta', parameters_billions: 0.5, size_on_disk_gb: 1.0, max_context_len: 1024, provenance: {}, is_downloaded: true }
        ];

        await page.route('**/api/models', route => { route.fulfill({ status: 200, json: modelState }); });
        await page.route('**/api/models/*', async route => {
            if (route.request().method() === 'GET') {
                const id = route.request().url().split('/').pop();
                const model = modelState.find(m => m.id === id);
                if (model) await route.fulfill({ status: 200, json: model });
                else await route.fulfill({ status: 404 });
            } else { route.fallback(); }
        });
        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({ status: 200, json: downloadState });
            } else if (route.request().method() === 'POST') {
                const postData = JSON.parse(route.request().postData());
                if (postData.model_id === 'mock-model-1') { downloadState['mock-model-1'] = { bytes_transferred: 50, total_bytes: 100, current_speed_bps: 1000000, start_time: 0, state: 'Downloading...' }; }
                await route.fulfill({ status: 202, body: '' });
            } else { route.fallback(); }
        });
        await page.route('**/api/generate', route => { route.fulfill({ status: 200, body: '{"token":"Generation after download."}\n', contentType: 'text/plain' }); });

        await page.goto('/');
        await expect(page.locator('#chat-model-select option')).not.toHaveCount(0);

        await page.locator('#prompt-input').fill('Test download');
        await page.locator('#send-btn').click();

        const progressContainer = page.locator('.download-progress-container');
        await expect(progressContainer).toBeVisible();
        await expect(progressContainer.locator('#dl-stats-mock-model-1')).toContainText('50.0%');

        delete downloadState['mock-model-1'];
        modelState[0].is_downloaded = true;
        await expect(page.locator('.ai-message').last()).toContainText('Generation after download.');
    });

    test('Chat UI aborts active generation when switching sessions', async ({ page }) => {
        const handleSwitchSessionRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        id: 'session-switch-target', title: 'Target Session', updated_at: 1678886400, email: 'mock@example.com',
                        messages: [{
                            role: 'assistant', content: 'Target loaded',
                            metadata: { id: 'msg-1', parent_id: null, timestamp: 1000 }
                        }]
                    }
                });
            } else { route.fallback(); }
        };
        await page.route('**/api/chat/sessions/session-switch-target*', handleSwitchSessionRoute);
        
        const handleSessionsList = route => { 
            route.fulfill({ 
                status: 200, 
                json: [
                    { id: 'session-switch-active', title: 'Active Session', updated_at: 1678886400, email: 'mock@example.com' },
                    { id: 'session-switch-target', title: 'Target Session', updated_at: 1678886500, email: 'mock@example.com' }
                ] 
            }); 
        };
        await page.route('**/api/chat/sessions', handleSessionsList);
        await page.route('**/api/chat/sessions?*', handleSessionsList);

        let continueGenerate;
        const generatePromise = new Promise(res => continueGenerate = res);
        
        await page.route('**/api/generate', async route => {
            await generatePromise;
            route.fulfill({ status: 200, body: '{"token":"Late"}\n', contentType: 'text/plain' }).catch(() => {});
        });

        let savedAbortedMessage = null;
        await page.route('**/api/chat/sessions/session-switch-active/messages', async route => {
            if (route.request().method() === 'POST') {
                const body = JSON.parse(route.request().postData());
                if (body.role === 'assistant') { savedAbortedMessage = body.content; }
                await route.fulfill({ status: 200 });
            } else { route.fallback(); }
        });

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Active Session' }).click();

        await page.locator('#prompt-input').fill('Tell me a long story');
        await page.locator('#send-btn').click();
        await expect(page.locator('#stop-btn')).toBeVisible();

        await page.locator('.session-item', { hasText: 'Target Session' }).click();
        await expect(page.locator('.ai-message').last()).toContainText('Target loaded');

        continueGenerate();
        await expect.poll(() => savedAbortedMessage).toContain('[Stopped]');
    });

    test('Chat UI aborts active generation when starting a new session', async ({ page }) => {
        const handleSessionsList = route => { 
            route.fulfill({ 
                status: 200, 
                json: [
                    { id: 'session-new-chat-active', title: 'Active Session', updated_at: 1678886400, email: 'mock@example.com' }
                ] 
            }); 
        };
        await page.route('**/api/chat/sessions', handleSessionsList);
        await page.route('**/api/chat/sessions?*', handleSessionsList);

        let continueGenerate;
        const generatePromise = new Promise(res => continueGenerate = res);
        
        await page.route('**/api/generate', async route => {
            await generatePromise;
            route.fulfill({ status: 200, body: '{"token":"Late"}\n', contentType: 'text/plain' }).catch(() => {});
        });

        let savedAbortedMessage = null;
        await page.route('**/api/chat/sessions/session-new-chat-active/messages', async route => {
            if (route.request().method() === 'POST') {
                const body = JSON.parse(route.request().postData());
                if (body.role === 'assistant') { savedAbortedMessage = body.content; }
                await route.fulfill({ status: 200 });
            } else { route.fallback(); }
        });

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Active Session' }).click();

        await page.locator('#prompt-input').fill('Tell me a long story');
        await page.locator('#send-btn').click();
        
        // Wait for generation to start
        await expect(page.locator('#stop-btn')).toBeVisible();

        // Click new chat mid-generation
        await page.locator('#new-chat-btn').click();
        await expect(page.locator('.ai-message').first()).toContainText('System: New chat session started.');

        continueGenerate();
        await expect.poll(() => savedAbortedMessage).toContain('[Stopped]');
    });

    test('Chat UI aborts active generation and safely clears state when clearChat is clicked', async ({ page }) => {
        const handleSessionsList = route => { 
            route.fulfill({ status: 200, json: [{ id: 'session-clear-abort', title: 'Clear Abort Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); 
        };
        await page.route('**/api/chat/sessions', handleSessionsList);
        await page.route('**/api/chat/sessions?*', handleSessionsList);

        let continueGenerate;
        const generatePromise = new Promise(res => continueGenerate = res);
        await page.route('**/api/generate', async route => {
            await generatePromise;
            route.fulfill({ status: 200, body: '{"token":"Late"}\n', contentType: 'text/plain' }).catch(() => {});
        });

        let deleteCalled = false;
        await page.route('**/api/chat/sessions/session-clear-abort', async route => {
            if (route.request().method() === 'DELETE') {
                deleteCalled = true;
                await route.fulfill({ status: 200 });
            } else { route.fallback(); }
        });

        page.on('dialog', dialog => dialog.accept());

        await page.goto('/');
        await page.locator('.session-item', { hasText: 'Clear Abort Chat' }).click();

        await page.locator('#prompt-input').fill('Tell me a story');
        await page.locator('#send-btn').click();
        await expect(page.locator('#stop-btn')).toBeVisible();

        // Click Clear Chat mid-generation
        await page.locator('#clear-chat-btn').click();
        await page.locator('#delete-confirm-btn').click();
        await expect(page.locator('.ai-message').first()).toContainText('System: New chat session started.');
        
        continueGenerate(); // Unblock the route to allow abort microtasks to settle
        await expect.poll(() => deleteCalled).toBe(true);
        await expect(page.locator('#stop-btn')).not.toBeVisible();
    });

    test('Chat UI aborts active generation and safely clears state when inline delete button is clicked', async ({ page }) => {
        const handleSessionsList = route => { 
            route.fulfill({ status: 200, json: [{ id: 'session-inline-abort', title: 'Inline Abort Chat', updated_at: 1678886400, email: 'mock@example.com' }] }); 
        };
        await page.route('**/api/chat/sessions', handleSessionsList);
        await page.route('**/api/chat/sessions?*', handleSessionsList);

        let continueGenerate;
        const generatePromise = new Promise(res => continueGenerate = res);
        await page.route('**/api/generate', async route => {
            await generatePromise;
            route.fulfill({ status: 200, body: '{"token":"Late"}\n', contentType: 'text/plain' }).catch(() => {});
        });

        let deleteCalled = false;
        await page.route('**/api/chat/sessions/session-inline-abort', async route => {
            if (route.request().method() === 'DELETE') {
                deleteCalled = true;
                await route.fulfill({ status: 200 });
            } else { route.fallback(); }
        });

        page.on('dialog', dialog => dialog.accept());

        await page.goto('/');
        const sessionItem = page.locator('.session-item', { hasText: 'Inline Abort Chat' });
        await sessionItem.click();

        await page.locator('#prompt-input').fill('Tell me a story');
        await page.locator('#send-btn').click();
        await expect(page.locator('#stop-btn')).toBeVisible();

        // Hover and click inline delete mid-generation
        await sessionItem.hover();
        await sessionItem.locator('button[title="Delete Chat"]').click();
        await page.locator('#delete-confirm-btn').click();
        
        await expect(page.locator('.ai-message').first()).toContainText('System: New chat session started.');
        
        continueGenerate(); // Unblock the route to allow abort microtasks to settle
        await expect.poll(() => deleteCalled).toBe(true);
        await expect(page.locator('#stop-btn')).not.toBeVisible();
    });

    test('Chat UI alerts user and prevents message send if session creation fails', async ({ page }) => {
        const handleSessionCreateFailRoute = async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({ status: 200, json: [] });
            } else if (route.request().method() === 'POST') {
                await route.fulfill({ status: 500, body: 'Internal Server Error' });
            } else { await route.fallback(); }
        };
        await page.route('**/api/chat/sessions', handleSessionCreateFailRoute);
        await page.route('**/api/chat/sessions?*', handleSessionCreateFailRoute);

        let alertText = null;
        page.on('dialog', dialog => { if (dialog.type() === 'alert') { alertText = dialog.message(); dialog.accept(); } });

        await page.goto('/');
        await expect(page.locator('#chat-model-select option')).not.toHaveCount(0);

        const input = page.locator('#prompt-input');
        await input.fill('Hello');
        await page.locator('#send-btn').click();

        await expect.poll(() => alertText).toContain('Failed to create a new chat session');
        
        await expect(page.locator('#send-btn')).toBeEnabled();
        await expect(page.locator('.user-message')).toHaveCount(0);
    });
});