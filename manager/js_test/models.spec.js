import { test, expect } from '@playwright/test';
import { AxeBuilder } from '@axe-core/playwright';
import { mockStaticAssets, mockEngineApis, setupPageErrorHandlers } from './utils.js';

test.describe('Mini Inference Engine - Models Directory UI', () => {
    test.beforeEach(async ({ page }, testInfo) => {
        setupPageErrorHandlers(page, testInfo);
        await mockStaticAssets(page);
        await mockEngineApis(page);
    });

    test('Models Directory renders the model configuration cards', async ({ page }) => {
        await page.goto('/models');
        const modelCards = page.locator('.model-dir-card');
        await expect(modelCards).toHaveCount(2);
        await expect(modelCards.first()).toContainText('Mock Chat Model');
    });

    test('Models Directory should not have any automatically detectable accessibility issues', async ({ page }) => {
        await page.goto('/models');
        await expect(page.locator('.model-dir-card')).toHaveCount(2);
        const accessibilityScanResults = await new AxeBuilder({ page }).analyze();
        expect(accessibilityScanResults.violations).toEqual([]);
    });

    test('Models Directory renders even if status API fails or is slow', async ({ page }) => {
        await page.route('**/api/status', () => { });
        await page.goto('/models');
        const modelCards = page.locator('.model-dir-card');
        await expect(modelCards).toHaveCount(2);
        await expect(modelCards.first()).toContainText('Mock Chat Model');
    });

    test('Models Directory handles streaming download progress', async ({ page }) => {
        const downloadState = {};
        const modelState = [
            { id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: true },
            { id: 'mock-comp-1', name: 'Mock Compressor', roles: ['ContextCompressor'], supported_backends: ['Candle'], arch: 'XLMRoberta', parameters_billions: 0.5, size_on_disk_gb: 1.0, max_context_len: 1024, provenance: {}, is_downloaded: false }
        ];

        await page.route('**/api/models', async route => {
            if (route.request().method() === 'GET') { await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(modelState) }); } 
            else { route.fallback(); }
        });
        
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
                await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(downloadState) });
            } else if (route.request().method() === 'POST') {
                const postData = JSON.parse(route.request().postData());
                if (postData.model_id === 'mock-comp-1') {
                    downloadState['mock-comp-1'] = { bytes_transferred: 52428800, total_bytes: 104857600, current_speed_bps: 10485760, start_time: Math.floor(Date.now() / 1000) - 5, state: 'Downloading...' };
                }
                await route.fulfill({ status: 202, body: '' });
            } else { route.fallback(); }
        });

        await page.goto('/models');
        const compCard = page.locator('#model-card-mock-comp-1');
        await expect(compCard).toHaveClass(/model-undownloaded/);
        
        await compCard.locator('.btn-download').click();
        const progressContainer = compCard.locator('.download-progress-container');
        await expect(progressContainer).toBeVisible();
        await expect(progressContainer.locator('.download-stats')).toContainText('50.0%');
        await expect(progressContainer.locator('.download-stats')).toContainText('10.0 MB/s');

        delete downloadState['mock-comp-1'];
        modelState[1].is_downloaded = true;
        await expect(async () => { await expect(compCard.locator('.badge-json', { hasText: 'Ready' })).toBeVisible(); }).toPass();
        await expect(compCard).not.toHaveClass(/model-undownloaded/);
    });

    test('Models Directory handles download cancellation', async ({ page }) => {
        const downloadState = {};
        const modelState = [{ id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: false }];
        await page.route('**/api/models', async route => { if (route.request().method() === 'GET') { await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(modelState) }); } else { route.fallback(); } });
        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') { await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(downloadState) }); }
            else if (route.request().method() === 'POST') {
                downloadState['mock-model-1'] = { bytes_transferred: 50, total_bytes: 100, current_speed_bps: 10, start_time: Math.floor(Date.now() / 1000), state: 'Downloading...' };
                await route.fulfill({ status: 202, body: '' });
            } else { route.fallback(); }
        });

        await page.goto('/models');
        const card = page.locator('#model-card-mock-model-1');
        await card.locator('.btn-download').click();

        const progressContainer = card.locator('.download-progress-container');
        await expect(progressContainer).toBeVisible();
        await progressContainer.locator('.dl-cancel-btn').click();
        await expect(progressContainer.locator('.download-stats')).toContainText('Download Canceled.');
    });

    test('Models Directory recovers from network interruptions during download', async ({ page }) => {
        const downloadState = {};
        let isDownloaded = false;
        
        await page.route('**/api/models', async route => {
            route.fulfill({ status: 200, json: [{ id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: isDownloaded }] });
        });
        await page.route('**/api/models/*', async route => {
            if (route.request().method() === 'GET') {
                const id = route.request().url().split('/').pop();
                if (id === 'mock-model-1') await route.fulfill({ status: 200, json: { id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: isDownloaded } });
                else await route.fulfill({ status: 404 });
            } else { route.fallback(); }
        });

        let simulateDrop = false;
        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') {
                if (simulateDrop) await route.fulfill({ status: 502, body: 'Bad Gateway' });
                else await route.fulfill({ status: 200, json: downloadState });
            } else if (route.request().method() === 'POST') {
                downloadState['mock-model-1'] = { bytes_transferred: 50, total_bytes: 100, current_speed_bps: 10, start_time: 0, state: "Downloading..." };
                await route.fulfill({ status: 202, body: '' });
            } else { route.fallback(); }
        });

        await page.goto('/models');
        const card = page.locator('#model-card-mock-model-1');
        await card.locator('.btn-download').click();

        const stats = card.locator('.download-stats');
        await expect(stats).toContainText('50.0%');
        simulateDrop = true;
        await expect(stats).toContainText('Retrying in 5s...');
        simulateDrop = false;
        downloadState['mock-model-1'].bytes_transferred = 100;
        await expect(stats).toContainText('100.0%', { timeout: 15000 });
        delete downloadState['mock-model-1'];
        isDownloaded = true;
        await expect(card.locator('.badge-json', { hasText: 'Ready' })).toBeVisible();
    });

    test('Models Directory handles download dropped by server (ServerDropped)', async ({ page }) => {
        await page.route('**/api/models', async route => { if (route.request().method() === 'GET') { await route.fulfill({ status: 200, json: [{ id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: false }] }); } else { route.fallback(); } });
        await page.route('**/api/models/mock-model-1', async route => { if (route.request().method() === 'GET') { await route.fulfill({ status: 200, json: { is_downloaded: false } }); } else { route.fallback(); } });

        let postCount = 0;
        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') {
                if (postCount === 1) { await route.fulfill({ status: 200, json: { 'mock-model-1': { bytes_transferred: 10, total_bytes: 100, current_speed_bps: 10, start_time: 0, state: 'Downloading...' } } }); postCount++; } 
                else { await route.fulfill({ status: 200, json: {} }); }
            } else if (route.request().method() === 'POST') { postCount++; await route.fulfill({ status: 202, body: '' }); } 
            else { route.fallback(); }
        });

        await page.goto('/models');
        const card = page.locator('#model-card-mock-model-1');
        await card.locator('.btn-download').click();
        await expect(card.locator('.download-stats')).toContainText('Download Stopped.', { timeout: 10000 });
        expect(postCount).toBe(2);
    });

    test('Models Directory retries on network errors or 429 Too Many Requests', async ({ page }) => {
        await page.route('**/api/models', async route => { if (route.request().method() === 'GET') { await route.fulfill({ status: 200, json: [{ id: 'mock-model-1', name: 'Mock Chat Model', roles: ['GeneralChat'], supported_backends: ['Candle'], arch: 'Llama', parameters_billions: 8.0, size_on_disk_gb: 4.0, max_context_len: 8192, provenance: {}, is_downloaded: false }] }); } else { route.fallback(); } });
        let postCalled = false;
        let errorServed = false;
        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'POST') { postCalled = true; await route.fulfill({ status: 202, body: '' }); }
            else if (route.request().method() === 'GET') {
                if (postCalled && !errorServed) { errorServed = true; await route.fulfill({ status: 502, body: 'Bad Gateway' }); }
                else if (postCalled && errorServed) { await route.fulfill({ status: 200, json: { 'mock-model-1': { bytes_transferred: 50, total_bytes: 100, current_speed_bps: 1000, start_time: 0, state: 'Downloading...' } } }); }
                else { await route.fulfill({ status: 200, json: {} }); }
            } else { route.fallback(); }
        });

        await page.goto('/models');
        const card = page.locator('#model-card-mock-model-1');
        await card.locator('.btn-download').click();
        await expect(card.locator('.download-stats')).toContainText('Retrying', { timeout: 6000 });
        await expect(card.locator('.download-stats')).toContainText('50.0%', { timeout: 10000 });
    });
});