import { test, expect } from '@playwright/test';
import { AxeBuilder } from '@axe-core/playwright';
import { mockStaticAssets, mockEngineApis, setupPageErrorHandlers } from './utils.js';

test.describe('Mini Inference Engine - Queue UI', () => {
    test.beforeEach(async ({ page }, testInfo) => {
        setupPageErrorHandlers(page, testInfo);
        await mockStaticAssets(page);
        await mockEngineApis(page);
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

        page.on('dialog', dialog => dialog.accept());

        await page.goto('/queue');

        const tbody = page.locator('#queue-tbody');
        await expect(tbody).toContainText('mock-model-1');
        await expect(tbody).toContainText('mock-model-2');

        await page.locator('tr', { hasText: 'mock-model-2' }).locator('button', { hasText: 'Cancel' }).click();
        expect(deleteCalled).toBe(true);

        await page.locator('tr', { hasText: 'mock-model-1' }).locator('button', { hasText: 'Pause' }).click();
        expect(pauseCalled).toBe(true);

        await page.locator('#clear-all-btn').click();
        expect(clearAllCalled).toBe(true);
    });

    test('Queue UI should not have any automatically detectable accessibility issues', async ({ page }) => {
        await page.goto('/queue');
        await expect(page.locator('#queue-tbody')).toContainText('No active or queued downloads.');
        const accessibilityScanResults = await new AxeBuilder({ page }).analyze();
        expect(accessibilityScanResults.violations).toEqual([]);
    });

    test('Queue UI sanitizes malicious XSS payloads in model IDs and status', async ({ page }) => {
        await page.route('**/api/downloads', async route => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    json: {
                        '<img src="x" onerror="alert(\'XSS-ID\')">malicious-model': { bytes_transferred: 50, total_bytes: 100, current_speed_bps: 10, start_time: 0, state: '<script>alert("XSS-STATUS")</script>Downloading...' }
                    }
                });
            } else { route.fallback(); }
        });

        let alertFired = false;
        page.on('dialog', dialog => { alertFired = true; dialog.dismiss(); });
        await page.goto('/queue');
        const tbody = page.locator('#queue-tbody');
        await expect(tbody).toContainText('malicious-model');
        await expect(tbody).toContainText('Downloading...');
        const rowHtml = await tbody.innerHTML();
        expect(rowHtml).not.toContain('<img');
        expect(rowHtml).not.toContain('<script');
        expect(alertFired).toBe(false);
    });
});