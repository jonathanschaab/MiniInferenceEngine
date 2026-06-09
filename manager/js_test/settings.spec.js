import { test, expect } from '@playwright/test';
import { AxeBuilder } from '@axe-core/playwright';
import { mockStaticAssets, mockEngineApis, setupPageErrorHandlers } from './utils.js';

test.describe('Mini Inference Engine - Settings UI', () => {
    test.beforeEach(async ({ page }, testInfo) => {
        setupPageErrorHandlers(page, testInfo);
        await mockStaticAssets(page);
        await mockEngineApis(page);
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

    test('Settings UI should not have any automatically detectable accessibility issues', async ({ page }) => {
        await page.goto('/settings');
        await expect(page.locator('#keys-tbody')).toContainText('Test Key');
        const accessibilityScanResults = await new AxeBuilder({ page }).analyze();
        expect(accessibilityScanResults.violations).toEqual([]);
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
        page.on('dialog', dialog => { alertFired = true; dialog.dismiss(); });

        await page.goto('/settings');

        const tbody = page.locator('#keys-tbody');
        await expect(tbody).toBeVisible();
        await expect(tbody).toContainText('<script>');
        await expect(tbody).toContainText('alert("XSS-Name")');
        await expect(tbody).toContainText('<img');
        expect(alertFired).toBe(false);
    });
});