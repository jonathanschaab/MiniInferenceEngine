import { test, expect } from '@playwright/test';
import { AxeBuilder } from '@axe-core/playwright';
import { mockStaticAssets, mockEngineApis, setupPageErrorHandlers } from './utils.js';

test.describe('Mini Inference Engine - Memory UI', () => {
    test.beforeEach(async ({ page }, testInfo) => {
        setupPageErrorHandlers(page, testInfo);
        await mockStaticAssets(page);
        await mockEngineApis(page);
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

    test('Memory UI should not have any automatically detectable accessibility issues', async ({ page }) => {
        await page.goto('/memory');
        await expect(page.locator('#txt-weights')).toContainText('Weights');
        const accessibilityScanResults = await new AxeBuilder({ page }).analyze();
        expect(accessibilityScanResults.violations).toEqual([]);
    });
});