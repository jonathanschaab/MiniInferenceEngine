import { test, expect } from '@playwright/test';
import { mockStaticAssets, mockEngineApis, setupPageErrorHandlers } from './utils.js';

test.describe('Mini Inference Engine - Common Globals', () => {
    test.beforeEach(async ({ page }, testInfo) => {
        setupPageErrorHandlers(page, testInfo);
        await mockStaticAssets(page);
        await mockEngineApis(page);
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

        await page.goto('/');
        apiCallCount = 0; 

        await page.evaluate(async () => {
            const promises = Array.from({ length: 5 }).map(() => SharedDownloadProgress.get());
            await Promise.all(promises);
        });

        expect(apiCallCount).toBe(1);
        await page.waitForTimeout(600);
        await page.evaluate(async () => await SharedDownloadProgress.get());
        expect(apiCallCount).toBe(2);
    });

    test('UI redirects to login on 401 Unauthorized API response', async ({ page }) => {
        await page.route('**/api/status', async route => { await route.fulfill({ status: 401, body: 'Unauthorized' }); });
        await page.route('**/auth/login', async route => {
            await route.fulfill({ status: 200, contentType: 'text/html', body: '<html><body>Mock Login Page</body></html>' });
        });
        await page.goto('/');
        await expect(page).toHaveURL(/.*\/auth\/login/);
    });
});