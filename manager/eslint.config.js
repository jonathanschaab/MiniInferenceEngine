import eslint from '@eslint/js';
import globals from 'globals';

export default [
  eslint.configs.recommended, // Applies the recommended rules
  {
    // Define which files this configuration applies to
    files: ['**/*.js'],
    // Configure language options, equivalent to parserOptions and env
    languageOptions: {
      ecmaVersion: 2021, // Supports ES2021 syntax (equivalent to ecmaVersion: 12)
      sourceType: 'module',
      // Define globals for the browser, Node.js, and Playwright test environments
      globals: {
        ...globals.browser, // Includes 'document', 'window', 'console', 'fetch', 'alert', 'confirm', 'setInterval', 'Option', 'URL', 'AbortController', 'TextDecoder', etc.
        ...globals.node,    // Includes 'process', '__dirname', 'require', 'module', etc.

        // Global variables from external libraries
        DOMPurify: 'readonly', // Used in memory.js, models.js, settings.js, stats.js
        Chart: 'readonly',     // Used in stats.js (Chart.js library)

        // Playwright test environment globals (from ui.spec.js)
        test: 'readonly',
        expect: 'readonly',

        // Custom global functions/variables explicitly exposed or used via HTML onclick
        fetchWithAuth: 'readonly',
        getGenerationParameters: 'readonly',
        injectNavbar: 'readonly',
        
        // Globals from chat.js
        initializeUI: 'readonly',
        updateDropdownCompatibility: 'readonly',
        loadSessions: 'readonly',
        loadSession: 'readonly',
        startNewSession: 'readonly',
        clearChat: 'readonly',
        appendMessage: 'readonly',
        ensureSession: 'readonly',
        appendMessageToDB: 'readonly',
        truncateMessagesInDB: 'readonly',
        sendMessage: 'readonly',
        regenerateLast: 'readonly',
        
        // Globals from stats.js
        openBenchmarkModal: 'readonly',
        updateBenchmarkCompatibility: 'readonly',
        closeModal: 'readonly',
        submitBenchmark: 'readonly',
        loadDashboard: 'readonly',
        renderSpeedChart: 'readonly',
        renderTokenChart: 'readonly',
        renderTokTimeChart: 'readonly',
        checkStatus: 'readonly',
        
        // Globals from memory.js
        formatMB: 'readonly',
        formatGB: 'readonly',
        formatTime: 'readonly',
        switchTab: 'readonly',
        updateDashboard: 'readonly',
        
        // Globals from settings.js and models.js
        openKeyModal: 'readonly',
        closeKeyModal: 'readonly',
        submitNewKey: 'readonly',
        deleteKey: 'readonly',
        loadKeys: 'readonly',
        loadModels: 'readonly'
      }
    },
    // Customize rules as needed.
    rules: {
      'no-unused-vars': 'warn', // Warn about unused variables instead of erroring
      'no-console': 'off'       // Allow console.log for development
    }
  }
];