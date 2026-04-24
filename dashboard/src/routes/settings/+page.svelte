<script lang="ts">
  import { settings } from '$lib/stores';
  import { updateSettings } from '$lib/api';
  
  let apiKey = $settings.apiKey;
  let apiSecret = $settings.apiSecret;
  let testnet = $settings.testnet;
  let tradingEnabled = $settings.tradingEnabled;
  let saving = false;
  let message = '';
  
  async function save() {
    saving = true;
    message = '';
    try {
      await updateSettings({ apiKey, apiSecret, testnet });
      settings.update(s => ({ ...s, apiKey, apiSecret, testnet, tradingEnabled }));
      message = 'Settings saved successfully';
    } catch (e) {
      message = 'Failed to save settings';
    }
    saving = false;
  }
</script>

<div class="settings-page">
  <h1>Settings</h1>
  
  <form on:submit|preventDefault={save}>
    <div class="section">
      <h2>API Credentials</h2>
      
      <div class="field">
        <label for="apiKey">API Key</label>
        <input 
          id="apiKey"
          type="password" 
          bind:value={apiKey} 
          placeholder="Enter your Binance API key"
        />
      </div>
      
      <div class="field">
        <label for="apiSecret">API Secret</label>
        <input 
          id="apiSecret"
          type="password" 
          bind:value={apiSecret} 
          placeholder="Enter your Binance API secret"
        />
      </div>
    </div>

    <div class="section">
      <h2>Trading Mode</h2>
      
      <div class="toggle-field">
        <label>
          <input type="checkbox" bind:checked={testnet} />
          <span>Use Testnet</span>
        </label>
        <p class="hint">When enabled, trades will be executed on Binance Futures Testnet</p>
      </div>
      
      <div class="toggle-field">
        <label class:danger={tradingEnabled && !testnet}>
          <input type="checkbox" bind:checked={tradingEnabled} />
          <span>Enable Trading</span>
        </label>
        {#if tradingEnabled && !testnet}
          <p class="warning">WARNING: Live trading with real funds is enabled!</p>
        {/if}
      </div>
    </div>

    <div class="actions">
      <button type="submit" disabled={saving}>
        {saving ? 'Saving...' : 'Save Settings'}
      </button>
    </div>
    
    {#if message}
      <p class="message" class:success={message.includes('success')}>
        {message}
      </p>
    {/if}
  </form>
</div>

<style>
  .settings-page { padding: 2rem; max-width: 600px; }
  
  .section { 
    background: #1e1e1e; 
    padding: 1.5rem; 
    border-radius: 8px; 
    margin-bottom: 1rem;
  }
  .section h2 { 
    margin: 0 0 1rem 0; 
    color: #888; 
    font-size: 0.9rem; 
    text-transform: uppercase; 
  }
  
  .field { margin-bottom: 1rem; }
  .field label { display: block; margin-bottom: 0.5rem; color: #aaa; }
  .field input { 
    width: 100%; 
    padding: 0.75rem; 
    background: #2a2a2a; 
    border: 1px solid #444;
    color: white;
    border-radius: 4px;
  }
  
  .toggle-field { margin-bottom: 1rem; }
  .toggle-field label { 
    display: flex; 
    align-items: center; 
    gap: 0.5rem; 
    cursor: pointer;
  }
  .toggle-field input { width: auto; }
  .hint { color: #666; font-size: 0.85rem; margin: 0.5rem 0 0 1.5rem; }
  .warning { color: #e74c3c; font-size: 0.85rem; margin: 0.5rem 0 0 1.5rem; }
  .danger span { color: #e74c3c; }
  
  .actions { margin-top: 1.5rem; }
  button {
    padding: 0.75rem 2rem;
    background: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  button:hover:not(:disabled) { background: #2980b9; }
  
  .message { 
    margin-top: 1rem; 
    padding: 0.75rem; 
    border-radius: 4px;
    background: #e74c3c;
  }
  .message.success { background: #27ae60; }
</style>
