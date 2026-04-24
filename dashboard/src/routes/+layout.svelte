<script lang="ts">
  import { page } from '$app/stores';
</script>

<div class="layout">
  <nav>
    <a href="/" class:active={$page.url.pathname === '/'}>Dashboard</a>
    <a href="/history" class:active={$page.url.pathname === '/history'}>History</a>
    <a href="/settings" class:active={$page.url.pathname === '/settings'}>Settings</a>
  </nav>
  
  <main>
    <slot />
  </main>
</div>

<style>
  :global(body) {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #121212;
    color: #fff;
  }
  
  .layout {
    display: flex;
    min-height: 100vh;
  }
  
  nav {
    width: 200px;
    background: #1a1a1a;
    padding: 2rem 0;
    border-right: 1px solid #333;
  }
  
  nav a {
    display: block;
    padding: 0.75rem 1.5rem;
    color: #888;
    text-decoration: none;
    border-left: 3px solid transparent;
  }
  
  nav a:hover {
    color: #fff;
    background: #222;
  }
  
  nav a.active {
    color: #3498db;
    border-left-color: #3498db;
    background: #222;
  }
  
  main {
    flex: 1;
    overflow-y: auto;
  }
</style>
