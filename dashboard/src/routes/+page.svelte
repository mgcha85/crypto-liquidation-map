<script lang="ts">
  import { tradingState, metrics } from '$lib/stores';
  
  $: signalText = $tradingState.lastSignal === 1 ? 'BUY' : 
                  $tradingState.lastSignal === -1 ? 'SELL' : 'HOLD';
  $: signalClass = $tradingState.lastSignal === 1 ? 'signal-buy' : 
                   $tradingState.lastSignal === -1 ? 'signal-sell' : 'signal-hold';
</script>

<div class="dashboard">
  <h1>Trading Dashboard</h1>
  
  <div class="status-bar">
    <span class="status" class:running={$tradingState.isRunning}>
      {$tradingState.isRunning ? 'RUNNING' : 'STOPPED'}
    </span>
    <span class="mode">{$tradingState.mode.toUpperCase()}</span>
    {#if $tradingState.isHalted}
      <span class="halted">HALTED: {$tradingState.haltReason}</span>
    {/if}
  </div>

  <div class="grid">
    <div class="card">
      <h3>Current Signal</h3>
      <div class="signal {signalClass}">{signalText}</div>
    </div>

    <div class="card">
      <h3>Position</h3>
      <div class="position">
        <p>Side: <strong>{$tradingState.position.side}</strong></p>
        {#if $tradingState.position.side !== 'NONE'}
          <p>Entry: ${$tradingState.position.entryPrice.toFixed(2)}</p>
          <p>Size: {$tradingState.position.size.toFixed(6)}</p>
          <p class:profit={$tradingState.position.unrealizedPnl > 0} 
             class:loss={$tradingState.position.unrealizedPnl < 0}>
            P&L: ${$tradingState.position.unrealizedPnl.toFixed(2)}
          </p>
        {/if}
      </div>
    </div>

    <div class="card">
      <h3>Equity</h3>
      <div class="equity">${$tradingState.equity.toFixed(2)}</div>
      <p class:profit={$tradingState.dailyPnl > 0} 
         class:loss={$tradingState.dailyPnl < 0}>
        Daily: ${$tradingState.dailyPnl.toFixed(2)}
      </p>
    </div>

    <div class="card">
      <h3>Metrics</h3>
      <table>
        <tr><td>Return</td><td>{($metrics.totalReturn * 100).toFixed(2)}%</td></tr>
        <tr><td>Sharpe</td><td>{$metrics.sharpeRatio.toFixed(2)}</td></tr>
        <tr><td>Max DD</td><td>{($metrics.maxDrawdown * 100).toFixed(2)}%</td></tr>
        <tr><td>Win Rate</td><td>{($metrics.winRate * 100).toFixed(1)}%</td></tr>
        <tr><td>Trades</td><td>{$metrics.totalTrades}</td></tr>
      </table>
    </div>
  </div>
</div>

<style>
  .dashboard { padding: 2rem; }
  .status-bar { display: flex; gap: 1rem; margin-bottom: 2rem; }
  .status { padding: 0.5rem 1rem; border-radius: 4px; background: #e74c3c; color: white; }
  .status.running { background: #27ae60; }
  .mode { padding: 0.5rem 1rem; background: #3498db; color: white; border-radius: 4px; }
  .halted { padding: 0.5rem 1rem; background: #f39c12; color: white; border-radius: 4px; }
  
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }
  .card { background: #1e1e1e; padding: 1.5rem; border-radius: 8px; }
  .card h3 { margin: 0 0 1rem 0; color: #888; font-size: 0.9rem; text-transform: uppercase; }
  
  .signal { font-size: 2rem; font-weight: bold; }
  .signal-buy { color: #27ae60; }
  .signal-sell { color: #e74c3c; }
  .signal-hold { color: #f39c12; }
  
  .equity { font-size: 2rem; font-weight: bold; color: #3498db; }
  .profit { color: #27ae60; }
  .loss { color: #e74c3c; }
  
  table { width: 100%; }
  td { padding: 0.25rem 0; }
  td:last-child { text-align: right; font-weight: bold; }
</style>
