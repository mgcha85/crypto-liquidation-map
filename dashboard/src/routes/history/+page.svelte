<script lang="ts">
  import { trades } from '$lib/stores';
  
  let dateFilter = '';
  let sideFilter = '';
  
  $: filteredTrades = $trades.filter(t => {
    if (dateFilter && !t.timestamp.startsWith(dateFilter)) return false;
    if (sideFilter && t.side !== sideFilter) return false;
    return true;
  });
</script>

<div class="history">
  <h1>Trade History</h1>
  
  <div class="filters">
    <input type="date" bind:value={dateFilter} placeholder="Filter by date" />
    <select bind:value={sideFilter}>
      <option value="">All Sides</option>
      <option value="LONG">LONG</option>
      <option value="SHORT">SHORT</option>
    </select>
  </div>

  <table>
    <thead>
      <tr>
        <th>Time</th>
        <th>Side</th>
        <th>Entry</th>
        <th>Exit</th>
        <th>P&L</th>
        <th>P&L %</th>
        <th>Reason</th>
      </tr>
    </thead>
    <tbody>
      {#each filteredTrades as trade}
        <tr>
          <td>{new Date(trade.timestamp).toLocaleString()}</td>
          <td class:long={trade.side === 'LONG'} class:short={trade.side === 'SHORT'}>
            {trade.side}
          </td>
          <td>${trade.entryPrice.toFixed(2)}</td>
          <td>${trade.exitPrice.toFixed(2)}</td>
          <td class:profit={trade.pnl > 0} class:loss={trade.pnl < 0}>
            ${trade.pnl.toFixed(2)}
          </td>
          <td class:profit={trade.pnlPct > 0} class:loss={trade.pnlPct < 0}>
            {(trade.pnlPct * 100).toFixed(2)}%
          </td>
          <td>{trade.exitReason}</td>
        </tr>
      {:else}
        <tr>
          <td colspan="7" class="empty">No trades yet</td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .history { padding: 2rem; }
  .filters { display: flex; gap: 1rem; margin-bottom: 1rem; }
  .filters input, .filters select { 
    padding: 0.5rem; 
    background: #2a2a2a; 
    border: 1px solid #444; 
    color: white;
    border-radius: 4px;
  }
  
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #333; }
  th { background: #1e1e1e; color: #888; font-size: 0.85rem; text-transform: uppercase; }
  
  .long { color: #27ae60; }
  .short { color: #e74c3c; }
  .profit { color: #27ae60; }
  .loss { color: #e74c3c; }
  .empty { text-align: center; color: #666; padding: 2rem; }
</style>
