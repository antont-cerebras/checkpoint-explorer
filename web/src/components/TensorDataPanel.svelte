<script lang="ts">
  import Heatmap from './Heatmap.svelte';
  import HistogramView from './HistogramView.svelte';
  import TensorStats from './TensorStats.svelte';
  import ValueGrid from './ValueGrid.svelte';

  export let name: string;
  export let dtype: string;

  type Tab = 'heatmap' | 'histogram' | 'stats' | 'grid';
  let tab: Tab | null = null; // null = nothing loaded yet (byte scans are opt-in)

  const tabs: { id: Tab; label: string }[] = [
    { id: 'heatmap', label: 'Heatmap' },
    { id: 'grid', label: 'Values' },
    { id: 'histogram', label: 'Histogram' },
    { id: 'stats', label: 'Statistics' },
  ];

  // Reset when the selected tensor changes so we don't scan on every selection.
  $: if (name) tab = null;
</script>

<div class="panel">
  <div class="tabbar">
    {#each tabs as t}
      <button class:active={tab === t.id} on:click={() => (tab = t.id)}>{t.label}</button>
    {/each}
    <span class="dim hint">reads tensor data on demand</span>
  </div>

  <div class="body">
    {#if tab === 'heatmap'}
      <Heatmap {name} />
    {:else if tab === 'grid'}
      <ValueGrid {name} />
    {:else if tab === 'histogram'}
      <HistogramView {name} {dtype} />
    {:else if tab === 'stats'}
      <TensorStats {name} />
    {:else}
      <p class="dim">Pick a view above to read this tensor's values.</p>
    {/if}
  </div>
</div>

<style>
  .panel {
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-panel);
  }
  .tabbar {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px;
    border-bottom: 1px solid var(--border);
  }
  .hint {
    margin-left: auto;
    font-size: 11px;
  }
  .body {
    padding: 12px;
    min-height: 120px;
  }
</style>
