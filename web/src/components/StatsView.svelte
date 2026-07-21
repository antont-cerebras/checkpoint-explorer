<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import { humanCount, humanSize } from '../lib/format';
  import JsonView from './JsonView.svelte';

  let stats: Record<string, unknown> | null = null;
  let err = '';

  onMount(async () => {
    try {
      stats = await api.stats();
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    }
  });

  const asNum = (v: unknown) => (typeof v === 'number' ? v : 0);
</script>

<div class="stats">
  {#if err}
    <p class="err">{err}</p>
  {:else if !stats}
    <p class="dim">loading…</p>
  {:else}
    <div class="cards">
      <div class="card"><span class="k">tensors</span><span class="v">{humanCount(asNum(stats.n_tensors))}</span></div>
      <div class="card"><span class="k">parameters</span><span class="v">{humanCount(asNum(stats.params))}</span></div>
      <div class="card"><span class="k">logical size</span><span class="v">{humanSize(asNum(stats.logical_bytes))}</span></div>
      <div class="card"><span class="k">on disk</span><span class="v">{humanSize(asNum(stats.disk_bytes))}</span></div>
      {#if stats.model_type}
        <div class="card"><span class="k">model</span><span class="v">{stats.model_type}</span></div>
      {/if}
    </div>
    <div class="full">
      <JsonView value={stats} />
    </div>
  {/if}
</div>

<style>
  .stats {
    padding: 16px;
    height: 100%;
    overflow: auto;
  }
  .cards {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 18px;
  }
  .card {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 10px 14px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    min-width: 120px;
  }
  .k {
    color: var(--fg-dim);
    font-size: 11px;
  }
  .v {
    font-size: 17px;
    color: var(--accent);
  }
  .err {
    color: var(--danger);
  }
</style>
