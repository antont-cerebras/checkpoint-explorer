<script lang="ts">
  import { tree, cachedStats } from '../stores/server';
  import { setTab, navigate, type DataTab } from '../stores/view';
  import type { StatsDto, TensorInfo, TreeNode } from '../lib/types';
  import { humanCount, humanSize, num, shape } from '../lib/format';
  import DataView from './DataView.svelte';
  import HistogramView from './HistogramView.svelte';
  import Spinner from './Spinner.svelte';

  export let tensor: string;
  export let tab: DataTab;

  $: info = findTensor($tree?.tree ?? [], tensor);

  function findTensor(nodes: TreeNode[], name: string): TensorInfo | null {
    for (const n of nodes) {
      if (n.kind === 'tensor' && n.info.name === name) return n.info;
      if (n.kind === 'group') {
        const found = findTensor(n.children, name);
        if (found) return found;
      }
    }
    return null;
  }

  function baseName(p: string): string {
    return p.split('/').pop() || p;
  }

  function offsets(layout: unknown): string | null {
    const l = layout as Record<string, { start?: number; end?: number }>;
    if (l && l.ByteRange) return `${l.ByteRange.start} – ${l.ByteRange.end} (within file data)`;
    return null;
  }

  // Whole-tensor statistics are shown on the Info tab, scanned on demand.
  let statsPromise: Promise<StatsDto> | null = null;
  $: if (tensor) statsPromise = null; // reset when the selected tensor changes
  function scan() {
    statsPromise = cachedStats(tensor);
  }

  const tabs: { id: DataTab; label: string }[] = [
    { id: 'info', label: 'Info' },
    { id: 'heatmap', label: 'Heatmap' },
    { id: 'values', label: 'Values' },
    { id: 'histogram', label: 'Histogram' },
  ];
</script>

<div class="detail">
  <div class="tabbar">
    {#each tabs as t}
      <button class:active={tab === t.id} on:click={() => setTab(t.id)}>{t.label}</button>
    {/each}
  </div>

  <div class="body">
    {#if !info}
      <p class="dim">Tensor not found: {tensor}</p>
    {:else if tab === 'info'}
      <h2>{info.name}</h2>
      <table>
        <tbody>
          <tr><th>Data Type</th><td class="mono">{info.dtype}</td></tr>
          <tr><th>Shape</th><td class="mono">{shape(info.shape)}</td></tr>
          <tr><th>Parameters</th><td class="mono">{humanCount(info.num_elements)} ({info.num_elements.toLocaleString()})</td></tr>
          <tr><th>Size</th><td class="mono">{humanSize(info.size_bytes)}</td></tr>
          {#if offsets(info.layout)}<tr><th>Data offsets</th><td class="mono">{offsets(info.layout)}</td></tr>{/if}
          <tr>
            <th>File</th>
            <td><button class="link src" title="Show this shard's byte-layout map" on:click={() => navigate({ kind: 'layout', file: baseName(info.source_path) })}>{info.source_path}</button></td>
          </tr>
        </tbody>
      </table>

      <div class="statsblock">
        <h3>Statistics</h3>
        {#if !statsPromise}
          <button on:click={scan}>Scan tensor</button>
          <span class="dim">reads the whole tensor's values</span>
        {:else}
          {#await statsPromise}
            <Spinner label="scanning tensor…" />
          {:then st}
            <table>
              <tbody>
                <tr><th>min</th><td class="mono">{num(st.min)}</td><th>max</th><td class="mono">{num(st.max)}</td></tr>
                <tr><th>mean</th><td class="mono">{num(st.mean)}</td><th>std</th><td class="mono">{num(st.std)}</td></tr>
                <tr><th>zeros</th><td class="mono">{(st.zero_fraction * 100).toFixed(2)}%</td><th>non-finite</th><td class="mono">{st.nonfinite.toLocaleString()}</td></tr>
              </tbody>
            </table>
            <span class="dim">scanned {st.count.toLocaleString()} elements in {st.elapsed_ms.toFixed(0)} ms</span>
          {:catch e}
            <p class="err">{e.message}</p>
          {/await}
        {/if}
      </div>
    {:else if tab === 'heatmap'}
      <DataView {tensor} kind="heatmap" />
    {:else if tab === 'values'}
      <DataView {tensor} kind="values" />
    {:else if tab === 'histogram'}
      <HistogramView name={tensor} dtype={info.dtype} />
    {/if}
  </div>
</div>

<style>
  .detail {
    height: 100%;
    display: flex;
    flex-direction: column;
  }
  .tabbar {
    flex: 0 0 auto;
    display: flex;
    gap: 6px;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
  }
  .body {
    flex: 1 1 auto;
    overflow: auto;
    padding: 14px 18px;
  }
  h2 {
    margin: 0 0 12px;
    font-size: 15px;
    color: var(--accent);
    word-break: break-all;
  }
  h3 {
    margin: 0 0 8px;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--fg-dim);
  }
  table {
    border-collapse: collapse;
    margin-bottom: 14px;
  }
  th {
    text-align: right;
    color: var(--fg-dim);
    font-weight: 400;
    padding: 2px 12px 2px 0;
    vertical-align: top;
    white-space: nowrap;
  }
  td {
    padding: 2px 18px 2px 0;
  }
  .statsblock {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: flex-start;
  }
  .statsblock table {
    margin: 0;
  }
  .src {
    word-break: break-all;
  }
  .link {
    background: none;
    border: none;
    padding: 0;
    text-align: left;
    color: var(--accent);
    text-decoration: underline;
    text-decoration-style: dotted;
    cursor: pointer;
    font: inherit;
  }
  .err {
    color: var(--danger);
  }
</style>
