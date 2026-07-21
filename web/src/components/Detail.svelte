<script lang="ts">
  import { tree } from '../stores/server';
  import { setTab, navigate, type DataTab } from '../stores/view';
  import type { TensorInfo, TreeNode } from '../lib/types';
  import { humanCount, humanSize, shape } from '../lib/format';
  import Heatmap from './Heatmap.svelte';
  import HistogramView from './HistogramView.svelte';
  import ValueGrid from './ValueGrid.svelte';
  import TensorStats from './TensorStats.svelte';

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

  const tabs: { id: DataTab; label: string }[] = [
    { id: 'info', label: 'Info' },
    { id: 'heatmap', label: 'Heatmap' },
    { id: 'values', label: 'Values' },
    { id: 'histogram', label: 'Histogram' },
    { id: 'stats', label: 'Statistics' },
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
          {#if offsets(info.layout)}
            <tr><th>Data offsets</th><td class="mono">{offsets(info.layout)}</td></tr>
          {/if}
          <tr>
            <th>File</th>
            <td>
              <button
                class="link src"
                title="Show this shard's byte-layout map"
                on:click={() => navigate({ kind: 'layout', file: baseName(info.source_path) })}
              >{info.source_path}</button>
            </td>
          </tr>
        </tbody>
      </table>
    {:else if tab === 'heatmap'}
      <Heatmap name={tensor} />
    {:else if tab === 'values'}
      <ValueGrid name={tensor} />
    {:else if tab === 'histogram'}
      <HistogramView name={tensor} dtype={info.dtype} />
    {:else if tab === 'stats'}
      <TensorStats name={tensor} />
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
    padding: 2px 0;
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
</style>
