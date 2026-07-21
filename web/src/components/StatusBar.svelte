<script lang="ts">
  import { screen, selectedId, visibleRows } from '../stores/view';
  import { humanCount, humanSize, shape } from '../lib/format';

  $: row = $visibleRows.find((r) => r.id === $selectedId) ?? null;

  function basename(p: string): string {
    return p.split('/').pop() || p;
  }
</script>

<div class="status">
  {#if $screen.kind !== 'tree'}
    <span class="dim">{$screen.kind}{$screen.kind === 'detail' ? ` · ${$screen.tensor}` : ''}</span>
  {:else if row && row.node.kind === 'tensor'}
    <span class="name">{row.node.info.name}</span>
    <span class="dim">·</span>
    <span class="mono">{row.node.info.dtype} [{shape(row.node.info.shape)}]</span>
    <span class="dim">·</span>
    <span class="mono">{humanSize(row.node.info.size_bytes)}</span>
    <span class="dim">·</span>
    <span class="dim">{basename(row.node.info.source_path)}</span>
  {:else if row && row.node.kind === 'group'}
    <span class="name">{row.node.name}</span>
    <span class="dim">·</span>
    <span class="mono">{humanCount(row.node.tensor_count)} tensors</span>
    <span class="dim">·</span>
    <span class="mono">{humanSize(row.node.total_size)}</span>
  {:else if row && row.node.kind === 'metadata'}
    <span class="name">{row.node.info.name}</span>
    <span class="dim">=</span>
    <span class="mono">{row.node.info.value}</span>
  {:else}
    <span class="dim">—</span>
  {/if}
</div>

<style>
  .status {
    flex: 0 0 auto;
    height: 24px;
    line-height: 24px;
    padding: 0 12px;
    border-top: 1px solid var(--border);
    background: var(--bg-panel);
    display: flex;
    gap: 8px;
    align-items: center;
    white-space: nowrap;
    overflow: hidden;
  }
  .name {
    color: var(--fg);
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mono {
    flex: 0 0 auto;
  }
</style>
