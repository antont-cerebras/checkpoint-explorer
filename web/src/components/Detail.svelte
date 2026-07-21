<script lang="ts">
  import { tree } from '../stores/server';
  import { selectedId } from '../stores/view';
  import { nodeId } from '../lib/flatten';
  import type { TreeNode } from '../lib/types';
  import { humanCount, humanSize, shape } from '../lib/format';
  import TensorDataPanel from './TensorDataPanel.svelte';

  $: node = find($tree?.tree ?? [], $selectedId);

  function find(nodes: TreeNode[], id: string | null): TreeNode | null {
    if (!id) return null;
    let out: TreeNode | null = null;
    const walk = (ns: TreeNode[], parentId: string) => {
      for (const n of ns) {
        if (out) return;
        const nid = nodeId(n, parentId);
        if (nid === id) {
          out = n;
          return;
        }
        if (n.kind === 'group') walk(n.children, nid);
      }
    };
    walk(nodes, '');
    return out;
  }
</script>

<div class="detail">
  {#if !node}
    <p class="empty dim">Select a tensor or group in the tree.</p>
  {:else if node.kind === 'tensor'}
    <h2>{node.info.name}</h2>
    <table>
      <tbody>
        <tr><th>dtype</th><td class="mono">{node.info.dtype}</td></tr>
        <tr><th>shape</th><td class="mono">{shape(node.info.shape)}</td></tr>
        <tr><th>params</th><td class="mono">{humanCount(node.info.num_elements)} ({node.info.num_elements.toLocaleString()})</td></tr>
        <tr><th>size</th><td class="mono">{humanSize(node.info.size_bytes)}</td></tr>
        <tr><th>source</th><td class="dim src">{node.info.source_path}</td></tr>
      </tbody>
    </table>
    <TensorDataPanel name={node.info.name} dtype={node.info.dtype} />
  {:else if node.kind === 'metadata'}
    <h2>{node.info.name}</h2>
    <table>
      <tbody>
        <tr><th>type</th><td class="mono">{node.info.value_type}</td></tr>
        <tr><th>value</th><td class="mono val">{node.info.value}</td></tr>
      </tbody>
    </table>
  {:else}
    <h2>{node.name}</h2>
    <table>
      <tbody>
        <tr><th>tensors</th><td class="mono">{humanCount(node.tensor_count)}</td></tr>
        <tr><th>params</th><td class="mono">{humanCount(node.params)}</td></tr>
        <tr><th>size</th><td class="mono">{humanSize(node.total_size)}</td></tr>
        {#if node.stored_size !== node.total_size}
          <tr><th>on disk</th><td class="mono">{humanSize(node.stored_size)}</td></tr>
        {/if}
      </tbody>
    </table>
  {/if}
</div>

<style>
  .detail {
    padding: 14px 18px;
  }
  .empty {
    margin-top: 40px;
    text-align: center;
  }
  h2 {
    margin: 0 0 12px;
    font-size: 15px;
    color: var(--accent);
    word-break: break-all;
  }
  table {
    border-collapse: collapse;
    margin-bottom: 16px;
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
  .src,
  .val {
    word-break: break-all;
  }
</style>
