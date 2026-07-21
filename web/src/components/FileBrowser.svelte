<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import type { FileNode } from '../lib/types';
  import { humanSize } from '../lib/format';

  let root: FileNode | null = null;
  let err = '';
  let expanded = new Set<string>();

  onMount(async () => {
    try {
      root = await api.files();
      if (root) expanded.add(root.path);
      expanded = expanded;
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    }
  });

  interface Row {
    node: FileNode;
    depth: number;
  }

  function flatten(node: FileNode, depth: number, out: Row[]) {
    out.push({ node, depth });
    if (node.kind === 'dir' && expanded.has(node.path)) {
      for (const c of node.children) flatten(c, depth + 1, out);
    }
  }

  $: rows = (() => {
    if (!root) return [] as Row[];
    const out: Row[] = [];
    flatten(root, 0, out);
    return out;
  })();

  function toggle(node: FileNode) {
    if (node.kind !== 'dir') return;
    if (expanded.has(node.path)) expanded.delete(node.path);
    else expanded.add(node.path);
    expanded = expanded;
  }
</script>

<div class="files">
  {#if err}
    <p class="err">{err}</p>
  {:else if !root}
    <p class="dim">loading…</p>
  {:else}
    {#each rows as { node, depth } (node.path + node.name)}
      <div
        class="row {node.kind}"
        style="padding-left:{8 + depth * 16}px"
        role="button"
        tabindex="-1"
        on:click={() => toggle(node)}
        on:keydown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggle(node);
          }
        }}
      >
        <span class="caret">{node.kind === 'dir' ? (expanded.has(node.path) ? '▾' : '▸') : ''}</span>
        <span class="icon">{node.kind === 'dir' ? '📁' : fileIcon(node.file_kind)}</span>
        <span class="name">{node.name}</span>
        <span class="meta dim">
          {#if node.kind === 'dir'}{node.files} files · {humanSize(node.size)}
          {:else}{node.file_kind} · {humanSize(node.size)}{/if}
        </span>
      </div>
    {/each}
  {/if}
</div>

<script context="module" lang="ts">
  function fileIcon(kind: string): string {
    switch (kind) {
      case 'Checkpoint':
        return '🧊';
      case 'Json':
        return '📋';
      case 'Text':
        return '📄';
      default:
        return '·';
    }
  }
</script>

<style>
  .files {
    height: 100%;
    overflow: auto;
    padding: 6px 0;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 6px;
    height: 24px;
    line-height: 24px;
    padding-right: 12px;
    white-space: nowrap;
    cursor: pointer;
  }
  .row:hover {
    background: var(--bg-hover);
  }
  .caret {
    flex: 0 0 12px;
    color: var(--fg-dim);
    text-align: center;
  }
  .icon {
    flex: 0 0 auto;
  }
  .name {
    flex: 0 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .row.dir .name {
    color: var(--group);
  }
  .meta {
    flex: 1 1 auto;
    text-align: right;
    font-size: 12px;
  }
  .err {
    color: var(--danger);
  }
</style>
