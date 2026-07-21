<script lang="ts">
  import { tree } from '../stores/server';
  import { expanded, selectedId, search, toggleExpanded } from '../stores/view';
  import { flatten, type Row } from '../lib/flatten';
  import { searchRows } from '../lib/search';
  import { humanCount, humanSize } from '../lib/format';

  const ROW_H = 22;
  let scrollEl: HTMLDivElement;
  let scrollTop = 0;
  let viewportH = 600;

  $: nodes = $tree?.tree ?? [];
  $: query = $search.trim();
  $: rows = query ? searchRows(nodes, query) : flatten(nodes, $expanded);

  // Virtualization: render only the rows overlapping the viewport (31k-safe).
  $: total = rows.length;
  $: first = Math.max(0, Math.floor(scrollTop / ROW_H) - 6);
  $: slice = rows.slice(first, first + Math.ceil(viewportH / ROW_H) + 12);

  function onScroll() {
    scrollTop = scrollEl.scrollTop;
  }

  function activate(row: Row) {
    if (row.hasChildren) toggleExpanded(row.id);
    selectedId.set(row.id);
  }

  function label(row: Row): string {
    const n = row.node;
    if (n.kind === 'group') return n.name;
    if (n.kind === 'tensor') return n.label ?? n.info.name.split('.').pop() ?? n.info.name;
    return n.info.name;
  }

  function rowMeta(row: Row): string {
    const n = row.node;
    if (n.kind === 'group') return `${humanCount(n.tensor_count)} · ${humanSize(n.total_size)}`;
    if (n.kind === 'tensor') return `${n.info.dtype} [${n.info.shape.join(',')}]`;
    return n.info.value;
  }
</script>

<div class="tree">
  <div class="searchbar">
    <input placeholder="search tensors (fuzzy)…" bind:value={$search} spellcheck="false" />
    {#if query}<span class="dim count">{total}</span>{/if}
  </div>
  <div class="rows" bind:this={scrollEl} on:scroll={onScroll} bind:clientHeight={viewportH}>
    <div class="spacer" style="height:{total * ROW_H}px">
      {#each slice as row, i (row.id)}
        <div
          class="row {row.node.kind}"
          class:sel={$selectedId === row.id}
          style="top:{(first + i) * ROW_H}px; padding-left:{6 + row.depth * 14}px"
          role="button"
          tabindex="-1"
          on:click={() => activate(row)}
          on:keydown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              activate(row);
            }
          }}
        >
          <span class="caret">{row.hasChildren ? ($expanded.has(row.id) ? '▾' : '▸') : ''}</span>
          <span class="lbl">{label(row)}</span>
          <span class="rmeta dim">{rowMeta(row)}</span>
        </div>
      {/each}
    </div>
  </div>
</div>

<style>
  .tree {
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  .searchbar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px;
    border-bottom: 1px solid var(--border);
  }
  .searchbar input {
    flex: 1 1 auto;
  }
  .count {
    flex: 0 0 auto;
  }
  .rows {
    position: relative;
    flex: 1 1 auto;
    overflow: auto;
  }
  .spacer {
    position: relative;
    width: 100%;
  }
  .row {
    position: absolute;
    left: 0;
    right: 0;
    height: 22px;
    line-height: 22px;
    display: flex;
    align-items: center;
    gap: 6px;
    padding-right: 8px;
    white-space: nowrap;
    cursor: pointer;
    overflow: hidden;
  }
  .row:hover {
    background: var(--bg-hover);
  }
  .row.sel {
    background: var(--bg-sel);
  }
  .caret {
    flex: 0 0 12px;
    color: var(--fg-dim);
    text-align: center;
  }
  .lbl {
    flex: 0 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .row.group .lbl {
    color: var(--group);
  }
  .row.tensor .lbl {
    color: var(--tensor);
  }
  .row.metadata .lbl {
    color: var(--meta);
  }
  .rmeta {
    flex: 1 1 auto;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 12px;
  }
</style>
