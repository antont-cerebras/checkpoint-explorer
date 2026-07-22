<script lang="ts">
  import { visibleRows, selectedId, expanded, searching, toggle, openDetail } from '../stores/view';
  import type { Row } from '../lib/flatten';
  import { humanCount, humanSize } from '../lib/format';
  import Dtype from './Dtype.svelte';

  const ROW_H = 22;
  let scrollEl: HTMLDivElement;
  let scrollTop = 0;
  let viewportH = 600;

  $: rows = $visibleRows;
  $: total = rows.length;
  $: first = Math.max(0, Math.floor(scrollTop / ROW_H) - 6);
  $: slice = rows.slice(first, first + Math.ceil(viewportH / ROW_H) + 12);

  function onScroll() {
    scrollTop = scrollEl.scrollTop;
  }

  function click(row: Row) {
    selectedId.set(row.id);
    if (row.node.kind === 'tensor') openDetail(row.node.info.name);
    else if (row.hasChildren) toggle(row.id);
  }

  // Keep the keyboard-selected row visible (essential at 31k rows).
  $: keepVisible($selectedId, rows);
  function keepVisible(id: string | null, rs: Row[]) {
    if (!scrollEl || id == null) return;
    const idx = rs.findIndex((r) => r.id === id);
    if (idx < 0) return;
    const top = idx * ROW_H;
    if (top < scrollEl.scrollTop) scrollEl.scrollTop = top;
    else if (top + ROW_H > scrollEl.scrollTop + viewportH) {
      scrollEl.scrollTop = top + ROW_H - viewportH;
    }
  }

  // While searching, results are a flat list, so show the FULL tensor name (the
  // compacted last-segment label is only meaningful within the indented tree).
  function label(row: Row, isSearching: boolean): string {
    const n = row.node;
    if (n.kind === 'group') return n.name;
    if (n.kind === 'tensor') {
      return isSearching ? n.info.name : (n.label ?? n.info.name.split('.').pop() ?? n.info.name);
    }
    return n.info.name;
  }

  // Hover tooltip: the detail fields that need no computation (native title, so it
  // isn't clipped by the scroll container).
  function rowTitle(row: Row): string | undefined {
    const n = row.node;
    if (n.kind === 'tensor') {
      const t = n.info;
      const lines = [
        t.name,
        `${t.dtype}  ${t.shape.join(' × ') || 'scalar'}`,
        `${humanCount(t.num_elements)} params · ${humanSize(t.size_bytes)}`,
        t.source_path,
      ];
      return lines.join('\n');
    }
    if (n.kind === 'group') {
      return `${n.name}\n${humanCount(n.tensor_count)} tensors · ${humanSize(n.total_size)}`;
    }
    return `${n.info.name} [${n.info.value_type}]: ${n.info.value}`;
  }

  function rowMeta(row: Row): string {
    const n = row.node;
    if (n.kind === 'group') {
      return ` (${humanCount(n.tensor_count)}, ${humanSize(n.total_size)})`;
    }
    if (n.kind === 'tensor') {
      return ` [${n.info.dtype}, ${n.info.shape.join('×')}, ${humanSize(n.info.size_bytes)}]`;
    }
    return ` [${n.info.value_type}]: ${n.info.value}`;
  }
</script>

<div class="rows" bind:this={scrollEl} on:scroll={onScroll} bind:clientHeight={viewportH}>
  <div class="spacer" style="height:{total * ROW_H}px">
    {#each slice as row, i (row.id)}
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
      <div
        class="row {row.node.kind}"
        class:sel={$selectedId === row.id}
        style="top:{(first + i) * ROW_H}px; padding-left:{6 + row.depth * 14}px"
        title={rowTitle(row)}
        role="button"
        tabindex="-1"
        on:click={() => click(row)}
      >
        <span class="caret">{row.hasChildren ? ($expanded.has(row.id) ? '▾' : '▸') : ''}</span>
        <span class="lbl">{label(row, $searching)}</span>
        {#if row.node.kind === 'tensor'}
          <span class="tmeta">
            <Dtype dtype={row.node.info.dtype} bubble={false} />
            <span class="dim">{row.node.info.shape.join('×')}</span>
            <span class="dim">{humanSize(row.node.info.size_bytes)}</span>
          </span>
        {:else}
          <span class="rmeta dim">{rowMeta(row)}</span>
        {/if}
      </div>
    {/each}
  </div>
</div>

<style>
  .rows {
    position: relative;
    height: 100%;
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
    padding-right: 10px;
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
    flex: 0 0 auto;
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
    flex: 0 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 12px;
  }
  .tmeta {
    flex: 0 1 auto;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    overflow: hidden;
    font-size: 12px;
  }
</style>
