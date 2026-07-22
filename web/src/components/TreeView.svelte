<script lang="ts">
  import { visibleRows, selectedId, expanded, searching, toggle, openDetail, navigate } from '../stores/view';
  import type { Row } from '../lib/flatten';
  import { humanCount, humanSize } from '../lib/format';
  import Dtype from './Dtype.svelte';
  import Shape from './Shape.svelte';

  const ROW_H = 22;
  let scrollEl: HTMLDivElement;
  let scrollTop = 0;
  let viewportH = 600;

  // Themed, interactive hover popover (fixed-position so the scroll container can't
  // clip it; stays open while hovered so its buttons are clickable).
  let tipRow: Row | null = null;
  let tipTop = 0;
  let tipLeft = 0;
  let copied = '';
  let hideTimer: ReturnType<typeof setTimeout> | undefined;

  function openTip(e: MouseEvent, row: Row) {
    // Non-tensor rows have no popover, but don't hard-close: let any open popover
    // linger (via the grace timer) so passing over one on the way to it is fine.
    if (row.node.kind !== 'tensor') {
      scheduleHide();
      return;
    }
    clearTimeout(hideTimer);
    // Place it beside the cursor horizontally but aligned to the row's top: reaching
    // it is a straight move right along the same full-width row (never crossing other
    // rows), while staying close to the pointer.
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    tipTop = r.top;
    tipLeft = e.clientX;
    copied = '';
    tipRow = row;
  }
  function keepTip() {
    clearTimeout(hideTimer);
  }
  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => (tipRow = null), 450);
  }
  async function copyVal(key: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      copied = key;
      setTimeout(() => copied === key && (copied = ''), 1000);
    } catch {
      /* clipboard unavailable */
    }
  }
  function baseName(p: string): string {
    return p.split('/').pop() || p;
  }
  $: tipInfo = tipRow && tipRow.node.kind === 'tensor' ? tipRow.node.info : null;
  $: tipStyle = tipRow
    ? `left:${Math.min(tipLeft + 8, window.innerWidth - 330)}px; top:${Math.max(8, Math.min(tipTop, window.innerHeight - 230))}px`
    : '';

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
        role="button"
        tabindex="-1"
        on:click={() => click(row)}
        on:mouseenter={(e) => openTip(e, row)}
        on:mouseleave={scheduleHide}
      >
        <span class="caret">{row.hasChildren ? ($expanded.has(row.id) ? '▾' : '▸') : ''}</span>
        <span class="lbl">{label(row, $searching)}</span>
        {#if row.node.kind === 'tensor'}
          <span class="tmeta">
            <Dtype dtype={row.node.info.dtype} bubble={false} />
            <Shape shape={row.node.info.shape} />
            <span class="dim">{humanSize(row.node.info.size_bytes)}</span>
          </span>
        {:else}
          <span class="rmeta dim">{rowMeta(row)}</span>
        {/if}
      </div>
    {/each}
  </div>
</div>

{#if tipInfo}
  <div class="rowpop" style={tipStyle} role="tooltip" on:mouseenter={keepTip} on:mouseleave={scheduleHide}>
    <div class="pname">{tipInfo.name}</div>
    <div class="prow">
      <span class="pk">dtype</span>
      <Dtype dtype={tipInfo.dtype} />
      <button class="cp" title="Copy dtype" on:click={() => copyVal('d', tipInfo.dtype)}>{copied === 'd' ? '✓' : '⧉'}</button>
    </div>
    <div class="prow">
      <span class="pk">shape</span>
      <Shape shape={tipInfo.shape} />
      <button class="cp" title="Copy shape" on:click={() => copyVal('s', tipInfo.shape.join('×'))}>{copied === 's' ? '✓' : '⧉'}</button>
    </div>
    <div class="prow"><span class="pk">params</span><span class="mono">{humanCount(tipInfo.num_elements)}</span></div>
    <div class="prow"><span class="pk">size</span><span class="mono">{humanSize(tipInfo.size_bytes)}</span></div>
    <div class="prow">
      <span class="pk">shard</span>
      <button class="link" title="Open this shard's layout" on:click={() => navigate({ kind: 'layout', file: baseName(tipInfo.source_path) })}>{baseName(tipInfo.source_path)}</button>
      <button class="cp" title="Copy file path" on:click={() => copyVal('f', tipInfo.source_path)}>{copied === 'f' ? '✓' : '⧉'}</button>
    </div>
    <div class="prow">
      <span class="pk">name</span>
      <button class="cp wide" on:click={() => copyVal('n', tipInfo.name)}>{copied === 'n' ? '✓ copied' : '⧉ copy name'}</button>
    </div>
  </div>
{/if}

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

  .rowpop {
    position: fixed;
    z-index: 30;
    min-width: 240px;
    max-width: 320px;
    padding: 8px 10px;
    background: var(--bg-elev);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.45);
    font-size: 12px;
  }
  .pname {
    color: var(--accent);
    word-break: break-all;
    margin-bottom: 6px;
  }
  .prow {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 2px 0;
  }
  .pk {
    flex: 0 0 46px;
    color: var(--fg-dim);
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.04em;
  }
  .cp {
    margin-left: auto;
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--fg-dim);
    cursor: pointer;
    font: inherit;
    padding: 0 5px;
    line-height: 16px;
  }
  .cp:hover {
    color: var(--accent);
    border-color: var(--accent);
  }
  .cp.wide {
    margin-left: 0;
  }
  .rowpop .link {
    background: none;
    border: none;
    padding: 0;
    color: var(--accent);
    text-decoration: underline;
    text-decoration-style: dotted;
    cursor: pointer;
    font: inherit;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
