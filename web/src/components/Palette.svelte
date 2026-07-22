<script lang="ts">
  import {
    paletteOpen,
    navigate,
    setAllExpanded,
    startSearch,
    filterByDtype,
    clearFilter,
  } from '../stores/view';
  import { tree } from '../stores/server';
  import { theme } from '../stores/theme';
  import { fuzzyScore } from '../lib/search';
  import type { TreeNode } from '../lib/types';

  interface Cmd {
    group: string;
    label: string;
    run: () => void;
  }

  const base: Cmd[] = [
    { group: 'Go', label: 'Tensor tree', run: () => navigate({ kind: 'tree' }) },
    { group: 'Go', label: 'File browser', run: () => navigate({ kind: 'files' }) },
    { group: 'Go', label: 'Byte layout map', run: () => navigate({ kind: 'layout' }) },
    { group: 'Go', label: 'Statistics', run: () => navigate({ kind: 'stats' }) },
    { group: 'Go', label: 'Health check', run: () => navigate({ kind: 'health' }) },
    { group: 'Tree', label: 'Expand all groups', run: () => setAllExpanded(true) },
    { group: 'Tree', label: 'Collapse all groups', run: () => setAllExpanded(false) },
    { group: 'Tree', label: 'Search tensors', run: () => { navigate({ kind: 'tree' }); startSearch(); } },
    { group: 'Theme', label: 'Theme: System', run: () => theme.set('system') },
    { group: 'Theme', label: 'Theme: Dark', run: () => theme.set('dark') },
    { group: 'Theme', label: 'Theme: Light', run: () => theme.set('light') },
    { group: 'Theme', label: 'Theme: Fallout', run: () => theme.set('fallout') },
  ];

  function distinctDtypes(nodes: TreeNode[]): string[] {
    const set = new Set<string>();
    const walk = (ns: TreeNode[]) => {
      for (const n of ns) {
        if (n.kind === 'tensor') set.add(n.info.dtype);
        else if (n.kind === 'group') walk(n.children);
      }
    };
    walk(nodes);
    return [...set].sort();
  }

  // Filter commands are data-driven: one per dtype present, plus a clear.
  $: dtypes = $tree ? distinctDtypes($tree.tree) : [];
  $: commands = [
    ...base,
    ...dtypes.map((d) => ({ group: 'Filter', label: `Filter dtype: ${d}`, run: () => filterByDtype(d) })),
    { group: 'Filter', label: 'Clear filter', run: clearFilter },
  ];

  let q = '';
  let sel = 0;

  $: filtered = q.trim()
    ? commands
        .map((c) => ({ c, s: fuzzyScore(q.trim(), `${c.group} ${c.label}`) }))
        .filter((x) => x.s >= 0)
        .sort((a, b) => b.s - a.s)
        .map((x) => x.c)
    : commands;
  $: if (sel >= filtered.length) sel = Math.max(0, filtered.length - 1);

  function run(c: Cmd) {
    paletteOpen.set(false);
    c.run();
  }
  function onKey(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.preventDefault();
      paletteOpen.set(false);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      sel = Math.min(filtered.length - 1, sel + 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      sel = Math.max(0, sel - 1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (filtered[sel]) run(filtered[sel]);
    }
  }
</script>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
<div
  class="backdrop"
  role="presentation"
  on:click={(e) => {
    if (e.target === e.currentTarget) paletteOpen.set(false);
  }}
>
  <div class="palette" role="dialog" aria-label="Command palette">
    <!-- svelte-ignore a11y-autofocus -->
    <input autofocus placeholder="Run a command…" bind:value={q} on:keydown={onKey} />
    <ul>
      {#each filtered as c, i}
        <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-noninteractive-element-interactions -->
        <li
          class:sel={i === sel}
          role="option"
          aria-selected={i === sel}
          on:click={() => run(c)}
          on:mousemove={() => (sel = i)}
        >
          <span class="cgroup">{c.group}</span><span class="clabel">{c.label}</span>
        </li>
      {/each}
      {#if !filtered.length}<li class="empty dim">no matching commands</li>{/if}
    </ul>
  </div>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.45);
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding-top: 12vh;
    z-index: 40;
  }
  .palette {
    width: 460px;
    max-width: 92vw;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    overflow: hidden;
  }
  input {
    width: 100%;
    border: none;
    border-bottom: 1px solid var(--border);
    border-radius: 0;
    padding: 10px 14px;
    font-size: 14px;
    background: var(--bg);
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 4px;
    max-height: 50vh;
    overflow: auto;
  }
  li {
    display: flex;
    align-items: baseline;
    gap: 10px;
    padding: 6px 10px;
    border-radius: 5px;
    cursor: pointer;
  }
  li.sel {
    background: var(--bg-sel);
  }
  .cgroup {
    flex: 0 0 54px;
    color: var(--fg-dim);
    font-size: 11px;
    text-transform: uppercase;
  }
  .empty {
    cursor: default;
  }
</style>
