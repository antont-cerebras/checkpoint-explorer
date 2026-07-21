<script lang="ts">
  import {
    screen,
    back,
    navigate,
    activateSelection,
    setAllExpanded,
    startSearch,
    setTab,
    type Screen,
  } from '../stores/view';

  interface Hint {
    keys: string;
    label: string;
    act: () => void;
  }

  // The tree footer mirrors the TUI's `tree_hint_lines` (src/ui.rs), minus the
  // TUI-only actions (quit / repack / rename / OSC-52 copy / command palette).
  const treeHints: Hint[] = [
    { keys: '↑↓', label: 'navigate', act: () => {} },
    { keys: '←→', label: 'parent/child', act: () => {} },
    { keys: '⇧↑↓', label: 'sibling', act: () => {} },
    { keys: 'Enter', label: 'open', act: activateSelection },
    { keys: 'Tab', label: 'files', act: () => navigate({ kind: 'files' }) },
    { keys: 'E/C', label: 'expand all', act: () => setAllExpanded(true) },
    { keys: '/', label: 'search', act: startSearch },
    { keys: 'h', label: 'health', act: () => navigate({ kind: 'health' }) },
    { keys: 's', label: 'stats', act: () => navigate({ kind: 'stats' }) },
    { keys: 'L', label: 'layout', act: () => navigate({ kind: 'layout' }) },
  ];

  const detailHints: Hint[] = [
    { keys: 'i', label: 'info', act: () => setTab('info') },
    { keys: 'm', label: 'heatmap', act: () => setTab('heatmap') },
    { keys: 'v', label: 'values', act: () => setTab('values') },
    { keys: 'h', label: 'histogram', act: () => setTab('histogram') },
    { keys: 'Esc/⌫', label: 'back', act: back },
  ];

  const otherHints: Hint[] = [{ keys: 'Esc/⌫', label: 'back', act: back }];

  function hintsFor(s: Screen): Hint[] {
    if (s.kind === 'tree') return treeHints;
    if (s.kind === 'detail') return detailHints;
    return otherHints;
  }

  $: hints = hintsFor($screen);
</script>

<div class="footer">
  {#each hints as h}
    <button class="hint" on:click={h.act} title={h.label}>
      <span class="k">{h.keys}</span><span class="l">{h.label}</span>
    </button>
  {/each}
</div>

<style>
  .footer {
    flex: 0 0 auto;
    display: flex;
    flex-wrap: wrap;
    gap: 4px 10px;
    padding: 5px 12px;
    border-top: 1px solid var(--border);
    background: var(--bg);
    font-size: 12px;
  }
  .hint {
    display: inline-flex;
    align-items: baseline;
    gap: 5px;
    background: none;
    border: none;
    padding: 1px 2px;
    cursor: pointer;
    color: var(--fg-dim);
  }
  .hint:hover {
    background: var(--bg-hover);
    border-radius: 3px;
  }
  .k {
    color: var(--accent);
    font-weight: 600;
  }
  .l {
    color: var(--fg-dim);
  }
</style>
