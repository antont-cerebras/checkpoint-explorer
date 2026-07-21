<script lang="ts">
  import { onMount } from 'svelte';
  import { get } from 'svelte/store';
  import { tree, treeError, ensureTree } from './stores/server';
  import {
    screen,
    searching,
    search,
    selectedId,
    visibleRows,
    back,
    forward,
    navigate,
    moveSelection,
    selectParent,
    enterChild,
    selectSibling,
    activateSelection,
    setAllExpanded,
    startSearch,
    exitSearch,
    setTab,
  } from './stores/view';
  import TreeView from './components/TreeView.svelte';
  import Detail from './components/Detail.svelte';
  import FileBrowser from './components/FileBrowser.svelte';
  import LayoutView from './components/LayoutView.svelte';
  import StatsView from './components/StatsView.svelte';
  import HealthView from './components/HealthView.svelte';
  import FilePreview from './components/FilePreview.svelte';
  import StatusBar from './components/StatusBar.svelte';
  import Footer from './components/Footer.svelte';
  import { theme } from './stores/theme';

  onMount(ensureTree);

  const PAGE = 20;

  function selectedRow() {
    const id = get(selectedId);
    return get(visibleRows).find((r) => r.id === id) ?? null;
  }

  async function copy(text: string) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      /* clipboard may be unavailable over plain http; ignore */
    }
  }

  function onKeydown(e: KeyboardEvent) {
    // Let real browser/system chords through.
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const s = get(screen);

    // --- search mode: the input is focused; only steal a few keys ---
    if (get(searching)) {
      if (e.key === 'Escape') {
        e.preventDefault();
        exitSearch();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        moveSelection(1);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        moveSelection(-1);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        activateSelection();
      }
      return; // everything else types into the search box
    }

    // --- global (any screen) ---
    if (e.key === 'Backspace') {
      e.preventDefault(); // don't let the browser navigate back
      back();
      return;
    }
    if (e.key === '\\') {
      e.preventDefault();
      forward();
      return;
    }

    if (s.kind === 'tree') {
      treeKey(e);
    } else if (s.kind === 'detail') {
      detailKey(e);
    } else {
      // files / layout / stats / health
      if (e.key === 'Escape' || (e.key === 'Tab' && s.kind === 'files')) {
        e.preventDefault();
        back();
      }
    }
  }

  function treeKey(e: KeyboardEvent) {
    switch (e.key) {
      case 'ArrowDown':
      case 'j':
        e.preventDefault();
        e.shiftKey ? selectSibling(true) : moveSelection(1);
        break;
      case 'ArrowUp':
      case 'k':
        e.preventDefault();
        e.shiftKey ? selectSibling(false) : moveSelection(-1);
        break;
      case 'ArrowLeft':
        e.preventDefault();
        selectParent();
        break;
      case 'ArrowRight':
        e.preventDefault();
        enterChild();
        break;
      case 'PageDown':
        e.preventDefault();
        moveSelection(PAGE);
        break;
      case 'PageUp':
        e.preventDefault();
        moveSelection(-PAGE);
        break;
      case 'Enter':
        e.preventDefault();
        activateSelection();
        break;
      case 'Tab':
        e.preventDefault();
        navigate({ kind: 'files' });
        break;
      case 'E':
        setAllExpanded(true);
        break;
      case 'C':
        setAllExpanded(false);
        break;
      case '/':
        e.preventDefault();
        startSearch();
        break;
      case 's':
        navigate({ kind: 'stats' });
        break;
      case 'h':
        navigate({ kind: 'health' });
        break;
      case 'L':
      case 'y': // no CLI-command copy in the browser; reuse for layout
        navigate({ kind: 'layout' });
        break;
      case 'f': {
        const r = selectedRow();
        if (r?.node.kind === 'tensor') copy(r.node.info.source_path);
        break;
      }
      case 'n': {
        const r = selectedRow();
        if (r?.node.kind === 'tensor') copy(r.node.info.name);
        break;
      }
    }
  }

  function detailKey(e: KeyboardEvent) {
    switch (e.key) {
      case 'Escape':
        e.preventDefault();
        back();
        break;
      case 's':
        setTab('stats');
        break;
      case 'h':
        setTab('histogram');
        break;
      case 'm':
        setTab('heatmap');
        break;
      case 'v':
        setTab('values');
        break;
      case 'i':
        setTab('info');
        break;
    }
  }
</script>

<svelte:window on:keydown={onKeydown} />

<div class="app">
  <header>
    <span class="title">Checkpoint&nbsp;Explorer</span>
    <span class="root" title={$tree?.root ?? ''}>{$tree?.root ?? '…'}</span>
    {#if $searching}
      <span class="search">
        /
        <!-- svelte-ignore a11y-autofocus -->
        <input
          autofocus
          spellcheck="false"
          placeholder="fuzzy filter tensors…"
          bind:value={$search}
        />
        <span class="dim">{$visibleRows.length} matches · Esc to exit</span>
      </span>
    {/if}
    <select class="theme" bind:value={$theme} title="Color theme" aria-label="Color theme">
      <option value="system">System</option>
      <option value="dark">Dark</option>
      <option value="light">Light</option>
    </select>
  </header>

  <main>
    {#if $treeError}
      <div class="error">Failed to load checkpoint: {$treeError}</div>
    {:else if $screen.kind === 'tree'}
      <TreeView />
    {:else if $screen.kind === 'detail'}
      <Detail tensor={$screen.tensor} tab={$screen.tab} />
    {:else if $screen.kind === 'files'}
      <FileBrowser />
    {:else if $screen.kind === 'layout'}
      <LayoutView />
    {:else if $screen.kind === 'stats'}
      <StatsView />
    {:else if $screen.kind === 'health'}
      <HealthView />
    {:else if $screen.kind === 'preview'}
      <FilePreview path={$screen.path} name={$screen.name} />
    {/if}
  </main>

  <StatusBar />
  <Footer />
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
    flex: 0 0 auto;
  }
  .title {
    font-weight: 600;
    color: var(--accent);
    flex: 0 0 auto;
  }
  .root {
    flex: 0 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--fg-dim);
  }
  .search {
    flex: 1 1 auto;
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--accent);
  }
  .search input {
    flex: 0 1 360px;
  }
  .theme {
    margin-left: auto;
    flex: 0 0 auto;
    font-size: 12px;
    padding: 2px 4px;
  }
  main {
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
  }
  .error {
    padding: 16px;
    color: var(--danger);
  }
</style>
