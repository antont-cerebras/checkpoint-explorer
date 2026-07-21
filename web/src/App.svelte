<script lang="ts">
  import { onMount } from 'svelte';
  import { tree, treeError, ensureTree } from './stores/server';
  import { screen, type Screen } from './stores/view';
  import TreeScreen from './components/TreeScreen.svelte';
  import FileBrowser from './components/FileBrowser.svelte';
  import LayoutView from './components/LayoutView.svelte';
  import StatsView from './components/StatsView.svelte';
  import HealthView from './components/HealthView.svelte';

  const tabs: { id: Screen; label: string }[] = [
    { id: 'tree', label: 'Tensors' },
    { id: 'files', label: 'Files' },
    { id: 'layout', label: 'Layout' },
    { id: 'stats', label: 'Stats' },
    { id: 'health', label: 'Health' },
  ];

  onMount(ensureTree);
</script>

<div class="app">
  <header>
    <span class="title">checkpoint-explorer</span>
    <span class="root dim" title={$tree?.root ?? ''}>{$tree?.root ?? '…'}</span>
    <nav>
      {#each tabs as tab}
        <button class:active={$screen === tab.id} on:click={() => screen.set(tab.id)}>
          {tab.label}
        </button>
      {/each}
    </nav>
  </header>

  <main>
    {#if $treeError}
      <div class="error">Failed to load checkpoint: {$treeError}</div>
    {:else if $screen === 'tree'}
      <TreeScreen />
    {:else if $screen === 'files'}
      <FileBrowser />
    {:else if $screen === 'layout'}
      <LayoutView />
    {:else if $screen === 'stats'}
      <StatsView />
    {:else if $screen === 'health'}
      <HealthView />
    {/if}
  </main>
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
  }
  .root {
    flex: 1 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  nav {
    display: flex;
    gap: 4px;
    flex: 0 0 auto;
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
