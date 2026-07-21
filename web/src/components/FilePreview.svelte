<script lang="ts">
  import { api } from '../lib/api';
  import { humanSize } from '../lib/format';
  import Spinner from './Spinner.svelte';

  export let path: string;
  export let name: string;

  let data: { text: string; truncated: boolean; size: number } | null = null;
  let err = '';
  let loading = true;

  $: load(path);
  async function load(p: string) {
    loading = true;
    err = '';
    data = null;
    try {
      data = await api.file(p);
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }
</script>

<div class="preview">
  <div class="head">
    <span class="name">{name}</span>
    {#if data}
      <span class="dim">· {humanSize(data.size)}{data.truncated ? ' · truncated to 1 MiB' : ''}</span>
    {/if}
  </div>
  {#if loading}
    <Spinner label="reading file…" />
  {:else if err}
    <p class="err">{err}</p>
  {:else if data}
    <pre>{data.text}</pre>
  {/if}
</div>

<style>
  .preview {
    height: 100%;
    display: flex;
    flex-direction: column;
  }
  .head {
    flex: 0 0 auto;
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
  }
  .name {
    color: var(--accent);
  }
  pre {
    flex: 1 1 auto;
    margin: 0;
    padding: 12px 16px;
    overflow: auto;
    white-space: pre;
    tab-size: 2;
    font-size: 12px;
    line-height: 1.5;
  }
  .err {
    padding: 14px;
    color: var(--danger);
  }
</style>
