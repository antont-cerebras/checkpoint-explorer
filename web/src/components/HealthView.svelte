<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import JsonView from './JsonView.svelte';

  let health: unknown[] | null = null;
  let check: Record<string, unknown> | null = null;
  let err = '';

  onMount(async () => {
    try {
      [health, check] = await Promise.all([api.health(), api.check()]);
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    }
  });
</script>

<div class="health">
  {#if err}
    <p class="err">{err}</p>
  {:else}
    <section>
      <h3>Structural check</h3>
      {#if check}<JsonView value={check} />{:else}<p class="dim">loading…</p>{/if}
    </section>
    <section>
      <h3>Index health</h3>
      {#if health && health.length}
        <JsonView value={health} />
      {:else if health}
        <p class="dim">No <code>model.safetensors.index.json</code> to reconcile.</p>
      {:else}
        <p class="dim">loading…</p>
      {/if}
    </section>
  {/if}
</div>

<style>
  .health {
    padding: 16px;
    height: 100%;
    overflow: auto;
  }
  section {
    margin-bottom: 22px;
  }
  h3 {
    margin: 0 0 8px;
    color: var(--accent);
    font-size: 14px;
  }
  .err {
    color: var(--danger);
  }
</style>
