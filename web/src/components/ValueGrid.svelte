<script lang="ts">
  import { cachedSample } from '../stores/server';
  import type { SampleDto } from '../lib/types';
  import { num } from '../lib/format';

  export let name: string;

  const R = 16;
  const C = 12;
  let data: SampleDto | null = null;
  let err = '';
  let loading = false;
  let rowOff = 0;
  let colOff = 0;
  let slice = 0;

  $: nSlices = data?.slices ?? 1;

  $: load(name, rowOff, colOff, slice);
  async function load(n: string, ro: number, co: number, sl: number) {
    loading = true;
    err = '';
    try {
      data = await cachedSample(n, { mode: 'window', rows: R, cols: C, row_off: ro, col_off: co, slice: sl });
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
      data = null;
    }
    loading = false;
  }

  function pan(dr: number, dc: number) {
    rowOff = Math.max(0, rowOff + dr * R);
    colOff = Math.max(0, colOff + dc * C);
  }
</script>

<div class="grid">
  <div class="controls">
    <div class="pan">
      <button on:click={() => pan(-1, 0)} title="up">↑</button>
      <button on:click={() => pan(1, 0)} title="down">↓</button>
      <button on:click={() => pan(0, -1)} title="left">←</button>
      <button on:click={() => pan(0, 1)} title="right">→</button>
    </div>
    {#if data}
      <span class="dim mono">rows {data.rows[0] ?? 0}… cols {data.cols[0] ?? 0}… of {data.total_rows}×{data.total_cols}</span>
    {/if}
    {#if nSlices > 1}
      <label>slice <input type="number" min="0" max={nSlices - 1} bind:value={slice} /></label>
    {/if}
  </div>

  {#if loading}
    <p class="dim">reading window…</p>
  {:else if err}
    <p class="err">{err}</p>
  {:else if data}
    <div class="tablewrap">
      <table>
        <thead>
          <tr>
            <th></th>
            {#each data.cols as c}<th class="dim">{c}</th>{/each}
          </tr>
        </thead>
        <tbody>
          {#each data.values as row, i}
            <tr>
              <th class="dim">{data.rows[i]}</th>
              {#each row as v}<td class="mono">{num(v)}</td>{/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>

<style>
  .controls {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }
  .pan {
    display: flex;
    gap: 3px;
  }
  .pan button {
    width: 26px;
    padding: 2px 0;
  }
  .controls input {
    width: 64px;
  }
  .tablewrap {
    overflow: auto;
    max-width: 100%;
  }
  table {
    border-collapse: collapse;
    font-size: 12px;
  }
  th,
  td {
    padding: 2px 8px;
    text-align: right;
    white-space: nowrap;
  }
  thead th {
    position: sticky;
    top: 0;
    background: var(--bg-panel);
  }
  .err {
    color: var(--danger);
  }
</style>
