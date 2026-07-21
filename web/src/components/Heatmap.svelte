<script lang="ts">
  import { cachedSample } from '../stores/server';
  import type { SampleDto } from '../lib/types';
  import { viridis } from '../lib/color';
  import { num } from '../lib/format';

  export let name: string;

  const MAX = 128;
  let canvas: HTMLCanvasElement;
  let data: SampleDto | null = null;
  let err = '';
  let loading = false;
  let slice = 0;
  let cell = 6;
  let hover = '';

  $: nSlices = data?.slices ?? 1;

  $: load(name, slice);
  async function load(n: string, sl: number) {
    loading = true;
    err = '';
    try {
      data = await cachedSample(n, { mode: 'grid', rows: MAX, cols: MAX, slice: sl });
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
      data = null;
    }
    loading = false;
  }

  $: if (data && canvas) draw(data);
  function draw(d: SampleDto) {
    const rows = d.values.length;
    const cols = rows ? d.values[0].length : 0;
    if (!rows || !cols) return;
    cell = Math.max(2, Math.min(16, Math.floor(620 / Math.max(rows, cols))));
    canvas.width = cols * cell;
    canvas.height = rows * cell;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const range = d.max - d.min || 1;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = d.values[i][j];
        ctx.fillStyle = Number.isFinite(v) ? viridis((v - d.min) / range) : '#000';
        ctx.fillRect(j * cell, i * cell, cell, cell);
      }
    }
  }

  function onMove(e: MouseEvent) {
    if (!data) return;
    const j = Math.floor(e.offsetX / cell);
    const i = Math.floor(e.offsetY / cell);
    const row = data.values[i];
    if (!row || j < 0 || j >= row.length) {
      hover = '';
      return;
    }
    hover = `[${data.rows[i]}, ${data.cols[j]}] = ${num(row[j])}`;
  }
</script>

<div class="heatmap">
  <div class="controls">
    {#if data}
      <span class="dim">viewing {data.values.length}×{data.values[0]?.length ?? 0} of {data.total_rows}×{data.total_cols}</span>
    {/if}
    {#if nSlices > 1}
      <label>slice
        <input type="number" min="0" max={nSlices - 1} bind:value={slice} />
        / {nSlices - 1}
      </label>
    {/if}
    <span class="hover mono">{hover}</span>
  </div>

  {#if loading}
    <p class="dim">scanning tensor…</p>
  {:else if err}
    <p class="err">{err}</p>
  {:else if data}
    <canvas bind:this={canvas} on:mousemove={onMove} on:mouseleave={() => (hover = '')}></canvas>
    <div class="scale">
      <span class="mono">{num(data.min)}</span>
      <span class="ramp"></span>
      <span class="mono">{num(data.max)}</span>
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
  .controls input {
    width: 70px;
  }
  .hover {
    margin-left: auto;
    color: var(--accent);
  }
  canvas {
    image-rendering: pixelated;
    border: 1px solid var(--border);
    max-width: 100%;
  }
  .scale {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
    font-size: 11px;
  }
  .ramp {
    width: 160px;
    height: 10px;
    border-radius: 2px;
    background: linear-gradient(
      to right,
      rgb(68, 1, 84),
      rgb(59, 82, 139),
      rgb(33, 145, 140),
      rgb(94, 201, 98),
      rgb(253, 231, 37)
    );
  }
  .err {
    color: var(--danger);
  }
</style>
