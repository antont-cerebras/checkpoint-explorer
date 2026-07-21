<script lang="ts">
  import { cachedHistogram } from '../stores/server';
  import type { HistogramDto } from '../lib/types';
  import { num } from '../lib/format';

  export let name: string;
  export let dtype: string;

  const BINS = 64;
  let canvas: HTMLCanvasElement;
  let data: HistogramDto | null = null;
  let err = '';
  let loading = false;
  let hover = '';

  $: load(name);
  async function load(n: string) {
    loading = true;
    err = '';
    try {
      data = await cachedHistogram(n, BINS);
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
      data = null;
    }
    loading = false;
  }

  const W = 660;
  const H = 240;
  const PAD = 28;

  $: if (data && canvas) draw(data);
  function draw(d: HistogramDto) {
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);
    const n = d.counts.length || 1;
    const max = Math.max(1, ...d.counts);
    const bw = (W - 2 * PAD) / n;
    ctx.fillStyle = '#6db3f2';
    for (let i = 0; i < d.counts.length; i++) {
      const h = (d.counts[i] / max) * (H - 2 * PAD);
      ctx.fillRect(PAD + i * bw, H - PAD - h, Math.max(1, bw - 1), h);
    }
    ctx.strokeStyle = '#2a2e39';
    ctx.beginPath();
    ctx.moveTo(PAD, H - PAD);
    ctx.lineTo(W - PAD, H - PAD);
    ctx.stroke();
  }

  function span(d: HistogramDto): [number, number] {
    if (d.bins.type === 'int') return [d.bins.start, d.bins.start + d.bins.step * d.counts.length];
    return [d.bins.lo, d.bins.hi];
  }

  function onMove(e: MouseEvent) {
    if (!data) return;
    const n = data.counts.length || 1;
    const bw = (W - 2 * PAD) / n;
    const i = Math.floor((e.offsetX - PAD) / bw);
    if (i < 0 || i >= data.counts.length) {
      hover = '';
      return;
    }
    hover = `bin ${i}: ${data.counts[i].toLocaleString()}`;
  }
</script>

<div class="hist">
  <div class="controls">
    {#if data}
      <span class="dim">{data.total.toLocaleString()} values · {data.counts.length} bins{data.nonfinite ? ` · ${data.nonfinite} non-finite` : ''}</span>
    {/if}
    <span class="hover mono">{hover}</span>
  </div>
  {#if loading}
    <p class="dim">scanning tensor…</p>
  {:else if err}
    <p class="err">{err}</p>
  {:else if data}
    <canvas bind:this={canvas} on:mousemove={onMove} on:mouseleave={() => (hover = '')}></canvas>
    <div class="axis mono">
      <span>{num(span(data)[0])}</span>
      <span class="dim">{dtype}</span>
      <span>{num(span(data)[1])}</span>
    </div>
  {/if}
</div>

<style>
  .controls {
    display: flex;
    gap: 12px;
    margin-bottom: 8px;
  }
  .hover {
    margin-left: auto;
    color: var(--accent);
  }
  canvas {
    border: 1px solid var(--border);
    max-width: 100%;
  }
  .axis {
    display: flex;
    justify-content: space-between;
    width: 660px;
    max-width: 100%;
    font-size: 11px;
    margin-top: 4px;
  }
  .err {
    color: var(--danger);
  }
</style>
