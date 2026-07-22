<script lang="ts">
  import { tree } from '../stores/server';
  import { screen, openDetail } from '../stores/view';
  import { api } from '../lib/api';
  import type { LayoutMap, Segment, TreeNode } from '../lib/types';
  import { humanSize } from '../lib/format';
  import { cssVar } from '../lib/color';
  import { theme } from '../stores/theme';
  import Spinner from './Spinner.svelte';
  import Dtype from './Dtype.svelte';
  import Shape from './Shape.svelte';

  let shards: string[] = [];
  let selected = '';
  let map: LayoutMap | null = null;
  let err = '';
  let loading = false;
  let canvas: HTMLCanvasElement;
  let barH = 640;
  let hover = '';

  $: shards = collect($tree?.tree ?? []);
  $: wanted = $screen.kind === 'layout' ? $screen.file : undefined;
  $: if (shards.length && !shards.includes(selected)) selected = shards[0];
  $: if (wanted && shards.includes(wanted)) selected = wanted;
  $: if (selected) load(selected);

  function collect(nodes: TreeNode[]): string[] {
    const set = new Set<string>();
    const walk = (ns: TreeNode[]) => {
      for (const n of ns) {
        if (n.kind === 'tensor') {
          const p = n.info.source_path;
          set.add(p.split('/').pop() || p);
        } else if (n.kind === 'group') {
          walk(n.children);
        }
      }
    };
    walk(nodes);
    return [...set].sort();
  }

  async function load(f: string) {
    loading = true;
    err = '';
    try {
      map = await api.layout(f);
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
      map = null;
    }
    loading = false;
  }

  const W = 90;
  $: if (map && canvas && barH && $theme) draw(map, barH);
  function draw(m: LayoutMap, h: number) {
    canvas.width = W;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, h);
    const total = m.total_len || 1;
    for (const s of m.segments) {
      const y = (s.start / total) * h;
      const sh = Math.max(0.5, ((s.end - s.start) / total) * h);
      ctx.fillStyle = cssVar(segVar(s.kind.kind));
      ctx.fillRect(0, y, W, sh);
    }
  }

  /** Theme CSS var for a segment kind (so colors follow the theme, incl. Fallout). */
  function segVar(k: string): string {
    return k === 'header' ? '--dtype' : k === 'gap' ? '--danger' : '--accent';
  }

  function segAt(clientY: number): Segment | undefined {
    if (!map) return undefined;
    const pos = (clientY / (canvas.clientHeight || 1)) * map.total_len;
    return map.segments.find((s) => pos >= s.start && pos < s.end);
  }

  function onMove(e: MouseEvent) {
    const seg = segAt(e.offsetY);
    hover = seg ? `${seg.name} — ${humanSize(seg.end - seg.start)}` : '';
  }

  function open(seg: Segment) {
    if (seg.kind.kind === 'tensor') openDetail(seg.name);
  }
</script>

<div class="layout">
  <div class="head">
    <label>shard
      <select bind:value={selected}>
        {#each shards as s}<option value={s}>{s}</option>{/each}
      </select>
    </label>
    {#if map}
      <span class="dim">{humanSize(map.total_len)} · header {humanSize(map.header_len)} · {map.tensor_count} tensors{map.metadata.length ? ` · ${map.metadata.length} metadata` : ''}</span>
    {/if}
  </div>

  {#if loading}
    <Spinner label="parsing layout…" />
  {:else if err}
    <p class="err">{err}</p>
  {:else if map}
    <div class="map">
      <div class="bar" bind:clientHeight={barH}>
        <canvas
          bind:this={canvas}
          on:mousemove={onMove}
          on:mouseleave={() => (hover = '')}
          on:click={(e) => {
            const s = segAt(e.offsetY);
            if (s) open(s);
          }}
        ></canvas>
      </div>
      <div class="side">
        <div class="legend">
          <span><i style="background:var(--dtype)"></i> header</span>
          <span><i style="background:var(--accent)"></i> tensor</span>
          <span><i style="background:var(--danger)"></i> gap</span>
          <span class="hover mono">{hover}</span>
        </div>
        <div class="seglist">
          {#each map.segments as s}
            <!-- svelte-ignore a11y-no-noninteractive-tabindex -->
            <div
              class="seg {s.kind.kind}"
              class:clickable={s.kind.kind === 'tensor'}
              role={s.kind.kind === 'tensor' ? 'button' : undefined}
              tabindex={s.kind.kind === 'tensor' ? 0 : undefined}
              on:click={() => open(s)}
              on:keydown={(e) => {
                if ((e.key === 'Enter' || e.key === ' ') && s.kind.kind === 'tensor') {
                  e.preventDefault();
                  open(s);
                }
              }}
            >
              <i class="dot" style="background:var({segVar(s.kind.kind)})"></i>
              <span class="sname">{s.name}</span>
              {#if s.kind.kind === 'tensor'}
                <Dtype dtype={s.kind.dtype} bubble={false} />
                <Shape shape={s.kind.shape} />
              {/if}
              <span class="ssize dim mono">{humanSize(s.end - s.start)}</span>
            </div>
          {/each}
        </div>
        {#if map.metadata.length}
          <table class="meta">
            <tbody>
              {#each map.metadata as [k, v]}<tr><th>{k}</th><td class="mono">{v}</td></tr>{/each}
            </tbody>
          </table>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .layout {
    height: 100%;
    display: flex;
    flex-direction: column;
    padding: 14px 18px;
  }
  .head {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .map {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    gap: 18px;
  }
  .bar {
    flex: 0 0 auto;
    width: 90px;
    height: 100%;
  }
  canvas {
    width: 90px;
    height: 100%;
    border: 1px solid var(--border);
    border-radius: 4px;
    display: block;
    cursor: pointer;
    image-rendering: pixelated;
  }
  .side {
    flex: 1 1 auto;
    min-width: 0;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .legend {
    display: flex;
    gap: 16px;
    align-items: center;
    font-size: 12px;
    color: var(--fg-dim);
    margin-bottom: 8px;
    flex-wrap: wrap;
  }
  .legend i {
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 2px;
    vertical-align: middle;
    margin-right: 4px;
  }
  .hover {
    color: var(--accent);
  }
  .seglist {
    flex: 1 1 auto;
    overflow: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .seg {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 10px;
    white-space: nowrap;
    border-bottom: 1px solid var(--border);
  }
  .seg:last-child {
    border-bottom: none;
  }
  .seg.clickable {
    cursor: pointer;
  }
  .seg.clickable:hover {
    background: var(--bg-hover);
  }
  .dot {
    flex: 0 0 auto;
    width: 9px;
    height: 9px;
    border-radius: 2px;
  }
  .sname {
    flex: 0 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ssize {
    flex: 1 1 auto;
    text-align: right;
    font-size: 12px;
  }
  .meta {
    flex: 0 0 auto;
    margin-top: 12px;
    border-collapse: collapse;
  }
  .meta th {
    text-align: right;
    color: var(--fg-dim);
    font-weight: 400;
    padding: 2px 12px 2px 0;
  }
  .err {
    color: var(--danger);
  }
</style>
