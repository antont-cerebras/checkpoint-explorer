<script lang="ts">
  import { tree } from '../stores/server';
  import { screen } from '../stores/view';
  import { api } from '../lib/api';
  import type { LayoutMap, TreeNode } from '../lib/types';
  import { humanSize } from '../lib/format';
  import Spinner from './Spinner.svelte';

  let shards: string[] = [];
  let selected = '';
  let map: LayoutMap | null = null;
  let err = '';
  let loading = false;
  let canvas: HTMLCanvasElement;
  let hover = '';

  $: shards = collect($tree?.tree ?? []);
  // Preselect the shard the file browser opened, else the first one.
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

  const W = 960;
  const H = 48;
  $: if (map && canvas) draw(map);
  function draw(m: LayoutMap) {
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);
    const total = m.total_len || 1;
    for (const s of m.segments) {
      const x = (s.start / total) * W;
      const w = Math.max(0.5, ((s.end - s.start) / total) * W);
      ctx.fillStyle = segColor(s.kind.kind);
      ctx.fillRect(x, 0, w, H);
    }
  }

  function segColor(k: string): string {
    return k === 'header' ? '#e2b877' : k === 'gap' ? '#e06c75' : '#6db3f2';
  }

  function onMove(e: MouseEvent) {
    if (!map) return;
    const pos = (e.offsetX / canvas.clientWidth) * map.total_len;
    const seg = map.segments.find((s) => pos >= s.start && pos < s.end);
    hover = seg ? `${seg.name} — ${humanSize(seg.end - seg.start)}` : '';
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
    <span class="hover mono">{hover}</span>
  </div>

  {#if loading}
    <Spinner label="parsing layout…" />
  {:else if err}
    <p class="err">{err}</p>
  {:else if map}
    <canvas bind:this={canvas} on:mousemove={onMove} on:mouseleave={() => (hover = '')}></canvas>
    <div class="legend">
      <span><i style="background:#e2b877"></i> header</span>
      <span><i style="background:#6db3f2"></i> tensor</span>
      <span><i style="background:#e06c75"></i> gap</span>
    </div>
    {#if map.metadata.length}
      <table class="meta">
        <tbody>
          {#each map.metadata as [k, v]}<tr><th>{k}</th><td class="mono">{v}</td></tr>{/each}
        </tbody>
      </table>
    {/if}
  {/if}
</div>

<style>
  .layout {
    padding: 16px;
    height: 100%;
    overflow: auto;
  }
  .head {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .hover {
    margin-left: auto;
    color: var(--accent);
  }
  canvas {
    width: 100%;
    max-width: 960px;
    height: 48px;
    border: 1px solid var(--border);
    border-radius: 3px;
    display: block;
  }
  .legend {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    font-size: 12px;
    color: var(--fg-dim);
  }
  .legend i {
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 2px;
    vertical-align: middle;
    margin-right: 4px;
  }
  .meta {
    margin-top: 16px;
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
