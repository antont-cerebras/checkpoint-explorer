// The client-owned VIEW STATE, mirroring the TUI's full-screen modal flow: one
// active screen at a time, a browser-style history stack (Backspace / \), and the
// persistent tree fold/selection/search state. Navigation + tree-cursor helpers
// live here so the global key handler (App.svelte) and the views share them.

import { derived, get, writable } from 'svelte/store';
import { flatten, nodeId, type Row } from '../lib/flatten';
import { searchRows } from '../lib/search';
import type { TreeNode } from '../lib/types';
import { tree as treeData } from './server';

export type DataTab = 'info' | 'heatmap' | 'values' | 'histogram';

export type Screen =
  | { kind: 'tree' }
  | { kind: 'detail'; tensor: string; tab: DataTab }
  | { kind: 'files' }
  | { kind: 'layout'; file?: string }
  | { kind: 'stats' }
  | { kind: 'health' }
  | { kind: 'preview'; path: string; name: string };

// The current screen is driven by the URL hash, so the browser's back/forward
// buttons work natively and every screen+mode has a shareable link.
export const screen = writable<Screen>(parseHash());

function screenToHash(s: Screen): string {
  const enc = encodeURIComponent;
  switch (s.kind) {
    case 'tree':
      return 'tree';
    case 'detail':
      return `detail?t=${enc(s.tensor)}&tab=${s.tab}`;
    case 'files':
      return 'files';
    case 'layout':
      return s.file ? `layout?file=${enc(s.file)}` : 'layout';
    case 'stats':
      return 'stats';
    case 'health':
      return 'health';
    case 'preview':
      return `preview?path=${enc(s.path)}&name=${enc(s.name)}`;
  }
}

function parseHash(): Screen {
  const raw = location.hash.replace(/^#/, '');
  const [kind, queryStr] = raw.split('?');
  const q = new URLSearchParams(queryStr ?? '');
  switch (kind) {
    case 'detail': {
      const t = q.get('t');
      const raw = q.get('tab') ?? 'info';
      const tab = (['info', 'heatmap', 'values', 'histogram'].includes(raw) ? raw : 'info') as DataTab;
      if (t) return { kind: 'detail', tensor: t, tab };
      break;
    }
    case 'files':
      return { kind: 'files' };
    case 'layout':
      return { kind: 'layout', file: q.get('file') ?? undefined };
    case 'stats':
      return { kind: 'stats' };
    case 'health':
      return { kind: 'health' };
    case 'preview': {
      const path = q.get('path');
      if (path) return { kind: 'preview', path, name: q.get('name') ?? path };
      break;
    }
  }
  return { kind: 'tree' };
}

// Keep the store in sync with the URL (covers the browser back/forward buttons).
window.addEventListener('hashchange', () => screen.set(parseHash()));

// Persistent tree state (survives screen changes, like the TUI).
export const expanded = writable<Set<string>>(new Set());
export const selectedId = writable<string | null>(null);
export const searching = writable<boolean>(false);
export const search = writable<string>('');

/** Active tensor facet filters (from clicking dtype / shape / dimension badges).
 * They stack — a tensor must match ALL of them (AND). */
export type Filter =
  | { kind: 'dtype'; value: string }
  | { kind: 'shape'; dims: number[] }
  | { kind: 'dim'; value: number };
export const filters = writable<Filter[]>([]);

function matchesFilter(shape: number[], dtype: string, f: Filter): boolean {
  if (f.kind === 'dtype') return dtype === f.value;
  if (f.kind === 'dim') return shape.includes(f.value);
  return shape.length === f.dims.length && shape.every((d, i) => d === f.dims[i]);
}

function sameFilter(a: Filter, b: Filter): boolean {
  if (a.kind === 'dtype' && b.kind === 'dtype') return a.value === b.value;
  if (a.kind === 'dim' && b.kind === 'dim') return a.value === b.value;
  if (a.kind === 'shape' && b.kind === 'shape')
    return a.dims.length === b.dims.length && a.dims.every((d, i) => d === b.dims[i]);
  return false;
}

export function filterLabel(f: Filter): string {
  if (f.kind === 'dtype') return `dtype ${f.value}`;
  if (f.kind === 'dim') return `dim ${f.value}`;
  return `shape ${f.dims.join('×') || 'scalar'}`;
}

/** Command palette (Space / `:`) open state. */
export const paletteOpen = writable<boolean>(false);

/** The flattened rows — fold-aware, or a flat list while searching / dtype-filtering.
 * Shared by TreeView (render) and the key handler (cursor movement). */
export const visibleRows = derived(
  [treeData, expanded, search, searching, filters],
  ([$t, $exp, $q, $searching, $f]) => {
    if (!$t) return [] as Row[];
    if ($searching && $q.trim()) return searchRows($t.tree, $q.trim());
    if ($f.length) return filterRows($t.tree, $f);
    return flatten($t.tree, $exp);
  },
);

/** Flat list of tensor rows matching ALL active facet filters. */
function filterRows(nodes: TreeNode[], fs: Filter[]): Row[] {
  const out: Row[] = [];
  const walk = (ns: TreeNode[], parentId: string) => {
    for (const n of ns) {
      const id = nodeId(n, parentId);
      if (n.kind === 'group') walk(n.children, id);
      else if (n.kind === 'tensor' && fs.every((f) => matchesFilter(n.info.shape, n.info.dtype, f)))
        out.push({ id, node: n, depth: 0, hasChildren: false });
    }
  };
  walk(nodes, '');
  return out;
}

/** Add a filter to the active set (deduped, AND-ed with the rest). */
function addFilter(f: Filter): void {
  filters.update((cur) => (cur.some((x) => sameFilter(x, f)) ? cur : [...cur, f]));
  searching.set(false);
  search.set('');
  navigate({ kind: 'tree' });
}
export function filterByDtype(value: string): void {
  addFilter({ kind: 'dtype', value });
}
export function filterByShape(dims: number[]): void {
  addFilter({ kind: 'shape', dims });
}
export function filterByDim(value: number): void {
  addFilter({ kind: 'dim', value });
}
export function removeFilter(index: number): void {
  filters.update((cur) => cur.filter((_, i) => i !== index));
}
export function clearFilter(): void {
  filters.set([]);
}

// Expand the synthetic root node once the tree first loads, so its children show.
let seededExpand = false;
treeData.subscribe((t) => {
  if (t && !seededExpand) {
    seededExpand = true;
    const first = t.tree[0];
    if (first) expanded.set(new Set([nodeId(first, '')]));
  }
});

// ---- navigation (URL hash = source of truth; browser back/forward just work) ----

export function navigate(s: Screen): void {
  const h = `#${screenToHash(s)}`;
  if (location.hash !== h) location.hash = h; // pushes a history entry
  screen.set(s); // optimistic (hashchange will confirm)
}
export function back(): void {
  history.back();
}
export function forward(): void {
  history.forward();
}

export function openDetail(tensor: string): void {
  navigate({ kind: 'detail', tensor, tab: 'info' });
}

/** Open a file from the browser: a safetensors shard jumps to its byte-layout map;
 * anything else gets a text preview. */
export function openFile(path: string, name: string, fileKind: string): void {
  if (fileKind === 'Checkpoint') navigate({ kind: 'layout', file: name });
  else navigate({ kind: 'preview', path, name });
}
export function setTab(tab: DataTab): void {
  const s = get(screen);
  if (s.kind === 'detail') navigate({ ...s, tab });
}

// ---- tree cursor movement (mirrors kernel::TreeState nav) ----

function rowIndex(): number {
  const rows = get(visibleRows);
  const id = get(selectedId);
  const i = rows.findIndex((r) => r.id === id);
  return i < 0 ? 0 : i;
}

function selectAt(i: number): void {
  const rows = get(visibleRows);
  if (!rows.length) return;
  const clamped = Math.max(0, Math.min(rows.length - 1, i));
  selectedId.set(rows[clamped].id);
}

export function moveSelection(delta: number): void {
  selectAt(rowIndex() + delta);
}

/** ← : jump to the parent group (nearest preceding shallower row). */
export function selectParent(): void {
  const rows = get(visibleRows);
  const i = rowIndex();
  const depth = rows[i]?.depth ?? 0;
  if (depth === 0) return;
  for (let k = i - 1; k >= 0; k--) {
    if (rows[k].depth < depth) {
      selectedId.set(rows[k].id);
      return;
    }
  }
}

/** → : enter the selected group (expand if needed) and move to its first child. */
export function enterChild(): void {
  const rows = get(visibleRows);
  const i = rowIndex();
  const row = rows[i];
  if (!row || !row.hasChildren) return;
  const exp = get(expanded);
  if (!exp.has(row.id)) {
    toggle(row.id);
    // first child is the next row once re-flattened
    const next = get(visibleRows)[i + 1];
    if (next && next.depth === row.depth + 1) selectedId.set(next.id);
  } else {
    const next = rows[i + 1];
    if (next && next.depth === row.depth + 1) selectedId.set(next.id);
  }
}

/** Shift+↑/↓ : previous/next sibling (same depth, without leaving the parent). */
export function selectSibling(forwardDir: boolean): void {
  const rows = get(visibleRows);
  const i = rowIndex();
  const depth = rows[i]?.depth ?? 0;
  const range = forwardDir
    ? Array.from({ length: rows.length - i - 1 }, (_, k) => i + 1 + k)
    : Array.from({ length: i }, (_, k) => i - 1 - k);
  for (const k of range) {
    if (rows[k].depth < depth) break;
    if (rows[k].depth === depth) {
      selectedId.set(rows[k].id);
      break;
    }
  }
}

/** Enter : expand/collapse a group, or open a tensor's detail. */
export function activateSelection(): void {
  const rows = get(visibleRows);
  const row = rows[rowIndex()];
  if (!row) return;
  if (row.node.kind === 'tensor') openDetail(row.node.info.name);
  else if (row.hasChildren) toggle(row.id);
}

export function toggle(id: string): void {
  expanded.update((set) => {
    const next = new Set(set);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  });
}

export function setAllExpanded(on: boolean): void {
  if (!on) {
    expanded.set(new Set());
    return;
  }
  const t = get(treeData);
  if (!t) return;
  const ids = new Set<string>();
  const walk = (nodes: typeof t.tree, parentId: string) => {
    for (const n of nodes) {
      if (n.kind === 'group') {
        const id = nodeId(n, parentId);
        ids.add(id);
        walk(n.children, id);
      }
    }
  };
  walk(t.tree, '');
  expanded.set(ids);
}

export function startSearch(): void {
  searching.set(true);
}
export function exitSearch(): void {
  searching.set(false);
  search.set('');
}
