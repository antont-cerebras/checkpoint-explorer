// The client-owned VIEW STATE, mirroring the TUI's full-screen modal flow: one
// active screen at a time, a browser-style history stack (Backspace / \), and the
// persistent tree fold/selection/search state. Navigation + tree-cursor helpers
// live here so the global key handler (App.svelte) and the views share them.

import { derived, get, writable } from 'svelte/store';
import { flatten, nodeId, type Row } from '../lib/flatten';
import { searchRows } from '../lib/search';
import { tree as treeData } from './server';

export type DataTab = 'info' | 'heatmap' | 'values' | 'histogram' | 'stats';

export type Screen =
  | { kind: 'tree' }
  | { kind: 'detail'; tensor: string; tab: DataTab }
  | { kind: 'files' }
  | { kind: 'layout' }
  | { kind: 'stats' }
  | { kind: 'health' };

export const screen = writable<Screen>({ kind: 'tree' });

// Persistent tree state (survives screen changes, like the TUI).
export const expanded = writable<Set<string>>(new Set());
export const selectedId = writable<string | null>(null);
export const searching = writable<boolean>(false);
export const search = writable<string>('');

/** The flattened, fold-/search-aware visible rows — shared by TreeView (render)
 * and the key handler (cursor movement). */
export const visibleRows = derived(
  [treeData, expanded, search, searching],
  ([$t, $exp, $q, $searching]) => {
    if (!$t) return [] as Row[];
    return $searching && $q.trim() ? searchRows($t.tree, $q.trim()) : flatten($t.tree, $exp);
  },
);

// ---- screen history (browser-style back/forward) ----

let stack: Screen[] = [{ kind: 'tree' }];
let cursor = 0;

export function navigate(s: Screen): void {
  stack = stack.slice(0, cursor + 1);
  stack.push(s);
  cursor = stack.length - 1;
  screen.set(s);
}
export function back(): void {
  if (cursor > 0) {
    cursor -= 1;
    screen.set(stack[cursor]);
  }
}
export function forward(): void {
  if (cursor < stack.length - 1) {
    cursor += 1;
    screen.set(stack[cursor]);
  }
}

export function openDetail(tensor: string): void {
  navigate({ kind: 'detail', tensor, tab: 'info' });
}
export function setTab(tab: DataTab): void {
  screen.update((s) => (s.kind === 'detail' ? { ...s, tab } : s));
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
