// The client-owned VIEW STATE (the product contract: server = data, client = view
// state). Which screen is active, tree fold set, selection, and search query.

import { writable } from 'svelte/store';

export type Screen = 'tree' | 'files' | 'layout' | 'stats' | 'health';

export const screen = writable<Screen>('tree');

/** Ids of expanded tree groups (see `lib/flatten.nodeId`). */
export const expanded = writable<Set<string>>(new Set());

/** The selected tree row id (`t:<name>` for a tensor, `m:<name>` for metadata). */
export const selectedId = writable<string | null>(null);

/** Live tree search query ('' = not searching). */
export const search = writable<string>('');

export function toggleExpanded(id: string): void {
  expanded.update((set) => {
    const next = new Set(set);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  });
}
