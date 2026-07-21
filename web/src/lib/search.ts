// A small fuzzy subsequence matcher for the tree search box — mirrors the TUI's
// "type to filter tensors" behavior without a dependency.

import type { TreeNode } from './types';
import { nodeId, type Row } from './flatten';

/** Subsequence score (higher = better); -1 if `needle` isn't a subsequence of `hay`. */
export function fuzzyScore(needle: string, hay: string): number {
  const n = needle.toLowerCase();
  const h = hay.toLowerCase();
  if (!n) return 0;
  let hi = 0;
  let score = 0;
  let streak = 0;
  for (const c of n) {
    let found = -1;
    for (let k = hi; k < h.length; k++) {
      if (h[k] === c) {
        found = k;
        break;
      }
    }
    if (found < 0) return -1;
    streak = found === hi ? streak + 1 : 0;
    score += 1 + streak * 2 + (found === 0 ? 5 : 0);
    hi = found + 1;
  }
  return score - h.length * 0.01; // gently prefer shorter (tighter) matches
}

/** Leaf (tensor/metadata) rows across the whole tree that match `query`, ranked. */
export function searchRows(tree: TreeNode[], query: string, limit = 1000): Row[] {
  const scored: { row: Row; score: number }[] = [];
  const walk = (nodes: TreeNode[], parentId: string) => {
    for (const node of nodes) {
      const id = nodeId(node, parentId);
      if (node.kind === 'group') {
        walk(node.children, id);
      } else {
        const name = node.info.name;
        const score = fuzzyScore(query, name);
        if (score >= 0) scored.push({ row: { id, node, depth: 0, hasChildren: false }, score });
      }
    }
  };
  walk(tree, '');
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit).map((s) => s.row);
}
