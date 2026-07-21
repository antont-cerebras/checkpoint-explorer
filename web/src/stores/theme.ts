// Color theme: dark / light / system (follow the OS). Persisted to localStorage;
// applied by setting `data-theme` on <html> (the CSS in app.css keys off it).

import { get, writable } from 'svelte/store';

export type Theme = 'system' | 'dark' | 'light';

const KEY = 'ce-theme';
const mq = window.matchMedia('(prefers-color-scheme: light)');

function load(): Theme {
  const v = localStorage.getItem(KEY);
  return v === 'dark' || v === 'light' || v === 'system' ? v : 'system';
}

/** Resolve the preference to a concrete theme and apply it to the document. */
function apply(pref: Theme): void {
  const resolved = pref === 'system' ? (mq.matches ? 'light' : 'dark') : pref;
  document.documentElement.setAttribute('data-theme', resolved);
}

export const theme = writable<Theme>(load());

theme.subscribe((t) => {
  localStorage.setItem(KEY, t);
  apply(t);
});

// Follow OS changes while on "system".
mq.addEventListener('change', () => {
  if (get(theme) === 'system') apply('system');
});
