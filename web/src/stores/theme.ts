// Color theme: dark / light / system (follow the OS). Persisted to localStorage;
// applied by setting `data-theme` on <html> (the CSS in app.css keys off it).

import { get, writable } from 'svelte/store';

export type Theme = 'system' | 'dark' | 'light' | 'fallout';

const KEY = 'ce-theme';
const mq = window.matchMedia('(prefers-color-scheme: light)');

function load(): Theme {
  const v = localStorage.getItem(KEY);
  return v === 'dark' || v === 'light' || v === 'system' || v === 'fallout' ? v : 'system';
}

// Vault-Boy head favicon for the Fallout theme (inline so it swaps live).
const FALLOUT_FAVICON =
  'data:image/svg+xml,' +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">` +
      `<rect width="32" height="32" rx="6" fill="#04140a"/>` +
      `<g fill="none" stroke="#35e46a" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">` +
      `<path d="M8 12 Q14 4 24 9"/>` + // hair
      `<path d="M8 12 Q8 22 15 25 Q22 27 24 13"/>` + // head/jaw
      `<path d="M10.5 15 q2.5 2 5 0.5"/>` + // wink
      `<path d="M10 19 q6 5 12 -1"/>` + // grin
      `<path d="M11 20 H21"/></g>` + // teeth
      `<circle cx="20" cy="14" r="1.2" fill="#35e46a"/></svg>`,
  );

/** Resolve the preference to a concrete theme and apply it to the document. */
function apply(pref: Theme): void {
  const resolved = pref === 'system' ? (mq.matches ? 'light' : 'dark') : pref;
  document.documentElement.setAttribute('data-theme', resolved);
  const link = document.querySelector<HTMLLinkElement>('link#favicon');
  if (link) link.href = resolved === 'fallout' ? FALLOUT_FAVICON : './favicon.svg';
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
