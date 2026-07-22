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

// Weathered radiation-trefoil favicon for the Fallout theme (the trefoil is a
// public-domain safety symbol; this is a stylized, paint-chipped rendering). Inline
// so it swaps live.
const BLADE = 'M2 -3.46 L6.5 -11.26 A13 13 0 0 0 -6.5 -11.26 L-2 -3.46 A4 4 0 0 1 2 -3.46 Z';
const FALLOUT_FAVICON =
  'data:image/svg+xml,' +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">` +
      `<rect width="32" height="32" rx="6" fill="#04140a"/>` +
      `<g transform="translate(16,16)" fill="#35e46a">` +
      `<path d="${BLADE}"/>` +
      `<path transform="rotate(120)" d="${BLADE}"/>` +
      `<path transform="rotate(240)" d="${BLADE}"/>` +
      `<circle r="2.6"/></g>` +
      // chipped/worn paint: dark specks eating into the green
      `<g fill="#04140a">` +
      `<circle cx="15" cy="5.5" r="1.1"/>` +
      `<circle cx="23.5" cy="21.5" r="1.3"/>` +
      `<circle cx="9" cy="22.5" r="1"/>` +
      `<path d="M19.5 8 l2.2 0.6 -1 2 z"/>` +
      `<path d="M11 18 l-1.6 1.2 0.4 1.6 z"/></g></svg>`,
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
