<!-- A small phosphor-green wink + thumbs-up, à la Vault Boy — shown only in the
     Fallout theme, just for fun. Line art in currentColor (the theme accent). -->
<svg class="vaultboy" viewBox="0 0 64 72" fill="none" stroke="currentColor" aria-hidden="true">
  <!-- hair tuft -->
  <path d="M20 15 Q26 4 40 9 Q47 11 46 19" stroke-width="2" stroke-linecap="round" />
  <!-- head -->
  <circle cx="32" cy="27" r="16" stroke-width="2" />
  <!-- winking left eye -->
  <path d="M23 26 q4 4 8 0" stroke-width="2" stroke-linecap="round" />
  <!-- open right eye -->
  <circle cx="40" cy="25" r="1.8" fill="currentColor" stroke="none" />
  <!-- big grin -->
  <path d="M22 33 q10 11 20 0" stroke-width="2" stroke-linecap="round" />
  <!-- thumbs-up fist -->
  <rect x="28" y="50" width="13" height="12" rx="3" stroke-width="2" />
  <path d="M41 54 q6 -1 6 -6 q0 -3 -3 -3 l-3 1" stroke-width="2" stroke-linejoin="round" />
</svg>

<style>
  .vaultboy {
    position: fixed;
    right: 14px;
    bottom: 52px;
    width: 66px;
    height: 74px;
    color: var(--accent);
    opacity: 0.55;
    pointer-events: none;
    z-index: 5;
    filter: drop-shadow(0 0 3px currentColor);
  }
</style>
