<!-- A stylized winking, thumbs-up vault dweller — original phosphor line-art (the
     real Vault Boy is Bethesda's, so this is just an homage), shown only in the
     Fallout theme, for fun. Drawn in currentColor (the theme accent). -->
<svg class="vaultboy" viewBox="0 0 80 116" fill="none" stroke="currentColor" aria-hidden="true">
  <g stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <!-- swept hair + fringe -->
    <path d="M23 27 Q19 11 36 8 Q54 6 58 23" />
    <path d="M31 11 30 20" />
    <path d="M39 9 38 19" />
    <path d="M47 10 48 20" />
    <!-- head + jaw -->
    <path d="M23 27 Q22 42 31 48 Q40 53 48 49 Q57 44 58 27" />
    <!-- ear -->
    <path d="M22 29 q-4 1 -3 6 q1 3 4 2" />
    <!-- brows -->
    <path d="M28 24 q4 -2 8 0" />
    <path d="M44 24 q4 -2 7 1" />
    <!-- winking eye (left) -->
    <path d="M29 30 q3 3 7 1" />
    <!-- open eye (right) -->
    <ellipse cx="47" cy="28" rx="2.6" ry="3.2" />
    <circle cx="47" cy="28.6" r="1" fill="currentColor" stroke="none" />
    <!-- nose -->
    <path d="M40 30 q-1 5 -3 7 q2 1 4 0" />
    <!-- big grin with teeth -->
    <path d="M30 40 Q40 51 50 39" />
    <path d="M33 41 H47" />
    <!-- neck -->
    <path d="M36 49 36 57" />
    <path d="M45 49 45 57" />
    <!-- jumpsuit shoulders + collar + zipper -->
    <path d="M25 64 Q40 56 56 64 L60 104 Q40 112 20 104 Z" />
    <path d="M34 59 40 65 46 59" />
    <path d="M40 65 40 100" />
    <!-- raised thumbs-up fist -->
    <rect x="30" y="74" width="16" height="14" rx="3" />
    <path d="M32 78 32 84 M37 78 37 84 M42 78 42 84" />
    <path d="M46 79 q9 -2 9 -11 q0 -4 -4 -4 l-5 2" />
  </g>
</svg>

<style>
  .vaultboy {
    position: fixed;
    right: 14px;
    bottom: 50px;
    width: 78px;
    height: 112px;
    color: var(--accent);
    opacity: 0.6;
    pointer-events: none;
    z-index: 5;
    filter: drop-shadow(0 0 3px currentColor);
  }
</style>
