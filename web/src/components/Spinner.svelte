<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  export let label = '';
  // The TUI's braille spinner frames (crates/core/src/progress.rs).
  const FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  let i = 0;
  let timer: ReturnType<typeof setInterval> | undefined;
  onMount(() => {
    timer = setInterval(() => (i = (i + 1) % FRAMES.length), 80);
  });
  onDestroy(() => clearInterval(timer));
</script>

<span class="spinner"><span class="frame">{FRAMES[i]}</span>{#if label}&nbsp;{label}{/if}</span>

<style>
  .spinner {
    color: var(--accent);
  }
  .frame {
    display: inline-block;
    width: 1ch;
    text-align: center;
  }
</style>
