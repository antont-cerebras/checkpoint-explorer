<script lang="ts">
  import { filterByDim, filterByShape } from '../stores/view';
  export let shape: number[];
</script>

{#if shape.length === 0}
  <button
    class="brk"
    title="Filter tensors with this shape"
    on:click|stopPropagation={() => filterByShape(shape)}>()</button
  >
{:else}
  <span class="shape">
    <button class="brk" title="Filter by this exact shape" on:click|stopPropagation={() => filterByShape(shape)}>(</button>
    {#each shape as d, i}
      <button class="dim" title="Filter tensors with a dimension of {d}" on:click|stopPropagation={() => filterByDim(d)}>{d}</button>{#if i < shape.length - 1}<span class="x">,&nbsp;</span>{:else if shape.length === 1}<span class="x">,</span>{/if}
    {/each}
    <button class="brk" title="Filter by this exact shape" on:click|stopPropagation={() => filterByShape(shape)}>)</button>
  </span>
{/if}

<style>
  .shape {
    display: inline-flex;
    align-items: center;
    gap: 1px;
    font-family: ui-monospace, monospace;
  }
  .dim,
  .brk {
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
    padding: 0 2px;
    border-radius: 3px;
    font: inherit;
  }
  .dim:hover,
  .brk:hover {
    background: var(--bg-hover);
    color: var(--accent);
  }
  .brk {
    color: var(--fg-dim);
  }
  .x {
    color: var(--fg-dim);
  }
</style>
