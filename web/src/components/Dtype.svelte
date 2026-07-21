<script lang="ts">
  import { dtypeInfo } from '../lib/dtype';
  import { filterByDtype } from '../stores/view';

  export let dtype: string;
  /** Show the styled hover bubble (off in tight/overflow-clipped rows like the tree,
   * where the native title is used instead). */
  export let bubble = true;
</script>

<button
  class="dtype"
  class:has-bubble={bubble}
  title={dtypeInfo(dtype)}
  on:click|stopPropagation={() => filterByDtype(dtype)}
>
  {dtype}
  {#if bubble}<span class="tip">{dtypeInfo(dtype)}</span>{/if}
</button>

<style>
  .dtype {
    position: relative;
    display: inline-block;
    font-size: 11px;
    color: var(--dtype);
    background: color-mix(in srgb, var(--dtype) 16%, transparent);
    border: 1px solid color-mix(in srgb, var(--dtype) 30%, transparent);
    border-radius: 4px;
    padding: 0 6px;
    line-height: 16px;
    cursor: pointer;
    font-family: inherit;
  }
  .dtype:hover {
    background: color-mix(in srgb, var(--dtype) 28%, transparent);
  }
  .tip {
    display: none;
    position: absolute;
    bottom: calc(100% + 6px);
    left: 0;
    width: 250px;
    padding: 8px 10px;
    background: var(--bg-elev);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 6px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    font-size: 12px;
    line-height: 1.45;
    white-space: normal;
    text-align: left;
    z-index: 20;
  }
  .has-bubble:hover .tip,
  .has-bubble:focus .tip {
    display: block;
  }
</style>
