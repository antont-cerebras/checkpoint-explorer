<script lang="ts">
  import TreeView from './TreeView.svelte';
  import Detail from './Detail.svelte';
</script>

<div class="split">
  <div class="left"><TreeView /></div>
  <div class="right"><Detail /></div>
</div>

<style>
  .split {
    display: flex;
    height: 100%;
  }
  .left {
    width: 46%;
    min-width: 300px;
    max-width: 620px;
    border-right: 1px solid var(--border);
    height: 100%;
  }
  .right {
    flex: 1 1 auto;
    min-width: 0;
    height: 100%;
    overflow: auto;
  }
</style>
