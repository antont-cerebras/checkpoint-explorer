<script lang="ts">
  import { cachedStats } from '../stores/server';
  import { num } from '../lib/format';
  import Spinner from './Spinner.svelte';

  export let name: string;
  $: promise = cachedStats(name);
</script>

{#await promise}
  <Spinner label="scanning tensor…" />
{:then s}
  <table>
    <tbody>
      <tr><th>count</th><td class="mono">{s.count.toLocaleString()}</td></tr>
      <tr><th>min</th><td class="mono">{num(s.min)}</td></tr>
      <tr><th>max</th><td class="mono">{num(s.max)}</td></tr>
      <tr><th>mean</th><td class="mono">{num(s.mean)}</td></tr>
      <tr><th>std</th><td class="mono">{num(s.std)}</td></tr>
      <tr><th>zeros</th><td class="mono">{s.zeros.toLocaleString()} ({(s.zero_fraction * 100).toFixed(2)}%)</td></tr>
      <tr><th>non-finite</th><td class="mono">{s.nonfinite.toLocaleString()}</td></tr>
      <tr><th>scan</th><td class="dim mono">{s.elapsed_ms.toFixed(1)} ms</td></tr>
    </tbody>
  </table>
{:catch e}
  <p class="err">{e.message}</p>
{/await}

<style>
  table {
    border-collapse: collapse;
  }
  th {
    text-align: right;
    color: var(--fg-dim);
    font-weight: 400;
    padding: 2px 12px 2px 0;
  }
  .err {
    color: var(--danger);
  }
</style>
