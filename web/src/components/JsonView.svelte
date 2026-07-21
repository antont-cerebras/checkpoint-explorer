<script lang="ts">
  // A small recursive renderer for the report JSON (stats/check/health) so the web
  // shows the same info as the TUI without hardcoding every field: arrays of like
  // objects become tables, objects become key/value lists, scalars are formatted.
  export let value: unknown;

  function isTable(v: unknown): v is Record<string, unknown>[] {
    return (
      Array.isArray(v) &&
      v.length > 0 &&
      v.every((x) => x !== null && typeof x === 'object' && !Array.isArray(x))
    );
  }

  function columns(arr: Record<string, unknown>[]): string[] {
    const set = new Set<string>();
    for (const o of arr) for (const k of Object.keys(o)) set.add(k);
    return [...set];
  }

  function isScalar(v: unknown): boolean {
    return v === null || typeof v !== 'object';
  }

  function fmt(v: unknown): string {
    if (v === null) return '—';
    if (typeof v === 'number') {
      if (Number.isInteger(v)) return v.toLocaleString();
      const a = Math.abs(v);
      return a >= 1e6 || a < 1e-4 ? v.toExponential(3) : String(v);
    }
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    return String(v);
  }
</script>

{#if isTable(value)}
  <table>
    <thead>
      <tr>{#each columns(value) as c}<th>{c}</th>{/each}</tr>
    </thead>
    <tbody>
      {#each value as row}
        <tr>
          {#each columns(value) as c}
            <td>
              {#if isScalar(row[c])}<span class="mono">{fmt(row[c])}</span>
              {:else}<svelte:self value={row[c]} />{/if}
            </td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
{:else if Array.isArray(value)}
  {#if value.length === 0}
    <span class="dim">(none)</span>
  {:else}
    <ul>
      {#each value as item}<li><svelte:self value={item} /></li>{/each}
    </ul>
  {/if}
{:else if value !== null && typeof value === 'object'}
  <dl>
    {#each Object.entries(value) as [k, v]}
      <dt>{k}</dt>
      <dd>
        {#if isScalar(v)}<span class="mono">{fmt(v)}</span>{:else}<svelte:self value={v} />{/if}
      </dd>
    {/each}
  </dl>
{:else}
  <span class="mono">{fmt(value)}</span>
{/if}

<style>
  dl {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 2px 14px;
    margin: 0;
  }
  dt {
    color: var(--fg-dim);
    text-align: right;
  }
  dd {
    margin: 0;
    min-width: 0;
  }
  ul {
    margin: 0;
    padding-left: 16px;
  }
  table {
    border-collapse: collapse;
    margin: 2px 0;
  }
  th {
    text-align: left;
    color: var(--fg-dim);
    font-weight: 400;
    border-bottom: 1px solid var(--border);
    padding: 2px 10px 2px 0;
  }
  td {
    padding: 2px 10px 2px 0;
    vertical-align: top;
  }
</style>
