<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import { humanCount } from '../lib/format';
  import Spinner from './Spinner.svelte';

  interface Finding {
    severity: string;
    subject: string;
    message: string;
  }
  interface Check {
    id: string;
    title: string;
    status: string;
    findings: Finding[];
  }
  interface CheckReport {
    summary: { files: number; tensors: number; params: number; errors: number; warnings: number };
    checks: Check[];
    healthy: boolean;
  }
  interface Health {
    index_path: string;
    missing_files: string[];
    extra_files: string[];
    missing_tensors: string[];
    extra_tensors: string[];
  }

  let check: CheckReport | null = null;
  let health: Health[] = [];
  let err = '';
  let loading = true;

  onMount(async () => {
    try {
      const [c, h] = await Promise.all([api.check(), api.health()]);
      check = c as unknown as CheckReport;
      health = (h as unknown as Health[]) ?? [];
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  });

  const STATUS: Record<string, { icon: string; cls: string; label: string }> = {
    pass: { icon: '✓', cls: 'ok', label: 'pass' },
    warn: { icon: '⚠', cls: 'warn', label: 'warning' },
    fail: { icon: '✗', cls: 'fail', label: 'fail' },
    na: { icon: '–', cls: 'na', label: 'n/a' },
  };
  const SEV: Record<string, string> = { error: 'fail', warning: 'warn', info: 'na' };

  // Findings-first: checks with problems on top, then passing, then n/a.
  const rank = (s: string) => (s === 'fail' ? 0 : s === 'warn' ? 1 : s === 'pass' ? 2 : 3);
  $: checks = check ? [...check.checks].sort((a, b) => rank(a.status) - rank(b.status)) : [];
  $: indexIssues = health.filter(
    (h) =>
      h.missing_files.length || h.extra_files.length || h.missing_tensors.length || h.extra_tensors.length,
  );
</script>

<div class="health">
  {#if loading}
    <Spinner label="running checks…" />
  {:else if err}
    <p class="err">{err}</p>
  {:else if check}
    <!-- overall banner -->
    <div class="banner {check.healthy ? (check.summary.warnings ? 'warn' : 'ok') : 'fail'}">
      <span class="big">{check.healthy ? (check.summary.warnings ? '⚠' : '✓') : '✗'}</span>
      <span class="msg">
        {#if !check.healthy}{check.summary.errors} error{check.summary.errors === 1 ? '' : 's'}
        {:else if check.summary.warnings}{check.summary.warnings} warning{check.summary.warnings === 1 ? '' : 's'} — no errors
        {:else}Healthy — all checks passed{/if}
      </span>
      <span class="sub dim">
        {check.summary.files} files · {check.summary.tensors.toLocaleString()} tensors · {humanCount(check.summary.params)} params
      </span>
    </div>

    <!-- structural checks -->
    <section>
      <h3>Structural checks</h3>
      <ul class="checks">
        {#each checks as c}
          {@const st = STATUS[c.status] ?? STATUS.na}
          <li>
            <div class="checkhead">
              <span class="badge {st.cls}" title={st.label}>{st.icon}</span>
              <span class="ctitle">{c.title}</span>
              {#if c.findings.length}<span class="dim">· {c.findings.length}</span>{/if}
            </div>
            {#if c.findings.length}
              <ul class="findings">
                {#each c.findings as f}
                  <li>
                    <span class="badge sm {SEV[f.severity] ?? 'na'}">{(STATUS[SEV[f.severity]] ?? STATUS.na).icon}</span>
                    <span class="subject">{f.subject}</span>
                    <span class="fmsg dim">{f.message}</span>
                  </li>
                {/each}
              </ul>
            {/if}
          </li>
        {/each}
      </ul>
    </section>

    <!-- index reconciliation -->
    <section>
      <h3>Index health</h3>
      {#if !health.length}
        <p class="dim">No <code>model.safetensors.index.json</code> to reconcile.</p>
      {:else if !indexIssues.length}
        <p class="ok-line"><span class="badge ok">✓</span> Index matches the files on disk.</p>
      {:else}
        {#each health as h}
          {@const lists = [
            ['Missing files', h.missing_files, 'fail'],
            ['Extra files (on disk, not in index)', h.extra_files, 'warn'],
            ['Missing tensors', h.missing_tensors, 'fail'],
            ['Extra tensors', h.extra_tensors, 'warn'],
          ]}
          {#each lists as [heading, items, cls]}
            {#if items.length}
              <div class="idxgroup">
                <div class="idxhead"><span class="badge sm {cls}">{cls === 'fail' ? '✗' : '⚠'}</span> {heading} <span class="dim">({items.length})</span></div>
                <ul class="idxlist">
                  {#each items as it}<li class="mono">{it}</li>{/each}
                </ul>
              </div>
            {/if}
          {/each}
        {/each}
      {/if}
    </section>
  {/if}
</div>

<style>
  .health {
    height: 100%;
    overflow: auto;
    padding: 18px 22px;
    max-width: 900px;
  }
  .banner {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    border: 1px solid var(--border);
    border-left-width: 4px;
    border-radius: 8px;
    background: var(--bg-panel);
    margin-bottom: 22px;
  }
  .banner.ok {
    border-left-color: var(--ok);
  }
  .banner.warn {
    border-left-color: var(--warn);
  }
  .banner.fail {
    border-left-color: var(--danger);
  }
  .banner .big {
    font-size: 26px;
  }
  .banner.ok .big {
    color: var(--ok);
  }
  .banner.warn .big {
    color: var(--warn);
  }
  .banner.fail .big {
    color: var(--danger);
  }
  .banner .msg {
    font-size: 15px;
  }
  .banner .sub {
    margin-left: auto;
    font-size: 12px;
  }
  h3 {
    margin: 0 0 10px;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--fg-dim);
  }
  section {
    margin-bottom: 26px;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .checks > li {
    padding: 8px 0;
    border-top: 1px solid var(--border);
  }
  .checkhead {
    display: flex;
    align-items: center;
    gap: 9px;
  }
  .ctitle {
    color: var(--fg);
  }
  .badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 4px;
    font-size: 12px;
    flex: 0 0 auto;
  }
  .badge.sm {
    width: 15px;
    height: 15px;
    font-size: 10px;
  }
  .badge.ok {
    background: color-mix(in srgb, var(--ok) 22%, transparent);
    color: var(--ok);
  }
  .badge.warn {
    background: color-mix(in srgb, var(--warn) 22%, transparent);
    color: var(--warn);
  }
  .badge.fail {
    background: color-mix(in srgb, var(--danger) 22%, transparent);
    color: var(--danger);
  }
  .badge.na {
    background: var(--bg-hover);
    color: var(--fg-dim);
  }
  .findings {
    margin: 6px 0 2px 27px;
  }
  .findings li {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 2px 0;
  }
  .subject {
    color: var(--warn);
  }
  .fmsg {
    font-size: 12px;
  }
  .idxgroup {
    margin-bottom: 12px;
  }
  .idxhead {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }
  .idxlist {
    margin-left: 24px;
  }
  .idxlist li {
    padding: 1px 0;
    color: var(--fg-dim);
  }
  .ok-line {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .err {
    color: var(--danger);
  }
</style>
