// Human-readable formatting helpers, mirroring the Rust side's byte/param display.

export function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 2 : 1)} ${units[i]}`;
}

export function humanCount(n: number): string {
  if (n < 1000) return String(n);
  const units = ['K', 'M', 'B', 'T'];
  let v = n / 1000;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return `${v.toFixed(v < 10 ? 2 : 1)}${units[i]}`;
}

export function shape(dims: number[]): string {
  return dims.length ? dims.join(' × ') : 'scalar';
}

/** A compact number for grid cells / stats (trims noise, keeps precision). */
export function num(v: number): string {
  if (!Number.isFinite(v)) return v > 0 ? '+∞' : v < 0 ? '-∞' : 'NaN';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(3);
  return Number(v.toPrecision(6)).toString();
}
