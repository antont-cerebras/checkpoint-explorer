// Typed fetch wrappers over the Rust `--web` JSON API. The server owns the data;
// these just fetch it. Errors surface the server's `{error}` envelope.

import type {
  FileNode,
  HistogramDto,
  LayoutMap,
  SampleDto,
  StatsDto,
  TensorInfo,
  TreeResponse,
} from './types';

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const msg = (body && (body as { error?: string }).error) || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return body as T;
}

const enc = encodeURIComponent;

export interface SampleParams {
  mode?: 'grid' | 'window' | 'edges';
  rows?: number;
  cols?: number;
  slice?: number;
  dtype?: string;
  row_off?: number;
  col_off?: number;
  raw?: number;
}

function qs(params: Record<string, string | number | undefined>): string {
  return Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== '')
    .map(([k, v]) => `${k}=${enc(String(v))}`)
    .join('&');
}

export const api = {
  tree: () => getJson<TreeResponse>('/api/tree'),
  files: () => getJson<FileNode>('/api/files'),
  stats: () => getJson<Record<string, unknown>>('/api/stats'),
  health: () => getJson<unknown[]>('/api/health'),
  check: () => getJson<Record<string, unknown> | null>('/api/check'),
  tensor: (name: string) => getJson<TensorInfo>(`/api/tensor?name=${enc(name)}`),
  layout: (file: string) => getJson<LayoutMap>(`/api/layout?file=${enc(file)}`),
  file: (path: string) =>
    getJson<{ path: string; name: string; size: number; truncated: boolean; text: string }>(
      `/api/file?path=${enc(path)}`,
    ),
  tensorStats: (name: string, dtype?: string) =>
    getJson<StatsDto>(`/api/tensor/stats?${qs({ name, dtype })}`),
  sample: (name: string, p: SampleParams) =>
    getJson<SampleDto>(`/api/tensor/sample?${qs({ name, ...p })}`),
  histogram: (name: string, bins?: number, dtype?: string) =>
    getJson<HistogramDto>(`/api/tensor/histogram?${qs({ name, bins, dtype })}`),
};
