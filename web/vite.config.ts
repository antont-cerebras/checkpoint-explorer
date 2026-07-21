import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// Plain SPA built to `dist/`, embedded into the Rust binary (rust-embed). Relative
// `base` so the assets load regardless of the mount path. In dev, proxy `/api` to
// a running `checkpoint-explorer --web` instance (default port 8080).
export default defineConfig({
  plugins: [svelte()],
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2020',
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8080',
    },
  },
});
