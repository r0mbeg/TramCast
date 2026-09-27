/// <reference types="vitest/config" />
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

const require = createRequire(import.meta.url)

// MapLibre 6 loads its worker next to its own module, which bundling breaks.
// The build emits the worker with its shared chunk under maplibre/, and the
// app points MapLibre there with setWorkerUrl (see src/map/worker.ts).
function maplibreWorker(): Plugin {
  const dist = join(dirname(require.resolve('maplibre-gl/package.json')), 'dist')
  return {
    name: 'tramcast-maplibre-worker',
    apply: 'build',
    generateBundle() {
      for (const file of ['maplibre-gl-worker.mjs', 'maplibre-gl-shared.mjs']) {
        this.emitFile({ type: 'asset', fileName: `maplibre/${file}`, source: readFileSync(join(dist, file)) })
      }
    },
  }
}

export default defineConfig({
  plugins: [react(), maplibreWorker()],
  // Served from node_modules in development, so the worker resolves as shipped.
  optimizeDeps: { exclude: ['maplibre-gl'] },
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8080' },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    chunkSizeWarningLimit: 1500,
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
