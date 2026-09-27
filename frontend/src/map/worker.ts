import { setWorkerUrl } from 'maplibre-gl'

// The production build emits MapLibre's worker under /maplibre/ (vite.config.ts).
// In development MapLibre is served from node_modules and finds it itself.
if (import.meta.env.PROD) {
  setWorkerUrl(new URL('/maplibre/maplibre-gl-worker.mjs', window.location.origin).href)
}
