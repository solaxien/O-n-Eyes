const SHELL = "p3-device-shell-v1";
self.addEventListener("install", (event) =>
  event.waitUntil(
    (async () => {
      const files = await fetch("/shell.json", { cache: "no-store" }).then(
        (r) => r.json(),
      );
      const cache = await caches.open(SHELL);
      await cache.addAll(
        [...files, "/shell.json"].map(
          (path) => new Request(path, { cache: "reload" }),
        ),
      );
      await self.skipWaiting();
    })(),
  ),
);
self.addEventListener("activate", (event) =>
  event.waitUntil(
    (async () => {
      for (const key of await caches.keys())
        if (key.startsWith("p3-device-shell-") && key !== SHELL)
          await caches.delete(key);
      await self.clients.claim();
    })(),
  ),
);
self.addEventListener("fetch", (event) => {
  const request = event.request,
    url = new URL(request.url);
  // This app has no inference API and never forwards generation requests.
  if (request.method !== "GET" || url.pathname.startsWith("/api/")) {
    event.respondWith(
      new Response("No server inference in this application.", { status: 405 }),
    );
    return;
  }
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/models/")) return; // Explicit checksum-verified model cache handles weights.
  event.respondWith(
    (async () => {
      const cache = await caches.open(SHELL);
      const cached = await cache.match(url.pathname);
      if (cached) return cached;
      return fetch(request);
    })(),
  );
});
