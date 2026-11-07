// Package this same browser app as a static-only Cloudflare Worker.
// Large model/runtime binaries are streamed from an immutable GitHub release.
import { readFile, writeFile, mkdir, readdir } from "node:fs/promises";
import { resolve, join } from "node:path";
import { createHash } from "node:crypto";

const destination = resolve(process.argv[2] || ".runs/browser-site");
const hosting = JSON.parse(
  await readFile(join(destination, ".openai/hosting.json"), "utf8"),
);
if (!hosting.project_id)
  throw Error("Persist the registered Sites identity first.");
const web = resolve("browser/dist"),
  models = resolve(process.env.P3_BROWSER_MODELS || ".models/browser/v3");
const release =
  "https://github.com/solaxien/O-n-Eyes/releases/download/browser-models-v3/";
const shell = JSON.parse(await readFile(join(web, "shell.json"), "utf8"));
const embedded = {},
  remote = {};
const type = (name) =>
  name.endsWith(".mjs") || name.endsWith(".js")
    ? "text/javascript; charset=utf-8"
    : name.endsWith(".html")
      ? "text/html; charset=utf-8"
      : name.endsWith(".css")
        ? "text/css; charset=utf-8"
        : name.endsWith(".json")
          ? "application/json"
          : name.endsWith(".wasm")
            ? "application/wasm"
            : "text/plain; charset=utf-8";
for (const asset of new Set([...shell, "/sw.js", "/shell.json"])) {
  if (asset === "/") continue;
  if (asset.endsWith(".wasm"))
    remote[asset] = { file: asset.split("/").at(-1), type: type(asset) };
  else
    embedded[asset] = {
      body: await readFile(join(web, asset.slice(1)), "utf8"),
      type: type(asset),
    };
}
embedded["/"] = embedded["/index.html"];
const manifest = JSON.parse(
  await readFile(join(models, "manifest.json"), "utf8"),
);
for (const file of Object.values(manifest.components).flatMap((c) => c.files))
  remote["/models/" + file.path] = {
    file: file.path,
    type: "application/octet-stream",
    sha256: file.sha256,
  };
for (const file of [
  "manifest.json",
  "vocab.json",
  "merges.txt",
  "schedules.json",
  "tokenizer-tests.json",
])
  embedded["/models/" + file] = {
    body: await readFile(join(models, file), "utf8"),
    type: type(file),
  };
const source = `// Static asset delivery only. There is no inference handler, prompt log or GPU host.
const embedded=${JSON.stringify(embedded)};
const remote=${JSON.stringify(remote)};
const release=${JSON.stringify(release)};
const common={'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp','Cross-Origin-Resource-Policy':'same-origin','X-Content-Type-Options':'nosniff'};
export default {async fetch(request){
  const url=new URL(request.url),path=url.pathname;
  if(!['GET','HEAD'].includes(request.method)||path.startsWith('/api/'))return new Response('No server inference exists.',{status:405,headers:common});
  if(embedded[path])return new Response(request.method==='HEAD'?null:embedded[path].body,{headers:{...common,'Content-Type':embedded[path].type,'Cache-Control':'no-cache'}});
  if(!remote[path])return new Response('Not found',{status:404,headers:common});
  const asset=remote[path],headers={};
  if(request.headers.has('Range'))headers.Range=request.headers.get('Range');
  const response=await fetch(release+asset.file,{method:request.method,headers,redirect:'follow'});
  if(!response.ok)return new Response('Static asset temporarily unavailable. Retry preparation.',{status:502,headers:common});
  const resultHeaders=new Headers({...common,'Content-Type':asset.type,'Cache-Control':'public,max-age=31536000,immutable'});
  for(const key of ['Content-Length','Content-Range','Accept-Ranges'])if(response.headers.has(key))resultHeaders.set(key,response.headers.get(key));
  if(asset.sha256)resultHeaders.set('X-Model-SHA256',asset.sha256);
  return new Response(response.body,{status:response.status,headers:resultHeaders});
}};
`;
await mkdir(join(destination, "worker"), { recursive: true });
await mkdir(join(destination, "dist/server"), { recursive: true });
await mkdir(join(destination, "dist/.openai"), { recursive: true });
await writeFile(join(destination, "worker/index.js"), source);
await writeFile(join(destination, "dist/server/index.js"), source);
await writeFile(
  join(destination, "dist/.openai/hosting.json"),
  JSON.stringify(hosting, null, 2) + "\n",
);
await writeFile(
  join(destination, "package.json"),
  JSON.stringify(
    {
      name: "on-eyes-browser-host",
      private: true,
      type: "module",
      scripts: { build: "node build.mjs" },
    },
    null,
    2,
  ) + "\n",
);
await writeFile(
  join(destination, "build.mjs"),
  `import {mkdir,copyFile} from 'node:fs/promises';\nawait mkdir('dist/server',{recursive:true});await mkdir('dist/.openai',{recursive:true});await copyFile('worker/index.js','dist/server/index.js');await copyFile('.openai/hosting.json','dist/.openai/hosting.json');\n`,
);
await writeFile(
  join(destination, ".gitignore"),
  "node_modules/\ndist/\n.openai/*\n!.openai/hosting.json\n",
);
console.log(
  JSON.stringify({
    destination,
    embedded: Object.keys(embedded).length,
    remote: Object.keys(remote).length,
    worker_bytes: Buffer.byteLength(source),
    sha256: createHash("sha256").update(source).digest("hex"),
  }),
);
