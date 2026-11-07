// Measure only this project's Chrome tab. Node 22+ and an authorized ADB phone.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {execFileSync, spawn} = require('node:child_process');
const args = {};
for (let i = 2; i < process.argv.length; i++) {
  const key = process.argv[i].replace(/^--/, '');
  args[key] = process.argv[i + 1]?.startsWith('--') || i + 1 === process.argv.length ? true : process.argv[++i];
}
for (const key of ['origin', 'out', 'manifest', 'adb']) if (!args[key]) throw Error(`Missing --${key}`);
const out = path.resolve(args.out), origin = new URL(args.origin).origin;
if (fs.existsSync(out)) throw Error('Use a new output directory; existing evidence is preserved.');
fs.mkdirSync(out, {recursive:true});
const write = (name, value) => fs.writeFileSync(path.join(out,name), JSON.stringify(value,null,2)+'\n');
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const adb = (...parts) => execFileSync(args.adb, ['-d', ...parts], {encoding:'utf8',timeout:15000}).trim();
const sleep = ms => new Promise(resolve => setTimeout(resolve,ms));
const samples = fs.readFileSync(args.manifest,'utf8').trim().split(/\r?\n/).map(line=>JSON.parse(line));
for(const sample of samples)if(!/^[a-z0-9_-]+$/i.test(sample.id))throw Error('Invalid sample filename');
const mode = args.mode || 'eco', keepLoaded = Boolean(args['keep-loaded']);
let ws, monitor, counter=0, phase='setup';
const pending=new Map(), network=[], networkState={};
function call(method,params={},sessionId) {
  const id=++counter;
  return new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>{pending.delete(id);reject(Error(`${method} timed out`));},360000);
    pending.set(id,{resolve,reject,timeout});
    ws.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));
  });
}
async function evaluate(expression) {
  const result=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
  if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function waitReady() {
  for(let attempt=0;attempt<120;attempt++) {
    if(await evaluate('Boolean(window.__p3?.state.ready)'))return;
    await sleep(500);
  }
  throw Error('Cached app did not become ready.');
}
(async()=>{
  const tabs=await fetch(`http://127.0.0.1:${args.port||49228}/json/list`).then(r=>r.json());
  const target=tabs.find(t=>t.type==='page'&&(!args.tab||t.id===args.tab)&&new URL(t.url).origin===origin);
  if(!target)throw Error('Authorized project tab not found.');
  ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
  ws.addEventListener('message',event=>{
    const m=JSON.parse(event.data);
    if(m.id&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);clearTimeout(p.timeout);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);return;}
    if(m.method==='Target.attachedToTarget'&&m.params.targetInfo.type==='worker')call('Network.enable',{},m.params.sessionId).catch(()=>{});
    if(m.method?.startsWith('Network.')){
      const p=m.params;
      if(['Network.requestWillBeSent','Network.responseReceived','Network.loadingFailed'].includes(m.method)){
        const record={at:new Date().toISOString(),phase,session:m.sessionId||'page',event:m.method,requestId:p.requestId,
          ...(p.request?{url:p.request.url,method:p.request.method,type:p.type}:{}),
          ...(p.response?{url:p.response.url,status:p.response.status,fromServiceWorker:Boolean(p.response.fromServiceWorker),fromDiskCache:Boolean(p.response.fromDiskCache)}:{}),
          ...(p.errorText?{error:p.errorText}: {})};
        network.push(record);fs.appendFileSync(path.join(out,'network.jsonl'),JSON.stringify(record)+'\n');
      }
    }
  });
  await call('Page.bringToFront');await call('Network.enable');
  await call('Target.setAutoAttach',{autoAttach:true,waitForDebuggerOnStart:false,flatten:true});
  await waitReady();
  const identity=await evaluate(`(async()=>({origin:location.origin,userAgent:navigator.userAgent,deviceMemory:navigator.deviceMemory,
    shellCaches:(await caches.keys()).filter(x=>x.startsWith('p3-device-shell-')),
    modelIndex:await(await caches.match('/model-index.json')).json(),
    model:await(await caches.match('/models/manifest.json')).json(),
    source:Object.fromEntries(await Promise.all(['app.js','worker.js','numerics.js','tokenizer.js','vision.js','cache.js','sw.js'].filter(x=>x!=='sw.js').map(async file=>[file,[...new Uint8Array(await crypto.subtle.digest('SHA-256',await(await caches.match('/'+file)).arrayBuffer()))].map(x=>x.toString(16).padStart(2,'0')).join('')])))}))()`);
  const source={};
  for(const file of ['app.js','worker.js','numerics.js','tokenizer.js','vision.js','cache.js','sw.js'])source[file]=hash(fs.readFileSync(path.join('browser/dist',file)));
  for(const [file,sha] of Object.entries(identity.source))if(source[file]!==sha)throw Error(`Cached source differs from local build: ${file}`);
  write('experiment.json',{schema:1,started:new Date().toISOString(),identity,source,
    cdp:await call('Browser.getVersion'),device:{model:adb('shell','getprop','ro.product.model'),build:adb('shell','getprop','ro.build.fingerprint')},
    mode,keepLoaded,offline:Boolean(args.offline),manifest:samples,manifest_sha256:hash(fs.readFileSync(args.manifest)),
    memory_scope:'All Chrome package process PSS, including other tabs. GPU allocations are not fully observable.'});
  if(args.offline){
    networkState.wifi=adb('shell','settings','get','global','wifi_on');
    networkState.mobile=adb('shell','settings','get','global','mobile_data');
    if(!['0','1'].includes(networkState.wifi)||!['0','1'].includes(networkState.mobile))throw Error('Could not determine network settings; no changes made.');
    write('network-original.json',networkState);
    // Remove only this project's reverse tunnel, never other device forwards.
    const reverse=adb('reverse','--list');
    if(reverse.split('\n').some(line=>line.split(/\s+/)[1]==='tcp:7861'))adb('reverse','--remove','tcp:7861');
    adb('shell','svc','data','disable');adb('shell','svc','wifi','disable');
    await sleep(2500);
    const canary=await evaluate(`(async()=>{try{await fetch('/offline-network-canary-'+Date.now(),{cache:'no-store'});return {failed:false,online:navigator.onLine};}catch(e){return {failed:true,online:navigator.onLine,error:e.message};}})()`);
    write('offline-state.json',{at:new Date().toISOString(),wifi:adb('shell','settings','get','global','wifi_on'),mobile:adb('shell','settings','get','global','mobile_data'),reverse:adb('reverse','--list'),canary});
    if(!canary.failed||canary.online)throw Error('Network isolation was not established.');
    phase='offline-reload';await call('Page.reload',{ignoreCache:false});await sleep(1500);await waitReady();
    if(await evaluate('navigator.onLine'))throw Error('Phone unexpectedly online after reload.');
  }
  if(args.python){
    monitor=spawn(args.python,['scripts/monitor_phone.py','--adb',args.adb,'--out',path.join(out,'memory.json'),'--seconds','1800','--stop',path.join(out,'monitor.stop')],{stdio:['ignore','ignore','inherit']});
    await sleep(2000);
  }
  const queue=args.warmup?[{id:'warmup',seed:20261107,prompt:'A photograph of a red apple on a wooden table.'},...samples]:samples;
  for(const sample of queue){
    phase=`generation:${sample.id}`;
    const started=new Date().toISOString();
    const config={prompt:sample.prompt,seed:sample.seed,mode,keepLoaded};
    const result=await evaluate(`(async()=>{const r=await window.__p3.generate(${JSON.stringify(config)});const b64=b=>new Promise((ok,no)=>{const f=new FileReader();f.onload=()=>ok(f.result.split(',')[1]);f.onerror=no;f.readAsDataURL(b);});return {metadata:r.metadata,raw:await b64(r.raw),enhanced:await b64(r.enhanced),online:navigator.onLine};})()`);
    const artifacts={};
    for(const kind of ['raw','enhanced']){
      const bytes=Buffer.from(result[kind],'base64'),file=`${sample.id}-${kind}.png`;
      fs.writeFileSync(path.join(out,file),bytes);artifacts[kind]={file,sha256:hash(bytes),bytes:bytes.length};
    }
    const record={sample,warmup:sample.id==='warmup',started,finished:new Date().toISOString(),metadata:result.metadata,online:result.online,artifacts};
    write(`${sample.id}.json`,record);
    console.log(JSON.stringify({id:sample.id,visible_ms:result.metadata.submit_to_visible_ms,warm:result.metadata.warm_models,online:result.online,dispatches:result.metadata.gpu_dispatches}));
  }
  phase='complete';
  const screenshot=await call('Page.captureScreenshot',{format:'png'});
  fs.writeFileSync(path.join(out,'phone-browser.png'),Buffer.from(screenshot.data,'base64'));
  const generationRequests=network.filter(x=>x.phase.startsWith('generation:')&&x.event==='Network.requestWillBeSent'&&x.url?.startsWith('http'));
  write('completion.json',{at:new Date().toISOString(),complete:samples.length,network_requests_during_generation:generationRequests,
    inference_requests:generationRequests.filter(x=>x.method!=='GET'||new URL(x.url).pathname.startsWith('/api/')),
    offline:Boolean(args.offline),network_capture:'CDP page and auto-attached dedicated workers; no response bodies or other tabs captured.'});
})().catch(error=>{write('error.json',{at:new Date().toISOString(),error:error.stack});console.error(error);process.exitCode=1;}).finally(async()=>{
  if(monitor){fs.writeFileSync(path.join(out,'monitor.stop'),'done');await Promise.race([new Promise(r=>monitor.once('exit',r)),sleep(15000)]);}
  try{
    if(networkState.wifi==='1')adb('shell','svc','wifi','enable');
    if(networkState.mobile==='1')adb('shell','svc','data','enable');
    if(networkState.wifi!==undefined)write('network-restored.json',{wifi:adb('shell','settings','get','global','wifi_on'),mobile:adb('shell','settings','get','global','mobile_data')});
  }catch(error){console.error('Network restoration needs attention:',error.message);process.exitCode=1;}
  for(const p of pending.values())clearTimeout(p.timeout);
  ws?.close();
});
