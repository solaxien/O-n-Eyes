// Development-only USB cache migration. Transfers verified model bytes, never
// prompts, generated images or inference results. Not used by the public app.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const [origin,tabId,modelDir,output,port='49228']=process.argv.slice(2);
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
(async()=>{
 const tabs=await fetch(`http://127.0.0.1:${port}/json/list`).then(r=>r.json());
 const tab=tabs.find(t=>t.id===tabId&&new URL(t.url).origin===new URL(origin).origin);
 if(!tab)throw Error('Project tab not found');
 const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((yes,no)=>{ws.onopen=yes;ws.onerror=no;});
 let serial=0;const pending=new Map();
 ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const {yes,no,timer}=pending.get(m.id);clearTimeout(timer);pending.delete(m.id);m.error||m.result.exceptionDetails?no(Error(JSON.stringify(m.error||m.result.exceptionDetails))):yes(m.result.result.value);}};
 const evaluate=expression=>new Promise((yes,no)=>{const id=++serial,timer=setTimeout(()=>no(Error('Cache transfer timed out')),120000);pending.set(id,{yes,no,timer});ws.send(JSON.stringify({id,method:'Runtime.evaluate',params:{expression,awaitPromise:true,returnByValue:true}}));});
 try{
  const manifestBytes=fs.readFileSync(path.join(modelDir,'manifest.json')),manifest=JSON.parse(manifestBytes);
  const expected=await evaluate(`(async()=>{await navigator.serviceWorker.ready;return (await (await fetch('/model-index.json')).json()).manifest_sha256;})()`);
  if(expected!==sha(manifestBytes))throw Error('Published model identity differs from local bundle');
  const files=[{path:'manifest.json',sha256:sha(manifestBytes)},...Object.values(manifest.components).flatMap(c=>c.files),...['vocab.json','merges.txt','schedules.json','tokenizer-tests.json'].map(name=>({path:name}))];
  const evidence={origin,started:new Date().toISOString(),transport:'ADB-forwarded Chrome DevTools; static model bytes only, to avoid a second cellular model download.',files:[]};
  for(const file of files){
   const bytes=fs.readFileSync(path.join(modelDir,file.path)),digest=sha(bytes);
   if(file.sha256&&file.sha256!==digest)throw Error('Local artifact checksum mismatch');
   const present=await evaluate(`(async()=>{const cache=await caches.open('p3-model-fp16-v1'),file=await cache.match('/models/'+${JSON.stringify(file.path)});return file?.headers.get('x-model-sha256')===${JSON.stringify(digest)};})()`);
   if(present){evidence.files.push({path:file.path,bytes:bytes.length,sha256:digest,already_cached:true});continue;}
   await evaluate(`window.__p3AssetTransfer=new Uint8Array(${bytes.length});true`);
   for(let offset=0;offset<bytes.length;offset+=2*1024*1024){const chunk=bytes.subarray(offset,offset+2*1024*1024).toString('base64');await evaluate(`(()=>{const text=atob(${JSON.stringify(chunk)}),offset=${offset};for(let i=0;i<text.length;i++)window.__p3AssetTransfer[offset+i]=text.charCodeAt(i);return true;})()`);}
   const verified=await evaluate(`(async()=>{const bytes=window.__p3AssetTransfer,hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');if(hash!==${JSON.stringify(digest)})throw Error('Device checksum mismatch');const cache=await caches.open('p3-model-fp16-v1');await cache.put('/models/'+${JSON.stringify(file.path)},new Response(bytes,{headers:{'Content-Type':${JSON.stringify(file.path.endsWith('.json')?'application/json':'application/octet-stream')},'x-model-sha256':hash}}));delete window.__p3AssetTransfer;return hash;})()`);
   evidence.files.push({path:file.path,bytes:bytes.length,sha256:verified});fs.writeFileSync(output,JSON.stringify(evidence,null,2));console.log(JSON.stringify({file:file.path,bytes:bytes.length,verified:true}));
  }
  evidence.completed=new Date().toISOString();fs.writeFileSync(output,JSON.stringify(evidence,null,2));
  await evaluate('window.__p3.prepare()');
 }finally{for(const p of pending.values())clearTimeout(p.timer);ws.close();}
})().catch(e=>{console.error(e);process.exit(1);});
