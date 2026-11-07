import {getAsset,prepareModels,MODEL_CACHE} from './cache.js';
const $=id=>document.getElementById(id),state={ready:false,busy:false,worker:null,urls:{},result:null};
const fail=message=>{$('error').textContent=message;$('error').hidden=false;};
let adapter;
try{
  if(!navigator.gpu)throw Error('WebGPU is unavailable in this browser. No server fallback is used.');
  adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  if(!adapter||!adapter.features.has('shader-f16'))throw Error('This browser needs a WebGPU adapter with FP16 support. No server fallback is used.');
  $('capability').textContent=`${adapter.info.vendor||'Device'} GPU available · ${navigator.deviceMemory||'unknown'} GB memory class. Models are checked before use.`;
  $('download-status').textContent='Preparing the offline browser runtime (about 29 MB on first use)…';
  if('serviceWorker' in navigator){await navigator.serviceWorker.register('/sw.js');await navigator.serviceWorker.ready;}else throw Error('Offline storage requires a browser with service-worker support.');
  $('prepare').disabled=false;$('download-status').textContent='Ready to download about 896 MB of model files. Completed files are reused after interruption.';
  try{const index=await fetch('/model-index.json').then(r=>r.json()),manifest=await(await getAsset('manifest.json',{sha256:index.manifest_sha256})).json(),cache=await caches.open(MODEL_CACHE);let complete=true;
    for(const file of Object.values(manifest.components).flatMap(c=>c.files)){const r=await cache.match('/models/'+file.path);if(!r||r.headers.get('x-model-sha256')!==file.sha256){complete=false;break;}}
    if(complete&&Object.keys(manifest.components).length===3){state.ready=true;$('generate').disabled=false;$('stage').textContent='Models cached. Ready for local generation.';$('download-status').textContent='Model files are cached on this device. Generation can run offline.';}
  }catch{}
}catch(error){$('capability').textContent=error.message;fail(error.message);}

$('prepare').onclick=async()=>{
  clearTimeout(state.idleTimer);state.worker?.terminate();state.worker=null;state.ready=false;
  $('error').hidden=true;$('prepare').disabled=true;$('generate').disabled=true;$('download-progress').hidden=false;$('cancel').hidden=false;
  const controller=new AbortController();state.controller=controller;
  try{
    await prepareModels(({file,bytes,total})=>{$('download-progress').value=bytes/total*100;$('download-status').textContent=`${(bytes/1e6).toFixed(0)} / ${(total/1e6).toFixed(0)} MB · ${file}`;},controller.signal);
    const persistent=await navigator.storage.persist().catch(()=>false);
    state.ready=true;$('generate').disabled=false;$('stage').textContent='Ready. Generation runs entirely in this browser.';$('download-status').textContent=`Model files are cached and verified. ${persistent?'Persistent storage granted.':'The browser may evict cached files when storage is low.'}`;
  }catch(error){fail(controller.signal.aborted?'Download paused. Completed files are retained; press Download to continue.':error.message);}
  finally{$('prepare').disabled=false;$('cancel').hidden=true;state.controller=null;}
};
$('cancel').onclick=()=>state.controller?.abort();

function showResult(result,visibleMs){
  for(const url of Object.values(state.urls))URL.revokeObjectURL(url);
  state.urls={raw:URL.createObjectURL(result.raw),enhanced:URL.createObjectURL(result.enhanced)};state.result=result;
  $('image').src=state.urls.enhanced;$('image').alt=result.metadata.prompt;$('image').hidden=false;$('placeholder').hidden=true;$('toggle').hidden=false;
  for(const button of document.querySelectorAll('[data-image]'))button.setAttribute('aria-pressed',String(button.dataset.image==='enhanced'));
  $('save').href=state.urls.enhanced;$('save').hidden=false;$('details').hidden=false;
  result.metadata.submit_to_result_ms=visibleMs;$('metadata').textContent=JSON.stringify(result.metadata,null,2);
  $('timing').textContent=`${(visibleMs/1000).toFixed(2)} s on this device · ${result.metadata.native_resolution} px native → 512 px display`;
  $('stage').textContent=result.metadata.cv.usable?'Generated locally. Pixel quality checks passed.':'Generated locally, but pixel checks flagged this image. Raw output is retained.';
}
async function generate({prompt,seed,mode,omit=[],keepLoaded=$('keep-loaded').checked}){
  if(!state.ready||state.busy)throw Error(state.busy?'A local generation is already running.':'Prepare the models first.');
  if(!prompt.trim())throw Error('Enter a description for your image.');
  if(!Number.isInteger(seed)||seed<0||seed>4294967295)throw Error('Use a whole-number seed from 0 to 4294967295.');
  state.busy=true;$('error').hidden=true;$('generate').disabled=true;$('prepare').disabled=true;$('generation-progress').hidden=false;
  clearTimeout(state.idleTimer);if(!keepLoaded&&state.worker){state.worker.terminate();state.worker=null;}
  const started=performance.now(),worker=state.worker||new Worker('/worker.js',{type:'module'});state.worker=worker;let success=false;
  try{return await new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>{worker.terminate();reject(Error('Local generation exceeded five minutes. Try Low memory mode and close other tabs.'));},300000);
    worker.onerror=e=>{clearTimeout(timeout);reject(Error(e.message||'The local worker stopped, possibly due to memory pressure. Try Low memory mode.'));};
    worker.onmessage=async({data})=>{
      if(data.type==='progress'){$('stage').textContent=data.message;$('generation-progress').value=data.progress;return;}
      clearTimeout(timeout);
      if(data.type==='error'){reject(Error(data.message));return;}
      if(data.type==='result'){try{showResult(data,performance.now()-started);await $('image').decode();data.metadata.submit_to_visible_ms=performance.now()-started;$('metadata').textContent=JSON.stringify(data.metadata,null,2);success=true;resolve(data);}catch(error){reject(error);}}
    };
    worker.postMessage({prompt,seed,mode,omit,keepLoaded});
  });}catch(error){fail(error.message);throw error;}finally{if(!keepLoaded||!success){worker.terminate();state.worker=null;}else state.idleTimer=setTimeout(()=>{worker.terminate();if(state.worker===worker)state.worker=null;},120000);state.busy=false;$('generate').disabled=!state.ready;$('prepare').disabled=false;}
}
addEventListener('pagehide',()=>state.worker?.terminate());
$('form').onsubmit=event=>{event.preventDefault();generate({prompt:$('prompt').value.trim(),seed:Number($('seed').value),mode:$('mode').value}).catch(()=>{});};
for(const button of document.querySelectorAll('[data-image]'))button.onclick=()=>{const kind=button.dataset.image;$('image').src=state.urls[kind];$('save').href=state.urls[kind];for(const other of document.querySelectorAll('[data-image]'))other.setAttribute('aria-pressed',String(other===button));};
// Reproducible diagnostics operate on this app only; no remote inference path exists.
window.__p3={generate,prepare:()=>new Promise((resolve,reject)=>{$('prepare').click();const timer=setInterval(()=>{if(!state.controller){clearInterval(timer);state.ready?resolve(true):reject(Error($('error').textContent));}},200);}),state};
