(async () => {
  const result = {date:new Date().toISOString(),origin:location.origin,userAgent:navigator.userAgent,
    secureContext:isSecureContext,webgpu:!!navigator.gpu,cores:navigator.hardwareConcurrency,
    deviceMemoryGiB:navigator.deviceMemory,crossOriginIsolated,storage:await navigator.storage.estimate(),
    wasm:typeof WebAssembly==='object',jsHeap:performance.memory?{limit:performance.memory.jsHeapSizeLimit,used:performance.memory.usedJSHeapSize}:null};
  if (!navigator.gpu) return result;
  const adapter = await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  if (!adapter) return {...result,adapter:null};
  result.adapter = {vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,
    description:adapter.info.description,isFallbackAdapter:adapter.info.isFallbackAdapter};
  result.features = [...adapter.features];
  result.limits = Object.fromEntries(['maxBufferSize','maxStorageBufferBindingSize','maxComputeWorkgroupStorageSize',
    'maxComputeInvocationsPerWorkgroup','maxStorageBuffersPerShaderStage'].map(k=>[k,adapter.limits[k]]));
  const device = await adapter.requestDevice({requiredFeatures:adapter.features.has('shader-f16')?['shader-f16']:[]});
  result.computations=[];
  for(const dtype of ['f32',...(adapter.features.has('shader-f16')?['f16']:[])]) {
    const code=`${dtype==='f16'?'enable f16;':''}\n@group(0) @binding(0) var<storage,read_write> out:array<f32>; @compute @workgroup_size(1) fn main() { let x:${dtype}=${dtype}(1.5); out[0]=f32(x*x+${dtype}(0.75)); }`;
    device.pushErrorScope('validation');
    const module=device.createShaderModule({code});
    const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
    const gpu=device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:gpu}}]});
    const encoder=device.createCommandEncoder();
    const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(1);pass.end();
    encoder.copyBufferToBuffer(gpu,0,read,0,4);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const value=new Float32Array(read.getMappedRange())[0];read.unmap();gpu.destroy();read.destroy();
    const error=await device.popErrorScope();
    result.computations.push({dtype,value,passed:value===3,error:error?.message||null});
  }
  // Bounded allocation probe, not a claim about total available GPU memory.
  device.pushErrorScope('out-of-memory');
  const memory=device.createBuffer({size:64*1024*1024,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(memory,0,new Uint32Array([38]));
  await device.queue.onSubmittedWorkDone();
  result.allocationProbe={bytes:64*1024*1024,error:(await device.popErrorScope())?.message||null};
  memory.destroy();device.destroy();return result;
})()
