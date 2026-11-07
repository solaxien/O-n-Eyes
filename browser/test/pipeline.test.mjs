import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {dpmStep,half,noise} from '../src/numerics.js';
import {enhance} from '../src/vision.js';
const fixture=JSON.parse(await readFile(new URL('./scheduler-fixtures.json',import.meta.url),'utf8'));
for(const data of fixture.cases)test(`DPM++ ${data.steps} steps match independent Diffusers FP32 reference`,()=>{let sample=new Float32Array(data.initial),previous=null;for(let i=0;i<data.rows.length;i++){({sample,previous}=dpmStep(sample,new Float32Array(data.rows[i].epsilon),data.sigmas,i,previous));const max=Math.max(...sample.map((x,j)=>Math.abs(x-data.rows[i].sample[j])));assert.ok(max<.00005,`step ${i}: ${max}`);}});
test('FP16 conversion handles boundary values and ties',()=>{for(const [x,expected] of [[0,0],[-0,0x8000],[1,0x3c00],[1.5,0x3e00],[-2,0xc000],[65504,0x7bff],[Infinity,0x7c00],[2**-24,1],[2**-25,0],[1+2**-11,0x3c00]])assert.equal(half(x),expected);assert.ok((half(NaN)&0x7fff)>0x7c00);});
test('Fresh seeds alter finite initial noise reproducibly',()=>{assert.deepEqual(noise(38,64),noise(38,64));assert.notDeepEqual(noise(38,64),noise(39,64));assert.ok(noise(0,4096).every(Number.isFinite));});
function image(size,hot=false){const out=new Uint8ClampedArray(size*size*4);for(let y=0;y<size;y++)for(let x=0;x<size;x++){const i=(y*size+x)*4;out.fill(50+x*2,i,i+3);out[i+3]=255;}if(hot)out.fill(255,(20*size+20)*4,(20*size+20)*4+3);return out;}
test('An injected isolated hot pixel is corrected while a gradient remains usable',()=>{const clean=image(64),raw=image(64,true),result=enhance(raw,64,64),i=(20*64+20)*4;assert.ok(result.usable);assert.ok(result.applied.includes('Isolated speckle correction'));assert.ok(Math.abs(result.pixels[i]-clean[i])<Math.abs(raw[i]-clean[i])*.1);});
test('Uniform failed decode is flagged and retains original pixels',()=>{const raw=new Uint8ClampedArray(32*32*4).fill(255),result=enhance(raw,32,32);assert.equal(result.usable,false);assert.deepEqual(result.pixels,raw);});
