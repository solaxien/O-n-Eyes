// Classical image operations, implemented locally without a learned enhancer.
const clamp=x=>Math.max(0,Math.min(255,Math.round(x)));
function inspect(rgb,width,height){
  const count=width*height,gray=new Float32Array(count),hist=new Uint32Array(256);let clipped=0,sum=0,square=0;
  for(let i=0;i<count;i++){const v=.299*rgb[i*4]+.587*rgb[i*4+1]+.114*rgb[i*4+2];gray[i]=v;hist[clamp(v)]++;if(v<3||v>252)clipped++;}
  const percentile=p=>{let c=0;for(let i=0;i<256;i++){c+=hist[i];if(c>=p*count)return i;}return 255;};
  let edges=0;for(let y=1;y<height-1;y++)for(let x=1;x<width-1;x++){const i=y*width+x,lap=gray[i-1]+gray[i+1]+gray[i-width]+gray[i+width]-4*gray[i];sum+=lap;square+=lap*lap;if(Math.abs(gray[i+1]-gray[i-1])+Math.abs(gray[i+width]-gray[i-width])>28)edges++;}
  const interior=(width-2)*(height-2);return {gray,metrics:{luminance_span:percentile(.99)-percentile(.01),contrast:percentile(.9)-percentile(.1),clipped_fraction:clipped/count,laplacian_variance:square/interior-(sum/interior)**2,edges}};
}
export function enhance(raw,width,height,omit=[]){
  const before=inspect(raw,width,height),pixels=new Uint8ClampedArray(raw),mask=new Uint8Array(width*height),medians=new Float32Array(width*height);let artifacts=0;
  for(let y=1;y<height-1;y++)for(let x=1;x<width-1;x++){
    const i=y*width+x,values=[];for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)values.push(before.gray[i+dy*width+dx]);values.sort((a,b)=>a-b);const median=values[4];medians[i]=median;
    // Isolated outliers only; preserve structured edges and clustered defects.
    const span=values[7]-values[1];if(Math.abs(before.gray[i]-median)>=48&&span<24){mask[i]=1;artifacts++;}
  }
  const fraction=artifacts/(width*height),usable=before.metrics.luminance_span>=8&&fraction<=.035;
  if(!usable&&!omit.includes('gate'))return {pixels,applied:['Pixel quality gate flagged decode'],before:before.metrics,after:before.metrics,artifact_fraction:fraction,usable:false};
  const applied=[];
  if(artifacts&&fraction<=.004&&!omit.includes('repair')){
    for(let i=0;i<mask.length;i++)if(mask[i])for(let c=0;c<3;c++){const a=[];for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)if(dx||dy)a.push(raw[(i+dy*width+dx)*4+c]);a.sort((a,b)=>a-b);pixels[i*4+c]=Math.round((a[3]+a[4])/2);}
    applied.push('Isolated speckle correction');
  }
  if(!omit.includes('edges')){
    const source=new Uint8ClampedArray(pixels);
    for(let y=1;y<height-1;y++)for(let x=1;x<width-1;x++){
      const i=y*width+x;let total=0,luma=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const j=i+dy*width+dx,d=before.gray[j]-before.gray[i],w=Math.exp(-.5*(dx*dx+dy*dy)-d*d/288);total+=w;luma+=w*before.gray[j];}
      const change=.2*(luma/total-before.gray[i]);for(let c=0;c<3;c++)pixels[i*4+c]=clamp(source[i*4+c]+change);
    }
    applied.push('Edge-aware bilateral luminance');
  }
  if(!omit.includes('color')){
    let neutral=0,means=[0,0,0];for(let i=0;i<pixels.length;i+=4){const rgb=[pixels[i],pixels[i+1],pixels[i+2]],l=(rgb[0]+rgb[1]+rgb[2])/3;if(l>45&&l<215&&Math.max(...rgb)-Math.min(...rgb)<15){neutral++;for(let c=0;c<3;c++)means[c]+=rgb[c]-l;}}
    if(neutral/(width*height)>=.35){means=means.map(x=>Math.max(-3,Math.min(3,x/neutral*.35)));if(Math.max(...means.map(Math.abs))>=1){for(let i=0;i<pixels.length;i+=4)for(let c=0;c<3;c++)pixels[i+c]=clamp(pixels[i+c]-means[c]);applied.push('Conservative neutral colour balance');}}
    if(before.metrics.contrast>=12&&before.metrics.contrast<55){for(let i=0;i<pixels.length;i+=4)for(let c=0;c<3;c++)pixels[i+c]=clamp(128+(pixels[i+c]-128)*1.035);applied.push('Bounded contrast correction');}
  }
  const after=inspect(pixels,width,height);
  if(!omit.includes('gate')&&(after.metrics.clipped_fraction>before.metrics.clipped_fraction+.025||(before.metrics.edges>100&&after.metrics.edges<before.metrics.edges*.72))){return {pixels:new Uint8ClampedArray(raw),applied:['Quality gate retained raw pixels'],before:before.metrics,after:before.metrics,artifact_fraction:fraction,usable};}
  return {pixels,applied,before:before.metrics,after:after.metrics,artifact_fraction:fraction,usable};
}
export async function png(pixels,width,height,display=512){const source=new OffscreenCanvas(width,height);source.getContext('2d').putImageData(new ImageData(pixels,width,height),0,0);const output=new OffscreenCanvas(display,Math.round(height*display/width)),ctx=output.getContext('2d');ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(source,0,0,output.width,output.height);return output.convertToBlob({type:'image/png'});}
