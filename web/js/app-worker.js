'use strict';

(function initWorkerModule(global) {
  const app = global.CameraDropApp;
  const state = app.state;
  const dom = app.dom;
  const ui = app.ui;
  const config = app.config;

  const base = new URL('.', location.href).href;
  const workerSource = `'use strict';
importScripts('${base}ort.webgpu.min.js');
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = {
  'ort-wasm-simd-threaded.wasm': '${base}ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.jsep.wasm': '${base}ort-wasm-simd-threaded.jsep.wasm',
};
const CONF=${config.CONF},ANCHOR_EXP=${config.ANCHOR_EXP},FULL_SZ=640,FAST_SZ=320,FULL_EVERY=6,SCAN_MAX=${Math.max(256, Number(config.SCAN_MAX_SIDE) || 960)};
const CANON_SIZE=1024,ANCHOR_OUT=2,ANCHOR_SIZE=56,ANCHOR_CENTER=ANCHOR_OUT+(ANCHOR_SIZE/2),SCAN_ANCHOR_SIZE=30;
let sess=null,frameCount=0,localizerMode='yolo';
const _lb={},_scan={},_refine={};
function clamp(v,lo,hi){return v<lo?lo:(v>hi?hi:v);}
function dist2(a,b){const dx=a.x-b.x,dy=a.y-b.y;return dx*dx+dy*dy;}
function letterbox(bm,SZ){
  const vw=bm.width,vh=bm.height,sc=Math.min(SZ/vw,SZ/vh);
  const nw=Math.round(vw*sc),nh=Math.round(vh*sc),px=(SZ-nw)>>1,py=(SZ-nh)>>1;
  if(!_lb[SZ]){_lb[SZ]={cvs:new OffscreenCanvas(SZ,SZ),f32:new Float32Array(3*SZ*SZ)};_lb[SZ].ctx=_lb[SZ].cvs.getContext('2d');}
  const{ctx,f32}=_lb[SZ];
  ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,SZ,SZ);ctx.drawImage(bm,px,py,nw,nh);
  const id=ctx.getImageData(0,0,SZ,SZ),N=SZ*SZ,d=id.data;
  for(let i=0;i<N;i++){f32[i]=d[i*4]/255;f32[N+i]=d[i*4+1]/255;f32[2*N+i]=d[i*4+2]/255;}
  return{f32,scale:sc,px,py};
}
function parseOutput(out,scale,px,py,vw,vh){
  const od=out.data,n=out.dims[1],FEAT=out.dims[2],dets=[];
  for(let i=0;i<n;i++){
    const o=i*FEAT,score=od[o+4];if(score<CONF)continue;
    const cls=od[o+5]|0;
    const x1=Math.max(0,(od[o]-px)/scale),y1=Math.max(0,(od[o+1]-py)/scale);
    const x2=Math.min(vw,(od[o+2]-px)/scale),y2=Math.min(vh,(od[o+3]-py)/scale);
    if(x2>x1&&y2>y1)dets.push({box:[x1,y1,x2,y2],score,cls});
  }
  return dets;
}
function ensureRefineBuffers(w,h){
  if(!_refine.cvs||_refine.cvs.width!==w||_refine.cvs.height!==h){
    _refine.cvs=new OffscreenCanvas(w,h);
    _refine.ctx=_refine.cvs.getContext('2d',{willReadFrequently:true});
    _refine.gray=new Uint8Array(w*h);
    _refine.blur=new Uint8Array(w*h);
  }
}
function detectAnchorQuad(bm,box){
  const margin=4;
  const x1=Math.max(0,Math.floor(box[0])-margin),y1=Math.max(0,Math.floor(box[1])-margin);
  const x2=Math.min(bm.width,Math.ceil(box[2])+margin),y2=Math.min(bm.height,Math.ceil(box[3])+margin);
  const w=x2-x1,h=y2-y1;
  if(w<6||h<6)return null;
  ensureRefineBuffers(w,h);
  _refine.ctx.drawImage(bm,x1,y1,w,h,0,0,w,h);
  const rgba=_refine.ctx.getImageData(0,0,w,h).data;
  const gray=_refine.gray,blur=_refine.blur;
  for(let i=0,j=0;i<gray.length;i++,j+=4)gray[i]=(rgba[j]*77+rgba[j+1]*150+rgba[j+2]*29)>>8;
  blurGray3(gray,blur,w,h);
  const pts=[];
  const thr=Math.max(48,Math.min(160,Math.round((w+h)*1.2)));
  for(let y=1;y<h-1;y++){
    const row=y*w;
    for(let x=1;x<w-1;x++){
      const idx=row+x;
      const gx=(blur[idx-w+1]+(blur[idx+1]<<1)+blur[idx+w+1])-(blur[idx-w-1]+(blur[idx-1]<<1)+blur[idx+w-1]);
      const gy=(blur[idx+w-1]+(blur[idx+w]<<1)+blur[idx+w+1])-(blur[idx-w-1]+(blur[idx-w]<<1)+blur[idx-w+1]);
      if(Math.abs(gx)+Math.abs(gy)>=thr)pts.push({x,y});
    }
  }
  if(pts.length<10)return null;
  let cx=0,cy=0;
  for(const p of pts){cx+=p.x;cy+=p.y;}
  cx/=pts.length;cy/=pts.length;
  let cxx=0,cxy=0,cyy=0;
  for(const p of pts){const dx=p.x-cx,dy=p.y-cy;cxx+=dx*dx;cxy+=dx*dy;cyy+=dy*dy;}
  const ang=0.5*Math.atan2(2*cxy,cxx-cyy);
  const ux=Math.cos(ang),uy=Math.sin(ang),vx=-uy,vy=ux;
  let minU=Infinity,maxU=-Infinity,minV=Infinity,maxV=-Infinity;
  for(const p of pts){
    const dx=p.x-cx,dy=p.y-cy,u=dx*ux+dy*uy,v=dx*vx+dy*vy;
    if(u<minU)minU=u;if(u>maxU)maxU=u;if(v<minV)minV=v;if(v>maxV)maxV=v;
  }
  const mk=(u,v)=>({x:x1+cx+u*ux+v*vx,y:y1+cy+u*uy+v*vy});
  return[mk(minU,minV),mk(maxU,minV),mk(minU,maxV),mk(maxU,maxV)];
}
function outerCornerFromDet(det,bm,fcx,fcy,inv){
  const pts=detectAnchorQuad(bm,det.box)||[
    {x:det.box[0],y:det.box[1]},
    {x:det.box[2],y:det.box[1]},
    {x:det.box[0],y:det.box[3]},
    {x:det.box[2],y:det.box[3]},
  ];
  let best=pts[0],bestD=-1;
  for(const p of pts){
    const dd=(p.x-fcx)*(p.x-fcx)+(p.y-fcy)*(p.y-fcy);
    if(dd>bestD){bestD=dd;best=p;}
  }
  return[best.x*inv,best.y*inv];
}
function buildCornersFromAnchorDetections(normals,brA,inv,frameBox){
  if(!brA||normals.length<3)return null;
  normals.sort((a,b)=>b.d.score-a.d.score);
  const t=normals.slice(0,3);
  t.sort((a,b)=>((b.cx-brA.cx)*(b.cx-brA.cx)+(b.cy-brA.cy)*(b.cy-brA.cy))-((a.cx-brA.cx)*(a.cx-brA.cx)+(a.cy-brA.cy)*(a.cy-brA.cy)));
  const tlA=t[0],rem=t.slice(1);
  const vx=brA.cx-tlA.cx,vy=brA.cy-tlA.cy;
  const cf=p=>vx*(p.cy-tlA.cy)-vy*(p.cx-tlA.cx);
  const trA=cf(rem[0])<0?rem[0]:rem[1],blA=cf(rem[0])<0?rem[1]:rem[0];
  if(!frameBox){
    return buildCornersFromCenters([{x:tlA.cx,y:tlA.cy},{x:trA.cx,y:trA.cy},{x:blA.cx,y:blA.cy},{x:brA.cx,y:brA.cy}],inv);
  }
  const[bx1,by1,bx2,by2]=frameBox;
  const fcx=(bx1+bx2)/2,fcy=(by1+by2)/2;
  const bc=a=>{
    const[b0,b1,b2,b3]=a.d.box;let best=[b0,b1],bd=-1;
    for(const c of[[b0,b1],[b2,b1],[b0,b3],[b2,b3]]){const dd=(c[0]-fcx)*(c[0]-fcx)+(c[1]-fcy)*(c[1]-fcy);if(dd>bd){bd=dd;best=c;}}
    return[best[0]*inv,best[1]*inv];
  };
  return{TL:bc(tlA),TR:bc(trA),BL:bc(blA),BR:bc(brA),outSize:Math.round(Math.max(bx2-bx1,by2-by1)*inv)};
}
function collectAnchorDetections(all,frameBox){
  const normals=[],brs=[];
  let bx1=0,by1=0,bx2=0,by2=0,ex=0,ey=0;
  if(frameBox){
    [bx1,by1,bx2,by2]=frameBox;
    ex=(bx2-bx1)*ANCHOR_EXP;
    ey=(by2-by1)*ANCHOR_EXP;
  }
  for(const d of all){
    if(d.cls!==2&&d.cls!==3)continue;
    const ax=(d.box[0]+d.box[2])/2,ay=(d.box[1]+d.box[3])/2;
    if(frameBox&&(ax<bx1-ex||ax>bx2+ex||ay<by1-ey||ay>by2+ey))continue;
    (d.cls===3?brs:normals).push({cx:ax,cy:ay,d});
  }
  return{normals,brs};
}
function assignCorners(all,vw,vh,inv){
  let bf=null;
  for(const d of all)if(d.cls===0&&(!bf||d.score>bf.score))bf=d;
  if(bf){
    const scoped=collectAnchorDetections(all,bf.box);
    if(scoped.normals.length>=3&&scoped.brs.length>=1){
      scoped.brs.sort((a,b)=>b.d.score-a.d.score);
      return buildCornersFromAnchorDetections(scoped.normals,scoped.brs[0],inv,bf.box);
    }
  }
  const globalAnchors=collectAnchorDetections(all,null);
  if(globalAnchors.normals.length<3||globalAnchors.brs.length<1)return null;
  globalAnchors.brs.sort((a,b)=>b.d.score-a.d.score);
  return buildCornersFromAnchorDetections(globalAnchors.normals,globalAnchors.brs[0],inv,bf?bf.box:null);
}

function ensureScanBuffers(w,h){
  if(!_scan.cvs||_scan.cvs.width!==w||_scan.cvs.height!==h){
    _scan.cvs=new OffscreenCanvas(w,h);
    _scan.ctx=_scan.cvs.getContext('2d',{willReadFrequently:true});
    _scan.gray=new Uint8Array(w*h);
    _scan.blur=new Uint8Array(w*h);
    _scan.bin=new Uint8Array(w*h);
  }
}
function blurGray3(src,dst,w,h){
  for(let y=0;y<h;y++){
    const y0=y>0?y-1:y,y1=y+1<h?y+1:y;
    for(let x=0;x<w;x++){
      const x0=x>0?x-1:x,x1=x+1<w?x+1:x;
      let sum=0,n=0;
      for(let yy=y0;yy<=y1;yy++){
        let idx=yy*w+x0;
        for(let xx=x0;xx<=x1;xx++,idx++){sum+=src[idx];n++;}
      }
      dst[y*w+x]=(sum/n)|0;
    }
  }
}
function otsuThreshold(gray){
  const hist=new Uint32Array(256);
  for(let i=0;i<gray.length;i++)hist[gray[i]]++;
  let total=gray.length,sum=0;
  for(let i=0;i<256;i++)sum+=i*hist[i];
  let sumB=0,wB=0,varMax=-1,thr=128;
  for(let t=0;t<256;t++){
    wB+=hist[t];
    if(!wB)continue;
    const wF=total-wB;
    if(!wF)break;
    sumB+=t*hist[t];
    const mB=sumB/wB,mF=(sum-sumB)/wF;
    const diff=mB-mF;
    const variance=wB*wF*diff*diff;
    if(variance>varMax){varMax=variance;thr=t;}
  }
  return thr;
}
function nextPowerOfTwoPlusOne(v){
  v=Math.max(3,Math.ceil(v));
  v-=1;
  v|=v>>1;
  v|=v>>2;
  v|=v>>4;
  v|=v>>8;
  v|=v>>16;
  return Math.max(3,v+2);
}
function buildIntegralGray(src,w,h){
  const integ=new Uint32Array((w+1)*(h+1));
  for(let y=1;y<=h;y++){
    let rowSum=0;
    const srcRow=(y-1)*w;
    const outRow=y*(w+1);
    const prevRow=(y-1)*(w+1);
    for(let x=1;x<=w;x++){
      rowSum+=src[srcRow+x-1];
      integ[outRow+x]=integ[prevRow+x]+rowSum;
    }
  }
  return integ;
}
function boxBlurGray(src,dst,w,h,ksize){
  if(ksize<=3){blurGray3(src,dst,w,h);return;}
  const integ=buildIntegralGray(src,w,h),stride=w+1,rad=ksize>>1;
  for(let y=0;y<h;y++){
    const y0=Math.max(0,y-rad),y1=Math.min(h-1,y+rad),iy0=y0*stride,iy1=(y1+1)*stride;
    const row=y*w;
    for(let x=0;x<w;x++){
      const x0=Math.max(0,x-rad),x1=Math.min(w-1,x+rad),area=(x1-x0+1)*(y1-y0+1);
      const sum=integ[iy1+x1+1]-integ[iy1+x0]-integ[iy0+x1+1]+integ[iy0+x0];
      dst[row+x]=(sum/area)|0;
    }
  }
}
function adaptiveThresholdMean(src,dst,w,h,blockSize,bias){
  const integ=buildIntegralGray(src,w,h),stride=w+1,rad=blockSize>>1;
  for(let y=0;y<h;y++){
    const y0=Math.max(0,y-rad),y1=Math.min(h-1,y+rad),iy0=y0*stride,iy1=(y1+1)*stride;
    const row=y*w;
    for(let x=0;x<w;x++){
      const x0=Math.max(0,x-rad),x1=Math.min(w-1,x+rad),area=(x1-x0+1)*(y1-y0+1);
      const mean=(integ[iy1+x1+1]-integ[iy1+x0]-integ[iy0+x1+1]+integ[iy0+x0])/area;
      dst[row+x]=src[row+x]>mean+bias?1:0;
    }
  }
}
function prepareScanBitmap(bm,adaptive){
  const vw=bm.width,vh=bm.height,sc=Math.min(1,SCAN_MAX/Math.max(vw,vh));
  const sw=Math.max(32,Math.round(vw*sc)),sh=Math.max(32,Math.round(vh*sc));
  ensureScanBuffers(sw,sh);
  _scan.ctx.drawImage(bm,0,0,sw,sh);
  const rgba=_scan.ctx.getImageData(0,0,sw,sh).data;
  const gray=_scan.gray,blur=_scan.blur,bin=_scan.bin;
  for(let i=0,j=0;i<gray.length;i++,j+=4)gray[i]=(rgba[j]*77+rgba[j+1]*150+rgba[j+2]*29)>>8;
  const blurUnit=Math.max(nextPowerOfTwoPlusOne(Math.min(sw,sh)*0.002),3);
  boxBlurGray(gray,blur,sw,sh,blurUnit);
  if(adaptive){
    const blockSize=nextPowerOfTwoPlusOne(Math.min(sw,sh)*0.05);
    adaptiveThresholdMean(blur,bin,sw,sh,blockSize,10);
  }else{
    const thr=otsuThreshold(blur);
    for(let i=0;i<bin.length;i++)bin[i]=blur[i]>=thr?1:0;
  }
  return{
    w:sw,
    h:sh,
    scale:sc,
    inv:1/sc,
    bin,
    skip:Math.max(1,(Math.min(sw,sh)/120)|0),
    mergeCutoff:Math.max(4,(sw/30)|0),
    anchorSize:SCAN_ANCHOR_SIZE,
    adaptive:!!adaptive,
  };
}
function makeAnchor(x,xmax,y,ymax){return{x,y,xmax,ymax};}
function cloneAnchor(a){return{x:a.x,y:a.y,xmax:a.xmax,ymax:a.ymax};}
function anchorMerge(a,b){a.x=Math.min(a.x,b.x);a.xmax=Math.max(a.xmax,b.xmax);a.y=Math.min(a.y,b.y);a.ymax=Math.max(a.ymax,b.ymax);return a;}
function anchorXAvg(a){return(a.x+a.xmax)*0.5;}
function anchorYAvg(a){return(a.y+a.ymax)*0.5;}
function anchorXRange(a){return Math.abs(a.x-a.xmax)*0.5;}
function anchorYRange(a){return Math.abs(a.y-a.ymax)*0.5;}
function anchorSize(a){const dx=a.x-a.xmax,dy=a.y-a.ymax;return dx*dx+dy*dy;}
function anchorMaxRange(a){return Math.max(Math.abs(a.x-a.xmax),Math.abs(a.y-a.ymax));}
function anchorCenter(a){return{x:anchorXAvg(a),y:anchorYAvg(a)};}
function isMergeable(a,b,maxDistance){
  if(Math.abs(anchorXAvg(a)-anchorXAvg(b))>maxDistance||Math.abs(anchorYAvg(a)-anchorYAvg(b))>maxDistance)return false;
  const denom=Math.max(1,anchorMaxRange(a));
  const ratio=anchorMaxRange(b)*10/denom;
  return ratio>6&&ratio<17;
}
function makeScanState(limits){return{state:0,tally:[0],limits};}
function makeState114(){return makeScanState([[0,0],[3.0,6.0],[3.0,6.0],[0,0],[3.0,6.0],[3.0,6.0]]);}
function makeState122(){return makeScanState([[0,0],[1.0,3.0],[0.5,1.5],[0,0],[0.5,1.5],[1.0,3.0]]);}
function popScanState(st){st.state-=2;st.tally.shift();st.tally.shift();}
function evaluateScanState(st){
  if(st.state!==6)return-1;
  for(let i=1;i<=5;i++)if((st.tally[i]||0)===0)return-1;
  const center=st.tally[3];
  for(let i=1;i<=5;i++){
    if(i===3)continue;
    const ratioMin=center/((st.tally[i]||0)+1);
    const ratioMax=center/Math.max(1,(st.tally[i]||0)-1);
    if(ratioMax<st.limits[i][0]||ratioMin>st.limits[i][1])return-1;
  }
  let size=0;
  for(let i=1;i<=5;i++)size+=st.tally[i]||0;
  return size;
}
function processScanState(st,active){
  const even=st.state===0||st.state===2||st.state===4;
  const odd=st.state===1||st.state===3||st.state===5;
  const isTransition=(even&&active)||(odd&&!active);
  if(isTransition){
    st.state+=1;
    st.tally.push(1);
    if(st.state===6){
      const res=evaluateScanState(st);
      popScanState(st);
      return res;
    }
    return-1;
  }
  const tail=st.tally.length-1;
  if(odd&&active)st.tally[tail]+=1;
  if(!active&&(st.state===2||st.state===4))st.tally[tail]+=1;
  return-1;
}
function makeEdgeState(){return{state:0,tally:[0]};}
function popEdgeState(st){st.state-=2;st.tally.shift();st.tally.shift();}
function processEdgeState(st,active){
  const isTransition=(st.state===0&&active)||(st.state===1&&!active);
  if(isTransition){
    st.state+=1;
    st.tally.push(1);
    if(st.state===2){
      const res=st.tally[1]||0;
      popEdgeState(st);
      return res;
    }
    return-1;
  }
  const tail=st.tally.length-1;
  if(st.state===1&&active)st.tally[tail]+=1;
  if(st.state===0&&!active)st.tally[tail]+=1;
  return-1;
}
function scanActive(scan,x,y){return x>=0&&x<scan.w&&y>=0&&y<scan.h&&scan.bin[y*scan.w+x]===0;}
function scanHorizontal(scan,stateFactory,points,y,xstart,xend){
  if(y<0||y>=scan.h)return false;
  if(xstart==null||xstart<0)xstart=0;
  if(xend==null||xend<0||xend>scan.w)xend=scan.w;
  const init=points.length,st=stateFactory();
  for(let x=xstart;x<xend;x++){
    const res=processScanState(st,scanActive(scan,x,y));
    if(res>0)points.push(makeAnchor(x-res,x-1,y,y));
  }
  const res=processScanState(st,false);
  if(res>0)points.push(makeAnchor(xend-res,xend-1,y,y));
  return points.length!==init;
}
function scanVertical(scan,stateFactory,points,x,xmax,ystart,yend){
  if(xmax==null||xmax<0)xmax=x;
  const xavg=Math.round((x+xmax)*0.5);
  if(xavg<0||xavg>=scan.w)return false;
  if(ystart==null||ystart<0)ystart=0;
  if(yend==null||yend<0||yend>scan.h)yend=scan.h;
  const init=points.length,st=stateFactory();
  for(let y=ystart;y<yend;y++){
    const res=processScanState(st,scanActive(scan,xavg,y));
    if(res>0)points.push(makeAnchor(xavg,xavg,y-res,y-1));
  }
  const res=processScanState(st,false);
  if(res>0)points.push(makeAnchor(xavg,xavg,yend-res,yend-1));
  return points.length!==init;
}
function scanDiagonal(scan,stateFactory,points,xstart,xend,ystart,yend){
  xend=Math.min(xend,scan.w);
  yend=Math.min(yend,scan.h);
  if(xstart<0){const off=-xstart;xstart+=off;ystart+=off;}
  if(ystart<0){const off=-ystart;xstart+=off;ystart+=off;}
  const init=points.length,st=stateFactory();
  let x=xstart,y=ystart;
  for(;x<xend&&y<yend;x++,y++){
    const res=processScanState(st,scanActive(scan,x,y));
    if(res>0)points.push(makeAnchor(x-res,x-1,y-res,y-1));
  }
  const res=processScanState(st,false);
  if(res>0)points.push(makeAnchor(x-res,x-1,y-res,y-1));
  return points.length!==init;
}
function t1ScanRows(scan,stateFactory,fun,skip,ystart,yend,xstart,xend){
  if(skip==null||skip<=0)skip=scan.skip;
  if(ystart==null||ystart<0)ystart=0;
  if(yend==null||yend<0||yend>scan.h)yend=scan.h;
  const points=[];
  const offsets=[0];
  const half=skip>>1;
  if(half>0)offsets.push(half);
  for(const off of offsets){
    for(let y=ystart+off;y<yend;y+=skip)scanHorizontal(scan,stateFactory,points,y,xstart,xend);
  }
  for(const p of points)fun(p);
}
function t2ScanColumn(scan,stateFactory,hint,fun){
  const points=[];
  const ystart=Math.round(hint.y-(3*anchorXRange(hint)));
  const yend=Math.round(hint.ymax+(3*anchorXRange(hint)));
  scanVertical(scan,stateFactory,points,hint.x,hint.xmax,ystart,yend);
  for(const p of points)fun(p);
}
function t3ScanDiagonal(scan,stateFactory,hint,fun){
  const confirms=[];
  const xstart=Math.round(anchorXAvg(hint)-(2*anchorYRange(hint)));
  const xend=Math.round(anchorXAvg(hint)+(2*anchorYRange(hint)));
  const ystart=Math.round(hint.y-anchorYRange(hint));
  const yend=Math.round(hint.ymax+anchorYRange(hint));
  if(!scanDiagonal(scan,stateFactory,confirms,xstart,xend,ystart,yend))return;
  let confirm=false;
  const merged=cloneAnchor(hint);
  for(const co of confirms){
    if(isMergeable(co,hint,scan.mergeCutoff)){
      confirm=true;
      anchorMerge(merged,co);
    }
  }
  if(confirm)fun(merged);
}
function t4ConfirmScan(scan,stateFactory,hint,mergeConfirms,fun){
  hint=cloneAnchor(hint);
  {
    const confirms=[];
    const xstart=Math.round(hint.x-anchorXRange(hint));
    const xend=Math.round(hint.xmax+anchorXRange(hint));
    const yavg=Math.round(anchorYAvg(hint));
    for(const y of [yavg-1,yavg,yavg+1])if(!scanHorizontal(scan,stateFactory,confirms,y,xstart,xend))return;
    let confirm=false;
    for(const co of confirms){
      if(isMergeable(co,hint,scan.mergeCutoff)){
        confirm=true;
        if(!mergeConfirms)break;
        anchorMerge(hint,co);
      }
    }
    if(!confirm)return;
  }
  {
    const confirms=[];
    const ystart=Math.round(hint.y-anchorYRange(hint));
    const yend=Math.round(hint.ymax+anchorYRange(hint));
    const xavg=Math.round(anchorXAvg(hint));
    for(const x of [xavg-1,xavg,xavg+1])if(!scanVertical(scan,stateFactory,confirms,x,x,ystart,yend))return;
    let confirm=false;
    for(const co of confirms){
      if(isMergeable(co,hint,scan.mergeCutoff)){
        confirm=true;
        if(!mergeConfirms)break;
        anchorMerge(hint,co);
      }
    }
    if(!confirm)return;
  }
  fun(hint);
}
function onT1Scan(scan,stateFactory,found,candidates,mergeConfirms){
  for(const c of candidates)if(isMergeable(c,found,scan.mergeCutoff))return;
  t2ScanColumn(scan,stateFactory,found,p=>{
    t3ScanDiagonal(scan,stateFactory,p,p2=>{
      t4ConfirmScan(scan,stateFactory,p2,mergeConfirms,p3=>{
        candidates.push(p3);
      });
    });
  });
}
function filterCandidates(candidates){
  if(candidates.length<3)return 0;
  candidates.sort((a,b)=>anchorSize(b)-anchorSize(a));
  let cutoff=0;
  for(let i=0;i<3;i++)cutoff+=anchorSize(candidates[i]);
  cutoff/=8;
  let i=0;
  for(;i<candidates.length;i++)if(anchorSize(candidates[i])<cutoff)break;
  if(i>3)i=3;
  if(i<candidates.length)candidates.length=i;
  return cutoff;
}
function sortPrimaryAnchors(anchors){
  if(anchors.length<3)return false;
  const c0=anchorCenter(anchors[0]),c1=anchorCenter(anchors[1]),c2=anchorCenter(anchors[2]);
  const edges=[
    {x:c1.x-c2.x,y:c1.y-c2.y},
    {x:c2.x-c0.x,y:c2.y-c0.y},
    {x:c0.x-c1.x,y:c0.y-c1.y},
  ];
  let topLeft=0,maxD=-1;
  for(let i=0;i<3;i++){
    const d=edges[i].x*edges[i].x+edges[i].y*edges[i].y;
    if(d>maxD){maxD=d;topLeft=i;}
  }
  const fix=i=>i<0?2:(i>=3?0:i);
  const dep=edges[fix(topLeft-1)];
  const inc=edges[fix(topLeft+1)];
  const rot={x:-inc.y,y:inc.x};
  const overlap={x:dep.x-rot.x,y:dep.y-rot.y};
  const depD=dep.x*dep.x+dep.y*dep.y;
  const ovD=overlap.x*overlap.x+overlap.y*overlap.y;
  const topRight=ovD<depD?fix(topLeft+1):fix(topLeft-1);
  const bottomLeft=ovD<depD?fix(topLeft-1):fix(topLeft+1);
  const ordered=[anchors[topLeft],anchors[topRight],anchors[bottomLeft]];
  anchors.length=0;
  anchors.push(ordered[0],ordered[1],ordered[2]);
  return true;
}
function validatePrimaryAnchors(anchors){
  if(anchors.length<3)return false;
  const tl=anchorCenter(anchors[0]),tr=anchorCenter(anchors[1]),bl=anchorCenter(anchors[2]);
  const ux=tr.x-tl.x,uy=tr.y-tl.y,vx=bl.x-tl.x,vy=bl.y-tl.y;
  const du=Math.hypot(ux,uy),dv=Math.hypot(vx,vy);
  const avgSize=(anchorMaxRange(anchors[0])+anchorMaxRange(anchors[1])+anchorMaxRange(anchors[2]))/3;
  if(du<avgSize*3||dv<avgSize*3)return false;
  const ratio=du/Math.max(1,dv);
  if(ratio<0.35||ratio>2.85)return false;
  const cos=(ux*vx+uy*vy)/Math.max(1e-6,du*dv);
  return Math.abs(cos)<0.55;
}
function normalizeScanBounds(scan,bounds){
  if(!bounds)return null;
  const xstart=clamp(Math.floor(bounds.xstart),0,scan.w-1);
  const xend=clamp(Math.ceil(bounds.xend),xstart+1,scan.w);
  const ystart=clamp(Math.floor(bounds.ystart),0,scan.h-1);
  const yend=clamp(Math.ceil(bounds.yend),ystart+1,scan.h);
  return{xstart,xend,ystart,yend};
}
function scanPrimaryAnchors(scan,bounds){
  const roi=normalizeScanBounds(scan,bounds);
  const candidates=[];
  t1ScanRows(scan,makeState114,p=>{onT1Scan(scan,makeState114,p,candidates,true);},scan.skip,roi?roi.ystart:-1,roi?roi.yend:-1,roi?roi.xstart:-1,roi?roi.xend:-1);
  const cutoff=filterCandidates(candidates);
  sortPrimaryAnchors(candidates);
  if(!validatePrimaryAnchors(candidates))return{anchors:[],cutoff:0};
  return{anchors:candidates,cutoff};
}
function addBottomRightCorner(scan,anchors,cutoff,bounds){
  const tl=anchorCenter(anchors[0]),tr=anchorCenter(anchors[1]),bl=anchorCenter(anchors[2]);
  const topScalar=anchorMaxRange(anchors[2])/Math.max(anchorMaxRange(anchors[1]),anchorMaxRange(anchors[0]));
  const guess1={x:bl.x+(tr.x-tl.x)*topScalar,y:bl.y+(tr.y-tl.y)*topScalar};
  const leftScalar=anchorMaxRange(anchors[1])/Math.max(anchorMaxRange(anchors[2]),anchorMaxRange(anchors[0]));
  const guess2={x:tr.x+(bl.x-tl.x)*leftScalar,y:tr.y+(bl.y-tl.y)*leftScalar};
  const center={x:(guess1.x+guess2.x)*0.5,y:(guess1.y+guess2.y)*0.5};
  const range=Math.max(anchorMaxRange(anchors[0]),anchorMaxRange(anchors[1]),anchorMaxRange(anchors[2]))*2;
  const roi=normalizeScanBounds(scan,bounds)||{xstart:0,xend:scan.w,ystart:0,yend:scan.h};
  const xstart=Math.max(roi.xstart,Math.floor(center.x-range));
  const xend=Math.min(roi.xend,Math.ceil(center.x+range));
  const ystart=Math.max(roi.ystart,Math.floor(center.y-range));
  const yend=Math.min(roi.yend,Math.ceil(center.y+range));
  const candidates=[];
  t1ScanRows(scan,makeState122,p=>{onT1Scan(scan,makeState122,p,candidates,false);},Math.max(1,scan.skip>>1),ystart,yend,xstart,xend);
  if(!candidates.length)return false;
  for(const c of candidates){
    if(anchorSize(c)>cutoff){
      anchors.push(c);
      return true;
    }
  }
  return false;
}
function buildScanBoundsFromRawCorners(scan,corners,inputToRawInv){
  if(!corners)return null;
  const factor=scan.scale/Math.max(1e-6,inputToRawInv);
  const pts=[corners.TL,corners.TR,corners.BL,corners.BR];
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  for(const p of pts){
    const x=p[0]*factor,y=p[1]*factor;
    if(x<minX)minX=x;if(x>maxX)maxX=x;if(y<minY)minY=y;if(y>maxY)maxY=y;
  }
  const pad=Math.max(12,Math.max(maxX-minX,maxY-minY)*0.12);
  return{xstart:minX-pad,xend:maxX+pad,ystart:minY-pad,yend:maxY+pad};
}
function anchorsToCenters(anchors){return anchors.map(a=>{const c=anchorCenter(a);return{x:c.x,y:c.y};});}
function lineIntersection(a0,a1,b0,b1){
  const ax=a1.x-a0.x,ay=a0.y-a1.y,adet=a1.x*a0.y-a0.x*a1.y;
  const bx=b1.x-b0.x,by=b0.y-b1.y,bdet=b1.x*b0.y-b0.x*b1.y;
  const D=ay*bx-ax*by;
  if(Math.abs(D)<1e-8)return null;
  return{x:(adet*bx-ax*bdet)/D,y:(ay*bdet-adet*by)/D};
}
function averagePoint(a,b){return{x:(a.x+b.x)*0.5,y:(a.y+b.y)*0.5};}
function calculateEdgeMidpoints(tl,tr,bl,br){
  const center=lineIntersection(tl,br,tr,bl);
  const fallback={top:averagePoint(tl,tr),right:averagePoint(tr,br),bottom:averagePoint(bl,br),left:averagePoint(tl,bl)};
  if(!center)return fallback;
  const leftRightInf=lineIntersection(tr,br,tl,bl);
  const topBottomInf=lineIntersection(tl,tr,bl,br);
  if(!leftRightInf||!topBottomInf)return fallback;
  const tmid=lineIntersection(tl,tr,center,leftRightInf);
  const rmid=lineIntersection(tr,br,center,topBottomInf);
  const bmid=lineIntersection(bl,br,center,leftRightInf);
  const lmid=lineIntersection(tl,bl,center,topBottomInf);
  if(!tmid||!rmid||!bmid||!lmid)return fallback;
  return{top:tmid,right:rmid,bottom:bmid,left:lmid};
}
function chaseEdge(scan,start,unit){
  let success=0;
  for(const i of [-2,-1,1,2]){
    const x=(start.x+unit.x*i)|0,y=(start.y+unit.y*i)|0;
    if(scanActive(scan,x,y))success++;
  }
  return success>=2;
}
function findEdgePoint(scan,u,v,mid){
  const distance={x:v.x-u.x,y:v.y-u.y};
  const distanceUnit={x:distance.x/512,y:distance.y/512};
  const out={x:distance.y/64,y:-distance.x/64};
  const inn={x:-out.x,y:-out.y};
  const start=mid?{x:mid.x+out.x*(scan.anchorSize/16),y:mid.y+out.y*(scan.anchorSize/16)}:averagePoint(u,v);
  for(const check of [out,inn]){
    const maxCheck=Math.max(Math.abs(check.x),Math.abs(check.y));
    if(maxCheck<1e-6)continue;
    const unit={x:check.x/maxCheck,y:check.y/maxCheck};
    const st=makeEdgeState();
    let i=0,j=0;
    while(Math.abs(i)<=Math.abs(check.x)&&Math.abs(j)<=Math.abs(check.y)){
      const x=start.x+i,y=start.y+j;
      if(x>=0&&x<scan.w&&y>=0&&y<scan.h){
        const size=processEdgeState(st,scanActive(scan,x|0,y|0));
        if(size>0){
          const edge={x:x-unit.x*size*0.5,y:y-unit.y*size*0.5};
          if(chaseEdge(scan,edge,distanceUnit))return edge;
        }
      }
      i+=unit.x;
      j+=unit.y;
    }
  }
  return null;
}
function refineCornersByEdges(scan,anchors){
  if(anchors.length<4)return null;
  const tl=anchorCenter(anchors[0]),tr=anchorCenter(anchors[1]),bl=anchorCenter(anchors[2]),br=anchorCenter(anchors[3]);
  const mids=calculateEdgeMidpoints(tl,tr,bl,br);
  const top=findEdgePoint(scan,tl,tr,mids.top);
  const right=findEdgePoint(scan,tr,br,mids.right);
  const bottom=findEdgePoint(scan,br,bl,mids.bottom);
  const left=findEdgePoint(scan,bl,tl,mids.left);
  if(!top||!right||!bottom||!left)return null;
  const TL=lineIntersection(top,{x:top.x+(tr.x-tl.x),y:top.y+(tr.y-tl.y)},left,{x:left.x+(bl.x-tl.x),y:left.y+(bl.y-tl.y)});
  const TR=lineIntersection(top,{x:top.x+(tr.x-tl.x),y:top.y+(tr.y-tl.y)},right,{x:right.x+(br.x-tr.x),y:right.y+(br.y-tr.y)});
  const BL=lineIntersection(bottom,{x:bottom.x+(br.x-bl.x),y:bottom.y+(br.y-bl.y)},left,{x:left.x+(bl.x-tl.x),y:left.y+(bl.y-tl.y)});
  const BR=lineIntersection(bottom,{x:bottom.x+(br.x-bl.x),y:bottom.y+(br.y-bl.y)},right,{x:right.x+(br.x-tr.x),y:right.y+(br.y-tr.y)});
  if(!TL||!TR||!BL||!BR)return null;
  const pad=Math.max(scan.w,scan.h)*0.25;
  for(const p of [TL,TR,BL,BR]){
    if(!Number.isFinite(p.x)||!Number.isFinite(p.y))return null;
    if(p.x<-pad||p.x>scan.w+pad||p.y<-pad||p.y>scan.h+pad)return null;
  }
  return{TL,TR,BL,BR,edges:{top,right,bottom,left}};
}
function solveHomography(srcPts,dstPts){
  const A=new Float64Array(64),b=new Float64Array(8);
  for(let i=0;i<4;i++){
    const x=srcPts[i][0],y=srcPts[i][1],u=dstPts[i][0],v=dstPts[i][1],row=i*2;
    A.set([x,y,1,0,0,0,-u*x,-u*y],row*8);
    A.set([0,0,0,x,y,1,-v*x,-v*y],(row+1)*8);
    b[row]=u;b[row+1]=v;
  }
  const n=8,M=new Float64Array(n*(n+1));
  for(let i=0;i<n;i++){
    for(let j=0;j<n;j++)M[i*(n+1)+j]=A[i*n+j];
    M[i*(n+1)+n]=b[i];
  }
  for(let c=0;c<n;c++){
    let p=c;
    for(let r=c+1;r<n;r++)if(Math.abs(M[r*(n+1)+c])>Math.abs(M[p*(n+1)+c]))p=r;
    if(Math.abs(M[p*(n+1)+c])<1e-8)return null;
    for(let j=0;j<=n;j++){
      const t=M[c*(n+1)+j];
      M[c*(n+1)+j]=M[p*(n+1)+j];
      M[p*(n+1)+j]=t;
    }
    for(let r=c+1;r<n;r++){
      const f=M[r*(n+1)+c]/M[c*(n+1)+c];
      for(let j=c;j<=n;j++)M[r*(n+1)+j]-=f*M[c*(n+1)+j];
    }
  }
  const h=new Float64Array(n);
  for(let i=n-1;i>=0;i--){
    h[i]=M[i*(n+1)+n];
    for(let j=i+1;j<n;j++)h[i]-=M[i*(n+1)+j]*h[j];
    h[i]/=M[i*(n+1)+i];
  }
  return[h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7],1];
}
function projectPoint(H,x,y){
  const z=H[6]*x+H[7]*y+H[8];
  if(Math.abs(z)<1e-8)return null;
  return[(H[0]*x+H[1]*y+H[2])/z,(H[3]*x+H[4]*y+H[5])/z];
}
function buildCornersFromCenters(anchors,inv){
  const src=[[ANCHOR_CENTER,ANCHOR_CENTER],[CANON_SIZE-ANCHOR_CENTER,ANCHOR_CENTER],[ANCHOR_CENTER,CANON_SIZE-ANCHOR_CENTER],[CANON_SIZE-ANCHOR_CENTER,CANON_SIZE-ANCHOR_CENTER]];
  const dst=anchors.map(a=>[a.x,a.y]);
  const H=solveHomography(src,dst);
  if(!H)return null;
  const tl=projectPoint(H,ANCHOR_OUT,ANCHOR_OUT),tr=projectPoint(H,CANON_SIZE-ANCHOR_OUT,CANON_SIZE-ANCHOR_OUT-(CANON_SIZE-2*ANCHOR_OUT)),bl=projectPoint(H,ANCHOR_OUT,CANON_SIZE-ANCHOR_OUT),br=projectPoint(H,CANON_SIZE-ANCHOR_OUT,CANON_SIZE-ANCHOR_OUT);
  if(!tl||!tr||!bl||!br)return null;
  const TL=[tl[0]*inv,tl[1]*inv],TR=[tr[0]*inv,tr[1]*inv],BL=[bl[0]*inv,bl[1]*inv],BR=[br[0]*inv,br[1]*inv];
  const outSize=Math.round(Math.max(
    Math.hypot(TR[0]-TL[0],TR[1]-TL[1]),
    Math.hypot(BR[0]-BL[0],BR[1]-BL[1]),
    Math.hypot(BL[0]-TL[0],BL[1]-TL[1]),
    Math.hypot(BR[0]-TR[0],BR[1]-TR[1])
  ));
  return{TL,TR,BL,BR,outSize};
}
function buildCornersFromQuad(quad,inv){
  const TL=[quad.TL.x*inv,quad.TL.y*inv],TR=[quad.TR.x*inv,quad.TR.y*inv],BL=[quad.BL.x*inv,quad.BL.y*inv],BR=[quad.BR.x*inv,quad.BR.y*inv];
  const outSize=Math.round(Math.max(
    Math.hypot(TR[0]-TL[0],TR[1]-TL[1]),
    Math.hypot(BR[0]-BL[0],BR[1]-BL[1]),
    Math.hypot(BL[0]-TL[0],BL[1]-TL[1]),
    Math.hypot(BR[0]-TR[0],BR[1]-TR[1])
  ));
  return{TL,TR,BL,BR,outSize};
}
function runScanPass(scan,rawInv,seedCorners){
  const bounds=buildScanBoundsFromRawCorners(scan,seedCorners,rawInv);
  const prim=scanPrimaryAnchors(scan,bounds);
  const debug={adaptive:!!scan.adaptive,primCount:prim.anchors.length,cutoff:prim.cutoff||0,hasSeed:!!seedCorners};
  if(prim.anchors.length<3)return{corners:null,debug};
  const all=[prim.anchors[0],prim.anchors[1],prim.anchors[2]];
  debug.addBr=addBottomRightCorner(scan,all,prim.cutoff,bounds);
  if(!debug.addBr)return{corners:null,debug};
  const refined=refineCornersByEdges(scan,all);
  debug.edgeRefined=!!refined;
  if(refined){
    const quad=buildCornersFromQuad(refined,rawInv*scan.inv);
    if(quad)return{corners:quad,debug};
  }
  const center=buildCornersFromCenters(anchorsToCenters(all),rawInv*scan.inv);
  debug.centerFallback=!!center;
  return{corners:center,debug};
}
function scanBitmapForCorners(bm,inputToRawInv,seedCorners){
  const rawInv=Number.isFinite(inputToRawInv)&&inputToRawInv>0?inputToRawInv:1;
  const fastScan=prepareScanBitmap(bm,false);
  const fastRes=runScanPass(fastScan,rawInv,seedCorners);
  if(fastRes.corners)return fastRes;
  const adaptiveScan=prepareScanBitmap(bm,true);
  const adaptiveRes=runScanPass(adaptiveScan,rawInv,seedCorners);
  if(adaptiveRes.corners)return adaptiveRes;
  return{corners:null,debug:{fast:fastRes.debug,adaptive:adaptiveRes.debug}};
}

async function init(mode,modelBuf){
  localizerMode=(mode==='scanner'||mode==='hybrid')?mode:'yolo';
  if(localizerMode==='scanner'&&!modelBuf){postMessage({type:'ready',ep:'scanner'});return;}
  if(!modelBuf){postMessage({type:'ready',ep:'scanner'});localizerMode='scanner';return;}
  const hasGPU=typeof navigator!=='undefined'&&!!navigator.gpu;
  if(hasGPU){try{
    sess=await ort.InferenceSession.create(modelBuf,{executionProviders:['webgpu'],graphOptimizationLevel:'all',preferredOutputLocation:'cpu'});
    postMessage({type:'ready',ep:localizerMode==='hybrid'?'hybrid/webgpu':'webgpu'});return;
  }catch(e){}}
  sess=await ort.InferenceSession.create(modelBuf,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  postMessage({type:'ready',ep:localizerMode==='hybrid'?'hybrid/wasm':'wasm'});
}
async function inferBitmap(bm,patchOk,forceFull){
  if(localizerMode==='scanner'){
    const tScan=performance.now();
    const scanRes=scanBitmapForCorners(bm,1,null);
    const ms=performance.now()-tScan;
    bm.close();
    if(scanRes&&scanRes.corners){
      postMessage({type:'corners',corners:scanRes.corners,ms,sz:scanRes.corners.outSize,forceFull,loc:'scanner',debug:scanRes.debug||null});
      return;
    }
    postMessage({type:'corners',corners:null,ms,sz:0,forceFull,loc:'scanner',debug:scanRes?scanRes.debug:null});
    return;
  }
  const useFull=forceFull||!patchOk||(frameCount%FULL_EVERY===0);
  frameCount++;
  const SZ=useFull?FULL_SZ:FAST_SZ;
  const vw=bm.width,vh=bm.height,snapSc=Math.min(1,SZ/Math.max(vw,vh));
  const sw=Math.round(vw*snapSc),sh=Math.round(vh*snapSc);
  const snap=snapSc<1?await createImageBitmap(bm,{resizeWidth:sw,resizeHeight:sh,resizeQuality:'low'}):bm;
  const c=letterbox(snap,SZ);
  const t0=performance.now();
  const r=await sess.run({images:new ort.Tensor('float32',c.f32,[1,3,SZ,SZ])});
  const ms=performance.now()-t0;
  const inv=1/snapSc;
  let corners=assignCorners(parseOutput(r['output0'],c.scale,c.px,c.py,sw,sh),sw,sh,inv);
  let loc='yolo';
  if(localizerMode==='hybrid'){
    const refined=scanBitmapForCorners(bm,1,corners);
    if(refined){
      corners=refined;
      loc='hybrid';
    }else if(!corners){
      const fallback=scanBitmapForCorners(bm,1,null);
      if(fallback){
        corners=fallback;
        loc='scanner';
        corners.scanDebug=fallback.debug||null;
      }
    }
  }
  if(snap!==bm)snap.close();
  bm.close();
  postMessage({type:'corners',corners,ms,sz:SZ,forceFull,loc});
}

let busy=false;
self.onmessage=async e=>{
  if(e.data.type==='init'){await init(e.data.mode,e.data.model||null);return;}
  if(e.data.type==='frame'){
    if(busy){e.data.bitmap.close();return;}
    busy=true;
    try{await inferBitmap(e.data.bitmap,e.data.patchOk,e.data.forceFull);}catch(ex){console.error('[W]',ex);}
    busy=false;
    postMessage({type:'idle'});
  }
};`;

  app.getYoloWorkerSource = function getYoloWorkerSource() {
    return workerSource;
  };

  app.getLocalizerMode = function getLocalizerMode() {
    let raw = null;
    if (typeof global.__CAMDROP_LOCALIZER_MODE === 'string') {
      raw = global.__CAMDROP_LOCALIZER_MODE;
    } else {
      try {
        const params = new URL(global.location.href).searchParams;
        raw = params.get('loc') || params.get('localizer') || config.LOCALIZER_MODE;
      } catch (error) {
        raw = config.LOCALIZER_MODE;
      }
    }
    return raw === 'scanner' || raw === 'hybrid' || raw === 'contour' ? raw : 'yolo';
  };

  app.getYoloWorkerCount = function getYoloWorkerCount() {
    const rawCount = Number(global.__CAMDROP_YOLO_WORKERS);
    if (!Number.isFinite(rawCount)) {
      return 1;
    }
    return Math.max(1, Math.min(8, Math.round(rawCount)));
  };

  app.findIdleYoloWorker = function findIdleYoloWorker() {
    if (!state.yoloWorkers || !state.yoloWorkers.length) {
      return null;
    }
    for (let i = 0; i < state.yoloWorkers.length; i++) {
      const worker = state.yoloWorkers[i];
      if (worker && worker.__ready && !worker.__busy) {
        return worker;
      }
    }
    return null;
  };

  app.refreshYoloPoolState = function refreshYoloPoolState() {
    let ready = 0;
    let busy = 0;
    for (let i = 0; i < state.yoloWorkers.length; i++) {
      const worker = state.yoloWorkers[i];
      if (!worker) continue;
      if (worker.__ready) ready++;
      if (worker.__busy) busy++;
    }
    state.yoloWorker = state.yoloWorkers.length ? state.yoloWorkers[0] : null;
    state.yoloReadyWorkers = ready;
    state.yoloActiveCount = busy;
    state.workerIdle = !!app.findIdleYoloWorker();
  };

  app.getLocalizerQueueLimit = function getLocalizerQueueLimit(kind) {
    const key = kind === 'precise' ? '__CAMDROP_PRECISE_QUEUE_MAX' : '__CAMDROP_YOLO_QUEUE_MAX';
    const override = Number(global[key]);
    const fallback = kind === 'precise' ? config.PRECISE_QUEUE_MAX : config.YOLO_QUEUE_MAX;
    return Math.max(1, Math.round(Number.isFinite(override) ? override : fallback));
  };

  app.closeLocalizerTask = function closeLocalizerTask(task) {
    if (!task) {
      return;
    }
    if (task.bitmap) {
      try { task.bitmap.close(); } catch (_) {}
    }
    if (task.render) {
      try { task.render.close(); } catch (_) {}
    }
  };

  app.canAcceptLocalizerTask = function canAcceptLocalizerTask(kind) {
    const queue = kind === 'precise' ? state.preciseQueue : state.yoloQueue;
    const limit = app.getLocalizerQueueLimit(kind);
    return queue.length < limit || !!app.findIdleYoloWorker();
  };

  app.enqueueLocalizerTask = function enqueueLocalizerTask(kind, task) {
    const queue = kind === 'precise' ? state.preciseQueue : state.yoloQueue;
    const limit = app.getLocalizerQueueLimit(kind);
    if (queue.length >= limit) {
      if (kind === 'precise') {
        state.preciseQueueDropCount++;
      } else {
        state.yoloQueueDropCount++;
      }
      app.closeLocalizerTask(task);
      return false;
    }
    queue.push(task);
    return true;
  };

  app.disposeYoloWorkers = function disposeYoloWorkers() {
    for (let i = 0; i < state.yoloWorkers.length; i++) {
      try {
        state.yoloWorkers[i].terminate();
      } catch (_) {}
    }
    for (let i = 0; i < state.preciseQueue.length; i++) {
      const task = state.preciseQueue[i];
      if (task.bitmap) task.bitmap.close();
      if (task.render) task.render.close();
    }
    for (let i = 0; i < state.yoloQueue.length; i++) {
      const task = state.yoloQueue[i];
      if (task.bitmap) task.bitmap.close();
      if (task.render) task.render.close();
    }
    state.preciseQueue = [];
    state.yoloQueue = [];
    state.yoloWorker = null;
    state.yoloWorkers = [];
    state.yoloWorkerPoolSize = 0;
    state.yoloReadyWorkers = 0;
    state.yoloActiveCount = 0;
    state.workerIdle = true;
    state._yoloInitPending = 0;
    state._yoloInitResolve = null;
    state._yoloInitReject = null;
  };

  app.initWorker = function initWorker(modelBuf) {
    return new Promise((resolve, reject) => {
      app.disposeYoloWorkers();

      const targetCount = app.getYoloWorkerCount();
      const blob = new Blob([workerSource], { type: 'text/javascript' });
      const workerUrl = URL.createObjectURL(blob);
      state.yoloWorkers = [];
      state.yoloWorkerPoolSize = targetCount;
      state._yoloInitPending = targetCount;
      state._yoloInitResolve = resolve;
      state._yoloInitReject = reject;

      for (let i = 0; i < targetCount; i++) {
        const worker = new Worker(workerUrl);
        worker.__ready = false;
        worker.__busy = false;
        worker.__task = null;
        worker.onmessage = app.onWorkerMsg.bind(worker);
        worker.onerror = app.onWorkerErr.bind(worker);
        state.yoloWorkers.push(worker);
      }
      app.refreshYoloPoolState();

      const mode = app.getLocalizerMode();
      for (let i = 0; i < state.yoloWorkers.length; i++) {
        const msg = { type: 'init', mode };
        const transfer = [];
        if (modelBuf) {
          const modelCopy = i === state.yoloWorkers.length - 1 ? modelBuf : modelBuf.slice(0);
          msg.model = modelCopy;
          transfer.push(modelCopy);
        }
        state.yoloWorkers[i].postMessage(msg, transfer);
      }
    });
  };

  app.onWorkerMsg = function onWorkerMsg(event) {
    const worker = this;
    const { type } = event.data;

    if (type === 'ready') {
      worker.__ready = true;
      worker.__busy = false;
      state.currentEP = event.data.ep || state.currentEP;
      console.log('[Worker] ready, EP =', event.data.ep);
      app.refreshYoloPoolState();

      if (state._yoloInitPending > 0) {
        state._yoloInitPending--;
        if (state._yoloInitPending === 0 && state._yoloInitResolve) {
          const resolve = state._yoloInitResolve;
          state._yoloInitResolve = null;
          state._yoloInitReject = null;
          resolve();
        }
      }
      app.pumpYoloQueue();
      return;
    }

    if (type === 'corners') {
      state.yoloMs = event.data.ms;
      state.localizerSource = event.data.loc || state.localizerSource || '-';
      state.localizerDebug = event.data.debug || (event.data.corners && event.data.corners.scanDebug) || null;
      state.yoloFpsArr.push(1000 / (performance.now() - state.yoloLastT));
      if (state.yoloFpsArr.length > 10) {
        state.yoloFpsArr.shift();
      }
      state.yoloLastT = performance.now();

      const task = worker.__task;
      const taskSeq = task && Number.isFinite(task.captureSeq) ? task.captureSeq : 0;
      const shouldApplyResult = !taskSeq || taskSeq >= state.lastAppliedLocalizerSeq;
      if (event.data.corners) {
        if (shouldApplyResult) {
          state.lastAppliedLocalizerSeq = taskSeq || state.lastAppliedLocalizerSeq;
          state.lastCorners = event.data.corners;
          app.initPatches();
        }
        if (task && task.forceFull && shouldApplyResult) {
          state.forceFullDoneCount++;
          if (state.pendingRender) {
            state.pendingRender.close();
          }
          if (task.render) {
            state.pendingRender = task.render;
            task.render = null;
          }
          app.renderFine();
        } else if (task && task.render && shouldApplyResult && state.fineGl && typeof app.renderDeskew === 'function') {
          app.renderDeskew(state.fineGl, dom.dskCvs, event.data.corners, 1.0, task.render, config.FINE_RENDER_SIZE);
          task.render.close();
          task.render = null;
          state.lastDeskewTime = performance.now();
          dom.dskCvs.style.opacity = '1';
          if (typeof app.noteCodeSceneVisible === 'function') {
            app.noteCodeSceneVisible('deskew-visible-direct');
          }
          if (typeof app.enqueueRecognizeFrame === 'function') {
            app.enqueueRecognizeFrame();
          }
        } else if (!state.lastDeskewTime && shouldApplyResult && state.fineGl && typeof app.renderDeskew === 'function') {
          app.renderDeskew(state.fineGl, dom.dskCvs, state.lastCorners, 1.0, dom.video, config.FINE_RENDER_SIZE);
          state.lastDeskewTime = performance.now();
          dom.dskCvs.style.opacity = '1';
          if (typeof app.noteCodeSceneVisible === 'function') {
            app.noteCodeSceneVisible('deskew-visible-bootstrap');
          }
          if (typeof app.enqueueRecognizeFrame === 'function') {
            app.enqueueRecognizeFrame();
          }
        }
      } else {
        if (shouldApplyResult) {
          state.patches = null;
          state.lastAHash = null;
        }
        if (task && task.render) {
          task.render.close();
          task.render = null;
        }
      }
      return;
    }

    if (type === 'idle') {
      if (worker.__task && worker.__task.render) {
        worker.__task.render.close();
      }
      worker.__task = null;
      worker.__busy = false;
      app.refreshYoloPoolState();
      app.pumpYoloQueue();
    }
  };

  app.onWorkerErr = function onWorkerErr(error) {
    const worker = this;
    console.error('[Worker]', error.message || error);
    if (worker.__task && worker.__task.render) {
      worker.__task.render.close();
    }
    worker.__task = null;
    worker.__ready = false;
    worker.__busy = false;
    app.refreshYoloPoolState();
    if (state._yoloInitReject) {
      const reject = state._yoloInitReject;
      state._yoloInitResolve = null;
      state._yoloInitReject = null;
      reject(error);
      return;
    }
    app.pumpYoloQueue();
  };

  app.dispatchYoloTask = function dispatchYoloTask(worker, task) {
    if (!worker || !task || !task.bitmap) {
      return false;
    }
    worker.__busy = true;
    worker.__task = task;
    app.refreshYoloPoolState();
    worker.postMessage({
      type: 'frame',
      bitmap: task.bitmap,
      patchOk: !!task.patchOk,
      forceFull: !!task.forceFull,
    }, [task.bitmap]);
    return true;
  };

  app.takeNextLocalizerTask = function takeNextLocalizerTask() {
    const hasPrecise = state.preciseQueue.length > 0;
    const hasRaw = state.yoloQueue.length > 0;
    if (hasPrecise && hasRaw) {
      if (state.lastLocalizerDispatchKind === 'precise') {
        state.lastLocalizerDispatchKind = 'raw';
        return state.yoloQueue.shift();
      }
      state.lastLocalizerDispatchKind = 'precise';
      return state.preciseQueue.shift();
    }
    if (hasPrecise) {
      state.lastLocalizerDispatchKind = 'precise';
      return state.preciseQueue.shift();
    }
    if (hasRaw) {
      state.lastLocalizerDispatchKind = 'raw';
      return state.yoloQueue.shift();
    }
    return null;
  };

  app.pumpYoloQueue = function pumpYoloQueue() {
    while (true) {
      const worker = app.findIdleYoloWorker();
      if (!worker) {
        break;
      }
      const task = typeof app.takeNextLocalizerTask === 'function'
        ? app.takeNextLocalizerTask()
        : (state.preciseQueue.length ? state.preciseQueue.shift() : (state.yoloQueue.length ? state.yoloQueue.shift() : null));
      if (!task) {
        break;
      }
      app.dispatchYoloTask(worker, task);
    }
  };

  app.onModelLoaded = function onModelLoaded(name) {
    const mode = app.getLocalizerMode();
    const scannerOnly = mode === 'scanner' || mode === 'contour';
    dom.loadBtn.textContent = scannerOnly ? '算法定位' : ('模型 ' + name.replace('.onnx', ''));
    dom.loadBtn.classList.remove('loading');
    dom.loadBtn.classList.add('ready');
    ui.setStatus(scannerOnly ? '算法定位已就绪' : ('模型已就绪 [' + state.currentEP + ']'));
    dom.scanHint.classList.add('hidden');

    if (!state.scanning) {
      if (typeof app.resetPipelineCounters === 'function') {
        app.resetPipelineCounters();
      }
      state.scanning = true;
      state.coarseGl = app.initGL(state.offDsk, { filterMode: 'linear' });
      if (!state.coarseGl && typeof document !== 'undefined' && typeof document.createElement === 'function') {
        const fallbackCanvas = document.createElement('canvas');
        fallbackCanvas.width = Math.max(1, state.offDsk && state.offDsk.width ? state.offDsk.width : 1);
        fallbackCanvas.height = Math.max(1, state.offDsk && state.offDsk.height ? state.offDsk.height : 1);
        state.offDsk = fallbackCanvas;
        state.coarseGl = app.initGL(state.offDsk, { filterMode: 'linear' });
        if (state.coarseGl) {
          console.warn('[GL] coarse fallback canvas enabled');
        }
      }
      state.fineGl = app.initGL(dom.dskCvs, { filterMode: config.FINE_DESKEW_FILTER });
      console.log('[GL] coarse+fine contexts ready', { coarse: !!state.coarseGl, fine: !!state.fineGl, offscreen: typeof OffscreenCanvas !== 'undefined' && state.offDsk instanceof OffscreenCanvas });
      app.startVideoFrameLoop();
    }
  };

  app.startDeskewLoop = function startDeskewLoop() {
    app.startVideoFrameLoop();
  };

  app.stopVideoFrameLoop = function stopVideoFrameLoop() {
    if (!state.videoFrameLoopRunning) {
      return;
    }

    state.videoFrameLoopRunning = false;
    state.deskLoopRunning = false;
    if (state.videoFrameWatchdogId) {
      clearTimeout(state.videoFrameWatchdogId);
      state.videoFrameWatchdogId = 0;
    }

    if (!state.videoFrameLoopRequestId) {
      return;
    }

    if (state.videoFrameLoopMode === 'rvfc' && typeof dom.video.cancelVideoFrameCallback === 'function') {
      dom.video.cancelVideoFrameCallback(state.videoFrameLoopRequestId);
    } else {
      cancelAnimationFrame(state.videoFrameLoopRequestId);
    }
    state.videoFrameLoopRequestId = 0;
  };

  app.scheduleVideoFrameLoop = function scheduleVideoFrameLoop() {
    if (!state.scanning || !state.videoFrameLoopRunning) {
      return;
    }
    if (state.videoFrameWatchdogId) {
      clearTimeout(state.videoFrameWatchdogId);
      state.videoFrameWatchdogId = 0;
    }

    if (typeof dom.video.requestVideoFrameCallback === 'function') {
      state.videoFrameLoopMode = 'rvfc';
      state.videoFrameLoopRequestId = dom.video.requestVideoFrameCallback(app.handleVideoFrame);
      state.videoFrameWatchdogId = setTimeout(() => {
        if (!state.scanning || !state.videoFrameLoopRunning || state.videoFrameLoopMode !== 'rvfc' || !state.videoFrameLoopRequestId) {
          return;
        }
        try {
          if (typeof dom.video.cancelVideoFrameCallback === 'function') {
            dom.video.cancelVideoFrameCallback(state.videoFrameLoopRequestId);
          }
        } catch (_) {}
        state.videoFrameLoopRequestId = 0;
        state.videoFrameLoopMode = 'raf';
        state.videoFrameWatchdogId = 0;
        state.videoFrameLoopRequestId = requestAnimationFrame((now) => {
          app.handleVideoFrame(now, null);
        });
      }, 350);
      return;
    }

    state.videoFrameLoopMode = 'raf';
    state.videoFrameLoopRequestId = requestAnimationFrame((now) => {
      app.handleVideoFrame(now, null);
    });
  };

  app.handleVideoFrame = function handleVideoFrame(now, metadata) {
    state.videoFrameLoopRequestId = 0;
    if (state.videoFrameWatchdogId) {
      clearTimeout(state.videoFrameWatchdogId);
      state.videoFrameWatchdogId = 0;
    }
    if (!state.scanning || !state.videoFrameLoopRunning) {
      return;
    }

    try {
      state.videoFrameCount++;
      state.currentFrameToken = null;
      if (metadata && Number.isFinite(metadata.presentedFrames)) {
        const presentedToken = 'pf:' + String(Math.round(metadata.presentedFrames));
        if (state.lastObservedPresentedFrameToken !== presentedToken) {
          state.currentFrameToken = presentedToken;
          state.lastObservedPresentedFrameToken = presentedToken;
        }
      }
      if (state.currentFrameToken === null && metadata && Number.isFinite(metadata.mediaTime)) {
        const mediaToken = 'mt:' + Number(metadata.mediaTime).toFixed(6);
        if (state.lastObservedMediaTimeToken !== mediaToken) {
          state.currentFrameToken = mediaToken;
          state.lastObservedMediaTimeToken = mediaToken;
        }
      }
      if (state.currentFrameToken === null) {
        const t = Number(dom.video && dom.video.currentTime);
        if (Number.isFinite(t)) {
          const timeToken = 'ct:' + t.toFixed(6);
          if (state.lastObservedVideoTimeToken === timeToken) {
            state.currentFrameToken = 'vf:' + String(state.videoFrameCount);
          } else {
            state.currentFrameToken = timeToken;
            state.lastObservedVideoTimeToken = timeToken;
          }
        } else {
          state.currentFrameToken = 'vf:' + String(state.videoFrameCount);
        }
      }
      const videoReady = dom.video && dom.video.readyState >= 2 && dom.video.videoWidth > 0 && dom.video.videoHeight > 0;
      if (!videoReady) {
        return;
      }
      if (typeof app.markCameraFrameProgress === 'function') {
        app.markCameraFrameProgress();
      }
      if (app.getLocalizerMode() === 'contour') {
        app.patchTrackLoop(metadata);
        app.sendFrameLoop(metadata);
      } else {
        app.patchTrackLoop(metadata);
        if (typeof app.runRawBlurPrecheck === 'function') {
          app.runRawBlurPrecheck(dom.video);
        }
        app.deskewLoop(metadata);
        app.sendFrameLoop(metadata);
      }
    } finally {
      app.scheduleVideoFrameLoop();
    }
  };

  app.startVideoFrameLoop = function startVideoFrameLoop() {
    if (state.videoFrameLoopRunning) {
      return;
    }

    state.videoFrameLoopRunning = true;
    state.deskLoopRunning = true;
    app.scheduleVideoFrameLoop();
  };

  app.sendFrameLoop = function sendFrameLoop() {
    if (!state.scanning) {
      return;
    }
    if (state.cameraTunePending) {
      return;
    }
    if (dom.video.readyState < 2) {
      return;
    }
    if (app.getLocalizerMode() === 'contour') {
      if (typeof app.runContourLocalizer === 'function') {
        app.runContourLocalizer();
      }
      return;
    }
    if (!state.yoloWorkers.length) {
      return;
    }

    const coarseTrackHandled = state.currentFrameToken !== null
      && state.lastCoarseHandledVideoTime !== null
      && state.lastCoarseHandledVideoTime === state.currentFrameToken;
    // Skip raw YOLO only when deskewLoop() has already processed this exact video frame.
    if (coarseTrackHandled) {
      state.coarseTrackFreshCount++;
      if (typeof app.refreshPerfBar === 'function') {
        app.refreshPerfBar();
      }
      return;
    }

    if (typeof app.claimVideoFrame === 'function' && !app.claimVideoFrame('lastYoloVideoTime')) {
      return;
    }

    if (typeof app.shouldSampleCameraCapture === 'function' && !app.shouldSampleCameraCapture('raw')) {
      if (typeof app.refreshPerfBar === 'function') {
        app.refreshPerfBar();
      }
      return;
    }

    if (typeof app.canAcceptLocalizerTask === 'function' && !app.canAcceptLocalizerTask('raw')) {
      state.yoloQueueDropCount++;
      return;
    }

    const patchOk = !!state.patches;
    const needBootstrapRender = !state.lastDeskewTime;
    const captureSeq = ++state.localizerCaptureSeq;
    const capture = needBootstrapRender
      ? Promise.all([createImageBitmap(dom.video), createImageBitmap(dom.video)]).then(([bitmap, render]) => ({ bitmap, render }))
      : createImageBitmap(dom.video).then((bitmap) => ({ bitmap, render: null }));
    capture.then(({ bitmap, render }) => {
      const enqueued = typeof app.enqueueLocalizerTask === 'function'
        ? app.enqueueLocalizerTask('raw', {
            bitmap,
            render,
            patchOk,
            forceFull: false,
            captureToken: state.currentFrameToken,
            captureSeq,
          })
        : false;
      if (enqueued) {
        app.pumpYoloQueue();
      }
    }).catch((err) => {
      console.warn('[YoloQueue] capture failed:', err);
    });
  };
})(window);







