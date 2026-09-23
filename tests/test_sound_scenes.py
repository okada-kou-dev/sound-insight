"""音声時計と確定済み区間に従う追加描画の回帰確認。"""
import base64
from pathlib import Path
import shutil
import subprocess
import unittest


class SoundSceneTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is needed for canvas timing checks")
    def test_past_only_flags_seek_and_actual_draw_paths(self):
        module = base64.b64encode(Path("src/frontend/playback.js").read_bytes()).decode()
        script = r'''
import assert from 'node:assert/strict';
const {melColumnAtTime,spectralHistory,terrainBarRows,detectedIntervals,drawNebula,drawWaterfall,drawSpectrogram,NEBULA_STYLE} = await import('data:text/javascript;base64,MODULE');
const track={duration:10,mel:{width:311,height:64,frame_seconds:.064,hop_seconds:.032},
 windows:{starts:[0,.032,.064,.096],ends:[.64,.672,.704,.736],flags:[false,true,true,false]}};
assert.equal(melColumnAtTime(track.mel,0),-1);
assert.equal(melColumnAtTime(track.mel,.063),-1);
assert.equal(melColumnAtTime(track.mel,.064),0);
assert.deepEqual(spectralHistory(track,0),[]);
assert.ok(spectralHistory(track,.671).every(row=>!row.anomaly));
assert.ok(spectralHistory(track,.704).some(row=>row.anomaly));
assert.ok(spectralHistory(track,10).length<=65);
// After the 8-second history fills, existing rows must retain their columns.
for(let column=250;column<310;column++) {
 const at=c=>.064+c*.032+1e-9;
 const a=spectralHistory(track,at(column)).map(r=>r.column);
 const b=spectralHistory(track,at(column+1)).map(r=>r.column);
 assert.ok(b.every(c=>c%4===0));
 assert.ok(a.filter(c=>b.includes(c)).length>=Math.min(a.length,b.length)-1,
  'sliding history must not replace every sampled row');
}
for(const time of [.063,.064,.671,.704,.736,6,10]) {
 for(const row of spectralHistory(track,time)) assert.ok(row.seconds<=time+1e-12);
}
// Rewinding must remove later detections, even after fully rendering the clip.
spectralHistory(track,10);
assert.ok(spectralHistory(track,.671).every(row=>!row.anomaly));
const clean={...track,reference_label:1,decision:'anomaly_candidate',windows:{...track.windows,flags:[false,false,false,false]}};
assert.ok(spectralHistory(clean,10).every(row=>!row.anomaly));
const calls=[];
const g=new Proxy({createRadialGradient:()=>({addColorStop(){}}),createLinearGradient:()=>({addColorStop(){}})}, {
 get:(obj,key)=>key in obj?obj[key]:(...args)=>{assert.ok(args.every(v=>typeof v!=='number'||Number.isFinite(v)));calls.push([key,...args]);},
 set:(obj,key,value)=>{obj[key]=value;calls.push([key,value]);return true;}
});
const canvas={width:600,height:320,getContext:()=>g}, pixels=new Uint8Array(311*64).fill(150);
const frequencies=new Uint8Array(512).fill(150);
calls.length=0;drawNebula(canvas,frequencies,.704,true);assert.ok(!calls.some(c=>c[0]==='fillStyle'&&String(c[1]).startsWith('rgba(255,110,135,')));
const first=JSON.stringify(calls);calls.length=0;drawNebula(canvas,frequencies,.704,false);assert.equal(JSON.stringify(calls),first,'anomaly flags must not change nebula colors or motion');
assert.equal(calls.filter(c=>c[0]==='arc').length,950);
assert.ok(!calls.some(c=>c[0]==='stroke'),'nebula remains points');
const drawing=()=>({points:calls.filter(c=>c[0]==='arc'),colors:calls.filter(c=>c[0]==='fillStyle').slice(-950)});
// The same audio change must not throw particles farther as playback advances.
// Loud inputs must still differ: the response must not saturate early.
const energyMotion=[];
for(const time of [1,5,10,30]) {
 calls.length=0;drawNebula(canvas,new Uint8Array(512).fill(200),time,false);const low=drawing();
 calls.length=0;drawNebula(canvas,new Uint8Array(512).fill(255),time,false);const high=drawing();
 assert.ok(high.points.every((p,i)=>p[3]>low.points[i][3]));
 energyMotion.push(Math.sqrt(low.points.reduce((sum,p,i)=>sum+(p[1]-high.points[i][1])**2+(p[2]-high.points[i][2])**2,0)/low.points.length));
}
assert.ok(Math.max(...energyMotion)<Math.min(...energyMotion)*1.1,'elapsed time must not amplify identical audio changes');
const changes=new Float32Array(512).fill(.08);
calls.length=0;drawNebula(canvas,frequencies,.704,false);const stable=drawing();
calls.length=0;drawNebula(canvas,frequencies,.704,false,false,changes);const reactive=drawing();
assert.notDeepEqual(reactive.points,stable.points);
assert.ok(reactive.points.every((p,i)=>p[3]>stable.points[i][3]));
assert.ok(reactive.colors.every(c=>!String(c[1]).startsWith('rgba(255,')),'audio changes alone must not trigger anomaly red');
for(const level of [0,255])for(const change of [-1,0,1]) {
 calls.length=0;drawNebula(canvas,new Uint8Array(512).fill(level),8.4,false,false,new Float32Array(512).fill(change));
 assert.ok(drawing().points.every(p=>Math.abs(p[1])+p[3]<400&&Math.abs(p[2])+p[3]<225),'points remain inside the logical panel');
 if(level===255&&change===1) assert.ok(drawing().points.every(p=>Math.abs(p[3]-(NEBULA_STYLE.particleBase+NEBULA_STYLE.particleRange)*800/750)<1e-9),'strong changes reach the configured particle size');
 if(level===0&&change===0) assert.ok(drawing().points.every(p=>p[3]<1.35*800/750/2),'quiet particles must be less than half their previous radius');
}
// Changing panel size/DPR must only apply a uniform transform, never change the swirl.
calls.length=0;drawNebula(canvas,frequencies,5.4,false);const sized=drawing();
const baseScale=calls.find(c=>c[0]==='scale').slice(1);
calls.length=0;drawNebula({...canvas,width:1200,height:640},frequencies,5.4,false);
assert.deepEqual(drawing(),sized);
assert.deepEqual(calls.find(c=>c[0]==='scale').slice(1),baseScale.map(v=>v*2));
// At 48 kHz, bins 180+ are beyond the source's 8 kHz audible range.
const changedSpectrum=frequencies.slice();changedSpectrum.fill(255,180);
calls.length=0;drawNebula(canvas,frequencies,.704,false);const nebulaBefore=JSON.stringify(drawing());
calls.length=0;drawNebula(canvas,changedSpectrum,.704,false);assert.equal(JSON.stringify(drawing()),nebulaBefore);
changedSpectrum.fill(220,0,180);
calls.length=0;drawNebula(canvas,changedSpectrum,.704,false);assert.notEqual(JSON.stringify(drawing()),nebulaBefore);
calls.length=0;drawWaterfall(canvas,track,pixels,.704);assert.ok(calls.some(c=>c[0]==='strokeStyle'&&c[1]==='#ff8396'));
const anomalyWidths=calls.filter(c=>c[0]==='lineWidth').map(c=>c[1]);
assert.equal(anomalyWidths.at(-1),2.5,'only the leading ridge has a strong foreground stroke');
assert.ok(anomalyWidths.slice(0,-2).every(w=>w===1),'historical ridges stay thin');
assert.ok(calls.some(c=>c[0]==='strokeStyle'&&c[1]==='#081321'),'leading ridge has a separating dark outline');
calls.length=0;drawWaterfall(canvas,clean,pixels,.704);
assert.deepEqual(calls.filter(c=>c[0]==='lineWidth').map(c=>c[1]),anomalyWidths,'foreground emphasis does not depend on anomaly classification');
// The peak-to-baseline difference is 1.5 times the former .58 vertical scale.
calls.length=0;drawWaterfall(canvas,track,new Uint8Array(311*64),.064);
const flatY=calls.find(c=>c[0]==='moveTo')[2];
calls.length=0;drawWaterfall(canvas,track,new Uint8Array(311*64).fill(255),.064);
assert.ok(Math.abs(flatY-calls.find(c=>c[0]==='moveTo')[2]-canvas.height*.58*1.5)<1e-9);
calls.length=0;drawWaterfall(canvas,track,new Uint8Array(311*64).fill(255),10);
assert.ok(calls.filter(c=>c[0]==='moveTo'||c[0]==='lineTo').every(c=>c[2]>=0&&c[2]<=canvas.height));
calls.length=0;drawWaterfall(canvas,track,pixels,.671);assert.ok(!calls.some(c=>c[0]==='strokeStyle'&&c[1]==='#ff8396'));
// Solid bars use only completed data and the same detected anomaly intervals.
calls.length=0;drawWaterfall(canvas,track,pixels,.704,detectedIntervals(track,.704),'bars');
const barFrames=JSON.stringify(calls);
assert.equal(calls.filter(c=>c[0]==='fill').length,terrainBarRows(track,pixels,.704,[]).length*32*2,'front-facing bars show front and top faces');
assert.ok(calls.filter(c=>c[0]==='globalAlpha').every(c=>c[1]===1),'overlapping history must remain opaque');
assert.equal(calls.filter(c=>c[0]==='strokeStyle'&&c[1]==='#ffc0c9').length,32,'only the front row has bright edges');
assert.ok(calls.some(c=>c[0]==='fillStyle'&&c[1]==='#ff8396'),'detected intervals remain red in bar mode');
for (const time of [0,.064,.704,8.1,10]) {
 calls.length=0;drawWaterfall(canvas,track,new Uint8Array(311*64).fill(255),time,[],'bars');
 assert.ok(calls.filter(c=>c[0]==='moveTo'||c[0]==='lineTo').every(c=>c[1]>=0&&c[1]<=canvas.width&&c[2]>=0&&c[2]<=canvas.height),'solid terrain stays inside the panel');
}
const futureBars=pixels.slice();for(let row=0;row<64;row++)for(let col=30;col<311;col++)futureBars[row*311+col]=255;
calls.length=0;drawWaterfall(canvas,track,futureBars,.704,detectedIntervals(track,.704),'bars');
assert.equal(JSON.stringify(calls),barFrames,'bar mode must not reveal future columns');
calls.length=0;drawWaterfall(canvas,track,pixels,.671,detectedIntervals(track,.671),'bars');
assert.ok(!calls.some(c=>c[0]==='fillStyle'&&c[1]==='#ff8396'),'rewind must remove future red bars');
// Crossing a new-row boundary cannot reshape/resize any already drawn column.
const at=c=>.064+c*.032+1e-9;
const varied=pixels.map((v,i)=>(i*37)%256);
for(const column of [7,15,247,255,303]) {
 const a=terrainBarRows(track,varied,at(column),[]), b=terrainBarRows(track,varied,at(column+1),[]);
 for(const old of a) {
  const kept=b.find(row=>row.column===old.column);
  if(kept) assert.deepEqual(kept,old,'historical heights are immutable');
 }
}
// During flow, an existing cuboid undergoes a uniform translation, not zoom/jitter.
const vertices=time=>{
 calls.length=0;drawWaterfall(canvas,clean,varied,time,[],'bars');
 return calls.filter(c=>c[0]==='moveTo'||c[0]==='lineTo').slice(0,12).map(c=>c.slice(1));
};
const firstCell=vertices(at(15)), nextCell=vertices(at(16));
const dx=nextCell[0][0]-firstCell[0][0],dy=nextCell[0][1]-firstCell[0][1];
assert.ok(Math.abs(dx)<1e-9&&dy<0,'front view history flows straight back without lateral drift');
assert.ok(nextCell.every((p,i)=>Math.abs(p[0]-firstCell[i][0]-dx)<1e-9&&Math.abs(p[1]-firstCell[i][1]-dy)<1e-9),'old geometry must flow rigidly even as a new row enters');
// A brighter future must not alter the current scene.
calls.length=0;drawWaterfall(canvas,track,pixels,.704);const before=JSON.stringify(calls);
const changed=pixels.slice();for(let row=0;row<64;row++)for(let col=30;col<311;col++)changed[row*311+col]=255;
calls.length=0;drawWaterfall(canvas,track,changed,.704);assert.equal(JSON.stringify(calls),before);
// Only completed columns enter the interpolation source, including after rewind.
const cropCalls=[],cropContext={drawImage:(...args)=>cropCalls.push(args)};
const scratch={getContext:()=>cropContext},source={width:311,height:64};
calls.length=0;drawSpectrogram(canvas,track,source,scratch,.063);
assert.equal(cropCalls.length,0);
for(const time of [10,.704,.064]) {
 drawSpectrogram(canvas,track,source,scratch,time);
 const count=melColumnAtTime(track.mel,time)+1;
 assert.equal(scratch.width,count);assert.equal(cropCalls.at(-1)[3],count);
 assert.equal(g.imageSmoothingEnabled,true);assert.equal(g.imageSmoothingQuality,'high');
 assert.equal(cropContext.imageSmoothingEnabled,false);
 assert.ok(calls.filter(c=>c[0]==='drawImage').at(-1)[4]<=canvas.width*time/track.duration+1e-9);
}
console.log('scene drawing: causal frames, flags, pause/seek determinism, no label coloring: passed');
'''.replace("MODULE", module)
        result = subprocess.run([shutil.which("node"), "--input-type=module"], input=script,
                                capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
