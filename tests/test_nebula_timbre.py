"""同じ音量の低音・高音・広帯域音が異なる点群になることを確認。"""
import base64
import json
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np


class NebulaTimbreTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is needed for canvas checks")
    def test_equal_level_tones_noise_and_frequency_mapping(self):
        rate, count = 48000, 1024
        t = np.arange(count) / rate
        rng = np.random.default_rng(42)
        sounds = [np.sin(2 * np.pi * hz * t) for hz in (187.5, 3000)]
        sounds.append(rng.normal(size=count))
        sounds.append(sounds[0] + .5 * sounds[2])
        spectra = []
        for sound in sounds:
            sound = sound / np.sqrt(np.mean(sound ** 2)) * .03
            magnitude = np.abs(np.fft.rfft(sound * np.hanning(count)))[:512] / count
            db = 20 * np.log10(np.maximum(magnitude, 1e-12))
            spectra.append(np.clip((db + 100) / 90 * 255, 0, 255).astype(int).tolist())
        module = base64.b64encode(Path("src/frontend/playback.js").read_bytes()).decode()
        script = r'''
import assert from 'node:assert/strict';
const {drawNebula,nebulaBands}=await import('data:text/javascript;base64,MODULE');
const spectra=SPECTRA;
let points=[];
const g=new Proxy({createRadialGradient:()=>({addColorStop(){}}),arc:(x,y,r)=>points.push([x,y,r])},
 {get:(o,k)=>k in o?o[k]:()=>{}});
const canvas={width:800,height:450,getContext:()=>g};
function render(spectrum,time=0,rate=48000) {
 points=[]; drawNebula(canvas,spectrum,time,false,false,null,rate);
 return points.map(p=>p.slice());
}
const stats=p=>{
 const weight=p.reduce((s,v)=>s+v[2]**2,0);
 return [Math.sqrt(p.reduce((s,v)=>s+v[0]**2*v[2]**2,0)/weight),
         Math.sqrt(p.reduce((s,v)=>s+v[1]**2*v[2]**2,0)/weight)];
};
const scenes=spectra.map(s=>render(s)), extents=scenes.map(stats);
// Measure principal axes, not bounding-box height: a diagonal line has height
// but no thickness. Bright particles must form an oval for every sound.
for (const points of scenes) {
 const weight=points.reduce((s,p)=>s+p[2]**2,0);
 const mx=points.reduce((s,p)=>s+p[0]*p[2]**2,0)/weight;
 const my=points.reduce((s,p)=>s+p[1]*p[2]**2,0)/weight;
 const xx=points.reduce((s,p)=>s+(p[0]-mx)**2*p[2]**2,0)/weight;
 const yy=points.reduce((s,p)=>s+(p[1]-my)**2*p[2]**2,0)/weight;
 const xy=points.reduce((s,p)=>s+(p[0]-mx)*(p[1]-my)*p[2]**2,0)/weight;
 const spread=Math.hypot(xx-yy,2*xy);
 assert.ok(Math.sqrt((xx+yy-spread)/(xx+yy+spread))>.4,
  'the cloud must have depth, not collapse into a diagonal sliver');
}
assert.ok(extents[1][1]>extents[0][1]*1.25,'high tones must visibly spread beyond the fuller low-tone core');
// The frequency distribution must affect the silhouette, not just reshuffle dots.
assert.ok(extents[1][1]/extents[1][0]>extents[0][1]/extents[0][0]*1.35,
 'low and high tones need different visible proportions');
const prominent=p=>p.filter(v=>v[2]>.9).length;
assert.ok(prominent(scenes[2])>Math.max(prominent(scenes[0]),prominent(scenes[1]))*1.5,
 'broadband sound must activate more of the cloud than a tone');
assert.ok(prominent(scenes[3])>prominent(scenes[0])*1.5,
 'adding noise to the same low tone must activate a wider range of particles');
for(const p of scenes) assert.ok(p.every(v=>v.every(Number.isFinite)&&Math.abs(v[0])+v[2]<400&&Math.abs(v[1])+v[2]<225));
const silence=render(new Uint8Array(512));
assert.ok(silence.every(p=>p[2]<.5),'silence must not make energetic particles');
assert.deepEqual(render(spectra[0],2),render(spectra[0],2),'pause/seek must reproduce the same frame');
const isolated=hz=>{const a=new Uint8Array(512);a[Math.round(hz/48000*1024)]=210;return a;};
assert.ok(stats(render(isolated(6000)))[1]>stats(render(isolated(200)))[1]*1.25,
 'higher audible bins must not be discarded');
const bandAtRate=rate=>{
 const s=new Uint8Array(512);s[Math.round(3000/rate*1024)]=210;
 const bands=nebulaBands(s,null,rate).bands;
 return bands.reduce((best,b,i)=>b.power>bands[best].power?i:best,0);
};
assert.ok(Math.abs(bandAtRate(16000)-bandAtRate(48000))<=1,
 'the same pitch must stay in the same region across AudioContext sample rates');
const motions=[0,5,10,30].map(time=>{
 const a=render(spectra[0],time), b=render(spectra[1],time);
 return Math.sqrt(a.reduce((s,p,i)=>s+(p[0]-b[i][0])**2+(p[1]-b[i][1])**2,0)/a.length);
});
assert.ok(Math.max(...motions)/Math.min(...motions)<1.15,'time must not amplify the same sound change');
console.log(JSON.stringify({extents,prominent:scenes.map(prominent),motions}));
'''.replace("MODULE", module).replace("SPECTRA", json.dumps(spectra))
        result = subprocess.run([shutil.which("node"), "--input-type=module"], input=script,
                                capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
