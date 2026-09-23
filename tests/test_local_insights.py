"""Attribution additivity, causal display, axis units and local/public isolation."""
import base64
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from src import nearest, playback
from src.audio import FeatureSet
from tests.test_playback import detail_result


def sample_result():
    model = nearest.NearestModel('insights-synthetic', np.zeros(64), np.ones(64),
                                np.zeros((5, 64)), .5, {}, {'window_threshold': .6})
    mel = np.zeros((64, 249))
    mel[0] = np.linspace(0, 8, 249)
    mel[1] = np.linspace(5, 0, 249)
    source = FeatureSet(np.zeros((1, 64)), mel, np.array([0.]), np.array([.64]))
    result = detail_result()
    scores = nearest.score(model, source)
    result.update(score=scores['score'], threshold=scores['threshold'], decision=scores['decision'],
                  _features=scores['features'], _window_scores=scores['window_scores'],
                  _window_flags=scores['window_flags'], _window_threshold=scores['window_threshold'],
                  _window_policy=scores['window_policy'])
    return model, result


class LocalInsightsTests(unittest.TestCase):
    def test_contributions_choose_a_whole_window_and_preserve_scores(self):
        model, result = sample_result()
        bands = nearest.band_contributions(model, result['_features'])
        np.testing.assert_allclose(bands.sum(axis=1), result['_window_scores'], rtol=1e-12)
        self.assertAlmostEqual(bands.mean(axis=0).sum(), result['score'], places=12)
        self.assertTrue(np.all(bands[:, 2:] == 0))
        # An independent minimum per band would understate the true score.
        self.assertTrue(np.any(np.diff(bands[:, 0]) > 0))
        self.assertTrue(np.any(np.diff(bands[:, 1]) < 0))
        altered = result['_features'].log_mel.copy()
        altered[:, 180:] += 100
        future = replace(result['_features'], log_mel=altered)
        np.testing.assert_array_equal(nearest.band_contributions(model, future)[:160], bands[:160])
        again = nearest.score(model, result['_features'])
        np.testing.assert_array_equal(again['window_scores'], result['_window_scores'])
        self.assertEqual(again['decision'], result['decision'])

    def test_optional_display_payload_has_common_scale_and_validates_additivity(self):
        model, result = sample_result()
        bands = nearest.band_contributions(model, result['_features'])
        public = playback.make_track(result)
        local = playback.make_track(result, contributions=bands)
        self.assertNotIn('contribution', public)
        for key in ('windows', 'score', 'threshold', 'decision', 'pcm', 'mel'):
            self.assertEqual(local[key], public[key])
        heatmap = local['contribution']
        self.assertEqual(heatmap['maximum'], .15)
        self.assertEqual(len(base64.b64decode(heatmap['pixels'])), len(bands) * 64)
        for broken in (bands * .9, bands[:, :63], np.full_like(bands, np.nan), -bands):
            with self.subTest(shape=broken.shape), self.assertRaises(ValueError):
                playback.make_track(result, contributions=broken)
        with patch('src.playback._component') as component:
            playback.show_player([public], key='public')
            self.assertFalse(component.return_value.call_args.kwargs['data']['local_insights'])
            playback.show_player([local], key='local', local_insights=True)
            self.assertTrue(component.return_value.call_args.kwargs['data']['local_insights'])

    def test_both_editions_compute_contributions_once_per_inspection(self):
        for namespace in ('local', 'public'):
            with self.subTest(namespace=namespace):
                _, result = sample_result()
                source = f'''
from src.ui import inspection_panel
from tests.test_local_insights import sample_result
model, _ = sample_result()
inspection_panel(model, [dict(path="synthetic.wav", name="Sample 1", label=0)], namespace="{namespace}")
'''
                with patch('src.service.inspect_one', return_value=result) as inspect, \
                     patch('src.nearest.band_contributions', wraps=nearest.band_contributions) as explain, \
                     patch('src.playback.show_player', return_value={'completed': []}) as player:
                    app = AppTest.from_string(source).run()
                    app.radio[0].set_value('1件だけ検査').run()
                    next(b for b in app.button if b.label == '単発検査を実行').click().run()
                    self.assertEqual(len(app.exception), 0)
                    self.assertEqual(inspect.call_count, 1)
                    self.assertEqual(explain.call_count, 1)
                    self.assertTrue(player.call_args.kwargs['local_insights'])
                    self.assertIn('contribution', player.call_args.args[0][0])
                    app.run()
                    self.assertEqual(inspect.call_count, 1)
                    self.assertEqual(explain.call_count, 1)

    @unittest.skipUnless(shutil.which('node'), 'Node required for drawing checks')
    def test_axis_coordinates_and_causal_contribution_cropping(self):
        module = base64.b64encode((playback.FRONTEND / 'playback.js').read_bytes()).decode()
        script = r'''
import assert from 'node:assert/strict';
const {axisTicks,timeTicks,drawContribution,contributionColor,drawAnomalyBars,detectedIntervals,drawWaterfall}=await import('data:text/javascript;base64,MODULE');
const track={duration:10,waveform:{peak:.25},mel:{hop_seconds:.032},contribution:{width:3,height:64},
 windows:{starts:[0,.032,.064],ends:[.64,.672,.704],maximum:2,threshold:.7,flags:[true,true,false]}};
assert.deepEqual(timeTicks(10),[0,2,4,6,8,10]);
assert.deepEqual(timeTicks(8),[0,2,4,6,8]);
for(const name of ['score','wave','mel','contribution']) {
 const axes=axisTicks(name,track);assert.equal(axes.xUnit,'s');assert.deepEqual(axes.x.map(t=>t.value),timeTicks(10));
 assert.ok(axes.y.every(t=>t.position>=0&&t.position<=1));
}
assert.deepEqual(axisTicks('wave',track).y.map(t=>t.value),[.25,0,-.25]);
assert.equal(axisTicks('mel',track).yUnit,'kHz');
assert.deepEqual(axisTicks('mel',track).y,axisTicks('contribution',track).y);
for(const mode of ['bars','line']) {
 const axes=axisTicks('waterfall',track,6,mode);assert.equal(axes.xUnit,'kHz');assert.equal(axes.yUnit,'dB');
 assert.ok(!('depth' in axes));assert.ok(!('depthUnit' in axes));
 assert.ok(axes.y.every(t=>t.position>=0&&t.position<=1));
}
const calls=[],crops=[],ctx={drawImage:(...a)=>calls.push(a)},crop={drawImage:(...a)=>crops.push(a)};
const canvas={width:1000,height:160,getContext:()=>ctx},source={},scratch={getContext:()=>crop};
drawContribution(canvas,track,source,scratch,.639);assert.equal(calls.length,0);assert.equal(crops.length,0);
for(const [time,count] of [[.64,1],[.704,3],[.65,1]]) {
 drawContribution(canvas,track,source,scratch,time);assert.equal(scratch.width,count);assert.equal(crops.at(-1)[3],count);
 const destination=calls.at(-1);assert.ok(destination[1]+destination[3]<=time/track.duration*canvas.width+1e-9);
 assert.equal(crop.imageSmoothingEnabled,false);
}
assert.deepEqual(contributionColor(-1),contributionColor(0));assert.deepEqual(contributionColor(2),contributionColor(1));
// Only a thin top bar: normal windows and unreached windows do not produce fills.
const fills=[],barCanvas={width:1000,height:160,getContext:()=>({fillRect:(...v)=>fills.push(v)})};
for(const [time,count] of [[.639,0],[.704,1],[.65,1],[0,0]]) {
 fills.length=0;drawAnomalyBars(barCanvas,detectedIntervals(track,time),time,10,2);
 assert.equal(fills.length,count);
 for(const [x,y,w,h] of fills){assert.equal(y,0);assert.equal(h,6);assert.ok(x+w<=time*100+1e-9);}
}
const normal={...track,windows:{...track.windows,flags:[false,false,false]}};
fills.length=0;drawAnomalyBars(barCanvas,detectedIntervals(normal,10),10,10);assert.equal(fills.length,0);
// The local leading LINE is slightly quieter; public and historical lines retain their widths.
const terrain={...track,mel:{width:100,height:64,frame_seconds:.064,hop_seconds:.032}};
function strokes(local){
 const widths=[];const ctx=new Proxy({stroke(){widths.push(this.lineWidth)},createRadialGradient(){return {addColorStop(){}}}},
  {get:(o,k)=>k in o?o[k]:()=>{}});
 drawWaterfall({width:600,height:200,getContext:()=>ctx},terrain,new Uint8Array(6400).fill(140),2,[],'line',local);
 return widths;
}
const before=strokes(false),after=strokes(true);assert.ok(before.length>2);
assert.deepEqual(after.slice(0,-2),before.slice(0,-2));
assert.deepEqual(before.slice(-2),[5,2.5]);
after.slice(-2).forEach((v,i)=>assert.equal(v,before.slice(-2)[i]*.85));
console.log('axes, causal crop/rewind, anomaly bars and local LINE styling passed');
'''.replace('MODULE', module)
        result = subprocess.run([shutil.which('node'), '--input-type=module'], input=script,
                                capture_output=True, text=True, encoding='utf-8', timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
