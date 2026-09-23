"""Playback conversion and audio-clock checks use synthetic data only."""
import base64
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from src.audio import AudioClip, FeatureSet
from src import playback


def detail_result(float_pcm=False):
    samples = np.full(16000 * 8, 0.125, dtype=np.float32)
    samples[12345] = 0.7 if float_pcm else 0.75
    samples[12346] = -0.625
    clip = AudioClip(samples, 16000, 1, 8.0, "synthetic", "selected")
    starts = np.array([0.0, 0.032, 0.064])
    features = FeatureSet(np.zeros((3, 320), dtype=np.float32),
                          np.linspace(-100, 20, 64 * 7, dtype=np.float32).reshape(64, 7),
                          starts, starts + 0.192)
    return {"status": "success", "source_name": "synthetic.wav", "model_id": "synthetic_only",
            "run_id": "synthetic_run", "item_index": 0, "score": 0.4, "threshold": 0.5,
            "decision": "within_reference", "_clip": clip, "_features": features,
            "_window_scores": np.array([0.2, 0.5, 0.5])}


class PlaybackTests(unittest.TestCase):
    def test_pcm16_roundtrip_preserves_every_sample_and_transient(self):
        result = detail_result()
        track = playback.make_track(result)
        self.assertEqual(track["pcm_format"], "s16le")
        reconstructed = np.frombuffer(base64.b64decode(track["pcm"]), dtype="<i2").astype(np.float32) / 32768
        np.testing.assert_array_equal(reconstructed, result["_clip"].samples)
        self.assertEqual(max(track["waveform"]["maximum"]), 0.75)
        self.assertEqual(min(track["waveform"]["minimum"]), -0.625)
        self.assertLess(len(json.dumps(track)), 450_000)
        self.assertNotIn("vectors", track)

    def test_float_pcm_is_not_clipped_quantized_or_normalized(self):
        result = detail_result(float_pcm=True)
        result["_clip"].samples[50] = np.float32(1.25)
        track = playback.make_track(result)
        self.assertEqual(track["pcm_format"], "f32le")
        reconstructed = np.frombuffer(base64.b64decode(track["pcm"]), dtype="<f4")
        np.testing.assert_array_equal(reconstructed, result["_clip"].samples)
        self.assertEqual(reconstructed[50], 1.25)

    def test_raw_scores_are_label_independent_without_relative_ranking(self):
        result = detail_result()
        track = playback.make_track(dict(result, reference_label=0, comparison="正常を正しく判定"))
        changed = playback.make_track(dict(result, reference_label=1, threshold=100, comparison="見逃し"))
        self.assertEqual(track["windows"], changed["windows"])
        self.assertEqual(track["pcm"], changed["pcm"])
        self.assertEqual(track["windows"]["scores"], [0.2, 0.5, 0.5])
        self.assertNotIn("relative", track["windows"])
        result["_window_scores"] = np.ones(3)
        constant = playback.make_track(result)
        self.assertEqual(constant["windows"]["scores"], [1.0, 1.0, 1.0])
        json.dumps(constant, allow_nan=False)

    def test_failure_missing_details_and_invalid_window_are_rejected(self):
        for result in ({"status": "invalid_input"}, {"status": "error"}, {"status": "success"}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                playback.make_track(result)
        result = detail_result()
        result["_window_scores"][1] = np.nan
        with self.assertRaises(ValueError):
            playback.make_track(result)
        result = detail_result()
        result["_features"].ends[-1] = 8.1
        with self.assertRaises(ValueError):
            playback.make_track(result)

    def test_user_text_is_only_data_and_only_completion_events_return_to_python(self):
        title = '</script><img src=x onerror="alert(1)">'
        track = playback.make_track(dict(detail_result(), source_name=title))
        with patch("src.playback._component") as component:
            playback.show_player([track], key="run", autoplay=True)
        call = component.return_value.call_args.kwargs
        self.assertEqual(call["data"]["tracks"][0]["title"], title)
        self.assertEqual([key for key in call if key.startswith("on_")], ["on_completed_change"])
        self.assertEqual(call["default"], {"completed": []})
        self.assertNotIn(title, (playback.FRONTEND / "playback.html").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            playback.show_player([], key="empty")

    def test_installed_streamlit_inline_component_mounts_and_reruns(self):
        source = '''
from src.playback import make_track, show_player
import numpy as np
from src.audio import AudioClip, FeatureSet
clip = AudioClip(np.full(128000, .125, dtype=np.float32),16000,1,8.,"s","c")
features = FeatureSet(np.zeros((1,320)),np.zeros((64,5)),np.array([0.]),np.array([.192]))
result = {"status":"success","_clip":clip,"_features":features,"_window_scores":np.array([.1]),"score":.1,"threshold":.2,"decision":"within_reference"}
show_player([make_track(result)], key="synthetic", autoplay=False)
'''
        app = AppTest.from_string(source).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("bidi_component")), 1)
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("bidi_component")), 1)

    def test_component_registers_in_each_independent_runtime(self):
        source = '''
from src.playback import _component
_component()(key="one",data={"tracks":[],"instance_id":"one"})
_component()(key="two",data={"tracks":[],"instance_id":"two"})
'''
        for runtime_number in range(3):
            with self.subTest(runtime=runtime_number):
                app = AppTest.from_string(source).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.get("bidi_component")), 2)
                app.run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.get("bidi_component")), 2)

    @unittest.skipUnless(shutil.which("node"), "JavaScript clock test requires Node; browser checks are separate")
    def test_javascript_audio_clock_rates_pause_seek_and_async_cleanup(self):
        # Execute the actual transport module against a controlled AudioContext.
        # No mocked timers drive the clock; all positions derive from context.currentTime.
        module = base64.b64encode((playback.FRONTEND / "playback.js").read_bytes()).decode("ascii")
        script = r'''
import assert from "node:assert/strict";
const {PlaybackTransport, decodePCM, windowAtTime, revealFraction, completionIds, melColor, detectedIntervals} = await import("data:text/javascript;base64,MODULE");
const sources = [];
let frequencyValue = 123;
const ctx = {state:"running",currentTime:0,destination:{},closeCount:0,
  resume:async function(){this.state="running";},close:async function(){this.closeCount++;this.state="closed";},
  createAnalyser:()=>({frequencyBinCount:512,connect(){},disconnect(){},getByteFrequencyData(out){out.fill(frequencyValue);}}),
  createBuffer:(channels,length,sampleRate)=>({copyToChannel:()=>{}}),
  createBufferSource:()=>{const source={playbackRate:{calls:[],setValueAtTime(rate,time){this.calls.push([rate,time]);}},connect:()=>{},disconnect:()=>{},start:()=>{},stop:()=>{}};sources.push(source);return source;}};
const pcm = Buffer.alloc(8);[4096,-8192,16384,0].forEach((value,index)=>pcm.writeInt16LE(value,index*2));
const track={id:"test",duration:10,sample_count:4,sample_rate:16000,pcm_format:"s16le",pcm:pcm.toString("base64")};
assert.deepEqual(Array.from(decodePCM(track)),[.125,-.25,.5,0]);
let ends=0;const transport=new PlaybackTransport(()=>ctx,()=>{},()=>ends++);
transport.setTrack(track);await transport.play();
assert.equal(transport.analyser.fftSize,1024);assert.equal(transport.analyser.smoothingTimeConstant,.22);
assert.equal(transport.sampleSpectrum()[0],123);
assert.equal(transport.spectrumChange[0],0,'first frame must not invent an onset');
frequencyValue=174;transport.sampleSpectrum();assert.ok(Math.abs(transport.spectrumChange[0]-.2)<1e-6);
const frozenChange=transport.spectrumChange[0];
transport.pause();frequencyValue=100;assert.equal(transport.sampleSpectrum()[0],174);
assert.equal(transport.spectrumChange[0],frozenChange);await transport.play();
transport.sampleSpectrum();assert.ok(transport.spectrumChange[0]<0);
ctx.currentTime=2;assert.equal(transport.position(),2);
transport.setRate(.25);ctx.currentTime=6;assert.equal(transport.position(),3);
transport.setRate(10);ctx.currentTime=6.5;assert.equal(transport.position(),8);
assert.deepEqual(sources[1].playbackRate.calls,[[1,0],[.25,2],[10,6]]);
transport.pause();ctx.currentTime=100;assert.equal(transport.position(),8);
await transport.play();const oldEnded=sources.at(-1).onended;
await transport.seek(1);assert.equal(transport.position(),1);assert.equal(transport.playing,true);
oldEnded();assert.equal(transport.playing,true);assert.equal(ends,0);
await transport.seek(10);assert.equal(transport.spectrum[0],0);assert.equal(transport.position(),10);assert.equal(transport.playing,false);
assert.ok(transport.spectrumChange.every(v=>v===0));assert.equal(transport.hasSpectrum,false);
await transport.play();assert.equal(transport.position(),0);assert.equal(transport.playing,true);
const timeline={windows:{ends:[.192,.224,.256]}};
assert.equal(windowAtTime(timeline,0),-1);
assert.equal(windowAtTime(timeline,.191),-1);
assert.equal(windowAtTime(timeline,.192),0);
assert.equal(windowAtTime(timeline,.24),1);
assert.equal(revealFraction(0,10),0);assert.equal(revealFraction(2.5,10),.25);assert.equal(revealFraction(12,10),1);
assert.deepEqual(completionIds([],track,9.99),[]);
assert.deepEqual(completionIds([],track,10),["test"]);
const done=["test"];assert.equal(completionIds(done,track,10),done);
const temporal={windows:{starts:[0,.032,.064,.096],ends:[.64,.672,.704,.736],flags:[false,true,true,false]}};
assert.deepEqual(detectedIntervals(temporal,.671),[]);
assert.deepEqual(detectedIntervals(temporal,.672),[[.032,.672]]);
assert.deepEqual(detectedIntervals(temporal,.704),[[.032,.704]]);
assert.deepEqual(detectedIntervals(temporal,.736),[[.032,.704]]);
assert.deepEqual(detectedIntervals({windows:{ends:[1]}},2),[]);
assert.notDeepEqual(melColor(.2),melColor(.8));
assert.notEqual(melColor(.8)[0],melColor(.8)[1]);
transport.stop();assert.equal(transport.position(),0);assert.equal(transport.playing,false);
await transport.play();const countBeforeSwitch=sources.length;
for(const id of ['previous3','previous2','previous1']) {
 transport.setTrack({...track,id});
 assert.equal(transport.playing,false);assert.equal(transport.position(),0);
}
assert.equal(sources.length,countBeforeSwitch,'manual switching must not start audio');
await transport.play();assert.equal(transport.playing,true);
transport.dispose();transport.dispose();assert.equal(ctx.closeCount,1);
let resumeResolve;let started=0;
const suspended={...ctx,state:"suspended",closeCount:0,resume:()=>new Promise(resolve=>{resumeResolve=resolve;}),createBufferSource:()=>{started++;return sources[0];}};
const pending=new PlaybackTransport(()=>suspended);pending.setTrack(track);
const promise=pending.play();pending.dispose();resumeResolve();await promise;
assert.equal(started,0);assert.equal(pending.playing,false);
console.log("clock: 1x -> .25x -> 10x; pause/seek/stale-ended/dispose: passed");
'''.replace("MODULE", module)
        completed = subprocess.run([shutil.which("node"), "--input-type=module"], input=script,
                                   capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
