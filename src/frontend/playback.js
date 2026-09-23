// Completion IDs are the only server events. Audio, seeking and rate changes stay local.
const instances = new WeakMap();

// SONIC NEBULAの調整値。音の分類やAI判定には使用しない。
export const NEBULA_STYLE = Object.freeze({
  changeSensitivity: 22, // 音の変化への感度
  angleResponse: 1.25,   // 変化によるねじれ
  levelSpread: .1,       // 音の強さによる伸縮
  changeSpread: 1.9,     // 音の変化による伸縮
  particleBase: .35,     // 粒の最小半径
  particleRange: 3,      // 粒の大小差
  levelReference: .0005, // 小さくすると弱い周波数も見えやすくなる
  viewDepth: .7,         // 輪郭の縦の厚み（大きいほど正面に近い）
});
const SPECTRUM_MIN_DB = -100, SPECTRUM_MAX_DB = -10;

function decodeBytes(encoded) {
  const binary = atob(encoded);
  return Uint8Array.from(binary, char => char.charCodeAt(0));
}

export function decodePCM(track) {
  const bytes = decodeBytes(track.pcm);
  const width = track.pcm_format === "s16le" ? 2 : track.pcm_format === "f32le" ? 4 : 0;
  if (!width || bytes.byteLength !== track.sample_count * width) throw new Error("音声データの長さ・形式が不正です。");
  const view = new DataView(bytes.buffer);
  const samples = new Float32Array(track.sample_count);
  for (let i = 0; i < samples.length; i++) {
    samples[i] = width === 2 ? view.getInt16(i * 2, true) / 32768 : view.getFloat32(i * 4, true);
    if (!Number.isFinite(samples[i])) throw new Error("音声データに非有限値があります。");
  }
  return samples;
}

// This clock follows AudioContext time, including immediate rate changes.
export class PlaybackTransport {
  constructor(contextFactory, onChange = () => {}, onEnded = () => {}) {
    this.contextFactory = contextFactory;
    this.onChange = onChange;
    this.onEnded = onEnded;
    this.context = null;
    this.analyser = null;
    this.spectrum = new Uint8Array(512);
    this.spectrumChange = new Float32Array(512);
    this.previousSpectrum = new Uint8Array(512);
    this.hasSpectrum = false;
    this.source = null;
    this.track = null;
    this.buffer = null;
    this.bufferId = null;
    this.duration = 0;
    this.offset = 0;
    this.anchor = 0;
    this.rate = 1;
    this.playing = false;
    this.disposed = false;
    this.generation = 0;
  }
  position(now = this.context ? this.context.currentTime : 0) {
    const position = this.offset + (this.playing ? Math.max(0, now - this.anchor) * this.rate : 0);
    return Math.max(0, Math.min(this.duration, position));
  }
  resetSpectrum() {
    if (this.analyser) this.analyser.disconnect();
    this.analyser = null;
    this.spectrum.fill(0);
    this.spectrumChange.fill(0);
    this.previousSpectrum.fill(0);
    this.hasSpectrum = false;
  }
  sampleSpectrum() {
    if (this.playing && this.analyser) {
      this.analyser.getByteFrequencyData(this.spectrum);
      for (let i = 0; i < this.spectrum.length; i++) {
        this.spectrumChange[i] = this.hasSpectrum ? (this.spectrum[i] - this.previousSpectrum[i]) / 255 : 0;
      }
      this.previousSpectrum.set(this.spectrum);
      this.hasSpectrum = true;
    }
    return this.spectrum;
  }
  setTrack(track) {
    this.pause();
    this.resetSpectrum();
    this.track = track;
    this.duration = track.duration;
    this.offset = 0;
    this.onChange();
  }
  async play(autoplay = false) {
    if (this.disposed || !this.track) return false;
    if (this.playing) return true;
    const generation = ++this.generation;
    if (!this.context) this.context = this.contextFactory();
    const context = this.context;
    // Initiate resume in the click call stack, before the first await.
    const resume = context.state === "running" ? Promise.resolve() : context.resume();
    if (autoplay) {
      await Promise.race([resume, new Promise(resolve => setTimeout(resolve, 250))]);
    } else {
      await resume;
    }
    if (this.disposed || generation !== this.generation) return false;
    if (context.state !== "running") return false;
    if (this.offset >= this.duration) { this.offset = 0; this.resetSpectrum(); }
    if (this.bufferId !== this.track.id) {
      const samples = decodePCM(this.track);
      this.buffer = context.createBuffer(1, samples.length, this.track.sample_rate);
      this.buffer.copyToChannel(samples, 0);
      this.bufferId = this.track.id;
    }
    const source = context.createBufferSource();
    source.buffer = this.buffer;
    if (!this.analyser) {
      this.analyser = context.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.smoothingTimeConstant = .22;
      this.analyser.minDecibels = SPECTRUM_MIN_DB;
      this.analyser.maxDecibels = SPECTRUM_MAX_DB;
      this.analyser.connect(context.destination);
    }
    source.connect(this.analyser);
    const now = context.currentTime;
    source.playbackRate.setValueAtTime(this.rate, now);
    source.onended = () => {
      if (this.disposed || generation !== this.generation || this.source !== source) return;
      this.offset = this.duration;
      this.playing = false;
      this.source = null;
      source.disconnect();
      this.onChange();
      this.onEnded();
    };
    this.source = source;
    this.anchor = now;
    this.playing = true;
    source.start(now, this.offset);
    this.onChange();
    return true;
  }
  pause() {
    ++this.generation;
    this.offset = this.position();
    this.playing = false;
    if (this.source) {
      this.source.onended = null;
      try { this.source.stop(); } catch (_) { /* Already ended. */ }
      this.source.disconnect();
      this.source = null;
    }
    this.onChange();
  }
  stop() {
    this.pause();
    this.offset = 0;
    this.resetSpectrum();
    this.onChange();
  }
  setRate(rate) {
    if (!Number.isFinite(rate) || rate < 0.25 || rate > 10) throw new Error("再生速度は0.25～10倍です。");
    const now = this.context ? this.context.currentTime : 0;
    this.offset = this.position(now);
    this.anchor = now;
    this.rate = rate;
    if (this.source) this.source.playbackRate.setValueAtTime(rate, now);
    this.onChange();
  }
  async seek(seconds) {
    const wasPlaying = this.playing;
    this.pause();
    this.resetSpectrum();
    this.offset = Math.min(this.duration, Math.max(0, seconds));
    this.onChange();
    // Seeking to the end is a completed position, not an implicit replay.
    return wasPlaying && this.offset < this.duration ? this.play(false) : false;
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.pause();
    this.buffer = null;
    this.resetSpectrum();
    if (this.context && this.context.state !== "closed") this.context.close().catch(() => {});
  }
}

function formatTime(seconds) {
  return `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

export function windowAtTime(track, time) {
  // A window becomes visible only once all its samples have been reached.
  const ends = track.windows.ends;
  let low = 0, high = ends.length;
  while (low < high) { const mid = (low + high) >> 1; if (ends[mid] <= time) low = mid + 1; else high = mid; }
  return low - 1;
}

export function revealFraction(time, duration) {
  return Math.max(0, Math.min(1, time / duration));
}

export function detectedIntervals(track, time) {
  const intervals = [], flags = track.windows.flags;
  if (!flags) return intervals;
  const last = windowAtTime(track, time);
  for (let i = 0; i <= last; i++) {
    if (!flags[i]) continue;
    const start = track.windows.starts[i], end = track.windows.ends[i];
    const previous = intervals.at(-1);
    if (previous && start <= previous[1]) previous[1] = Math.max(previous[1], end);
    else intervals.push([start, end]);
  }
  return intervals;
}

export function completionIds(previous, track, time) {
  return time >= track.duration && !previous.includes(track.id) ? [...previous, track.id] : previous;
}

export function melColor(value) {
  // Fixed display range (-49.6 to -13.6 dB); never normalize individual recordings.
  // The source remains the original log-mel, encoded over -100 to +20 dB.
  const stops = [[5, 1, 24], [62, 0, 145], [15, 57, 255], [0, 237, 225], [217, 255, 20], [255, 114, 0], [255, 31, 100]];
  const point = Math.max(0, Math.min(1, (value - .42) / .30)) * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(point));
  const amount = point - index;
  return stops[index].map((entry, channel) => Math.round(entry * (1 - amount) + stops[index + 1][channel] * amount));
}

// All visual timing follows the original audio clock, not wall-clock animation.
export function melColumnAtTime(mel, time) {
  const column = Math.min(mel.width - 1, Math.floor((time - mel.frame_seconds) / mel.hop_seconds));
  return Math.max(-1, column);
}

export function spectralHistory(track, time, intervals = detectedIntervals(track, time)) {
  const mel = track.mel, last = Math.floor(melColumnAtTime(mel, time) / 4) * 4, rows = [];
  // Anchor decimation to absolute columns. Sliding the stride origin replaces
  // every historical row after 8 seconds and makes the entire terrain shimmer.
  for (let column = Math.max(0, last - 4 * 63); column <= last; column += 4) {
    const seconds = mel.frame_seconds + column * mel.hop_seconds;
    rows.push({column, seconds, anomaly: intervals.some(([start, end]) => seconds >= start && seconds <= end)});
  }
  return rows;
}

export function drawSpectrogram(canvas, track, source, scratch, time) {
  const mel = track.mel, count = melColumnAtTime(mel, time) + 1;
  if (count <= 0) return;
  // Crop BEFORE interpolation so the smoothed leading edge cannot reveal a
  // future column. Resizing scratch also discards any history after a rewind.
  scratch.width = count; scratch.height = mel.height;
  const cropped = scratch.getContext("2d");
  cropped.imageSmoothingEnabled = false;
  cropped.drawImage(source, 0, 0, count, mel.height, 0, 0, count, mel.height);
  const context = canvas.getContext("2d");
  context.imageSmoothingEnabled = true; context.imageSmoothingQuality = "high";
  const end = mel.frame_seconds + (count - 1) * mel.hop_seconds;
  context.drawImage(scratch, 0, 0, canvas.width * Math.min(1, end / track.duration), canvas.height);
}

export function contributionColor(value) {
  // A common 0 ... 0.15 scale, not per-record normalization or an anomaly mask.
  const stops = [[8, 19, 33], [31, 153, 190], [151, 255, 222]];
  const point = Math.max(0, Math.min(1, value)) * 2, i = Math.min(1, Math.floor(point));
  return stops[i].map((v, channel) => Math.round(v + (stops[i + 1][channel] - v) * (point - i)));
}

export function drawContribution(canvas, track, source, scratch, time) {
  const count = windowAtTime(track, time) + 1;
  if (!track.contribution || !source || count <= 0) return;
  const height = track.contribution.height;
  // Crop before interpolation, including on rewind. Each bin ends at its window's completion.
  scratch.width = count; scratch.height = height;
  const cropped = scratch.getContext("2d");
  cropped.imageSmoothingEnabled = false;
  cropped.drawImage(source, 0, 0, count, height, 0, 0, count, height);
  const ends = track.windows.ends, hop = ends.length > 1 ? ends[1] - ends[0] : .032;
  const start = Math.max(0, ends[0] - hop), end = ends[count - 1];
  const context = canvas.getContext("2d");
  context.imageSmoothingEnabled = true; context.imageSmoothingQuality = "high";
  context.drawImage(scratch, start / track.duration * canvas.width, 0,
                    (end - start) / track.duration * canvas.width, canvas.height);
}

export function drawAnomalyBars(canvas, intervals, time, duration, pixelRatio = 1) {
  const context = canvas.getContext("2d");
  context.fillStyle = "#ff8396";
  for (const [start, end] of intervals) {
    const left = Math.max(0, start) / duration * canvas.width;
    const right = Math.min(time, end, duration) / duration * canvas.width;
    if (right > left) context.fillRect(left, 0, right - left, 3 * pixelRatio);
  }
}

function hzToMel(hz) {
  // Slaney scale used by the 64-band feature extractor, not a linear-Hz image.
  return hz < 1000 ? hz / (200 / 3) : 15 + Math.log(hz / 1000) / (Math.log(6.4) / 27);
}

export function timeTicks(duration) {
  return Array.from({length: Math.floor(duration / 2) + 1}, (_, i) => i * 2);
}

export function axisTicks(name, track, time = 0, mode = "bars") {
  const label = value => Number(value.toPrecision(3)).toString();
  const tick = (value, position) => ({value, position, label: label(value)});
  const frequencyFraction = hz => (hzToMel(hz) / hzToMel(8000) * 65 - .5) / 64;
  const x = timeTicks(track.duration).map(v => tick(v, v / track.duration));
  if (name === "mel" || name === "contribution") {
    return {x, y: [.1, .5, 2, 6].map(khz => tick(khz, 1 - frequencyFraction(khz * 1000))), xUnit: "s", yUnit: "kHz"};
  }
  if (name === "wave") {
    const peak = Math.max(track.waveform.peak, 1e-8);
    return {x, y: [peak, 0, -peak].map(v => tick(v, .5 - v / peak * .44)), xUnit: "s", yUnit: "amp"};
  }
  if (name === "score") {
    const top = Math.max(track.windows.maximum, track.windows.flags ? (track.windows.threshold ?? track.threshold) * 1.1 : 0, 1e-12);
    return {x, y: [top, 0].map(v => tick(v, .91 - v / top * .8)), xUnit: "s", yUnit: "score"};
  }
  const bars = mode === "bars";
  return {
    x: [.1, .5, 2, 6].map(khz => tick(khz, bars ? .08 + frequencyFraction(khz * 1000) * .84 :
      .05 + ((hzToMel(khz * 1000) / hzToMel(8000) * 65 - 1) / 63) * .9)),
    y: (bars ? [-14, -25, -40] : [0, -30, -60]).map(db => tick(db, bars ?
      .97 - Math.pow(Math.max(0, Math.min(1, (db + 49.6) / 36)), 1.6) * .87 * .5 :
      .94 - Math.pow((db + 100) / 120, 1.6) * .87)),
    xUnit: "kHz", yUnit: "dB",
  };
}

function sceneBackground(context, width, height) {
  context.clearRect(0, 0, width, height);
  const background = context.createRadialGradient(width * .5, height * .5, 0, width * .5, height * .5, width * .7);
  background.addColorStop(0, "#12223a"); background.addColorStop(1, "#050c18");
  context.fillStyle = background; context.fillRect(0, 0, width, height);
}

// Log-frequency groups give each pitch range a stable place in the cloud.
// Decode dB before averaging: byte values themselves are not sound energy.
export function nebulaBands(frequencies, changes = null, sampleRate = 48000) {
  const count = 24, upper = Math.min(8000, sampleRate / 2), lower = 60;
  const bands = [], binHz = sampleRate / (frequencies.length * 2);
  let total = 0, bass = 0, treble = 0;
  for (let band = 0; band < count; band++) {
    const lo = lower * (upper / lower) ** (band / count);
    const hi = lower * (upper / lower) ** ((band + 1) / count);
    const first = Math.max(1, Math.min(frequencies.length - 1, Math.floor(lo / binHz)));
    const end = Math.min(frequencies.length, Math.max(first + 1, Math.ceil(hi / binHz)));
    let power = 0, change = 0;
    for (let i = first; i < end; i++) {
      const value = frequencies[i] || 0;
      const db = SPECTRUM_MIN_DB + value / 255 * (SPECTRUM_MAX_DB - SPECTRUM_MIN_DB);
      power += value ? 10 ** (db / 10) : 0;
      change += changes?.[i] || 0;
    }
    power /= end - first;
    const magnitude = Math.sqrt(power);
    // Smooth compression has no flat ceiling for ordinary loud inputs.
    const amp = magnitude / (magnitude + NEBULA_STYLE.levelReference);
    const delta = Math.tanh(change / (end - first) * NEBULA_STYLE.changeSensitivity);
    bands.push({amp, delta, power});
    total += power;
    if (Math.sqrt(lo * hi) < 500) bass += power;
    if (Math.sqrt(lo * hi) >= 2000) treble += power;
  }
  let breadth = 0;
  if (total > 0) for (const band of bands) {
    const p = band.power / total;
    if (p > 0) breadth -= p * Math.log(p) / Math.log(count);
  }
  return {bands, bass: total ? bass / total : 0, treble: total ? treble / total : 0, breadth};
}

export function drawNebula(canvas, frequencies, time, active, reduceMotion = false, changes = null, sampleRate = 48000) {
  const context = canvas.getContext("2d");
  const width = canvas.width, height = canvas.height;
  context.clearRect(0, 0, width, height);
  const bg = context.createRadialGradient(width*.5,height*.5,0,width*.5,height*.5,width*.65);
  bg.addColorStop(0,"#101c38"); bg.addColorStop(1,"#02040c");
  context.fillStyle = bg; context.fillRect(0,0,width,height);
  // Preserve the 16:9 geometry at a fixed logical resolution so the swirl
  // does not change shape with panel size or DPR.
  const w = 800, h = w * 9 / 16, scale = Math.min(width / w, height / h);
  const {bands, bass, treble, breadth} = nebulaBands(frequencies, changes, sampleRate);
  const t = reduceMotion ? 0 : time, R = Math.min(w,h)*.375;
  context.save(); context.translate(width/2,height/2); context.scale(scale,scale);
  // Normal compositing preserves blue-violet points instead of white overlap.
  context.globalCompositeOperation = "source-over";
  // Low bands live at the core, high bands outside; noise excites many rings.
  // Keep the point swirl, but do not scatter every frequency throughout it.
  for (let k=0;k<950;k++) {
    const band = Math.floor(k / 950 * bands.length), u = band / (bands.length - 1);
    const amp = bands[band].amp, delta = bands[band].delta * (reduceMotion ? .15 : 1), impact = Math.abs(delta);
    const a = k*2.399 + t*.08 + amp*.12 + delta*NEBULA_STYLE.angleResponse;
    const scatter = breadth * amp * Math.sin(k*12.9898) * .15;
    const response = .95 + .35*Math.tanh(amp*NEBULA_STYLE.levelSpread + delta*NEBULA_STYLE.changeSpread);
    const r = R * (.5 + .5*Math.sqrt(u)) * response * (1 + scatter);
    // Apply the swirl to both axes: twisting x alone collapses some rings
    // into diagonal lines when the phase difference approaches pi/2.
    const angle = a+r*.013;
    const x = Math.cos(angle)*r*(1.3+.1*bass-.15*treble);
    const y = Math.sin(angle)*r*(NEBULA_STYLE.viewDepth+.13*treble+.02*breadth);
    const size = (NEBULA_STYLE.particleBase + NEBULA_STYLE.particleRange * Math.pow(Math.min(1,amp+impact*.35),.85))*(w/750);
    const alpha = .45 + .49*Math.sqrt(amp);
    context.fillStyle=`hsla(${190+u*85},90%,${48+amp*21}%,${alpha})`;
    context.beginPath(); context.arc(x,y,size,0,Math.PI*2); context.fill();
  }
  context.restore();
}

// Fixed time/frequency cells: advancing playback only translates old geometry.
// Average completed frames locally, never rescale a whole scene by its newest peak.
const TERRAIN_TIMING = { stride: 4, rows: 64 }; // 128 ms per row, 8.192 s of history.
export function terrainBarRows(track, pixels, time, intervals) {
  const {stride, rows: rowCount} = TERRAIN_TIMING;
  const mel = track.mel, last = Math.floor(melColumnAtTime(mel, time) / stride) * stride;
  const rows = [];
  for (let column = Math.max(0, last - stride * (rowCount - 1)); column <= last; column += stride) {
    const seconds = mel.frame_seconds + column * mel.hop_seconds;
    const values = [];
    for (let band = 0; band < mel.height; band += 2) {
      let sum = 0, count = 0;
      for (let b = band; b < Math.min(band + 2, mel.height); b++) {
        for (let c = Math.max(0, column - 3); c <= column; c++) {
          sum += pixels[b * mel.width + c] / 255; count++;
        }
      }
      // Same fixed -49.6 ... -13.6 dB display range as SPECTROGRAM.
      // The full transport range (-100 ... +20 dB) otherwise flattens the relief.
      values.push(Math.pow(Math.max(0, Math.min(1, (sum / count - .42) / .30)), 1.6));
    }
    rows.push({column, seconds, values, anomaly: intervals.some(([start, end]) => seconds >= start && seconds <= end)});
  }
  return rows;
}

function drawTerrainBars(context, width, height, track, pixels, time, intervals) {
  const rows = terrainBarRows(track, pixels, time, intervals);
  const historySeconds = TERRAIN_TIMING.stride * TERRAIN_TIMING.rows * track.mel.hop_seconds;
  // Front-facing, slightly elevated orthographic camera. History moves straight back.
  const project = (band, depth, value) => [
    width * (.08 + band * .84),
    height * (.97 - depth * .51 - value * .87 * .5),
  ];
  const face = (points, color) => {
    context.beginPath();
    points.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y));
    context.closePath(); context.fillStyle = color; context.fill();
  };
  context.save();
  context.globalAlpha = 1;
  context.shadowBlur = 0;
  // Opaque tinted-glass shading gives depth without translucent history stacking.
  // Shared screen-space gradients avoid thousands of gradient allocations per frame.
  const normalFace = context.createLinearGradient(0, 0, 0, height);
  normalFace.addColorStop(0, "#599aaa");
  normalFace.addColorStop(.45, "#244b62");
  normalFace.addColorStop(.8, "#102c40");
  normalFace.addColorStop(1, "#285e70");
  const anomalyFace = context.createLinearGradient(0, 0, 0, height);
  anomalyFace.addColorStop(0, "#e49bab");
  anomalyFace.addColorStop(.45, "#884556");
  anomalyFace.addColorStop(.8, "#422336");
  anomalyFace.addColorStop(1, "#9a5065");
  for (const row of rows) {
    const depth = Math.max(0, (time - row.seconds) / historySeconds);
    const back = Math.min(1, depth + .8 / TERRAIN_TIMING.rows);
    const leading = row === rows[rows.length - 1];
    for (let band = 0; band < row.values.length; band++) {
      const left = (band + .06) / row.values.length, right = (band + .94) / row.values.length;
      const value = row.values[band];
      const a = project(left, depth, value), b = project(right, depth, value);
      const c = project(right, back, value), d = project(left, back, value);
      // From directly in front, only the shaded front and the lit top are visible.
      face([a, project(left, depth, 0), project(right, depth, 0), b], row.anomaly ? anomalyFace : normalFace);
      face([a, b, c, d], row.anomaly ? "#ff8396" : "#70b9cc");
      context.strokeStyle = row.anomaly ? "#984f65" : "#396578";
      context.lineWidth = Math.max(.4, width / 1100);
      context.stroke();
      if (leading) {
        context.beginPath(); context.moveTo(...a); context.lineTo(...b);
        context.strokeStyle = row.anomaly ? "#ffc0c9" : "#cbffed";
        context.lineWidth = Math.max(1, width / 550); context.stroke();
      }
    }
  }
  context.restore();
}

export function drawWaterfall(canvas, track, pixels, time, intervals = detectedIntervals(track, time), mode = "line", localInsights = false) {
  const context = canvas.getContext("2d"), width = canvas.width, height = canvas.height;
  sceneBackground(context, width, height);
  if (!pixels) return;
  if (mode === "bars") {
    drawTerrainBars(context, width, height, track, pixels, time, intervals);
    return;
  }
  const rows = spectralHistory(track, time, intervals), mel = track.mel;
  const now = Math.max(0, time), historySeconds = 4 * 63 * mel.hop_seconds;
  const leadingScale = localInsights ? .85 : 1;
  context.save();
  for (const row of rows) {
    const leading = row === rows[rows.length - 1];
    const depth = Math.min(1, Math.max(0, (now - row.seconds) / historySeconds)), scale = 1 - depth * .48;
    // Leave headroom for 1.5x height, including the oldest, farthest rows.
    const baseY = height * .94 - depth * height * .44;
    context.beginPath();
    for (let band = 0; band < mel.height; band++) {
      const value = Math.pow(pixels[band * mel.width + row.column] / 255, 1.6);
      const x = width * .5 + (band / (mel.height - 1) - .5) * width * .9 * scale;
      const y = baseY - value * height * .87 * scale;
      band ? context.lineTo(x, y) : context.moveTo(x, y);
    }
    if (leading) {
      // A dark outline separates the current ridge from all previous lines.
      context.globalAlpha = 1;
      context.strokeStyle = "#081321";
      context.lineWidth = Math.max(5, width / 600 * 5) * leadingScale;
      context.stroke();
      context.shadowColor = row.anomaly ? "#ff8396" : "#9bffe4";
      context.shadowBlur = Math.max(4, width / 600 * 6) * leadingScale;
    } else {
      context.globalAlpha = .55;
      context.shadowBlur = 0;
    }
    context.strokeStyle = row.anomaly ? "#ff8396" : leading ? "#c6fff0" : `rgba(112,185,210,${.35 + (1-depth) * .4})`;
    context.lineWidth = Math.max(1, width / 600) * (leading ? 2.5 * leadingScale : 1);
    context.stroke();
  }
  context.restore();
}

export default function render({parentElement, data, setStateValue}) {
  const signature = `${data.instance_id}:${Boolean(data.local_insights)}:${data.tracks.map(track => track.id).join(":")}`;
  const previous = instances.get(parentElement);
  // Streamlit invokes render on data/theme changes without invoking old cleanup.
  if (previous && previous.signature === signature) return previous.cleanup;
  if (previous) previous.cleanup();
  const root = parentElement.querySelector(".sound-player");
  if (!root || !data.tracks.length) return () => {};
  const field = name => root.querySelector(`[data-field="${name}"]`);
  const action = name => root.querySelector(`[data-action="${name}"]`);
  const tracks = data.tracks;
  const localInsights = data.local_insights === true;
  root.classList.toggle("local-insights", localInsights);
  root.querySelector(".contribution-panel").hidden = !localInsights;
  const listenerController = new AbortController();
  const listen = (element, event, callback) => element.addEventListener(event, callback, {signal: listenerController.signal});
  let disposed = false;
  let index = 0;
  let frame = 0;
  let lastDraw = -Infinity;
  let images = {};
  let selectedWindow = -2;
  let busy = false;
  let completed = [];
  let changingTrack = false;
  let melPixels = null;
  let terrainMode = "bars";
  const melScratch = document.createElement("canvas");
  const contributionScratch = document.createElement("canvas");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const seek = root.querySelector(".seek");
  const continuous = root.querySelector('[data-control="continuous"]');
  const speed = root.querySelector('[data-control="rate"]');
  const canvasNames = ["wave", "mel", "score", "nebula", "waterfall", ...(localInsights ? ["contribution"] : [])];
  const canvases = Object.fromEntries(canvasNames.map(name => [name, root.querySelector(`[data-canvas="${name}"]`)]));
  const axes = {};
  // Wrappers are only installed for local mode; shared public rendering stays unchanged.
  if (!localInsights) {
    root.querySelectorAll(".axis-shell").forEach(shell => shell.replaceWith(shell.querySelector("canvas")));
    root.querySelectorAll(".axis-heading-unit").forEach(unit => unit.remove());
  } else for (const [name, canvas] of Object.entries(canvases)) {
    if (name === "nebula") continue;
    let shell = canvas.closest(".axis-shell");
    if (!shell) {
      shell = document.createElement("div"); shell.className = "axis-shell" + (name === "waterfall" ? " terrain-axes" : "");
      canvas.before(shell); shell.append(canvas);
      for (const part of ["x", "y"]) {
        const node = document.createElement("div"); node.className = `axis-${part}`; shell.append(node);
      }
      const unit = document.createElement("span"); unit.className = "axis-heading-unit";
      canvas.closest(".signal-panel,.scene-panel").querySelector(".signal-label").append(unit);
    }
    axes[name] = {shell, signature: "", unit: canvas.closest(".signal-panel,.scene-panel").querySelector(".axis-heading-unit")};
  }

  function drawAxisLabels(name, track, time) {
    const view = axes[name]; if (!view) return;
    const values = axisTicks(name, track, time, terrainMode);
    const signature = JSON.stringify([values.x, values.y, values.xUnit, values.yUnit]);
    const paint = (part, ticks, unit = "") => {
      const container = view.shell.querySelector(`.axis-${part}`);
      container.replaceChildren();
      for (const item of ticks) {
        const span = document.createElement("span"); span.className = "axis-tick";
        span.textContent = item.label;
        span.style[part === "x" ? "left" : "top"] = `${item.position * 100}%`;
        if (part === "x" && item.position === 0) span.classList.add("axis-edge-start");
        if (part === "x" && item.position === 1) span.classList.add("axis-edge-end");
        container.append(span);
      }
      if (unit) { const span = document.createElement("span"); span.className = "axis-unit"; span.textContent = unit; container.append(span); }
    };
    if (signature !== view.signature) {
      paint("x", values.x, values.xUnit); paint("y", values.y);
      view.unit.textContent = values.yUnit; view.signature = signature;
    }
  }
  const AudioContextType = window.AudioContext || window.webkitAudioContext;

  function notice(text, tone = "") {
    field("notice").textContent = text;
    field("notice").dataset.tone = tone;
  }
  function transportChanged() {
    if (disposed) return;
    const playing = transport.playing;
    action("play").textContent = playing ? "Ⅱ 一時停止" : "▶ 再生";
    action("play").setAttribute("aria-label", playing ? "音声を一時停止" : "音声を再生");
    field("status").textContent = playing ? "PLAYING · 音声と同期中" : transport.offset >= transport.duration ? "再生完了" : transport.offset ? "一時停止" : "再生待ち";
    root.dataset.playing = String(playing);
    root.dataset.rate = String(transport.rate);
    draw();
    if (playing && !frame) frame = requestAnimationFrame(tick);
  }
  const transport = new PlaybackTransport(
    () => {
      if (!AudioContextType) throw new Error("このブラウザはWeb Audioに対応していません。");
      return new AudioContextType({latencyHint: "interactive"});
    },
    transportChanged,
    () => {
      if (continuous.checked && index + 1 < tracks.length) choose(index + 1, true);
      else notice("再生が完了しました。再生ボタンで同じ音声を聴き直せます。");
    },
  );

  async function start(autoplay = false) {
    if (disposed) return;
    if (busy && !autoplay) transport.pause();
    busy = true;
    notice(autoplay ? "自動再生を準備しています…" : "音声を準備しています…");
    try {
      const started = await transport.play(autoplay);
      if (disposed) return;
      if (started) {
        window.dispatchEvent(new CustomEvent("si-player-start", {detail: data.instance_id}));
        notice("再生中も速度を変更できます。再生速度を変えてもAIの検出精度は変わりません。");
      } else if (!transport.playing) {
        notice("ブラウザが自動再生を保留しました。「▶ 再生」を押すと音声を開始します。", "warning");
      }
    } catch (error) {
      transport.pause();
      notice(`音声を再生できません: ${error.message || error}。再生ボタンから再試行してください。`, "error");
    } finally {
      busy = false;
    }
  }

  function setSpeed(value) {
    transport.setRate(value);
    speed.value = String(value);
    field("rate").textContent = `${value.toFixed(2)}×`;
    for (const button of root.querySelectorAll("[data-rate]")) {
      const active = Number(button.dataset.rate) === value;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    }
  }

  function baseCanvas(width, height) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }

  function buildImages() {
    const track = tracks[index];
    melPixels = decodeBytes(track.mel.pixels);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    for (const [name, canvas] of Object.entries(canvases)) {
      if (localInsights) drawAxisLabels(name, track, transport.position());
      const box = canvas.getBoundingClientRect();
      const width = Math.max(1, Math.round(box.width * dpr));
      const height = Math.max(1, Math.round(box.height * dpr));
      canvas.width = width;
      canvas.height = height;
      if (["nebula", "waterfall"].includes(name)) continue;
      const base = baseCanvas(width, height);
      const context = base.getContext("2d");
      context.fillStyle = name === "mel" ? "#071521" : "#0d1b2800";
      context.fillRect(0, 0, width, height);
      if (name !== "mel" && name !== "contribution") {
        context.strokeStyle = "#30485a66";
        context.lineWidth = dpr * 0.5;
        const grid = localInsights ? timeTicks(track.duration).filter(t => t > 0 && t < track.duration).map(t => t / track.duration) : [.2, .4, .6, .8];
        for (const part of grid) {
          context.beginPath(); context.moveTo(part * width, 0); context.lineTo(part * width, height); context.stroke();
        }
      }
      if (name === "wave") {
        const wave = track.waveform;
        const scale = Math.max(wave.peak, 1e-8);
        const gradient = context.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, "#48caffaa"); gradient.addColorStop(1, "#63f5ccbb");
        context.strokeStyle = gradient;
        context.lineWidth = Math.max(dpr * 0.7, width / wave.minimum.length * 0.8);
        context.beginPath();
        for (let i = 0; i < wave.minimum.length; i++) {
          const x = wave.times[i] / track.duration * width;
          context.moveTo(x, height / 2 - wave.maximum[i] / scale * height * 0.44);
          context.lineTo(x, height / 2 - wave.minimum[i] / scale * height * 0.44);
        }
        context.stroke();
      } else if (name === "mel" || name === "contribution") {
        const mel = name === "mel" ? track.mel : track.contribution;
        if (!mel) { images[name] = base; continue; }
        const pixels = decodeBytes(mel.pixels);
        const small = baseCanvas(mel.width, mel.height);
        const smallContext = small.getContext("2d");
        const image = smallContext.createImageData(mel.width, mel.height);
        for (let row = 0; row < mel.height; row++) {
          for (let column = 0; column < mel.width; column++) {
            const color = (name === "mel" ? melColor : contributionColor)(pixels[row * mel.width + column] / 255);
            const position = ((mel.height - 1 - row) * mel.width + column) * 4;
            image.data.set([...color, 255], position);
          }
        }
        smallContext.putImageData(image, 0, 0);
        images[name] = small;
        continue;
      } else {
        const values = track.windows;
        const top = Math.max(values.maximum, values.flags ? (values.threshold ?? track.threshold) * 1.1 : 0, 1e-12);
        const xAt = i => values.ends[i] / track.duration * width;
        const yAt = i => height * 0.91 - values.scores[i] / top * height * 0.8;
        const gradient = context.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, "#63f5cc55"); gradient.addColorStop(1, "#63f5cc00");
        context.beginPath(); context.moveTo(xAt(0), height);
        for (let i = 0; i < values.scores.length; i++) context.lineTo(xAt(i), yAt(i));
        context.lineTo(xAt(values.scores.length - 1), height); context.closePath();
        context.fillStyle = gradient; context.fill();
        context.beginPath();
        for (let i = 0; i < values.scores.length; i++) i ? context.lineTo(xAt(i), yAt(i)) : context.moveTo(xAt(i), yAt(i));
        context.strokeStyle = "#74f8cd"; context.lineWidth = 1.4 * dpr; context.stroke();
        if (values.flags) {
          for (let i = 1; i < values.scores.length; i++) {
            if (!values.flags[i]) continue;
            context.beginPath(); context.moveTo(xAt(i-1), yAt(i-1)); context.lineTo(xAt(i), yAt(i));
            context.strokeStyle = "#ff8fa0"; context.lineWidth = 2.8 * dpr; context.stroke();
          }
          const y = height * .91 - (values.threshold ?? track.threshold) / top * height * .8;
          context.setLineDash([4*dpr, 3*dpr]); context.strokeStyle = "#dcc8a4";
          context.beginPath(); context.moveTo(0,y); context.lineTo(width,y); context.stroke(); context.setLineDash([]);
        }
        if (!localInsights) {
          context.fillStyle = "#7493a8"; context.font = `${9 * dpr}px sans-serif`;
          context.fillText(top.toFixed(3), 4 * dpr, 10 * dpr); context.fillText("0", 4 * dpr, height - 3 * dpr);
        }
      }
      images[name] = base;
    }
    draw();
  }

  function draw() {
    if (disposed || changingTrack) return;
    const track = tracks[index];
    const time = transport.position();
    const fraction = revealFraction(time, track.duration);
    const newCompleted = completionIds(completed, track, time);
    if (newCompleted !== completed) { completed = newCompleted; setStateValue("completed", completed); }
    field("decision").textContent = track.decision === "anomaly_candidate" ? "AI判定：異常音" : "AI判定：正常音";
    field("decision").dataset.tone = track.decision === "anomaly_candidate" ? "attention" : "normal";
    field("clip-score").textContent = track.score.toFixed(4);
    root.dataset.position = time.toFixed(3);
    field("position").textContent = formatTime(time);
    seek.value = String(time);
    seek.setAttribute("aria-valuetext", `${time.toFixed(2)}秒 / ${track.duration.toFixed(2)}秒`);
    field("orbit-progress").style.strokeDashoffset = String(477.522 * (1 - fraction));
    const windowIndex = windowAtTime(track, time);
    const active = windowIndex >= 0 && track.windows.flags?.[windowIndex] === true;
    const intervals = detectedIntervals(track, time);
    root.dataset.detection = active ? "anomaly" : "normal";
    if (selectedWindow !== windowIndex) {
      selectedWindow = windowIndex;
      field("window-score").textContent = windowIndex < 0 ? "—" : track.windows.scores[windowIndex].toFixed(3);
      field("window-range").textContent = windowIndex < 0 ? "音声の開始を待っています" : `${track.windows.starts[windowIndex].toFixed(2)}〜${track.windows.ends[windowIndex].toFixed(2)}秒の文脈窓`;
    }
    for (const [name, canvas] of Object.entries(canvases)) {
      if (localInsights) drawAxisLabels(name, track, time);
      if (name === "nebula") { drawNebula(canvas, transport.spectrum, time, active, reduceMotion, transport.spectrumChange, transport.context?.sampleRate || 48000); continue; }
      if (name === "waterfall") { drawWaterfall(canvas, track, melPixels, time, intervals, terrainMode, localInsights); continue; }
      if (!images[name]) continue;
      const context = canvas.getContext("2d");
      const width = canvas.width, height = canvas.height;
      const x = fraction * width;
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#081321"; context.fillRect(0, 0, width, height);
      if (name === "contribution") {
        drawContribution(canvas, track, images[name], contributionScratch, time);
        drawAnomalyBars(canvas, intervals, time, track.duration, Math.min(window.devicePixelRatio || 1, 2));
        context.strokeStyle = "#e0fff7"; context.lineWidth = Math.min(window.devicePixelRatio || 1, 2);
        context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
        continue;
      }
      // Clip the source bitmap; future audio is genuinely not drawn.
      const visibleX = name === "score" ? (windowIndex < 0 ? 0 : track.windows.ends[windowIndex] / track.duration * width) : x;
      if (name === "mel") drawSpectrogram(canvas, track, images[name], melScratch, time);
      else if (visibleX > 0) context.drawImage(images[name], 0, 0, visibleX, height, 0, 0, visibleX, height);
      for (const [start, end] of intervals) {
        const left = start / track.duration * width, right = Math.min(x, end / track.duration * width);
        context.fillStyle = name === "mel" ? "#d548602b" : "#d5486055";
        context.fillRect(left, 0, Math.max(0, right-left), height);
        context.fillStyle = "#ff8396";
        context.fillRect(left, 0, Math.max(0, right-left), 3 * Math.min(window.devicePixelRatio || 1, 2));
      }
      const trail = context.createLinearGradient(Math.max(0, x - width * 0.055), 0, x + 0.01, 0);
      trail.addColorStop(0, "rgba(99,245,204,0)"); trail.addColorStop(1, "rgba(99,245,204,.16)");
      context.fillStyle = trail; context.fillRect(Math.max(0, x - width * 0.055), 0, width * 0.055, height);
      context.strokeStyle = active ? "#ff9baa" : "#e0fff7"; context.lineWidth = (active ? 2 : 1) * Math.min(window.devicePixelRatio || 1, 2);
      context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
      if (name === "score" && windowIndex >= 0) {
        const top = Math.max(track.windows.maximum, track.windows.flags ? (track.windows.threshold ?? track.threshold) * 1.1 : 0, 1e-12);
        const y = height * 0.91 - track.windows.scores[windowIndex] / top * height * 0.8;
        context.beginPath(); context.arc(visibleX, y, 3 * Math.min(window.devicePixelRatio || 1, 2), 0, 2 * Math.PI);
        context.fillStyle = "#f0fff9"; context.fill();
      }
    }
  }

  function tick(timestamp) {
    frame = 0;
    if (disposed || !transport.playing) return;
    transport.sampleSpectrum();
    if (timestamp - lastDraw >= (reduceMotion ? 80 : 0)) { draw(); lastDraw = timestamp; }
    frame = requestAnimationFrame(tick);
  }

  function choose(nextIndex, play = false) {
    if (nextIndex < 0 || nextIndex >= tracks.length) return;
    changingTrack = true;
    index = nextIndex;
    selectedWindow = -2;
    const track = tracks[index];
    transport.setTrack(track);
    changingTrack = false;
    field("track-index").textContent = `SAMPLE ${String(index + 1).padStart(2, "0")} / ${String(tracks.length).padStart(2, "0")}`;
    field("reference").textContent = track.reference_label === 0 ? "公開元: 正常" : track.reference_label === 1 ? "公開元: 異常" : "";
    field("threshold").textContent = track.threshold.toFixed(4);
    field("duration").textContent = formatTime(track.duration);
    field("duration-axis").textContent = `${track.duration.toFixed(1)}s`;
    seek.max = String(track.duration);
    action("previous").disabled = index === 0;
    action("next").disabled = index + 1 === tracks.length;
    for (const button of field("playlist").querySelectorAll("button")) {
      const selected = Number(button.dataset.index) === index;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-current", String(selected));
    }
    buildImages();
    if (play) start(false);
  }

  const playlist = field("playlist");
  playlist.replaceChildren();
  tracks.forEach((track, trackIndex) => {
    const button = document.createElement("button");
    button.dataset.index = String(trackIndex);
    const number = document.createElement("span"); number.className = "track-number"; number.textContent = String(trackIndex + 1).padStart(2, "0");
    const title = document.createElement("span"); title.className = "track-title"; title.textContent = track.title;
    const decision = document.createElement("span"); decision.className = "track-decision"; decision.textContent = track.reference_label === 1 ? "公開元：異常" : "公開元：正常";
    button.append(number, title, decision);
    listen(button, "click", () => choose(trackIndex, true));
    playlist.append(button);
  });
  field("playlist-count").textContent = `${tracks.length}件 · 検査した順序`;
  root.querySelector(".playlist").hidden = tracks.length === 1;
  listen(action("play"), "click", () => {
    if (transport.playing) { transport.pause(); notice("一時停止中です。再生ボタンで続きから再開します。"); }
    else start(false);
  });
  listen(action("stop"), "click", () => { transport.stop(); notice("停止しました。再生位置を先頭へ戻しました。"); });
  listen(action("previous"), "click", () => { choose(index - 1); notice("音声を切り替えました。再生ボタンで開始します。"); });
  listen(action("next"), "click", () => { choose(index + 1); notice("音声を切り替えました。再生ボタンで開始します。"); });
  listen(seek, "input", () => transport.seek(Number(seek.value)).catch(error => notice(error.message, "error")));
  listen(speed, "input", () => setSpeed(Number(speed.value)));
  for (const button of root.querySelectorAll("[data-terrain-mode]")) listen(button, "click", () => {
    terrainMode = button.dataset.terrainMode;
    for (const option of root.querySelectorAll("[data-terrain-mode]")) {
      option.setAttribute("aria-pressed", String(option.dataset.terrainMode === terrainMode));
    }
    draw();
  });
  for (const button of root.querySelectorAll("[data-rate]")) listen(button, "click", () => setSpeed(Number(button.dataset.rate)));
  for (const canvas of [canvases.wave, canvases.mel, canvases.score, ...(localInsights ? [canvases.contribution] : [])]) {
    listen(canvas, "click", event => {
      const box = canvas.getBoundingClientRect();
      transport.seek((event.clientX - box.left) / box.width * tracks[index].duration).catch(error => notice(error.message, "error"));
    });
  }
  listen(window, "si-player-start", event => {
    if (event.detail !== data.instance_id && transport.playing) {
      transport.pause(); notice("別のプレーヤーを開始したため一時停止しました。");
    }
  });
  const resizeObserver = new ResizeObserver(() => { if (!disposed) buildImages(); });
  resizeObserver.observe(root);
  choose(0, false);
  setSpeed(1);
  let attemptedAutoplay = false;
  if (data.autoplay) {
    attemptedAutoplay = true;
    queueMicrotask(() => { if (!disposed) start(true); });
  }
  const cleanup = () => {
    if (disposed) return;
    disposed = true;
    listenerController.abort();
    resizeObserver.disconnect();
    if (frame) cancelAnimationFrame(frame);
    transport.dispose();
    images = {};
    instances.delete(parentElement);
  };
  instances.set(parentElement, {signature, cleanup, transport, attemptedAutoplay});
  return cleanup;
}
