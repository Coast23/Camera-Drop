'use strict';

(function initBenchPlayerPage(global) {
  const codebook = global.CameraDropCodebook;
  if (!codebook) return;

  const DICT_OPTIONS = {
    best_v2: { mode: 'best', base: './best_v2' },
    best: { mode: 'best', base: './best' },
    builtin: { mode: 'builtin', base: '' },
  };

  function normalizeDictChoice(raw) {
    const value = String(raw || '').trim().toLowerCase();
    if (value === 'builtin' || value === 'builtin-gen') return 'builtin';
    if (value === 'best_v2' || value === './best_v2' || value.endsWith('/best_v2') || value.endsWith('\best_v2')) return 'best_v2';
    if (value === 'best' || value === './best' || value.endsWith('/best') || value.endsWith('\best')) return 'best';
    return 'best_v2';
  }

  function getInitialDictChoice() {
    const params = new URLSearchParams(location.search);
    return normalizeDictChoice(params.get('dict') || params.get('dict-base'));
  }

  function getDictSpec(choice) {
    return DICT_OPTIONS[normalizeDictChoice(choice)] || DICT_OPTIONS.best_v2;
  }

  function getCurrentDictChoice() {
    return normalizeDictChoice(dom.dictMode ? dom.dictMode.value : getInitialDictChoice());
  }

  const dom = {
    canvas: document.getElementById('playCanvas'),
    fpsRange: document.getElementById('fpsRange'),
    fpsInput: document.getElementById('fpsInput'),
    seedInput: document.getElementById('seedInput'),
    frameCountInput: document.getElementById('frameCountInput'),
    dictMode: document.getElementById('dictMode'),
    dictStatus: document.getElementById('dictStatus'),
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    stepBtn: document.getElementById('stepBtn'),
    fsBtn: document.getElementById('fsBtn'),
    statSeq: document.getElementById('statSeq'),
    statPlayFps: document.getElementById('statPlayFps'),
    shareText: document.getElementById('shareText'),
  };

  const ctx = dom.canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;

  const state = {
    dict: codebook.genDict(),
    frameSet: [],
    playing: false,
    rafId: 0,
    lastTs: 0,
    carryMs: 0,
    seq: 1,
    fpsWindowTs: 0,
    fpsWindowFrames: 0,
  };

  function clampInt(v, lo, hi, fallback) {
    const n = Number(v);
    if (!Number.isFinite(n)) return fallback;
    const i = Math.round(n);
    if (i < lo) return lo;
    if (i > hi) return hi;
    return i;
  }

  function getSeed() {
    return clampInt(dom.seedInput.value, 1, 0x7fffffff, 114514);
  }

  function getFps() {
    return clampInt(dom.fpsInput.value, 1, 60, 12);
  }

  function getFrameCount() {
    return codebook.BENCH_FRAME_SET;
  }

  function updateShareLink() {
    const q = new URLSearchParams();
    const choice = getCurrentDictChoice();
    const spec = getDictSpec(choice);
    q.set('seed', String(getSeed()));
    q.set('fps', String(getFps()));
    q.set('frames', String(getFrameCount()));
    q.set('dict', choice);
    if (spec.mode !== 'builtin') {
      q.set('dict-base', spec.base);
    }
    const url = 'bench-recognizer.html?' + q.toString();
    dom.shareText.textContent = url;
  }

  function updateControls() {
    dom.startBtn.disabled = state.playing;
    dom.stopBtn.disabled = !state.playing;
  }

  function renderCurrent() {
    const idx = ((state.seq - 1) % getFrameCount() + getFrameCount()) % getFrameCount();
    const frame = state.frameSet[idx];
    ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);
    if (frame && frame.canvas) {
      ctx.drawImage(frame.canvas, 0, 0);
    }
    dom.statSeq.textContent = String(state.seq);
  }

  function rebuildFrameSet() {
    const seed = getSeed();
    const frames = codebook.buildFrameSet(state.dict, seed, getFrameCount());
    for (let i = 0; i < frames.length; i++) {
      const canvas = document.createElement('canvas');
      canvas.width = codebook.IMG_SIZE;
      canvas.height = codebook.IMG_SIZE;
      const frameCtx = canvas.getContext('2d');
      frameCtx.imageSmoothingEnabled = false;
      codebook.drawFrame(frameCtx, state.dict, frames[i].seq, seed);
      frames[i].canvas = canvas;
    }
    state.frameSet = frames;
    if (state.seq > frames.length) state.seq = 1;
    renderCurrent();
  }

  function tick(ts) {
    if (!state.playing) return;
    if (!state.lastTs) state.lastTs = ts;
    const dt = ts - state.lastTs;
    state.lastTs = ts;
    state.carryMs += dt;

    const frameMs = 1000 / getFps();
    let renderedNow = 0;
    while (state.carryMs >= frameMs) {
      state.carryMs -= frameMs;
      renderCurrent();
      renderedNow++;
      state.seq++;
      if (state.seq > getFrameCount()) state.seq = 1;
    }

    if (!state.fpsWindowTs) state.fpsWindowTs = ts;
    state.fpsWindowFrames += renderedNow;
    const winDt = ts - state.fpsWindowTs;
    if (winDt >= 500) {
      dom.statPlayFps.textContent = ((state.fpsWindowFrames * 1000) / winDt).toFixed(2);
      state.fpsWindowFrames = 0;
      state.fpsWindowTs = ts;
    }

    state.rafId = requestAnimationFrame(tick);
  }

  function startPlay() {
    if (state.playing) return;
    state.playing = true;
    state.lastTs = 0;
    state.carryMs = 0;
    state.fpsWindowTs = 0;
    state.fpsWindowFrames = 0;
    updateControls();
    state.rafId = requestAnimationFrame(tick);
  }

  function stopPlay() {
    if (!state.playing) return;
    state.playing = false;
    if (state.rafId) cancelAnimationFrame(state.rafId);
    state.rafId = 0;
    updateControls();
  }

  async function loadDict() {
    const choice = getCurrentDictChoice();
    const spec = getDictSpec(choice);
    dom.dictStatus.value = 'loading';
    state.dict = await codebook.loadDict(spec.mode, spec.base);
    dom.dictStatus.value = state.dict.source || (spec.mode === 'builtin' ? 'builtin-gen' : spec.base);
    rebuildFrameSet();
    updateShareLink();
  }

  function syncFpsInputs(fromRange) {
    if (fromRange) dom.fpsInput.value = dom.fpsRange.value;
    else dom.fpsRange.value = dom.fpsInput.value;
    const fps = getFps();
    dom.fpsInput.value = String(fps);
    dom.fpsRange.value = String(fps);
    updateShareLink();
  }

  function applyQuery() {
    const q = new URLSearchParams(location.search);
    if (q.has('seed')) dom.seedInput.value = q.get('seed');
    if (q.has('fps')) {
      dom.fpsInput.value = q.get('fps');
      dom.fpsRange.value = q.get('fps');
    }
    dom.dictMode.value = getInitialDictChoice();
    dom.frameCountInput.value = String(getFrameCount());
    dom.frameCountInput.readOnly = true;
    syncFpsInputs(false);
  }

  dom.fpsRange.addEventListener('input', () => syncFpsInputs(true));
  dom.fpsInput.addEventListener('change', () => syncFpsInputs(false));
  dom.seedInput.addEventListener('change', () => {
    state.seq = 1;
    rebuildFrameSet();
    updateShareLink();
  });
  dom.frameCountInput.addEventListener('change', () => {
    dom.frameCountInput.value = String(getFrameCount());
    updateShareLink();
  });
  dom.dictMode.addEventListener('change', loadDict);
  dom.startBtn.addEventListener('click', startPlay);
  dom.stopBtn.addEventListener('click', stopPlay);
  dom.stepBtn.addEventListener('click', () => {
    stopPlay();
    state.seq++;
    if (state.seq > getFrameCount()) state.seq = 1;
    renderCurrent();
  });
  dom.fsBtn.addEventListener('click', async () => {
    if (!document.fullscreenElement) await dom.canvas.requestFullscreen();
    else await document.exitFullscreen();
  });

  applyQuery();
  updateControls();
  updateShareLink();
  loadDict();
})(window);

