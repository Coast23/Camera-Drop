'use strict';

(function initBenchRecognizerPage(global) {
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
    return normalizeDictChoice(dom && dom.dictSelect ? dom.dictSelect.value : getInitialDictChoice());
  }

  function getCurrentDictSpec() {
    return getDictSpec(getCurrentDictChoice());
  }

  function getCurrentDictKey() {
    const spec = getCurrentDictSpec();
    return spec.mode === 'builtin' ? 'builtin' : ('best|' + spec.base);
  }

  function applyRecognizerDictGlobals() {
    const spec = getCurrentDictSpec();
    global.__CAMDROP_RECOG_DICT_MODE = spec.mode;
    global.__CAMDROP_RECOG_DICT_BASE = spec.base;
  }

  const BENCH_ASSET_VERSION = '20260307a';
  const MATCH_WORKER_URL = 'js/bench-match-worker.js?v=' + BENCH_ASSET_VERSION;

  const defaultRecogWorkers = (() => {
    const hc = Number(global.navigator && global.navigator.hardwareConcurrency) || 4;
    return Math.max(1, Math.min(4, Math.floor(hc / 2) || 1));
  })();

  const defaultYoloWorkers = (() => {
    const hc = Number(global.navigator && global.navigator.hardwareConcurrency) || 4;
    return Math.max(1, Math.min(2, hc >= 6 ? 2 : 1));
  })();

  const dom = {
    seedInput: document.getElementById('benchSeed'),
    frameCountInput: document.getElementById('benchFrames'),
    modeSelect: document.getElementById('benchMode'),
    dictSelect: document.getElementById('benchDict'),
    expectedFpsInput: document.getElementById('benchExpectedFps'),
    startBtn: document.getElementById('benchStart'),
    stopBtn: document.getElementById('benchStop'),
    resetBtn: document.getElementById('benchReset'),
    shareText: document.getElementById('benchShare'),
    elapsed: document.getElementById('mElapsed'),
    decoded: document.getElementById('mDecoded'),
    unique: document.getElementById('mUnique'),
    skipped: document.getElementById('mSkipped'),
    duplicate: document.getElementById('mDup'),
    unknown: document.getElementById('mUnknown'),
    symbolAcc: document.getElementById('mSymAcc'),
    patternAcc: document.getElementById('mPatAcc'),
    colorAcc: document.getElementById('mColAcc'),
    best10SymAcc: document.getElementById('mBest10SymAcc'),
    best10PatAcc: document.getElementById('mBest10PatAcc'),
    best10ColAcc: document.getElementById('mBest10ColAcc'),
    top2Gap: document.getElementById('mTop2Gap'),
    decodeFps: document.getElementById('mDecodeFps'),
    uniqueFps: document.getElementById('mUniqueFps'),
    yoloFps: document.getElementById('mYoloFps'),
    recogMs: document.getElementById('mRecogMs'),
    queue: document.getElementById('mQueue'),
    expectedGen: document.getElementById('mExpectedGen'),
    missingApprox: document.getElementById('mMissingApprox'),
    lastSeq: document.getElementById('mLastSeq'),
  };

  const bench = {
    minRecogId: 0,
    captureEpoch: 0,
    running: false,
    startTime: 0,
    frameSet: [],
    frameVersion: 0,
    seen: new Set(),
    decodedCount: 0,
    uniqueCount: 0,
    skippedCount: 0,
    duplicateCount: 0,
    unknownCount: 0,
    symbolTotal: 0,
    symbolCorrect: 0,
    patternCorrect: 0,
    colorCorrect: 0,
    recogMsSum: 0,
    lastSeq: '-',
    lastHamming: '-',
    lastBest10SymAcc: 0,
    lastBest10PatAcc: 0,
    lastBest10ColAcc: 0,
    lastTop2Gap: 0,
    matchWorkers: [],
    matchQueue: [],
    matchToken: 0,
    matchPoolSize: 0,
    matchFrameVersion: -1,
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

  function getFrameCount() {
    return codebook.BENCH_FRAME_SET;
  }

  function getExpectedFps() {
    return clampInt(dom.expectedFpsInput.value, 1, 120, 20);
  }

  function getWorkerCount() {
    const params = new URLSearchParams(location.search);
    if (params.has('recog-workers')) {
      return clampInt(params.get('recog-workers'), 1, 8, defaultRecogWorkers);
    }
    if (params.has('workers')) {
      return clampInt(params.get('workers'), 1, 8, defaultRecogWorkers);
    }
    return defaultRecogWorkers;
  }

  function getYoloWorkerCount() {
    const params = new URLSearchParams(location.search);
    if (params.has('yolo-workers')) {
      return clampInt(params.get('yolo-workers'), 1, 4, defaultYoloWorkers);
    }
    return defaultYoloWorkers;
  }

  function getQueueLimits() {
    const expectedFps = getExpectedFps();
    const recogWorkers = getWorkerCount();
    const yoloWorkers = getYoloWorkerCount();
    return {
      yolo: Math.max(8, Math.min(24, Math.ceil(expectedFps * 0.8) + yoloWorkers * 2)),
      precise: Math.max(10, Math.min(28, Math.ceil(expectedFps) + yoloWorkers * 3)),
      recog: Math.max(8, Math.min(24, Math.ceil(expectedFps * 0.6) + recogWorkers * 2)),
    };
  }

  function getMatchWorkerCount() {
    const params = new URLSearchParams(location.search);
    if (params.has('match-workers')) {
      return clampInt(params.get('match-workers'), 1, 4, Math.max(1, Math.min(2, getWorkerCount())));
    }
    return Math.max(1, Math.min(2, getWorkerCount()));
  }

  function disposeMatchWorkers() {
    for (let i = 0; i < bench.matchWorkers.length; i++) {
      try {
        bench.matchWorkers[i].terminate();
      } catch (_) {}
    }
    for (let i = 0; i < bench.matchQueue.length; i++) {
      const item = bench.matchQueue[i];
      if (item && item.payloadBuf) {
        try {
          item.payloadBuf = null;
        } catch (_) {}
      }
    }
    bench.matchWorkers = [];
    bench.matchQueue = [];
    bench.matchPoolSize = 0;
    bench.matchFrameVersion = -1;
  }

  function findIdleMatchWorker() {
    for (let i = 0; i < bench.matchWorkers.length; i++) {
      const worker = bench.matchWorkers[i];
      if (worker && worker.__ready && !worker.__busy) {
        return worker;
      }
    }
    return null;
  }

  function pumpMatchQueue() {
    while (bench.matchQueue.length) {
      const worker = findIdleMatchWorker();
      if (!worker) break;
      const item = bench.matchQueue.shift();
      worker.__busy = true;
      worker.postMessage({
        type: 'match',
        token: item.token,
        payloadBuf: item.payloadBuf,
      }, [item.payloadBuf]);
    }
  }

  function applyMatchResult(data) {
    if (!data || data.seq < 0 || !data.n) {
      bench.unknownCount++;
      return;
    }
    bench.lastBest10SymAcc = Number(data.bestSymAcc) || 0;
    bench.lastBest10PatAcc = Number(data.bestPatAcc) || 0;
    bench.lastBest10ColAcc = Number(data.bestColAcc) || 0;
    bench.lastTop2Gap = Number(data.top2Gap) || 0;
    bench.lastSeq = data.seq;
    bench.lastHamming = bench.lastBest10SymAcc.toFixed(3) + '%';
    if (bench.seen.has(data.seq)) bench.duplicateCount++;
    else {
      bench.seen.add(data.seq);
      bench.uniqueCount++;
    }
    bench.symbolTotal += data.n;
    bench.symbolCorrect += data.symbolCorrect;
    bench.patternCorrect += data.patternCorrect;
    bench.colorCorrect += data.colorCorrect;
  }

  function initMatchWorkers() {
    const targetCount = getMatchWorkerCount();
    if (bench.matchWorkers.length === targetCount && bench.matchFrameVersion === bench.frameVersion) {
      return;
    }
    disposeMatchWorkers();
    bench.matchPoolSize = targetCount;
    bench.matchFrameVersion = bench.frameVersion;

    const frames = bench.frameSet.map((frame) => ({
      seq: frame.seq,
      payloadSymbols: Array.from(frame.payloadSymbols),
      payloadBytes: Array.from(frame.payloadBytes),
    }));

    for (let i = 0; i < targetCount; i++) {
      const worker = new Worker(MATCH_WORKER_URL);
      worker.__ready = false;
      worker.__busy = false;
      worker.onmessage = (event) => {
        const data = event.data;
        if (data.type === 'ready') {
          worker.__ready = true;
          worker.__busy = false;
          pumpMatchQueue();
          return;
        }
        worker.__busy = false;
        if (bench.running && data.type === 'match') {
          applyMatchResult(data);
        }
        pumpMatchQueue();
      };
      worker.onerror = () => {
        worker.__ready = false;
        worker.__busy = false;
      };
      bench.matchWorkers.push(worker);
      worker.postMessage({
        type: 'init',
        payloadSymbolCount: codebook.PAYLOAD_SYMBOLS,
        frames,
      });
    }
  }

  function enqueueDecodedMatch(payloadBuf, ms) {
    bench.decodedCount++;
    bench.recogMsSum += Number(ms) || 0;
    initMatchWorkers();
    bench.matchQueue.push({
      token: ++bench.matchToken,
      payloadBuf: payloadBuf.slice(0),
    });
    pumpMatchQueue();
  }

  function syncRuntimeConfig() {
    const workerCount = getWorkerCount();
    const yoloWorkerCount = getYoloWorkerCount();
    const queueLimits = getQueueLimits();
    global.__CAMDROP_RECOG_WORKERS = workerCount;
    global.__CAMDROP_YOLO_WORKERS = yoloWorkerCount;
    global.__CAMDROP_YOLO_QUEUE_MAX = queueLimits.yolo;
    global.__CAMDROP_PRECISE_QUEUE_MAX = queueLimits.precise;
    global.__CAMDROP_RECOG_QUEUE_MAX = queueLimits.recog;

    const app = global.CameraDropApp;
    if (!app || !app.state) {
      return;
    }

    app.state.recogDictPromise = null;

    const dictKey = getCurrentDictKey();
    applyRecognizerDictGlobals();

    if (typeof app.disposeRecognizerWorkers === 'function'
        && app.state.recogWorkers
        && app.state.recogWorkers.length
        && (app.state.recogWorkers.length !== workerCount || app.state.recogWorkerDictConfigKey !== dictKey)) {
      app.disposeRecognizerWorkers();
    }

    if (typeof app.disposeYoloWorkers === 'function'
        && app.state.yoloWorkers
        && app.state.yoloWorkers.length
        && app.state.yoloWorkers.length !== yoloWorkerCount) {
      app.disposeYoloWorkers();
    }
  }

  function rebuildFrameSet() {
    bench.frameSet = codebook.buildFrameSet(null, getSeed(), getFrameCount());
    bench.frameVersion++;
    dom.frameCountInput.value = String(getFrameCount());
    dom.frameCountInput.readOnly = true;
    if (dom.modeSelect) {
      dom.modeSelect.innerHTML = '<option value="nearest">nearest10</option>';
      dom.modeSelect.value = 'nearest';
      dom.modeSelect.disabled = true;
    }
    syncRuntimeConfig();
    disposeMatchWorkers();
    initMatchWorkers();
  }

  function resetMetrics() {
    const app = global.CameraDropApp;
    if (app && typeof app.resetPipelineCounters === 'function') {
      app.resetPipelineCounters();
    }
    if (app && typeof app.flushRecognizerOrderedResults === 'function') {
      app.flushRecognizerOrderedResults();
    }
    bench.startTime = performance.now();
    bench.minRecogId = app && app.state ? ((app.state.recogSeq | 0) + 1) : 0;
    bench.captureEpoch = app && typeof app.bumpRecognizerCaptureEpoch === 'function' ? app.bumpRecognizerCaptureEpoch() : 0;
    bench.seen = new Set();
    bench.decodedCount = 0;
    bench.uniqueCount = 0;
    bench.skippedCount = 0;
    bench.duplicateCount = 0;
    bench.unknownCount = 0;
    bench.symbolTotal = 0;
    bench.symbolCorrect = 0;
    bench.patternCorrect = 0;
    bench.colorCorrect = 0;
    bench.recogMsSum = 0;
    bench.lastSeq = '-';
    bench.lastHamming = '-';
    bench.lastBest10SymAcc = 0;
    bench.lastBest10PatAcc = 0;
    bench.lastBest10ColAcc = 0;
    bench.lastTop2Gap = 0;
  }

  function updateShare() {
    const q = new URLSearchParams();
    const choice = getCurrentDictChoice();
    const spec = getCurrentDictSpec();
    q.set('seed', String(getSeed()));
    q.set('fps', String(getExpectedFps()));
    q.set('frames', String(getFrameCount()));
    q.set('dict', choice);
    if (spec.mode !== 'builtin') {
      q.set('dict-base', spec.base);
    }
    dom.shareText.textContent = 'bench-player.html?' + q.toString();
  }

  function updateStats() {
    const app = global.CameraDropApp;
    const elapsedMs = bench.running ? (performance.now() - bench.startTime) : 0;
    const elapsedSec = Math.max(elapsedMs / 1000, 1e-6);
    const symAcc = bench.symbolTotal ? (100 * bench.symbolCorrect / bench.symbolTotal) : 0;
    const patAcc = bench.symbolTotal ? (100 * bench.patternCorrect / bench.symbolTotal) : 0;
    const colAcc = bench.symbolTotal ? (100 * bench.colorCorrect / bench.symbolTotal) : 0;
    const expectedGen = Math.round(elapsedSec * getExpectedFps());
    const missingApprox = Math.max(0, expectedGen - bench.decodedCount);
    const yoloFpsArr = app && app.state ? app.state.yoloFpsArr : [];
    const yoloFps = yoloFpsArr && yoloFpsArr.length
      ? (yoloFpsArr.reduce((a, b) => a + b, 0) / yoloFpsArr.length)
      : 0;
    const preciseQueueLen = app && app.state ? app.state.preciseQueue.length : 0;
    const yoloQueueLen = app && app.state ? app.state.yoloQueue.length : 0;
    const recogQueueLen = app && app.state ? app.state.recogQueue.length : 0;
    const preciseDrop = app && app.state ? (app.state.preciseQueueDropCount || 0) : 0;
    const yoloDrop = app && app.state ? (app.state.yoloQueueDropCount || 0) : 0;
    const recogDrop = app && app.state ? (app.state.recogQueueDropCount || 0) : 0;
    const matchQueueLen = bench.matchQueue.length;
    const queueLen = preciseQueueLen + yoloQueueLen + recogQueueLen + matchQueueLen;
    const avgMs = bench.decodedCount > 0 ? (bench.recogMsSum / bench.decodedCount) : 0;

    dom.elapsed.textContent = elapsedSec.toFixed(2) + 's';
    dom.decoded.textContent = String(bench.decodedCount);
    dom.unique.textContent = String(bench.uniqueCount);
    dom.skipped.textContent = String(bench.skippedCount);
    dom.duplicate.textContent = String(bench.duplicateCount);
    dom.unknown.textContent = String(bench.unknownCount);
    dom.symbolAcc.textContent = symAcc.toFixed(3) + '%';
    dom.patternAcc.textContent = patAcc.toFixed(3) + '%';
    dom.colorAcc.textContent = colAcc.toFixed(3) + '%';
    dom.best10SymAcc.textContent = bench.lastBest10SymAcc.toFixed(3) + '%';
    dom.best10PatAcc.textContent = bench.lastBest10PatAcc.toFixed(3) + '%';
    dom.best10ColAcc.textContent = bench.lastBest10ColAcc.toFixed(3) + '%';
    dom.top2Gap.textContent = bench.lastTop2Gap.toFixed(3) + 'pp';
    dom.decodeFps.textContent = (bench.decodedCount / elapsedSec).toFixed(2);
    dom.uniqueFps.textContent = (bench.uniqueCount / elapsedSec).toFixed(2);
    dom.yoloFps.textContent = yoloFps.toFixed(2);
    dom.recogMs.textContent = avgMs.toFixed(2) + 'ms';
    dom.queue.textContent = String(queueLen)
      + ' (p' + preciseQueueLen + '/y' + yoloQueueLen + '/r' + recogQueueLen + '/m' + matchQueueLen + ')'
      + ' drop(' + preciseDrop + '/' + yoloDrop + '/' + recogDrop + ')';
    dom.expectedGen.textContent = String(expectedGen);
    dom.missingApprox.textContent = String(missingApprox);
    dom.lastSeq.textContent = String(bench.lastSeq) + ' / ACC=' + String(bench.lastHamming);
  }

  function startBench() {
    if (bench.running) return;
    hookRecognizer();
    syncRuntimeConfig();
    disposeMatchWorkers();
    initMatchWorkers();
    resetMetrics();
    bench.running = true;
    dom.startBtn.disabled = true;
    dom.stopBtn.disabled = false;
  }

  function stopBench() {
    if (!bench.running) return;
    bench.running = false;
    disposeMatchWorkers();
    dom.startBtn.disabled = false;
    dom.stopBtn.disabled = true;
    updateStats();
  }

  function bindRecognizerWorkers() {
    const app = global.CameraDropApp;
    if (!app || !app.state || !app.state.recogWorkers || !app.state.recogWorkers.length) {
      return;
    }
    for (let i = 0; i < app.state.recogWorkers.length; i++) {
      const worker = app.state.recogWorkers[i];
      if (!worker) continue;
      worker.onmessage = app.handleRecognizerWorkerMsg || app.handleRecognizerMsg;
      worker.onerror = app.handleRecognizerErr;
    }
  }

  function hookRecognizer() {
    const app = global.CameraDropApp;
    if (!app || typeof app.handleRecognizerMsg !== 'function') return false;
    if (!app.__benchHooked) {
      app.__benchHooked = true;
      const originalHandle = app.handleRecognizerMsg;
      const originalSkip = typeof app.reportRecognizerSkip === 'function' ? app.reportRecognizerSkip : null;

      app.handleRecognizerMsg = function wrappedHandleRecognizerMsg(event) {
        const d = event && event.data;
        if (bench.running && d && d.type === 'result') {
          if ((Number(d.epoch) | 0) !== (bench.captureEpoch | 0)) {
            return originalHandle.call(this, event);
          }
          if (typeof d.id === 'number' && d.id < (bench.minRecogId || 0)) {
            return originalHandle.call(this, event);
          }
          if (d.skipped) {
            bench.skippedCount++;
          } else if (d.payloadBuf) {
            enqueueDecodedMatch(d.payloadBuf, d.ms);
          }
        }
        return originalHandle.call(this, event);
      };

      if (originalSkip) {
        app.reportRecognizerSkip = function wrappedReportRecognizerSkip() {
          if (bench.running) {
            bench.skippedCount++;
          }
          return originalSkip.apply(this, arguments);
        };
      }
    }
    bindRecognizerWorkers();
    return true;
  }

  function applyQuery() {
    const q = new URLSearchParams(location.search);
    if (q.has('seed')) dom.seedInput.value = q.get('seed');
    if (q.has('fps')) dom.expectedFpsInput.value = q.get('fps');
    if (dom.dictSelect) {
      dom.dictSelect.value = getInitialDictChoice();
    }
    dom.frameCountInput.value = String(getFrameCount());
    applyRecognizerDictGlobals();
  }

  dom.startBtn.addEventListener('click', startBench);
  dom.stopBtn.addEventListener('click', stopBench);
  dom.resetBtn.addEventListener('click', () => {
    resetMetrics();
    updateStats();
  });
  dom.seedInput.addEventListener('change', () => {
    rebuildFrameSet();
    updateShare();
  });
  dom.frameCountInput.addEventListener('change', () => {
    dom.frameCountInput.value = String(getFrameCount());
    updateShare();
  });
  dom.expectedFpsInput.addEventListener('change', updateShare);
  if (dom.dictSelect) {
    dom.dictSelect.addEventListener('change', () => {
      syncRuntimeConfig();
      updateShare();
    });
  }

  applyQuery();
  rebuildFrameSet();
  updateShare();
  resetMetrics();
  updateStats();
  stopBench();

  const hookTimer = setInterval(() => {
    if (hookRecognizer()) clearInterval(hookTimer);
  }, 100);
  setInterval(updateStats, 200);
})(window);
