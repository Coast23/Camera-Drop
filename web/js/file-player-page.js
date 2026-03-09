(function (global) {
  'use strict';

  const codec = global.CamDropRectCodec;
  const render = global.CamDropRectRender;
  const common = global.CamDropRectTransferCommon;

  const dom = {
    fileInput: document.getElementById('fileInput'),
    widthInput: document.getElementById('widthInput'),
    heightInput: document.getElementById('heightInput'),
    strideInput: document.getElementById('strideInput'),
    marginInput: document.getElementById('marginInput'),
    reservedInput: document.getElementById('reservedInput'),
    fpsInput: document.getElementById('fpsInput'),
    packetCountInput: document.getElementById('packetCountInput'),
    renderScaleInput: document.getElementById('renderScaleInput'),
    prepareBtn: document.getElementById('prepareBtn'),
    playBtn: document.getElementById('playBtn'),
    stopBtn: document.getElementById('stopBtn'),
    fullscreenBtn: document.getElementById('fullscreenBtn'),
    copyScannerBtn: document.getElementById('copyScannerBtn'),
    openScannerLink: document.getElementById('openScannerLink'),
    playerCanvas: document.getElementById('playerCanvas'),
    prepareProgressBar: document.getElementById('prepareProgressBar'),
    prepareProgressText: document.getElementById('prepareProgressText'),
    statusText: document.getElementById('statusText'),
    topStatus: document.getElementById('topStatus'),
    layoutInfo: document.getElementById('layoutInfo'),
    fileInfo: document.getElementById('fileInfo'),
    packetInfo: document.getElementById('packetInfo'),
    playbackInfo: document.getElementById('playbackInfo'),
    shareInfo: document.getElementById('shareInfo'),
    frameInfo: document.getElementById('frameInfo'),
    liveFps: document.getElementById('liveFps'),
    loopInfo: document.getElementById('loopInfo'),
    canvasInfo: document.getElementById('canvasInfo'),
    logBox: document.getElementById('logBox'),
  };

  const state = {
    layout: null,
    file: null,
    fileBytes: 0,
    frameCanvases: [],
    playing: false,
    preparing: false,
    prepareRunId: 0,
    prepareWorkers: [],
    prepareWorkerWaiters: [],
    prepareJob: null,
    prepareDone: 0,
    prepareTotal: 0,
    rafId: 0,
    playStartedAt: 0,
    lastDrawnOrdinal: -1,
    drawCount: 0,
    playCursor: -1,
    lastFrameAt: 0,
  };

  function log(line) {
    const text = '[' + new Date().toLocaleTimeString() + '] ' + line;
    dom.logBox.textContent = text + '\n' + dom.logBox.textContent;
  }

  function setStatus(text) {
    dom.statusText.textContent = text;
    dom.topStatus.textContent = text;
  }

  function updatePrepareProgress(done, total, options) {
    const cfg = options || {};
    const fill = dom.prepareProgressBar;
    const text = dom.prepareProgressText;
    if (!fill || !text) {
      return;
    }
    const totalNum = Math.max(0, Number(total) || 0);
    const doneNum = Math.max(0, Number(done) || 0);
    state.prepareDone = doneNum;
    state.prepareTotal = totalNum;
    if (cfg.indeterminate) {
      fill.classList.add('indeterminate');
      fill.style.width = '26%';
      text.textContent = cfg.label || '准备中';
      return;
    }
    fill.classList.remove('indeterminate');
    const clampedDone = totalNum > 0 ? Math.min(doneNum, totalNum) : doneNum;
    const ratio = totalNum > 0 ? (clampedDone / totalNum) : 0;
    fill.style.width = (ratio * 100).toFixed(1) + '%';
    if (cfg.label) {
      text.textContent = cfg.label;
      return;
    }
    if (totalNum > 0) {
      text.textContent = clampedDone + ' / ' + totalNum + ' (' + (ratio * 100).toFixed(1) + '%)';
      return;
    }
    text.textContent = clampedDone + ' / 0';
  }

  function resetPrepareProgress(label) {
    updatePrepareProgress(0, 0, { label: label || '0 / 0' });
  }

  function clampNumber(value, fallback, lo, hi) {
    const n = Number(value);
    if (!Number.isFinite(n)) {
      return fallback;
    }
    return Math.max(lo, Math.min(hi, n));
  }

  function nowMs() {
    return global.performance && typeof global.performance.now === 'function'
      ? global.performance.now()
      : Date.now();
  }

  function yieldToBrowser() {
    if (global.scheduler && typeof global.scheduler.yield === 'function') {
      return global.scheduler.yield();
    }
    return new Promise(function (resolve) {
      global.setTimeout(resolve, 0);
    });
  }

  function assetVersion() {
    if (typeof global.__CAMDROP_ASSET_VERSION === 'string' && global.__CAMDROP_ASSET_VERSION) {
      return global.__CAMDROP_ASSET_VERSION;
    }
    return String(Date.now());
  }

  function codecBase() {
    if (typeof global.__CAMDROP_RECT_CODEC_BASE === 'string' && global.__CAMDROP_RECT_CODEC_BASE) {
      return global.__CAMDROP_RECT_CODEC_BASE;
    }
    return new URL('./js/vendor', global.location.href).href;
  }

  function workerScriptUrl() {
    return './js/file-player-prepare-worker.js?v=' + encodeURIComponent(assetVersion());
  }

  function getPrepareWorkerTargetCount() {
    const hc = Math.max(0, Math.round(Number(global.navigator && global.navigator.hardwareConcurrency) || 0));
    if (!hc) {
      return 3;
    }
    return Math.max(2, Math.min(4, Math.max(2, hc - 1)));
  }

  function countBusyPrepareWorkers() {
    let busy = 0;
    for (let i = 0; i < state.prepareWorkers.length; i++) {
      if (state.prepareWorkers[i] && state.prepareWorkers[i].__busy) {
        busy += 1;
      }
    }
    return busy;
  }

  function hasPlaybackStartFrame() {
    return !!(state.frameCanvases && state.frameCanvases[0]);
  }

  function countReadyFrames() {
    let ready = 0;
    for (let i = 0; i < state.frameCanvases.length; i++) {
      if (state.frameCanvases[i]) {
        ready += 1;
      }
    }
    return ready;
  }

  function contiguousReadyFrames() {
    let ready = 0;
    while (ready < state.frameCanvases.length && state.frameCanvases[ready]) {
      ready += 1;
    }
    return ready;
  }

  function closeFrame(frame) {
    if (frame && typeof frame.close === 'function') {
      try {
        frame.close();
      } catch (_) {}
    }
  }

  function clearPreparedFrames() {
    for (let i = 0; i < state.frameCanvases.length; i++) {
      closeFrame(state.frameCanvases[i]);
    }
    state.frameCanvases = [];
    state.playCursor = -1;
    state.lastFrameAt = 0;
    state.lastDrawnOrdinal = -1;
    dom.frameInfo.textContent = 'frame - / -';
    dom.loopInfo.textContent = '0';
    dom.liveFps.textContent = '-';
    dom.canvasInfo.textContent = '-';
    resetPrepareProgress();
  }

  function canUsePrepareWorkers() {
    return typeof global.Worker === 'function'
      && typeof global.OffscreenCanvas !== 'undefined'
      && typeof global.ImageBitmap !== 'undefined';
  }

  function notifyPrepareWorkerWaiters() {
    if (!state.prepareWorkerWaiters.length) {
      return;
    }
    const waiters = state.prepareWorkerWaiters.splice(0, state.prepareWorkerWaiters.length);
    for (let i = 0; i < waiters.length; i++) {
      waiters[i]();
    }
  }

  function terminatePrepareWorkers() {
    for (let i = 0; i < state.prepareWorkers.length; i++) {
      const worker = state.prepareWorkers[i];
      if (!worker) {
        continue;
      }
      try {
        worker.terminate();
      } catch (_) {}
    }
    state.prepareWorkers = [];
    state.prepareWorkerWaiters = [];
  }

  function settlePrepareJob(job, result, error) {
    if (!job || job.settled) {
      return;
    }
    job.settled = true;
    if (state.prepareJob === job) {
      state.prepareJob = null;
    }
    if (error) {
      job.reject(error);
      return;
    }
    job.resolve(result || null);
  }

  function resetLoopPlaybackClock(now) {
    state.playStartedAt = Number.isFinite(now) ? now : performance.now();
    state.lastDrawnOrdinal = -1;
    state.playCursor = -1;
    state.lastFrameAt = 0;
    state.drawCount = 0;
    dom.loopInfo.textContent = '0';
  }

  function maybeFinishPrepareJob(job) {
    if (!job || job.settled || job.cancelled) {
      return;
    }
    if (!job.dispatchDone || job.doneCount < job.packetCount) {
      return;
    }
    dom.playbackInfo.textContent = '已预生成，等待播放';
    updatePrepareProgress(job.packetCount, job.packetCount);
    if (state.frameCanvases[0]) {
      drawFrame(0);
    }
    if (state.playing) {
      resetLoopPlaybackClock(nowMs());
    }
    setStatus('预生成完成');
    log('prepared ' + job.packetCount + ' frames for ' + (state.file ? state.file.name : 'file') + ' using ' + job.workerCount + ' workers');
    settlePrepareJob(job, { cancelled: false }, null);
  }

  function failPrepareJob(runId, error) {
    const job = state.prepareJob;
    if (!job || job.runId !== runId || job.settled) {
      return;
    }
    terminatePrepareWorkers();
    if (runId === state.prepareRunId) {
      state.preparing = false;
    }
    updateButtons();
    settlePrepareJob(job, null, error instanceof Error ? error : new Error(String(error)));
  }

  function getIdlePrepareWorker() {
    for (let i = 0; i < state.prepareWorkers.length; i++) {
      const worker = state.prepareWorkers[i];
      if (worker && worker.__ready && !worker.__busy) {
        return worker;
      }
    }
    return null;
  }

  async function waitForIdlePrepareWorker(runId) {
    while (runId === state.prepareRunId) {
      const worker = getIdlePrepareWorker();
      if (worker) {
        return worker;
      }
      await new Promise(function (resolve) {
        state.prepareWorkerWaiters.push(resolve);
      });
    }
    return null;
  }

  function handlePrepareWorkerError(event) {
    const worker = this;
    if (worker) {
      worker.__busy = false;
      if (!worker.__ready && typeof worker.__rejectReady === 'function') {
        worker.__rejectReady(new Error(event && event.message ? event.message : 'prepare worker init error'));
      }
    }
    notifyPrepareWorkerWaiters();
    const message = event && event.message ? event.message : 'prepare worker error';
    const runId = worker && Number.isFinite(worker.__runId) ? worker.__runId : state.prepareRunId;
    failPrepareJob(runId, new Error(message));
  }

  function handlePrepareWorkerMessage(event) {
    const worker = this;
    const data = event && event.data ? event.data : {};
    if (data.type === 'ready') {
      if (worker) {
        worker.__ready = true;
        worker.__busy = false;
        if (typeof worker.__resolveReady === 'function') {
          worker.__resolveReady(true);
        }
        worker.__resolveReady = null;
        worker.__rejectReady = null;
      }
      notifyPrepareWorkerWaiters();
      return;
    }

    if (worker) {
      worker.__busy = false;
    }
    notifyPrepareWorkerWaiters();

    const runId = Number(data.runId) || 0;
    if (data.type === 'frame') {
      if (!runId || runId !== state.prepareRunId) {
        closeFrame(data.bitmap);
        return;
      }
      const job = state.prepareJob;
      if (!job || job.runId !== runId || job.settled) {
        closeFrame(data.bitmap);
        return;
      }
      const index = Math.max(0, Math.min(job.packetCount - 1, Math.round(Number(data.index) || 0)));
      if (data.bitmap) {
        const prev = state.frameCanvases[index];
        if (prev && prev !== data.bitmap) {
          closeFrame(prev);
        }
        state.frameCanvases[index] = data.bitmap;
      }
      job.doneCount += 1;
      if (index === 0 && state.frameCanvases[0]) {
        drawFrame(0);
      }
      const elapsedSec = Math.max((nowMs() - job.startedAt) / 1000, 1e-6);
      const prepFps = job.doneCount / elapsedSec;
      updatePrepareProgress(job.doneCount, job.packetCount);
      setStatus('预生成 ' + job.doneCount + ' / ' + job.packetCount);
      dom.playbackInfo.textContent = '预生成中 | ' + job.doneCount + ' / ' + job.packetCount + ' | ' + prepFps.toFixed(1) + ' 帧/秒 | workers ' + countBusyPrepareWorkers() + '/' + job.workerCount;
      updateButtons();
      maybeFinishPrepareJob(job);
      return;
    }

    if (data.type === 'error') {
      if (runId && runId === state.prepareRunId) {
        failPrepareJob(runId, new Error(data.message || 'prepare worker failed'));
      }
    }
  }

  async function ensurePrepareWorkerPool() {
    const targetCount = getPrepareWorkerTargetCount();
    const version = assetVersion();
    const base = codecBase();
    let reusable = state.prepareWorkers.length === targetCount;
    if (reusable) {
      for (let i = 0; i < state.prepareWorkers.length; i++) {
        const worker = state.prepareWorkers[i];
        if (!worker || worker.__assetVersion !== version || worker.__codecBase !== base) {
          reusable = false;
          break;
        }
      }
    }
    if (!reusable) {
      terminatePrepareWorkers();
      state.prepareWorkers = [];
      for (let i = 0; i < targetCount; i++) {
        const worker = new Worker(workerScriptUrl());
        worker.__assetVersion = version;
        worker.__codecBase = base;
        worker.__ready = false;
        worker.__busy = false;
        worker.__runId = 0;
        worker.onmessage = handlePrepareWorkerMessage;
        worker.onerror = handlePrepareWorkerError;
        worker.__readyPromise = new Promise(function (resolve, reject) {
          worker.__resolveReady = resolve;
          worker.__rejectReady = reject;
        });
        state.prepareWorkers.push(worker);
        worker.postMessage({
          type: 'init',
          assetVersion: version,
          codecBase: base,
        });
      }
    }
    const readyPromises = [];
    for (let i = 0; i < state.prepareWorkers.length; i++) {
      readyPromises.push(state.prepareWorkers[i].__readyPromise || Promise.resolve(true));
    }
    await Promise.all(readyPromises);
    return state.prepareWorkers;
  }

  function cancelPreparation() {
    if (!state.preparing) {
      return false;
    }
    const runId = state.prepareRunId;
    state.prepareRunId += 1;
    state.preparing = false;
    if (state.rafId) {
      cancelAnimationFrame(state.rafId);
      state.rafId = 0;
    }
    state.playing = false;
    const job = state.prepareJob;
    if (job && job.runId === runId) {
      job.cancelled = true;
      settlePrepareJob(job, { cancelled: true }, null);
    }
    terminatePrepareWorkers();
    clearPreparedFrames();
    dom.playbackInfo.textContent = '已取消';
    resetPrepareProgress('已取消');
    updateButtons();
    return true;
  }

  function buildPageUrl(pathname, layout, extras) {
    const url = new URL(pathname, global.location.href);
    const params = common.makeLayoutQuery(layout || common.readLayoutInputs(dom));
    if (extras && typeof extras === 'object') {
      Object.keys(extras).forEach(function (key) {
        const value = extras[key];
        if (value == null || value === '') {
          return;
        }
        params.set(key, String(value));
      });
    }
    url.search = params.toString();
    return url.href;
  }

  function getScannerCaptureExtras() {
    const fps = clampNumber(dom.fpsInput.value, 12, 0.2, 240);
    return {
      fps: String(Math.round(fps * 1000) / 1000),
      'samples-per-code': 3,
    };
  }

  function updateButtons() {
    const hasFrames = hasPlaybackStartFrame();
    dom.prepareBtn.disabled = state.preparing;
    dom.playBtn.disabled = !hasFrames || state.playing;
    dom.stopBtn.disabled = !state.playing && !state.preparing;
    dom.fullscreenBtn.disabled = !hasFrames;
  }

  function refreshShareLink() {
    const href = buildPageUrl('./file-scanner.html', state.layout || common.readLayoutInputs(dom), getScannerCaptureExtras());
    dom.openScannerLink.href = href;
    dom.shareInfo.textContent = href;
  }

  async function syncLayout(pushUrl) {
    const layout = await common.applyLayoutInputs(dom);
    state.layout = layout;
    dom.layoutInfo.textContent = common.formatLayout(layout);
    refreshShareLink();
    if (pushUrl && history && history.replaceState) {
      history.replaceState(null, '', buildPageUrl('./file-player.html', layout));
    }
    return layout;
  }

  function drawFrame(index) {
    const frame = state.frameCanvases[index];
    if (!frame) {
      return;
    }
    if (dom.playerCanvas.width !== frame.width || dom.playerCanvas.height !== frame.height) {
      dom.playerCanvas.width = frame.width;
      dom.playerCanvas.height = frame.height;
      dom.canvasInfo.textContent = frame.width + ' x ' + frame.height;
    }
    const ctx = dom.playerCanvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, dom.playerCanvas.width, dom.playerCanvas.height);
    ctx.drawImage(frame, 0, 0);
    dom.frameInfo.textContent = 'frame ' + (index + 1) + ' / ' + state.frameCanvases.length;
  }

  function stopPlayback() {
    if (state.rafId) {
      cancelAnimationFrame(state.rafId);
      state.rafId = 0;
    }
    state.playing = false;
    state.playCursor = -1;
    state.lastFrameAt = 0;
    state.lastDrawnOrdinal = -1;
    updateButtons();
  }

  function playbackTick(now) {
    if (!state.playing) {
      return;
    }
    const fps = clampNumber(dom.fpsInput.value, 12, 0.2, 240);
    const frameDuration = 1000 / fps;

    if (state.preparing) {
      const readyPrefix = contiguousReadyFrames();
      if (readyPrefix <= 0) {
        dom.playbackInfo.textContent = '播放中 | 等待首帧';
        state.rafId = requestAnimationFrame(playbackTick);
        return;
      }
      if (state.playCursor < 0) {
        state.playCursor = 0;
        state.drawCount = 1;
        state.lastFrameAt = now;
        drawFrame(0);
      } else if ((now - state.lastFrameAt) >= frameDuration) {
        const nextIndex = state.playCursor + 1;
        if (nextIndex < readyPrefix) {
          state.playCursor = nextIndex;
          state.drawCount += 1;
          state.lastFrameAt = now;
          drawFrame(nextIndex);
        }
      }
      const nextNeeded = state.playCursor + 1;
      dom.liveFps.textContent = fps.toFixed(1);
      dom.loopInfo.textContent = '0';
      if (nextNeeded < readyPrefix) {
        dom.playbackInfo.textContent = '播放中 | 顺序播放 ' + (state.playCursor + 1) + ' / ' + state.frameCanvases.length;
      } else {
        dom.playbackInfo.textContent = '播放中 | 等待下一帧 ' + Math.min(nextNeeded + 1, state.frameCanvases.length) + ' / ' + state.frameCanvases.length;
      }
      state.rafId = requestAnimationFrame(playbackTick);
      return;
    }

    if (!state.frameCanvases.length || !hasPlaybackStartFrame()) {
      state.rafId = requestAnimationFrame(playbackTick);
      return;
    }

    const ordinal = Math.floor((now - state.playStartedAt) / frameDuration);
    if (ordinal !== state.lastDrawnOrdinal) {
      state.lastDrawnOrdinal = ordinal;
      state.drawCount += 1;
      const frameIndex = ordinal % state.frameCanvases.length;
      drawFrame(frameIndex);
      dom.loopInfo.textContent = String(Math.floor(ordinal / state.frameCanvases.length));
      dom.liveFps.textContent = fps.toFixed(1);
      dom.playbackInfo.textContent = '播放中 | target ' + fps.toFixed(1) + ' fps | 已绘制 ' + state.drawCount + ' 帧';
    }
    state.rafId = requestAnimationFrame(playbackTick);
  }

  function startPlayback() {
    if (!hasPlaybackStartFrame() || state.playing) {
      return;
    }
    state.playing = true;
    resetLoopPlaybackClock(performance.now());
    setStatus('播放中');
    updateButtons();
    state.rafId = requestAnimationFrame(playbackTick);
  }

  async function prepareFramesOnMain(file, fileBytes, prepareRunId, renderScale, packetCount) {
    const capacity = common.getCapacityInfo ? common.getCapacityInfo(state.layout) : null;
    const encoder = await codec.createEncoder(fileBytes, file.name);
    try {
      const recommended = Math.max(1, encoder.packetCountRecommended());
      const actualPacketCount = packetCount > 0 ? packetCount : common.autoPacketCount(recommended);
      const redundancyRatio = recommended > 0 ? (actualPacketCount / recommended) : 1;
      const startedAt = nowMs();
      let lastYieldAt = startedAt;
      let lastUiAt = 0;
      clearPreparedFrames();
      dom.packetInfo.textContent = '推荐 ' + recommended + ' 帧，准备 ' + actualPacketCount + ' 帧'
        + ' | 冗余 ' + redundancyRatio.toFixed(2) + 'x'
        + (capacity && capacity.fileBytesPerFrame ? (' | 单码 file ' + common.formatBytes(capacity.fileBytesPerFrame)) : '');
      dom.playbackInfo.textContent = '预生成中 | 0 / ' + actualPacketCount;
      updatePrepareProgress(0, actualPacketCount);
      for (let i = 0; i < actualPacketCount; i++) {
        if (prepareRunId !== state.prepareRunId) {
          return { cancelled: true };
        }
        const packet = await encoder.getPacket();
        const frameCanvas = document.createElement('canvas');
        await render.renderPacketToCanvas(frameCanvas, packet, {
          scale: renderScale,
          cooperativeYield: true,
          yieldEveryRows: 14,
          yieldFn: yieldToBrowser,
        });
        state.frameCanvases.push(frameCanvas);
        if (i === 0) {
          drawFrame(0);
        }
        updateButtons();
        const done = i + 1;
        const now = nowMs();
        updatePrepareProgress(done, actualPacketCount);
        if ((now - lastUiAt) >= 80 || done === actualPacketCount) {
          const elapsedSec = Math.max((now - startedAt) / 1000, 1e-6);
          const prepFps = done / elapsedSec;
          setStatus('预生成 ' + done + ' / ' + actualPacketCount);
          dom.playbackInfo.textContent = '预生成中 | ' + done + ' / ' + actualPacketCount + ' | ' + prepFps.toFixed(1) + ' 帧/秒';
          lastUiAt = now;
        }
        if (done < actualPacketCount && (now - lastYieldAt) >= 24) {
          lastYieldAt = now;
          await yieldToBrowser();
        }
      }
      dom.playbackInfo.textContent = '已预生成，等待播放';
      updatePrepareProgress(actualPacketCount, actualPacketCount);
      drawFrame(0);
      if (state.playing) {
        resetLoopPlaybackClock(nowMs());
      }
      setStatus('预生成完成');
      log('prepared ' + actualPacketCount + ' frames for ' + file.name + ' (' + fileBytes.length + ' bytes)');
      return { cancelled: false };
    } finally {
      encoder.destroy();
    }
  }

  async function prepareFramesWithWorkers(file, fileBytes, prepareRunId, renderScale, packetCount) {
    clearPreparedFrames();
    dom.packetInfo.textContent = '初始化渲染 workers';
    dom.playbackInfo.textContent = '后台预生成启动中';
    updatePrepareProgress(0, 0, { indeterminate: true, label: '初始化渲染 workers' });

    const workers = await ensurePrepareWorkerPool();
    if (prepareRunId !== state.prepareRunId) {
      return { cancelled: true };
    }

    const capacity = common.getCapacityInfo ? common.getCapacityInfo(state.layout) : null;
    const encoder = await codec.createEncoder(fileBytes, file.name);
    let job = null;
    try {
      const recommended = Math.max(1, encoder.packetCountRecommended());
      const actualPacketCount = packetCount > 0 ? packetCount : common.autoPacketCount(recommended);
      const redundancyRatio = recommended > 0 ? (actualPacketCount / recommended) : 1;
      state.frameCanvases = new Array(actualPacketCount);
      dom.packetInfo.textContent = '推荐 ' + recommended + ' 帧，准备 ' + actualPacketCount + ' 帧'
        + ' | 冗余 ' + redundancyRatio.toFixed(2) + 'x'
        + (capacity && capacity.fileBytesPerFrame ? (' | 单码 file ' + common.formatBytes(capacity.fileBytesPerFrame)) : '')
        + ' | workers ' + workers.length;
      dom.playbackInfo.textContent = '预生成中 | 0 / ' + actualPacketCount + ' | workers 0/' + workers.length;
      updatePrepareProgress(0, actualPacketCount);

      let resolveJob = null;
      let rejectJob = null;
      const resultPromise = new Promise(function (resolve, reject) {
        resolveJob = resolve;
        rejectJob = reject;
      });
      job = {
        runId: prepareRunId,
        startedAt: nowMs(),
        packetCount: actualPacketCount,
        doneCount: 0,
        workerCount: workers.length,
        dispatchDone: false,
        cancelled: false,
        settled: false,
        resolve: resolveJob,
        reject: rejectJob,
      };
      state.prepareJob = job;

      for (let index = 0; index < actualPacketCount; index++) {
        if (prepareRunId !== state.prepareRunId || job.cancelled || job.settled) {
          break;
        }
        const worker = await waitForIdlePrepareWorker(prepareRunId);
        if (!worker || prepareRunId !== state.prepareRunId || job.cancelled || job.settled) {
          break;
        }
        const packet = await encoder.getPacket();
        if (prepareRunId !== state.prepareRunId || job.cancelled || job.settled) {
          break;
        }
        worker.__busy = true;
        worker.__runId = prepareRunId;
        worker.postMessage({
          type: 'render',
          runId: prepareRunId,
          index: index,
          layout: state.layout,
          renderScale: renderScale,
          assetVersion: assetVersion(),
          codecBase: codecBase(),
          packetBytes: packet.buffer,
        }, [packet.buffer]);
        if (((index + 1) % 4) === 0) {
          await yieldToBrowser();
        }
      }

      if (state.prepareJob === job && !job.settled && !job.cancelled) {
        job.dispatchDone = true;
        maybeFinishPrepareJob(job);
      }
      return await resultPromise;
    } catch (error) {
      failPrepareJob(prepareRunId, error);
      throw error;
    } finally {
      encoder.destroy();
      if (job && state.prepareJob === job && !job.settled && prepareRunId !== state.prepareRunId) {
        settlePrepareJob(job, { cancelled: true }, null);
      }
    }
  }

  async function prepareFrames() {
    const file = dom.fileInput.files && dom.fileInput.files[0];
    if (!file) {
      setStatus('请先选择文件');
      return;
    }
    stopPlayback();
    cancelPreparation();
    state.preparing = true;
    const prepareRunId = ++state.prepareRunId;
    updateButtons();
    try {
      setStatus('应用布局');
      updatePrepareProgress(0, 0, { indeterminate: true, label: '应用布局' });
      await syncLayout(true);
      if (prepareRunId !== state.prepareRunId) {
        return;
      }

      setStatus('读取文件');
      updatePrepareProgress(0, 0, { indeterminate: true, label: '读取文件' });
      const fileBytes = new Uint8Array(await file.arrayBuffer());
      if (prepareRunId !== state.prepareRunId) {
        return;
      }
      state.file = file;
      state.fileBytes = fileBytes.length;
      dom.fileInfo.textContent = file.name + ' | ' + common.formatBytes(fileBytes.length);

      const rawPacketCount = Math.max(0, Math.round(Number(dom.packetCountInput.value) || 0));
      const renderScale = Math.max(1, Math.min(3, Math.round(Number(dom.renderScaleInput.value) || 1)));

      let result;
      if (canUsePrepareWorkers()) {
        result = await prepareFramesWithWorkers(file, fileBytes, prepareRunId, renderScale, rawPacketCount);
      } else {
        setStatus('创建编码器');
        updatePrepareProgress(0, 0, { indeterminate: true, label: '创建编码器' });
        result = await prepareFramesOnMain(file, fileBytes, prepareRunId, renderScale, rawPacketCount);
      }
      if (result && result.cancelled) {
        return;
      }
    } finally {
      if (prepareRunId === state.prepareRunId) {
        state.preparing = false;
      }
      updateButtons();
    }
  }

  async function copyScannerLink() {
    await syncLayout(true);
    const href = buildPageUrl('./file-scanner.html', state.layout, getScannerCaptureExtras());
    await common.copyText(href);
    setStatus('扫描页链接已复制');
    log('copied scanner link');
  }

  async function init() {
    try {
      const queryLayout = common.readLayoutFromQuery(location.search);
      if (Object.keys(queryLayout).length) {
        common.writeLayoutInputs(dom, {
          imgWidth: queryLayout.imgWidth || Number(dom.widthInput.value) || 1024,
          imgHeight: queryLayout.imgHeight || Number(dom.heightInput.value) || 1024,
          stride: queryLayout.stride || Number(dom.strideInput.value) || 9,
          margin: queryLayout.margin || Number(dom.marginInput.value) || 8,
          reservedCornerSide: queryLayout.reservedCornerSide || Number(dom.reservedInput.value) || 6,
        });
      }
      setStatus('加载 codec');
      await codec.loadModule();
      await syncLayout(false);
      dom.fileInfo.textContent = '-';
      dom.packetInfo.textContent = '-';
      dom.playbackInfo.textContent = '-';
      dom.frameInfo.textContent = 'frame - / -';
      dom.canvasInfo.textContent = '-';
      dom.liveFps.textContent = '-';
      resetPrepareProgress();
      setStatus('就绪');
      updateButtons();
    } catch (error) {
      setStatus('初始化失败');
      log(error && error.stack ? error.stack : String(error));
      throw error;
    }
  }

  dom.prepareBtn.addEventListener('click', function () {
    prepareFrames().catch(function (error) {
      setStatus('预生成失败');
      log(error && error.stack ? error.stack : String(error));
      updateButtons();
    });
  });

  dom.playBtn.addEventListener('click', function () {
    startPlayback();
  });

  dom.stopBtn.addEventListener('click', function () {
    if (state.playing) {
      stopPlayback();
      setStatus(state.preparing ? '继续预生成' : (hasPlaybackStartFrame() ? '已停止' : '等待文件'));
      return;
    }
    if (cancelPreparation()) {
      setStatus('已取消预生成');
      return;
    }
    stopPlayback();
    setStatus(hasPlaybackStartFrame() ? '已停止' : '等待文件');
  });

  dom.fullscreenBtn.addEventListener('click', function () {
    if (dom.playerCanvas.requestFullscreen) {
      dom.playerCanvas.requestFullscreen().catch(function (error) {
        log(error && error.stack ? error.stack : String(error));
      });
    }
  });

  dom.copyScannerBtn.addEventListener('click', function () {
    copyScannerLink().catch(function (error) {
      setStatus('复制失败');
      log(error && error.stack ? error.stack : String(error));
    });
  });

  [
    dom.widthInput,
    dom.heightInput,
    dom.strideInput,
    dom.marginInput,
    dom.reservedInput,
    dom.fpsInput,
    dom.packetCountInput,
    dom.renderScaleInput,
  ].forEach(function (input) {
    input.addEventListener('change', refreshShareLink);
  });

  global.addEventListener('beforeunload', function () {
    cancelPreparation();
    terminatePrepareWorkers();
    clearPreparedFrames();
  });

  init().catch(function () {});
})(window);
