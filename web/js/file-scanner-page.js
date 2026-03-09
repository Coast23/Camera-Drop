(function (global) {
  'use strict';

  const app = global.CameraDropApp;
  const codec = global.CamDropRectCodec;
  const recognizer = global.CamDropRectRecognizer;
  const common = global.CamDropRectTransferCommon;
  const samplingUi = global.CamDropCaptureSamplingUi;

  const dom = {
    widthInput: document.getElementById('widthInput'),
    heightInput: document.getElementById('heightInput'),
    strideInput: document.getElementById('strideInput'),
    marginInput: document.getElementById('marginInput'),
    reservedInput: document.getElementById('reservedInput'),
    captureFpsInput: document.getElementById('captureFpsInput'),
    samplesPerCodeInput: document.getElementById('samplesPerCodeInput'),
    applyLayoutBtn: document.getElementById('applyLayoutBtn'),
    resetBtn: document.getElementById('resetBtn'),
    saveBtn: document.getElementById('saveBtn'),
    cameraInfoBtn: document.getElementById('cameraInfoBtn'),
    statusText: document.getElementById('statusText'),
    topStatus: document.getElementById('topStatus'),
    layoutInfo: document.getElementById('layoutInfo'),
    progressText: document.getElementById('progressText'),
    fileInfo: document.getElementById('fileInfo'),
    locInfo: document.getElementById('locInfo'),
    decodeInfo: document.getElementById('decodeInfo'),
    dedupeInfo: document.getElementById('dedupeInfo'),
    cameraInfo: document.getElementById('cameraInfo'),
    statsTop: document.getElementById('statsTop'),
    progressFill: document.getElementById('progressFill'),
    deskewedCanvas: document.getElementById('deskewedCanvas'),
    decodeBar: document.getElementById('decodeBar'),
    statusBar: document.getElementById('statusBar'),
    logBox: document.getElementById('logBox'),
    cameraMeta: document.getElementById('cameraMeta'),
    loadBtn: document.getElementById('loadBtn'),
    scanHint: document.getElementById('scanHint'),
    hintText: document.getElementById('hintText'),
  };

  const state = {
    layout: null,
    decoder: null,
    outputBytes: null,
    outputName: '',
    loopId: 0,
    lastDeskewTime: 0,
    inFlight: 0,
    decodedPackets: 0,
    acceptedPackets: 0,
    errorCount: 0,
    lastDecodeMs: 0,
    lastAvgPatternDist: 0,
    lastUiAt: 0,
    decodeSession: 1,
    cameraGateActive: false,
    cameraGateTimer: 0,
    cvGatePatched: false,
    pendingDeskewToken: 0,
  };


  function readCaptureSampling() {
    if (!samplingUi) {
      return { fps: 0, samplesPerCode: 3 };
    }
    return samplingUi.readInputs(dom);
  }

  function initCaptureSamplingInputs() {
    if (!samplingUi) {
      return readCaptureSampling();
    }
    const querySampling = samplingUi.parseQuery(global.location.search);
    samplingUi.writeInputs(dom, querySampling);
    return syncCaptureSampling(false);
  }

  function syncCaptureSampling(pushUrl) {
    const sampling = samplingUi ? samplingUi.applyGlobals(readCaptureSampling()) : readCaptureSampling();
    if (pushUrl && global.history && typeof global.history.replaceState === 'function') {
      global.history.replaceState(null, '', buildPageUrl(state.layout || common.readLayoutInputs(dom), sampling));
    }
    return sampling;
  }

  function buildCaptureSamplingExtras(sampling) {
    return samplingUi ? samplingUi.buildExtras(sampling || readCaptureSampling()) : {};
  }

  function log(line) {
    if (!dom.logBox) {
      return;
    }
    const prefix = '[' + new Date().toLocaleTimeString() + '] ';
    dom.logBox.textContent = prefix + line + '\n' + dom.logBox.textContent;
  }

  function setStatus(text) {
    if (dom.statusText) dom.statusText.textContent = text;
    if (dom.topStatus) dom.topStatus.textContent = text;
  }

  function setStartButton(label, disabled) {
    const btn = app && app.dom ? app.dom.startBtn : null;
    if (!btn) {
      return;
    }
    btn.style.display = 'block';
    btn.textContent = label;
    btn.disabled = !!disabled;
  }

  function cameraHasUsableFrame() {
    const video = app && app.dom ? app.dom.video : null;
    return !!(app && app.state && app.state.cameraReady
      && video && video.readyState >= 2
      && video.videoWidth > 0 && video.videoHeight > 0);
  }

  function stopCameraGatePoll() {
    if (state.cameraGateTimer) {
      clearInterval(state.cameraGateTimer);
      state.cameraGateTimer = 0;
    }
  }

  function finishCameraGateReady() {
    stopCameraGatePoll();
    if (app && app.ui) {
      app.ui.setMsg('资源已就绪，点击开始扫描');
      app.ui.setProg(1.0);
      app.ui.setStatus('等待开始扫描');
    }
    if (dom.hintText) {
      dom.hintText.textContent = '请将码放入画面';
    }
    setStartButton('开始扫描', false);
  }

  function scheduleCameraGatePoll() {
    if (state.cameraGateTimer) {
      return;
    }
    state.cameraGateTimer = setInterval(function () {
      if (cameraHasUsableFrame()) {
        finishCameraGateReady();
        return;
      }
      if (app && typeof app.startCamera === 'function' && !app.state.cameraStartPromise && !app.state.cameraReady) {
        app.startCamera(false).catch(function () {});
      }
    }, 400);
  }

  function beginCameraGatePrewarm() {
    if (state.cameraGateActive) {
      return;
    }
    state.cameraGateActive = true;
    setStartButton('正在预热摄像头…', true);
    if (dom.hintText) {
      dom.hintText.textContent = '等待摄像头首帧';
    }
    if (app && app.ui) {
      app.ui.setMsg('OpenCV 就绪，正在预热摄像头…');
      app.ui.setProg(0.9);
      app.ui.setStatus('正在预热摄像头');
    }
    if (!app || typeof app.startCamera !== 'function') {
      state.cameraGateActive = false;
      setStartButton('开始扫描', false);
      return;
    }
    Promise.resolve(app.startCamera(false)).then(function (ok) {
      if (ok || cameraHasUsableFrame()) {
        finishCameraGateReady();
        return;
      }
      if (app && app.ui) {
        app.ui.setMsg('请先允许摄像头权限，并等待首帧显示');
        app.ui.setStatus('等待摄像头首帧');
      }
      setStartButton('等待摄像头首帧…', true);
      scheduleCameraGatePoll();
    }).catch(function (error) {
      log('camera prewarm failed: ' + (error && error.message ? error.message : String(error)));
      if (app && app.ui) {
        app.ui.setMsg('摄像头预热失败，请等待首帧或重试');
        app.ui.setStatus('等待摄像头首帧');
      }
      setStartButton('等待摄像头首帧…', true);
      scheduleCameraGatePoll();
    }).finally(function () {
      state.cameraGateActive = false;
    });
  }

  function patchCvReadyGate() {
    if (state.cvGatePatched) {
      return;
    }
    state.cvGatePatched = true;
    const gated = function () {
      if (app && app.state) {
        app.state.cvReady = true;
      }
      beginCameraGatePrewarm();
    };
    global.onCvReady = gated;
    if (app) {
      app.onCvReady = gated;
    }
  }

  function buildPageUrl(layout, sampling) {
    return common.makeShareUrl('./file-scanner.html', layout || state.layout || common.readLayoutInputs(dom), buildCaptureSamplingExtras(sampling));
  }

  function updateLayoutCanvas(layout) {
    if (!layout || !dom.deskewedCanvas) {
      return;
    }
    dom.deskewedCanvas.width = layout.imgWidth;
    dom.deskewedCanvas.height = layout.imgHeight;
  }

  function destroyDecoder() {
    if (state.decoder && typeof state.decoder.destroy === 'function') {
      try {
        state.decoder.destroy();
      } catch (_) {}
    }
    state.decoder = null;
  }

  async function resetDecoder() {
    destroyDecoder();
    state.decoder = await codec.createDecoder();
    state.outputBytes = null;
    state.outputName = '';
    state.lastDeskewTime = 0;
    state.pendingDeskewToken = 0;
    state.inFlight = 0;
    state.decodedPackets = 0;
    state.acceptedPackets = 0;
    state.errorCount = 0;
    state.lastDecodeMs = 0;
    state.lastAvgPatternDist = 0;
    state.decodeSession++;
    if (dom.saveBtn) {
      dom.saveBtn.disabled = true;
    }
  }

  async function syncLayout(pushUrl) {
    const layout = await common.applyLayoutInputs(dom);
    state.layout = layout;
    updateLayoutCanvas(layout);
    recognizer.dispose();
    await resetDecoder();
    if (dom.layoutInfo) {
      dom.layoutInfo.textContent = common.formatLayout(layout);
    }
    if (pushUrl && global.history && typeof global.history.replaceState === 'function') {
      global.history.replaceState(null, '', buildPageUrl(layout));
    }
    return layout;
  }

  function toggleCameraMeta() {
    if (!dom.cameraMeta) {
      return;
    }
    const nextHidden = !dom.cameraMeta.hidden;
    dom.cameraMeta.hidden = nextHidden;
    if (dom.cameraInfoBtn) {
      dom.cameraInfoBtn.textContent = nextHidden ? '显示相机信息' : '隐藏相机信息';
    }
  }

  async function finalizeOutputIfReady() {
    if (!state.decoder || state.outputBytes || !state.decoder.isComplete()) {
      return;
    }
    state.outputName = state.decoder.getFilename() || ('camera_drop_' + Date.now() + '.bin');
    state.outputBytes = state.decoder.getFileBytes();
    if (dom.saveBtn) {
      dom.saveBtn.disabled = !(state.outputBytes && state.outputBytes.length);
    }
    setStatus('接收完成');
    log('file complete: ' + state.outputName + ' (' + common.formatBytes(state.outputBytes.length) + ')');
  }

  function updateUi(force) {
    const now = Date.now();
    if (!force && now - state.lastUiAt < 120) {
      return;
    }
    state.lastUiAt = now;

    const uniqueBlocks = state.decoder ? state.decoder.getUniqueBlockCount() : 0;
    const requiredBlocks = state.decoder ? state.decoder.getRequiredBlockCount() : 0;
    const ratio = requiredBlocks > 0 ? Math.max(0, Math.min(1, uniqueBlocks / requiredBlocks)) : 0;
    const cornersOk = !!(app && app.state && app.state.lastCorners);
    const yoloMs = app && app.state ? (Number(app.state.yoloMs) || 0) : 0;
    const dskFps = app && app.state ? (Number(app.state.dskFps) || 0) : 0;
    const qYolo = app && app.state && Array.isArray(app.state.yoloQueue) ? app.state.yoloQueue.length : 0;
    const qPrecise = app && app.state && Array.isArray(app.state.preciseQueue) ? app.state.preciseQueue.length : 0;
    const qRecog = state.inFlight;
    const layout = state.layout;
    const fileText = state.outputBytes
      ? (state.outputName + ' | ' + common.formatBytes(state.outputBytes.length))
      : '-';

    if (dom.progressFill) {
      dom.progressFill.style.width = (ratio * 100).toFixed(2) + '%';
    }
    if (dom.progressText) {
      dom.progressText.textContent = requiredBlocks
        ? (uniqueBlocks + ' / ' + requiredBlocks + ' 块 (' + common.formatPercent(ratio * 100, 2) + ')')
        : '等待有效 packet';
    }
    if (dom.fileInfo) {
      dom.fileInfo.textContent = fileText;
    }
    if (dom.locInfo) {
      dom.locInfo.textContent = (cornersOk ? 'corners ok' : 'corners -')
        + ' | yolo ' + yoloMs.toFixed(1) + 'ms'
        + ' | deskew ' + dskFps.toFixed(1) + 'fps';
    }
    if (dom.decodeInfo) {
      dom.decodeInfo.textContent = 'decoded ' + state.decodedPackets
        + ' | accepted ' + state.acceptedPackets
        + ' | ms ' + state.lastDecodeMs.toFixed(1)
        + ' | dist ' + state.lastAvgPatternDist.toFixed(2);
    }
    if (dom.dedupeInfo) {
      const sampling = readCaptureSampling();
      dom.dedupeInfo.textContent = 'inflight ' + qRecog
        + ' | yoloQ ' + qYolo
        + ' | preciseQ ' + qPrecise
        + ' | errors ' + state.errorCount
        + ' | sample ' + (sampling.fps > 0 ? (sampling.fps + 'fps x' + sampling.samplesPerCode) : 'off');
    }
    if (dom.cameraInfo) {
      const ready = !!(app && app.state && app.state.cameraReady);
      const video = app && app.dom ? app.dom.video : null;
      const vw = video && video.videoWidth ? video.videoWidth : 0;
      const vh = video && video.videoHeight ? video.videoHeight : 0;
      dom.cameraInfo.textContent = (ready ? 'ready' : 'waiting')
        + ' | video ' + vw + 'x' + vh
        + ' | layout ' + (layout ? (layout.imgWidth + 'x' + layout.imgHeight) : '-');
    }
    if (dom.statsTop) {
      dom.statsTop.textContent = 'packets ' + state.decodedPackets + ' / ' + state.acceptedPackets
        + ' | unique ' + uniqueBlocks + ' / ' + requiredBlocks;
    }
    if (dom.decodeBar) {
      dom.decodeBar.textContent = '文件识别: inflight ' + qRecog
        + ' | decoded ' + state.decodedPackets
        + ' | accepted ' + state.acceptedPackets
        + ' | unique ' + uniqueBlocks + '/' + requiredBlocks
        + ' | dist ' + state.lastAvgPatternDist.toFixed(2);
    }
    if (dom.statusBar && dom.topStatus && dom.statusBar.textContent) {
      dom.topStatus.textContent = dom.statusBar.textContent;
    }
  }

  async function handleDecodedPacket(decoded, session) {
    if (session !== state.decodeSession) {
      return;
    }
    state.decodedPackets++;
    state.lastDecodeMs = Number(decoded && decoded.ms) || 0;
    state.lastAvgPatternDist = Number(decoded && decoded.avgPatternDist) || 0;
    try {
      state.decoder.processPacket(decoded.packetBytes);
      state.acceptedPackets++;
      await finalizeOutputIfReady();
    } catch (error) {
      state.errorCount++;
      log('packet reject: ' + (error && error.message ? error.message : String(error)));
    }
    updateUi(true);
  }

  function queueDeskewDecode() {
    if (!app || !app.state || !app.dom || !state.decoder) {
      return;
    }
    const benchDeskewCanvas = app.dom.dskCvs || dom.deskewedCanvas;
    if (!benchDeskewCanvas || !benchDeskewCanvas.width || !benchDeskewCanvas.height) {
      return;
    }
    const token = Number(app.state.lastDeskewTime) || 0;
    if (!token || token === state.lastDeskewTime) {
      return;
    }
    state.pendingDeskewToken = token;
    if (state.inFlight > 0) {
      return;
    }
    consumeDeskewDecodeQueue();
  }

  async function consumeDeskewDecodeQueue() {
    if (state.inFlight > 0) {
      return;
    }
    const token = Number(state.pendingDeskewToken) || 0;
    if (!token || token === state.lastDeskewTime) {
      return;
    }
    const benchDeskewCanvas = app && app.dom ? (app.dom.dskCvs || dom.deskewedCanvas) : dom.deskewedCanvas;
    if (!benchDeskewCanvas || !benchDeskewCanvas.width || !benchDeskewCanvas.height) {
      return;
    }
    state.pendingDeskewToken = 0;
    state.lastDeskewTime = token;
    const session = state.decodeSession;
    state.inFlight++;
    try {
      const decoded = await recognizer.decodeCanonicalCanvas(benchDeskewCanvas);
      await handleDecodedPacket(decoded, session);
    } catch (error) {
      if (session === state.decodeSession) {
        state.errorCount++;
        log('decode failed: ' + (error && error.message ? error.message : String(error)));
      }
    } finally {
      state.inFlight = Math.max(0, state.inFlight - 1);
      updateUi(true);
      if (state.pendingDeskewToken && state.pendingDeskewToken !== state.lastDeskewTime) {
        consumeDeskewDecodeQueue();
      }
    }
  }

  function loop() {
    try {
      queueDeskewDecode();
      updateUi(false);
    } catch (error) {
      log('loop error: ' + (error && error.message ? error.message : String(error)));
    }
    state.loopId = requestAnimationFrame(loop);
  }

  function saveOutput() {
    if (!state.outputBytes || !state.outputBytes.length) {
      setStatus('还没有完整文件');
      return;
    }
    common.downloadBytes(state.outputBytes, state.outputName || ('camera_drop_' + Date.now() + '.bin'));
  }

  async function init() {
    initCaptureSamplingInputs();
    patchCvReadyGate();
    if (dom.scanHint && dom.hintText) {
      dom.hintText.textContent = '等待摄像头预热';
    }
    if (dom.cameraMeta) {
      dom.cameraMeta.hidden = true;
    }
    const queryLayout = common.readLayoutFromQuery(global.location.search);
    if (Object.keys(queryLayout).length) {
      common.writeLayoutInputs(dom, {
        imgWidth: queryLayout.imgWidth || Number(dom.widthInput && dom.widthInput.value) || 1024,
        imgHeight: queryLayout.imgHeight || Number(dom.heightInput && dom.heightInput.value) || 1024,
        stride: queryLayout.stride || Number(dom.strideInput && dom.strideInput.value) || 9,
        margin: queryLayout.margin || Number(dom.marginInput && dom.marginInput.value) || 8,
        reservedCornerSide: queryLayout.reservedCornerSide || Number(dom.reservedInput && dom.reservedInput.value) || 6,
      });
    }

    setStatus('加载 codec');
    await codec.loadModule();
    await syncLayout(false);
    setStatus('等待 OpenCV 与摄像头就绪');
    updateUi(true);
    state.loopId = requestAnimationFrame(loop);
    if (app && app.state && app.state.cvReady) {
      beginCameraGatePrewarm();
    }
    log('scanner ready: ' + buildPageUrl(state.layout));
  }

  dom.applyLayoutBtn.addEventListener('click', function () {
    syncLayout(true).then(function () {
      setStatus('布局已应用');
      updateUi(true);
    }).catch(function (error) {
      setStatus('布局应用失败');
      log(error && error.stack ? error.stack : String(error));
    });
  });

  dom.resetBtn.addEventListener('click', function () {
    resetDecoder().then(function () {
      recognizer.dispose();
      setStatus('接收状态已重置');
      updateUi(true);
    }).catch(function (error) {
      setStatus('重置失败');
      log(error && error.stack ? error.stack : String(error));
    });
  });

  dom.saveBtn.addEventListener('click', function () {
    saveOutput();
  });

  dom.cameraInfoBtn.addEventListener('click', function () {
    toggleCameraMeta();
  });

  [dom.widthInput, dom.heightInput, dom.strideInput, dom.marginInput, dom.reservedInput].forEach(function (input) {
    if (!input) return;
    input.addEventListener('change', function () {
      if (global.history && typeof global.history.replaceState === 'function') {
        const layoutLike = common.readLayoutInputs(dom);
        global.history.replaceState(null, '', buildPageUrl(layoutLike));
      }
    });
  });

  [dom.captureFpsInput, dom.samplesPerCodeInput].forEach(function (input) {
    if (!input) return;
    input.addEventListener('change', function () {
      syncCaptureSampling(true);
      updateUi(true);
    });
  });

  init().catch(function (error) {
    setStatus('初始化失败');
    log(error && error.stack ? error.stack : String(error));
  });
})(window);
