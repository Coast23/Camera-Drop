(function (global) {
  'use strict';

  const dom = {
    refreshBtn: document.getElementById('refreshBtn'),
    startBtn: document.getElementById('startBtn'),
    tuneBtn: document.getElementById('tuneBtn'),
    snapshotBtn: document.getElementById('snapshotBtn'),
    stopBtn: document.getElementById('stopBtn'),
    video: document.getElementById('video'),
    overlayText: document.getElementById('overlayText'),
    statusText: document.getElementById('statusText'),
    topStatus: document.getElementById('topStatus'),
    deviceText: document.getElementById('deviceText'),
    frameText: document.getElementById('frameText'),
    rvfcText: document.getElementById('rvfcText'),
    sceneText: document.getElementById('sceneText'),
    summaryChips: document.getElementById('summaryChips'),
    apiTableBody: document.getElementById('apiTableBody'),
    supportedPre: document.getElementById('supportedPre'),
    requestedPre: document.getElementById('requestedPre'),
    capabilitiesPre: document.getElementById('capabilitiesPre'),
    settingsPre: document.getElementById('settingsPre'),
    logPre: document.getElementById('logPre'),
  };

  const state = {
    stream: null,
    track: null,
    rvfcToken: 0,
    rvfcMode: 'idle',
    frameCount: 0,
    lastFrameNow: 0,
    frameIntervals: [],
    sceneStats: null,
  };

  const API_ROWS = [
    { id: 'mediaDevices', label: 'navigator.mediaDevices', why: '媒体入口' },
    { id: 'getSupportedConstraints', label: 'mediaDevices.getSupportedConstraints()', why: '判断约束键是否被浏览器识别' },
    { id: 'getUserMedia', label: 'mediaDevices.getUserMedia()', why: '请求后置摄像头流' },
    { id: 'enumerateDevices', label: 'mediaDevices.enumerateDevices()', why: '列设备和后置相机标签' },
    { id: 'srcObject', label: 'HTMLVideoElement.srcObject', why: '把 MediaStream 绑到预览视频' },
    { id: 'videoPlay', label: 'HTMLVideoElement.play()', why: '驱动视频真正播放' },
    { id: 'rvfc', label: 'requestVideoFrameCallback()', why: '按真实视频帧驱动处理流水线' },
    { id: 'getCapabilities', label: 'MediaStreamTrack.getCapabilities()', why: '看曝光/白平衡/ISO/缩放能力' },
    { id: 'getSettings', label: 'MediaStreamTrack.getSettings()', why: '看当前实际生效参数' },
    { id: 'getConstraints', label: 'MediaStreamTrack.getConstraints()', why: '看当前 track 上的约束' },
    { id: 'applyConstraints', label: 'MediaStreamTrack.applyConstraints()', why: '下发一步到位参数和后续重学结果' },
    { id: 'createImageBitmap', label: 'createImageBitmap(video)', why: '抓视频帧进 worker / 精定位队列' },
    { id: 'offscreenCanvas', label: 'OffscreenCanvas', why: '相机采样、blur、deskew、识别预处理' },
    { id: 'worker', label: 'Worker', why: 'YOLO / 识别多 worker 消费' },
  ];

  function log(line) {
    const prefix = '[' + new Date().toLocaleTimeString() + '] ';
    const existing = dom.logPre.textContent === '-' ? '' : dom.logPre.textContent;
    dom.logPre.textContent = prefix + line + (existing ? '\n' + existing : '');
  }

  function setStatus(text) {
    dom.statusText.textContent = text;
    dom.topStatus.textContent = text;
  }

  function safeJson(value) {
    if (value == null) {
      return value;
    }
    if (Array.isArray(value)) {
      return value.map(safeJson);
    }
    if (typeof value === 'object') {
      const out = {};
      Object.keys(value).forEach(function (key) {
        const next = value[key];
        if (typeof next !== 'function') {
          out[key] = safeJson(next);
        }
      });
      return out;
    }
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value : String(value);
    }
    return value;
  }

  function pretty(value) {
    return JSON.stringify(safeJson(value), null, 2);
  }

  function clampNumeric(value, lo, hi) {
    if (!Number.isFinite(value)) {
      return null;
    }
    if (Number.isFinite(lo) && value < lo) value = lo;
    if (Number.isFinite(hi) && value > hi) value = hi;
    return value;
  }

  function capabilityRange(cap, fallbackMin, fallbackMax) {
    if (!cap) {
      return { lo: fallbackMin, hi: fallbackMax, step: NaN };
    }
    return {
      lo: Number.isFinite(Number(cap.min)) ? Number(cap.min) : fallbackMin,
      hi: Number.isFinite(Number(cap.max)) ? Number(cap.max) : fallbackMax,
      step: Number.isFinite(Number(cap.step)) ? Number(cap.step) : NaN,
    };
  }

  function buildRequestedConstraints() {
    const supported = navigator.mediaDevices && typeof navigator.mediaDevices.getSupportedConstraints === 'function'
      ? navigator.mediaDevices.getSupportedConstraints()
      : {};
    const isLandscape = typeof global.matchMedia === 'function'
      ? global.matchMedia('all and (orientation:landscape)').matches
      : (global.innerWidth >= global.innerHeight);
    const video = {
      facingMode: { ideal: 'environment' },
      width: { ideal: 9999 },
      height: { ideal: 9999 },
    };
    if (supported.frameRate) {
      video.frameRate = { ideal: 60, max: 60 };
    }
    if (supported.aspectRatio) {
      video.aspectRatio = isLandscape ? (16 / 9) : (9 / 16);
    }
    if (supported.resizeMode) {
      video.resizeMode = 'none';
    }
    if (supported.focusMode) {
      video.focusMode = 'continuous';
    }
    if (supported.exposureMode) {
      video.exposureMode = 'continuous';
    }
    if (supported.whiteBalanceMode) {
      video.whiteBalanceMode = 'continuous';
    }
    return { audio: false, video: video };
  }

  async function enumerateVideoDevices() {
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.enumerateDevices !== 'function') {
      return [];
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter(function (device) {
      return device.kind === 'videoinput';
    }).map(function (device, index) {
      return {
        index: index,
        label: device.label || '(未授权时通常拿不到标签)',
        deviceId: device.deviceId,
        groupId: device.groupId,
      };
    });
  }

  function sampleSceneStats() {
    const video = dom.video;
    const srcW = video && video.videoWidth;
    const srcH = video && video.videoHeight;
    if (!srcW || !srcH || typeof global.OffscreenCanvas === 'undefined') {
      return null;
    }
    const N = 24;
    const canvas = new OffscreenCanvas(N, N);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const marginRatio = 0.18;
    const marginX = Math.round(srcW * marginRatio);
    const marginY = Math.round(srcH * marginRatio);
    const sampleX = Math.max(0, Math.min(srcW - 1, marginX));
    const sampleY = Math.max(0, Math.min(srcH - 1, marginY));
    const sampleW = Math.max(1, srcW - 2 * marginX);
    const sampleH = Math.max(1, srcH - 2 * marginY);
    ctx.drawImage(video, sampleX, sampleY, sampleW, sampleH, 0, 0, N, N);
    const data = ctx.getImageData(0, 0, N, N).data;
    let sum = 0;
    let min = 255;
    let max = 0;
    for (let i = 0; i < N * N; i++) {
      const gray = (data[i * 4] * 77 + data[i * 4 + 1] * 150 + data[i * 4 + 2] * 29) >> 8;
      sum += gray;
      if (gray < min) min = gray;
      if (gray > max) max = gray;
    }
    return {
      mean: sum / (N * N),
      min: min,
      max: max,
      range: max - min,
      width: srcW,
      height: srcH,
    };
  }

  function computeSceneExposureRatio(sceneStats) {
    if (!sceneStats || !Number.isFinite(Number(sceneStats.mean))) {
      return 1;
    }
    const mean = Math.max(1, Number(sceneStats.mean));
    const target = 118;
    const tolerance = 10;
    if (Math.abs(mean - target) <= tolerance) {
      return 1;
    }
    return clampNumeric(target / mean, 0.28, 1.18);
  }

  function pickTargetColorTemperature(caps, settings) {
    if (!caps || !caps.colorTemperature) {
      return null;
    }
    const range = capabilityRange(caps.colorTemperature, NaN, NaN);
    const step = Number.isFinite(range.step) && range.step > 0 ? range.step : 50;
    const preferred = clampNumeric(6500, range.lo, range.hi);
    const current = settings && Number.isFinite(Number(settings.colorTemperature))
      ? Number(settings.colorTemperature)
      : NaN;
    const margin = step * 2;
    if (Number.isFinite(current) && current > range.lo + margin && current < range.hi - margin) {
      return current;
    }
    return preferred;
  }

  function pickTargetExposureTime(caps, settings, sceneStats) {
    if (!caps || !caps.exposureTime) {
      return null;
    }
    const range = capabilityRange(caps.exposureTime, NaN, NaN);
    const current = settings && Number.isFinite(Number(settings.exposureTime))
      ? Number(settings.exposureTime)
      : NaN;
    if (!Number.isFinite(current)) {
      return clampNumeric(range.lo, range.lo, range.hi);
    }
    const ratio = computeSceneExposureRatio(sceneStats);
    let target = clampNumeric(current * ratio, range.lo, range.hi);
    if (Number.isFinite(range.step) && range.step > 0 && target !== null) {
      target = Math.round(target / range.step) * range.step;
    }
    return clampNumeric(target, range.lo, range.hi);
  }

  function pickTargetIso(caps, settings, sceneStats) {
    if (!caps || !caps.iso) {
      return null;
    }
    const range = capabilityRange(caps.iso, NaN, NaN);
    const current = settings && Number.isFinite(Number(settings.iso))
      ? Number(settings.iso)
      : NaN;
    const ceiling = Math.min(range.hi, Math.max(range.lo, 400));
    if (!Number.isFinite(current)) {
      return clampNumeric(400, range.lo, ceiling);
    }
    const ratio = computeSceneExposureRatio(sceneStats);
    const isoRatio = clampNumeric(Math.sqrt(ratio), 0.45, 1.12);
    return clampNumeric(current * isoRatio, range.lo, ceiling);
  }

  function pickTargetExposureCompensation(caps, settings, sceneStats) {
    if (!caps || !caps.exposureCompensation) {
      return null;
    }
    const range = capabilityRange(caps.exposureCompensation, NaN, NaN);
    const current = settings && Number.isFinite(Number(settings.exposureCompensation))
      ? Number(settings.exposureCompensation)
      : 0;
    let target = Math.min(current, -1.3333334);
    if (sceneStats && Number.isFinite(Number(sceneStats.mean))) {
      const extraStops = Math.min(2.0, Math.max(0, (Number(sceneStats.mean) - 118) / 36));
      target = Math.min(target, -1.3333334 - extraStops);
    }
    if (Number.isFinite(range.step) && range.step > 0) {
      target = Math.round(target / range.step) * range.step;
    }
    return clampNumeric(target, range.lo, range.hi);
  }

  async function applyAdvancedStep(track, label, step) {
    if (!track || typeof track.applyConstraints !== 'function' || !step || !Object.keys(step).length) {
      return { label: label, ok: false, skipped: true };
    }
    try {
      await track.applyConstraints({ advanced: [step] });
      log(label + ' 已应用: ' + JSON.stringify(step));
      return { label: label, ok: true, step: step };
    } catch (error) {
      log(label + ' 失败: ' + (error && error.message ? error.message : String(error)));
      return { label: label, ok: false, step: step, error: error };
    }
  }

  function updateButtons() {
    const active = !!state.track;
    dom.startBtn.disabled = active;
    dom.tuneBtn.disabled = !active;
    dom.snapshotBtn.disabled = !active;
    dom.stopBtn.disabled = !active;
  }

  function stateClass(ok, warn) {
    if (ok) return 'state-good';
    if (warn) return 'state-warn';
    return 'state-bad';
  }

  function renderApiTable() {
    const mediaDevices = navigator.mediaDevices || null;
    const videoProto = global.HTMLVideoElement ? global.HTMLVideoElement.prototype : null;
    const track = state.track;
    const checks = {
      mediaDevices: !!mediaDevices,
      getSupportedConstraints: !!(mediaDevices && typeof mediaDevices.getSupportedConstraints === 'function'),
      getUserMedia: !!(mediaDevices && typeof mediaDevices.getUserMedia === 'function'),
      enumerateDevices: !!(mediaDevices && typeof mediaDevices.enumerateDevices === 'function'),
      srcObject: !!(dom.video && 'srcObject' in dom.video),
      videoPlay: !!(dom.video && typeof dom.video.play === 'function'),
      rvfc: !!(videoProto && typeof videoProto.requestVideoFrameCallback === 'function'),
      getCapabilities: !!(track && typeof track.getCapabilities === 'function'),
      getSettings: !!(track && typeof track.getSettings === 'function'),
      getConstraints: !!(track && typeof track.getConstraints === 'function'),
      applyConstraints: !!(track && typeof track.applyConstraints === 'function'),
      createImageBitmap: typeof global.createImageBitmap === 'function',
      offscreenCanvas: typeof global.OffscreenCanvas === 'function',
      worker: typeof global.Worker === 'function',
    };

    dom.apiTableBody.innerHTML = API_ROWS.map(function (row) {
      const ok = !!checks[row.id];
      const stateText = ok ? '可用' : (state.track ? '不可用' : '待相机确认');
      return '<tr>'
        + '<td>' + row.label + '</td>'
        + '<td class="' + stateClass(ok, !state.track && /getCapabilities|getSettings|getConstraints|applyConstraints/.test(row.id)) + '">' + stateText + '</td>'
        + '<td>' + row.why + '</td>'
        + '</tr>';
    }).join('');

    const chips = [
      { label: 'mediaDevices', ok: checks.mediaDevices },
      { label: 'getUserMedia', ok: checks.getUserMedia },
      { label: 'applyConstraints', ok: checks.applyConstraints, warn: !checks.applyConstraints && !state.track },
      { label: 'RVFC', ok: checks.rvfc },
      { label: 'OffscreenCanvas', ok: checks.offscreenCanvas },
      { label: 'Worker', ok: checks.worker },
    ];
    dom.summaryChips.innerHTML = chips.map(function (chip) {
      const cls = chip.ok ? 'good' : (chip.warn ? 'warn' : 'bad');
      const text = chip.ok ? 'ok' : (chip.warn ? 'pending' : 'no');
      return '<span class="chip ' + cls + '">' + chip.label + ': ' + text + '</span>';
    }).join('');
  }

  async function renderSnapshot() {
    const supported = navigator.mediaDevices && typeof navigator.mediaDevices.getSupportedConstraints === 'function'
      ? navigator.mediaDevices.getSupportedConstraints()
      : null;
    dom.supportedPre.textContent = pretty(supported);
    dom.requestedPre.textContent = pretty(buildRequestedConstraints());

    const devices = await enumerateVideoDevices();
    dom.deviceText.textContent = devices.length
      ? devices.map(function (device) { return device.label; }).join(' | ')
      : '没有拿到 videoinput';

    if (!state.track) {
      dom.capabilitiesPre.textContent = '-';
      dom.settingsPre.textContent = '-';
      renderApiTable();
      updateButtons();
      return;
    }

    let capabilities = null;
    let settings = null;
    let constraints = null;
    try { capabilities = state.track.getCapabilities ? state.track.getCapabilities() : null; } catch (_) {}
    try { settings = state.track.getSettings ? state.track.getSettings() : null; } catch (_) {}
    try { constraints = state.track.getConstraints ? state.track.getConstraints() : null; } catch (_) {}
    state.sceneStats = sampleSceneStats();

    dom.capabilitiesPre.textContent = pretty(capabilities);
    dom.settingsPre.textContent = pretty({
      settings: settings,
      constraints: constraints,
      scene: state.sceneStats,
    });
    dom.sceneText.textContent = state.sceneStats
      ? ('mean ' + state.sceneStats.mean.toFixed(1) + ' | range ' + state.sceneStats.range)
      : '-';
    renderApiTable();
    updateButtons();
  }

  async function waitForFirstFrame(video, timeoutMs) {
    return new Promise(function (resolve) {
      if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
        resolve(true);
        return;
      }
      let done = false;
      let rvfcId = 0;
      let timer = 0;
      let poll = 0;
      function finish(ok) {
        if (done) return;
        done = true;
        if (timer) clearTimeout(timer);
        if (poll) clearInterval(poll);
        if (rvfcId && typeof video.cancelVideoFrameCallback === 'function') {
          try { video.cancelVideoFrameCallback(rvfcId); } catch (_) {}
        }
        video.removeEventListener('loadeddata', onReady);
        video.removeEventListener('canplay', onReady);
        resolve(ok);
      }
      function onReady() {
        if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
          finish(true);
        }
      }
      video.addEventListener('loadeddata', onReady);
      video.addEventListener('canplay', onReady);
      if (typeof video.requestVideoFrameCallback === 'function') {
        try {
          rvfcId = video.requestVideoFrameCallback(function () { finish(true); });
        } catch (_) {}
      }
      poll = setInterval(onReady, 50);
      timer = setTimeout(function () { finish(false); }, timeoutMs);
    });
  }

  function stopFrameMonitor() {
    if (!state.rvfcToken) {
      return;
    }
    if (state.rvfcMode === 'rvfc' && typeof dom.video.cancelVideoFrameCallback === 'function') {
      try { dom.video.cancelVideoFrameCallback(state.rvfcToken); } catch (_) {}
    } else if (state.rvfcMode === 'raf') {
      cancelAnimationFrame(state.rvfcToken);
    }
    state.rvfcToken = 0;
    state.rvfcMode = 'idle';
  }

  function startFrameMonitor() {
    stopFrameMonitor();
    state.frameCount = 0;
    state.lastFrameNow = 0;
    state.frameIntervals = [];
    function onFrame(now) {
      if (!state.stream) {
        stopFrameMonitor();
        return;
      }
      state.frameCount += 1;
      if (state.lastFrameNow) {
        const delta = now - state.lastFrameNow;
        if (delta > 0) {
          state.frameIntervals.push(delta);
          if (state.frameIntervals.length > 90) {
            state.frameIntervals.shift();
          }
        }
      }
      state.lastFrameNow = now;
      const avg = state.frameIntervals.length
        ? (1000 / (state.frameIntervals.reduce(function (a, b) { return a + b; }, 0) / state.frameIntervals.length))
        : 0;
      dom.rvfcText.textContent = (state.rvfcMode === 'rvfc' ? 'RVFC ' : 'RAF ')
        + state.frameCount + ' frames'
        + (avg ? (' | ' + avg.toFixed(2) + ' fps') : '');
      dom.overlayText.textContent = state.rvfcMode + ' | ' + (avg ? avg.toFixed(2) + ' fps' : 'warming up');
      if (state.rvfcMode === 'rvfc' && typeof dom.video.requestVideoFrameCallback === 'function') {
        state.rvfcToken = dom.video.requestVideoFrameCallback(onFrame);
      } else {
        state.rvfcToken = requestAnimationFrame(onFrame);
      }
    }
    if (typeof dom.video.requestVideoFrameCallback === 'function') {
      state.rvfcMode = 'rvfc';
      state.rvfcToken = dom.video.requestVideoFrameCallback(onFrame);
    } else {
      state.rvfcMode = 'raf';
      state.rvfcToken = requestAnimationFrame(onFrame);
    }
  }

  async function startCamera() {
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
      setStatus('浏览器不支持 getUserMedia');
      renderApiTable();
      return;
    }
    await stopCamera();
    setStatus('请求摄像头');
    dom.overlayText.textContent = 'requesting camera';
    const constraints = buildRequestedConstraints();
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      state.stream = stream;
      state.track = stream.getVideoTracks && stream.getVideoTracks().length ? stream.getVideoTracks()[0] : null;
      dom.video.srcObject = stream;
      const playRet = dom.video.play();
      if (playRet && typeof playRet.catch === 'function') {
        await playRet.catch(function () {});
      }
      const firstFrameOk = await waitForFirstFrame(dom.video, 1800);
      dom.frameText.textContent = firstFrameOk
        ? (dom.video.videoWidth + ' x ' + dom.video.videoHeight)
        : 'timeout';
      setStatus(firstFrameOk ? '摄像头已就绪' : '摄像头已打开，但首帧超时');
      log('camera started');
      startFrameMonitor();
      await renderSnapshot();
    } catch (error) {
      state.stream = null;
      state.track = null;
      setStatus('请求摄像头失败');
      dom.overlayText.textContent = 'camera failed';
      log('getUserMedia failed: ' + (error && error.message ? error.message : String(error)));
      await renderSnapshot();
    }
  }

  async function stopCamera() {
    stopFrameMonitor();
    if (state.stream && typeof state.stream.getTracks === 'function') {
      state.stream.getTracks().forEach(function (track) {
        try { track.stop(); } catch (_) {}
      });
    }
    state.stream = null;
    state.track = null;
    try { dom.video.pause(); } catch (_) {}
    dom.video.srcObject = null;
    dom.overlayText.textContent = 'camera idle';
    dom.frameText.textContent = '-';
    dom.rvfcText.textContent = '-';
    dom.sceneText.textContent = '-';
    updateButtons();
    renderApiTable();
    await renderSnapshot();
  }

  async function applyRecommendedTune() {
    if (!state.track) {
      return;
    }
    const caps = state.track.getCapabilities ? state.track.getCapabilities() : null;
    const settings = state.track.getSettings ? state.track.getSettings() : null;
    const scene = sampleSceneStats();
    const steps = [];
    const base = {};

    if (caps && Array.isArray(caps.focusMode) && caps.focusMode.includes('continuous')) {
      base.focusMode = 'continuous';
    }
    if (caps && Array.isArray(caps.exposureMode) && caps.exposureMode.includes('continuous')) {
      base.exposureMode = 'continuous';
    }
    if (caps && Array.isArray(caps.whiteBalanceMode) && caps.whiteBalanceMode.includes('continuous')) {
      base.whiteBalanceMode = 'continuous';
    }
    if (caps && Array.isArray(caps.resizeMode) && caps.resizeMode.includes('none')) {
      base.resizeMode = 'none';
    }
    if (caps && caps.torch === true) {
      base.torch = false;
    }
    if (caps && caps.frameRate && Number.isFinite(Number(caps.frameRate.max))) {
      base.frameRate = Math.max(24, Math.min(60, Number(caps.frameRate.max)));
    }
    if (caps && caps.zoom) {
      const zoom = clampNumeric(1, Number(caps.zoom.min), Number(caps.zoom.max));
      if (zoom !== null) {
        base.zoom = zoom;
      }
    }
    if (Object.keys(base).length) {
      steps.push({ label: 'base', step: base });
    }

    const manual = {};
    if (caps && Array.isArray(caps.whiteBalanceMode) && caps.whiteBalanceMode.includes('manual') && caps.colorTemperature) {
      const wb = pickTargetColorTemperature(caps, settings);
      if (wb !== null) {
        manual.whiteBalanceMode = 'manual';
        manual.colorTemperature = wb;
      }
    }
    if (caps && Array.isArray(caps.exposureMode) && caps.exposureMode.includes('manual') && caps.exposureTime) {
      const exposureTime = pickTargetExposureTime(caps, settings, scene);
      if (exposureTime !== null) {
        manual.exposureMode = 'manual';
        manual.exposureTime = exposureTime;
      }
    }
    if (caps && caps.iso) {
      const iso = pickTargetIso(caps, settings, scene);
      if (iso !== null) {
        manual.iso = iso;
      }
    }
    if (Object.keys(manual).length) {
      steps.push({ label: 'manual', step: manual });
    }

    const bias = {};
    if (caps && caps.exposureCompensation) {
      const exposureCompensation = pickTargetExposureCompensation(caps, settings, scene);
      if (exposureCompensation !== null) {
        bias.exposureCompensation = exposureCompensation;
      }
    }
    if (Object.keys(bias).length) {
      steps.push({ label: 'bias', step: bias });
    }

    if (!steps.length) {
      log('没有可应用的推荐参数');
      return;
    }

    setStatus('应用一步到位参数');
    for (let i = 0; i < steps.length; i++) {
      await applyAdvancedStep(state.track, steps[i].label, steps[i].step);
    }
    await new Promise(function (resolve) { global.setTimeout(resolve, 120); });
    setStatus('推荐参数已应用');
    await renderSnapshot();
  }

  async function init() {
    renderApiTable();
    await renderSnapshot();
    updateButtons();

    dom.refreshBtn.addEventListener('click', function () {
      renderSnapshot().catch(function (error) {
        log('refresh failed: ' + (error && error.message ? error.message : String(error)));
      });
    });
    dom.startBtn.addEventListener('click', function () {
      startCamera();
    });
    dom.tuneBtn.addEventListener('click', function () {
      applyRecommendedTune().catch(function (error) {
        setStatus('参数应用失败');
        log('tune failed: ' + (error && error.message ? error.message : String(error)));
      });
    });
    dom.snapshotBtn.addEventListener('click', function () {
      renderSnapshot().catch(function (error) {
        log('snapshot failed: ' + (error && error.message ? error.message : String(error)));
      });
    });
    dom.stopBtn.addEventListener('click', function () {
      stopCamera().then(function () {
        setStatus('摄像头已停止');
      });
    });
    global.addEventListener('beforeunload', function () {
      stopCamera();
    });
  }

  init().catch(function (error) {
    setStatus('初始化失败');
    log(error && error.stack ? error.stack : String(error));
  });
})(window);
