'use strict';

(function initCameraModule(global) {
  const app = global.CameraDropApp;
  const state = app.state;
  const config = app.config;
  const dom = app.dom;
  const ui = app.ui;

  const tuneSceneSampleN = Math.max(12, Number(config.CAMERA_TUNE_SCENE_SAMPLE_N) || 24);
  const tuneSceneCanvas = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(tuneSceneSampleN, tuneSceneSampleN)
    : (() => {
        const cvs = document.createElement('canvas');
        cvs.width = tuneSceneSampleN;
        cvs.height = tuneSceneSampleN;
        return cvs;
      })();
  const tuneSceneCtx = tuneSceneCanvas.getContext('2d', { willReadFrequently: true });

  function buildVideoConstraints() {
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

    return video;
  }

  function getVideoTrack(stream) {
    const tracks = stream && typeof stream.getVideoTracks === 'function'
      ? stream.getVideoTracks()
      : [];
    return tracks.length ? tracks[0] : null;
  }

  function hasLiveStream(stream) {
    const track = getVideoTrack(stream);
    return !!(track && track.readyState !== 'ended');
  }

  function isBenchCameraStream(stream) {
    return !!(global.__benchCam && global.__benchCam.stream && stream === global.__benchCam.stream);
  }

  function safeJson(value) {
    if (value == null) {
      return value;
    }
    if (Array.isArray(value)) {
      return value.map((item) => safeJson(item));
    }
    if (typeof value === 'object') {
      const out = {};
      for (const key of Object.keys(value)) {
        const next = value[key];
        if (typeof next === 'function') {
          continue;
        }
        out[key] = safeJson(next);
      }
      return out;
    }
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value : String(value);
    }
    return value;
  }

  function updateCameraMeta(track, note) {
    if (!dom.cameraMeta) {
      return;
    }

    let capabilities = null;
    let settings = null;
    let constraints = null;
    try {
      capabilities = track && typeof track.getCapabilities === 'function'
        ? safeJson(track.getCapabilities())
        : '(unavailable)';
    } catch (error) {
      capabilities = '(error: ' + (error && error.message ? error.message : String(error)) + ')';
    }
    try {
      settings = track && typeof track.getSettings === 'function'
        ? safeJson(track.getSettings())
        : '(unavailable)';
    } catch (error) {
      settings = '(error: ' + (error && error.message ? error.message : String(error)) + ')';
    }
    try {
      constraints = track && typeof track.getConstraints === 'function'
        ? safeJson(track.getConstraints())
        : '(unavailable)';
    } catch (error) {
      constraints = '(error: ' + (error && error.message ? error.message : String(error)) + ')';
    }

    const payload = {
      note: note || '',
      ready: !!state.cameraReady,
      retuneWanted: !!state.cameraRetuneWanted,
      scene: state.cameraSceneStats ? safeJson(state.cameraSceneStats) : null,
      requested: safeJson(buildVideoConstraints()),
      track: track ? {
        label: track.label || '',
        readyState: track.readyState || '',
        enabled: !!track.enabled,
        muted: !!track.muted,
      } : '(no track)',
      constraints,
      capabilities,
      settings,
    };
    dom.cameraMeta.textContent = JSON.stringify(payload, null, 2);
  }

  function safePlayVideo() {
    if (!dom.video) {
      return;
    }
    try {
      const ret = dom.video.play();
      if (ret && typeof ret.catch === 'function') {
        ret.catch(() => {});
      }
    } catch (_) {}
  }

  function videoHasFrame() {
    return !!(dom.video
      && dom.video.readyState >= 2
      && dom.video.videoWidth > 0
      && dom.video.videoHeight > 0);
  }

  function markVideoFrameProgress() {
    const t = Number(dom.video && dom.video.currentTime);
    if (Number.isFinite(t)) {
      state.lastVideoFrameTime = t;
    }
    state.lastVideoFrameTickAt = performance.now();
    state.cameraReady = true;
  }

  function waitForVideoReady(timeoutMs) {
    return new Promise((resolve) => {
      if (videoHasFrame()) {
        markVideoFrameProgress();
        resolve(true);
        return;
      }

      const video = dom.video;
      const timeout = Math.max(200, Number(timeoutMs) || 0);
      let done = false;
      let timer = 0;
      let poll = 0;
      let rvfcId = 0;

      const finish = (ok) => {
        if (done) {
          return;
        }
        done = true;
        if (timer) clearTimeout(timer);
        if (poll) clearInterval(poll);
        if (video && rvfcId && typeof video.cancelVideoFrameCallback === 'function') {
          try { video.cancelVideoFrameCallback(rvfcId); } catch (_) {}
        }
        if (video) {
          video.removeEventListener('loadeddata', onMaybeReady);
          video.removeEventListener('canplay', onMaybeReady);
          video.removeEventListener('playing', onMaybeReady);
        }
        if (ok) {
          markVideoFrameProgress();
        }
        resolve(ok);
      };

      const onMaybeReady = () => {
        if (videoHasFrame()) {
          finish(true);
        }
      };

      if (video) {
        video.addEventListener('loadeddata', onMaybeReady);
        video.addEventListener('canplay', onMaybeReady);
        video.addEventListener('playing', onMaybeReady);
        if (typeof video.requestVideoFrameCallback === 'function') {
          try {
            rvfcId = video.requestVideoFrameCallback(() => finish(videoHasFrame()));
          } catch (_) {}
        }
      }

      poll = setInterval(() => {
        safePlayVideo();
        if (videoHasFrame()) {
          finish(true);
        }
      }, 60);
      timer = setTimeout(() => finish(videoHasFrame()), timeout);
    });
  }

  function clampNumeric(value, lo, hi) {
    if (!Number.isFinite(value)) {
      return null;
    }
    if (Number.isFinite(lo) && value < lo) {
      value = lo;
    }
    if (Number.isFinite(hi) && value > hi) {
      value = hi;
    }
    return value;
  }

  function sampleTuneSceneStats(source) {
    const srcW = source ? (source.videoWidth || source.width || 0) : 0;
    const srcH = source ? (source.videoHeight || source.height || 0) : 0;
    if (!srcW || !srcH || !tuneSceneCtx) {
      return null;
    }

    const marginRatio = Math.max(0, Math.min(0.35, Number(config.CAMERA_TUNE_SCENE_MARGIN_RATIO) || 0.18));
    const marginX = Math.round(srcW * marginRatio);
    const marginY = Math.round(srcH * marginRatio);
    const sampleX = Math.max(0, Math.min(srcW - 1, marginX));
    const sampleY = Math.max(0, Math.min(srcH - 1, marginY));
    const sampleW = Math.max(1, srcW - 2 * marginX);
    const sampleH = Math.max(1, srcH - 2 * marginY);
    const N = tuneSceneSampleN;

    tuneSceneCtx.drawImage(source, sampleX, sampleY, sampleW, sampleH, 0, 0, N, N);
    const data = tuneSceneCtx.getImageData(0, 0, N, N).data;

    let sum = 0;
    let min = 255;
    let max = 0;
    let gradSum = 0;
    let gradCount = 0;
    const gray = new Uint8Array(N * N);

    for (let i = 0; i < gray.length; i++) {
      const g = (data[i * 4] * 77 + data[i * 4 + 1] * 150 + data[i * 4 + 2] * 29) >> 8;
      gray[i] = g;
      sum += g;
      if (g < min) min = g;
      if (g > max) max = g;
    }

    for (let y = 1; y < N - 1; y++) {
      const row = y * N;
      for (let x = 1; x < N - 1; x++) {
        const idx = row + x;
        gradSum += Math.abs(gray[idx + 1] - gray[idx - 1]);
        gradSum += Math.abs(gray[idx + N] - gray[idx - N]);
        gradCount += 2;
      }
    }

    return {
      mean: sum / gray.length,
      range: max - min,
      min,
      max,
      blur: gradCount ? (gradSum / gradCount) : 0,
      width: srcW,
      height: srcH,
    };
  }

  function isTuneSceneUsable(stats) {
    if (!stats) {
      return false;
    }
    const minLuma = Math.max(0, Number(config.CAMERA_TUNE_SCENE_MIN_LUMA) || 22);
    const maxLuma = Math.min(255, Number(config.CAMERA_TUNE_SCENE_MAX_LUMA) || 242);
    const minRange = Math.max(8, Number(config.CAMERA_TUNE_SCENE_MIN_RANGE) || 26);
    const minBlur = Math.max(1, Number(config.CAMERA_TUNE_SCENE_MIN_BLUR) || 4);
    return stats.mean >= minLuma
      && stats.mean <= maxLuma
      && stats.range >= minRange
      && stats.blur >= minBlur;
  }

  function hasRecentCodeSeen(now) {
    const ttlMs = Math.max(300, Number(config.CAMERA_TUNE_RECENT_CODE_TTL_MS) || 1400);
    if (state.lastDeskewTime && (now - state.lastDeskewTime) <= ttlMs) {
      return true;
    }
    if (state.lastCoarseGateTime && (now - state.lastCoarseGateTime) <= ttlMs) {
      return true;
    }
    return false;
  }

  function sceneStatsChanged(current, baseline) {
    if (!current || !baseline) {
      return false;
    }
    if (Math.abs((current.mean || 0) - (baseline.mean || 0)) >= 16) {
      return true;
    }
    if (Math.abs((current.range || 0) - (baseline.range || 0)) >= 18) {
      return true;
    }
    const baseBlur = Math.max(1, Number(baseline.blur) || 0);
    const curBlur = Math.max(0, Number(current.blur) || 0);
    const blurRatio = Math.abs(curBlur - baseBlur) / baseBlur;
    return blurRatio >= 0.35;
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

  function pickTargetColorTemperature(caps, settings) {
    if (!caps || !caps.colorTemperature) {
      return null;
    }
    const { lo, hi, step } = capabilityRange(caps.colorTemperature, NaN, NaN);
    const preferred = clampNumeric(Number(config.CAMERA_WB_TARGET_K) || 6500, lo, hi);
    const current = settings && Number.isFinite(Number(settings.colorTemperature))
      ? Number(settings.colorTemperature)
      : NaN;
    const edgeGuardSteps = Math.max(1, Number(config.CAMERA_WB_EDGE_GUARD_STEPS) || 2);
    const margin = Number.isFinite(step) && step > 0
      ? step * edgeGuardSteps
      : Math.max(100, (hi - lo) * 0.08);
    if (Number.isFinite(current) && current > lo + margin && current < hi - margin) {
      return current;
    }
    return preferred;
  }

  function computeSceneExposureRatio(sceneStats) {
    if (!sceneStats || !Number.isFinite(Number(sceneStats.mean))) {
      return null;
    }
    const mean = Math.max(1, Number(sceneStats.mean));
    const target = Math.max(24, Math.min(220, Number(config.CAMERA_TUNE_TARGET_LUMA) || 118));
    const tolerance = Math.max(0, Number(config.CAMERA_TUNE_TARGET_LUMA_TOLERANCE) || 10);
    if (Math.abs(mean - target) <= tolerance) {
      return 1;
    }
    const minRatio = Math.max(0.1, Math.min(1, Number(config.CAMERA_TUNE_EXPOSURE_MIN_RATIO) || 0.28));
    const maxRatio = Math.max(1, Number(config.CAMERA_TUNE_EXPOSURE_MAX_RATIO) || 1.18);
    return clampNumeric(target / mean, minRatio, maxRatio);
  }

  function pickTargetIso(caps, settings, sceneStats) {
    if (!caps || !caps.iso) {
      return null;
    }
    const { lo, hi } = capabilityRange(caps.iso, NaN, NaN);
    const cur = settings && Number.isFinite(Number(settings.iso))
      ? Number(settings.iso)
      : NaN;
    const ceiling = Math.min(hi, Math.max(lo, Number(config.CAMERA_ISO_MAX) || 400));
    const sceneRatio = computeSceneExposureRatio(sceneStats);
    if (Number.isFinite(cur) && Number.isFinite(sceneRatio)) {
      const minRatio = Math.max(0.1, Math.min(1, Number(config.CAMERA_TUNE_ISO_MIN_RATIO) || 0.45));
      const maxRatio = Math.max(1, Number(config.CAMERA_TUNE_ISO_MAX_RATIO) || 1.12);
      const isoRatio = clampNumeric(Math.sqrt(sceneRatio), minRatio, maxRatio);
      return clampNumeric(cur * isoRatio, lo, ceiling);
    }
    if (Number.isFinite(cur)) {
      return clampNumeric(cur, lo, ceiling);
    }
    return clampNumeric(400, lo, ceiling);
  }

  function pickTargetExposureTime(caps, settings, sceneStats) {
    if (!caps || !caps.exposureTime) {
      return null;
    }
    const { lo, hi, step } = capabilityRange(caps.exposureTime, NaN, NaN);
    const cur = settings && Number.isFinite(Number(settings.exposureTime))
      ? Number(settings.exposureTime)
      : NaN;
    const sceneRatio = computeSceneExposureRatio(sceneStats);
    if (Number.isFinite(cur) && Number.isFinite(sceneRatio)) {
      let target = clampNumeric(cur * sceneRatio, lo, hi);
      if (Number.isFinite(step) && step > 0 && target !== null) {
        target = Math.round(target / step) * step;
      }
      return clampNumeric(target, lo, hi);
    }
    const darkenRatio = Math.min(1, Math.max(0.4, Number(config.CAMERA_EXPOSURE_DARKEN_RATIO) || 0.82));
    if (Number.isFinite(cur)) {
      return clampNumeric(cur * darkenRatio, lo, hi);
    }
    return clampNumeric(lo, lo, hi);
  }

  function pickTargetExposureCompensation(caps, settings, sceneStats) {
    if (!caps || !caps.exposureCompensation) {
      return null;
    }
    const { lo, hi, step } = capabilityRange(caps.exposureCompensation, NaN, NaN);
    const cur = settings && Number.isFinite(Number(settings.exposureCompensation))
      ? Number(settings.exposureCompensation)
      : 0;
    const targetBase = Number.isFinite(Number(config.CAMERA_EXPOSURE_COMP_TARGET))
      ? Number(config.CAMERA_EXPOSURE_COMP_TARGET)
      : -1.3333334;
    let target = Math.min(cur, targetBase);
    if (sceneStats && Number.isFinite(Number(sceneStats.mean))) {
      const mean = Number(sceneStats.mean);
      const targetLuma = Math.max(24, Math.min(220, Number(config.CAMERA_TUNE_TARGET_LUMA) || 118));
      if (mean > targetLuma) {
        const extraStops = Math.min(2.0, (mean - targetLuma) / 36);
        target = Math.min(target, targetBase - extraStops);
      }
    }
    if (!Number.isFinite(target)) {
      target = targetBase;
    }
    if (Number.isFinite(step) && step > 0) {
      target = Math.round(target / step) * step;
    }
    return clampNumeric(target, lo, hi);
  }

  async function applyTrackAdvancedStep(track, step) {
    if (!step || !Object.keys(step).length) {
      return false;
    }
    try {
      await track.applyConstraints({ advanced: [step] });
      return true;
    } catch (error) {
      console.warn('[Camera] tuneTrack step skipped:', step, error && error.message ? error.message : error);
      return false;
    }
  }

  function buildBaseTuneStep(caps) {
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
    return base;
  }

  function buildLearnedTuneProfile(caps, settings, sceneStats) {
    const manual = {};
    if (caps && Array.isArray(caps.whiteBalanceMode) && caps.whiteBalanceMode.includes('manual') && caps.colorTemperature) {
      const colorTemperature = pickTargetColorTemperature(caps, settings);
      if (colorTemperature !== null) {
        manual.whiteBalanceMode = 'manual';
        manual.colorTemperature = colorTemperature;
      }
    }
    if (caps && Array.isArray(caps.exposureMode) && caps.exposureMode.includes('manual') && caps.exposureTime) {
      const exposureTime = pickTargetExposureTime(caps, settings, sceneStats);
      if (exposureTime !== null) {
        manual.exposureMode = 'manual';
        manual.exposureTime = exposureTime;
      }
    }
    if (caps && caps.iso) {
      const iso = pickTargetIso(caps, settings, sceneStats);
      if (iso !== null) {
        manual.iso = iso;
      }
    }

    const bias = {};
    const exposureCompensation = pickTargetExposureCompensation(caps, settings, sceneStats);
    if (exposureCompensation !== null) {
      bias.exposureCompensation = exposureCompensation;
    }

    return { manual, bias };
  }

  function readTrackCapabilities(track, fallback) {
    try {
      return track && typeof track.getCapabilities === 'function' ? track.getCapabilities() : (fallback || null);
    } catch (_) {
      return fallback || null;
    }
  }

  function readTrackSettings(track, fallback) {
    try {
      return track && typeof track.getSettings === 'function' ? track.getSettings() : (fallback || null);
    } catch (_) {
      return fallback || null;
    }
  }

  function settingsNumber(settings, key) {
    return settings && Number.isFinite(Number(settings[key])) ? Number(settings[key]) : null;
  }

  function snapshotTuneExpectation(settings) {
    return {
      whiteBalanceMode: settings && typeof settings.whiteBalanceMode === 'string' ? settings.whiteBalanceMode : null,
      colorTemperature: settingsNumber(settings, 'colorTemperature'),
      exposureMode: settings && typeof settings.exposureMode === 'string' ? settings.exposureMode : null,
      exposureTime: settingsNumber(settings, 'exposureTime'),
      iso: settingsNumber(settings, 'iso'),
      exposureCompensation: settingsNumber(settings, 'exposureCompensation'),
      focusMode: settings && typeof settings.focusMode === 'string' ? settings.focusMode : null,
      zoom: settingsNumber(settings, 'zoom'),
    };
  }

  async function applyTuneProfileSteps(track, profile) {
    if (!profile) {
      return;
    }
    if (profile.base && Object.keys(profile.base).length) {
      await applyTrackAdvancedStep(track, profile.base);
    }
    if (profile.manual && Object.keys(profile.manual).length) {
      await applyTrackAdvancedStep(track, profile.manual);
    }
    if (profile.bias && Object.keys(profile.bias).length) {
      await applyTrackAdvancedStep(track, profile.bias);
    }
  }

  function persistTuneProfile(track, profile, fallbackSettings, extra) {
    const settings = readTrackSettings(track, fallbackSettings);
    state.cameraTuneProfile = {
      base: profile && profile.base ? { ...profile.base } : {},
      manual: profile && profile.manual ? { ...profile.manual } : {},
      bias: profile && profile.bias ? { ...profile.bias } : {},
      expected: snapshotTuneExpectation(settings),
      hadCode: !!(extra && extra.hadCode),
      scene: extra && extra.scene ? { ...extra.scene } : null,
    };
    state.cameraLastTuneAt = performance.now();
    state.cameraLastTuneCheckAt = state.cameraLastTuneAt;
    return settings;
  }

  function tuneProfileLooksDrifted(track, profile) {
    if (!profile || !profile.expected) {
      return false;
    }
    const settings = readTrackSettings(track, null);
    if (!settings) {
      return false;
    }

    const expected = profile.expected;
    if (expected.focusMode && settings.focusMode && settings.focusMode !== expected.focusMode) {
      return true;
    }
    if (expected.whiteBalanceMode && settings.whiteBalanceMode && settings.whiteBalanceMode !== expected.whiteBalanceMode) {
      return true;
    }
    if (expected.exposureMode && settings.exposureMode && settings.exposureMode !== expected.exposureMode) {
      return true;
    }

    const wbDrift = Math.max(50, Number(config.CAMERA_TUNE_WB_DRIFT_K) || 250);
    const expDriftRatio = Math.max(0.1, Number(config.CAMERA_TUNE_EXPOSURE_DRIFT_RATIO) || 0.35);
    const isoDrift = Math.max(10, Number(config.CAMERA_TUNE_ISO_DRIFT) || 80);

    const currentColorTemperature = settingsNumber(settings, 'colorTemperature');
    if (expected.colorTemperature !== null && currentColorTemperature !== null
        && Math.abs(currentColorTemperature - expected.colorTemperature) > wbDrift) {
      return true;
    }

    const currentExposureTime = settingsNumber(settings, 'exposureTime');
    if (expected.exposureTime !== null && currentExposureTime !== null && expected.exposureTime > 0) {
      const diffRatio = Math.abs(currentExposureTime - expected.exposureTime) / Math.max(expected.exposureTime, 1e-3);
      if (diffRatio > expDriftRatio) {
        return true;
      }
    }

    const currentIso = settingsNumber(settings, 'iso');
    if (expected.iso !== null && currentIso !== null && Math.abs(currentIso - expected.iso) > isoDrift) {
      return true;
    }

    const currentExposureCompensation = settingsNumber(settings, 'exposureCompensation');
    if (expected.exposureCompensation !== null && currentExposureCompensation !== null
        && Math.abs(currentExposureCompensation - expected.exposureCompensation) > 0.26) {
      return true;
    }

    return false;
  }

  async function learnTuneProfile(track, reason) {
    if (!track || typeof track.applyConstraints !== 'function') {
      return false;
    }

    const caps = readTrackCapabilities(track, null);
    if (!caps) {
      return false;
    }

    const preStats = sampleTuneSceneStats(dom.video);
    state.cameraSceneStats = preStats;
    if (!isTuneSceneUsable(preStats)) {
      state.cameraRetuneWanted = true;
      updateCameraMeta(track, reason || 'tune-scene-invalid');
      return false;
    }

    const profile = {
      base: buildBaseTuneStep(caps),
      manual: {},
      bias: {},
    };

    state.cameraTunePending = true;
    try {
      if (profile.base && Object.keys(profile.base).length) {
        await applyTrackAdvancedStep(track, profile.base);
      }

      // Let continuous AE/AWB settle briefly before locking the learned values.
      const settleMs = String(reason || '').includes('code-visible')
        ? Math.max(60, Number(config.CAMERA_CODE_SETTLE_MS) || 140)
        : Math.max(200, Number(config.CAMERA_SETTLE_MS) || 500);
      await new Promise((resolve) => global.setTimeout(resolve, settleMs));
      const sceneStats = sampleTuneSceneStats(dom.video) || preStats;
      state.cameraSceneStats = sceneStats;
      if (!isTuneSceneUsable(sceneStats)) {
        state.cameraRetuneWanted = true;
        updateCameraMeta(track, reason || 'tune-scene-invalid');
        return false;
      }
      const settledSettings = readTrackSettings(track, null);
      const learned = buildLearnedTuneProfile(caps, settledSettings, sceneStats);
      profile.manual = learned.manual;
      profile.bias = learned.bias;
      await applyTuneProfileSteps(track, profile);
      persistTuneProfile(track, profile, settledSettings, {
        hadCode: hasRecentCodeSeen(performance.now()),
        scene: sceneStats,
      });
      state.cameraRetuneWanted = false;
      updateCameraMeta(track, reason || 'tuned-learned');
      return true;
    } finally {
      state.cameraTunePending = false;
    }
  }

  async function applyBootstrapTuneProfile(track, reason) {
    if (!track || typeof track.applyConstraints !== 'function') {
      return false;
    }

    const caps = readTrackCapabilities(track, null);
    if (!caps) {
      return false;
    }

    const settings = readTrackSettings(track, null);
    const sceneStats = sampleTuneSceneStats(dom.video);
    state.cameraSceneStats = sceneStats;

    const profile = {
      base: buildBaseTuneStep(caps),
      manual: {},
      bias: {},
    };
    const learned = buildLearnedTuneProfile(caps, settings, sceneStats);
    profile.manual = learned.manual;
    profile.bias = learned.bias;

    if (!Object.keys(profile.base).length && !Object.keys(profile.manual).length && !Object.keys(profile.bias).length) {
      return false;
    }

    state.cameraTunePending = true;
    try {
      await applyTuneProfileSteps(track, profile);
      persistTuneProfile(track, profile, null, {
        hadCode: false,
        scene: sceneStats || null,
      });
      state.cameraRetuneWanted = true;
      updateCameraMeta(track, reason || 'bootstrap-tuned');
      return true;
    } finally {
      state.cameraTunePending = false;
    }
  }

  async function reapplyTuneProfile(track, reason) {
    if (!track || typeof track.applyConstraints !== 'function' || !state.cameraTuneProfile) {
      return false;
    }

    const caps = readTrackCapabilities(track, null);
    const profile = {
      base: (state.cameraTuneProfile.base && Object.keys(state.cameraTuneProfile.base).length)
        ? state.cameraTuneProfile.base
        : buildBaseTuneStep(caps),
      manual: state.cameraTuneProfile.manual || {},
      bias: state.cameraTuneProfile.bias || {},
    };

    state.cameraTunePending = true;
    try {
      await applyTuneProfileSteps(track, profile);
      const sceneStats = sampleTuneSceneStats(dom.video);
      state.cameraSceneStats = sceneStats;
      persistTuneProfile(track, profile, null, {
        hadCode: state.cameraTuneProfile && state.cameraTuneProfile.hadCode,
        scene: sceneStats || (state.cameraTuneProfile ? state.cameraTuneProfile.scene : null),
      });
      updateCameraMeta(track, reason || 'tuned-reapplied');
      return true;
    } finally {
      state.cameraTunePending = false;
    }
  }

  app.ensureCameraTunedForScan = async function ensureCameraTunedForScan(reason, options) {
    if (state.cameraTunePromise) {
      return state.cameraTunePromise;
    }

    const stream = state.cameraStream || dom.video.srcObject;
    if (!stream || isBenchCameraStream(stream)) {
      return false;
    }

    const track = getVideoTrack(stream);
    if (!track || track.readyState === 'ended') {
      return false;
    }

    const reuseOnly = !!(options && options.reuseOnly);
    const forceLearn = !!(options && options.forceLearn);

    state.cameraTunePromise = Promise.resolve().then(async () => {
      if (forceLearn) {
        return learnTuneProfile(track, reason || 'tuned-learned');
      }
      if (reuseOnly) {
        return state.cameraTuneProfile ? reapplyTuneProfile(track, reason || 'tuned-reapplied') : false;
      }
      if (state.cameraTuneProfile) {
        return reapplyTuneProfile(track, reason || 'tuned-reapplied');
      }
      return learnTuneProfile(track, reason || 'tuned-learned');
    }).finally(() => {
      state.cameraTunePromise = null;
    });

    return state.cameraTunePromise;
  };

  app.maybeRetuneCameraFromCodeScene = async function maybeRetuneCameraFromCodeScene(reason) {
    if (!state.scanning || state.cameraTunePending || state.cameraTunePromise) {
      return false;
    }

    const stream = state.cameraStream || dom.video.srcObject;
    if (!stream || isBenchCameraStream(stream)) {
      return false;
    }

    const track = getVideoTrack(stream);
    if (!track || track.readyState === 'ended') {
      return false;
    }

    const now = performance.now();
    if (!hasRecentCodeSeen(now)) {
      return false;
    }

    const sceneStats = sampleTuneSceneStats(dom.video);
    state.cameraSceneStats = sceneStats;
    if (!isTuneSceneUsable(sceneStats)) {
      state.cameraRetuneWanted = true;
      state.cameraLastCodeRetuneAttemptAt = now;
      updateCameraMeta(track, reason || 'code-scene-invalid');
      return false;
    }

    const profile = state.cameraTuneProfile;
    const needInitialLearn = !profile;
    const needCodeLearn = !!(profile && !profile.hadCode);
    const needSceneRetune = !!state.cameraRetuneWanted;
    const needDriftRetune = !!(profile && tuneProfileLooksDrifted(track, profile));
    const needShiftRetune = !!(profile && profile.scene && sceneStats && sceneStatsChanged(sceneStats, profile.scene));

    const cooldownMs = Math.max(
      500,
      Number(config.CAMERA_CODE_RETUNE_COOLDOWN_MS)
      || Number(config.CAMERA_TUNE_REAPPLY_COOLDOWN_MS)
      || 900
    );
    const lastAttemptAt = Number(state.cameraLastCodeRetuneAttemptAt) || 0;
    const lastTuneAt = Number(state.cameraLastTuneAt) || 0;
    const gateFrom = (needInitialLearn || needCodeLearn)
      ? lastAttemptAt
      : Math.max(lastTuneAt, lastAttemptAt);
    if ((now - gateFrom) < cooldownMs) {
      return false;
    }

    if (!(needInitialLearn || needCodeLearn || needSceneRetune || needDriftRetune || needShiftRetune)) {
      return false;
    }

    state.cameraLastCodeRetuneAttemptAt = now;
    console.warn('[Camera] relearning tune from visible code scene', {
      needInitialLearn,
      needCodeLearn,
      needSceneRetune,
      needDriftRetune,
      needShiftRetune,
      sceneStats,
      reason: reason || 'code-visible-relearn',
    });
    return app.ensureCameraTunedForScan(reason || 'code-visible-relearn', { forceLearn: true });
  };

  app.noteCodeSceneVisible = function noteCodeSceneVisible(reason) {
    Promise.resolve(app.maybeRetuneCameraFromCodeScene(reason)).catch((error) => {
      console.warn('[Camera] code-scene retune failed:', error && error.message ? error.message : error);
    });
  };

  function attachStream(stream) {
    dom.video.autoplay = true;
    dom.video.playsInline = true;
    dom.video.muted = true;
    dom.video.setAttribute('autoplay', '');
    dom.video.setAttribute('playsinline', '');
    dom.video.setAttribute('muted', '');
    dom.video.srcObject = stream;
    dom.video.onloadedmetadata = () => {
      safePlayVideo();
    };
    safePlayVideo();
  }

  app.markCameraFrameProgress = markVideoFrameProgress;

  app.stopCamera = function stopCamera() {
    if (state.cameraWatchdogId) {
      clearInterval(state.cameraWatchdogId);
      state.cameraWatchdogId = 0;
    }
    state.cameraWatchdogRunning = false;
    const stream = state.cameraStream || dom.video.srcObject;
    state.cameraStream = null;
    state.cameraReady = false;
    updateCameraMeta(null, 'stopped');
    if (stream && typeof stream.getTracks === 'function') {
      const tracks = stream.getTracks();
      for (let i = 0; i < tracks.length; i++) {
        try { tracks[i].stop(); } catch (_) {}
      }
    }
    try {
      dom.video.pause();
    } catch (_) {}
    dom.video.srcObject = null;
  };

  app.ensureCameraPlayback = async function ensureCameraPlayback(timeoutMs) {
    const stream = state.cameraStream || dom.video.srcObject;
    if (!stream || !hasLiveStream(stream)) {
      state.cameraReady = false;
      state.cameraRetuneWanted = true;
      updateCameraMeta(null, 'no-live-stream');
      return false;
    }
    if (dom.video.srcObject !== stream) {
      dom.video.srcObject = stream;
    }
    safePlayVideo();
    const ok = await waitForVideoReady(timeoutMs || config.CAMERA_RECOVER_RETRY_MS);
    state.cameraReady = ok;
    const track = getVideoTrack(stream);
    if (ok) {
      state.cameraMissCount = 0;
      if (!isBenchCameraStream(stream) && !state.cameraTuneProfile && track) {
        try {
          await applyBootstrapTuneProfile(track, 'playback-bootstrap');
        } catch (error) {
          console.warn('[Camera] playback bootstrap tune failed:', error && error.message ? error.message : error);
        }
      }
      if (state.scanning && !isBenchCameraStream(stream)) {
        state.cameraRetuneWanted = true;
        if (state.cameraTuneProfile) {
          await app.ensureCameraTunedForScan('playback-reapply', { reuseOnly: true });
        }
      }
    }
    updateCameraMeta(track, ok ? 'playback-ok' : 'playback-timeout');
    return ok;
  };

  app.startCamera = async function startCamera(forceRestart) {
    if (state.cameraStartPromise) {
      return state.cameraStartPromise;
    }

    state.cameraStartPromise = (async () => {
      state.cameraReady = false;
      try {
        if (forceRestart) {
          app.stopCamera();
        } else if (state.cameraStream && hasLiveStream(state.cameraStream)) {
          attachStream(state.cameraStream);
          const reusedOk = await app.ensureCameraPlayback(config.CAMERA_RECOVER_RETRY_MS);
          if (!reusedOk) {
            console.warn('[Camera] existing stream had no visible frame yet');
          }
          if (typeof app.armCameraWatchdog === 'function') {
            app.armCameraWatchdog();
          }
          return reusedOk;
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: buildVideoConstraints(),
          audio: false,
        });
        state.cameraStream = stream;
        attachStream(stream);

        const track = getVideoTrack(stream);
        updateCameraMeta(track, 'stream-opened');

        const ready = await waitForVideoReady(config.CAMERA_READY_TIMEOUT_MS);
        state.cameraReady = ready;
        if (ready && track && !isBenchCameraStream(stream) && !state.cameraTuneProfile) {
          try {
            await applyBootstrapTuneProfile(track, 'stream-bootstrap');
          } catch (error) {
            console.warn('[Camera] bootstrap tune failed:', error && error.message ? error.message : error);
          }
        }
        if (ready && state.scanning && track && !isBenchCameraStream(stream)) {
          state.cameraRetuneWanted = true;
          if (state.cameraTuneProfile) {
            await app.ensureCameraTunedForScan('stream-reapply', { reuseOnly: true });
          }
        }
        updateCameraMeta(track, ready ? 'first-frame-ok' : 'first-frame-timeout');
        if (!ready) {
          console.warn('[Camera] first frame timeout, watchdog will retry');
          safePlayVideo();
        }

        if (track && !isBenchCameraStream(stream) && typeof track.getSettings === 'function') {
          try {
            console.log('[Camera] settings', track.getSettings());
          } catch (_) {}
        }
        if (typeof app.armCameraWatchdog === 'function') {
          app.armCameraWatchdog();
        }
        return ready;
      } catch (error) {
        ui.setMsg('Camera error: ' + error.message);
        ui.setStatus('Camera error');
        state.cameraReady = false;
        updateCameraMeta(null, 'camera-error');
        return false;
      } finally {
        state.cameraStartPromise = null;
      }
    })();

    return state.cameraStartPromise;
  };

  app.armCameraWatchdog = function armCameraWatchdog() {
    if (state.cameraWatchdogId) {
      return;
    }
    const intervalMs = Math.max(500, Number(config.CAMERA_WATCHDOG_MS) || 1500);
    state.cameraWatchdogId = setInterval(() => {
      if (state.cameraWatchdogRunning) {
        return;
      }
      state.cameraWatchdogRunning = true;
      Promise.resolve().then(async () => {
        if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
          return;
        }
        const stream = state.cameraStream || dom.video.srcObject;
        if (!stream) {
          return;
        }
        const track = getVideoTrack(stream);
        if (!track || track.readyState === 'ended') {
          state.cameraRetuneWanted = true;
          await app.startCamera(true);
          return;
        }
        if (isBenchCameraStream(stream)) {
          safePlayVideo();
          return;
        }
        const now = performance.now();
        const currentTime = Number(dom.video && dom.video.currentTime);
        const timeAdvanced = Number.isFinite(currentTime)
          && currentTime > state.lastVideoFrameTime + 1e-4;
        const frameFresh = state.lastVideoFrameTickAt
          && (now - state.lastVideoFrameTickAt) <= (intervalMs * 1.5);
        const healthy = videoHasFrame() && !dom.video.paused && (timeAdvanced || frameFresh);

        if (healthy) {
          markVideoFrameProgress();
          state.cameraMissCount = 0;
          const recheckMs = Math.max(400, Number(config.CAMERA_TUNE_RECHECK_MS) || 1200);
          const cooldownMs = Math.max(300, Number(config.CAMERA_TUNE_REAPPLY_COOLDOWN_MS) || 900);
          if (state.scanning && !state.cameraTunePending && !state.cameraTunePromise
              && (now - state.cameraLastTuneCheckAt) >= recheckMs) {
            state.cameraLastTuneCheckAt = now;
            const sceneStats = sampleTuneSceneStats(dom.video);
            state.cameraSceneStats = sceneStats;
            const sceneUsable = isTuneSceneUsable(sceneStats);
            const codeFresh = hasRecentCodeSeen(now);
            const profile = state.cameraTuneProfile;
            const drifted = !!(profile && tuneProfileLooksDrifted(track, profile));
            const sceneShifted = !!(profile && profile.scene && sceneStats && sceneStatsChanged(sceneStats, profile.scene));

            if (!sceneUsable) {
              state.cameraRetuneWanted = true;
              return;
            }

            const canRetuneNow = (now - state.cameraLastTuneAt) >= cooldownMs;
            const needInitialLearn = !profile;
            const needCodeLearn = !!(profile && !profile.hadCode && codeFresh);
            const needSceneRetune = !!(codeFresh && state.cameraRetuneWanted);
            const needDriftRetune = !!(codeFresh && drifted);
            const needShiftRetune = !!(codeFresh && sceneShifted);

            if (canRetuneNow && (needInitialLearn || needCodeLearn || needSceneRetune || needDriftRetune || needShiftRetune)) {
              console.warn('[Camera] learning tune from current scene', { needInitialLearn, needCodeLearn, needSceneRetune, needDriftRetune, needShiftRetune, codeFresh, sceneStats });
              await app.ensureCameraTunedForScan('watchdog-relearn', { forceLearn: true });
            }
          }
          return;
        }

        state.cameraMissCount++;
        state.cameraRetuneWanted = true;
        safePlayVideo();
        const recovered = await app.ensureCameraPlayback(config.CAMERA_RECOVER_RETRY_MS);
        if (recovered) {
          return;
        }
        if (state.lastVideoFrameTickAt > 0 && state.cameraMissCount >= 2 && !state.cameraStartPromise) {
          console.warn('[Camera] watchdog restarting stalled stream');
          state.cameraMissCount = 0;
          await app.startCamera(true);
        }
      }).finally(() => {
        state.cameraWatchdogRunning = false;
      });
    }, intervalMs);
  };

  updateCameraMeta(null, 'idle');

  function scheduleCameraRecovery() {
    setTimeout(() => {
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
        return;
      }
      if (state.cameraStream) {
        app.ensureCameraPlayback(config.CAMERA_RECOVER_RETRY_MS).then((ok) => {
          if (!ok && !isBenchCameraStream(state.cameraStream) && !state.cameraStartPromise) {
            app.startCamera(true);
          }
        }).catch(() => {
          if (!state.cameraStartPromise) {
            app.startCamera(true);
          }
        });
        return;
      }
      if (state.scanning && !state.cameraStartPromise) {
        app.startCamera(false);
      }
    }, 80);
  }

  if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        scheduleCameraRecovery();
      }
    });
  }
  global.addEventListener('pageshow', scheduleCameraRecovery);
  global.addEventListener('focus', scheduleCameraRecovery);
})(window);
