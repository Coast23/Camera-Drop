'use strict';

(function initCameraModule(global) {
  const app = global.CameraDropApp;
  const state = app.state;
  const config = app.config;
  const dom = app.dom;
  const ui = app.ui;

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
      requested: safeJson(buildVideoConstraints()),
      tuneProfile: state.cameraTuneProfile ? safeJson({
        base: state.cameraTuneProfile.base || {},
        manual: state.cameraTuneProfile.manual || {},
        bias: state.cameraTuneProfile.bias || {},
        expected: state.cameraTuneProfile.expected || null,
      }) : null,
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

  function pickTargetIso(caps, settings) {
    if (!caps || !caps.iso) {
      return null;
    }
    const { lo, hi } = capabilityRange(caps.iso, NaN, NaN);
    const cur = settings && Number.isFinite(Number(settings.iso))
      ? Number(settings.iso)
      : NaN;
    const ceiling = Math.min(hi, Math.max(lo, Number(config.CAMERA_ISO_MAX) || 400));
    if (Number.isFinite(cur)) {
      return clampNumeric(cur, lo, ceiling);
    }
    return clampNumeric(400, lo, ceiling);
  }

  function pickTargetExposureTime(caps, settings) {
    if (!caps || !caps.exposureTime) {
      return null;
    }
    const { lo, hi, step } = capabilityRange(caps.exposureTime, NaN, NaN);
    const cur = settings && Number.isFinite(Number(settings.exposureTime))
      ? Number(settings.exposureTime)
      : NaN;
    const darkenRatio = Math.min(1, Math.max(0.4, Number(config.CAMERA_EXPOSURE_DARKEN_RATIO) || 0.82));
    if (Number.isFinite(cur)) {
      let target = clampNumeric(cur * darkenRatio, lo, hi);
      if (Number.isFinite(step) && step > 0 && target !== null) {
        target = Math.round(target / step) * step;
      }
      return clampNumeric(target, lo, hi);
    }
    return clampNumeric(lo, lo, hi);
  }

  function pickTargetExposureCompensation(caps, settings) {
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

  function buildFixedTuneProfile(caps, settings) {
    const manual = {};
    if (caps && Array.isArray(caps.whiteBalanceMode) && caps.whiteBalanceMode.includes('manual') && caps.colorTemperature) {
      const colorTemperature = pickTargetColorTemperature(caps, settings);
      if (colorTemperature !== null) {
        manual.whiteBalanceMode = 'manual';
        manual.colorTemperature = colorTemperature;
      }
    }
    if (caps && Array.isArray(caps.exposureMode) && caps.exposureMode.includes('manual') && caps.exposureTime) {
      const exposureTime = pickTargetExposureTime(caps, settings);
      if (exposureTime !== null) {
        manual.exposureMode = 'manual';
        manual.exposureTime = exposureTime;
      }
    }
    if (caps && caps.iso) {
      const iso = pickTargetIso(caps, settings);
      if (iso !== null) {
        manual.iso = iso;
      }
    }

    const bias = {};
    const exposureCompensation = pickTargetExposureCompensation(caps, settings);
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
    };
    state.cameraLastTuneAt = performance.now();
    return settings;
  }

  async function applyFixedTuneProfile(track, reason) {
    if (!track || typeof track.applyConstraints !== 'function') {
      return false;
    }

    const caps = readTrackCapabilities(track, null);
    if (!caps) {
      return false;
    }

    const settings = readTrackSettings(track, null);

    const profile = {
      base: buildBaseTuneStep(caps),
      manual: {},
      bias: {},
    };
    const fixed = buildFixedTuneProfile(caps, settings);
    profile.manual = fixed.manual;
    profile.bias = fixed.bias;

    if (!Object.keys(profile.base).length && !Object.keys(profile.manual).length && !Object.keys(profile.bias).length) {
      return false;
    }

    state.cameraTunePending = true;
    try {
      await applyTuneProfileSteps(track, profile);
      persistTuneProfile(track, profile, settings);
      updateCameraMeta(track, reason || 'fixed-tuned');
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
      persistTuneProfile(track, profile, null);
      updateCameraMeta(track, reason || 'fixed-reapplied');
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

    state.cameraTunePromise = Promise.resolve().then(async () => {
      if (reuseOnly) {
        return state.cameraTuneProfile ? reapplyTuneProfile(track, reason || 'fixed-reapplied') : false;
      }
      if (state.cameraTuneProfile) {
        return reapplyTuneProfile(track, reason || 'fixed-reapplied');
      }
      return applyFixedTuneProfile(track, reason || 'fixed-tuned');
    }).finally(() => {
      state.cameraTunePromise = null;
    });

    return state.cameraTunePromise;
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
    state.cameraTunePending = false;
    state.cameraTunePromise = null;
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
      if (!isBenchCameraStream(stream) && track) {
        try {
          if (state.cameraTuneProfile) {
            await reapplyTuneProfile(track, 'playback-reapply');
          } else {
            await applyFixedTuneProfile(track, 'playback-fixed');
          }
        } catch (error) {
          console.warn('[Camera] playback tune failed:', error && error.message ? error.message : error);
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
        if (ready && track && !isBenchCameraStream(stream)) {
          try {
            if (state.cameraTuneProfile) {
              await reapplyTuneProfile(track, 'stream-reapply');
            } else {
              await applyFixedTuneProfile(track, 'stream-fixed');
            }
          } catch (error) {
            console.warn('[Camera] stream tune failed:', error && error.message ? error.message : error);
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
          return;
        }

        state.cameraMissCount++;
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
