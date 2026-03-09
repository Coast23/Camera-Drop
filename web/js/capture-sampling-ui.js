(function (global) {
  'use strict';

  const api = global.CamDropCaptureSamplingUi = global.CamDropCaptureSamplingUi || {};

  function firstQueryValue(params, keys) {
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      if (params.has(key)) {
        return params.get(key);
      }
    }
    return null;
  }

  function clampSamples(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) {
      return 3;
    }
    return Math.max(1, Math.min(6, Math.round(n)));
  }

  function normalizeFps(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) {
      return 0;
    }
    return Math.round(n * 1000) / 1000;
  }

  api.normalize = function normalize(sampling) {
    const src = sampling || {};
    return {
      fps: normalizeFps(src.fps),
      samplesPerCode: clampSamples(src.samplesPerCode),
    };
  };

  api.parseQuery = function parseQuery(search) {
    const params = search instanceof URLSearchParams ? search : new URLSearchParams(search || global.location.search || '');
    return api.normalize({
      fps: firstQueryValue(params, ['fps', 'code-fps', 'target-fps']),
      samplesPerCode: firstQueryValue(params, ['samples-per-code', 'captures-per-code', 'samples']),
    });
  };

  api.readInputs = function readInputs(dom) {
    return api.normalize({
      fps: dom && dom.captureFpsInput ? dom.captureFpsInput.value : 0,
      samplesPerCode: dom && dom.samplesPerCodeInput ? dom.samplesPerCodeInput.value : 3,
    });
  };

  api.writeInputs = function writeInputs(dom, sampling) {
    const normalized = api.normalize(sampling);
    if (dom && dom.captureFpsInput && normalized.fps > 0) {
      dom.captureFpsInput.value = String(normalized.fps);
    }
    if (dom && dom.samplesPerCodeInput) {
      dom.samplesPerCodeInput.value = String(normalized.samplesPerCode);
    }
    return normalized;
  };

  api.applyGlobals = function applyGlobals(sampling) {
    const normalized = api.normalize(sampling);
    global.__CAMDROP_TARGET_CODE_FPS = normalized.fps;
    global.__CAMDROP_CAPTURES_PER_CODE = normalized.samplesPerCode;
    return normalized;
  };

  api.buildExtras = function buildExtras(sampling) {
    const normalized = api.normalize(sampling);
    const extras = {
      'samples-per-code': String(normalized.samplesPerCode),
    };
    if (normalized.fps > 0) {
      extras.fps = String(normalized.fps);
    }
    return extras;
  };
})(window);
