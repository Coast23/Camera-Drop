(function (global) {
  'use strict';

  const samplingUi = global.CamDropCaptureSamplingUi;

  const dom = {
    captureFpsInput: document.getElementById('captureFpsInput'),
    samplesPerCodeInput: document.getElementById('samplesPerCodeInput'),
  };

  function buildPageUrl() {
    const url = new URL(global.location.href);
    const params = url.searchParams;
    ['fps', 'code-fps', 'target-fps', 'samples-per-code', 'captures-per-code', 'samples'].forEach(function (key) {
      params.delete(key);
    });
    const extras = samplingUi ? samplingUi.buildExtras(samplingUi.readInputs(dom)) : {};
    Object.keys(extras).forEach(function (key) {
      params.set(key, extras[key]);
    });
    url.search = params.toString();
    return url.toString();
  }

  function syncSampling(pushUrl) {
    if (!samplingUi) {
      return;
    }
    samplingUi.applyGlobals(samplingUi.readInputs(dom));
    if (pushUrl && global.history && typeof global.history.replaceState === 'function') {
      global.history.replaceState(null, '', buildPageUrl());
    }
  }

  function init() {
    if (!samplingUi) {
      return;
    }
    const querySampling = samplingUi.parseQuery(global.location.search);
    samplingUi.writeInputs(dom, querySampling);
    syncSampling(false);
    [dom.captureFpsInput, dom.samplesPerCodeInput].forEach(function (input) {
      if (!input) return;
      input.addEventListener('change', function () {
        syncSampling(true);
      });
    });
  }

  init();
})(window);
