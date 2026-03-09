(function (global) {
  const api = global.CamDropRectCv = global.CamDropRectCv || {};
  const state = {
    ready: !!(global.cv && global.cv.Mat),
    errored: false,
  };

  const prevReady = typeof global.onCvReady === 'function' ? global.onCvReady : null;
  const prevError = typeof global.onCvError === 'function' ? global.onCvError : null;

  function markReady() {
    state.ready = true;
  }

  function markError() {
    state.errored = true;
  }

  global.onCvReady = function onCvReady() {
    markReady();
    if (prevReady && prevReady !== global.onCvReady) {
      try {
        prevReady();
      } catch (_) {}
    }
  };

  global.onCvError = function onCvError(error) {
    markError();
    if (prevError && prevError !== global.onCvError) {
      try {
        prevError(error);
      } catch (_) {}
    }
  };

  api.isReady = function isReady() {
    return !!(state.ready && global.cv && global.cv.Mat);
  };

  api.waitReady = function waitReady(timeoutMs) {
    const deadline = Date.now() + Math.max(1000, Number(timeoutMs) || 30000);
    return new Promise((resolve, reject) => {
      function tick() {
        if (api.isReady()) {
          resolve();
          return;
        }
        if (state.errored) {
          reject(new Error('opencv reported load error'));
          return;
        }
        if (Date.now() > deadline) {
          reject(new Error('opencv wait timeout'));
          return;
        }
        setTimeout(tick, 50);
      }
      tick();
    });
  };
})(window);
