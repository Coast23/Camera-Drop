var _cvWaitStart = Date.now();
(function waitCv() {
  if (typeof cv !== 'undefined' && cv.Mat) {
    console.log('[CV] wasm ready after', Date.now() - _cvWaitStart, 'ms');
    setTimeout(function() {
      if (typeof onCvReady === 'function') onCvReady();
    }, 0);
  } else if (Date.now() - _cvWaitStart > 60000) {
    console.error('[CV] wasm wait timeout (60s)');
    setTimeout(function() {
      if (typeof onCvError === 'function') onCvError(null);
    }, 0);
  } else {
    setTimeout(waitCv, 100);
  }
})();
