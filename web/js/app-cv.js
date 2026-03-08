'use strict';

(function initCvModule(global) {
  const app = global.CameraDropApp;
  const state = app.state;
  const dom = app.dom;
  const ui = app.ui;

  const MSG_READY_TO_START = '\u8d44\u6e90\u5df2\u5c31\u7eea\uff0c\u70b9\u51fb\u5f00\u59cb\u626b\u63cf';
  const MSG_CV_STATUS = 'OpenCV \u5c31\u7eea';
  const MSG_CV_FAIL = 'OpenCV \u52a0\u8f7d\u5931\u8d25: ';
  const MSG_CV_TIMEOUT = 'OpenCV \u52a0\u8f7d\u8d85\u65f6\uff0c\u53ef\u5148\u7ee7\u7eed';
  const MSG_SKIP_CV = '\u8df3\u8fc7 OpenCV\uff0c\u4ec5\u68c0\u6d4b';
  const STATUS_READY_TO_START = '\u7b49\u5f85\u5f00\u59cb\u626b\u63cf';

  function showStartButton(label, disabled, background) {
    dom.startBtn.textContent = label;
    dom.startBtn.style.display = 'block';
    dom.startBtn.style.background = background || '#2563eb';
    dom.startBtn.disabled = !!disabled;
  }

  app.onCvReady = function onCvReady() {
    state.cvReady = true;
    console.log('[CV] ready');
    ui.setMsg(MSG_READY_TO_START);
    ui.setProg(1.0);
    ui.setStatus(STATUS_READY_TO_START);
    showStartButton('\u5f00\u59cb\u626b\u63cf', false, '#2563eb');
  };

  app.onCvError = function onCvError(event) {
    const src = event && event.target ? event.target.src : 'opencv.js';
    console.error('[CV] load error:', src);
    ui.setMsg(MSG_CV_FAIL + src);
    ui.setStatus('OpenCV error');
    showStartButton(MSG_SKIP_CV, false, '#b45309');
  };

  setTimeout(() => {
    if (!state.cvReady) {
      console.warn('[CV] timeout after 30s');
      ui.setMsg(MSG_CV_TIMEOUT);
      ui.setStatus(MSG_CV_STATUS);
      showStartButton(MSG_SKIP_CV, false, '#b45309');
    }
  }, 30000);

  global.onCvReady = app.onCvReady;
  global.onCvError = app.onCvError;
})(window);
