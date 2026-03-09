'use strict';

(function initCvModule(global) {
  const app = global.CameraDropApp;
  const state = app.state;
  const dom = app.dom;
  const ui = app.ui;

  const MSG_READY_TO_START = '资源已就绪，点击开始扫描';
  const MSG_CV_STATUS = 'OpenCV 就绪';
  const MSG_CV_FAIL = 'OpenCV 加载失败: ';
  const MSG_CV_TIMEOUT = 'OpenCV 加载超时，可先继续';
  const MSG_SKIP_CV = '跳过 OpenCV，仅检测';
  const STATUS_READY_TO_START = '等待开始扫描';

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
    showStartButton('开始扫描', false, '#2563eb');
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
