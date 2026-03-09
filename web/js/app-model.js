'use strict';

(function initModelModule(global) {
  const app = global.CameraDropApp;
  const dom = app.dom;
  const ui = app.ui;

  app.onStart = async function onStart() {
    const originalLabel = dom.startBtn.textContent || '开始扫描';
    dom.startBtn.disabled = true;
    dom.startBtn.textContent = '正在准备...';
    ui.setMsg('正在请求摄像头...');
    ui.setProg(0.72);
    ui.setStatus('准备摄像头');

    const ok = await app.startCamera(false);
    if (!ok) {
      ui.setMsg('请先允许摄像头权限，并等待首帧显示后再开始扫描');
      ui.setStatus('等待摄像头首帧');
      dom.startBtn.textContent = originalLabel;
      dom.startBtn.disabled = false;
      return;
    }

    ui.setMsg('摄像头已就绪，正在锁定固定参数...');
    ui.setProg(0.84);
    ui.setStatus('锁定固定相机参数');
    if (typeof app.ensureCameraTunedForScan === 'function') {
      try {
        await app.ensureCameraTunedForScan('scan-start');
      } catch (error) {
        console.warn('[Camera] ensureCameraTunedForScan failed:', error && error.message ? error.message : error);
        ui.setStatus('相机参数锁定失败，继续启动');
      }
    }

    dom.startBtn.textContent = originalLabel;
    ui.setMsg('相机已就绪，正在启动识别...');
    ui.setProg(0.9);

    if (typeof app.getLocalizerMode === 'function') {
      const mode = app.getLocalizerMode();
      if (mode === 'scanner') {
        try {
          ui.setMsg('正在启动算法定位...');
          await app.initWorker(null);
          ui.setProg(1.0);
          ui.setMsg('就绪');
          await app.utils.sleep(300);
          dom.initOver.classList.add('hidden');
          app.onModelLoaded('scanner');
        } catch (error) {
          ui.setStatus('算法定位启动失败: ' + error.message);
          dom.startBtn.textContent = originalLabel;
          dom.startBtn.disabled = false;
        }
        return;
      }
      if (mode === 'contour') {
        ui.setMsg('正在启动轮廓定位...');
        ui.setProg(1.0);
        await app.utils.sleep(100);
        dom.initOver.classList.add('hidden');
        app.onModelLoaded('contour');
        return;
      }
    }

    const modelCandidates = ['model/best_dynamic.onnx', 'model/combined.onnx'];
    let modelLoaded = false;

    for (const modelPath of modelCandidates) {
      try {
        ui.setMsg('正在加载 ' + modelPath.split('/').pop() + '...');
        const response = await fetch(modelPath);
        if (!response.ok) {
          continue;
        }

        const buffer = await response.arrayBuffer();
        ui.setProg(0.94);
        ui.setMsg('正在编译 WebAssembly...');
        await app.initWorker(buffer);
        ui.setProg(1.0);
        ui.setMsg('就绪');
        await app.utils.sleep(300);
        dom.initOver.classList.add('hidden');
        app.onModelLoaded(modelPath.split('/').pop());
        modelLoaded = true;
        break;
      } catch (error) {
        console.warn('Failed to load ' + modelPath + ':', error);
      }
    }

    if (!modelLoaded) {
      ui.setMsg('未找到默认模型，请使用“加载模型”按钮选择 .onnx 文件');
      ui.setProg(1.0);
      await app.utils.sleep(800);
      dom.initOver.classList.add('hidden');
      ui.setStatus('请加载 .onnx 模型');
    }
  };

  app.pickModel = function pickModel() {
    dom.filePicker.click();
  };

  dom.filePicker.onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) {
      return;
    }

    dom.loadBtn.textContent = 'Loading...';
    dom.loadBtn.classList.add('loading');

    try {
      const buffer = await file.arrayBuffer();
      await app.initWorker(buffer);
      app.onModelLoaded(file.name);
    } catch (error) {
      ui.setStatus('Model load failed: ' + error.message);
      dom.loadBtn.textContent = 'Load Model';
      dom.loadBtn.classList.remove('loading');
    }

    event.target.value = '';
  };

  global.onStart = app.onStart;
  global.pickModel = app.pickModel;
})(window);
