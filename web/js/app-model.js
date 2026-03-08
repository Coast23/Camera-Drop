'use strict';

(function initModelModule(global) {
  const app = global.CameraDropApp;
  const dom = app.dom;
  const ui = app.ui;

  app.onStart = async function onStart() {
    const originalLabel = dom.startBtn.textContent || '\u5f00\u59cb\u626b\u63cf';
    dom.startBtn.disabled = true;
    dom.startBtn.textContent = '\u6b63\u5728\u51c6\u5907...';
    ui.setMsg('\u6b63\u5728\u8bf7\u6c42\u6444\u50cf\u5934...');
    ui.setProg(0.72);
    ui.setStatus('\u51c6\u5907\u6444\u50cf\u5934');

    const ok = await app.startCamera(false);
    if (!ok) {
      ui.setMsg('\u8bf7\u5148\u5141\u8bb8\u6444\u50cf\u5934\u6743\u9650\uff0c\u5e76\u7b49\u5f85\u9996\u5e27\u663e\u793a\u540e\u518d\u5f00\u59cb\u626b\u63cf');
      ui.setStatus('\u7b49\u5f85\u6444\u50cf\u5934\u9996\u5e27');
      dom.startBtn.textContent = originalLabel;
      dom.startBtn.disabled = false;
      return;
    }

    ui.setMsg('\u6444\u50cf\u5934\u5df2\u5c31\u7eea\uff0c\u6b63\u5728\u9501\u5b9a\u53c2\u6570...');
    ui.setProg(0.84);
    ui.setStatus('\u9501\u5b9a\u76f8\u673a\u53c2\u6570');
    if (typeof app.ensureCameraTunedForScan === 'function') {
      try {
        await app.ensureCameraTunedForScan('scan-start', { forceLearn: true });
      } catch (error) {
        console.warn('[Camera] ensureCameraTunedForScan failed:', error && error.message ? error.message : error);
        ui.setStatus('\u76f8\u673a\u53c2\u6570\u9501\u5b9a\u5931\u8d25\uff0c\u7ee7\u7eed\u542f\u52a8');
      }
    }

    dom.startBtn.textContent = originalLabel;
    ui.setMsg('\u76f8\u673a\u5df2\u5c31\u7eea\uff0c\u6b63\u5728\u542f\u52a8\u8bc6\u522b...');
    ui.setProg(0.9);

    if (typeof app.getLocalizerMode === 'function') {
      const mode = app.getLocalizerMode();
      if (mode === 'scanner') {
        try {
          ui.setMsg('\u6b63\u5728\u542f\u52a8\u7b97\u6cd5\u5b9a\u4f4d...');
          await app.initWorker(null);
          ui.setProg(1.0);
          ui.setMsg('\u5c31\u7eea');
          await app.utils.sleep(300);
          dom.initOver.classList.add('hidden');
          app.onModelLoaded('scanner');
        } catch (error) {
          ui.setStatus('\u7b97\u6cd5\u5b9a\u4f4d\u542f\u52a8\u5931\u8d25: ' + error.message);
          dom.startBtn.textContent = originalLabel;
          dom.startBtn.disabled = false;
        }
        return;
      }
      if (mode === 'contour') {
        ui.setMsg('\u6b63\u5728\u542f\u52a8\u8f6e\u5ed3\u5b9a\u4f4d...');
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
        ui.setMsg('\u6b63\u5728\u52a0\u8f7d ' + modelPath.split('/').pop() + '...');
        const response = await fetch(modelPath);
        if (!response.ok) {
          continue;
        }

        const buffer = await response.arrayBuffer();
        ui.setProg(0.94);
        ui.setMsg('\u6b63\u5728\u7f16\u8bd1 WebAssembly...');
        await app.initWorker(buffer);
        ui.setProg(1.0);
        ui.setMsg('\u5c31\u7eea');
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
      ui.setMsg('\u672a\u627e\u5230\u9ed8\u8ba4\u6a21\u578b\uff0c\u8bf7\u4f7f\u7528\u201c\u52a0\u8f7d\u6a21\u578b\u201d\u6309\u94ae\u9009\u62e9 .onnx \u6587\u4ef6');
      ui.setProg(1.0);
      await app.utils.sleep(800);
      dom.initOver.classList.add('hidden');
      ui.setStatus('\u8bf7\u52a0\u8f7d .onnx \u6a21\u578b');
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
