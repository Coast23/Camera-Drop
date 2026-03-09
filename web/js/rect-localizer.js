(function (global) {
  const api = global.CamDropRectLocalizer = global.CamDropRectLocalizer || {};

  async function fetchArrayBuffer(url) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) {
      throw new Error('failed to fetch ' + url + ': ' + res.status);
    }
    return await res.arrayBuffer();
  }

  api.createYoloLocalizer = async function createYoloLocalizer(cfg) {
    if (!global.CameraDropApp || typeof global.CameraDropApp.getYoloWorkerSource !== 'function') {
      throw new Error('yolo worker source unavailable');
    }
    const workerSource = global.CameraDropApp.getYoloWorkerSource();
    const modelUrl = (cfg && cfg.modelUrl) || './model/best_dynamic.onnx';
    const modelBuf = await fetchArrayBuffer(modelUrl);
    const blob = new Blob([workerSource], { type: 'text/javascript' });
    const workerUrl = URL.createObjectURL(blob);
    const worker = new Worker(workerUrl);
    let readyResolve;
    let readyReject;
    let readyDone = false;
    let readyEP = 'unknown';
    let pending = null;
    const ready = new Promise((resolve, reject) => {
      readyResolve = resolve;
      readyReject = reject;
    });

    worker.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === 'ready') {
        readyEP = data.ep || readyEP;
        readyDone = true;
        readyResolve();
        return;
      }
      if (data.type === 'corners' && pending) {
        const cur = pending;
        pending = null;
        cur.resolve({
          corners: data.corners || null,
          ms: Number(data.ms) || 0,
          ep: readyEP,
          loc: data.loc || 'yolo',
        });
      }
    };

    worker.onerror = (event) => {
      const err = new Error(event && event.message ? event.message : 'yolo worker error');
      if (!readyDone) {
        readyReject(err);
        return;
      }
      if (pending) {
        const cur = pending;
        pending = null;
        cur.reject(err);
      }
    };

    worker.postMessage({ type: 'init', mode: 'yolo', model: modelBuf }, [modelBuf]);
    await ready;

    return {
      ep: readyEP,
      async detect(sourceCanvas, timeoutMs) {
        if (pending) {
          throw new Error('yolo detect already pending');
        }
        const bitmap = await createImageBitmap(sourceCanvas);
        return await new Promise((resolve, reject) => {
          const timer = setTimeout(() => {
            if (!pending) {
              return;
            }
            pending = null;
            try { bitmap.close(); } catch (_) {}
            reject(new Error('yolo detect timeout'));
          }, Math.max(1000, Number(timeoutMs) || 8000));
          pending = {
            resolve(value) {
              clearTimeout(timer);
              resolve(value);
            },
            reject(error) {
              clearTimeout(timer);
              reject(error);
            },
          };
          worker.postMessage({ type: 'frame', bitmap, patchOk: false, forceFull: true }, [bitmap]);
        });
      },
      dispose() {
        if (pending) {
          const cur = pending;
          pending = null;
          cur.reject(new Error('yolo localizer disposed'));
        }
        worker.terminate();
        URL.revokeObjectURL(workerUrl);
      },
    };
  };
})(window);
