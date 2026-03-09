(function (global) {
  const dom = {
    scene: document.getElementById('sceneCanvas'),
    code: document.getElementById('codeCanvas'),
    deskew: document.getElementById('deskewCanvas'),
  };

  const state = {
    cvReady: false,
  };

  global.onCvReady = function onCvReady() {
    state.cvReady = true;
  };

  function waitCvReady(timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    return new Promise((resolve, reject) => {
      function tick() {
        if (state.cvReady && global.cv && cv.Mat) {
          resolve();
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
  }

  function makeRng(seed) {
    let s = seed >>> 0;
    return function rand() {
      s ^= s << 13;
      s ^= s >>> 17;
      s ^= s << 5;
      return (s >>> 0) / 4294967296;
    };
  }

  function drawBackground(ctx, w, h, rand, distortion) {
    const r1 = (rand() * 140 + 20) | 0;
    const g1 = (rand() * 140 + 20) | 0;
    const b1 = (rand() * 140 + 20) | 0;
    const r2 = (rand() * 140 + 20) | 0;
    const g2 = (rand() * 140 + 20) | 0;
    const b2 = (rand() * 140 + 20) | 0;
    const grad = ctx.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, 'rgb(' + r1 + ',' + g1 + ',' + b1 + ')');
    grad.addColorStop(1, 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
    const distractors = distortion === 'full' ? 14 : distortion === 'light' ? 10 : 6;
    for (let i = 0; i < distractors; i++) {
      const x = rand() * w;
      const y = rand() * h;
      const s = 32 + rand() * 180;
      const alpha = distortion === 'full' ? 0.42 : 0.30;
      ctx.fillStyle = 'rgba(' + ((rand() * 255) | 0) + ',' + ((rand() * 255) | 0) + ',' + ((rand() * 255) | 0) + ',' + alpha + ')';
      ctx.fillRect(x, y, s, s);
    }
  }

  function drawOccluders(ctx, w, h, rand, distortion) {
    const n = distortion === 'full' ? 4 : distortion === 'light' ? 2 : 0;
    for (let i = 0; i < n; i++) {
      const x = rand() * w;
      const y = rand() * h;
      const ww = 80 + rand() * 180;
      const hh = 20 + rand() * 60;
      const gray = 40 + ((rand() * 100) | 0);
      ctx.fillStyle = 'rgba(' + gray + ',' + gray + ',' + gray + ',' + (distortion === 'full' ? 0.48 : 0.32) + ')';
      ctx.fillRect(x, y, ww, hh);
    }
  }

  function initPose(rand, distortion, camW, camH) {
    let scale = 0.72;
    if (distortion === 'light') scale = 0.64;
    if (distortion === 'full') scale = 0.58;
    return {
      cx: camW * (0.45 + rand() * 0.10),
      cy: camH * (0.45 + rand() * 0.10),
      scale,
      angle: (rand() - 0.5) * (distortion === 'full' ? 0.20 : 0.10),
    };
  }

  function updatePose(pose, rand, distortion, camW, camH, codeW, codeH) {
    const j = distortion === 'full' ? 1.0 : distortion === 'light' ? 0.75 : 0.55;
    pose.cx += (rand() - 0.5) * 22 * j;
    pose.cy += (rand() - 0.5) * 16 * j;
    pose.angle += (rand() - 0.5) * 0.030 * j;
    pose.scale += (rand() - 0.5) * 0.020 * j;
    const minScale = distortion === 'full' ? 0.48 : 0.56;
    const maxScale = distortion === 'full' ? 0.75 : 0.82;
    if (pose.scale < minScale) pose.scale = minScale;
    if (pose.scale > maxScale) pose.scale = maxScale;
    if (pose.angle < -0.28) pose.angle = -0.28;
    if (pose.angle > 0.28) pose.angle = 0.28;
    const halfW = codeW * pose.scale * 0.52;
    const halfH = codeH * pose.scale * 0.52;
    if (pose.cx < halfW) pose.cx = halfW;
    if (pose.cx > camW - halfW) pose.cx = camW - halfW;
    if (pose.cy < halfH) pose.cy = halfH;
    if (pose.cy > camH - halfH) pose.cy = camH - halfH;
  }

  function drawCodeOnScene(sceneCtx, codeCanvas, pose, distortion, rand, camW, camH) {
    const scale = pose.scale;
    const angle = pose.angle;
    const cx = pose.cx;
    const cy = pose.cy;
    const codeW = codeCanvas.width;
    const codeH = codeCanvas.height;
    sceneCtx.save();
    sceneCtx.translate(cx, cy);
    sceneCtx.rotate(angle);
    sceneCtx.scale(scale, scale);
    if (distortion === 'none') sceneCtx.filter = 'none';
    else if (distortion === 'light') sceneCtx.filter = 'blur(0.6px)';
    else sceneCtx.filter = 'blur(1.2px)';
    sceneCtx.drawImage(codeCanvas, -codeW / 2, -codeH / 2, codeW, codeH);
    sceneCtx.filter = 'none';
    sceneCtx.restore();
    if (distortion !== 'none') {
      const lines = distortion === 'full' ? 30 : 12;
      for (let i = 0; i < lines; i++) {
        const y = (rand() * camH) | 0;
        const a = distortion === 'full' ? 0.10 : 0.06;
        sceneCtx.fillStyle = 'rgba(255,255,255,' + a + ')';
        sceneCtx.fillRect(0, y, camW, 1);
      }
    }
  }

  function makeGtCornersFromPose(codeW, codeH, pose) {
    const halfW = codeW / 2;
    const halfH = codeH / 2;
    const cs = Math.cos(pose.angle);
    const sn = Math.sin(pose.angle);
    function map(lx, ly) {
      const x = lx * pose.scale;
      const y = ly * pose.scale;
      return [pose.cx + x * cs - y * sn, pose.cy + x * sn + y * cs];
    }
    return {
      TL: map(-halfW, -halfH),
      TR: map(halfW, -halfH),
      BR: map(halfW, halfH),
      BL: map(-halfW, halfH),
    };
  }

  function getRawPlacement(camW, camH, codeW, codeH) {
    const pad = Math.max(24, Math.round(Math.min(camW, camH) * 0.06));
    const availW = Math.max(1, camW - pad * 2);
    const availH = Math.max(1, camH - pad * 2);
    const scale = Math.min(availW / Math.max(1, codeW), availH / Math.max(1, codeH), 1);
    const drawW = Math.max(1, Math.round(codeW * scale));
    const drawH = Math.max(1, Math.round(codeH * scale));
    const dx = ((camW - drawW) / 2) | 0;
    const dy = ((camH - drawH) / 2) | 0;
    return { dx, dy, drawW, drawH };
  }

  function makeGtCornersFromRaw(camW, camH, codeW, codeH) {
    const placement = getRawPlacement(camW, camH, codeW, codeH);
    const dx = placement.dx;
    const dy = placement.dy;
    const drawW = placement.drawW;
    const drawH = placement.drawH;
    return {
      TL: [dx, dy],
      TR: [dx + drawW - 1, dy],
      BR: [dx + drawW - 1, dy + drawH - 1],
      BL: [dx, dy + drawH - 1],
    };
  }

  function compareUnits(decoded, expected) {
    const n = Math.min(decoded.length, expected.length);
    let symbolCorrect = 0;
    let patternCorrect = 0;
    let colorCorrect = 0;
    for (let i = 0; i < n; i++) {
      const d = decoded[i];
      const e = expected[i];
      if (d === e) symbolCorrect++;
      if ((d & 0x0f) === (e & 0x0f)) patternCorrect++;
      if ((d >> 4) === (e >> 4)) colorCorrect++;
    }
    return { n, symbolCorrect, patternCorrect, colorCorrect };
  }

  function bytesEqual(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false;
    }
    return true;
  }

  function makeDeterministicFile(seed, size) {
    const rand = makeRng(seed >>> 0);
    const out = new Uint8Array(size);
    for (let i = 0; i < size; i++) {
      out[i] = (rand() * 256) | 0;
    }
    return out;
  }

  function getDisplaySize(layout, cfg) {
    const aspect = Number(cfg.codeAspect) > 0 ? Number(cfg.codeAspect) : (layout.imgWidth / layout.imgHeight);
    const shortSide = Math.max(240, Number(cfg.codeShortSide) || 720);
    if (aspect >= 1) {
      return { width: Math.round(shortSide * aspect), height: shortSide };
    }
    return { width: shortSide, height: Math.round(shortSide / aspect) };
  }

  async function fetchArrayBuffer(url) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) {
      throw new Error('failed to fetch ' + url + ': ' + res.status);
    }
    return await res.arrayBuffer();
  }

  async function createYoloLocalizer(cfg) {
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
    let readyEP = 'unknown';
    let pending = null;
    let readyDone = false;
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
            if (!pending) return;
            pending = null;
            try { bitmap.close(); } catch (_) {}
            reject(new Error('yolo detect timeout'));
          }, Math.max(1000, timeoutMs || 8000));
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
  }

  async function runCase(cfg) {
    await waitCvReady(Math.max(1000, Number(cfg.timeoutMs) || 30000));
    const layout = await global.CamDropRectCodec.getLayout();
    const cornerMode = cfg && cfg.cornerMode === 'yolo' ? 'yolo' : 'gt';
    const canonical = document.createElement('canvas');
    const displaySize = getDisplaySize(layout, cfg);
    dom.code.width = displaySize.width;
    dom.code.height = displaySize.height;
    dom.deskew.width = layout.imgWidth;
    dom.deskew.height = layout.imgHeight;
    const camW = cfg.rawInput ? displaySize.width : 1280;
    const camH = cfg.rawInput ? displaySize.height : 720;
    dom.scene.width = camW;
    dom.scene.height = camH;
    const sceneCtx = dom.scene.getContext('2d', { willReadFrequently: true });
    const codeCtx = dom.code.getContext('2d', { willReadFrequently: true });
    sceneCtx.imageSmoothingEnabled = true;
    codeCtx.imageSmoothingEnabled = true;

    const fileSeed = (cfg.seed >>> 0) ^ 0x51f15eed;
    const fileBytes = makeDeterministicFile(fileSeed, Math.max(256, Number(cfg.fileBytes) || 65536));
    const fileName = cfg.fileName || 'rect_e2e_payload.bin';

    const encoder = await global.CamDropRectCodec.createEncoder(fileBytes, fileName);
    const decoder = await global.CamDropRectCodec.createDecoder();
    const yoloLocalizer = cornerMode === 'yolo' ? await createYoloLocalizer(cfg) : null;
    let phase = 'setup';
    try {
      const tCase0 = performance.now();
      phase = 'frames';
      const frames = Math.max(1, Math.round((Number(cfg.fps) || 1) * (Number(cfg.durationSec) || 1)));
      const poseRand = makeRng((cfg.seed ^ 0x9e3779b9) >>> 0);
      const pose = initPose(poseRand, cfg.distortion || 'none', camW, camH);
      let symbolTotal = 0;
      let symbolCorrect = 0;
      let patternCorrect = 0;
      let colorCorrect = 0;
      let exactPacketFrames = 0;
      let completedAt = -1;
      const perFrame = [];

      for (let i = 0; i < frames; i++) {
        phase = 'frame ' + (i + 1) + ' getPacket';
        const packetBytes = await encoder.getPacket();
        phase = 'frame ' + (i + 1) + ' packetToUnits';
        const expectedUnits = await global.CamDropRectCodec.packetToUnits(packetBytes);
        phase = 'frame ' + (i + 1) + ' renderUnits';
        await global.CamDropRectRender.renderUnitsToCanvas(canonical, expectedUnits, { scale: 1 });
        phase = 'frame ' + (i + 1) + ' drawCode';
        codeCtx.clearRect(0, 0, dom.code.width, dom.code.height);
        codeCtx.drawImage(canonical, 0, 0, dom.code.width, dom.code.height);

        let gtCorners;
        if (cfg.rawInput) {
          phase = 'frame ' + (i + 1) + ' rawScene';
          const placement = getRawPlacement(camW, camH, dom.code.width, dom.code.height);
          sceneCtx.fillStyle = '#000';
          sceneCtx.fillRect(0, 0, camW, camH);
          sceneCtx.drawImage(dom.code, placement.dx, placement.dy, placement.drawW, placement.drawH);
          gtCorners = makeGtCornersFromRaw(camW, camH, dom.code.width, dom.code.height);
        } else {
          phase = 'frame ' + (i + 1) + ' posedScene';
          updatePose(pose, poseRand, cfg.distortion || 'none', camW, camH, dom.code.width, dom.code.height);
          drawBackground(sceneCtx, camW, camH, poseRand, cfg.distortion || 'none');
          drawCodeOnScene(sceneCtx, dom.code, pose, cfg.distortion || 'none', poseRand, camW, camH);
          drawOccluders(sceneCtx, camW, camH, poseRand, cfg.distortion || 'none');
          gtCorners = makeGtCornersFromPose(dom.code.width, dom.code.height, pose);
        }

        let corners = gtCorners;
        let localizerMs = 0;
        let localizerOk = true;
        if (cornerMode === 'yolo') {
          phase = 'frame ' + (i + 1) + ' yolo';
          const yoloRes = await yoloLocalizer.detect(dom.scene, Number(cfg.yoloTimeoutMs) || 12000);
          corners = yoloRes.corners;
          localizerMs = yoloRes.ms || 0;
          localizerOk = !!corners;
        }
        if (!corners) {
          symbolTotal += expectedUnits.length;
          perFrame.push({
            index: i + 1,
            symbolAcc: 0,
            patternAcc: 0,
            colorAcc: 0,
            packetExact: false,
            decoderAccepted: false,
            avgPatternDist: 64,
            localizerMs,
            localizerOk,
          });
          continue;
        }

        phase = 'frame ' + (i + 1) + ' warp';
        await global.CamDropRectRecognizer.warpSceneToCanvas(dom.scene, corners, dom.deskew);
        phase = 'frame ' + (i + 1) + ' decodeCanonical';
        const decoded = await global.CamDropRectRecognizer.decodeCanonicalCanvas(dom.deskew);
        phase = 'frame ' + (i + 1) + ' compare';
        const cmp = compareUnits(decoded.units, expectedUnits);
        const packetExact = bytesEqual(decoded.packetBytes, packetBytes);
        symbolTotal += cmp.n;
        symbolCorrect += cmp.symbolCorrect;
        patternCorrect += cmp.patternCorrect;
        colorCorrect += cmp.colorCorrect;
        if (packetExact) exactPacketFrames++;
        let decoderAccepted = false;
        try {
          phase = 'frame ' + (i + 1) + ' processPacket';
          decoder.processPacket(decoded.packetBytes);
          decoderAccepted = true;
        } catch (_) {
          decoderAccepted = false;
        }
        phase = 'frame ' + (i + 1) + ' isComplete';
        if (completedAt < 0 && decoder.isComplete()) {
          completedAt = i + 1;
        }
        perFrame.push({
          index: i + 1,
          symbolAcc: cmp.n ? (cmp.symbolCorrect / cmp.n) * 100 : 0,
          patternAcc: cmp.n ? (cmp.patternCorrect / cmp.n) * 100 : 0,
          colorAcc: cmp.n ? (cmp.colorCorrect / cmp.n) * 100 : 0,
          packetExact,
          decoderAccepted,
          avgPatternDist: decoded.avgPatternDist,
          localizerMs,
          localizerOk,
        });
      }

      phase = 'finalize';
      const fileComplete = decoder.isComplete();
      const outBytes = fileComplete ? decoder.getFileBytes() : new Uint8Array(0);
      const outName = fileComplete ? decoder.getFilename() : '';
      const fileExact = fileComplete && outName === fileName && bytesEqual(outBytes, fileBytes);
      const elapsedMs = performance.now() - tCase0;
      const throughputFps = frames > 0 && elapsedMs > 0 ? (frames * 1000 / elapsedMs) : 0;
      const realtimeOK = throughputFps >= (Number(cfg.fps) || 1);

      return {
        frames,
        symbolAcc: symbolTotal ? (symbolCorrect / symbolTotal) * 100 : 0,
        patternAcc: symbolTotal ? (patternCorrect / symbolTotal) * 100 : 0,
        colorAcc: symbolTotal ? (colorCorrect / symbolTotal) * 100 : 0,
        exactPacketFrames,
        exactPacketRatio: frames ? (exactPacketFrames / frames) * 100 : 0,
        completedAt,
        fileComplete,
        fileExact,
        fileName,
        decodedName: outName,
        fileBytes: fileBytes.length,
        decodedBytes: outBytes.length,
        cornerMode,
        yoloEP: yoloLocalizer ? yoloLocalizer.ep : '',
        elapsedMs,
        throughputFps,
        realtimeOK,
        perFrame,
      };
    } catch (err) {
      throw new Error('runCase failed at ' + phase + ': ' + (err && err.message ? err.message : String(err)));
    } finally {
      if (yoloLocalizer) {
        yoloLocalizer.dispose();
      }
      encoder.destroy();
      decoder.destroy();
    }
  }

  global.__rectE2E = { runCase };
})(window);
