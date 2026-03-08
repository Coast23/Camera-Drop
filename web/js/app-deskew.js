'use strict';

(function initDeskewModule(global) {
  const app = global.CameraDropApp;
  const state = app.state;
  const dom = app.dom;
  const config = app.config;

  function clamp(v, lo, hi) {
    return v < lo ? lo : (v > hi ? hi : v);
  }

  app.initGL = function initGL(canvas, options) {
    const opts = options || {};
    const gl = canvas.getContext('webgl', {
      antialias: false,
      preserveDrawingBuffer: true,
    });

    if (!gl) {
      console.warn('[GL] WebGL unavailable');
      return null;
    }

    const vs = 'attribute vec2 a; void main(){gl_Position=vec4(a,0,1);}';
    const fs = `
      precision highp float;
      uniform sampler2D u_tex;
      uniform mat3 u_H;
      uniform vec2 u_out, u_src;
      void main() {
        vec2 ic = vec2(gl_FragCoord.x, u_out.y - gl_FragCoord.y);
        vec3 p = u_H * vec3(ic, 1.0);
        vec2 uv = (p.xy / p.z) / u_src;
        if (uv.x<0.||uv.x>1.||uv.y<0.||uv.y>1.)
          { gl_FragColor=vec4(0.15,0.15,0.15,1.); return; }
        gl_FragColor = texture2D(u_tex, uv);
      }`;

    function makeShader(type, source) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      return shader;
    }

    const program = gl.createProgram();
    gl.attachShader(program, makeShader(gl.VERTEX_SHADER, vs));
    gl.attachShader(program, makeShader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(program);
    gl.useProgram(program);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    const aLoc = gl.getAttribLocation(program, 'a');
    gl.enableVertexAttribArray(aLoc);
    gl.vertexAttribPointer(aLoc, 2, gl.FLOAT, false, 0, 0);

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const filter = opts.filterMode === 'nearest' ? gl.NEAREST : gl.LINEAR;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.uniform1i(gl.getUniformLocation(program, 'u_tex'), 0);

    return {
      gl,
      tex: texture,
      hLoc: gl.getUniformLocation(program, 'u_H'),
      outSzLoc: gl.getUniformLocation(program, 'u_out'),
      srcSzLoc: gl.getUniformLocation(program, 'u_src'),
      defaultFilterMode: opts.filterMode === 'nearest' ? 'nearest' : 'linear',
    };
  };

  app.computeH = function computeH(srcPts, dstPts) {
    const A = new Float64Array(64);
    const b = new Float64Array(8);

    for (let i = 0; i < 4; i++) {
      const [x, y] = [srcPts[i][0], srcPts[i][1]];
      const [u, v] = [dstPts[i][0], dstPts[i][1]];
      const row = i * 2;
      A.set([x, y, 1, 0, 0, 0, -u * x, -u * y], row * 8);
      A.set([0, 0, 0, x, y, 1, -v * x, -v * y], (row + 1) * 8);
      b[row] = u;
      b[row + 1] = v;
    }

    const n = 8;
    const M = new Float64Array(n * (n + 1));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        M[i * (n + 1) + j] = A[i * n + j];
      }
      M[i * (n + 1) + n] = b[i];
    }

    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) {
        if (Math.abs(M[r * (n + 1) + c]) > Math.abs(M[p * (n + 1) + c])) {
          p = r;
        }
      }

      for (let j = 0; j <= n; j++) {
        const t = M[c * (n + 1) + j];
        M[c * (n + 1) + j] = M[p * (n + 1) + j];
        M[p * (n + 1) + j] = t;
      }

      for (let r = c + 1; r < n; r++) {
        const f = M[r * (n + 1) + c] / M[c * (n + 1) + c];
        for (let j = c; j <= n; j++) {
          M[r * (n + 1) + j] -= f * M[c * (n + 1) + j];
        }
      }
    }

    const h = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      h[i] = M[i * (n + 1) + n];
      for (let j = i + 1; j < n; j++) {
        h[i] -= M[i * (n + 1) + j] * h[j];
      }
      h[i] /= M[i * (n + 1) + i];
    }

    return new Float32Array([h[0], h[3], h[6], h[1], h[4], h[7], h[2], h[5], 1]);
  };

  app.getCanonicalDeskewInset = function getCanonicalDeskewInset(outSize) {
    const canonicalInset = Math.max(0, Number(config.DESKEW_CANONICAL_INSET) || 0);
    const baseSize = Math.max(1, Number(config.FINE_RENDER_SIZE) || 1024);
    return canonicalInset * (outSize / baseSize);
  };

  app.renderDeskew = function renderDeskew(ctx, targetCanvas, corners, expand, imageSource, outOverride) {
    if (!ctx || !corners) {
      return;
    }

    const src = imageSource || dom.video;
    if (!src || (src === dom.video && dom.video.readyState < 2)) {
      return;
    }

    const { TL, TR, BL, BR, outSize } = corners;
    const ratio = expand || 1;
    const cx = (TL[0] + TR[0] + BL[0] + BR[0]) / 4;
    const cy = (TL[1] + TR[1] + BL[1] + BR[1]) / 4;
    const scalePoint = ([px, py]) => [cx + (px - cx) * ratio, cy + (py - cy) * ratio];
    const [eTL, eTR, eBL, eBR] = [TL, TR, BL, BR].map(scalePoint);

    const out = Math.max(64, Math.round(Number.isFinite(outOverride) ? outOverride : (outSize * ratio)));
    const sw = src.videoWidth || src.width;
    const sh = src.videoHeight || src.height;
    const { gl, tex, hLoc, outSzLoc, srcSzLoc } = ctx;

    if (targetCanvas.width !== out || targetCanvas.height !== out) {
      targetCanvas.width = out;
      targetCanvas.height = out;
      gl.viewport(0, 0, out, out);
    }

    const inset = app.getCanonicalDeskewInset(out);
    const maxCoord = Math.max(inset, out - inset);
    const H = app.computeH(
      [[inset, inset], [maxCoord, inset], [inset, maxCoord], [maxCoord, maxCoord]],
      [eTL, eTR, eBL, eBR]
    );

    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    gl.uniformMatrix3fv(hLoc, false, H);
    gl.uniform2f(outSzLoc, out, out);
    gl.uniform2f(srcSzLoc, sw, sh);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  app.renderFine = function renderFine() {
    if (!state.pendingRender) {
      return;
    }

    app.renderDeskew(state.fineGl, dom.dskCvs, state.lastCorners, 1.0, state.pendingRender, config.FINE_RENDER_SIZE);
    state.pendingRender.close();
    state.pendingRender = null;
    state.lastDeskewTime = performance.now();
    dom.dskCvs.style.opacity = '1';
    if (typeof app.enqueueRecognizeFrame === 'function') {
      app.enqueueRecognizeFrame();
    }
  };


  app.recordDeskewFrame = function recordDeskewFrame(now) {
    const t = Number.isFinite(now) ? now : performance.now();
    state.dskFpsArr.push(1000 / Math.max(1, t - state.dskLastT));
    if (state.dskFpsArr.length > 12) {
      state.dskFpsArr.shift();
    }
    state.dskLastT = t;
    state.dskFps = state.dskFpsArr.reduce((a, b) => a + b, 0) / Math.max(1, state.dskFpsArr.length);
  };

  app.refreshPerfBar = function refreshPerfBar() {
    const yoloFps = state.yoloFpsArr.length
      ? (state.yoloFpsArr.reduce((a, b) => a + b, 0) / state.yoloFpsArr.length).toFixed(1)
      : '-';
    const locSource = state.localizerSource || '-';
    dom.perfBar.textContent = 'Deskew ' + state.dskFps.toFixed(1)
      + 'fps  blur ' + state.rawBlurScore.toFixed(1) + '/' + state.coarseBlurScore.toFixed(1) + '/' + state.fineBlurScore.toFixed(1)
      + '  Loc ' + locSource + ' ' + yoloFps + 'fps ' + state.yoloMs.toFixed(0) + 'ms [' + state.currentEP + ']'
      + '  Gate vf/t/s/d/q/ok ' + state.videoFrameCount + '/' + state.coarseTrackFreshCount + '/' + state.coarseHashSameCount + '/' + state.coarseHashDiffCount + '/' + state.preciseEnqueueCount + '/' + state.forceFullDoneCount
      + '  Drop p/y/r ' + state.preciseQueueDropCount + '/' + state.yoloQueueDropCount + '/' + state.recogQueueDropCount;
  };

  app.computeAHashFromSource = function computeAHashFromSource(source, marginRatio) {
    const N = config.AHASH_N;
    const srcW = source ? (source.videoWidth || source.width || 0) : 0;
    const srcH = source ? (source.videoHeight || source.height || 0) : 0;
    if (!srcW || !srcH) {
      return 0n;
    }

    const ratio = Number.isFinite(marginRatio) ? marginRatio : 0.08;
    const marginX = Math.round(srcW * ratio);
    const marginY = Math.round(srcH * ratio);
    const innerW = Math.max(1, srcW - 2 * marginX);
    const innerH = Math.max(1, srcH - 2 * marginY);

    state.ahCtx.drawImage(source, marginX, marginY, innerW, innerH, 0, 0, N, N);
    const data = state.ahCtx.getImageData(0, 0, N, N).data;
    const gray = new Uint8Array(N * N);
    let sum = 0;

    for (let i = 0; i < N * N; i++) {
      gray[i] = (data[i * 4] * 77 + data[i * 4 + 1] * 150 + data[i * 4 + 2] * 29) >> 8;
      sum += gray[i];
    }

    const avg = sum / (N * N);
    let hash = 0n;
    for (let i = 0; i < N * N; i++) {
      if (gray[i] >= avg) {
        hash |= (1n << BigInt(i));
      }
    }
    return hash;
  };

  app.computeAHash = function computeAHash() {
    return app.computeAHashFromSource(state.offDsk, 0.08);
  };

  app.getBlurThreshold = function getBlurThreshold(stage) {
    const rawOverride = Number(global.__CAMDROP_RAW_BLUR_THRESH);
    const coarseOverride = Number(global.__CAMDROP_COARSE_BLUR_THRESH);
    const fineOverride = Number(global.__CAMDROP_FINE_BLUR_THRESH);
    if (stage === 'fine') {
      return Number.isFinite(fineOverride) ? fineOverride : config.FINE_BLUR_THRESH;
    }
    if (stage === 'raw') {
      return Number.isFinite(rawOverride) ? rawOverride : config.RAW_BLUR_THRESH;
    }
    return Number.isFinite(coarseOverride) ? coarseOverride : config.COARSE_BLUR_THRESH;
  };

  app.getRawBlurSampleRect = function getRawBlurSampleRect(source) {
    const srcW = source ? (source.videoWidth || source.width || 0) : 0;
    const srcH = source ? (source.videoHeight || source.height || 0) : 0;
    const corners = state.lastCorners;
    if (!srcW || !srcH || !corners) {
      return null;
    }
    const pts = [corners.TL, corners.TR, corners.BL, corners.BR];
    if (!pts.every((pt) => Array.isArray(pt) && pt.length >= 2 && Number.isFinite(pt[0]) && Number.isFinite(pt[1]))) {
      return null;
    }
    let minX = srcW;
    let minY = srcH;
    let maxX = 0;
    let maxY = 0;
    for (let i = 0; i < pts.length; i++) {
      const x = pts[i][0];
      const y = pts[i][1];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
    const w = Math.max(1, maxX - minX);
    const h = Math.max(1, maxY - minY);
    const padX = Math.max(12, w * 0.08);
    const padY = Math.max(12, h * 0.08);
    const x = clamp(minX + padX, 0, srcW - 1);
    const y = clamp(minY + padY, 0, srcH - 1);
    const right = clamp(maxX - padX, x + 1, srcW);
    const bottom = clamp(maxY - padY, y + 1, srcH);
    return {
      x,
      y,
      width: Math.max(1, right - x),
      height: Math.max(1, bottom - y),
    };
  };

  app.measureBlurScore = function measureBlurScore(source, options) {
    const srcW = source ? (source.videoWidth || source.width || 0) : 0;
    const srcH = source ? (source.videoHeight || source.height || 0) : 0;
    if (!srcW || !srcH || !state.blurCtx) {
      return 0;
    }

    const N = config.BLUR_SAMPLE_N;
    const srcRect = options && options.srcRect ? options.srcRect : null;
    let sampleX = 0;
    let sampleY = 0;
    let sampleW = srcW;
    let sampleH = srcH;
    if (srcRect
        && Number.isFinite(srcRect.x)
        && Number.isFinite(srcRect.y)
        && Number.isFinite(srcRect.width)
        && Number.isFinite(srcRect.height)
        && srcRect.width > 1
        && srcRect.height > 1) {
      sampleX = clamp(Math.round(srcRect.x), 0, srcW - 1);
      sampleY = clamp(Math.round(srcRect.y), 0, srcH - 1);
      const right = clamp(Math.round(srcRect.x + srcRect.width), sampleX + 1, srcW);
      const bottom = clamp(Math.round(srcRect.y + srcRect.height), sampleY + 1, srcH);
      sampleW = Math.max(1, right - sampleX);
      sampleH = Math.max(1, bottom - sampleY);
    } else {
      const marginRatio = options && Number.isFinite(options.marginRatio) ? options.marginRatio : 0.08;
      const marginX = Math.round(srcW * marginRatio);
      const marginY = Math.round(srcH * marginRatio);
      sampleX = marginX;
      sampleY = marginY;
      sampleW = Math.max(1, srcW - 2 * marginX);
      sampleH = Math.max(1, srcH - 2 * marginY);
    }

    state.blurCtx.drawImage(source, sampleX, sampleY, sampleW, sampleH, 0, 0, N, N);
    const data = state.blurCtx.getImageData(0, 0, N, N).data;
    const gray = state.blurGray;

    for (let i = 0; i < gray.length; i++) {
      gray[i] = (data[i * 4] * 77 + data[i * 4 + 1] * 150 + data[i * 4 + 2] * 29) >> 8;
    }

    let sum = 0;
    let count = 0;
    for (let y = 1; y < N - 1; y++) {
      const row = y * N;
      for (let x = 1; x < N - 1; x++) {
        const idx = row + x;
        sum += Math.abs(gray[idx + 1] - gray[idx - 1]);
        sum += Math.abs(gray[idx + N] - gray[idx - N]);
        count += 2;
      }
    }

    return count ? (sum / count) : 0;
  };

  app.runRawBlurPrecheck = function runRawBlurPrecheck(source) {
    if (config.RAW_BLUR_ENABLED === false) {
      state.rawBlurScore = 0;
      state.rawBlurPass = true;
      return true;
    }
    if (!source || typeof app.measureBlurScore !== 'function') {
      state.rawBlurPass = true;
      return true;
    }
    const sampleRect = typeof app.getRawBlurSampleRect === 'function'
      ? app.getRawBlurSampleRect(source)
      : null;
    const blurScore = app.measureBlurScore(source, sampleRect ? { srcRect: sampleRect } : { marginRatio: 0.08 });
    state.rawBlurScore = blurScore;

    const hasTrack = !!(sampleRect && state.lastCorners);
    const trackTtlMs = Math.max(200, Number(config.RAW_BLUR_TRACK_TTL_MS) || 1200);
    const trackFresh = !!(state.lastDeskewTime && (performance.now() - state.lastDeskewTime) <= trackTtlMs);
    const gateEnabled = hasTrack && trackFresh;
    const passed = !gateEnabled || blurScore >= app.getBlurThreshold('raw');
    const blocking = config.RAW_BLUR_BLOCKING === true;

    state.rawBlurPass = passed;
    if (gateEnabled && !passed) {
      state.rawBlurRejectCount++;
    }
    return blocking ? passed : true;
  };

  app.hammingDist = function hammingDist(a, b) {
    let diff = a ^ b;
    let count = 0;
    while (diff) {
      count += Number(diff & 1n);
      diff >>= 1n;
    }
    return count;
  };

  app.deskewLoop = function deskewLoop() {
    if (typeof app.getLocalizerMode === 'function' && app.getLocalizerMode() === 'contour') {
      return;
    }
    if (!state.scanning) {
      return;
    }
    if (state.cameraTunePending) {
      return;
    }

    state.deskewLoopCount++;
    if (!state.coarseGl) {
      state.deskewSkipNoGlCount++;
      return;
    }
    if (!state.lastCorners) {
      state.deskewSkipNoCornersCount++;
      return;
    }
    if (dom.video.readyState < 2) {
      state.deskewSkipNotReadyCount++;
      return;
    }

    if (typeof app.claimVideoFrame === 'function' && !app.claimVideoFrame('lastDeskewVideoTime')) {
      state.deskewSkipClaimCount++;
      return;
    }

    app.renderDeskew(state.coarseGl, state.offDsk, state.lastCorners, config.DESKEW_EXP);

    const coarseBlur = app.measureBlurScore(state.offDsk, { marginRatio: 0.08 });
    state.coarseBlurScore = coarseBlur;
    const coarsePassed = coarseBlur >= app.getBlurThreshold('coarse');
    const coarseBlocking = config.COARSE_BLUR_BLOCKING === true;
    if (coarsePassed || !coarseBlocking) {
      if (!coarsePassed) {
        state.blurRejectCount++;
      }
      state.lastCoarseHandledVideoTime = state.currentFrameToken;
      state.lastCoarseGateTime = performance.now();
      const newHash = app.computeAHash();
      const hashDist = state.lastAHash === null ? (config.AHASH_THRESH + 1) : app.hammingDist(newHash, state.lastAHash);
      if (state.lastAHash !== null && hashDist <= config.AHASH_THRESH) {
        state.coarseHashSameCount++;
      } else {
        state.coarseHashDiffCount++;
        state.lastAHash = newHash;
        if (typeof app.canAcceptLocalizerTask === 'function' && !app.canAcceptLocalizerTask('precise')) {
          state.preciseQueueDropCount++;
          return;
        }
        const captureSeq = ++state.localizerCaptureSeq;
        Promise.all([createImageBitmap(dom.video), createImageBitmap(dom.video)]).then(([worker, render]) => {
          const enqueued = typeof app.enqueueLocalizerTask === 'function'
            ? app.enqueueLocalizerTask('precise', {
                bitmap: worker,
                render,
                patchOk: false,
                forceFull: true,
                captureToken: state.currentFrameToken,
                captureSeq,
              })
            : false;
          if (enqueued) {
            state.preciseEnqueueCount++;
            if (typeof app.pumpYoloQueue === 'function') {
              app.pumpYoloQueue();
            }
          }
        }).catch((err) => {
          console.warn('[PreciseQueue] capture failed:', err);
        });
      }
    } else {
      state.blurRejectCount++;
    }

    const now = performance.now();
    app.recordDeskewFrame(now);
    app.refreshPerfBar();
  };

  setInterval(() => {
    if (!state.scanning || !state.lastDeskewTime) {
      return;
    }
    if (performance.now() - state.lastDeskewTime > 3000) {
      dom.dskCvs.style.opacity = '0.3';
    }
  }, 500);

  app.downloadDeskewed = function downloadDeskewed() {
    const url = dom.dskCvs.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = 'deskewed_' + Date.now() + '.png';
    a.click();
  };

  global.downloadDeskewed = app.downloadDeskewed;
})(window);



