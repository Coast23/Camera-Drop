'use strict';

(function initPatchModule(global) {
  const app = global.CameraDropApp;
  const state = app.state;
  const dom = app.dom;
  const config = app.config;

  app.initPatches = function initPatches() {
    if (!state.lastCorners || dom.video.readyState < 2) {
      return;
    }

    const points = [
      state.lastCorners.TL,
      state.lastCorners.TR,
      state.lastCorners.BL,
      state.lastCorners.BR,
    ];

    state.patches = points.map(([cx, cy]) => {
      const x0 = Math.round(cx - config.PATCH_SZ / 2);
      const y0 = Math.round(cy - config.PATCH_SZ / 2);
      const patchCanvas = new OffscreenCanvas(config.PATCH_SZ, config.PATCH_SZ);
      const patchCtx = patchCanvas.getContext('2d', { willReadFrequently: true });
      patchCtx.drawImage(dom.video, x0, y0, config.PATCH_SZ, config.PATCH_SZ, 0, 0, config.PATCH_SZ, config.PATCH_SZ);
      const image = patchCtx.getImageData(0, 0, config.PATCH_SZ, config.PATCH_SZ);
      const gray = new Float32Array(config.PATCH_SZ * config.PATCH_SZ);

      for (let i = 0; i < gray.length; i++) {
        gray[i] = (image.data[i * 4] + image.data[i * 4 + 1] + image.data[i * 4 + 2]) / 765;
      }

      return { cx, cy, gray };
    });
  };

  app.trackPatches = function trackPatches() {
    if (!state.patches || dom.video.readyState < 2) {
      return null;
    }

    const mode = typeof app.getLocalizerMode === 'function' ? app.getLocalizerMode() : 'yolo';
    const searchRadius = mode === 'contour'
      ? Math.max(config.PATCH_SRCH, Number(config.PATCH_SRCH_CONTOUR) || 0)
      : config.PATCH_SRCH;
    const searchArea = config.PATCH_SZ + 2 * searchRadius;
    const newPoints = [];

    for (const patch of state.patches) {
      const x0 = Math.round(patch.cx - config.PATCH_SZ / 2 - searchRadius);
      const y0 = Math.round(patch.cy - config.PATCH_SZ / 2 - searchRadius);

      state.srchCtx.drawImage(dom.video, x0, y0, searchArea, searchArea, 0, 0, searchArea, searchArea);
      const image = state.srchCtx.getImageData(0, 0, searchArea, searchArea);

      let bestSAD = Infinity;
      let bestDx = 0;
      let bestDy = 0;

      for (let dy = -searchRadius; dy <= searchRadius; dy++) {
        for (let dx = -searchRadius; dx <= searchRadius; dx++) {
          let sad = 0;
          const oy = searchRadius + dy;
          const ox = searchRadius + dx;

          for (let py = 0; py < config.PATCH_SZ; py++) {
            for (let px = 0; px < config.PATCH_SZ; px++) {
              const si = ((oy + py) * searchArea + (ox + px)) * 4;
              const g = (image.data[si] + image.data[si + 1] + image.data[si + 2]) / 765;
              sad += Math.abs(g - patch.gray[py * config.PATCH_SZ + px]);
            }
          }

          if (sad < bestSAD) {
            bestSAD = sad;
            bestDx = dx;
            bestDy = dy;
          }
        }
      }

      if (bestSAD / (config.PATCH_SZ * config.PATCH_SZ) > config.PATCH_SAD_MAX) {
        return null;
      }

      patch.cx += bestDx;
      patch.cy += bestDy;
      newPoints.push([patch.cx, patch.cy]);
    }

    return newPoints;
  };

  app.patchTrackLoop = function patchTrackLoop() {
    if (!state.scanning) {
      return;
    }
    if (!state.patches || !state.lastCorners) {
      return;
    }
    if (typeof app.claimVideoFrame === 'function' && !app.claimVideoFrame('lastPatchVideoTime')) {
      return;
    }

    const isContour = typeof app.getLocalizerMode === 'function' && app.getLocalizerMode() === 'contour';
    const points = app.trackPatches();
    if (!points) {
      state.patches = null;
      state.lastAHash = null;
      if (isContour) {
        state.lastContourRunAt = 0;
        state.contourTrackHash = null;
        state.contourNeedRefine = true;
      }
      return;
    }

    const [TL, TR, BL, BR] = points;
    state.lastCorners = { TL, TR, BL, BR, outSize: state.lastCorners.outSize };

    if (!isContour) {
      return;
    }

    state.localizerSource = 'contour-track';
    const now = performance.now();
    if (state.fineGl && typeof app.renderDeskew === 'function') {
      app.renderDeskew(state.fineGl, dom.dskCvs, state.lastCorners, 1.0, dom.video, config.FINE_RENDER_SIZE);
      state.lastDeskewTime = now;
      dom.dskCvs.style.opacity = '1';
      if (typeof app.measureBlurScore === 'function') {
        state.fineBlurScore = app.measureBlurScore(dom.dskCvs, { marginRatio: 0.08 });
      }
      if (typeof app.updateContourHashGate === 'function') {
        app.updateContourHashGate(dom.dskCvs);
      }
    }
    if (typeof app.recordDeskewFrame === 'function') {
      app.recordDeskewFrame(now);
    }
    if (typeof app.enqueueRecognizeFrame === 'function') {
      app.enqueueRecognizeFrame();
    }
    if (typeof app.refreshPerfBar === 'function') {
      app.refreshPerfBar();
    }
  };
})(window);
