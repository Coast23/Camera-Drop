'use strict';

(function initContourLocalizerModule(global) {
  const app = global.CameraDropApp;
  const state = app.state;
  const dom = app.dom;
  const config = app.config;

  const TYPE_NONE = 0;
  const TYPE_NORMAL = 1;
  const TYPE_BR = 2;

  const CONTOUR_MAX_SIDE = 1280;
  const ROI_NORM_SIZE = 64;
  const MIN_ASPECT = 0.72;

  const CANON_SIZE = 1024;
  const ANCHOR_OUT = 2;
  const ANCHOR_SIZE = 56;
  const ANCHOR_CENTER = ANCHOR_OUT + (ANCHOR_SIZE * 0.5);
  const ANCHOR_SPAN = CANON_SIZE - (ANCHOR_CENTER * 2);

  const NORMAL_SEQUENCES = [
    { name: 'full', layers: [56, 42, 28, 14], outerScale: 1.0 },
    { name: 'inner', layers: [42, 28, 14], outerScale: 56 / 42 },
  ];

  const BR_COLOR_ORDER = ['Y', 'G', 'M', 'C'];
  const BR_OUTER_SAMPLES = [
    { dx: -21, dy: -21, color: 'Y' },
    { dx: 21, dy: -21, color: 'G' },
    { dx: -21, dy: 21, color: 'M' },
    { dx: 21, dy: 21, color: 'C' },
  ];
  const BR_INNER_SAMPLES = [
    { dx: -10.5, dy: -10.5, color: 'Y' },
    { dx: 10.5, dy: -10.5, color: 'G' },
    { dx: -10.5, dy: 10.5, color: 'M' },
    { dx: 10.5, dy: 10.5, color: 'C' },
  ];
  const BR_BLACK_SAMPLES = [
    { dx: 0, dy: 0 },
    { dx: 0, dy: -17.5 },
    { dx: 17.5, dy: 0 },
    { dx: 0, dy: 17.5 },
    { dx: -17.5, dy: 0 },
  ];
  const BR_SEARCH_FACTORS = [0.94, 1.0, 1.06];

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function ensureContourCanvas(width, height) {
    if (!state.contourCanvas || state.contourCanvas.width !== width || state.contourCanvas.height !== height) {
      state.contourCanvas = new OffscreenCanvas(width, height);
      state.contourCtx = state.contourCanvas.getContext('2d', { willReadFrequently: true });
    }
    return state.contourCanvas;
  }

  function getScaleForVideo() {
    const vw = dom.video.videoWidth || dom.video.width || 0;
    const vh = dom.video.videoHeight || dom.video.height || 0;
    const maxSide = Math.max(vw, vh);
    if (!maxSide) {
      return 1;
    }
    return Math.min(1, CONTOUR_MAX_SIDE / maxSide);
  }

  function getScaleForSize(width, height) {
    const maxSide = Math.max(width || 0, height || 0);
    if (!maxSide) {
      return 1;
    }
    return Math.min(1, CONTOUR_MAX_SIDE / maxSide);
  }

  function offsetDetectedCorners(corners, dx, dy) {
    if (!corners) {
      return null;
    }
    return {
      TL: [corners.TL[0] + dx, corners.TL[1] + dy],
      TR: [corners.TR[0] + dx, corners.TR[1] + dy],
      BL: [corners.BL[0] + dx, corners.BL[1] + dy],
      BR: [corners.BR[0] + dx, corners.BR[1] + dy],
      outSize: corners.outSize,
      assignmentMode: corners.assignmentMode,
    };
  }

  function normalizeContourRoi(rect, srcW, srcH) {
    if (!rect || !srcW || !srcH) {
      return null;
    }
    const x0 = clamp(Math.floor(rect.x), 0, srcW - 1);
    const y0 = clamp(Math.floor(rect.y), 0, srcH - 1);
    const x1 = clamp(Math.ceil(rect.x + rect.w), x0 + 1, srcW);
    const y1 = clamp(Math.ceil(rect.y + rect.h), y0 + 1, srcH);
    return {
      x: x0,
      y: y0,
      w: Math.max(1, x1 - x0),
      h: Math.max(1, y1 - y0),
    };
  }

  function sortQuad(points) {
    let tl = points[0];
    let tr = points[0];
    let br = points[0];
    let bl = points[0];
    let minSum = Infinity;
    let maxSum = -Infinity;
    let minDiff = Infinity;
    let maxDiff = -Infinity;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const sum = p.x + p.y;
      const diff = p.y - p.x;
      if (sum < minSum) {
        minSum = sum;
        tl = p;
      }
      if (sum > maxSum) {
        maxSum = sum;
        br = p;
      }
      if (diff < minDiff) {
        minDiff = diff;
        tr = p;
      }
      if (diff > maxDiff) {
        maxDiff = diff;
        bl = p;
      }
    }
    return [tl, tr, br, bl];
  }

  function makeRotatedRect(center, width, height, angle) {
    return new cv.RotatedRect(
      new cv.Point(center.x, center.y),
      new cv.Size(width, height),
      angle
    );
  }

  function scaleRotatedRect(rect, scale) {
    return makeRotatedRect(rect.center, rect.size.width * scale, rect.size.height * scale, rect.angle);
  }

  function warpRotatedRect(src, rect, expandFactor) {
    const scaled = expandFactor && expandFactor !== 1 ? scaleRotatedRect(rect, expandFactor) : rect;
    const corners = cv.RotatedRect.points(scaled);
    const ordered = sortQuad(corners);
    const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
      ordered[0].x, ordered[0].y,
      ordered[1].x, ordered[1].y,
      ordered[2].x, ordered[2].y,
      ordered[3].x, ordered[3].y,
    ]);
    const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      ROI_NORM_SIZE, 0,
      ROI_NORM_SIZE, ROI_NORM_SIZE,
      0, ROI_NORM_SIZE,
    ]);
    const transform = cv.getPerspectiveTransform(srcPts, dstPts);
    const norm = new cv.Mat();
    cv.warpPerspective(src, norm, transform, new cv.Size(ROI_NORM_SIZE, ROI_NORM_SIZE), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
    srcPts.delete();
    dstPts.delete();
    transform.delete();
    return norm;
  }

  function scoreNormalAnchorRoi(roi) {
    const center = (ROI_NORM_SIZE - 1) * 0.5;
    const sums = [0, 0, 0, 0];
    const counts = [0, 0, 0, 0];
    const data = roi.data;
    for (let y = 0; y < roi.rows; y++) {
      const rowOff = y * roi.cols;
      for (let x = 0; x < roi.cols; x++) {
        const white = data[rowOff + x] / 255;
        const dist = Math.max(Math.abs((x + 0.5) - center), Math.abs((y + 0.5) - center));
        let band = 0;
        if (dist >= 24) {
          band = 0;
        } else if (dist >= 16) {
          band = 1;
        } else if (dist >= 8) {
          band = 2;
        } else {
          band = 3;
        }
        sums[band] += white;
        counts[band]++;
      }
    }
    const outerWhite = sums[0] / Math.max(1, counts[0]);
    const ringBlack = sums[1] / Math.max(1, counts[1]);
    const innerWhite = sums[2] / Math.max(1, counts[2]);
    const centerBlack = sums[3] / Math.max(1, counts[3]);
    return {
      outerWhite,
      ringBlack,
      innerWhite,
      centerBlack,
      score: outerWhite + innerWhite + (1 - ringBlack) + (1 - centerBlack),
    };
  }

  function getContourInfo(contours, index) {
    const contour = contours.get(index);
    try {
      const rect = cv.minAreaRect(contour);
      const sideMin = Math.min(rect.size.width, rect.size.height);
      const sideMax = Math.max(rect.size.width, rect.size.height);
      return {
        index,
        rect,
        center: { x: rect.center.x, y: rect.center.y },
        size: (rect.size.width + rect.size.height) * 0.5,
        sideMin,
        sideMax,
        aspect: sideMin / Math.max(1e-6, sideMax),
      };
    } finally {
      contour.delete();
    }
  }

  function getSiblingCandidates(hierarchy, start) {
    const out = [];
    let index = start;
    while (index >= 0) {
      out.push(index);
      const ptr = hierarchy.intPtr(0, index);
      index = ptr ? ptr[0] : -1;
    }
    return out;
  }

  function findBestNestedChild(index, infos, hierarchy) {
    const parent = infos[index];
    const ptr = hierarchy.intPtr(0, index);
    if (!parent || !ptr || ptr[2] < 0) {
      return -1;
    }
    const siblings = getSiblingCandidates(hierarchy, ptr[2]);
    let best = -1;
    let bestScore = Infinity;
    for (let i = 0; i < siblings.length; i++) {
      const childIndex = siblings[i];
      const child = infos[childIndex];
      if (!child || child.aspect < MIN_ASPECT || child.size >= parent.size * 0.96) {
        continue;
      }
      const drift = Math.hypot(child.center.x - parent.center.x, child.center.y - parent.center.y) / Math.max(1, parent.size);
      if (drift > 0.16) {
        continue;
      }
      const ratio = child.size / Math.max(1, parent.size);
      const score = drift * 4 + Math.abs(ratio - 0.68);
      if (score < bestScore) {
        bestScore = score;
        best = childIndex;
      }
    }
    return best;
  }

  function buildNestedChain(index, infos, hierarchy, limit) {
    const chain = [];
    let current = index;
    const maxDepth = limit || 4;
    while (current >= 0 && chain.length < maxDepth) {
      const info = infos[current];
      if (!info) {
        break;
      }
      chain.push(info);
      current = findBestNestedChild(current, infos, hierarchy);
    }
    return chain;
  }

  function fitNormalChain(chain) {
    if (!chain || chain.length < 3) {
      return null;
    }
let best = null;
    for (let s = 0; s < NORMAL_SEQUENCES.length; s++) {
      const spec = NORMAL_SEQUENCES[s];
      if (chain.length < spec.layers.length) {
        continue;
      }
      const obs = chain.slice(0, spec.layers.length);
      let scale = 0;
      for (let i = 0; i < spec.layers.length; i++) {
        scale += obs[i].size / spec.layers[i];
      }
      scale /= spec.layers.length;
      let sizeErr = 0;
      let driftErr = 0;
      let aspectErr = 0;
      for (let i = 0; i < obs.length; i++) {
        const expected = spec.layers[i] * scale;
        sizeErr += Math.abs(obs[i].size - expected) / Math.max(1, expected);
        driftErr = Math.max(
          driftErr,
          Math.hypot(obs[i].center.x - obs[0].center.x, obs[i].center.y - obs[0].center.y) / Math.max(1, scale * ANCHOR_SIZE)
        );
        aspectErr = Math.max(aspectErr, Math.abs(1 - obs[i].aspect));
      }
      const score = (sizeErr / obs.length) + driftErr * 1.6 + aspectErr * 0.5;
      if (!best || score < best.score) {
        best = {
          name: spec.name,
          layers: spec.layers.slice(),
          outerScale: spec.outerScale,
          scale,
          sizeErr: sizeErr / obs.length,
          driftErr,
          aspectErr,
          score,
        };
      }
    }
    if (!best) {
      return null;
    }
    if (best.sizeErr > 0.095 || best.driftErr > 0.12 || best.aspectErr > 0.2) {
      return null;
    }
    return best;
  }

  function deduplicateCandidates(candidates) {
    const out = [];
    const used = new Array(candidates.length).fill(false);
    for (let i = 0; i < candidates.length; i++) {
      if (used[i]) {
        continue;
      }
      let best = candidates[i];
      used[i] = true;
      for (let j = i + 1; j < candidates.length; j++) {
        if (used[j]) {
          continue;
        }
        const dx = candidates[i].center.x - candidates[j].center.x;
        const dy = candidates[i].center.y - candidates[j].center.y;
        const limit = Math.min(candidates[i].outerSize, candidates[j].outerSize) * 0.45;
        if (Math.hypot(dx, dy) < limit) {
          if (candidates[j].rank > best.rank) {
            best = candidates[j];
          }
          used[j] = true;
        }
      }
      out.push(best);
    }
    return out;
  }

  function orderNormalTriple(items) {
    const c0 = items[0].center;
    const c1 = items[1].center;
    const c2 = items[2].center;
    const edges = [
      { x: c1.x - c2.x, y: c1.y - c2.y },
      { x: c2.x - c0.x, y: c2.y - c0.y },
      { x: c0.x - c1.x, y: c0.y - c1.y },
    ];
    let topLeft = 0;
    let maxD = -1;
    for (let i = 0; i < edges.length; i++) {
      const d = edges[i].x * edges[i].x + edges[i].y * edges[i].y;
      if (d > maxD) {
        maxD = d;
        topLeft = i;
      }
    }
    const fix = (i) => {
      if (i < 0) return 2;
      if (i >= 3) return 0;
      return i;
    };
    const dep = edges[fix(topLeft - 1)];
    const inc = edges[fix(topLeft + 1)];
    const rot = { x: -inc.y, y: inc.x };
    const overlap = { x: dep.x - rot.x, y: dep.y - rot.y };
    const depD = dep.x * dep.x + dep.y * dep.y;
    const ovD = overlap.x * overlap.x + overlap.y * overlap.y;
    const topRight = ovD < depD ? fix(topLeft + 1) : fix(topLeft - 1);
    const bottomLeft = ovD < depD ? fix(topLeft - 1) : fix(topLeft + 1);
    return [items[topLeft], items[topRight], items[bottomLeft]];
  }

  function validateNormalTriple(tl, tr, bl) {
    const ux = tr.center.x - tl.center.x;
    const uy = tr.center.y - tl.center.y;
    const vx = bl.center.x - tl.center.x;
    const vy = bl.center.y - tl.center.y;
    const du = Math.hypot(ux, uy);
    const dv = Math.hypot(vx, vy);
    const avgSize = (tl.outerSize + tr.outerSize + bl.outerSize) / 3;
    if (!Number.isFinite(du) || !Number.isFinite(dv) || du < avgSize * 6 || dv < avgSize * 6) {
      return null;
    }
    const ratio = du / Math.max(1, dv);
    if (ratio < 0.45 || ratio > 2.2) {
      return null;
    }
    const cos = (ux * vx + uy * vy) / Math.max(1e-6, du * dv);
    if (Math.abs(cos) > 0.42) {
      return null;
    }
    return {
      du,
      dv,
      ratio,
      cos,
      score: (du + dv) / Math.max(1, avgSize),
    };
  }

  function addPointScaled(origin, u, du, v, dv) {
    return {
      x: origin.x + u.x * du + v.x * dv,
      y: origin.y + u.y * du + v.y * dv,
    };
  }

  function sampleRgbNearest(data, width, height, x, y) {
    const xi = clamp(Math.round(x), 0, width - 1);
    const yi = clamp(Math.round(y), 0, height - 1);
    const idx = (yi * width + xi) * 4;
    return [data[idx], data[idx + 1], data[idx + 2]];
  }

  function samplePatchRgb(data, width, height, center, u, v, dx, dy, patchScale) {
    const offsets = [
      [0, 0],
      [-patchScale, -patchScale],
      [patchScale, -patchScale],
      [-patchScale, patchScale],
      [patchScale, patchScale],
    ];
    let sr = 0;
    let sg = 0;
    let sb = 0;
    for (let i = 0; i < offsets.length; i++) {
      const p = addPointScaled(center, u, dx + offsets[i][0], v, dy + offsets[i][1]);
      const rgb = sampleRgbNearest(data, width, height, p.x, p.y);
      sr += rgb[0];
      sg += rgb[1];
      sb += rgb[2];
    }
    const inv = 1 / offsets.length;
    return [sr * inv, sg * inv, sb * inv];
  }

  function colorPatchScore(rgb, label) {
    const r = rgb[0];
    const g = rgb[1];
    const b = rgb[2];
    const hi = label === 'Y'
      ? (r + g) * 0.5
      : label === 'G'
        ? g
        : label === 'M'
          ? (r + b) * 0.5
          : (g + b) * 0.5;
    const lo = label === 'Y'
      ? b
      : label === 'G'
        ? (r + b) * 0.5
        : label === 'M'
          ? g
          : r;
    const sat = Math.max(r, g, b) - Math.min(r, g, b);
    return clamp((hi - lo + sat * 0.4) / 255, 0, 1.4);
  }

  function blackPatchScore(rgb) {
    const r = rgb[0];
    const g = rgb[1];
    const b = rgb[2];
    const lum = (r + g + b) / 3;
    const sat = Math.max(r, g, b) - Math.min(r, g, b);
    return clamp((255 - lum - sat * 0.1) / 255, 0, 1.2);
  }

  function scoreBottomRightAt(data, width, height, center, u, v, factor) {
    let outerScore = 0;
    let innerScore = 0;
    let blackScore = 0;
    const patchScale = 1.2 * factor;
    for (let i = 0; i < BR_OUTER_SAMPLES.length; i++) {
      const item = BR_OUTER_SAMPLES[i];
      const rgb = samplePatchRgb(data, width, height, center, u, v, item.dx * factor, item.dy * factor, patchScale);
      outerScore += colorPatchScore(rgb, item.color);
    }
    for (let i = 0; i < BR_INNER_SAMPLES.length; i++) {
      const item = BR_INNER_SAMPLES[i];
      const rgb = samplePatchRgb(data, width, height, center, u, v, item.dx * factor, item.dy * factor, patchScale * 0.8);
      innerScore += colorPatchScore(rgb, item.color);
    }
    for (let i = 0; i < BR_BLACK_SAMPLES.length; i++) {
      const item = BR_BLACK_SAMPLES[i];
      const rgb = samplePatchRgb(data, width, height, center, u, v, item.dx * factor, item.dy * factor, patchScale * 0.7);
      blackScore += blackPatchScore(rgb);
    }
    return {
      outerScore,
      innerScore,
      blackScore,
      total: outerScore * 1.0 + innerScore * 0.9 + blackScore * 0.8,
    };
  }

  function axesFromOuterRect(rect) {
    const pts = sortQuad(cv.RotatedRect.points(rect));
    return {
      u: {
        x: (pts[1].x - pts[0].x) / ANCHOR_SIZE,
        y: (pts[1].y - pts[0].y) / ANCHOR_SIZE,
      },
      v: {
        x: (pts[3].x - pts[0].x) / ANCHOR_SIZE,
        y: (pts[3].y - pts[0].y) / ANCHOR_SIZE,
      },
    };
  }

  function scoreBottomRightRect(data, width, height, candidate) {
    const axes = axesFromOuterRect(candidate.rect);
    const score = scoreBottomRightAt(data, width, height, candidate.center, axes.u, axes.v, 1.0);
    return {
      total: score.total,
      outerScore: score.outerScore,
      innerScore: score.innerScore,
      blackScore: score.blackScore,
    };
  }

  function searchBottomRightCandidate(shapeCandidates, predicted, avgOuterSize, data, width, height, debug) {
    if (!shapeCandidates || !shapeCandidates.length) {
      return null;
    }
let best = null;
    for (let i = 0; i < shapeCandidates.length; i++) {
      const candidate = shapeCandidates[i];
      const dist = Math.hypot(candidate.center.x - predicted.x, candidate.center.y - predicted.y);
      if (dist > avgOuterSize * 2.6) {
        continue;
      }
      const sizeRatio = candidate.outerSize / Math.max(1, avgOuterSize);
      if (sizeRatio < 0.6 || sizeRatio > 1.45) {
        continue;
      }
      const score = scoreBottomRightRect(data, width, height, candidate);
      const total = score.total - dist / Math.max(1, avgOuterSize) * 0.25;
      if (!best || total > best.total) {
        best = {
          candidate,
          total,
          raw: score,
          dist,
        };
      }
    }
    if (!best) {
      return null;
    }
    if (debug) {
      debug.candidate = {
        x: Number(best.candidate.center.x.toFixed(2)),
        y: Number(best.candidate.center.y.toFixed(2)),
        size: Number(best.candidate.outerSize.toFixed(2)),
        dist: Number(best.dist.toFixed(2)),
        total: Number(best.total.toFixed(3)),
        outer: Number(best.raw.outerScore.toFixed(3)),
        inner: Number(best.raw.innerScore.toFixed(3)),
        black: Number(best.raw.blackScore.toFixed(3)),
      };
    }
    if (best.raw.outerScore < 1.7 || best.raw.innerScore < 1.6 || best.raw.blackScore < 1.9 || best.total < 5.3) {
      return null;
    }
    return {
      center: best.candidate.center,
      outerSize: best.candidate.outerSize,
      rect: best.candidate.rect,
      type: TYPE_BR,
      rank: best.total,
      score: best.total,
    };
  }
  function searchBottomRightAnchor(imageData, width, height, tl, tr, bl, shapeCandidates, debug) {
    const u = {
      x: (tr.center.x - tl.center.x) / ANCHOR_SPAN,
      y: (tr.center.y - tl.center.y) / ANCHOR_SPAN,
    };
    const v = {
      x: (bl.center.x - tl.center.x) / ANCHOR_SPAN,
      y: (bl.center.y - tl.center.y) / ANCHOR_SPAN,
    };
    const uLen = Math.hypot(u.x, u.y);
    const vLen = Math.hypot(v.x, v.y);
    if (uLen < 0.01 || vLen < 0.01) {
      return null;
    }
    const predicted = addPointScaled(tl.center, u, ANCHOR_SPAN, v, ANCHOR_SPAN);
    if (debug) {
      debug.predicted = {
        x: Number(predicted.x.toFixed(2)),
        y: Number(predicted.y.toFixed(2)),
      };
      debug.uLen = Number(uLen.toFixed(4));
      debug.vLen = Number(vLen.toFixed(4));
    }
    const avgOuterSize = (tl.outerSize + tr.outerSize + bl.outerSize) / 3;
    const candidateBr = searchBottomRightCandidate(shapeCandidates, predicted, avgOuterSize, imageData.data, width, height, debug);
    if (candidateBr) {
      return candidateBr;
    }

    let best = null;
    let bestDU = 0;
    let bestDV = 0;
    let bestFactor = 1;
    for (let fi = 0; fi < BR_SEARCH_FACTORS.length; fi++) {
      const factor = BR_SEARCH_FACTORS[fi];
      for (let du = -24; du <= 24; du += 4) {
        for (let dv = -24; dv <= 24; dv += 4) {
          const center = addPointScaled(predicted, u, du, v, dv);
          const score = scoreBottomRightAt(imageData.data, width, height, center, u, v, factor);
          if (!best || score.total > best.total) {
            best = { center, total: score.total, outerScore: score.outerScore, innerScore: score.innerScore, blackScore: score.blackScore };
            bestDU = du;
            bestDV = dv;
            bestFactor = factor;
          }
        }
      }
    }
    if (!best) {
      return null;
    }
    for (let factor = bestFactor - 0.04; factor <= bestFactor + 0.04; factor += 0.02) {
      for (let du = bestDU - 4; du <= bestDU + 4; du += 1) {
        for (let dv = bestDV - 4; dv <= bestDV + 4; dv += 1) {
          const center = addPointScaled(predicted, u, du, v, dv);
          const score = scoreBottomRightAt(imageData.data, width, height, center, u, v, factor);
          if (score.total > best.total) {
            best = { center, total: score.total, outerScore: score.outerScore, innerScore: score.innerScore, blackScore: score.blackScore };
            bestDU = du;
            bestDV = dv;
            bestFactor = factor;
          }
        }
      }
    }
    if (debug) {
      debug.bestOffset = { du: bestDU, dv: bestDV, factor: Number(bestFactor.toFixed(3)) };
      debug.bestScore = {
        total: Number(best.total.toFixed(3)),
        outer: Number(best.outerScore.toFixed(3)),
        inner: Number(best.innerScore.toFixed(3)),
        black: Number(best.blackScore.toFixed(3)),
      };
      debug.center = {
        x: Number(best.center.x.toFixed(2)),
        y: Number(best.center.y.toFixed(2)),
      };
    }
    if (best.outerScore < 1.8 || best.innerScore < 1.7 || best.blackScore < 2.0 || best.total < 5.8) {
      return null;
    }
    return {
      center: best.center,
      outerSize: avgOuterSize,
      type: TYPE_BR,
      rank: best.total,
      score: best.total,
    };
  }

  function solveHomography(srcPts, dstPts) {
    const A = new Float64Array(64);
    const b = new Float64Array(8);
    for (let i = 0; i < 4; i++) {
      const x = srcPts[i][0];
      const y = srcPts[i][1];
      const u = dstPts[i][0];
      const v = dstPts[i][1];
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
      let pivot = c;
      for (let r = c + 1; r < n; r++) {
        if (Math.abs(M[r * (n + 1) + c]) > Math.abs(M[pivot * (n + 1) + c])) {
          pivot = r;
        }
      }
      if (Math.abs(M[pivot * (n + 1) + c]) < 1e-8) {
        return null;
      }
      for (let j = 0; j <= n; j++) {
        const tmp = M[c * (n + 1) + j];
        M[c * (n + 1) + j] = M[pivot * (n + 1) + j];
        M[pivot * (n + 1) + j] = tmp;
      }
      for (let r = c + 1; r < n; r++) {
        const factor = M[r * (n + 1) + c] / M[c * (n + 1) + c];
        for (let j = c; j <= n; j++) {
          M[r * (n + 1) + j] -= factor * M[c * (n + 1) + j];
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
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  }

  function projectPoint(H, x, y) {
    const z = H[6] * x + H[7] * y + H[8];
    if (Math.abs(z) < 1e-8) {
      return null;
    }
    return [
      (H[0] * x + H[1] * y + H[2]) / z,
      (H[3] * x + H[4] * y + H[5]) / z,
    ];
  }

  function buildCornersFromCenters(centers, invScale) {
    const src = [
      [ANCHOR_CENTER, ANCHOR_CENTER],
      [CANON_SIZE - ANCHOR_CENTER, ANCHOR_CENTER],
      [ANCHOR_CENTER, CANON_SIZE - ANCHOR_CENTER],
      [CANON_SIZE - ANCHOR_CENTER, CANON_SIZE - ANCHOR_CENTER],
    ];
    const dst = centers.map((item) => [item.x, item.y]);
    const H = solveHomography(src, dst);
    if (!H) {
      return null;
    }
    const tl = projectPoint(H, ANCHOR_OUT, ANCHOR_OUT);
    const tr = projectPoint(H, CANON_SIZE - ANCHOR_OUT, ANCHOR_OUT);
    const bl = projectPoint(H, ANCHOR_OUT, CANON_SIZE - ANCHOR_OUT);
    const br = projectPoint(H, CANON_SIZE - ANCHOR_OUT, CANON_SIZE - ANCHOR_OUT);
    if (!tl || !tr || !bl || !br) {
      return null;
    }
    const TL = [tl[0] * invScale, tl[1] * invScale];
    const TR = [tr[0] * invScale, tr[1] * invScale];
    const BL = [bl[0] * invScale, bl[1] * invScale];
    const BR = [br[0] * invScale, br[1] * invScale];
    const outSize = Math.round(Math.max(
      Math.hypot(TR[0] - TL[0], TR[1] - TL[1]),
      Math.hypot(BR[0] - BL[0], BR[1] - BL[1]),
      Math.hypot(BL[0] - TL[0], BL[1] - TL[1]),
      Math.hypot(BR[0] - TR[0], BR[1] - TR[1])
    ));
    return { TL, TR, BL, BR, outSize };
  }

  app.buildContourRoiFromCorners = function buildContourRoiFromCorners(corners) {
    if (!corners) {
      return null;
    }
    const pts = [corners.TL, corners.TR, corners.BL, corners.BR];
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const minX = Math.min.apply(null, xs);
    const maxX = Math.max.apply(null, xs);
    const minY = Math.min.apply(null, ys);
    const maxY = Math.max.apply(null, ys);
    const span = Math.max(maxX - minX, maxY - minY, Number(config.CONTOUR_ROI_MIN) || 0);
    const pad = Math.max(48, span * Math.max(0, Number(config.CONTOUR_ROI_MARGIN) || 0));
    return {
      x: minX - pad,
      y: minY - pad,
      w: (maxX - minX) + pad * 2,
      h: (maxY - minY) + pad * 2,
    };
  };

  app.updateContourHashGate = function updateContourHashGate(source) {
    if (!source || typeof app.computeAHashFromSource !== 'function' || typeof app.hammingDist !== 'function') {
      return false;
    }
    const nextHash = app.computeAHashFromSource(source, 0.08);
    const thresh = Math.max(0, Number(config.CONTOUR_AHASH_THRESH));
    const changed = state.contourTrackHash !== null && app.hammingDist(nextHash, state.contourTrackHash) > thresh;
    state.contourTrackHash = nextHash;
    if (changed) {
      state.contourNeedRefine = true;
    }
    return changed;
  };

  function chooseContourCorners(normals, shapeCandidates, imageData, width, height, invScale, debug) {
    if (!normals || normals.length < 3) {
      return null;
    }
    const pool = normals.slice().sort((a, b) => b.rank - a.rank).slice(0, Math.min(10, normals.length));
let best = null;
    for (let i = 0; i < pool.length; i++) {
      for (let j = i + 1; j < pool.length; j++) {
        for (let k = j + 1; k < pool.length; k++) {
          const ordered = orderNormalTriple([pool[i], pool[j], pool[k]]);
          const tl = ordered[0];
          const tr = ordered[1];
          const bl = ordered[2];
          const geom = validateNormalTriple(tl, tr, bl);
          if (!geom) {
            continue;
          }
          const brDebug = debug && debug.bestSet == null ? {} : null;
          const br = searchBottomRightAnchor(imageData, width, height, tl, tr, bl, shapeCandidates, brDebug);
          if (!br) {
            continue;
          }
          const corners = buildCornersFromCenters([tl.center, tr.center, bl.center, br.center], invScale);
          if (!corners) {
            continue;
          }
          const total = tl.rank + tr.rank + bl.rank + br.rank * 1.8 + geom.score * 0.04;
          if (!best || total > best.total) {
            best = {
              total,
              tl,
              tr,
              bl,
              br,
              corners,
              geom,
              brDebug,
            };
          }
        }
      }
    }
    if (!best) {
      return null;
    }
    if (debug) {
      debug.bestSet = {
        total: Number(best.total.toFixed(3)),
        geom: {
          du: Number(best.geom.du.toFixed(2)),
          dv: Number(best.geom.dv.toFixed(2)),
          ratio: Number(best.geom.ratio.toFixed(3)),
          cos: Number(best.geom.cos.toFixed(3)),
        },
        tl: { x: Number(best.tl.center.x.toFixed(2)), y: Number(best.tl.center.y.toFixed(2)), size: Number(best.tl.outerSize.toFixed(2)), rank: Number(best.tl.rank.toFixed(3)) },
        tr: { x: Number(best.tr.center.x.toFixed(2)), y: Number(best.tr.center.y.toFixed(2)), size: Number(best.tr.outerSize.toFixed(2)), rank: Number(best.tr.rank.toFixed(3)) },
        bl: { x: Number(best.bl.center.x.toFixed(2)), y: Number(best.bl.center.y.toFixed(2)), size: Number(best.bl.outerSize.toFixed(2)), rank: Number(best.bl.rank.toFixed(3)) },
        br: { x: Number(best.br.center.x.toFixed(2)), y: Number(best.br.center.y.toFixed(2)), size: Number(best.br.outerSize.toFixed(2)), rank: Number(best.br.rank.toFixed(3)) },
        brDebug: best.brDebug,
      };
    }
    best.corners.assignmentMode = 'anchor-centers';
    return best.corners;
  }

  app.detectContourCorners = function detectContourCorners(options) {
    if (typeof cv === 'undefined' || !cv || dom.video.readyState < 2) {
      return null;
    }
    const opts = options || {};
    const srcW = dom.video.videoWidth || dom.video.width;
    const srcH = dom.video.videoHeight || dom.video.height;
    const roi = normalizeContourRoi(opts.roiSrcRect, srcW, srcH);
    const roiX = roi ? roi.x : 0;
    const roiY = roi ? roi.y : 0;
    const roiW = roi ? roi.w : srcW;
    const roiH = roi ? roi.h : srcH;
    const scale = roi ? getScaleForSize(roiW, roiH) : getScaleForVideo();
    const width = Math.max(64, Math.round(roiW * scale));
    const height = Math.max(64, Math.round(roiH * scale));
    ensureContourCanvas(width, height);
    state.contourCtx.drawImage(dom.video, roiX, roiY, roiW, roiH, 0, 0, width, height);
    const imageData = state.contourCtx.getImageData(0, 0, width, height);
    const rgba = cv.matFromImageData(imageData);
    const gray = new cv.Mat();
    const binary = new cv.Mat();
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    try {
      cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
      let blockSize = Math.max(3, Math.round(Math.min(width, height) / 20));
      if ((blockSize & 1) === 0) {
        blockSize++;
      }
      cv.adaptiveThreshold(gray, binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, 5);
      cv.findContours(binary, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);

      const whiteRatio = cv.countNonZero(binary) / Math.max(1, width * height);
      const minAnchor = Math.max(10, Math.round(Math.min(width, height) * 0.01));
      const maxAnchor = Math.max(minAnchor + 4, Math.round(Math.min(width, height) * 0.12));
      const infos = new Array(contours.size());
      for (let i = 0; i < contours.size(); i++) {
        infos[i] = getContourInfo(contours, i);
      }
      const shapeCandidates = [];
      const normalCandidates = [];
      const samples = [];
      let childContours = 0;
      let sizePassCount = 0;
      let fitPassCount = 0;
      let roiPassCount = 0;

      for (let i = 0; i < contours.size(); i++) {
        const h = hierarchy.intPtr(0, i);
        if (!h || h[2] < 0) {
          continue;
        }
        childContours++;
        const info = infos[i];
        if (info.aspect < MIN_ASPECT || info.sideMin < minAnchor || info.sideMax > maxAnchor) {
          continue;
        }
        sizePassCount++;
        const chain = buildNestedChain(i, infos, hierarchy, 4);
        const fit = fitNormalChain(chain);
        if (!fit) {
          if (samples.length < 20) {
            samples.push({
              cx: Number(info.center.x.toFixed(2)),
              cy: Number(info.center.y.toFixed(2)),
              size: Number(info.size.toFixed(2)),
              aspect: Number(info.aspect.toFixed(3)),
              chain: chain.map((item) => Number(item.size.toFixed(2))),
              fit: null,
            });
          }
          continue;
        }
        fitPassCount++;
        const outerRect = scaleRotatedRect(info.rect, fit.outerScale);
        const roi = warpRotatedRect(binary, outerRect, 1.0);
        const template = scoreNormalAnchorRoi(roi);
        roi.delete();
        const rank = template.score - fit.score * 3 + (chain.length >= 4 ? 0.18 : 0);
        shapeCandidates.push({
          type: TYPE_NONE,
          center: { x: info.center.x, y: info.center.y },
          outerSize: (info.size * fit.outerScale),
          rank,
          fit,
          template,
          chainLength: chain.length,
          rect: outerRect,
        });
        if (samples.length < 20) {
          samples.push({
            cx: Number(info.center.x.toFixed(2)),
            cy: Number(info.center.y.toFixed(2)),
            size: Number(info.size.toFixed(2)),
            aspect: Number(info.aspect.toFixed(3)),
            chain: chain.map((item) => Number(item.size.toFixed(2))),
            fit: {
              name: fit.name,
              sizeErr: Number(fit.sizeErr.toFixed(3)),
              driftErr: Number(fit.driftErr.toFixed(3)),
              score: Number(fit.score.toFixed(3)),
            },
            roi: {
              outerWhite: Number(template.outerWhite.toFixed(3)),
              ringBlack: Number(template.ringBlack.toFixed(3)),
              innerWhite: Number(template.innerWhite.toFixed(3)),
              centerBlack: Number(template.centerBlack.toFixed(3)),
              score: Number(template.score.toFixed(3)),
            },
          });
        }
        if (template.outerWhite < 0.68 || template.innerWhite < 0.62 || template.ringBlack > 0.36 || template.centerBlack > 0.24 || template.score < 3.25) {
          continue;
        }
        roiPassCount++;
        normalCandidates.push({
          type: TYPE_NORMAL,
          center: { x: info.center.x, y: info.center.y },
          outerSize: (info.size * fit.outerScale),
          rank,
          fit,
          template,
          chainLength: chain.length,
          rect: outerRect,
        });
      }

      const dedupedShapes = deduplicateCandidates(shapeCandidates).sort((a, b) => b.rank - a.rank);
      const dedupedNormals = deduplicateCandidates(normalCandidates).sort((a, b) => b.rank - a.rank);
      const chooserDebug = {};
      let corners = chooseContourCorners(dedupedNormals, dedupedShapes, imageData, width, height, 1 / scale, chooserDebug);
      if (corners && (roiX || roiY)) {
        corners = offsetDetectedCorners(corners, roiX, roiY);
      }
      state.contourDebug = {
        scale,
        srcW,
        srcH,
        width,
        height,
        scanMode: opts.scanMode || (roi ? 'roi' : 'full'),
        roi: roi ? { x: roiX, y: roiY, w: roiW, h: roiH } : null,
        blockSize,
        whiteRatio: Number(whiteRatio.toFixed(4)),
        contourCount: contours.size(),
        childContours,
        sizePassCount,
        fitPassCount,
        roiPassCount,
        minAnchor,
        maxAnchor,
        shapeCount: dedupedShapes.length,
        normalCount: dedupedNormals.length,
        normals: dedupedNormals.slice(0, 12).map((item) => ({
          x: Number(item.center.x.toFixed(2)),
          y: Number(item.center.y.toFixed(2)),
          size: Number(item.outerSize.toFixed(2)),
          rank: Number(item.rank.toFixed(3)),
          chainLength: item.chainLength,
          fit: item.fit.name,
          roi: Number(item.template.score.toFixed(3)),
        })),
        assignmentMode: corners && corners.assignmentMode ? corners.assignmentMode : null,
        bestSet: chooserDebug.bestSet || null,
        corners,
        samples,
      };
      return corners;
    } finally {
      hierarchy.delete();
      contours.delete();
      binary.delete();
      gray.delete();
      rgba.delete();
    }
  };

  app.shouldRunContourLocalizer = function shouldRunContourLocalizer(now) {
    if (typeof app.getLocalizerMode !== 'function' || app.getLocalizerMode() !== 'contour') {
      return false;
    }
    if (!state.scanning || dom.video.readyState < 2 || typeof cv === 'undefined' || !cv) {
      return false;
    }
    if (state.contourBusy) {
      return false;
    }
    if (!state.lastCorners || !state.patches) {
      return true;
    }
    if (state.contourNeedRefine) {
      return true;
    }
    const t = Number.isFinite(now) ? now : performance.now();
    const interval = Math.max(0, Number(config.CONTOUR_REDETECT_MS) || 0);
    return interval > 0 && (t - (state.lastContourRunAt || 0)) >= interval;
  };

  app.runContourLocalizer = function runContourLocalizer() {
    if (!app.shouldRunContourLocalizer()) {
      return false;
    }
    if (typeof app.claimVideoFrame === 'function' && !app.claimVideoFrame('lastContourVideoTime')) {
      return false;
    }
    state.contourBusy = true;
    try {
      const t0 = performance.now();
      let corners = null;
      let scanPath = 'full';
      const roiRect = state.lastCorners && typeof app.buildContourRoiFromCorners === 'function'
        ? app.buildContourRoiFromCorners(state.lastCorners)
        : null;
      if (roiRect) {
        corners = app.detectContourCorners({ roiSrcRect: roiRect, scanMode: 'roi' });
        scanPath = 'roi';
      }
      if (!corners) {
        corners = app.detectContourCorners({ scanMode: 'full' });
        scanPath = scanPath === 'roi' ? 'roi->full' : 'full';
      }
      const now = performance.now();
      state.lastContourRunAt = now;
      state.yoloMs = now - t0;
      state.yoloFpsArr.push(1000 / Math.max(1, now - state.yoloLastT));
      if (state.yoloFpsArr.length > 10) {
        state.yoloFpsArr.shift();
      }
      state.yoloLastT = now;
      state.localizerSource = scanPath;
      state.currentEP = 'contour';
      if (state.contourDebug) {
        state.contourDebug.scanPath = scanPath;
      }
      state.localizerDebug = state.contourDebug || null;
      if (!corners) {
        state.patches = null;
        state.lastAHash = null;
        state.contourTrackHash = null;
        state.contourNeedRefine = true;
        return false;
      }
      state.lastCorners = corners;
      if (typeof app.initPatches === 'function') {
        app.initPatches();
      }
      if (state.fineGl && typeof app.renderDeskew === 'function') {
        app.renderDeskew(state.fineGl, dom.dskCvs, corners, 1.0, dom.video, config.FINE_RENDER_SIZE);
        state.lastDeskewTime = now;
        dom.dskCvs.style.opacity = '1';
        if (typeof app.measureBlurScore === 'function') {
          state.fineBlurScore = app.measureBlurScore(dom.dskCvs, { marginRatio: 0.08 });
        }
        if (typeof app.updateContourHashGate === 'function') {
          app.updateContourHashGate(dom.dskCvs);
        }
      }
      state.contourNeedRefine = false;
      if (typeof app.recordDeskewFrame === 'function') {
        app.recordDeskewFrame(now);
      }
      if (typeof app.enqueueRecognizeFrame === 'function') {
        app.enqueueRecognizeFrame();
      }
      if (typeof app.refreshPerfBar === 'function') {
        app.refreshPerfBar();
      }
      return true;
    } catch (error) {
      console.warn('[ContourLocalizer]', error);
      return false;
    } finally {
      state.contourBusy = false;
    }
  };
})(window);