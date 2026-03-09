'use strict';

(function initCodebookModule(global) {
  const IMG_SIZE = 1024;
  const GRID_SIZE = 112;
  const STRIDE = 9;
  const MARGIN = 8;
  const TILE_SIZE = 8;
  const NUM_PATTERNS = 16;
  const NUM_COLORS = 4;
  const P_BITS = 4;
  const BENCH_FRAME_SET = 10;
  const DEFAULT_CODE_ASPECT = 1.0;
  const DEFAULT_SHORT_SIDE = IMG_SIZE;

  const COLORS = [
    [255, 255, 0],
    [0, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
  ];

  const ANCHOR_OUT_START = 2;
  const ANCHOR_L1_SIZE = 56;
  const ANCHOR_L2_INSET = 7;
  const ANCHOR_L2_SIZE = 42;
  const ANCHOR_L3_INSET = 14;
  const ANCHOR_L3_SIZE = 28;
  const ANCHOR_L4_INSET = 21;
  const ANCHOR_L4_SIZE = 14;

  const pop8 = new Uint8Array(256);
  for (let i = 1; i < 256; i++) pop8[i] = pop8[i >> 1] + (i & 1);

  const canonicalCanvas = document.createElement('canvas');
  canonicalCanvas.width = IMG_SIZE;
  canonicalCanvas.height = IMG_SIZE;
  const canonicalCtx = canonicalCanvas.getContext('2d');
  canonicalCtx.imageSmoothingEnabled = false;

  function popcnt32(v) {
    v >>>= 0;
    return pop8[v & 255] + pop8[(v >>> 8) & 255] + pop8[(v >>> 16) & 255] + pop8[(v >>> 24) & 255];
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

  function clampNumber(value, lo, hi, fallback) {
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    if (n < lo) return lo;
    if (n > hi) return hi;
    return n;
  }

  function clampInt(value, lo, hi, fallback) {
    return Math.round(clampNumber(value, lo, hi, fallback));
  }

  function normalizeRenderOptions(options) {
    const codeAspect = clampNumber(options && options.codeAspect, 0.25, 4.0, DEFAULT_CODE_ASPECT);
    const shortSide = clampInt(options && (options.shortSide || options.baseShortSide), 240, 4096, DEFAULT_SHORT_SIDE);
    return { codeAspect, shortSide };
  }

  function getCanvasDimensions(options) {
    const render = normalizeRenderOptions(options);
    if (render.codeAspect >= 1) {
      return {
        width: Math.max(1, Math.round(render.shortSide * render.codeAspect)),
        height: render.shortSide,
      };
    }
    return {
      width: render.shortSide,
      height: Math.max(1, Math.round(render.shortSide / render.codeAspect)),
    };
  }

  function expandMask16(mask16) {
    let lo = 0;
    let hi = 0;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        if (((mask16 >>> (r * 4 + c)) & 1) === 0) continue;
        const r8 = r << 1;
        const c8 = c << 1;
        const idx = [r8 * 8 + c8, r8 * 8 + c8 + 1, (r8 + 1) * 8 + c8, (r8 + 1) * 8 + c8 + 1];
        for (let k = 0; k < 4; k++) {
          const p = idx[k];
          if (p < 32) lo = (lo | ((1 << p) >>> 0)) >>> 0;
          else hi = (hi | ((1 << (p - 32)) >>> 0)) >>> 0;
        }
      }
    }
    return { lo, hi };
  }

  function genDict() {
    const cand = [];
    for (let i = 0; i < (1 << 16); i++) {
      const p = popcnt32(i);
      if (p >= 6 && p <= 10) cand.push(i);
    }

    const pick = [0x00FF];
    const dist = new Int16Array(cand.length);
    for (let i = 0; i < cand.length; i++) dist[i] = popcnt32(cand[i] ^ pick[0]);

    for (let k = 1; k < NUM_PATTERNS; k++) {
      let best = -1;
      let maxDist = -1;
      for (let i = 0; i < cand.length; i++) {
        if (dist[i] > maxDist) {
          maxDist = dist[i];
          best = i;
        }
      }
      const selected = cand[best];
      pick.push(selected);
      for (let i = 0; i < cand.length; i++) {
        const d = popcnt32(cand[i] ^ selected);
        if (d < dist[i]) dist[i] = d;
      }
    }

    const lo = new Uint32Array(NUM_PATTERNS);
    const hi = new Uint32Array(NUM_PATTERNS);
    for (let i = 0; i < NUM_PATTERNS; i++) {
      const m = expandMask16(pick[i]);
      lo[i] = m.lo;
      hi[i] = m.hi;
    }
    return { lo, hi, source: 'builtin-gen' };
  }

  function maskFrom8x8Grayscale(imageData) {
    let lo = 0;
    let hi = 0;
    const data = imageData.data;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const idx = (r * 8 + c) * 4;
        const gray = data[idx];
        if (gray >= 128) continue;
        const bit = r * 8 + c;
        if (bit < 32) lo = (lo | ((1 << bit) >>> 0)) >>> 0;
        else hi = (hi | ((1 << (bit - 32)) >>> 0)) >>> 0;
      }
    }
    return { lo, hi };
  }

  function loadPatternMask(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = 8;
        c.height = 8;
        const x = c.getContext('2d');
        x.imageSmoothingEnabled = false;
        x.drawImage(img, 0, 0, 8, 8);
        resolve(maskFrom8x8Grayscale(x.getImageData(0, 0, 8, 8)));
      };
      img.onerror = () => reject(new Error('failed to load ' + url));
      img.src = url + '?t=' + Date.now();
    });
  }

  async function loadPatternDirDict(baseUrl) {
    const lo = new Uint32Array(NUM_PATTERNS);
    const hi = new Uint32Array(NUM_PATTERNS);
    for (let i = 0; i < NUM_PATTERNS; i++) {
      const name = i.toString(16).padStart(2, '0');
      const mask = await loadPatternMask(baseUrl + '/' + name + '.png');
      lo[i] = mask.lo;
      hi[i] = mask.hi;
    }
    return { lo, hi, source: baseUrl };
  }

  async function loadDict(mode, baseUrl) {
    if (mode === 'builtin') return genDict();
    try {
      return await loadPatternDirDict(baseUrl || './best');
    } catch (_) {
      return genDict();
    }
  }

  function isMaskOn(maskLo, maskHi, bit) {
    if (bit < 32) return ((maskLo >>> bit) & 1) !== 0;
    return ((maskHi >>> (bit - 32)) & 1) !== 0;
  }

  function isAnchorReserved(r, c) {
    if (r < 6 && c < 6) return true;
    if (r < 6 && c > 105) return true;
    if (r > 105 && c < 6) return true;
    if (r > 105 && c > 105) return true;
    return false;
  }

  function isCalibrationCell(r, c) {
    return r === 0 && c >= 6 && c < 14;
  }

  function isHeaderCell(r, c) {
    return r === 0 && c >= 14 && c < 46;
  }

  function isPayloadCell(r, c) {
    if (isAnchorReserved(r, c)) return false;
    if (isCalibrationCell(r, c)) return false;
    if (isHeaderCell(r, c)) return false;
    return true;
  }

  function drawRectRgb(ctx, x, y, w, h, color) {
    ctx.fillStyle = 'rgb(' + color[0] + ',' + color[1] + ',' + color[2] + ')';
    ctx.fillRect(x, y, w, h);
  }

  function drawNormalAnchor(ctx, x0, y0) {
    drawRectRgb(ctx, x0, y0, ANCHOR_L1_SIZE, ANCHOR_L1_SIZE, [255, 255, 255]);
    drawRectRgb(ctx, x0 + ANCHOR_L2_INSET, y0 + ANCHOR_L2_INSET, ANCHOR_L2_SIZE, ANCHOR_L2_SIZE, [0, 0, 0]);
    drawRectRgb(ctx, x0 + ANCHOR_L3_INSET, y0 + ANCHOR_L3_INSET, ANCHOR_L3_SIZE, ANCHOR_L3_SIZE, [255, 255, 255]);
    drawRectRgb(ctx, x0 + ANCHOR_L4_INSET, y0 + ANCHOR_L4_INSET, ANCHOR_L4_SIZE, ANCHOR_L4_SIZE, [0, 0, 0]);
  }

  function drawBrAnchor(ctx, x0, y0) {
    const h1 = ANCHOR_L1_SIZE >> 1;
    drawRectRgb(ctx, x0, y0, h1, h1, [255, 255, 0]);
    drawRectRgb(ctx, x0 + h1, y0, h1, h1, [0, 255, 0]);
    drawRectRgb(ctx, x0, y0 + h1, h1, h1, [255, 0, 255]);
    drawRectRgb(ctx, x0 + h1, y0 + h1, h1, h1, [0, 255, 255]);
    drawRectRgb(ctx, x0 + ANCHOR_L2_INSET, y0 + ANCHOR_L2_INSET, ANCHOR_L2_SIZE, ANCHOR_L2_SIZE, [0, 0, 0]);

    const h3 = ANCHOR_L3_SIZE >> 1;
    const ix = x0 + ANCHOR_L3_INSET;
    const iy = y0 + ANCHOR_L3_INSET;
    drawRectRgb(ctx, ix, iy, h3, h3, [255, 255, 0]);
    drawRectRgb(ctx, ix + h3, iy, h3, h3, [0, 255, 0]);
    drawRectRgb(ctx, ix, iy + h3, h3, h3, [255, 0, 255]);
    drawRectRgb(ctx, ix + h3, iy + h3, h3, h3, [0, 255, 255]);
    drawRectRgb(ctx, x0 + ANCHOR_L4_INSET, y0 + ANCHOR_L4_INSET, ANCHOR_L4_SIZE, ANCHOR_L4_SIZE, [0, 0, 0]);
  }

  function drawAnchors(ctx) {
    const tlx = ANCHOR_OUT_START;
    const tly = ANCHOR_OUT_START;
    const trx = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    const tryy = ANCHOR_OUT_START;
    const blx = ANCHOR_OUT_START;
    const bly = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    const brx = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    const bry = IMG_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
    drawNormalAnchor(ctx, tlx, tly);
    drawNormalAnchor(ctx, trx, tryy);
    drawNormalAnchor(ctx, blx, bly);
    drawBrAnchor(ctx, brx, bry);
  }

  function drawSymbolCell(ctx, dict, r, c, symbol) {
    const pat = symbol & 0x0F;
    const colorIdx = symbol >> P_BITS;
    const col = COLORS[colorIdx];
    const maskLo = dict.lo[pat];
    const maskHi = dict.hi[pat];
    const sx = MARGIN + c * STRIDE;
    const sy = MARGIN + r * STRIDE;
    ctx.fillStyle = 'rgb(' + col[0] + ',' + col[1] + ',' + col[2] + ')';
    for (let pr = 0; pr < TILE_SIZE; pr++) {
      for (let pc = 0; pc < TILE_SIZE; pc++) {
        const bit = pr * TILE_SIZE + pc;
        if (!isMaskOn(maskLo, maskHi, bit)) continue;
        ctx.fillRect(sx + pc, sy + pr, 1, 1);
      }
    }
  }

  const payloadCells = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (isPayloadCell(r, c)) payloadCells.push([r, c]);
    }
  }

  function generateHeaderSymbols(seq, rand) {
    const out = new Uint8Array(32);
    out[0] = seq & 0x3F;
    out[1] = (seq >> 6) & 0x3F;
    out[2] = (seq >> 12) & 0x3F;
    out[3] = (seq >> 18) & 0x3F;
    for (let i = 4; i < 32; i++) out[i] = (rand() * 64) | 0;
    return out;
  }

  function generatePayloadSymbols(seq, seed) {
    const rand = makeRng((seed ^ Math.imul(seq, 2654435761)) >>> 0);
    const out = new Uint8Array(payloadCells.length);
    for (let i = 0; i < payloadCells.length; i++) {
      let s = (rand() * 64) | 0;
      if (i === 0) s = seq & 0x3F;
      else if (i === 1) s = (seq >> 6) & 0x3F;
      else if (i === 2) s = (seq >> 12) & 0x3F;
      else if (i === 3) s = (seq >> 18) & 0x3F;
      out[i] = s;
    }
    return out;
  }

  function pack6Bits(symbols) {
    const out = new Uint8Array(Math.ceil(symbols.length * 6 / 8));
    let writeIdx = 0;
    let buffer = 0;
    let bits = 0;
    for (let i = 0; i < symbols.length; i++) {
      buffer = (buffer << 6) | (symbols[i] & 0x3F);
      bits += 6;
      while (bits >= 8) {
        out[writeIdx++] = (buffer >>> (bits - 8)) & 0xFF;
        bits -= 8;
      }
    }
    if (bits > 0) out[writeIdx++] = (buffer << (8 - bits)) & 0xFF;
    return out.subarray(0, writeIdx);
  }

  function drawCanonicalFrame(ctx, dict, seq, seed) {
    const rand = makeRng((seed ^ Math.imul(seq, 2654435761)) >>> 0);
    const header = generateHeaderSymbols(seq, rand);
    const payload = generatePayloadSymbols(seq, seed);

    drawRectRgb(ctx, 0, 0, IMG_SIZE, IMG_SIZE, [0, 0, 0]);
    drawAnchors(ctx);

    for (let i = 0; i < 8; i++) {
      drawRectRgb(ctx, MARGIN + (6 + i) * STRIDE, MARGIN, TILE_SIZE, TILE_SIZE, COLORS[i % NUM_COLORS]);
    }

    let hi = 0;
    for (let c = 14; c < 46; c++) {
      drawSymbolCell(ctx, dict, 0, c, header[hi++]);
    }

    for (let i = 0; i < payloadCells.length; i++) {
      const rc = payloadCells[i];
      drawSymbolCell(ctx, dict, rc[0], rc[1], payload[i]);
    }

    return { headerSymbols: header, payloadSymbols: payload };
  }

  function drawFrame(ctx, dict, seq, seed, options) {
    const render = normalizeRenderOptions(options);
    const canvas = ctx.canvas;
    const dims = getCanvasDimensions(render);
    if (canvas.width !== dims.width || canvas.height !== dims.height) {
      canvas.width = dims.width;
      canvas.height = dims.height;
    }
    ctx.imageSmoothingEnabled = false;
    drawRectRgb(ctx, 0, 0, canvas.width, canvas.height, [0, 0, 0]);
    const rendered = drawCanonicalFrame(canonicalCtx, dict, seq, seed);
    ctx.drawImage(canonicalCanvas, 0, 0, canvas.width, canvas.height);
    rendered.renderOptions = render;
    return rendered;
  }

  function buildFrameSet(dict, seed, frameCount) {
    const total = frameCount || BENCH_FRAME_SET;
    const frames = [];
    for (let seq = 1; seq <= total; seq++) {
      const payloadSymbols = generatePayloadSymbols(seq, seed);
      frames.push({
        seq,
        payloadSymbols,
        payloadBytes: pack6Bits(payloadSymbols),
      });
    }
    return frames;
  }

  function unpack6Bits(bytes, symbolCount) {
    const out = new Uint8Array(symbolCount);
    let bitPos = 0;
    for (let i = 0; i < symbolCount; i++) {
      let v = 0;
      for (let b = 0; b < 6; b++) {
        const p = bitPos + b;
        const by = bytes[p >> 3];
        const bit = (by >> (7 - (p & 7))) & 1;
        v = (v << 1) | bit;
      }
      out[i] = v;
      bitPos += 6;
    }
    return out;
  }

  function decodeSeqFromPayloadBytes(payloadBytes) {
    if (!payloadBytes || payloadBytes.length < 3) return -1;
    const syms = unpack6Bits(payloadBytes, 4);
    return syms[0] | (syms[1] << 6) | (syms[2] << 12) | (syms[3] << 18);
  }

  function hammingBytes(a, b) {
    const n = Math.min(a.length, b.length);
    let total = 0;
    for (let i = 0; i < n; i++) {
      total += pop8[(a[i] ^ b[i]) & 255];
    }
    return total + Math.abs(a.length - b.length) * 8;
  }

  function findNearestFrame(decodedPayloadBytes, frames) {
    let best = null;
    for (let i = 0; i < frames.length; i++) {
      const frame = frames[i];
      const dist = hammingBytes(decodedPayloadBytes, frame.payloadBytes);
      if (!best || dist < best.hamming) {
        best = { frame, hamming: dist };
      }
    }
    return best;
  }

  function compareSymbols(decoded, expected) {
    let symbolCorrect = 0;
    let patternCorrect = 0;
    let colorCorrect = 0;
    const n = Math.min(decoded.length, expected.length);
    for (let i = 0; i < n; i++) {
      const ds = decoded[i];
      const es = expected[i];
      if (ds === es) symbolCorrect++;
      if ((ds & 0x0F) === (es & 0x0F)) patternCorrect++;
      if ((ds >> 4) === (es >> 4)) colorCorrect++;
    }
    return { n, symbolCorrect, patternCorrect, colorCorrect };
  }

  function isBetterSymbolMatch(a, b) {
    if (!b) return true;
    if (a.cmp.symbolCorrect !== b.cmp.symbolCorrect) return a.cmp.symbolCorrect > b.cmp.symbolCorrect;
    if (a.cmp.patternCorrect !== b.cmp.patternCorrect) return a.cmp.patternCorrect > b.cmp.patternCorrect;
    if (a.cmp.colorCorrect !== b.cmp.colorCorrect) return a.cmp.colorCorrect > b.cmp.colorCorrect;
    return a.hamming < b.hamming;
  }

  function findBestFrameBySymbols(decodedSymbols, frames, decodedPayloadBytes) {
    let best = null;
    let second = null;
    for (let i = 0; i < frames.length; i++) {
      const frame = frames[i];
      const cmp = compareSymbols(decodedSymbols, frame.payloadSymbols);
      const hamming = decodedPayloadBytes ? hammingBytes(decodedPayloadBytes, frame.payloadBytes) : 0;
      const candidate = { frame, cmp, hamming };
      if (isBetterSymbolMatch(candidate, best)) {
        second = best;
        best = candidate;
      } else if (isBetterSymbolMatch(candidate, second)) {
        second = candidate;
      }
    }
    const bestSymbolAcc = best && best.cmp.n ? best.cmp.symbolCorrect / best.cmp.n : 0;
    const secondSymbolAcc = second && second.cmp.n ? second.cmp.symbolCorrect / second.cmp.n : 0;
    return {
      best,
      second,
      top2Gap: best && second ? (bestSymbolAcc - secondSymbolAcc) : 0,
    };
  }

  global.CameraDropCodebook = {
    IMG_SIZE,
    GRID_SIZE,
    STRIDE,
    MARGIN,
    TILE_SIZE,
    NUM_PATTERNS,
    NUM_COLORS,
    P_BITS,
    BENCH_FRAME_SET,
    DEFAULT_CODE_ASPECT,
    DEFAULT_SHORT_SIDE,
    COLORS,
    PAYLOAD_SYMBOLS: payloadCells.length,
    makeRng,
    genDict,
    loadDict,
    isPayloadCell,
    normalizeRenderOptions,
    getCanvasDimensions,
    drawFrame,
    buildFrameSet,
    generatePayloadSymbols,
    pack6Bits,
    decodeSeqFromPayloadBytes,
    unpack6Bits,
    hammingBytes,
    findNearestFrame,
    compareSymbols,
    findBestFrameBySymbols,
  };
})(window);
