'use strict';

(function initRecognizerModule(global) {
  const app = global.CameraDropApp;
  const state = app.state;
  const dom = app.dom;
  const config = app.config;

  const WORKER_SRC = `'use strict';
const GRID_SIZE = 112;
const STRIDE = 9;
const MARGIN = 8;
const TILE_SIZE = 8;
const NUM_PATTERNS = 16;
const P_BITS = 4;
const RECOG_AHASH_N = 16;
const RECOG_AHASH_THRESH = 2;
const NORM_SIZE = 1024;
const DRIFT_MAX = 7;
const COOL_INIT = 0xFE;
const COOL_NONE = 0xFF;
const PRIO_INIT = 0xFFFE;
const HASH_FAST_N = 5;
const HASH_ALL_N = 9;
const SEARCH_PRIMARY = [
  [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
  [1, 1], [-1, -1], [1, -1], [-1, 1],
];
const SEARCH_EXTENDED = [
  [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
  [1, 1], [-1, -1], [1, -1], [-1, 1],
  [2, 0], [0, 2], [-2, 0], [0, -2],
  [2, 1], [1, 2], [-1, 2], [-2, 1],
  [-2, -1], [-1, -2], [1, -2], [2, -1],
  [2, 2], [-2, -2], [2, -2], [-2, 2],
];
const CELL_KIND_CAL = 0;
const CELL_KIND_HEADER = 1;
const CELL_KIND_PAYLOAD = 2;
const HASH_ORDER = [4, 5, 7, 3, 1, 8, 0, 2, 6];
const CELL_SAMPLE_SIZE = TILE_SIZE + 2;
const SAMPLE_AREA = CELL_SAMPLE_SIZE * CELL_SAMPLE_SIZE;
const LUMA_RECHECK_DIST64 = 5;
const LUMA_RECHECK_DIST16 = 1;
const BINARY_RECHECK_DIST64 = 3;
const BINARY_RECHECK_DIST16 = 0;
const RECHECK_MIN_GAP = 2;
const RECHECK_SCORE_FLOOR = 8;
const BINARY_BLOCK_SIZE = 5;
const BINARY_SHARP_BLOCK_SIZE = 7;
const BINARY_THRESHOLD_BIAS = 0;
const BITGRID_RECHECK_DIST64 = 8;
const BITGRID_RECHECK_DIST16 = 1;
const BITGRID_ACCEPT_GAIN = 2;
const BITGRID_ACCEPT_GAIN_HINT = 1;
const DRIFT_PAIRS = [
  [-1, -1], [0, -1], [1, -1],
  [-1, 0],  [0, 0],  [1, 0],
  [-1, 1],  [0, 1],  [1, 1],
];
const HEAP_IDX_BITS = 14;
const HEAP_IDX_MASK = (1 << HEAP_IDX_BITS) - 1;
const COLORS = [
  [255, 255, 0],
  [0, 255, 0],
  [0, 255, 255],
  [255, 0, 255],
];
const BEST_COLOR_FLOOR = 48.0;
const COLOR_VOTE_MIN_SPAN = 12.0;
const COLOR_VOTE_MIN_GAP = 6.0;
const COLOR_VOTE_STRONG_SPAN = 32.0;
const COLOR_VOTE_STRONG_GAP = 12.0;
const COLOR_VOTE_ABS_WEIGHT = 0.35;
const COLOR_VOTE_REL_WEIGHT = 0.65;
const ANCHOR_OUT_START = 2;
const ANCHOR_L1_SIZE = 56;
const ANCHOR_L2_INSET = 7;
const ANCHOR_L2_SIZE = 42;
const ANCHOR_L3_INSET = 14;
const ANCHOR_L3_SIZE = 28;
const ANCHOR_L4_INSET = 21;
const ANCHOR_L4_SIZE = 14;
const IDENTITY_COLOR_MATRIX = new Float32Array([
  1, 0, 0,
  0, 1, 0,
  0, 0, 1,
]);
let colorRefs = COLORS.map((ref) => ref.slice());
let colorVoteRefs = COLORS.map((ref) => stretchNormalizedColorSample(ref));
let colorBias = [0, 0, 0];
let colorMatrix = new Float32Array(IDENTITY_COLOR_MATRIX);
let colorMatrixActive = false;

let normCvs = null;
let normCtx = null;
let hashCvs = null;
let hashCtx = null;
let decodeLayout = null;
let decodeBuffers = null;
const subwindowMap = (() => {
  const out = [];
  for (let idx = 0; idx < 9; idx++) {
    const ox = idx % 3;
    const oy = (idx / 3) | 0;
    const map = new Uint8Array(64);
    let k = 0;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        map[k++] = (oy + r) * CELL_SAMPLE_SIZE + (ox + c);
      }
    }
    out.push(map);
  }
  return out;
})();
const block16Map = (() => {
  const out = new Uint8Array(64);
  for (let i = 0; i < 64; i++) {
    const r = i >> 3;
    const c = i & 7;
    out[i] = ((r >> 1) * 4) + (c >> 1);
  }
  return out;
})();

const pop8 = new Uint8Array(256);
for (let i = 1; i < 256; i++) {
  pop8[i] = pop8[i >> 1] + (i & 1);
}

function popcnt32(v) {
  v >>>= 0;
  return pop8[v & 255] + pop8[(v >>> 8) & 255] + pop8[(v >>> 16) & 255] + pop8[(v >>> 24) & 255];
}

function hammingDist(a, b) {
  let diff = a ^ b;
  let count = 0;
  while (diff) {
    count += Number(diff & 1n);
    diff >>= 1n;
  }
  return count;
}

function expandMask16(mask16) {
  let lo = 0;
  let hi = 0;

  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const bit = (mask16 >>> (r * 4 + c)) & 1;
      if (!bit) {
        continue;
      }

      const r8 = r << 1;
      const c8 = c << 1;
      const idx = [
        r8 * 8 + c8,
        r8 * 8 + c8 + 1,
        (r8 + 1) * 8 + c8,
        (r8 + 1) * 8 + c8 + 1,
      ];

      for (let k = 0; k < 4; k++) {
        const p = idx[k];
        if (p < 32) {
          lo = (lo | ((1 << p) >>> 0)) >>> 0;
        } else {
          hi = (hi | ((1 << (p - 32)) >>> 0)) >>> 0;
        }
      }
    }
  }

  return { lo, hi };
}

function maskIsOn(maskLo, maskHi, bit) {
  if (bit < 32) {
    return ((maskLo >>> bit) & 1) !== 0;
  }
  return ((maskHi >>> (bit - 32)) & 1) !== 0;
}

function compressMask64To16(maskLo, maskHi) {
  let out = 0;
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const base = (r << 4) + (c << 1);
      let on = 0;
      on += maskIsOn(maskLo, maskHi, base) ? 1 : 0;
      on += maskIsOn(maskLo, maskHi, base + 1) ? 1 : 0;
      on += maskIsOn(maskLo, maskHi, base + 8) ? 1 : 0;
      on += maskIsOn(maskLo, maskHi, base + 9) ? 1 : 0;
      if (on >= 2) {
        out |= (1 << (r * 4 + c));
      }
    }
  }
  return out >>> 0;
}

function buildDict16(sourceDict) {
  const out = new Uint16Array(NUM_PATTERNS);
  for (let i = 0; i < NUM_PATTERNS; i++) {
    out[i] = compressMask64To16(sourceDict.lo[i], sourceDict.hi[i]);
  }
  return out;
}

function genDict() {
  const cand = [];
  for (let i = 0; i < (1 << 16); i++) {
    const p = popcnt32(i);
    if (p >= 6 && p <= 10) {
      cand.push(i);
    }
  }

  const pick = [0x00FF];
  const dist = new Int16Array(cand.length);
  for (let i = 0; i < cand.length; i++) {
    dist[i] = popcnt32(cand[i] ^ pick[0]);
  }

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
      if (d < dist[i]) {
        dist[i] = d;
      }
    }
  }

  const lo = new Uint32Array(NUM_PATTERNS);
  const hi = new Uint32Array(NUM_PATTERNS);
  for (let i = 0; i < NUM_PATTERNS; i++) {
    const mask = expandMask16(pick[i]);
    lo[i] = mask.lo;
    hi[i] = mask.hi;
  }

  return { lo, hi };
}

let dict = genDict();
let dict16 = buildDict16(dict);
let dictSource = 'builtin-gen';

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

function matchPattern(maskLo, maskHi) {
  let bestPat = 0;
  let bestDist = 65;

  for (let i = 0; i < NUM_PATTERNS; i++) {
    const d = popcnt32(maskLo ^ dict.lo[i]) + popcnt32(maskHi ^ dict.hi[i]);
    if (d < bestDist) {
      bestDist = d;
      bestPat = i;
    }
  }

  return { bestPat, bestDist };
}

function normalizeChannel(v, ch) {
  const base = v - colorBias[ch];
  const row = ch * 3;
  return clampByte(colorMatrix[row + ch] * base);
}

function applyColorTransform(r, g, b) {
  const x = r - colorBias[0];
  const y = g - colorBias[1];
  const z = b - colorBias[2];
  return [
    clampByte(colorMatrix[0] * x + colorMatrix[1] * y + colorMatrix[2] * z),
    clampByte(colorMatrix[3] * x + colorMatrix[4] * y + colorMatrix[5] * z),
    clampByte(colorMatrix[6] * x + colorMatrix[7] * y + colorMatrix[8] * z),
  ];
}

function normalizeRgbSample(rgb) {
  return applyColorTransform(rgb[0], rgb[1], rgb[2]);
}

function clampByte(v) {
  return v < 0 ? 0 : (v > 255 ? 255 : v);
}

function stretchNormalizedColorSample(rgb) {
  const nr = rgb[0];
  const ng = rgb[1];
  const nb = rgb[2];
  const maxv = Math.max(nr, ng, nb, 1);
  let minv = Math.min(nr, ng, nb, BEST_COLOR_FLOOR);
  if (minv >= maxv) {
    minv = 0;
  }
  const adjust = 255 / Math.max(1, maxv - minv);
  return [
    clampByte((nr - minv) * adjust),
    clampByte((ng - minv) * adjust),
    clampByte((nb - minv) * adjust),
  ];
}

function fixColorSample(rgb) {
  return stretchNormalizedColorSample(normalizeRgbSample(rgb));
}

function relativeColorDist(a, b) {
  const arg = a[0] - a[1];
  const agb = a[1] - a[2];
  const abr = a[2] - a[0];
  const brg = b[0] - b[1];
  const bgb = b[1] - b[2];
  const bbr = b[2] - b[0];
  const d0 = arg - brg;
  const d1 = agb - bgb;
  const d2 = abr - bbr;
  return d0 * d0 + d1 * d1 + d2 * d2;
}

function nearestColor(r, g, b) {
  const x = r - colorBias[0];
  const y = g - colorBias[1];
  const z = b - colorBias[2];
  const nr = clampByte(colorMatrix[0] * x + colorMatrix[1] * y + colorMatrix[2] * z);
  const ng = clampByte(colorMatrix[3] * x + colorMatrix[4] * y + colorMatrix[5] * z);
  const nb = clampByte(colorMatrix[6] * x + colorMatrix[7] * y + colorMatrix[8] * z);
  const span = Math.max(nr, ng, nb) - Math.min(nr, ng, nb);
  const voteSample = stretchNormalizedColorSample([nr, ng, nb]);
  let best = 0;
  let minDist = Number.POSITIVE_INFINITY;
  let secondDist = Number.POSITIVE_INFINITY;

  for (let i = 0; i < colorRefs.length; i++) {
    const ref = colorRefs[i];
    const dr = nr - ref[0];
    const dg = ng - ref[1];
    const db = nb - ref[2];
    const absDist = dr * dr + dg * dg + db * db;
    const relDist = relativeColorDist(voteSample, colorVoteRefs[i]);
    const d = absDist * COLOR_VOTE_ABS_WEIGHT + relDist * COLOR_VOTE_REL_WEIGHT;
    if (d < minDist) {
      secondDist = minDist;
      minDist = d;
      best = i;
    } else if (d < secondDist) {
      secondDist = d;
    }
  }

  return { idx: best, dist: minDist, secondDist, span };
}

function clamp(v, lo, hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

function calcCooldown(previous, idx) {
  if (idx === 4) return 4;
  if ((idx & 1) === 0) return COOL_NONE;
  if ((previous ^ idx) === 6) return COOL_NONE;
  return idx;
}

function packHeapNode(idx, prio) {
  return ((prio & 0xFFFF) << HEAP_IDX_BITS) | (idx & HEAP_IDX_MASK);
}

function unpackHeapIdx(node) {
  return node & HEAP_IDX_MASK;
}

function heapPush(heap, node) {
  heap.push(node);
  let i = heap.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (heap[p] <= node) break;
    heap[i] = heap[p];
    i = p;
  }
  heap[i] = node;
}

function heapPop(heap) {
  if (heap.length === 0) return -1;
  const top = heap[0];
  const last = heap.pop();
  if (heap.length === 0) return top;
  let i = 0;
  while (true) {
    const l = (i << 1) + 1;
    if (l >= heap.length) break;
    const r = l + 1;
    let child = l;
    if (r < heap.length && heap[r] < heap[l]) child = r;
    if (heap[child] >= last) break;
    heap[i] = heap[child];
    i = child;
  }
  heap[i] = last;
  return top;
}

function ensureDecodeLayout() {
  if (decodeLayout) return decodeLayout;

  const xs = [];
  const ys = [];
  const kinds = [];
  const rows = [];
  const cols = [];
  const rcToIdx = new Int32Array(GRID_SIZE * GRID_SIZE);
  rcToIdx.fill(-1);

  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (isAnchorReserved(r, c)) continue;
      const idx = xs.length;
      rows.push(r);
      cols.push(c);
      xs.push(MARGIN + c * STRIDE);
      ys.push(MARGIN + r * STRIDE);
      if (isCalibrationCell(r, c)) kinds.push(CELL_KIND_CAL);
      else if (isHeaderCell(r, c)) kinds.push(CELL_KIND_HEADER);
      else kinds.push(CELL_KIND_PAYLOAD);
      rcToIdx[r * GRID_SIZE + c] = idx;
    }
  }

  const n = xs.length;
  const neighbors = new Int32Array(n * 4);
  neighbors.fill(-1);
  for (let i = 0; i < n; i++) {
    const r = rows[i];
    const c = cols[i];
    neighbors[i * 4] = (c + 1 < GRID_SIZE) ? rcToIdx[r * GRID_SIZE + c + 1] : -1;
    neighbors[i * 4 + 1] = (c - 1 >= 0) ? rcToIdx[r * GRID_SIZE + c - 1] : -1;
    neighbors[i * 4 + 2] = (r + 1 < GRID_SIZE) ? rcToIdx[(r + 1) * GRID_SIZE + c] : -1;
    neighbors[i * 4 + 3] = (r - 1 >= 0) ? rcToIdx[(r - 1) * GRID_SIZE + c] : -1;
  }

  const seeds = [];
  const seen = new Uint8Array(n);
  const pushSeed = (r, c, prio) => {
    const idx = rcToIdx[r * GRID_SIZE + c];
    if (idx < 0 || seen[idx]) return;
    seen[idx] = 1;
    seeds.push(packHeapNode(idx, prio));
  };
  pushSeed(0, 6, 0);
  pushSeed(0, 105, 0);
  pushSeed(111, 6, 0);
  pushSeed(111, 105, 0);
  pushSeed(6, 0, 1);
  pushSeed(6, 111, 1);
  pushSeed(105, 0, 1);
  pushSeed(105, 111, 1);

  decodeLayout = {
    count: n,
    x: Int16Array.from(xs),
    y: Int16Array.from(ys),
    kind: Uint8Array.from(kinds),
    neighbors,
    seeds,
  };
  return decodeLayout;
}

function ensureDecodeBuffers() {
  const layout = ensureDecodeLayout();
  if (decodeBuffers && decodeBuffers.count === layout.count) {
    return decodeBuffers;
  }
  decodeBuffers = {
    count: layout.count,
    pending: new Uint8Array(layout.count),
    driftX: new Int8Array(layout.count),
    driftY: new Int8Array(layout.count),
    priority: new Uint16Array(layout.count),
    cooldown: new Uint8Array(layout.count),
    symbol: new Uint8Array(layout.count),
    grayFrame: new Uint8Array(NORM_SIZE * NORM_SIZE),
    grayTemp: new Uint8Array(NORM_SIZE * NORM_SIZE),
    lumaFrame: new Uint8Array(NORM_SIZE * NORM_SIZE),
    binFrame: new Uint8Array(NORM_SIZE * NORM_SIZE),
    satFrame: new Uint32Array((NORM_SIZE + 1) * (NORM_SIZE + 1)),
    cell10: new Uint8Array(SAMPLE_AREA),
    cell8: new Uint8Array(64),
    block16: new Uint16Array(16),
  };
  return decodeBuffers;
}

function computeColorSignal(r, g, b) {
  const maxv = r > g ? (r > b ? r : b) : (g > b ? g : b);
  const minv = r < g ? (r < b ? r : b) : (g < b ? g : b);
  return maxv - minv;
}

function buildSignalFrames(data, grayFrame, lumaFrame) {
  const b0 = colorBias[0];
  const b1 = colorBias[1];
  const b2 = colorBias[2];
  const m0 = colorMatrix[0];
  const m1 = colorMatrix[1];
  const m2 = colorMatrix[2];
  const m3 = colorMatrix[3];
  const m4 = colorMatrix[4];
  const m5 = colorMatrix[5];
  const m6 = colorMatrix[6];
  const m7 = colorMatrix[7];
  const m8 = colorMatrix[8];
  let sumGray = 0;
  let sumLuma = 0;
  let sumLumaSq = 0;
  let hiClip = 0;
  let loClip = 0;
  for (let src = 0, dst = 0; dst < grayFrame.length; dst++, src += 4) {
    const x = data[src] - b0;
    const y = data[src + 1] - b1;
    const z = data[src + 2] - b2;
    let r = m0 * x + m1 * y + m2 * z;
    let g = m3 * x + m4 * y + m5 * z;
    let b = m6 * x + m7 * y + m8 * z;
    r = r < 0 ? 0 : (r > 255 ? 255 : r);
    g = g < 0 ? 0 : (g > 255 ? 255 : g);
    b = b < 0 ? 0 : (b > 255 ? 255 : b);
    const maxv = r > g ? (r > b ? r : b) : (g > b ? g : b);
    const minv = r < g ? (r < b ? r : b) : (g < b ? g : b);
    const gray = maxv - minv;
    const luma = (((r * 77) + (g * 150) + (b * 29)) / 256) | 0;
    grayFrame[dst] = gray;
    lumaFrame[dst] = luma;
    sumGray += gray;
    sumLuma += luma;
    sumLumaSq += luma * luma;
    if (maxv >= 248) hiClip++;
    if (maxv <= 16) loClip++;
  }
  const n = grayFrame.length || 1;
  const lumaMean = sumLuma / n;
  const grayMean = sumGray / n;
  const lumaVar = Math.max(0, (sumLumaSq / n) - (lumaMean * lumaMean));
  const lumaStd = Math.sqrt(lumaVar);
  const hiClipRatio = hiClip / n;
  const loClipRatio = loClip / n;
  const washedOut = hiClipRatio >= 0.020 || (lumaMean >= 176 && grayMean <= 34) || (lumaMean >= 188 && lumaStd <= 34);
  const lowContrast = lumaStd <= 26 || grayMean <= 18 || (loClipRatio >= 0.18 && lumaMean <= 84);
  return {
    lumaMean,
    lumaStd,
    grayMean,
    hiClipRatio,
    loClipRatio,
    washedOut,
    lowContrast,
  };
}

function sharpenGray(grayFrame, grayTemp, width, height, amount) {
  grayTemp.set(grayFrame);
  const gain = Number.isFinite(amount) ? amount : 0.6;

  for (let y = 1; y < height - 1; y++) {
    const row = y * width;
    for (let x = 1; x < width - 1; x++) {
      const idx = row + x;
      const center = grayFrame[idx];
      const lap = center * 4
        - grayFrame[idx - 1]
        - grayFrame[idx + 1]
        - grayFrame[idx - width]
        - grayFrame[idx + width];
      const next = center + gain * lap;
      grayTemp[idx] = next < 0 ? 0 : (next > 255 ? 255 : next);
    }
  }

  grayFrame.set(grayTemp);
}

function buildIntegralGray(grayFrame, satFrame, width, height) {
  satFrame.fill(0);
  const stride = width + 1;
  let srcIdx = 0;

  for (let y = 0; y < height; y++) {
    let rowSum = 0;
    const satRow = (y + 1) * stride;
    const satPrev = y * stride;
    for (let x = 0; x < width; x++) {
      rowSum += grayFrame[srcIdx++];
      satFrame[satRow + x + 1] = satFrame[satPrev + x + 1] + rowSum;
    }
  }
}

function adaptiveThresholdGray(grayFrame, binFrame, satFrame, width, height, blockSize, thresholdBias) {
  const stride = width + 1;
  const radius = blockSize >> 1;
  const bias = Number.isFinite(thresholdBias) ? thresholdBias : 0;

  for (let y = 0; y < height; y++) {
    const y0 = y - radius < 0 ? 0 : y - radius;
    const y1 = y + radius >= height ? (height - 1) : (y + radius);
    const top = y0 * stride;
    const bottom = (y1 + 1) * stride;
    const row = y * width;

    for (let x = 0; x < width; x++) {
      const x0 = x - radius < 0 ? 0 : x - radius;
      const x1 = x + radius >= width ? (width - 1) : (x + radius);
      const area = (x1 - x0 + 1) * (y1 - y0 + 1);
      const sum = satFrame[bottom + x1 + 1] - satFrame[top + x1 + 1] - satFrame[bottom + x0] + satFrame[top + x0];
      const mean = (sum / area) | 0;
      binFrame[row + x] = grayFrame[row + x] > (mean + bias) ? 255 : 0;
    }
  }
}

function sampleRectMeanRgb(data, x0, y0, size) {
  const sx = clamp(x0, 0, NORM_SIZE - size);
  const sy = clamp(y0, 0, NORM_SIZE - size);
  let sr = 0;
  let sg = 0;
  let sb = 0;
  let n = 0;
  for (let y = 0; y < size; y++) {
    let idx = ((sy + y) * NORM_SIZE + sx) * 4;
    for (let x = 0; x < size; x++) {
      sr += data[idx];
      sg += data[idx + 1];
      sb += data[idx + 2];
      idx += 4;
      n++;
    }
  }
  return [sr / n, sg / n, sb / n];
}

function colorRectScore(r, g, b, colorIdx) {
  switch (colorIdx) {
    case 0: return (r + g) - (b * 2);
    case 1: return (g * 2) - (r + b);
    case 2: return (g + b) - (r * 2);
    case 3: return (r + b) - (g * 2);
    default: return 0;
  }
}

function sampleRectSelectiveRgb(data, x0, y0, size, scorePixel, keepRatio) {
  const sx = clamp(x0, 0, NORM_SIZE - size);
  const sy = clamp(y0, 0, NORM_SIZE - size);
  const samples = [];
  for (let y = 0; y < size; y++) {
    let idx = ((sy + y) * NORM_SIZE + sx) * 4;
    for (let x = 0; x < size; x++) {
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      samples.push([scorePixel(r, g, b), r, g, b]);
      idx += 4;
    }
  }
  samples.sort((a, b) => b[0] - a[0]);
  const keep = Math.max(4, Math.min(samples.length, Math.round(samples.length * keepRatio)));
  let sr = 0;
  let sg = 0;
  let sb = 0;
  for (let i = 0; i < keep; i++) {
    sr += samples[i][1];
    sg += samples[i][2];
    sb += samples[i][3];
  }
  return [sr / keep, sg / keep, sb / keep];
}

function sampleRectStrongColorRgb(data, x0, y0, size, colorIdx) {
  return sampleRectSelectiveRgb(data, x0, y0, size, (r, g, b) => colorRectScore(r, g, b, colorIdx), 0.45);
}

function sampleRectDarkRgb(data, x0, y0, size) {
  return sampleRectSelectiveRgb(data, x0, y0, size, (r, g, b) => -(r + g + b), 0.45);
}

function averageRgb(a, b) {
  return [
    (a[0] + b[0]) * 0.5,
    (a[1] + b[1]) * 0.5,
    (a[2] + b[2]) * 0.5,
  ];
}

function averageRgbs(samples) {
  let sr = 0;
  let sg = 0;
  let sb = 0;
  const n = samples.length || 1;
  for (let i = 0; i < samples.length; i++) {
    sr += samples[i][0];
    sg += samples[i][1];
    sb += samples[i][2];
  }
  return [sr / n, sg / n, sb / n];
}

function subtractRgb(a, b) {
  return [
    Math.max(0, a[0] - b[0]),
    Math.max(0, a[1] - b[1]),
    Math.max(0, a[2] - b[2]),
  ];
}

function sampleRectBrightRgb(data, x0, y0, size) {
  return sampleRectSelectiveRgb(data, x0, y0, size, (r, g, b) => r + g + b, 0.45);
}

function sampleAnchorWhiteRgb(data, baseX, baseY) {
  const outerInset = 2;
  const outerSize = 8;
  const innerBaseX = baseX + ANCHOR_L3_INSET;
  const innerBaseY = baseY + ANCHOR_L3_INSET;
  const innerHalf = ANCHOR_L3_SIZE >> 1;
  const innerInset = 1;
  const innerSize = 6;
  return averageRgbs([
    sampleRectBrightRgb(data, baseX + outerInset, baseY + outerInset, outerSize),
    sampleRectBrightRgb(data, baseX + ANCHOR_L1_SIZE - outerInset - outerSize, baseY + outerInset, outerSize),
    sampleRectBrightRgb(data, baseX + outerInset, baseY + ANCHOR_L1_SIZE - outerInset - outerSize, outerSize),
    sampleRectBrightRgb(data, baseX + ANCHOR_L1_SIZE - outerInset - outerSize, baseY + ANCHOR_L1_SIZE - outerInset - outerSize, outerSize),
    sampleRectBrightRgb(data, innerBaseX + innerInset, innerBaseY + innerInset, innerSize),
    sampleRectBrightRgb(data, innerBaseX + innerHalf + innerInset, innerBaseY + innerInset, innerSize),
    sampleRectBrightRgb(data, innerBaseX + innerInset, innerBaseY + innerHalf + innerInset, innerSize),
    sampleRectBrightRgb(data, innerBaseX + innerHalf + innerInset, innerBaseY + innerHalf + innerInset, innerSize),
  ]);
}

function sampleAnchorBlackRgb(data, baseX, baseY) {
  return averageRgbs([
    sampleRectDarkRgb(data, baseX + ANCHOR_L2_INSET + 7, baseY + ANCHOR_L2_INSET + 7, 8),
    sampleRectDarkRgb(data, baseX + ANCHOR_L4_INSET + 3, baseY + ANCHOR_L4_INSET + 3, 8),
  ]);
}

function invert3x3(m) {
  const a = m[0], b = m[1], c = m[2];
  const d = m[3], e = m[4], f = m[5];
  const g = m[6], h = m[7], i = m[8];
  const A = (e * i) - (f * h);
  const B = -((d * i) - (f * g));
  const C = (d * h) - (e * g);
  const D = -((b * i) - (c * h));
  const E = (a * i) - (c * g);
  const F = -((a * h) - (b * g));
  const G = (b * f) - (c * e);
  const H = -((a * f) - (c * d));
  const I = (a * e) - (b * d);
  const det = (a * A) + (b * B) + (c * C);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-6) {
    return null;
  }
  const invDet = 1 / det;
  return [A * invDet, D * invDet, G * invDet, B * invDet, E * invDet, H * invDet, C * invDet, F * invDet, I * invDet];
}

function fitLinearColorMatrix(actualRows, desiredRows) {
  const ata = new Float64Array(9);
  const atbR = new Float64Array(3);
  const atbG = new Float64Array(3);
  const atbB = new Float64Array(3);
  for (let i = 0; i < actualRows.length; i++) {
    const row = actualRows[i];
    const want = desiredRows[i];
    const x = row[0];
    const y = row[1];
    const z = row[2];
    ata[0] += x * x;
    ata[1] += x * y;
    ata[2] += x * z;
    ata[3] += y * x;
    ata[4] += y * y;
    ata[5] += y * z;
    ata[6] += z * x;
    ata[7] += z * y;
    ata[8] += z * z;
    atbR[0] += x * want[0];
    atbR[1] += y * want[0];
    atbR[2] += z * want[0];
    atbG[0] += x * want[1];
    atbG[1] += y * want[1];
    atbG[2] += z * want[1];
    atbB[0] += x * want[2];
    atbB[1] += y * want[2];
    atbB[2] += z * want[2];
  }
  const inv = invert3x3(ata);
  if (!inv) {
    return null;
  }
  const solve = (rhs) => [
    (inv[0] * rhs[0]) + (inv[1] * rhs[1]) + (inv[2] * rhs[2]),
    (inv[3] * rhs[0]) + (inv[4] * rhs[1]) + (inv[5] * rhs[2]),
    (inv[6] * rhs[0]) + (inv[7] * rhs[1]) + (inv[8] * rhs[2]),
  ];
  const rowR = solve(atbR);
  const rowG = solve(atbG);
  const rowB = solve(atbB);
  return new Float32Array([
    rowR[0], rowR[1], rowR[2],
    rowG[0], rowG[1], rowG[2],
    rowB[0], rowB[1], rowB[2],
  ]);
}

function computeMatrixResidual(matrix, actualRows, desiredRows) {
  let total = 0;
  let count = 0;
  for (let i = 0; i < actualRows.length; i++) {
    const row = actualRows[i];
    const want = desiredRows[i];
    const r = (matrix[0] * row[0]) + (matrix[1] * row[1]) + (matrix[2] * row[2]);
    const g = (matrix[3] * row[0]) + (matrix[4] * row[1]) + (matrix[5] * row[2]);
    const b = (matrix[6] * row[0]) + (matrix[7] * row[1]) + (matrix[8] * row[2]);
    total += ((r - want[0]) * (r - want[0])) + ((g - want[1]) * (g - want[1])) + ((b - want[2]) * (b - want[2]));
    count += 3;
  }
  return count > 0 ? (total / count) : Number.POSITIVE_INFINITY;
}

function setFallbackColorCalibration(black, white) {
  colorBias = [black[0], black[1], black[2]];
  colorMatrix = new Float32Array([
    255 / Math.max(16, white[0] - black[0]), 0, 0,
    0, 255 / Math.max(16, white[1] - black[1]), 0,
    0, 0, 255 / Math.max(16, white[2] - black[2]),
  ]);
  colorMatrixActive = false;
}

function estimateColorCalibration(data) {
  const baseX = NORM_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
  const baseY = NORM_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
  const outerInset = 2;
  const outerSize = 8;
  const innerBaseX = baseX + ANCHOR_L3_INSET;
  const innerBaseY = baseY + ANCHOR_L3_INSET;
  const innerHalf = ANCHOR_L3_SIZE >> 1;
  const innerInset = 1;
  const innerSize = 6;
  const tlBaseX = ANCHOR_OUT_START;
  const tlBaseY = ANCHOR_OUT_START;
  const trBaseX = NORM_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
  const trBaseY = ANCHOR_OUT_START;
  const blBaseX = ANCHOR_OUT_START;
  const blBaseY = NORM_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;
  const refs = COLORS.map((ref) => ref.slice());

  const yOuter = sampleRectStrongColorRgb(data, baseX + outerInset, baseY + outerInset, outerSize, 0);
  const gOuter = sampleRectStrongColorRgb(data, baseX + ANCHOR_L1_SIZE - outerInset - outerSize, baseY + outerInset, outerSize, 1);
  const mOuter = sampleRectStrongColorRgb(data, baseX + outerInset, baseY + ANCHOR_L1_SIZE - outerInset - outerSize, outerSize, 3);
  const cOuter = sampleRectStrongColorRgb(data, baseX + ANCHOR_L1_SIZE - outerInset - outerSize, baseY + ANCHOR_L1_SIZE - outerInset - outerSize, outerSize, 2);
  const yInner = sampleRectStrongColorRgb(data, innerBaseX + innerInset, innerBaseY + innerInset, innerSize, 0);
  const gInner = sampleRectStrongColorRgb(data, innerBaseX + innerHalf + innerInset, innerBaseY + innerInset, innerSize, 1);
  const mInner = sampleRectStrongColorRgb(data, innerBaseX + innerInset, innerBaseY + innerHalf + innerInset, innerSize, 3);
  const cInner = sampleRectStrongColorRgb(data, innerBaseX + innerHalf + innerInset, innerBaseY + innerHalf + innerInset, innerSize, 2);
  refs[0] = averageRgb(yOuter, yInner);
  refs[1] = averageRgb(gOuter, gInner);
  refs[2] = averageRgb(cOuter, cInner);
  refs[3] = averageRgb(mOuter, mInner);

  const white = averageRgbs([
    sampleAnchorWhiteRgb(data, tlBaseX, tlBaseY),
    sampleAnchorWhiteRgb(data, trBaseX, trBaseY),
    sampleAnchorWhiteRgb(data, blBaseX, blBaseY),
  ]);
  const black = averageRgbs([
    sampleAnchorBlackRgb(data, tlBaseX, tlBaseY),
    sampleAnchorBlackRgb(data, trBaseX, trBaseY),
    sampleAnchorBlackRgb(data, blBaseX, blBaseY),
    sampleAnchorBlackRgb(data, baseX, baseY),
  ]);

  setFallbackColorCalibration(black, white);

  const actualRows = [refs[0], refs[1], refs[2], refs[3], white].map((rgb) => subtractRgb(rgb, black));
  const desiredRows = [
    COLORS[0].slice(),
    COLORS[1].slice(),
    COLORS[2].slice(),
    COLORS[3].slice(),
    [255, 255, 255],
  ];
  const matrix = fitLinearColorMatrix(actualRows, desiredRows);
  if (matrix) {
    const residual = computeMatrixResidual(matrix, actualRows, desiredRows);
    let maxCoeff = 0;
    for (let i = 0; i < matrix.length; i++) {
      const c = Math.abs(matrix[i]);
      if (c > maxCoeff) {
        maxCoeff = c;
      }
    }
    if (Number.isFinite(residual) && residual <= 4800 && maxCoeff <= 4.0) {
      colorBias = [black[0], black[1], black[2]];
      colorMatrix = matrix;
      colorMatrixActive = true;
    }
  }

  colorRefs = COLORS.map((ref) => ref.slice());
  colorVoteRefs = colorRefs.map((rgb) => stretchNormalizedColorSample(rgb));
}

function preprocessSymbolFrame(data, buffers, sharpenHint, sharpenStrength) {
  const stats = buildSignalFrames(data, buffers.grayFrame, buffers.lumaFrame);
  if (sharpenHint && sharpenStrength > 0) {
    sharpenGray(buffers.lumaFrame, buffers.grayTemp, NORM_SIZE, NORM_SIZE, sharpenStrength);
  }
  buildIntegralGray(buffers.lumaFrame, buffers.satFrame, NORM_SIZE, NORM_SIZE);
  const binaryHint = !!sharpenHint || stats.washedOut || stats.lowContrast;
  adaptiveThresholdGray(
    buffers.lumaFrame,
    buffers.binFrame,
    buffers.satFrame,
    NORM_SIZE,
    NORM_SIZE,
    binaryHint ? BINARY_SHARP_BLOCK_SIZE : BINARY_BLOCK_SIZE,
    BINARY_THRESHOLD_BIAS
  );
  return {
    primaryFrame: buffers.grayFrame,
    lumaFrame: buffers.lumaFrame,
    bitgridFrame: buffers.binFrame,
    lumaHint: stats.washedOut || stats.lowContrast,
    bitgridHint: binaryHint,
    frameStats: stats,
  };
}

function sampleCell10(signalFrame, x0, y0, cell10) {
  const sx = clamp(x0 - 1, 0, NORM_SIZE - CELL_SAMPLE_SIZE);
  const sy = clamp(y0 - 1, 0, NORM_SIZE - CELL_SAMPLE_SIZE);
  let sum = 0;
  let k = 0;

  for (let r = 0; r < CELL_SAMPLE_SIZE; r++) {
    const row = (sy + r) * NORM_SIZE + sx;
    for (let c = 0; c < CELL_SAMPLE_SIZE; c++) {
      const v = signalFrame[row + c];
      cell10[k++] = v;
      sum += v;
    }
  }

  return { sx, sy, sum };
}

function sampleCell10Luma(data, x0, y0, cell10) {
  const sx = clamp(x0 - 1, 0, NORM_SIZE - CELL_SAMPLE_SIZE);
  const sy = clamp(y0 - 1, 0, NORM_SIZE - CELL_SAMPLE_SIZE);
  let sum = 0;
  let k = 0;

  for (let r = 0; r < CELL_SAMPLE_SIZE; r++) {
    let idx = ((sy + r) * NORM_SIZE + sx) * 4;
    for (let c = 0; c < CELL_SAMPLE_SIZE; c++) {
      const corrected = applyColorTransform(data[idx], data[idx + 1], data[idx + 2]);
      const v = ((corrected[0] * 77) + (corrected[1] * 150) + (corrected[2] * 29)) / 256;
      cell10[k++] = v;
      sum += v;
      idx += 4;
    }
  }

  return { sx, sy, sum };
}

function hashSubwindow10(cell10, driftIdx, block16, threshold) {
  const map = subwindowMap[driftIdx];
  block16.fill(0);
  let maskLo = 0;
  let maskHi = 0;

  for (let i = 0; i < 64; i++) {
    const v = cell10[map[i]];
    if (v > threshold) {
      if (i < 32) maskLo = (maskLo | ((1 << i) >>> 0)) >>> 0;
      else maskHi = (maskHi | ((1 << (i - 32)) >>> 0)) >>> 0;
    }
    block16[block16Map[i]] += v;
  }

  return { maskLo, maskHi, mask16: hashBlock16(block16) };
}

function hashBlock16(block16) {
  let sum = 0;
  for (let i = 0; i < 16; i++) sum += block16[i];
  const threshold = sum / 16;
  let mask = 0;
  for (let i = 0; i < 16; i++) {
    if (block16[i] > threshold) {
      mask |= (1 << i);
    }
  }
  return mask >>> 0;
}

function matchPattern16(mask16) {
  let bestPat = 0;
  let bestDist = 17;
  for (let i = 0; i < NUM_PATTERNS; i++) {
    const d = popcnt32((mask16 ^ dict16[i]) >>> 0);
    if (d < bestDist) {
      bestDist = d;
      bestPat = i;
    }
  }
  return { bestPat, bestDist };
}

function matchPatternCombined(mask16, maskLo, maskHi) {
  let bestPat = 0;
  let bestDist64 = 65;
  let bestDist16 = 17;

  for (let i = 0; i < NUM_PATTERNS; i++) {
    const d64 = popcnt32(maskLo ^ dict.lo[i]) + popcnt32(maskHi ^ dict.hi[i]);
    const d16 = popcnt32((mask16 ^ dict16[i]) >>> 0);
    if (d64 < bestDist64 || (d64 === bestDist64 && d16 < bestDist16)) {
      bestDist64 = d64;
      bestDist16 = d16;
      bestPat = i;
    }
  }

  return {
    bestPat,
    bestDist64,
    bestDist16,
    bestScore: (bestDist64 << 2) + bestDist16,
  };
}

function driftIndexFromOffset(dx, dy) {
  const sx = dx < 0 ? -1 : (dx > 0 ? 1 : 0);
  const sy = dy < 0 ? -1 : (dy > 0 ? 1 : 0);
  return (sy + 1) * 3 + (sx + 1);
}

function chooseSearchDriftIndices(cooldown) {
  if (cooldown === COOL_INIT) {
    return HASH_ORDER;
  }
  return HASH_ORDER.slice(0, HASH_FAST_N);
}

function shouldPreferBitgridCandidate(primary, candidate, forceHint) {
  const gain64 = primary.bestDist64 - candidate.bestDist64;
  if (candidate.bestDist64 === 0 && primary.bestDist64 >= 3) {
    return true;
  }
  if (forceHint) {
    if (gain64 >= BITGRID_ACCEPT_GAIN_HINT) {
      return true;
    }
    return candidate.bestDist64 === primary.bestDist64 && candidate.bestDist16 + 1 < primary.bestDist16;
  }
  if (candidate.bestDist64 > 2) {
    return false;
  }
  if (gain64 >= BITGRID_ACCEPT_GAIN) {
    return true;
  }
  return primary.bestDist64 >= 10 && gain64 > 0 && candidate.bestDist16 <= primary.bestDist16;
}

function shouldPreferLumaCandidate(primary, candidate, forceHint) {
  const gain64 = primary.bestDist64 - candidate.bestDist64;
  if (candidate.bestDist64 === 0 && primary.bestDist64 >= 2) {
    return true;
  }
  if (candidate.bestDist64 > primary.bestDist64) {
    return false;
  }
  if (forceHint) {
    if (gain64 >= 1 && candidate.bestDist16 <= primary.bestDist16 + 1) {
      return true;
    }
    return gain64 === 0 && candidate.bestDist16 + 1 < primary.bestDist16;
  }
  if (gain64 >= 2) {
    return true;
  }
  if (primary.bestDist64 >= 8 && gain64 >= 1 && candidate.bestDist16 <= primary.bestDist16) {
    return true;
  }
  return primary.bestDist64 >= 10 && gain64 === 0 && candidate.bestDist16 + 1 < primary.bestDist16;
}

function decodeColorFromMask(data, x0, y0, patIdx) {
  const bestMaskLo = dict.lo[patIdx];
  const bestMaskHi = dict.hi[patIdx];
  const cntAll = new Uint16Array(4);
  const cntStrong = new Uint16Array(4);
  const distAll = new Float64Array(4);
  const distStrong = new Float64Array(4);
  let validAll = 0;
  let validStrong = 0;

  for (let pr = 0; pr < TILE_SIZE; pr++) {
    for (let pc = 0; pc < TILE_SIZE; pc++) {
      const bit = pr * TILE_SIZE + pc;
      if (!maskIsOn(bestMaskLo, bestMaskHi, bit)) continue;
      const idx = ((y0 + pr) * NORM_SIZE + (x0 + pc)) * 4;
      const m = nearestColor(data[idx], data[idx + 1], data[idx + 2]);
      const gap = Math.max(0, Math.sqrt(m.secondDist) - Math.sqrt(m.dist));
      if (m.span < COLOR_VOTE_MIN_SPAN && gap < COLOR_VOTE_MIN_GAP) {
        continue;
      }
      cntAll[m.idx]++;
      distAll[m.idx] += m.dist;
      validAll++;
      if (m.span >= COLOR_VOTE_STRONG_SPAN || gap >= COLOR_VOTE_STRONG_GAP) {
        cntStrong[m.idx]++;
        distStrong[m.idx] += m.dist;
        validStrong++;
      }
    }
  }

  function pickBest(cntBuf, distBuf, valid) {
    if (valid <= 0) {
      return null;
    }
    let bestColor = 0;
    let bestCnt = -1;
    let bestAvg = Number.POSITIVE_INFINITY;
    for (let i = 0; i < 4; i++) {
      const cnt = cntBuf[i];
      if (cnt <= 0) continue;
      const avg = distBuf[i] / cnt;
      if (cnt > bestCnt || (cnt === bestCnt && avg < bestAvg)) {
        bestColor = i;
        bestCnt = cnt;
        bestAvg = avg;
      }
    }
    return { bestColor, bestAvgColorDist: bestAvg };
  }

  const strongPick = pickBest(cntStrong, distStrong, validStrong);
  if (strongPick && validStrong >= Math.max(3, (validAll >> 2))) {
    return strongPick;
  }
  const allPick = pickBest(cntAll, distAll, validAll);
  if (allPick) {
    return allPick;
  }
  return { bestColor: 0, bestAvgColorDist: Number.POSITIVE_INFINITY };
}

function decodeCellAdaptive(data, frames, x0, y0, cooldown, cell10, cell8, block16) {
  void cell8;
  const primaryFrame = frames.primaryFrame;
  const lumaFrame = frames.lumaFrame;
  const lumaHint = !!frames.lumaHint;
  const bitgridFrame = frames.bitgridFrame;
  const bitgridHint = !!frames.bitgridHint;
  const searchDriftIndices = chooseSearchDriftIndices(cooldown);

  const decodeCandidate = (signalFrame) => {
    let bestPat = 0;
    let bestDist16 = 17;
    let bestDist64 = 65;
    let bestDx = 0;
    let bestDy = 0;
    let bestSampleX = clamp(x0, 0, NORM_SIZE - TILE_SIZE);
    let bestSampleY = clamp(y0, 0, NORM_SIZE - TILE_SIZE);
    let bestRadius = 0;

    const consider = (sample, driftIdx, hit) => {
      const ox = driftIdx % 3;
      const oy = (driftIdx / 3) | 0;
      const sampleX = sample.sx + ox;
      const sampleY = sample.sy + oy;
      const dx = sampleX - x0;
      const dy = sampleY - y0;
      const radius = Math.abs(dx) + Math.abs(dy);

      if (hit.bestDist64 < bestDist64
          || (hit.bestDist64 === bestDist64 && hit.bestDist16 < bestDist16)
          || (hit.bestDist64 === bestDist64 && hit.bestDist16 === bestDist16 && radius < bestRadius)) {
        bestDist16 = hit.bestDist16;
        bestDist64 = hit.bestDist64;
        bestPat = hit.bestPat;
        bestDx = dx;
        bestDy = dy;
        bestSampleX = sampleX;
        bestSampleY = sampleY;
        bestRadius = radius;
      }
    };

    const evaluateSample = (sample, driftIndices) => {
      const threshold = sample.sum / SAMPLE_AREA;
      for (let i = 0; i < driftIndices.length; i++) {
        const driftIdx = driftIndices[i];
        const hashes = hashSubwindow10(cell10, driftIdx, block16, threshold);
        const hit = matchPatternCombined(hashes.mask16, hashes.maskLo, hashes.maskHi);
        consider(sample, driftIdx, hit);
        if (bestDist64 <= 2 && bestDist16 === 0 && driftIdx === 4) {
          return true;
        }
      }
      return false;
    };

    let sample = sampleCell10(signalFrame, x0, y0, cell10);
    evaluateSample(sample, searchDriftIndices);

    if (bestDist64 > 8 || bestDist16 > 1) {
      for (let i = 0; i < SEARCH_EXTENDED.length; i++) {
        const off = SEARCH_EXTENDED[i];
        if (Math.abs(off[0]) <= 1 && Math.abs(off[1]) <= 1) {
          continue;
        }
        sample = sampleCell10(signalFrame, x0 + off[0], y0 + off[1], cell10);
        evaluateSample(sample, [4]);
        if (bestDist64 <= 2 && bestDist16 === 0) {
          break;
        }
      }
    }

    return {
      bestPat,
      bestDist16,
      bestDist64,
      bestDx,
      bestDy,
      bestSampleX,
      bestSampleY,
    };
  };

  let best = decodeCandidate(primaryFrame);

  if (lumaFrame && (lumaHint || best.bestDist64 > LUMA_RECHECK_DIST64 || (best.bestDist64 > 4 && best.bestDist16 > LUMA_RECHECK_DIST16))) {
    const luma = decodeCandidate(lumaFrame);
    if (shouldPreferLumaCandidate(best, luma, lumaHint)) {
      best = luma;
    }
  }

  if (bitgridFrame && (bitgridHint || best.bestDist64 > BITGRID_RECHECK_DIST64 || (best.bestDist64 > 6 && best.bestDist16 > BITGRID_RECHECK_DIST16))) {
    const bitgrid = decodeCandidate(bitgridFrame);
    if (shouldPreferBitgridCandidate(best, bitgrid, bitgridHint)) {
      best = bitgrid;
    }
  }

  const color = decodeColorFromMask(data, best.bestSampleX, best.bestSampleY, best.bestPat);
  const symbol = ((color.bestColor << P_BITS) | best.bestPat) & 0x3F;
  return {
    symbol,
    bestDist: (best.bestDist64 << 2) + best.bestDist16,
    driftIdx: driftIndexFromOffset(best.bestDx, best.bestDy),
    driftX: best.bestDx,
    driftY: best.bestDy,
    bestAvgColorDist: color.bestAvgColorDist,
  };
}

function tryQueueNeighbor(next, driftX, driftY, prio, cooldown, layout, buffers, heap) {
  if (next < 0 || buffers.pending[next] === 0) return;
  if (buffers.priority[next] <= prio) return;
  buffers.driftX[next] = driftX;
  buffers.driftY[next] = driftY;
  buffers.priority[next] = prio;
  buffers.cooldown[next] = cooldown;
  heapPush(heap, packHeapNode(next, prio));
}

function queueAdjacents(idx, driftX, driftY, prio, cooldown, layout, buffers, heap) {
  const b = idx * 4;
  tryQueueNeighbor(layout.neighbors[b], driftX, driftY, prio, cooldown, layout, buffers, heap);
  tryQueueNeighbor(layout.neighbors[b + 1], driftX, driftY, prio, cooldown, layout, buffers, heap);
  tryQueueNeighbor(layout.neighbors[b + 2], driftX, driftY, prio, cooldown, layout, buffers, heap);
  tryQueueNeighbor(layout.neighbors[b + 3], driftX, driftY, prio, cooldown, layout, buffers, heap);
}

function queueAggressive(idx, driftX, driftY, prio, cooldown, layout, buffers, heap) {
  const b = idx * 4;
  const right = layout.neighbors[b];
  const left = layout.neighbors[b + 1];
  const down = layout.neighbors[b + 2];
  const up = layout.neighbors[b + 3];

  if (right >= 0 && left >= 0) {
    const rr = layout.neighbors[right * 4];
    const rrr = rr >= 0 ? layout.neighbors[rr * 4] : -1;
    const ll = layout.neighbors[left * 4 + 1];
    const lll = ll >= 0 ? layout.neighbors[ll * 4 + 1] : -1;
    tryQueueNeighbor(rr, driftX, driftY, prio, cooldown, layout, buffers, heap);
    tryQueueNeighbor(rrr, driftX, driftY, prio, cooldown, layout, buffers, heap);
    tryQueueNeighbor(ll, driftX, driftY, prio, cooldown, layout, buffers, heap);
    tryQueueNeighbor(lll, driftX, driftY, prio, cooldown, layout, buffers, heap);
  }

  if (up >= 0 && down >= 0) {
    const uu = layout.neighbors[up * 4 + 3];
    const uuu = uu >= 0 ? layout.neighbors[uu * 4 + 3] : -1;
    const dd = layout.neighbors[down * 4 + 2];
    const ddd = dd >= 0 ? layout.neighbors[dd * 4 + 2] : -1;
    tryQueueNeighbor(uu, driftX, driftY, prio, cooldown, layout, buffers, heap);
    tryQueueNeighbor(uuu, driftX, driftY, prio, cooldown, layout, buffers, heap);
    tryQueueNeighbor(dd, driftX, driftY, prio, cooldown, layout, buffers, heap);
    tryQueueNeighbor(ddd, driftX, driftY, prio, cooldown, layout, buffers, heap);
  }
}

function decodeByPriority(data, frames) {
  const layout = ensureDecodeLayout();
  const buffers = ensureDecodeBuffers();

  buffers.pending.fill(1);
  buffers.driftX.fill(0);
  buffers.driftY.fill(0);
  buffers.priority.fill(PRIO_INIT);
  buffers.cooldown.fill(COOL_INIT);

  const heap = [];
  for (let i = 0; i < layout.seeds.length; i++) heapPush(heap, layout.seeds[i]);

  let decoded = 0;
  let sumPatternDist = 0;
  let patternCount = 0;

  while (heap.length > 0 && decoded < layout.count) {
    const node = heapPop(heap);
    const idx = unpackHeapIdx(node);
    if (buffers.pending[idx] === 0) continue;

    buffers.pending[idx] = 0;
    decoded++;

    const prevErr = buffers.priority[idx];
    const prevCooldown = buffers.cooldown[idx];
    const cell = decodeCellAdaptive(
      data,
      frames,
      layout.x[idx] + buffers.driftX[idx],
      layout.y[idx] + buffers.driftY[idx],
      prevCooldown,
      buffers.cell10, buffers.cell8, buffers.block16
    );

    const ndx = clamp(buffers.driftX[idx] + cell.driftX, -DRIFT_MAX, DRIFT_MAX);
    const ndy = clamp(buffers.driftY[idx] + cell.driftY, -DRIFT_MAX, DRIFT_MAX);
    const nextCooldown = calcCooldown(prevCooldown, cell.driftIdx);

    queueAdjacents(idx, ndx, ndy, cell.bestDist, nextCooldown, layout, buffers, heap);
    if (prevErr < 3 && cell.bestDist < 3 && prevCooldown === 4 && nextCooldown === 4) {
      queueAggressive(idx, ndx, ndy, cell.bestDist, nextCooldown, layout, buffers, heap);
    }

    buffers.driftX[idx] = ndx;
    buffers.driftY[idx] = ndy;
    buffers.priority[idx] = cell.bestDist;
    buffers.cooldown[idx] = nextCooldown;
    buffers.symbol[idx] = cell.symbol;

    if (layout.kind[idx] !== CELL_KIND_CAL) {
      sumPatternDist += cell.bestDist;
      patternCount++;
    }
  }

  if (decoded < layout.count) {
    for (let idx = 0; idx < layout.count; idx++) {
      if (buffers.pending[idx] === 0) continue;
      const cell = decodeCellAdaptive(data, frames, layout.x[idx], layout.y[idx], COOL_INIT, buffers.cell10, buffers.cell8, buffers.block16);
      buffers.pending[idx] = 0;
      buffers.symbol[idx] = cell.symbol;
      buffers.priority[idx] = cell.bestDist;
      if (layout.kind[idx] !== CELL_KIND_CAL) {
        sumPatternDist += cell.bestDist;
        patternCount++;
      }
    }
  }

  const headerSymbols = [];
  const payloadSymbols = [];
  for (let i = 0; i < layout.count; i++) {
    if (layout.kind[i] === CELL_KIND_HEADER) headerSymbols.push(buffers.symbol[i]);
    else if (layout.kind[i] === CELL_KIND_PAYLOAD) payloadSymbols.push(buffers.symbol[i]);
  }

  return {
    headerSymbols,
    payloadSymbols,
    avgPatternDist: patternCount > 0 ? (sumPatternDist / patternCount) : 0,
  };
}

function decodeByLinearFast(data, frames) {
  const layout = ensureDecodeLayout();
  const buffers = ensureDecodeBuffers();
  const rowDriftX = new Int8Array(GRID_SIZE);
  const rowDriftY = new Int8Array(GRID_SIZE);
  let sumPatternDist = 0;
  let patternCount = 0;

  for (let i = 0; i < layout.count; i++) {
    const yCell = ((layout.y[i] - MARGIN) / STRIDE) | 0;
    const dx = rowDriftX[yCell];
    const dy = rowDriftY[yCell];
    const cell = decodeCellAdaptive(
      data,
      frames,
      layout.x[i] + dx,
      layout.y[i] + dy,
      4,
      buffers.cell10, buffers.cell8, buffers.block16
    );
    buffers.symbol[i] = cell.symbol;
    if (cell.bestDist <= 10) {
      rowDriftX[yCell] = clamp(dx + cell.driftX, -DRIFT_MAX, DRIFT_MAX);
      rowDriftY[yCell] = clamp(dy + cell.driftY, -DRIFT_MAX, DRIFT_MAX);
    }
    if (layout.kind[i] !== CELL_KIND_CAL) {
      sumPatternDist += cell.bestDist;
      patternCount++;
    }
  }

  const headerSymbols = [];
  const payloadSymbols = [];
  for (let i = 0; i < layout.count; i++) {
    if (layout.kind[i] === CELL_KIND_HEADER) headerSymbols.push(buffers.symbol[i]);
    else if (layout.kind[i] === CELL_KIND_PAYLOAD) payloadSymbols.push(buffers.symbol[i]);
  }
  return {
    headerSymbols,
    payloadSymbols,
    avgPatternDist: patternCount > 0 ? (sumPatternDist / patternCount) : 0,
  };
}

function decodeAdaptiveByQueue(data, frames, queueLen) {
  void queueLen;
  return decodeByPriority(data, frames);
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

  if (bits > 0) {
    out[writeIdx++] = (buffer << (8 - bits)) & 0xFF;
  }

  return { bytes: out.subarray(0, writeIdx), tailBits: bits };
}

function computeDeskewHash() {
  const margin = Math.round(NORM_SIZE * 0.08);
  const inner = NORM_SIZE - margin * 2;
  hashCtx.drawImage(normCvs, margin, margin, inner, inner, 0, 0, RECOG_AHASH_N, RECOG_AHASH_N);
  const d = hashCtx.getImageData(0, 0, RECOG_AHASH_N, RECOG_AHASH_N).data;

  const gray = new Uint8Array(RECOG_AHASH_N * RECOG_AHASH_N);
  let sum = 0;
  for (let i = 0; i < gray.length; i++) {
    gray[i] = (d[i * 4] * 77 + d[i * 4 + 1] * 150 + d[i * 4 + 2] * 29) >> 8;
    sum += gray[i];
  }

  const avg = sum / gray.length;
  let hash = 0n;
  for (let i = 0; i < gray.length; i++) {
    if (gray[i] >= avg) {
      hash |= (1n << BigInt(i));
    }
  }

  return hash;
}

function ensureWorkCanvas() {
  if (!normCvs) {
    normCvs = new OffscreenCanvas(NORM_SIZE, NORM_SIZE);
    normCtx = normCvs.getContext('2d', { willReadFrequently: true });
  }
  if (!hashCvs) {
    hashCvs = new OffscreenCanvas(RECOG_AHASH_N, RECOG_AHASH_N);
    hashCtx = hashCvs.getContext('2d', { willReadFrequently: true });
  }
}

function decodeFrame(id, epoch, frame, queueLen, sharpenHint, sharpenStrength) {
  ensureWorkCanvas();
  normCtx.drawImage(frame, 0, 0, NORM_SIZE, NORM_SIZE);
  if (frame && typeof frame.close === 'function') {
    frame.close();
  }

  computeDeskewHash();

  const t0 = performance.now();
  const image = normCtx.getImageData(0, 0, NORM_SIZE, NORM_SIZE);
  const data = image.data;
  estimateColorCalibration(data);
  const buffers = ensureDecodeBuffers();
  const frames = preprocessSymbolFrame(data, buffers, sharpenHint, sharpenStrength);
  const decoded = decodeAdaptiveByQueue(data, frames, queueLen | 0);
  const headerPacked = pack6Bits(decoded.headerSymbols);
  const payloadPacked = pack6Bits(decoded.payloadSymbols);
  const ms = performance.now() - t0;

  const headerBuf = headerPacked.bytes.buffer.slice(0);
  const payloadBuf = payloadPacked.bytes.buffer.slice(0);

  self.postMessage({
    type: 'result',
    id,
    epoch,
    skipped: false,
    ms,
    avgPatternDist: decoded.avgPatternDist,
    headerTailBits: headerPacked.tailBits,
    payloadTailBits: payloadPacked.tailBits,
    headerBuf,
    payloadBuf,
  }, [headerBuf, payloadBuf]);
}

self.onmessage = (event) => {
  const data = event.data;
  if (!data) {
    return;
  }

  if (data.type === 'init') {
    try {
      if (data.dictLo && data.dictHi && data.dictLo.length === NUM_PATTERNS && data.dictHi.length === NUM_PATTERNS) {
        dict = {
          lo: new Uint32Array(data.dictLo),
          hi: new Uint32Array(data.dictHi),
        };
        dict16 = buildDict16(dict);
        dictSource = data.dictSource || 'external';
      } else {
        dict = genDict();
        dict16 = buildDict16(dict);
        dictSource = 'builtin-gen';
      }
      self.postMessage({ type: 'ready', dictSource });
    } catch (err) {
      self.postMessage({
        type: 'error',
        id: -1,
        message: err && err.message ? err.message : String(err),
      });
    }
    return;
  }

  if (data.type !== 'frame') {
    return;
  }

  try {
    decodeFrame(data.id, data.epoch | 0, data.bitmap, data.queueLen || 0, !!data.sharpenHint, Number(data.sharpenStrength) || 0);
  } catch (err) {
    if (data.bitmap) {
      try { data.bitmap.close(); } catch (_) {}
    }
    self.postMessage({
      type: 'error',
      id: data.id,
      message: err && err.message ? err.message : String(err),
    });
  }
};`;

  function toHexPreview(bytes, maxCount) {
    const out = [];
    const n = Math.min(bytes.length, maxCount);
    for (let i = 0; i < n; i++) {
      out.push(bytes[i].toString(16).padStart(2, '0'));
    }
    return out.join(' ');
  }

  const RECOG_NUM_PATTERNS = 16;

  app.maskFrom8x8Gray = function maskFrom8x8Gray(imageData) {
    let lo = 0;
    let hi = 0;
    const d = imageData.data;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const idx = (r * 8 + c) * 4;
        const gray = d[idx];
        if (gray >= 128) {
          continue;
        }
        const bit = r * 8 + c;
        if (bit < 32) {
          lo = (lo | ((1 << bit) >>> 0)) >>> 0;
        } else {
          hi = (hi | ((1 << (bit - 32)) >>> 0)) >>> 0;
        }
      }
    }
    return { lo, hi };
  };

  app.loadPatternMask = function loadPatternMask(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const cvs = document.createElement('canvas');
        cvs.width = 8;
        cvs.height = 8;
        const ctx = cvs.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(img, 0, 0, 8, 8);
        const imageData = ctx.getImageData(0, 0, 8, 8);
        resolve(app.maskFrom8x8Gray(imageData));
      };
      img.onerror = () => reject(new Error('failed to load ' + url));
      img.src = url + '?t=' + Date.now();
    });
  };

  app.loadPatternDirDict = async function loadPatternDirDict(baseUrl) {
    const lo = new Uint32Array(RECOG_NUM_PATTERNS);
    const hi = new Uint32Array(RECOG_NUM_PATTERNS);
    for (let i = 0; i < RECOG_NUM_PATTERNS; i++) {
      const name = i.toString(16).padStart(2, '0');
      const m = await app.loadPatternMask(baseUrl + '/' + name + '.png');
      lo[i] = m.lo;
      hi[i] = m.hi;
    }
    return { lo, hi };
  };

  app.getRecognizerDictConfig = function getRecognizerDictConfig() {
    const defaultMode = config && config.RECOG_DICT_MODE === 'builtin' ? 'builtin' : 'best';
    const globalMode = typeof global.__CAMDROP_RECOG_DICT_MODE === 'string' ? global.__CAMDROP_RECOG_DICT_MODE : '';
    const mode = globalMode === 'builtin' ? 'builtin' : (globalMode === 'best' ? 'best' : defaultMode);
    const baseUrl = typeof global.__CAMDROP_RECOG_DICT_BASE === 'string' && global.__CAMDROP_RECOG_DICT_BASE
      ? global.__CAMDROP_RECOG_DICT_BASE
      : ((config && typeof config.RECOG_DICT_BASE === 'string' && config.RECOG_DICT_BASE) ? config.RECOG_DICT_BASE : './best');
    return {
      mode,
      baseUrl,
      key: mode === 'builtin' ? 'builtin' : ('best|' + baseUrl),
    };
  };

  app.ensureRecognizerDict = function ensureRecognizerDict() {
    const cfg = app.getRecognizerDictConfig();
    if (state.recogDictPromise && state.recogDictConfigKey === cfg.key) {
      return state.recogDictPromise;
    }

    state.recogDictConfigKey = cfg.key;
    state.recogDictPromise = (async () => {
      if (cfg.mode === 'builtin') {
        if (state.recogDictConfigKey === cfg.key) {
          state.recogDictSource = 'builtin-gen';
        }
        return null;
      }
      try {
        const dict = await app.loadPatternDirDict(cfg.baseUrl);
        if (state.recogDictConfigKey === cfg.key) {
          state.recogDictSource = cfg.baseUrl;
        }
        return dict;
      } catch (err) {
        console.warn('[RecognizerDict] fallback to builtin-gen:', err && err.message ? err.message : err);
        if (state.recogDictConfigKey === cfg.key) {
          state.recogDictSource = 'builtin-gen';
        }
        return null;
      }
    })();
    return state.recogDictPromise;
  };

  app.updateRecognizeBar = function updateRecognizeBar(result, skipped) {
    if (!dom.decodeBar) {
      return;
    }

    const q = state.recogQueue.length;
    const pool = state.recogWorkerPoolSize || (state.recogWorkers ? state.recogWorkers.length : 0);
    const busy = state.recogActiveCount || 0;
    const idle = Math.max(0, pool - busy);

    if (!result) {
      dom.decodeBar.textContent = `识别: 队列${q} workers ${idle}/${pool} 等待首帧`;
      return;
    }

    if (skipped) {
      dom.decodeBar.textContent = `识别: 队列${q} workers ${idle}/${pool} 缓存命中 解码${state.recogDecodeCount} 跳过${state.recogSkipCount}`;
      return;
    }

    const headHex = toHexPreview(result.headerBytes, 8);
    const payloadHex = toHexPreview(result.payloadBytes, 8);
    dom.decodeBar.textContent =
      `识别: 队列${q} workers ${idle}/${pool} head ${result.headerBytes.length}B [${headHex}] payload ${result.payloadBytes.length}B [${payloadHex}]`
      + ` avgD ${result.avgPatternDist.toFixed(1)} ${result.ms.toFixed(1)}ms`;
  };

  app.reportRecognizerSkip = function reportRecognizerSkip() {
    state.recogSkipCount++;
    app.updateRecognizeBar(state.recogLastResult, true);
  };

  app.bumpRecognizerCaptureEpoch = function bumpRecognizerCaptureEpoch() {
    state.recogCaptureEpoch = (((state.recogCaptureEpoch | 0) + 1) & 0x7fffffff) || 1;
    state.recogLastHash = null;
    return state.recogCaptureEpoch;
  };

  app.getRecognizerWorkerCount = function getRecognizerWorkerCount() {
    const rawCount = Number(global.__CAMDROP_RECOG_WORKERS);
    if (!Number.isFinite(rawCount)) {
      return 1;
    }
    return Math.max(1, Math.min(8, Math.round(rawCount)));
  };

  app.findIdleRecognizerWorker = function findIdleRecognizerWorker() {
    if (!state.recogWorkers || !state.recogWorkers.length) {
      return null;
    }
    for (let i = 0; i < state.recogWorkers.length; i++) {
      const worker = state.recogWorkers[i];
      if (worker && worker.__ready && !worker.__busy) {
        return worker;
      }
    }
    return null;
  };

  app.refreshRecognizerPoolState = function refreshRecognizerPoolState() {
    let ready = 0;
    let busy = 0;
    if (state.recogWorkers && state.recogWorkers.length) {
      for (let i = 0; i < state.recogWorkers.length; i++) {
        const worker = state.recogWorkers[i];
        if (!worker) {
          continue;
        }
        if (worker.__ready) {
          ready++;
        }
        if (worker.__busy) {
          busy++;
        }
      }
    }
    state.recogWorker = state.recogWorkers && state.recogWorkers.length ? state.recogWorkers[0] : null;
    state.recogReadyWorkers = ready;
    state.recogActiveCount = busy;
    state.recogWorkerIdle = !!app.findIdleRecognizerWorker();
  };

  app.getRecognizerQueueLimit = function getRecognizerQueueLimit() {
    const override = Number(global.__CAMDROP_RECOG_QUEUE_MAX);
    const fallback = Number(config.RECOG_QUEUE_MAX) || 4;
    return Math.max(1, Math.round(Number.isFinite(override) ? override : fallback));
  };

  app.enqueueRecognizerTask = function enqueueRecognizerTask(task) {
    const limit = app.getRecognizerQueueLimit();
    if (state.recogQueue.length >= limit) {
      state.recogQueueDropCount++;
      if (task && task.bitmap) {
        try { task.bitmap.close(); } catch (_) {}
      }
      return false;
    }
    state.recogQueue.push(task);
    return true;
  };

  app.disposeRecognizerWorkers = function disposeRecognizerWorkers() {
    if (state.recogWorkers && state.recogWorkers.length) {
      for (let i = 0; i < state.recogWorkers.length; i++) {
        try {
          state.recogWorkers[i].terminate();
        } catch (_) {}
      }
    }
    if (state.recogQueue && state.recogQueue.length) {
      for (let i = 0; i < state.recogQueue.length; i++) {
        const item = state.recogQueue[i];
        if (item && item.bitmap) {
          try {
            item.bitmap.close();
          } catch (_) {}
        }
      }
    }
    state.recogQueue = [];
    state.recogWorker = null;
    state.recogWorkers = [];
    state.recogWorkerPoolSize = 0;
    state.recogReadyWorkers = 0;
    state.recogActiveCount = 0;
    state.recogWorkerIdle = true;
    state.recogWorkerDictConfigKey = '';
    state.recogLastHash = null;
    state.recogPendingResults = new Map();
    state.recogNextCommitId = Math.max(1, (state.recogSeq | 0) + 1);
    state.recogSessionId = ((state.recogSessionId | 0) + 1) | 0;
  };

  app.flushRecognizerOrderedResults = function flushRecognizerOrderedResults() {
    const pending = state.recogPendingResults;
    if (!(pending instanceof Map)) {
      return;
    }
    while (pending.has(state.recogNextCommitId)) {
      const entry = pending.get(state.recogNextCommitId);
      pending.delete(state.recogNextCommitId);
      state.recogNextCommitId++;
      app.handleRecognizerMsg.call(entry && entry.worker ? entry.worker : null, { data: entry.data });
    }
  };

  app.enqueueRecognizerOrderedResult = function enqueueRecognizerOrderedResult(worker, data) {
    const id = Number(data && data.id);
    if (!Number.isFinite(id) || id <= 0) {
      app.handleRecognizerMsg.call(worker || null, { data });
      return;
    }
    if (!(state.recogPendingResults instanceof Map)) {
      state.recogPendingResults = new Map();
    }
    if (id < state.recogNextCommitId) {
      return;
    }
    state.recogPendingResults.set(id, { worker: worker || null, data });
    app.flushRecognizerOrderedResults();
  };

  app.initRecognizerWorker = function initRecognizerWorker() {
    const targetCount = app.getRecognizerWorkerCount();
    const dictCfg = app.getRecognizerDictConfig();
    if (state.recogWorkers
        && state.recogWorkers.length === targetCount
        && state.recogWorkerDictConfigKey === dictCfg.key) {
      return;
    }

    app.disposeRecognizerWorkers();

    const blob = new Blob([WORKER_SRC], { type: 'text/javascript' });
    const workerUrl = URL.createObjectURL(blob);
    state.recogWorkers = [];
    state.recogWorkerPoolSize = targetCount;
    state.recogWorkerDictConfigKey = dictCfg.key;

    for (let i = 0; i < targetCount; i++) {
      const worker = new Worker(workerUrl);
      worker.__ready = false;
      worker.__busy = false;
      worker.__taskId = 0;
      worker.__taskEpoch = 0;
      worker.__sessionId = state.recogSessionId;
      worker.onmessage = app.handleRecognizerWorkerMsg;
      worker.onerror = app.handleRecognizerErr;
      state.recogWorkers.push(worker);
    }
    app.refreshRecognizerPoolState();

    const workerList = state.recogWorkers;
    app.ensureRecognizerDict().then((dict) => {
      if (state.recogWorkers !== workerList || state.recogWorkerDictConfigKey !== dictCfg.key) {
        return;
      }
      const msg = {
        type: 'init',
        dictSource: state.recogDictSource || 'builtin-gen',
      };
      if (dict) {
        msg.dictLo = Array.from(dict.lo);
        msg.dictHi = Array.from(dict.hi);
      }
      for (let i = 0; i < workerList.length; i++) {
        try {
          workerList[i].postMessage(msg);
        } catch (_) {}
      }
    }).catch(() => {
      if (state.recogWorkers !== workerList || state.recogWorkerDictConfigKey !== dictCfg.key) {
        return;
      }
      for (let i = 0; i < workerList.length; i++) {
        try {
          workerList[i].postMessage({ type: 'init', dictSource: 'builtin-gen' });
        } catch (_) {}
      }
    });
  };

  app.handleRecognizerMsg = function handleRecognizerMsg(event) {
    const data = event.data;
    const worker = this;

    if (data.type === 'ready') {
      if (worker) {
        worker.__ready = true;
        worker.__busy = false;
        worker.__taskId = 0;
        worker.__taskEpoch = 0;
      }
      state.recogDictSource = data.dictSource || state.recogDictSource || 'builtin-gen';
      app.refreshRecognizerPoolState();
      app.pumpRecognizeQueue();
      return;
    }

    if (data.type === 'error') {
      console.error('[RecognizeWorker]', data.message);
      app.updateRecognizeBar(state.recogLastResult, true);
      app.pumpRecognizeQueue();
      return;
    }

    if (data.type !== 'result') {
      app.pumpRecognizeQueue();
      return;
    }

    if (data.skipped) {
      state.recogSkipCount++;
      app.updateRecognizeBar(state.recogLastResult, true);
      app.pumpRecognizeQueue();
      return;
    }

    const result = {
      id: data.id,
      ms: data.ms,
      avgPatternDist: data.avgPatternDist,
      headerTailBits: data.headerTailBits,
      payloadTailBits: data.payloadTailBits,
      headerBytes: new Uint8Array(data.headerBuf),
      payloadBytes: new Uint8Array(data.payloadBuf),
    };

    state.recogMs = result.ms;
    state.recogDecodeCount++;
    state.recogLastResult = result;

    app.updateRecognizeBar(result, false);
    app.pumpRecognizeQueue();
  };

  app.handleRecognizerWorkerMsg = function handleRecognizerWorkerMsg(event) {
    const data = event && event.data;
    const worker = this;
    if (!data) {
      return;
    }

    if (data.type === 'ready') {
      app.handleRecognizerMsg.call(worker || null, event);
      return;
    }

    const sessionId = worker && Number.isFinite(worker.__sessionId) ? worker.__sessionId : -1;
    if (sessionId !== (state.recogSessionId | 0)) {
      return;
    }

    const taskId = worker && Number.isFinite(worker.__taskId) ? worker.__taskId : 0;
    const taskEpoch = worker && Number.isFinite(worker.__taskEpoch) ? worker.__taskEpoch : 0;
    if (worker) {
      worker.__busy = false;
      worker.__taskId = 0;
      worker.__taskEpoch = 0;
    }
    app.refreshRecognizerPoolState();
    app.pumpRecognizeQueue();

    if (data.type === 'error') {
      console.error('[RecognizeWorker]', data.message);
      if (taskId > 0) {
        app.enqueueRecognizerOrderedResult(worker || null, {
          type: 'result',
          id: taskId,
          skipped: true,
          message: data.message || 'worker error',
          epoch: taskEpoch,
        });
      } else {
        app.handleRecognizerMsg.call(worker || null, event);
      }
      return;
    }

    if (data.type !== 'result') {
      app.handleRecognizerMsg.call(worker || null, event);
      return;
    }

    const orderedData = (taskId > 0 || taskEpoch > 0)
      ? Object.assign({}, data, {
        id: taskId > 0 ? taskId : Number(data.id) || 0,
        epoch: taskEpoch > 0 ? taskEpoch : (Number(data.epoch) || 0),
      })
      : data;
    app.enqueueRecognizerOrderedResult(worker || null, orderedData);
  };

  app.handleRecognizerErr = function handleRecognizerErr(error) {
    const worker = this;
    console.error('[RecognizeWorker]', error.message || error);
    const taskId = worker && Number.isFinite(worker.__taskId) ? worker.__taskId : 0;
    const taskEpoch = worker && Number.isFinite(worker.__taskEpoch) ? worker.__taskEpoch : 0;
    if (worker) {
      worker.__busy = false;
      worker.__ready = false;
      worker.__taskId = 0;
      worker.__taskEpoch = 0;
    }
    app.refreshRecognizerPoolState();
    app.pumpRecognizeQueue();
    if (taskId > 0) {
      app.enqueueRecognizerOrderedResult(worker || null, {
        type: 'result',
        id: taskId,
        skipped: true,
        message: error && error.message ? error.message : 'worker crash',
        epoch: taskEpoch,
      });
    }
  };

  app.pumpRecognizeQueue = function pumpRecognizeQueue() {
    if (!state.recogWorkers || !state.recogWorkers.length) {
      return;
    }

    while (state.recogQueue.length) {
      const worker = app.findIdleRecognizerWorker();
      if (!worker) {
        break;
      }
      const item = state.recogQueue.shift();
      worker.__busy = true;
      worker.__taskId = item.id;
      worker.__taskEpoch = item.epoch | 0;
      app.refreshRecognizerPoolState();
      worker.postMessage({
        type: 'frame',
        id: item.id,
        epoch: item.epoch | 0,
        bitmap: item.bitmap,
        queueLen: state.recogQueue.length,
        sharpenHint: !!item.sharpenHint,
        sharpenStrength: item.sharpenStrength || 0,
      }, [item.bitmap]);
    }
  };

  app.captureRecognizerBitmap = function captureRecognizerBitmap() {
    if (!dom.dskCvs || !dom.dskCvs.width || !dom.dskCvs.height) {
      return Promise.reject(new Error('deskew canvas unavailable'));
    }
    const width = dom.dskCvs.width;
    const height = dom.dskCvs.height;

    if (typeof OffscreenCanvas === 'function') {
      if (!state.recogSnapCvs || state.recogSnapCvs.width !== width || state.recogSnapCvs.height !== height) {
        state.recogSnapCvs = new OffscreenCanvas(width, height);
        state.recogSnapCtx = state.recogSnapCvs.getContext('2d');
      }
      state.recogSnapCtx.clearRect(0, 0, width, height);
      state.recogSnapCtx.drawImage(dom.dskCvs, 0, 0, width, height);
      if (typeof state.recogSnapCvs.transferToImageBitmap === 'function') {
        return Promise.resolve(state.recogSnapCvs.transferToImageBitmap());
      }
      return createImageBitmap(state.recogSnapCvs);
    }

    const snap = document.createElement('canvas');
    snap.width = width;
    snap.height = height;
    const snapCtx = snap.getContext('2d');
    snapCtx.drawImage(dom.dskCvs, 0, 0, width, height);
    return createImageBitmap(snap);
  };

  app.enqueueRecognizeFrame = function enqueueRecognizeFrame() {
    if (!dom.dskCvs || !dom.dskCvs.width || !dom.dskCvs.height) {
      return;
    }

    let sharpenHint = false;
    let sharpenStrength = 0;
    if (typeof app.measureBlurScore === 'function') {
      const blurScore = app.measureBlurScore(dom.dskCvs, { marginRatio: 0.08 });
      const fineThresh = app.getBlurThreshold('fine');
      const fineBlocking = config.FINE_BLUR_BLOCKING === true;
      state.fineBlurScore = blurScore;
      if (blurScore < fineThresh) {
        if (fineBlocking) {
          app.reportRecognizerSkip();
          return;
        }
        sharpenHint = true;
        sharpenStrength = Math.max(0, Number(config.FINE_SHARPEN_STRENGTH) || 0);
      }
      const sharpenMargin = Math.max(0, Number(config.FINE_SHARPEN_MARGIN) || 0);
      if (sharpenMargin > 0 && blurScore < fineThresh + sharpenMargin) {
        sharpenHint = true;
        sharpenStrength = Math.max(sharpenStrength, Math.max(0, Number(config.FINE_SHARPEN_STRENGTH) || 0));
      }
    }

    if (typeof app.computeAHashFromSource === 'function' && typeof app.hammingDist === 'function') {
      const nextHash = app.computeAHashFromSource(dom.dskCvs, 0.08);
      const dedupeThresh = Math.max(0, Number(config.RECOG_MAIN_AHASH_THRESH) || 0);
      if (state.recogLastHash !== null && app.hammingDist(nextHash, state.recogLastHash) <= dedupeThresh) {
        app.reportRecognizerSkip();
        return;
      }
      state.recogLastHash = nextHash;
    }

    app.initRecognizerWorker();

    if (state.recogQueue.length >= app.getRecognizerQueueLimit() && !app.findIdleRecognizerWorker()) {
      state.recogQueueDropCount++;
      return;
    }

    const captureEpoch = state.recogCaptureEpoch | 0;
    app.captureRecognizerBitmap().then((bitmap) => {
      const id = (state.recogSeq | 0) + 1;
      const enqueued = app.enqueueRecognizerTask({ id, epoch: captureEpoch, bitmap, sharpenHint, sharpenStrength });
      if (!enqueued) {
        return;
      }
      state.recogSeq = id;
      app.updateRecognizeBar(state.recogLastResult, false);
      app.pumpRecognizeQueue();
    }).catch((err) => {
      console.warn('[RecognizeQueue] capture failed:', err);
    });
  };

  app.tryRecognizeDeskewed = app.enqueueRecognizeFrame;
  app.updateRecognizeBar(state.recogLastResult, false);
})(window);

