(function (global) {
  'use strict';

  const api = global.CamDropRectRecognizer = global.CamDropRectRecognizer || {};
  const INTERNAL_TO_RENDER_COLOR = [0, 1, 3, 2];

  const state = {
    workers: [],
    queue: [],
    nextTaskId: 1,
    layoutKey: '',
    dictKey: '',
    workerSource: '',
    readyPromise: null,
  };

  function defaultWorkerCount() {
    const hc = Number(global.navigator && global.navigator.hardwareConcurrency) || 4;
    return Math.max(1, Math.min(2, Math.floor(hc / 2) || 1));
  }

  function getWorkerCount() {
    const override = Number(global.__CAMDROP_RECT_RECOG_WORKERS);
    return Math.max(1, Math.min(8, Math.round(Number.isFinite(override) ? override : defaultWorkerCount())));
  }

  function getQueueLimit() {
    const override = Number(global.__CAMDROP_RECT_RECOG_QUEUE_MAX);
    return Math.max(1, Math.min(256, Math.round(Number.isFinite(override) ? override : 24)));
  }

  function getBaseWorkerSource() {
    if (typeof global.CamDropBenchRecognizerWorkerSource === 'string' && global.CamDropBenchRecognizerWorkerSource) {
      return global.CamDropBenchRecognizerWorkerSource;
    }
    if (global.CameraDropApp && typeof global.CameraDropApp.getRecognizerWorkerSource === 'function') {
      return global.CameraDropApp.getRecognizerWorkerSource();
    }
    throw new Error('bench recognizer worker source is not loaded');
  }

  function replaceOne(source, pattern, replacement, label) {
    const next = source.replace(pattern, replacement);
    if (next === source) {
      throw new Error('rect recognizer transform failed at ' + label);
    }
    return next;
  }

  function buildWorkerSource() {
    let source = String(getBaseWorkerSource() || '').replace(/\r\n/g, '\n');

    source = replaceOne(
      source,
      /const GRID_SIZE = 112;\nconst STRIDE = 9;\nconst MARGIN = 8;\nconst TILE_SIZE = 8;\nconst NUM_PATTERNS = 16;\nconst P_BITS = 4;\nconst RECOG_AHASH_N = 16;\nconst RECOG_AHASH_THRESH = 2;\nconst NORM_SIZE = 1024;/,
      [
        'let GRID_ROWS = 112;',
        'let GRID_COLS = 112;',
        'let STRIDE = 9;',
        'let MARGIN = 8;',
        'const TILE_SIZE = 8;',
        'const NUM_PATTERNS = 16;',
        'const P_BITS = 4;',
        'const RECOG_AHASH_N = 16;',
        'const RECOG_AHASH_THRESH = 2;',
        'let NORM_W = 1024;',
        'let NORM_H = 1024;',
        'let RESERVED_SIDE = 6;'
      ].join('\n'),
      'const block'
    );

    source = replaceOne(
      source,
      /function isAnchorReserved\(r, c\) \{[\s\S]*?function matchPattern\(maskLo, maskHi\) \{/,
      [
        'function isAnchorReserved(r, c) {',
        '  if (r < RESERVED_SIDE && c < RESERVED_SIDE) return true;',
        '  if (r < RESERVED_SIDE && c >= GRID_COLS - RESERVED_SIDE) return true;',
        '  if (r >= GRID_ROWS - RESERVED_SIDE && c < RESERVED_SIDE) return true;',
        '  if (r >= GRID_ROWS - RESERVED_SIDE && c >= GRID_COLS - RESERVED_SIDE) return true;',
        '  return false;',
        '}',
        '',
        'function isCalibrationCell(r, c) {',
        '  void r;',
        '  void c;',
        '  return false;',
        '}',
        '',
        'function isHeaderCell(r, c) {',
        '  void r;',
        '  void c;',
        '  return false;',
        '}',
        '',
        'function isPayloadCell(r, c) {',
        '  return !isAnchorReserved(r, c);',
        '}',
        '',
        'function matchPattern(maskLo, maskHi) {'
      ].join('\n'),
      'reserved/header block'
    );

    source = replaceOne(
      source,
      /function ensureDecodeLayout\(\) \{[\s\S]*?\n\}\n\nfunction ensureDecodeBuffers\(\) \{/,
      [
        'function ensureDecodeLayout() {',
        '  if (decodeLayout) return decodeLayout;',
        '',
        '  const xs = [];',
        '  const ys = [];',
        '  const kinds = [];',
        '  const rows = [];',
        '  const cols = [];',
        '  const rcToIdx = new Int32Array(GRID_ROWS * GRID_COLS);',
        '  rcToIdx.fill(-1);',
        '',
        '  for (let r = 0; r < GRID_ROWS; r++) {',
        '    for (let c = 0; c < GRID_COLS; c++) {',
        '      if (isAnchorReserved(r, c)) continue;',
        '      const idx = xs.length;',
        '      rows.push(r);',
        '      cols.push(c);',
        '      xs.push(MARGIN + c * STRIDE);',
        '      ys.push(MARGIN + r * STRIDE);',
        '      kinds.push(CELL_KIND_PAYLOAD);',
        '      rcToIdx[r * GRID_COLS + c] = idx;',
        '    }',
        '  }',
        '',
        '  const n = xs.length;',
        '  const neighbors = new Int32Array(n * 4);',
        '  neighbors.fill(-1);',
        '  for (let i = 0; i < n; i++) {',
        '    const r = rows[i];',
        '    const c = cols[i];',
        '    neighbors[i * 4] = (c + 1 < GRID_COLS) ? rcToIdx[r * GRID_COLS + c + 1] : -1;',
        '    neighbors[i * 4 + 1] = (c - 1 >= 0) ? rcToIdx[r * GRID_COLS + c - 1] : -1;',
        '    neighbors[i * 4 + 2] = (r + 1 < GRID_ROWS) ? rcToIdx[(r + 1) * GRID_COLS + c] : -1;',
        '    neighbors[i * 4 + 3] = (r - 1 >= 0) ? rcToIdx[(r - 1) * GRID_COLS + c] : -1;',
        '  }',
        '',
        '  const seeds = [];',
        '  const seen = new Uint8Array(n);',
        '  const pushSeed = (r, c, prio) => {',
        '    if (r < 0 || c < 0 || r >= GRID_ROWS || c >= GRID_COLS) return;',
        '    const idx = rcToIdx[r * GRID_COLS + c];',
        '    if (idx < 0 || seen[idx]) return;',
        '    seen[idx] = 1;',
        '    seeds.push(packHeapNode(idx, prio));',
        '  };',
        '  const nearTop = Math.max(0, Math.min(GRID_ROWS - 1, RESERVED_SIDE));',
        '  const nearBottom = Math.max(0, Math.min(GRID_ROWS - 1, GRID_ROWS - RESERVED_SIDE - 1));',
        '  const nearLeft = Math.max(0, Math.min(GRID_COLS - 1, RESERVED_SIDE));',
        '  const nearRight = Math.max(0, Math.min(GRID_COLS - 1, GRID_COLS - RESERVED_SIDE - 1));',
        '  pushSeed(0, nearLeft, 0);',
        '  pushSeed(0, nearRight, 0);',
        '  pushSeed(GRID_ROWS - 1, nearLeft, 0);',
        '  pushSeed(GRID_ROWS - 1, nearRight, 0);',
        '  pushSeed(nearTop, 0, 1);',
        '  pushSeed(nearTop, GRID_COLS - 1, 1);',
        '  pushSeed(nearBottom, 0, 1);',
        '  pushSeed(nearBottom, GRID_COLS - 1, 1);',
        '',
        '  decodeLayout = {',
        '    count: n,',
        '    x: Int16Array.from(xs),',
        '    y: Int16Array.from(ys),',
        '    row: Int16Array.from(rows),',
        '    col: Int16Array.from(cols),',
        '    kind: Uint8Array.from(kinds),',
        '    neighbors,',
        '    seeds,',
        '  };',
        '  return decodeLayout;',
        '}',
        '',
        'function ensureDecodeBuffers() {'
      ].join('\n'),
      'ensureDecodeLayout'
    );

    const exactReplacements = [
      ['new Uint8Array(NORM_SIZE * NORM_SIZE)', 'new Uint8Array(NORM_W * NORM_H)'],
      ['new Uint32Array((NORM_SIZE + 1) * (NORM_SIZE + 1))', 'new Uint32Array((NORM_W + 1) * (NORM_H + 1))'],
      ['clamp(x0, 0, NORM_SIZE - size)', 'clamp(x0, 0, NORM_W - size)'],
      ['clamp(y0, 0, NORM_SIZE - size)', 'clamp(y0, 0, NORM_H - size)'],
      ['((sy + y) * NORM_SIZE + sx)', '((sy + y) * NORM_W + sx)'],
      ['((sy + r) * NORM_SIZE + sx)', '((sy + r) * NORM_W + sx)'],
      ['(sy + r) * NORM_SIZE + sx', '(sy + r) * NORM_W + sx'],
      ['((y0 + pr) * NORM_SIZE + (x0 + pc))', '((y0 + pr) * NORM_W + (x0 + pc))'],
      ['clamp(x0 - 1, 0, NORM_SIZE - CELL_SAMPLE_SIZE)', 'clamp(x0 - 1, 0, NORM_W - CELL_SAMPLE_SIZE)'],
      ['clamp(y0 - 1, 0, NORM_SIZE - CELL_SAMPLE_SIZE)', 'clamp(y0 - 1, 0, NORM_H - CELL_SAMPLE_SIZE)'],
      ['clamp(x0, 0, NORM_SIZE - TILE_SIZE)', 'clamp(x0, 0, NORM_W - TILE_SIZE)'],
      ['clamp(y0, 0, NORM_SIZE - TILE_SIZE)', 'clamp(y0, 0, NORM_H - TILE_SIZE)'],
      ['const baseX = NORM_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;', 'const baseX = NORM_W - ANCHOR_OUT_START - ANCHOR_L1_SIZE;'],
      ['const baseY = NORM_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;', 'const baseY = NORM_H - ANCHOR_OUT_START - ANCHOR_L1_SIZE;'],
      ['const trBaseX = NORM_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;', 'const trBaseX = NORM_W - ANCHOR_OUT_START - ANCHOR_L1_SIZE;'],
      ['const blBaseY = NORM_SIZE - ANCHOR_OUT_START - ANCHOR_L1_SIZE;', 'const blBaseY = NORM_H - ANCHOR_OUT_START - ANCHOR_L1_SIZE;'],
      ['sharpenGray(buffers.lumaFrame, buffers.grayTemp, NORM_SIZE, NORM_SIZE, sharpenStrength);', 'sharpenGray(buffers.lumaFrame, buffers.grayTemp, NORM_W, NORM_H, sharpenStrength);'],
      ['buildIntegralGray(buffers.lumaFrame, buffers.satFrame, NORM_SIZE, NORM_SIZE);', 'buildIntegralGray(buffers.lumaFrame, buffers.satFrame, NORM_W, NORM_H);'],
      ['    NORM_SIZE,\n    NORM_SIZE,', '    NORM_W,\n    NORM_H,'],
      ['const rowDriftX = new Int8Array(GRID_SIZE);', 'const rowDriftX = new Int8Array(GRID_ROWS);'],
      ['const rowDriftY = new Int8Array(GRID_SIZE);', 'const rowDriftY = new Int8Array(GRID_ROWS);']
    ];

    exactReplacements.forEach(([from, to]) => {
      if (!source.includes(from)) {
        throw new Error('rect recognizer transform failed at ' + from);
      }
      source = source.split(from).join(to);
    });

    source = replaceOne(
      source,
      /function decodeByPriority\(data, frames\) \{([\s\S]*?)const headerSymbols = \[\];[\s\S]*?return \{\n    headerSymbols,\n    payloadSymbols,\n    avgPatternDist: patternCount > 0 \? \(sumPatternDist \/ patternCount\) : 0,\n  \};\n\}/,
      [
        'function decodeByPriority(data, frames) {$1const units = new Uint8Array(layout.count);',
        '  for (let i = 0; i < layout.count; i++) {',
        '    units[i] = buffers.symbol[i];',
        '  }',
        '  return {',
        '    units,',
        '    avgPatternDist: patternCount > 0 ? (sumPatternDist / patternCount) : 0,',
        '  };',
        '}'
      ].join('\n'),
      'decodeByPriority return'
    );

    source = replaceOne(
      source,
      /function decodeByLinearFast\(data, frames\) \{([\s\S]*?)const headerSymbols = \[\];[\s\S]*?return \{\n    headerSymbols,\n    payloadSymbols,\n    avgPatternDist: patternCount > 0 \? \(sumPatternDist \/ patternCount\) : 0,\n  \};\n\}/,
      [
        'function decodeByLinearFast(data, frames) {$1const units = new Uint8Array(layout.count);',
        '  for (let i = 0; i < layout.count; i++) {',
        '    units[i] = buffers.symbol[i];',
        '  }',
        '  return {',
        '    units,',
        '    avgPatternDist: patternCount > 0 ? (sumPatternDist / patternCount) : 0,',
        '  };',
        '}'
      ].join('\n'),
      'decodeByLinearFast return'
    );

    source = replaceOne(
      source,
      /function computeDeskewHash\(\) \{[\s\S]*?\n\}\n\nfunction ensureWorkCanvas\(\) \{/,
      [
        'function computeDeskewHash() {',
        '  const marginX = Math.round(NORM_W * 0.08);',
        '  const marginY = Math.round(NORM_H * 0.08);',
        '  const innerW = Math.max(1, NORM_W - marginX * 2);',
        '  const innerH = Math.max(1, NORM_H - marginY * 2);',
        '  hashCtx.drawImage(normCvs, marginX, marginY, innerW, innerH, 0, 0, RECOG_AHASH_N, RECOG_AHASH_N);',
        '  const d = hashCtx.getImageData(0, 0, RECOG_AHASH_N, RECOG_AHASH_N).data;',
        '',
        '  const gray = new Uint8Array(RECOG_AHASH_N * RECOG_AHASH_N);',
        '  let sum = 0;',
        '  for (let i = 0; i < gray.length; i++) {',
        '    gray[i] = (d[i * 4] * 77 + d[i * 4 + 1] * 150 + d[i * 4 + 2] * 29) >> 8;',
        '    sum += gray[i];',
        '  }',
        '',
        '  const avg = sum / gray.length;',
        '  let hash = 0n;',
        '  for (let i = 0; i < gray.length; i++) {',
        '    if (gray[i] >= avg) {',
        '      hash |= (1n << BigInt(i));',
        '    }',
        '  }',
        '',
        '  return hash;',
        '}',
        '',
        'function ensureWorkCanvas() {'
      ].join('\n'),
      'computeDeskewHash'
    );

    source = replaceOne(
      source,
      /function ensureWorkCanvas\(\) \{[\s\S]*?\n\}\n\nfunction decodeFrame\(/,
      [
        'function ensureWorkCanvas() {',
        '  if (!normCvs || normCvs.width !== NORM_W || normCvs.height !== NORM_H) {',
        '    normCvs = new OffscreenCanvas(NORM_W, NORM_H);',
        '    normCtx = normCvs.getContext(\'2d\', { willReadFrequently: true });',
        '  }',
        '  if (!hashCvs) {',
        '    hashCvs = new OffscreenCanvas(RECOG_AHASH_N, RECOG_AHASH_N);',
        '    hashCtx = hashCvs.getContext(\'2d\', { willReadFrequently: true });',
        '  }',
        '}',
        '',
        'function decodeFrame('
      ].join('\n'),
      'ensureWorkCanvas'
    );

    source = replaceOne(
      source,
      /function decodeFrame\(id, epoch, frame, queueLen, sharpenHint, sharpenStrength\) \{[\s\S]*?\nself\.onmessage = \(event\) => \{/,
      [
        'function decodeFrame(id, epoch, frame, queueLen, sharpenHint, sharpenStrength) {',
        '  ensureWorkCanvas();',
        '  normCtx.clearRect(0, 0, NORM_W, NORM_H);',
        '  normCtx.drawImage(frame, 0, 0, NORM_W, NORM_H);',
        '  if (frame && typeof frame.close === \"function\") {',
        '    frame.close();',
        '  }',
        '',
        '  computeDeskewHash();',
        '',
        '  const t0 = performance.now();',
        '  const image = normCtx.getImageData(0, 0, NORM_W, NORM_H);',
        '  const data = image.data;',
        '  estimateColorCalibration(data);',
        '  const buffers = ensureDecodeBuffers();',
        '  const frames = preprocessSymbolFrame(data, buffers, sharpenHint, sharpenStrength);',
        '  const decoded = decodeAdaptiveByQueue(data, frames, queueLen | 0);',
        '  const ms = performance.now() - t0;',
        '',
        '  const unitsBuf = decoded.units.buffer.slice(0);',
        '',
        '  self.postMessage({',
        '    type: \"result\",',
        '    id,',
        '    epoch,',
        '    skipped: false,',
        '    ms,',
        '    avgPatternDist: decoded.avgPatternDist,',
        '    unitsBuf,',
        '  }, [unitsBuf]);',
        '}',
        '',
        'self.onmessage = (event) => {'
      ].join('\n'),
      'decodeFrame'
    );

    source = replaceOne(
      source,
      /if \(data\.type === 'init'\) \{[\s\S]*?\n    return;\n  \}/,
      [
        'if (data.type === \"init\") {',
        '  try {',
        '    GRID_ROWS = Math.max(1, Math.round(Number(data.gridRows) || GRID_ROWS));',
        '    GRID_COLS = Math.max(1, Math.round(Number(data.gridCols) || GRID_COLS));',
        '    STRIDE = Math.max(1, Math.round(Number(data.stride) || STRIDE));',
        '    MARGIN = Math.max(0, Math.round(Number(data.margin) || MARGIN));',
        '    NORM_W = Math.max(64, Math.round(Number(data.normWidth) || NORM_W));',
        '    NORM_H = Math.max(64, Math.round(Number(data.normHeight) || NORM_H));',
        '    RESERVED_SIDE = Math.max(1, Math.round(Number(data.reservedSide) || RESERVED_SIDE));',
        '    decodeLayout = null;',
        '    decodeBuffers = null;',
        '    normCvs = null;',
        '    normCtx = null;',
        '    hashCvs = null;',
        '    hashCtx = null;',
        '    if (data.dictLo && data.dictHi && data.dictLo.length === NUM_PATTERNS && data.dictHi.length === NUM_PATTERNS) {',
        '      dict = {',
        '        lo: new Uint32Array(data.dictLo),',
        '        hi: new Uint32Array(data.dictHi),',
        '      };',
        '      dict16 = buildDict16(dict);',
        '      dictSource = data.dictSource || \"external\";',
        '    } else {',
        '      dict = genDict();',
        '      dict16 = buildDict16(dict);',
        '      dictSource = \"builtin-gen\";',
        '    }',
        '    self.postMessage({ type: \"ready\", dictSource, gridRows: GRID_ROWS, gridCols: GRID_COLS, normWidth: NORM_W, normHeight: NORM_H });',
        '  } catch (err) {',
        '    self.postMessage({',
        '      type: \"error\",',
        '      id: -1,',
        '      message: err && err.message ? err.message : String(err),',
        '    });',
        '  }',
        '  return;',
        '}'
      ].join('\n'),
      'init message block'
    );

    source = replaceOne(
      source,
      /function decodeFrame\(id, epoch, frame, queueLen, sharpenHint, sharpenStrength\) \{[\s\S]*?\n\}\n\nself\.onmessage = \(event\) => \{/,
      [
        'function postDecodedResult(id, epoch, decoded, ms) {',
        '  const unitsBuf = decoded.units.buffer.slice(0);',
        '  self.postMessage({',
        '    type: "result",',
        '    id,',
        '    epoch,',
        '    skipped: false,',
        '    ms,',
        '    avgPatternDist: decoded.avgPatternDist,',
        '    unitsBuf,',
        '  }, [unitsBuf]);',
        '}',
        '',
        'function decodeNormalizedRgba(data, queueLen, sharpenHint, sharpenStrength) {',
        '  estimateColorCalibration(data);',
        '  const buffers = ensureDecodeBuffers();',
        '  const frames = preprocessSymbolFrame(data, buffers, sharpenHint, sharpenStrength);',
        '  return decodeAdaptiveByQueue(data, frames, queueLen | 0);',
        '}',
        '',
        'function decodeFrame(id, epoch, frame, queueLen, sharpenHint, sharpenStrength) {',
        '  ensureWorkCanvas();',
        '  normCtx.clearRect(0, 0, NORM_W, NORM_H);',
        '  normCtx.drawImage(frame, 0, 0, NORM_W, NORM_H);',
        '  if (frame && typeof frame.close === "function") {',
        '    frame.close();',
        '  }',
        '  computeDeskewHash();',
        '  const t0 = performance.now();',
        '  const image = normCtx.getImageData(0, 0, NORM_W, NORM_H);',
        '  const decoded = decodeNormalizedRgba(image.data, queueLen, sharpenHint, sharpenStrength);',
        '  const ms = performance.now() - t0;',
        '  postDecodedResult(id, epoch, decoded, ms);',
        '}',
        '',
        'function decodeFromRgba(id, epoch, rgbaBuf, srcWidth, srcHeight, queueLen, sharpenHint, sharpenStrength) {',
        '  const width = Math.max(1, Math.round(Number(srcWidth) || 0));',
        '  const height = Math.max(1, Math.round(Number(srcHeight) || 0));',
        '  if (width !== NORM_W || height !== NORM_H) {',
        '    throw new Error("rgba payload size mismatch");',
        '  }',
        '  const data = new Uint8ClampedArray(rgbaBuf);',
        '  if (data.length !== width * height * 4) {',
        '    throw new Error("rgba payload length mismatch");',
        '  }',
        '  ensureWorkCanvas();',
        '  const image = normCtx.createImageData(width, height);',
        '  image.data.set(data);',
        '  normCtx.putImageData(image, 0, 0);',
        '  computeDeskewHash();',
        '  const t0 = performance.now();',
        '  const decoded = decodeNormalizedRgba(data, queueLen, sharpenHint, sharpenStrength);',
        '  const ms = performance.now() - t0;',
        '  postDecodedResult(id, epoch, decoded, ms);',
        '}',
        '',
        'self.onmessage = (event) => {'
      ].join('\n'),
      'rgba decode path'
    );

    source = replaceOne(
      source,
      /if \(data\.type !== 'frame'\) \{[\s\S]*?\n\};/,
      [
        'if (data.type !== "frame" && data.type !== "rgba") {',
        '  return;',
        '}',
        '',
        'try {',
        '  if (data.type === "rgba") {',
        '    decodeFromRgba(data.id, data.epoch | 0, data.rgbaBuf, data.srcWidth, data.srcHeight, data.queueLen || 0, !!data.sharpenHint, Number(data.sharpenStrength) || 0);',
        '  } else {',
        '    decodeFrame(data.id, data.epoch | 0, data.frame || data.bitmap, data.queueLen || 0, !!data.sharpenHint, Number(data.sharpenStrength) || 0);',
        '  }',
        '} catch (err) {',
        '  if (data.frame && typeof data.frame.close === "function") {',
        '    try { data.frame.close(); } catch (_) {}',
        '  }',
        '  if (data.bitmap && typeof data.bitmap.close === "function") {',
        '    try { data.bitmap.close(); } catch (_) {}',
        '  }',
        '  self.postMessage({',
        '    type: "error",',
        '    id: data.id,',
        '    message: err && err.message ? err.message : String(err),',
        '  });',
        '}',
        '};'
      ].join('\n'),
      'rgba dispatch'
    );

    if (source.includes('GRID_SIZE')) {
      throw new Error('GRID_SIZE still remains after rect transform');
    }
    if (source.includes('NORM_SIZE')) {
      throw new Error('NORM_SIZE still remains after rect transform');
    }

    return source;
  }

  function bigIntMaskToWords(mask) {
    const value = typeof mask === 'bigint' ? mask : BigInt(mask || 0);
    const lo = Number(value & 0xffffffffn) >>> 0;
    const hi = Number((value >> 32n) & 0xffffffffn) >>> 0;
    return { lo, hi };
  }

  function buildPatternDict() {
    const patterns = global.CamDropRectRender && Array.isArray(global.CamDropRectRender.PATTERNS)
      ? global.CamDropRectRender.PATTERNS
      : null;
    if (!patterns || patterns.length !== 16) {
      throw new Error('CamDropRectRender.PATTERNS is unavailable');
    }
    const lo = new Uint32Array(patterns.length);
    const hi = new Uint32Array(patterns.length);
    for (let i = 0; i < patterns.length; i++) {
      const words = bigIntMaskToWords(patterns[i]);
      lo[i] = words.lo;
      hi[i] = words.hi;
    }
    return { lo, hi, key: Array.from(lo).join(',') + '|' + Array.from(hi).join(',') };
  }

  function layoutKeyOf(layout) {
    return [
      layout && layout.imgWidth,
      layout && layout.imgHeight,
      layout && layout.gridRows,
      layout && layout.gridCols,
      layout && layout.stride,
      layout && layout.margin,
      layout && layout.reservedCornerSide,
      getWorkerCount(),
    ].join('|');
  }

  function closeBitmap(bitmap) {
    if (bitmap && typeof bitmap.close === 'function') {
      try { bitmap.close(); } catch (_) {}
    }
  }

  function cleanupTaskPayload(task) {
    if (!task) {
      return;
    }
    if (task.kind === 'bitmap') {
      closeBitmap(task.bitmap);
      task.bitmap = null;
      return;
    }
    if (task.kind === 'rgba') {
      task.rgbaBuf = null;
    }
  }

  function disposeWorkers() {
    for (let i = 0; i < state.workers.length; i++) {
      const worker = state.workers[i];
      if (!worker) continue;
      if (worker.__task) {
        cleanupTaskPayload(worker.__task);
        worker.__task.reject(new Error('rect recognizer disposed'));
        worker.__task = null;
      }
      try { worker.terminate(); } catch (_) {}
    }
    while (state.queue.length) {
      const task = state.queue.shift();
      if (!task) continue;
      cleanupTaskPayload(task);
      task.reject(new Error('rect recognizer disposed'));
    }
    state.workers = [];
    state.readyPromise = null;
    state.layoutKey = '';
    state.dictKey = '';
  }

  function findIdleWorker() {
    for (let i = 0; i < state.workers.length; i++) {
      const worker = state.workers[i];
      if (worker && worker.__ready && !worker.__task) {
        return worker;
      }
    }
    return null;
  }

  function pumpQueue() {
    while (state.queue.length) {
      const worker = findIdleWorker();
      if (!worker) {
        return;
      }
      const task = state.queue.shift();
      if (!task) continue;
      worker.__task = task;
      if (task.kind === 'rgba') {
        worker.postMessage({
          type: 'rgba',
          id: task.id,
          epoch: 1,
          queueLen: state.queue.length,
          sharpenHint: !!task.sharpenHint,
          sharpenStrength: Number(task.sharpenStrength) || 0,
          srcWidth: task.srcWidth,
          srcHeight: task.srcHeight,
          rgbaBuf: task.rgbaBuf,
        }, [task.rgbaBuf]);
        task.rgbaBuf = null;
        continue;
      }
      worker.postMessage({
        type: 'frame',
        id: task.id,
        epoch: 1,
        queueLen: state.queue.length,
        sharpenHint: !!task.sharpenHint,
        sharpenStrength: Number(task.sharpenStrength) || 0,
        frame: task.bitmap,
      }, [task.bitmap]);
      task.bitmap = null;
    }
  }

  async function ensureWorkers(layout) {
    const dict = buildPatternDict();
    const nextLayoutKey = layoutKeyOf(layout);
    if (!state.workerSource) {
      state.workerSource = buildWorkerSource();
    }
    if (state.readyPromise && state.layoutKey === nextLayoutKey && state.dictKey === dict.key) {
      return state.readyPromise;
    }

    disposeWorkers();
    state.layoutKey = nextLayoutKey;
    state.dictKey = dict.key;

    const workerCount = getWorkerCount();
    const blob = new Blob([state.workerSource], { type: 'text/javascript' });
    const workerUrl = URL.createObjectURL(blob);
    state.readyPromise = new Promise((resolve, reject) => {
      let readyCount = 0;
      let settled = false;
      const finishReject = (error) => {
        if (settled) return;
        settled = true;
        try { URL.revokeObjectURL(workerUrl); } catch (_) {}
        reject(error);
      };
      const finishResolve = () => {
        if (settled) return;
        settled = true;
        try { URL.revokeObjectURL(workerUrl); } catch (_) {}
        resolve();
      };

      for (let i = 0; i < workerCount; i++) {
        const worker = new Worker(workerUrl);
        worker.__ready = false;
        worker.__task = null;
        worker.onmessage = function (event) {
          const data = event && event.data ? event.data : null;
          if (!data) {
            return;
          }
          if (data.type === 'ready') {
            worker.__ready = true;
            readyCount++;
            if (readyCount >= workerCount) {
              finishResolve();
              pumpQueue();
            }
            return;
          }
          if (data.type === 'error') {
            if (!worker.__ready) {
              finishReject(new Error(data.message || 'rect recognizer init failed'));
              return;
            }
            const task = worker.__task;
            worker.__task = null;
            if (task) {
              task.reject(new Error(data.message || 'rect recognizer worker error'));
            }
            pumpQueue();
            return;
          }
          if (data.type === 'result') {
            const task = worker.__task;
            worker.__task = null;
            if (task) {
              task.resolve(data);
            }
            pumpQueue();
          }
        };
        worker.onerror = function (event) {
          const error = new Error(event && event.message ? event.message : 'rect recognizer worker crashed');
          if (!worker.__ready) {
            finishReject(error);
            return;
          }
          const task = worker.__task;
          worker.__task = null;
          if (task) {
            task.reject(error);
          }
          pumpQueue();
        };
        state.workers.push(worker);
        worker.postMessage({
          type: 'init',
          dictSource: 'rect-render',
          dictLo: Array.from(dict.lo),
          dictHi: Array.from(dict.hi),
          gridRows: layout.gridRows,
          gridCols: layout.gridCols,
          stride: layout.stride,
          margin: layout.margin,
          reservedSide: layout.reservedCornerSide,
          normWidth: layout.imgWidth,
          normHeight: layout.imgHeight,
        });
      }
    });

    return state.readyPromise;
  }

  function remapUnitsInternalToRender(units) {
    const out = new Uint8Array(units.length);
    for (let i = 0; i < units.length; i++) {
      const value = units[i] & 0x3f;
      const pattern = value & 0x0f;
      const color = (value >> 4) & 0x03;
      out[i] = ((INTERNAL_TO_RENDER_COLOR[color] & 0x03) << 4) | pattern;
    }
    return out;
  }

  let readbackCanvas = null;
  let readbackCtx = null;

  function ensureReadbackCanvas(width, height) {
    if (!readbackCanvas || readbackCanvas.width !== width || readbackCanvas.height !== height) {
      if (typeof OffscreenCanvas === 'function') {
        readbackCanvas = new OffscreenCanvas(width, height);
      } else {
        readbackCanvas = document.createElement('canvas');
        readbackCanvas.width = width;
        readbackCanvas.height = height;
      }
      readbackCtx = readbackCanvas.getContext('2d', { willReadFrequently: true });
    }
    return readbackCtx;
  }

  async function createBitmapFromCanvas(canvas) {
    if (!canvas) {
      throw new Error('canonical canvas is unavailable');
    }
    if (typeof createImageBitmap === 'function') {
      return await createImageBitmap(canvas);
    }
    throw new Error('createImageBitmap is not available');
  }

  function extractRgbaFromCanvas(canvas, layout) {
    if (!canvas) {
      throw new Error('canonical canvas is unavailable');
    }
    const width = Math.max(1, Math.round((layout && layout.imgWidth) || canvas.width || 0));
    const height = Math.max(1, Math.round((layout && layout.imgHeight) || canvas.height || 0));
    const ctx = ensureReadbackCanvas(width, height);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(canvas, 0, 0, width, height);
    const rgba = ctx.getImageData(0, 0, width, height).data;
    return {
      rgbaBuf: rgba.buffer.slice(0),
      srcWidth: width,
      srcHeight: height,
    };
  }

  async function decodeBitmap(bitmap, options) {
    const layout = await global.CamDropRectCodec.getLayout();
    await ensureWorkers(layout);
    if (state.queue.length >= getQueueLimit() && !findIdleWorker()) {
      closeBitmap(bitmap);
      throw new Error('rect recognizer queue is full');
    }
    const result = await new Promise((resolve, reject) => {
      state.queue.push({
        id: state.nextTaskId++,
        kind: 'bitmap',
        bitmap,
        sharpenHint: !!(options && options.sharpenHint),
        sharpenStrength: Number(options && options.sharpenStrength) || 0,
        resolve,
        reject,
      });
      pumpQueue();
    });
    const rawUnits = new Uint8Array(result.unitsBuf || new ArrayBuffer(0));
    const units = remapUnitsInternalToRender(rawUnits);
    const packetBytes = await global.CamDropRectCodec.unitsToPacket(units);
    return {
      layout,
      units,
      packetBytes,
      avgPatternDist: Number(result.avgPatternDist) || 0,
      ms: Number(result.ms) || 0,
    };
  }


  async function decodeRgba(rgbaPayload, options) {
    const layout = await global.CamDropRectCodec.getLayout();
    await ensureWorkers(layout);
    if (!rgbaPayload || !(rgbaPayload.rgbaBuf instanceof ArrayBuffer)) {
      throw new Error('rgba payload is unavailable');
    }
    if (state.queue.length >= getQueueLimit() && !findIdleWorker()) {
      throw new Error('rect recognizer queue is full');
    }
    const result = await new Promise((resolve, reject) => {
      state.queue.push({
        id: state.nextTaskId++,
        kind: 'rgba',
        rgbaBuf: rgbaPayload.rgbaBuf,
        srcWidth: rgbaPayload.srcWidth,
        srcHeight: rgbaPayload.srcHeight,
        sharpenHint: !!(options && options.sharpenHint),
        sharpenStrength: Number(options && options.sharpenStrength) || 0,
        resolve,
        reject,
      });
      pumpQueue();
    });
    const rawUnits = new Uint8Array(result.unitsBuf || new ArrayBuffer(0));
    const units = remapUnitsInternalToRender(rawUnits);
    const packetBytes = await global.CamDropRectCodec.unitsToPacket(units);
    return {
      layout,
      units,
      packetBytes,
      avgPatternDist: Number(result.avgPatternDist) || 0,
      ms: Number(result.ms) || 0,
    };
  }

  api.dispose = function dispose() {
    disposeWorkers();
  };

  api.decodeCanonicalBitmap = async function decodeCanonicalBitmap(bitmap, options) {
    return await decodeBitmap(bitmap, options || null);
  };

  api.decodeCanonicalCanvas = async function decodeCanonicalCanvas(canvas, options) {
    const layout = await global.CamDropRectCodec.getLayout();
    const rgbaPayload = extractRgbaFromCanvas(canvas, layout);
    return await decodeRgba(rgbaPayload, options || null);
  };

  api.warpSceneToCanvas = async function warpSceneToCanvas(sceneCanvas, corners, outCanvas) {
    if (!global.cv || !cv.Mat) {
      throw new Error('opencv is not ready');
    }
    const layout = await global.CamDropRectCodec.getLayout();
    const dstCanvas = outCanvas || document.createElement('canvas');
    dstCanvas.width = layout.imgWidth;
    dstCanvas.height = layout.imgHeight;

    const src = cv.imread(sceneCanvas);
    const dst = new cv.Mat();
    const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
      corners.TL[0], corners.TL[1],
      corners.TR[0], corners.TR[1],
      corners.BR[0], corners.BR[1],
      corners.BL[0], corners.BL[1],
    ]);
    const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      layout.imgWidth - 1, 0,
      layout.imgWidth - 1, layout.imgHeight - 1,
      0, layout.imgHeight - 1,
    ]);
    const M = cv.getPerspectiveTransform(srcTri, dstTri);
    cv.warpPerspective(src, dst, M, new cv.Size(layout.imgWidth, layout.imgHeight), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
    cv.imshow(dstCanvas, dst);
    src.delete();
    dst.delete();
    srcTri.delete();
    dstTri.delete();
    M.delete();
    return dstCanvas;
  };
})(window);
