(function (global) {
  'use strict';

  const api = global.CamDropRectRender = global.CamDropRectRender || {};

  const BUILTIN_PATTERNS = [
    72909780498219007n,
    18410503204342530817n,
    9277662557957324543n,
    18446459269879480448n,
    4359176317332586044n,
    18446601679374391320n,
    18392435895879811071n,
    6821915192918016n,
    3474557959208187952n,
    7914543405472768n,
    18446744073692774400n,
    17361641481138401520n,
    1085102596360827120n,
    17361641477348724495n,
    18138078526599232n,
    1099511627775n,
  ];
  const PATTERN_COUNT = 16;
  const TILE_SIZE = 8;

  const COLORS = [
    'rgb(255,255,0)',
    'rgb(0,255,0)',
    'rgb(255,0,255)',
    'rgb(0,255,255)',
  ];

  let patterns = BUILTIN_PATTERNS.slice(0);
  let patternSource = 'builtin';
  let patternDict = null;
  let patternPromise = null;
  let patternCacheKey = '';
  let readCanvas = null;
  let readCtx = null;

  function assetVersion() {
    if (typeof global.__CAMDROP_ASSET_VERSION === 'string' && global.__CAMDROP_ASSET_VERSION) {
      return global.__CAMDROP_ASSET_VERSION;
    }
    return '';
  }

  function withAssetVersion(url) {
    const version = assetVersion();
    if (!version) {
      return url;
    }
    return url + (url.indexOf('?') >= 0 ? '&' : '?') + 'v=' + encodeURIComponent(version);
  }

  function getPatternBase() {
    if (typeof global.__CAMDROP_RECT_PATTERN_BASE === 'string' && global.__CAMDROP_RECT_PATTERN_BASE) {
      return global.__CAMDROP_RECT_PATTERN_BASE.replace(/\/$/, '');
    }
    return typeof document === 'undefined' ? '../best_v2' : './best_v2';
  }

  function getPatternCacheKey() {
    return getPatternBase() + '|' + assetVersion();
  }

  function ensureReadContext() {
    if (readCtx) {
      return readCtx;
    }
    if (typeof OffscreenCanvas !== 'undefined') {
      readCanvas = new OffscreenCanvas(TILE_SIZE, TILE_SIZE);
    } else {
      readCanvas = document.createElement('canvas');
      readCanvas.width = TILE_SIZE;
      readCanvas.height = TILE_SIZE;
    }
    readCtx = readCanvas.getContext('2d', { willReadFrequently: true });
    return readCtx;
  }

  async function loadBitmap(url) {
    if (typeof fetch === 'function' && typeof createImageBitmap === 'function') {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error('failed to load ' + url + ': ' + response.status);
      }
      const blob = await response.blob();
      return await createImageBitmap(blob);
    }

    if (typeof document !== 'undefined') {
      return await new Promise(function (resolve, reject) {
        const img = new Image();
        img.onload = function () { resolve(img); };
        img.onerror = function () { reject(new Error('failed to load ' + url)); };
        img.src = url;
      });
    }

    throw new Error('pattern image loading is unavailable');
  }

  function closeBitmap(bitmap) {
    if (bitmap && typeof bitmap.close === 'function') {
      try {
        bitmap.close();
      } catch (_) {}
    }
  }

  async function loadPatternMask(url) {
    const bitmap = await loadBitmap(url);
    try {
      const ctx = ensureReadContext();
      ctx.clearRect(0, 0, TILE_SIZE, TILE_SIZE);
      ctx.drawImage(bitmap, 0, 0, TILE_SIZE, TILE_SIZE);
      const data = ctx.getImageData(0, 0, TILE_SIZE, TILE_SIZE).data;
      let mask = 0n;
      for (let i = 0; i < TILE_SIZE * TILE_SIZE; i++) {
        const gray = (data[i * 4] * 77 + data[i * 4 + 1] * 150 + data[i * 4 + 2] * 29) >> 8;
        if (gray < 128) {
          mask |= (1n << BigInt(i));
        }
      }
      return mask;
    } finally {
      closeBitmap(bitmap);
    }
  }

  function maskToWords(mask) {
    const value = typeof mask === 'bigint' ? mask : BigInt(mask || 0);
    return {
      lo: Number(value & 0xffffffffn) >>> 0,
      hi: Number((value >> 32n) & 0xffffffffn) >>> 0,
    };
  }

  function rebuildPatternDict() {
    const lo = new Uint32Array(patterns.length);
    const hi = new Uint32Array(patterns.length);
    for (let i = 0; i < patterns.length; i++) {
      const words = maskToWords(patterns[i]);
      lo[i] = words.lo;
      hi[i] = words.hi;
    }
    patternDict = {
      lo,
      hi,
      key: Array.from(lo).join(',') + '|' + Array.from(hi).join(','),
      source: patternSource,
    };
    api.PATTERNS = patterns.slice(0);
  }

  async function ensurePatterns() {
    const nextKey = getPatternCacheKey();
    if (patternPromise && patternCacheKey === nextKey) {
      return patternPromise;
    }

    patternCacheKey = nextKey;
    patternPromise = (async function () {
      const base = getPatternBase();
      try {
        const loaded = await Promise.all(
          Array.from({ length: PATTERN_COUNT }, function (_, i) {
            const name = i.toString(16).padStart(2, '0');
            return loadPatternMask(withAssetVersion(base + '/' + name + '.png'));
          })
        );
        patterns = loaded.slice(0);
        patternSource = base;
      } catch (error) {
        console.warn('[RectRender] pattern load fallback to builtin:', error && error.message ? error.message : error);
        patterns = BUILTIN_PATTERNS.slice(0);
        patternSource = 'builtin';
      }
      rebuildPatternDict();
      return api.PATTERNS;
    })();

    return patternPromise;
  }

  function isReservedCell(layout, r, c) {
    const side = layout && Number.isFinite(layout.reservedCornerSide) ? layout.reservedCornerSide : 6;
    if (r < side && c < side) return true;
    if (r < side && c >= layout.gridCols - side) return true;
    if (r >= layout.gridRows - side && c < side) return true;
    if (r >= layout.gridRows - side && c >= layout.gridCols - side) return true;
    return false;
  }

  function drawNormalAnchor(ctx, x0, y0, scale) {
    const l1 = 56 * scale;
    const l2Inset = 7 * scale;
    const l2 = 42 * scale;
    const l3Inset = 14 * scale;
    const l3 = 28 * scale;
    const l4Inset = 21 * scale;
    const l4 = 14 * scale;
    ctx.fillStyle = '#fff';
    ctx.fillRect(x0, y0, l1, l1);
    ctx.fillStyle = '#000';
    ctx.fillRect(x0 + l2Inset, y0 + l2Inset, l2, l2);
    ctx.fillStyle = '#fff';
    ctx.fillRect(x0 + l3Inset, y0 + l3Inset, l3, l3);
    ctx.fillStyle = '#000';
    ctx.fillRect(x0 + l4Inset, y0 + l4Inset, l4, l4);
  }

  function drawBrAnchor(ctx, layout, scale) {
    const x0 = (layout.imgWidth - 2 - 56) * scale;
    const y0 = (layout.imgHeight - 2 - 56) * scale;
    const h1 = 28 * scale;
    const h3 = 14 * scale;
    ctx.fillStyle = COLORS[0];
    ctx.fillRect(x0, y0, h1, h1);
    ctx.fillStyle = COLORS[1];
    ctx.fillRect(x0 + h1, y0, h1, h1);
    ctx.fillStyle = COLORS[2];
    ctx.fillRect(x0, y0 + h1, h1, h1);
    ctx.fillStyle = COLORS[3];
    ctx.fillRect(x0 + h1, y0 + h1, h1, h1);
    ctx.fillStyle = '#000';
    ctx.fillRect(x0 + 7 * scale, y0 + 7 * scale, 42 * scale, 42 * scale);
    ctx.fillStyle = COLORS[0];
    ctx.fillRect(x0 + 14 * scale, y0 + 14 * scale, h3, h3);
    ctx.fillStyle = COLORS[1];
    ctx.fillRect(x0 + 28 * scale, y0 + 14 * scale, h3, h3);
    ctx.fillStyle = COLORS[2];
    ctx.fillRect(x0 + 14 * scale, y0 + 28 * scale, h3, h3);
    ctx.fillStyle = COLORS[3];
    ctx.fillRect(x0 + 28 * scale, y0 + 28 * scale, h3, h3);
    ctx.fillStyle = '#000';
    ctx.fillRect(x0 + 21 * scale, y0 + 21 * scale, 14 * scale, 14 * scale);
  }

  function drawAnchors(ctx, layout, scale) {
    drawNormalAnchor(ctx, 2 * scale, 2 * scale, scale);
    drawNormalAnchor(ctx, (layout.imgWidth - 2 - 56) * scale, 2 * scale, scale);
    drawNormalAnchor(ctx, 2 * scale, (layout.imgHeight - 2 - 56) * scale, scale);
    drawBrAnchor(ctx, layout, scale);
  }

  function drawCell(ctx, layout, scale, row, col, value) {
    const patternIdx = value & 0x0f;
    const colorIdx = (value >> 4) & 0x03;
    const mask = patterns[patternIdx];
    const x = (layout.margin + col * layout.stride) * scale;
    const y = (layout.margin + row * layout.stride) * scale;
    ctx.fillStyle = COLORS[colorIdx];
    for (let pr = 0; pr < 8; pr++) {
      for (let pc = 0; pc < 8; pc++) {
        const bit = BigInt(pr * 8 + pc);
        if (((mask >> bit) & 1n) !== 0n) {
          ctx.fillRect(x + pc * scale, y + pr * scale, scale, scale);
        }
      }
    }
  }

  rebuildPatternDict();

  api.PATTERNS = patterns.slice(0);
  api.COLORS = COLORS.slice(0);
  api.isReservedCell = isReservedCell;
  api.ensurePatterns = ensurePatterns;
  api.getPatternSource = function getPatternSource() {
    return patternSource;
  };
  api.getPatternDict = async function getPatternDict() {
    await ensurePatterns();
    return {
      lo: new Uint32Array(patternDict.lo),
      hi: new Uint32Array(patternDict.hi),
      key: patternDict.key,
      source: patternDict.source,
    };
  };

  function cooperativeYield() {
    if (global.scheduler && typeof global.scheduler.yield === 'function') {
      return global.scheduler.yield();
    }
    return new Promise(function (resolve) {
      global.setTimeout(resolve, 0);
    });
  }

  api.renderUnitsToCanvas = async function renderUnitsToCanvas(canvas, units, options) {
    const cfg = options || {};
    const layout = await global.CamDropRectCodec.getLayout();
    await ensurePatterns();
    const scale = Math.max(1, cfg.scale || 1);
    const data = units instanceof Uint8Array ? units : new Uint8Array(units || 0);
    const useCooperativeYield = cfg.cooperativeYield === true;
    const yieldEveryRows = Math.max(0, Math.round(Number(cfg.yieldEveryRows) || 0));
    const yieldFn = typeof cfg.yieldFn === 'function' ? cfg.yieldFn : cooperativeYield;
    if (data.length !== layout.unitCount) {
      throw new Error('unit count mismatch: expected ' + layout.unitCount + ', got ' + data.length);
    }
    canvas.width = layout.imgWidth * scale;
    canvas.height = layout.imgHeight * scale;
    const ctx = canvas.getContext('2d', { willReadFrequently: false });
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawAnchors(ctx, layout, scale);
    let index = 0;
    for (let r = 0; r < layout.gridRows; r++) {
      for (let c = 0; c < layout.gridCols; c++) {
        if (isReservedCell(layout, r, c)) {
          continue;
        }
        drawCell(ctx, layout, scale, r, c, data[index++]);
      }
      if (useCooperativeYield && yieldEveryRows > 0 && (r + 1) < layout.gridRows && ((r + 1) % yieldEveryRows) === 0) {
        await yieldFn();
      }
    }
    return canvas;
  };

  api.renderPacketToCanvas = async function renderPacketToCanvas(canvas, packetBytes, options) {
    const units = await global.CamDropRectCodec.packetToUnits(packetBytes);
    return api.renderUnitsToCanvas(canvas, units, options);
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
