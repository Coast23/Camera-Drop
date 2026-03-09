(function (global) {
  const api = global.CamDropRectTransferCommon = global.CamDropRectTransferCommon || {};

  const QUERY_KEYS = {
    imgWidth: ['w', 'width', 'imgWidth'],
    imgHeight: ['h', 'height', 'imgHeight'],
    stride: ['stride'],
    margin: ['margin'],
    reservedCornerSide: ['reserved', 'reservedCornerSide'],
  };

  function firstParam(params, keys) {
    for (let i = 0; i < keys.length; i++) {
      const value = params.get(keys[i]);
      if (value != null && value !== '') {
        return value;
      }
    }
    return null;
  }

  api.clampInt = function clampInt(value, lo, hi, fallback) {
    const n = Number(value);
    if (!Number.isFinite(n)) {
      return fallback;
    }
    const i = Math.round(n);
    if (i < lo) return lo;
    if (i > hi) return hi;
    return i;
  };

  api.formatBytes = function formatBytes(bytes) {
    const n = Math.max(0, Number(bytes) || 0);
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    if (n < 1024 * 1024 * 1024) return (n / (1024 * 1024)).toFixed(2) + ' MB';
    return (n / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  };

  api.formatPercent = function formatPercent(value, digits) {
    const n = Number(value);
    if (!Number.isFinite(n)) {
      return '-';
    }
    return n.toFixed(Number.isFinite(digits) ? digits : 1) + '%';
  };

  api.getCapacityInfo = function getCapacityInfo(layout) {
    if (!layout) {
      return null;
    }
    const rsDataSize = Math.max(1, Number(layout.rsDataSize) || 1);
    const rsBlockSize = Math.max(1, Number(layout.rsBlockSize) || 1);
    const packetBytes = Math.max(0, Number(layout.packetBytes) || 0);
    const encodedPacketBytes = Math.max(packetBytes, Number(layout.encodedPacketBytes) || 0);
    const fountainPayloadSize = Math.max(0, Number(layout.fountainPayloadSize) || 0);
    const fountainChunkSize = Math.max(0, Number(layout.fountainChunkSize) || 0);
    const rsBlocksPerChunk = fountainPayloadSize > 0 ? Math.max(1, Math.round(fountainPayloadSize / rsDataSize)) : 0;
    const chunksPerFrame = rsBlocksPerChunk > 0
      ? Math.max(1, Math.round(encodedPacketBytes / (rsBlocksPerChunk * rsBlockSize)))
      : 0;
    return {
      packetBytes,
      encodedPacketBytes,
      fountainPayloadSize,
      fountainChunkSize,
      rsBlocksPerChunk,
      chunksPerFrame,
      fileBytesPerFrame: chunksPerFrame * fountainChunkSize,
    };
  };

  api.formatLayout = function formatLayout(layout) {
    if (!layout) {
      return '-';
    }
    const capacity = api.getCapacityInfo(layout);
    return layout.imgWidth + 'x' + layout.imgHeight +
      ' | grid ' + layout.gridCols + 'x' + layout.gridRows +
      ' | stride ' + layout.stride +
      ' | margin ' + layout.margin +
      ' | reserved ' + layout.reservedCornerSide +
      ' | packet ' + layout.packetBytes + 'B' +
      (capacity && capacity.fileBytesPerFrame ? (' | file ' + capacity.fileBytesPerFrame + 'B') : '');
  };

  api.readLayoutFromQuery = function readLayoutFromQuery(search) {
    const params = search instanceof URLSearchParams ? search : new URLSearchParams(search || global.location.search || '');
    const out = {};
    Object.keys(QUERY_KEYS).forEach((field) => {
      const raw = firstParam(params, QUERY_KEYS[field]);
      if (raw != null) {
        const n = Number(raw);
        if (Number.isFinite(n)) {
          out[field] = Math.round(n);
        }
      }
    });
    return out;
  };

  api.writeLayoutInputs = function writeLayoutInputs(dom, layout) {
    if (!dom || !layout) {
      return;
    }
    if (dom.widthInput) dom.widthInput.value = layout.imgWidth;
    if (dom.heightInput) dom.heightInput.value = layout.imgHeight;
    if (dom.strideInput) dom.strideInput.value = layout.stride;
    if (dom.marginInput) dom.marginInput.value = layout.margin;
    if (dom.reservedInput) dom.reservedInput.value = layout.reservedCornerSide;
  };

  api.readLayoutInputs = function readLayoutInputs(dom) {
    return {
      imgWidth: api.clampInt(dom && dom.widthInput && dom.widthInput.value, 64, 10000, 2536),
      imgHeight: api.clampInt(dom && dom.heightInput && dom.heightInput.value, 64, 10000, 1456),
      stride: api.clampInt(dom && dom.strideInput && dom.strideInput.value, 1, 128, 9),
      margin: api.clampInt(dom && dom.marginInput && dom.marginInput.value, 0, 512, 8),
      reservedCornerSide: api.clampInt(dom && dom.reservedInput && dom.reservedInput.value, 1, 64, 6),
    };
  };

  api.applyLayout = async function applyLayout(layoutLike) {
    const opts = layoutLike || {};
    return await global.CamDropRectCodec.configureLayout({
      imgWidth: opts.imgWidth,
      imgHeight: opts.imgHeight,
      stride: opts.stride,
      margin: opts.margin,
      reservedCornerSide: opts.reservedCornerSide,
    });
  };

  api.applyLayoutInputs = async function applyLayoutInputs(dom) {
    return await api.applyLayout(api.readLayoutInputs(dom));
  };

  api.makeLayoutQuery = function makeLayoutQuery(layout) {
    const params = new URLSearchParams();
    if (!layout) {
      return params;
    }
    params.set('w', String(layout.imgWidth));
    params.set('h', String(layout.imgHeight));
    params.set('stride', String(layout.stride));
    params.set('margin', String(layout.margin));
    params.set('reserved', String(layout.reservedCornerSide));
    return params;
  };

  api.makeShareUrl = function makeShareUrl(pathname, layout, extras) {
    const url = new URL(pathname, global.location.href);
    const params = api.makeLayoutQuery(layout);
    if (extras && typeof extras === 'object') {
      Object.keys(extras).forEach((key) => {
        const value = extras[key];
        if (value == null || value === '') {
          return;
        }
        params.set(key, String(value));
      });
    }
    url.search = params.toString();
    return url.href;
  };

  api.hashBytesFNV1a = function hashBytesFNV1a(bytes) {
    const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes || 0);
    let hash = 0x811c9dc5;
    for (let i = 0; i < data.length; i++) {
      hash ^= data[i];
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash >>> 0;
  };

  api.autoPacketCount = function autoPacketCount(recommended) {
    return Math.max(1, Math.round(Number(recommended) || 1));
  };

  api.downloadBytes = function downloadBytes(bytes, fileName) {
    const blob = new Blob([bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes || 0)]);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName || ('camera_drop_' + Date.now() + '.bin');
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  api.copyText = async function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const input = document.createElement('textarea');
    input.value = text;
    document.body.appendChild(input);
    input.select();
    document.execCommand('copy');
    input.remove();
  };
})(window);
