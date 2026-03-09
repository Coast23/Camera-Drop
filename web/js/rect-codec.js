(function (global) {
  const api = global.CamDropRectCodec = global.CamDropRectCodec || {};

  let modulePromise = null;
  let layoutPromise = null;
  let assetVersionCache = '';

  function assetVersion() {
    if (assetVersionCache) {
      return assetVersionCache;
    }
    if (typeof global.__CAMDROP_ASSET_VERSION === 'string' && global.__CAMDROP_ASSET_VERSION) {
      assetVersionCache = global.__CAMDROP_ASSET_VERSION;
      return assetVersionCache;
    }
    assetVersionCache = String(Date.now());
    return assetVersionCache;
  }

  function withAssetVersion(url) {
    const v = encodeURIComponent(assetVersion());
    return url + (url.includes('?') ? '&' : '?') + 'v=' + v;
  }

  function baseUrl() {
    if (typeof global.__CAMDROP_RECT_CODEC_BASE === 'string' && global.__CAMDROP_RECT_CODEC_BASE) {
      return global.__CAMDROP_RECT_CODEC_BASE.replace(/\/$/, '');
    }
    return typeof document === 'undefined' ? './vendor' : './js/vendor';
  }

  function loadScript(url) {
    return new Promise((resolve, reject) => {
      if (typeof document === 'undefined') {
        try {
          global.importScripts(url);
          resolve();
        } catch (error) {
          reject(error instanceof Error ? error : new Error(String(error)));
        }
        return;
      }
      const s = document.createElement('script');
      s.src = url;
      s.async = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('failed to load ' + url));
      document.head.appendChild(s);
    });
  }

  function readCString(mod, ptr, maxLen) {
    const bytes = mod.HEAPU8.subarray(ptr, ptr + maxLen);
    let end = 0;
    while (end < bytes.length && bytes[end] !== 0) {
      end++;
    }
    return new TextDecoder('utf-8').decode(bytes.subarray(0, end));
  }

  function getLastError(mod) {
    const ptr = mod._malloc(1024);
    try {
      const n = mod._cdr_rect_get_last_error(ptr, 1024);
      if (!n) {
        return '';
      }
      return readCString(mod, ptr, 1024);
    } finally {
      mod._free(ptr);
    }
  }

  function getExports(mod) {
    if (mod.__camdropRectExports) {
      return mod.__camdropRectExports;
    }
    mod.__camdropRectExports = {
      resetLayout: mod.cwrap('cdr_rect_reset_layout', 'number', []),
      setLayout: mod.cwrap('cdr_rect_set_layout', 'number', ['number', 'number', 'number', 'number', 'number']),
      getImgWidth: mod.cwrap('cdr_rect_get_img_width', 'number', []),
      getImgHeight: mod.cwrap('cdr_rect_get_img_height', 'number', []),
      getStride: mod.cwrap('cdr_rect_get_stride', 'number', []),
      getMargin: mod.cwrap('cdr_rect_get_margin', 'number', []),
      getReservedCornerSide: mod.cwrap('cdr_rect_get_reserved_corner_side', 'number', []),
      getGridRows: mod.cwrap('cdr_rect_get_grid_rows', 'number', []),
      getGridCols: mod.cwrap('cdr_rect_get_grid_cols', 'number', []),
      getBitsPerUnit: mod.cwrap('cdr_rect_get_bits_per_unit', 'number', []),
      getUnitCount: mod.cwrap('cdr_rect_get_unit_count', 'number', []),
      getPacketCapacityBytes: mod.cwrap('cdr_rect_get_packet_capacity_bytes', 'number', []),
      getEncodedPacketBytes: mod.cwrap('cdr_rect_get_encoded_packet_bytes', 'number', []),
      getRsDataSize: mod.cwrap('cdr_rect_get_rs_data_size', 'number', []),
      getRsParitySize: mod.cwrap('cdr_rect_get_rs_parity_size', 'number', []),
      getRsBlockSize: mod.cwrap('cdr_rect_get_rs_block_size', 'number', []),
      getFountainPayloadSize: mod.cwrap('cdr_rect_get_fountain_payload_size', 'number', []),
      getFountainChunkSize: mod.cwrap('cdr_rect_get_fountain_chunk_size', 'number', []),
      packetBytesToUnits: mod.cwrap('cdr_rect_packet_bytes_to_units', 'number', ['number', 'number', 'number', 'number']),
      unitsToPacketBytes: mod.cwrap('cdr_rect_units_to_packet_bytes', 'number', ['number', 'number', 'number', 'number']),
      encoderCreate: mod.cwrap('cdr_rect_encoder_create', 'number', []),
      encoderDestroy: mod.cwrap('cdr_rect_encoder_destroy', null, ['number']),
      encoderInit: mod.cwrap('cdr_rect_encoder_init', 'number', ['number', 'number', 'number', 'number', 'number']),
      encoderPacketCountRecommended: mod.cwrap('cdr_rect_encoder_packet_count_recommended', 'number', ['number']),
      encoderGetPacket: mod.cwrap('cdr_rect_encoder_get_packet', 'number', ['number', 'number', 'number']),
      decoderCreate: mod.cwrap('cdr_rect_decoder_create', 'number', []),
      decoderDestroy: mod.cwrap('cdr_rect_decoder_destroy', null, ['number']),
      decoderReset: mod.cwrap('cdr_rect_decoder_reset', null, ['number']),
      decoderProcessPacket: mod.cwrap('cdr_rect_decoder_process_packet', 'number', ['number', 'number', 'number']),
      decoderIsComplete: mod.cwrap('cdr_rect_decoder_is_complete', 'number', ['number']),
      decoderGetUniqueBlockCount: mod.cwrap('cdr_rect_decoder_get_unique_block_count', 'number', ['number']),
      decoderGetRequiredBlockCount: mod.cwrap('cdr_rect_decoder_get_required_block_count', 'number', ['number']),
      decoderGetFilename: mod.cwrap('cdr_rect_decoder_get_filename', 'number', ['number', 'number', 'number']),
      decoderGetFileSize: mod.cwrap('cdr_rect_decoder_get_file_size', 'number', ['number']),
      decoderCopyFile: mod.cwrap('cdr_rect_decoder_copy_file', 'number', ['number', 'number', 'number']),
    };
    return mod.__camdropRectExports;
  }

  function readLayout(mod) {
    const ex = getExports(mod);
    return Object.freeze({
      imgWidth: ex.getImgWidth(),
      imgHeight: ex.getImgHeight(),
      stride: ex.getStride(),
      margin: ex.getMargin(),
      reservedCornerSide: ex.getReservedCornerSide(),
      gridRows: ex.getGridRows(),
      gridCols: ex.getGridCols(),
      bitsPerUnit: ex.getBitsPerUnit(),
      unitCount: ex.getUnitCount(),
      packetBytes: ex.getPacketCapacityBytes(),
      encodedPacketBytes: ex.getEncodedPacketBytes(),
      rsDataSize: ex.getRsDataSize(),
      rsParitySize: ex.getRsParitySize(),
      rsBlockSize: ex.getRsBlockSize(),
      fountainPayloadSize: ex.getFountainPayloadSize(),
      fountainChunkSize: ex.getFountainChunkSize(),
    });
  }

  function toFiniteInt(value, fallback) {
    if (value == null || value === '') {
      return fallback;
    }
    const n = Number(value);
    if (!Number.isFinite(n)) {
      throw new Error('invalid layout value: ' + value);
    }
    return Math.round(n);
  }

  function buildLayoutRequest(current, options) {
    const cfg = options || {};
    const stride = toFiniteInt(cfg.stride, current.stride);
    const margin = toFiniteInt(cfg.margin, current.margin);
    const reservedCornerSide = toFiniteInt(
      cfg.reservedCornerSide != null ? cfg.reservedCornerSide : cfg.reserved,
      current.reservedCornerSide
    );

    const inputGridCols = cfg.gridCols != null ? cfg.gridCols : cfg.cols;
    const inputGridRows = cfg.gridRows != null ? cfg.gridRows : cfg.rows;
    const inputWidth = cfg.imgWidth != null ? cfg.imgWidth : cfg.width;
    const inputHeight = cfg.imgHeight != null ? cfg.imgHeight : cfg.height;

    const gridCols = inputGridCols == null || inputGridCols === '' ? null : toFiniteInt(inputGridCols, null);
    const gridRows = inputGridRows == null || inputGridRows === '' ? null : toFiniteInt(inputGridRows, null);

    let imgWidth = inputWidth == null || inputWidth === '' ? null : toFiniteInt(inputWidth, null);
    let imgHeight = inputHeight == null || inputHeight === '' ? null : toFiniteInt(inputHeight, null);

    if (imgWidth == null && gridCols != null) {
      imgWidth = margin * 2 + gridCols * stride;
    }
    if (imgHeight == null && gridRows != null) {
      imgHeight = margin * 2 + gridRows * stride;
    }

    return {
      imgWidth: toFiniteInt(imgWidth, current.imgWidth),
      imgHeight: toFiniteInt(imgHeight, current.imgHeight),
      stride,
      margin,
      reservedCornerSide,
    };
  }

  async function applyLayout(mod, options) {
    if (!options || typeof options !== 'object') {
      throw new Error('layout options must be an object');
    }
    const ex = getExports(mod);
    const current = readLayout(mod);
    const req = buildLayoutRequest(current, options);
    const rc = ex.setLayout(req.imgWidth, req.imgHeight, req.stride, req.margin, req.reservedCornerSide);
    if (rc < 0) {
      throw new Error(getLastError(mod) || ('setLayout failed: ' + rc));
    }
    const layout = readLayout(mod);
    layoutPromise = Promise.resolve(layout);
    return layout;
  }

  function getBootstrapLayout() {
    if (!global.__CAMDROP_RECT_LAYOUT || typeof global.__CAMDROP_RECT_LAYOUT !== 'object') {
      return null;
    }
    return global.__CAMDROP_RECT_LAYOUT;
  }

  api.loadModule = function loadModule() {
    if (modulePromise) {
      return modulePromise;
    }
    modulePromise = (async () => {
      if (typeof global.CameraDropRectCodecModule !== 'function') {
        await loadScript(withAssetVersion(baseUrl() + '/camera_drop_rect_codec.js'));
      }
      if (typeof global.CameraDropRectCodecModule !== 'function') {
        throw new Error('camera_drop_rect_codec module factory missing');
      }
      const mod = await global.CameraDropRectCodecModule({
        locateFile: (path) => withAssetVersion(baseUrl() + '/' + path),
      });
      getExports(mod);
      const bootstrapLayout = getBootstrapLayout();
      if (bootstrapLayout) {
        await applyLayout(mod, bootstrapLayout);
      } else {
        layoutPromise = Promise.resolve(readLayout(mod));
      }
      return mod;
    })();
    return modulePromise;
  };

  api.getLayout = function getLayout() {
    if (layoutPromise) {
      return layoutPromise;
    }
    layoutPromise = api.loadModule().then((mod) => readLayout(mod));
    return layoutPromise;
  };

  api.resetLayout = async function resetLayout() {
    const mod = await api.loadModule();
    const ex = getExports(mod);
    const rc = ex.resetLayout();
    if (rc < 0) {
      throw new Error(getLastError(mod) || ('resetLayout failed: ' + rc));
    }
    const layout = readLayout(mod);
    layoutPromise = Promise.resolve(layout);
    return layout;
  };

  api.configureLayout = async function configureLayout(options) {
    const mod = await api.loadModule();
    return applyLayout(mod, options);
  };

  api.packetToUnits = async function packetToUnits(bytes) {
    const mod = await api.loadModule();
    const ex = getExports(mod);
    const layout = await api.getLayout();
    const input = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes || 0);
    if (input.length !== layout.packetBytes) {
      throw new Error('packet byte length mismatch: expected ' + layout.packetBytes + ', got ' + input.length);
    }
    const inPtr = mod._malloc(input.length);
    const outPtr = mod._malloc(layout.unitCount);
    try {
      mod.HEAPU8.set(input, inPtr);
      const rc = ex.packetBytesToUnits(inPtr, input.length, outPtr, layout.unitCount);
      if (rc < 0) {
        throw new Error(getLastError(mod) || ('packetToUnits failed: ' + rc));
      }
      return new Uint8Array(mod.HEAPU8.slice(outPtr, outPtr + layout.unitCount));
    } finally {
      mod._free(inPtr);
      mod._free(outPtr);
    }
  };

  api.unitsToPacket = async function unitsToPacket(units) {
    const mod = await api.loadModule();
    const ex = getExports(mod);
    const layout = await api.getLayout();
    const input = units instanceof Uint8Array ? units : new Uint8Array(units || 0);
    if (input.length !== layout.unitCount) {
      throw new Error('unit count mismatch: expected ' + layout.unitCount + ', got ' + input.length);
    }
    const inPtr = mod._malloc(input.length);
    const outPtr = mod._malloc(layout.packetBytes);
    try {
      mod.HEAPU8.set(input, inPtr);
      const rc = ex.unitsToPacketBytes(inPtr, input.length, outPtr, layout.packetBytes);
      if (rc < 0) {
        throw new Error(getLastError(mod) || ('unitsToPacket failed: ' + rc));
      }
      return new Uint8Array(mod.HEAPU8.slice(outPtr, outPtr + layout.packetBytes));
    } finally {
      mod._free(inPtr);
      mod._free(outPtr);
    }
  };

  class RectEncoder {
    constructor(mod, handle, layout) {
      this.mod = mod;
      this.handle = handle;
      this.ex = getExports(mod);
      this.layout = Object.freeze({ ...layout });
      this._destroyed = false;
    }

    init(fileBytes, fileName) {
      const bytes = fileBytes instanceof Uint8Array ? fileBytes : new Uint8Array(fileBytes || 0);
      const encodedName = new TextEncoder().encode(fileName || '');
      const dataPtr = this.mod._malloc(Math.max(1, bytes.length));
      const namePtr = this.mod._malloc(Math.max(1, encodedName.length));
      try {
        if (bytes.length) {
          this.mod.HEAPU8.set(bytes, dataPtr);
        }
        if (encodedName.length) {
          this.mod.HEAPU8.set(encodedName, namePtr);
        }
        const rc = this.ex.encoderInit(this.handle, dataPtr, bytes.length, namePtr, encodedName.length);
        if (rc < 0) {
          throw new Error(getLastError(this.mod) || ('encoderInit failed: ' + rc));
        }
      } finally {
        this.mod._free(dataPtr);
        this.mod._free(namePtr);
      }
    }

    packetCountRecommended() {
      const rc = this.ex.encoderPacketCountRecommended(this.handle);
      if (rc < 0) {
        throw new Error(getLastError(this.mod) || ('packetCountRecommended failed: ' + rc));
      }
      return rc;
    }

    getLayout() {
      return this.layout;
    }

    async getPacket() {
      const outPtr = this.mod._malloc(this.layout.packetBytes);
      try {
        const rc = this.ex.encoderGetPacket(this.handle, outPtr, this.layout.packetBytes);
        if (rc < 0) {
          throw new Error(getLastError(this.mod) || ('encoderGetPacket failed: ' + rc));
        }
        return new Uint8Array(this.mod.HEAPU8.slice(outPtr, outPtr + this.layout.packetBytes));
      } finally {
        this.mod._free(outPtr);
      }
    }

    destroy() {
      if (this._destroyed) {
        return;
      }
      this.ex.encoderDestroy(this.handle);
      this._destroyed = true;
    }
  }

  class RectDecoder {
    constructor(mod, handle, layout) {
      this.mod = mod;
      this.handle = handle;
      this.ex = getExports(mod);
      this.layout = Object.freeze({ ...layout });
      this._destroyed = false;
    }

    reset() {
      this.ex.decoderReset(this.handle);
    }

    getLayout() {
      return this.layout;
    }

    processPacket(packetBytes) {
      const bytes = packetBytes instanceof Uint8Array ? packetBytes : new Uint8Array(packetBytes || 0);
      if (bytes.length !== this.layout.packetBytes) {
        throw new Error('packet byte length mismatch: expected ' + this.layout.packetBytes + ', got ' + bytes.length);
      }
      const ptr = this.mod._malloc(Math.max(1, bytes.length));
      try {
        if (bytes.length) {
          this.mod.HEAPU8.set(bytes, ptr);
        }
        let rc;
        try {
          rc = this.ex.decoderProcessPacket(this.handle, ptr, bytes.length);
        } catch (error) {
          const message = error && error.message ? error.message : String(error);
          if (/memory access out of bounds|RuntimeError/i.test(message)) {
            try {
              this.ex.decoderDestroy(this.handle);
            } catch (_) {}
            this.handle = this.ex.decoderCreate();
            if (!this.handle) {
              throw new Error('decoder trap and recreate failed: ' + (getLastError(this.mod) || message));
            }
            throw new Error('decoder trap: reset state and dropped current packet');
          }
          throw error;
        }
        if (rc < 0) {
          throw new Error(getLastError(this.mod) || ('decoderProcessPacket failed: ' + rc));
        }
      } finally {
        this.mod._free(ptr);
      }
    }
    isComplete() {
      return this.ex.decoderIsComplete(this.handle) > 0;
    }

    getUniqueBlockCount() {
      const rc = this.ex.decoderGetUniqueBlockCount(this.handle);
      if (rc < 0) {
        throw new Error(getLastError(this.mod) || ('decoderGetUniqueBlockCount failed: ' + rc));
      }
      return rc;
    }

    getRequiredBlockCount() {
      const rc = this.ex.decoderGetRequiredBlockCount(this.handle);
      if (rc < 0) {
        throw new Error(getLastError(this.mod) || ('decoderGetRequiredBlockCount failed: ' + rc));
      }
      return rc;
    }

    getProgressRatio() {
      const required = this.getRequiredBlockCount();
      if (!required) {
        return 0;
      }
      return Math.max(0, Math.min(1, this.getUniqueBlockCount() / required));
    }

    getFilename() {
      const size = Math.max(1, this.ex.decoderGetFilename(this.handle, 0, 0));
      const ptr = this.mod._malloc(size + 1);
      try {
        const rc = this.ex.decoderGetFilename(this.handle, ptr, size + 1);
        if (rc < 0) {
          throw new Error(getLastError(this.mod) || ('decoderGetFilename failed: ' + rc));
        }
        return readCString(this.mod, ptr, size + 1);
      } finally {
        this.mod._free(ptr);
      }
    }

    getFileBytes() {
      const size = this.ex.decoderGetFileSize(this.handle);
      if (size < 0) {
        throw new Error(getLastError(this.mod) || ('decoderGetFileSize failed: ' + size));
      }
      const ptr = this.mod._malloc(Math.max(1, size));
      try {
        const rc = this.ex.decoderCopyFile(this.handle, ptr, Math.max(1, size));
        if (rc < 0) {
          throw new Error(getLastError(this.mod) || ('decoderCopyFile failed: ' + rc));
        }
        return new Uint8Array(this.mod.HEAPU8.slice(ptr, ptr + rc));
      } finally {
        this.mod._free(ptr);
      }
    }

    destroy() {
      if (this._destroyed) {
        return;
      }
      this.ex.decoderDestroy(this.handle);
      this._destroyed = true;
    }
  }

  api.createEncoder = async function createEncoder(fileBytes, fileName) {
    const mod = await api.loadModule();
    const layout = await api.getLayout();
    const ex = getExports(mod);
    const handle = ex.encoderCreate();
    if (!handle) {
      throw new Error(getLastError(mod) || 'encoderCreate failed');
    }
    const encoder = new RectEncoder(mod, handle, layout);
    encoder.init(fileBytes, fileName || '');
    return encoder;
  };

  api.createDecoder = async function createDecoder() {
    const mod = await api.loadModule();
    const layout = await api.getLayout();
    const ex = getExports(mod);
    const handle = ex.decoderCreate();
    if (!handle) {
      throw new Error(getLastError(mod) || 'decoderCreate failed');
    }
    return new RectDecoder(mod, handle, layout);
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
