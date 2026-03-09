(function (global) {
  'use strict';

  let runtimePromise = null;
  let runtimeKey = '';
  let activeRunId = 0;
  let cachedLayout = null;
  let cachedLayoutKey = '';
  let cachedCanvas = null;
  let cachedCanvasKey = '';

  function withAssetVersion(url, version) {
    if (!version) {
      return url;
    }
    return url + (url.indexOf('?') >= 0 ? '&' : '?') + 'v=' + encodeURIComponent(version);
  }

  function makeLayoutKey(layout) {
    if (!layout || typeof layout !== 'object') {
      return '';
    }
    return [
      Number(layout.imgWidth) || 0,
      Number(layout.imgHeight) || 0,
      Number(layout.stride) || 0,
      Number(layout.margin) || 0,
      Number(layout.reservedCornerSide) || 0,
    ].join('x');
  }

  async function ensureRuntime(assetVersion, codecBase) {
    const version = typeof assetVersion === 'string' ? assetVersion : '';
    const base = (typeof codecBase === 'string' && codecBase ? codecBase : './vendor').replace(/\/$/, '');
    const key = version + '|' + base;
    if (runtimePromise && runtimeKey === key && global.CamDropRectCodec && global.CamDropRectRender) {
      return runtimePromise;
    }
    runtimeKey = key;
    global.__CAMDROP_ASSET_VERSION = version;
    global.__CAMDROP_RECT_CODEC_BASE = base;
    global.importScripts(
      withAssetVersion('./rect-codec.js', version),
      withAssetVersion('./rect-render.js', version)
    );
    runtimePromise = (async function () {
      await global.CamDropRectCodec.loadModule();
      return true;
    })();
    return runtimePromise;
  }

  async function ensureLayout(layout) {
    const key = makeLayoutKey(layout);
    if (cachedLayout && cachedLayoutKey === key) {
      return cachedLayout;
    }
    cachedLayout = await global.CamDropRectCodec.configureLayout(layout || {});
    cachedLayoutKey = makeLayoutKey(cachedLayout);
    cachedCanvas = null;
    cachedCanvasKey = '';
    return cachedLayout;
  }

  function getCanvas(width, height) {
    const key = String(width) + 'x' + String(height);
    if (!cachedCanvas || cachedCanvasKey !== key) {
      cachedCanvas = new OffscreenCanvas(width, height);
      cachedCanvasKey = key;
    }
    return cachedCanvas;
  }

  async function canvasToBitmap(canvas) {
    if (typeof canvas.transferToImageBitmap === 'function') {
      return canvas.transferToImageBitmap();
    }
    if (typeof global.createImageBitmap === 'function') {
      return await global.createImageBitmap(canvas);
    }
    throw new Error('ImageBitmap is not supported in this browser');
  }

  function post(type, payload, transfer) {
    global.postMessage(Object.assign({ type: type }, payload || {}), transfer || []);
  }

  async function handleInit(message) {
    await ensureRuntime(message.assetVersion, message.codecBase);
    post('ready', {});
  }

  async function handleRender(message) {
    const runId = Math.max(1, Math.round(Number(message.runId) || 0));
    activeRunId = runId;
    await ensureRuntime(message.assetVersion, message.codecBase);
    const layout = await ensureLayout(message.layout || {});
    if (activeRunId !== runId) {
      return;
    }
    const scale = Math.max(1, Math.min(3, Math.round(Number(message.renderScale) || 1)));
    const packetBytes = message.packetBytes instanceof ArrayBuffer
      ? new Uint8Array(message.packetBytes)
      : new Uint8Array(message.packetBytes || 0);
    const canvas = getCanvas(layout.imgWidth * scale, layout.imgHeight * scale);
    await global.CamDropRectRender.renderPacketToCanvas(canvas, packetBytes, { scale: scale });
    if (activeRunId !== runId) {
      return;
    }
    const bitmap = await canvasToBitmap(canvas);
    if (activeRunId !== runId) {
      if (bitmap && typeof bitmap.close === 'function') {
        bitmap.close();
      }
      return;
    }
    post('frame', {
      runId: runId,
      index: Math.max(0, Math.round(Number(message.index) || 0)),
      bitmap: bitmap,
    }, [bitmap]);
  }

  global.onmessage = function (event) {
    const data = event && event.data ? event.data : {};
    if (data.type === 'cancel') {
      if ((Number(data.runId) || 0) === activeRunId) {
        activeRunId = 0;
      }
      return;
    }
    if (data.type === 'init') {
      handleInit(data).catch(function (error) {
        post('error', {
          runId: 0,
          message: error && error.message ? error.message : String(error),
        });
      });
      return;
    }
    if (data.type !== 'render') {
      return;
    }
    handleRender(data).catch(function (error) {
      post('error', {
        runId: Number(data.runId) || 0,
        message: error && error.message ? error.message : String(error),
      });
    });
  };
})(typeof globalThis !== 'undefined' ? globalThis : self);
