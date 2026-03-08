'use strict';

(function initContext(global) {
  const app = global.CameraDropApp || {};

  const config = {
    PATCH_SZ: 16,
    PATCH_SRCH: 10,
    PATCH_SRCH_CONTOUR: 16,
    PATCH_SAD_MAX: 0.18,
    AHASH_N: 16,
    AHASH_THRESH: 30,
    BLUR_SAMPLE_N: 48,
    RAW_BLUR_ENABLED: true,
    RAW_BLUR_BLOCKING: false,
    RAW_BLUR_THRESH: 8,
    RAW_BLUR_TRACK_TTL_MS: 1200,
    COARSE_BLUR_BLOCKING: false,
    COARSE_BLUR_THRESH: 8,
    FINE_BLUR_BLOCKING: false,
    FINE_BLUR_THRESH: 10,
    FINE_SHARPEN_MARGIN: 3,
    FINE_SHARPEN_STRENGTH: 0.6,
    FINE_RENDER_SIZE: 1024,
    DESKEW_CANONICAL_INSET: 2,
    FINE_DESKEW_FILTER: 'nearest',
    CAMERA_READY_TIMEOUT_MS: 1600,
    CAMERA_RECOVER_RETRY_MS: 900,
    CAMERA_WATCHDOG_MS: 1500,
    CAMERA_SETTLE_MS: 500,
    CAMERA_WB_TARGET_K: 6500,
    CAMERA_WB_EDGE_GUARD_STEPS: 2,
    CAMERA_EXPOSURE_DARKEN_RATIO: 0.82,
    CAMERA_ISO_MAX: 400,
    CAMERA_EXPOSURE_COMP_TARGET: -1.3333334,
    CAMERA_TUNE_RECHECK_MS: 1200,
    CAMERA_TUNE_REAPPLY_COOLDOWN_MS: 900,
    CAMERA_TUNE_WB_DRIFT_K: 250,
    CAMERA_TUNE_EXPOSURE_DRIFT_RATIO: 0.35,
    CAMERA_TUNE_ISO_DRIFT: 80,
    CAMERA_TUNE_RECENT_CODE_TTL_MS: 1400,
    CAMERA_TUNE_SCENE_MARGIN_RATIO: 0.18,
    CAMERA_TUNE_SCENE_SAMPLE_N: 24,
    CAMERA_TUNE_SCENE_MIN_LUMA: 22,
    CAMERA_TUNE_SCENE_MAX_LUMA: 242,
    CAMERA_TUNE_SCENE_MIN_RANGE: 26,
    CAMERA_TUNE_SCENE_MIN_BLUR: 4,
    YOLO_QUEUE_MAX: 4,
    PRECISE_QUEUE_MAX: 6,
    RECOG_QUEUE_MAX: 16,
    RECOG_DICT_MODE: 'best',
    RECOG_DICT_BASE: './best_v2',
    RECOG_MAIN_AHASH_THRESH: 4,
    DESKEW_EXP: 1.10,
    CONF: 0.35,
    ANCHOR_EXP: 0.05,
    LOCALIZER_MODE: 'yolo',
    SCAN_MAX_SIDE: 960,
    CONTOUR_AHASH_THRESH: 24,
    CONTOUR_REDETECT_MS: 1200,
    CONTOUR_ROI_MARGIN: 0.38,
    CONTOUR_ROI_MIN: 320,
  };

  const searchArea = config.PATCH_SZ + 2 * Math.max(config.PATCH_SRCH, config.PATCH_SRCH_CONTOUR || 0);

  const state = {
    cvReady: false,
    scanning: false,
    currentEP: 'wasm',
    localizerSource: '-',
    lastCorners: null,
    deskLoopRunning: false,

    yoloWorker: null,
    yoloWorkers: [],
    yoloWorkerPoolSize: 0,
    yoloReadyWorkers: 0,
    yoloActiveCount: 0,
    workerIdle: true,
    yoloMs: 0,
    yoloFpsArr: [],
    yoloLastT: performance.now(),
    videoFrameLoopMode: 'raf',
    videoFrameLoopRunning: false,
    videoFrameLoopRequestId: 0,
    videoFrameWatchdogId: 0,
    currentFrameToken: null,
    lastObservedPresentedFrameToken: null,
    lastObservedMediaTimeToken: null,
    lastObservedVideoTimeToken: null,
    localizerCaptureSeq: 0,
    lastAppliedLocalizerSeq: 0,

    cameraStream: null,
    cameraReady: false,
    cameraStartPromise: null,
    cameraWatchdogId: 0,
    cameraWatchdogRunning: false,
    cameraMissCount: 0,
    lastVideoFrameTime: -1,
    lastVideoFrameTickAt: 0,
    cameraTuneProfile: null,
    cameraTunePromise: null,
    cameraTunePending: false,
    cameraRetuneWanted: false,
    cameraSceneStats: null,
    cameraLastTuneAt: 0,
    cameraLastTuneCheckAt: 0,

    patches: null,

    lastAHash: null,
    preciseQueue: [],
    yoloQueue: [],
    lastLocalizerDispatchKind: 'precise',
    pendingRender: null,

    srchCvs: new OffscreenCanvas(searchArea, searchArea),
    srchCtx: null,
    ahCvs: new OffscreenCanvas(config.AHASH_N, config.AHASH_N),
    ahCtx: null,

    offDsk: new OffscreenCanvas(1, 1),
    coarseGl: null,
    fineGl: null,
    blurCvs: new OffscreenCanvas(config.BLUR_SAMPLE_N, config.BLUR_SAMPLE_N),
    blurCtx: null,
    blurGray: new Uint8Array(config.BLUR_SAMPLE_N * config.BLUR_SAMPLE_N),
    rawBlurScore: 0,
    rawBlurPass: true,
    coarseBlurScore: 0,
    fineBlurScore: 0,
    blurRejectCount: 0,
    rawBlurRejectCount: 0,
    deskewLoopCount: 0,
    deskewSkipNoGlCount: 0,
    deskewSkipNoCornersCount: 0,
    deskewSkipNotReadyCount: 0,
    deskewSkipClaimCount: 0,
    videoFrameCount: 0,
    lastCoarseHandledVideoTime: null,
    coarseTrackFreshCount: 0,
    coarseHashSameCount: 0,
    coarseHashDiffCount: 0,
    preciseEnqueueCount: 0,
    forceFullDoneCount: 0,
    yoloQueueDropCount: 0,
    preciseQueueDropCount: 0,
    recogQueueDropCount: 0,
    lastYoloVideoTime: -1,
    lastDeskewVideoTime: -1,
    lastPatchVideoTime: -1,
    lastContourRunAt: 0,
    contourTrackHash: null,
    contourNeedRefine: false,

    dskFpsArr: [],
    dskLastT: performance.now(),
    dskFps: 0,
    lastDeskewTime: 0,
    lastCoarseGateTime: 0,

    recogLastHash: null,
    recogMs: 0,
    recogDecodeCount: 0,
    recogSkipCount: 0,
    recogLastResult: null,
    recogWorker: null,
    recogWorkerIdle: true,
    recogWorkers: [],
    recogWorkerPoolSize: 0,
    recogReadyWorkers: 0,
    recogActiveCount: 0,
    recogQueue: [],
    recogSeq: 0,
    recogCaptureEpoch: 1,
    recogNextCommitId: 1,
    recogPendingResults: new Map(),
    recogSessionId: 0,
    recogDictPromise: null,
    recogDictConfigKey: '',
    recogWorkerDictConfigKey: '',
    recogDictSource: 'builtin-gen',
  };

  state.srchCtx = state.srchCvs.getContext('2d', { willReadFrequently: true });
  state.ahCtx = state.ahCvs.getContext('2d', { willReadFrequently: true });
  state.blurCtx = state.blurCvs.getContext('2d', { willReadFrequently: true });

  const dom = {
    video: document.getElementById('video'),
    overlay: document.getElementById('overlay'),
    dskCvs: document.getElementById('deskewedCanvas'),
    decodeBar: document.getElementById('decodeBar'),
    cameraMeta: document.getElementById('cameraMeta'),
    scanHint: document.getElementById('scanHint'),
    hintText: document.getElementById('hintText'),
    statusBar: document.getElementById('statusBar'),
    perfBar: document.getElementById('perfBar'),
    loadBtn: document.getElementById('loadBtn'),
    initOver: document.getElementById('initOverlay'),
    initMsg: document.getElementById('initMsg'),
    progBar: document.getElementById('progBar'),
    startBtn: document.getElementById('startBtn'),
    filePicker: document.getElementById('filePicker'),
  };

  const ui = {
    setMsg(message) {
      dom.initMsg.textContent = message;
    },
    setProg(value) {
      dom.progBar.style.width = (value * 100) + '%';
    },
    setStatus(message) {
      dom.statusBar.textContent = message;
    },
  };

  const utils = {
    sleep(ms) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    },
  };

  app.claimVideoFrame = function claimVideoFrame(slotName) {
    const token = state.currentFrameToken;
    if (token !== null && token !== undefined) {
      if (state[slotName] === token) {
        return false;
      }
      state[slotName] = token;
      return true;
    }
    const t = Number(dom.video && dom.video.currentTime);
    if (!Number.isFinite(t)) {
      return true;
    }
    if (Math.abs((state[slotName] ?? -1) - t) < 1e-6) {
      return false;
    }
    state[slotName] = t;
    return true;
  };

  app.resetPipelineCounters = function resetPipelineCounters() {
    state.videoFrameCount = 0;
    state.deskewLoopCount = 0;
    state.deskewSkipNoGlCount = 0;
    state.deskewSkipNoCornersCount = 0;
    state.deskewSkipNotReadyCount = 0;
    state.deskewSkipClaimCount = 0;
    state.currentFrameToken = null;
    state.lastLocalizerDispatchKind = 'precise';
    state.lastObservedPresentedFrameToken = null;
    state.lastObservedMediaTimeToken = null;
    state.lastObservedVideoTimeToken = null;
    state.localizerCaptureSeq = 0;
    state.lastAppliedLocalizerSeq = 0;
    state.lastCoarseHandledVideoTime = null;
    state.lastCoarseGateTime = 0;
    state.coarseTrackFreshCount = 0;
    state.coarseHashSameCount = 0;
    state.coarseHashDiffCount = 0;
    state.preciseEnqueueCount = 0;
    state.forceFullDoneCount = 0;
    state.yoloQueueDropCount = 0;
    state.preciseQueueDropCount = 0;
    state.recogQueueDropCount = 0;
  };

  app.config = config;
  app.state = state;
  app.dom = dom;
  app.ui = ui;
  app.utils = utils;

  global.CameraDropApp = app;
})(window);
