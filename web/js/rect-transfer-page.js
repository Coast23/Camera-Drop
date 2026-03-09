(function (global) {
  const dom = {
    fileInput: document.getElementById('fileInput'),
    scaleInput: document.getElementById('scaleInput'),
    prepareBtn: document.getElementById('prepareBtn'),
    prevBtn: document.getElementById('prevBtn'),
    nextBtn: document.getElementById('nextBtn'),
    verifyBtn: document.getElementById('verifyBtn'),
    status: document.getElementById('status'),
    layoutInfo: document.getElementById('layoutInfo'),
    packetInfo: document.getElementById('packetInfo'),
    frameInfo: document.getElementById('frameInfo'),
    verifyInfo: document.getElementById('verifyInfo'),
    logBox: document.getElementById('logBox'),
    preview: document.getElementById('preview'),
  };

  const state = {
    file: null,
    packets: [],
    frameIndex: 0,
  };

  function log(line) {
    dom.logBox.textContent = line + '\n' + dom.logBox.textContent;
  }

  function setStatus(text) {
    dom.status.textContent = text;
  }

  async function renderCurrent() {
    if (!state.packets.length) {
      dom.frameInfo.textContent = '-';
      return;
    }
    const scale = Math.max(1, Math.min(4, Number(dom.scaleInput.value) || 1));
    await global.CamDropRectRender.renderPacketToCanvas(dom.preview, state.packets[state.frameIndex], { scale });
    dom.frameInfo.textContent = (state.frameIndex + 1) + ' / ' + state.packets.length;
    dom.prevBtn.disabled = state.frameIndex <= 0;
    dom.nextBtn.disabled = state.frameIndex >= state.packets.length - 1;
  }

  async function prepare() {
    const file = dom.fileInput.files && dom.fileInput.files[0];
    if (!file) {
      setStatus('select a file first');
      return;
    }
    setStatus('loading wasm');
    const layout = await global.CamDropRectCodec.getLayout();
    dom.layoutInfo.textContent = layout.imgWidth + 'x' + layout.imgHeight + '  grid ' + layout.gridCols + 'x' + layout.gridRows + '  packet ' + layout.packetBytes + 'B';
    setStatus('reading file');
    const fileBytes = new Uint8Array(await file.arrayBuffer());
    setStatus('encoding');
    const encoder = await global.CamDropRectCodec.createEncoder(fileBytes, file.name);
    try {
      const recommended = Math.max(1, encoder.packetCountRecommended());
      const packetCount = Math.min(recommended, 256);
      const packets = [];
      for (let i = 0; i < packetCount; i++) {
        packets.push(await encoder.getPacket());
      }
      state.file = file;
      state.packets = packets;
      state.frameIndex = 0;
      dom.packetInfo.textContent = 'recommended ' + recommended + ', prepared ' + packetCount;
      dom.verifyInfo.textContent = '-';
      dom.verifyBtn.disabled = false;
      await renderCurrent();
      setStatus('ready');
      log('prepared ' + packetCount + ' packets for ' + file.name + ' (' + fileBytes.length + ' bytes)');
    } finally {
      encoder.destroy();
    }
  }

  async function verifyRoundtrip() {
    if (!state.file || !state.packets.length) {
      return;
    }
    setStatus('verifying');
    const original = new Uint8Array(await state.file.arrayBuffer());
    const decoder = await global.CamDropRectCodec.createDecoder();
    try {
      for (let i = 0; i < state.packets.length; i++) {
        decoder.processPacket(state.packets[i]);
        if (decoder.isComplete()) {
          break;
        }
      }
      if (!decoder.isComplete()) {
        dom.verifyInfo.textContent = 'not complete after ' + state.packets.length + ' packets';
        setStatus('verify incomplete');
        return;
      }
      const outName = decoder.getFilename();
      const outBytes = decoder.getFileBytes();
      let ok = outBytes.length === original.length;
      if (ok) {
        for (let i = 0; i < original.length; i++) {
          if (original[i] !== outBytes[i]) {
            ok = false;
            break;
          }
        }
      }
      dom.verifyInfo.textContent = ok ? ('ok, name=' + (outName || '(none)') + ', size=' + outBytes.length) : ('mismatch, decoded=' + outBytes.length + ' bytes');
      setStatus(ok ? 'verify ok' : 'verify mismatch');
    } finally {
      decoder.destroy();
    }
  }

  dom.prepareBtn.addEventListener('click', () => {
    prepare().catch((err) => {
      setStatus('error');
      log(err && err.stack ? err.stack : String(err));
    });
  });
  dom.prevBtn.addEventListener('click', () => {
    if (state.frameIndex > 0) {
      state.frameIndex--;
      renderCurrent().catch(console.error);
    }
  });
  dom.nextBtn.addEventListener('click', () => {
    if (state.frameIndex + 1 < state.packets.length) {
      state.frameIndex++;
      renderCurrent().catch(console.error);
    }
  });
  dom.verifyBtn.addEventListener('click', () => {
    verifyRoundtrip().catch((err) => {
      setStatus('error');
      log(err && err.stack ? err.stack : String(err));
    });
  });
  dom.scaleInput.addEventListener('change', () => {
    renderCurrent().catch(console.error);
  });
})(window);
