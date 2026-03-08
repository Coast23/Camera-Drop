'use strict';

const pop8 = new Uint8Array(256);
for (let i = 1; i < 256; i++) pop8[i] = pop8[i >> 1] + (i & 1);

let payloadSymbolCount = 0;
let frames = [];

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

function hammingBytes(a, b) {
  const n = Math.min(a.length, b.length);
  let total = 0;
  for (let i = 0; i < n; i++) total += pop8[(a[i] ^ b[i]) & 255];
  return total + Math.abs(a.length - b.length) * 8;
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

function isBetter(a, b) {
  if (!b) return true;
  if (a.cmp.symbolCorrect !== b.cmp.symbolCorrect) return a.cmp.symbolCorrect > b.cmp.symbolCorrect;
  if (a.cmp.patternCorrect !== b.cmp.patternCorrect) return a.cmp.patternCorrect > b.cmp.patternCorrect;
  if (a.cmp.colorCorrect !== b.cmp.colorCorrect) return a.cmp.colorCorrect > b.cmp.colorCorrect;
  return a.hamming < b.hamming;
}

self.onmessage = (event) => {
  const data = event.data;
  if (!data) return;

  if (data.type === 'init') {
    payloadSymbolCount = data.payloadSymbolCount | 0;
    frames = (data.frames || []).map((frame) => ({
      seq: frame.seq | 0,
      payloadSymbols: new Uint8Array(frame.payloadSymbols || []),
      payloadBytes: new Uint8Array(frame.payloadBytes || []),
    }));
    self.postMessage({ type: 'ready' });
    return;
  }

  if (data.type !== 'match') return;

  const payloadBytes = new Uint8Array(data.payloadBuf || new ArrayBuffer(0));
  const decodedSymbols = unpack6Bits(payloadBytes, payloadSymbolCount);
  let best = null;
  let second = null;

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    const cmp = compareSymbols(decodedSymbols, frame.payloadSymbols);
    const hamming = hammingBytes(payloadBytes, frame.payloadBytes);
    const candidate = { frame, cmp, hamming };
    if (isBetter(candidate, best)) {
      second = best;
      best = candidate;
    } else if (isBetter(candidate, second)) {
      second = candidate;
    }
  }

  const bestSymAcc = best && best.cmp.n ? (100 * best.cmp.symbolCorrect / best.cmp.n) : 0;
  const bestPatAcc = best && best.cmp.n ? (100 * best.cmp.patternCorrect / best.cmp.n) : 0;
  const bestColAcc = best && best.cmp.n ? (100 * best.cmp.colorCorrect / best.cmp.n) : 0;
  const secondSymAcc = second && second.cmp.n ? (100 * second.cmp.symbolCorrect / second.cmp.n) : 0;

  self.postMessage({
    type: 'match',
    token: data.token,
    seq: best ? best.frame.seq : -1,
    n: best ? best.cmp.n : 0,
    symbolCorrect: best ? best.cmp.symbolCorrect : 0,
    patternCorrect: best ? best.cmp.patternCorrect : 0,
    colorCorrect: best ? best.cmp.colorCorrect : 0,
    bestSymAcc,
    bestPatAcc,
    bestColAcc,
    top2Gap: best && second ? (bestSymAcc - secondSymAcc) : 0,
  });
};
