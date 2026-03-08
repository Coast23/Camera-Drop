if (typeof SharedArrayBuffer === 'undefined') {
  ort.env.wasm.numThreads = 1;
} else {
  ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);
}
ort.env.wasm.wasmPaths = {
  'ort-wasm-simd-threaded.wasm':      'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.jsep.wasm': 'ort-wasm-simd-threaded.jsep.wasm',
};
