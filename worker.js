// 전사 워커: 모델 로드와 추론을 메인 스레드 밖에서 처리한다.
import { pipeline, Tensor } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

let asr = null, loadedKey = null;

// 네트워크 기록: 이 워커가 낸 요청을 메인 화면의 '네트워크 증명' 패널에 알린다.
// 이 코드는 요청을 읽어 알리기만 하고 막거나 바꾸지 않는다.
try {
  new PerformanceObserver((l) => {
    const rows = l.getEntries().map((e) => ({ url: e.name, type: e.initiatorType, bytes: e.encodedBodySize || 0 })).filter((r) => /^https?:/.test(r.url));
    if (rows.length) self.postMessage({ type: 'net', kind: 'resource', rows });
  }).observe({ type: 'resource', buffered: true });
} catch {}
const reportSend = (method, url) => { try { self.postMessage({ type: 'net', kind: 'send', method, url: String(url) }); } catch {} };
const _fetch = self.fetch.bind(self);
self.fetch = (input, init) => {
  try {
    const req = typeof Request !== 'undefined' && input instanceof Request ? input : null;
    const method = String((init && init.method) || (req && req.method) || 'GET').toUpperCase();
    const body = (init && init.body) || (req && req.body);
    if ((method !== 'GET' && method !== 'HEAD') || body) reportSend(method, req ? req.url : input);
  } catch {}
  return _fetch(input, init);
};
try {
  const xo = XMLHttpRequest.prototype.open, xs = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (m, u, ...r) { this.__m = String(m).toUpperCase(); this.__u = u; return xo.call(this, m, u, ...r); };
  XMLHttpRequest.prototype.send = function (b) { if ((this.__m && this.__m !== 'GET' && this.__m !== 'HEAD') || b) reportSend(this.__m || 'XHR', this.__u); return xs.call(this, b); };
} catch {}
try {
  const WS = self.WebSocket;
  if (WS) self.WebSocket = new Proxy(WS, { construct(t, args) { reportSend('WS', args[0]); return Reflect.construct(t, args); } });
} catch {}

// revision: 모델 저장소의 커밋 해시(2026-11-09 조회). main이 바뀌어도 같은 파일을 받게 고정한다.
const MODELS = {
  turbo: { id: 'onnx-community/whisper-large-v3-turbo', revision: '360ebcde2559d60bb474678be3c1de9ef347d01a' },
  small: { id: 'onnx-community/whisper-small', revision: '36050c46d777d46dc4b5f43f6d90574fc38f8732' },
  base:  { id: 'onnx-community/whisper-base', revision: '27a07b4b52b3a4c5886134bb4320e93e1a4bbc7c' },
};
const DTYPES = {
  webgpu: { encoder_model: 'fp16', decoder_model_merged: 'q4' },
  wasm:   { encoder_model: 'q8',   decoder_model_merged: 'q8' },
};

// 언어 감지: 라이브러리는 언어를 안 주면 영어로 간주하므로, 모델이 첫 토큰으로 예측하는 언어 확률을 직접 읽는다.
// (30초 구간 하나에 대해 인코더를 한 번 더 돌리는 비용이 든다)
async function detectLang(slice) {
  const gc = asr.model.generation_config;
  const f = await asr.processor(slice);
  const out = await asr.model({
    input_features: f.input_features,
    decoder_input_ids: new Tensor('int64', BigInt64Array.from([BigInt(gc.decoder_start_token_id)]), [1, 1]),
  });
  const d = out.logits.data;
  const langs = Object.entries(gc.lang_to_id).map(([k, v]) => [k.slice(2, -2), Number(v)]);
  let mx = -Infinity; for (const [, id] of langs) if (d[id] > mx) mx = d[id];
  let z = 0, best = null, bp = 0;
  const ps = langs.map(([c, id]) => { const e = Math.exp(d[id] - mx); z += e; return [c, e]; });
  for (const [c, e] of ps) if (e > bp) { bp = e; best = c; }
  try { out.logits.dispose && out.logits.dispose(); f.input_features.dispose && f.input_features.dispose(); } catch {}
  return [best, bp / z];
}

self.onmessage = async (e) => {
  const m = e.data;
  const id = m.id;
  const post = (o) => self.postMessage({ ...o, id });
  try {
    if (m.type === 'load') {
      if (!Object.hasOwn(MODELS, m.model) || !Object.hasOwn(DTYPES, m.device)) throw new Error('지원하지 않는 모델 또는 장치입니다.');
      const cfg = MODELS[m.model];
      const key = m.model + '|' + m.device;
      if (loadedKey !== key) {
        asr = null; loadedKey = null;   // 로드가 실패해도 이전 모델이 남아 있는 것처럼 보이지 않게 한다
        const t0 = performance.now();
        asr = await pipeline('automatic-speech-recognition', cfg.id, {
          device: m.device, dtype: DTYPES[m.device], revision: cfg.revision,
          progress_callback: (p) => post({ type: 'progress', p }),
        });
        loadedKey = key;
        post({ type: 'ready', ms: performance.now() - t0 });
      } else post({ type: 'ready', ms: 0 });
    } else if (m.type === 'run') {
      if (!asr) throw new Error('모델이 준비되지 않았습니다.');
      const t0 = performance.now();
      // 겹치는 구간을 합치지 않고, 30초 창을 순서대로 처리한다(Whisper 원래 방식).
      // 창 안의 마지막 문장은 잘렸을 수 있어, 그 앞 문장의 끝에서 다음 창을 시작한다.
      // base: 이 오디오 조각이 원본에서 시작하는 시각(초). 멈춘 구간을 건너뛰고 이어서 할 때 쓴다.
      const SR = 16000, WIN = 30 * SR, total = m.audio.length, base = m.base || 0, totalSec = m.total || (base + total / SR);
      let pos = 0, lastLang = null;
      while (pos < total) {
        const slice = m.audio.subarray(pos, Math.min(pos + WIN, total));
        // 소리가 거의 없는 창(디지털 무음 수준)은 모델에 넣지 않는다. 넣으면 없는 문장을 지어낸다.
        let peak = 0;
        for (let i = 0; i < slice.length; i++) { const v = slice[i] < 0 ? -slice[i] : slice[i]; if (v > peak) peak = v; }
        let cs = [], used = null, prob = null;
        if (peak >= 0.002) {
          let language = m.language;
          if (language === 'auto') {
            try { [used, prob] = await detectLang(slice); language = used; lastLang = used; }
            catch { language = lastLang || 'en'; used = language + '?'; }   // 감지에 실패하면 직전 구간의 언어(없으면 영어)
          } else used = language;
          const r = await asr(slice, { language, task: 'transcribe', return_timestamps: true });
          cs = (r.chunks || []).filter(c => c.text && c.text.trim());
        }
        const isLast = pos + WIN >= total;
        let adv = WIN, keep = cs;
        if (!isLast && cs.length > 1) {
          const last = cs[cs.length - 1];
          const lastEnd = last.timestamp[1];
          // 마지막 문장이 창 끝까지 이어지면(잘림 가능) 버리고 그 앞 문장 끝에서 다시 시작
          if (lastEnd == null || lastEnd >= 29.5) {
            keep = cs.slice(0, -1);
            const prevEnd = keep[keep.length - 1].timestamp[1];
            if (prevEnd != null && prevEnd > 5) adv = Math.round(prevEnd * SR);
            else keep = cs;
          }
        }
        const off = base + pos / SR;
        const out = keep.map(c => ({ text: c.text, timestamp: [c.timestamp[0] + off, (c.timestamp[1] ?? c.timestamp[0] + 2) + off] }));
        pos += adv;
        const next = base + Math.min(pos, total) / SR;
        // 창 하나가 끝날 때마다 결과를 바로 보낸다(메인이 모으고, 멈춤 감시에도 쓴다)
        post({ type: 'chunk', frac: Math.min(1, next / totalSec), keep: out, next, lang: used, langProb: prob });
      }
      post({ type: 'done', ms: performance.now() - t0 });
    }
  } catch (err) {
    if (m.type === 'load') { asr = null; loadedKey = null; }
    post({ type: 'error', message: String((err && err.message) || err) });
  }
};
