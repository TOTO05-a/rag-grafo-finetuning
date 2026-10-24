import express from 'express';
import { ObjectId } from 'mongodb';
import { HttpError, embed, chat, cosine, normKey } from './lib.js';

// Campos por estrategia: [máx. caracteres, obligatorio]
export const SCHEMAS = {
  transfer: { text: [2000, 1], label: [60, 1] },
  sft: { prompt: [2000, 1], response: [4000, 1] },
  fewshot: { input: [1500, 1], output: [1500, 1] },
  instruction: { instruction: [600, 1], input: [2000, 0], output: [4000, 1] },
};
const MAX_ITEMS = { transfer: 300, sft: 1000, fewshot: 50, instruction: 1000 };

export function cleanItem(strategy, body) {
  const out = {};
  for (const [k, [max, req]] of Object.entries(SCHEMAS[strategy])) {
    const v = String(body?.[k] ?? '').trim();
    if (req && !v) throw new HttpError(400, `El campo "${k}" es obligatorio`);
    if (v.length > max) throw new HttpError(400, `El campo "${k}" supera ${max} caracteres`);
    out[k] = v;
  }
  return out;
}
export function toJSONL(strategy, items) {
  const row = it => strategy === 'sft' ? { messages: [{ role: 'user', content: it.prompt }, { role: 'assistant', content: it.response }] }
    : strategy === 'instruction' ? { instruction: it.instruction, input: it.input, output: it.output }
    : strategy === 'fewshot' ? { input: it.input, output: it.output } : { text: it.text, label: it.label };
  return items.map(i => JSON.stringify(row(i))).join('\n') + '\n';
}

// ---- Transfer learning: cabeza softmax entrenada sobre embeddings congelados ----
const l2n = v => { const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1; return v.map(x => x / n); };
const softmax = z => { const m = Math.max(...z), e = z.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
const logits = (m, x) => m.W.map((w, c) => w.reduce((s, wj, j) => s + wj * x[j], m.b[c]));
export function trainSoftmax(X, y, C, { epochs = 300, lr = 0.8, l2 = 1e-3 } = {}) {
  const d = X[0].length, n = X.length, m = { W: Array.from({ length: C }, () => new Array(d).fill(0)), b: new Array(C).fill(0) };
  for (let e = 0; e < epochs; e++) {
    const gW = m.W.map(w => w.map(() => 0)), gb = new Array(C).fill(0);
    X.forEach((x, i) => { const p = softmax(logits(m, x)); for (let c = 0; c < C; c++) { const err = p[c] - (y[i] === c ? 1 : 0); gb[c] += err; for (let j = 0; j < d; j++) gW[c][j] += err * x[j]; } });
    for (let c = 0; c < C; c++) { m.b[c] -= lr * gb[c] / n; for (let j = 0; j < d; j++) m.W[c][j] -= lr * (gW[c][j] / n + l2 * m.W[c][j]); }
  }
  return m;
}
export const predictProba = (m, x) => softmax(logits(m, x));
export const accuracy = (m, X, y) => X.length ? X.filter((x, i) => predictProba(m, x).indexOf(Math.max(...predictProba(m, x))) === y[i]).length / X.length : null;

// ---- Domain-specific: perfil inyectado en extracción y respuestas ----
export function domainPrompt(p) {
  if (!p) return '';
  const g = (p.glossary || []).slice(0, 60).map(t => `- ${t.term}: ${t.definition}`).join('\n');
  return `\n\nPERFIL DE DOMINIO${p.name ? ` (${p.name})` : ''}:` + (p.entityTypes?.length ? `\nTipos de entidad preferidos: ${p.entityTypes.join(', ')}.` : '')
    + (p.style ? `\nEstilo/criterio: ${p.style}` : '') + (g ? `\nGlosario (úsalo para interpretar y unificar términos):\n${g}` : '');
}
export async function loadDomainPrompt(db, uid) { return domainPrompt(await db.collection('tuning').findOne({ userId: uid, kind: 'domain' })); }
const parseGlossary = t => String(t || '').split('\n').map(l => l.split(':')).filter(a => a.length >= 2 && a[0].trim()).map(a => ({ term: a[0].trim().slice(0, 80), definition: a.slice(1).join(':').trim().slice(0, 300) })).slice(0, 100);

export function tuneRouter({ db, auth, wrap }) {
  const col = db.collection('tuning'), chunks = db.collection('chunks'), r = express.Router();
  r.use(auth);
  const list = (uid, s) => col.find({ userId: uid, kind: 'example', strategy: s }).sort({ createdAt: 1 }).toArray();
  const need = (items, n, what) => { if (items.length < n) throw new HttpError(400, `Necesitas al menos ${n} ${what} (tienes ${items.length})`); };

  r.get('/domain', wrap(async (req, res) => res.json(await col.findOne({ userId: req.uid, kind: 'domain' }) || {})));
  r.put('/domain', wrap(async (req, res) => {
    const doc = { name: String(req.body.name || '').trim().slice(0, 80), style: String(req.body.style || '').trim().slice(0, 600),
      entityTypes: String(req.body.entityTypes || '').split(',').map(s => s.trim().slice(0, 40)).filter(Boolean).slice(0, 20), glossary: parseGlossary(req.body.glossary) };
    if (!doc.name && !doc.glossary.length && !doc.style && !doc.entityTypes.length) throw new HttpError(400, 'El perfil está vacío');
    await col.updateOne({ userId: req.uid, kind: 'domain' }, { $set: doc, $setOnInsert: { createdAt: new Date() } }, { upsert: true });
    res.json({ ok: true, ...doc });
  }));
  r.post('/domain/glossary', wrap(async (req, res) => {
    const sample = await chunks.find({ userId: req.uid }, { projection: { text: 1 } }).limit(12).toArray();
    need(sample, 1, 'fragmentos indexados (procesa un documento en Ingesta)');
    const out = await chat([{ role: 'system', content: 'Extrae del texto los 15 términos técnicos más propios del dominio con definición breve basada SOLO en el texto. Responde JSON: {"terms":[{"term","definition"}]}.' }, { role: 'user', content: sample.map(c => c.text.slice(0, 700)).join('\n---\n') }], true);
    try { res.json({ terms: (JSON.parse(out.replace(/^```json|```$/gim, '')).terms || []).slice(0, 20) }); } catch { throw new HttpError(502, 'El modelo no devolvió JSON válido; reintenta'); }
  }));

  r.post('/transfer/train', wrap(async (req, res) => {
    const items = await list(req.uid, 'transfer'), labels = [...new Set(items.map(i => i.label))];
    need(items, 6, 'ejemplos'); if (labels.length < 2) throw new HttpError(400, 'Necesitas al menos 2 etiquetas distintas');
    const few = labels.filter(l => items.filter(i => i.label === l).length < 2); if (few.length) throw new HttpError(400, `Cada etiqueta necesita 2+ ejemplos: ${few.join(', ')}`);
    const X = (await embed(items.map(i => i.text))).map(l2n), y = items.map(i => labels.indexOf(i.label));
    const hold = items.length >= 10 ? i => i % 5 === 4 : () => false;
    const tr = X.map((_, i) => i).filter(i => !hold(i));
    const model = trainSoftmax(tr.map(i => X[i]), tr.map(i => y[i]), labels.length);
    const testIdx = X.map((_, i) => i).filter(i => hold(i));
    const m = { trainAccuracy: accuracy(model, tr.map(i => X[i]), tr.map(i => y[i])), testAccuracy: accuracy(model, testIdx.map(i => X[i]), testIdx.map(i => y[i])), examples: items.length, heldOut: testIdx.length };
    await col.updateOne({ userId: req.uid, kind: 'model' }, { $set: { labels, ...model, ...m, trainedAt: new Date() } }, { upsert: true });
    res.json({ labels, ...m, note: 'Se entrenó solo la capa clasificadora; el modelo de embeddings permanece congelado.' });
  }));
  r.post('/transfer/predict', wrap(async (req, res) => {
    const model = await col.findOne({ userId: req.uid, kind: 'model' }); if (!model) throw new HttpError(400, 'Primero entrena el clasificador');
    const text = String(req.body.text || '').trim().slice(0, 2000); if (!text) throw new HttpError(400, 'Texto vacío');
    const p = predictProba(model, l2n((await embed([text]))[0]));
    res.json({ ranking: model.labels.map((l, i) => ({ label: l, prob: +p[i].toFixed(4) })).sort((a, b) => b.prob - a.prob) });
  }));
  r.post('/fewshot/run', wrap(async (req, res) => {
    const items = await list(req.uid, 'fewshot'); need(items, 1, 'ejemplos'); const input = String(req.body.input || '').trim().slice(0, 1500); if (!input) throw new HttpError(400, 'Entrada vacía');
    const k = Math.min(Math.max(parseInt(req.body.k) || 4, 1), 8), v = await embed([input, ...items.map(i => i.input)]);
    const used = items.map((it, i) => ({ it, s: cosine(v[0], v[i + 1]) })).sort((a, b) => b.s - a.s).slice(0, k).reverse().map(x => x.it);
    const msgs = [{ role: 'system', content: 'Imita exactamente el formato y criterio de los ejemplos previos.' + await loadDomainPrompt(db, req.uid) }, ...used.flatMap(u => [{ role: 'user', content: u.input }, { role: 'assistant', content: u.output }]), { role: 'user', content: input }];
    res.json({ output: await chat(msgs), examplesUsed: used.length, note: 'Few-shot por prompting: no se modifican pesos.' });
  }));
  r.post('/instruction/run', wrap(async (req, res) => {
    const instruction = String(req.body.instruction || '').trim().slice(0, 600), input = String(req.body.input || '').trim().slice(0, 2000); if (!instruction) throw new HttpError(400, 'Instrucción vacía');
    const demos = (await list(req.uid, 'instruction')).filter(i => normKey(i.instruction) === normKey(instruction)).slice(-3);
    const msgs = [{ role: 'system', content: 'Sigue la instrucción al pie de la letra.' + await loadDomainPrompt(db, req.uid) }, ...demos.flatMap(d => [{ role: 'user', content: `${d.instruction}\n\n${d.input}` }, { role: 'assistant', content: d.output }]), { role: 'user', content: `${instruction}\n\n${input}` }];
    res.json({ output: await chat(msgs), demos: demos.length, note: 'Se ejecuta con la instrucción y demos guardadas; no se modifican pesos.' });
  }));

  r.get('/:s/export', wrap(async (req, res) => {
    if (!SCHEMAS[req.params.s]) throw new HttpError(404, 'Estrategia desconocida');
    const items = await list(req.uid, req.params.s); need(items, 1, 'ejemplos');
    res.set({ 'Content-Type': 'application/jsonl; charset=utf-8', 'Content-Disposition': `attachment; filename="${req.params.s}.jsonl"` }).send(toJSONL(req.params.s, items));
  }));
  r.get('/:s', wrap(async (req, res) => { if (!SCHEMAS[req.params.s]) throw new HttpError(404, 'Estrategia desconocida'); res.json({ items: await list(req.uid, req.params.s), max: MAX_ITEMS[req.params.s], model: req.params.s === 'transfer' ? await col.findOne({ userId: req.uid, kind: 'model' }, { projection: { W: 0, b: 0 } }) : undefined }); }));
  r.post('/:s', wrap(async (req, res) => {
    const s = req.params.s; if (!SCHEMAS[s]) throw new HttpError(404, 'Estrategia desconocida');
    const item = cleanItem(s, req.body);
    if (await col.countDocuments({ userId: req.uid, kind: 'example', strategy: s }) >= MAX_ITEMS[s]) throw new HttpError(400, `Máximo ${MAX_ITEMS[s]} ejemplos`);
    const { insertedId } = await col.insertOne({ userId: req.uid, kind: 'example', strategy: s, ...item, createdAt: new Date() });
    res.status(201).json({ _id: insertedId, ...item });
  }));
  r.delete('/:s/:id', wrap(async (req, res) => {
    if (!SCHEMAS[req.params.s] || !ObjectId.isValid(req.params.id)) throw new HttpError(400, 'Solicitud inválida');
    res.json({ deleted: (await col.deleteOne({ _id: new ObjectId(req.params.id), userId: req.uid, kind: 'example', strategy: req.params.s })).deletedCount });
  }));
  return r;
}
