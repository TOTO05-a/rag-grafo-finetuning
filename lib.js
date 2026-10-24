import pdf from 'pdf-parse/lib/pdf-parse.js';
import mammoth from 'mammoth';
import * as cheerio from 'cheerio';

export class HttpError extends Error { constructor(s, m) { super(m); this.status = s; } }

export function chunkText(text, size = 1200, overlap = 150) {
  const t = text.replace(/\r/g, '').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
  const out = []; let i = 0;
  while (i < t.length) {
    let end = Math.min(i + size, t.length);
    if (end < t.length) { const cut = Math.max(t.lastIndexOf('\n', end), t.lastIndexOf('. ', end)); if (cut > i + size / 2) end = cut + 1; }
    const piece = t.slice(i, end).trim(); if (piece) out.push(piece);
    if (end >= t.length) break; i = end - overlap;
  }
  return out;
}
export function cosine(a, b) {
  let d = 0, x = 0, y = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] ** 2; y += b[i] ** 2; }
  return d / (Math.sqrt(x * y) || 1);
}
export const normKey = s => String(s).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();

export async function extractText(file) {
  const ext = (file.originalname.split('.').pop() || '').toLowerCase();
  const buf = file.buffer;
  if (ext === 'pdf') return (await pdf(buf)).text;
  if (ext === 'docx') return (await mammoth.extractRawText({ buffer: buf })).value;
  if (['html', 'htm'].includes(ext)) { const $ = cheerio.load(buf.toString('utf8')); $('script,style,noscript').remove(); return $('body').text() || $.root().text(); }
  if (ext === 'json') { try { return JSON.stringify(JSON.parse(buf.toString('utf8')), null, 1); } catch { throw new HttpError(400, `JSON inválido: ${file.originalname}`); } }
  if (['txt', 'md', 'markdown', 'csv'].includes(ext)) return buf.toString('utf8');
  throw new HttpError(400, `Formato no soportado: .${ext}`);
}

async function or(path, body) {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) throw new HttpError(503, 'Falta OPENROUTER_API_KEY en .env');
  const r = await fetch('https://openrouter.ai/api/v1/' + path, { method: 'POST', headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new HttpError(502, `OpenRouter ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
}
export async function embed(texts) {
  const out = [];
  for (let i = 0; i < texts.length; i += 64) {
    const j = await or('embeddings', { model: process.env.EMBED_MODEL || 'openai/text-embedding-3-small', input: texts.slice(i, i + 64) });
    out.push(...j.data.sort((a, b) => a.index - b.index).map(d => d.embedding));
  }
  return out;
}
export async function chat(messages, json = false) {
  const j = await or('chat/completions', { model: process.env.CHAT_MODEL || 'openai/gpt-4o-mini', messages, temperature: 0.1, ...(json && { response_format: { type: 'json_object' } }) });
  return j.choices?.[0]?.message?.content || '';
}
const s = (v, n) => String(v ?? '').trim().slice(0, n);
export function validateExtraction(raw) {
  let o; try { o = JSON.parse(String(raw).replace(/^```json|```$/gim, '').trim()); } catch { return { entities: [], relations: [] }; }
  const entities = (Array.isArray(o.entities) ? o.entities : []).map(e => ({ name: s(e?.name, 120), type: s(e?.type || 'Concepto', 40), description: s(e?.description, 400) })).filter(e => normKey(e.name).length > 1);
  const names = new Set(entities.map(e => normKey(e.name)));
  const relations = (Array.isArray(o.relations) ? o.relations : []).map(r => ({ source: s(r?.source, 120), target: s(r?.target, 120), type: s(r?.type, 60).toLowerCase(), evidence: s(r?.evidence, 300) }))
    .filter(r => r.type && names.has(normKey(r.source)) && names.has(normKey(r.target)) && normKey(r.source) !== normKey(r.target));
  return { entities, relations };
}
