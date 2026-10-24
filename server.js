import 'dotenv/config';
import express from 'express';
import cookieParser from 'cookie-parser';
import rateLimit from 'express-rate-limit';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import multer from 'multer';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MongoClient, ObjectId } from 'mongodb';
import { tuneRouter, loadDomainPrompt } from './tune.js';
import { HttpError, chunkText, cosine, normKey, extractText, embed, chat, validateExtraction } from './lib.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const SECRET = process.env.JWT_SECRET;
if (!SECRET || SECRET.length < 16) { console.error('Define JWT_SECRET (16+ caracteres) en .env'); process.exit(1); }
const client = new MongoClient(process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017');
await client.connect();
const db = client.db(process.env.MONGODB_DB || 'rag_grafo');
const [users, docs, chunks, nodes, edges] = ['users', 'documents', 'chunks', 'nodes', 'edges'].map(n => db.collection(n));
await Promise.all([users.createIndex({ email: 1 }, { unique: true }), nodes.createIndex({ userId: 1, key: 1 }, { unique: true }),
  edges.createIndex({ userId: 1, key: 1 }, { unique: true }), chunks.createIndex({ userId: 1, docId: 1 }), docs.createIndex({ userId: 1 })]);

const app = express();
app.use(express.json({ limit: '1mb' }), cookieParser());
app.use(express.static(path.join(here, 'public')));
app.use('/vendor', express.static(path.join(here, 'node_modules/vis-network/standalone/umd')));
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024, files: 5 } });
const wrap = f => (req, res, next) => f(req, res, next).catch(next);

const auth = wrap(async (req, res, next) => {
  try { req.uid = new ObjectId(jwt.verify(req.cookies.token || '', SECRET).sub); } catch { throw new HttpError(401, 'No autenticado'); }
  next();
});
const setSession = (res, id) => res.cookie('token', jwt.sign({ sub: String(id) }, SECRET, { expiresIn: '7d' }), { httpOnly: true, sameSite: 'lax', secure: process.env.NODE_ENV === 'production', maxAge: 7 * 864e5 });
app.use('/api/tune', tuneRouter({ db, auth, wrap }));
const limiter = rateLimit({ windowMs: 15 * 60e3, limit: 30 });

app.post('/api/auth/register', limiter, wrap(async (req, res) => {
  const email = String(req.body.email || '').trim().toLowerCase(), password = String(req.body.password || '');
  if (!/^\S+@\S+\.\S+$/.test(email) || password.length < 8) throw new HttpError(400, 'Correo válido y contraseña de 8+ caracteres requeridos');
  try { const r = await users.insertOne({ email, hash: await bcrypt.hash(password, 12), createdAt: new Date() }); setSession(res, r.insertedId).json({ email }); }
  catch (e) { if (e.code === 11000) throw new HttpError(409, 'Ese correo ya está registrado'); throw e; }
}));
app.post('/api/auth/login', limiter, wrap(async (req, res) => {
  const u = await users.findOne({ email: String(req.body.email || '').trim().toLowerCase() });
  if (!u || !(await bcrypt.compare(String(req.body.password || ''), u.hash))) throw new HttpError(401, 'Credenciales incorrectas');
  setSession(res, u._id).json({ email: u.email });
}));
app.post('/api/auth/logout', (req, res) => res.clearCookie('token').json({ ok: true }));
app.get('/api/auth/me', auth, wrap(async (req, res) => res.json({ email: (await users.findOne({ _id: req.uid })).email })));

const DEFAULT_INSTR = 'Extrae personas, organizaciones, conceptos, tecnologías y lugares relevantes, con relaciones explícitas en el texto.';
async function extractFromChunk(text, related, instructions) {
  const sys = `Eres un extractor de grafos de conocimiento. ${instructions}\nResponde SOLO JSON: {"entities":[{"name","type","description"}],"relations":[{"source","target","type","evidence"}]}. Usa solo información del TEXTO (el CONTEXTO es de apoyo). "source" y "target" deben ser nombres presentes en entities. No inventes.`;
  return validateExtraction(await chat([{ role: 'system', content: sys }, { role: 'user', content: `CONTEXTO RELACIONADO:\n${related || '(ninguno)'}\n\nTEXTO:\n${text}` }], true));
}

app.post('/api/ingest', auth, upload.array('files', 5), wrap(async (req, res) => {
  const instructions = String(req.body.instructions || '').trim().slice(0, 1500) || DEFAULT_INSTR;
  const domainP = await loadDomainPrompt(db, req.uid);
  const sources = (req.files || []).map(f => ({ title: f.originalname, getText: () => extractText(f) }));
  const pasted = String(req.body.text || '').trim();
  if (pasted) sources.push({ title: String(req.body.title || '').trim().slice(0, 120) || 'Texto manual', getText: async () => pasted });
  if (!sources.length) throw new HttpError(400, 'Envía texto o al menos un archivo');
  const existing = await chunks.find({ userId: req.uid }, { projection: { text: 1, vector: 1 } }).toArray();
  const report = [];
  for (const src of sources) {
    const text = (await src.getText()).trim();
    if (text.length < 20) { report.push({ title: src.title, error: 'Sin texto extraíble' }); continue; }
    const parts = chunkText(text), vectors = await embed(parts);
    const { insertedId: docId } = await docs.insertOne({ userId: req.uid, title: src.title, chars: text.length, instructions, createdAt: new Date() });
    await chunks.insertMany(parts.map((t, i) => ({ userId: req.uid, docId, index: i, text: t, vector: vectors[i] })));
    let nN = 0, nE = 0;
    for (let i = 0; i < Math.min(parts.length, 30); i++) {
      const related = existing.map(c => ({ c, s: cosine(vectors[i], c.vector) })).sort((a, b) => b.s - a.s).slice(0, 2).filter(x => x.s > 0.35).map(x => x.c.text.slice(0, 500)).join('\n---\n');
      const { entities, relations } = await extractFromChunk(parts[i], related, instructions + domainP);
      for (const e of entities) {
        const r = await nodes.updateOne({ userId: req.uid, key: normKey(e.name) }, { $setOnInsert: { name: e.name, type: e.type, description: e.description, createdAt: new Date() }, $addToSet: { docIds: docId } }, { upsert: true });
        nN += r.upsertedCount;
      }
      for (const r of relations) {
        const sk = normKey(r.source), tk = normKey(r.target);
        const u = await edges.updateOne({ userId: req.uid, key: `${sk}|${r.type}|${tk}` }, { $setOnInsert: { from: sk, to: tk, type: r.type }, $addToSet: { docIds: docId, evidence: { docId, chunk: i, text: r.evidence } } }, { upsert: true });
        nE += u.upsertedCount;
      }
    }
    await docs.updateOne({ _id: docId }, { $set: { chunks: parts.length, newNodes: nN, newEdges: nE } });
    existing.push(...parts.map((t, i) => ({ text: t, vector: vectors[i] })));
    report.push({ title: src.title, chunks: parts.length, newNodes: nN, newEdges: nE, truncated: parts.length > 30 });
  }
  res.json({ report });
}));

app.get('/api/documents', auth, wrap(async (req, res) => res.json(await docs.find({ userId: req.uid }, { projection: { instructions: 0 } }).sort({ createdAt: -1 }).toArray())));
app.get('/api/graph', auth, wrap(async (req, res) => {
  const [n, e, d] = await Promise.all([nodes.find({ userId: req.uid }).toArray(), edges.find({ userId: req.uid }).toArray(), docs.find({ userId: req.uid }, { projection: { title: 1 } }).toArray()]);
  res.json({ nodes: n, edges: e, documents: Object.fromEntries(d.map(x => [x._id, x.title])) });
}));
app.post('/api/query', auth, wrap(async (req, res) => {
  const q = String(req.body.question || '').trim().slice(0, 500); if (!q) throw new HttpError(400, 'Pregunta vacía');
  const [qv] = await embed([q]);
  const all = await chunks.find({ userId: req.uid }).toArray();
  const top = all.map(c => ({ c, s: cosine(qv, c.vector) })).sort((a, b) => b.s - a.s).slice(0, 5);
  if (!top.length) throw new HttpError(400, 'Aún no hay documentos indexados');
  const ctx = top.map((x, i) => `[${i + 1}] ${x.c.text}`).join('\n\n');
  const answer = await chat([{ role: 'system', content: 'Responde en español solo con el CONTEXTO y cita fragmentos como [n]. Si no alcanza, dilo.' + await loadDomainPrompt(db, req.uid) }, { role: 'user', content: `CONTEXTO:\n${ctx}\n\nPREGUNTA: ${q}` }]);
  res.json({ answer, sources: top.map(x => ({ docId: x.c.docId, score: +x.s.toFixed(3), text: x.c.text.slice(0, 200) })) });
}));

app.use((err, req, res, next) => {
  const status = err.status || (err.code === 'LIMIT_FILE_SIZE' ? 413 : 500);
  if (status === 500) console.error(err);
  res.status(status).json({ error: status === 500 ? 'Error interno' : err.message });
});
app.listen(process.env.PORT || 3000, () => console.log(`http://localhost:${process.env.PORT || 3000}`));
