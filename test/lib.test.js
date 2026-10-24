import test from 'node:test'; import assert from 'node:assert';
import { chunkText, cosine, normKey, validateExtraction, extractText } from '../lib.js';
test('chunkText divide en fragmentos acotados', () => { const c = chunkText('Frase de prueba. '.repeat(300), 500, 50); assert.ok(c.length > 5 && c.every(x => x.length <= 500)); });
test('cosine', () => { assert.equal(cosine([1, 0], [1, 0]), 1); assert.equal(cosine([1, 0], [0, 1]), 0); });
test('normKey unifica duplicados', () => assert.equal(normKey('  Álvaro  Pérez '), normKey('alvaro perez')));
test('validateExtraction descarta relaciones huérfanas y JSON inválido', () => {
  const r = validateExtraction('{"entities":[{"name":"RAG","type":"Concepto"},{"name":"MongoDB"}],"relations":[{"source":"RAG","target":"MongoDB","type":"usa"},{"source":"RAG","target":"X","type":"usa"}]}');
  assert.equal(r.entities.length, 2); assert.equal(r.relations.length, 1);
  assert.deepEqual(validateExtraction('no json'), { entities: [], relations: [] });
});
test('extractText: html, json y formato no soportado', async () => {
  assert.match(await extractText({ originalname: 'a.html', buffer: Buffer.from('<body><script>x</script><p>Hola</p></body>') }), /Hola/);
  assert.match(await extractText({ originalname: 'a.json', buffer: Buffer.from('{"a":1}') }), /"a": 1/);
  await assert.rejects(extractText({ originalname: 'a.exe', buffer: Buffer.from('') }));
});
