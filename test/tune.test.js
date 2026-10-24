import test from 'node:test'; import assert from 'node:assert';
import { cleanItem, toJSONL, trainSoftmax, predictProba, accuracy, domainPrompt } from '../tune.js';
test('trainSoftmax aprende clases separables', () => {
  const X = [[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1], [0.1, 0.9], [0.2, 0.8]], y = [0, 0, 0, 1, 1, 1];
  const m = trainSoftmax(X, y, 2); assert.equal(accuracy(m, X, y), 1);
  assert.ok(predictProba(m, [1, 0])[0] > 0.8);
});
test('cleanItem valida obligatorios y longitud', () => {
  assert.throws(() => cleanItem('sft', { prompt: 'hola' })); assert.throws(() => cleanItem('transfer', { text: 'x'.repeat(2001), label: 'a' }));
  assert.deepEqual(cleanItem('instruction', { instruction: 'Resume', output: 'ok' }), { instruction: 'Resume', input: '', output: 'ok' });
});
test('toJSONL: SFT en formato chat y válido línea a línea', () => {
  const l = toJSONL('sft', [{ prompt: 'p', response: 'r' }]).trim().split('\n').map(JSON.parse);
  assert.deepEqual(l[0].messages.map(m => m.role), ['user', 'assistant']);
});
test('domainPrompt incluye glosario y vacío sin perfil', () => {
  assert.equal(domainPrompt(null), ''); assert.match(domainPrompt({ name: 'Salud', glossary: [{ term: 'HTA', definition: 'hipertensión' }] }), /HTA: hipertensión/);
});
