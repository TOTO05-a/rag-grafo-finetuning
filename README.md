# Grafo RAG — RAG + grafo de conocimiento (Node.js, Express, MongoDB, OpenRouter)

## Qué hace
1. Registro/login (bcrypt + JWT en cookie httpOnly; todas las rutas de datos filtran por usuario).
2. Pantalla **Ingesta** (`/`): texto o archivos PDF, DOCX, TXT, MD, CSV, JSON, HTML + instrucciones de refinamiento configurables.
3. Pipeline RAG real: extracción de texto → fragmentos (1200 car., solape 150) → embeddings (OpenRouter) → para cada fragmento se recuperan los 2 más similares ya guardados (coseno) como contexto → el LLM extrae entidades y relaciones en JSON.
4. Validación (JSON, longitudes, relaciones con extremos existentes) y deduplicación (clave normalizada sin tildes/mayúsculas; índices únicos en MongoDB). Cada nodo/relación conserva los documentos de origen y la evidencia.
5. Pantalla **Grafo** (`/graph.html`): vis-network con búsqueda, zoom, selección, detalle, vecinos navegables, colores por tipo y consulta RAG al corpus.
6. MongoDB: colecciones `users`, `documents`, `chunks`, `nodes`, `edges`.

## Instalación
```bash
npm install
cp .env.example .env   # completa JWT_SECRET y OPENROUTER_API_KEY
# MongoDB local (mongod) o Atlas gratuito en MONGODB_URI
npm start              # http://localhost:3000
npm test
```
Modelos configurables en `.env` (`CHAT_MODEL`, `EMBED_MODEL`). Elige modelos baratos o gratuitos de OpenRouter para no generar gasto.

## Demo en clase
Regístrate → pega un texto (o sube un PDF) → procesa → abre "Grafo" → busca un nodo, haz clic y recorre sus relaciones → pregunta algo en el panel RAG.

## Explicación para defenderlo
- **RAG ≠ Fine-Tuning**: no se cambian los pesos del modelo; se recupera contexto relevante y se inyecta en el prompt (la analogía del "examen a libro abierto" de la clase).
- Los embeddings convierten texto en vectores; la similitud de coseno encuentra fragmentos parecidos (igual que `vector.service.js` de la clase, pero con embeddings reales).
- El LLM no dibuja nada: devuelve JSON; el servidor lo valida y lo guarda; vis-network lo dibuja.

## Limitaciones
- La búsqueda vectorial se calcula en Node (suficiente para cientos de fragmentos; para escalar usar Atlas Vector Search).
- Se extraen entidades de los primeros 30 fragmentos por documento (control de coste).
- Sin borrado de documentos, sin protección CSRF más allá de `SameSite=Lax`, sin reranker.

---
# Estrategias de Fine-Tuning (`/tune.html`, pestaña "Estrategias")

Cinco pestañas en la misma interfaz, todas conectadas al backend (`tune.js`, colección `tuning`). **Ninguna modifica los pesos de un LLM**: OpenRouter no ofrece fine-tuning y no hay GPU local. Lo que sí es real en cada una:

| Estrategia | Qué ocurre realmente | Cómo probarla |
|---|---|---|
| Transfer Learning | Entrenamiento real de una capa softmax (descenso de gradiente, en Node) sobre embeddings congelados. Reporta precisión de entrenamiento y en ejemplos retenidos (si hay 10+). | Añade 6+ ejemplos con 2+ etiquetas (2+ por etiqueta) → *Entrenar* → *Predecir*. |
| SFT | Curación de pares prompt→respuesta y exportación JSONL (formato chat). **No se entrena aquí.** Alternativa realista: usar el JSONL en Unsloth/Colab con un modelo abierto. | Añade pares → *Exportar JSONL*. |
| Few-shot | Prompting: se eligen hasta *k* ejemplos por similitud de embeddings y se envían como demostraciones. Sin entrenamiento. | Añade ejemplos → escribe entrada → *Ejecutar*. |
| Instruction | Dataset (instrucción, entrada, salida) exportable en formato Alpaca; al ejecutar una instrucción idéntica se usan hasta 3 ejemplos como demos. Sin entrenamiento. | Añade ejemplos → *Ejecutar* / *Exportar*. |
| Dominio | Perfil (glosario, tipos de entidad, estilo) inyectado en la extracción del grafo, en las preguntas RAG y en Few-shot/Instruction. El glosario se puede sugerir desde tus documentos. Sin entrenamiento. | Guarda el perfil → procesa un documento nuevo en Ingesta → compara el grafo. |

Validaciones: campos obligatorios y máximos de longitud, límites de ejemplos por estrategia, aislamiento por usuario, errores claros del servidor. Requiere `OPENROUTER_API_KEY` (embeddings/chat); usa modelos baratos o gratuitos.
