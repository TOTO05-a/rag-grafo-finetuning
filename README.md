# RAG-Grafo: documentos, IA y grafos de conocimiento

Este proyecto lo hice para trabajar con inteligencia artificial de una forma más visual e interactiva. La idea es poder ingresar documentos, procesarlos con IA y convertir la información en un grafo que permita explorar entidades, sus relaciones y hacer preguntas sobre el contenido.

También integré una sección con cinco estrategias relacionadas con el ajuste y uso de modelos de IA, cada una con un funcionamiento diferente.

## ¿Qué puede hacer?

- Crear una cuenta e iniciar sesión.
- Ingresar texto o cargar documentos PDF, DOCX, TXT, Markdown, CSV, JSON y HTML.
- Procesar la información mediante RAG y generar un grafo de conocimiento.
- Explorar nodos y relaciones, buscar información y consultar el contenido con preguntas.
- Cambiar entre cinco estrategias de IA desde una misma interfaz.
- Guardar información y ejemplos asociados a cada usuario.

## Tecnologías utilizadas

- **Node.js y Express:** servidor y lógica de la aplicación.
- **MongoDB:** almacenamiento de usuarios, documentos y datos del grafo.
- **OpenRouter:** acceso a modelos de lenguaje y embeddings.
- **HTML, CSS y JavaScript:** interfaz de usuario.
- **vis-network:** visualización interactiva del grafo.

## ¿Cómo funciona?

Primero ingreso un texto o documento. El sistema extrae su contenido, lo divide en fragmentos y utiliza embeddings para encontrar información relacionada. Después, un modelo de IA ayuda a identificar entidades y relaciones, que se validan y guardan en MongoDB.

Finalmente, puedo visualizar el resultado como un grafo, explorar sus conexiones y realizar preguntas utilizando la información recuperada de los documentos.

## Las cinco estrategias de IA

En la sección **Estrategias** (`/tune.html`) integré cinco opciones dentro del mismo proyecto. No todas realizan entrenamiento real, por lo que es importante distinguir qué hace cada una.

| Estrategia | Funcionamiento |
|---|---|
| Transfer Learning | Entrena un clasificador sobre embeddings existentes para realizar predicciones. El modelo que genera los embeddings permanece sin cambios. |
| Supervised Fine-Tuning (SFT) | Permite preparar ejemplos de entrada y respuesta y exportarlos en formato JSONL para entrenar un modelo compatible en otra herramienta. |
| Few-shot | Utiliza ejemplos similares como referencia para orientar las respuestas de la IA, sin modificar los pesos del modelo. |
| Instruction Fine-Tuning | Permite organizar ejemplos de instrucciones, entradas y salidas, y utilizarlos como demostraciones o exportarlos para entrenamiento posterior. |
| Domain-Specific Fine-Tuning | Permite definir un perfil de dominio con un glosario, tipos de entidades y estilo para adaptar la extracción de información y las consultas. |

**Importante:** la aplicación no realiza actualmente el entrenamiento de los pesos de un modelo de lenguaje. Transfer Learning sí entrena un clasificador sobre embeddings congelados; las demás estrategias preparan datos o adaptan el contexto y las instrucciones. Esto permite explorar sus diferencias sin presentar como entrenamiento algo que no lo es.

## Instalación y ejecución

Necesito tener Node.js instalado, acceso a MongoDB local o Atlas y una clave de OpenRouter para las funciones que utilizan modelos externos.

1. Instalo las dependencias:

   ```bash
   npm install
   ```

2. Copio `.env.example` a un archivo llamado `.env` y configuro las variables necesarias, incluyendo `JWT_SECRET` y `OPENROUTER_API_KEY`.

3. Inicio MongoDB y compruebo que la conexión configurada sea correcta.

4. Inicio la aplicación:

   ```bash
   npm start
   ```

5. Abro [http://localhost:3000](http://localhost:3000) en el navegador.

Para ejecutar las pruebas:

```bash
npm test
```

Los modelos utilizados se pueden configurar mediante las variables `CHAT_MODEL` y `EMBED_MODEL`. Las llamadas a OpenRouter pueden tener costes según el modelo y las condiciones de uso, por lo que conviene comprobar sus precios antes de utilizarlas.

## Cómo probar el proyecto

1. Creo una cuenta e inicio sesión.
2. Ingreso un texto o cargo un documento desde la pantalla principal.
3. Proceso el contenido y abro la sección **Grafo**.
4. Exploro las entidades y relaciones generadas y pruebo las consultas RAG.
5. Entro en **Estrategias** y pruebo las cinco opciones con sus respectivos ejemplos.

## Lo que aprendí y lo que queda pendiente

Este proyecto me permite trabajar con RAG, embeddings, extracción de entidades, grafos de conocimiento y diferentes formas de adaptar el comportamiento de una IA.

Todavía hay limitaciones. La búsqueda de fragmentos se realiza en Node.js y está pensada para conjuntos de datos relativamente pequeños. Algunas funciones dependen de MongoDB y OpenRouter, y las estrategias de entrenamiento de modelos de lenguaje requieren herramientas adicionales.

Las pruebas automatizadas cubren parte de la lógica del proyecto, pero todavía es necesario verificar el funcionamiento completo con las conexiones reales configuradas.

---

Este proyecto sigue en desarrollo. La idea es continuar mejorándolo, probar sus funciones y explorar nuevas formas de integrar inteligencia artificial en una aplicación práctica.