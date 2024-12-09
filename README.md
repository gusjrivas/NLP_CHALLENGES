# Desafíos de la materia Procesamiento del Lenguaje Natural
## Posgrado CEIA - UBA - Cohorte 16 2024

![Texto alternativo](logoFIUBA.jpg)


![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17%2B-orange.svg)](https://www.tensorflow.org/install)
### Alumno: Gustavo Julián Rivas - N° SIU: a1620

En este repositorio se encuentran las soluciones para los desafios de la materia Procesamiento de lenguaje natural de la Especialización en Inteligencia Artificial (CEIA) de la Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA).

# Desafío 1: Análisis de Similaridad entre Documentos
En este desafío, se realizó un análisis de similaridad de documentos utilizando el conjunto de datos 20 Newsgroups. Se aplicaron las siguientes técnicas:

Vectorización y análisis de similaridad: Los documentos fueron vectorizados utilizando TF-IDF para representar el contenido textual en un espacio de características numéricas. Posteriormente, se midió la similaridad de coseno entre un documento seleccionado y los 5 documentos más similares, buscando analizar si los documentos más similares pertenecen a la misma categoría.

Modelos de ML: Se entrenaron modelos de Naive Bayes (Multinomial y ComplementNB) para realizar la tarea de clasificación de documentos. Para maximizar el rendimiento, se utilizó una búsqueda aleatoria de hiperparámetros, optimizando la métrica f1-score macro en el conjunto de datos de prueba.

Vectorización de palabras: Se estudió la similaridad entre palabras, seleccionando 5 palabras y analizando las palabras más similares utilizando el mismo enfoque de vectorización.

El notebook con la solución se encuentra en  https://github.com/gusjrivas/NLP_CHALLENGES/tree/main/Desafio_1

# Desafio 2: Word2Vec y Análisis de Similaridad entre Palabras
En este desafío, se trabajó con una muestra de reseñas de películas SST-2 (Stanford Sentiment Treebank 2) este es uno de los conjuntos de datos incluidos en GLUE y está diseñado específicamente para el análisis de sentimientos. SST-2 consiste en frases cortas extraídas de reseñas de películas y sus sentimientos. Se estudió la similaridad entre palabras, seleccionando 5 palabras y analizando las palabras más similares utilizando el mismo enfoque de vectorización.

El notebook con la solución se encuentra en  https://github.com/gusjrivas/NLP_CHALLENGES/tree/main/Desafio_2


# Desafío 3: Generación de Texto con LSTM basado en el libro Martín Fierro
En este desafío, se trabajó con una publicación de Martín Fierro para entrenar un modelo LSTM de generación de texto. Se trabajó en lo siguiente:

Carga y Preparación de Datos: Se cargaron los datos del publicación-martin-fierro.pdf y se preprocesaron para segmentar el texto en palabras. Luego, se entrenó un Tokenizer para convertir las palabras en índices numéricos. Posteriormente, se dividió el conjunto de datos en entrenamiento y validación, y se aplicó padding para asegurar que todas las secuencias tuvieran la misma longitud.

Entrenamiento del Modelo LSTM: Este modelo NLP utiliza una arquitectura secuencial con embeddings y LSTM para procesar texto, optimizado para tareas como generación y predicción de palabras en secuencia. Comienza con una capa de embeddings que convierte cada palabra en un vector denso de 16 dimensiones, facilitando una representación numérica del texto. A continuación, dos capas LSTM con 32 unidades cada una capturan relaciones secuenciales en el texto, permitiendo que el modelo aprenda patrones contextuales a largo plazo, mientras que una capa de dropout ayuda a reducir el sobreajuste, mejorando la capacidad de generalización del modelo.

La capa final es una densa con activación softmax, que convierte las características aprendidas en una probabilidad sobre el vocabulario, permitiendo al modelo predecir la siguiente palabra. 

el notebook con la solución se encuentra en  https://github.com/gusjrivas/NLP_CHALLENGES/tree/main/Desafio_3


# Desafío 4: LSTM Bot QA

En este desafío, se trabajó con un conjunto de datos de conversaciones para entrenar un bot de preguntas y respuestas (QA) utilizando una arquitectura encoder-decoder con LSTM y embeddings preentrenados de FastText. A continuación, se detallan las etapas y características principales del proyecto:

## 1. Carga y Preparación de Datos

- Se utilizó el archivo `data_volunteers.json`, que contiene diálogos entre un bot y usuarios.
- Las oraciones de entrada y salida fueron limpiadas y preparadas añadiendo los tokens `<sos>` y `<eos>` para marcar el comienzo y final de las oraciones.
- Se descartaron las oraciones que superaban una longitud de 8 palabras para garantizar consistencia en las secuencias.

## 2. Tokenización y Padding

- Se utilizó el `Tokenizer` de Keras para convertir las oraciones en secuencias de tokens.
- Se crearon los diccionarios `word2idx_inputs` y `word2idx_outputs`.
- Las secuencias fueron ajustadas con `padding` para tener una longitud uniforme, generando:
  - `encoder_input_sequences`
  - `decoder_input_sequences`
  - `decoder_targets` (salidas categorizadas).

## 3. Preparación de Embeddings

- Se utilizaron embeddings preentrenados de FastText para inicializar la capa de embeddings del modelo.
- La clase `FasttextEmbeddings` fue implementada para manejar la matriz de embeddings.
- Se generó una matriz de embeddings adaptada al vocabulario del conjunto de datos.

## 4. Modelos Implementados

### Modelo 1: LSTM Encoder-Decoder

- **Estructura**:
  - Encoder: Una capa LSTM con 256 unidades.
  - Decoder: Una capa LSTM con 256 unidades conectada a una capa densa con activación `softmax`.
- **Resultados**:
  - Respuestas coherentes y estructuradas, aunque limitadas por el tamaño del dataset.

### Modelo 2: Bidirectional LSTM con Atención

- **Estructura**:
  - Encoder: Una capa Bidirectional LSTM con 512 unidades, concatenando los estados hacia adelante y hacia atrás.
  - Decoder: Una LSTM con 1024 unidades y un mecanismo de atención para resaltar las partes relevantes de la entrada.
  - Regularización: Se aplicó L2 en la capa densa de salida.
- **Resultados**:
  - Mejor capacidad para responder preguntas complejas, retrasando el overfitting gracias a la atención.

## 5. Entrenamiento

- Ambos modelos fueron entrenados con:
  - Callbacks: `EarlyStopping`, `ModelCheckpoint` y `ReduceLROnPlateau`.
  - Optimizador: Adam con tasas de aprendizaje adaptativas.
  - Pérdida: `categorical_crossentropy`.
- El mejor modelo fue guardado automáticamente según la métrica de validación.

## 6. Inferencia y Generación de Respuestas

- Los modelos fueron evaluados con preguntas predefinidas y aleatorias. Algunos ejemplos de respuestas incluyen:

### Modelo 1
- **Pregunta**: "Where are you from?"
  - **Respuesta**: "I am from the US"
- **Pregunta**: "Do you read?"
  - **Respuesta**: "I do not like to read"

### Modelo 2
- **Pregunta**: "Do you read?"
  - **Respuesta**: "I like to read"
- **Pregunta**: "Hello, do you play any sports?"
  - **Respuesta**: "Yes"

Aunque el tamaño reducido del dataset limitó el desempeño, el modelo con atención logró un mejor rendimiento en general.

## 7. Conclusiones

- La incorporación de atención y Bidirectional LSTM demostró ser beneficiosa para mejorar la calidad de las respuestas.
- Se identificaron limitaciones significativas debido al dataset, lo que sugiere la necesidad de un conjunto de datos más grande para explorar modelos más complejos.

El notebook con la solución se encuentra en:  https://github.com/gusjrivas/NLP_CHALLENGES/tree/main/Desafio_4




