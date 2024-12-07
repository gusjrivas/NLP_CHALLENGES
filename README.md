# Desafíos de la materia Procesamiento del Lenguaje Natural
## Posgrado CEIA - UBA - Cohorte 16 2024

![Texto alternativo](logoFIUBA.jpg)


![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17%2B-orange.svg)](https://www.tensorflow.org/install)
### Alumno: Gustavo Julián Rivas - N° SIU: a1620

En este repositorio se encuentran las soluciones para los desafios de la materia Procesamiento de lenguaje natural de la Especialización en Inteligencia Artificial (CEIA) de la Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA).

### Desafío 1: Análisis de Similaridad entre Documentos
En este desafío, se realizó un análisis de similaridad de documentos utilizando el conjunto de datos 20 Newsgroups. Se aplicaron las siguientes técnicas:

Vectorización y análisis de similaridad: Los documentos fueron vectorizados utilizando TF-IDF para representar el contenido textual en un espacio de características numéricas. Posteriormente, se midió la similaridad de coseno entre un documento seleccionado y los 5 documentos más similares, buscando analizar si los documentos más similares pertenecen a la misma categoría.

Modelos de ML: Se entrenaron modelos de Naive Bayes (Multinomial y ComplementNB) para realizar la tarea de clasificación de documentos. Para maximizar el rendimiento, se utilizó una búsqueda aleatoria de hiperparámetros, optimizando la métrica f1-score macro en el conjunto de datos de prueba.

Vectorización de palabras: Se estudió la similaridad entre palabras, seleccionando 5 palabras y analizando las palabras más similares utilizando el mismo enfoque de vectorización.

El notebook con la solución se encuentra en  https://github.com/gusjrivas/NLP_CHALLENGES/tree/main/Desafio_1

### Desafio 2: Word2Vec y Análisis de Similaridad entre Palabras
En este desafío, se trabajó con una muestra de reseñas de películas SST-2 (Stanford Sentiment Treebank 2) este es uno de los conjuntos de datos incluidos en GLUE y está diseñado específicamente para el análisis de sentimientos. SST-2 consiste en frases cortas extraídas de reseñas de películas y sus sentimientos. Se estudió la similaridad entre palabras, seleccionando 5 palabras y analizando las palabras más similares utilizando el mismo enfoque de vectorización.

El notebook con la solución se encuentra en  https://github.com/gusjrivas/NLP_CHALLENGES/tree/main/Desafio_2


### Desafío 3: Generación de Texto con LSTM basado en el libro Martín Fierro
En este desafío, se trabajó con una publicación de Martín Fierro para entrenar un modelo LSTM de generación de texto. Se trabajó en lo siguiente:

Carga y Preparación de Datos: Se cargaron los datos del publicación-martin-fierro.pdf y se preprocesaron para segmentar el texto en palabras. Luego, se entrenó un Tokenizer para convertir las palabras en índices numéricos. Posteriormente, se dividió el conjunto de datos en entrenamiento y validación, y se aplicó padding para asegurar que todas las secuencias tuvieran la misma longitud.

Entrenamiento del Modelo LSTM: Este modelo NLP utiliza una arquitectura secuencial con embeddings y LSTM para procesar texto, optimizado para tareas como generación y predicción de palabras en secuencia. Comienza con una capa de embeddings que convierte cada palabra en un vector denso de 16 dimensiones, facilitando una representación numérica del texto. A continuación, dos capas LSTM con 32 unidades cada una capturan relaciones secuenciales en el texto, permitiendo que el modelo aprenda patrones contextuales a largo plazo, mientras que una capa de dropout ayuda a reducir el sobreajuste, mejorando la capacidad de generalización del modelo.

La capa final es una densa con activación softmax, que convierte las características aprendidas en una probabilidad sobre el vocabulario, permitiendo al modelo predecir la siguiente palabra. 

el notebook con la solución se encuentra en  https://github.com/gusjrivas/NLP_CHALLENGES/tree/main/Desafio_3


