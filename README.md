# 🧠 Proyecto RAG con OpenRouter + Qdrant + SentenceTransformers

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** que combina un motor de búsqueda semántica con un modelo de lenguaje grande (LLM) para responder preguntas basadas en una base de conocimiento local.

---

## 🚀 Descripción General

El sistema utiliza tres componentes principales:

1. **SentenceTransformers**  
   Genera embeddings (representaciones numéricas) de texto para comparar similitud semántica.

2. **Qdrant**  
   Base de datos vectorial que almacena los embeddings y permite búsquedas eficientes por similitud.

3. **OpenRouter (modelo GPT)**  
   Genera respuestas coherentes y contextuales usando la información recuperada desde Qdrant.

---

## 🧩 Estructura del Proyecto

