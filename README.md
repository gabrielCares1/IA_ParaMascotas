# 🩺 NutriVet AI: Asistente Nutricional Preventivo con Arquitectura RAG

**NutriVet AI** es una solución de ingeniería de software que aplica modelos de lenguaje de gran tamaño (LLMs) y técnicas de recuperación de información para transformar la medicina preventiva en mascotas. El sistema genera planes nutricionales orgánicos y personalizados, mitigando riesgos de salud hereditarios mediante el análisis de datos clínicos y genéticos en tiempo real.

---

## 🚀 El Desafío Técnico
En la nutrición animal existe una brecha crítica entre el conocimiento clínico y la dieta diaria. La mayoría de los propietarios dependen de alimentos comerciales genéricos que no consideran las predisposiciones genéticas. Este proyecto resuelve dicho problema implementando una arquitectura de **Generación Aumentada por Recuperación (RAG)**, garantizando que las respuestas de la IA no sean "alucinaciones", sino que estén fundamentadas exclusivamente en literatura veterinaria oficial.

---

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.10+
* **Orquestación de IA:** [LangChain](https://www.langchain.com/) (LCEL & Chain of Thought).
* **Modelos de Lenguaje:** GPT-4o a través de GitHub Models API.
* **Embeddings:** text-embedding-3-small (OpenAI).
* **Base de Datos Vectorial:** [ChromaDB](https://www.trychroma.com/) (Persistencia y búsqueda semántica).
* **Procesamiento de Documentos:** PyPDF para la ingesta de guías clínicas de la **WSAVA**.
* **Frontend:** [Streamlit](https://streamlit.io/) para una interfaz web reactiva.

---

## 📂 Estructura del Proyecto
```text
Proyecto IA para mascotas/
├── app.py                # Aplicación principal (Interfaz + Lógica RAG)
├── datos_razas.json      # Base de conocimientos genéticos (Estructura JSON)
├── WSAVA-Manual.pdf      # Literatura médica de referencia (Contexto clínico)
├── README.md             # Documentación técnica
└── requirements.txt      # Dependencias del sistema
