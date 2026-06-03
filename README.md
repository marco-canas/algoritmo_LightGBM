Este es un **Diseño de Curso Completo** estructurado bajo un enfoque teórico-práctico y progresivo. Está pensado para profesionales o investigadores en Ciencia de Datos, Epidemiología Matemática o Meteorología que desean dominar el algoritmo **LightGBM** adaptado específicamente a la naturaleza compleja, no lineal y autorregresiva de las series temporales de salud y clima.

---

# Diseño de Curso: Machine Learning para Clima y Salud con LightGBM

## 📋 Ficha Técnica del Curso

* **Nombre del Curso:** Predicción de Epidemias y Variables Climáticas con LightGBM: Enfoque de Series de Tiempo
* **Duración Total:** 40 horas (sugerido: 8 semanas, 5 horas por semana)
* **Nivel:** Intermedio - Avanzado
* **Requisitos Previos:** Python intermedio (Pandas, NumPy), nociones básicas de Machine Learning (regresión) y conceptos elementales de series de tiempo (rezagos, estacionalidad).

---

## 🎯 Objetivos de Aprendizaje

Al finalizar este curso, el estudiante será capaz de:

1. Transformar datos crudos de planillas epidemiológicas y estaciones meteorológicas en matrices aptas para algoritmos basados en árboles de decisión.
2. Implementar estrategias robustas de validación cruzada temporal sin cometer fuga de datos (*data leakage*).
3. Aplicar técnicas avanzadas de reducción de atributos para mitigar la colinealidad del clima.
4. Optimizar hiperparámetros críticos de LightGBM para evitar el subajuste (líneas planas) y el sobreajuste (memorización de ruido).
5. Interpretar las predicciones del modelo desde una perspectiva biológica y climática.

---

## 📚 Estructura de Módulos y Contenido

### Módulo 1: Fundamentos de LightGBM y Particularidades de Clima y Salud

* **Clase 1.1:** Arquitectura de LightGBM: Crecimiento por hojas (*Leaf-wise*) vs. Crecimiento por niveles (*Level-wise*). ¿Por qué es óptimo para grandes volúmenes de datos exógenos?
* **Clase 1.2:** La naturaleza de los datos meteorológicos (continuos, cíclicos, colineales) y epidemiológicos (conteos discretos, distribución sesgada, ceros estructurales).
* **Clase 1.3:** Preparación del entorno: Configuración de entornos virtuales, LightGBM, Scikit-Learn y herramientas de visualización.

### Módulo 2: Ingeniería de Características (Feature Engineering) Temporal

* **Clase 2.1:** Dinámica Autorregresiva: Creación y lógica biológica de los rezagos de la variable objetivo (ej. casos de dengue a las semanas $t-1$ a $t-4$).
* **Clase 2.2:** Dinámica Exógena y Retardos Biológicos: Modelado del impacto desfasado del clima (ej. cómo la lluvia de hace 6 semanas afecta los casos de hoy).
* **Clase 2.3:** Transformaciones matemáticas aplicadas a la salud: Transformación logarítmica ($\ln(x+1)$) y Box-Cox para estabilizar la varianza ante brotes epidémicos.
* **Clase 2.4:** Manejo correcto de la estacionalidad: El dilema de incluir variables de calendario (mes/semana) vs. depender puramente de la biología y la meteorología.

### Módulo 3: Selección de Atributos y Control de Redundancia

* **Clase 3.1:** El problema de la alta dimensionalidad al multiplicar rezagos climáticos.
* **Clase 3.2:** Algoritmo de Reducción de Atributos vía Correlación de Spearman.
* **Clase 3.3:** Programación de filtros personalizados: Selección de los componentes con mayor fuerza de asociación y eliminación automática de variables predictoras internamente colineales mediante umbrales estrictos.

### Módulo 4: Validación Cruzada en Series de Tiempo

* **Clase 4.1:** Por qué la validación cruzada tradicional (K-Fold aleatorio) destruye la estructura temporal y genera *data leakage*.
* **Clase 4.2:** Configuración de `TimeSeriesSplit` de Scikit-Learn. Determinación del horizonte de pronóstico óptimo ($H$) y del tamaño máximo de la ventana de entrenamiento (`max_train_size`).

* **Clase 4.3:** Monitoreo del comportamiento de las métricas de error (MAE, RMSE) a lo largo de múltiples ventanas del pasado (Folds).

### Módulo 5: Ajuste Fino y Búsqueda en Cuadrícula (Grid Search)

* **Clase 5.1:** Comprensión de los hiperparámetros anti-overfitting y anti-underfitting en LightGBM:
* Complejidad: `max_depth`, `num_leaves`, `min_child_samples`.
* Ritmo de aprendizaje: `learning_rate` y `n_estimators`.
* Regularización: `reg_alpha` (L1) y `reg_lambda` (L2).
* Aleatoriedad: `colsample_bytree` y `subsample`.


* **Clase 5.2:** Automatización de una grilla de hiperparámetros manual integrada sobre `TimeSeriesSplit`.
* **Clase 5.3:** Estrategias para evitar predicciones rígidas ("líneas planas") en el conjunto de test.

### Módulo 6: Evaluación Avanzada e Interpretabilidad Climático-Epidemiológica

* **Clase 6.1:** Diagnóstico del reporte tabular de desempeño: Análisis de brechas (*gaps*) entre métricas de entrenamiento y validación.
* **Clase 6.2:** Identificación de quiebres estructurales y anomalías temporales (ej. brotes epidémicos imprevistos inducidos por eventos climáticos extremos).
* **Clase 6.3:** Interpretabilidad del modelo: Uso de Importancia de Variables intrínseca de LightGBM y valores SHAP (*SHapley Additive exPlanations*) para mapear el peso real del clima sobre la salud.

---

## 🛠️ Metodología y Estrategia Evaluativa

El curso se plantea bajo la metodología de **Aprendizaje Basado en Proyectos (ABP)**.

1. **Laboratorios Semanales (40%):** Código guiado paso a paso para limpiar datos, estructurar desfases temporales y entrenar modelos base.
2. **Talleres de Diagnóstico (20%):** Ejercicios donde se entrega un modelo defectuoso (con subajuste u overfitting extremo) para que el estudiante corrija su arquitectura y su grilla de hiperparámetros.
3. **Proyecto Final Integrador (40%):** * **Entregable:** Construcción de un pipeline completo en Python utilizando un dataset real de una enfermedad transmitida por vectores (Dengue, Malaria o Zika) combinado con registros de satélites meteorológicos o estaciones locales.
* **Defensa:** Presentación de un reporte tabular de evaluación cruzada y gráficos de ajuste histórico vs. pronósticos fuera de muestra, justificando las métricas obtenidas mediante la lógica del ecosistema analizado.



---

## 💻 Stack Tecnológico Recomendado

* **Lenguaje:** Python 3.9+
* **Librerías Clave:** `lightgbm`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl` (para ingesta de reportes hospitalarios/climáticos), `shap`.

