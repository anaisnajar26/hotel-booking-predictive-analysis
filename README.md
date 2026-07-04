# 🏨 Análisis Predictivo de Reservas Hoteleras

Este proyecto aborda el flujo completo de ciencia de datos sobre un conjunto de datos del sector hotelero con **más de 36.000 registros**. El objetivo principal es identificar patrones de comportamiento de los clientes y desarrollar modelos predictivos avanzados capaces de anticipar la demanda y las cancelaciones de reservas para optimizar la toma de decisiones estratégicas y gestionar el overbooking de forma segura.

---

## 📂 Estructura del Repositorio

El proyecto está estructurado de forma cronológica siguiendo las fases reales de un desarrollo en Ciencia de Datos:

### 📁 01-data-extraction-sql/
* **Contenido:** Estructuración de la base de datos en SQL.
* **Hito:** Consultas métricas iniciales para segmentar el comportamiento de las cancelaciones según los canales de distribución y el perfil del cliente.

### 📁 02-eda-powerbi/
* **Contenido:** Presentación ejecutiva de resultados.
* **Hito:** Identificación visual de los KPI clave de ocupación, estacionalidad de tarifas en los meses de verano y el impacto crítico del canal *Online* (36.5% de tasa de cancelación).

### 📁 03-statistical-analysis-r/
* **Contenido:** Scripts de análisis estadístico avanzado en R.
* **Hito:** Modelado y validación de hipótesis matemáticas para confirmar la correlación entre el tiempo de anticipación de la reserva (*lead time*) y el riesgo real de cancelación.

### 📁 04-machine-learning-python/
* **Contenido:** Pipeline completo de Machine Learning en Python con preparación de datos limpia (prevención de *data leakage*), balanceo de clases y ordenación temporal estricta.
* **Modelos Evaluados:** CART (Árbol de Decisión), Random Forest, XGBoost, LightGBM y Redes Neuronales Profundas (MLP).
* **Analítica Avanzada:** 
  * Explicabilidad de modelos mediante **SHAP Values** (global, local y curvas de dependencia).
  * **Threshold Tuning:** Ajuste del umbral de decisión al 70% para maximizar la precisión de negocio.
  * **Simulador Financiero:** Optimización del beneficio económico del hotel basándose en penalizaciones por riesgo reputacional ante overbooking.

---

## 🛠️ Tecnologías Utilizadas
* **Extracción de Datos:** SQL 
* **Visualización de Negocio:** Microsoft Power BI & Microsoft Excel
* **Análisis Estadístico:** R / RStudio
* **Ciencia de Datos e Inteligencia Artificial:** Python (Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM)
* **Explicabilidad de Modelos:** SHAP (SHapley Additive exPlanations)
* **Documentación Técnica:** LaTeX

---

## 🏆 Conclusiones Principales del Proyecto
* **El ADN del Cancelador:** Las variables de mayor impacto son el *lead time* (antelación extrema de la reserva) y el número de peticiones especiales (a menor interacción, menor compromiso emocional y mayor riesgo de fuga).
* **Optimización Financiera:** El modelo ganador (**LightGBM**) combinado con un ajuste estratégico del umbral al 70% permite al hotel ejecutar políticas de overbooking seguras, minimizando las habitaciones vacías imprevistas sin poner en riesgo la reputación de la marca.
