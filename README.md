# Recreación de Kandie et al. (2025) con datos simulados

Actividad final de Machine Learning II. Se recrea, con datos simulados, el análisis de supervivencia de:

> Kandie, A. K., Atitwa, E. B. y Nyaluke, W. (2025). *Perspectives on Judicial Efficiency: Case Study of Predictive Modeling in Kenyan High Courts*. International Journal for Court Administration, 16(2), 5. https://doi.org/10.36745/ijca.627

El artículo modela el tiempo que tarda un caso en resolverse en los tribunales superiores de Kenia con Kaplan-Meier, Cox y modelos paramétricos (Weibull, exponencial, log-normal y log-logístico). El trabajo de grado aplica la misma lógica a los tiempos de ciclo de los casos de compras para estimar el personal que requiere cada equipo.

## Qué hace el proyecto

1. Construye una base simulada de 91.405 casos calibrada contra los estadísticos que publica el artículo (Tabla 1). Lo que el artículo no reporta se declara como supuesto (S1 a S7, en `src/config.py` y en `results/supuestos.csv`).
2. Aplica a esa base el mismo procedimiento del código R de los autores: descriptivos, prueba t, Kaplan-Meier con log-rank, Cox, los cuatro modelos paramétricos y la comparación por AIC.
3. Compara los resultados con las Tablas 1 a 8 del artículo y explica las diferencias. Como referencia, repite el análisis con la base original compartida por los autores (si está disponible).
4. Extensión: cambia una sola condición, el mecanismo de observación (incluir los casos abiertos como censurados en lugar de descartarlos), y mide el efecto contra la duración verdadera que conoce el simulador.

## Estructura

```
proyecto/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── recreacion_kandie_2025.ipynb   Notebook principal (análisis, resultados e interpretación)
├── src/
│   ├── config.py        Tablas del artículo, supuestos S1-S7, rutas y colores
│   ├── simulacion.py    Proceso generador, calibración y construcción de las bases
│   ├── modelos.py       Kaplan-Meier, log-rank, Cox, modelos AFT, AIC, Cox-Snell, C-index
│   ├── eda.py           Descriptivos, prueba t y tablas artículo vs. simulado vs. real
│   ├── pipeline.py      Corre el procedimiento completo y guarda/carga tablas
│   ├── extension.py     Experimento de la extensión (censura)
│   └── graficos.py      Todas las figuras
├── data/
│   ├── casos_simulados.csv        Base simulada que se analiza (equivalente a la del artículo)
│   ├── poblacion_latente.csv.gz   Todos los casos latentes, con su duración verdadera
│   ├── parametros_calibrados.csv  Forma, escala y efecto de apelación calibrados
│   └── raw/                       Base original de los autores (NO se versiona)
├── results/      Tablas en CSV (prefijo sim_ = simulado, real_ = base original, comparacion_, extension_)
└── figures/      Figuras en PNG, numeradas en el orden del notebook
```

## Cómo ejecutar

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/recreacion_kandie_2025.ipynb
```

Ejecutar todas las celdas en orden. Tiempo aproximado: 2 a 3 minutos.

- Si `data/parametros_calibrados.csv` existe, el notebook carga los datos simulados ya generados. Para regenerarlos desde cero, borrar los tres archivos de `data/` (excepto `raw/`); la calibración tarda alrededor de un minuto y, con la semilla fija (`SEMILLA = 2025` en `src/config.py`), produce los mismos datos.
- Si `data/raw/high_court_cases.csv` existe, el notebook rehace la réplica con la base original. Si no existe, usa las tablas ya guardadas en `results/real_*.csv`.

## Sobre la base original

Los autores indican en el artículo (Ethics and Consent) que la base no puede publicarse completa. Por eso `data/raw/` está en `.gitignore` y el repositorio solo incluye resultados agregados (coeficientes y tablas) obtenidos con ella. Para correr la réplica con datos reales, copiar el archivo como `data/raw/high_court_cases.csv`.

## Supuestos de la simulación

| Id | Supuesto | Justificación |
|---|---|---|
| S1 | Proporción de casos por tipo | No reportada; tomada de la base compartida |
| S2 | Proporción de apelaciones por tipo | No reportada; tomada de la base compartida |
| S3 | Efecto de la apelación común a todos los tipos | Calibrado contra la diferencia de medias de la Tabla 1 |
| S4 | Radicación uniforme entre 2012 y 2022 | El artículo solo indica el periodo |
| S5 | Solo entran casos dispuestos entre 2018 y 2022 | Inferido de `disposal_year` en la base compartida |
| S6 | Tiempo latente Weibull por tipo de caso | Calibrado contra media, mediana y DE de la Tabla 1 |
| S7 | Tiempo en meses completos | Reproduce las duraciones de 0 meses del original |

## Equivalencias R → Python

| R (código de los autores) | Python (`lifelines`) |
|---|---|
| `survfit` | `KaplanMeierFitter` |
| `coxph` | `CoxPHFitter` |
| `survreg(dist = "weibull" / "lognormal" / "loglogistic")` | `WeibullAFTFitter` / `LogNormalAFTFitter` / `LogLogisticAFTFitter` |
| `survreg(dist = "exponential")` | `ExponentialAFTFitter` (en `src/modelos.py`) |

Con la base original, este código reproduce las Tablas 3 a 8 del artículo hasta el tercer decimal.

## Paleta de las figuras

Las cinco categorías usan una paleta categórica validada para distinción con daltonismo; además, toda figura con varias series tiene leyenda o etiquetas, así que el color nunca es la única forma de identificar una serie.
