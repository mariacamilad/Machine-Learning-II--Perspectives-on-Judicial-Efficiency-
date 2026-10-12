"""
Parámetros del estudio original, supuestos de simulación y rutas del proyecto.

Toda cifra que viene del artículo se marca con la tabla o sección de origen:
Kandie, Atitwa y Nyaluke (2025), "Perspectives on Judicial Efficiency: Case Study
of Predictive Modeling in Kenyan High Courts", IJCA 16(2), doi:10.36745/ijca.627.

Toda cifra que NO aparece en el artículo se declara como supuesto (S1, S2, ...)
en el diccionario SUPUESTOS, con su justificación.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
DATA_RAW = DATA / "raw"
RESULTS = RAIZ / "results"
FIGURES = RAIZ / "figures"

ARCHIVO_SIMULADO = DATA / "casos_simulados.csv"
ARCHIVO_LATENTE = DATA / "poblacion_latente.csv"
ARCHIVO_PARAMETROS = DATA / "parametros_calibrados.csv"
ARCHIVO_REAL = DATA_RAW / "high_court_cases.csv"   # no se versiona (ver README)

SEMILLA = 2025

# ---------------------------------------------------------------------------
# Lo que reporta el artículo
# ---------------------------------------------------------------------------
TIPOS = ["ANTI-CORRUPTION", "CIVIL", "CONSTITUTIONAL", "CRIMINAL", "FAMILY"]
REFERENCIA = "ANTI-CORRUPTION"          # categoría base en Cox y AFT (Tablas 3-7)

N_ARTICULO = 91_405                      # Sección 3.1.1

# Tabla 1: estadísticos del tiempo en meses por tipo de caso y por apelación
TABLA1 = {
    #                  media  mediana min  max   sd
    "ANTI-CORRUPTION": (14.22,  8, 0,  72, 16.01),
    "CIVIL":           (30.89, 23, 0, 130, 27.11),
    "CONSTITUTIONAL":  (21.54, 14, 0, 129, 22.24),
    "CRIMINAL":        (15.91,  8, 0, 126, 20.81),
    "FAMILY":          (40.19, 35, 0, 129, 30.14),
    "Non-Appeals(0)":  (26.22, 15, 0, 130, 28.47),
    "Appeals(1)":      (30.08, 24, 0, 129, 23.09),
}

# Tabla 2: prueba t (Welch) de tiempo según apelación
TABLA2 = {"t": -20.67, "gl": 48116, "ic_inf": -4.22, "ic_sup": -3.49,
          "media_0": 26.22, "media_1": 30.08}

# Tabla 3: Cox PH (coef, HR, IC inf, IC sup, p)
TABLA3 = {
    "CIVIL":          (-0.690, 0.502, 0.456, 0.553),
    "CONSTITUTIONAL": (-0.470, 0.625, 0.566, 0.690),
    "CRIMINAL":       (-0.073, 0.930, 0.844, 1.024),
    "FAMILY":         (-1.105, 0.331, 0.301, 0.365),
    "Appeals":        (-0.312, 0.732, 0.720, 0.743),
}

# Tablas 4-7: modelos paramétricos de survreg (coeficiente, error estándar)
TABLAS_AFT = {
    "Weibull": {           # Tabla 4
        "Intercept": (2.767, 0.047), "Appeals": (0.130, 0.008),
        "CIVIL": (0.674, 0.047), "CONSTITUTIONAL": (0.432, 0.048),
        "CRIMINAL": (0.310, 0.047), "FAMILY": (0.992, 0.047)},
    "Exponential": {       # Tabla 5
        "Intercept": (2.714, 0.052), "Appeals": (0.153, 0.009),
        "CIVIL": (0.679, 0.052), "CONSTITUTIONAL": (0.443, 0.053),
        "CRIMINAL": (0.307, 0.052), "FAMILY": (1.017, 0.052)},
    "Log-normal": {        # Tabla 6
        "Intercept": (2.110, 0.058), "Appeals": (0.506, 0.010),
        "CIVIL": (0.627, 0.058), "CONSTITUTIONAL": (0.537, 0.060),
        "CRIMINAL": (0.181, 0.058), "FAMILY": (1.259, 0.058)},
    "Log-logistic": {      # Tabla 7
        "Intercept": (2.125, 0.058), "Appeals": (0.484, 0.010),
        "CIVIL": (0.704, 0.059), "CONSTITUTIONAL": (0.588, 0.060),
        "CRIMINAL": (0.237, 0.059), "FAMILY": (1.330, 0.059)},
}

# Tabla 8: AIC
TABLA8 = {"Cox PH": (5, 1915086.7), "Weibull": (7, 713383.0),
          "Exponential": (6, 714643.8), "Log-normal": (7, 721860.8),
          "Log-logistic": (7, 723709.4)}

# ---------------------------------------------------------------------------
# Supuestos de simulación (no reportados en el artículo)
# ---------------------------------------------------------------------------
# S1. Proporción de casos por tipo. El artículo no la reporta; el Rmd de los
#     autores la calcula (sección "Introduction for 4.1") pero no se publicó.
#     Se toma del conjunto de datos compartido por los autores.
PROP_TIPO = {"ANTI-CORRUPTION": 0.0045, "CIVIL": 0.3371, "CONSTITUTIONAL": 0.0761,
             "CRIMINAL": 0.3431, "FAMILY": 0.2392}

# S2. Proporción de casos apelados dentro de cada tipo. No reportada.
#     Fuente: conjunto de datos compartido por los autores.
PROP_APELACION = {"ANTI-CORRUPTION": 0.215, "CIVIL": 0.412, "CONSTITUTIONAL": 0.004,
                  "CRIMINAL": 0.299, "FAMILY": 0.027}

# S3. Efecto de la apelación: multiplicador del tiempo (modelo AFT) igual al
#     coeficiente de la Tabla 4: exp(0.130) = 1.139, común a todos los tipos.
GAMMA_APELACION = 0.130

# S4. Llegada de casos: fechas de radicación uniformes entre ene-2012 y dic-2022
#     (132 meses). El artículo solo dice "datos de 2012 a 2022".
MESES_VENTANA = 132

# S5. Ventana de observación: el conjunto incluye solo casos DISPUESTOS entre
#     ene-2018 y dic-2022 (mes 72 a 132). No se menciona en el artículo; se
#     infiere de la columna disposal_year del conjunto compartido (99.99 % de
#     los casos tienen Event = 1 y disposición 2018-2022).
INICIO_DISPOSICION = 72

# S6. Forma del tiempo latente: Weibull con forma y escala propias de cada tipo
#     de caso, calibradas para reproducir media, mediana y desviación de la
#     Tabla 1 después de aplicar S4, S5 y el redondeo a meses completos.
# S7. El tiempo reportado es el número de meses completos (piso), lo que genera
#     duraciones de 0 meses como en el original (min = 0 en la Tabla 1).

SUPUESTOS = {
    "S1": ("Proporción de casos por tipo",
           "No reportada en el artículo; tomada del conjunto compartido por los autores."),
    "S2": ("Proporción de apelaciones por tipo",
           "No reportada; tomada del conjunto compartido por los autores."),
    "S3": ("Efecto de la apelación sobre el tiempo",
           "Razón de tiempos exp(0.130)=1.139 tomada de la Tabla 4 (Weibull), común a todos los tipos."),
    "S4": ("Llegada de casos",
           "Radicación uniforme entre ene-2012 y dic-2022; el artículo solo indica el periodo."),
    "S5": ("Mecanismo de observación",
           "Solo entran casos dispuestos entre ene-2018 y dic-2022 (inferido de disposal_year)."),
    "S6": ("Distribución del tiempo latente",
           "Weibull por tipo de caso, calibrada contra media, mediana y DE de la Tabla 1."),
    "S7": ("Medición del tiempo",
           "Meses completos (piso), lo que produce duraciones de 0 meses."),
}

# Paleta (validada para distinción por daltonismo, ver README)
COLORES_TIPO = {"ANTI-CORRUPTION": "#2a78d6", "CIVIL": "#eb6834",
                "CONSTITUTIONAL": "#1baf7a", "CRIMINAL": "#eda100",
                "FAMILY": "#e87ba4"}
COLORES_APELACION = {0: "#2a78d6", 1: "#eb6834"}
