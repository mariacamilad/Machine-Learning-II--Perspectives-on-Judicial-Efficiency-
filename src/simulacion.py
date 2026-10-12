"""
Generación de datos simulados con las características reportadas por el artículo.

Lógica del proceso generador (ver supuestos S1-S7 en config.py):

1. Cada caso latente recibe un tipo (S1), una marca de apelación (S2) y una
   fecha de radicación uniforme en la ventana de 132 meses (S4).
2. Su duración latente sigue una Weibull AFT propia de su tipo (S6):
       T = lambda_tipo * exp(gamma * Apelacion) * E^(1/k_tipo),  E ~ Exp(1)
   donde gamma es el efecto de apelación (S3).
3. El caso aparece en la base solo si se dispuso entre ene-2018 y dic-2022 (S5).
4. Se registra el número de meses completos (S7).

Los parámetros (k_tipo, lambda_tipo) se calibran por simulación para que los
datos observados reproduzcan la media, mediana y desviación estándar de la Tabla 1.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config as cfg

DIAS_POR_MES = 30.4375
FECHA_INICIO = pd.Timestamp("2012-01-01")


# ---------------------------------------------------------------------------
# Piezas del proceso generador
# ---------------------------------------------------------------------------
def tiempo_latente(k, lam, apelacion, e, gamma):
    """Duración latente en meses (Weibull AFT) a partir de choques E ~ Exp(1)."""
    return lam * np.exp(gamma * apelacion) * e ** (1.0 / k)


def observado_en_ventana(radicacion, duracion):
    """S5: True si la fecha de disposición cae entre el mes 72 y el mes 132."""
    disposicion = radicacion + duracion
    return (disposicion >= cfg.INICIO_DISPOSICION) & (disposicion < cfg.MESES_VENTANA)


# ---------------------------------------------------------------------------
# Calibración contra la Tabla 1
# ---------------------------------------------------------------------------
def _momentos_observados(k, lam, apel, e, u, gamma):
    t = tiempo_latente(k, lam, apel, e, gamma)
    rad = u * cfg.MESES_VENTANA
    obs = t[observado_en_ventana(rad, t)]
    meses = np.floor(obs)
    # La mediana se compara en escala continua (mediana de meses completos = m
    # implica mediana continua en [m, m+1)); así la función objetivo es suave.
    return meses.mean(), np.median(obs), meses.std(ddof=1), (meses == 0).mean()


def calibrar_tipo(tipo, gamma, semilla, n_latente=200_000):
    """Busca (k, lambda) que reproducen media, mediana y DE observadas del tipo."""
    rng = np.random.default_rng(semilla)
    media_obj, mediana_obj, _, _, sd_obj = cfg.TABLA1[tipo]
    apel = (rng.random(n_latente) < cfg.PROP_APELACION[tipo]).astype(float)
    e = rng.exponential(size=n_latente)          # números aleatorios comunes
    u = rng.random(n_latente)

    def perdida(theta):
        k, lam = np.exp(theta)
        m, med, sd, _ = _momentos_observados(k, lam, apel, e, u, gamma)
        return (((m - media_obj) / media_obj) ** 2
                + ((med - (mediana_obj + 0.5)) / (mediana_obj + 0.5)) ** 2
                + ((sd - sd_obj) / sd_obj) ** 2)

    inicio = np.log([1.0, media_obj])
    res = minimize(perdida, inicio, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000})
    k, lam = np.exp(res.x)
    m, med, sd, ceros = _momentos_observados(k, lam, apel, e, u, gamma)
    return {"case_type": tipo, "forma_k": k, "escala_lambda": lam, "gamma": gamma,
            "media_obj": media_obj, "media_sim": m,
            "mediana_obj": mediana_obj, "mediana_sim_continua": med,
            "sd_obj": sd_obj, "sd_sim": sd, "prop_ceros_sim": ceros,
            "perdida": res.fun}


def calibrar_dado_gamma(gamma, semilla=cfg.SEMILLA):
    return pd.DataFrame([calibrar_tipo(t, gamma, semilla + i)
                         for i, t in enumerate(cfg.TIPOS)])


def diferencia_apelacion(parametros, semilla=cfg.SEMILLA):
    """Media(apelados) - media(no apelados) en la base observada que genera `parametros`."""
    base = base_tipo_articulo(generar_poblacion_latente(parametros, semilla=semilla))
    medias = base.groupby("Appeals")["time_in_months"].mean()
    return medias[1] - medias[0]


def calibrar(semilla=cfg.SEMILLA, gamma_min=0.0, gamma_max=1.5, tol=0.01):
    """
    Calibración en dos niveles:
      - nivel interno: (k, lambda) de cada tipo contra media, mediana y DE (Tabla 1);
      - nivel externo: gamma común contra la diferencia de medias por apelación
        de la Tabla 1 (30.08 - 26.22 = 3.86 meses), por bisección.
    """
    objetivo = cfg.TABLA1["Appeals(1)"][0] - cfg.TABLA1["Non-Appeals(0)"][0]
    historial = []

    def brecha(gamma):
        params = calibrar_dado_gamma(gamma, semilla)
        d = diferencia_apelacion(params, semilla)
        historial.append((gamma, d))
        return d - objetivo, params

    lo, hi = gamma_min, gamma_max
    f_lo, _ = brecha(lo)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        f_mid, _ = brecha(mid)
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    gamma = (lo + hi) / 2
    _, params = brecha(gamma)
    params.attrs["historial_gamma"] = historial
    return params


# ---------------------------------------------------------------------------
# Generación de la población latente y de las dos vistas de los datos
# ---------------------------------------------------------------------------
def generar_poblacion_latente(parametros, n_observados=cfg.N_ARTICULO,
                              semilla=cfg.SEMILLA, lote=50_000):
    """
    Genera casos latentes por tipo hasta acumular el número de casos observados
    que exige S1 en la ventana de disposición. Devuelve todos los casos latentes
    (observados y no observados), lo que permite construir tanto la base "tipo
    artículo" como la base de cohorte usada en la extensión.
    """
    rng = np.random.default_rng(semilla + 1)
    bloques = []
    for _, fila in parametros.iterrows():
        tipo = fila["case_type"]
        meta = int(round(n_observados * cfg.PROP_TIPO[tipo]))
        partes, acumulado = [], 0
        while acumulado < meta:
            apel = (rng.random(lote) < cfg.PROP_APELACION[tipo]).astype(int)
            t = tiempo_latente(fila["forma_k"], fila["escala_lambda"], apel,
                               rng.exponential(size=lote), fila["gamma"])
            rad = rng.random(lote) * cfg.MESES_VENTANA
            sel = observado_en_ventana(rad, t)
            parte = pd.DataFrame({"case_type": tipo, "Appeals": apel,
                                  "radicacion_mes": rad, "T_latente": t,
                                  "en_ventana": sel})
            faltan = meta - acumulado
            if sel.sum() >= faltan:          # recorta en el caso observado número `meta`
                corte = np.flatnonzero(sel)[faltan - 1] + 1
                parte = parte.iloc[:corte]
            partes.append(parte)
            acumulado += int(parte["en_ventana"].sum())
        bloques.append(pd.concat(partes, ignore_index=True))
    latente = pd.concat(bloques, ignore_index=True)
    latente.insert(0, "id", np.arange(1, len(latente) + 1))
    return latente


def _agregar_fechas(df, meses_desde_inicio, prefijo):
    fechas = FECHA_INICIO + pd.to_timedelta(meses_desde_inicio * DIAS_POR_MES, unit="D")
    df[f"{prefijo}_date"] = fechas.dt.date
    df[f"{prefijo}_year"] = fechas.dt.year
    return df


def base_tipo_articulo(latente):
    """Vista equivalente a la base del artículo: solo casos dispuestos en 2018-2022."""
    df = latente.loc[latente["en_ventana"], ["id", "case_type", "Appeals",
                                            "radicacion_mes", "T_latente"]].copy()
    df["time_in_months"] = np.floor(df["T_latente"]).astype(int)
    df["time_in_years"] = df["time_in_months"] // 12
    df["Event"] = 1
    df = _agregar_fechas(df, df["radicacion_mes"], "filed")
    df = _agregar_fechas(df, df["radicacion_mes"] + df["T_latente"], "disposal")
    columnas = ["id", "case_type", "Appeals", "filed_date", "filed_year",
                "disposal_date", "disposal_year", "time_in_years",
                "time_in_months", "Event"]
    return df[columnas].reset_index(drop=True)


def base_cohorte(latente):
    """
    Vista alternativa (extensión): todos los casos radicados en 2012-2022,
    seguidos hasta dic-2022; los que siguen abiertos se registran como censurados.
    """
    df = latente[["id", "case_type", "Appeals", "radicacion_mes", "T_latente"]].copy()
    seguimiento = cfg.MESES_VENTANA - df["radicacion_mes"]
    df["Event"] = (df["T_latente"] <= seguimiento).astype(int)
    df["time_in_months"] = np.floor(np.minimum(df["T_latente"], seguimiento)).astype(int)
    return df[["id", "case_type", "Appeals", "time_in_months", "Event"]].reset_index(drop=True)


def base_verdad(latente):
    """Duraciones latentes completas (sin censura ni selección): la verdad del simulador."""
    df = latente[["id", "case_type", "Appeals", "T_latente"]].copy()
    df["time_in_months"] = np.floor(df["T_latente"]).astype(int)
    df["Event"] = 1
    return df[["id", "case_type", "Appeals", "time_in_months", "Event"]]


# ---------------------------------------------------------------------------
# Datos reales (opcionales, solo como referencia)
# ---------------------------------------------------------------------------
NOMBRES_ORIGINALES = {"ANTI-CORRUPTION AND ECONOMIC CRIMES": "ANTI-CORRUPTION",
                      "CONSTITUTIONAL & HUMAN RIGHTS": "CONSTITUTIONAL"}


def cargar_real(ruta=cfg.ARCHIVO_REAL):
    """Carga la base compartida por los autores con las mismas etiquetas del Rmd."""
    df = pd.read_csv(ruta)
    df["case_type"] = df["case_type"].replace(NOMBRES_ORIGINALES)
    return df
