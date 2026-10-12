"""
Corre el procedimiento completo del artículo sobre cualquier base con las columnas
case_type, Appeals, time_in_months y Event, y guarda/carga las tablas resultantes.
Se usa igual para la base simulada y para la base real de referencia.
"""
import warnings

import pandas as pd

from . import config as cfg
from . import eda
from . import modelos as m


def correr(df, con_diagnosticos=True):
    warnings.filterwarnings("ignore")
    r = {"n": len(df), "n_sin_ceros": int((df["time_in_months"] > 0).sum())}
    r["tabla1_tipo"] = eda.resumen(df, "case_type")
    r["tabla1_apelacion"] = eda.resumen(df, "Appeals")
    r["prueba_t"] = eda.prueba_t(df)

    limpio = m.sin_ceros(df)
    r["logrank_tipo"] = m.logrank(limpio, "case_type").p_value
    r["logrank_apelacion"] = m.logrank(limpio, "Appeals").p_value
    r["km_largo"] = m.curvas_km_largas(limpio, "case_type")

    r["cox"] = m.ajustar_cox(df)
    r["tabla_cox"] = m.tabla_cox(r["cox"])
    r["parametricos"] = m.ajustar_parametricos(df)
    r["tablas_aft"] = {k: m.tabla_aft(f, k) for k, f in r["parametricos"].items()}
    r["escalas"] = pd.Series({k: m.escala_survreg(f, k) for k, f in r["parametricos"].items()},
                             name="Scale (survreg)")
    r["aic"] = m.tabla_aic(r["cox"], r["parametricos"], r["n"], r["n_sin_ceros"])
    r["weibull_hr"] = m.weibull_aft_a_hr(r["parametricos"]["Weibull"])

    if con_diagnosticos:
        todos = {"Cox PH": r["cox"], **r["parametricos"]}
        r["prueba_ph"] = m.prueba_ph(r["cox"], df)
        r["cox_snell"] = m.residuos_cox_snell(todos, df)
        r["cox_snell_desvio"] = m.distancia_cox_snell(r["cox_snell"])
        r["c_index"] = m.indices_concordancia(todos, df)
    return r


# Tablas que se guardan en results/ (las de la base real permiten correr el
# notebook sin el archivo original, que no se versiona).
_TABLAS = ["tabla1_tipo", "tabla1_apelacion", "prueba_t", "km_largo", "tabla_cox",
           "escalas", "aic", "weibull_hr", "prueba_ph", "cox_snell_desvio", "c_index"]


def guardar(r, prefijo):
    cfg.RESULTS.mkdir(exist_ok=True, parents=True)
    for clave in _TABLAS:
        if clave in r:
            r[clave].to_csv(cfg.RESULTS / f"{prefijo}_{clave}.csv")
    for modelo, t in r["tablas_aft"].items():
        t.to_csv(cfg.RESULTS / f"{prefijo}_aft_{modelo.lower()}.csv")


def cargar(prefijo):
    """Reconstruye las tablas guardadas (sin los objetos ajustados)."""
    r = {}
    for clave in _TABLAS:
        ruta = cfg.RESULTS / f"{prefijo}_{clave}.csv"
        if ruta.exists():
            t = pd.read_csv(ruta, index_col=0)
            r[clave] = t.iloc[:, 0] if t.shape[1] == 1 else t
    if "km_largo" in r:
        r["km_largo"] = pd.read_csv(cfg.RESULTS / f"{prefijo}_km_largo.csv", index_col=0)
    r["tablas_aft"] = {mod: pd.read_csv(cfg.RESULTS / f"{prefijo}_aft_{mod.lower()}.csv",
                                        index_col=0) for mod in cfg.TABLAS_AFT}
    return r
