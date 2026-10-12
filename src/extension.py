"""
Extensión: se modifica UNA condición del experimento, el mecanismo de observación.

Diseño A (artículo): solo entran los casos dispuestos entre 2018 y 2022.
Diseño B (cohorte):  entran todos los casos radicados entre 2012 y 2022; los que
                     siguen abiertos a dic-2022 quedan como censurados.

Ambos diseños salen de la MISMA población latente, así que la única diferencia es
cómo se observa. Como en la simulación conocemos la duración completa de cada caso,
podemos medir cuál de los dos diseños se acerca más a la verdad.
"""
import numpy as np
import pandas as pd
from lifelines.utils import restricted_mean_survival_time

from . import config as cfg
from . import modelos as m
from . import simulacion as sim

HORIZONTE_RMST = 60   # meses: horizonte de planeación para la duración media restringida


def _resumen_km(df, etiqueta):
    filas = []
    for tipo, kmf in m.km_por_grupo(m.sin_ceros(df), "case_type").items():
        filas.append({"diseno": etiqueta, "case_type": tipo,
                      "mediana": kmf.median_survival_time_,
                      "resuelto_12m": 1 - float(kmf.survival_function_at_times(12).iloc[0]),
                      f"RMST_{HORIZONTE_RMST}": restricted_mean_survival_time(kmf, t=HORIZONTE_RMST)})
    return pd.DataFrame(filas)


def ejecutar(latente):
    bases = {"Verdad (duración completa)": sim.base_verdad(latente),
             "A: artículo (solo dispuestos 2018-2022)": sim.base_tipo_articulo(latente),
             "B: cohorte con censura": sim.base_cohorte(latente)}

    tamanos = pd.DataFrame({k: {"n": len(v), "eventos": int(v["Event"].sum()),
                                "censura_%": 100 * (1 - v["Event"].mean())}
                            for k, v in bases.items()}).T

    resumen = pd.concat([_resumen_km(v, k) for k, v in bases.items()], ignore_index=True)

    coef = {}
    for k, v in bases.items():
        w = m.ajustar_parametricos(v)["Weibull"]
        coef[k] = m.weibull_aft_a_hr(w)["TR=exp(beta)"]
    razones_tiempo = pd.DataFrame(coef)

    curvas = {k: m.curvas_km_largas(m.sin_ceros(v), "case_type") for k, v in bases.items()}
    return {"tamanos": tamanos, "resumen": resumen,
            "razones_tiempo": razones_tiempo, "curvas": curvas}


def error_relativo(resumen, metrica):
    """Error relativo de cada diseño frente a la verdad, por tipo de caso."""
    p = resumen.pivot(index="case_type", columns="diseno", values=metrica)
    verdad = p.pop("Verdad (duración completa)")
    return (p.sub(verdad, axis=0).div(verdad, axis=0) * 100).round(1)
