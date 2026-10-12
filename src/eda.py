"""Análisis exploratorio (Sección 3.1 del artículo) y tablas comparativas."""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg

DUR = "time_in_months"


def resumen(df, por):
    """Tabla 1: media, mediana, mínimo, máximo y DE del tiempo en meses."""
    t = df.groupby(por)[DUR].agg(media="mean", mediana="median", minimo="min",
                                 maximo="max", de="std", n="count")
    return t


def distribucion_tipos(df):
    """Conteo y porcentaje de casos por tipo (sección 'Introduction for 4.1' del Rmd)."""
    c = df["case_type"].value_counts().reindex(cfg.TIPOS)
    return pd.DataFrame({"n": c, "porcentaje": (100 * c / c.sum()).round(1)})


def prueba_t(df):
    """Tabla 2: prueba t de Welch, igual que t.test(time ~ Appeals) en R."""
    g0 = df.loc[df["Appeals"] == 0, DUR]
    g1 = df.loc[df["Appeals"] == 1, DUR]
    r = stats.ttest_ind(g0, g1, equal_var=False)
    ic = r.confidence_interval(0.95)
    return pd.Series({"t": r.statistic, "gl": r.df, "p": r.pvalue,
                      "ic_inf": ic.low, "ic_sup": ic.high,
                      "media_0": g0.mean(), "media_1": g1.mean(),
                      "de_0": g0.std(), "de_1": g1.std()})


# ---------------------------------------------------------------------------
# Tablas "artículo vs real vs simulado"
# ---------------------------------------------------------------------------
def comparar_tabla1(sim, real=None):
    filas = []
    for etiqueta, (media, mediana, mn, mx, de) in cfg.TABLA1.items():
        if etiqueta in cfg.TIPOS:
            sub_s = sim[sim["case_type"] == etiqueta]
            sub_r = None if real is None else real[real["case_type"] == etiqueta]
        else:
            a = 1 if "(1)" in etiqueta else 0
            sub_s = sim[sim["Appeals"] == a]
            sub_r = None if real is None else real[real["Appeals"] == a]
        fila = {"grupo": etiqueta,
                "media_art": media, "media_sim": sub_s[DUR].mean(),
                "mediana_art": mediana, "mediana_sim": sub_s[DUR].median(),
                "max_art": mx, "max_sim": sub_s[DUR].max(),
                "de_art": de, "de_sim": sub_s[DUR].std()}
        if sub_r is not None:
            fila.update({"media_real": sub_r[DUR].mean(), "de_real": sub_r[DUR].std()})
        filas.append(fila)
    return pd.DataFrame(filas).set_index("grupo")


def comparar_tabla2(t_sim, t_real=None):
    art = pd.Series({"t": cfg.TABLA2["t"], "gl": cfg.TABLA2["gl"],
                     "ic_inf": cfg.TABLA2["ic_inf"], "ic_sup": cfg.TABLA2["ic_sup"],
                     "media_0": cfg.TABLA2["media_0"], "media_1": cfg.TABLA2["media_1"],
                     "de_0": 4.22, "de_1": 3.49})   # DE tal como las escribe el texto 3.1.2
    t = pd.DataFrame({"articulo": art, "simulado": t_sim[art.index]})
    if t_real is not None:
        t["real"] = t_real[art.index]
    return t


def comparar_cox(tab_sim, tab_real=None):
    art = pd.DataFrame(cfg.TABLA3, index=["coef", "HR", "IC95_inf", "IC95_sup"]).T
    t = pd.DataFrame({"coef_art": art["coef"], "coef_sim": tab_sim["coef"],
                      "HR_art": art["HR"], "HR_sim": tab_sim["HR"],
                      "IC_sim": [f"[{a:.3f}, {b:.3f}]" for a, b in
                                 zip(tab_sim["IC95_inf"], tab_sim["IC95_sup"])],
                      "p_sim": tab_sim["p"]})
    if tab_real is not None:
        t.insert(1, "coef_real", tab_real["coef"])
        t.insert(4, "HR_real", tab_real["HR"])
    return t.reindex(tab_sim.index)


def comparar_aft(tablas_sim, tablas_real=None):
    filas = []
    for modelo, coefs in cfg.TABLAS_AFT.items():
        for cov, (b, se) in coefs.items():
            fila = {"modelo": modelo, "covariable": cov, "coef_art": b,
                    "coef_sim": tablas_sim[modelo].loc[cov, "coef"],
                    "se_sim": tablas_sim[modelo].loc[cov, "se"]}
            if tablas_real is not None:
                fila["coef_real"] = tablas_real[modelo].loc[cov, "coef"]
            filas.append(fila)
    return pd.DataFrame(filas)


def comparar_aic(aic_sim, aic_real=None):
    t = pd.DataFrame({"AIC_art": {k: v[1] for k, v in cfg.TABLA8.items()},
                      "AIC_sim": aic_sim["AIC"]})
    if aic_real is not None:
        t["AIC_real"] = aic_real["AIC"]
    t["rango_art"] = t["AIC_art"].rank().astype(int)
    t["rango_sim"] = t["AIC_sim"].rank().astype(int)
    return t
