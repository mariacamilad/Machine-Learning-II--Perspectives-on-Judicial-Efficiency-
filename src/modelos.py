"""
Modelos de supervivencia del artículo, implementados con lifelines.

Equivalencias con el código R de los autores (Kandie Final Results.Rmd):

    survfit(Surv(t, e) ~ g)                -> KaplanMeierFitter por grupo
    ggsurvplot(..., pval = TRUE)           -> multivariate_logrank_test
    coxph(Surv(t, e) ~ case_type + Appeals)-> CoxPHFitter (empates de Efron, igual que R)
    survreg(..., dist = "weibull")         -> WeibullAFTFitter
    survreg(..., dist = "exponential")     -> ExponentialAFTFitter (definido aquí)
    survreg(..., dist = "lognormal")       -> LogNormalAFTFitter
    survreg(..., dist = "loglogistic")     -> LogLogisticAFTFitter
    AIC(...)                               -> AIC_ (paramétricos) / AIC_partial_ (Cox)

Igual que en el Rmd, Cox se ajusta con todos los casos y los modelos paramétricos
solo con duraciones > 0 (clean_data), porque log(0) no está definido.
"""
import numpy as np
import pandas as pd
from autograd import numpy as anp
from scipy.stats import norm
from lifelines import (CoxPHFitter, KaplanMeierFitter, LogLogisticAFTFitter,
                       LogNormalAFTFitter, NelsonAalenFitter, WeibullAFTFitter)
from lifelines.fitters import ParametricRegressionFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index

from . import config as cfg

DUR, EVT = "time_in_months", "Event"
DUMMIES = [f"ct_{t}" for t in cfg.TIPOS if t != cfg.REFERENCIA]
COVARIABLES = ["Appeals"] + DUMMIES
ORDEN = ["Intercept", "Appeals", "CIVIL", "CONSTITUTIONAL", "CRIMINAL", "FAMILY"]


# ---------------------------------------------------------------------------
# Preparación
# ---------------------------------------------------------------------------
def matriz_diseno(df):
    """Dummies de tipo de caso con ANTI-CORRUPTION como referencia (igual que R)."""
    tipo = pd.Categorical(df["case_type"], categories=cfg.TIPOS)
    dummies = pd.get_dummies(tipo, prefix="ct", drop_first=True).astype(float)
    dummies.index = df.index
    return pd.concat([df[[DUR, EVT, "Appeals"]].astype(float), dummies], axis=1)


def sin_ceros(df):
    """clean_data del Rmd: elimina duraciones de 0 meses."""
    return df[df[DUR] > 0].copy()


def _renombrar(indice):
    return [str(i).replace("ct_", "") for i in indice]


# ---------------------------------------------------------------------------
# Kaplan-Meier y log-rank
# ---------------------------------------------------------------------------
def km_por_grupo(df, grupo=None):
    """Devuelve {nivel: KaplanMeierFitter}. grupo=None ajusta la curva global."""
    if grupo is None:
        return {"Global": KaplanMeierFitter(label="Global").fit(df[DUR], df[EVT])}
    salida = {}
    for nivel, sub in df.groupby(grupo, sort=True):
        salida[nivel] = KaplanMeierFitter(label=str(nivel)).fit(sub[DUR], sub[EVT])
    return salida


def tabla_km(kmf, tiempos=(12, 24, 36, 48, 60, 72, 84, 96, 108, 120)):
    """Probabilidad de seguir abierto en meses fijos (como summary(km, times=...))."""
    s = kmf.survival_function_at_times(list(tiempos))
    ic = kmf.confidence_interval_survival_function_
    inf = [ic.iloc[ic.index.searchsorted(t, side="right") - 1, 0] for t in tiempos]
    sup = [ic.iloc[ic.index.searchsorted(t, side="right") - 1, 1] for t in tiempos]
    return pd.DataFrame({"mes": tiempos, "S(t)": s.values, "IC_inf": inf, "IC_sup": sup})


def curvas_km_largas(df, grupo):
    """Curvas KM en formato largo (para guardar en results/ y comparar)."""
    filas = []
    for nivel, kmf in km_por_grupo(df, grupo).items():
        sf = kmf.survival_function_
        filas.append(pd.DataFrame({grupo: nivel, "mes": sf.index, "S": sf.iloc[:, 0].values}))
    return pd.concat(filas, ignore_index=True)


def logrank(df, grupo):
    return multivariate_logrank_test(df[DUR], df[grupo], df[EVT])


def medianas_km(df, grupo="case_type"):
    return pd.Series({k: v.median_survival_time_ for k, v in km_por_grupo(df, grupo).items()},
                     name="mediana_KM")


# ---------------------------------------------------------------------------
# Cox de riesgos proporcionales
# ---------------------------------------------------------------------------
def ajustar_cox(df):
    return CoxPHFitter().fit(matriz_diseno(df), duration_col=DUR, event_col=EVT)


def tabla_cox(cph):
    s = cph.summary
    t = pd.DataFrame({"coef": s["coef"], "HR": s["exp(coef)"],
                      "IC95_inf": s["exp(coef) lower 95%"],
                      "IC95_sup": s["exp(coef) upper 95%"], "p": s["p"]})
    t.index = _renombrar(t.index)
    return t.reindex([c for c in ORDEN if c in t.index])


def prueba_ph(cph, df):
    """Prueba de residuos de Schoenfeld (el artículo no la reporta)."""
    r = proportional_hazard_test(cph, matriz_diseno(df), time_transform="rank")
    t = r.summary[["test_statistic", "p"]].copy()
    t.index = _renombrar(t.index)
    return t


# ---------------------------------------------------------------------------
# Modelos paramétricos (forma AFT, igual que survreg)
# ---------------------------------------------------------------------------
class ExponentialAFTFitter(ParametricRegressionFitter):
    """
    Exponencial con la parametrización de survreg: log(T) = X*beta + W,
    con W ~ Gumbel de escala fija 1. Es la Weibull con forma rho = 1.
    """
    _fitted_parameter_names = ["lambda_"]

    def _cumulative_hazard(self, params, T, Xs):
        lambda_ = anp.exp(anp.dot(Xs["lambda_"], params["lambda_"]))
        return T / lambda_


def ajustar_parametricos(df):
    """Ajusta los cuatro modelos de las Tablas 4-7 sobre duraciones > 0."""
    d = matriz_diseno(sin_ceros(df))
    formula = "1 + " + " + ".join(COVARIABLES)
    return {
        "Weibull": WeibullAFTFitter().fit(d, DUR, EVT),
        "Exponential": ExponentialAFTFitter().fit(d, DUR, EVT,
                                                   regressors={"lambda_": formula}),
        "Log-normal": LogNormalAFTFitter().fit(d, DUR, EVT),
        "Log-logistic": LogLogisticAFTFitter().fit(d, DUR, EVT),
    }


_PARAM_UBICACION = {"Weibull": "lambda_", "Exponential": "lambda_",
                    "Log-normal": "mu_", "Log-logistic": "alpha_"}
_PARAM_ESCALA = {"Weibull": "rho_", "Log-normal": "sigma_", "Log-logistic": "beta_"}


def tabla_aft(fitter, nombre):
    """Coeficientes en la escala de survreg (log del tiempo), como Tablas 4-7."""
    s = fitter.summary.loc[_PARAM_UBICACION[nombre]]
    t = pd.DataFrame({"coef": s["coef"], "exp(coef)": s["exp(coef)"],
                      "se": s["se(coef)"], "z": s["z"], "p": s["p"]})
    t.index = _renombrar(t.index)
    return t.reindex(ORDEN)


def escala_survreg(fitter, nombre):
    """
    Parámetro 'Scale' que imprime survreg. Weibull: sigma = 1/rho;
    log-normal: sigma; log-logística: 1/beta; exponencial: 1 (fijo).
    """
    if nombre == "Exponential":
        return 1.0
    v = float(np.exp(fitter.params_.loc[(_PARAM_ESCALA[nombre], "Intercept")]))
    return v if nombre == "Log-normal" else 1.0 / v


def weibull_aft_a_hr(fitter):
    """
    Conversión clave para interpretar la Tabla 4: en una Weibull, el modelo AFT y
    el de riesgos proporcionales son el mismo modelo. Si log(T) = X*beta + sigma*W,
    entonces HR = exp(-beta / sigma) y la razón de tiempos (TR) = exp(beta).
    """
    t = tabla_aft(fitter, "Weibull").drop(index="Intercept")
    sigma = escala_survreg(fitter, "Weibull")
    return pd.DataFrame({"beta_AFT": t["coef"], "TR=exp(beta)": np.exp(t["coef"]),
                         "HR=exp(-beta/sigma)": np.exp(-t["coef"] / sigma)})


# ---------------------------------------------------------------------------
# Comparación de modelos
# ---------------------------------------------------------------------------
def tabla_aic(cph, parametricos, n_cox, n_param):
    filas = [{"modelo": "Cox PH", "gl": len(cph.params_), "logLik": cph.log_likelihood_,
              "AIC": cph.AIC_partial_, "n": n_cox, "verosimilitud": "parcial"}]
    for nombre, f in parametricos.items():
        filas.append({"modelo": nombre, "gl": len(f.params_), "logLik": f.log_likelihood_,
                      "AIC": f.AIC_, "n": n_param, "verosimilitud": "completa"})
    return pd.DataFrame(filas).set_index("modelo")


def _lineal(fitter, parametro, X):
    coef = fitter.params_.loc[parametro]
    Xc = X.copy()
    Xc["Intercept"] = 1.0
    return Xc[coef.index].values @ coef.values


def hazard_acumulado(fitter, nombre, df):
    """H(t_i | x_i) de cada caso: base de los residuos de Cox-Snell."""
    d = matriz_diseno(df)
    t = d[DUR].values
    if nombre == "Cox PH":
        h0 = fitter.baseline_cumulative_hazard_.iloc[:, 0]
        pos = np.searchsorted(h0.index.values, t, side="right") - 1
        base = np.where(pos >= 0, h0.values[np.clip(pos, 0, None)], 0.0)
        return base * fitter.predict_partial_hazard(d[COVARIABLES]).values
    if nombre in ("Weibull", "Exponential"):
        lam = np.exp(_lineal(fitter, "lambda_", d))
        rho = 1.0 if nombre == "Exponential" else np.exp(fitter.params_.loc[("rho_", "Intercept")])
        return (t / lam) ** rho
    if nombre == "Log-normal":
        mu = _lineal(fitter, "mu_", d)
        sigma = np.exp(fitter.params_.loc[("sigma_", "Intercept")])
        return -norm.logsf((np.log(t) - mu) / sigma)
    if nombre == "Log-logistic":
        alpha = np.exp(_lineal(fitter, "alpha_", d))
        beta = np.exp(fitter.params_.loc[("beta_", "Intercept")])
        return np.log1p((t / alpha) ** beta)
    raise ValueError(nombre)


def residuos_cox_snell(modelos, df):
    """
    Si el modelo es correcto, los residuos r_i = H(t_i|x_i) se comportan como una
    muestra censurada de una Exp(1): su hazard acumulado de Nelson-Aalen debe caer
    sobre la recta de 45 grados. Se evalúa sobre duraciones > 0 para todos.
    """
    d = sin_ceros(df)
    salida = {}
    for nombre, f in modelos.items():
        r = hazard_acumulado(f, nombre, d)
        na = NelsonAalenFitter().fit(r, d[EVT].values)
        salida[nombre] = na.cumulative_hazard_.iloc[:, 0]
    return salida


def distancia_cox_snell(curvas, cuantil=0.99):
    """Resumen numérico: distancia media absoluta a la recta de 45 grados."""
    filas = {}
    for nombre, h in curvas.items():
        x = h.index.values
        lim = np.quantile(x, cuantil)
        m = (x > 0) & (x <= lim)
        filas[nombre] = np.mean(np.abs(h.values[m] - x[m]))
    return pd.Series(filas, name="desvio_medio_45")


def indices_concordancia(modelos, df):
    """C de Harrell de cada modelo, todos evaluados sobre los mismos casos (> 0 meses)."""
    d = matriz_diseno(sin_ceros(df))
    filas = {}
    for nombre, f in modelos.items():
        if nombre == "Cox PH":
            score = -f.predict_partial_hazard(d[COVARIABLES]).values
        else:
            # Con escala común, la mediana predicha es monótona en el predictor
            # lineal de ubicación: se usa directamente (más alto = más largo).
            score = _lineal(f, _PARAM_UBICACION[nombre], d)
        filas[nombre] = concordance_index(d[DUR], score, d[EVT])
    return pd.Series(filas, name="C_index")
