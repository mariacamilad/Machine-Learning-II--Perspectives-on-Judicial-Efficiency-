"""Figuras del proyecto. Todas se guardan en figures/ con el mismo estilo."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg

TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"
FONDO = "#fcfcfb"
GRIS = "#8a8984"
# Tres fuentes de resultados (artículo / real / simulado): primeras tres ranuras
COLOR_FUENTE = {"Artículo": "#2a78d6", "Datos reales": "#eb6834", "Simulado": "#1baf7a"}
MARCADOR_FUENTE = {"Artículo": "o", "Datos reales": "s", "Simulado": "D"}


def estilo():
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9.5,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 160,
        "lines.linewidth": 2,
    })


def guardar(fig, nombre):
    cfg.FIGURES.mkdir(exist_ok=True, parents=True)
    fig.savefig(cfg.FIGURES / f"{nombre}.png", bbox_inches="tight")
    return fig


def _paneles(n, ncol=3, ancho=4.0, alto=3.0, compartir=True):
    nfil = int(np.ceil(n / ncol))
    fig, ejes = plt.subplots(nfil, ncol, figsize=(ancho * ncol, alto * nfil),
                             sharex=compartir, sharey=compartir, squeeze=False)
    ejes = ejes.ravel()
    for ax in ejes[n:]:
        ax.set_visible(False)
    return fig, ejes[:n]


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------
def calibracion(parametros, comparacion_t1, nombre="01_calibracion_tabla1"):
    """Objetivo (Tabla 1) vs simulado para media, mediana y DE por tipo de caso."""
    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    pares = [("media", "Media (meses)"), ("mediana", "Mediana (meses)"), ("de", "Desv. estándar")]
    tipos = cfg.TIPOS[::-1]
    y = np.arange(len(tipos))
    for ax, (clave, titulo) in zip(ejes, pares):
        art = comparacion_t1.loc[tipos, f"{clave}_art"]
        sim = comparacion_t1.loc[tipos, f"{clave}_sim"]
        ax.hlines(y, art, sim, color=REJILLA, lw=3, zorder=1)
        ax.scatter(art, y, s=60, facecolor=FONDO, edgecolor=COLOR_FUENTE["Artículo"],
                   lw=2, label="Tabla 1 del artículo", zorder=3)
        ax.scatter(sim, y, s=36, color=COLOR_FUENTE["Simulado"], label="Simulado", zorder=4)
        ax.set_title(titulo, loc="left")
        ax.set_yticks(y, tipos)
    ejes[0].legend(loc="lower right")
    fig.suptitle("La simulación reproduce los estadísticos descriptivos de la Tabla 1",
                 x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    return guardar(fig, nombre)


def brecha_apelacion(historial, objetivo, gamma_elegido, nombre="02_calibracion_apelacion"):
    h = pd.DataFrame(historial, columns=["gamma", "brecha"]).sort_values("gamma")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.scatter(h["gamma"], h["brecha"], s=36, color=COLOR_FUENTE["Simulado"], zorder=3,
               label="Evaluaciones de la bisección")
    ax.axhline(objetivo, color=TINTA_2, lw=1, ls="--")
    ax.text(0.30, objetivo + 0.3, f"Objetivo Tabla 1: {objetivo:.2f} meses",
            color=TINTA_2, fontsize=8.5)
    ax.axvline(0.130, color=COLOR_FUENTE["Artículo"], lw=1, ls=":")
    ax.text(0.14, h["brecha"].min(), "γ de la Tabla 4\n(0.130)", color=TINTA_2, fontsize=8.5)
    ax.axvline(gamma_elegido, color=GRIS, lw=1)
    ax.text(gamma_elegido - 0.01, h["brecha"].min(), f"γ calibrado\n({gamma_elegido:.3f})",
            color=TINTA_2, fontsize=8.5, ha="right")
    ax.set_xlabel("γ (efecto de apelación en el tiempo latente, escala log)")
    ax.set_ylabel("Media apelados − no apelados")
    ax.set_title("Calibración del efecto de apelación", loc="left")
    fig.tight_layout()
    return guardar(fig, nombre)


# ---------------------------------------------------------------------------
# Exploratorio (Figuras 1 y 2 del artículo)
# ---------------------------------------------------------------------------
def histogramas(df, titulo, nombre):
    fig, ejes = _paneles(len(cfg.TIPOS), compartir=False, alto=2.8)
    for ax, tipo in zip(ejes, cfg.TIPOS):
        x = df.loc[df["case_type"] == tipo, "time_in_months"]
        ax.hist(x, bins=np.arange(0, 135, 5), color=cfg.COLORES_TIPO[tipo],
                edgecolor=FONDO, linewidth=1)
        ax.set_title(tipo, loc="left", fontsize=10)
        ax.set_xlim(0, 132)
    for ax in ejes[-3:]:
        ax.set_xlabel("Tiempo en meses")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    return guardar(fig, nombre)


def boxplot_tipos(df, titulo, nombre):
    fig, ax = plt.subplots(figsize=(8, 4))
    datos = [df.loc[df["case_type"] == t, "time_in_months"] for t in cfg.TIPOS]
    bp = ax.boxplot(datos, patch_artist=True, widths=0.55,
                    medianprops=dict(color=TINTA, lw=1.5),
                    flierprops=dict(marker=".", ms=2, alpha=0.3, markeredgecolor=GRIS),
                    whiskerprops=dict(color=TINTA_2), capprops=dict(color=TINTA_2))
    for caja, t in zip(bp["boxes"], cfg.TIPOS):
        caja.set(facecolor=cfg.COLORES_TIPO[t], edgecolor=TINTA_2, alpha=0.85)
    ax.set_xticks(range(1, 6), cfg.TIPOS, fontsize=8.5)
    ax.set_ylabel("Tiempo en meses")
    ax.set_title(titulo, loc="left")
    fig.tight_layout()
    return guardar(fig, nombre)


# ---------------------------------------------------------------------------
# Kaplan-Meier (Figura 3 del artículo)
# ---------------------------------------------------------------------------
def km(curvas, colores, titulo, nombre, p_valor=None, etiquetas=None):
    """curvas: {nivel: KaplanMeierFitter}."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for nivel, kmf in curvas.items():
        etiqueta = etiquetas.get(nivel, nivel) if etiquetas else nivel
        sf = kmf.survival_function_.iloc[:, 0]
        ax.step(sf.index, sf.values, where="post", color=colores[nivel], label=etiqueta)
        ic = kmf.confidence_interval_survival_function_
        ax.fill_between(ic.index, ic.iloc[:, 0], ic.iloc[:, 1], step="post",
                        color=colores[nivel], alpha=0.15, lw=0)
    if p_valor is not None:
        texto = "p < 0.0001" if p_valor < 1e-4 else f"p = {p_valor:.4f}"
        ax.text(5, 0.08, f"Log-rank {texto}", color=TINTA_2, fontsize=9)
    ax.set_xlim(0, 132)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(0, 140, 12))
    ax.set_xlabel("Tiempo en meses")
    ax.set_ylabel("Probabilidad de seguir abierto")
    ax.set_title(titulo, loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return guardar(fig, nombre)


def km_sim_vs_real(largo_sim, largo_real, nombre="07_km_simulado_vs_real"):
    fig, ejes = _paneles(len(cfg.TIPOS), alto=2.9)
    for ax, tipo in zip(ejes, cfg.TIPOS):
        for largo, estilo_linea, etiqueta in [(largo_real, "--", "Datos reales"),
                                              (largo_sim, "-", "Simulado")]:
            c = largo[largo["case_type"] == tipo]
            ax.step(c["mes"], c["S"], where="post", ls=estilo_linea,
                    color=cfg.COLORES_TIPO[tipo] if etiqueta == "Simulado" else TINTA_2,
                    label=etiqueta, lw=2 if etiqueta == "Simulado" else 1.5)
        ax.set_title(tipo, loc="left", fontsize=10)
        ax.set_xlim(0, 132)
        ax.set_xticks(range(0, 140, 24))
    ejes[0].legend(loc="upper right")
    for ax in ejes[-3:]:
        ax.set_xlabel("Tiempo en meses")
    fig.suptitle("Curvas de Kaplan-Meier: datos simulados frente a datos reales",
                 x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    return guardar(fig, nombre)


# ---------------------------------------------------------------------------
# Comparación de coeficientes e interpretación
# ---------------------------------------------------------------------------
def coeficientes(comp_aft, comp_cox, nombre="08_coeficientes_comparados"):
    """Coeficientes del artículo, de los datos reales y de la simulación, por modelo."""
    modelos = ["Cox PH"] + list(cfg.TABLAS_AFT)
    fig, ejes = _paneles(len(modelos), ncol=3, alto=3.0, compartir=False)
    covs = ["Appeals", "CIVIL", "CONSTITUTIONAL", "CRIMINAL", "FAMILY"]
    y = np.arange(len(covs))[::-1]
    for ax, modelo in zip(ejes, modelos):
        if modelo == "Cox PH":
            series = {"Artículo": [cfg.TABLA3[c][0] for c in covs],
                      "Simulado": comp_cox.loc[covs, "coef_sim"].values}
            if "coef_real" in comp_cox:
                series["Datos reales"] = comp_cox.loc[covs, "coef_real"].values
            titulo = "Cox PH (log HR)"
        else:
            sub = comp_aft[comp_aft["modelo"] == modelo].set_index("covariable").loc[covs]
            series = {"Artículo": sub["coef_art"].values, "Simulado": sub["coef_sim"].values}
            if "coef_real" in sub:
                series["Datos reales"] = sub["coef_real"].values
            titulo = f"{modelo} (log razón de tiempos)"
        desplaz = {"Artículo": 0.18, "Datos reales": 0.0, "Simulado": -0.18}
        for fuente, vals in series.items():
            ax.scatter(vals, y + desplaz[fuente], s=34, marker=MARCADOR_FUENTE[fuente],
                       color=COLOR_FUENTE[fuente], label=fuente, zorder=3)
        ax.axvline(0, color=TINTA_2, lw=1)
        ax.set_yticks(y, covs)
        ax.set_title(titulo, loc="left", fontsize=10)
    manejadores, etiquetas = ejes[0].get_legend_handles_labels()
    fig.suptitle("Coeficientes estimados: artículo, réplica con datos reales y simulación",
                 x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(manejadores, etiquetas, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.995))
    return guardar(fig, nombre)


def hr_cox_vs_weibull(tabla, nombre="09_interpretacion_weibull"):
    """
    tabla: índice = covariables; columnas 'HR Cox', 'HR Weibull (convertido)',
    'exp(β) leído como HR (artículo)'.
    """
    colores = ["#2a78d6", "#1baf7a", "#e34948"]
    marcas = ["o", "D", "X"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(tabla))[::-1]
    for i, col in enumerate(tabla.columns):
        ax.scatter(tabla[col], y + (0.2 - 0.2 * i), s=45, marker=marcas[i],
                   color=colores[i], label=col, zorder=3)
    ax.axvline(1, color=TINTA_2, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 4)
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(y, tabla.index)
    ax.set_xlabel("Razón de riesgos de resolución (escala log)\n← más lento que la referencia  |  más rápido →")
    ax.set_title("Bien convertido, el Weibull dice lo mismo que Cox", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return guardar(fig, nombre)


def cox_snell(curvas, titulo, nombre):
    modelos = list(curvas)
    fig, ejes = _paneles(len(modelos), alto=3.0, compartir=False)
    for ax, mod in zip(ejes, modelos):
        h = curvas[mod]
        lim = np.quantile(h.index.values, 0.995)
        h = h[h.index <= lim]
        ax.plot([0, lim], [0, lim], color=TINTA_2, lw=1, ls="--")
        ax.step(h.index, h.values, where="post", color=COLOR_FUENTE["Simulado"])
        ax.set_title(mod, loc="left", fontsize=10)
    for ax in ejes[-3:]:
        ax.set_xlabel("Residuo de Cox-Snell")
    ejes[0].set_ylabel("H acumulado (Nelson-Aalen)")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    return guardar(fig, nombre)


# ---------------------------------------------------------------------------
# Extensión
# ---------------------------------------------------------------------------
COLOR_DISENO = {"Verdad (duración completa)": TINTA,
                "A: artículo (solo dispuestos 2018-2022)": "#eb6834",
                "B: cohorte con censura": "#2a78d6"}
LINEA_DISENO = {"Verdad (duración completa)": ":",
                "A: artículo (solo dispuestos 2018-2022)": "-",
                "B: cohorte con censura": "--"}


def extension_km(curvas, nombre="11_extension_km"):
    fig, ejes = _paneles(len(cfg.TIPOS), alto=2.9)
    for ax, tipo in zip(ejes, cfg.TIPOS):
        for diseno, largo in curvas.items():
            c = largo[largo["case_type"] == tipo]
            ax.step(c["mes"], c["S"], where="post", color=COLOR_DISENO[diseno],
                    ls=LINEA_DISENO[diseno], label=diseno,
                    lw=2.4 if diseno.startswith("Verdad") else 1.8)
        ax.set_title(tipo, loc="left", fontsize=10)
        ax.set_xlim(0, 132)
        ax.set_xticks(range(0, 140, 24))
    ejes[0].legend(loc="upper right", fontsize=7.5)
    for ax in ejes[-3:]:
        ax.set_xlabel("Tiempo en meses")
    fig.suptitle("Extensión: excluir los casos abiertos subestima la duración",
                 x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    return guardar(fig, nombre)


def extension_medianas(resumen, nombre="12_extension_medianas"):
    p = resumen.pivot(index="case_type", columns="diseno", values="mediana").loc[cfg.TIPOS[::-1]]
    y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hlines(y, p.min(axis=1), p.max(axis=1), color=REJILLA, lw=3, zorder=1)
    marcas = {"Verdad (duración completa)": "|",
              "A: artículo (solo dispuestos 2018-2022)": "o", "B: cohorte con censura": "D"}
    for diseno in COLOR_DISENO:
        es_verdad = marcas[diseno] == "|"
        ax.scatter(p[diseno], y, color=COLOR_DISENO[diseno], marker=marcas[diseno],
                   s=260 if es_verdad else 40, lw=2.5, label=diseno, zorder=5 if es_verdad else 3)
    ax.set_yticks(y, p.index)
    ax.set_xlabel("Mediana de duración (meses, Kaplan-Meier)")
    ax.set_title("Mediana estimada por diseño frente a la verdad simulada", loc="left")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    return guardar(fig, nombre)
