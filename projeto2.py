"""
Projeto 2 - Comunicações Móveis (UnB)
Combinadores de diversidade SC, MRC e EGC: outage, SEP (4-ASK) e capacidade ergódica.

Item 1  Curvas sob Rayleigh para L = 1..4 ramos, teoria x simulação
        (EGC: apenas simulação).                                  -> Figura 1
Item 2  Comparação SC x MRC x EGC (OP, SEP e capacidade).         -> Figura 2
Item 3  Ganho de diversidade das três métricas, Rayleigh x Rice
        (Rice apenas simulado).                                   -> Figuras 3 e 4

Convenções: linha contínua = expressão analítica; marcadores = simulação;
linha tracejada = resultado só simulado.

Simulação
  - SNRs por ramo gamma_l = gbar |h_l|^2 vindas do canal Rayleigh (ou Rice) gerado.
  - OP e capacidade: calculadas na SNR combinada.
  - SEP (Item 1): simulada símbolo a símbolo (sorteio 4-ASK, y_l = sqrt(gamma_l) a + n_l,
    combinação SC/MRC/EGC, decisão por distância mínima, contagem de erros).

Ganho de diversidade (Item 3)
  Ganho(L) = SNR média por ramo exigida com L = 1 menos a exigida com L ramos, em dB,
  para o mesmo nível da métrica (SEP = 1e-3, OP = 1e-3, C = 4 bit/s/Hz). Por
  homogeneidade dos combinadores, a OP exige apenas quantis da SNR combinada.
  O ganho inclui o ganho de arranjo (10 log10 L para MRC e EGC com canal sem
  desvanecimento), que é o limite do ganho quando K -> infinito.

Uso:
    python projeto2_diversidade.py
    python projeto2_diversidade.py --N 500000 --Ns 500000 --Ng 500000 --K 5
    python projeto2_diversidade.py --salvar figuras      # também grava PDF e PNG
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator
from scipy.optimize import brentq
from scipy.special import comb, erfc
from scipy.stats import gamma as gamma_dist

trapz = getattr(np, "trapezoid", None) or np.trapz

# =============================================================================
# Parâmetros
# =============================================================================
M_ASK = 4                                    # 4-ASK, como nos slides
A_M = 2 * (M_ASK - 1) / M_ASK                # a_M = 2(M-1)/M
B_M = 6 / (M_ASK**2 - 1)                     # b_M = 6/(M^2-1)
# Constelação com Es/N0 = 1 para ruído de variância 1 por ramo (N0 = 2):
# níveis D*{-3,-1,1,3}, D^2 = 2/5 (Es = 5 D^2 = 2). Assim gamma_l = beta_l^2.
D_ASK = np.sqrt(2.0 / 5.0)
MIN_ERROS = 20                               # mínimo de erros para aceitar um ponto de SEP

L_LISTA = (1, 2, 3, 4)                       # Item 1
L_GANHO = tuple(range(1, 9))                 # Item 3: ganho em função de L
L_REF = 4                                    # L usado nas comparações do Item 2
K_FINA = (0, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20)   # Item 3: ganho em função de K (K = 0: Rayleigh)
L_K = (2, 4, 8)                              # Item 3: ordens de diversidade na curva ganho x K
SNR_SEP_DB = np.arange(0, 31, 2)
SNR_CAP_DB = np.arange(-5, 26, 2)
GBAR_OP_DB = 10.0
GTH_DB = np.arange(-10, 21, 2)

ALVO_SEP, ALVO_OP, ALVO_CAP = 1e-3, 1e-3, 4.0   # níveis para o ganho de diversidade

COMBINADORES = ("SC", "MRC", "EGC")
TEM_TEORIA = {"SC": True, "MRC": True, "EGC": False}
CORES = {"SC": "tab:blue", "MRC": "tab:red", "EGC": "tab:green"}
MARCADORES = {"SC": "o", "MRC": "s", "EGC": "^"}

# Quadratura de Gauss-Legendre em theta in (0, pi/2) (integral de Craig)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(128)
_THETA = (_GL_X + 1.0) * np.pi / 4.0
_W_THETA = _GL_W * np.pi / 4.0
_S_THETA = B_M / (2.0 * np.sin(_THETA) ** 2)

# Grade (SNR normalizada, média 1) para a capacidade analítica
_U = np.linspace(0.0, 80.0, 80_001)


def db2lin(x) -> np.ndarray:
    return 10.0 ** (np.asarray(x, dtype=float) / 10.0)


# =============================================================================
# Canal e combinadores
# =============================================================================
def potencia_de_normais(re: np.ndarray, im: np.ndarray, K: float = 0.0) -> np.ndarray:
    """
    Potência de canal |h|^2 de média unitária a partir de normais padrão (re, im) já sorteadas.
    K é o fator de Rice (K = 0 equivale a Rayleigh). A SNR de cada ramo é gamma_l = gbar * |h_l|^2.
    Reutilizar (re, im) para vários K (números aleatórios comuns) reduz o ruído nas comparações.
    """
    los = np.sqrt(K / (K + 1.0))
    dif = np.sqrt(0.5 / (K + 1.0))
    return (los + dif * re) ** 2 + (dif * im) ** 2


def gera_potencia_canal(rng, N: int, L: int, K: float = 0.0, dtype=np.float64) -> np.ndarray:
    """N realizações de L ramos independentes, com fator de Rice K (K = 0: Rayleigh)."""
    return potencia_de_normais(rng.standard_normal((N, L), dtype=dtype),
                               rng.standard_normal((N, L), dtype=dtype), K)


def combina(g: np.ndarray, tipo: str) -> np.ndarray:
    """SNR na saída do combinador (g tem shape (N, L))."""
    if tipo == "SC":
        return g.max(axis=1)
    if tipo == "MRC":
        return g.sum(axis=1)
    if tipo == "EGC":                                    # Eq. (25)
        return np.sqrt(g).sum(axis=1) ** 2 / g.shape[1]
    raise ValueError(f"combinador desconhecido: {tipo}")


def pe_ask(gamma: np.ndarray) -> np.ndarray:
    """Pe(gamma) = a_M Q(sqrt(b_M gamma)) para M-ASK em AWGN."""
    return A_M * 0.5 * erfc(np.sqrt(B_M * gamma / 2.0))


# =============================================================================
# Amostras e simulação (Item 1)
# =============================================================================
@dataclass
class Amostras:
    """Canal Rayleigh, símbolos e ruído da simulação símbolo a símbolo."""
    g0: np.ndarray                               # potência de canal (N, Lmax), gbar = 1
    idx: np.ndarray                              # índice do símbolo transmitido (Ns,)
    ruido: np.ndarray                            # ruído de variância 1 por ramo (Ns, Lmax)
    cache: Dict[Tuple[str, int], np.ndarray]     # SNR combinada (gbar = 1)
    g32: np.ndarray                              # g0[:Ns] em float32
    a: np.ndarray                                # amplitude do símbolo transmitido (Ns,)

    @classmethod
    def cria(cls, rng, N: int, Ns: int, Lmax: int) -> "Amostras":
        g0 = gera_potencia_canal(rng, N, Lmax)
        idx = rng.integers(0, M_ASK, Ns).astype(np.int8)
        ruido = rng.standard_normal((Ns, Lmax), dtype=np.float32)
        cache = {(tp, L): combina(g0[:, :L], tp) for tp in COMBINADORES for L in L_LISTA}
        a = (D_ASK * (2 * idx - 3)).astype(np.float32)
        return cls(g0, idx, ruido, cache, g0[:Ns].astype(np.float32), a)


def sim_sep(am: Amostras, tp: str, L: int, snr_db) -> np.ndarray:
    """
    SEP simulada símbolo a símbolo. Em cada ramo, y_l = beta_l a + n_l, beta_l = sqrt(gamma_l).
      SC : z = y_k / beta_k, k = argmax beta_l
      MRC: z = sum(beta_l y_l) / sum(beta_l^2)
      EGC: z = sum(y_l) / sum(beta_l)
    Decisão pelo símbolo mais próximo (limiares em -2D, 0 e 2D).
    """
    g, n, a, idx = am.g32[:, :L], am.ruido[:, :L], am.a, am.idx
    linhas = np.arange(idx.size)
    saida = np.empty(len(snr_db))
    for i, gbar in enumerate(db2lin(snr_db)):
        gam = np.float32(gbar) * g
        beta = np.sqrt(gam)
        y = beta * a[:, None] + n
        if tp == "SC":
            k = beta.argmax(axis=1)
            z = y[linhas, k] / beta[linhas, k]
        elif tp == "MRC":
            z = (beta * y).sum(axis=1) / gam.sum(axis=1)
        else:
            z = y.sum(axis=1) / beta.sum(axis=1)
        dec = (z > -2 * D_ASK).astype(np.int8) + (z > 0) + (z > 2 * D_ASK)
        erros = np.count_nonzero(dec != idx)
        saida[i] = erros / idx.size if erros >= MIN_ERROS else np.nan
    return saida


def sim_op(am: Amostras, tp: str, L: int, gth_db) -> np.ndarray:
    gc = am.cache[(tp, L)]
    limiar = db2lin(gth_db) / db2lin(GBAR_OP_DB)
    return np.searchsorted(np.sort(gc), limiar) / gc.size


def sim_cap(am: Amostras, tp: str, L: int, snr_db) -> np.ndarray:
    gc = am.cache[(tp, L)]
    return np.array([np.log2(1.0 + g * gc).mean() for g in db2lin(snr_db)])


# =============================================================================
# Teoria, Rayleigh (gbar é a SNR média POR RAMO)
# =============================================================================
def _mgf(tipo: str, L: int, gb: np.ndarray) -> np.ndarray:
    sg = _S_THETA[None, :] * gb[:, None]
    if tipo == "SC":
        return sum((-1) ** l * comb(L - 1, l) * L / (l + 1 + sg) for l in range(L))
    return (1.0 + sg) ** (-L)                            # MRC: Gamma(L, gbar)


def teo_sep(tipo: str, L: int, snr_db) -> np.ndarray:
    """Pe = (a_M/pi) int_0^{pi/2} MGF(b_M / (2 sin^2 theta)) dtheta."""
    return A_M / np.pi * (_mgf(tipo, L, db2lin(snr_db)) @ _W_THETA)


def teo_op(tipo: str, L: int, gth_db) -> np.ndarray:
    gb, gth = db2lin(GBAR_OP_DB), db2lin(gth_db)
    if tipo == "SC":
        return (1.0 - np.exp(-gth / gb)) ** L            # Eq. (9)
    return gamma_dist.cdf(gth, a=L, scale=gb)


def teo_cap(tipo: str, L: int, snr_db) -> np.ndarray:
    if tipo == "SC":
        pdf = L * (1.0 - np.exp(-_U)) ** (L - 1) * np.exp(-_U)
    else:
        pdf = gamma_dist.pdf(_U, a=L)
    gb = db2lin(snr_db)[:, None]
    return trapz(np.log2(1.0 + gb * _U[None, :]) * pdf[None, :], _U, axis=1)


# =============================================================================
# Métricas do Item 1
# =============================================================================
@dataclass(frozen=True)
class Metrica:
    nome: str
    x_sim: np.ndarray
    x_teo: np.ndarray
    xlabel: str
    ylabel: str
    titulo: Callable[[str], str]
    sim: Callable[[Amostras, str, int, np.ndarray], np.ndarray]
    teo: Callable[[str, int, np.ndarray], np.ndarray]
    log: bool = True
    ylim: Optional[Tuple[float, float]] = None


METRICAS = (
    Metrica("SEP", SNR_SEP_DB, np.linspace(SNR_SEP_DB[0], SNR_SEP_DB[-1], 120),
            r"SNR média por ramo, $\bar{\gamma}_l$ (dB)", r"$\bar{P}_s$ (SEP, 4-ASK)",
            lambda tp: tp, sim_sep, teo_sep, ylim=(1e-5, 1)),
    Metrica("OP", GTH_DB, np.linspace(GTH_DB[0], GTH_DB[-1], 200),
            r"Limiar de SNR, $\gamma_{\mathrm{th}}$ (dB)", r"$P_{\mathrm{out}}$ ($\bar{\gamma}_l = 10$ dB)",
            lambda tp: tp, sim_op, teo_op, ylim=(1e-5, 1)),
    Metrica("CAP", SNR_CAP_DB, np.linspace(SNR_CAP_DB[0], SNR_CAP_DB[-1], 120),
            r"SNR média por ramo, $\bar{\gamma}_l$ (dB)", r"$C_{\mathrm{erg}}$ (bit/s/Hz)",
            lambda tp: tp, sim_cap, teo_cap, log=False),
)

Resultados = Dict[Tuple[str, str, int], Tuple[np.ndarray, Optional[np.ndarray]]]


def calcula_item1(am: Amostras) -> Resultados:
    """(simulação, teoria) para cada (métrica, combinador, L). Teoria = None no EGC."""
    return {(m.nome, tp, L): (m.sim(am, tp, L, m.x_sim),
                              m.teo(tp, L, m.x_teo) if TEM_TEORIA[tp] else None)
            for m in METRICAS for tp in COMBINADORES for L in L_LISTA}


# =============================================================================
# Ganho de diversidade (Item 3)
# =============================================================================
def _snr_para_nivel(media: Callable[[float], float], alvo: float, tipo: str) -> float:
    """SNR média por ramo (dB) em que a métrica atinge o alvo (busca por bissecção)."""
    if tipo == "SEP":
        f = lambda s: np.log10(max(media(s), 1e-300)) - np.log10(alvo)
    else:
        f = lambda s: media(s) - alvo
    return brentq(f, -30.0, 70.0, xtol=1e-3)


def snr_sep_para_alvo(gc: np.ndarray, alvo: float = None) -> float:
    """SNR média por ramo (dB) em que a SEP média atinge o alvo, dada a SNR combinada gc (gbar = 1)."""
    alvo = ALVO_SEP if alvo is None else alvo
    return _snr_para_nivel(lambda s: pe_ask(np.float32(db2lin(s)) * gc).mean(dtype=np.float64), alvo, "SEP")


def snr_cap_para_alvo(gc: np.ndarray) -> float:
    """SNR média por ramo (dB) em que a capacidade ergódica atinge ALVO_CAP."""
    return _snr_para_nivel(lambda s: np.log2(1.0 + np.float32(db2lin(s)) * gc).mean(dtype=np.float64),
                           ALVO_CAP, "CAP")


def _gc32(g: np.ndarray, tp: str, L: int) -> np.ndarray:
    return combina(g[:, :L], tp)


def ganhos(g: np.ndarray, Ls, combinadores=COMBINADORES) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Ganho de diversidade (dB) em relação a L = 1, para cada métrica e combinador.
    g: potências de canal de média unitária, shape (N, max(Ls)), float32.
    """
    ref = _gc32(g, "SC", 1)                              # L = 1: os três combinadores coincidem
    base = {"SEP": snr_sep_para_alvo(ref), "CAP": snr_cap_para_alvo(ref), "OP": np.quantile(ref, ALVO_OP)}
    out = {met: {tp: np.zeros(len(Ls)) for tp in combinadores} for met in ("SEP", "OP", "CAP")}
    for tp in combinadores:
        for i, L in enumerate(Ls):
            if L == 1:
                continue                                 # ganho nulo por definição
            gc = _gc32(g, tp, L)
            out["SEP"][tp][i] = base["SEP"] - snr_sep_para_alvo(gc)
            out["CAP"][tp][i] = base["CAP"] - snr_cap_para_alvo(gc)
            # OP depende só de gamma_th/gbar: ganho = razão entre quantis
            out["OP"][tp][i] = 10 * np.log10(np.quantile(gc, ALVO_OP) / base["OP"])
    return out


# =============================================================================
# Verificações de consistência
# =============================================================================
def verifica(am: Amostras) -> None:
    L, gb = L_REF, db2lin(10)
    print(f"Verificações (Rayleigh, L = {L}, gbar = 10 dB)")
    print(f"  E[gamma_l]   sim = {(gb * am.g0).mean():8.3f}   teoria = {gb:8.3f}")
    print(f"  E[gamma_SC]  sim = {(gb * am.cache[('SC', L)]).mean():8.3f}   "
          f"teoria = {gb * sum(1 / k for k in range(1, L + 1)):8.3f}")
    print(f"  E[gamma_MRC] sim = {(gb * am.cache[('MRC', L)]).mean():8.3f}   teoria = {L * gb:8.3f}")
    print("  SEP em 10 dB: símbolos x Pe(gamma) médio x teoria")
    for tp in COMBINADORES:
        s_pe = pe_ask(gb * am.cache[(tp, L)]).mean()
        s_teo = f"{teo_sep(tp, L, [10])[0]:.3e}" if TEM_TEORIA[tp] else "   --   "
        print(f"    {tp:<3} símbolos = {sim_sep(am, tp, L, [10])[0]:.3e}   Pe médio = {s_pe:.3e}   teoria = {s_teo}")
    print()


def resumo(g_ray: dict, g_ric: dict, K: float) -> None:
    i = L_GANHO.index(L_REF)
    print(f"Ganho de diversidade (dB) para L = {L_REF}  [alvos: SEP={ALVO_SEP:g}, OP={ALVO_OP:g}, C={ALVO_CAP:g}]")
    print(f"  {'métrica':<8}{'comb.':<6}{'Rayleigh':>10}{'Rice K=' + format(K, 'g'):>12}")
    for met in ("SEP", "OP", "CAP"):
        for tp in COMBINADORES:
            print(f"  {met:<8}{tp:<6}{g_ray[met][tp][i]:>10.2f}{g_ric[met][tp][i]:>12.2f}")
    print()


def testa_afirmacao(g_ray: dict, g_ric: dict, K: float, g_K: dict) -> None:
    """
    Afirmação (slide 11): a vantagem relativa da diversidade é maior em Rayleigh que em Rice.
    Teste 1: ganho(Rayleigh) > ganho(Rice, K fixo) para toda métrica, combinador e L = 2..8.
    Teste 2 (MRC, ganho x K): nenhum K > 0 supera o Rayleigh além da tolerância TOL, e o ganho
    não aumenta com K além de TOL. TOL absorve o ruído de Monte Carlo: para K muito pequeno o
    canal Rice é praticamente Rayleigh e as curvas são estatisticamente indistinguíveis.
    """
    TOL = 0.3                                            # dB
    d1 = np.concatenate([g_ray[m][tp][1:] - g_ric[m][tp][1:] for m in g_ray for tp in COMBINADORES])
    print(f"Teste 1: ganho(Rayleigh) > ganho(Rice, K={K:g}) em {int((d1 > 0).sum())} de {d1.size} casos "
          f"(menor margem: {d1.min():.2f} dB)")
    total = estrito = acima = viol = 0
    for m in g_K:
        for L, y in g_K[m].items():
            total += len(y) - 1
            estrito += int((y[0] > y[1:]).sum())
            acima += int((y[1:] - y[0] > TOL).sum())
            viol += int((np.diff(y) > TOL).sum())
    print(f"Teste 2: ganho(K=0) > ganho(K>0) em {estrito} de {total} casos "
          f"(os demais dentro de ±{TOL} dB, ruído de Monte Carlo); "
          f"K>0 acima de Rayleigh por mais que a tolerância: {acima}; aumentos com K acima da tolerância: {viol}")
    veredito = (d1 > 0).all() and acima == 0 and viol == 0
    print(f"-> afirmação {'VERDADEIRA' if veredito else 'NÃO se sustenta em todos os casos'}\n")


# =============================================================================
# Estilo e figuras (formato de periódico, duas colunas: 7,16 pol)
# =============================================================================
LARGURA = 7.16

ESTILO = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "grid.linewidth": 0.4, "grid.alpha": 0.35, "grid.color": "0.5",
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

# Paleta de Okabe-Ito (distinguível por daltônicos); marcadores distintos para impressão P&B
CORES = {"SC": "#0072B2", "MRC": "#D55E00", "EGC": "#009E73"}
MARCADORES = {"SC": "o", "MRC": "s", "EGC": "^"}
MEW, MS = 0.7, 3.6


def _marca(ax, x, y, cor, mk, **kw):
    """Marcadores vazados (simulação)."""
    ax.plot(x, y, ls=kw.pop("ls", "none"), marker=mk, mfc="white", mec=cor, mew=MEW, ms=MS, color=cor, **kw)


def _formata(ax, titulo, xlabel, ylabel, log=False, ylim=None):
    ax.set_title(titulo, loc="left", fontweight="bold", pad=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major")
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if ylim:
        ax.set_ylim(*ylim)


def _alvos() -> dict:
    """Rótulos dos níveis usados no ganho de diversidade."""
    p10 = lambda x: rf"10^{{{int(round(np.log10(x)))}}}"
    return {"SEP": rf"$\bar{{P}}_s={p10(ALVO_SEP)}$", "OP": rf"$P_{{\mathrm{{out}}}}={p10(ALVO_OP)}$",
            "CAP": rf"$C_{{\mathrm{{erg}}}}={ALVO_CAP:g}$ bit/s/Hz"}


def _legenda(fig, handles, ncol):
    fig.legend(handles=handles, loc="outside upper center", ncol=ncol,
               handlelength=2.3, columnspacing=1.4, handletextpad=0.5)


def _proxy_estilo():
    """Entradas de legenda para a convenção teoria / simulação."""
    kw = dict(color="k", mfc="white", mew=MEW, ms=MS)
    return [Line2D([], [], lw=1.0, label="Teoria", **kw),
            Line2D([], [], ls="none", marker="o", label="Simulação", **kw),
            Line2D([], [], ls="--", lw=1.0, marker="o", label="Somente simulação", **kw)]


# Estilo dos slides da disciplina (Figuras 2-7): fonte Computer Modern, cores do tab10,
# curvas teóricas contínuas e simulação com marcadores "+" pretos, grade tracejada.
ESTILO_SLIDES = {
    "font.family": "serif", "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "lines.linewidth": 2.0,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.top": False, "ytick.right": False,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "grid.linestyle": "--", "grid.linewidth": 0.6, "grid.alpha": 0.6, "grid.color": "0.6",
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "figure.dpi": 100, "savefig.dpi": 300,
}
NOME_COMB = {"SC": "Selection Combining (SC)", "MRC": "Maximum Ratio Combining (MRC)",
             "EGC": "Equal Gain Combining (EGC)"}
COR_SLIDE = {  # OP/SEP usam o tab10; a capacidade usa azul, vermelho, verde e roxo, como no slide
    "OP": {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"},
    "SEP": {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"},
    "CAP": {1: "blue", 2: "red", 3: "green", 4: "purple"},
}
FORMATO_SLIDE = {  # título, rótulos, limites e posição da legenda de cada métrica
    "OP": dict(titulo="OP, {c}, SNR média por ramo = 10 dB", xlabel=r"Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)",
               ylabel="Probabilidade de Outage", xlim=(-10, 20), ylim=(1e-5, 1), leg="lower right"),
    "SEP": dict(titulo="SEP, {c} – 4-ASK", xlabel="SNR média por ramo (dB)", ylabel="SEP",
                xlim=(0, 30), ylim=(1e-5, 1), leg="lower left"),
    "CAP": dict(titulo="Capacidade ergódica, {c}", xlabel="SNR média por ramo (dB)",
                ylabel="Capacidade ergódica (bits/s/Hz)", xlim=(-5, 25), ylim=(0, 10), leg="upper left"),
}


import warnings
warnings.filterwarnings("ignore", message="Glyph .* missing from font")  # cmr10 não tem acentos; cai em STIX


COR_COMB = {"SC": "tab:blue", "MRC": "tab:red", "EGC": "tab:green"}
TITULO_ITEM2 = {
    "OP": "OP, SC, MRC e EGC (L={L}), SNR média por ramo = 10 dB",
    "SEP": "SEP, SC, MRC e EGC (L={L}) – 4-ASK",
    "CAP": "Capacidade ergódica, SC, MRC e EGC (L={L})",
}
PLUS = dict(ls="none", marker="+", color="k", ms=5, mew=1.5)      # marcador da simulação


def _eixo(ax, m, titulo, ylabel=True, ylim=None):
    """Rótulos, limites, escala e grade de um painel de métrica (formato dos slides)."""
    f = FORMATO_SLIDE[m.nome]
    ax.set(title=titulo, xlabel=f["xlabel"], xlim=f["xlim"], ylim=ylim or f["ylim"])
    if ylabel:
        ax.set_ylabel(f["ylabel"])
    if m.log:
        ax.set_yscale("log")
    ax.grid(True, which="both" if m.log else "major")


def _painel_item1(ax, m, tp, res, ylabel=True, legenda=True, titulo=None):
    """Item 1: curvas L = 1..4 de um combinador (teoria em linha, simulação em '+')."""
    for L in L_LISTA:
        sim, teo = res[(m.nome, tp, L)]
        x, y = (m.x_teo, teo) if teo is not None else (m.x_sim, sim)   # EGC: só simulação
        ax.plot(x, y, color=COR_SLIDE[m.nome][L], label=f"$L={L}$")
        ax.plot(m.x_sim, sim, **PLUS)
    ax.plot([], [], **PLUS, label="Simulação")
    _eixo(ax, m, titulo or NOME_COMB[tp], ylabel)
    if legenda:
        ax.legend(loc=FORMATO_SLIDE[m.nome]["leg"])


def _painel_item2(ax, m, res, ylabel=True, titulo=None, legenda=True):
    """Item 2: SC, MRC e EGC juntos (L = L_REF), com L = 1 como referência."""
    ax.plot(m.x_teo, res[(m.nome, "MRC", 1)][1], ":", color="0.4", lw=1.6, label="$L=1$ (sem diversidade)")
    for tp in COMBINADORES:
        sim, teo = res[(m.nome, tp, L_REF)]
        if teo is not None:
            ax.plot(m.x_teo, teo, color=COR_COMB[tp], label=tp)
        else:
            ax.plot(m.x_sim, sim, color=COR_COMB[tp], label="EGC (só simulação)")
        ax.plot(m.x_sim, sim, **PLUS)
    ax.plot([], [], **PLUS, label="Simulação")
    _eixo(ax, m, titulo or TITULO_ITEM2[m.nome].format(L=L_REF), ylabel, (0, 12) if m.nome == "CAP" else None)
    if legenda:
        ax.legend(loc=FORMATO_SLIDE[m.nome]["leg"])


def _fecha(fig):
    fig.canvas.draw()                                    # fixa fonte/estilo antes de sair do rc_context
    return fig


def figuras_item1(res: Resultados, individuais: bool) -> Dict[str, plt.Figure]:
    """Item 1: uma figura por métrica com os três combinadores lado a lado (eixo y comum);
    com `individuais`, também uma figura por (métrica, combinador), no tamanho dos slides."""
    figs = {}
    with plt.rc_context(ESTILO_SLIDES):
        for m in METRICAS:
            fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True, layout="constrained")
            for j, (ax, tp) in enumerate(zip(axs, COMBINADORES)):
                _painel_item1(ax, m, tp, res, ylabel=(j == 0), legenda=(j == 0))
            fig.suptitle(FORMATO_SLIDE[m.nome]["titulo"].format(c="SC, MRC e EGC"))
            figs[f"grupo1_{m.nome}"] = _fecha(fig)
            if individuais:
                for tp in COMBINADORES:
                    fig, ax = plt.subplots(figsize=(7.0, 3.9), layout="constrained")
                    _painel_item1(ax, m, tp, res, titulo=FORMATO_SLIDE[m.nome]["titulo"].format(c=NOME_COMB[tp]))
                    figs[f"fig1_{m.nome}_{tp}"] = _fecha(fig)
    return figs


def figuras_item2(res: Resultados, individuais: bool) -> Dict[str, plt.Figure]:
    """Item 2: bloco 1x3 com a comparação SC x MRC x EGC (OP, SEP e capacidade);
    com `individuais`, também uma figura por métrica, no tamanho dos slides."""
    figs = {}
    por_nome = {m.nome: m for m in METRICAS}
    with plt.rc_context(ESTILO_SLIDES):
        fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.8), layout="constrained")
        for ax, nome in zip(axs, ("OP", "SEP", "CAP")):
            m = por_nome[nome]
            _painel_item2(ax, m, res, ylabel=True, legenda=(nome == "CAP"),    # legenda única
                          titulo=TITULO_ITEM2[nome].format(L=L_REF).replace(", SNR média por ramo = 10 dB", ""))
        figs["grupo2_comparacao"] = _fecha(fig)
        if individuais:
            for m in METRICAS:
                fig, ax = plt.subplots(figsize=(7.0, 3.9), layout="constrained")
                _painel_item2(ax, m, res)
                figs[f"fig2_{m.nome}_comparacao"] = _fecha(fig)
    return figs


def fig3_ganho_vs_L(g_ray: dict, g_ric: dict, K: float):
    """Item 3 (a): ganho de diversidade em função de L, Rayleigh x Rice, três combinadores."""
    fig, axs = plt.subplots(1, 3, figsize=(LARGURA, 2.9), layout="constrained")
    nomes = {"SEP": "SEP", "OP": "Outage", "CAP": "Capacidade"}
    alvos = _alvos()
    for letra, ax, met in zip("abc", axs, ("SEP", "OP", "CAP")):
        for tp in COMBINADORES:
            ax.plot(L_GANHO, g_ray[met][tp], "-", color=CORES[tp], marker=MARCADORES[tp], ms=MS)
            ax.plot(L_GANHO, g_ric[met][tp], "--", color=CORES[tp], marker=MARCADORES[tp],
                    mfc="white", mew=MEW, ms=MS)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _formata(ax, f"({letra}) {nomes[met]}: {alvos[met]}", "Número de ramos, $L$", "Ganho de diversidade (dB)")
    h = [Line2D([], [], color=CORES[tp], marker=MARCADORES[tp], mfc="white", mew=MEW, ms=MS, lw=1.0, label=tp)
         for tp in COMBINADORES]
    h += [Line2D([], [], color="k", lw=1.0, marker="o", ms=MS, label="Rayleigh"),
          Line2D([], [], color="k", lw=1.0, ls="--", marker="o", mfc="white", mew=MEW, ms=MS,
                 label=f"Rice, $K={K:g}$")]
    _legenda(fig, h, ncol=5)
    return fig


def fig4_ganho_vs_K(g_K: dict):
    """
    Item 3 (b), figura principal: ganho de diversidade do MRC em função do fator de Rice K.
    Em K = 0 (Rayleigh) o ganho é máximo; cai com K e tende ao ganho de arranjo 10 log10 L
    (canal sem desvanecimento). A região sombreada é o ganho obtido contra o desvanecimento.
    """
    fig, axs = plt.subplots(1, 3, figsize=(LARGURA, 2.9), layout="constrained")
    nomes = {"SEP": "SEP", "OP": "Outage", "CAP": "Capacidade"}
    alvos = _alvos()
    cor = {L: plt.cm.viridis(c) for L, c in zip(L_K, (0.05, 0.42, 0.78))}
    mk = dict(zip(L_K, ("o", "s", "D")))
    K = np.array(K_FINA, dtype=float)
    for letra, ax, met in zip("abc", axs, ("SEP", "OP", "CAP")):
        for L in L_K:
            y, piso = g_K[met][L], 10 * np.log10(L)
            ax.fill_between(K, piso, y, color=cor[L], alpha=0.16, lw=0)
            ax.axhline(piso, color=cor[L], ls="--", lw=0.8)
            ax.plot(K, y, "-", color=cor[L], marker=mk[L], mfc="white", mew=MEW, ms=MS)
            ax.text(K[0] + 0.3, piso - 0.25, rf"$10\log_{{10}}{L}$", color=cor[L], fontsize=6.8,
                    ha="left", va="top")
        ax.set_xlim(-0.8, K[-1] + 0.8)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xticklabels(["0", "5", "10", "15", "20"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylim(bottom=0)
        _formata(ax, f"({letra}) {nomes[met]}: {alvos[met]}", r"Fator de Rice, $K$ ($K=0$: Rayleigh)", "Ganho de diversidade (dB)")
    h = [Line2D([], [], color=cor[L], marker=mk[L], mfc="white", mew=MEW, ms=MS, lw=1.0,
                label=f"MRC, $L={L}$") for L in L_K]
    h += [Patch(facecolor="0.5", alpha=0.3, label="Ganho contra o desvanecimento"),
          Line2D([], [], color="0.4", ls="--", lw=0.8, label=r"Ganho de arranjo ($K\to\infty$)")]
    _legenda(fig, h, ncol=5)
    return fig


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=2_000_000, help="amostras de canal no Item 1 (OP e capacidade)")
    ap.add_argument("--Ns", type=int, default=1_000_000, help="símbolos simulados na SEP do Item 1 (<= N)")
    ap.add_argument("--Ng", type=int, default=3_000_000, help="amostras de canal no Item 3 (ganho)")
    ap.add_argument("--K", type=float, default=5.0, help="fator de Rice na comparação com Rayleigh")
    ap.add_argument("--individuais", action="store_true", help="além das figuras agrupadas, gera uma figura por curva/métrica (tamanho de slide)")
    ap.add_argument("--salvar", metavar="PASTA", default=None, help="salva PDF (vetorial) e PNG 300 dpi; sem isso, só exibe")
    a = ap.parse_args()

    plt.rcParams.update(ESTILO)
    rng = np.random.default_rng()                # sem semente: resultados variam a cada execução

    # Item 1 e Item 2 (curvas)
    am = Amostras.cria(rng, a.N, min(a.Ns, a.N), max(L_LISTA))
    verifica(am)
    res = calcula_item1(am)

    # Item 3 (ganhos): mesmas normais base para Rayleigh, Rice e a varredura em K
    # (números aleatórios comuns: as comparações entre canais deixam de ter ruído de sorteios distintos)
    Lg = max(L_GANHO)
    re = rng.standard_normal((a.Ng, Lg), dtype=np.float32)
    im = rng.standard_normal((a.Ng, Lg), dtype=np.float32)
    g_ray = ganhos(potencia_de_normais(re, im, 0.0), L_GANHO)
    g_ric = ganhos(potencia_de_normais(re, im, a.K), L_GANHO)

    g_K = {met: {L: np.zeros(len(K_FINA)) for L in L_K} for met in ("SEP", "OP", "CAP")}
    for i, k in enumerate(K_FINA):                       # ganho do MRC em função de K
        gk = ganhos(potencia_de_normais(re[:, :max(L_K)], im[:, :max(L_K)], k), (1,) + L_K, ("MRC",))
        for met in g_K:
            for j, L in enumerate(L_K):
                g_K[met][L][i] = gk[met]["MRC"][j + 1]

    resumo(g_ray, g_ric, a.K)
    testa_afirmacao(g_ray, g_ric, a.K, g_K)

    figuras = {**figuras_item1(res, a.individuais),
               **figuras_item2(res, a.individuais),
               "fig3_item3_ganho_vs_L": fig3_ganho_vs_L(g_ray, g_ric, a.K),
               "fig4_item3_ganho_vs_K": fig4_ganho_vs_K(g_K)}
    if a.salvar:
        os.makedirs(a.salvar, exist_ok=True)
        for nome, fig in figuras.items():
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(a.salvar, f"{nome}.{ext}"))
    plt.show()


if __name__ == "__main__":
    main()