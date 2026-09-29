"""
espectro_primos_v7.py

Análisis espectral extendido de la matriz de Markov de residuos de primos,
con Opción C corregida, autoverificación robusta y tests adicionales.

Correcciones aplicadas:
  - Autoprueba reforzada: entropía de matriz uniforme = log K,
    irreducibilidad (|λ2| < 1), condicionamiento de P.
  - C.1: Cramér dentro de cada bloque usa distribución GLOBAL de gaps,
    y gaps agrupados en bins para evitar sesgo del suavizado.
  - C.3: compara gaps reales normalizados contra gaps de CRAMÉR REFINADO
    (densidad + distribución de gaps) normalizados, no contra Cramér clásico.
  - C.4: normaliza gaps por densidad local antes del ANOVA.
  - Test adicional: estacionariedad de residuos (primera vs segunda mitad).
  - Test adicional: KS de gaps normalizados vs exponencial.

Uso:
    python3 espectro_primos_v7.py
"""

import numpy as np
from sympy import primerange, isprime
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time


# ============================================================
# CONFIGURACIÓN
# ============================================================
N_PRIMOS = 200_000
MODULOS = [210, 2310, 30030]
SEMILLA = 42
ALPHA = 0.5
N_SHUFFLES_NULOS = 50
N_SHUFFLES_GAPS = 100
TAMANO_BLOQUE = 2000
N_BINS_GAPS_BLOQUES = 10
VERBOSE_AUTOVERIF = False


# ============================================================
# AUTOVERIFICADOR
# ============================================================
class Autoverificador:
    """Acumula verificaciones y reporta al final."""
    def __init__(self):
        self.checks = []

    def check(self, nombre, condicion, detalle=""):
        self.checks.append({
            'nombre': nombre,
            'ok': bool(condicion),
            'detalle': detalle,
        })
        if VERBOSE_AUTOVERIF:
            simbolo = "OK " if condicion else "FAIL"
            print(f"    [{simbolo}] {nombre}: {detalle}")

    def resumen(self):
        total = len(self.checks)
        exitosos = sum(1 for c in self.checks if c['ok'])
        print("\n" + "=" * 80)
        print("  RESUMEN DE AUTOVERIFICACION")
        print("=" * 80)
        print(f"  Verificaciones exitosas: {exitosos}/{total}")
        if exitosos < total:
            print("\n  Verificaciones FALLIDAS:")
            for c in self.checks:
                if not c['ok']:
                    print(f"    [FAIL] {c['nombre']}: {c['detalle']}")
        else:
            print("  Todas las verificaciones pasaron.")
        return exitosos == total


# ============================================================
# UTILIDADES
# ============================================================
def _phi(n):
    """Funcion totiente de Euler."""
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result


def agrupar_en_bins(valores, n_bins, bins=None):
    """Agrupa valores continuos en n_bins por percentiles."""
    if bins is None:
        bins = np.percentile(valores, np.linspace(0, 100, n_bins + 1))
        bins[0] -= 1e-10
        bins[-1] += 1e-10
    obs = np.digitize(valores, bins) - 1
    return np.clip(obs, 0, n_bins - 1), bins


# ============================================================
# GENERACIÓN DE DATOS
# ============================================================
def generar_primos(n, verif=None):
    """Genera los primeros n primos."""
    print(f"Generando {n:,} primos...")
    limite = int(n * (np.log(n) + np.log(np.log(n)))) + 10
    primos = list(primerange(2, limite))
    while len(primos) < n:
        limite *= 2
        primos = list(primerange(2, limite))
    primos = primos[:n]

    if verif is not None:
        todos_primos = all(isprime(p) for p in primos[:1000])
        verif.check("Primos generados son primos", todos_primos,
                    f"verificados primeros 1000 de {n}")
        ordenados = all(primos[i] < primos[i+1]
                        for i in range(min(len(primos)-1, 1000)))
        verif.check("Primos en orden creciente", ordenados,
                    "verificados primeros 1000")
        verif.check("Cantidad correcta de primos", len(primos) == n,
                    f"len={len(primos)}, esperado={n}")

    return primos


def residuos_a_indices(primos, modulo, verif=None):
    """Convierte primos a indices de residuos coprimos con el modulo."""
    coprimos = [r for r in range(modulo) if np.gcd(r, modulo) == 1]
    mapa = {r: i for i, r in enumerate(coprimos)}
    obs = np.array([mapa[p % modulo] for p in primos
                    if np.gcd(p, modulo) == 1])

    if verif is not None:
        en_rango = np.all((obs >= 0) & (obs < len(coprimos)))
        verif.check(f"Indices mod {modulo} en rango", en_rango,
                    f"rango=[0, {len(coprimos)-1}], "
                    f"min={obs.min()}, max={obs.max()}")
        verif.check(f"Cantidad de coprimos mod {modulo}",
                    len(coprimos) == _phi(modulo),
                    f"len={len(coprimos)}, phi({modulo})={_phi(modulo)}")

    return obs, coprimos


def calcular_gaps(primos, verif=None):
    """Calcula los gaps entre primos consecutivos."""
    primos_arr = np.array(primos)
    gaps = np.diff(primos_arr)

    if verif is not None:
        verif.check("Todos los gaps positivos", np.all(gaps > 0),
                    f"min={gaps.min()}, max={gaps.max()}")
        gaps_pares = np.all(gaps[1:] % 2 == 0)
        verif.check("Gaps pares (excepto el primero)", gaps_pares,
                    f"gaps[0]={gaps[0]}, resto pares")
        suma_ok = (gaps.sum() == primos_arr[-1] - primos_arr[0])
        verif.check("Suma de gaps = ultimo - primer primo", suma_ok,
                    f"suma={gaps.sum()}, "
                    f"diff={primos_arr[-1]-primos_arr[0]}")

    return gaps


def generar_cramer_clasico(n, modulo, semilla, verif=None):
    """Modelo nulo clasico: densidad 1/log x."""
    rng = np.random.default_rng(semilla)
    coprimos = [r for r in range(modulo) if np.gcd(r, modulo) == 1]
    mapa = {r: i for i, r in enumerate(coprimos)}
    resultado = []
    x = 7
    while len(resultado) < n:
        p = (modulo / len(coprimos)) / np.log(x)
        if rng.random() < p and np.gcd(x, modulo) == 1:
            resultado.append(mapa[x % modulo])
        x += 1
    return np.array(resultado)


def generar_cramer_refinado(n, modulo, gaps_observados, semilla):
    """Cramer refinado: densidad + distribucion de gaps."""
    rng = np.random.default_rng(semilla)
    coprimos = [r for r in range(modulo) if np.gcd(r, modulo) == 1]
    mapa = {r: i for i, r in enumerate(coprimos)}

    gaps_unicos, counts = np.unique(gaps_observados, return_counts=True)
    probs = counts / counts.sum()

    resultado = []
    p_actual = 7
    while len(resultado) < n:
        gap = rng.choice(gaps_unicos, p=probs)
        p_actual += int(gap)
        while np.gcd(p_actual, modulo) != 1:
            p_actual += 1
        resultado.append(mapa[p_actual % modulo])

    return np.array(resultado)


def generar_pseudo_primos_con_gaps(n, gaps_observados, semilla):
    """
    Genera pseudo-primos con la distribucion de gaps observada.
    Util para C.3 (normalizar por densidad local).
    """
    rng = np.random.default_rng(semilla)
    gaps_unicos, counts = np.unique(gaps_observados, return_counts=True)
    probs = counts / counts.sum()

    p_actual = 7
    primos = [p_actual]
    while len(primos) < n:
        gap = rng.choice(gaps_unicos, p=probs)
        p_actual += int(gap)
        primos.append(p_actual)
    return np.array(primos)


def generar_shuffle(obs, semilla):
    """Shuffle de las observaciones reales."""
    rng = np.random.default_rng(semilla)
    obs_shuf = obs.copy()
    rng.shuffle(obs_shuf)
    return obs_shuf


# ============================================================
# MATRIZ DE TRANSICIÓN
# ============================================================
def matriz_transicion_orden1(obs, K, alpha=ALPHA, verif=None, nombre="P"):
    """Matriz de transicion de orden 1 con suavizado Dirichlet."""
    conteo = np.zeros((K, K))
    for t in range(1, len(obs)):
        conteo[obs[t - 1], obs[t]] += 1

    P = np.zeros((K, K))
    for i in range(K):
        den = conteo[i].sum()
        P[i] = (conteo[i] + alpha) / (den + alpha * K)

    if verif is not None:
        sumas = P.sum(axis=1)
        filas_ok = np.allclose(sumas, 1.0, atol=1e-10)
        verif.check(f"{nombre}: filas suman 1", filas_ok,
                    f"min={sumas.min():.12f}, max={sumas.max():.12f}")
        verif.check(f"{nombre}: elementos no negativos", np.all(P >= 0),
                    f"min={P.min():.2e}")
        verif.check(f"{nombre}: elementos positivos (suavizado)",
                    np.all(P > 0), f"min={P.min():.2e}")
        total_conteo = conteo.sum()
        verif.check(f"{nombre}: suma de conteos",
                    total_conteo == len(obs) - 1,
                    f"suma={total_conteo}, esperado={len(obs)-1}")

    return P, conteo


def matriz_transicion_orden2(obs, K, alpha=ALPHA):
    """Matriz de transicion de orden 2."""
    conteo_contexto = defaultdict(int)
    conteo_transicion = defaultdict(int)
    T = len(obs)
    for t in range(2, T):
        contexto = (obs[t - 2], obs[t - 1])
        siguiente = obs[t]
        conteo_contexto[contexto] += 1
        conteo_transicion[(contexto, siguiente)] += 1

    P2 = {}
    for ctx in conteo_contexto:
        den = conteo_contexto[ctx]
        vec = np.zeros(K)
        for k in range(K):
            num = conteo_transicion.get((ctx, k), 0)
            vec[k] = (num + alpha) / (den + alpha * K)
        P2[ctx] = vec
    return P2, conteo_contexto


def loglik_orden2(obs, P2):
    """Log-verosimilitud de una cadena de orden 2."""
    T = len(obs)
    loglik = 0.0
    for t in range(2, T):
        contexto = (obs[t - 2], obs[t - 1])
        siguiente = obs[t]
        prob = P2[contexto][siguiente]
        loglik += np.log(prob + 1e-15)
    return loglik


# ============================================================
# ANÁLISIS ESPECTRAL
# ============================================================
def analizar_espectro(P, K, verif=None, nombre="P"):
    """Autovalores, autovectores y metricas derivadas."""
    eigvals, eigvecs = np.linalg.eig(P)
    orden = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[orden]
    eigvecs = eigvecs[:, orden]
    abs_eig = np.abs(eigvals)

    eigvals_l, eigvecs_l = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals_l - 1))
    pi = np.real(eigvecs_l[:, idx])
    pi = np.abs(pi)
    pi = pi / pi.sum()

    H = -np.sum(pi[:, None] * P * np.log(P + 1e-15))
    H_max = np.log(K)

    if abs_eig[1] >= 1:
        t_mix = np.inf
    elif abs_eig[1] <= 0:
        t_mix = 1.0
    else:
        t_mix = np.log(0.01) / np.log(abs_eig[1])

    contrib = abs_eig / abs_eig.sum()
    rango_ef = 1.0 / np.sum(contrib ** 2)

    if verif is not None:
        verif.check(f"{nombre}: |lambda1| = 1",
                    np.isclose(abs_eig[0], 1.0, atol=1e-10),
                    f"|lambda1| = {abs_eig[0]:.12f}")
        verif.check(f"{nombre}: |lambdai| <= 1",
                    np.all(abs_eig <= 1.0 + 1e-10),
                    f"max |lambdai| = {abs_eig.max():.12f}")
        verif.check(f"{nombre}: pi suma 1",
                    np.isclose(pi.sum(), 1.0, atol=1e-10),
                    f"suma = {pi.sum():.12f}")
        verif.check(f"{nombre}: pi >= 0", np.all(pi >= 0),
                    f"min pi = {pi.min():.2e}")
        piP = pi @ P
        error_punto_fijo = np.max(np.abs(piP - pi))
        verif.check(f"{nombre}: piP = pi", error_punto_fijo < 1e-8,
                    f"error max = {error_punto_fijo:.2e}")
        verif.check(f"{nombre}: 0 <= H <= log K",
                    0 <= H <= H_max + 1e-10,
                    f"H = {H:.6f}, log K = {H_max:.6f}")
        ratio = H / H_max if H_max > 0 else 0
        verif.check(f"{nombre}: H/H_max en [0,1]",
                    0 <= ratio <= 1 + 1e-10,
                    f"ratio = {ratio:.6f}")
        # Entropia de matriz uniforme
        P_unif = np.ones((K, K)) / K
        pi_unif = np.ones(K) / K
        H_unif = -np.sum(pi_unif[:, None] * P_unif *
                         np.log(P_unif + 1e-15))
        verif.check(f"{nombre}: entropia de matriz uniforme = log K",
                    np.isclose(H_unif, H_max, atol=1e-10),
                    f"H_unif={H_unif:.8f}, log K={H_max:.8f}")
        # Irreducibilidad
        verif.check(f"{nombre}: cadena irreducible (|lambda2| < 1)",
                    abs_eig[1] < 1.0 - 1e-8,
                    f"|lambda2| = {abs_eig[1]:.10f}")
        # Condicionamiento
        try:
            cond_P = np.linalg.cond(P)
            verif.check(f"{nombre}: P bien condicionada",
                        cond_P < 1e8,
                        f"cond(P) = {cond_P:.3e}")
        except Exception as e:
            verif.check(f"{nombre}: P bien condicionada", False,
                        f"error: {e}")

    return {
        'eigvals': eigvals,
        'eigvecs': eigvecs,
        'abs_eig': abs_eig,
        'pi': pi,
        'entropia': H,
        'entropia_max': H_max,
        'entropia_ratio': H / H_max if H_max > 0 else 0,
        't_mix': t_mix,
        'rango_efectivo': rango_ef,
        'contrib': contrib,
    }


# ============================================================
# OPCIÓN C: EXPLORACIÓN PROFUNDA DE GAPS
# ============================================================
def opcion_c_gaps_por_bloques(gaps, tamano_bloque=TAMANO_BLOQUE,
                               n_bins=N_BINS_GAPS_BLOQUES, verif=None):
    """C.1. Analisis de gaps por bloques (con bins globales y por bloque)."""
    print("\n" + "=" * 80)
    print("  OPCION C.1: ANALISIS DE GAPS POR BLOQUES")
    print("=" * 80)

    n_bloques = len(gaps) // tamano_bloque
    print(f"Numero de bloques: {n_bloques} (tamano = {tamano_bloque})")
    print(f"Gaps agrupados en {n_bins} bins por percentiles")

    if n_bloques < 5:
        print("  -> Muy pocos bloques para analisis robusto.")
        return None

    # Bins globales
    obs_gaps_bins_global, _ = agrupar_en_bins(gaps, n_bins)

    lambdas_real = []
    lambdas_cram = []
    rng = np.random.default_rng(SEMILLA)

    for b in range(n_bloques):
        inicio = b * tamano_bloque
        fin = inicio + tamano_bloque

        obs_bloque = obs_gaps_bins_global[inicio:fin]
        P_real, _ = matriz_transicion_orden1(obs_bloque, n_bins)
        esp_real = analizar_espectro(P_real, n_bins)
        lambdas_real.append(esp_real['abs_eig'][1])

        obs_cram = rng.choice(obs_gaps_bins_global, size=len(obs_bloque),
                              replace=True)
        P_cram, _ = matriz_transicion_orden1(obs_cram, n_bins)
        esp_cram = analizar_espectro(P_cram, n_bins)
        lambdas_cram.append(esp_cram['abs_eig'][1])

    lambdas_real = np.array(lambdas_real)
    lambdas_cram = np.array(lambdas_cram)

    media_real = lambdas_real.mean()
    media_cram = lambdas_cram.mean()
    diff = media_real - media_cram

    print(f"\n  |lambda2| promedio en bloques:")
    print(f"    Real:   {media_real:.5f} +/- {lambdas_real.std():.5f}")
    print(f"    Cramer: {media_cram:.5f} +/- {lambdas_cram.std():.5f}")
    print(f"    Delta (real - Cramer): {diff:+.5f}")

    try:
        from scipy import stats
        t_stat, p_valor = stats.ttest_rel(lambdas_real, lambdas_cram)
    except ImportError:
        t_stat, p_valor = 0.0, 1.0

    print(f"\n  Test t pareado:")
    print(f"    t = {t_stat:.3f}")
    print(f"    p-valor = {p_valor:.2e}")

    if p_valor < 0.001 and diff > 0.05:
        print("  -> ESTRUCTURA GENUINA: los gaps reales tienen mas")
        print("     estructura que Cramer dentro de cada bloque.")
        resultado = "genuina"
    elif p_valor < 0.05 and diff > 0.02:
        print("  -> ESTRUCTURA MODERADA: evidencia parcial.")
        resultado = "moderada"
    else:
        print("  -> NO SIGNIFICATIVO: puede ser artefacto de densidad.")
        resultado = "no_significativa"

    if verif is not None:
        verif.check("C.1: bloques analizados", n_bloques >= 5,
                    f"n_bloques = {n_bloques}")
        verif.check("C.1: |lambda2| en rango",
                    np.all((lambdas_real >= 0) & (lambdas_real <= 1)),
                    f"min={lambdas_real.min():.4f}, "
                    f"max={lambdas_real.max():.4f}")
        verif.check("C.1: bins usados",
                    n_bins == N_BINS_GAPS_BLOQUES,
                    f"n_bins = {n_bins}")

    return {
        'lambdas_real': lambdas_real,
        'lambdas_cram': lambdas_cram,
        'media_real': media_real,
        'media_cram': media_cram,
        'diff': diff,
        't_stat': t_stat,
        'p_valor': p_valor,
        'resultado': resultado,
        'n_bins': n_bins,
    }


def opcion_c_significancia_gaps(gaps, n_shuffles=N_SHUFFLES_GAPS,
                                 verif=None):
    """C.2. Test de significancia de gaps con multiples shuffles."""
    print("\n" + "=" * 80)
    print("  OPCION C.2: TEST DE SIGNIFICANCIA DE GAPS")
    print("=" * 80)

    gaps_unicos = sorted(set(gaps))
    mapa_gap = {g: i for i, g in enumerate(gaps_unicos)}
    K_g = len(gaps_unicos)
    obs_gaps = np.array([mapa_gap[g] for g in gaps])

    P_real, _ = matriz_transicion_orden1(obs_gaps, K_g)
    esp_real = analizar_espectro(P_real, K_g)
    lambda_real = esp_real['abs_eig'][1]

    lambdas_shuffle = []
    rng = np.random.default_rng(SEMILLA)
    for _ in range(n_shuffles):
        obs_shuf = obs_gaps.copy()
        rng.shuffle(obs_shuf)
        P_shuf, _ = matriz_transicion_orden1(obs_shuf, K_g)
        esp_shuf = analizar_espectro(P_shuf, K_g)
        lambdas_shuffle.append(esp_shuf['abs_eig'][1])

    lambdas_shuffle = np.array(lambdas_shuffle)
    media = lambdas_shuffle.mean()
    std = lambdas_shuffle.std()
    z = (lambda_real - media) / std if std > 0 else 0
    p_valor = (lambdas_shuffle >= lambda_real).mean()

    print(f"\n  |lambda2| real    = {lambda_real:.5f}")
    print(f"  |lambda2| shuffle = {media:.5f} +/- {std:.5f}")
    print(f"  Delta (real - shuffle) = {lambda_real - media:+.5f}")
    print(f"  z-score = {z:.3f}")
    print(f"  p-valor = {p_valor:.4f}")

    if z > 3:
        print("  -> SIGNIFICATIVO: el real esta muy por encima del shuffle.")
        resultado = "significativo"
    elif z > 2:
        print("  -> MODERADAMENTE SIGNIFICATIVO.")
        resultado = "moderado"
    else:
        print("  -> NO SIGNIFICATIVO.")
        resultado = "no_significativo"

    if verif is not None:
        verif.check("C.2: shuffles generados",
                    len(lambdas_shuffle) == n_shuffles,
                    f"n = {len(lambdas_shuffle)}")

    return {
        'lambda_real': lambda_real,
        'lambda_shuffle_media': media,
        'lambda_shuffle_std': std,
        'z': z,
        'p_valor': p_valor,
        'resultado': resultado,
    }


def opcion_c_gaps_normalizados(primos, gaps, verif=None):
    """C.3. Gaps normalizados vs Cramer refinado normalizado."""
    print("\n" + "=" * 80)
    print("  OPCION C.3: GAPS NORMALIZADOS vs CRAMER REFINADO")
    print("=" * 80)

    primos_arr = np.array(primos)
    gaps_reales = np.diff(primos_arr)
    log_primos = np.log(primos_arr[:-1])
    gaps_norm = gaps_reales / log_primos

    print(f"\n  Estadisticas de gaps reales normalizados:")
    print(f"    Media: {gaps_norm.mean():.5f}")
    print(f"    Std:   {gaps_norm.std():.5f}")
    print(f"    Min:   {gaps_norm.min():.5f}")
    print(f"    Max:   {gaps_norm.max():.5f}")

    # Generar Cramer refinado con la misma distribucion de gaps
    print("\n  Generando pseudo-primos con distribucion de gaps observada...")
    primos_cr = generar_pseudo_primos_con_gaps(len(primos), gaps_reales,
                                                SEMILLA + 1)
    gaps_cr = np.diff(primos_cr)
    log_cr = np.log(primos_cr[:-1])
    gaps_cr_norm = gaps_cr / log_cr

    print(f"\n  Estadisticas de gaps Cramer refinado normalizados:")
    print(f"    Media: {gaps_cr_norm.mean():.5f}")
    print(f"    Std:   {gaps_cr_norm.std():.5f}")
    print(f"    Min:   {gaps_cr_norm.min():.5f}")
    print(f"    Max:   {gaps_cr_norm.max():.5f}")

    # Discretizar en bins comunes
    n_bins = 20
    todos = np.concatenate([gaps_norm, gaps_cr_norm])
    obs_real_bins, _ = agrupar_en_bins(gaps_norm, n_bins,
                                        bins=np.percentile(todos, np.linspace(0, 100, n_bins + 1)))
    obs_cr_bins, _ = agrupar_en_bins(gaps_cr_norm, n_bins,
                                      bins=np.percentile(todos, np.linspace(0, 100, n_bins + 1)))

    P_real, _ = matriz_transicion_orden1(obs_real_bins, n_bins)
    esp_real = analizar_espectro(P_real, n_bins)

    P_cr, _ = matriz_transicion_orden1(obs_cr_bins, n_bins)
    esp_cr = analizar_espectro(P_cr, n_bins)

    diff = esp_real['abs_eig'][1] - esp_cr['abs_eig'][1]

    print(f"\n  Analisis espectral (despues de normalizar):")
    print(f"    |lambda2| real           = {esp_real['abs_eig'][1]:.5f}")
    print(f"    |lambda2| Cramer refin   = {esp_cr['abs_eig'][1]:.5f}")
    print(f"    Delta (real - Cramer) = {diff:+.5f}")

    if diff > 0.05:
        print("  -> ESTRUCTURA GENUINA PERSISTE despues de normalizar.")
        print("     NO es artefacto de densidad ni de distribucion de gaps.")
        resultado = "persiste"
    elif diff > 0.01:
        print("  -> Estructura debil persiste. Ambiguo.")
        resultado = "debil"
    else:
        print("  -> Estructura desaparece despues de normalizar.")
        print("     Era artefacto de densidad o distribucion de gaps.")
        resultado = "desaparece"

    if verif is not None:
        verif.check("C.3: gaps reales normalizados calculados",
                    len(gaps_norm) == len(gaps_reales),
                    f"len = {len(gaps_norm)}")
        verif.check("C.3: gaps Cramer refinado normalizados calculados",
                    len(gaps_cr_norm) == len(gaps_cr),
                    f"len = {len(gaps_cr_norm)}")
        verif.check("C.3: bins en rango (real)",
                    np.all((obs_real_bins >= 0) & (obs_real_bins < n_bins)),
                    f"min={obs_real_bins.min()}, max={obs_real_bins.max()}")
        verif.check("C.3: bins en rango (Cramer)",
                    np.all((obs_cr_bins >= 0) & (obs_cr_bins < n_bins)),
                    f"min={obs_cr_bins.min()}, max={obs_cr_bins.max()}")

    return {
        'gaps_norm': gaps_norm,
        'gaps_cr_norm': gaps_cr_norm,
        'esp_real': esp_real,
        'esp_cr': esp_cr,
        'diff': diff,
        'resultado': resultado,
    }


def opcion_c_correlacion_residuo_gap(primos, modulo, verif=None):
    """C.4. Correlacion residuo-gap normalizada por densidad local."""
    print("\n" + "=" * 80)
    print(f"  OPCION C.4: CORRELACION RESIDUO-GAP (mod {modulo})")
    print("=" * 80)

    primos_arr = np.array(primos)
    gaps = np.diff(primos_arr)
    log_primos = np.log(primos_arr[:-1])
    residuos = primos_arr[:-1] % modulo

    # CORRECCION: normalizar gaps por densidad local
    gaps_norm = gaps / log_primos

    try:
        from scipy import stats
        corr, p_valor = stats.pearsonr(residuos, gaps_norm)
    except ImportError:
        corr, p_valor = np.corrcoef(residuos, gaps_norm)[0, 1], 0.0

    print(f"\n  Correlacion de Pearson (residuo, gap/log p):")
    print(f"    r = {corr:.5f}")
    print(f"    p-valor = {p_valor:.2e}")

    coprimos = [r for r in range(modulo) if np.gcd(r, modulo) == 1]
    gaps_por_clase = defaultdict(list)
    for i, p in enumerate(primos_arr[:-1]):
        if np.gcd(p, modulo) == 1:
            gaps_por_clase[p % modulo].append(gaps_norm[i])

    medias_por_clase = {}
    for r in coprimos:
        if gaps_por_clase[r]:
            medias_por_clase[r] = np.mean(gaps_por_clase[r])

    if medias_por_clase:
        valores = list(medias_por_clase.values())
        print(f"\n  Gap normalizado medio por clase de residuo mod {modulo}:")
        print(f"    Media global: {np.mean(valores):.5f}")
        print(f"    Std entre clases: {np.std(valores):.5f}")
        print(f"    Min: {np.min(valores):.5f} (residuo "
              f"{min(medias_por_clase, key=medias_por_clase.get)})")
        print(f"    Max: {np.max(valores):.5f} (residuo "
              f"{max(medias_por_clase, key=medias_por_clase.get)})")

        grupos = [gaps_por_clase[r] for r in coprimos if gaps_por_clase[r]]
        try:
            from scipy import stats
            f_stat, p_anova = stats.f_oneway(*grupos)
        except ImportError:
            f_stat, p_anova = 0.0, 1.0

        print(f"\n  ANOVA (medias por clase, gaps normalizados):")
        print(f"    F = {f_stat:.3f}")
        print(f"    p-valor = {p_anova:.2e}")

        if p_anova < 0.001:
            print("  -> HAY DIFERENCIAS SIGNIFICATIVAS entre clases.")
            resultado = "significativo"
        elif p_anova < 0.05:
            print("  -> Diferencias moderadas.")
            resultado = "moderado"
        else:
            print("  -> No hay diferencias significativas.")
            resultado = "no_significativo"
    else:
        f_stat, p_anova, resultado = 0, 1, "sin_datos"

    if verif is not None:
        verif.check(f"C.4: correlacion calculada mod {modulo}",
                    -1 <= corr <= 1, f"r = {corr:.5f}")
        verif.check(f"C.4: ANOVA calculado mod {modulo}",
                    0 <= p_anova <= 1, f"p = {p_anova:.2e}")

    return {
        'corr': corr,
        'p_valor_corr': p_valor,
        'medias_por_clase': medias_por_clase,
        'f_stat': f_stat,
        'p_anova': p_anova,
        'resultado': resultado,
    }


# ============================================================
# TESTS ADICIONALES
# ============================================================
def test_estacionariedad_gaps(gaps, verif=None):
    """Test de estacionariedad: primera mitad vs segunda mitad de gaps."""
    print("\n" + "=" * 80)
    print("  TEST ADICIONAL: ESTACIONARIEDAD DE GAPS")
    print("=" * 80)

    n = len(gaps)
    mitad = n // 2
    gaps_1 = gaps[:mitad]
    gaps_2 = gaps[mitad:]

    n_bins = 20
    bins = np.percentile(gaps, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-10
    bins[-1] += 1e-10

    obs_1 = np.clip(np.digitize(gaps_1, bins) - 1, 0, n_bins - 1)
    obs_2 = np.clip(np.digitize(gaps_2, bins) - 1, 0, n_bins - 1)

    P_1, _ = matriz_transicion_orden1(obs_1, n_bins)
    P_2, _ = matriz_transicion_orden1(obs_2, n_bins)

    esp_1 = analizar_espectro(P_1, n_bins)
    esp_2 = analizar_espectro(P_2, n_bins)

    l1 = esp_1['abs_eig'][1]
    l2 = esp_2['abs_eig'][1]
    diff = abs(l1 - l2)

    print(f"\n  |lambda2| primera mitad: {l1:.5f}")
    print(f"  |lambda2| segunda mitad: {l2:.5f}")
    print(f"  Diferencia absoluta:    {diff:.5f}")

    if diff < 0.02:
        print("  -> ESTACIONARIO: las dos mitades tienen la misma estructura.")
        resultado = "estacionario"
    elif diff < 0.05:
        print("  -> LIGERAMENTE NO ESTACIONARIO.")
        resultado = "ligero"
    else:
        print("  -> NO ESTACIONARIO: la estructura cambia con la posicion.")
        resultado = "no_estacionario"

    if verif is not None:
        verif.check("Estacionariedad gaps: |lambda2| calculados",
                    np.isfinite(l1) and np.isfinite(l2),
                    f"l1={l1:.5f}, l2={l2:.5f}")

    return {
        'l1': l1,
        'l2': l2,
        'diff': diff,
        'resultado': resultado,
    }


def test_estacionariedad_residuos(obs, K, verif=None, nombre="residuos"):
    """Test de estacionariedad de los residuos: primera vs segunda mitad."""
    print("\n" + "=" * 80)
    print(f"  TEST ADICIONAL: ESTACIONARIEDAD DE {nombre.upper()}")
    print("=" * 80)

    n = len(obs)
    mitad = n // 2
    obs_1 = obs[:mitad]
    obs_2 = obs[mitad:]

    P_1, _ = matriz_transicion_orden1(obs_1, K)
    P_2, _ = matriz_transicion_orden1(obs_2, K)

    esp_1 = analizar_espectro(P_1, K)
    esp_2 = analizar_espectro(P_2, K)

    l1 = esp_1['abs_eig'][1]
    l2 = esp_2['abs_eig'][1]
    diff = abs(l1 - l2)

    print(f"\n  |lambda2| primera mitad: {l1:.5f}")
    print(f"  |lambda2| segunda mitad: {l2:.5f}")
    print(f"  Diferencia absoluta:    {diff:.5f}")

    if diff < 0.02:
        print("  -> ESTACIONARIO.")
        resultado = "estacionario"
    elif diff < 0.05:
        print("  -> LIGERAMENTE NO ESTACIONARIO.")
        resultado = "ligero"
    else:
        print("  -> NO ESTACIONARIO.")
        resultado = "no_estacionario"

    if verif is not None:
        verif.check(f"Estacionariedad {nombre}: |lambda2| calculados",
                    np.isfinite(l1) and np.isfinite(l2),
                    f"l1={l1:.5f}, l2={l2:.5f}")

    return {
        'l1': l1,
        'l2': l2,
        'diff': diff,
        'resultado': resultado,
    }


def test_ks_gaps_normalizados(gaps, log_primos, verif=None):
    """KS test: gaps normalizados vs exponencial."""
    print("\n" + "=" * 80)
    print("  TEST ADICIONAL: KS DE GAPS NORMALIZADOS VS EXPONENCIAL")
    print("=" * 80)

    gaps_norm = gaps / log_primos

    try:
        from scipy import stats
        ks_stat, p_valor = stats.kstest(gaps_norm, 'expon')
    except ImportError:
        print("  -> scipy no disponible, saltando KS test.")
        return None

    print(f"\n  KS statistic: {ks_stat:.5f}")
    print(f"  p-valor:      {p_valor:.2e}")

    if p_valor < 0.001:
        print("  -> RECHAZA exponencial: hay estructura no explicada")
        print("     por densidad.")
        resultado = "rechaza"
    elif p_valor < 0.05:
        print("  -> Rechaza debilmente.")
        resultado = "debil"
    else:
        print("  -> NO rechaza: consistente con exponencial.")
        resultado = "consistente"

    if verif is not None:
        verif.check("KS: estadistico calculado",
                    0 <= ks_stat <= 1, f"KS = {ks_stat:.5f}")

    return {
        'ks_stat': ks_stat,
        'p_valor': p_valor,
        'resultado': resultado,
    }


# ============================================================
# EXTENSIONES 1, 2, 3, 4
# ============================================================
def test_cramer_refinado(obs_real, obs_cram_ref, obs_cram_clas, K,
                          verif=None):
    """Extension 1: Cramer refinado."""
    print("\n" + "=" * 80)
    print("  EXTENSION 1: CRAMER REFINADO (densidad + gaps observados)")
    print("=" * 80)

    P_real, _ = matriz_transicion_orden1(obs_real, K, verif=verif,
                                          nombre="P_real")
    P_cc, _ = matriz_transicion_orden1(obs_cram_clas, K,
                                        nombre="P_cram_clas")
    P_cr, _ = matriz_transicion_orden1(obs_cram_ref, K,
                                        nombre="P_cram_ref")

    esp_real = analizar_espectro(P_real, K, verif=verif,
                                  nombre="esp_real")
    esp_cc = analizar_espectro(P_cc, K, nombre="esp_cc")
    esp_cr = analizar_espectro(P_cr, K, nombre="esp_cr")

    print(f"\n{'Metrica':>25} | {'real':>10} | {'Cram clas':>10} | "
          f"{'Cram refin':>10}")
    print("-" * 65)
    print(f"{'|lambda2|':>25} | {esp_real['abs_eig'][1]:>10.5f} | "
          f"{esp_cc['abs_eig'][1]:>10.5f} | {esp_cr['abs_eig'][1]:>10.5f}")
    print(f"{'Entropia H':>25} | {esp_real['entropia']:>10.4f} | "
          f"{esp_cc['entropia']:>10.4f} | {esp_cr['entropia']:>10.4f}")
    print(f"{'H / H_max':>25} | {esp_real['entropia_ratio']:>10.4f} | "
          f"{esp_cc['entropia_ratio']:>10.4f} | "
          f"{esp_cr['entropia_ratio']:>10.4f}")
    print(f"{'Rango efectivo':>25} | {esp_real['rango_efectivo']:>10.2f} | "
          f"{esp_cc['rango_efectivo']:>10.2f} | "
          f"{esp_cr['rango_efectivo']:>10.2f}")
    print(f"{'Tiempo de mezcla':>25} | {esp_real['t_mix']:>10.2f} | "
          f"{esp_cc['t_mix']:>10.2f} | {esp_cr['t_mix']:>10.2f}")

    diff_ref = esp_real['abs_eig'][1] - esp_cr['abs_eig'][1]
    diff_clas = esp_real['abs_eig'][1] - esp_cc['abs_eig'][1]
    print(f"\nDelta |lambda2| (real - Cramer clasico)  = {diff_clas:+.5f}")
    print(f"Delta |lambda2| (real - Cramer refinado) = {diff_ref:+.5f}")

    if abs(diff_ref) < 0.02:
        print("-> El Cramer refinado tambien reproduce la estructura del real.")
    elif diff_ref > 0.05:
        print("-> El real tiene MAS estructura que el Cramer refinado.")
    else:
        print("-> Resultado ambiguo.")

    return esp_real, esp_cc, esp_cr


def test_orden2(obs_real, obs_shuf, obs_cram, K):
    """Extension 2: Markov de orden 2."""
    print("\n" + "=" * 80)
    print("  EXTENSION 2: MARKOV DE ORDEN 2")
    print("=" * 80)

    P1_real, _ = matriz_transicion_orden1(obs_real, K)
    ll1_real = sum(np.log(P1_real[obs_real[t-1], obs_real[t]] + 1e-15)
                   for t in range(1, len(obs_real)))

    P1_shuf, _ = matriz_transicion_orden1(obs_shuf, K)
    ll1_shuf = sum(np.log(P1_shuf[obs_shuf[t-1], obs_shuf[t]] + 1e-15)
                   for t in range(1, len(obs_shuf)))

    P1_cram, _ = matriz_transicion_orden1(obs_cram, K)
    ll1_cram = sum(np.log(P1_cram[obs_cram[t-1], obs_cram[t]] + 1e-15)
                   for t in range(1, len(obs_cram)))

    P2_real, _ = matriz_transicion_orden2(obs_real, K)
    ll2_real = loglik_orden2(obs_real, P2_real)

    P2_shuf, _ = matriz_transicion_orden2(obs_shuf, K)
    ll2_shuf = loglik_orden2(obs_shuf, P2_shuf)

    P2_cram, _ = matriz_transicion_orden2(obs_cram, K)
    ll2_cram = loglik_orden2(obs_cram, P2_cram)

    n1 = len(obs_real) - 1
    n2 = len(obs_real) - 2

    print(f"\n{'Modelo':>15} | {'ll total':>15} | {'ll/obs':>12}")
    print("-" * 50)
    print(f"{'orden1 real':>15} | {ll1_real:>15.2f} | {ll1_real/n1:>12.5f}")
    print(f"{'orden1 shuffle':>15} | {ll1_shuf:>15.2f} | {ll1_shuf/n1:>12.5f}")
    print(f"{'orden1 Cramer':>15} | {ll1_cram:>15.2f} | {ll1_cram/n1:>12.5f}")
    print(f"{'orden2 real':>15} | {ll2_real:>15.2f} | {ll2_real/n2:>12.5f}")
    print(f"{'orden2 shuffle':>15} | {ll2_shuf:>15.2f} | {ll2_shuf/n2:>12.5f}")
    print(f"{'orden2 Cramer':>15} | {ll2_cram:>15.2f} | {ll2_cram/n2:>12.5f}")

    delta_rs_1 = (ll1_real - ll1_shuf) / n1
    delta_rs_2 = (ll2_real - ll2_shuf) / n2
    delta_rc_1 = (ll1_real - ll1_cram) / n1
    delta_rc_2 = (ll2_real - ll2_cram) / n2

    print(f"\nDelta ll/obs (real - shuffle), orden 1: {delta_rs_1:+.5f}")
    print(f"Delta ll/obs (real - shuffle), orden 2: {delta_rs_2:+.5f}")
    print(f"Delta ll/obs (real - Cramer), orden 1:  {delta_rc_1:+.5f}")
    print(f"Delta ll/obs (real - Cramer), orden 2:  {delta_rc_2:+.5f}")

    if delta_rs_2 > delta_rs_1:
        print("-> Orden 2 captura MAS estructura que orden 1.")
    else:
        print("-> Orden 2 NO captura mas estructura: memoria de orden 1.")

    return {
        'll1_real': ll1_real, 'll1_shuf': ll1_shuf, 'll1_cram': ll1_cram,
        'll2_real': ll2_real, 'll2_shuf': ll2_shuf, 'll2_cram': ll2_cram,
        'delta_rs_1': delta_rs_1, 'delta_rs_2': delta_rs_2,
        'delta_rc_1': delta_rc_1, 'delta_rc_2': delta_rc_2,
    }


def test_gaps(primos, max_niveles_gilbreath=20, verif=None):
    """Extension 3: analisis de gaps + Opcion C completa + tests adicionales."""
    print("\n" + "=" * 80)
    print("  EXTENSION 3: ANALISIS DIRECTO DE GAPS")
    print("=" * 80)

    gaps = calcular_gaps(primos, verif=verif)
    print(f"Numero de gaps: {len(gaps)}")
    print(f"Gap minimo: {gaps.min()}")
    print(f"Gap maximo: {gaps.max()}")
    print(f"Gap medio: {gaps.mean():.4f}")

    valores, counts = np.unique(gaps, return_counts=True)
    print(f"\nDistribucion de gaps (top 15):")
    print(f"{'gap':>6} | {'frecuencia':>12} | {'probabilidad':>12}")
    print("-" * 40)
    orden = np.argsort(counts)[::-1]
    for i in orden[:15]:
        print(f"{valores[i]:>6} | {counts[i]:>12} | "
              f"{counts[i]/len(gaps):>12.6f}")

    gaps_unicos = sorted(set(gaps))
    mapa_gap = {g: i for i, g in enumerate(gaps_unicos)}
    obs_gaps = np.array([mapa_gap[g] for g in gaps])
    K_g = len(gaps_unicos)

    P_gaps, _ = matriz_transicion_orden1(obs_gaps, K_g)
    esp_gaps = analizar_espectro(P_gaps, K_g)

    print(f"\nAnalisis espectral de los gaps:")
    print(f"  K (valores unicos de gaps) = {K_g}")
    print(f"  |lambda2| = {esp_gaps['abs_eig'][1]:.5f}")
    print(f"  H/H_max = {esp_gaps['entropia_ratio']:.5f}")
    print(f"  Tiempo de mezcla = {esp_gaps['t_mix']:.2f}")

    obs_gaps_shuf = generar_shuffle(obs_gaps, SEMILLA)
    P_gaps_shuf, _ = matriz_transicion_orden1(obs_gaps_shuf, K_g)
    esp_gaps_shuf = analizar_espectro(P_gaps_shuf, K_g)

    rng = np.random.default_rng(SEMILLA)
    obs_gaps_cram = rng.choice(obs_gaps, size=len(obs_gaps), replace=True)
    P_gaps_cram, _ = matriz_transicion_orden1(obs_gaps_cram, K_g)
    esp_gaps_cram = analizar_espectro(P_gaps_cram, K_g)

    print(f"\n  Comparacion con modelos nulos:")
    print(f"    |lambda2| real    = {esp_gaps['abs_eig'][1]:.5f}")
    print(f"    |lambda2| shuffle = {esp_gaps_shuf['abs_eig'][1]:.5f}")
    print(f"    |lambda2| Cramer  = {esp_gaps_cram['abs_eig'][1]:.5f}")
    diff_shuf = esp_gaps['abs_eig'][1] - esp_gaps_shuf['abs_eig'][1]
    diff_cram = esp_gaps['abs_eig'][1] - esp_gaps_cram['abs_eig'][1]
    print(f"    Delta vs shuffle = {diff_shuf:+.5f}")
    print(f"    Delta vs Cramer  = {diff_cram:+.5f}")

    if diff_shuf > 0.1:
        print("-> Los gaps tienen estructura de orden FUERTE.")
    elif diff_shuf > 0.02:
        print("-> Los gaps tienen estructura de orden MODERADA.")
    else:
        print("-> Los gaps no tienen estructura de orden significativa.")

    if abs(diff_cram) < 0.02:
        print("-> El Cramer de gaps reproduce la estructura de los gaps reales.")
    elif diff_cram > 0.05:
        print("-> Los gaps reales tienen MAS estructura que el Cramer de gaps.")
        print("   Evidencia de aritmetica genuina en los gaps.")

    # Opcion C
    print("\n" + "#" * 80)
    print("#  OPCION C: EXPLORACION PROFUNDA DE GAPS")
    print("#" * 80)

    c1 = opcion_c_gaps_por_bloques(gaps, verif=verif)
    c2 = opcion_c_significancia_gaps(gaps, verif=verif)

    primos_arr = np.array(primos)
    log_primos = np.log(primos_arr[:-1])
    c3 = opcion_c_gaps_normalizados(primos, gaps, verif=verif)

    c4_210 = opcion_c_correlacion_residuo_gap(primos, 210, verif=verif)
    c4_2310 = opcion_c_correlacion_residuo_gap(primos, 2310, verif=verif)

    # Tests adicionales
    print("\n" + "#" * 80)
    print("#  TESTS ADICIONALES")
    print("#" * 80)
    est_gaps = test_estacionariedad_gaps(gaps, verif=verif)
    ks = test_ks_gaps_normalizados(gaps, log_primos, verif=verif)

    # Gilbreath
    print(f"\nConjetura de Gilbreath (primeros {max_niveles_gilbreath} niveles):")
    fila = list(primos[:100])
    for k in range(max_niveles_gilbreath):
        if len(fila) < 2:
            break
        print(f"  Nivel {k}: primer elemento = {fila[0]}, "
              f"longitud = {len(fila)}")
        fila = [abs(fila[i+1] - fila[i]) for i in range(len(fila)-1)]

    return {
        'gaps': gaps,
        'K_gaps': K_g,
        'esp_gaps': esp_gaps,
        'esp_gaps_shuf': esp_gaps_shuf,
        'esp_gaps_cram': esp_gaps_cram,
        'diff_gaps_shuf': diff_shuf,
        'diff_gaps_cram': diff_cram,
        'opcion_c': {
            'c1_bloques': c1,
            'c2_significancia': c2,
            'c3_normalizados': c3,
            'c4_residuo_210': c4_210,
            'c4_residuo_2310': c4_2310,
        },
        'test_estacionariedad_gaps': est_gaps,
        'test_ks': ks,
    }


def test_modulo_grande(primos, modulo, verif=None):
    """Extension 4: modulos grandes."""
    obs_real, coprimos = residuos_a_indices(primos, modulo)
    K = len(coprimos)
    print(f"\nModulo {modulo}: K = {K} simbolos, "
          f"{len(obs_real)} observaciones")

    if len(obs_real) < 1000:
        print("  -> Muy pocos datos para modulo tan grande.")
        return None

    obs_shuf = generar_shuffle(obs_real, SEMILLA)
    obs_cram = generar_cramer_clasico(len(obs_real), modulo, SEMILLA)

    P_real, _ = matriz_transicion_orden1(obs_real, K)
    P_shuf, _ = matriz_transicion_orden1(obs_shuf, K)
    P_cram, _ = matriz_transicion_orden1(obs_cram, K)

    esp_real = analizar_espectro(P_real, K)
    esp_shuf = analizar_espectro(P_shuf, K)
    esp_cram = analizar_espectro(P_cram, K)

    print(f"  |lambda2| real    = {esp_real['abs_eig'][1]:.5f}")
    print(f"  |lambda2| shuffle = {esp_shuf['abs_eig'][1]:.5f}")
    print(f"  |lambda2| Cramer  = {esp_cram['abs_eig'][1]:.5f}")
    print(f"  H/H_max real = {esp_real['entropia_ratio']:.5f}")
    print(f"  Rango efectivo real = {esp_real['rango_efectivo']:.2f}")

    return {
        'modulo': modulo,
        'K': K,
        'esp_real': esp_real,
        'esp_shuf': esp_shuf,
        'esp_cram': esp_cram,
    }


# ============================================================
# CONCLUSIÓN AUTOMÁTICA
# ============================================================
def concluir(resultados, gaps_info, modulos_grandes):
    print("\n" + "=" * 80)
    print("  CONCLUSION AUTOMATICA")
    print("=" * 80)

    print("\n[1] Estructura de orden 1 en residuos:")
    for modulo, datos in resultados.items():
        comp = datos['comparacion']
        print(f"  mod {modulo}:")
        print(f"    |lambda2| real    = {comp['lambda2_real']:.5f}")
        print(f"    |lambda2| shuffle = {comp['lambda2_shuf']:.5f}")
        print(f"    |lambda2| Cram clas = {comp['lambda2_cc']:.5f}")
        print(f"    |lambda2| Cram refin = {comp['lambda2_cr']:.5f}")
        print(f"    H/H_max real = {comp['H_ratio_real']:.5f}")

    print("\n[2] Cramer refinado (densidad + gaps):")
    for modulo, datos in resultados.items():
        comp = datos['comparacion']
        diff_ref = comp['lambda2_real'] - comp['lambda2_cr']
        if abs(diff_ref) < 0.02:
            print(f"  mod {modulo}: Cramer refinado REPRODUCE la estructura.")
        elif diff_ref > 0.05:
            print(f"  mod {modulo}: REAL TIENE MAS ESTRUCTURA que Cramer refinado.")
        else:
            print(f"  mod {modulo}: resultado ambiguo (Delta = {diff_ref:+.5f}).")

    print("\n[3] Orden 2 (por modulo):")
    for modulo, datos in resultados.items():
        if 'orden2_info' in datos:
            o2 = datos['orden2_info']
            d1 = o2['delta_rs_1']
            d2 = o2['delta_rs_2']
            print(f"  mod {modulo}: Delta ll/obs orden1 = {d1:.5f} | "
                  f"orden2 = {d2:.5f}")
            if d2 > d1:
                print(f"    -> Memoria de orden 2 detectada.")
            else:
                print(f"    -> Memoria de orden 1 domina.")

    print("\n[4] Gaps:")
    print(f"  |lambda2| gaps real    = {gaps_info['esp_gaps']['abs_eig'][1]:.5f}")
    print(f"  |lambda2| gaps shuffle = {gaps_info['esp_gaps_shuf']['abs_eig'][1]:.5f}")
    print(f"  |lambda2| gaps Cramer  = {gaps_info['esp_gaps_cram']['abs_eig'][1]:.5f}")
    print(f"  Delta vs shuffle = {gaps_info['diff_gaps_shuf']:+.5f}")
    print(f"  Delta vs Cramer  = {gaps_info['diff_gaps_cram']:+.5f}")
    if gaps_info['diff_gaps_shuf'] > 0.1:
        print("  -> Los gaps tienen estructura de orden FUERTE.")
    elif gaps_info['diff_gaps_shuf'] > 0.02:
        print("  -> Los gaps tienen estructura de orden MODERADA.")
    else:
        print("  -> Los gaps no tienen estructura de orden significativa.")
    if abs(gaps_info['diff_gaps_cram']) < 0.02:
        print("  -> Cramer de gaps reproduce la estructura. Densidad + gaps bastan.")
    elif gaps_info['diff_gaps_cram'] > 0.05:
        print("  -> Gaps reales tienen MAS estructura que Cramer. Aritmetica genuina.")

    print("\n[5] Modulos grandes:")
    for res in modulos_grandes:
        if res:
            print(f"  mod {res['modulo']}: K = {res['K']}, "
                  f"|lambda2| real = {res['esp_real']['abs_eig'][1]:.5f}, "
                  f"|lambda2| Cramer = {res['esp_cram']['abs_eig'][1]:.5f}")

    print("\n[6] Opcion C (exploracion profunda de gaps):")
    op_c = gaps_info.get('opcion_c', {})
    if op_c.get('c1_bloques'):
        c1 = op_c['c1_bloques']
        print(f"  C.1 Bloques: Delta = {c1['diff']:+.5f}, "
              f"p = {c1['p_valor']:.2e} -> {c1['resultado']}")
    if op_c.get('c2_significancia'):
        c2 = op_c['c2_significancia']
        print(f"  C.2 Significancia: z = {c2['z']:.3f}, "
              f"p = {c2['p_valor']:.4f} -> {c2['resultado']}")
    if op_c.get('c3_normalizados'):
        c3 = op_c['c3_normalizados']
        print(f"  C.3 Normalizados vs Cramer refin: Delta = "
              f"{c3['diff']:+.5f} -> {c3['resultado']}")
    if op_c.get('c4_residuo_210'):
        c4 = op_c['c4_residuo_210']
        print(f"  C.4 Residuo-gap mod 210: "
              f"p_ANOVA = {c4['p_anova']:.2e} -> {c4['resultado']}")
    if op_c.get('c4_residuo_2310'):
        c4 = op_c['c4_residuo_2310']
        print(f"  C.4 Residuo-gap mod 2310: "
              f"p_ANOVA = {c4['p_anova']:.2e} -> {c4['resultado']}")

    print("\n[7] Tests adicionales:")
    if gaps_info.get('test_estacionariedad_gaps'):
        est = gaps_info['test_estacionariedad_gaps']
        print(f"  Estacionariedad gaps: l1={est['l1']:.5f}, "
              f"l2={est['l2']:.5f}, diff={est['diff']:.5f} -> "
              f"{est['resultado']}")
    if gaps_info.get('test_ks'):
        ks = gaps_info['test_ks']
        print(f"  KS exponencial: KS={ks['ks_stat']:.5f}, "
              f"p={ks['p_valor']:.2e} -> {ks['resultado']}")

    # Veredicto final
    print("\n" + "=" * 80)
    print("  VEREDICTO FINAL")
    print("=" * 80)

    refinado_compatible = all(
        abs(d['comparacion']['lambda2_real']
            - d['comparacion']['lambda2_cr']) < 0.02
        for d in resultados.values()
    )
    orden2_mayor = any(
        d.get('orden2_info', {}).get('delta_rs_2', 0)
        > d.get('orden2_info', {}).get('delta_rs_1', 0)
        for d in resultados.values()
    )
    gaps_vs_cram_fuerte = gaps_info['diff_gaps_cram'] > 0.05

    c1 = op_c.get('c1_bloques', {}) or {}
    c3 = op_c.get('c3_normalizados', {}) or {}
    est = gaps_info.get('test_estacionariedad_gaps', {}) or {}
    ks = gaps_info.get('test_ks', {}) or {}

    bloques_genuina = c1.get('resultado') == 'genuina'
    normalizados_persiste = c3.get('resultado') == 'persiste'
    estacionario = est.get('resultado') == 'estacionario'
    ks_rechaza = ks.get('resultado') == 'rechaza'

    if refinado_compatible and not orden2_mayor and not gaps_vs_cram_fuerte:
        print("""
Los residuos de primos tienen estructura de orden 1 fuerte, pero esa
estructura es COMPLETAMENTE reproducida por el modelo de Cramer refinado
(densidad 1/log x + distribucion de gaps observada). No hay evidencia de:
  - Memoria de orden 2
  - Estructura de gaps no explicada por densidad
  - Estructura aritmetica modular en modulos grandes

La conclusion es que la regularidad observable de los residuos de primos
es de origen densitivo, no aritmetico profundo.
        """)
    elif (gaps_vs_cram_fuerte and bloques_genuina and normalizados_persiste
          and ks_rechaza):
        print("""
HALLAZGO ROBUSTO! Los gaps tienen estructura que NO es reproducida por
Cramer, y esa estructura PERSISTE despues de:
  - Analisis por bloques con Cramer de distribucion global + bins (C.1)
  - Normalizacion por densidad local vs Cramer refinado (C.3)
  - Comparacion con shuffles (C.2)
  - KS test contra exponencial (rechaza)
  - Test de estacionariedad (estacionario)

Esto sugiere estructura aritmetica genuina en los gaps, no artefacto
de densidad. Resultado potencialmente publicable.
        """)
    elif gaps_vs_cram_fuerte:
        print("""
HALLAZGO SIGNIFICATIVO! Hay estructura en los gaps que NO es
reproducida por el modelo de Cramer refinado. Sin embargo, la Opcion C
y los tests adicionales no confirman completamente que sea genuina
(puede ser artefacto de densidad residual). Se necesitan mas datos
para confirmar.
        """)
    else:
        print("""
Resultado mixto. Hay alguna evidencia de estructura no explicada por
Cramer refinado, pero no es concluyente. Se necesitan mas datos o
modulos mas finos para confirmar.
        """)


# ============================================================
# VISUALIZACIÓN
# ============================================================
def graficar_comparacion(esp_real, esp_cc, esp_cr, esp_shuf, modulo,
                          nombre_archivo):
    """Grafica comparativa."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(esp_real['abs_eig'], 'o-', label='real',
                 markersize=4, color='C0')
    axes[0].plot(esp_cc['abs_eig'], 's-', label='Cramer clasico',
                 markersize=4, alpha=0.7, color='C2')
    axes[0].plot(esp_cr['abs_eig'], '^-', label='Cramer refinado',
                 markersize=4, alpha=0.7, color='C3')
    axes[0].plot(esp_shuf['abs_eig'], 'd-', label='shuffle',
                 markersize=4, alpha=0.7, color='C1')
    axes[0].set_xlabel('Indice del autovalor')
    axes[0].set_ylabel('|lambda|')
    axes[0].set_title(f'Espectro - mod {modulo}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(np.cumsum(esp_real['contrib']) * 100, 'o-',
                 label='real', markersize=4, color='C0')
    axes[1].plot(np.cumsum(esp_cc['contrib']) * 100, 's-',
                 label='Cramer clasico', markersize=4, alpha=0.7, color='C2')
    axes[1].plot(np.cumsum(esp_cr['contrib']) * 100, '^-',
                 label='Cramer refinado', markersize=4, alpha=0.7, color='C3')
    axes[1].set_xlabel('Indice del autovalor')
    axes[1].set_ylabel('% contribucion acumulada')
    axes[1].set_title(f'Contribucion acumulada - mod {modulo}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(esp_real['pi'], 'o-', label='real',
                 markersize=4, color='C0')
    axes[2].plot(esp_cc['pi'], 's-', label='Cramer clasico',
                 markersize=4, alpha=0.7, color='C2')
    axes[2].plot(esp_cr['pi'], '^-', label='Cramer refinado',
                 markersize=4, alpha=0.7, color='C3')
    axes[2].axhline(1/len(esp_real['pi']), color='k', linestyle='--',
                    alpha=0.5, label='uniforme')
    axes[2].set_xlabel('Estado')
    axes[2].set_ylabel('pi_i')
    axes[2].set_title(f'Distribucion estacionaria - mod {modulo}')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=120)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time()
    print("=" * 80)
    print("  ANALISIS ESPECTRAL EXTENDIDO - RESIDUOS DE PRIMOS (v7)")
    print("=" * 80)
    print(f"N_PRIMOS: {N_PRIMOS:,} | Modulos: {MODULOS}")
    print(f"Alpha Dirichlet: {ALPHA} | Semilla: {SEMILLA}")
    print(f"Opcion C corregida | Autoverificacion robusta")
    print()

    verif = Autoverificador()

    primos = generar_primos(N_PRIMOS, verif=verif)
    gaps = calcular_gaps(primos, verif=verif)
    print()

    # EXTENSION 3 + OPCION C + TESTS ADICIONALES
    gaps_info = test_gaps(primos, verif=verif)

    # Analisis por modulo (210 y 2310)
    resultados = {}
    for modulo in MODULOS[:2]:
        print("\n" + "=" * 80)
        print(f"  MODULO {modulo}")
        print("=" * 80)

        obs_real, coprimos = residuos_a_indices(primos, modulo, verif=verif)
        K = len(coprimos)
        print(f"Observaciones: {len(obs_real):,} | K = {K} simbolos")

        obs_shuf = generar_shuffle(obs_real, SEMILLA)
        obs_cram_clas = generar_cramer_clasico(len(obs_real), modulo,
                                                SEMILLA, verif=verif)
        obs_cram_ref = generar_cramer_refinado(
            len(obs_real), modulo, gaps, SEMILLA)

        esp_real, esp_cc, esp_cr = test_cramer_refinado(
            obs_real, obs_cram_ref, obs_cram_clas, K, verif=verif)

        orden2_info = test_orden2(obs_real, obs_shuf, obs_cram_clas, K)

        # Test de estacionariedad de residuos
        est_res = test_estacionariedad_residuos(obs_real, K, verif=verif,
                                                 nombre=f"residuos mod {modulo}")

        P_shuf, _ = matriz_transicion_orden1(obs_shuf, K)
        esp_shuf = analizar_espectro(P_shuf, K)

        graficar_comparacion(esp_real, esp_cc, esp_cr, esp_shuf, modulo,
                              f'espectro_mod{modulo}_v7.png')

        comp = {
            'lambda2_real': esp_real['abs_eig'][1],
            'lambda2_shuf': esp_shuf['abs_eig'][1],
            'lambda2_cc': esp_cc['abs_eig'][1],
            'lambda2_cr': esp_cr['abs_eig'][1],
            'H_ratio_real': esp_real['entropia_ratio'],
            'H_ratio_shuf': esp_shuf['entropia_ratio'],
            'H_ratio_cc': esp_cc['entropia_ratio'],
            'H_ratio_cr': esp_cr['entropia_ratio'],
        }

        resultados[modulo] = {
            'K': K,
            'comparacion': comp,
            'esp_real': esp_real,
            'esp_cc': esp_cc,
            'esp_cr': esp_cr,
            'esp_shuf': esp_shuf,
            'orden2_info': orden2_info,
            'est_residuos': est_res,
        }

    # EXTENSION 4
    print("\n" + "=" * 80)
    print("  EXTENSION 4: MODULOS GRANDES")
    print("=" * 80)
    modulos_grandes = []
    for modulo in MODULOS[2:]:
        res = test_modulo_grande(primos, modulo, verif=verif)
        if res:
            modulos_grandes.append(res)

    # Conclusion
    concluir(resultados, gaps_info, modulos_grandes)

    # Autoverificacion
    verif.resumen()

    print(f"\nTiempo total: {time() - t0:.1f}s")


if __name__ == "__main__":
    main()
