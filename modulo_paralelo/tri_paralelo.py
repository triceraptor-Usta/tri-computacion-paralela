#!/usr/bin/env python3
"""Preprocesamiento paralelo de tomas CSI del proyecto Tri y medición de speedup y eficiencia.

Cada toma (un CSV comprimido con la CSI de los 4 nodos ESP32-S3) se procesa de forma independiente:
lectura del CSV, alineación de los nodos por rx_seq, normalización de amplitud y fase diferencial
(`cavityvoxel_former.data.tramas.tramas_de`). Al final se resume la toma en un perfil medio por nodo
(4 x 52), que es lo que consume el reconocedor por vecino más cercano.

El script ejecuta el mismo lote fijo de tomas con:
  * una versión secuencial (línea base, T1),
  * multiprocessing.Pool con p procesos,
  * ThreadPoolExecutor con p hilos (para mostrar el efecto del GIL),
y reporta tiempo de pared, speedup S = T1 / Tp y eficiencia E = S / p. También comprueba que los perfiles
paralelos son idénticos a los secuenciales.

Uso (desde cualquier carpeta):
  python tri_paralelo.py --datos D:/Proyectos/2026/Tri/tri_datos_v2 \
                         --repo  D:/Proyectos/2026/Tri/triceraptor-Usta/cavityvoxel-former \
                         --tomas 96 --workers 1 2 4 8 12 16 --reps 3 --salida resultados
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_tramas = None  # módulo tramas, importado en cada proceso por _init


def _init(repo: str) -> None:
    """Inicializador de cada proceso: hace importable el paquete del repositorio de Tri."""
    global _tramas
    src = str(Path(repo) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from cavityvoxel_former.data import tramas  # noqa: E402
    _tramas = tramas


def procesa_toma(ruta: str) -> tuple[str, np.ndarray | None, str]:
    """Una toma -> (nombre, perfil medio por nodo (4, 52) o None, error)."""
    try:
        d = _tramas.tramas_de(Path(ruta))
        perfil = np.nanmean(d["a"], axis=0).astype(np.float64)  # (4, 52)
        return Path(ruta).name, perfil, ""
    except (ValueError, OSError, EOFError) as e:
        return Path(ruta).name, None, str(e)


def selecciona_tomas(datos: Path, n: int) -> list[str]:
    """Lote fijo y reproducible: las n primeras tomas de la malla en orden alfabético."""
    todas = sorted((datos / "raw" / "fase_3_grid_completo").glob("grid_*.csv.gz"))
    if len(todas) < n:
        raise SystemExit("Se pidieron %d tomas y la carpeta tiene menos." % n)
    return [str(p) for p in todas[:n]]


def corre_secuencial(rutas: list[str]) -> list:
    return [procesa_toma(r) for r in rutas]


def corre_procesos(rutas: list[str], p: int, repo: str) -> list:
    with Pool(processes=p, initializer=_init, initargs=(repo,)) as pool:
        return pool.map(procesa_toma, rutas, chunksize=1)


def corre_hilos(rutas: list[str], p: int) -> list:
    with ThreadPoolExecutor(max_workers=p) as ex:
        return list(ex.map(procesa_toma, rutas))


def cronometra(fn, *args) -> tuple[float, list]:
    t0 = time.perf_counter()
    r = fn(*args)
    return time.perf_counter() - t0, r


def iguales(a: list, b: list) -> bool:
    da = {n: v for n, v, _ in a}
    db = {n: v for n, v, _ in b}
    if da.keys() != db.keys():
        return False
    for k in da:
        x, y = da[k], db[k]
        if (x is None) != (y is None):
            return False
        if x is not None and not np.array_equal(x, y, equal_nan=True):
            return False
    return True


def grafica(filas: list[dict], salida: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ps = sorted({f["p"] for f in filas})
    ax1.plot(ps, ps, ls=":", color="0.5", label="Ideal")
    for modo, color, marca, etiqueta in (("procesos", "#1f5fa8", "o", "multiprocessing"),
                                          ("hilos", "#c0392b", "s", "threading")):
        fs = sorted((f for f in filas if f["modo"] == modo), key=lambda f: f["p"])
        ax1.plot([f["p"] for f in fs], [f["speedup"] for f in fs], marker=marca, color=color, label=etiqueta)
        ax2.plot([f["p"] for f in fs], [f["eficiencia"] for f in fs], marker=marca, color=color, label=etiqueta)
    ax1.set_xlabel("Trabajadores $p$")
    ax1.set_ylabel("Speedup $S_p$")
    ax2.set_xlabel("Trabajadores $p$")
    ax2.set_ylabel("Eficiencia $E_p$")
    ax2.set_ylim(0, 1.1)
    for ax in (ax1, ax2):
        ax.set_xticks(ps)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(salida / "speedup_eficiencia.pdf")
    fig.savefig(salida / "speedup_eficiencia.png", dpi=200)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datos", type=Path, required=True, help="carpeta con raw/<fase>/*.csv.gz")
    ap.add_argument("--repo", type=Path, required=True, help="repositorio cavityvoxel-former")
    ap.add_argument("--tomas", type=int, default=96)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sin-hilos", action="store_true", help="no medir la versión con threading")
    ap.add_argument("--salida", type=Path, default=Path("resultados"))
    a = ap.parse_args(argv)
    for f in (sys.stdout, sys.stderr):
        if hasattr(f, "reconfigure"):
            f.reconfigure(encoding="utf-8", errors="replace")
    a.salida.mkdir(parents=True, exist_ok=True)
    repo = str(a.repo.resolve())
    _init(repo)

    rutas = selecciona_tomas(a.datos, a.tomas)
    print("Lote: %d tomas | CPU lógicas: %d | repeticiones: %d" % (len(rutas), os.cpu_count(), a.reps), flush=True)

    tiempos_seq, ref = [], None
    for _ in range(a.reps):
        t, ref = cronometra(corre_secuencial, rutas)
        tiempos_seq.append(t)
    t1 = statistics.median(tiempos_seq)
    print("Secuencial: %.2f s (mediana)" % t1, flush=True)

    filas = []
    modos = [("procesos", lambda p: corre_procesos(rutas, p, repo))]
    if not a.sin_hilos:
        modos.append(("hilos", lambda p: corre_hilos(rutas, p)))
    for modo, fn in modos:
        for p in a.workers:
            ts, ok = [], True
            for _ in range(a.reps):
                t, r = cronometra(fn, p)
                ts.append(t)
                ok &= iguales(ref, r)
            tp = statistics.median(ts)
            s = t1 / tp
            filas.append({"modo": modo, "p": p, "tiempo_s": tp, "tiempo_min_s": min(ts), "tiempo_max_s": max(ts),
                          "speedup": s, "eficiencia": s / p, "resultado_identico": ok})
            print("  %-8s p=%2d  T=%.2f s  S=%.2f  E=%.2f  idéntico=%s" % (modo, p, tp, s, s / p, ok), flush=True)

    with open(a.salida / "rendimiento.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    errores = [(n, e) for n, v, e in ref if v is None]
    meta = {"tomas": len(rutas), "reps": a.reps, "t1_mediana_s": t1, "t1_todas_s": tiempos_seq,
            "tomas_con_error": errores, "cpu": platform.processor(), "cpu_logicas": os.cpu_count(),
            "sistema": platform.platform(), "python": platform.python_version(), "numpy": np.__version__}
    (a.salida / "entorno.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    grafica(filas, a.salida)
    print("Resultados en %s" % a.salida.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
