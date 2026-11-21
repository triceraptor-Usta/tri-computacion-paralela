# Módulo paralelo del proyecto Tri

Avance del proyecto integrador de Computación Paralela y Distribuida.

`tri_paralelo.py` paraleliza el preprocesamiento de las tomas CSI de Tri con `multiprocessing` y lo compara con
una versión secuencial y con una versión con `threading`. Cada toma es un CSV comprimido con la CSI de los cuatro
nodos ESP32-S3; procesarla significa leer el CSV, alinear los nodos por `rx_seq`, normalizar la amplitud, calcular
la fase diferencial y resumir la toma en un perfil medio por nodo (4 × 52). Las tomas son independientes entre sí,
así que el problema se reparte por tomas (paralelismo de datos) con `Pool.map(chunksize=1)`.

El script mide el tiempo de pared de cada configuración (mediana de varias repeticiones), calcula

- speedup: S_p = T_1 / T_p, donde T_1 es la versión secuencial;
- eficiencia: E_p = S_p / p;

y comprueba que los perfiles calculados en paralelo son idénticos, bit a bit, a los secuenciales.

## Requisitos

- Python 3.9 o superior, con `numpy` y `matplotlib`.
- El repositorio [`cavityvoxel-former`](https://github.com/triceraptor-Usta/cavityvoxel-former) (de él se usa
  `src/cavityvoxel_former/data/tramas.py`).
- La carpeta de datos de Tri con `raw/fase_3_grid_completo/*.csv.gz`.

## Ejecución

```powershell
python tri_paralelo.py --datos D:/Proyectos/2026/Tri/tri_datos_v2 --repo D:/Proyectos/2026/Tri/triceraptor-Usta/cavityvoxel-former --tomas 64 --workers 1 2 4 8 12 16 --reps 3 --salida resultados
```

El lote es fijo y reproducible: las `--tomas` primeras tomas de la malla en orden alfabético. Con
`--sin-hilos` se omite la comparación con `threading`.

## Salidas (`resultados/`)

| Archivo | Contenido |
|---|---|
| `rendimiento.csv` | modo, p, tiempo mediano, mínimo y máximo, speedup, eficiencia y si el resultado es idéntico |
| `entorno.json` | CPU, sistema operativo, versiones de Python y NumPy, T_1 de cada repetición |
| `speedup_eficiencia.pdf` / `.png` | gráfica de speedup y eficiencia frente al número de trabajadores |

El tiempo de cada configuración paralela incluye crear y cerrar el `Pool`, que en Windows arranca cada proceso
con `spawn`. No se excluye porque forma parte del costo real de paralelizar.
