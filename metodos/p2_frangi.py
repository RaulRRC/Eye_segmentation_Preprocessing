"""Pipeline 2: filtro de Frangi (Hessiano) + umbral por histéresis."""
import numpy as np
from skimage.filters import frangi, apply_hysteresis_threshold

from comun.postprocesamiento import limpiar

NOMBRE = "p2_frangi"
DESCRIPCION = "Frangi multiescala + histéresis (crecimiento desde semillas)"

CONFIG_PRE = {}   # usa la base: verde + CLAHE + Gauss

PARAMS = {
    "sigmas": (1, 2, 3, 4, 5),
    "beta": 0.5,
    "pct_bajo": 88,     # percentil dentro del FOV: umbral de crecimiento
    "pct_alto": 95,     # percentil dentro del FOV: semillas
    "area_min": 100,
}


def respuesta(pre, p):
    v = frangi(pre.final.astype(np.float64) / 255.0, sigmas=p["sigmas"],
               beta=p["beta"], black_ridges=not pre.config["invertir"])
    v[pre.fov_erosionado == 0] = 0
    return v / (v.max() + 1e-12)


def segmentar(pre, p):
    v = respuesta(pre, p)
    dentro = v[pre.fov_erosionado == 1]
    bajo, alto = np.percentile(dentro, p["pct_bajo"]), np.percentile(dentro, p["pct_alto"])
    binaria = apply_hysteresis_threshold(v, bajo, alto).astype(np.uint8)
    return limpiar(binaria, pre.fov_erosionado, p["area_min"])
