"""Pipeline 1: Black-Hat multiescala + umbral de Otsu."""
import cv2
import numpy as np
from skimage.filters import threshold_otsu

from comun.postprocesamiento import limpiar

NOMBRE = "p1_blackhat"
DESCRIPCION = "Black-Hat multiescala + Otsu"

CONFIG_PRE = {"ruido": None}   # el Black-Hat ya suaviza; el ruido se trata después

PARAMS = {
    "kernels": [7, 11, 15],    # px a la escala de trabajo (vasos finos -> gruesos)
    "suavizado": 3,
    "factor_otsu": 1.0,        # >1 más estricto, <1 más permisivo
    "area_min": 100,
}


def respuesta(pre, p):
    """Black-Hat = cierre - imagen: resalta lo oscuro más estrecho que el kernel."""
    g = pre.final
    r = np.zeros_like(g, dtype=np.float32)
    for k in p["kernels"]:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        r = np.maximum(r, cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se).astype(np.float32))
    if p["suavizado"]:
        r = cv2.GaussianBlur(r, (p["suavizado"], p["suavizado"]), 0)
    return r


def segmentar(pre, p):
    r = respuesta(pre, p)
    t = threshold_otsu(r[pre.fov_erosionado == 1]) * p["factor_otsu"]
    return limpiar((r > t).astype(np.uint8), pre.fov_erosionado, p["area_min"])
