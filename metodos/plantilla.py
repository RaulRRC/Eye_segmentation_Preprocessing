"""
PLANTILLA para un método nuevo.
1. Copia este archivo como metodos/<tu_metodo>.py
2. Rellena PARAMS, CONFIG_PRE (opcional) y segmentar()
3. Ejecuta:  python ejecutar.py --metodos <tu_metodo> --solo 5

Reglas del grupo:
- segmentar() recibe un objeto Preprocesado y devuelve una máscara 0/1 (uint8)
  del mismo tamaño que pre.final.
- No leer manual1 (ground truth) dentro del método.
- Si necesitas otro preprocesamiento, cámbialo en CONFIG_PRE, no en comun/.
"""
import numpy as np

from comun.postprocesamiento import limpiar

NOMBRE = "plantilla"
DESCRIPCION = "Umbral global simple (solo como ejemplo)"

# Cambios respecto a comun.preprocesamiento.CONFIG_BASE (vacío = usar la base)
CONFIG_PRE = {}

PARAMS = {
    "percentil": 15,     # los píxeles más oscuros se consideran vaso
    "area_min": 100,
}


def segmentar(pre, p):
    img = pre.final                     # gris uint8, vasos oscuros
    dentro = img[pre.fov_erosionado == 1]
    t = np.percentile(dentro, p["percentil"])
    binaria = (img < t).astype(np.uint8)
    return limpiar(binaria, pre.fov_erosionado, p["area_min"])
