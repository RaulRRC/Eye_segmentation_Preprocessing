"""Postprocesamiento compartido: utilidades que cualquier método puede usar."""

import cv2
import numpy as np


def aplicar_fov(binaria, fov):
    return (binaria.astype(np.uint8) & fov.astype(np.uint8))


def quitar_componentes_pequenas(binaria, area_min=100):
    """Elimina regiones conexas con menos de area_min píxeles (ruido)."""
    binaria = binaria.astype(np.uint8)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    conservar = np.zeros(n, dtype=np.uint8)
    conservar[1:] = stats[1:, cv2.CC_STAT_AREA] >= area_min
    return conservar[etiquetas]


def rellenar_huecos(binaria, area_max=30):
    """Rellena huecos pequeños dentro de los vasos (p. ej. el reflejo central)."""
    inv = 1 - binaria.astype(np.uint8)
    huecos = quitar_componentes_pequenas(inv, area_max + 1)
    return 1 - huecos


def limpiar(binaria, fov, area_min=100):
    """Limpieza estándar: FOV + quitar ruido pequeño."""
    return quitar_componentes_pequenas(aplicar_fov(binaria, fov), area_min)
