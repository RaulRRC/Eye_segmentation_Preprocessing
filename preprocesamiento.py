"""
BASE COMPARTIDA: carga de datos y preprocesamiento.

Todos los métodos parten de aquí. Si alguien cambia algo en este archivo,
avisar al grupo, porque afecta los resultados de todos.

Uso típico:
    from comun.preprocesamiento import listar_casos, cargar_caso, preprocesar
    casos = listar_casos("HRF")
    img, fov = cargar_caso(casos[0], escala=0.5)
    pre = preprocesar(img, fov)          # usa CONFIG_BASE
    pre.final                            # imagen lista para segmentar
    pre.pasos["verde"]                   # cualquier paso intermedio
"""

import os
import glob
from dataclasses import dataclass, field

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# CONFIGURACIÓN BASE (acordada por el grupo)
# ----------------------------------------------------------------------------
CONFIG_BASE = {
    "canal": "verde",            # "verde" | "luminancia" | "verde_rojo"
    "rellenar_fov": True,        # rellena el exterior del FOV para evitar el anillo
    "erosion_fov_px": 5,
    "iluminacion": None,         # None | "resta_fondo" | "division_fondo"
    "kernel_fondo": 51,          # tamaño (impar) del filtro de mediana para estimar fondo
    "clahe": True,
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "ruido": "gauss",            # None | "gauss" | "mediana" | "bilateral"
    "ruido_param": 1.0,          # sigma (gauss), tamaño (mediana), d (bilateral)
    "invertir": False,           # True -> vasos claros sobre fondo oscuro
}


# ----------------------------------------------------------------------------
# DATOS
# ----------------------------------------------------------------------------
@dataclass
class Caso:
    nombre: str          # ej. "01_dr"
    ruta_img: str
    ruta_fov: str
    ruta_gt: str

    @property
    def clase(self):
        return {"h": "sano", "dr": "retinopatía", "g": "glaucoma"}.get(
            self.nombre.split("_")[-1], "?")


def listar_casos(base_dir):
    """Empareja images/, mask/ y manual1/ por nombre (01_dr, 01_g, 01_h...)."""
    casos = []
    for ruta_img in sorted(glob.glob(os.path.join(base_dir, "images", "*"))):
        nombre = os.path.splitext(os.path.basename(ruta_img))[0]
        ruta_fov = os.path.join(base_dir, "mask", f"{nombre}_mask.tif")
        ruta_gt = os.path.join(base_dir, "manual1", f"{nombre}.tif")
        if os.path.exists(ruta_fov) and os.path.exists(ruta_gt):
            casos.append(Caso(nombre, ruta_img, ruta_fov, ruta_gt))
        else:
            print(f"[aviso] faltan archivos para {nombre}, se omite")
    return casos


def leer_binaria(ruta):
    m = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if m is None:  # algunos .tif no los abre OpenCV
        from skimage.io import imread
        m = imread(ruta)
        if m.ndim == 3:
            m = m[..., 0]
    return (m > 127).astype(np.uint8)


def redimensionar(img, escala, interp=cv2.INTER_AREA):
    if escala == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * escala), int(h * escala)), interpolation=interp)


def cargar_caso(caso, escala=0.5, cache_dir=".cache"):
    """Devuelve (img_bgr, fov) ya reescalados. Guarda una caché en disco
    porque leer los JPG de 3504x2336 una y otra vez es lento al iterar."""
    ruta_cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        ruta_cache = os.path.join(cache_dir, f"{caso.nombre}_{escala}.npz")
        if os.path.exists(ruta_cache):
            d = np.load(ruta_cache)
            return d["img"], d["fov"]

    img = cv2.imread(caso.ruta_img, cv2.IMREAD_COLOR)
    fov = leer_binaria(caso.ruta_fov)
    img = redimensionar(img, escala)
    fov = (redimensionar(fov * 255, escala) > 127).astype(np.uint8)

    if ruta_cache:
        np.savez_compressed(ruta_cache, img=img, fov=fov)
    return img, fov


def cargar_gt(caso):
    """Ground truth a resolución ORIGINAL. Solo para evaluar, nunca para segmentar."""
    return leer_binaria(caso.ruta_gt)


# ----------------------------------------------------------------------------
# PASOS DE PREPROCESAMIENTO (cada uno se puede usar por separado)
# ----------------------------------------------------------------------------
def erosionar_fov(fov, px=5):
    if px <= 0:
        return fov
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(fov, k)


def extraer_canal(img_bgr, canal="verde"):
    """El canal verde tiene el mayor contraste vaso/fondo en fondo de ojo:
    el rojo está saturado y el azul es muy ruidoso."""
    if canal == "verde":
        return img_bgr[:, :, 1].copy()
    if canal == "luminancia":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    if canal == "verde_rojo":  # combinación ponderada, útil en imágenes oscuras
        g = img_bgr[:, :, 1].astype(np.float32)
        r = img_bgr[:, :, 2].astype(np.float32)
        return np.clip(0.75 * g + 0.25 * r, 0, 255).astype(np.uint8)
    raise ValueError(f"canal desconocido: {canal}")


def rellenar_fuera_fov(gris, fov, iteraciones=None):
    """Extiende los valores del borde del FOV hacia afuera, para que los filtros
    no detecten el salto ojo/fondo negro como si fuera un vaso."""
    out = gris.astype(np.float32).copy()
    out[fov == 0] = out[fov == 1].mean()
    # Suaviza la transición en una franja cercana al borde
    borde = cv2.dilate(fov, np.ones((31, 31), np.uint8)) - fov
    suav = cv2.blur(out, (31, 31))
    out[borde == 1] = suav[borde == 1]
    return np.clip(out, 0, 255).astype(np.uint8)


def corregir_iluminacion(gris, metodo="resta_fondo", kernel=51):
    """Estima el fondo con una mediana grande (los vasos desaparecen) y lo
    elimina para compensar iluminación no uniforme (centro brillante, bordes oscuros)."""
    if metodo is None:
        return gris
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    fondo = cv2.medianBlur(gris, kernel).astype(np.float32)
    g = gris.astype(np.float32)
    if metodo == "resta_fondo":
        r = g - fondo
    elif metodo == "division_fondo":
        r = g / (fondo + 1.0)
    else:
        raise ValueError(f"método de iluminación desconocido: {metodo}")
    return cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def aplicar_clahe(gris, clip=2.0, tile=8):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gris)


def reducir_ruido(gris, metodo="gauss", param=1.0):
    if metodo is None:
        return gris
    if metodo == "gauss":
        return cv2.GaussianBlur(gris, (0, 0), float(param))
    if metodo == "mediana":
        k = int(param)
        k = k if k % 2 == 1 else k + 1
        return cv2.medianBlur(gris, max(k, 3))
    if metodo == "bilateral":
        return cv2.bilateralFilter(gris, int(param) if param >= 3 else 5, 30, 5)
    raise ValueError(f"método de ruido desconocido: {metodo}")


# ----------------------------------------------------------------------------
# PIPELINE DE PREPROCESAMIENTO COMPLETO
# ----------------------------------------------------------------------------
@dataclass
class Preprocesado:
    final: np.ndarray                 # imagen gris uint8 lista para segmentar
    fov: np.ndarray                   # FOV (0/1) a la escala de trabajo
    fov_erosionado: np.ndarray        # FOV recortado para evitar el borde
    img_bgr: np.ndarray               # imagen original a la escala de trabajo
    config: dict
    pasos: dict = field(default_factory=dict)   # intermedios, para visualizar


def preprocesar(img_bgr, fov, config=None):
    """Aplica los pasos en orden: canal -> relleno FOV -> iluminación ->
    CLAHE -> ruido -> (inversión). Pasa solo las claves que quieras cambiar:
        preprocesar(img, fov, {"clahe_clip": 3.0})
    """
    cfg = dict(CONFIG_BASE)
    if config:
        desconocidas = set(config) - set(CONFIG_BASE)
        if desconocidas:
            raise KeyError(f"claves de config desconocidas: {desconocidas}")
        cfg.update(config)

    pasos = {}
    g = extraer_canal(img_bgr, cfg["canal"]);                        pasos["canal"] = g
    if cfg["rellenar_fov"]:
        g = rellenar_fuera_fov(g, fov);                              pasos["relleno_fov"] = g
    if cfg["iluminacion"]:
        g = corregir_iluminacion(g, cfg["iluminacion"], cfg["kernel_fondo"])
        pasos["iluminacion"] = g
    if cfg["clahe"]:
        g = aplicar_clahe(g, cfg["clahe_clip"], cfg["clahe_tile"]);  pasos["clahe"] = g
    if cfg["ruido"]:
        g = reducir_ruido(g, cfg["ruido"], cfg["ruido_param"]);      pasos["ruido"] = g
    if cfg["invertir"]:
        g = 255 - g;                                                 pasos["invertida"] = g

    return Preprocesado(final=g, fov=fov,
                        fov_erosionado=erosionar_fov(fov, cfg["erosion_fov_px"]),
                        img_bgr=img_bgr, config=cfg, pasos=pasos)


def visualizar_pasos(pre, ruta_salida=None, titulo=""):
    """Figura con cada paso del preprocesamiento (útil para el video)."""
    import matplotlib.pyplot as plt
    paneles = [("original", cv2.cvtColor(pre.img_bgr, cv2.COLOR_BGR2RGB))]
    paneles += list(pre.pasos.items())
    fig, ax = plt.subplots(1, len(paneles), figsize=(4.5 * len(paneles), 4))
    for a, (nombre, im) in zip(np.atleast_1d(ax), paneles):
        a.imshow(im, cmap=None if im.ndim == 3 else "gray")
        a.set_title(nombre)
        a.axis("off")
    if titulo:
        fig.suptitle(titulo)
    plt.tight_layout()
    if ruta_salida:
        plt.savefig(ruta_salida, dpi=90)
        plt.close()
    else:
        plt.show()
