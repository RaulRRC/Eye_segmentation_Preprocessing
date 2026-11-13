"""Métricas y evaluación común para todos los métodos."""

import os
import time

import cv2
import numpy as np
import pandas as pd

from comun.preprocesamiento import cargar_caso, cargar_gt, preprocesar


def dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    s = pred.sum() + gt.sum()
    return 2.0 * np.logical_and(pred, gt).sum() / s if s else 1.0


def jaccard(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    u = np.logical_or(pred, gt).sum()
    return np.logical_and(pred, gt).sum() / u if u else 1.0


def a_tamano_original(pred, forma):
    h, w = forma
    r = cv2.resize(pred.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    return (r > 127).astype(np.uint8)


def segmentar_caso(metodo, caso, escala, params=None, config_pre=None):
    """Ejecuta un método sobre un caso y devuelve la máscara a tamaño original."""
    img, fov = cargar_caso(caso, escala)
    cfg = dict(getattr(metodo, "CONFIG_PRE", {}) or {})
    cfg.update(config_pre or {})
    pre = preprocesar(img, fov, cfg)
    p = dict(metodo.PARAMS)
    p.update(params or {})
    pred = metodo.segmentar(pre, p)
    gt = cargar_gt(caso)
    return a_tamano_original(pred, gt.shape) & a_tamano_original(fov, gt.shape), gt


def evaluar(metodo, casos, escala=0.5, params=None, config_pre=None,
            dir_mascaras=None, verbose=True):
    filas = []
    if dir_mascaras:
        os.makedirs(dir_mascaras, exist_ok=True)
    for i, caso in enumerate(casos, 1):
        t0 = time.time()
        pred, gt = segmentar_caso(metodo, caso, escala, params, config_pre)
        d, j = dice(pred, gt), jaccard(pred, gt)
        filas.append({"imagen": caso.nombre, "clase": caso.clase, "dice": d, "jaccard": j})
        if dir_mascaras:
            cv2.imwrite(os.path.join(dir_mascaras, f"{caso.nombre}.png"), pred * 255)
        if verbose:
            print(f"  [{i}/{len(casos)}] {caso.nombre}: Dice={d:.4f} IoU={j:.4f} "
                  f"({time.time() - t0:.1f}s)")
    return pd.DataFrame(filas)


def resumen(df):
    r = df.groupby("clase")[["dice", "jaccard"]].mean()
    r.loc["TOTAL media"] = df[["dice", "jaccard"]].mean()
    r.loc["TOTAL desv"] = df[["dice", "jaccard"]].std()
    return r


def barrido(metodo, casos, clave, valores, escala=0.5, en="params"):
    """Varía un parámetro (en="params") o una opción de preprocesamiento
    (en="pre") y devuelve el Dice/IoU medio para cada valor."""
    filas = []
    for v in valores:
        kw = {"params": {clave: v}} if en == "params" else {"config_pre": {clave: v}}
        df = evaluar(metodo, casos, escala, verbose=False, **kw)
        filas.append({clave: v, "dice": df.dice.mean(), "jaccard": df.jaccard.mean()})
        print(f"  {clave}={v}: Dice={filas[-1]['dice']:.4f}")
    return pd.DataFrame(filas)
