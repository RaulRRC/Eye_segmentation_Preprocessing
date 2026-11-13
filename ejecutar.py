"""
Ejecuta y compara métodos de segmentación.

Ejemplos:
    python ejecutar.py                                  # p1 y p2 en todo HRF
    python ejecutar.py --metodos p2_frangi --solo 5     # iteración rápida
    python ejecutar.py --metodos p1_blackhat mi_metodo --base /content/drive/MyDrive/HRF
    python ejecutar.py --ver-pre                        # solo figuras del preprocesamiento
    python ejecutar.py --metodos p1_blackhat --barrido factor_otsu 0.8 1.0 1.2

En Colab / Jupyter:
    import sys; sys.path.append("/ruta/a/proyecto_vasos")
    y desde ahí importar comun.* y metodos.* normalmente.
"""

import argparse
import importlib
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from comun.preprocesamiento import listar_casos, cargar_caso, preprocesar, visualizar_pasos
from comun.evaluacion import evaluar, resumen, barrido


def casos_representativos(casos):
    """Uno sano, uno con retinopatía y uno con glaucoma, para pruebas rápidas."""
    sel = []
    for clase in ("sano", "retinopatía", "glaucoma"):
        sel += [c for c in casos if c.clase == clase][:1]
    return sel or casos[:3]


def figura_comparacion(caso, metodos, dir_masks, dir_res):
    img = cv2.cvtColor(cv2.imread(caso.ruta_img), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(caso.ruta_gt, cv2.IMREAD_GRAYSCALE)
    paneles = [("Original", img), ("Ground truth", gt)]
    for m in metodos:
        ruta = os.path.join(dir_masks, m, f"{caso.nombre}.png")
        if os.path.exists(ruta):
            paneles.append((m, cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)))
    fig, ax = plt.subplots(1, len(paneles), figsize=(4.5 * len(paneles), 3.5))
    for a, (t, im) in zip(ax, paneles):
        a.imshow(im, cmap=None if im.ndim == 3 else "gray")
        a.set_title(t)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_res, f"comparacion_{caso.nombre}.png"), dpi=90)
    plt.close()


def convertir(v):
    for tipo in (int, float):
        try:
            return tipo(v)
        except ValueError:
            pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HRF")
    ap.add_argument("--metodos", nargs="+", default=["p1_blackhat", "p2_frangi"])
    ap.add_argument("--escala", type=float, default=0.5)
    ap.add_argument("--solo", type=int, default=0, help="usar solo N imágenes")
    ap.add_argument("--salida", default="predicted_masks")
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--ver-pre", action="store_true")
    ap.add_argument("--barrido", nargs="+", metavar=("PARAM", "VALORES"),
                    help="parámetro y valores, en imágenes representativas")
    ap.add_argument("--barrido-pre", action="store_true",
                    help="el parámetro del barrido es de preprocesamiento")
    a = ap.parse_args()

    os.makedirs(a.resultados, exist_ok=True)
    casos = listar_casos(a.base)
    if a.solo:
        casos = casos[:a.solo]
    print(f"{len(casos)} imágenes en {a.base}")
    if not casos:
        return

    metodos = {n: importlib.import_module(f"metodos.{n}") for n in a.metodos}

    if a.ver_pre:
        for c in casos_representativos(casos):
            img, fov = cargar_caso(c, a.escala)
            for n, m in metodos.items():
                pre = preprocesar(img, fov, getattr(m, "CONFIG_PRE", {}))
                visualizar_pasos(pre, os.path.join(a.resultados, f"pre_{n}_{c.nombre}.png"),
                                 f"{n} - {c.nombre}")
        print("Figuras de preprocesamiento guardadas en", a.resultados)
        return

    if a.barrido:
        clave, valores = a.barrido[0], [convertir(v) for v in a.barrido[1:]]
        sub = casos_representativos(casos)
        for n, m in metodos.items():
            print(f"\nBarrido {n}: {clave}")
            df = barrido(m, sub, clave, valores, a.escala,
                         en="pre" if a.barrido_pre else "params")
            df.to_csv(os.path.join(a.resultados, f"barrido_{n}_{clave}.csv"), index=False)
            plt.figure(figsize=(6, 4))
            plt.plot(df[clave].astype(str), df["dice"], "o-", label="Dice")
            plt.plot(df[clave].astype(str), df["jaccard"], "s--", label="IoU")
            plt.xlabel(clave); plt.title(f"{n}: sensibilidad a {clave}")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(a.resultados, f"barrido_{n}_{clave}.png"), dpi=90)
            plt.close()
        return

    tabla = []
    for n, m in metodos.items():
        print(f"\n=== {n}: {getattr(m, 'DESCRIPCION', '')} ===")
        df = evaluar(m, casos, a.escala, dir_mascaras=os.path.join(a.salida, n))
        df.to_csv(os.path.join(a.resultados, f"metricas_{n}.csv"), index=False)
        r = resumen(df)
        print(r.round(4))
        fila = r.loc["TOTAL media"].rename(n)
        tabla.append(fila)

    comp = pd.DataFrame(tabla)
    comp.to_csv(os.path.join(a.resultados, "comparacion_metodos.csv"))
    print("\n=== COMPARACIÓN ===\n", comp.round(4))

    for c in casos_representativos(casos):
        figura_comparacion(c, list(metodos), a.salida, a.resultados)


if __name__ == "__main__":
    main()
