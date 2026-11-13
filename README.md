# Segmentación de Vasos Sanguíneos en Retina – HRF

**Curso:** Visión por Computador 2026-II – Universidad Nacional de Ingenierías
**Docente:** Mag. Elian R. Laura Riveros
**Grupo:** Equipo 1 – Dataset HRF (High-Resolution Fundus)

Este proyecto segmenta los vasos sanguíneos de imágenes de fondo de ojo usando **solo procesamiento digital de imágenes clásico**. Las reglas del examen no permiten machine learning ni deep learning. Se comparan al menos dos pipelines distintos contra la segmentación manual del dataset, usando el coeficiente de Dice y el índice de Jaccard.

---

## Tabla de contenido

1. [Requisitos](#requisitos)
2. [Dataset](#dataset)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Inicio rápido](#inicio-rápido)
5. [Uso en Google Colab](#uso-en-google-colab)
6. [Preprocesamiento compartido](#preprocesamiento-compartido)
7. [Métodos implementados](#métodos-implementados)
8. [Cómo agregar un método nuevo](#cómo-agregar-un-método-nuevo)
9. [Comandos de `ejecutar.py`](#comandos-de-ejecutarpy)
10. [Métricas](#métricas)
11. [Archivos de salida](#archivos-de-salida)
12. [Flujo de trabajo del grupo](#flujo-de-trabajo-del-grupo)
13. [Entregables](#entregables)
14. [Problemas frecuentes](#problemas-frecuentes)

---

## Requisitos

- Python 3.9 o superior
- Librerías:

```bash
pip install opencv-python scikit-image numpy pandas matplotlib
```

En Colab todas estas librerías ya vienen instaladas.

---

## Dataset

Se usa la carpeta **`HRF`**, que contiene 45 imágenes de 3504×2336 px. Hay 15 de pacientes sanos (`_h`), 15 con retinopatía diabética (`_dr`) y 15 con glaucoma (`_g`).

```
HRF/
├── images/    01_dr.JPG, 01_g.jpg, 01_h.jpg, ...   → entrada del algoritmo
├── mask/      01_dr_mask.tif, ...                  → campo de visión (FOV)
└── manual1/   01_dr.tif, ...                       → ground truth (solo para evaluar)
```

| Carpeta   | Contenido                         | Uso                                    |
|-----------|-----------------------------------|----------------------------------------|
| `images`  | Fotos RGB de la retina            | Entrada de los métodos                 |
| `mask`    | Círculo blanco del FOV            | Recortar el fondo negro exterior       |
| `manual1` | Segmentación manual de un experto | **Solo** para calcular Dice y Jaccard  |

> ⚠️ `manual1` **nunca** debe usarse dentro de un método de segmentación.

---

## Estructura del proyecto

```
proyecto_vasos/
├── README.md
├── ejecutar.py                 ← corre, evalúa y compara métodos
├── comun/                      ← BASE COMPARTIDA (los cambios se avisan al grupo)
│   ├── preprocesamiento.py       carga, caché, canal, FOV, iluminación, CLAHE, ruido
│   ├── postprocesamiento.py      FOV, eliminar ruido pequeño, rellenar huecos
│   └── evaluacion.py             Dice, Jaccard, evaluación y barridos
└── metodos/                    ← un archivo por método; cada integrante trabaja aquí
    ├── plantilla.py              punto de partida para métodos nuevos
    ├── p1_blackhat.py            Pipeline 1
    └── p2_frangi.py              Pipeline 2
```

La idea es que **el preprocesamiento, el postprocesamiento y la evaluación sean idénticos para todos**. Así, cualquier diferencia en las métricas se debe al método de segmentación y no a detalles de implementación.

---

## Inicio rápido

```bash
# 1. Probar con 5 imágenes (rápido)
python ejecutar.py --base RUTA/A/HRF --solo 5

# 2. Correr los dos pipelines en las 45 imágenes
python ejecutar.py --base RUTA/A/HRF

# 3. Ver la tabla comparativa
cat resultados/comparacion_metodos.csv
```

---

## Uso en Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

# Descomprimir el proyecto (si se subió como zip)
!unzip -q /content/drive/MyDrive/proyecto_vasos.zip -d /content/

%cd /content/proyecto_vasos
!python ejecutar.py --base "/content/drive/MyDrive/RUTA/HRF" --solo 5
```

Para trabajar de forma interactiva en un notebook:

```python
import sys
sys.path.append("/content/proyecto_vasos")

from comun.preprocesamiento import listar_casos, cargar_caso, preprocesar, visualizar_pasos
from comun.evaluacion import evaluar, resumen
from metodos import p1_blackhat

casos = listar_casos("/content/drive/MyDrive/RUTA/HRF")
img, fov = cargar_caso(casos[0], escala=0.5)
pre = preprocesar(img, fov)
visualizar_pasos(pre)                       # muestra cada paso

df = evaluar(p1_blackhat, casos[:5])
resumen(df)
```

> Si modificas un archivo `.py` mientras usas el notebook, reinicia el entorno de ejecución o usa `importlib.reload(...)` para que se carguen los cambios.

---

## Preprocesamiento compartido

`comun/preprocesamiento.py` aplica los pasos en este orden:

```
imagen RGB → canal → relleno fuera del FOV → corrección de iluminación → CLAHE → reducción de ruido → (inversión)
```

### Configuración (`CONFIG_BASE`)

| Clave            | Defecto     | Opciones                                       | Para qué sirve |
|------------------|-------------|------------------------------------------------|----------------|
| `canal`          | `"verde"`   | `"verde"`, `"luminancia"`, `"verde_rojo"`      | El canal verde tiene el mayor contraste entre vaso y fondo |
| `rellenar_fov`   | `True`      | `True` / `False`                               | Evita el falso anillo en el borde del ojo |
| `erosion_fov_px` | `5`         | entero ≥ 0                                     | Recorta el borde del FOV, donde hay artefactos |
| `iluminacion`    | `None`      | `None`, `"resta_fondo"`, `"division_fondo"`    | Compensa la iluminación no uniforme |
| `kernel_fondo`   | `51`        | entero impar                                   | Tamaño de la mediana con la que se estima el fondo |
| `clahe`          | `True`      | `True` / `False`                               | Ecualización adaptativa del contraste |
| `clahe_clip`     | `2.0`       | float                                          | Valores más altos dan más contraste y más ruido |
| `clahe_tile`     | `8`         | entero                                         | Tamaño de la rejilla de CLAHE |
| `ruido`          | `"gauss"`   | `None`, `"gauss"`, `"mediana"`, `"bilateral"`  | Suavizado |
| `ruido_param`    | `1.0`       | float / int                                    | Sigma (Gauss), tamaño (mediana) o diámetro (bilateral) |
| `invertir`       | `False`     | `True` / `False`                               | Vasos claros sobre fondo oscuro |

Cada método puede cambiar solo las claves que necesite mediante su propio `CONFIG_PRE`, sin tocar la base:

```python
CONFIG_PRE = {"iluminacion": "resta_fondo", "clahe_clip": 3.0}
```

### Resolución de trabajo

Las imágenes se procesan a **escala 0.5** (1752×1168) por velocidad. La máscara predicha se vuelve a llevar al tamaño original antes de compararla con `manual1`. Si cambian la escala, los tamaños de kernel y los sigmas de los métodos deben escalarse en la misma proporción.

### Caché

`cargar_caso()` guarda en `.cache/` las imágenes ya reescaladas, para no releer los JPG grandes en cada prueba. Si cambias la escala o el dataset, borra esa carpeta.

---

## Métodos implementados

### Pipeline 1 – Black-Hat multiescala + Otsu (`p1_blackhat`)

1. **Preprocesamiento:** canal verde, CLAHE y sin suavizado previo.
2. **Realce:** transformada Black-Hat (cierre − imagen) con kernels elípticos de 7, 11 y 15 px. De cada píxel se toma el máximo entre escalas, lo que resalta estructuras oscuras más estrechas que el kernel, es decir, los vasos finos y gruesos.
3. **Segmentación:** umbral de Otsu calculado solo dentro del FOV, multiplicado por `factor_otsu`.
4. **Postprocesamiento:** se aplica el FOV y se eliminan las componentes de menos de `area_min` píxeles.

| Parámetro     | Defecto       | Efecto |
|---------------|---------------|--------|
| `kernels`     | `[7, 11, 15]` | Grosor máximo de vaso detectable |
| `factor_otsu` | `1.0`         | Más de 1 da menos falsos positivos y pierde vasos finos |
| `area_min`    | `100`         | Más alto elimina más ruido, pero puede cortar vasos |

**Ventajas:** es simple, rápido e interpretable.
**Limitaciones:** es sensible a lesiones oscuras (hemorragias) y al borde del disco óptico, y el umbral global falla con contraste variable.

### Pipeline 2 – Frangi + histéresis (`p2_frangi`)

1. **Preprocesamiento:** canal verde, CLAHE y filtro gaussiano.
2. **Realce:** filtro de **Frangi**, que analiza los autovalores de la matriz Hessiana en varias escalas (σ = 1…5). Responde fuerte a estructuras tubulares y débil a manchas o bordes.
3. **Segmentación:** umbral por **histéresis**. Los píxeles por encima del percentil alto funcionan como semillas, y la región crece hacia los vecinos conectados que superan el percentil bajo. Es una forma de **crecimiento de regiones**.
4. **Postprocesamiento:** se aplica el FOV y se eliminan las componentes pequeñas.

| Parámetro  | Defecto           | Efecto |
|------------|-------------------|--------|
| `sigmas`   | `(1, 2, 3, 4, 5)` | Rango de grosores de vaso |
| `beta`     | `0.5`             | Sensibilidad a estructuras tipo mancha frente a tubo |
| `pct_alto` | `95`              | Cantidad de semillas |
| `pct_bajo` | `88`              | Hasta dónde crecen las regiones; más bajo da más vasos y más ruido |

**Ventajas:** se adapta a vasos de distinto grosor, conserva la conectividad y es más robusto a manchas.
**Limitaciones:** es más lento, tiende a engrosar vasos finos y puede responder al borde del disco óptico.

---

## Cómo agregar un método nuevo

1. Copia la plantilla:
   ```bash
   cp metodos/plantilla.py metodos/<iniciales>_<idea>.py
   ```
2. Define en ese archivo:
   - `NOMBRE` y `DESCRIPCION`
   - `CONFIG_PRE`: solo lo que cambia respecto a `CONFIG_BASE`
   - `PARAMS`: los parámetros de tu método
   - `segmentar(pre, p)`: recibe un objeto `Preprocesado` y devuelve una máscara `uint8` con valores 0/1 del mismo tamaño que `pre.final`
3. Pruébalo:
   ```bash
   python ejecutar.py --metodos <iniciales>_<idea> --solo 5
   ```

### Qué contiene `pre` (objeto `Preprocesado`)

| Atributo             | Descripción |
|----------------------|-------------|
| `pre.final`          | Imagen gris `uint8` preprocesada, con vasos oscuros salvo que `invertir=True` |
| `pre.pasos`          | Diccionario con cada paso intermedio (`"canal"`, `"clahe"`, ...) |
| `pre.fov`            | FOV (0/1) a la escala de trabajo |
| `pre.fov_erosionado` | FOV recortado; es el que se recomienda usar |
| `pre.img_bgr`        | Imagen a color a la escala de trabajo |
| `pre.config`         | Configuración de preprocesamiento usada |

### Ideas de métodos para iterar

- Filtros de **Gabor 2D** en varias orientaciones (aparecen en la diapositiva 9)
- **Matched filter** (kernel gaussiano invertido rotado en 12 orientaciones)
- **Umbral adaptativo** (`cv2.adaptiveThreshold`) sobre el Black-Hat
- **Top-hat con elementos lineales** rotados (line detector)
- Combinación de la respuesta de Black-Hat y Frangi antes de umbralizar

---

## Comandos de `ejecutar.py`

| Opción | Qué hace |
|--------|----------|
| *(sin opciones)* | Corre `p1_blackhat` y `p2_frangi` en todo el dataset |
| `--base RUTA` | Ruta de la carpeta HRF (por defecto `HRF`) |
| `--metodos A B C` | Métodos a correr, por nombre de archivo en `metodos/` |
| `--solo N` | Usa solo las primeras N imágenes |
| `--escala 0.5` | Escala de trabajo |
| `--ver-pre` | Solo genera las figuras del preprocesamiento, en 3 imágenes representativas |
| `--barrido PARAM v1 v2 ...` | Estudio de sensibilidad de un parámetro del método |
| `--barrido-pre` | Junto con `--barrido`, indica que el parámetro es de preprocesamiento |
| `--salida DIR` | Carpeta de las máscaras (por defecto `predicted_masks`) |
| `--resultados DIR` | Carpeta de métricas y figuras (por defecto `resultados`) |

Ejemplos:

```bash
python ejecutar.py --metodos p1_blackhat --barrido factor_otsu 0.8 1.0 1.2 1.4
python ejecutar.py --metodos p2_frangi --barrido pct_bajo 82 85 88 90
python ejecutar.py --metodos p2_frangi --barrido clahe_clip 1 2 3 4 --barrido-pre
python ejecutar.py --metodos p1_blackhat p2_frangi rv_gabor --base /content/drive/MyDrive/HRF
```

Los barridos usan 3 imágenes representativas (una sana, una con retinopatía y una con glaucoma) para que sean rápidos. El valor elegido debe confirmarse después con el dataset completo.

---

## Métricas

Donde **A** es la máscara predicha y **B** el ground truth (`manual1`):

- **Coeficiente de Dice:** DSC = 2·|A ∩ B| / (|A| + |B|)
- **Índice de Jaccard (IoU):** IoU = |A ∩ B| / |A ∪ B|

Ambos van de 0 a 1, y 1 significa coincidencia perfecta. Se reportan por imagen, por clase (sano, retinopatía, glaucoma) y como media y desviación estándar de todo el dataset.

> Como referencia, los métodos clásicos bien ajustados suelen alcanzar un Dice de aproximadamente 0.65–0.75 en HRF.

---

## Archivos de salida

```
predicted_masks/
├── p1_blackhat/01_dr.png, ...     máscaras binarias (0/255) a tamaño original
└── p2_frangi/...
resultados/
├── metricas_<metodo>.csv          Dice y Jaccard por imagen
├── comparacion_metodos.csv        tabla resumen entre métodos
├── comparacion_<imagen>.png       original | GT | método 1 | método 2 ...
├── pre_<metodo>_<imagen>.png      pasos del preprocesamiento (--ver-pre)
└── barrido_<metodo>_<param>.png   curvas de sensibilidad (--barrido)
```


