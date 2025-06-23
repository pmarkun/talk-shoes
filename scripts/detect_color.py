#!/usr/bin/env python3
"""
Detecta a cor predominante e busca o nome mais próximo numa paleta.
Agora aceita métricas RGB, LAB ou HSV e vem com paleta de 16 cores balanceada.
"""

from pathlib import Path
import argparse, json, yaml, colorsys, math
import numpy as np, cv2
from sklearn.cluster import KMeans

# ---------- Etapas já vistas (crop + brighten + K-Means) -------------------
def center_crop(img, margin=0.1):
    h, w = img.shape[:2]
    mx = int(w*margin) if margin < 1 else int(margin)
    my = int(h*margin) if margin < 1 else int(margin)
    return img[my:h-my, mx:w-mx]

def brighten(img, p=95):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    v   = hsv[..., 2]
    hsv[..., 2] = np.clip(v * (255. / max(np.percentile(v, p), 1)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def dominant_rgb(path, margin=0.1, k=5, sample=40_000):
    bgr = cv2.imread(str(path))
    bgr = brighten(center_crop(bgr, margin))
    pix = bgr.reshape(-1, 3)
    if len(pix) > sample:
        pix = pix[np.random.choice(len(pix), sample, replace=False)]
    km   = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(pix)
    dom  = km.cluster_centers_[np.bincount(km.labels_).argmax()].astype(int)
    return tuple(int(c) for c in dom[::-1])           # RGB

# ---------- Paleta revisada ------------------------------------------------
PALETTE16 = {
    "Preto":       "#000000",
    "Branco":      "#ffffff",
    "Cinza":       "#7f7f7f",
    "Vermelho":    "#e6194b",
    "Laranja":     "#f58231",
    "Amarelo":     "#ffe119",
    "Verde":       "#3cb44b",
    "Azul":        "#4363d8",
    "Ciano":       "#42d4f4",
    "Roxo":        "#911eb4",
    "Magenta":     "#f032e6",
    "Marrom":      "#8b4513",
    "Rosa":        "#fabed4",
    "Oliva":       "#808000",
    "Marinho":     "#000075",
    "Lima":        "#bfef45",
}


# ---------- Conversões -----------------------------------------------------
def hex2rgb(h): h=h.lstrip('#'); return tuple(int(h[i:i+2],16) for i in (0,2,4))
def rgb2lab(rgb):  # via OpenCV
    return cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0,0]

# ---------- Métricas de distância -----------------------------------------
def dist_rgb(c1, c2): return np.linalg.norm(np.array(c1) - c2)
def dist_lab(c1, c2):
    return np.linalg.norm(rgb2lab(c1) - rgb2lab(c2))
def dist_hsv(c1, c2):
    h1,s1,v1 = colorsys.rgb_to_hsv(*[x/255 for x in c1])
    h2,s2,v2 = colorsys.rgb_to_hsv(*[x/255 for x in c2])
    dh = min(abs(h1-h2), 1-abs(h1-h2)) * 2   # peso maior para H
    return math.sqrt(dh*dh + (s1-s2)**2 + (v1-v2)**2)

DIST_FUN = {"rgb": dist_rgb, "lab": dist_lab, "hsv": dist_hsv}

def load_palette(path):
    if path is None: return PALETTE16
    if path.suffix.lower()==".json": return json.loads(path.read_text())
    if path.suffix.lower() in {".yml",".yaml"}:
        return yaml.safe_load(path.read_text())
    raise ValueError("Use JSON ou YAML na paleta.")

def nearest(color, palette, metric):
    d = DIST_FUN[metric]
    items = [(name, hex2rgb(hex), d(color, hex2rgb(hex)))
             for name, hex in palette.items()]
    name, rgb, _ = min(items, key=lambda x: x[2])
    return name, rgb

# ---------- CLI ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Cor predominante + nome mais próximo")
    ap.add_argument("image", type=Path)
    ap.add_argument("--metric", choices=("rgb","lab","hsv"), default="lab",
                    help="Métrica p/ proximidade (default: lab)")
    ap.add_argument("--palette", type=Path,
                    help="Arquivo JSON/YAML com a paleta")
    ap.add_argument("--margin", type=float, default=.1,
                    help="Margem central (fração ou px)")
    args = ap.parse_args()

    dom = dominant_rgb(args.image, args.margin)
    pal = load_palette(args.palette)
    name, rgb_pal = nearest(dom, pal, args.metric)

    def hx(r): return "#{:02x}{:02x}{:02x}".format(*r)
    print(f"Dominante exata : {dom}  {hx(dom)}")
    print(f"Mais próxima    : {name}  {rgb_pal}  {hx(rgb_pal)} (métrica={args.metric})")

if __name__ == "__main__":
    main()