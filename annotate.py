import os
import json
from pathlib import Path
from typing import Union

import pandas as pd
from PIL import Image, ImageDraw

def relative_bbox(bbox_absoluto: list[int], bbox_referencia: list[int]) -> list[int]:
    """
    Recebe dois bboxes no formato [x1, y1, x2, y2] e retorna o bbox_absoluto
    ajustado para o sistema de coordenadas local do bbox_referencia.

    Ex: se bbox_absoluto = [120, 200, 180, 260] e bbox_referencia = [100, 150, 300, 350],
    retorna [20, 50, 80, 110]
    """
    ax1, ay1, ax2, ay2 = bbox_absoluto
    rx1, ry1, _, _ = bbox_referencia
    return [ax1 + rx1, ay1 + ry1, ax2 + rx1, ay2 + ry1]

def draw_bboxes_from_json(
    data_file: Union[str, Path, pd.DataFrame],
    base_path: Union[str, Path],
    output_base: Union[str, Path] = "output",
    shoe_color: tuple[int, int, int] = (255, 0, 0),        # vermelho
    demo_color: tuple[int, int, int] = (0, 255, 0),        # verde
    line_width: int = 4
) -> None:
    """
    Lê um arquivo JSON ou DataFrame no formato do exemplo, constrói os caminhos
    das imagens, desenha os bboxes e salva cópias anotadas.

    Parameters
    ----------
    data_file : str | Path | pd.DataFrame
        Caminho para o JSON ou um DataFrame já carregado.
    base_path : str | Path
        Diretório raiz onde estão as pastas de imagens.
    output_base : str | Path, default "output"
        Diretório onde as imagens anotadas serão gravadas.
    shoe_color : tuple[int, int, int], default (255, 0, 0)
        Cor RGB das caixas de calçados.
    demo_color : tuple[int, int, int], default (0, 255, 0)
        Cor RGB das caixas demográficas (faces).
    line_width : int, default 4
        Espessura das linhas dos retângulos.
    """
    # Carrega dados
    if isinstance(data_file, (str, Path)):
        with open(data_file, "r", encoding="utf-8") as fp:
            raw = json.load(fp)
        df = pd.DataFrame(raw)
    elif isinstance(data_file, pd.DataFrame):
        df = data_file.copy()
    else:
        raise TypeError("data_file deve ser str, Path ou DataFrame")

    base_path = Path(base_path)
    output_base = Path(output_base)

    # Garante que estruturas de saída existam
    for folder in df["folder"].unique():
        (output_base / folder).mkdir(parents=True, exist_ok=True)

    # Percorre cada linha
    for idx, row in df.iterrows():
        img_path = base_path / row["folder"] / row["filename"]
        if not img_path.is_file():
            print(f"[AVISO] Imagem não encontrada: {img_path}")
            continue

        # Abre imagem e prepara para desenhar transpondo rotações do EXIF
    
        with Image.open(img_path).convert("RGB") as im:
            # Corrige a orientação da imagem
            exif = im.getexif()
            orientation = exif.get(274)
            if orientation == 3:
                im = im.rotate(180, expand=True)
            elif orientation == 6:
                im = im.rotate(270, expand=True)
            elif orientation == 8:
                im = im.rotate(90, expand=True)
            draw = ImageDraw.Draw(im)

            # Desenha bbox da pessoa
            x1, y1, x2, y2 = map(int, row["bbox"])
            draw.rectangle([(x1, y1), (x2, y2)],
                           outline=(255, 255, 255),
                           width=line_width)
            
            # Desenha bboxes de calçados
            for shoe in row["shoes"]:
                # Em alguns casos shoe["bbox"] pode ser lista de listas
                for box in shoe["bbox"]:
                    x1, y1, x2, y2 = map(int, relative_bbox(box, row["bbox"]))
                    draw.rectangle([(x1, y1), (x2, y2)],
                                   outline=shoe_color,
                                   width=line_width)

            # Desenha bboxes demográficos

            x1, y1, x2, y2 = map(int, relative_bbox(row["demographic"]["bbox"], row["bbox"]))
            draw.rectangle([(x1, y1), (x2, y2)],
                            outline=demo_color,
                            width=line_width)

            # Salva resultado
            out_path = output_base / row["folder"] / f"{Path(row['filename']).stem}_{idx}_bbox.jpg"
            im.save(out_path, "JPEG")
            print(f"[OK] Salvo em {out_path}")

# Exemplo de uso:
# draw_bboxes_from_json("dados.json", base_path="/imagens/maratona")

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Desenha caixas delimitadoras em imagens a partir de um JSON.")
    parser.add_argument("data_file", type=str, help="Caminho para o arquivo JSON ou DataFrame")
    parser.add_argument("base_path", type=str, help="Caminho base para as imagens")
  
    args = parser.parse_args()

    draw_bboxes_from_json(
        data_file=args.data_file,
        base_path=args.base_path,
    )