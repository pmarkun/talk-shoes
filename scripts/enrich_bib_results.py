import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import yaml
from tqdm import tqdm

from placa_peito import extract_run_data


def load_config() -> Dict[str, Any]:
    """Loads configuration from config.yaml or config.yaml-sample."""
    cfg_path = Path("config.yaml")
    if not cfg_path.is_file():
        cfg_path = Path("config.yaml-sample")
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    return {}


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    raise ValueError("JSON de entrada deve ser uma lista de objetos")


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def process_item(item: Dict[str, Any], categories: List[str], colours: str | None, max_retries: int, error_dir: Path | None) -> Dict[str, Any]:
    image_path = item.get("image")
    if not image_path:
        item["error"] = "caminho da imagem ausente"
        return item

    if item.get("number") and item.get("category"):
        # já processado
        return item

    img_path = Path(image_path)
    if not img_path.is_file():
        item["error"] = "imagem nao encontrada"
        return item

    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        item["error"] = "falha ao abrir imagem"
        return item

    attempt = 0
    last_err: str | None = None
    while attempt < max_retries:
        try:
            result = extract_run_data(img, categories, colours=colours)
            if isinstance(result, dict):
                item.update(result)
            return item
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            attempt += 1
    item["error"] = last_err or "erro desconhecido"
    if error_dir:
        error_dir.mkdir(parents=True, exist_ok=True)
        error_img = error_dir / f"{img_path.stem}_error.jpg"
        cv2.imwrite(str(error_img), img)
        with open(error_dir / "errors.log", "a", encoding="utf-8") as fh:
            fh.write(f"{img_path}: {item['error']}\n")
    return item


def main(input_json: str, output_json: str, max_workers: int = 4, max_retries: int = 3, error_log: bool = False) -> None:
    cfg = load_config()
    settings = cfg.get("settings", {})
    categories = settings.get("bib_categories", [])
    colours = settings.get("bib_colours")

    data = load_json(Path(input_json))
    resume_data: List[Dict[str, Any]] = []
    if Path(output_json).is_file():
        resume_data = load_json(Path(output_json))
        processed_ids = {
            item.get("image") for item in resume_data if item.get("number")
        }
    else:
        processed_ids = set()

    to_process = [item for item in data if item.get("image") not in processed_ids]

    error_dir = Path("error_logs") if error_log else None

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_item, item, categories, colours, max_retries, error_dir)
            for item in to_process
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processando"):
            results.append(fut.result())

    combined = resume_data + results
    save_json(Path(output_json), combined)

    success = sum(1 for it in combined if it.get("number"))
    failures = len(combined) - success
    print("\nResumo:")
    print(f"Total processado: {len(combined)}")
    print(f"Com número detectado: {success}")
    print(f"Falhas: {failures}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enriquece JSON de provas com info da placa de peito")
    parser.add_argument("input", help="Arquivo JSON de entrada")
    parser.add_argument("output", help="Arquivo JSON de saída")
    parser.add_argument("--workers", type=int, default=4, help="Número de threads")
    parser.add_argument("--retries", type=int, default=3, help="Tentativas em caso de erro")
    parser.add_argument("--log-errors", action="store_true", help="Salvar imagens e mensagens de erro")

    args = parser.parse_args()
    main(args.input, args.output, args.workers, args.retries, args.log_errors)
