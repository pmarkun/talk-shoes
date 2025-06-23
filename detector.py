from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
import json
from tqdm import tqdm
from torchvision import transforms, models
from transformers import ViTForImageClassification
from ultralytics import YOLO

from insightface.app import FaceAnalysis
from placa_peito import extract_run_data
import uuid
import threading

import yaml

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ShoesAIAnalyzer:
    def __init__(self,
             config_path: str = 'config.yaml'):
        """ Inicializa o analisador com os parâmetros e carrega modelo e classes. """
        logger.info("Inicializando ShoesAIAnalyzer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Utilizando dispositivo: {self.device}")
        self.config = None

        # Carrega os path do config.xml
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        output_cfg = self.config.get("output", {})
        self.save_crops = output_cfg.get("save_crops", False)
        self.output_path = Path(output_cfg.get("path", "output"))
        self.min_confidence_cls = output_cfg.get("min_confidence", 0.5)
        self.save_uncertain = output_cfg.get("save_uncertain", False)

        #Carrega o Classificador de Tênis
        self.classes = self._load_classify_shoes_classes(self.config["models"]["classify_shoes_classes"])
        self.classify_shoes_model = self._load_classify_shoes_model(self.config["models"]["classify_shoes_model"])
        
        #Carrega o Detector de Tênis
        self.detect_shoes_model = self._load_yolo_model(self.config["models"]["detect_shoes_model"])

        #Carrega o Detector de Pessoas
        self.detect_person_model = self._load_yolo_model(self.config["models"]["detect_person_model"])

        #Carrega o Detector de Placa de Peito (Bib)
        bib_model_path = self.config["models"].get("detect_bib_model")
        self.detect_bib_model = self._load_yolo_model(bib_model_path) if bib_model_path else None

        #Categorias e instruções de cor para corrida (opcionais)
        self.bib_settings = (
            self.config.get("settings", {}).get('bib_detect', {})
        )

        #Carrega o Detector de Faces
        self.detect_face_model = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.detect_face_model.prepare(ctx_id=0, det_size=(640, 640))
        self.detect_face_transform = self._define_transform() # Pode ser igual ou diferente se necessário para FairFace
        
        #Carrega o classificador de gênero
        self.classify_gender_model = self._load_classify_gender_model(self.config["models"]["classify_gender_model"])
        self.classify_transform = self._define_transform() # Transformação padrão para classificação ViT e gênero

        self.confidence = self.config["settings"]["confidence"] # Confiança mínima para detecção de face
        logger.info("ShoesAIAnalyzer inicializado com sucesso.")


    def _load_classify_shoes_classes(self, classes_path) -> list:
        """
        Carrega as classes do arquivo classes.txt.
        """
        logger.info(f"Carregando classes de {classes_path}")
        try:
            with open(classes_path, "r") as f:
                classes = [line.strip() for line in f if line.strip()]
            if not classes:
                logger.warning(f"Arquivo de classes {classes_path} está vazio ou não contém classes válidas.")
                return []
            logger.info(f"Classes carregadas: {classes}")
            return classes
        except FileNotFoundError:
            logger.error(f"Arquivo de classes {classes_path} não encontrado.")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar classes de {classes_path}: {e}")
            raise

    def _load_classify_gender_model(self, model_path: str) -> torch.nn.Module:
        """
        Carrega um checkpoint FairFace (.pth) e o prepara para inferência.
        """
        logger.info(f"Carregando modelo de gênero/idade/raça de {model_path}")
        try:
            model = models.resnet34(weights=None) # Use weights=None para evitar warning de deprecated 'pretrained'
            model.fc = torch.nn.Linear(model.fc.in_features, 18)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device).eval()
            logger.info("Modelo de gênero/idade/raça carregado.")
            return model
        except FileNotFoundError:
            logger.error(f"Arquivo do modelo de gênero {model_path} não encontrado.")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de gênero {model_path}: {e}")
            raise

    def _load_classify_shoes_model(self, model_path: str) -> torch.nn.Module:
        """
        Carrega o modelo VIT16 treinado (.pth) e o coloca em modo de avaliação.
        """
        logger.info(f"Carregando modelo de classificação de calçados de {model_path}")
        try:
            model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                num_labels=len(self.classes),
                ignore_mismatched_sizes=True # Útil se o num_labels no checkpoint for diferente
            )
            # A maneira correta de adaptar o classificador ViT é geralmente substituir model.classifier
            # Se o state_dict já contém o classificador treinado, esta recriação pode não ser necessária
            # ou precisa corresponder exatamente à arquitetura usada no treinamento.
            # Assumindo que o state_dict é para o modelo inteiro, incluindo o classificador adaptado:
            model.classifier = torch.nn.Sequential( # type: ignore
                torch.nn.Dropout(0.3),
                torch.nn.Linear(model.classifier.in_features, len(self. classes))
            )
            state_dict = torch.load(model_path, map_location=self.device)

            # Se o state_dict foi salvo APENAS para o classificador e você quer carregar em um ViT pré-treinado:
            # model.classifier.load_state_dict(state_dict)
            # OU, se foi salvo o modelo inteiro:
            model.load_state_dict(state_dict)
            model.to(self.device).eval() # type: ignore

            logger.info("Modelo de classificação de calçados carregado.")
            return model
        except FileNotFoundError:
            logger.error(f"Arquivo do modelo de classificação {model_path} não encontrado.")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de classificação {model_path}: {e}")
            raise

    def _load_yolo_model(self, model_path: str):
        """
        Carrega o modelo YOLO de detecção de objetos.
        """
        logger.info(f"Carregando modelo de detecção de objetos de {model_path}")
        try:
            model = YOLO(model_path)
            logger.info("Modelo de detecção de objetos carregado.")
            return model
        except FileNotFoundError: # YOLO pode levantar outros erros se o arquivo estiver corrompido
            logger.error(f"Arquivo do modelo de detecção {model_path} não encontrado.")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de detecção {model_path}: {e}")
            raise

    def _define_transform(self):
        """
        Define as transformações padrão para as imagens de classificação (ViT e Gênero).
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def _open_image(self, image_path: Path) -> Image.Image | None:
        """Abre e prepara uma imagem PIL a partir de um caminho."""
        try:
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB") # type: ignore
        except FileNotFoundError:
            logger.error(f"Arquivo de imagem não encontrado: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Erro ao abrir ou converter imagem {image_path}: {e}")
            return None

    def detect_person_batch(self, image_paths: list[Path]) -> list[tuple[Path, np.ndarray | None, list]]:
        """
        Detecta objetos em várias imagens. Retorna lista de (caminho, array_img, yolo_results).
        Se uma imagem falhar ao carregar, array_img será None e yolo_results será [].
        """
        loaded_images_data = [] # Lista de (path, pil_image)
        for p in image_paths:
            pil_img = self._open_image(p)
            if pil_img:
                loaded_images_data.append((p, pil_img))
            else:
                # Adiciona placeholder para manter a correspondência de índice se uma imagem falhar
                loaded_images_data.append((p, None))

        # Prepara imagens para YOLO apenas se foram carregadas com sucesso
        valid_pil_images = [data[1] for data in loaded_images_data if data[1] is not None]
        valid_paths = [data[0] for data in loaded_images_data if data[1] is not None]

        if not valid_pil_images:
            return [(p, None, []) for p, _ in loaded_images_data] # Retorna Nones para todas

        np_imgs_for_yolo = [np.array(pil_img) for pil_img in valid_pil_images]
        
        verbose_level = logger.getEffectiveLevel() <= logging.DEBUG # Seja verboso se INFO ou DEBUG
        yolo_batch_results = list(self.detect_person_model(np_imgs_for_yolo, iou=0.5, stream=True, verbose=verbose_level))

        # Mapeia os resultados de volta para os caminhos originais, incluindo falhas
        output_results = []
        yolo_result_idx = 0
        for original_path, original_pil_img in loaded_images_data:
            if original_pil_img:
                # Imagem foi processada pelo YOLO
                np_array = np_imgs_for_yolo[yolo_result_idx] # Reusa o array já convertido
                yolo_res = yolo_batch_results[yolo_result_idx]
                output_results.append((original_path, np_array, [yolo_res])) # YOLO retorna lista de resultados, mesmo para uma img
                yolo_result_idx += 1
            else:
                # Imagem falhou ao carregar
                output_results.append((original_path, None, []))
        
        return output_results


    def crop_images(self, image_array: np.ndarray, yolo_results, classes_to_crop: list = [0], confidence: float = 0.5) -> list[dict]:
        """Extrai crops de sapatos da imagem."""
        crops = []
        if image_array is None or not yolo_results:
            return crops

        for r in yolo_results: # r é um ultralytics.engine.results.Results object
            if r is None or r.boxes is None: # Adicionado para segurança
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls in classes_to_crop and conf >= confidence:
                    crop_pil = Image.fromarray(image_array[y1:y2, x1:x2])
                    crops.append({
                        "img": crop_pil,
                        "box": (x1, y1, x2, y2),
                        "confidence": conf, # Adiciona confiança da detecção
                        "class_detection": cls       # Adiciona classe da detecção YOLO
                    })
        return crops

    def detect_bibs(self, pil_image: Image.Image) -> list[dict]:
        """Detecta placas de peito na imagem de uma pessoa e lê seu conteúdo."""
        if self.detect_bib_model is None:
            return []

        np_img = np.array(pil_image)
        results = self.detect_bib_model(np_img, stream=True, verbose=False)

        crops = self.crop_images(np_img, results, classes_to_crop=[0], confidence=0.3)
        bibs = []
        for crop in crops:
            run_data = extract_run_data(
                np.array(crop["img"]),
                self.bib_settings.get("categories", []),
                colours=self.bib_settings.get("colours", ""),
            )
            bib_info = {
                "bbox": crop["box"],
                "confidence": crop["confidence"],
            }
            if isinstance(run_data, dict):
                bib_info.update(run_data)
            bibs.append(bib_info)
        return bibs

    def classify_batch(self, pil_images: list[Image.Image]) -> list[dict]:
        """Classifica uma lista de PIL Images (crops de sapatos) em lote."""
        if not pil_images:
            return []

        stack = [self.classify_transform(img) for img in pil_images]
        batch = torch.stack(stack).to(self.device) # type: ignore

        with torch.no_grad():
            logits = self.classify_shoes_model(batch).logits # Para ViTForImageClassification
            probs  = torch.softmax(logits, dim=1)
            max_p, idx = probs.max(dim=1)

        results = []
        for p_val, i_val in zip(max_p.cpu(), idx.cpu()):
            prob_item = p_val.item()
            class_idx = i_val.item()
            
            predicted_label = self.classes[class_idx] # type: ignore
            results.append({"label": predicted_label, "prob": prob_item})
        
        return results

    def _save_shoe_crop(self, img: Image.Image, label: str, prob: float) -> None:
        """Salva um crop classificado opcionalmente em disco."""
        if not self.save_crops:
            return
        certain = prob >= self.min_confidence_cls
        if not certain and not self.save_uncertain:
            return

        folder = "certain" if certain else "uncertain"
        dest_dir = self.output_path / "classes" / folder / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = dest_dir / f"crop_{uuid.uuid4().hex}.jpg"
        try:
            img.save(filename)
        except Exception as e:
            logger.error(f"Erro ao salvar crop em {filename}: {e}")


    def detect_faces(self, pil_image: Image.Image) -> list[Image.Image]:
        """Detecta rostos em uma imagem PIL e retorna uma lista de crops de rostos (PIL Images)."""
        if pil_image is None:
            return []
        
        logger.debug("Detectando rostos...")
        # insightface espera BGR numpy array
        img_bgr = np.array(pil_image.convert("RGB"))[:, :, ::-1] # PIL RGB para BGR numpy
        
        try:
            detected_faces_info = self.detect_face_model.get(img_bgr)
        except Exception as e:
            logger.error(f"Erro durante a detecção de faces com InsightFace: {e}")
            return []

        logger.debug(f"InsightFace encontrou {len(detected_faces_info)} rostos.")
        
        face_crops = []
        for face_info in detected_faces_info:
            if face_info.det_score >= self.confidence:
                x1, y1, x2, y2 = face_info.bbox.astype(int)
                # Garante que as coordenadas estão dentro dos limites da imagem
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(pil_image.width, x2), min(pil_image.height, y2)
                if x1 < x2 and y1 < y2: # Verifica se o crop é válido
                    img = {
                        "img": pil_image.crop((x1, y1, x2, y2)),
                        "box": (x1, y1, x2, y2),
                    }
                    face_crops.append(img)
        logger.debug(f"Retornando {len(face_crops)} rostos após filtro de confiança ({self.confidence}).")
        return face_crops

    from typing import List, Dict, Optional

    def _pick_primary_face(
        self,
        faces: List[Dict],
        shoes: List[Dict],
        person_bbox: List[int],
        h_weight: float = 0.7,
        x_weight: float = 0.3,
    ) -> Optional[Dict]:
        """
        Heuristic to choose the face that *most likely* belongs to the runner
        whose shoes were detected.

        Parameters
        ----------
        faces : list of dict
            Output from your gender classifier (`demographic_data_list`), each with key "bbox".
        shoes : list of dict
            Your `shoes` list (built just above), each with key "bbox".
        person_bbox : [px1, py1, px2, py2]
            Bounding box of the cropped runner (`person_crop["box"]`).
        inside_margin : int
            Pixel tolerance for the “face centre inside person box” test.
        h_weight / x_weight : float
            Weights for the vertical‑position score vs. horizontal‑alignment score.
            They must sum to 1.0.

        Returns
        -------
        dict | None
            The chosen face dict, or None if nothing passes the inclusion test.
        """

        if not faces:
            return None

        px1, py1, px2, py2 = person_bbox
        pcx = (px1 + px2) / 2  # runner’s centre‑x

        # Average centre‑x of all shoe bboxes (falls back to runner centre if none)
        if shoes:
            scx = sum((sx1 + sx2) / 2 for s in shoes for sx1, _, sx2, _ in [s["bbox"]]) / len(shoes)
        else:
            scx = pcx

        candidates = []
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # 2. “Higher is better” vertical score  (top of image is y=0)
            height_score = 1 - (y1 - py1) / max(1, (py2 - py1))

            # 3. Horizontal alignment with shoe line
            align_score = 1 - abs(cx - scx) / max(1, (px2 - px1))

            score = h_weight * height_score + x_weight * align_score
            candidates.append((score, face))

        if not candidates:
            return None

        # highest‑scoring face wins
        return max(candidates, key=lambda t: t[0])[1]

    def classify_gender_batch(self, face_pil_images: list[Image.Image]) -> list[dict]:
        """Classifica gênero, idade e raça para uma lista de crops de rostos."""
        if not face_pil_images:
            return []

        gender_labels = ['male', 'female']
        age_labels    = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'] # Ajustado range para ser consistente
        race_labels   = ['White', 'Black', 'Latino_Hispanic', 'East Asian',
                         'Southeast Asian', 'Indian', 'Middle Eastern']

        # Usa self.face_transform se for diferente, senão self.transform
        stack = [self.classify_transform(img["img"]) for img in face_pil_images] # type: ignore
        batch = torch.stack(stack).to(self.device)

        with torch.no_grad():
            logits = self.classify_gender_model(batch) # (B, 18)

        # Separa logits para gênero, idade e raça
        gender_logits = logits[:, :2]
        age_logits = logits[:, 2:11]
        race_logits = logits[:, 11:]

        gender_probs = torch.softmax(gender_logits, dim=1)
        age_probs = torch.softmax(age_logits, dim=1)
        race_probs = torch.softmax(race_logits, dim=1)

        max_gender_p, gender_idx = gender_probs.max(dim=1)
        max_age_p, age_idx = age_probs.max(dim=1)
        max_race_p, race_idx = race_probs.max(dim=1)

        demographic_results = []
        for i in range(len(face_pil_images)):
            demographic_results.append({
                "gender": {"label": gender_labels[int(gender_idx[i].item())], "prob": max_gender_p[i].item()},
                "age":    {"label": age_labels[int(age_idx[i].item())],    "prob": max_age_p[i].item()},
                "race":   {"label": race_labels[int(race_idx[i].item())],   "prob": max_race_p[i].item()},
                "bbox":   face_pil_images[i]["box"], # type: ignore
            })
            logger.debug(f"Rosto {i+1} classificado: {demographic_results[-1]}")
        return demographic_results

    def _create_initial_record(self, image_path: Path, folder_root: Path, width: int = 0, height: int = 0) -> dict:
        """Cria a estrutura base de um registro para uma imagem."""
        record = {
            "filename": str(image_path.relative_to(folder_root)), # Salva como string
            "folder": image_path.parent.name,
            "original_width": width,
            "original_height": height,
        }
        return record

    def process_images(
        self,
        folder: str,
        max_images: int | None = None,
        skip_files: set[str] | None = None,
        output_path: str | None = None,
        resume_df: pd.DataFrame | None = None,
        cancel_event: 'threading.Event | None' = None,
    ) -> pd.DataFrame:
        """
        Processa todas as imagens, detecta calçados, classifica marcas e analisa rostos.
        Permite pular arquivos já processados e salvar resultados incrementalmente.
        """
        confidence = self.config["settings"]["confidence"] # type: ignore
        logger.info(f"Iniciando processamento de imagens em '{folder}', confidence={confidence}")
        folder_path = Path(folder)
        valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
        
        image_files = [p for p in folder_path.rglob("*") if p.suffix.lower() in valid_ext]
        image_files.sort()
        if max_images is not None:
            image_files = image_files[:max_images]
        logger.info(f"Encontradas {len(image_files)} imagens para processar.")

        # Carrega registros existentes se fornecido para retomar
        df_full = resume_df.copy() if resume_df is not None else pd.DataFrame()
        processed = set()
        if not df_full.empty and "filename" in df_full.columns:
            processed = set(df_full["filename"].astype(str).tolist())
        if skip_files:
            processed.update(skip_files)

        all_records_data: list[dict] = []

        batch_size = 8  # Para detecção de objetos YOLO
        for i in tqdm(range(0, len(image_files), batch_size), desc="Processando Lotes de Imagens"):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Processamento cancelado pelo usuário.")
                break
            batch_image_paths = [p for p in image_files[i : i + batch_size] if str(p.relative_to(folder_path)) not in processed]
            if not batch_image_paths:
                continue

            batch_records: list[dict] = []
            
            # Detecta e retorna as pessoas detectados em batch
            batch_detect_person_data = self.detect_person_batch(batch_image_paths)

            # Processa cada imagem do lote individualmente para demais etapas
            for image_path, img_array, yolo_results_list in batch_detect_person_data:
                if cancel_event is not None and cancel_event.is_set():
                    logger.info("Cancelamento solicitado - interrompendo o processamento do lote atual.")
                    break
                if img_array is None: # Imagem falhou ao carregar
                    logger.warning(f"Pulando processamento adicional para imagem com falha na leitura: {image_path}")
                    continue

                image_height, image_width = img_array.shape[:2]

                # 1. Detecção e Classificação de Calçados
                # yolo_results_list contém um único item Results se a detecção foi bem-sucedida
                person_crops_data = self.crop_images(img_array, yolo_results_list, confidence=0.9, classes_to_crop=[0]) # Classe 0 para sapatos

                if person_crops_data:
                    for person_crop in person_crops_data:
                        record = self._create_initial_record(image_path, folder_path, image_width, image_height)
                        record["bbox"] = person_crop["box"]
                        face_crops_pil = self.detect_faces(person_crop["img"]) # Usa o método que retorna PIL images
                        if face_crops_pil:
                            demographic_data_list = self.classify_gender_batch(face_crops_pil)
                        else:
                            demographic_data_list = []
                        shoe_crops = self.detect_shoes_model(person_crop["img"], stream=True, verbose=False) # Executa detecção de sapatos
                        shoe_crops_data = self.crop_images(np.array(person_crop["img"]), shoe_crops, classes_to_crop=[0])
                        shoe_classifications = self.classify_batch([shoe_crop["img"] for shoe_crop in shoe_crops_data])
                        shoes = []
                        for crop_info, classification_info in zip(shoe_crops_data, shoe_classifications):
                            foot = {}
                            foot["label"] = classification_info["label"]
                            foot["prob"] = classification_info.get("prob")
                            foot["bbox"] = crop_info["box"]
                            foot["confidence"] = crop_info["confidence"]
                            self._save_shoe_crop(crop_info["img"], foot["label"], foot["prob"])
                            shoes.append(foot)

                        
                        bibs = None
                        if self.bib_settings.get("enabled", False) and self.detect_bib_model:
                            try:
                                bibs = self.detect_bibs(person_crop["img"])
                            except:
                                from time import sleep
                                sleep(1) # Espera um pouco para evitar problemas de concorrência com o modelo YOLO
                                bibs = self.detect_bibs(person_crop["img"])
                        record["shoes"] = shoes
                        record["bib"] = bibs[0] if bibs else None
                        record["demographic"] = self._pick_primary_face(
                            faces=demographic_data_list,     # your face detections
                            shoes=shoes,                     # the shoe list you just built
                            person_bbox=record["bbox"]       # bbox of the runner
                        )

                        if shoes:
                            all_records_data.append(record)
                            batch_records.append(record)
                    
                

            if batch_records:
                batch_df = pd.DataFrame(batch_records)
                df_full = pd.concat([df_full, batch_df], ignore_index=True)
                processed.update(batch_df["filename"].astype(str).tolist())
                if output_path:
                    try:
                        df_full.to_json(output_path, default_handler=str)
                    except Exception as e:
                        logger.error(f"Erro ao salvar dados parciais em {output_path}: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if cancel_event is not None and cancel_event.is_set():
                logger.info("Processamento cancelado durante o lote.")
                break
        
        logger.info(f"Processamento de {len(df_full)} imagens concluído.")
        return df_full

    def processFolder(
        self,
        folder: str,
        max_images: int | None = None,
        skip_files: set[str] | None = None,
        output_path: str | None = None,
        resume_df: pd.DataFrame | None = None,
        cancel_event: 'threading.Event | None' = None,
    ) -> pd.DataFrame:
        """
        Alias para process_images, mantendo consistência se usado externamente.
        """
        logger.info(f"Processando pasta (via processFolder alias): {folder}")
        return self.process_images(
            folder,
            max_images=max_images,
            skip_files=skip_files,
            output_path=output_path,
            resume_df=resume_df,
            cancel_event=cancel_event,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise de marcas de calçados e demografia em imagens.")
    parser.add_argument("folder", help="Pasta contendo as imagens a serem analisadas.")
    parser.add_argument("--output", default="output/processed_dataset.json", help="Pasta para salvar os resultados (dataset.json, tasks).")
    parser.add_argument(
        "--bib-categories",
        help="Lista de categorias separadas por vírgula para sobrescrever o config.",
    )
    parser.add_argument(
        "--bib-colours",
        help="Texto adicional mapeando cores para categorias a ser enviado ao modelo.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Processa apenas as N primeiras imagens encontradas.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continua o processamento carregando o JSON de saída existente.",
    )
    
    args = parser.parse_args()

    if args.folder:
        analyzer = ShoesAIAnalyzer()

        existing_df = None
        skip_files: set[str] = set()
        if args.resume and Path(args.output).is_file():
            try:
                existing_df = pd.read_json(args.output)
                if "filename" in existing_df.columns:
                    skip_files = set(existing_df["filename"].astype(str).tolist())
                logger.info(f"Retomando processamento a partir de {len(skip_files)} registros.")
            except Exception as e:
                logger.error(f"Erro ao carregar JSON existente {args.output}: {e}")

        df_processed = analyzer.processFolder(
            args.folder,
            max_images=args.max_images,
            skip_files=skip_files,
            output_path=args.output,
            resume_df=existing_df,
        )

        df_output_path = args.output
        try:
            df_processed.to_json(df_output_path, default_handler=str)
            logger.info(f"DataFrame processado salvo em: {df_output_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar DataFrame processado em {df_output_path}: {e}")