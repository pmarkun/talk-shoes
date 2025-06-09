import argparse
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch # For torch.cuda.empty_cache()
import sys
import re # For sanitizing folder names

#insert parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from detector import ShoesAIAnalyzer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_shoe_crops(input_folder: Path,
                    output_folder: Path,
                    config_path: Path,
                    detection_confidence_threshold: float,
                    classification_confidence_threshold: float, # Novo parâmetro
                    batch_size: int = 8,
                    shoe_class_id: int = 0):
    """
    Iterates through images, detects shoes, classifies them, and saves crops
    to folders based on classification and confidence.
    """
    if not input_folder.is_dir():
        logger.error(f"Input folder {input_folder} does not exist.")
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Detection confidence threshold: {detection_confidence_threshold}")
    logger.info(f"Classification confidence threshold: {classification_confidence_threshold}") # Log novo param
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Shoe class ID for cropping (detection): {shoe_class_id}")

    try:
        analyzer = ShoesAIAnalyzer(config_path=str(config_path))
        logger.info("ShoesAIAnalyzer initialized successfully.")
        if not analyzer.classes:
            logger.warning("Shoe classification classes are not loaded. Classification might not work as expected.")
    except Exception as e:
        logger.error(f"Failed to initialize ShoesAIAnalyzer: {e}")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted([p for p in input_folder.rglob("*") if p.suffix.lower() in image_extensions])

    if not image_paths:
        logger.info("No images found in the input folder.")
        return

    logger.info(f"Found {len(image_paths)} images to process.")

    total_crops_saved = 0
    unknown_folder_name = "Unknown_Classification" # Pasta para classificações de baixa confiança

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing image batches"):
        batch_paths = image_paths[i : i + batch_size]
        
        pil_images_data = []
        for img_path in batch_paths:
            pil_img = analyzer._open_image(img_path)
            if pil_img:
                pil_images_data.append({"path": img_path, "pil_image": pil_img, "np_image": np.array(pil_img)})
            else:
                logger.warning(f"Skipping {img_path} due to loading error.")

        if not pil_images_data:
            continue

        np_images_for_yolo = [data["np_image"] for data in pil_images_data]

        try:
            batch_yolo_results = list(analyzer.detect_shoes_model(np_images_for_yolo, stream=True, verbose=False))
        except Exception as e:
            logger.error(f"Error during batch shoe detection: {e}")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        for idx, data in enumerate(pil_images_data):
            original_path = data["path"]
            full_np_image = data["np_image"]
            yolo_results_for_image = [batch_yolo_results[idx]]

            try:
                detected_shoe_crops_data = analyzer.crop_images(
                    full_np_image,
                    yolo_results_for_image,
                    classes_to_crop=[shoe_class_id], # Crop apenas a classe de tênis
                    confidence=detection_confidence_threshold # Confiança da DETECÇÃO
                )
            except Exception as e:
                logger.error(f"Error cropping shoes for {original_path}: {e}")
                continue

            if detected_shoe_crops_data:
                # Coleta todas as imagens de crops de tênis para classificar em lote
                pil_shoe_images_for_classification = [crop_data["img"] for crop_data in detected_shoe_crops_data]

                if pil_shoe_images_for_classification:
                    try:
                        classification_results = analyzer.classify_batch(pil_shoe_images_for_classification)
                    except Exception as e:
                        logger.error(f"Error during batch shoe classification for {original_path}: {e}")
                        classification_results = [{"label": "ErrorInClassification", "prob": 0.0}] * len(pil_shoe_images_for_classification)


                    # Agora, itere sobre os crops detectados e seus resultados de classificação
                    for crop_idx, (crop_data, class_res) in enumerate(zip(detected_shoe_crops_data, classification_results)):
                        crop_pil_image = crop_data["img"]
                        predicted_label = class_res.get("label", "LabelError")
                        classification_prob = class_res.get("prob", 0.0)

                        # Determina a pasta de destino com base na confiança da CLASSIFICAÇÃO
                        if classification_prob >= classification_confidence_threshold:
                            target_subfolder_name = predicted_label
                        else:
                            target_subfolder_name = unknown_folder_name
                        
                        relative_image_path = original_path.relative_to(input_folder)
                        # Mantém a estrutura de subpastas da imagem original dentro da pasta da classe
                        final_crop_output_dir = output_folder / target_subfolder_name / relative_image_path.parent
                        final_crop_output_dir.mkdir(parents=True, exist_ok=True)

                        base_name = original_path.stem
                        # Adiciona info da classe e prob ao nome do arquivo para fácil identificação (opcional)
                        # class_prob_str = f"{sanitize_foldername(predicted_label)}_{classification_prob:.2f}"
                        # crop_filename = f"{base_name}_shoe_{crop_idx}_{class_prob_str}.png"
                        crop_filename = f"{base_name}_shoe_{crop_idx}.png" # Nome mais simples
                        
                        output_crop_path = final_crop_output_dir / crop_filename
                        try:
                            crop_pil_image.save(output_crop_path)
                            total_crops_saved += 1
                        except Exception as e:
                            logger.error(f"Failed to save crop {output_crop_path}: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Processing complete. Total shoe crops saved: {total_crops_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect, classify shoes in images, and save crops to class-specific folders.")
    parser.add_argument("input_folder", type=str,
                        help="Folder containing images to process (searches recursively).")
    parser.add_argument("output_folder", type=str,
                        help="Folder where shoe crops will be saved in class-specific subfolders.", default='output')
    parser.add_argument("--config_path", type=str, default="config.yaml",
                        help="Path to the configuration YAML file for ShoesAIAnalyzer.")
    parser.add_argument("--detection_confidence", type=float, default=0.5, # Renomeado para clareza
                        help="Minimum confidence threshold for DETECTING a shoe.")
    parser.add_argument("--classification_confidence", type=float, default=0.7, # Novo argumento
                        help="Minimum confidence threshold for CLASSIFICATION to save in a specific class folder; otherwise, saved in 'Unknown'.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of images to process in a single batch for detection.")
    parser.add_argument("--shoe_class_id", type=int, default=0,
                        help="Class ID representing 'shoe' in the YOLO detection model.")

    args = parser.parse_args()

    input_p = Path(args.input_folder)
    output_p = Path(args.output_folder)
    config_p = Path(args.config_path)

    save_shoe_crops(input_folder=input_p,
                    output_folder=output_p,
                    config_path=config_p,
                    detection_confidence_threshold=args.detection_confidence,
                    classification_confidence_threshold=args.classification_confidence,
                    batch_size=args.batch_size,
                    shoe_class_id=args.shoe_class_id)