import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from tqdm import tqdm
import numpy as np
import torch
from detector import ShoesAIAnalyzer


analyzer = ShoesAIAnalyzer()
analyzer.init(classify_model_path="../models/classify_model.pth",
                           detect_model_path="../models/detect_model.pt",
                           classes_path="../models/classes.txt")


IMAGES_PATH = "/home/markun/devel/datasets/olympikus/prova0604"
OUTPUT_PATH = IMAGES_PATH + "_processed"

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load YOLO model (use "yolo11n.pt" or your custom model, e.g., YOLO("path/to/best.pt"))
person_model = YOLO("yolo11n.pt")
person_model.to(device)

count = {
    "goodShoes": 0,
    "badShoes": 0
}


def detect_persons_yolo(image_path,confidence=0.8):
    """
    Opens an image and uses YOLO to detect persons (class 0).
    Returns the PIL image and a list of bounding boxes [x1, y1, x2, y2] for each detection.
    """
    # Open image with PIL
    image = Image.open(image_path).convert("RGB")
    
    # Run YOLO detection on the image
    results = person_model.predict(image_path, conf=confidence,iou=0.3,classes=[0],
                            verbose=False)  # Pass the image path to the model
    boxes = []
    
    for result in results:
        for d in result.boxes:
            person = save_one_box(d.xyxy,
                         result.orig_img.copy(),
                         file=Path(OUTPUT_PATH) / f'{Path(image_path).stem}.jpg',
                         BGR=True, save=False)
            shoe_detection = model_shoes.predict(person, verbose=False)
            
            hasShoes = "badShoes"
            if len(shoe_detection) > 0:
                for s in shoe_detection[0].boxes:
                    if s.conf > 0.5:
                        hasShoes = "goodShoes"
                count[hasShoes] += 1
            else:
                count[hasShoes] += 1


            file_path = Path(OUTPUT_PATH) / f'{hasShoes}/{Path(image_path).stem}.jpg'
            if not file_path.exists():
                os.makedirs(file_path.parent, exist_ok=True)  
                save_one_box(d.xyxy,
                         result.orig_img.copy(),
                         file=file_path,
                         BGR=True, save=True)
    return results

def get_predictions(path):
        predictions = []
        regions = []
        img, objects = analyzer.detect_objects(path)
        for object in objects:
            crops = analyzer.crop_images(img, objects)
            orig_width = img.shape[1]
            orig_height = img.shape[0]
            for crop in crops:
                pred = analyzer.classify_image(crop["img"], 0.5)
                x1,y1,x2,y2 = crop["box"]
                x_pct = (x1 / orig_width) * 100
                y_pct = (y1 / orig_height) * 100
                w_pct = ((x2 - x1) / orig_width) * 100
                h_pct = ((y2 - y1) / orig_height) * 100
                
                region = {
                                "from_name": from_name,
                                "id": str(uuid4())[:4],
                                "to_name": to_name,
                                "score": pred["prob"],
                                "readonly": False,
                                "type": "rectanglelabels",
                                "value": {
                                    "height": h_pct,
                                    "rectanglelabels": [
                                        pred["label"]
                                    ],
                                    "rotation": 0,
                                    "width": w_pct,
                                    "x": x_pct,
                                    "y": y_pct
                                }
                            }
                
                regions.append(region)
        
        from pprint import pprint    
        pprint(regions)
        all_scores = [region["score"] for region in regions if "score" in region]
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        
        prediction = {
            "result": regions,
            "model_version": self.get("model_version"),
            "score": avg_score
        }
        predictions.append(prediction)