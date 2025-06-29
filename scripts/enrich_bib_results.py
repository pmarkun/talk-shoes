import json
from pathlib import Path
import sys

#append parent directory to sys.path to import placa_peito
sys.path.append(str(Path(__file__).resolve().parent.parent))
from detector import ShoesAIAnalyzer
from placa_peito import extract_run_data
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
#import multithreading
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from time import sleep

DEBUG = False 
client = ShoesAIAnalyzer(config_path="config.yaml")

base_path = Path("/home/markun/devel/storage/olympikus/prova_poa/")


data = json.load(open("output/processed_dataset_logical.json", "r"))



def crop_person_image(entry):
    filename = base_path / entry["filename"]
    #crop the image to the bounding box in ["demograhics"]["bbox"]
    bbox = entry["bbox"]
    #load the image
    person_img = Image.open(filename)
    # crop the image
    person_img = person_img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    return person_img

def crop_bib_number(person_img, yolo_results):
        """
        Detects the bib number in the given image using YOLO results.
        Yields cropped images of detected bib numbers.
        """
        if not yolo_results:
            return
        for r in yolo_results: # r é um ultralytics.engine.results.Results object
                if r is None or r.boxes is None: # Adicionado para segurança
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls == 0 and conf >= 0.5:
                        #Check if the box is valid
                        if x1 >= x2 or y1 >= y2:
                            continue
                        if x1 < 0 or y1 < 0 or x2 > person_img.size[0] or y2 > person_img.size[1]:
                            continue
                        
                        crop_pil = person_img.crop((x1, y1, x2, y2))
                        
                        if DEBUG:
                            crop_filename = f"tmp/{entry['filename'].split('/')[-1].replace('.jpg', '')}_crop_{i}.jpg"
                            crop_pil.save(crop_filename)
                        
                        #yield result
                        crop = {
                            "img": crop_pil,
                            "box": (x1, y1, x2, y2),
                            "confidence": conf, # Adiciona confiança da detecção
                            "class_detection": cls       # Adiciona classe da detecção YOLO
                        }

                        yield crop

def process_crop(crop, categories, prompt, entry_copy):
    """Process a single crop and return the result with entry data"""
    try:
        result = extract_run_data(np.array(crop["img"]), categories, prompt)
        entry_copy["bib"] = result
        return entry_copy
    except Exception as e:
        sleep(3)
        try:
            result = extract_run_data(np.array(crop["img"]), categories, prompt)
            entry_copy["bib"] = result
        except Exception as e:
            print(f"Error processing crop: {e}")
            return None
        
bibs_data = []
max_workers = 8  # Adjust based on your system and API limits
output_file = Path("output/bibs_data_progressive.json")
already_processed = {}
if output_file.is_file():
    with open(output_file, 'r') as f:
        bibs_data = json.load(f)
        already_processed = {entry['filename']: entry["bbox"] for entry in bibs_data}


with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_entry = {}
    
    for i, entry in enumerate(tqdm(data)):
        if entry["filename"] in already_processed and already_processed[entry["filename"]] == entry["bbox"]:
            if DEBUG:
                print(f"Skipping already processed entry: {entry['filename']}")
            continue
        person_img = crop_person_image(entry)

        yolo_results = client.detect_bib_model(person_img, verbose=False)  
        for crop in crop_bib_number(person_img, yolo_results):
            # Create a copy of entry for each crop to avoid race conditions
            entry_copy = entry.copy()
            
            # Submit the task to the thread pool
            future = executor.submit(
                process_crop, 
                crop, 
                ["5K", "10K", "21K", "42K"], 
                "10K -> Laranja, 21K -> Azul, 42K -> Preta ou Vermelha (desafio 21+42)",
                entry_copy
            )
            future_to_entry[future] = entry_copy
    
    futures_list = list(future_to_entry.keys())
    # Collect results as they complete
    for future in tqdm(as_completed(futures_list), total=len(futures_list)):
        result = future.result()
        if result is not None:
            if DEBUG:
                print(f"Filename: {result['filename']} -> Category: {result['bib'].get('category','Não encontrado')}, Number: {result['bib'].get('number', 'Não encontrado')}")
            bibs_data.append(result)
            with open(output_file, 'w') as f:
                json.dump(bibs_data, f, indent=2, ensure_ascii=False)