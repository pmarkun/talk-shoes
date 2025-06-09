import json
import uuid
from datetime import datetime, timezone
import argparse # For command-line arguments

def convert_to_ls_format(abs_x_min, abs_y_min, abs_x_max, abs_y_max, img_width, img_height):
    """
    Converts absolute pixel coordinates to Label Studio's percentage-based format.
    x, y are top-left corner.
    """
    if img_width == 0 or img_height == 0:
        return 0, 0, 0, 0 # Avoid division by zero
    x = (abs_x_min / img_width) * 100
    y = (abs_y_min / img_height) * 100
    width = ((abs_x_max - abs_x_min) / img_width) * 100
    height = ((abs_y_max - abs_y_min) / img_height) * 100
    return x, y, width, height

def create_ls_tasks_with_predictions_only(
    input_json_filepath, # Changed from input_data to input_json_filepath
    image_base_url,
    ls_project_id=7,
    model_version_name="ShoesAI_Import",
    prediction_rect_label_default="Validar",
    rect_from_name="shoes",
    choice_from_name="brand",
    to_name="image"
):
    """
    Transforms the input data from a JSON file into a list of Label Studio tasks,
    containing only predictions. Each bounding box will be labeled with
    `prediction_rect_label_default`.
    """
    tasks = []

    try:
        with open(input_json_filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_json_filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_filepath}. Ensure it's valid JSON.")
        return []

    # Check for expected top-level keys to ensure data structure is roughly correct
    required_keys = ["filename", "folder", "original_width", "original_height", "shoes"]
    if not all(key in data for key in required_keys):
        print(f"Error: Input JSON from {input_json_filepath} is missing one or more required keys: {required_keys}")
        return []
    
    # Assuming all top-level dictionaries have the same set of string indices ("0", "1", ...)
    # and that 'filename' is a reliable indicator of the number of items.
    if not data["filename"]:
        print(f"Warning: No filenames found in the input JSON from {input_json_filepath}. No tasks will be generated.")
        return []
        
    num_items = len(data["filename"])

    for i in range(num_items):
        idx_str = str(i)

        # Basic check if the current index exists in all essential parts
        if not all(idx_str in data[key] for key in required_keys):
            print(f"Warning: Data for index '{idx_str}' is incomplete. Skipping this item.")
            continue

        original_w = data["original_width"][idx_str]
        original_h = data["original_height"][idx_str]
        folder = data["folder"][idx_str]
        filename = data["filename"][idx_str]

        image_path_parts = [image_base_url.rstrip('/')]
        if folder:
            image_path_parts.extend(folder.strip('/').split('/'))
        image_path_parts.append(filename.lstrip('/'))
        image_url = "/".join(filter(None, image_path_parts))

        prediction_results_list = []
        current_task_id = i + 1
        current_prediction_obj_id = i + 10001 # Arbitrary offset for prediction object ID

        prediction_score = 0.0
        shoes_data_for_item = data["shoes"].get(idx_str, []) # Gracefully handle missing shoe data for an index
        
        if shoes_data_for_item:
            first_shoe_info = shoes_data_for_item[0]
            if "confidence" in first_shoe_info:
                 prediction_score = first_shoe_info["confidence"]
            elif "prob" in first_shoe_info and first_shoe_info["prob"]:
                 prediction_score = first_shoe_info["prob"][0]

        for shoe_info in shoes_data_for_item:
            # Ensure essential keys are present in shoe_info
            if not ("bbox" in shoe_info and shoe_info["bbox"] and "label" in shoe_info):
                print(f"Warning: Incomplete shoe data for index '{idx_str}'. Skipping this shoe entry.")
                continue
            
            shoe_bbox = shoe_info["bbox"][0]
            brand_labels = shoe_info["label"]
            brand_label = brand_labels[0] if brand_labels else "Desconhecido"

            ls_x, ls_y, ls_width, ls_height = convert_to_ls_format(
                shoe_bbox[0], shoe_bbox[1], shoe_bbox[2], shoe_bbox[3],
                original_w, original_h
            )

            region_id = uuid.uuid4().hex[:10]

            prediction_rect_item = {
                "type": "rectanglelabels", "id": region_id, "from_name": rect_from_name, "to_name": to_name,
                "original_width": original_w, "original_height": original_h, "image_rotation": 0,
                "value": {"rotation": 0, "x": ls_x, "y": ls_y, "width": ls_width, "height": ls_height,
                          "rectanglelabels": [prediction_rect_label_default]}
            }
            prediction_choice_item = {
                "type": "choices", "id": region_id, "from_name": choice_from_name, "to_name": to_name,
                "original_width": original_w, "original_height": original_h, "image_rotation": 0,
                "value": {"x": ls_x, "y": ls_y, "width": ls_width, "height": ls_height,
                          "choices": [brand_label]}
            }
            prediction_results_list.extend([prediction_rect_item, prediction_choice_item])

        now_iso_for_prediction = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        prediction_object = {
            "id": current_prediction_obj_id, "model_version": model_version_name,
            "created_at": now_iso_for_prediction, "result": prediction_results_list,
            "score": prediction_score, "project": ls_project_id, "task": current_task_id
        }

        task = {
            "id": current_task_id,
            "data": {"image": image_url},
            "predictions": [prediction_object],
            "project": ls_project_id
        }
        tasks.append(task)

    return tasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert custom JSON to Label Studio tasks format (predictions only).")
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("output_file", help="Path for the output Label Studio JSON tasks file.")
    parser.add_argument("--image_base_url", required=True,
                        help="Base URL for the images (e.g., 'gs://your-bucket/', 'http://localhost:8080/images/').")
    parser.add_argument("--project_id", type=int, default=7,
                        help="Label Studio project ID (default: 7).")
    parser.add_argument("--model_version", default="ShoesAI_Import_Script",
                        help="Model version string for predictions (default: ShoesAI_Import_Script).")
    parser.add_argument("--rect_label", default="Validar",
                        help="Default label for rectangle predictions (default: Validar).")
    parser.add_argument("--rect_from_name", default="shoes",
                        help="The 'from_name' for rectangle labels in LS config (default: shoes).")
    parser.add_argument("--choice_from_name", default="brand",
                        help="The 'from_name' for choices in LS config (default: brand).")
    parser.add_argument("--to_name", default="image",
                        help="The 'to_name' for image tagging in LS config (default: image).")

    args = parser.parse_args()

    # --- Generate Label Studio tasks ---
    ls_tasks = create_ls_tasks_with_predictions_only(
        args.input_file,
        args.image_base_url,
        ls_project_id=args.project_id,
        model_version_name=args.model_version,
        prediction_rect_label_default=args.rect_label,
        rect_from_name=args.rect_from_name,
        choice_from_name=args.choice_from_name,
        to_name=args.to_name
    )

    if ls_tasks:
        # --- Save to output file ---
        try:
            with open(args.output_file, 'w') as f:
                json.dump(ls_tasks, f, indent=4)
            print(f"Successfully transformed data and saved to {args.output_file}")
            print(f"Generated {len(ls_tasks)} tasks.")
        except IOError:
            print(f"Error: Could not write to output file {args.output_file}")
    else:
        print("No tasks were generated. Please check input file and logs.")

    # Example of how to run from the command line:
    # python your_script_name.py input_data.json output_ls_tasks.json --image_base_url "gs://my-image-bucket"