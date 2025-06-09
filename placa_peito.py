import base64
import os
from google import genai
from google.genai import types
import easyocr
import yaml
import cv2

#load the configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
    gemini_api_key = config.get("gemini_api_key", os.environ.get("GEMINI_API_KEY"))

reader = easyocr.Reader(['pt', 'en'], gpu=True)  # Inicializa o OCR com suporte a GPU

PROMPT = """This is a label for a runner. Extract runner number and run category."""
HELPER = """You can use the following mapping colour -> category:"""

def extract_ocr(image):
    raw = reader.readtext(
        image,
        detail=1,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=1280,
        mag_ratio=1.0,
        slope_ths=0.1,
        ycenter_ths=0.3,
        height_ths=0.2,
        width_ths=0.2,
        add_margin=0.01,
        min_size=5,
        paragraph=False
    )
    
    results = []
    for detection in raw:
        text = detection[1]
        bbox = detection[0]
        #infer font size from bbox
        font_size = int((bbox[2][0] - bbox[0][0]) * 0.5)  # Aproximar tamanho da fonte
        results.append({
            "text": text,
            "bbox": bbox,
            "font_size": font_size
        })
    return results


def extract_run_data(image, category_list, colours=None):
    """Extracts the run data from the image and returns it as a dictionary."""
    # Convert numpy array to bytes in memory
    is_success, buffer = cv2.imencode(".jpg", image)
    if not is_success:
        return {"error": "Failed to encode image"}
    
    # Get bytes directly from the buffer
    image_bytes = buffer.tobytes()

    prompt = PROMPT
    if colours:
        prompt += "\n" + HELPER + "\n" + colours

    client = genai.Client(
        api_key=gemini_api_key,
    )

    model = "gemini-2.0-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=image_bytes,
            ),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=100,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            properties = {
                "number": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "category": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    enum = category_list,
                ),
            },
        ),
    )

    result = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    return result.parsed