import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import io
import base64
from placa_peito import extract_ocr, extract_run_data

# Carregar o modelo YOLOv8

model = YOLO("models/yolo-peito.pt")

# Lista de classes para detecção
classes = [
    "Peito",
]


# Função para processar a imagem e realizar a detecção
def process_image(image):
    image = np.array(image)
    results = model(image)
    coords = results[0].boxes.xyxy.cpu().numpy()  # Coordenadas das caixas delimitadoras
    scores = results[0].boxes.conf.cpu().numpy()  # Confiança das detecções
    class_ids = results[0].boxes.cls.cpu().numpy()  # IDs das classes detectadas
    img_with_boxes = image.copy()
    crops = []
    for i in range(len(coords)):
            x1, y1, x2, y2 = map(int, coords[i])  # Converter para inteiros
            score = scores[i]
            class_id = int(class_ids[i])
            class_name = classes[class_id] if class_id < len(classes) else "Desconhecido"
            
            # Desenhar retângulo
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Adicionar texto
            label = f"{class_name}: {score:.2f}"
            cv2.putText(img_with_boxes, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # Extrair a região de interesse (ROI)
            crop = image[y1:y2, x1:x2]
            crops.append(crop)

    # Converter a imagem com caixas para o formato PIL
    detections = {
        "boxes": coords,
        "scores": scores,
        "class_ids": class_ids,
        "img": img_with_boxes,
        "crops": crops
    }       
    return detections
        
st.set_page_config(layout="wide")
st.title("Teste de Yolo")
st.write("Faça upload de uma imagem para detectar e classificar as regiões de interesse.")

# Upload da imagem
with st.sidebar:
    uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png", "webp"])
    categories = st.text_input("Provas (separadas por vírgula)", "5K, 21K, 42K")
    categories = [cat.strip() for cat in categories.split(",") if cat.strip()]
    colours = st.text_area("Informações adicionais para o modelo (opcional) ex: azul -> 42km")

col1, col2 = st.columns(2)

with col1:
    if uploaded_file is not None:
        # Ler a imagem carregada
        # Converter os bytes da imagem para um objeto PIL Image
        image = uploaded_file.read()

        pil_image = Image.open(io.BytesIO(image))
        #rotacionar a imagem se necessário de acordo com a orientação EXIF
        #checa se a imagem tem metadados EXIF
        if hasattr(pil_image, '_getexif'):
            exif = pil_image._getexif()
            if exif is not None:
                orientation = exif.get(274)
                if orientation == 3:
                    pil_image = pil_image.rotate(180, expand=True)
                elif orientation == 6:
                    pil_image = pil_image.rotate(270, expand=True)
                elif orientation == 8:
                    pil_image = pil_image.rotate(90, expand=True)

        # Processar a imagem e obter as detecções
        detections = process_image(pil_image)
        
        # Exibir os resultados
        st.image(detections["img"], caption="Imagem com Detecções", width=400)
    else:
        st.warning("Por favor, faça upload de uma imagem.")

with col2:
    if uploaded_file is not None:
        st.write("Detecções:")
        for i in range(len(detections["boxes"])):
            #exibir crops
            st.image(detections["crops"][i], caption=f"Crop {i+1}", width=200)
            #extrair texto da imagem
            # Inicializa OCR
            
            results = extract_run_data(
                detections["crops"][i],
                categories,
                colours=colours if colours else None
            )
            #results é um json { "number": "123", "category": "5K" }

            #exibir o texto extraído de cada crop de forma visual
            if "error" in results:
                st.error(f"Erro ao processar a imagem: {results['error']}")
            else:
                st.write(f"Número do corredor: {results.get('number', 'Desconhecido')}")
                st.write(f"Categoria da corrida: {results.get('category', 'Desconhecida')}")

            box = detections["boxes"][i]
            score = detections["scores"][i]
            class_id = int(detections["class_ids"][i])
            class_name = classes[class_id] if class_id < len(classes) else "Desconhecido"
            #st.write(f"Classe: {class_name}, Confiança: {score:.2f}, Caixa: {box}")