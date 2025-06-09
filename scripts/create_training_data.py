import os
import json
from PIL import Image
from io import BytesIO
from argparse import ArgumentParser
from tqdm import tqdm

def process_annotations_from_gcs(json_path, output_dir, local_dir=None):
    """
    Processa anotações em um arquivo JSON, corta a imagem com base nas bounding boxes
    e salva os crops em pastas organizadas por marca. As imagens são carregadas diretamente
    de um bucket do Google Cloud Storage.

    Args:
        json_path (str): Caminho para o arquivo JSON contendo as anotações.
        output_dir (str): Diretório onde os crops serão salvos.
        local_dir (str): Diretório local onde as imagens estão armazenadas.
    Retorna:
        None
    """
    # Inicializar o cliente do Google Cloud Storage
    if not local_dir:  
        pass#client = storage.Client()

    # Carregar o JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
        print(data)
    
        # Iterar pelas anotações
        for pic in tqdm(data):
            # Verificar se a anotação contém "choices" (marca)
            for annotation in pic["annotations"]:
                for i,result in enumerate(annotation["result"]):
                    if "value" in result and "choices" in result["value"]:
                        # Obter a marca (choice label)
                        brand = result["value"]["choices"][0]
                        if brand == "Desconhecido":
                            continue

                        # Obter as coordenadas da bounding box
                        x = result["value"]["x"]
                        y = result["value"]["y"]
                        width = result["value"]["width"]
                        height = result["value"]["height"]

                        # Converter coordenadas relativas para coordenadas absolutas
                        img_width = result["original_width"]
                        img_height = result["original_height"]
                        abs_x = int(x / 100 * img_width)
                        abs_y = int(y / 100 * img_height)
                        abs_width = int(width / 100 * img_width)
                        abs_height = int(height / 100 * img_height)

                        # Obter o caminho da imagem no bucket
                        image_path = pic["data"]["image"]  # Certifique-se de que o JSON contém o caminho da imagem no bucket
                        if local_dir:
                            file_path = local_dir + image_path.replace("gs://","").split("/", 1)[1]
                            
                            if os.path.exists(file_path):
                                image = Image.open(file_path)
                                #rotate image from exif
                                exif = image._getexif()
                                if exif is not None:
                                    orientation = exif.get(274)
                                    if orientation == 3:
                                        image = image.rotate(180, expand=True)
                                    elif orientation == 6:
                                        image = image.rotate(270, expand=True)
                                    elif orientation == 8:
                                        image = image.rotate(90, expand=True)
                                image_filename = os.path.basename(file_path)
                            else:
                                continue
                        else:
                            bucket_name = image_path.split("/")[2]  # Exemplo: gs://bucket_name/path/to/image.jpg
                            bucket = client.get_bucket(bucket_name)
                            blob = bucket.blob(image_path)

                            # Fazer o download da imagem diretamente para a memória
                            image_data = blob.download_as_bytes()
                            image = Image.open(BytesIO(image_data))

                        # Cortar a imagem
                        crop = image.crop((abs_x, abs_y, abs_x + abs_width, abs_y + abs_height))

                        # Criar a pasta para a marca, se não existir
                        brand_dir = os.path.join(output_dir, brand)
                        os.makedirs(brand_dir, exist_ok=True)

                        # Salvar o crop na pasta correspondente
                        crop_filename = f"crop_{i}_{image_filename}"
                        crop.save(os.path.join(brand_dir, crop_filename))

    print(f"Processamento concluído. Crops salvos em: {output_dir}")

if __name__ == "__main__":
    """
    Função principal para processar as anotações do Label Studio e salvar os crops.
    """

    parser = ArgumentParser(description="Processa anotações de imagens do Label Studio e salva crops em pastas por marca.")
    parser.add_argument("json_path", help="Caminho para o arquivo JSON com as anotações.")
    parser.add_argument("output_dir", help="Diretório onde os crops serão salvos.")
    parser.add_argument("--local_dir", help="Diretório local onde as imagens estão armazenadas (opcional).")
    args = parser.parse_args()
    process_annotations_from_gcs(args.json_path, args.output_dir, args.local_dir)