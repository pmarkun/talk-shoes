# Detecção e Classificação de Tênis

Este projeto utiliza **YOLO** e **Vision Transformer (ViT)** para detectar e classificar tênis em imagens. A aplicação é construída com **Streamlit**, permitindo uma interface interativa para upload de imagens e visualização dos resultados.

## Estrutura do Projeto
.
├── .streamlit/
│   └── config.toml        # Configuração de tema do Streamlit
├── app.py               # Aplicação principal em Streamlit
├── detector.py                # Classe ShoeDetector para detecção e classificação
├── models/
│   ├── vit16_shoesclassify.pth    # Modelo ViT treinado para classificação
│   ├── vit16_shoesclassify.txt # Classes do modelo ViT
│   └── yolo_shoescrop.pth         # Modelo YOLO para detecção de tênis
├── static/
│   └── 3270NerdFont-Regular.ttf   # Fonte customizada para labels

## Requisitos

Certifique-se de quet em git-lfs instalado - os arquivos de modelo estão hospedados lá 

    apt-get install git-lfs
    git lfs install
Certifique-se de ter o Python 3.8+ instalado. As dependências podem ser instaladas com:

    pip install -r requirements.txt


### Principais Dependências

- `streamlit`
- `torch`
- `transformers`
- `ultralytics`
- `Pillow`
- `numpy`

## Como Executar

1. Certifique-se de que os modelos estão no diretório `models/`:
   - `yolo_shoescrop.pth`: Modelo YOLO para detecção.
   - `vit16_shoesclassify.pth`: Modelo ViT para classificação.
   - `vit16_shoesclassify.txt`: Arquivo com as classes do modelo ViT.

2. Execute o aplicativo Streamlit:

    streamlit run app.py

3. Acesse o aplicativo no navegador em `http://localhost:8501`.

## Funcionalidades

- **Upload de Imagens**: Faça upload de imagens no formato JPG, JPEG, PNG ou WEBP.
- **Detecção de Tênis**: O modelo YOLO detecta regiões de interesse (bounding boxes) contendo tênis.
- **Classificação de Tênis**: O modelo ViT classifica os tênis detectados em uma das seguintes categorias:
  - Adidas
  - Asics
  - Fila
  - Inconclusivo
  - Mizuno
  - New Balance
  - Nike
  - Olympikus
  - Outro
- **Visualização**:
  - Imagem com bounding boxes e labels.
  - Crops das regiões detectadas com a classificação e probabilidade.
- **Estatísticas**:
  - Total de tênis detectados.
  - Distribuição por marca na barra lateral.

## Configuração do Tema

O tema do Streamlit pode ser ajustado no arquivo `.streamlit/config.toml`. Atualmente, está configurado para o tema claro:

    [theme]
    base="light"

## Exemplo de Uso

1. Faça upload de uma imagem.
2. Aguarde o processamento.
3. Visualize:
   - A imagem processada com bounding boxes e labels.
   - As regiões detectadas lado a lado.
   - Estatísticas na barra lateral.

## Créditos

- **YOLO**: Utilizado para detecção de objetos.
- **ViT**: Modelo de classificação baseado em Vision Transformers.
- **Streamlit**: Framework para construção de interfaces interativas.

## Licença

Este projeto é distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.