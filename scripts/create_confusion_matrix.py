import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import ViTForImageClassification

# ----------------------------- CONFIG -----------------------------
DATASET_DIR  = "~/devel/storage/olympikus/train/data2004/"
CLASSES_PATH = "models/classify_shoes_classes.txt"
MODEL_PATH   = "models/classify_shoes_model.pth"
BATCH_SIZE   = 32

OUT_COUNTS_CSV = "confusion_counts.csv"
OUT_PROB_CSV   = "confusion_prob.csv"
# ------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_classes(classes_path):
    with open(classes_path) as f:
        classes = [line.strip() for line in f if line.strip()]

    return classes

def normalize_confusion_matrix(cm):
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_prob  = np.divide(cm_counts, row_sums,
                        where=row_sums != 0)   # evita divisão por zero
    return cm_prob

def generateConfusionMatrix(dataset_dir, classes, model_path):
    # ---------- 2. Dataset ----------
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root=DATASET_DIR, transform=val_transform)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---------- 3. Modelo ----------
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(classes),
        ignore_mismatched_sizes=True,
    )
    # substitui a cabeça
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(model.classifier.in_features, len(classes))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    # ---------- 4. Coletar predições ----------
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Inferência"):
            imgs   = imgs.to(device)
            labels = labels.to(device)
            outputs = model(pixel_values=imgs).logits
            preds   = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # ---------- 5. Matriz de confusão ----------
    cm_counts = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    
    return cm_counts


from argparse import ArgumentParser

if __name__ == "__main__":
    # ---------- 1. Argumentos ----------
    parser = ArgumentParser(description="Gera matriz de confusão")
    parser.add_argument("--dataset_dir", type=str, help="Diretório do dataset")
    parser.add_argument("--classes_path", type=str, default=CLASSES_PATH,
                        help="Caminho do arquivo de classes")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Caminho do modelo")
    parser.add_argument("--output", type=str, default=OUT_COUNTS_CSV,
                        help="Caminho do arquivo de saída")
    parser.add_argument("--in_confusion", type=str)
    args = parser.parse_args()

    classes = load_classes(args.classes_path)
    
    if args.in_confusion:
        #read confusion matrix
        cm = pd.read_csv(args.in_confusion, index_col=0).values

        y_true, y_pred = [], []

        for true_idx in range(cm.shape[0]):
            for pred_idx in range(cm.shape[1]):
                n = cm[true_idx, pred_idx]
                if n > 0:
                    y_true.extend([true_idx] * n)
                    y_pred.extend([pred_idx] * n)
        print("Relatório de Classificação:")
        print(classification_report(y_true, y_pred, target_names=classes, labels=np.unique(y_true)))
        
    if args.dataset_dir:
        cm_counts = generateConfusionMatrix(
            args.dataset_dir, classes, args.model_path
        )
        cm_prob = normalize_confusion_matrix(cm_counts)
        # ---------- 7. Converter em DataFrame ----------
        df_counts = pd.DataFrame(cm_counts, index=classes, columns=classes)
        df_prob   = pd.DataFrame(cm_prob,   index=classes, columns=classes)

        # ---------- 8. Exportar ----------
        df_counts.to_csv(args.output, float_format="%.0f")
        df_prob.to_csv("prob_"+args.output, float_format="%.6f")