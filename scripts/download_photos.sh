#!/usr/bin/env bash

# Uso: ./baixar_drive_folder.sh "https://drive.google.com/drive/folders/ID_DA_PASTA" "/caminho/de/destino"

set -e

# ─── PARÂMETROS ────────────────────────────────────────────────────────
FOLDER_URL="$1"
DEST_DIR="$2"

# ─── VERIFICAÇÕES ───────────────────────────────────────────────────────
if [[ -z "$FOLDER_URL" || -z "$DEST_DIR" ]]; then
  echo "Uso: $0 <URL da pasta do Google Drive> <pasta de destino>"
  exit 1
fi

# ─── EXTRAI O ID DA PASTA ───────────────────────────────────────────────
FOLDER_ID=$(echo "$FOLDER_URL" | grep -oE 'folders/([^/?]+)' | cut -d/ -f2)

if [[ -z "$FOLDER_ID" ]]; then
  echo "Não foi possível extrair o ID da pasta da URL fornecida."
  exit 1
fi

# ─── CRIA A PASTA DE DESTINO, SE NECESSÁRIO ─────────────────────────────
mkdir -p "$DEST_DIR"

# ─── EXECUTA O RCLONE COPY COM SUPORTE A SHARED-WITH-ME + RESUME ────────
rclone copy \
  "gdrive:" \
  "$DEST_DIR" \
  --drive-root-folder-id "$FOLDER_ID" \
  --progress \
  --checkers=8 \
  --transfers=4 \
  --copy-links \
  --create-empty-src-dirs \
  --retries=10 \
  --low-level-retries=10 \
  --retries-sleep 30s

# Dica: use `--log-file=rclone.log` se quiser um log completo

