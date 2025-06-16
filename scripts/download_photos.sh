#!/usr/bin/env bash

set -euo pipefail

# ─── HELP ────────────────────────────────────────────────────────
show_help() {
  echo "Uso:"
  echo "  $0 <link_unico> <destino>"
  echo "  $0 --multiple <arquivo.txt> <destino>"
  echo
  echo "Flags:"
  echo "  --check        Ativa verificação de hash dos arquivos baixados"
  exit 1
}

# ─── PARSE ARGUMENTOS ────────────────────────────────────────────
CHECKSUM=false
MULTIPLE_MODE=false
LIST_FILE=""
DEST_DIR=""
LINK=""

POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --multiple)
      MULTIPLE_MODE=true
      LIST_FILE="$2"
      DEST_DIR="$3"
      shift 3
      ;;
    --check)
      CHECKSUM=true
      shift
      ;;
    -h|--help)
      show_help
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL[@]}"

if ! $MULTIPLE_MODE; then
  LINK="${1:-}"
  DEST_DIR="${2:-}"
  [[ -z "$LINK" || -z "$DEST_DIR" ]] && show_help
fi

[[ -z "$DEST_DIR" ]] && show_help
mkdir -p "$DEST_DIR"

# ─── FUNÇÃO: EXTRAI ID E BAIXA UMA PASTA ─────────────────────────
baixar_pasta() {
  local url="$1"
  local subdir="$2"

  echo "📥 Baixando: $url → $DEST_DIR/$subdir"

  local folder_id
  folder_id=$(echo "$url" | grep -oE 'folders/([^/?]+)' | cut -d/ -f2)

  if [[ -z "$folder_id" ]]; then
    echo "❌ Erro ao extrair o ID da pasta de: $url"
    return 1
  fi

  mkdir -p "$DEST_DIR/$subdir"

  rclone copy \
    "gdrive:" \
    "$DEST_DIR/$subdir" \
    --drive-root-folder-id "$folder_id" \
    --progress \
    --transfers=4 \
    --checkers=8 \
    --retries=10 \
    --low-level-retries=10 \
    --retries-sleep 30s \
    $($CHECKSUM && echo "--checksum")
}

# ─── EXECUÇÃO EM MODO MÚLTIPLO ───────────────────────────────────
if $MULTIPLE_MODE; then
  while IFS=',' read -r url subdir; do
    [[ -z "$url" || -z "$subdir" ]] && continue
    baixar_pasta "$url" "$subdir"
  done < "$LIST_FILE"

# ─── EXECUÇÃO EM MODO ÚNICO ──────────────────────────────────────
else
  baixar_pasta "$LINK" ""
fi
