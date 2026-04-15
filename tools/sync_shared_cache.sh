#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
MODE="${1:-all}"

SHARED_CACHE_ROOT="/vla/users/niejunnan/.cache"
USER_CACHE_ROOT="${HOME}/.cache"
CACHE_ITEMS=("uv" "huggingface")

usage() {
  cat <<EOF
Usage:
  bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/${SCRIPT_NAME} [copy|link|all]

Modes:
  copy  Copy current user's caches from ${USER_CACHE_ROOT} to ${SHARED_CACHE_ROOT}
  link  Link current user's caches in ${USER_CACHE_ROOT} to ${SHARED_CACHE_ROOT}
  all   copy + link
EOF
}

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

copy_cache_dir() {
  local name="$1"
  local src="${USER_CACHE_ROOT}/${name}"
  local dst="${SHARED_CACHE_ROOT}/${name}"

  if [[ ! -d "$src" ]]; then
    echo "[copy] skip '${name}': source not found: ${src}"
    return 0
  fi

  mkdir -p "$dst"

  if command -v rsync >/dev/null 2>&1; then
    echo "[copy] rsync ${src}/ -> ${dst}/"
    rsync -a --delete --info=progress2 "${src}/" "${dst}/"
  else
    echo "[copy] rsync not found, fallback to cp -a ${src}/. -> ${dst}/"
    cp -a "${src}/." "${dst}/"
  fi
}

link_cache_dir() {
  local name="$1"
  local src="${SHARED_CACHE_ROOT}/${name}"
  local dst="${USER_CACHE_ROOT}/${name}"

  if [[ ! -d "$src" ]]; then
    echo "[link] ERROR: shared cache not found: ${src}"
    return 1
  fi

  mkdir -p "$USER_CACHE_ROOT"

  if [[ -L "$dst" ]]; then
    rm -f "$dst"
  elif [[ -e "$dst" ]]; then
    local bak="${dst}.bak.$(timestamp)"
    echo "[link] move existing ${dst} -> ${bak}"
    mv "$dst" "$bak"
  fi

  ln -s "$src" "$dst"
  echo "[link] ${dst} -> ${src}"
}

do_copy() {
  mkdir -p "$SHARED_CACHE_ROOT"
  for item in "${CACHE_ITEMS[@]}"; do
    copy_cache_dir "$item"
  done
}

do_link() {
  for item in "${CACHE_ITEMS[@]}"; do
    link_cache_dir "$item"
  done
}

case "$MODE" in
  copy)
    do_copy
    ;;
  link)
    do_link
    ;;
  all)
    do_copy
    do_link
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "ERROR: unknown mode '${MODE}'"
    usage
    exit 1
    ;;
esac

echo "Done."
