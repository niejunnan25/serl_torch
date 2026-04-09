#!/usr/bin/env bash

codex_try_init_conda() {
    if [[ "${_CODEX_CONDA_INITIALIZED:-0}" == "1" ]]; then
        return 0
    fi

    if command -v conda >/dev/null 2>&1; then
        local conda_hook
        conda_hook="$(conda shell.bash hook 2>/dev/null)" || conda_hook=""
        if [[ -n "$conda_hook" ]]; then
            eval "$conda_hook"
            _CODEX_CONDA_INITIALIZED=1
            return 0
        fi
    fi

    if [[ -n "${CONDA_EXE:-}" ]]; then
        local conda_root conda_sh
        conda_root="$(cd "$(dirname "$(dirname "$CONDA_EXE")")" && pwd)"
        conda_sh="$conda_root/etc/profile.d/conda.sh"
        if [[ -f "$conda_sh" ]]; then
            # shellcheck source=/dev/null
            source "$conda_sh"
            _CODEX_CONDA_INITIALIZED=1
            return 0
        fi
    fi

    return 1
}

codex_activate_conda() {
    local requested_prefix="${1:-}"
    local requested_name="${2:-}"
    shift 2 || true
    local fallback_names=("$@")

    if [[ -n "${CONDA_PREFIX:-}" && -z "$requested_prefix" && -z "$requested_name" ]]; then
        return 0
    fi

    if ! codex_try_init_conda; then
        if [[ -n "$requested_prefix" || -n "$requested_name" ]]; then
            echo "ERROR: conda is not available in this shell, but an env override was requested." >&2
            return 1
        fi
        return 0
    fi

    if [[ -n "$requested_prefix" ]]; then
        conda activate "$requested_prefix"
        return 0
    fi

    if [[ -n "$requested_name" ]]; then
        conda activate "$requested_name"
        return 0
    fi

    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        return 0
    fi

    local fallback_name
    for fallback_name in "${fallback_names[@]}"; do
        if [[ -n "$fallback_name" ]] && conda activate "$fallback_name" >/dev/null 2>&1; then
            return 0
        fi
    done

    return 0
}

codex_python_bin() {
    local preferred="${1:-python}"
    if command -v "$preferred" >/dev/null 2>&1; then
        printf '%s\n' "$preferred"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        printf 'python3\n'
        return 0
    fi
    printf '%s\n' "$preferred"
}
