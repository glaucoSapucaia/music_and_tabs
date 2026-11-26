#!/bin/bash
# Reaplica patch do resampy (necessário após recriar o venv)
# O resampy <=0.4.3 usa pkg_resources, removido do setuptools moderno.
# Este script substitui por importlib.resources.

set -e

# Vai para o diretório raiz do app (pai de scripts/)
cd "$(dirname "$0")/.."

FILTERS_PY=".venv/lib/python3.11/site-packages/resampy/filters.py"

if [ ! -f "$FILTERS_PY" ]; then
    echo "[erro] $FILTERS_PY não encontrado. Rode 'uv sync' primeiro."
    exit 1
fi

if grep -q "pkg_resources" "$FILTERS_PY"; then
    sed -i 's/^import pkg_resources$/from importlib import resources as _importlib_resources/' "$FILTERS_PY"
    sed -i 's/pkg_resources.resource_filename(__name__, fname)/str(_importlib_resources.files(__name__).joinpath(fname))/' "$FILTERS_PY"
    echo "[ok] Patch aplicado em $FILTERS_PY"
else
    echo "[ok] Patch já aplicado (ou não necessário)"
fi
