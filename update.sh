#!/usr/bin/env bash
# Refresh the data and (optionally) open the page.
#   ./update.sh            rebuild current season
#   ./update.sh --open     rebuild, then open site/index.html
#   ./update.sh --season 2025 --no-download   (any build_data.py flags pass through)
set -euo pipefail
cd "$(dirname "$0")"

OPEN=0; ARGS=()
for a in "$@"; do [[ "$a" == "--open" ]] && OPEN=1 || ARGS+=("$a"); done

PY=$(command -v python3 || command -v python) || { echo "Python 3 is required."; exit 1; }

# The venv stores absolute paths, so it breaks if this folder is moved or copied.
# Recreate it whenever it can't import its packages.
if ! .venv/bin/python -c "import pandas, pyarrow, numpy" 2>/dev/null; then
  echo "Setting up local Python environment in .venv…"
  rm -rf .venv
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

.venv/bin/python build_data.py ${ARGS[@]+"${ARGS[@]}"}
.venv/bin/python make_standalone.py >/dev/null   # site/NFL-EPA.html: one file to share, no setup needed

if [[ $OPEN == 1 ]]; then
  case "$(uname)" in
    Darwin) open site/index.html ;;
    Linux) xdg-open site/index.html >/dev/null 2>&1 || true ;;
  esac
fi
