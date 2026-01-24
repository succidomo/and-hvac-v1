#!/usr/bin/env bash
set -euo pipefail

echo "ENTRYPOINT START"
echo "whoami: $(whoami)"
echo "pwd: $(pwd)"

# Find EnergyPlus root directory reliably
EPLUS_BIN="$(command -v energyplus || command -v EnergyPlus || true)"
if [ -n "${EPLUS_BIN}" ]; then
  EPLUS_DIR="$(dirname "${EPLUS_BIN}")"
else
  # Fallback to common install patterns in NREL images
  EPLUS_DIR="$(ls -d /EnergyPlus-* /usr/local/EnergyPlus-* 2>/dev/null | head -n 1 || true)"
fi

echo "EPLUS_BIN: ${EPLUS_BIN:-<none>}"
echo "EPLUS_DIR: ${EPLUS_DIR:-<none>}"

if [ -z "${EPLUS_DIR}" ] || [ ! -d "${EPLUS_DIR}" ]; then
  echo "ERROR: Could not locate EnergyPlus install directory."
  echo "Try: docker run --rm -it eplus-hello bash -lc 'ls -la /; which energyplus; echo \$PATH'"
  exit 1
fi

# Make Python find pyenergyplus
export PYTHONPATH="${EPLUS_DIR}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${EPLUS_DIR}:${LD_LIBRARY_PATH:-}"

echo "PYTHONPATH: ${PYTHONPATH}"
echo "Running python..."
python3 /app/run_sim_train.py "$@"
