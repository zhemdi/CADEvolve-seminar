#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Paths to scripts + configs
# -------------------------

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Relative paths w.r.t. the bash script
SCRIPT_DIR="$SCRIPT_ROOT/../data/"
UTILS_DIR="$SCRIPT_ROOT/./utils"

SAMPLING_PY="$UTILS_DIR/canonicalization_run/sampling.py"
CANON_PY="$UTILS_DIR/canonicalization_run/canonicalization.py"
ROT_PY="$UTILS_DIR/rotation_augmentation/mixed_rotation.py"

CFG_SAMPLING="$UTILS_DIR/canonicalization_run/cfg_sampling.yaml"
CFG_CANON="$UTILS_DIR/canonicalization_run/cfg_canonicalization.yaml"
CFG_ROT="$UTILS_DIR/rotation_augmentation/cfg_rotation.yaml"

# -------------------------
# Data root (single source of truth)
# -------------------------

OUT_DIR="$SCRIPT_DIR/results"

# -------------------------
# Helpers
# -------------------------
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

# -------------------------
# Ensure dirs exist
# -------------------------
mkdir -p "$OUT_DIR/logs"

# -------------------------
# 1) Sampling
# -------------------------
log "Step 1: sampling"
(
  cd "$OUT_DIR"
  python "$SAMPLING_PY" --config "$CFG_SAMPLING"
)

# -------------------------
# 2) Canonicalization
# -------------------------
log "Step 2: canonicalization"
(
  cd "$OUT_DIR"
  python "$CANON_PY" --config "$CFG_CANON"
)

# -------------------------
# 3) Flatten canonicalized/binarized -> canonicalized_flat
#    (collect all *.py from nested dirs into a single folder)
# -------------------------
log "Step 3: flatten canonicalized/binarized -> canonicalized_flat"

CANON_DIR="$OUT_DIR/canonicalized/binarized"
FLAT_DIR="$OUT_DIR/canonicalized_flat"

rm -rf "$FLAT_DIR"
mkdir -p "$FLAT_DIR"

while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  dst="$FLAT_DIR/$base"

  if [[ -e "$dst" ]]; then
    suf="$(python - <<PY
import hashlib
print(hashlib.md5("$f".encode("utf-8")).hexdigest()[:8])
PY
)"
    dst="$FLAT_DIR/${base%.py}__${suf}.py"
  fi

  cp -f "$f" "$dst"
done < <(find "$CANON_DIR" -type f -name "*.py" -print0)

log "Flatten done: $(find "$FLAT_DIR" -maxdepth 1 -type f -name "*.py" | wc -l) scripts in canonicalized_flat"

# -------------------------
# 4) Rotation augmentation
# -------------------------
log "Step 4: rotation augmentation"
(
  cd "$OUT_DIR"
  python "$ROTATED_PY" --config "$CFG_ROT"
)

log "Pipeline finished OK"
