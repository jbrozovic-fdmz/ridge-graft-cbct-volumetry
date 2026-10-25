#!/usr/bin/env bash
set -euo pipefail

CASE="${1:?case}"; XMIN="${2:?}"; XMAX="${3:?}"; YMIN="${4:?}"; YMAX="${5:?}"; ZMIN="${6:?}"; STEP="${7:-0.2}"
CC="CloudCompare"
DIR="cloudcompare_${CASE}"
U="${DIR}"
BOX="${XMIN}:${YMIN}:${ZMIN}:${XMAX}:${YMAX}:31"
mkdir -p "$DIR"

CORNERS="${DIR}/_grid_corners.asc"
python3 - "$XMIN" "$XMAX" "$YMIN" "$YMAX" "$ZMIN" > "$CORNERS" <<'PYC'
import sys
xmin, xmax, ymin, ymax, zmin = map(float, sys.argv[1:6])
for x in (xmin + 1e-3, xmax - 1e-3):
    for y in (ymin + 1e-3, ymax - 1e-3):
        print(f"{x:.5f} {y:.5f} {zmin + 1e-3:.5f}")
PYC

for m in pre post; do
  "$CC" -SILENT -NO_TIMESTAMP -AUTO_SAVE OFF -C_EXPORT_FMT BIN \
        -O "${U}/${m}_cc.stl" -SAMPLE_MESH DENSITY 400 \
        -O "${U}/_grid_corners.asc" \
        -MERGE_CLOUDS \
        -CROP "$BOX" \
        -RASTERIZE -GRID_STEP "$STEP" -VERT_DIR 2 -PROJ MAX -OUTPUT_CLOUD \
        -SAVE_CLOUDS FILE "${U}/${m}_dem.bin" >/dev/null 2>&1
done

rm -f "${DIR}/VolumeCalculationReport.txt"
"$CC" -SILENT -NO_TIMESTAMP -AUTO_SAVE OFF \
      -O "${U}/post_dem.bin" -O "${U}/pre_dem.bin" \
      -VOLUME -GRID_STEP "$STEP" -VERT_DIR 2 >/dev/null 2>&1

cat "${DIR}/VolumeCalculationReport.txt"
