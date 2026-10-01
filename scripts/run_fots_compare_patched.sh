#!/usr/bin/env bash
# Staged FOTS sim-vs-real marker calibration: SWEEP each model's shape params to its best fit,
# then render the best-fit 2x4 grid  [Real | FOTS | HydroShear | Ours] x [dilate, shear]
# on the same ContactDepth backdrop -> grid_sweep_best.png.
#
#   ours       = ElastomerTaxel (our compressibility + bonded-layer thickness)
#   FOTS       = FOTS's published marker model driven by our ContactDepth
#   HydroShear = the HydroFOTS reference model, ported (scripts/hydroshear_ref.py), on our geometry
#
# Each cell is ONE sweep (--sweep ... --save-markers): the sweep finds its own best fit and writes
# that result straight into the grid npz -- no separate best-fit render step. Only SHAPE params are
# swept; dilate_scale/shear_scale are best-fit analytically. Geometry (ball_radius, depth, centers)
# is NOT identifiable from markers -- set it by hand below. Edit the ranges and re-run.
#
# Caching: the npz is kept across runs; each sweep stores its (geometry+motion+model+grid) signature,
# so an unchanged sweep is SKIPPED (and re-emits its cached BEST line for the staged dilate->shear
# chaining). Pass CLEAN=1 to wipe and recompute everything; --force-markers (FORCE=1) to overwrite.
set -euo pipefail
cd "$(dirname "$0")/.."                        # repo root (dexterous_hands/)
PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs/fots_compare}
mkdir -p "$OUT"
MARKERS="$OUT/markers_sweep.npz"               # every sweep saves its best cell here -> grid_sweep_best.png
[ "${CLEAN:-0}" = 1 ] && rm -f "$MARKERS"
FORCE=$([ "${FORCE:-0}" = 1 ] && echo "--force-markers" || echo "")

# ----- geometry + motion (shared) -------------------------------------------------------------
COMMON="--no-boundary --ball-radius 3.0 --depth 1.5 --depth-query sdf --control-mode position"
DILATE="--frame 0 --motion dilate --center-px 136 124"
SHEAR="--frame 1 --motion shear --shear-start 136 118 0 --shear-stop 133 135 -5"

# ----- sweep ranges (comma-separated; edit these) ---------------------------------------------
# OURS (ElastomerTaxel): dilate = dilation_reg x compressibility (thickness + lambda_d FIXED);
#                        shear  = lambda_s x shear_scale.
LAMBDA_D=${LAMBDA_D:-800000}                                       # fixed local-bulge width (1/m^2; not swept)
OURS_THICKNESS=${OURS_THICKNESS:-4.25}                            # fixed bonded-layer thickness (mm; not swept)
COMPRESSIBILITY=${COMPRESSIBILITY:-"0.6,0.7,0.8,0.9"}            # dilate y: local/global mix in [0,1]
OURS_LAMBDA_S=${OURS_LAMBDA_S:-"3e4,1e5,3e5,1e6"}                # shear x: locality (1/m^2)
OURS_SHEAR_SCALE=${OURS_SHEAR_SCALE:-"1.0,1.25,1.5,1.75,2.0"}    # shear y: shear_scale (shape axis w/ boundary)
# FOTS (1/px^2):
FOTS_LAMBDA_D=${FOTS_LAMBDA_D:-"0.0008,0.00125,0.002,0.003"}
FOTS_LAMBDA_S=${FOTS_LAMBDA_S:-"0.001,0.003,0.01,0.03"}
# HydroShear ported ref (1/m^2). Reference trains lambda_d 11000-22000, lambda_s 7000-13000, mu 0.5-0.75.
HS_LAMBDA_D=${HS_LAMBDA_D:-"11000,16000,22000"}
HS_LAMBDA_S=${HS_LAMBDA_S:-"7000,10000,13000"}
HS_MU=${HS_MU:-0.6}

# ----- helpers --------------------------------------------------------------------------------
FLOAT="[0-9.eE+-]+"
# run a sweep that auto-saves its best cell to $MARKERS; emit all output (for BEST parsing). Non-fatal.
swp () { $PY scripts/fots_marker_compare.py "$@" --save-markers "$MARKERS" $FORCE 2>&1 || echo "  WARN: sweep failed"; }
show () { echo "$1" | grep -E "rmse\*=|BEST:|saved best-fit|cached sweep" || true; }   # surface the key lines
best_param () { echo "$1" | grep "BEST:" | grep -oE "$2=$FLOAT" | head -1 | cut -d= -f2; }  # "key=" off BEST line

# ===== OURS (ElastomerTaxel) ================================================================
echo "== [1/3] OURS dilate sweep: dilation_reg x compressibility  (thickness=${OURS_THICKNESS}mm, lambda_d=$LAMBDA_D fixed) =="
T=$(swp $COMMON $DILATE --marker-model elastomer --lambda-d "$LAMBDA_D" --elastomer-thickness "$OURS_THICKNESS" --sweep \
    --sweep-x compressibility --sweep-x-values "$COMPRESSIBILITY" --out "$OUT/sweep_dilate.png")
show "$T"
COMPRESS=$(best_param "$T" compressibility)
DSCALE=$(echo "$T" | grep "BEST:" | grep -oE "'dilate': $FLOAT" | grep -oE "$FLOAT$")
DILATION="--compressibility $COMPRESS --elastomer-thickness $OURS_THICKNESS --lambda-d $LAMBDA_D"
echo ">> ours dilation calibrated: compressibility=$COMPRESS thickness=${OURS_THICKNESS}mm dilate_scale=$DSCALE"

echo "== OURS shear sweep: lambda_s x shear_scale  (dilation fixed from above) =="
show "$(swp $COMMON $SHEAR --marker-model elastomer $DILATION --dilate-scale "$DSCALE" --sweep \
    --sweep-x lambda_s    --sweep-x-values "$OURS_LAMBDA_S" \
    --sweep-y shear_scale --sweep-y-values "$OURS_SHEAR_SCALE" --out "$OUT/sweep_shear.png")"

# ===== FOTS =================================================================================
echo "== [2/3] FOTS dilate sweep: fots_lambda_d =="
T=$(swp $COMMON $DILATE --marker-model fots --sweep \
    --sweep-x fots_lambda_d --sweep-x-values "$FOTS_LAMBDA_D" --out "$OUT/sweep_fots_dilate.png")
show "$T"; FOTS_LD=$(best_param "$T" fots_lambda_d)
echo "== FOTS shear sweep: fots_lambda_s  (fots_lambda_d=$FOTS_LD fixed) =="
show "$(swp $COMMON $SHEAR --marker-model fots --fots-lambda-d "$FOTS_LD" --sweep \
    --sweep-x fots_lambda_s --sweep-x-values "$FOTS_LAMBDA_S" --out "$OUT/sweep_fots_shear.png")"

# ===== HydroShear (ported HydroFOTS ref) ====================================================
echo "== [3/3] HydroShear dilate sweep: hydroshear_lambda_d  (mu=$HS_MU) =="
T=$(swp $COMMON $DILATE --marker-model hydroshear --hydroshear-mu "$HS_MU" --sweep \
    --sweep-x hydroshear_lambda_d --sweep-x-values "$HS_LAMBDA_D" --out "$OUT/sweep_hydroshear_dilate.png")
show "$T"; HS_LD=$(best_param "$T" hydroshear_lambda_d)
echo "== HydroShear shear sweep: hydroshear_lambda_s  (lambda_d=$HS_LD mu=$HS_MU fixed) =="
show "$(swp $COMMON $SHEAR --marker-model hydroshear --hydroshear-lambda-d "$HS_LD" --hydroshear-mu "$HS_MU" --sweep \
    --sweep-x hydroshear_lambda_s --sweep-x-values "$HS_LAMBDA_S" --out "$OUT/sweep_hydroshear_shear.png")"

# ----- 2x4 grid: [Real FOTS HydroShear Ours] x [dilate shear], one shared depth colorbar -------
[ -f "$MARKERS" ] || { echo "ERROR: no $MARKERS -- every sweep failed (see WARNs above)." >&2; exit 1; }
$PY scripts/fots_plot_grid.py --npz "$MARKERS" --out "$OUT/grid_sweep_best.png" --figsize 14 5
echo "Done -> $OUT/grid_sweep_best.png  (best params per cell are in the BEST: lines above + cached in $MARKERS)"
