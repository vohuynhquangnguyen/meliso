#!/bin/bash


#===================================================================================================
# SCRIPT DESCRIPTION
#===================================================================================================
# @author: Huynh Quang Nguyen Vo
# SLURM job array for the MELISO+ per-row variance benchmark (`benchmark_sigma_sq.py`).
#
# One `sbatch` submits one array task per matrix (17 tasks = the paper's 7 real + 10 binary
# matrices in inputs/matrices/imc-rk). Each task runs one MPI job: rank 0 holds the array, rank 1
# (root) drives N_BENCH MVMs with the same x. The whole matrix sits on one array (cell_rows x
# cell_cols = matrix shape), so it is programmed once (W&V) and then read N_BENCH times.
# sigma_sq_i = Var_l(y_i^(l)) / ||x||^2, measured in the solvers' domain (rows of A normalized to unit
# norm, y after the min-max reversal), so the files feed imcrk_synthetic.py (SIGMA_SQ_FILE) directly.
#
# Tasks are independent: a failed matrix does not stop the others.
#
# Usage (from the repo root):
#   mkdir -p logs && sbatch run_benchmark_sigma_sq_meliso.sh
#   sbatch --array=2 run_benchmark_sigma_sq_meliso.sh         # rerun one matrix (index into MATRICES)
#
# Requirements:
#   - the rebuilt simulator in ./build (make all from the fork's dev branch)
#   - a device YAML with ReadNoise.sigmaReadNoise > 0 (otherwise every read is identical and
#     sigma_sq = 0); checked at the start of each task
#===================================================================================================


#===================================================================================================
# SLURM DIRECTIVES
#===================================================================================================
# --array must cover the MATRICES list below (0 .. number of matrices - 1).
# Largest matrix (well1850, 1850x712) at N_BENCH=100: ~3 min on a laptop-class core, ~2.5x that
# on Pete (per-read time, 66x66 quickstart), so 2 h per task is a wide margin.
#SBATCH --partition express
#SBATCH --time 1:00:00
#SBATCH --array=0-16
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=8GB
#SBATCH --mail-user=lucius.vo@okstate.edu
#SBATCH --mail-type=END
#SBATCH --output=logs/BenchmarkSigmaSq_%A_%a.out
#SBATCH --job-name=BenchmarkSigmaSq
#===================================================================================================


#===================================================================================================
# ENVIRONMENT SETUP
#===================================================================================================
echo "[INFO] Setting up the environment..."

module load anaconda3/2022.10
module load gcc/7.5.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpienv38

export PYTHONPATH="${PYTHONPATH:-}:./build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:./build/"

echo "[INFO] Environment setup complete."
#===================================================================================================


set -euo pipefail


#===================================================================================================
# EXPERIMENT CONFIGURATION
#===================================================================================================
MATERIAL="EpiRAM"
DEVICE_YAML="device_EpiRAM.yaml"            # under config_files/; must have sigmaReadNoise > 0
N_BENCH=100                                 # rel. std-error of each sigma_sq_i ~ sqrt(2/N_BENCH)

# One array task per entry (file name without .npz in MATRIX_DIR). The folder also holds
# G43_binary, delaunay_n10_binary and lshp_265_binary, which are not in the paper's set; to run
# them too, append them here and extend --array to 0-19.
MATRIX_DIR="inputs/matrices/imc-rk"
MATRICES=(
    cz628_rowminmax gre_512_rowminmax mcfe_rowminmax nos6_rowminmax
    pde900_rowminmax well1033_rowminmax well1850_rowminmax
    G14_binary G2_binary cage_binary can_838_binary dwt_310_binary
    jagmesh1_binary lshp_406_binary lshp_577_binary lshp_778_binary netz4504_dual_binary
)

# x must not be constant: min-max maps a constant x to zeros and the result is then computed digitally,
# so every read would be identical
INPUT_VECTOR_PATH="inputs/vectors/input_x.txt"
# Two files per matrix from the same reads (see benchmark_sigma_sq.py):
#   OUT_ROOT/rownorm/sigma_sq_<matrix>.npy  solver domain -> imcrk_synthetic.py SIGMA_SQ_FILE
#   OUT_ROOT/sigma_sq_<matrix>.npy          scaled hardware output y_s -> plot_sigma_sq_spread.py
OUT_ROOT="inputs/sigma_sq/meliso/${MATERIAL}"
CFG_DIR="config_files/sigma_sq/${MATERIAL}"

export DT=1 OVERRIDE=0 ITER_LIMIT=21
export EC=0               # plain MVM. EC=1 re-programs A (two W&V encodings) on every sample
export MELISO_REENCODE=0  # write once, read many
export OMP_NUM_THREADS=1  # read noise draws from one shared RNG inside NeuroSim's OpenMP loop
export NORMALIZE_ROWS=1   # same row normalization as the solvers


#===================================================================================================
# SELECT THIS TASK'S MATRIX
#===================================================================================================
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with sbatch (job array); SLURM_ARRAY_TASK_ID is not set}"
if [ "$TASK_ID" -ge "${#MATRICES[@]}" ]; then
    echo "[ERROR] Array index ${TASK_ID} is past the end of MATRICES (${#MATRICES[@]} entries)."
    exit 1
fi
mat="${MATRICES[$TASK_ID]}"
A_FILE="${MATRIX_DIR}/${mat}.npz"

if [ ! -f "$A_FILE" ]; then
    echo "[ERROR] ${A_FILE} not found."
    exit 1
fi

if ! python3 -c "
import sys, yaml
rn = yaml.safe_load(open(sys.argv[1]))['Device'].get('ReadNoise', {})
sys.exit(0 if rn.get('enabled') and float(rn.get('sigmaReadNoise', 0)) > 0 else 1)
" "config_files/${DEVICE_YAML}"; then
    echo "[ERROR] config_files/${DEVICE_YAML} is missing or has no read noise (ReadNoise.enabled with sigmaReadNoise > 0)."
    echo "[ERROR] With sigmaReadNoise = 0 every read is identical and sigma_sq = 0 for every row."
    exit 1
fi

read -r ROWS COLS < <(python3 -c "import numpy as np; print(*np.load('${A_FILE}')['A'].shape)")


#===================================================================================================
# EXECUTION
#===================================================================================================
mkdir -p "$OUT_ROOT/rownorm" "$CFG_DIR"

# Experiment YAML: the whole matrix on one array, no error correction.
CFG="${CFG_DIR}/${mat}.yaml"
cat > "$CFG" <<EOF
exp_params:
  turnOnHardware: 1
  turnOnScaling: 0
  errCorr: 0
  inputEncoding: "signed"
  matrix_name: "sigma_sq_${MATERIAL}_${mat}"
  matrix_file: "${A_FILE}"
  distributed:
    decomposition_dir: "/tmp/"
    mca_rows: 1
    mca_cols: 1

device_config:
  root: "config_files/"
  cell_rows: ${ROWS}
  cell_cols: ${COLS}
  assignment:
    ${DEVICE_YAML}: [[-1]]
EOF

SIGMA_SQ_OUT="${OUT_ROOT}/rownorm/sigma_sq_${mat}.npy"
SIGMA_SQ_SCALED_OUT="${OUT_ROOT}/sigma_sq_${mat}.npy"
echo "[INFO] Task ${TASK_ID}: ${mat} (${ROWS}x${COLS}), material=${MATERIAL}, device=${DEVICE_YAML}, N_BENCH=${N_BENCH}"
echo "[INFO] Config: ${CFG}"

EXP_CONFIG_FILE="$CFG" A_FILE="$A_FILE" XVEC_PATH="$INPUT_VECTOR_PATH" \
N_BENCH="$N_BENCH" SIGMA_SQ_OUT="$SIGMA_SQ_OUT" SIGMA_SQ_SCALED_OUT="$SIGMA_SQ_SCALED_OUT" \
TMPDIR="${TMPDIR:-/tmp}/sigma_sq_${SLURM_ARRAY_JOB_ID:-local}_${mat}" \
    mpiexec -n 2 python3 benchmark_sigma_sq.py

echo "[INFO] DONE: ${mat} -> ${SIGMA_SQ_OUT} (solver domain), ${SIGMA_SQ_SCALED_OUT} (scaled)"
