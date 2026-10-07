#!/bin/bash
# ===========================================================================
# Slurm Job Submission Script for ATLAS Pipeline (Green Planet, UCI)
#
# 4-phase batch/array architecture (see pipeline/executor.py module docstring
# for the general design; this script is the Slurm implementation of it):
#
#   1. batch array   : each array task runs parsing + mass_calculating +
#                       post_processing on its own slice of files, capped at
#                       FILES_PER_JOB files -- a FRESH process per task.
#   2. scan          : single job, after ALL batch tasks succeed --
#                       writes logs/global_ranges.json
#   3. histogram array: each array task builds histograms for its slice of
#                       processed SQLite shards -> histograms/batch_N.root
#   4. merge         : single job, after ALL histogram tasks succeed --
#                       hadd's batch_N.root files, aggregates stats, plots
#
# WHY an array instead of one big job: a single long-lived process was
# observed to accumulate memory well beyond what EventAccumulator tracks
# (chunk_yield_threshold_bytes) and eventually gets OOM-killed by Slurm --
# reproduced twice, at file ~220/5000 (8 threads) and file ~382/500
# (4 threads). See CLAUDE.md "env_threshold_memory_mb is dead config" for the
# full writeup. Splitting work across many independent array tasks sidesteps
# this entirely: no single process ever accumulates enough history to
# approach the ~23.5GB physical memory of an `atlas`-partition node.
#
# Usage: bash submit_slurm.sh
# ===========================================================================

set -e

# --- Configuration ---------------------------------------------------------
CONFIG="config.yaml"

TOTAL_FILES=100        # 500 big run, 250 maybe 40min, - total files to process across the whole run
FILES_PER_JOB=50        # 150 - per-job file cap -- comfortably under the ~220-382
                         # file range where single long-running jobs OOM'd.
                         # Lower this further if a batch task still OOMs.
NUM_JOBS=$(( (TOTAL_FILES + FILES_PER_JOB - 1) / FILES_PER_JOB ))  # ceil division
MAX_CONCURRENT=8         # throttle: at most this many array tasks run at once
                         # (the atlas partition only has ~11 nodes total --
                         # be a reasonable neighbor to other users)

CPUS_PER_JOB=4
BATCH_MEM="16G"
BATCH_WALLTIME="12:00:00"
SCAN_WALLTIME="01:00:00"
HIST_WALLTIME="01:00:00"
MERGE_WALLTIME="02:00:00"
PARTITION="atlas"

# Slurm account/allocation, if this cluster enforces one. Leave blank and try
# submitting first -- sbatch's own error message will name what's required
# if one is needed (see CLAUDE.md / earlier investigation: sacctmgr is
# broken on gpatlas1, so this couldn't be looked up directly).
ACCOUNT=""

# Nodes to skip entirely. c-5-6 caused every task there to fail instantly
# (/nfspool-0 wasn't mounted -- see CLAUDE.md) due to a failed Infiniband
# link that hadn't tripped the cluster's own health check; admin manually
# drained the node on 2026-07-29, so Slurm now excludes it automatically
# and this no longer needs to be listed here. Leave blank; add
# comma-separated node names here again if another node needs excluding.
EXCLUDE_NODES=""

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"

ACCOUNT_DIRECTIVE=""
if [ -n "$ACCOUNT" ]; then
    ACCOUNT_DIRECTIVE="#SBATCH --account=${ACCOUNT}"
fi

EXCLUDE_DIRECTIVE=""
if [ -n "$EXCLUDE_NODES" ]; then
    EXCLUDE_DIRECTIVE="#SBATCH --exclude=${EXCLUDE_NODES}"
fi

# --- Read run_name and base_output_dir from config -------------------------
read RUN_NAME BASE_DIR <<< $(python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
m = c.get('run_metadata', {})
print(m.get('run_name', 'pipeline_run'), m.get('base_output_dir', './output'))")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${BASE_DIR}/${RUN_NAME}_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

cp "${PIPELINE_DIR}/${CONFIG}" "${LOG_DIR}/config.yaml.orig"
cp "${PIPELINE_DIR}/submit_slurm.sh" "${LOG_DIR}/"

# --- Per-run config copy with max_files_to_process overridden ---------------
# main.py has no --max-files-to-process CLI flag, and this script must not
# silently rewrite the user's actual config.yaml (would destroy its comments
# on every run). Instead: copy it once per run and surgically patch just
# that one line with sed, preserving everything else -- this copy (not the
# original config.yaml) is what every array task/phase below actually reads.
RUN_CONFIG="${LOG_DIR}/config_run.yaml"
cp "${PIPELINE_DIR}/${CONFIG}" "${RUN_CONFIG}"
sed -i -E "s/^([[:space:]]*max_files_to_process:)[[:space:]]*.*/\1 ${FILES_PER_JOB}  # overridden by submit_slurm.sh (per-job cap for batch array)/" "${RUN_CONFIG}"

echo "=============================================="
echo "ATLAS Pipeline Submission (Slurm / Green Planet)"
echo "=============================================="
echo "Config:        ${CONFIG} (per-job override: ${RUN_CONFIG})"
echo "Run dir:       ${RUN_DIR}"
echo "Total files:   ${TOTAL_FILES}"
echo "Files/job:     ${FILES_PER_JOB}"
echo "Array size:    ${NUM_JOBS} (max ${MAX_CONCURRENT} concurrent)"
echo "Partition:     ${PARTITION}"
echo "Account:       ${ACCOUNT:-(none set - omitting --account)}"
echo "CPUs/job:      ${CPUS_PER_JOB}"
echo "Mem/batch job: ${BATCH_MEM}"
echo "=============================================="

# --- Phase 1: batch array — parsing + mass_calculating + post_processing ---
BATCH_JOB_ID=$(sbatch --parsable --array=1-${NUM_JOBS}%${MAX_CONCURRENT} <<EOF
#!/bin/bash
#SBATCH --job-name=atlas_batch
#SBATCH --partition=${PARTITION}
${ACCOUNT_DIRECTIVE}
${EXCLUDE_DIRECTIVE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${BATCH_MEM}
#SBATCH --time=${BATCH_WALLTIME}
#SBATCH --output=${LOG_DIR}/batch_%a.out
#SBATCH --error=${LOG_DIR}/batch_%a.err

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"

python -u main.py --config "${RUN_CONFIG}" --run-dir "${RUN_DIR}" \
    --batch-job-index \${SLURM_ARRAY_TASK_ID} --total-batch-jobs ${NUM_JOBS} \
    --tasks parsing,mass_calculating,post_processing
EOF
)
echo "Submitted batch array:    ${BATCH_JOB_ID} (${NUM_JOBS} tasks)"

# --- Phase 2: scan-only (writes logs/global_ranges.json) -------------------
SCAN_JOB_ID=$(sbatch --parsable --dependency=afterok:${BATCH_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=atlas_scan
#SBATCH --partition=${PARTITION}
${ACCOUNT_DIRECTIVE}
${EXCLUDE_DIRECTIVE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${BATCH_MEM}
#SBATCH --time=${SCAN_WALLTIME}
#SBATCH --output=${LOG_DIR}/scan.out
#SBATCH --error=${LOG_DIR}/scan.err

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"

python -u main.py --config "${RUN_CONFIG}" --run-dir "${RUN_DIR}" --scan-only
EOF
)
echo "Submitted scan job:       ${SCAN_JOB_ID} (depends on ${BATCH_JOB_ID})"

# --- Phase 3: histogram array ------------------------------------------------
HIST_JOB_ID=$(sbatch --parsable --array=1-${NUM_JOBS}%${MAX_CONCURRENT} --dependency=afterok:${SCAN_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=atlas_hist
#SBATCH --partition=${PARTITION}
${ACCOUNT_DIRECTIVE}
${EXCLUDE_DIRECTIVE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${BATCH_MEM}
#SBATCH --time=${HIST_WALLTIME}
#SBATCH --output=${LOG_DIR}/hist_%a.out
#SBATCH --error=${LOG_DIR}/hist_%a.err

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"

python -u main.py --config "${RUN_CONFIG}" --run-dir "${RUN_DIR}" \
    --batch-job-index \${SLURM_ARRAY_TASK_ID} --total-batch-jobs ${NUM_JOBS} \
    --tasks histogram_creation
EOF
)
echo "Submitted histogram array: ${HIST_JOB_ID} (${NUM_JOBS} tasks, depends on ${SCAN_JOB_ID})"

# --- Phase 4: merge — hadd histograms + aggregate stats + plots -------------
MERGE_JOB_ID=$(sbatch --parsable --dependency=afterok:${HIST_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=atlas_merge
#SBATCH --partition=${PARTITION}
${ACCOUNT_DIRECTIVE}
${EXCLUDE_DIRECTIVE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=${MERGE_WALLTIME}
#SBATCH --output=${LOG_DIR}/merge.out
#SBATCH --error=${LOG_DIR}/merge.err

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"

python -u main.py --config "${RUN_CONFIG}" --run-dir "${RUN_DIR}" --merge-only
EOF
)
echo "Submitted merge job:      ${MERGE_JOB_ID} (depends on ${HIST_JOB_ID})"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Cancel:   scancel ${BATCH_JOB_ID} ${SCAN_JOB_ID} ${HIST_JOB_ID} ${MERGE_JOB_ID}"
echo "Logs:     ${LOG_DIR}/"
echo "Output:   ${RUN_DIR}/"
