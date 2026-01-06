#!/bin/bash
#
# Run preference generation for all centuries.
#
# Usage:
#   ./scripts/run_all.sh --dataset prism --model_size 8B
#   ./scripts/run_all.sh --dataset ca --model_size 8B --centuries C013,C014,C015
#   ./scripts/run_all.sh --dataset prism --model_size 8B --n_runs 3 --resume_from C015
#   ./scripts/run_all.sh --dataset prism --model_size 8B --user_ids all
#   ./scripts/run_all.sh --dataset prism --model_size 8B --user_ids 1,2,3
#

set -e

# Default values
DATASET=""
MODEL_SIZE="8B"
N_RUNS=3
N_SAMPLES=""  # Empty means all
START_IDX=0
RESUME_FROM=""
CENTURIES="C013,C014,C015,C016,C017,C018,C019,C020,C021"
USER_IDS=""  # Empty means no user profiles

# NAS paths
NAS_BASE="/nas/ucb/rachel/historical-prefs"
LOG_DIR="${NAS_BASE}/logs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --n_runs)
            N_RUNS="$2"
            shift 2
            ;;
        --n_samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --start_idx)
            START_IDX="$2"
            shift 2
            ;;
        --centuries)
            CENTURIES="$2"
            shift 2
            ;;
        --resume_from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --user_ids)
            USER_IDS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --dataset {ca,prism} [options]"
            echo ""
            echo "Options:"
            echo "  --dataset      Dataset to process (required: ca or prism)"
            echo "  --model_size   Model size (default: 8B)"
            echo "  --n_runs       Number of runs per comparison (default: 3)"
            echo "  --n_samples    Number of samples to process (default: all)"
            echo "  --start_idx    Starting index (default: 0)"
            echo "  --centuries    Comma-separated list of centuries (default: all C013-C021)"
            echo "  --resume_from  Resume from this century (skip earlier ones)"
            echo "  --user_ids     User IDs to generate for: 'all' or comma-separated (e.g., '1,2,3')"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required"
    echo "Usage: $0 --dataset {ca,prism} [options]"
    exit 1
fi

if [[ "$DATASET" != "ca" && "$DATASET" != "prism" ]]; then
    echo "Error: --dataset must be 'ca' or 'prism'"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Convert centuries to array
IFS=',' read -ra CENTURY_ARRAY <<< "$CENTURIES"

# Handle resume_from
SKIP_UNTIL_FOUND=false
if [[ -n "$RESUME_FROM" ]]; then
    SKIP_UNTIL_FOUND=true
fi

# Print configuration
echo "=========================================="
echo "Historical Preferences Generation"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model size: $MODEL_SIZE"
echo "Centuries: ${CENTURY_ARRAY[*]}"
echo "Runs per comparison: $N_RUNS"
if [[ -n "$N_SAMPLES" ]]; then
    echo "Samples: $N_SAMPLES"
else
    echo "Samples: all"
fi
echo "Start index: $START_IDX"
if [[ -n "$RESUME_FROM" ]]; then
    echo "Resuming from: $RESUME_FROM"
fi
if [[ -n "$USER_IDS" ]]; then
    echo "User IDs: $USER_IDS"
else
    echo "User IDs: none (single-user mode)"
fi
echo "Log directory: $LOG_DIR"
echo "=========================================="
echo ""

# Get total samples for dataset
if [[ -z "$N_SAMPLES" ]]; then
    PAIRWISE_FILE="${NAS_BASE}/data/${DATASET}/questions_pairwise.csv"
    if [[ -f "$PAIRWISE_FILE" ]]; then
        # Count lines minus header
        N_SAMPLES=$(($(wc -l < "$PAIRWISE_FILE") - 1))
        echo "Auto-detected $N_SAMPLES total pairwise comparisons"
    else
        echo "Warning: Could not find $PAIRWISE_FILE to count samples"
        echo "Please specify --n_samples manually"
        exit 1
    fi
fi

# Process each century
for CENTURY in "${CENTURY_ARRAY[@]}"; do
    # Handle resume
    if [[ "$SKIP_UNTIL_FOUND" == "true" ]]; then
        if [[ "$CENTURY" == "$RESUME_FROM" ]]; then
            SKIP_UNTIL_FOUND=false
        else
            echo "Skipping $CENTURY (resuming from $RESUME_FROM)"
            continue
        fi
    fi
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/${DATASET}_${MODEL_SIZE}_${CENTURY}_${TIMESTAMP}.log"
    
    echo ""
    echo "=========================================="
    echo "Processing: $DATASET / $MODEL_SIZE / $CENTURY"
    echo "Start time: $(date)"
    echo "Log file: $LOG_FILE"
    echo "=========================================="
    
    # Build command
    CMD="python generate_preferences.py"
    CMD="$CMD --dataset $DATASET"
    CMD="$CMD --model_size $MODEL_SIZE"
    CMD="$CMD --model_century $CENTURY"
    CMD="$CMD --n_samples $N_SAMPLES"
    CMD="$CMD --start_idx $START_IDX"
    CMD="$CMD --n_runs $N_RUNS"
    
    if [[ -n "$USER_IDS" ]]; then
        CMD="$CMD --user_ids $USER_IDS"
    fi
    
    # Run generation
    $CMD 2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "Finished $CENTURY at $(date)"
    echo "Log saved to: $LOG_FILE"
done

echo ""
echo "=========================================="
echo "All centuries completed"
echo "=========================================="

