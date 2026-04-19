#!/bin/bash
#
# Detailed profiling entrypoint for the Triton track.
#
# Usage:
#   ./benchmark_detailed_triton_track.sh <folder_name>
#   ./benchmark_detailed_triton_track.sh <folder_name> --warmup 2 --runs 5
#   ./benchmark_detailed_triton_track.sh glm_asr_triton_example
#   ./benchmark_detailed_triton_track.sh --attention-only --seq-len 512
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    echo "GLM-ASR Detailed Operator Profiling (Triton Track)"
    echo ""
    echo "Usage: $0 [folder_name] [options]"
    echo ""
    echo "Options:"
    echo "  --audio PATH      Path to test audio file"
    echo "  --runs N          Number of profiling runs (default: 3)"
    echo "  --warmup N        Number of explicit warmup runs (default: 1)"
    echo "  --nsys            Run Nsight Systems profiling"
    echo "  --attention-only  Only profile attention operations"
    echo "  --linear-only     Only profile linear/GEMM operations"
    echo "  --seq-len N       Sequence length for micro-benchmarks (default: 256)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Available Triton folders:"
    for dir in "$SCRIPT_DIR"/glm_asr_triton_*/; do
        if [ -d "$dir" ]; then
            echo "  - $(basename "$dir")"
        fi
    done
}

for arg in "$@"; do
    if [ "$arg" == "-h" ] || [ "$arg" == "--help" ]; then
        show_help
        exit 0
    fi
done

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

cd "$SCRIPT_DIR"

if [[ "$1" == --* ]]; then
    python benchmark_detailed_triton_track.py "$@"
else
    FOLDER="$1"
    shift

    if [[ "$FOLDER" != *triton* ]]; then
        echo "Error: '$FOLDER' is not a Triton track folder"
        exit 1
    fi

    if [ ! -d "$SCRIPT_DIR/$FOLDER" ]; then
        echo "Error: Folder '$FOLDER' not found in $SCRIPT_DIR"
        exit 1
    fi

    python benchmark_detailed_triton_track.py "$FOLDER" "$@"
fi
