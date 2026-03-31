#!/bin/bash
# Run all Study 1 analyses sequentially on a single GPU.
#
# Pipeline order:
#   1. Model comparison (fast, slow)     — MC approximators, calibration, confusion
#   2. Parameter estimation (fast, slow) — inference networks, posteriors, PPC metrics
#   3. Analyses                          — age correlations from posteriors
#   4. PPC analysis                      — PPC figures from saved metrics
#   5. Figure 1 schematic               — DDM vs. OUM illustration
#
# All scripts use JAX backend and the priors defined in sfi_functions.py.
# Expected total runtime: ~8-12 hours on RTX 4080 Super.

set -e

# Resolve script directory so the script can be called from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use PYTHON env variable if set, otherwise fall back to whichever python3 is active.
PYTHON="${PYTHON:-python3}"

echo "=== [1/7] Model comparison — fast tasks ==="
$PYTHON run_model_comparison_fast.py 2>&1 | tee /tmp/mc_fast.log
echo ""

echo "=== [2/7] Model comparison — slow tasks ==="
$PYTHON run_model_comparison_slow.py 2>&1 | tee /tmp/mc_slow.log
echo ""

echo "=== [3/7] Parameter estimation — fast tasks ==="
$PYTHON run_parameter_estimation_fast.py 2>&1 | tee /tmp/param_est_fast.log
echo ""

echo "=== [4/7] Parameter estimation — slow tasks ==="
$PYTHON run_parameter_estimation_slow.py 2>&1 | tee /tmp/param_est_slow.log
echo ""

echo "=== [5/7] Age correlation analyses ==="
$PYTHON run_analyses.py 2>&1 | tee /tmp/analyses.log
echo ""

echo "=== [6/7] PPC analysis ==="
$PYTHON run_ppc_analysis.py 2>&1 | tee /tmp/ppc.log
echo ""

echo "=== [7/7] Figure 1 schematic ==="
$PYTHON run_figure1_schematic.py 2>&1 | tee /tmp/figure1.log
echo ""

echo "=== All Study 1 analyses complete ==="
echo "Figures saved to: sfi/figures/"
echo "Posteriors saved to: sfi/sfi_data/"
