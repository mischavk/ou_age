#!/bin/bash
# Run all Study 2 (IAT) analyses sequentially on a single GPU.
#
# Pipeline order:
#   1. Model comparison        — MC approximator, calibration, confusion
#   2. Parameter estimation    — inference networks, posteriors, PPC metrics
#   3. Analyses                — age trends, demographic subgroups
#   4. PPC analysis            — PPC figures from saved metrics
#
# All scripts use JAX backend and the priors defined in iat_functions.py.

set -eo pipefail

# Resolve script directory so the script can be called from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use PYTHON env variable if set, otherwise fall back to whichever python3 is active.
PYTHON="${PYTHON:-python3}"

echo "=== [1/4] Model comparison ==="
$PYTHON run_model_comparison.py 2>&1 | tee /tmp/iat_mc.log
echo ""

echo "=== [2/4] Parameter estimation ==="
$PYTHON run_parameter_estimation.py 2>&1 | tee /tmp/iat_param_est.log
echo ""

echo "=== [3/4] Age analyses ==="
$PYTHON run_analyses.py 2>&1 | tee /tmp/iat_analyses.log
echo ""

echo "=== [4/4] PPC analysis ==="
$PYTHON run_ppc_analysis.py 2>&1 | tee /tmp/iat_ppc.log
echo ""

echo "=== All IAT analyses complete ==="
echo "Figures saved to: iat/figures/"
echo "Results saved to: iat/iat_data/estimates/"
