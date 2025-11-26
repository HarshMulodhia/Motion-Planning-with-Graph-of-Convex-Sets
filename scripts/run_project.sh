#!/bin/bash

# GCS-Guided Deep RL Project - Execution Script
# Usage: bash run_project.sh [option]
# Options: quick, train, eval, compare, demo, all

set -e  # Exit on error

echo "======================================================================"
echo "GCS-Guided Deep RL for Robot Manipulation"
echo "======================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_info() {
    echo -e "${YELLOW}[i]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    log_error "Python not found. Please install Python 3.8+"
    exit 1
fi

# Default action
ACTION=${1:-all}

case $ACTION in
    quick)
        log_info "Running quick setup and test..."
        log_step "Installing dependencies..."
        pip install -q -r requirements.txt

        log_step "Creating gripper URDF..."
        python3 scripts/create_gripper.py

        log_step "Testing configuration space..."
        python3 scripts/test_config_space.py

        log_info "Quick setup complete!"
        ;;

    train)
        log_info "Starting training pipeline..."
        log_step "Installing dependencies..."
        pip install -q -r requirements.txt

        log_step "Creating gripper URDF..."
        python3 scripts/create_gripper.py

        log_info "Starting training (this will take 2-3 hours on GPU)..."
        python3 scripts/train_ycb_grasp.py

        log_info "Training complete!"
        ;;

    eval)
        log_info "Evaluating trained policy..."
        python3 scripts/evaluate_grasp.py
        log_info "Evaluation complete!"
        ;;

    compare)
        log_info "Comparing planning methods..."
        python3 scripts/compare_methods.py
        log_info "Comparison complete!"
        ;;

    demo)
        log_info "Generating demo video..."
        log_info "This will create ~1800 frames and compile into MP4 (takes ~30 min)..."
        python3 scripts/generate_demo_video.py

        if [ -f "results/demo_video.mp4" ]; then
            log_info "Video saved to: results/demo_video.mp4"
        fi
        ;;

    all)
        log_info "Running complete pipeline..."

        log_step "1/5: Setup"
        pip install -q -r requirements.txt
        python3 scripts/create_gripper.py
        python3 scripts/test_config_space.py

        log_step "2/5: Training (this will take 2-3 hours on GPU)"
        python3 scripts/train_ycb_grasp.py

        log_step "3/5: Evaluation"
        python3 scripts/evaluate_grasp.py

        log_step "4/5: Method Comparison"
        python3 scripts/compare_methods.py

        log_step "5/5: Demo Video Generation"
        python3 scripts/generate_demo_video.py

        log_info "All steps complete!"
        ;;

    *)
        echo "Usage: bash run_project.sh [option]"
        echo ""
        echo "Options:"
        echo "  quick     - Quick setup and test (5 min)"
        echo "  train     - Train RL policy (2-3 hours)"
        echo "  eval      - Evaluate policy (5 min)"
        echo "  compare   - Compare methods (20 min)"
        echo "  demo      - Generate video (30 min)"
        echo "  all       - Run complete pipeline (6 hours)"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
log_step "Process complete!"
echo "======================================================================"
