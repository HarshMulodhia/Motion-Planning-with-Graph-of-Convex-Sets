#!/bin/bash

# GCS + RL Grasp Project - Unified Execution Script
# Updated with video visualization support for GCS, RL, and Hybrid methods

# Usage:
# bash run_project.sh [command]

# Commands:
# setup - Installs dependencies and runs the project initializer.
# train - Trains the PPO model.
# evaluate - Evaluates the trained model on test objects.
# compare - Compares GCS, RL, and Hybrid methods.
# visualize - Creates video simulations of all three methods (GCS, RL, Hybrid) on a test object.
# visualize_knife - Creates video simulations for the "knife" object (GCS, RL, Hybrid).
# all - Runs the full pipeline: setup -> train -> evaluate -> compare.
# cleanup - Removes all generated files (models, logs, results).
# help - Shows this help message.

# Example:
# bash run_project.sh all
# bash run_project.sh visualize_knife

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
PYTHON_CMD="python3"
VIDEO_OUTPUT_DIR="results/videos"

# --- Colors for Logging ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\n\033[1;34m'
NC='\033[0m' # No Color

# --- Logging Functions ---
log_header() {
    echo -e "${BLUE}▶ $1${NC}"
}

log_step() {
    echo -e " ${GREEN}✓${NC} $1"
}

log_info() {
    echo -e " ${YELLOW}ℹ${NC} $1"
}

log_error() {
    echo -e " ${RED}✗ ERROR:${NC} $1"
}

# --- Main Functions ---

# Function to set up the project
setup() {
    log_header "STEP 1: PROJECT SETUP"
    log_info "This will install dependencies and validate the project structure."
    
    if ! command -v $PYTHON_CMD &> /dev/null; then
        log_error "$PYTHON_CMD is not found. Please install Python 3.8+."
        exit 1
    fi
    
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt not found! Cannot install dependencies."
        exit 1
    fi
    
    log_step "Installing dependencies from requirements.txt..."
    pip install -q -r requirements.txt
    log_step "Dependencies installed."
    
    if [ ! -f "init_project.py" ]; then
        log_error "init_project.py not found! Cannot initialize project."
        log_info "This file was created during the debugging process. Please restore it."
        exit 1
    fi
    
    log_step "Running project initializer..."
    $PYTHON_CMD init_project.py
    log_step "Project setup and validation complete."
}

# Function to train the model
train() {
    log_header "STEP 2: TRAINING MODEL"
    
    if [ ! -f "scripts/train_ycb_grasp.py" ]; then
        log_error "scripts/train_ycb_grasp.py not found!"
        exit 1
    fi
    
    log_info "Starting model training. This may take 3-5 minutes..."
    $PYTHON_CMD scripts/train_ycb_grasp.py
    log_step "Training complete. Model saved to 'models/ycb_grasp/'."
}

# Function to evaluate the model
evaluate() {
    log_header "STEP 3: EVALUATING MODEL"
    
    if [ ! -f "scripts/evaluate_grasp.py" ]; then
        log_error "scripts/evaluate_grasp.py not found!"
        exit 1
    fi
    
    if [ ! -f "models/ycb_grasp/final_model.zip" ]; then
        log_error "Trained model not found! Please run the 'train' step first."
        exit 1
    fi
    
    log_info "Evaluating policy on test objects..."
    $PYTHON_CMD scripts/evaluate_grasp.py
    log_step "Evaluation complete. Results saved to 'results/grasp_eval_results.json'."
}

# Function to compare methods
compare() {
    log_header "STEP 4: COMPARING METHODS"
    
    if [ ! -f "scripts/compare_methods.py" ]; then
        log_error "scripts/compare_methods.py not found!"
        exit 1
    fi
    
    if [ ! -f "models/ycb_grasp/final_model.zip" ]; then
        log_error "Trained model not found! Please run the 'train' step first."
        exit 1
    fi
    
    log_info "Comparing Pure GCS vs. Pure RL vs. Hybrid methods..."
    $PYTHON_CMD scripts/compare_methods.py
    log_step "Comparison complete. Results saved to 'results/comparison_results.json'."
}

# Function to create video visualizations
visualize() {
    log_header "STEP 5: CREATING VIDEO VISUALIZATIONS"
    
    local test_object="${1:-rubiks_cube}"
    
    if [ ! -f "models/ycb_grasp/final_model.zip" ]; then
        log_error "Trained model not found! Please run the 'train' step first."
        exit 1
    fi
    
    log_info "Creating video directory: $VIDEO_OUTPUT_DIR"
    mkdir -p "$VIDEO_OUTPUT_DIR"
    
    log_info "Generating video simulations for: $test_object"
    log_step "Creating GCS visualization (pure GCS planning)..."
    log_step "Creating RL visualization (pure learned policy)..."
    log_step "Creating Hybrid visualization (GCS + RL combined)..."
    
    # Run visualization
    $PYTHON_CMD scripts/visualize_policies.py "$test_object"
    
    log_step "Videos saved to: $VIDEO_OUTPUT_DIR"
    log_info "Video files:"
    log_info "  - GCS method: $VIDEO_OUTPUT_DIR/gcs_*.mp4"
    log_info "  - RL method: $VIDEO_OUTPUT_DIR/rl_*.mp4"
    log_info "  - Hybrid method: $VIDEO_OUTPUT_DIR/hybrid_*.mp4"
}

# Special function for knife visualization
visualize_knife() {
    log_header "CREATING VIDEO VISUALIZATIONS FOR KNIFE OBJECT"
    visualize "knife"
}

# Function to visualize GCS only (no RL dependency)
visualize_gcs_only() {
    log_header "VISUALIZING PURE GCS METHOD"
    
    local test_object="${1:-knife}"
    
    if [ ! -f "scripts/visualize_gcs_only.py" ]; then
        log_error "scripts/visualize_gcs_only.py not found!"
        exit 1
    fi
    
    log_info "Creating GCS-only visualizations for: $test_object"
    $PYTHON_CMD scripts/visualize_gcs_only.py "$test_object"
    
    log_step "GCS visualization complete (no RL model needed)"
}


# Function to run the full pipeline
run_all() {
    log_header "RUNNING FULL PIPELINE"
    setup
    train
    evaluate
    compare
    log_header "✅ FULL PIPELINE COMPLETE"
    log_info "All results are in the 'results/' directory."
}



# Function to clean up generated files
cleanup() {
    log_header "CLEANING UP PROJECT"
    log_info "This will remove all generated directories: models, logs, results, __pycache__"
    read -p "Are you sure? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf models logs results .pytest_cache
        find . -type d -name '__pycache__' -exec rm -r {} + 2>/dev/null
        log_step "Cleanup complete."
    else
        log_info "Cleanup cancelled."
    fi
}

# --- Command-line Argument Parsing ---
show_help() {
    echo "GCS + RL Grasp Project - Unified Execution Script"
    echo ""
    echo "Usage: bash run_project.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup              Installs dependencies and initializes the project."
    echo "  train              Trains the PPO model."
    echo "  evaluate           Evaluates the trained model."
    echo "  compare            Compares GCS, RL, and Hybrid methods."
    echo "  visualize          Creates video simulations (default object: rubiks_cube)."
    echo "  visualize_knife    Creates video simulations specifically for the knife object."
    echo "  all                (Default) Runs the full pipeline: setup -> train -> evaluate -> compare."
    echo "  cleanup            Removes all generated files."
    echo "  help               Shows this help message."
    echo ""
    echo "Examples:"
    echo "  bash run_project.sh all"
    echo "  bash run_project.sh visualize rubiks_cube"
    echo "  bash run_project.sh visualize_knife"
    echo "  bash run_project.sh train && bash run_project.sh visualize_knife"
}

# Main entry point
main() {
    COMMAND=${1:-all} # Default to 'all' if no command is provided
    
    case $COMMAND in
        setup)
            setup
            ;;
        train)
            train
            ;;
        evaluate)
            evaluate
            ;;
        compare)
            compare
            ;;
        visualize)
            test_object=${2:-rubiks_cube}
            visualize "$test_object"
            ;;
        visualize_knife)
            visualize_knife
            ;;
        gcs_only|gcs-only)
            test_object=${2:-knife}
            visualize_gcs_only "$test_object"
            ;;
        all)
            run_all
            ;;
        cleanup)
            cleanup
            ;;
        help|*)
            show_help
            ;;
    esac
}

main "$@"
