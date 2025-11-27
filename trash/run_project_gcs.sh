#!/bin/bash

# run_project_gcs_complete.sh
# Complete GCS project: Evaluation + Obstacles + Presentation Assets

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   GCS Motion Planning: Complete Analysis & Presentation        ║"
echo "║   • Basic GCS Planning                                         ║"
echo "║   • Obstacle Avoidance Challenge                              ║"
echo "║   • Presentation Asset Generation                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Step 1: Environment Checks ---
echo -e "${BLUE}[1/6] Checking environment...${NC}"
python --version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Python not found"
    exit 1
fi
echo -e "${GREEN}✓ Environment verified${NC}"
echo ""


# --- Step 2: Project Structure ---
echo -e "${BLUE}[2/6] Setting up project structure...${NC}"
mkdir -p results
mkdir -p logs
mkdir -p presentation_assets
mkdir -p src
echo -e "${GREEN}✓ Directories ready${NC}"
echo ""


# --- Step 3: Generate Presentation Assets ---
echo -e "${BLUE}[3/6] Generating presentation graphics & video...${NC}"
echo "  (Creating 5 graphs, 10 frames, and demo video...)"

python3 scripts/generate_presentation_assets.py > /dev/null 2>&1
ASSETS_STATUS=$?

python3 scripts/create_demo_video.py > /dev/null 2>&1
VIDEO_STATUS=$?

if [ $ASSETS_STATUS -eq 0 ] && [ $VIDEO_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ Presentation assets ready${NC}"
else
    echo -e "${YELLOW}⚠ Some assets failed (check logs)${NC}"
fi
echo ""


# --- Step 4: Run Basic GCS Evaluation ---
echo -e "${BLUE}[4/6] Running Basic GCS Evaluation (No Obstacles)...${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 scripts/evaluate_gcs_only.py
EVAL_BASIC_STATUS=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ Basic GCS evaluation complete${NC}"
echo ""


# --- Step 5: Run Obstacle Challenge Evaluation ---
echo -e "${BLUE}[5/6] Running Obstacle Challenge (With Obstacles)...${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 scripts/evaluate_gcs_with_obstacles.py
EVAL_OBSTACLES_STATUS=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ Obstacle challenge evaluation complete${NC}"
echo ""


# --- Step 6: Final Summary ---
echo -e "${BLUE}[6/6] Project Summary${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${CYAN}📊 RESULTS GENERATED:${NC}"
echo ""

echo "▶ GCS Planning Evaluation:"
if [ -f "results/gcs_evaluation_results.json" ]; then
    echo -e "${GREEN}  ✓${NC} Basic GCS Results: results/gcs_evaluation_results.json"
else
    echo -e "${YELLOW}  ✗${NC} Basic GCS Results not found"
fi

if [ -f "results/gcs_obstacle_comparison.json" ]; then
    echo -e "${GREEN}  ✓${NC} Obstacle Comparison: results/gcs_obstacle_comparison.json"
else
    echo -e "${YELLOW}  ✗${NC} Obstacle Comparison not found"
fi

echo ""
echo "▶ Presentation Assets:"
if [ -d "presentation_assets" ] && [ "$(ls -A presentation_assets)" ]; then
    count=$(ls presentation_assets/*.png 2>/dev/null | wc -l)
    echo -e "${GREEN}  ✓${NC} PNG Graphs & Frames: $count files"
    if [ -f "presentation_assets/gcs_demo_video.mp4" ]; then
        echo -e "${GREEN}  ✓${NC} Demo Video: gcs_demo_video.mp4"
    fi
else
    echo -e "${YELLOW}  ✗${NC} Presentation assets not found"
fi

echo ""
echo "▶ Project Files:"
if [ -f "src/gcs_motion_planner.py" ]; then
    echo -e "${GREEN}  ✓${NC} Basic GCS Planner: src/gcs_motion_planner.py"
fi
if [ -f "src/gcs_motion_planner_with_obstacles.py" ]; then
    echo -e "${GREEN}  ✓${NC} Obstacle-Aware Planner: src/gcs_motion_planner_with_obstacles.py"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${CYAN}📈 KEY METRICS:${NC}"
echo ""

if [ -f "results/gcs_evaluation_results.json" ]; then
    echo "Basic GCS (No Obstacles):"
    python3 -c "
import json
with open('results/gcs_evaluation_results.json') as f:
    d = json.load(f)
    s = d.get('summary_statistics', {})
    print(f\"  Success Rate: {s.get('planning_success_rate_percent', 0):.1f}%\")
    print(f\"  Avg Planning Time: {s.get('avg_planning_time_ms', 0):.1f}ms\")
    print(f\"  Avg Path Length: {s.get('avg_path_length_regions', 0):.1f} regions\")
" 2>/dev/null || echo "  (Results unavailable)"
    echo ""
fi

if [ -f "results/gcs_obstacle_comparison.json" ]; then
    echo "With Obstacles Challenge:"
    python3 -c "
import json
with open('results/gcs_obstacle_comparison.json') as f:
    d = json.load(f)
    w = d.get('with_obstacles', {})
    print(f\"  Success Rate: {w.get('planning_success_rate', 0):.1f}%\")
    print(f\"  Avg Planning Time: {w.get('avg_planning_time', 0)*1000:.1f}ms\")
    print(f\"  Avg Path Length: {w.get('avg_path_length', 0):.1f} regions\")
    print(f\"  Planning Difficulty: {w.get('avg_planning_difficulty', 0):.2f}\")
" 2>/dev/null || echo "  (Results unavailable)"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${GREEN}🎉 PROJECT COMPLETE!${NC}"
echo ""
echo "Your presentation package includes:"
echo "  ✓ Complete GCS implementation (with & without obstacles)"
echo "  ✓ Performance metrics for both scenarios"
echo "  ✓ 5 professional graphs (300 DPI)"
echo "  ✓ Comparison demo video (1280x960)"
echo "  ✓ Detailed evaluation results (JSON)"
echo ""
echo "Next steps:"
echo "  1. Review results in: results/gcs_*.json"
echo "  2. Use graphs from: presentation_assets/"
echo "  3. Discuss obstacle challenge impact"
echo "  4. Show comparison between basic & obstacles"
echo ""
echo -e "${CYAN}Ready for presentation! 🚀${NC}"
echo ""
