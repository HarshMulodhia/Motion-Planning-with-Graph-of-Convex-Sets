@echo off
REM GCS-Guided Deep RL Project - Execution Script (Windows)
REM Usage: run_project.bat [option]
REM Options: quick, train, eval, compare, demo, all

setlocal enabledelayedexpansion

echo ======================================================================
echo GCS-Guided Deep RL for Robot Manipulation
echo ======================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    exit /b 1
)

REM Default action
if "%1%"=="" set ACTION=all
if not "%1%"=="" set ACTION=%1%

if /i "%ACTION%"=="quick" (
    echo [*] Running quick setup and test...
    echo [+] Installing dependencies...
    pip install -q -r requirements.txt

    echo [+] Creating gripper URDF...
    python create_gripper.py

    echo [+] Testing configuration space...
    python test_config_space.py

    echo [*] Quick setup complete!
    goto end
)

if /i "%ACTION%"=="train" (
    echo [*] Starting training pipeline...
    echo [+] Installing dependencies...
    pip install -q -r requirements.txt

    echo [+] Creating gripper URDF...
    python create_gripper.py

    echo [*] Starting training (this will take 2-3 hours on GPU)...
    python train_ycb_grasp.py

    echo [*] Training complete!
    goto end
)

if /i "%ACTION%"=="eval" (
    echo [*] Evaluating trained policy...
    python evaluate_grasp.py
    echo [*] Evaluation complete!
    goto end
)

if /i "%ACTION%"=="compare" (
    echo [*] Comparing planning methods...
    python compare_methods.py
    echo [*] Comparison complete!
    goto end
)

if /i "%ACTION%"=="demo" (
    echo [*] Generating demo video...
    echo [*] This will create ~1800 frames and compile into MP4 (takes ~30 min)...
    python generate_demo_video.py

    if exist "results\demo_video.mp4" (
        echo [+] Video saved to: results\demo_video.mp4
    )
    goto end
)

if /i "%ACTION%"=="all" (
    echo [*] Running complete pipeline...

    echo [+] 1/5: Setup
    pip install -q -r requirements.txt
    python create_gripper.py
    python test_config_space.py

    echo [+] 2/5: Training (this will take 2-3 hours on GPU)
    python train_ycb_grasp.py

    echo [+] 3/5: Evaluation
    python evaluate_grasp.py

    echo [+] 4/5: Method Comparison
    python compare_methods.py

    echo [+] 5/5: Demo Video Generation
    python generate_demo_video.py

    echo [*] All steps complete!
    goto end
)

echo Usage: run_project.bat [option]
echo.
echo Options:
echo   quick     - Quick setup and test (5 min)
echo   train     - Train RL policy (2-3 hours)
echo   eval      - Evaluate policy (5 min)
echo   compare   - Compare methods (20 min)
echo   demo      - Generate video (30 min)
echo   all       - Run complete pipeline (6 hours)
exit /b 1

:end
echo.
echo ======================================================================
echo [+] Process complete!
echo ======================================================================
endlocal
