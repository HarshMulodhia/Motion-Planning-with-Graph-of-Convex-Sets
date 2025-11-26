# ⚡ Quick Reference Card

## Installation (5 minutes)

```bash
pip install -r requirements.txt
python create_gripper.py
python test_config_space.py  # Verify setup
```

---

## Full Pipeline (6 hours)

### 1️⃣ Train (2-3 hours)
```bash
python train_ycb_grasp.py
# Outputs: models/ycb_grasp/final_model.zip
```

### 2️⃣ Evaluate (5 minutes)
```bash
python evaluate_grasp.py
# Outputs: results/grasp_eval_results.json
```

### 3️⃣ Compare Methods (20 minutes)
```bash
python compare_methods.py
# Outputs: results/comparison_results.json
```

### 4️⃣ Generate Demo (30 minutes)
```bash
python generate_demo_video.py
# Outputs: results/demo_video.mp4 (2GB)
```

---

## Quick Test (10 minutes)

```bash
# Just evaluate pre-trained model
python evaluate_grasp.py

# Compare without training
python compare_methods.py
```

---

## Key Commands for Presentation

### Monitor Training
```bash
tensorboard --logdir logs/ycb_grasp
# Open http://localhost:6006
```

### View Results
```bash
cat results/grasp_eval_results.json  # Evaluation metrics
cat results/comparison_results.json  # Method comparison
```

### Check Model
```bash
ls -lh models/ycb_grasp/
file results/demo_video.mp4  # Verify MP4
```

---

## Expected Results

| Metric | Value |
|--------|-------|
| Training Time (GPU) | 2-3 hours |
| Success Rate | 88-94% |
| Evaluation Time | 5 min |
| Demo Video | ~2GB |

---

## Troubleshooting Commands

```bash
# Check dependencies
python -c "import stable_baselines3; print('OK')"

# Test PyBullet
python -c "import pybullet; print('OK')"

# Debug logging
DEBUG=1 python train_ycb_grasp.py
```

---

## File Map

| File | Purpose | Run Time |
|------|---------|----------|
| create_gripper.py | Setup | <1 min |
| test_config_space.py | Verify | 2 min |
| train_ycb_grasp.py | Train | 2-3 hrs |
| evaluate_grasp.py | Test | 5 min |
| compare_methods.py | Benchmark | 20 min |
| generate_demo_video.py | Visualize | 30 min |

---

## Output Structure

```
After training:
✅ models/ycb_grasp/final_model.zip (50MB)
✅ logs/ycb_grasp/events.* (TensorBoard)

After evaluation:
✅ results/grasp_eval_results.json
✅ results/comparison_results.json
✅ results/demo_video.mp4 (2GB)
```

---

## One-Liner Commands

```bash
# Setup
python create_gripper.py && python test_config_space.py

# Train + Eval
python train_ycb_grasp.py && python evaluate_grasp.py

# Full pipeline
python train_ycb_grasp.py && python evaluate_grasp.py && python compare_methods.py && python generate_demo_video.py
```

---

## Presentation Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Create gripper: `python create_gripper.py`
- [ ] Test setup: `python test_config_space.py`
- [ ] Train model: `python train_ycb_grasp.py` (takes 2-3 hours)
- [ ] Evaluate: `python evaluate_grasp.py`
- [ ] Compare: `python compare_methods.py`
- [ ] Generate demo: `python generate_demo_video.py`
- [ ] Open demo video: `results/demo_video.mp4`
- [ ] Show results: `cat results/grasp_eval_results.json`

---

## Success Indicators

✅ **Training succeeds**: Model saved to `models/ycb_grasp/final_model.zip`
✅ **Evaluation works**: Results in `results/grasp_eval_results.json`
✅ **Comparison runs**: Results in `results/comparison_results.json`
✅ **Video generated**: `results/demo_video.mp4` exists and is 2GB

---

## Pro Tips

1. **Speed up training**: Increase `num_envs=8` in config
2. **Monitor progress**: Use TensorBoard (`tensorboard --logdir logs/ycb_grasp`)
3. **Debug issues**: Enable logging with `logging.basicConfig(level=logging.DEBUG)`
4. **Save time**: Skip training if model exists (just run evaluate_grasp.py)
5. **Quick test**: Use `total_timesteps=100_000` for testing

---

## Emergency Help

| Problem | Fix |
|---------|-----|
| CUDA OOM | Reduce `num_envs` to 2 |
| Video fails | Install ffmpeg: `apt-get install ffmpeg` |
| Model not found | Run `train_ycb_grasp.py` first |
| Imports fail | `pip install -r requirements.txt` again |
| PyBullet error | Use `p.DIRECT` mode instead of GUI |

---

**Last Updated**: 2025-11-26
**Status**: ✅ Production Ready
