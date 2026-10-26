# SBC-EVFI

A structure-guided bidirectional cascading framework for frame interpolation from sparse RGB keyframes and low-resolution simulated events. 

## Project Structure

```
├── main/train/                 # Core training and inference modules
│   ├── run.py                  # Training entry point (Phase 1 / Phase 2)
│   ├── model.py                # Network architecture (StructureEncoder, LIF, offset/deblur/fusion head)
│   ├── trainer.py              # Training loop and parameter grouping
│   ├── phases.py               # Two-stage training logic
│   ├── cascade.py              # Cascade inference and fusion module
│   ├── data.py                 # Event data loading and preprocessing
│   ├── evaluator.py            # Evaluation metric computation
│   ├── event_mask.py           # Blurry region prediction and deblurring
│   ├── metrics.py              # Metrics aggregation
│   ├── eval_packets.py         # Packet evaluation
│   └── color_diag.py           # Color diagnostic tool
├── encoder/run_snn_preview.py  # SNN grayscale reconstruction preview
├── encoder/                    # Event data encoding and preprocessing
│   ├── run_encoder_main.py     # Event extraction, keyframe indexing, packaging
│   ├── prepare_data.py         # Organize training data directories
│   └── run_decoder_side_pipeline.py  # Unpacking and decoding preview
├── datasets/                   # Datasets
│   ├── GT/                     # Original video frames (train/val subdirectories)
│   ├── data/                   # Training event data (decompress-ing/)
│   └── test/data/              # Small-scale test data
├── experiments/                # Experiment outputs
├── logs/                     # Training logs
└── pre_doc/                    # Method documentation
    ├── core_method.md          # Core method
    ├── command.md              # Usage guide
    └── data.md                 # Data pipeline
```

## Main Pipeline

### 1. Event Encoding and Data Preparation

Extract event streams from video frames and organize them into a training-ready format.

```bash
# Event encoding (single sequence)
python encoder/run_encoder_main.py \
    --frames-dir input/000 \
    --output-root output \
    --key-interval 9 \
    --event-threshold 12 \
    --event-downsample 2

# Batch data preparation (for training)
python encoder/prepare_data.py \
    --dataset-root datasets/GT \
    --train-dir datasets/data \
    --splits train,val \
    --key-interval 9 \
    --event-threshold 12 \
    --event-downsample 2 \
    --device cuda
```

### 2. Two-Stage Training

The current core training scheme, entry point at `main/train/run.py`.

**Data Flow Overview**:

- **Phase 1 (Motion Estimation Warm-up)**: event stream → event encoder + structure encoder → motion estimation → warp → blur prediction → deblurring
- **Phase 2 (Cascade + Fusion Joint Training)**: Phase 1 output as cascade source → motion estimation → warp → deblur → fusion → blur prediction → deblur correction

**Key Parameters**:

| Parameter | Description | Phase 1 Default | Phase 2 Default |
|:---|:---|:---|:---|
| `--lr` | Learning rate | 1e-4 | 2e-5 |
| `--blurry-lr` | BlurryPredictor independent learning rate | same as lr | 4e-6 |
| `--t-anchor-lr` | Temporal anchor parameter learning rate | — | 10× lr |
| `--lpips-w` | LPIPS perceptual loss weight | 0.1 | 0.1 |
| `--wsm-w` | Weight smoothing regularization weight | — | 0.005 |
| `--fusion-w` | Fusion loss weight | — | 1.0 |
| `--delta-range` | Fusion delta correction range | — | 0.15 |

**Run Commands**:

```bash
# Phase 1: Motion estimation warm-up (20 epochs)
python -m main.train.run \
    --phase 1 \
    --exp-name test_v21 \
    --data-root datasets/data \
    --gt-root datasets/GT \
    --device cuda \
    --phase1-epochs 20

# Phase 2: Cascade + Fusion joint training (10 epochs, requires Phase 1 completion)
python -m main.train.run \
    --phase 2 \
    --exp-name test_v21 \
    --data-root datasets/data \
    --gt-root datasets/GT \
    --device cuda \
    --phase2-epochs 10 \
    --fusion-w 1.0 \
    --lpips-w 0.1 \
    --wsm-w 0.005 \
    --lr 2e-5 \
    --blurry-lr 4e-6 \
    --resume \
    --resume-path experiments/test_v21/phase1/ckpt/ckpt_phase1_best.pth
```

### 3. SNN Grayscale Reconstruction Preview

```bash
python encoder/run_snn_preview.py \
    --packet-dir datasets/data/val/029/decompress-ing/packets/packet_0000_0000_0099.pkt \
    --gt-dir datasets/GT/val/029 \
    --num-pairs 3 \
    --device cuda
```

## Experiment Management

For each experiment, it is recommended to organize under `experiments/<name>/`. Training outputs are automatically written to:

```
experiments/<name>/
├── phase1/
│   ├── ckpt/       # Model checkpoint
│   ├── logs/       # Training logs
│   └── preview/    # Preview videos
└── phase2/
    ├── ckpt/
    ├── logs/
    └── preview/
```

Comparison group evaluation data is placed in `compare_DATA/`, including `metrics_kf3_server.csv` (baseline method metrics on the validation set).

## Environment

```bash
conda activate motion-skip
```

Dependencies: PyTorch, NumPy, OpenCV, lpips, and in-project modules.