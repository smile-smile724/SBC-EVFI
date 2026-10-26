"""Event-driven SNN — Two-Phase Training with StructureEncoder + BlurryMaskPredictor.

Phase 1: Offset Warmup + Deblur
         Pipeline: events → offset_head → warp → blurry_predictor → deblur_head
         Train: structure_encoder, offset_head, blurry_predictor, deblur_head
Phase 2: Cascade + Fusion Joint Training
         Pipeline: cascade_ref → offset_head → warp → [cascade out] → fusion → blurry_predictor → deblur_head → delta
         encoder_body frozen (Phase 1 weights)
         Train: structure_encoder, offset_head_fwd/bwd, deblur_head, fusion_head, blurry_predictor_fwd/bwd

Run (from project root):
    python -m main.train.run --phase 1 --exp-name test_v21 --data-root datasets/data --gt-root datasets/GT --device cuda

Phase 1 → Phase 2:
    python -m main.train.run --phase 2 --exp-name test_v21 --data-root ... --gt-root ... --device cuda --fusion-w 1.0 --resume --resume-path experiments/test_v21/phase1/ckpt/ckpt_phase1_best.pth
"""

import argparse
from pathlib import Path

from .trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Two-Phase Event-driven SNN Training with Structural Encoder")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--key-interval", type=int, default=9)
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--phase1-epochs", type=int, default=3)
    parser.add_argument("--phase2-epochs", type=int, default=3)
    parser.add_argument("--preview-seq", type=str, default=None)
    parser.add_argument("--preview-interval", type=int, default=1,
                        help="generate preview every N epochs (default: 1, every epoch)")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warp-w", type=float, default=None,
                        help="override warp loss weight (default: phase-dependent: 1.0/0.5)")
    parser.add_argument("--lpips-w", type=float, default=0.1)
    parser.add_argument("--wsm-w", type=float, default=0.005)
    parser.add_argument("--fusion-w", type=float, default=1.0,
                        help="Phase 2 fusion loss weight (joint training always enabled)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--mask-kernel-size", type=int, default=5)
    parser.add_argument("--mask-sigma", type=float, default=0.5)
    parser.add_argument("--delta-scale", type=float, default=10.0)
    parser.add_argument("--delta-range", type=float, default=0.15,
                        help="delta_correction range (tanh * delta_range), for Phase 2")
    parser.add_argument("--blurry-sparsity-w", type=float, default=0.0,
                        help="Phase 2: L1 penalty on blurry_mask mean (sparsity, disabled by default)")
    parser.add_argument("--offset-w", type=float, default=0.3,
                        help="Phase 2 offset FT warp weight (default: 0.3)")
    parser.add_argument("--blurry-lr", type=float, default=None,
                        help="Phase 2: separate lr for BlurryPredictor (default: None = same as --lr)")
    parser.add_argument("--t-anchor-lr", type=float, default=None,
                        help="Phase 2: separate lr for t_anchor params (t_scale,t_bias_proj,t_conv), default: 10x --lr")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default=None)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()