#!/usr/bin/env python3
"""
End-to-End Data Preparation Script

Pipeline: Original frames -> .pkt packet files -> decompress-ing/

Training only uses data from decompress-ing/ directory, so this script
cleans up all intermediate files (compress/, compress-ing/, decompress/, visualization/)
and keeps only decompress-ing/ for each sequence.

Usage:
    python encoder/prepare_data.py \
        --dataset-root datasets \
        --train-dir datasets/.cache \
        --splits train,val,test

Note: Only decompress-ing/ is kept as training data. Packet files (compress/)
are removed to save disk space.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def list_image_files(sequence_dir: Path) -> list[Path]:
    files = [path for path in sequence_dir.iterdir() if is_image_file(path)]
    files.sort(key=lambda item: item.name)
    return files


def discover_sequence_dirs(split_root: Path, max_depth: int = 4) -> list[Path]:
    """Discover all sequence directories containing image files."""
    if not split_root.exists() or split_root.is_file():
        return []

    sequence_dirs: list[Path] = []
    seen: set[Path] = set()
    stack: list[tuple[Path, int]] = [(split_root, 0)]
    
    while stack:
        current, depth = stack.pop()
        if current in seen or depth > max_depth:
            continue
        seen.add(current)
        
        try:
            image_files = list_image_files(current)
        except Exception:
            image_files = []
        
        if image_files:
            sequence_dirs.append(current)
            continue
        
        try:
            children = [item for item in current.iterdir() if item.is_dir()]
        except Exception:
            children = []
        
        for child in sorted(children, key=lambda item: item.name, reverse=True):
            stack.append((child, depth + 1))
    
    sequence_dirs.sort(key=lambda item: str(item))
    return sequence_dirs


def run_full_pipeline(
    sequence_dir: Path,
    output_root: Path,
    key_interval: int = 9,
    packet_size: int = 100,
    event_threshold: int = 12,
    event_downsample: int = 2,
    latency_budget_ms: float = 30.0,
    gpu_worker_count: int = 2,
    device: str = "cuda",
    skip_if_exists: bool = True,
) -> Path | None:
    """
    Run full encode-decode pipeline.
    
    Returns:
        Path to decompress-ing directory or None if failed
    """
    video_stem = sequence_dir.name
    video_output_dir = output_root / video_stem
    compress_dir = video_output_dir / "compress"
    decompress_ing_dir = video_output_dir / "decompress-ing"
    
    # Check if already processed
    if skip_if_exists and decompress_ing_dir.exists():
        has_keyframes = any(decompress_ing_dir.glob("keyframe_*.png"))
        has_events = any(decompress_ing_dir.glob("*.npz"))
        
        if has_keyframes and has_events:
            print(f"[e2e] Skipping {video_stem} (already processed)")
            return decompress_ing_dir
    
    print(f"[e2e] Processing: {video_stem}")
    
    encoder_script = Path(__file__).parent / "run_encoder_main.py"
    decoder_script = Path(__file__).parent / "run_decoder_side_pipeline.py"
    
    # ===== Step 1: Encode =====
    print(f"[e2e]   Step 1/2: Encoding...")
    encode_cmd = [
        sys.executable, str(encoder_script),
        "--frames-dir", str(sequence_dir),
        "--output-root", str(output_root),
        "--key-interval", str(key_interval),
        "--packet-size", str(packet_size),
        "--event-threshold", str(event_threshold),
        "--max-events-per-frame", "0",
        "--packet-ext", ".pkt",
        "--event-downsample", str(event_downsample),
        "--latency-budget-ms", str(latency_budget_ms),
        "--emit-manifest",
        "--device", device,
        "--event-batch-size", "64",
        "--gpu-worker-count", str(gpu_worker_count),
        "--disable-event-visualization",
    ]
    
    try:
        result = subprocess.run(
            encode_cmd,
            check=True,
            capture_output=False,
            timeout=3600
        )
        
        if not compress_dir.exists() or not any(compress_dir.glob("packet_*.pkt")):
            print(f"[e2e]   ✗ Encoding failed: no packet files")
            return None
        
        print(f"[e2e]   ✓ Encoding complete")
    except subprocess.TimeoutExpired:
        print(f"[e2e]   ✗ Encoding timeout")
        return None
    except Exception as e:
        print(f"[e2e]   ✗ Encoding failed: {e}")
        return None
    
    # ===== Step 2: Decode =====
    print(f"[e2e]   Step 2/2: Decoding...")
    decode_cmd = [
        sys.executable, str(decoder_script),
        "--packet-dir", str(compress_dir),
        "--output-root", str(output_root),
        "--video-stem", video_stem,
        "--disable-visualization",
    ]
    
    try:
        result = subprocess.run(
            decode_cmd,
            check=True,
            capture_output=False,
            timeout=1800
        )
        
        if not decompress_ing_dir.exists():
            print(f"[e2e]   ✗ Decoding failed: decompress-ing not created")
            return None

        # Support two possible decompress-ing layouts:
        # 1) flat: decompress-ing contains keyframe_*.png and events.npz
        # 2) packetized: decompress-ing/packets/<packet>/keyframes/*.png and events.npz
        has_keyframes = any(decompress_ing_dir.glob("keyframe_*.png"))
        has_events = any(decompress_ing_dir.glob("*.npz"))

        packets_dir = decompress_ing_dir / "packets"
        if packets_dir.exists():
            # check for nested keyframes
            if not has_keyframes:
                try:
                    for packet_sub in sorted(packets_dir.iterdir()):
                        kf_dir = packet_sub / "keyframes"
                        if kf_dir.exists() and any(kf_dir.glob("*.png")):
                            has_keyframes = True
                            break
                except Exception:
                    pass
            # check for nested events.npz
            if not has_events:
                try:
                    for packet_sub in sorted(packets_dir.iterdir()):
                        if any(packet_sub.glob("events.npz")):
                            has_events = True
                            break
                except Exception:
                    pass

        if has_keyframes and has_events:
            print(f"[e2e]   ✓ Decoding complete")
            print(f"[e2e]   Step 3/3: Cleaning up intermediate files...")
            _cleanup_intermediate_files(video_output_dir)
            return decompress_ing_dir
        else:
            # debug: show what we found
            print(f"[e2e]   ✗ Missing outputs in decompress-ing: keyframes={has_keyframes} events={has_events} path={decompress_ing_dir}")
            return None
            
    except Exception as e:
        print(f"[e2e]   ✗ Decoding failed: {e}")
        return None


def _cleanup_intermediate_files(video_output_dir: Path) -> None:
    """
    Remove unnecessary intermediate files after encoding-decoding pipeline.

    Keeps only:
    - decompress-ing/ (training data, the only directory needed for training)

    Removes:
    - compress/ (packet files, not needed for training)
    - compress-ing/ (encoder intermediate output, not needed for training)
    - decompress/ (decoder output directory, not needed for training)
    - visualization/ (event visualization videos, not needed)
    """
    import shutil

    dirs_to_remove = []

    # Check compress directory (packets, not needed for training)
    compress_dir = video_output_dir / "compress"
    if compress_dir.exists():
        dirs_to_remove.append(("compress", compress_dir))

    # Check compress-ing directory
    compress_ing_dir = video_output_dir / "compress-ing"
    if compress_ing_dir.exists():
        dirs_to_remove.append(("compress-ing", compress_ing_dir))

    # Check decompress directory
    decompress_dir = video_output_dir / "decompress"
    if decompress_dir.exists():
        dirs_to_remove.append(("decompress", decompress_dir))

    # Check visualization directory
    visualization_dir = video_output_dir / "visualization"
    if visualization_dir.exists():
        dirs_to_remove.append(("visualization", visualization_dir))
    
    # Remove directories
    for dir_name, dir_path in dirs_to_remove:
        try:
            shutil.rmtree(dir_path)
            print(f"[e2e]     Removed {dir_name}/")
        except Exception as e:
            print(f"[e2e]     Warning: Failed to remove {dir_name}/: {e}")


def _process_single_sequence(args_tuple) -> tuple[str, bool]:
    """Worker function for parallel sequence processing."""
    seq_dir, output_root, video_stem, encoder_config = args_tuple
    
    try:
        decompress_dir = run_full_pipeline(
            sequence_dir=seq_dir,
            output_root=output_root,
            **encoder_config
        )
        return (video_stem, decompress_dir is not None)
    except Exception as e:
        print(f"[e2e] ✗ Error processing {video_stem}: {e}")
        return (video_stem, False)


def process_split_parallel(
    split_name: str,
    split_root: Path,
    cache_root: Path,
    encoder_config: dict[str, Any],
    parallel_sequences: int = 1,
) -> dict[str, int]:
    """Process all sequences in a split with parallel execution."""
    from multiprocessing import Pool
    
    sequence_dirs = discover_sequence_dirs(split_root)
    if not sequence_dirs:
        print(f"[e2e] No sequences found in {split_root}")
        return {"total": 0, "success": 0, "failed": 0}
    
    print(f"\n{'='*70}")
    print(f"[e2e] Processing split: {split_name}")
    print(f"[e2e] Sequences: {len(sequence_dirs)}")
    print(f"[e2e] Parallel workers: {parallel_sequences}")
    print(f"[e2e] Output: {cache_root}/{split_name}/")
    print(f"{'='*70}\n")
    
    # Prepare tasks
    tasks = []
    for seq_dir in sequence_dirs:
        tasks.append((
            seq_dir,
            cache_root / split_name,
            seq_dir.name,
            encoder_config
        ))
    
    # Process sequences in parallel
    success_count = 0
    failed_count = 0
    
    if parallel_sequences > 1:
        print(f"[e2e] Starting parallel processing with {parallel_sequences} workers...")
        
        with Pool(processes=min(parallel_sequences, len(tasks))) as pool:
            results = pool.map(_process_single_sequence, tasks)
            
            for video_stem, success in results:
                if success:
                    success_count += 1
                    print(f"[e2e] ✓ Completed: {video_stem}")
                else:
                    failed_count += 1
                    print(f"[e2e] ✗ Failed: {video_stem}")
    else:
        # Sequential processing (original behavior)
        for idx, seq_dir in enumerate(sequence_dirs, 1):
            print(f"\n[{idx}/{len(sequence_dirs)}] Processing {seq_dir.name}...")
            
            decompress_dir = run_full_pipeline(
                sequence_dir=seq_dir,
                output_root=cache_root / split_name,
                **encoder_config
            )
            
            if decompress_dir is not None:
                success_count += 1
            else:
                failed_count += 1
    
    stats = {
        "total": len(sequence_dirs),
        "success": success_count,
        "failed": failed_count,
    }
    
    print(f"\n{'='*70}")
    print(f"[e2e] Split {split_name} complete:")
    print(f"  Total:   {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"{'='*70}\n")
    
    return stats


def process_split(
    split_name: str,
    split_root: Path,
    cache_root: Path,
    encoder_config: dict[str, Any],
    parallel_sequences: int = 1,
) -> dict[str, int]:
    """Process all sequences in a split (with optional parallelism)."""
    return process_split_parallel(
        split_name=split_name,
        split_root=split_root,
        cache_root=cache_root,
        encoder_config=encoder_config,
        parallel_sequences=parallel_sequences,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare end-to-end encoded-decoded dataset for SNN training"
    )
    
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"),
                        help="Root directory of dataset (default: datasets).")
    parser.add_argument("--train-dir", type=Path, default=None,
                        help="Optional explicit training input root where encode-decode outputs will be written. If provided, use as output root instead of <dataset-root>/.cache")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="Comma-separated splits to process (default: train,val,test)")
    
    parser.add_argument("--key-interval", type=int, default=9,
                        help="Keyframe interval (default: 9)")
    parser.add_argument("--packet-size", type=int, default=100,
                        help="Packet size in frames (default: 100)")
    parser.add_argument("--event-threshold", type=int, default=12,
                        help="Event threshold (default: 12)")
    parser.add_argument("--event-downsample", type=int, default=2,
                        help="Event downsample factor (default: 2)")
    parser.add_argument("--latency-budget-ms", type=float, default=30.0,
                        help="Latency budget in ms (default: 30)")
    parser.add_argument("--gpu-worker-count", type=int, default=2,
                        help="GPU worker count (default: 2)")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"],
                        help="Device for encoding (default: cuda)")
    parser.add_argument("--skip-if-exists", action="store_true", default=True,
                        help="Skip if outputs already exist (default: True)")
    parser.add_argument("--parallel-sequences", type=int, default=1,
                        help="Number of sequences to process in parallel (default: 1). Set >1 for multi-sequence parallel processing.")
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Output root: prefer explicit --train-dir if provided, otherwise use <dataset_root>/.cache
    cache_root = Path(args.train_dir).resolve() if getattr(args, "train_dir", None) is not None else (args.dataset_root / ".cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    
    splits = [s.strip() for s in args.splits.split(",")]
    
    encoder_config = {
        "key_interval": args.key_interval,
        "packet_size": args.packet_size,
        "event_threshold": args.event_threshold,
        "event_downsample": args.event_downsample,
        "latency_budget_ms": args.latency_budget_ms,
        "gpu_worker_count": args.gpu_worker_count,
        "device": args.device,
        "skip_if_exists": args.skip_if_exists,
    }
    
    print(f"\n{'#'*70}")
    print(f"# End-to-End Data Preparation (Encode + Decode)")
    print(f"{'#'*70}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Cache dir:    {cache_root}")
    print(f"Splits:       {', '.join(splits)}")
    print(f"Encoder config:")
    for key, value in encoder_config.items():
        print(f"  {key}: {value}")
    print(f"{'#'*70}\n")
    
    total_stats = {"total": 0, "success": 0, "failed": 0}
    
    for split_name in splits:
        split_root = args.dataset_root / split_name
        if not split_root.exists():
            print(f"[e2e] Warning: Split '{split_name}' not found at {split_root}, skipping")
            continue
        
        stats = process_split(
            split_name=split_name,
            split_root=split_root,
            cache_root=cache_root,
            encoder_config=encoder_config,
            parallel_sequences=args.parallel_sequences,
        )
        total_stats["total"] += stats["total"]
        total_stats["success"] += stats["success"]
        total_stats["failed"] += stats["failed"]
    
    print(f"\n{'='*70}")
    print(f"[e2e] Overall Summary:")
    print(f"  Total sequences:      {total_stats['total']}")
    print(f"  Successfully processed: {total_stats['success']}")
    print(f"  Failed:               {total_stats['failed']}")
    print(f"{'='*70}\n")
    
    if total_stats["failed"] > 0:
        print(f"[e2e] WARNING: {total_stats['failed']} sequences failed!")
        return 1
    
    print("[e2e] All sequences processed successfully!")
    print(f"[e2e] Cache location: {cache_root}")
    print(f"\n[e2e] Generated structure:")
    print(f"  {cache_root}/")
    print(f"  ├── train/")
    print(f"  │   └── <sequence_name>/")
    print(f"  │       └── decompress-ing/    <- ONLY dir needed for training")
    print(f"  ├── val/")
    print(f"  └── test/")
    print(f"\n[e2e] Training should use data from decompress-ing/ directories")
    print(f"[e2e] Note: Only decompress-ing/ is kept; compress/, compress-ing/, decompress/, visualization/ are removed")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())