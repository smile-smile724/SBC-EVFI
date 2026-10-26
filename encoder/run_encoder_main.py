
from __future__ import annotations
import time

import argparse
import gzip
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import tarfile
import struct
from pathlib import Path
from statistics import mean
from typing import Dict, List
try:
    import torch
except Exception:
    torch = None

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encoding-side main program: extract pixel events (Module A), "
            "select keyframes (Module B), and build hybrid transmission packets (Module C)."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Input video path.")
    group.add_argument(
        "--frames-dir",
        type=Path,
        help="Input directory containing sequential frame images (names sorted lexicographically).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root output directory; final path is output/<video_stem>/.",
    )
    parser.add_argument("--key-interval", type=int, default=10, help="Fixed keyframe interval.")
    parser.add_argument(
        "--packet-size",
        type=int,
        default=60,
        help="Transmission packet size in frames (non-overlapping).",
    )
    parser.add_argument("--event-threshold", type=int, default=12, help="Pixel-diff threshold for events.")
    parser.add_argument(
        "--event-format",
        type=str,
        choices=["npz", "jsonl"],
        default="npz",
        help="Event payload format for packets and local export.",
    )
    parser.add_argument(
        "--max-events-per-frame",
        type=int,
        default=0,
        help="Maximum events exported per frame (0 means unlimited, recommended for full transmission).",
    )
    parser.add_argument(
        "--spatial-grid-size",
        type=int,
        default=1,
        help="Spatial aggregation grid size in pixels (1 disables aggregation; larger values keep only the strongest event per grid cell).",
    )
    parser.add_argument(
        "--event-refractory-frames",
        type=int,
        default=0,
        help="Suppress repeated events at the same pixel within N frames (0 disables).",
    )
    parser.add_argument(
        "--adjacent-event-merge-connectivity",
        type=int,
        choices=[0, 4, 8],
        default=0,
        help="Merge adjacent same-polarity event blobs with connected components (0 disables; 4 or 8 sets connectivity).",
    )
    parser.add_argument(
        "--event-delta-quant-bits",
        type=int,
        choices=[8, 16],
        default=16,
        help="Quantization bits for event delta in NPZ payload (16 keeps original precision, 8 reduces payload).",
    )
    parser.add_argument(
        "--event-delta-quant-scale",
        type=int,
        default=2,
        help="Scale for delta quantization when bits=8; reconstructed delta ~= q * scale.",
    )
    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        default=30.0,
        help="Latency budget written into packet metadata.",
    )
    parser.add_argument(
        "--emit-manifest",
        action="store_true",
        help="If set, write top-level transmission manifest to compress/ (default: off).",
    )
    parser.add_argument(
        "--disable-event-visualization",
        action="store_true",
        help="Disable event visualization video export.",
    )
    parser.add_argument(
        "--packet-compression",
        type=str,
        choices=["gz", "xz", "none"],
        default="gz",
        help="Compression for packet tarball: gz (default), xz (better compression), or none (fast).",
    )
    parser.add_argument(
        "--packet-ext",
        type=str,
        default=None,
        help="Custom packet filename extension to use (e.g. .pkt or .bin). If provided, overrides default .tgz/.tar.xz/.tar.",
    )
    parser.add_argument(
        "--event-downsample",
        type=int,
        default=1,
        help="Downsample factor for event extraction (integer >=1). Events are detected on the downsampled image and coordinates are scaled back to original resolution for packaging/visualization.",
    )
    parser.add_argument(
        "--event-batch-size",
        type=int,
        default=16,
        help="Number of consecutive frame pairs to process together on the chosen device before results are merged back in time order.",
    )
    parser.add_argument(
        "--event-payload-compression",
        type=str,
        choices=["gz", "none"],
        default="gz",
        help="Compression for per-packet event payload: gz (npz compressed) or none (uncompressed npz, faster write).",
    )
    parser.add_argument(
        "--write-global-event-npz",
        action="store_true",
        help="Also write compress-ing/pixel_events.npz (disabled by default to avoid duplicate event serialization).",
    )
    parser.add_argument(
        "--gpu-worker-count",
        type=int,
        default=2,
        help="Number of CUDA worker processes to use for batched event extraction when CUDA is enabled.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device for event extraction. auto prefers CUDA when available; CPU remains the fallback.",
    )
    return parser.parse_args()


def resolve_compute_device(device_name: str) -> str:
    preferred = str(device_name or "auto").strip().lower()
    if preferred == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if preferred == "cuda" and (torch is None or not torch.cuda.is_available()):
        return "cpu"
    return preferred


def build_events_from_arrays(
    frame_id: int,
    pts_ms: int,
    xs: np.ndarray,
    ys: np.ndarray,
    deltas: np.ndarray,
    polarity: int,
) -> List[dict]:
    events: List[dict] = []
    if xs.size == 0:
        return events
    for x, y, d in zip(xs.tolist(), ys.tolist(), deltas.tolist()):
        events.append(
            {
                "frame_id": int(frame_id),
                "pts_ms": int(pts_ms),
                "x": int(x),
                "y": int(y),
                "polarity": int(polarity),
                "delta": int(d),
            }
        )
    return events


def extract_pixel_events(
    gray_prev: np.ndarray,
    gray_cur: np.ndarray,
    frame_id: int,
    pts_ms: int,
    threshold: int,
    spatial_grid_size: int,
    max_events_per_frame: int,
    compute_device: str = "cpu",
) -> tuple[List[dict], np.ndarray, np.ndarray, int, bool]:
    if compute_device == "cuda" and torch is not None and torch.cuda.is_available():
        prev_t = torch.as_tensor(gray_prev, dtype=torch.int16, device="cuda")
        cur_t = torch.as_tensor(gray_cur, dtype=torch.int16, device="cuda")
        delta_t = cur_t - prev_t

        on_mask_t = delta_t >= int(threshold)
        off_mask_t = delta_t <= -int(threshold)
        on_coords_t = torch.nonzero(on_mask_t, as_tuple=False)
        off_coords_t = torch.nonzero(off_mask_t, as_tuple=False)

        on_mask = on_mask_t.detach().cpu().numpy()
        off_mask = off_mask_t.detach().cpu().numpy()

        if on_coords_t.numel() > 0:
            on_ys = on_coords_t[:, 0].detach().cpu().numpy()
            on_xs = on_coords_t[:, 1].detach().cpu().numpy()
            on_deltas = delta_t[on_mask_t].detach().cpu().numpy()
        else:
            on_ys = np.empty((0,), dtype=np.int64)
            on_xs = np.empty((0,), dtype=np.int64)
            on_deltas = np.empty((0,), dtype=np.int64)

        if off_coords_t.numel() > 0:
            off_ys = off_coords_t[:, 0].detach().cpu().numpy()
            off_xs = off_coords_t[:, 1].detach().cpu().numpy()
            off_deltas = (-delta_t[off_mask_t]).detach().cpu().numpy()
        else:
            off_ys = np.empty((0,), dtype=np.int64)
            off_xs = np.empty((0,), dtype=np.int64)
            off_deltas = np.empty((0,), dtype=np.int64)
    else:
        delta = gray_cur.astype(np.int16) - gray_prev.astype(np.int16)

        on_mask = delta >= int(threshold)
        off_mask = delta <= -int(threshold)
        on_y, on_x = np.where(on_mask)
        off_y, off_x = np.where(off_mask)

        on_ys = on_y.astype(np.int64, copy=False)
        on_xs = on_x.astype(np.int64, copy=False)
        on_deltas = delta[on_y, on_x].astype(np.int64, copy=False)
        off_ys = off_y.astype(np.int64, copy=False)
        off_xs = off_x.astype(np.int64, copy=False)
        off_deltas = (-delta[off_y, off_x]).astype(np.int64, copy=False)

    events = build_events_from_arrays(frame_id, pts_ms, on_xs, on_ys, on_deltas, polarity=1)
    events.extend(build_events_from_arrays(frame_id, pts_ms, off_xs, off_ys, off_deltas, polarity=-1))

    raw_event_count = len(events)

    if spatial_grid_size > 1 and events:
        grid = int(spatial_grid_size)
        selected: dict[tuple[int, int], dict] = {}
        for event in events:
            cell_x = int(event["x"]) // grid
            cell_y = int(event["y"]) // grid
            key = (cell_x, cell_y)
            current = selected.get(key)
            if current is None or abs(int(event["delta"])) > abs(int(current["delta"])):
                selected[key] = event
        events = list(selected.values())

    was_truncated = False
    if max_events_per_frame > 0 and raw_event_count > max_events_per_frame:
        events.sort(key=lambda item: item["delta"], reverse=True)
        events = events[:max_events_per_frame]
        was_truncated = True

    return events, on_mask, off_mask, raw_event_count, was_truncated


def extract_pixel_events_batch(
    gray_prev_batch: List[np.ndarray],
    gray_cur_batch: List[np.ndarray],
    frame_ids: List[int],
    pts_ms_batch: List[int],
    threshold: int,
    spatial_grid_size: int,
    max_events_per_frame: int,
    compute_device: str = "cpu",
    compact_arrays_only: bool = False,
) -> List[dict]:
    batch_len = len(frame_ids)
    results: List[dict] = [
        {
            "frame_id": int(frame_ids[idx]),
            "pts_ms": int(pts_ms_batch[idx]),
            "events": [],
            "on_mask": None,
            "off_mask": None,
            "raw_count": 0,
            "was_truncated": False,
        }
        for idx in range(batch_len)
    ]

    if batch_len <= 0:
        return results

    if compute_device == "cuda" and torch is not None and torch.cuda.is_available():
        prev_stack = np.stack(gray_prev_batch, axis=0)
        cur_stack = np.stack(gray_cur_batch, axis=0)
        prev_t = torch.from_numpy(prev_stack).to(device="cuda", dtype=torch.int16)
        cur_t = torch.from_numpy(cur_stack).to(device="cuda", dtype=torch.int16)

        delta_t = cur_t - prev_t
        on_mask_t = delta_t >= int(threshold)
        off_mask_t = delta_t <= -int(threshold)

        on_coords = torch.nonzero(on_mask_t, as_tuple=False)
        off_coords = torch.nonzero(off_mask_t, as_tuple=False)
        on_values = delta_t[on_mask_t]
        off_values = (-delta_t[off_mask_t])

        on_masks = on_mask_t.detach().cpu().numpy()
        off_masks = off_mask_t.detach().cpu().numpy()
        for idx in range(batch_len):
            results[idx]["on_mask"] = on_masks[idx]
            results[idx]["off_mask"] = off_masks[idx]

        if compact_arrays_only:
            on_np = on_coords.detach().cpu().numpy() if on_coords.numel() > 0 else np.empty((0, 3), dtype=np.int64)
            off_np = off_coords.detach().cpu().numpy() if off_coords.numel() > 0 else np.empty((0, 3), dtype=np.int64)
            on_val_np = on_values.detach().cpu().numpy() if on_coords.numel() > 0 else np.empty((0,), dtype=np.int64)
            off_val_np = off_values.detach().cpu().numpy() if off_coords.numel() > 0 else np.empty((0,), dtype=np.int64)

            for idx in range(batch_len):
                on_sel = on_np[:, 0] == idx if on_np.size > 0 else np.zeros((0,), dtype=bool)
                off_sel = off_np[:, 0] == idx if off_np.size > 0 else np.zeros((0,), dtype=bool)

                on_frame = on_np[on_sel] if on_np.size > 0 else np.empty((0, 3), dtype=np.int64)
                off_frame = off_np[off_sel] if off_np.size > 0 else np.empty((0, 3), dtype=np.int64)

                x_parts: list[np.ndarray] = []
                y_parts: list[np.ndarray] = []
                polarity_parts: list[np.ndarray] = []
                delta_parts: list[np.ndarray] = []

                if on_frame.size > 0:
                    x_parts.append(on_frame[:, 2].astype(np.int16, copy=False))
                    y_parts.append(on_frame[:, 1].astype(np.int16, copy=False))
                    polarity_parts.append(np.ones(on_frame.shape[0], dtype=np.int8))
                    delta_parts.append(on_val_np[on_sel].astype(np.int16, copy=False))

                if off_frame.size > 0:
                    x_parts.append(off_frame[:, 2].astype(np.int16, copy=False))
                    y_parts.append(off_frame[:, 1].astype(np.int16, copy=False))
                    polarity_parts.append(-np.ones(off_frame.shape[0], dtype=np.int8))
                    delta_parts.append(off_val_np[off_sel].astype(np.int16, copy=False))

                if x_parts:
                    results[idx]["event_arrays"] = {
                        "x": np.concatenate(x_parts, axis=0),
                        "y": np.concatenate(y_parts, axis=0),
                        "polarity": np.concatenate(polarity_parts, axis=0),
                        "delta": np.concatenate(delta_parts, axis=0),
                    }
                else:
                    results[idx]["event_arrays"] = {
                        "x": np.empty((0,), dtype=np.int16),
                        "y": np.empty((0,), dtype=np.int16),
                        "polarity": np.empty((0,), dtype=np.int8),
                        "delta": np.empty((0,), dtype=np.int16),
                    }

                results[idx]["raw_count"] = int(results[idx]["event_arrays"]["x"].size)
                results[idx]["was_truncated"] = False

            return results

        if on_coords.numel() > 0:
            on_np = on_coords.detach().cpu().numpy()
            on_val_np = on_values.detach().cpu().numpy()
            for row in range(on_np.shape[0]):
                batch_idx = int(on_np[row, 0])
                y = int(on_np[row, 1])
                x = int(on_np[row, 2])
                results[batch_idx]["events"].append(
                    {
                        "frame_id": int(frame_ids[batch_idx]),
                        "pts_ms": int(pts_ms_batch[batch_idx]),
                        "x": int(x),
                        "y": int(y),
                        "polarity": 1,
                        "delta": int(on_val_np[row]),
                    }
                )

        if off_coords.numel() > 0:
            off_np = off_coords.detach().cpu().numpy()
            off_val_np = off_values.detach().cpu().numpy()
            for row in range(off_np.shape[0]):
                batch_idx = int(off_np[row, 0])
                y = int(off_np[row, 1])
                x = int(off_np[row, 2])
                results[batch_idx]["events"].append(
                    {
                        "frame_id": int(frame_ids[batch_idx]),
                        "pts_ms": int(pts_ms_batch[batch_idx]),
                        "x": int(x),
                        "y": int(y),
                        "polarity": -1,
                        "delta": int(off_val_np[row]),
                    }
                )

        for item in results:
            raw_events = item["events"]
            raw_count = len(raw_events)
            if spatial_grid_size > 1 and raw_events:
                grid = int(spatial_grid_size)
                selected: dict[tuple[int, int], dict] = {}
                for event in raw_events:
                    cell_x = int(event["x"]) // grid
                    cell_y = int(event["y"]) // grid
                    key = (cell_x, cell_y)
                    current = selected.get(key)
                    if current is None or abs(int(event["delta"])) > abs(int(current["delta"])):
                        selected[key] = event
                raw_events = list(selected.values())

            was_truncated = False
            if max_events_per_frame > 0 and raw_count > max_events_per_frame:
                raw_events.sort(key=lambda item: item["delta"], reverse=True)
                raw_events = raw_events[:max_events_per_frame]
                was_truncated = True

            item["events"] = raw_events
            item["raw_count"] = raw_count
            item["was_truncated"] = was_truncated

        return results

    for idx in range(batch_len):
        events, on_mask, off_mask, raw_count, was_truncated = extract_pixel_events(
            gray_prev=gray_prev_batch[idx],
            gray_cur=gray_cur_batch[idx],
            frame_id=frame_ids[idx],
            pts_ms=pts_ms_batch[idx],
            threshold=threshold,
            spatial_grid_size=spatial_grid_size,
            max_events_per_frame=max_events_per_frame,
            compute_device="cpu",
        )
        results[idx]["events"] = events
        results[idx]["on_mask"] = on_mask
        results[idx]["off_mask"] = off_mask
        results[idx]["raw_count"] = raw_count
        results[idx]["was_truncated"] = was_truncated

    return results


def extract_pixel_events_batch_worker(payload: tuple) -> list[dict]:
    (
        local_device_index,
        gray_prev_batch,
        gray_cur_batch,
        frame_ids,
        pts_ms_batch,
        threshold,
        spatial_grid_size,
        max_events_per_frame,
        compact_arrays_only,
    ) = payload

    if torch is None or not torch.cuda.is_available():
        return extract_pixel_events_batch(
            gray_prev_batch=list(gray_prev_batch),
            gray_cur_batch=list(gray_cur_batch),
            frame_ids=list(frame_ids),
            pts_ms_batch=list(pts_ms_batch),
            threshold=int(threshold),
            spatial_grid_size=int(spatial_grid_size),
            max_events_per_frame=int(max_events_per_frame),
            compute_device="cpu",
            compact_arrays_only=bool(compact_arrays_only),
        )

    torch.cuda.set_device(int(local_device_index))
    return extract_pixel_events_batch(
        gray_prev_batch=list(gray_prev_batch),
        gray_cur_batch=list(gray_cur_batch),
        frame_ids=list(frame_ids),
        pts_ms_batch=list(pts_ms_batch),
        threshold=int(threshold),
        spatial_grid_size=int(spatial_grid_size),
        max_events_per_frame=int(max_events_per_frame),
        compute_device="cuda",
        compact_arrays_only=bool(compact_arrays_only),
    )


def split_ranges(total_items: int, parts: int) -> list[tuple[int, int]]:
    total_items = max(0, int(total_items))
    parts = max(1, int(parts))
    if total_items == 0:
        return []
    parts = min(parts, total_items)
    base = total_items // parts
    remainder = total_items % parts
    ranges: list[tuple[int, int]] = []
    start = 0
    for idx in range(parts):
        size = base + (1 if idx < remainder else 0)
        end = start + size
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


def compose_event_visualization(frame_bgr: np.ndarray, on_mask: np.ndarray, off_mask: np.ndarray) -> np.ndarray:
    event_map = np.zeros_like(frame_bgr, dtype=np.uint8)
    event_map[on_mask] = (0, 255, 0)
    event_map[off_mask] = (0, 0, 255)
    overlay = cv2.addWeighted(frame_bgr, 0.75, event_map, 0.85, 0.0)
    return np.hstack([frame_bgr, overlay])


def compose_event_visualization_from_events(frame_bgr: np.ndarray, events: List[dict]) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    on_mask = np.zeros((h, w), dtype=bool)
    off_mask = np.zeros((h, w), dtype=bool)
    for e in events:
        x = int(e.get("x", -1))
        y = int(e.get("y", -1))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if int(e.get("polarity", 1)) >= 0:
            on_mask[y, x] = True
        else:
            off_mask[y, x] = True
    return compose_event_visualization(frame_bgr, on_mask, off_mask)


def apply_event_refractory(
    events: List[dict],
    frame_id: int,
    refractory_frames: int,
    last_on_frame: np.ndarray,
    last_off_frame: np.ndarray,
) -> tuple[List[dict], int]:
    if refractory_frames <= 0 or not events:
        return events, 0

    kept: List[dict] = []
    dropped = 0
    limit = int(refractory_frames)
    fid = int(frame_id)

    for ev in events:
        x = int(ev.get("x", -1))
        y = int(ev.get("y", -1))
        if x < 0 or y < 0 or y >= last_on_frame.shape[0] or x >= last_on_frame.shape[1]:
            continue

        pol = int(ev.get("polarity", 1))
        if pol >= 0:
            prev = int(last_on_frame[y, x])
            if prev >= 0 and (fid - prev) <= limit:
                dropped += 1
                continue
            last_on_frame[y, x] = fid
        else:
            prev = int(last_off_frame[y, x])
            if prev >= 0 and (fid - prev) <= limit:
                dropped += 1
                continue
            last_off_frame[y, x] = fid

        kept.append(ev)

    return kept, dropped


def merge_adjacent_same_change_events(
    events: List[dict],
    frame_shape: tuple[int, int],
    connectivity: int,
) -> tuple[List[dict], int, int]:
    if connectivity not in (4, 8) or len(events) <= 1:
        return events, 0, 0

    h, w = int(frame_shape[0]), int(frame_shape[1])
    if h <= 0 or w <= 0:
        return events, 0, 0

    representative_indices: List[int] = []
    dropped_total = 0
    merged_component_total = 0

    for polarity in (1, -1):
        polarity_indices = [idx for idx, event in enumerate(events) if int(event.get("polarity", 1)) == polarity]
        if not polarity_indices:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        for idx in polarity_indices:
            x = int(events[idx].get("x", -1))
            y = int(events[idx].get("y", -1))
            if 0 <= x < w and 0 <= y < h:
                mask[y, x] = 1

        num_labels, labels = cv2.connectedComponents(mask, connectivity=int(connectivity))
        if num_labels <= 1:
            representative_indices.extend(polarity_indices)
            continue

        groups: dict[int, List[int]] = {}
        for idx in polarity_indices:
            x = int(events[idx].get("x", -1))
            y = int(events[idx].get("y", -1))
            if 0 <= x < w and 0 <= y < h:
                label = int(labels[y, x])
                if label > 0:
                    groups.setdefault(label, []).append(idx)

        for label in sorted(groups):
            group = groups[label]
            if not group:
                continue
            if len(group) > 1:
                merged_component_total += 1
                dropped_total += len(group) - 1

            best_idx = max(
                group,
                key=lambda idx: (
                    abs(int(events[idx].get("delta", 0))),
                    int(events[idx].get("delta", 0)),
                    -int(idx),
                ),
            )
            representative_indices.append(int(best_idx))

    if not representative_indices:
        return [], dropped_total, merged_component_total

    representative_indices = sorted(set(representative_indices))
    merged_events = [events[idx] for idx in representative_indices]
    return merged_events, dropped_total, merged_component_total


def encode_png_lossless(frame_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(
        ".png",
        frame_bgr,
        [int(cv2.IMWRITE_PNG_COMPRESSION), 0],
    )
    if not ok:
        raise RuntimeError("failed to encode keyframe to png")
    return bytes(buf)


def read_file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def write_custom_packet(meta_payload: bytes, entries: list[tuple[str, bytes]]) -> bytes:
    header = {
        "format": "mspkt-v1",
        "meta_size": len(meta_payload),
        "entries": [{"name": name, "size": len(payload)} for name, payload in entries],
    }
    header_payload = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    out = io.BytesIO()
    out.write(b"MSPKT001")
    out.write(struct.pack("<I", len(header_payload)))
    out.write(header_payload)
    out.write(meta_payload)
    for _, payload in entries:
        out.write(payload)
    return out.getvalue()


def add_bytes_to_tar(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def build_event_payload(
    events: List[dict],
    event_format: str,
    delta_quant_bits: int = 16,
    delta_quant_scale: int = 2,
    compress: bool = True,
) -> tuple[str, bytes]:
    if event_format == "jsonl":
        payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in events).encode("utf-8")
        return "events.jsonl", payload

    # NPZ v2 frame-bucket layout: store frame_id/pts_ms once per frame and use
    # offsets to index contiguous event arrays, reducing repeated metadata.
    events_sorted = sorted(events, key=lambda e: (int(e["frame_id"]), int(e["pts_ms"]), int(e["y"]), int(e["x"])))

    frame_ids: List[int] = []
    frame_pts_ms: List[int] = []
    frame_offsets: List[int] = [0]
    x_list: List[int] = []
    y_list: List[int] = []
    polarity_list: List[int] = []
    delta_list: List[int] = []

    current_fid = None
    current_pts = 0
    current_count = 0
    for ev in events_sorted:
        fid = int(ev["frame_id"])
        pts = int(ev["pts_ms"])
        if current_fid is None:
            current_fid = fid
            current_pts = pts
        elif fid != current_fid:
            frame_ids.append(int(current_fid))
            frame_pts_ms.append(int(current_pts))
            frame_offsets.append(frame_offsets[-1] + current_count)
            current_fid = fid
            current_pts = pts
            current_count = 0

        x_list.append(int(ev["x"]))
        y_list.append(int(ev["y"]))
        polarity_list.append(int(ev["polarity"]))
        delta_list.append(int(ev["delta"]))
        current_count += 1

    if current_fid is not None:
        frame_ids.append(int(current_fid))
        frame_pts_ms.append(int(current_pts))
        frame_offsets.append(frame_offsets[-1] + current_count)

    delta_scale = max(1, int(delta_quant_scale))
    if int(delta_quant_bits) == 8:
        delta_arr = np.asarray(np.round(np.asarray(delta_list, dtype=np.float32) / float(delta_scale)), dtype=np.int16)
        delta_arr = np.clip(delta_arr, -127, 127).astype(np.int8)
    else:
        delta_arr = np.asarray(delta_list, dtype=np.int16)
        delta_scale = 1

    buf = io.BytesIO()
    if compress:
        np.savez_compressed(
            buf,
            layout_version=np.asarray([2], dtype=np.int8),
            delta_quant_bits=np.asarray([int(delta_quant_bits)], dtype=np.int8),
            delta_scale=np.asarray([int(delta_scale)], dtype=np.int16),
            frame_ids=np.asarray(frame_ids, dtype=np.int32),
            frame_pts_ms=np.asarray(frame_pts_ms, dtype=np.int32),
            frame_offsets=np.asarray(frame_offsets, dtype=np.int64),
            x=np.asarray(x_list, dtype=np.int16),
            y=np.asarray(y_list, dtype=np.int16),
            polarity=np.asarray(polarity_list, dtype=np.int8),
            delta=delta_arr,
        )
    else:
        np.savez(
            buf,
            layout_version=np.asarray([2], dtype=np.int8),
            delta_quant_bits=np.asarray([int(delta_quant_bits)], dtype=np.int8),
            delta_scale=np.asarray([int(delta_scale)], dtype=np.int16),
            frame_ids=np.asarray(frame_ids, dtype=np.int32),
            frame_pts_ms=np.asarray(frame_pts_ms, dtype=np.int32),
            frame_offsets=np.asarray(frame_offsets, dtype=np.int64),
            x=np.asarray(x_list, dtype=np.int16),
            y=np.asarray(y_list, dtype=np.int16),
            polarity=np.asarray(polarity_list, dtype=np.int8),
            delta=delta_arr,
        )
    return "events.npz", buf.getvalue()


def build_event_payload_from_frame_arrays(
    frame_event_arrays_by_frame: Dict[int, dict[str, np.ndarray]],
    frame_order: List[int],
    delta_quant_bits: int = 16,
    delta_quant_scale: int = 2,
    compress: bool = True,
) -> tuple[str, bytes]:
    frame_ids: List[int] = []
    frame_pts_ms: List[int] = []
    frame_offsets: List[int] = [0]
    x_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    polarity_chunks: List[np.ndarray] = []
    delta_chunks: List[np.ndarray] = []

    total_count = 0
    for frame_id in frame_order:
        record = frame_event_arrays_by_frame.get(int(frame_id))
        if not record:
            continue
        x_arr = np.asarray(record.get("x", []), dtype=np.int16)
        y_arr = np.asarray(record.get("y", []), dtype=np.int16)
        polarity_arr = np.asarray(record.get("polarity", []), dtype=np.int8)
        delta_arr = np.asarray(record.get("delta", []), dtype=np.int16)
        pts_ms = int(record.get("pts_ms", 0))

        frame_ids.append(int(frame_id))
        frame_pts_ms.append(int(pts_ms))
        x_chunks.append(x_arr)
        y_chunks.append(y_arr)
        polarity_chunks.append(polarity_arr)
        delta_chunks.append(delta_arr)
        total_count += int(x_arr.size)
        frame_offsets.append(total_count)

    if delta_quant_bits == 8:
        if delta_chunks:
            delta_concat = np.concatenate(delta_chunks, axis=0).astype(np.float32)
            delta_scale = max(1, int(delta_quant_scale))
            delta_arr = np.asarray(np.round(delta_concat / float(delta_scale)), dtype=np.int16)
            delta_arr = np.clip(delta_arr, -127, 127).astype(np.int8)
        else:
            delta_arr = np.empty((0,), dtype=np.int8)
            delta_scale = max(1, int(delta_quant_scale))
    else:
        delta_arr = np.concatenate(delta_chunks, axis=0).astype(np.int16) if delta_chunks else np.empty((0,), dtype=np.int16)
        delta_scale = 1

    buf = io.BytesIO()
    if compress:
        np.savez_compressed(
            buf,
            layout_version=np.asarray([2], dtype=np.int8),
            delta_quant_bits=np.asarray([int(delta_quant_bits)], dtype=np.int8),
            delta_scale=np.asarray([int(delta_scale)], dtype=np.int16),
            frame_ids=np.asarray(frame_ids, dtype=np.int32),
            frame_pts_ms=np.asarray(frame_pts_ms, dtype=np.int32),
            frame_offsets=np.asarray(frame_offsets, dtype=np.int64),
            x=np.concatenate(x_chunks, axis=0) if x_chunks else np.empty((0,), dtype=np.int16),
            y=np.concatenate(y_chunks, axis=0) if y_chunks else np.empty((0,), dtype=np.int16),
            polarity=np.concatenate(polarity_chunks, axis=0) if polarity_chunks else np.empty((0,), dtype=np.int8),
            delta=delta_arr,
        )
    else:
        np.savez(
            buf,
            layout_version=np.asarray([2], dtype=np.int8),
            delta_quant_bits=np.asarray([int(delta_quant_bits)], dtype=np.int8),
            delta_scale=np.asarray([int(delta_scale)], dtype=np.int16),
            frame_ids=np.asarray(frame_ids, dtype=np.int32),
            frame_pts_ms=np.asarray(frame_pts_ms, dtype=np.int32),
            frame_offsets=np.asarray(frame_offsets, dtype=np.int64),
            x=np.concatenate(x_chunks, axis=0) if x_chunks else np.empty((0,), dtype=np.int16),
            y=np.concatenate(y_chunks, axis=0) if y_chunks else np.empty((0,), dtype=np.int16),
            polarity=np.concatenate(polarity_chunks, axis=0) if polarity_chunks else np.empty((0,), dtype=np.int8),
            delta=delta_arr,
        )
    return "events.npz", buf.getvalue()


def build_packet_blob(
    packet_meta: dict,
    keyframe_file_by_id: Dict[int, tuple[str, bytes]],
    events: List[dict],
    event_format: str,
    delta_quant_bits: int,
    delta_quant_scale: int,
    compression: str = "gz",
    packet_ext: str | None = None,
    event_payload_override: tuple[str, bytes] | None = None,
) -> tuple[bytes, str]:
    if event_payload_override is not None:
        event_name, event_payload = event_payload_override
    else:
        event_name, event_payload = build_event_payload(
            events=events,
            event_format=event_format,
            delta_quant_bits=delta_quant_bits,
            delta_quant_scale=delta_quant_scale,
        )
    meta_payload = json.dumps(packet_meta, ensure_ascii=False, indent=2).encode("utf-8")

    if packet_ext is not None:
        entries: list[tuple[str, bytes]] = []
        for frame_id in sorted(keyframe_file_by_id):
            archive_name, payload = keyframe_file_by_id[frame_id]
            entries.append((archive_name, payload))
        entries.append((event_name, event_payload))
        return write_custom_packet(meta_payload, entries), event_name

    buffer = io.BytesIO()
    mode = "w:gz" if compression == "gz" else ("w:xz" if compression == "xz" else "w")
    with tarfile.open(fileobj=buffer, mode=mode) as tar:
        add_bytes_to_tar(tar, "meta.json", meta_payload)
        for frame_id in sorted(keyframe_file_by_id):
            archive_name, payload = keyframe_file_by_id[frame_id]
            add_bytes_to_tar(tar, archive_name, payload)
        add_bytes_to_tar(tar, event_name, event_payload)
    return buffer.getvalue(), event_name


def p95(values: List[int]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * 0.95))
    return float(ordered[idx])


def build_packet_windows(total_frames: int, packet_len: int) -> List[dict]:
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if packet_len <= 0:
        raise ValueError("packet_len must be > 0")
    windows: List[dict] = []
    start = 0
    window_id = 0
    while start < total_frames:
        end = min(start + packet_len - 1, total_frames - 1)
        span = end - start + 1
        is_tail_fallback = end == (total_frames - 1) and span < packet_len
        windows.append(
            {
                "window_id": window_id,
                "start_frame_id": int(start),
                "end_frame_id": int(end),
                "is_tail_fallback": bool(is_tail_fallback),
                "reason": "tail_fallback" if is_tail_fallback else "fixed_interval",
            }
        )
        if end == total_frames - 1:
            break
        start += packet_len
        window_id += 1
    return windows


def build_keyframe_ids(total_frames: int, key_interval: int) -> List[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if key_interval <= 0:
        raise ValueError("key_interval must be > 0")

    ids = list(range(0, total_frames, key_interval))
    last_frame_id = total_frames - 1
    if ids[-1] != last_frame_id:
        ids.append(last_frame_id)
    return ids


def main() -> int:
    # ====== timing: full pipeline and per-stage time statistics ======
    t0 = time.perf_counter()
    t_event_start = t0
    args = parse_args()
    compute_device = resolve_compute_device(args.device)
    if args.device == "cuda" and compute_device != "cuda":
        print("[warn] CUDA requested but not available; falling back to CPU")
    elif compute_device == "cuda":
        print("[info] using CUDA for event extraction")
    # If user requests a custom packet extension, force no compression to keep .pkt/.bin raw
    if args.packet_ext:
        args.packet_compression = "none"

    # support either a video file or a directory of frames
    cap = None
    frame_paths = None
    frame_index = 0
    if args.frames_dir is not None:
        if not args.frames_dir.exists():
            raise FileNotFoundError(f"frames directory not found: {args.frames_dir}")
        if not args.frames_dir.is_dir():
            raise NotADirectoryError(f"frames path is not a directory: {args.frames_dir}")
        # gather image files sorted lexicographically (assumes zero-padded names like 00000000.png)
        frame_paths = sorted([p for p in args.frames_dir.iterdir() if p.is_file()])
        total_frames = len(frame_paths)
        if total_frames <= 0:
            raise RuntimeError(f"no frame files found in: {args.frames_dir}")
        # read first frame to probe width/height
        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError(f"failed to read first frame: {frame_paths[0]}")
        fps = 30.0
        source_height, source_width = first.shape[:2]
        video_stem = args.frames_dir.name
        video_output_dir = args.output_root / video_stem
    else:
        if args.video is None:
            raise RuntimeError("either --video or --frames-dir must be provided")
        if not args.video.exists():
            raise FileNotFoundError(f"video not found: {args.video}")
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {args.video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError("invalid total frame count from source video")
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_stem = args.video.stem
        video_output_dir = args.output_root / video_stem
    compress_ing_dir = video_output_dir / "compress-ing"
    compress_dir = video_output_dir / "compress"
    visualization_dir = video_output_dir / "visualization"

    compress_ing_dir.mkdir(parents=True, exist_ok=True)
    compress_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    if args.packet_size <= 0:
        raise ValueError("packet_size must be > 0")

    packet_windows = build_packet_windows(
        total_frames=total_frames,
        packet_len=args.packet_size,
    )
    keyframe_ids = build_keyframe_ids(total_frames=total_frames, key_interval=args.key_interval)
    keyframe_set = set(keyframe_ids)

    keyframe_file_by_id: Dict[int, tuple[str, bytes]] = {}
    events_by_frame: Dict[int, List[dict]] = {}
    frame_raw_event_count: Dict[int, int] = {}
    gray_prev = None
    last_frame_bgr = None
    truncated_frame_count = 0
    dropped_event_total = 0
    raw_event_total = 0
    exported_event_total = 0
    refractory_dropped_total = 0
    adjacent_merge_dropped_total = 0
    adjacent_merge_component_total = 0
    last_on_frame: np.ndarray | None = None
    last_off_frame: np.ndarray | None = None

    events_jsonl_gz_path = compress_ing_dir / "pixel_events.jsonl.gz"
    events_npz_path = compress_ing_dir / "pixel_events.npz"
    all_events: List[dict] = []
    events_fp = None
    if args.event_format == "jsonl":
        events_fp = gzip.open(events_jsonl_gz_path, "wt", encoding="utf-8")

    vis_video_path = visualization_dir / "event_overlay.mp4"
    vis_writer = None
    vis_enabled = not args.disable_event_visualization
    event_batch_size = max(1, int(args.event_batch_size))
    fast_npz_mode = (
        args.event_format == "npz"
        and int(args.event_refractory_frames) <= 0
        and int(args.adjacent_event_merge_connectivity) == 0
        and int(args.spatial_grid_size) <= 1
        and int(args.max_events_per_frame) == 0
    )
    gpu_worker_count = 1
    if compute_device == "cuda" and torch is not None and torch.cuda.is_available():
        visible_gpu_count = max(1, int(torch.cuda.device_count()))
        gpu_worker_count = max(1, min(int(args.gpu_worker_count), visible_gpu_count))
    batch_executor = None
    if compute_device == "cuda" and gpu_worker_count > 1:
        batch_executor = ProcessPoolExecutor(max_workers=gpu_worker_count, mp_context=get_context("spawn"))
    frame_buffer: List[dict] = []
    frame_event_arrays_by_frame: Dict[int, dict[str, np.ndarray]] = {}

    def consume_buffered_pairs(batch_pairs: int) -> None:
        nonlocal last_on_frame, last_off_frame
        nonlocal vis_writer
        nonlocal truncated_frame_count, dropped_event_total, raw_event_total, exported_event_total
        nonlocal refractory_dropped_total, adjacent_merge_dropped_total, adjacent_merge_component_total

        if batch_pairs <= 0 or len(frame_buffer) <= 1:
            return

        batch_pairs = min(batch_pairs, len(frame_buffer) - 1)
        batch_prev = [frame_buffer[idx]["gray"] for idx in range(batch_pairs)]
        batch_cur = [frame_buffer[idx + 1]["gray"] for idx in range(batch_pairs)]
        batch_frame_ids = [int(frame_buffer[idx + 1]["frame_id"]) for idx in range(batch_pairs)]
        batch_pts_ms = [int(frame_buffer[idx + 1]["pts_ms"]) for idx in range(batch_pairs)]

        if batch_executor is not None and gpu_worker_count > 1 and batch_pairs > 1:
            sub_ranges = split_ranges(batch_pairs, gpu_worker_count)
            futures = []
            for worker_idx, (start, end) in enumerate(sub_ranges):
                payload = (
                    int(worker_idx % gpu_worker_count),
                    batch_prev[start:end],
                    batch_cur[start:end],
                    batch_frame_ids[start:end],
                    batch_pts_ms[start:end],
                    int(args.event_threshold),
                    int(args.spatial_grid_size),
                    int(args.max_events_per_frame),
                    bool(fast_npz_mode),
                )
                futures.append((start, end, batch_executor.submit(extract_pixel_events_batch_worker, payload)))

            batch_results = [None] * batch_pairs
            for start, end, future in futures:
                sub_results = future.result()
                for offset, result in enumerate(sub_results):
                    batch_results[start + offset] = result
        else:
            batch_results = extract_pixel_events_batch(
                gray_prev_batch=batch_prev,
                gray_cur_batch=batch_cur,
                frame_ids=batch_frame_ids,
                pts_ms_batch=batch_pts_ms,
                threshold=args.event_threshold,
                spatial_grid_size=args.spatial_grid_size,
                max_events_per_frame=args.max_events_per_frame,
                compute_device=compute_device,
                compact_arrays_only=fast_npz_mode,
            )

        for idx, result in enumerate(batch_results):
            current_item = frame_buffer[idx + 1]
            current_frame_bgr = current_item["frame_bgr"]
            frame_id_local = int(result["frame_id"])
            raw_count = int(result.get("raw_count", 0))
            was_truncated = bool(result.get("was_truncated", False))

            if fast_npz_mode:
                event_arrays = result.get("event_arrays", {})
                x_arr = np.asarray(event_arrays.get("x", []), dtype=np.int16)
                y_arr = np.asarray(event_arrays.get("y", []), dtype=np.int16)
                polarity_arr = np.asarray(event_arrays.get("polarity", []), dtype=np.int8)
                delta_arr = np.asarray(event_arrays.get("delta", []), dtype=np.int16)
                frame_event_arrays_by_frame[frame_id_local] = {
                    "pts_ms": int(result["pts_ms"]),
                    "x": x_arr,
                    "y": y_arr,
                    "polarity": polarity_arr,
                    "delta": delta_arr,
                }
                frame_events = []
                raw_count = int(x_arr.size)
            else:
                frame_events = list(result.get("events", []))

            if int(args.event_refractory_frames) > 0 and frame_events:
                h, w = current_item["gray"].shape[:2]
                if last_on_frame is None or last_off_frame is None:
                    last_on_frame = np.full((h, w), -1, dtype=np.int32)
                    last_off_frame = np.full((h, w), -1, dtype=np.int32)
                frame_events, dropped_ref = apply_event_refractory(
                    events=frame_events,
                    frame_id=frame_id_local,
                    refractory_frames=args.event_refractory_frames,
                    last_on_frame=last_on_frame,
                    last_off_frame=last_off_frame,
                )
                refractory_dropped_total += int(dropped_ref)

            if int(args.adjacent_event_merge_connectivity) in (4, 8) and frame_events:
                frame_events, dropped_adjacent, merged_components = merge_adjacent_same_change_events(
                    events=frame_events,
                    frame_shape=current_item["gray"].shape[:2],
                    connectivity=int(args.adjacent_event_merge_connectivity),
                )
                adjacent_merge_dropped_total += int(dropped_adjacent)
                adjacent_merge_component_total += int(merged_components)

            frame_raw_event_count[frame_id_local] = int(raw_count)
            raw_event_total += int(raw_count)
            if was_truncated:
                truncated_frame_count += 1

            exported_count = int(frame_event_arrays_by_frame[frame_id_local]["x"].size) if fast_npz_mode else int(len(frame_events))
            exported_event_total += int(exported_count)
            dropped_event_total += int(raw_count - exported_count)

            events_by_frame[frame_id_local] = frame_events
            if args.event_format == "jsonl":
                for item in frame_events:
                    events_fp.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif not fast_npz_mode:
                all_events.extend(frame_events)

            source_path = current_item.get("source_path")
            if frame_id_local in keyframe_set:
                if source_path is not None:
                    source_name = Path(source_path).name
                    keyframe_file_by_id[frame_id_local] = (f"keyframes/{source_name}", read_file_bytes(Path(source_path)))
                else:
                    keyframe_file_by_id[frame_id_local] = (
                        f"keyframes/frame_{frame_id_local:04d}.png",
                        encode_png_lossless(current_frame_bgr),
                    )

            if vis_enabled:
                if fast_npz_mode:
                    on_mask = result.get("on_mask")
                    off_mask = result.get("off_mask")
                    if on_mask is None or off_mask is None:
                        vis_frame = current_frame_bgr
                    elif ds > 1:
                        on_mask_full = np.repeat(np.repeat(on_mask, ds, axis=0), ds, axis=1)
                        off_mask_full = np.repeat(np.repeat(off_mask, ds, axis=0), ds, axis=1)
                        on_mask_full = on_mask_full[:source_height, :source_width]
                        off_mask_full = off_mask_full[:source_height, :source_width]
                        vis_frame = compose_event_visualization(current_frame_bgr, on_mask_full, off_mask_full)
                    else:
                        vis_frame = compose_event_visualization(current_frame_bgr, on_mask, off_mask)
                elif ds > 1 and frame_events:
                    vis_events: List[dict] = []
                    for ev in frame_events:
                        vx = int(min(source_width - 1, int(ev.get("x", 0)) * ds + ds // 2))
                        vy = int(min(source_height - 1, int(ev.get("y", 0)) * ds + ds // 2))
                        vis_ev = dict(ev)
                        vis_ev["x"] = int(vx)
                        vis_ev["y"] = int(vy)
                        vis_events.append(vis_ev)
                    vis_frame = compose_event_visualization_from_events(current_frame_bgr, vis_events)
                else:
                    vis_frame = compose_event_visualization_from_events(current_frame_bgr, frame_events)
                if vis_writer is None:
                    h, w = vis_frame.shape[:2]
                    vis_writer = cv2.VideoWriter(
                        str(vis_video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(1.0, fps),
                        (w, h),
                    )
                vis_writer.write(vis_frame)

        del frame_buffer[:batch_pairs]


    # ====== Stage 1: event extraction, keyframe selection, visualization (per-frame) ======
    t_event_visual_start = time.perf_counter()
    frame_id = 0
    ds = int(max(1, args.event_downsample))
    while True:
        if frame_paths is not None:
            if frame_index >= total_frames:
                break
            frame_bgr = cv2.imread(str(frame_paths[frame_index]))
            ok = frame_bgr is not None
            frame_index += 1
        else:
            ok, frame_bgr = cap.read()
        if not ok:
            break

        # prepare original gray and optionally downsample for event extraction
        gray_cur_orig = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if ds > 1:
            small_w = max(1, source_width // ds)
            small_h = max(1, source_height // ds)
            gray_cur = cv2.resize(gray_cur_orig, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            gray_cur = gray_cur_orig

        last_frame_bgr = frame_bgr
        pts_ms = int(round((frame_id * 1000.0) / fps))

        current_source_path = frame_paths[frame_index - 1] if frame_paths is not None else None
        frame_buffer.append(
            {
                "frame_id": int(frame_id),
                "frame_bgr": frame_bgr,
                "gray": gray_cur,
                "pts_ms": int(pts_ms),
                "source_path": current_source_path,
            }
        )

        if frame_id in keyframe_set:
            if current_source_path is not None:
                keyframe_file_by_id[frame_id] = (
                    f"keyframes/{Path(current_source_path).name}",
                    read_file_bytes(Path(current_source_path)),
                )
            else:
                keyframe_file_by_id[frame_id] = (f"keyframes/frame_{frame_id:04d}.png", encode_png_lossless(frame_bgr))

        while len(frame_buffer) > event_batch_size:
            consume_buffered_pairs(event_batch_size)

        frame_id += 1

    while len(frame_buffer) > 1:
        consume_buffered_pairs(len(frame_buffer) - 1)

    if batch_executor is not None:
        batch_executor.shutdown(wait=True)

    t_event_visual_end = time.perf_counter()

    # ====== Stage 2: event stream write, keyframe write ======
    t_write_start = time.perf_counter()


    if cap is not None:
        cap.release()
    if events_fp is not None:
        events_fp.close()
    if args.event_format == "npz":
        if args.write_global_event_npz:
            if fast_npz_mode:
                _name, payload = build_event_payload_from_frame_arrays(
                    frame_event_arrays_by_frame=frame_event_arrays_by_frame,
                    frame_order=sorted(frame_event_arrays_by_frame.keys()),
                    delta_quant_bits=args.event_delta_quant_bits,
                    delta_quant_scale=args.event_delta_quant_scale,
                    compress=(args.event_payload_compression == "gz"),
                )
            else:
                _name, payload = build_event_payload(
                    events=all_events,
                    event_format="npz",
                    delta_quant_bits=args.event_delta_quant_bits,
                    delta_quant_scale=args.event_delta_quant_scale,
                    compress=(args.event_payload_compression == "gz"),
                )
            events_npz_path.write_bytes(payload)
        elif events_npz_path.exists():
            events_npz_path.unlink()
        if events_jsonl_gz_path.exists():
            events_jsonl_gz_path.unlink()
    else:
        if events_npz_path.exists():
            events_npz_path.unlink()
    if vis_writer is not None:
        vis_writer.release()
    t_write_end = time.perf_counter()


    # ====== Stage 3: packaging ======
    t_pack_start = time.perf_counter()


    if frame_id != total_frames:
        total_frames = frame_id
        packet_windows = build_packet_windows(
            total_frames=total_frames,
            packet_len=args.packet_size,
        )
        keyframe_ids = build_keyframe_ids(total_frames=total_frames, key_interval=args.key_interval)
        keyframe_set = set(keyframe_ids)

    last_frame_id = total_frames - 1
    if last_frame_id in keyframe_set and last_frame_id not in keyframe_file_by_id:
        if last_frame_bgr is None:
            raise RuntimeError("tail fallback frame missing")
        if frame_paths is not None:
            source_path = frame_paths[last_frame_id]
            keyframe_file_by_id[last_frame_id] = (f"keyframes/{source_path.name}", read_file_bytes(source_path))
        else:
            keyframe_file_by_id[last_frame_id] = (f"keyframes/frame_{last_frame_id:04d}.png", encode_png_lossless(last_frame_bgr))


    keyframes_dir = compress_ing_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    for fid, (archive_name, payload) in keyframe_file_by_id.items():
        out_name = Path(archive_name).name
        (keyframes_dir / out_name).write_bytes(payload)

    keyframe_index_path = compress_ing_dir / "keyframe_index.json"
    keyframe_index_payload = {
        "source_video": str(args.video),
        "total_frames": int(total_frames),
        "fps": float(fps),
        "packet_compression": str(args.packet_compression),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "key_interval": int(args.key_interval),
        "keyframe_ids": keyframe_ids,
    }
    keyframe_index_path.write_text(json.dumps(keyframe_index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    t_pack_end = time.perf_counter()

    packets_dir = compress_dir
    packets_dir.mkdir(parents=True, exist_ok=True)

    packet_entries: List[dict] = []
    packet_sizes: List[int] = []

    for i, window in enumerate(packet_windows):
        start_fid = int(window["start_frame_id"])
        end_fid = int(window["end_frame_id"])

        window_events: List[dict] = []
        packet_event_payload_override: tuple[str, bytes] | None = None
        if fast_npz_mode:
            frame_order = [fid for fid in range(start_fid, end_fid + 1) if fid in frame_event_arrays_by_frame]
            event_name_tmp, event_payload_tmp = build_event_payload_from_frame_arrays(
                frame_event_arrays_by_frame=frame_event_arrays_by_frame,
                frame_order=frame_order,
                delta_quant_bits=args.event_delta_quant_bits,
                delta_quant_scale=args.event_delta_quant_scale,
                compress=(args.event_payload_compression == "gz"),
            )
            packet_event_payload_override = (event_name_tmp, event_payload_tmp)
            window_event_count = int(
                sum(np.asarray(frame_event_arrays_by_frame[fid]["x"]).size for fid in frame_order)
            )
        else:
            for fid in range(start_fid, end_fid + 1):
                window_events.extend(events_by_frame.get(fid, []))
            window_event_count = int(len(window_events))

        packet_keyframe_ids = [fid for fid in keyframe_ids if start_fid <= fid <= end_fid]
        if not packet_keyframe_ids:
            raise RuntimeError(f"no keyframe in packet window [{start_fid}, {end_fid}]")

        packet_keyframes: Dict[int, tuple[str, bytes]] = {}
        for fid in packet_keyframe_ids:
            payload = keyframe_file_by_id.get(fid)
            if payload is None:
                raise RuntimeError(f"missing keyframe payload for frame_id={fid}")
            packet_keyframes[fid] = payload

        # codec reflects chosen packet container and actual keyframe extension(s)
        if args.packet_ext:
            codec = f"mspkt-v1(raw+{args.event_format})"
        else:
            exts: set[str] = set()
            for _fid, (archive_name, _payload) in packet_keyframes.items():
                ext = Path(archive_name).suffix.lower().lstrip(".")
                if not ext:
                    ext = "bin"
                exts.add(ext)
            if len(exts) == 1:
                kf_ext = next(iter(exts))
            else:
                kf_ext = "mixed(" + ",".join(sorted(exts)) + ")"

            if args.packet_compression == "gz":
                codec = f"tar+gzip({kf_ext}+{args.event_format})"
            elif args.packet_compression == "xz":
                codec = f"tar+xz({kf_ext}+{args.event_format})"
            else:
                codec = f"tar({kf_ext}+{args.event_format})"

        packet_meta = {
            "packet_id": i,
            "start_frame_id": int(start_fid),
            "end_frame_id": int(end_fid),
            "codec": codec,
            "event_format": args.event_format,
            "event_downsample": int(args.event_downsample),
            "latency_budget_ms": float(args.latency_budget_ms),
            "keyframe_ids": packet_keyframe_ids,
            "keyframe_reason": str(window["reason"]),
            "event_count": int(window_event_count),
            "event_count_raw": int(sum(frame_raw_event_count.get(fid, 0) for fid in range(start_fid, end_fid + 1))),
        }
        # Provide an explicit per-frame mapping to prev/next keyframe ids to ease decoder-side forward/back splits
        frame_keyframe_map: Dict[int, Dict[str, int]] = {}
        for fid in range(start_fid, end_fid + 1):
            # find prev and next keyframe from global keyframe_ids
            prev_k = keyframe_ids[0]
            next_k = keyframe_ids[-1]
            for kid in keyframe_ids:
                if kid <= fid:
                    prev_k = kid
                if kid >= fid:
                    next_k = kid
                    break
            frame_keyframe_map[int(fid)] = {"prev_keyframe": int(prev_k), "next_keyframe": int(next_k)}
        packet_meta["frame_keyframe_map"] = frame_keyframe_map
        raw_count = int(packet_meta["event_count_raw"])
        if raw_count > 0:
            packet_meta["event_retention_ratio"] = float(packet_meta["event_count"] / raw_count)
        else:
            packet_meta["event_retention_ratio"] = 1.0

        packet_blob, event_name = build_packet_blob(
            packet_meta=packet_meta,
            keyframe_file_by_id=packet_keyframes,
            events=window_events,
            event_format=args.event_format,
            delta_quant_bits=args.event_delta_quant_bits,
            delta_quant_scale=args.event_delta_quant_scale,
            compression=args.packet_compression,
            packet_ext=args.packet_ext,
            event_payload_override=packet_event_payload_override,
        )
        checksum = hashlib.sha256(packet_blob).hexdigest()

        ext_map = {"gz": ".tgz", "xz": ".tar.xz", "none": ".tar"}
        if args.packet_ext:
            ext = args.packet_ext if args.packet_ext.startswith(".") else "." + args.packet_ext
        else:
            ext = ext_map.get(args.packet_compression, ".tgz")
        packet_filename = f"packet_{i:04d}_{start_fid:04d}_{end_fid:04d}" + ext
        packet_path = packets_dir / packet_filename
        packet_path.write_bytes(packet_blob)

        size_bytes = len(packet_blob)
        packet_sizes.append(size_bytes)
        keyframe_refs = [
            str(packet_path.relative_to(video_output_dir)) + f"::{packet_keyframes[fid][0]}"
            for fid in packet_keyframe_ids
        ]
        packet_entries.append(
            {
                "packet_id": i,
                "start_frame_id": int(start_fid),
                "end_frame_id": int(end_fid),
                "keyframe_refs": keyframe_refs,
                "event_chunk_ref": str(packet_path.relative_to(video_output_dir)) + f"::{event_name}",
                "codec": codec,
                "event_format": args.event_format,
                "event_count": int(packet_meta["event_count"]),
                "event_count_raw": int(packet_meta["event_count_raw"]),
                "event_retention_ratio": float(packet_meta["event_retention_ratio"]),
                "size_bytes": int(size_bytes),
                "latency_budget_ms": float(args.latency_budget_ms),
                "checksum": checksum,
                "is_tail_fallback": bool(window["is_tail_fallback"]),
                "window_id": int(window["window_id"]),
                "reason": str(window["reason"]),
            }
        )

    # ====== timing summary ======
    t1 = time.perf_counter()
    timing = {
        "total_wall_sec": t1 - t0,
        "event_visualization_sec": t_event_visual_end - t_event_visual_start,
        "event_and_keyframe_write_sec": t_write_end - t_write_start,
        "packets_build_sec": t_pack_end - t_pack_start,
        "total_excluding_event_visualization_sec": (t1 - t0) - (t_event_visual_end - t_event_visual_start),
    }

    bundle_manifest = {
        "source_video": str(args.video),
        "total_frames": int(total_frames),
        "fps": float(fps),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "key_interval": int(args.key_interval),
        "packet_size": int(args.packet_size),
        "packet_count": len(packet_entries),
        "keyframe_count": len(keyframe_ids),
        "max_events_per_frame": int(args.max_events_per_frame),
        "spatial_grid_size": int(args.spatial_grid_size),
        "event_refractory_frames": int(args.event_refractory_frames),
        "event_refractory_dropped_total": int(refractory_dropped_total),
        "adjacent_event_merge_connectivity": int(args.adjacent_event_merge_connectivity),
        "adjacent_event_merge_dropped_total": int(adjacent_merge_dropped_total),
        "adjacent_event_merge_component_total": int(adjacent_merge_component_total),
        "events_truncated": bool(truncated_frame_count > 0),
        "truncated_frame_count": int(truncated_frame_count),
        "dropped_event_total": int(dropped_event_total),
        "event_count_raw_total": int(raw_event_total),
        "event_count_exported_total": int(exported_event_total),
        "event_retention_ratio": float((exported_event_total / raw_event_total) if raw_event_total > 0 else 1.0),
        "event_format": args.event_format,
        "event_delta_quant_bits": int(args.event_delta_quant_bits),
        "event_delta_quant_scale": int(args.event_delta_quant_scale),
        "tail_fallback_count": 1 if keyframe_ids and keyframe_ids[-1] == (total_frames - 1) and (total_frames - 1) % args.key_interval != 0 else 0,
        "packet_size_bytes_mean": float(mean(packet_sizes)) if packet_sizes else 0.0,
        "packet_size_bytes_p95": p95(packet_sizes),
        "output_layout": {
            "compress_ing": str(compress_ing_dir.relative_to(video_output_dir)),
            "compress": str(compress_dir.relative_to(video_output_dir)),
            "visualization": str(visualization_dir.relative_to(video_output_dir)),
            "event_jsonl_gz": None if args.event_format == "npz" else str(events_jsonl_gz_path.relative_to(video_output_dir)),
            "event_npz": str(events_npz_path.relative_to(video_output_dir)) if (args.event_format == "npz" and args.write_global_event_npz) else None,
            "keyframe_index": str(keyframe_index_path.relative_to(video_output_dir)),
            "event_visualization_video": None if args.disable_event_visualization else str(vis_video_path.relative_to(video_output_dir)),
        },
        "packets": packet_entries,
        "meta": {
            "timing": timing
        },
    }

    if args.emit_manifest:
        manifest_path = compress_dir / "transmission_bundle.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(bundle_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        # ensure we do NOT leave a transmissible top-level manifest in `compress/`
        candidate = compress_dir / "transmission_bundle.json"
        if candidate.exists():
            try:
                candidate.unlink()
            except Exception:
                pass
        # keep a local debug copy in compress-ing for traceability
        local_manifest = compress_ing_dir / "transmission_bundle.local.json"
        local_manifest.write_text(json.dumps(bundle_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[encoder-main] video={args.video}")
    print(f"[encoder-main] total_frames={total_frames} fps={fps:.6f}")
    print(f"[encoder-main] key_interval={args.key_interval} packet_size={args.packet_size}")
    print(f"[encoder-main] packets={len(packet_entries)} tail_fallback={bundle_manifest['tail_fallback_count']}")
    print(
        f"[encoder-main] events exported/raw={bundle_manifest['event_count_exported_total']}/{bundle_manifest['event_count_raw_total']} "
        f"retention={bundle_manifest['event_retention_ratio']:.4f}"
    )
    if int(args.adjacent_event_merge_connectivity) in (4, 8):
        print(
            f"[encoder-main] adjacent_merge_conn={args.adjacent_event_merge_connectivity} "
            f"dropped={adjacent_merge_dropped_total} components={adjacent_merge_component_total}"
        )
    print(f"[encoder-main] mean_packet_bytes={bundle_manifest['packet_size_bytes_mean']:.2f}")
    print(f"[encoder-main] output_dir={video_output_dir}")
    if args.emit_manifest:
        print(f"[encoder-main] manifest={manifest_path}")
    else:
        print(f"[encoder-main] manifest= (not emitted; use --emit-manifest to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())