from __future__ import annotations

import argparse
import io
import json
import tarfile
import struct
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode-side pipeline: unpack hybrid packets and render a decode-side visualization video."
    )
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=Path("output") / "demo" / "compress",
        help="Directory containing packet_* tarballs (any supported compression).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root output directory; final path is output/<video_stem>/.",
    )
    parser.add_argument(
        "--video-stem",
        type=str,
        default="",
        help="Video stem used to resolve output/<video_stem>/; inferred from packet-dir when omitted.",
    )
    parser.add_argument(
        "--visualization-fps",
        type=float,
        default=30.0,
        help="FPS for decode-side visualization video.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("output") / "demo" / "compress-ing" / "transmission_bundle.local.json",
        help="Optional encoder-side local manifest used to infer fps and total_frames.",
    )
    parser.add_argument(
        "--disable-visualization",
        action="store_true",
        help="Only unpack packets and write stats, without a video preview.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device for decoder-side processing (for future GPU-based inference).",
    )
    parser.add_argument(
        "--fast-npz",
        action="store_true",
        help="Enable fast NPZ path: keep frame-bucket arrays instead of immediate per-event dict expansion.",
    )
    parser.add_argument(
        "--event-payload-compression",
        type=str,
        choices=["gz", "none"],
        default="gz",
        help="Event payload compression mode (informational; decoder will auto-detect payload).",
    )
    return parser.parse_args()


def infer_video_stem(packet_dir: Path, explicit_stem: str) -> str:
    if explicit_stem:
        return explicit_stem
    if packet_dir.name == "compress" and packet_dir.parent.name:
        return packet_dir.parent.name
    return packet_dir.name


def load_source_manifest(source_manifest: Path) -> dict[str, Any]:
    if not source_manifest.exists():
        return {}
    try:
        return json.loads(source_manifest.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_packet_files(packet_dir: Path) -> list[Path]:
    candidates = sorted([p for p in packet_dir.iterdir() if p.is_file() and p.name.startswith("packet_")])
    if not candidates:
        raise FileNotFoundError(f"no packet_ files found in {packet_dir}")
    return candidates


def read_tar_bytes(tar: tarfile.TarFile, member_name: str) -> bytes | None:
    try:
        member = tar.getmember(member_name)
    except KeyError:
        return None
    extracted = tar.extractfile(member)
    if extracted is None:
        return None
    return extracted.read()


def unpack_custom_packet(packet_path: Path, packet_out: Path | None = None, args: argparse.Namespace | None = None) -> dict[str, Any]:
    data = packet_path.read_bytes()
    if len(data) < 12 or data[:8] != b"MSPKT001":
        raise ValueError("not a custom packet")
    header_len = struct.unpack_from("<I", data, 8)[0]
    header_start = 12
    header_end = header_start + header_len
    header = json.loads(data[header_start:header_end].decode("utf-8"))
    meta_size = int(header.get("meta_size", 0))
    entries = header.get("entries", []) or []

    cursor = header_end
    meta_bytes = data[cursor : cursor + meta_size]
    cursor += meta_size

    summary: dict[str, Any] = {
        "packet_name": packet_path.name,
        "packet_out": None,
        "size_bytes": packet_path.stat().st_size,
        "meta": json.loads(meta_bytes.decode("utf-8")) if meta_bytes else {},
        "event_support": 1,
        "spatial_grid_size": 1,
        "keyframes": [],
        "keyframe_ids": [],
        "frame_keyframe_map": {},
        "event_count": 0,
        "event_payload_name": None,
    }
    if isinstance(summary["meta"].get("keyframe_ids"), list):
        summary["keyframe_ids"] = [int(v) for v in summary["meta"].get("keyframe_ids", [])]
    summary["event_support"] = int(summary["meta"].get("event_downsample", 1) or 1)
    summary["spatial_grid_size"] = int(summary["meta"].get("spatial_grid_size", 1) or 1)
    raw_frame_keyframe_map = summary["meta"].get("frame_keyframe_map", {})
    if isinstance(raw_frame_keyframe_map, dict):
        for frame_id, mapping in raw_frame_keyframe_map.items():
            if not isinstance(mapping, dict):
                continue
            try:
                fid = int(frame_id)
                prev_keyframe = int(mapping["prev_keyframe"])
                next_keyframe = int(mapping["next_keyframe"])
            except Exception:
                continue
            summary["frame_keyframe_map"][fid] = {
                "prev_keyframe": prev_keyframe,
                "next_keyframe": next_keyframe,
            }

    for idx, entry in enumerate(entries):
        name = str(entry.get("name", ""))
        size = int(entry.get("size", 0))
        payload = data[cursor : cursor + size]
        cursor += size
        # write payload to disk if packet_out provided
        if packet_out is not None:
            target = packet_out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

        if name.startswith("keyframes/"):
            # load image for in-memory summary as before
            summary["keyframes"].append(load_image(payload))
        else:
            summary["event_payload_name"] = name
            events_or_arrays = decode_events_payload(name, payload, fast_npz=(args.fast_npz if args is not None else False))
            if isinstance(events_or_arrays, dict):
                # fast path: keep arrays grouped under `event_arrays`
                summary.setdefault("event_arrays", {}).update(events_or_arrays)
                # compute count if offsets present
                if "frame_offsets" in events_or_arrays:
                    offsets = events_or_arrays.get("frame_offsets", [])
                    total = int(offsets[-1]) if offsets else 0
                else:
                    total = int(sum(len(v) for v in events_or_arrays.get("x", [])))
                summary["event_count"] += total
            else:
                summary["event_count"] += len(events_or_arrays)
                summary.setdefault("events", []).extend(events_or_arrays)

    return summary


def decode_events_payload(payload_name: str, payload_bytes: bytes, fast_npz: bool = False) -> list[dict[str, Any]] | dict[str, Any]:
    if payload_name.endswith(".npz"):
        buf = io.BytesIO(payload_bytes)
        with np.load(buf, allow_pickle=False) as data:
            # v2 frame-bucket layout
            if "frame_offsets" in data and "frame_ids" in data:
                frame_ids = data["frame_ids"].tolist()
                frame_pts = data["frame_pts_ms"].tolist() if "frame_pts_ms" in data else [0] * len(frame_ids)
                frame_offsets = data["frame_offsets"].tolist()
                delta_scale = int(data["delta_scale"][0]) if "delta_scale" in data else 1
                if fast_npz:
                    # return compact arrays and metadata
                    return {
                        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
                        "frame_pts_ms": np.asarray(frame_pts, dtype=np.int32),
                        "frame_offsets": np.asarray(frame_offsets, dtype=np.int64),
                        "delta_scale": int(delta_scale),
                        "x": data["x"] if "x" in data else np.empty((0,), dtype=np.int16),
                        "y": data["y"] if "y" in data else np.empty((0,), dtype=np.int16),
                        "polarity": data["polarity"] if "polarity" in data else np.empty((0,), dtype=np.int8),
                        "delta": data["delta"] if "delta" in data else np.empty((0,), dtype=np.int16),
                    }

                x = data["x"].tolist() if "x" in data else []
                y = data["y"].tolist() if "y" in data else []
                polarity = data["polarity"].tolist() if "polarity" in data else []
                delta = data["delta"].tolist() if "delta" in data else []

                events = []
                for i, fid in enumerate(frame_ids):
                    start = int(frame_offsets[i])
                    end = int(frame_offsets[i + 1])
                    pts = int(frame_pts[i]) if i < len(frame_pts) else 0
                    for idx in range(start, end):
                        events.append(
                            {
                                "frame_id": int(fid),
                                "pts_ms": int(pts),
                                "x": int(x[idx]),
                                "y": int(y[idx]),
                                "polarity": int(polarity[idx]),
                                "delta": int(delta[idx]) * int(delta_scale),
                            }
                        )
                return events

            # legacy flat layout
            frame_id = data["frame_id"].tolist() if "frame_id" in data else []
            pts_ms = data["pts_ms"].tolist() if "pts_ms" in data else []
            x = data["x"].tolist() if "x" in data else []
            y = data["y"].tolist() if "y" in data else []
            polarity = data["polarity"].tolist() if "polarity" in data else []
            delta = data["delta"].tolist() if "delta" in data else []

            events = []
            for idx in range(len(frame_id)):
                events.append(
                    {
                        "frame_id": int(frame_id[idx]),
                        "pts_ms": int(pts_ms[idx]),
                        "x": int(x[idx]),
                        "y": int(y[idx]),
                        "polarity": int(polarity[idx]),
                        "delta": int(delta[idx]),
                    }
                )
            return events

    # fallback: jsonl text
    text = payload_bytes.decode("utf-8")
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def load_image(payload: bytes) -> np.ndarray:
    data = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("failed to decode image payload")
    return frame


def restore_event_coordinates(
    events: list[dict[str, Any]],
    event_support: int,
    source_h: int,
    source_w: int,
) -> list[dict[str, Any]]:
    support = max(1, int(event_support))
    if support <= 1 or not events:
        return events

    restored: list[dict[str, Any]] = []
    for event in events:
        x = int(event.get("x", -1))
        y = int(event.get("y", -1))
        if x < 0 or y < 0:
            continue
        restored_event = dict(event)
        restored_event["x"] = int(min(source_w - 1, x * support + support // 2))
        restored_event["y"] = int(min(source_h - 1, y * support + support // 2))
        restored.append(restored_event)
    return restored


def draw_text_block(canvas: np.ndarray, lines: list[str], origin=(18, 34), line_gap: int = 26) -> None:
    x0, y0 = origin
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            str(line),
            (x0, y0 + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )


def make_overlay_frame(base_frame: np.ndarray, events: list[dict[str, Any]], event_support: int = 1) -> np.ndarray:
    event_layer = np.zeros_like(base_frame, dtype=np.float32)
    h, w = base_frame.shape[:2]
    support = max(1, int(event_support))
    k = max(3, support * 4 + 1)
    if k % 2 == 0:
        k += 1
    for event in events:
        x = int(event["x"])
        y = int(event["y"])
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if int(event.get("polarity", 1)) >= 0:
            event_layer[y, x, 1] += 1.0
        else:
            event_layer[y, x, 2] += 1.0

    event_layer = cv2.GaussianBlur(event_layer, (k, k), 0)
    event_layer = np.clip(event_layer * 320.0, 0, 255).astype(np.uint8)

    # Keep base luminance unchanged; only add colored event hints.
    return cv2.addWeighted(base_frame, 0.55, event_layer, 1.15, 0.0)


def make_event_base_frame(events: list[dict[str, Any]], frame_shape: tuple[int, int], event_support: int = 1) -> np.ndarray:
    h, w = frame_shape[:2]
    pos = np.zeros((h, w), dtype=np.float32)
    neg = np.zeros((h, w), dtype=np.float32)
    support = max(1, int(event_support))
    k = max(5, support * 6 + 1)
    if k % 2 == 0:
        k += 1

    for event in events:
        x = int(event.get("x", -1))
        y = int(event.get("y", -1))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        mag = float(max(1, int(event.get("delta", 1))))
        if int(event.get("polarity", 1)) >= 0:
            pos[y, x] += mag
        else:
            neg[y, x] += mag

    pos = cv2.GaussianBlur(pos, (k, k), 0)
    neg = cv2.GaussianBlur(neg, (k, k), 0)

    # log scaling keeps dense bursts visible while retaining weak events
    pos = np.log1p(pos)
    neg = np.log1p(neg)
    scale = max(float(pos.max()), float(neg.max()), 1e-6)
    pos = np.clip(pos / scale, 0.0, 1.0)
    neg = np.clip(neg / scale, 0.0, 1.0)

    # strong, saturated color mapping: positive=cyan/green, negative=magenta/red
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :, 1] = np.clip(pos * 255.0, 0, 255).astype(np.uint8)
    canvas[:, :, 0] = np.clip(pos * 120.0 + neg * 255.0, 0, 255).astype(np.uint8)
    canvas[:, :, 2] = np.clip(neg * 255.0, 0, 255).astype(np.uint8)

    # emphasize event field so it visually dominates the preview
    canvas = cv2.addWeighted(canvas, 1.0, cv2.GaussianBlur(canvas, (0, 0), 2.2), 0.45, 0.0)
    bg = np.zeros_like(canvas)
    return cv2.addWeighted(bg, 0.0, canvas, 1.0, 0.0)


def add_keyframe_corner_inset(canvas: np.ndarray, keyframe: np.ndarray | None, label: str = "keyframe") -> np.ndarray:
    if keyframe is None:
        return canvas
    h, w = canvas.shape[:2]
    inset_w = max(120, w // 5)
    inset_h = max(68, h // 7)
    inset = cv2.resize(keyframe, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    pad = 18
    x0 = max(0, w - inset_w - pad)
    y0 = max(0, h - inset_h - pad)
    x1 = min(w, x0 + inset_w)
    y1 = min(h, y0 + inset_h)

    # dark backing plate so the inset does not blend into the event field
    canvas[y0 - 4 : y1 + 4, x0 - 4 : x1 + 4] = 0
    canvas[y0:y1, x0:x1] = inset
    cv2.rectangle(canvas, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (255, 255, 255), 2)
    cv2.putText(canvas, label, (x0, max(24, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def group_events_by_frame(events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        frame_id = int(event.get("frame_id", -1))
        if frame_id < 0:
            continue
        grouped.setdefault(frame_id, []).append(event)
    return grouped


def build_keyframe_timeline(packet_summaries: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    keyframes: dict[int, np.ndarray] = {}
    for summary in packet_summaries:
        meta = summary.get("meta", {})
        keyframe_ids = meta.get("keyframe_ids", []) or []
        frames = summary.get("keyframes", []) or []
        for frame_id, frame in zip(keyframe_ids, frames):
            keyframes[int(frame_id)] = frame
    return keyframes


def infer_frame_span(packet_summaries: list[dict[str, Any]]) -> tuple[int, int]:
    starts = []
    ends = []
    for summary in packet_summaries:
        meta = summary.get("meta", {})
        if "start_frame_id" in meta:
            starts.append(int(meta["start_frame_id"]))
        if "end_frame_id" in meta:
            ends.append(int(meta["end_frame_id"]))
    if not starts or not ends:
        return 0, -1
    return min(starts), max(ends)


def infer_source_video_params(source_manifest: dict[str, Any], keyframe_by_id: dict[int, np.ndarray]) -> tuple[float, int, int]:
    fps = float(source_manifest.get("fps", 30.0) or 30.0)

    width = int(source_manifest.get("source_width", 0) or 0)
    height = int(source_manifest.get("source_height", 0) or 0)

    if (width <= 0 or height <= 0) and keyframe_by_id:
        first = keyframe_by_id[min(keyframe_by_id.keys())]
        height, width = first.shape[:2]

    if width <= 0 or height <= 0:
        width, height = 1280, 720

    return fps, width, height


def apply_events_to_luma(reference_frame: np.ndarray, events: list[dict[str, Any]], event_support: int = 1) -> np.ndarray:
    # Events were extracted from grayscale deltas; apply them to luma to keep keyframe chroma.
    ycrcb = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2YCrCb)
    y_ref = ycrcb[:, :, 0].astype(np.float32)
    h, w = y_ref.shape
    support = max(1, int(event_support))
    k = max(3, support * 4 + 1)
    if k % 2 == 0:
        k += 1
    pos_map = np.zeros((h, w), dtype=np.float32)
    neg_map = np.zeros((h, w), dtype=np.float32)

    for event in events:
        x = int(event.get("x", -1))
        y_pos = int(event.get("y", -1))
        if x < 0 or y_pos < 0 or x >= w or y_pos >= h:
            continue
        delta = int(event.get("delta", 0))
        if int(event.get("polarity", 1)) < 0:
            neg_map[y_pos, x] += float(max(1, abs(delta)))
        else:
            pos_map[y_pos, x] += float(max(1, abs(delta)))

    pos_map = cv2.GaussianBlur(pos_map, (k, k), 0)
    neg_map = cv2.GaussianBlur(neg_map, (k, k), 0)
    event_energy = np.maximum(pos_map, neg_map)
    if event_energy.max() <= 1e-6:
        ycrcb[:, :, 0] = y_ref.astype(np.uint8)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Normalize event response and use it strictly as a replacement region, not a blend.
    event_mask = event_energy >= float(max(0.02, 0.08 * float(event_energy.max())))
    y_event = y_ref.copy()
    pos_norm = pos_map / float(max(pos_map.max(), 1e-6))
    neg_norm = neg_map / float(max(neg_map.max(), 1e-6))

    # positive events brighten, negative events darken; the event region is overwritten directly.
    y_event = np.clip(y_event + (pos_norm * 120.0) - (neg_norm * 120.0), 0, 255)
    y_out = y_ref.copy()
    y_out[event_mask] = y_event[event_mask]

    ycrcb[:, :, 0] = y_out.astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def pack_overview_frame(packet_meta: dict[str, Any], keyframes: list[np.ndarray], events: list[dict[str, Any]], frame_size=(1280, 720)) -> np.ndarray:
    canvas = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    canvas[:] = (18, 18, 24)

    header_lines = [
        f"packet_id={packet_meta.get('packet_id')}  window={packet_meta.get('start_frame_id')}-{packet_meta.get('end_frame_id')}  size_bytes={packet_meta.get('size_bytes')}",
        f"keyframes={len(keyframes)}  events={len(events)}  codec={packet_meta.get('codec')}",
        f"latency_budget_ms={packet_meta.get('latency_budget_ms')}  reason={packet_meta.get('reason')}",
    ]
    draw_text_block(canvas, header_lines, origin=(18, 34), line_gap=28)

    usable_top = 90
    left_w = frame_size[0] // 2 - 18
    right_w = frame_size[0] - left_w - 36
    pane_h = frame_size[1] - usable_top - 18

    if keyframes:
        base = keyframes[0]
    else:
        base = np.zeros((pane_h, left_w, 3), dtype=np.uint8)

    base = cv2.resize(base, (left_w, pane_h), interpolation=cv2.INTER_AREA)
    overlay = make_overlay_frame(base, events)

    canvas[usable_top : usable_top + pane_h, 18 : 18 + left_w] = base
    canvas[usable_top : usable_top + pane_h, 18 + left_w + 18 : 18 + left_w + 18 + right_w] = cv2.resize(
        overlay, (right_w, pane_h), interpolation=cv2.INTER_AREA
    )

    bottom_y = usable_top + pane_h - 28
    cv2.putText(canvas, "raw keyframe", (18, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "event overlay",
        (18 + left_w + 24, bottom_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def pack_frame_reconstruction_frame(
    frame_id: int,
    pts_ms: int,
    reference_frame: np.ndarray,
    events: list[dict[str, Any]],
    packet_meta: dict[str, Any],
    frame_size=(1280, 720),
) -> np.ndarray:
    canvas = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    canvas[:] = (18, 18, 24)

    header_lines = [
        f"frame_id={frame_id}  packet={packet_meta.get('packet_id')}  window={packet_meta.get('start_frame_id')}-{packet_meta.get('end_frame_id')}",
        f"pts_ms={pts_ms}  events_in_frame={len(events)}  ref_keyframe={packet_meta.get('ref_keyframe_id')}",
        f"codec={packet_meta.get('codec')}  latency_budget_ms={packet_meta.get('latency_budget_ms')}  reconstruction_preview=frame-wise",
    ]
    draw_text_block(canvas, header_lines, origin=(18, 34), line_gap=28)

    usable_top = 90
    left_w = frame_size[0] // 2 - 18
    right_w = frame_size[0] - left_w - 36
    pane_h = frame_size[1] - usable_top - 18

    base = cv2.resize(reference_frame, (left_w, pane_h), interpolation=cv2.INTER_AREA)
    overlay = make_overlay_frame(base, events)

    canvas[usable_top : usable_top + pane_h, 18 : 18 + left_w] = base
    canvas[usable_top : usable_top + pane_h, 18 + left_w + 18 : 18 + left_w + 18 + right_w] = cv2.resize(
        overlay, (right_w, pane_h), interpolation=cv2.INTER_AREA
    )

    bottom_y = usable_top + pane_h - 28
    cv2.putText(canvas, "reference keyframe", (18, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "frame-wise event overlay",
        (18 + left_w + 24, bottom_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def unpack_packet(packet_path: Path, unpack_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    # normalize stem: strip known compression extensions so unpack dir is clean
    packet_name = packet_path.name
    packet_stem = packet_name
    for ext in (".tar.xz", ".tar.gz", ".tgz", ".tar", ".gz", ".xz"):
        if packet_stem.endswith(ext):
            packet_stem = packet_stem[: -len(ext)]
            break
    parts = packet_stem.split("_")
    packet_id = parts[1] if len(parts) > 1 else packet_stem
    packet_out = unpack_root / packet_stem
    packet_out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "packet_name": packet_path.name,
        "packet_out": str(packet_out),
        "size_bytes": packet_path.stat().st_size,
        "meta": {},
        "event_support": 1,
        "keyframes": [],
        "keyframe_ids": [],
        "frame_keyframe_map": {},
        "event_count": 0,
        "event_payload_name": None,
    }

    raw = packet_path.read_bytes()
    if len(raw) >= 8 and raw[:8] == b"MSPKT001":
        packet_out = unpack_root / packet_stem
        packet_out.mkdir(parents=True, exist_ok=True)
        summary = unpack_custom_packet(packet_path, packet_out, args)
        summary["packet_out"] = str(packet_out)
        # write a light debug manifest to unpack dir
        (packet_out / "meta.json").write_text(json.dumps(summary.get("meta", {}), ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    # allow tarfile to auto-detect compression (gz, xz, none)
    with tarfile.open(packet_path, mode="r:*") as tar:
        members = tar.getmembers()
        for member in members:
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read()
            target = packet_out / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        meta_bytes = read_tar_bytes(tar, "meta.json")
        if meta_bytes is not None:
            summary["meta"] = json.loads(meta_bytes.decode("utf-8"))
            if isinstance(summary["meta"].get("keyframe_ids"), list):
                summary["keyframe_ids"] = [int(v) for v in summary["meta"].get("keyframe_ids", [])]
            summary["event_support"] = int(summary["meta"].get("event_downsample", 1) or 1)
            summary["spatial_grid_size"] = int(summary["meta"].get("spatial_grid_size", 1) or 1)
            raw_frame_keyframe_map = summary["meta"].get("frame_keyframe_map", {})
            if isinstance(raw_frame_keyframe_map, dict):
                for frame_id, mapping in raw_frame_keyframe_map.items():
                    if not isinstance(mapping, dict):
                        continue
                    try:
                        fid = int(frame_id)
                        prev_keyframe = int(mapping["prev_keyframe"])
                        next_keyframe = int(mapping["next_keyframe"])
                    except Exception:
                        continue
                    summary["frame_keyframe_map"][fid] = {
                        "prev_keyframe": prev_keyframe,
                        "next_keyframe": next_keyframe,
                    }

        keyframe_members = sorted([m.name for m in members if m.name.startswith("keyframes/") and not m.isdir()])
        for name in keyframe_members:
            payload = read_tar_bytes(tar, name)
            if payload is None:
                continue
            frame = load_image(payload)
            summary["keyframes"].append(frame)

        event_member = next((m.name for m in members if m.name.endswith(".npz") or m.name.endswith(".jsonl")), None)
        if event_member is not None:
            payload = read_tar_bytes(tar, event_member)
            if payload is not None:
                summary["event_payload_name"] = event_member
                events_or_arrays = decode_events_payload(event_member, payload, fast_npz=args.fast_npz)
                if isinstance(events_or_arrays, dict):
                    summary.setdefault("event_arrays", {}).update(events_or_arrays)
                    # compute total events
                    offsets = events_or_arrays.get("frame_offsets")
                    total = int(offsets[-1]) if offsets is not None and len(offsets) > 0 else 0
                    summary["event_count"] = total
                else:
                    summary["event_count"] = len(events_or_arrays)
                    summary["events"] = events_or_arrays

    return summary


def main() -> int:
    args = parse_args()
    packet_dir = args.packet_dir.resolve()
    if not packet_dir.exists():
        raise FileNotFoundError(f"packet dir not found: {packet_dir}")

    source_manifest = load_source_manifest(args.source_manifest.resolve())

    video_stem = infer_video_stem(packet_dir, args.video_stem)
    video_output_dir = args.output_root / video_stem
    decompress_ing_dir = video_output_dir / "decompress-ing"
    decompress_dir = video_output_dir / "decompress"
    visualization_dir = video_output_dir / "visualization"

    decompress_ing_dir.mkdir(parents=True, exist_ok=True)
    decompress_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    packet_files = list_packet_files(packet_dir)
    unpack_root = decompress_ing_dir / "packets"
    unpack_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for packet_path in packet_files:
        summaries.append(unpack_packet(packet_path, unpack_root, args))

    event_by_frame: dict[int, list[dict[str, Any]]] = {}
    for summary in summaries:
        if "events" in summary and summary.get("events"):
            for event in summary.get("events", []):
                frame_id = int(event.get("frame_id", -1))
                if frame_id < 0:
                    continue
                event_by_frame.setdefault(frame_id, []).append(event)
        elif "event_arrays" in summary:
            arrs = summary["event_arrays"]
            frame_ids = list(np.asarray(arrs.get("frame_ids", [])).tolist())
            offsets = list(np.asarray(arrs.get("frame_offsets", [0])).tolist())
            x = np.asarray(arrs.get("x", []))
            y = np.asarray(arrs.get("y", []))
            polarity = np.asarray(arrs.get("polarity", []))
            delta = np.asarray(arrs.get("delta", []))
            delta_scale = int(arrs.get("delta_scale", 1))
            for i, fid in enumerate(frame_ids):
                start = int(offsets[i])
                end = int(offsets[i + 1]) if i + 1 < len(offsets) else int(len(x))
                pts = int(np.asarray(arrs.get("frame_pts_ms", [0] * len(frame_ids))[i]))
                events_list = []
                for idx in range(start, end):
                    events_list.append(
                        {
                            "frame_id": int(fid),
                            "pts_ms": int(pts),
                            "x": int(x[idx]),
                            "y": int(y[idx]),
                            "polarity": int(polarity[idx]),
                            "delta": int(delta[idx]) * int(delta_scale),
                        }
                    )
                if events_list:
                    event_by_frame.setdefault(int(fid), []).extend(events_list)

    keyframe_by_id = build_keyframe_timeline(summaries)
    start_frame_id, end_frame_id = infer_frame_span(summaries)
    if end_frame_id < start_frame_id:
        end_frame_id = max(keyframe_by_id.keys(), default=-1)

    source_fps, source_width, source_height = infer_source_video_params(source_manifest, keyframe_by_id)

    packet_metas = [summary.get("meta", {}) for summary in summaries]
    packet_windows: list[dict[str, Any]] = []
    for item in summaries:
        meta = item.get("meta", {})
        start = int(meta.get("start_frame_id", -1))
        end = int(meta.get("end_frame_id", -1))
        if start < 0 or end < start:
            continue
        packet_windows.append(
            {
                "packet_name": item["packet_name"],
                "packet_id": int(meta.get("packet_id", -1)),
                "start_frame_id": start,
                "end_frame_id": end,
                "frame_count": end - start + 1,
                "event_count": int(item.get("event_count", 0)),
                "size_bytes": int(item.get("size_bytes", 0)),
                "reason": meta.get("keyframe_reason", meta.get("reason", "")),
            }
        )

    packet_sizes = [int(item["size_bytes"]) for item in summaries]
    event_counts = [int(item.get("event_count", 0)) for item in summaries]
    event_support = max(1, max((int(item.get("event_support", 1) or 1) for item in summaries), default=1))
    spatial_grid_size = max(1, max((int(item.get("spatial_grid_size", 1) or 1) for item in summaries), default=1))
    event_render_support = max(event_support, spatial_grid_size)
    frame_keyframe_map: dict[int, dict[str, int]] = {}
    for item in summaries:
        for frame_id, mapping in item.get("frame_keyframe_map", {}).items():
            if not isinstance(mapping, dict):
                continue
            try:
                fid = int(frame_id)
                prev_keyframe = int(mapping["prev_keyframe"])
                next_keyframe = int(mapping["next_keyframe"])
            except Exception:
                continue
            frame_keyframe_map[fid] = {
                "prev_keyframe": prev_keyframe,
                "next_keyframe": next_keyframe,
            }

    summary_payload = {
        "video_stem": video_stem,
        "packet_dir": str(packet_dir),
        "packet_count": len(summaries),
        "packet_size_bytes_mean": float(mean(packet_sizes)) if packet_sizes else 0.0,
        "packet_size_bytes_p95": float(sorted(packet_sizes)[int(round((len(packet_sizes) - 1) * 0.95))]) if packet_sizes else 0.0,
        "event_count_total": int(sum(event_counts)),
        "frame_count_estimate": int(sum(item["frame_count"] for item in packet_windows)),
        "source_fps": float(source_fps),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "event_support": int(event_support),
        "spatial_grid_size": int(spatial_grid_size),
        "event_render_support": int(event_render_support),
        "packet_windows": packet_windows,
        "frame_keyframe_map": frame_keyframe_map,
        "packets": [
            {
                "packet_name": item["packet_name"],
                "packet_out": item["packet_out"],
                "size_bytes": int(item["size_bytes"]),
                "event_count": int(item.get("event_count", 0)),
                "event_payload_name": item.get("event_payload_name"),
                "meta": item.get("meta", {}),
            }
            for item in summaries
        ],
    }

    summary_path = decompress_ing_dir / "decoder_bundle.local.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    reconstructed_path = decompress_dir / "decoder_reconstructed.mp4"
    vis_path = visualization_dir / "decoder_frame_reconstruction_preview.mp4"
    if not args.disable_visualization:
        fps = max(1.0, float(source_fps))
        reconstructed_writer = cv2.VideoWriter(
            str(reconstructed_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (int(source_width), int(source_height)),
        )
        preview_writer = cv2.VideoWriter(
            str(vis_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (int(source_width), int(source_height)),
        )

        first_ref_id = min(keyframe_by_id.keys(), default=0)
        previous_frame = keyframe_by_id.get(first_ref_id, np.zeros((source_height, source_width, 3), dtype=np.uint8)).copy()
        if previous_frame.shape[1] != source_width or previous_frame.shape[0] != source_height:
            previous_frame = cv2.resize(previous_frame, (source_width, source_height), interpolation=cv2.INTER_AREA)

        for packet_window in sorted(packet_windows, key=lambda item: int(item["start_frame_id"])):
            for frame_id in range(int(packet_window["start_frame_id"]), int(packet_window["end_frame_id"]) + 1):
                frame_events = restore_event_coordinates(
                    event_by_frame.get(frame_id, []),
                    event_support=event_render_support,
                    source_h=source_height,
                    source_w=source_width,
                )

                if frame_id in keyframe_by_id:
                    frame = keyframe_by_id[frame_id].copy()
                    if frame.shape[1] != source_width or frame.shape[0] != source_height:
                        frame = cv2.resize(frame, (source_width, source_height), interpolation=cv2.INTER_AREA)
                else:
                    frame = apply_events_to_luma(previous_frame, frame_events, event_support=event_support)

                reconstructed_writer.write(frame)

                ref_keyframe_id = frame_keyframe_map.get(frame_id, {}).get("prev_keyframe", frame_id)
                ref_keyframe = keyframe_by_id.get(ref_keyframe_id, previous_frame)
                if ref_keyframe.shape[1] != source_width or ref_keyframe.shape[0] != source_height:
                    ref_keyframe = cv2.resize(ref_keyframe, (source_width, source_height), interpolation=cv2.INTER_AREA)
                preview_base = make_event_base_frame(frame_events, (source_height, source_width), event_support=event_render_support)
                preview_frame = add_keyframe_corner_inset(
                    preview_base,
                    ref_keyframe,
                    label=f"keyframe {ref_keyframe_id:04d}",
                )
                preview_writer.write(preview_frame)

                previous_frame = frame

        reconstructed_writer.release()
        preview_writer.release()

    print(f"[decoder] packet_dir={packet_dir}")
    print(f"[decoder] packet_count={summary_payload['packet_count']} event_count_total={summary_payload['event_count_total']}")
    print(f"[decoder] frame_count_estimate={summary_payload['frame_count_estimate']}")
    print(f"[decoder] source_fps={summary_payload['source_fps']:.6f}")
    print(f"[decoder] source_resolution={summary_payload['source_width']}x{summary_payload['source_height']}")
    print(f"[decoder] output_dir={video_output_dir}")
    print(f"[decoder] summary={summary_path}")
    if not args.disable_visualization:
        print(f"[decoder] reconstructed={reconstructed_path}")
        print(f"[decoder] visualization={vis_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())