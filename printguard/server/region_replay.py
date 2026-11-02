"""Offline comparison of baseline and overlapping camera views on recorded video."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import av
import numpy as np
import onnxruntime as ort

from printguard.engine import regions, vision


def replay(args: argparse.Namespace) -> dict[str, Any]:
    """Scores recorded frames once per baseline or tile, without hub connections.

    Args:
        args: Parsed local file paths, sampling and inference options.

    Returns:
        Summary measurements, per-region geometry and labelled-frame outcomes.
    """
    args.output.mkdir(parents=True, exist_ok=True)
    assets = vision.assets_from_dicts(json.loads((args.models / 'metadata.json').read_text()), json.loads((args.models / 'prototypes.json').read_text())['prototypes'])
    transform = json.loads(args.transform.read_text())
    full_transform = dict(transform, crop=None)
    labels: dict[tuple[str, int], str] = {}
    if args.labels:
        reviewed = json.loads(args.labels.read_text())
        if reviewed['session_id'] != args.recording.name:
            raise ValueError('label session does not match recording directory')
        labels = {(row['segment'], row['frame_index_zero_based']): row['proposed_label'] for row in reviewed['examples'] if row['selected']}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    if args.provider == 'cuda':
        ort.preload_dlls(directory='')
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        providers: list[Any] = [('CUDAExecutionProvider', {'use_tf32': '0'})]
    else:
        providers = ['CPUExecutionProvider']
    model = ort.InferenceSession(str(args.models / 'encoder_float32.onnx'), sess_options=options, providers=providers)
    model.disable_fallback()
    if model.get_providers()[0] != ('CUDAExecutionProvider' if args.provider == 'cuda' else 'CPUExecutionProvider'):
        raise RuntimeError('requested inference provider is unavailable')
    input_name = model.get_inputs()[0].name
    for _ in range(10):
        model.run(None, {input_name: np.zeros((1, 3, 224, 224), dtype=np.float32)})
    timings: dict[str, list[float]] = defaultdict(list)
    values: dict[str, list[float]] = defaultdict(list)
    labelled: dict[str, Counter] = defaultdict(Counter)
    streaks: dict[str, int] = defaultdict(int)
    longest: dict[str, int] = defaultdict(int)
    first_crossing: dict[str, dict[str, Any]] = {}
    geometry: dict[str, Any] = {}
    decoded = sampled = 0
    elapsed = 0.0
    files = sorted(args.recording.glob('*.mp4'))
    if not files:
        raise ValueError('recording directory contains no MP4 segments')
    source_files = []
    started = time.perf_counter()
    previous_decoded: int | None = None
    with (args.output / 'frames.jsonl').open('w') as output:
        for segment_index, path in enumerate(files):
            source_files.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            with av.open(str(path)) as video:
                stream = video.streams.video[0]
                duration = float(stream.duration * stream.time_base)
                first_pts = None
                for index, frame in enumerate(video.decode(video=0)):
                    decoded += 1
                    if first_pts is None:
                        first_pts = float(frame.time)
                    dense = args.dense_tail > 0 and segment_index >= len(files) - args.dense_tail
                    if not dense and index % args.stride and (path.name, index) not in labels:
                        continue
                    sampled += 1
                    rgb = frame.to_ndarray(format='rgb24')
                    transform_started = time.perf_counter()
                    full = vision.transform(rgb, **full_transform)
                    full_transform_ms = (time.perf_counter() - transform_started) * 1000
                    height, width = full.shape[:2]
                    grids = {f'tiles_{size}': regions.plan_regions(width, height, size, args.overlap) for size in args.tile_sizes}
                    for name, grid in grids.items():
                        key = f'{name}_{width}x{height}'
                        if key not in geometry:
                            coverage = np.zeros((height, width), dtype=np.uint16)
                            for region in grid:
                                x0, y0, x1, y1 = region.visible_box
                                coverage[y0:y1, x0:x1] += 1
                            geometry[key] = {'regions': [asdict(r) for r in grid], 'covered_fraction': float(np.mean(coverage > 0)), 'minimum_views_per_pixel': int(coverage.min())}
                    row: dict[str, Any] = {'segment': path.name, 'frame_index': index, 'pts': frame.pts, 'time_base': str(frame.time_base), 'elapsed_video_s': elapsed + float(frame.time) - first_pts, 'label': labels.get((path.name, index)), 'modes': {}}
                    adjacent = previous_decoded is not None and decoded == previous_decoded + 1
                    if not adjacent:
                        streaks.clear()
                    mode_names = ['saved_crop', 'uncropped_center', *grids]
                    for name in mode_names:
                        start = time.perf_counter()
                        if name == 'saved_crop':
                            views = [('saved_crop', vision.transform(rgb, **transform))]
                        elif name == 'uncropped_center':
                            views = [('uncropped_center', full)]
                        else:
                            views = [(r.name, regions.region_image(full, r)) for r in grids[name]]
                        results = []
                        for view_name, image in views:
                            tensor = vision.preprocess(image, assets)
                            embedding = model.run(None, {input_name: tensor})[0][0]
                            results.append({'region': view_name, 'score': vision.defect_score(vision.classify(embedding, assets))})
                        score = max(r['score'] for r in results)
                        milliseconds = (time.perf_counter() - start) * 1000 + (0 if name == 'saved_crop' else full_transform_ms)
                        timings[name].append(milliseconds)
                        values[name].append(score)
                        above = score >= args.threshold
                        streaks[name] = streaks[name] + 1 if above else 0
                        longest[name] = max(longest[name], streaks[name])
                        if above and name not in first_crossing:
                            first_crossing[name] = {'segment': path.name, 'frame_index': index, 'elapsed_video_s': row['elapsed_video_s']}
                        if row['label']:
                            labelled[name][row['label'] + ('_above' if above else '_below')] += 1
                        row['modes'][name] = {'score': score, 'milliseconds': milliseconds, 'regions': results}
                    output.write(json.dumps(row) + '\n')
                    previous_decoded = decoded
            elapsed += duration
            print(json.dumps({'segment_finished': segment_index, 'sampled': sampled}), flush=True)
    summary = {
        'session': args.recording.name, 'provider': model.get_providers()[0],
        'model_sha256': hashlib.sha256((args.models / 'encoder_float32.onnx').read_bytes()).hexdigest(),
        'saved_transform': transform, 'stride': args.stride, 'dense_tail_segments': args.dense_tail,
        'decoded_frames': decoded, 'sampled_frames': sampled, 'video_seconds': elapsed,
        'wall_seconds': time.perf_counter() - started, 'threshold': args.threshold,
        'aggregation': 'Maximum region score per frame. Offline measurement only.',
        'timing_scope': 'Serial transform, view extraction, preprocessing, inference and scoring per mode; shared full-frame transform charged once to each relevant mode. Excludes decode and grid planning; warmed model.',
        'limitations': 'One camera and retained recordings. Labels are owner-selected examples from this print, not a held-out accuracy estimate. First crossings are first observed sampled crossings. Streaks reset on unsampled gaps; no live watchdog or physical action is simulated.',
        'sources': source_files, 'geometry': geometry,
        'modes': {name: {'views_per_frame': 1 if name not in grids else len(grids[name]), 'score_min': min(scores), 'score_max': max(scores), 'score_median': float(np.median(scores)), 'frames_above_threshold': sum(s >= args.threshold for s in scores), 'longest_adjacent_frame_streak': longest[name], 'first_crossing': first_crossing.get(name), 'labelled_counts': dict(labelled[name]), 'milliseconds_median': float(np.median(timings[name])), 'milliseconds_p95': float(np.percentile(timings[name], 95))} for name, scores in values.items()},
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    """Parses offline replay options and writes local comparison artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--transform', type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provider', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--dense-tail', type=int, default=2)
    parser.add_argument('--tile-sizes', type=int, nargs='+', default=[320, 192])
    parser.add_argument('--overlap', type=float, default=.25)
    parser.add_argument('--threshold', type=float, default=.75)
    args = parser.parse_args()
    if args.stride < 1 or args.dense_tail < 0 or not 0 <= args.threshold <= 1:
        parser.error('stride must be positive, dense-tail nonnegative and threshold between zero and one')
    for size in args.tile_sizes:
        regions.plan_regions(640, 360, size, args.overlap)
    replay(args)


if __name__ == '__main__':
    main()
