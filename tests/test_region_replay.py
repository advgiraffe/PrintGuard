"""Offline sampling and aggregation with deterministic decoding and inference."""

from __future__ import annotations

from argparse import Namespace
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from printguard.server import region_replay


def test_sparse_gaps_break_streaks_and_reviewed_frames_are_included(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording = tmp_path / 'session'
    recording.mkdir()
    for index in range(3):
        (recording / f'{index}.mp4').write_bytes(b'recorded-source-placeholder')
    models = tmp_path / 'models'
    models.mkdir()
    (models / 'metadata.json').write_text(json.dumps({'preprocessing': {'normalise_mean': [0, 0, 0], 'normalise_std': [1, 1, 1]}}))
    (models / 'prototypes.json').write_text('{"prototypes":{}}')
    (models / 'encoder_float32.onnx').write_bytes(b'model-placeholder')
    transform = tmp_path / 'transform.json'
    transform.write_text('{"rotation":0,"crop":null,"brightness":1,"contrast":1,"sharpness":0}')
    labels = tmp_path / 'labels.json'
    labels.write_text(json.dumps({'session_id': 'session', 'examples': [{'segment': '0.mp4', 'frame_index_zero_based': 1, 'selected': True, 'proposed_label': 'failure'}]}))

    class Video:
        """Four decoded frames per segment."""

        streams = SimpleNamespace(video=[SimpleNamespace(duration=4, time_base=Fraction(1, 1))])

        def __enter__(self) -> Video:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def decode(self, video: int) -> list[SimpleNamespace]:
            return [SimpleNamespace(time=i, pts=i, time_base=Fraction(1, 1), to_ndarray=lambda format: np.zeros((32, 64, 3), np.uint8)) for i in range(4)]

    class Model:
        """Finite, varying above-threshold scores for aggregation checks."""

        calls = 0

        def disable_fallback(self) -> None:
            pass

        def get_providers(self) -> list[str]:
            return ['CPUExecutionProvider']

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name='input')]

        def run(self, *args: object) -> list[np.ndarray]:
            self.calls += 1
            return [np.array([[.8 + (self.calls % 10) / 100]])]

    monkeypatch.setattr(region_replay.av, 'open', lambda _: Video())
    monkeypatch.setattr(region_replay.ort, 'InferenceSession', lambda *args, **kwargs: Model())
    monkeypatch.setattr(region_replay.vision, 'classify', lambda embedding, assets: float(embedding[0]))
    monkeypatch.setattr(region_replay.vision, 'defect_score', lambda result: result)
    output = tmp_path / 'results'
    result = region_replay.replay(Namespace(recording=recording, models=models, transform=transform, labels=labels, output=output, provider='cpu', stride=100, dense_tail=1, tile_sizes=[32], overlap=.25, threshold=.75))
    assert result['decoded_frames'] == 12
    assert result['sampled_frames'] == 7
    rows = [json.loads(line) for line in (output / 'frames.jsonl').read_text().splitlines()]
    assert [(r['segment'], r['frame_index']) for r in rows] == [('0.mp4', 0), ('0.mp4', 1), ('1.mp4', 0), ('2.mp4', 0), ('2.mp4', 1), ('2.mp4', 2), ('2.mp4', 3)]
    for mode in result['modes'].values():
        assert mode['longest_adjacent_frame_streak'] == 4
        assert mode['frames_above_threshold'] == 7
        assert mode['labelled_counts'] == {'failure_above': 1}
        assert mode['first_crossing']['frame_index'] == 0
    for row in rows:
        for mode in row['modes'].values():
            assert mode['score'] == max(r['score'] for r in mode['regions'])
    assert result['geometry']['tiles_32_64x32']['covered_fraction'] == 1
