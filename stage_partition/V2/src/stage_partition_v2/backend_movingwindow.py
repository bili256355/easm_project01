from __future__ import annotations

from dataclasses import asdict
import inspect
import numpy as np
import pandas as pd

from .config import MovingWindowConfig


def _import_movingwindow():
    try:
        from sktime.detection.skchange_cp import MovingWindow
        return MovingWindow
    except Exception:
        try:
            from skchange.change_detectors import MovingWindow
            return MovingWindow
        except Exception as e:
            raise ImportError(
                'MovingWindow backend requires sktime/skchange in the user environment. '
                'Install the dependencies listed in requirements_stage_partition_v2.txt.'
            ) from e


def _movingwindow_api_mode(MovingWindow) -> str:
    params = inspect.signature(MovingWindow.__init__).parameters
    if 'threshold_scale' in params:
        return 'threshold_api'
    return 'legacy_api'


def movingwindow_api_mode() -> str:
    MovingWindow = _import_movingwindow()
    return _movingwindow_api_mode(MovingWindow)


def _build_detector(cfg: MovingWindowConfig):
    MovingWindow = _import_movingwindow()
    mode = _movingwindow_api_mode(MovingWindow)
    if mode == 'threshold_api':
        detector = MovingWindow(
            bandwidth=cfg.bandwidth,
            threshold_scale=cfg.threshold_scale,
            level=cfg.level,
            min_detection_interval=cfg.min_detection_interval,
        )
        compat = {
            'api_mode': mode,
            'uses_threshold_scale': True,
            'uses_level': True,
            'uses_min_detection_interval': True,
        }
    else:
        detector = MovingWindow(
            bandwidth=cfg.bandwidth,
            selection_method='local_optimum',
        )
        compat = {
            'api_mode': mode,
            'uses_threshold_scale': False,
            'uses_level': False,
            'uses_min_detection_interval': False,
            'compat_note': (
                'Installed MovingWindow backend uses the pre-threshold API. '
                'threshold_scale/level/min_detection_interval are not supported by this backend and were not applied.'
            ),
        }
    return detector, compat


def _map_point_indices_to_days(points: pd.Series, day_index: np.ndarray | None) -> pd.Series:
    if day_index is None:
        return points.astype(int)
    mapped = []
    arr = np.asarray(day_index, dtype=int)
    for p in points.astype(int).tolist():
        if p < 0:
            mapped.append(int(arr[0]))
        elif p >= len(arr):
            mapped.append(int(arr[-1]))
        else:
            mapped.append(int(arr[p]))
    return pd.Series(mapped, name='changepoint', dtype=int)


def _to_points_series(obj) -> pd.Series:
    if isinstance(obj, pd.Series):
        if np.issubdtype(obj.dtype, np.number):
            return obj.astype(int)
        return pd.Series(obj.index.astype(int), dtype=int)
    if isinstance(obj, pd.DataFrame):
        if 'ilocs' in obj.columns:
            ilocs = obj['ilocs']
            vals = []
            for x in ilocs:
                if hasattr(x, 'left'):
                    vals.append(int(x.left))
                else:
                    vals.append(int(x))
            return pd.Series(vals, name='changepoint', dtype=int)
        if obj.shape[1] >= 1:
            return pd.Series(obj.iloc[:, 0].astype(int).to_numpy(), name='changepoint')
    arr = np.asarray(obj).astype(int).ravel()
    return pd.Series(arr, name='changepoint', dtype=int)


def _to_score_series(obj, n_days: int, day_index: np.ndarray | None = None) -> pd.Series:
    if isinstance(obj, pd.Series):
        s = obj.astype(float)
    elif isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            s = obj.iloc[:, 0].astype(float)
        else:
            s = obj.mean(axis=1).astype(float)
    else:
        arr = np.asarray(obj, dtype=float).ravel()
        s = pd.Series(arr, name='score')
    if day_index is not None and len(day_index) == len(s):
        s.index = np.asarray(day_index, dtype=int)
    else:
        s.index = np.arange(len(s))
    return s


def run_movingwindow(state_matrix: np.ndarray, cfg: MovingWindowConfig, day_index: np.ndarray | None = None) -> dict:
    X = pd.DataFrame(state_matrix)
    detector, compat = _build_detector(cfg)
    points_obj = detector.fit_predict(X)
    points_local = _to_points_series(points_obj)
    points = _map_point_indices_to_days(points_local, day_index)
    try:
        scores_obj = detector.predict_scores(X)
    except Exception:
        scores_obj = detector.transform_scores(X)
    scores = _to_score_series(scores_obj, state_matrix.shape[0], day_index=day_index)
    try:
        segments = detector.predict_segments(X)
    except Exception:
        segments = pd.DataFrame(columns=['ilocs', 'labels'])
    return {
        'backend_name': 'movingwindow',
        'detector': detector,
        'points': points,
        'points_local': points_local,
        'scores': scores,
        'segments': segments,
        'config': {**asdict(cfg), **compat},
    }
