import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

# === Sliding buffer ===
class TimeSeriesBuffer:
    def __init__(self, window_sec: float):
        self.window_sec = window_sec
        self.t = deque()
        self.v = deque()

    def append(self, t: float, val: float):
        self.t.append(float(t))
        self.v.append(float(val))
        self._trim()

    def _trim(self):
        if not self.t:
            return
        t_latest = self.t[-1]
        while self.t and (t_latest - self.t[0] > self.window_sec):
            self.t.popleft()
            self.v.popleft()

    def as_arrays(self):
        return np.asarray(self.t, dtype=float), np.asarray(self.v, dtype=float)


class StreamSimulator:
    def __init__(self, df, step_sec: float = 1.0, agg: str = "mean"):
        self.t = df["time_sec"].to_numpy(dtype=float)
        self.v = df["value"].to_numpy(dtype=float)
        self.idx = 0
        self.step_sec = step_sec
        self.current_time = self.t[0]
        self.agg = agg

    def step_one_second(self):
        if self.idx >= len(self.t):
            return None, []

        t_start = self.current_time
        t_end = t_start + self.step_sec
        vals = []

        while self.idx < len(self.t) and self.t[self.idx] < t_end:
            vals.append(self.v[self.idx])
            self.idx += 1

        self.current_time = t_end

        agg_val = None
        if vals:
            if self.agg == "median":
                agg_val = float(np.median(vals))
            else:
                agg_val = float(np.mean(vals))

        return (int(t_start), agg_val), [(t_start, v) for v in vals]
    
    def finished(self):
        return self.idx >= len(self.t)

    


def resample_uniform(t, v, fs=4.0, gap_sec=3.0):
    if len(t) < 2:
        return t, v
    t0, t1 = t[0], t[-1]
    tt = np.arange(t0, t1, 1/fs)
    vv = np.interp(tt, t, v)

    gaps = np.diff(t)
    big_gap_idx = np.where(gaps > gap_sec)[0]
    for i in big_gap_idx:
        left, right = t[i], t[i+1]
        mask = (tt > left) & (tt < right)
        vv[mask] = np.nan
    return tt, vv


@dataclass
class Metrics:
    slope_bpm_min: float
    sdnn: float
    rmssd: float
    pnn5: float
    baseline_med: float
    vmin: float
    vmax: float
    missing_ratio: float
    est_fs: float


def compute_metrics_on_window(t, v) -> Optional[Metrics]:
    if len(t) < 8:
        return None
    mask = ~np.isnan(v)
    t = t[mask]; v = v[mask]
    if len(t) < 8:
        return None

    # тренд
    x = t - t[0]
    a, b = np.polyfit(x, v, 1)
    slope_bpm_min = a * 60.0

    # вариабельность
    sdnn = float(np.nanstd(v))
    dv = np.diff(v)
    rmssd = float(np.sqrt(np.nanmean(dv**2))) if len(dv) else np.nan
    pnn5 = float(np.mean(np.abs(dv) > 5.0) * 100) if len(dv) else np.nan

    baseline_med = float(np.nanmedian(v))
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    missing_ratio = float(np.mean(~mask))
    est_fs = float(1.0 / np.median(np.diff(t))) if len(t) > 1 else np.nan

    return Metrics(slope_bpm_min, sdnn, rmssd, pnn5, baseline_med, vmin, vmax, missing_ratio, est_fs)


def check_alerts(m: Metrics):
    alerts = []
    if m.baseline_med < 110:
        alerts.append("⚠️ Брадикардия (<110 bpm)")
    if m.baseline_med > 160:
        alerts.append("⚠️ Тахикардия (>160 bpm)")
    if m.sdnn < 3:
        alerts.append("⚠️ Снижение вариабельности (SDNN < 3)")
    return alerts
