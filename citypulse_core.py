"""CityPulse core: the scoring maths shared by the training pipeline and the dashboard.

Nothing here needs TensorFlow, so the Streamlit app can import it and recompute
statistical + Isolation Forest evidence live (what-if simulator, citizen reports,
feed outages) with exactly the same formulas the pipeline used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["rainfall", "traffic_congestion", "incidents_30m", "aqi"]
NAMES = {"rainfall": "Rainfall", "traffic_congestion": "Traffic congestion",
         "incidents_30m": "Road incidents", "aqi": "Air quality (AQI)"}
UNITS = {"rainfall": "mm/h", "traffic_congestion": "%", "incidents_30m": "in 30 min", "aqi": "index"}
FMT = {"rainfall": "{:.1f}", "traffic_congestion": "{:.0f}", "incidents_30m": "{:.0f}", "aqi": "{:.0f}"}
# Minimum robust sigma per feed (same floors as the original Core Engine).
FLOOR = {"rainfall": 0.5, "traffic_congestion": 4.0, "incidents_30m": 1.5, "aqi": 6.0}
# Rain on its own is weather, not a civic problem: its statistical evidence is damped.
STAT_FEED_WEIGHT = {"rainfall": 0.6, "traffic_congestion": 1.0, "incidents_30m": 1.0, "aqi": 1.0}
Z_CLIP = (-3.0, 8.0)
DELTA_STEPS = 3            # 15-minute change (5-minute data)
ISO_COLS = [f"z_{f}" for f in FEATURES] + [f"dz_{f}" for f in FEATURES]
CHANNELS = ["stat", "iso", "lstm"]
MERGE_KM = 9.0            # zone episodes closer than this (and overlapping in time) form one event
STRONG = 0.5               # a channel counts as "strong" at >= 0.5 on its calibrated 0-1 scale


# ----------------------------------------------------------------------------- baseline
def fit_reference(train: pd.DataFrame) -> dict:
    """Per zone x hour median and robust sigma (1.4826 * MAD, floored), from training rows only."""
    refs = {}
    for f in FEATURES:
        g = train.groupby(["zone_id", "hour"])[f]
        med = g.median()
        mad = g.apply(lambda s: np.median(np.abs(s - np.median(s))))
        sig = np.maximum(1.4826 * mad, FLOOR[f])
        refs[f] = {
            "table": [{"zone_id": z, "hour": int(h), "median": float(m), "sigma": float(s)}
                      for (z, h), m, s in zip(med.index, med.values, sig.values)],
            "global_median": float(train[f].median()),
            "global_sigma": float(max(1.4826 * np.median(np.abs(train[f] - train[f].median())), FLOOR[f])),
        }
    return refs


def apply_reference(frame: pd.DataFrame, refs: dict) -> pd.DataFrame:
    out = frame.copy()
    for f in FEATURES:
        t = pd.DataFrame(refs[f]["table"]).set_index(["zone_id", "hour"])
        key = pd.MultiIndex.from_arrays([out["zone_id"], out["hour"]])
        med = t["median"].reindex(key).to_numpy()
        sig = t["sigma"].reindex(key).to_numpy()
        med = np.where(np.isfinite(med), med, refs[f]["global_median"])
        sig = np.where(np.isfinite(sig), sig, refs[f]["global_sigma"])
        out[f"{f}_typical"] = med
        out[f"{f}_sigma"] = sig
        out[f"z_{f}"] = (out[f].to_numpy() - med) / sig
    return out


def add_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """15-minute change of each z-score, within zone (sudden jumps are informative)."""
    out = frame.sort_values(["zone_id", "timestamp"]).copy()
    for f in FEATURES:
        out[f"dz_{f}"] = out.groupby("zone_id")[f"z_{f}"].diff(DELTA_STEPS).fillna(0.0)
    return out


def model_matrix(frame: pd.DataFrame, drop: str | None = None) -> np.ndarray:
    """Clipped z + delta features. `drop` imputes one feed as 'typical' (z = 0) to model an outage."""
    X = frame[ISO_COLS].to_numpy(dtype=float).copy()
    X = np.clip(X, Z_CLIP[0], Z_CLIP[1])
    if drop is not None:
        i = FEATURES.index(drop)
        X[:, i] = 0.0
        X[:, len(FEATURES) + i] = 0.0
    return X


# ----------------------------------------------------------------------------- channel scores
def stat_score(frame: pd.DataFrame, drop: str | None = None) -> np.ndarray:
    """Noisy-OR of per-feed evidence. One very unusual feed can score high; several moderate
    feeds compound. z = 1 -> no evidence, z >= 5 -> full evidence for that feed."""
    keep = np.ones(len(frame))
    for f in FEATURES:
        if f == drop:
            continue
        s = np.clip((frame[f"z_{f}"].to_numpy(dtype=float) - 1.0) / 4.0, 0, 1) * STAT_FEED_WEIGHT[f]
        keep *= (1 - s)
    return 1 - keep


def calibrate(raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map a raw anomaly score to 0-1 using NORMAL validation percentiles.
    lo (median of normal) -> 0, hi (99.5th pct of normal) -> 0.63, 2*hi -> 0.86. Never hard-saturates."""
    x = np.clip((np.asarray(raw, dtype=float) - lo) / max(hi - lo, 1e-9), 0, None)
    return 1 - np.exp(-x)


def iso_raw(model, frame: pd.DataFrame, drop: str | None = None) -> np.ndarray:
    return -model.decision_function(model_matrix(frame, drop))


def hybrid(stat, iso, lstm, w: dict) -> np.ndarray:
    return 100 * (w["stat"] * np.asarray(stat) + w["iso"] * np.asarray(iso) + w["lstm"] * np.asarray(lstm))


def strong_count(stat, iso, lstm) -> np.ndarray:
    return ((np.asarray(stat) >= STRONG).astype(int) + (np.asarray(iso) >= STRONG).astype(int)
            + (np.asarray(lstm) >= STRONG).astype(int))


# ----------------------------------------------------------------------------- decision layer
def run_detector(frame: pd.DataFrame, thr: dict, prefix: str = "") -> pd.DataFrame:
    """Temporal decision layer.

    Watch     : hybrid >= watch.
    Confirmed : hybrid >= confirmed AND >= 2 of 3 evidence channels strong AND
                (2 consecutive watch readings OR hybrid >= critical).
    Episode   : starts at a confirmed anchor, backfills watch readings up to 15 min before it,
                and stays open while the zone remains at watch level (hysteresis).
    Critical  : flagged reading with hybrid >= critical.
    """
    h = frame[f"{prefix}hybrid"].to_numpy()
    agree = frame[f"{prefix}strong"].to_numpy()
    out = pd.DataFrame(index=frame.index)
    zone = frame["zone_id"].to_numpy()
    ts = frame["timestamp"].to_numpy()
    n = len(frame)
    watch = h >= thr["watch"]
    run = np.zeros(n, dtype=int)
    flag = np.zeros(n, dtype=int)
    anchor = np.zeros(n, dtype=int)
    order = np.lexsort((ts, zone))
    gap = np.timedelta64(10, "m")
    back = np.timedelta64(15, "m")
    prev_i = None
    for i in order:
        cont = prev_i is not None and zone[prev_i] == zone[i] and (ts[i] - ts[prev_i]) <= gap
        run[i] = (run[prev_i] + 1 if (cont and watch[prev_i]) else 1) if watch[i] else 0
        prev_i = i
    anchor = ((h >= thr["confirmed"]) & (agree >= 2) & ((run >= 2) | (h >= thr["critical"]))).astype(int)
    # forward pass with hysteresis + backfill
    prev_i = None
    zone_rows: list[int] = []
    for i in order:
        if prev_i is None or zone[prev_i] != zone[i]:
            zone_rows = []
        cont = prev_i is not None and zone[prev_i] == zone[i] and (ts[i] - ts[prev_i]) <= gap
        if anchor[i]:
            flag[i] = 1
            for j in reversed(zone_rows):
                if ts[i] - ts[j] > back or not watch[j]:
                    break
                flag[j] = 1
        elif cont and flag[prev_i] and watch[i]:
            flag[i] = 1
        zone_rows.append(i)
        prev_i = i
    out[f"{prefix}warning_run"] = run
    out[f"{prefix}confirmed_anchor"] = anchor
    out[f"{prefix}final_flag"] = flag
    out[f"{prefix}state"] = np.select(
        [(flag == 1) & (h >= thr["critical"]), flag == 1, watch],
        ["Critical", "Confirmed", "Watch"], default="Normal")
    return out


def pulse_from_hybrid(h, thr: dict) -> np.ndarray:
    """Resident-facing 0-100 pulse anchored to the calibrated thresholds:
    100-70 calm (below watch), 70-50 watch, 50-25 confirmed, 25-0 critical."""
    h = np.asarray(h, dtype=float)
    xp = [0, thr["watch"], thr["confirmed"], thr["critical"], 100]
    return np.interp(h, xp, [100, 70, 50, 25, 0]).round(0)


def episodes(frame: pd.DataFrame, flag_col: str = "final_flag", gap_min: int = 10) -> list[dict]:
    """Contiguous flagged runs per zone."""
    f = frame[frame[flag_col] == 1].sort_values(["zone_id", "timestamp"])
    eps = []
    for z, g in f.groupby("zone_id", sort=False):
        t = g["timestamp"].to_list()
        s = p = t[0]
        for x in t[1:]:
            if (x - p) > pd.Timedelta(minutes=gap_min):
                eps.append({"zone_id": z, "start": s, "end": p})
                s = x
            p = x
        eps.append({"zone_id": z, "start": s, "end": p})
    return eps


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def describe_pattern(zs: dict) -> str:
    """Plain-language name for an anomaly from which feeds are elevated (data-driven, not from labels)."""
    hot = [f for f, z in sorted(zs.items(), key=lambda kv: -kv[1]) if z >= 2.0]
    s = set(hot)
    if {"rainfall", "traffic_congestion"} <= s or ({"rainfall", "incidents_30m"} <= s):
        return "Heavy rain with traffic disruption"
    if "aqi" in s and len(s) == 1:
        return "Air-quality spike"
    if "aqi" in s:
        return "Air-quality spike with road disruption"
    if {"traffic_congestion", "incidents_30m"} <= s:
        return "Traffic and incident cluster"
    if "traffic_congestion" in s:
        return "Unusual congestion"
    if "incidents_30m" in s:
        return "Cluster of road incidents"
    if "rainfall" in s:
        return "Heavy rain"
    return "Unusual combination of readings"
