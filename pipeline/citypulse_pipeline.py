"""CityPulse v3 pipeline: city_data.csv -> calibrated hybrid detector -> dashboard data contract.

    python pipeline/citypulse_pipeline.py --input city_data.csv --out data

What changed vs v2.4/2.5
  * Isolation Forest and LSTM both learn on *deviation-from-typical* features (hour-aware z-scores
    and their 15-min change) instead of raw values, so they spend capacity on anomalies, not on the
    daily cycle. Both are fitted on clean training rows only.
  * LSTM error is taken on the latest 3 steps (no dilution over the 12-step window), per feed.
  * Both model scores are calibrated on NORMAL validation rows only (events excluded), with a
    non-saturating mapping. v2.4 calibrated on all validation rows, so the injected event itself
    set the ceiling and squashed the LSTM to ~0.
  * Fusion weights and all three thresholds are chosen on the validation split, then frozen.
    The test split is scored exactly once. (v2.5 lowered thresholds after looking at test.)
  * 8 labelled scenarios + 4 decoys across validation and test, so metrics are not n = 1.
  * Every row is also scored with each feed knocked out, so the dashboard's "feed outage" switch
    shows real re-scored evidence, not a cosmetic change.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import citypulse_core as cc  # noqa: E402

SEED = 42
SEQ_LEN = 12
LSTM_TAIL = 3
ISO_PARAMS = dict(n_estimators=300, max_samples=512, random_state=SEED, n_jobs=-1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================ 1. load
ALIASES = {
    "timestamp": ["timestamp", "datetime", "date_time", "time"],
    "zone_id": ["zone_id", "zone", "zoneid"], "zone_name": ["zone_name", "zonename", "area_name"],
    "lat": ["lat", "latitude"], "lon": ["lon", "lng", "longitude"],
    "rainfall": ["rainfall", "rain", "precipitation"], "temperature": ["temperature", "temp"],
    "aqi": ["aqi", "us_aqi"], "traffic_congestion": ["traffic_congestion", "traffic", "congestion"],
    "incident_count": ["incident_count", "incidents"], "incidents_30m": ["incidents_30m"],
    "ground_truth_event": ["ground_truth_event", "is_event"], "event_label": ["event_label", "label"],
}


def load(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in raw.columns]
    ren = {}
    for canon, opts in ALIASES.items():
        for o in opts:
            if o in raw.columns and canon not in ren.values():
                ren[o] = canon
                break
    df = raw.rename(columns=ren)
    need = ["timestamp", "zone_id", "zone_name", "lat", "lon", "rainfall", "temperature", "aqi",
            "traffic_congestion", "incident_count"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"city_data.csv is missing columns: {miss}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    df["ground_truth_event"] = df.get("ground_truth_event", 0)
    df["ground_truth_event"] = df["ground_truth_event"].fillna(0).astype(int)
    df["event_label"] = df.get("event_label", "").fillna("").astype(str)
    df["incident_count"] = df["incident_count"].astype(float)
    df["decoy"] = ""
    return df


# ============================================================================ 2. scenarios
# Original two events are kept (A = 22 Sep flood in MN+MS, B = 23 Sep SC incident/AQI cluster).
# New ones use smooth onset/offset envelopes so "time to detect" is meaningful.
SCENARIOS = [
    # (id, day, hh:mm, minutes, zones, effects, label, description)
    ("S1", 4, "11:00", 45, ["CS"], {"traffic_congestion": 30, "incident_count": 1.2},
     "traffic_incident_cluster", "Road accident cluster blocking the main corridor"),
    ("S2", 5, "07:00", 100, ["VN"], {"aqi": 75, "traffic_congestion": 8},
     "smog_buildup", "Gradual smog build-up (e.g. waste burning upwind)"),
    ("S3", 6, "09:00", 60, ["VN", "MS"], {"rainfall": 18, "traffic_congestion": 28, "incident_count": 0.8},
     "waterlogging", "Waterlogging after a cloudburst on the western side"),
    ("S4", 6, "16:00", 110, ["CS"], {"aqi": 85, "incident_count": 0.4},
     "smog_buildup", "Evening air-quality spike in the central business area"),
    ("S5", 6, "22:00", 50, ["MN"], {"traffic_congestion": 45, "incident_count": 0.9},
     "signal_failure_gridlock", "Night-time gridlock after a traffic-signal failure"),
]
DECOYS = [  # plausible busy moments that should NOT become events
    ("D1", 5, "02:00", 40, ["MN", "MS", "VN", "CS", "SC"], {"rainfall": 5}, "light citywide shower at night"),
    ("D2", 4, "14:00", 20, ["SC"], {"traffic_congestion": 11}, "brief traffic bump near the bus stand"),
    ("D3", 6, "13:00", 25, ["MN"], {"aqi": 14}, "small afternoon AQI rise"),
    ("D4", 6, "03:30", 30, ["VN", "MS"], {"rainfall": 4}, "drizzle before dawn"),
]


def envelope(n: int) -> np.ndarray:
    """Ramp up over the first 35%, plateau, ease off over the last 20%."""
    x = np.linspace(0, 1, n)
    up = np.clip(x / 0.35, 0, 1)
    down = np.clip((1 - x) / 0.20, 0, 1)
    return np.minimum(up, down) ** 1.2


def inject(df: pd.DataFrame, rng) -> pd.DataFrame:
    day0 = df.timestamp.min().normalize()
    tmax = df.timestamp.max()
    catalogue = []

    def window(day, hhmm, minutes):
        h, m = map(int, hhmm.split(":"))
        s = day0 + pd.Timedelta(days=day, hours=h, minutes=m)
        return s, s + pd.Timedelta(minutes=minutes)

    for sid, day, hhmm, mins, zones, eff, label, desc in SCENARIOS:
        s, e = window(day, hhmm, mins)
        if e > tmax:
            continue
        for z in zones:
            m = (df.zone_id == z) & df.timestamp.between(s, e)
            n = int(m.sum())
            if n == 0:
                continue
            env = envelope(n) * rng.uniform(0.85, 1.15, n)
            for f, amp in eff.items():
                if f == "incident_count":
                    df.loc[m, f] += rng.poisson(amp * env * 1.6 + 0.2)
                else:
                    df.loc[m, f] += amp * env
            df.loc[m, "ground_truth_event"] = 1
            df.loc[m, "event_label"] = f"{sid}_{label}"
        catalogue.append(dict(id=sid, kind="event", label=label, description=desc, zones=zones,
                              start=str(s), end=str(e)))
    for sid, day, hhmm, mins, zones, eff, desc in DECOYS:
        s, e = window(day, hhmm, mins)
        if e > tmax:
            continue
        for z in zones:
            m = (df.zone_id == z) & df.timestamp.between(s, e) & (df.ground_truth_event == 0)
            n = int(m.sum())
            env = envelope(max(n, 1))[:n]
            for f, amp in eff.items():
                df.loc[m, f] += amp * env
            df.loc[m, "decoy"] = sid
        catalogue.append(dict(id=sid, kind="decoy", label="decoy", description=desc, zones=zones,
                              start=str(s), end=str(e)))
    df["traffic_congestion"] = df["traffic_congestion"].clip(0, 100)
    df["rainfall"] = df["rainfall"].clip(lower=0)
    df["incidents_30m"] = df.groupby("zone_id")["incident_count"].transform(
        lambda s: s.rolling(6, min_periods=1).sum())
    # Original labels -> readable ids
    df["event_label"] = df["event_label"].replace({"A_flood_disruption": "A_flood_disruption",
                                                   "B_incident_cluster_aqi": "B_incident_cluster_aqi"})
    for lab, desc, kind in [("A_flood_disruption", "Localised flood disruption (original demo event A)", "event"),
                            ("B_incident_cluster_aqi", "Incident cluster with AQI spike (original demo event B)",
                             "event")]:
        m = df.event_label == lab
        if m.any():
            catalogue.append(dict(id=lab.split("_")[0], kind=kind, label=lab, description=desc,
                                  zones=sorted(df.loc[m, "zone_id"].unique().tolist()),
                                  start=str(df.loc[m, "timestamp"].min()), end=str(df.loc[m, "timestamp"].max())))
    return df, sorted(catalogue, key=lambda c: c["start"])


# ============================================================================ 3. LSTM
def build_sequences(frame: pd.DataFrame, X: np.ndarray):
    """Zone-wise sliding windows ending at each row. Returns windows and the row index they end at."""
    seqs, ends = [], []
    pos = pd.Series(np.arange(len(frame)), index=frame.index)
    for _, g in frame.groupby("zone_id", sort=False):
        g = g.sort_values("timestamp")
        idx = pos.loc[g.index].to_numpy()
        vals = X[idx]
        # pad the first SEQ_LEN-1 steps by repeating the first reading so every row gets a score
        pad = np.repeat(vals[:1], SEQ_LEN - 1, axis=0)
        v = np.vstack([pad, vals])
        w = np.lib.stride_tricks.sliding_window_view(v, (SEQ_LEN, v.shape[1]))[:, 0]
        seqs.append(w)
        ends.append(idx)
    return np.concatenate(seqs).astype(np.float32), np.concatenate(ends)


def train_lstm(Xtr: np.ndarray):
    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    n_f = Xtr.shape[2]
    inp = keras.Input(shape=(SEQ_LEN, n_f))
    x = keras.layers.LSTM(48, return_sequences=True)(inp)
    x = keras.layers.LSTM(16)(x)                      # bottleneck
    x = keras.layers.RepeatVector(SEQ_LEN)(x)
    x = keras.layers.LSTM(16, return_sequences=True)(x)
    x = keras.layers.LSTM(48, return_sequences=True)(x)
    out = keras.layers.TimeDistributed(keras.layers.Dense(n_f))(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(2e-3), loss="mse")
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    hist = model.fit(Xtr, Xtr, validation_split=0.15, epochs=60, batch_size=128, shuffle=True,
                     callbacks=cb, verbose=0)
    return model, hist.history


def lstm_errors(model, seqs: np.ndarray) -> np.ndarray:
    """Per-feed squared error on the most recent LSTM_TAIL steps -> shape (n, n_features)."""
    rec = model.predict(seqs, verbose=0, batch_size=1024)
    return np.square(seqs - rec)[:, -LSTM_TAIL:, :].mean(axis=1)


LSTM_SCALE = 3.0  # z-scores are divided by 3 before entering the LSTM


def lstm_inputs(frame, drop=None):
    Xm = cc.model_matrix(frame, drop)[:, :len(cc.FEATURES)] / LSTM_SCALE
    return Xm


def lstm_row_error(per_feed: np.ndarray) -> np.ndarray:
    return per_feed.sum(axis=1)


# ============================================================================ 4. evaluation helpers
def truth_episodes(frame):
    t = frame[frame.ground_truth_event == 1]
    out = []
    for lab, g in t.groupby("event_label"):
        out.append(dict(label=lab, start=g.timestamp.min(), end=g.timestamp.max(),
                        zones=sorted(g.zone_id.unique())))
    return sorted(out, key=lambda d: d["start"])


def event_metrics(frame, det, days: float):
    """Event-level matching: a truth event is detected if any flagged episode in one of its zones
    overlaps [start, end + 20 min]. Unmatched flagged episodes are false alarms."""
    f = frame[["timestamp", "zone_id", "ground_truth_event", "event_label", "decoy"]].join(det)
    eps = cc.episodes(f)
    truths = truth_episodes(f)
    matched_eps = set()
    rows = []
    for tr in truths:
        hit = [k for k, e in enumerate(eps) if e["zone_id"] in tr["zones"]
               and e["start"] <= tr["end"] + pd.Timedelta(minutes=20) and e["end"] >= tr["start"]]
        matched_eps.update(hit)
        conf = f[(f.zone_id.isin(tr["zones"])) & (f.confirmed_anchor == 1)
                 & f.timestamp.between(tr["start"], tr["end"] + pd.Timedelta(minutes=20))]
        watch = f[(f.zone_id.isin(tr["zones"])) & (f.state != "Normal")
                  & f.timestamp.between(tr["start"] - pd.Timedelta(minutes=10), tr["end"])]
        lag = (conf.timestamp.min() - tr["start"]).total_seconds() / 60 if len(conf) else None
        wlag = (watch.timestamp.min() - tr["start"]).total_seconds() / 60 if len(watch) else None
        peak = f[(f.zone_id.isin(tr["zones"])) & f.timestamp.between(tr["start"], tr["end"])]
        rows.append(dict(label=tr["label"], start=str(tr["start"]), end=str(tr["end"]), zones=tr["zones"],
                         detected=bool(hit), minutes_to_confirm=lag, minutes_to_first_warning=wlag,
                         peak_state=(peak.state.map({"Normal": 0, "Watch": 1, "Confirmed": 2, "Critical": 3})
                                     .max() if len(peak) else 0)))
    fa = [e for k, e in enumerate(eps) if k not in matched_eps]
    decoy_rows = f[f.decoy != ""]
    decoys_fired = sorted(decoy_rows.loc[decoy_rows.final_flag == 1, "decoy"].unique().tolist())
    tp = sum(r["detected"] for r in rows)
    lags = [r["minutes_to_confirm"] for r in rows if r["minutes_to_confirm"] is not None]
    ev_prec = (len(eps) - len(fa)) / len(eps) if eps else 0.0
    ev_rec = tp / len(rows) if rows else 0.0
    return dict(true_events=len(rows), detected=tp, recall=ev_rec, precision=ev_prec,
                f1=(2 * ev_prec * ev_rec / (ev_prec + ev_rec)) if ev_prec + ev_rec else 0.0,
                false_alarm_episodes=len(fa), false_alarms_per_day=len(fa) / max(days, 1e-9),
                mean_minutes_to_confirm=float(np.mean(lags)) if lags else None,
                decoys_total=int(decoy_rows.decoy.nunique()), decoys_fired=decoys_fired,
                per_event=rows,
                false_alarm_list=[dict(zone_id=e["zone_id"], start=str(e["start"]), end=str(e["end"])) for e in fa])


def row_metrics(y, flag, scores: dict):
    m = dict(precision=float(precision_score(y, flag, zero_division=0)),
             recall=float(recall_score(y, flag, zero_division=0)),
             f1=float(f1_score(y, flag, zero_division=0)), positives=int(y.sum()), rows=int(len(y)))
    if y.nunique() > 1:
        for k, s in scores.items():
            m[f"roc_auc_{k}"] = float(roc_auc_score(y, s))
            m[f"pr_auc_{k}"] = float(average_precision_score(y, s))
    return m


# ============================================================================ main
def main(inp: str, out_dir: str, source: str):
    rng = np.random.default_rng(SEED)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log("loading", inp)
    df = load(inp)
    df, catalogue = inject(df, rng)
    df["hour"] = df.timestamp.dt.hour
    log(f"{len(df):,} rows, {df.zone_id.nunique()} zones, {df.timestamp.min()} -> {df.timestamp.max()}")

    # ---- chronological split (60 / 20 / 20 of unique timestamps)
    ts = np.sort(df.timestamp.unique())
    c1, c2 = ts[int(len(ts) * 0.60)], ts[int(len(ts) * 0.80)]
    df["split"] = np.where(df.timestamp < c1, "train", np.where(df.timestamp < c2, "val", "test"))
    for s in ["train", "val", "test"]:
        d = df[df.split == s]
        log(f"  {s:5s} {d.timestamp.min()} -> {d.timestamp.max()}  events={d[d.ground_truth_event == 1].event_label.nunique()}")
    tr_clean = df[(df.split == "train") & (df.ground_truth_event == 0)]

    # ---- robust baseline (train only) + deltas
    refs = cc.fit_reference(tr_clean)
    df = cc.add_deltas(cc.apply_reference(df, refs)).sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    is_tr = (df.split == "train").to_numpy()
    is_val = (df.split == "val").to_numpy()
    normal = (df.ground_truth_event == 0).to_numpy()

    # ---- statistical channel
    df["stat_score"] = cc.stat_score(df)

    # ---- Isolation Forest (train, clean rows)
    log("fitting Isolation Forest")
    iso = IsolationForest(contamination="auto", **ISO_PARAMS)
    iso.fit(cc.model_matrix(df[is_tr & normal]))
    df["iso_raw"] = cc.iso_raw(iso, df)
    iso_lo = float(np.percentile(df.loc[is_val & normal, "iso_raw"], 50))
    iso_hi = float(np.percentile(df.loc[is_val & normal, "iso_raw"], 99.5))
    df["iso_score"] = cc.calibrate(df["iso_raw"], iso_lo, iso_hi)

    # ---- LSTM autoencoder (train windows that contain no event)
    log("building LSTM sequences")
    X_all = lstm_inputs(df)
    seqs, ends = build_sequences(df, X_all)
    ev = df.ground_truth_event.to_numpy()
    win_has_event = np.zeros(len(ends), dtype=bool)
    # a window is clean if none of its SEQ_LEN rows is an event (rows are zone-sorted & contiguous)
    ev_by_end = pd.Series(ev).rolling(SEQ_LEN, min_periods=1).max().to_numpy()
    win_has_event = ev_by_end[ends] > 0
    train_win = is_tr[ends] & ~win_has_event
    log(f"training LSTM autoencoder on {train_win.sum():,} clean windows")
    model, hist = train_lstm(seqs[train_win])
    per_feed = np.zeros((len(df), len(cc.FEATURES)))
    per_feed[ends] = lstm_errors(model, seqs)
    df["lstm_error"] = lstm_row_error(per_feed)
    for k, f in enumerate(cc.FEATURES):
        df[f"lstm_err_{f}"] = per_feed[:, k]
    lstm_lo = float(np.percentile(df.loc[is_val & normal, "lstm_error"], 50))
    lstm_hi = float(np.percentile(df.loc[is_val & normal, "lstm_error"], 99.5))
    df["lstm_score"] = cc.calibrate(df["lstm_error"], lstm_lo, lstm_hi)
    log(f"LSTM epochs={len(hist['loss'])} final val_loss={hist['val_loss'][-1]:.4f}")

    # ---- fusion weights (validation PR-AUC)
    yv = df.loc[is_val, "ground_truth_event"]
    S = df.loc[is_val, ["stat_score", "iso_score", "lstm_score"]].to_numpy()
    best = None
    grid = np.round(np.arange(0.1, 0.81, 0.05), 2)
    for ws, wi in product(grid, grid):
        wl = round(1 - ws - wi, 2)
        if wl < 0.1:
            continue
        ap = average_precision_score(yv, S @ np.array([ws, wi, wl]))
        bal = -np.std([ws, wi, wl])  # tie-break toward balanced weights
        if best is None or (ap, bal) > (best[0], best[1]):
            best = (ap, bal, dict(stat=float(ws), iso=float(wi), lstm=float(wl)))
    W = best[2]
    log(f"fusion weights {W}  (val PR-AUC {best[0]:.3f})")

    def score_block(prefix="", drop=None):
        if drop is None:
            st_, is_, ls_ = df["stat_score"], df["iso_score"], df["lstm_score"]
        else:
            st_ = cc.stat_score(df, drop)
            is_ = cc.calibrate(cc.iso_raw(iso, df, drop), iso_lo, iso_hi)
            pf = np.zeros((len(df), len(cc.FEATURES)))
            s2, e2 = build_sequences(df, lstm_inputs(df, drop))
            pf[e2] = lstm_errors(model, s2)
            ls_ = cc.calibrate(lstm_row_error(pf), lstm_lo, lstm_hi)
        df[f"{prefix}hybrid"] = cc.hybrid(st_, is_, ls_, W)
        df[f"{prefix}strong"] = cc.strong_count(st_, is_, ls_)
        return st_, is_, ls_

    score_block()

    # ---- thresholds (validation only)
    val = df[is_val].copy()
    val_days = (val.timestamp.max() - val.timestamp.min()).total_seconds() / 86400
    watch = float(np.percentile(val.loc[val.ground_truth_event == 0, "hybrid"], 99.0))
    cands = []
    for conf in np.arange(np.ceil(watch) + 2, 95, 1.0):
        for crit_gap in (8, 12, 16, 20):
            thr = dict(watch=watch, confirmed=float(conf), critical=float(min(conf + crit_gap, 98)))
            det = cc.run_detector(val, thr)
            em = event_metrics(val, det, val_days)
            rm = f1_score(val.ground_truth_event, det.final_flag, zero_division=0)
            lag = em["mean_minutes_to_confirm"] if em["mean_minutes_to_confirm"] is not None else 999
            cands.append(((em["f1"], -em["false_alarms_per_day"], -lag, rm, -crit_gap), thr, em))
    cands.sort(key=lambda c: c[0], reverse=True)
    THR = cands[0][1]
    # critical = upper half of validation event evidence, but never below confirmed + 8
    ev_h = val.loc[val.ground_truth_event == 1, "hybrid"]
    THR["critical"] = float(np.clip(np.percentile(ev_h, 55), THR["confirmed"] + 8, 98)) if len(ev_h) else THR["critical"]
    THR = {k: round(v, 1) for k, v in THR.items()}
    log(f"thresholds {THR}")

    # ---- detector on every row; outage variants
    det = cc.run_detector(df, THR)
    df = df.join(det)
    for f in cc.FEATURES:
        score_block(prefix=f"wo_{f}__", drop=f)
        d2 = cc.run_detector(df, THR, prefix=f"wo_{f}__")
        df[f"wo_{f}__state"] = d2[f"wo_{f}__state"]
    df["pulse"] = cc.pulse_from_hybrid(df["hybrid"], THR)

    # ---- evaluation
    metrics = {}
    for s in ["val", "test"]:
        d = df[df.split == s]
        days = (d.timestamp.max() - d.timestamp.min()).total_seconds() / 86400
        dcols = ["warning_run", "confirmed_anchor", "final_flag", "state"]
        metrics[s] = dict(
            rows=row_metrics(d.ground_truth_event, d.final_flag,
                             dict(hybrid=d.hybrid, stat=d.stat_score, iso=d.iso_score, lstm=d.lstm_score)),
            events=event_metrics(d, d[dcols], days), days=round(days, 2))
        # ablation: same decision layer, one channel at a time (weights renormalised)
        abl = {}
        for ch, col in [("stat", "stat_score"), ("iso", "iso_score"), ("lstm", "lstm_score")]:
            tmp = d[["timestamp", "zone_id", "ground_truth_event", "event_label", "decoy"]].copy()
            tmp["hybrid"] = 100 * d[col]
            tmp["strong"] = 3 * (d[col] >= cc.STRONG)
            dd = cc.run_detector(tmp, THR)
            em = event_metrics(tmp, dd, days)
            abl[ch] = dict(event_recall=em["recall"], false_alarms_per_day=em["false_alarms_per_day"],
                           pr_auc=metrics[s]["rows"].get(f"pr_auc_{ch}"))
        abl["hybrid"] = dict(event_recall=metrics[s]["events"]["recall"],
                             false_alarms_per_day=metrics[s]["events"]["false_alarms_per_day"],
                             pr_auc=metrics[s]["rows"].get("pr_auc_hybrid"))
        metrics[s]["ablation"] = abl
        e = metrics[s]["events"]
        log(f"[{s}] events {e['detected']}/{e['true_events']}  FA/day {e['false_alarms_per_day']:.2f}  "
            f"mean lag {e['mean_minutes_to_confirm']}  row F1 {metrics[s]['rows']['f1']:.3f}  "
            f"PR-AUC hybrid {metrics[s]['rows'].get('pr_auc_hybrid', float('nan')):.3f} "
            f"stat {metrics[s]['rows'].get('pr_auc_stat', float('nan')):.3f} "
            f"iso {metrics[s]['rows'].get('pr_auc_iso', float('nan')):.3f} "
            f"lstm {metrics[s]['rows'].get('pr_auc_lstm', float('nan')):.3f}  decoys fired {e['decoys_fired']}")

    # ---- city-level events (merge zone episodes that overlap in time and sit within MERGE_KM)
    zones = df.groupby("zone_id")[["zone_name", "lat", "lon"]].first()
    eps = cc.episodes(df)
    eps.sort(key=lambda e: e["start"])
    groups: list[list[dict]] = []
    for e in eps:
        placed = False
        for g in groups:
            if any(e["start"] <= x["end"] + pd.Timedelta(minutes=10) and e["end"] >= x["start"] - pd.Timedelta(minutes=10)
                   and cc.haversine_km(zones.at[e["zone_id"], "lat"], zones.at[e["zone_id"], "lon"],
                                       zones.at[x["zone_id"], "lat"], zones.at[x["zone_id"], "lon"]) <= cc.MERGE_KM for x in g):
                g.append(e)
                placed = True
                break
        if not placed:
            groups.append([e])
    events = []
    for g in groups:
        zs = sorted({e["zone_id"] for e in g})
        s, e_ = min(x["start"] for x in g), max(x["end"] for x in g)
        rows = df[df.zone_id.isin(zs) & df.timestamp.between(s, e_)]
        flagged = rows[rows.final_flag == 1]
        pk = flagged.loc[flagged.hybrid.idxmax()]
        zsc = {f: float(pk[f"z_{f}"]) for f in cc.FEATURES}
        dur = (e_ - s).total_seconds() / 60 + 5
        comp = dict(magnitude=float(pk.hybrid / 100), spread=min(len(zs) / 3, 1.0),
                    duration=min(dur / 90, 1.0), agreement=float(flagged.strong.mean() / 3))
        sev = 100 * (0.45 * comp["magnitude"] + 0.20 * comp["spread"] + 0.15 * comp["duration"]
                     + 0.20 * comp["agreement"])
        lf = {f: float(pk[f"lstm_err_{f}"]) for f in cc.FEATURES}
        tot = sum(lf.values()) or 1.0
        evidence = []
        for f in sorted(cc.FEATURES, key=lambda f: -zsc[f]):
            if zsc[f] >= 2:
                evidence.append(dict(feed=f, value=float(pk[f]), typical=float(pk[f"{f}_typical"]), z=zsc[f]))
        truth = rows.loc[rows.ground_truth_event == 1, "event_label"]
        first_anchor = flagged.loc[flagged.confirmed_anchor == 1, "timestamp"]
        events.append(dict(
            start=str(s), end=str(e_), zone_ids=zs, zone_names=[zones.at[z, "zone_name"] for z in zs],
            peak_time=str(pk.timestamp), peak_zone=pk.zone_id, peak_hybrid=round(float(pk.hybrid), 1),
            peak_state=str(pk.state), severity=round(float(sev), 1), duration_min=int(dur),
            components={k: round(v, 3) for k, v in comp.items()},
            channels=dict(stat=round(float(pk.stat_score), 3), iso=round(float(pk.iso_score), 3),
                          lstm=round(float(pk.lstm_score), 3)),
            lstm_feed_share={f: round(v / tot, 3) for f, v in lf.items()},
            z_scores={f: round(v, 2) for f, v in zsc.items()}, evidence=evidence,
            pattern=cc.describe_pattern(zsc),
            confirmed_at=str(first_anchor.min()) if len(first_anchor) else None,
            split=str(pk.split),
            matches_injected_scenario=truth.iloc[0] if len(truth) else None,
        ))
    events.sort(key=lambda e: e["start"])
    for i, e in enumerate(events, 1):
        e["event_id"] = f"EV{i:03d}"
    log(f"{len(events)} city-level events")

    # ---- export
    keep = (["timestamp", "zone_id", "zone_name", "lat", "lon", "split", "temperature", "incident_count",
             "ground_truth_event", "event_label", "decoy"]
            + cc.FEATURES + [f"{f}_typical" for f in cc.FEATURES] + [f"{f}_sigma" for f in cc.FEATURES]
            + [f"z_{f}" for f in cc.FEATURES] + [f"dz_{f}" for f in cc.FEATURES]
            + ["stat_score", "iso_score", "lstm_score", "lstm_error"] + [f"lstm_err_{f}" for f in cc.FEATURES]
            + ["hybrid", "strong", "warning_run", "confirmed_anchor", "final_flag", "state", "pulse"]
            + [c for f in cc.FEATURES for c in (f"wo_{f}__hybrid", f"wo_{f}__strong", f"wo_{f}__state")])
    sc = df[keep].copy()
    for c in sc.select_dtypes("float").columns:
        sc[c] = sc[c].round(4)
    sc.sort_values(["timestamp", "zone_id"]).to_csv(out / "scored.csv", index=False)
    # IF training matrix so the app can refit the identical model for live what-if scoring
    pd.DataFrame(cc.model_matrix(df[is_tr & normal]), columns=cc.ISO_COLS).round(4).to_csv(
        out / "iso_train.csv", index=False)
    meta = dict(
        version="3.0", generated=time.strftime("%Y-%m-%d %H:%M"), data_source=source,
        simulated=["traffic_congestion", "incident_count", "all labelled events and decoys"],
        zones=zones.reset_index().to_dict("records"),
        split=dict(train_end=str(pd.Timestamp(c1)), val_end=str(pd.Timestamp(c2)),
                   rule="chronological 60/20/20 by timestamp"),
        weights=W, thresholds=THR, strong_channel=cc.STRONG,
        calibration=dict(iso_lo=iso_lo, iso_hi=iso_hi, lstm_lo=lstm_lo, lstm_hi=lstm_hi,
                         fitted_on="validation rows with no labelled event"),
        iso_params={k: v for k, v in ISO_PARAMS.items() if k != "n_jobs"},
        lstm=dict(seq_len=SEQ_LEN, tail=LSTM_TAIL, epochs=len(hist["loss"]),
                  final_val_loss=float(hist["val_loss"][-1]), architecture="LSTM 48-16 | 16-48 autoencoder"),
        refs=refs, metrics=metrics, scenarios=catalogue,
        note="Correlations are shown as possible links, not confirmed causes. Metrics are measured on "
             "injected scenarios and do not claim real-world accuracy.")
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (out / "events.json").write_text(json.dumps(events, indent=2, default=str))
    try:
        model.save(out / "lstm_autoencoder.keras")
    except Exception as ex:  # optional artefact
        log("could not save keras model:", ex)
    log("exported ->", sorted(p.name for p in out.iterdir()))
    return df, meta, events


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="city_data.csv")
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--source", default="Open-Meteo (real rainfall, temperature and AQI)")
    a = ap.parse_args()
    main(a.input, a.out, a.source)
