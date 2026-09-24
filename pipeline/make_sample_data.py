"""Offline stand-in for city_data.csv (same logic/schema as CityPulse_Core_Engine.ipynb).

Only used when the real Open-Meteo-based city_data.csv is not available.
Every row produced here is simulated - the pipeline records that in model_meta.json.
"""
import numpy as np
import pandas as pd

SEED = 42
ZONES = pd.DataFrame([
    ("MN", "Malviya Nagar", 26.8540, 75.8180, 1.00, 1.0, 0),
    ("VN", "Vaishali Nagar", 26.9124, 75.7273, 0.95, 0.9, -4),
    ("MS", "Mansarovar", 26.8490, 75.7620, 0.90, 0.9, 3),
    ("CS", "C-Scheme", 26.9075, 75.8000, 1.10, 1.1, 6),
    ("SC", "Sindhi Camp", 26.9195, 75.7997, 1.20, 1.2, 10),
], columns=["zone_id", "zone_name", "lat", "lon", "traffic_factor", "incident_factor", "aqi_offset"])


def make(start="2026-09-18 00:00", days=7):
    rng = np.random.default_rng(SEED)
    ts = pd.date_range(start, periods=days * 24, freq="h")
    h = ts.hour.values
    hourly = pd.DataFrame({
        "timestamp": ts,
        "temperature": 28 + 6 * np.sin((h - 9) * np.pi / 12) + rng.normal(0, 0.8, len(ts)),
        "rain_mm_h": np.where(rng.random(len(ts)) < 0.04, rng.gamma(2, 1.2, len(ts)), 0.0),
        "aqi": 95 + 18 * np.sin((h - 4) * np.pi / 12) + rng.normal(0, 6, len(ts)),
    }).set_index("timestamp")
    city = hourly.resample("5min").interpolate("time")
    city["rain_mm_h"] = city["rain_mm_h"].clip(lower=0)
    city = city.reset_index()

    frames = []
    for z in ZONES.itertuples():
        d = city.copy()
        d["zone_id"], d["zone_name"], d["lat"], d["lon"] = z.zone_id, z.zone_name, z.lat, z.lon
        d["rainfall"] = (d["rain_mm_h"] * rng.lognormal(0, 0.15, len(d))).clip(0)
        d["aqi"] = d["aqi"] + z.aqi_offset + rng.normal(0, 3, len(d))
        hf = d.timestamp.dt.hour + d.timestamp.dt.minute / 60
        base = 22 + 40 * np.exp(-((hf - 9) ** 2) / 3) + 46 * np.exp(-((hf - 18.5) ** 2) / 4)
        d["traffic_congestion"] = (base * z.traffic_factor + 1.2 * d["rainfall"].clip(0, 10)
                                   + rng.normal(0, 4, len(d))).clip(0, 100)
        lam = z.incident_factor * (0.12 + 0.007 * d["traffic_congestion"] + 0.02 * d["rainfall"].clip(0, 10))
        d["incident_count"] = rng.poisson(lam)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["ground_truth_event"], df["event_label"], df["scenario_injected"] = 0, "", False

    day0 = df.timestamp.min().normalize()
    A0, A1 = day0 + pd.Timedelta(days=days - 3, hours=18), day0 + pd.Timedelta(days=days - 3, hours=18, minutes=45)
    B0, B1 = day0 + pd.Timedelta(days=days - 2, hours=20), day0 + pd.Timedelta(days=days - 2, hours=20, minutes=40)
    mA = df.timestamp.between(A0, A1) & df.zone_id.isin(["MN", "MS"])
    n = mA.sum()
    df.loc[mA, "rainfall"] += rng.uniform(15, 30, n)
    df.loc[mA, "traffic_congestion"] = (df.loc[mA, "traffic_congestion"] + rng.uniform(25, 40, n)).clip(0, 100)
    df.loc[mA, "incident_count"] += rng.poisson(3, n)
    df.loc[mA, ["ground_truth_event", "scenario_injected"]] = [1, True]
    df.loc[mA, "event_label"] = "A_flood_disruption"
    mB = df.timestamp.between(B0, B1) & (df.zone_id == "SC")
    n = mB.sum()
    df.loc[mB, "aqi"] += rng.uniform(60, 100, n)
    df.loc[mB, "incident_count"] += rng.poisson(2.5, n)
    df.loc[mB, "traffic_congestion"] = (df.loc[mB, "traffic_congestion"] + rng.uniform(20, 30, n)).clip(0, 100)
    df.loc[mB, ["ground_truth_event", "scenario_injected"]] = [1, True]
    df.loc[mB, "event_label"] = "B_incident_cluster_aqi"

    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("zone_id")
    for f in ["rainfall", "traffic_congestion", "aqi"]:
        df[f] = g[f].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["incidents_30m"] = g["incident_count"].transform(lambda s: s.rolling(6, min_periods=1).sum())
    keep = ["timestamp", "zone_id", "zone_name", "lat", "lon", "rainfall", "temperature", "aqi",
            "traffic_congestion", "incident_count", "incidents_30m", "ground_truth_event", "event_label",
            "scenario_injected"]
    return df[keep]


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "city_data.csv"
    make().to_csv(out, index=False)
    print("wrote", out)
