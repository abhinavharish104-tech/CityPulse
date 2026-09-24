"""CityPulse v3 — live civic health dashboard for Jaipur.

Reads the data contract written by pipeline/citypulse_pipeline.py (data/scored.csv, events.json,
model_meta.json, iso_train.csv). TensorFlow is NOT needed here: LSTM scores are precomputed, while
statistical and Isolation Forest evidence are recomputed live for citizen reports and what-if.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit.components.v1 as components
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest

import citypulse_core as cc
import cp_llm
import cp_recorder
import cp_ui as ui

st.set_page_config(page_title="CityPulse — Jaipur", page_icon="💗", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(ui.CSS, unsafe_allow_html=True)
DATA = Path(__file__).parent / "data"
PLOT_CFG = {"displayModeBar": False}


# =============================================================================== data
@st.cache_data(show_spinner=False)
def load():
    df = pd.read_csv(DATA / "scored.csv", parse_dates=["timestamp"])
    for c in ["event_label", "decoy"]:
        df[c] = df[c].fillna("")
    meta = json.loads((DATA / "model_meta.json").read_text())
    events = json.loads((DATA / "events.json").read_text())
    for e in events:
        for k in ["start", "end", "peak_time", "confirmed_at"]:
            e[k] = pd.Timestamp(e[k]) if e.get(k) not in (None, "None") else None
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    return df, meta, events


@st.cache_resource(show_spinner=False)
def iso_model(params_json: str):
    """Refit the Isolation Forest on the exported training matrix (same params, same seed)."""
    X = pd.read_csv(DATA / "iso_train.csv")[cc.ISO_COLS].to_numpy()
    return IsolationForest(contamination="auto", n_jobs=1, **json.loads(params_json)).fit(X)


if not (DATA / "scored.csv").exists():
    st.error("No detector output found. Run `python pipeline/citypulse_pipeline.py --input city_data.csv` "
             "to create the data folder, then reload.")
    st.stop()

DF, META, EVENTS = load()
THR, W, CAL = META["thresholds"], META["weights"], META["calibration"]
ISO = iso_model(json.dumps(META["iso_params"]))
ZONES = pd.DataFrame(META["zones"])
ZNAME = dict(zip(ZONES.zone_id, ZONES.zone_name))
TS = sorted(DF.timestamp.unique())
N = len(TS)
STATE_RANK = {"Normal": 0, "Watch": 1, "Confirmed": 2, "Critical": 3}


def secrets_safe():
    try:
        return st.secrets
    except Exception:
        return None


LLM = cp_llm.config(secrets_safe())

# =============================================================================== language
L10N = {
    "en": dict(calm_all="All five neighbourhoods are calm",
               calm_sub="Rain, traffic, road incidents and air quality are all within their usual range for this hour.",
               critical="{z} needs attention now", critical_pl="{z} need attention now", disrupted="{z} is disrupted right now",
               disrupted_pl="{z} are disrupted right now", watch="Keep an eye on {z}",
               feed_line="{f} {v} (usually {t})",
               link=" Signals rising together suggest a possible link, not a confirmed cause.",
               typical="Everything typical for this hour", slightly="Slightly unusual readings",
               pulse_lbl="city pulse out of 100", usually="usually",
               feeds_all="All 4 feeds live", feeds_some="{n} of 4 feeds live, confidence {c}%",
               states={"Normal": "Calm", "Watch": "Watch", "Confirmed": "Disrupted", "Critical": "Critical"}),
    "hi": dict(calm_all="सभी पाँच इलाक़े सामान्य हैं",
               calm_sub="बारिश, ट्रैफ़िक, सड़क दुर्घटनाएँ और हवा की गुणवत्ता, सब इस समय के सामान्य दायरे में हैं।",
               critical="{z} पर तुरंत ध्यान दें", critical_pl="{z} पर तुरंत ध्यान दें", disrupted="{z} में इस समय बाधा है",
               disrupted_pl="{z} में इस समय बाधा है", watch="{z} पर नज़र रखें",
               feed_line="{f} {v} (आमतौर पर {t})",
               link=" साथ-साथ बढ़ते संकेत एक संभावित संबंध दिखाते हैं, पुष्ट कारण नहीं।",
               typical="इस समय सब सामान्य", slightly="थोड़ी असामान्य रीडिंग",
               pulse_lbl="शहर की नब्ज़, 100 में से", usually="आमतौर पर",
               feeds_all="चारों फ़ीड चालू", feeds_some="4 में से {n} फ़ीड चालू, भरोसा {c}%",
               states={"Normal": "सामान्य", "Watch": "निगरानी", "Confirmed": "बाधित", "Critical": "गंभीर"}),
}
FEED_HI = {"rainfall": "बारिश", "traffic_congestion": "ट्रैफ़िक जाम", "incidents_30m": "सड़क दुर्घटनाएँ",
           "aqi": "वायु गुणवत्ता (AQI)"}


def fmt_feed(f, v):
    return f"{cc.FMT[f].format(v)}{'%' if f == 'traffic_congestion' else ''}" + (
        " mm/h" if f == "rainfall" else "")


def feed_name(f, lang):
    return FEED_HI[f] if lang == "hi" else cc.NAMES[f]


def join_names(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


# =============================================================================== state
ss = st.session_state
ss.setdefault("t_idx", max(0, next((i for i, t in enumerate(TS) if t >= pd.Timestamp(META["split"]["val_end"])), 0)))
ss.setdefault("playing", False)
ss.setdefault("reports", [])
ss.setdefault("chat", [])
ss.setdefault("alert_seen", -1)
ss.setdefault("llm_cache", {})

if "t" in st.query_params and not ss.get("_qp_done"):
    try:
        _t = pd.Timestamp(st.query_params["t"])
        ss.t_idx = int(np.clip(np.searchsorted(np.array(TS, dtype="datetime64[ns]"), np.datetime64(_t)), 0, N - 1))
    except Exception:
        pass
    ss._qp_done = True
def jump_to(ts):
    i = int(np.searchsorted(np.array(TS, dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(ts))))
    ss.t_idx = int(np.clip(i, 0, N - 1))
    ss.playing = False


# =============================================================================== sidebar
with st.sidebar:
    st.markdown(ui.BRAND, unsafe_allow_html=True)
    mode = st.radio("View", ["Resident", "City operations"], horizontal=True,
                    help="City operations adds detector scores, channel evidence and tables.")
    lang = "hi" if st.radio("Summary language", ["English", "हिन्दी"], horizontal=True) == "हिन्दी" else "en"
    my_zone = st.selectbox("My neighbourhood", ZONES.zone_id, format_func=ZNAME.get,
                           index=list(ZONES.zone_id).index("SC") if "SC" in ZNAME else 0)
    T = L10N[lang]

    st.subheader("Replay")
    st.toggle("Play the week", key="playing")
    st.select_slider("Speed", options=[1, 2, 3, 6], value=2, key="speed",
                     format_func=lambda x: f"{x * 5} min per tick")
    st.caption("Jump to a detected event")
    for e in EVENTS:
        lbl = f"{e['start']:%a %d %b, %H:%M}  {', '.join(e['zone_names'])}"
        st.button(lbl, key=f"jump_{e['event_id']}", on_click=jump_to,
                  args=(e["start"] - pd.Timedelta(minutes=45),), width="stretch")

    st.subheader("Test a feed outage")
    outage = st.selectbox("Feed availability", [None] + cc.FEATURES,
                          format_func=lambda f: "All feeds live" if f is None else f"{cc.NAMES[f]} offline")
    st.subheader("AI summaries")
    use_ai = st.toggle("Let AI write the summaries", value=LLM is not None, disabled=LLM is None)
    st.caption(f"Connected: {LLM['provider'].title()} ({LLM['model']})" if LLM else
               "Add an API key in Streamlit secrets to enable. Template summaries are used until then.")


# =============================================================================== live rescoring
def rescore(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["stat_score"] = cc.stat_score(out)
    out["iso_score"] = cc.calibrate(cc.iso_raw(ISO, out), CAL["iso_lo"], CAL["iso_hi"])
    out["hybrid"] = cc.hybrid(out.stat_score, out.iso_score, out.lstm_score, W)
    out["strong"] = cc.strong_count(out.stat_score, out.iso_score, out.lstm_score)
    return out


@st.cache_data(show_spinner=False, max_entries=16)
def with_reports(reports: tuple) -> pd.DataFrame:
    """Citizen reports become incidents in that zone at that time and flow through the detector."""
    d = DF.copy()
    if not reports:
        return d
    touched = pd.Series(False, index=d.index)
    for zone, t_iso, n in reports:
        t = pd.Timestamp(t_iso)
        m = (d.zone_id == zone) & d.timestamp.between(t, t + pd.Timedelta(minutes=25))
        d.loc[m, "incidents_30m"] += n
        d.loc[(d.zone_id == zone) & (d.timestamp == t), "incident_count"] += n
        touched |= (d.zone_id == zone) & d.timestamp.between(t, t + pd.Timedelta(minutes=45))
    d["z_incidents_30m"] = (d.incidents_30m - d.incidents_30m_typical) / d.incidents_30m_sigma
    d["dz_incidents_30m"] = d.groupby("zone_id")["z_incidents_30m"].diff(cc.DELTA_STEPS).fillna(0.0)
    d.loc[touched] = rescore(d.loc[touched])
    det = cc.run_detector(d, THR)
    for c in det.columns:
        d[c] = det[c]
    d["pulse"] = cc.pulse_from_hybrid(d.hybrid, THR)
    return d


def reports_key():
    return tuple((r["zone"], str(r["time"]), r["n"]) for r in ss.reports)


D = with_reports(reports_key())
if outage:
    V = D.assign(h=D[f"wo_{outage}__hybrid"], strong_v=D[f"wo_{outage}__strong"], st_v=D[f"wo_{outage}__state"])
else:
    V = D.assign(h=D.hybrid, strong_v=D.strong, st_v=D.state)
V["pulse_v"] = cc.pulse_from_hybrid(V.h, THR)
feeds_live = [f for f in cc.FEATURES if f != outage]

# =============================================================================== time control
PLAY_EVERY = 1.0  # seconds between ticks while the week plays


def _slider_moved():
    ss.t_idx = ss.t_slider


if not ss.playing:
    ss.t_slider = ss.t_idx
st.select_slider("Replay time", options=list(range(N)), key="t_slider", on_change=_slider_moved,
                 format_func=lambda i: pd.Timestamp(TS[i]).strftime("%a %d %b %Y, %H:%M"),
                 label_visibility="collapsed", disabled=ss.playing)


def snap_at(i):
    now = pd.Timestamp(TS[i])
    return now, V[V.timestamp == now].set_index("zone_id").loc[ZONES.zone_id]


NOW, snap = snap_at(ss.t_idx)


# =============================================================================== summaries
def zone_reason(r, lang, max_feeds=2):
    hot = sorted([f for f in feeds_live if r[f"z_{f}"] >= 2.0], key=lambda f: -r[f"z_{f}"])[:max_feeds]
    if not hot:
        return L10N[lang]["slightly"] if r.st_v == "Watch" else L10N[lang]["typical"]
    return "; ".join(f"{feed_name(f, lang)} {fmt_feed(f, r[f])} ({L10N[lang]['usually']} "
                     f"{fmt_feed(f, r[f + '_typical'])})" for f in hot)


def template_summary(snap, lang):
    T = L10N[lang]
    worst = snap.sort_values("h", ascending=False)
    ranks = worst.st_v.map(STATE_RANK)
    top = worst.iloc[0]
    names = lambda st_: [ZNAME[z] for z in worst.index[worst.st_v == st_]]
    if ranks.max() == 0:
        return T["calm_all"], T["calm_sub"], "Normal"
    if (worst.st_v == "Critical").any():
        n = names("Critical")
        head, state = (T["critical_pl"] if len(n) > 1 else T["critical"]).format(z=join_names(n)), "Critical"
    elif (worst.st_v == "Confirmed").any():
        n = names("Confirmed")
        head, state = (T["disrupted_pl"] if len(n) > 1 else T["disrupted"]).format(z=join_names(n)), "Confirmed"
    else:
        head, state = T["watch"].format(z=join_names(names("Watch"))), "Watch"
    hot = sorted([f for f in feeds_live if top[f"z_{f}"] >= 2.0], key=lambda f: -top[f"z_{f}"])
    if not hot:
        sub = (f"{ZNAME[top.name]} में रीडिंग इस समय के सामान्य स्तर से थोड़ी ऊपर है।" if lang == "hi" else
               f"Readings in {ZNAME[top.name]} are drifting above normal for this hour.")
        return head, sub, state
    parts = [T["feed_line"].format(f=feed_name(f, lang), v=fmt_feed(f, top[f]), t=fmt_feed(f, top[f + "_typical"]))
             for f in hot[:2]]
    if lang == "en":
        parts = [p_ if i == 0 else p_[0].lower() + p_[1:] for i, p_ in enumerate(parts)]
        sub = f"{' and '.join(parts)} in {ZNAME[top.name]} at this hour."
    else:
        sub = f"{ZNAME[top.name]}: {' और '.join(parts)}।"
    if len(hot) >= 2:
        sub += T["link"]
    return head, sub, state


def snapshot_facts(snap, now=None):
    now = now if now is not None else NOW
    return dict(time=f"{now:%A %d %B %Y, %H:%M}", feeds_available=feeds_live,
                zones=[dict(zone=ZNAME[z], status=ui.STATE_WORD[r.st_v], pulse=int(r.pulse_v),
                            unusual_feeds=[dict(feed=cc.NAMES[f], now=round(float(r[f]), 1),
                                                usual_for_this_hour=round(float(r[f + "_typical"]), 1))
                                           for f in feeds_live if r[f"z_{f}"] >= 2.0])
                       for z, r in snap.iterrows()])


def ai_text(kind: str, task: str, facts: dict, max_tokens=300):
    """Cached, grounded LLM call. Never runs during auto-play."""
    if not (use_ai and LLM) or ss.playing:
        return None, "off"
    key = hashlib.sha1((kind + task + json.dumps(facts, sort_keys=True, default=str)).encode()).hexdigest()
    if key not in ss.llm_cache:
        ss.llm_cache[key] = cp_llm.grounded(LLM, task, facts, max_tokens)
    return ss.llm_cache[key]


def hex_rgb(h, a=255):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)] + [a]


def city_map(snap):
    """Native Streamlit map (deck.gl + CARTO dark basemap, no token needed)."""
    m = snap.reset_index()
    m["name"] = m.zone_id.map(ZNAME)
    m["word"] = m.st_v.map(T["states"])
    m["p"] = m.pulse_v.astype(int)
    m["r"] = 230 + (100 - m.pulse_v) * 6
    m["r_halo"] = m["r"] * 2.4
    m["fill"] = [hex_rgb(ui.STATE[s_]) for s_ in m.st_v]
    m["halo"] = [hex_rgb(ui.STATE[s_], 0 if s_ == "Normal" else 70) for s_ in m.st_v]
    # labels above the dot, except where two neighbourhoods sit close together (C-Scheme under Sindhi Camp)
    m["off"] = [[0, 22] if z == "CS" else [0, -22] for z in m.zone_id]
    data = m[["lon", "lat", "name", "word", "p", "r", "r_halo", "fill", "halo", "off"]].to_dict("records")
    layers = [
        pdk.Layer("ScatterplotLayer", data, get_position=["lon", "lat"], get_radius="r_halo",
                  get_fill_color="halo", pickable=False),
        pdk.Layer("ScatterplotLayer", data, get_position=["lon", "lat"], get_radius="r", get_fill_color="fill",
                  stroked=True, get_line_color=[21, 18, 46, 255], line_width_min_pixels=2, pickable=True),
        pdk.Layer("TextLayer", data, get_position=["lon", "lat"], get_text="name", get_size=14,
                  get_color=[237, 232, 247, 255], get_pixel_offset="off", font_weight=700,
                  outline_width=3, outline_color=[15, 12, 36, 255], font_settings={"sdf": True}),
    ]
    deck = pdk.Deck(layers=layers, map_provider="carto", map_style=pdk.map_styles.CARTO_DARK,
                    initial_view_state=pdk.ViewState(latitude=float((m.lat.min() + m.lat.max()) / 2),
                                                     longitude=float((m.lon.min() + m.lon.max()) / 2),
                                                     zoom=11.1, pitch=0),
                    tooltip={"html": "<b>{name}</b><br/>{word}, pulse {p}",
                             "style": {"backgroundColor": ui.RAISED, "color": ui.TEXT, "border": f"1px solid {ui.LINE}",
                                       "borderRadius": "8px", "fontFamily": "Atkinson Hyperlegible, sans-serif"}})
    st.pydeck_chart(deck, height=400)


ops = mode == "City operations"


@st.fragment(run_every=PLAY_EVERY if ss.playing else None)
def live_board():
    """Hero + vital monitor + map. While playing only this fragment re-runs, not the page."""
    if ss.playing:
        ss.t_idx = (ss.t_idx + ss.get("speed", 1)) % N
    now, snap = snap_at(ss.t_idx)
    head, sub, hstate = template_summary(snap, lang)
    ai_used = False
    if hstate != "Normal" and not ss.playing:
        lang_word = "Hindi" if lang == "hi" else "English"
        txt, status = ai_text("hero", f"In {lang_word}, write ONE short sentence (max 35 words) telling a resident "
                                      f"what is unusual right now and where. Mention at most two readings with their "
                                      f"usual value. Frame co-occurring signals as a possible link.",
                              snapshot_facts(snap, now))
        if txt:
            sub, ai_used = txt, True
    city_pulse = int(round(snap.pulse_v.mean()))
    conf = T["feeds_all"] if not outage else T["feeds_some"].format(n=3, c=75)
    when = f"{now:%A %d %B, %H:%M}" + ("   ▶ playing the week" if ss.playing else "")
    st.markdown(ui.hero(head, sub, when, city_pulse, hstate, conf, ai_used), unsafe_allow_html=True)
    st.markdown(f'<div class="source">Rainfall, temperature and AQI: <b>{ui.esc(META["data_source"])}</b>. '
                f'Traffic, incidents and all labelled events: <b>simulated</b>.'
                + (f' <b>{len(ss.reports)} citizen report(s)</b> included.' if ss.reports else "") + '</div>',
                unsafe_allow_html=True)
    mine = snap.loc[my_zone]
    if STATE_RANK[mine.st_v] >= STATE_RANK[ss.get("alert_level", "Confirmed")] and ss.alert_seen != ss.t_idx:
        st.toast(f"{ZNAME[my_zone]}: {T['states'][mine.st_v]}. {zone_reason(mine, lang, 1)}", icon="🔔")
        ss.alert_seen = ss.t_idx
    if True:
        rows = [dict(zone_name=ZNAME[z], pulse=r.pulse_v, state=r.st_v, state_word=T["states"][r.st_v],
                     why=zone_reason(r, lang), mine=(z == my_zone), seed=sum(map(ord, z)))
                for z, r in snap.sort_values("h", ascending=False).iterrows()]
        st.markdown(ui.monitor(rows), unsafe_allow_html=True)
        if outage:
            st.caption(f"{cc.NAMES[outage]} is switched off. Every score has been re-computed by all three "
                       f"detectors without it, so the pulse keeps working with slightly less certainty.")
    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        city_map(snap)
    with right:
        recent = [e for e in EVENTS if e["start"] <= now <= e["end"] + pd.Timedelta(hours=3)]
        st.markdown("#### Last three hours")
        if not recent:
            st.markdown('<div class="tick">No disruptions. Every neighbourhood has stayed within its usual range.</div>',
                        unsafe_allow_html=True)
        for e in recent:
            live = now <= e["end"]
            st.markdown(f'<div class="tick" style="border-left:4px solid {ui.STATE[e["peak_state"]]}">'
                        f'<b>{"Ongoing" if live else "Cleared at " + format(e["end"], "%H:%M")}</b><br>'
                        f'{ui.esc(e["pattern"])} in {ui.esc(join_names(e["zone_names"]))}, since {e["start"]:%H:%M}.</div>',
                        unsafe_allow_html=True)
        bad = snap[snap.st_v != "Normal"]
        st.markdown("#### Allow extra time in" if len(bad) else "#### Getting around")
        st.markdown(("".join(f'<div class="tick">{ui.chip(r.st_v, T["states"][r.st_v])} {ui.esc(ZNAME[z])}</div>'
                             for z, r in bad.iterrows())) if len(bad) else
                    '<div class="tick">Normal travel times expected across all five neighbourhoods.</div>',
                    unsafe_allow_html=True)
    if ops:
        st.markdown("#### Detector channels at this moment")
        t = snap.reset_index()[["zone_id", "st_v", "h", "stat_score", "iso_score", "lstm_score", "strong_v",
                                "warning_run"]]
        t.insert(1, "zone", t.zone_id.map(ZNAME))
        st.dataframe(t.drop(columns="zone_id").rename(columns={
            "st_v": "state", "h": "hybrid", "stat_score": "statistics", "iso_score": "isolation forest",
            "lstm_score": "LSTM", "strong_v": "channels agreeing", "warning_run": "readings in a row"}).round(3),
            hide_index=True, width="stretch")
    if ss.playing:
        st.caption("The week is playing. The tabs below refresh when you pause.")


live_board()
NOW, snap = snap_at(ss.t_idx)
tabs = st.tabs(["Timeline", "Events", "Why this score", "What if", "Report a problem",
                "Ask CityPulse", "Alerts", "How it works"])

# =============================================================================== 2. timeline
FEED_COL = {"rainfall": "#7CB7FF", "traffic_congestion": ui.MARIGOLD, "incidents_30m": ui.ROSE, "aqi": "#B79CFF"}


def shade_events(fig, zone, t0, t1, rows):
    for e in EVENTS:
        if zone in e["zone_ids"] and e["end"] >= t0 and e["start"] <= t1 and e["start"] <= NOW:
            for r in rows:
                fig.add_vrect(x0=e["start"], x1=min(e["end"] + pd.Timedelta(minutes=5), t1), row=r, col=1,
                              fillcolor=ui.STATE[e["peak_state"]], opacity=.13, line_width=0)


with tabs[0]:
    c1, c2, c3 = st.columns([1, 1.1, 1.3])
    tz = c1.selectbox("Neighbourhood", ZONES.zone_id, format_func=ZNAME.get,
                      index=list(ZONES.zone_id).index(snap.h.idxmax() if snap.h.max() >= THR["watch"] else my_zone),
                      key="tl_zone")
    view = c2.radio("View", ["Live recorder", "Full chart"], horizontal=True, key="tl_view")
    span = c3.radio("Window", ["6 hours", "24 hours", "Whole week"], horizontal=True, index=1,
                    disabled=view == "Live recorder")
    if view == "Live recorder":
        st.caption(f"Readings for {ZNAME[tz]} are drawn one by one from six hours before {NOW:%a %H:%M}, the way a "
                   f"monitor records them. Use pause, speed and 6 h back inside the recorder.")
        zd = V[V.zone_id == tz]
        data = cp_recorder.payload(zd, ZNAME[tz], NOW, THR, [e for e in EVENTS if tz in e["zone_ids"]],
                                   T["states"], outage)
        html_doc = cp_recorder.recorder_html(data)
        if hasattr(st, "iframe"):          # Streamlit >= 1.52
            st.iframe(html_doc, height=800)
        else:
            components.html(html_doc, height=800, scrolling=False)
with tabs[0]:
    if view == "Full chart":
        hrs = {"6 hours": 6, "24 hours": 24}.get(span)
        t0 = NOW - pd.Timedelta(hours=hrs) if hrs else TS[0]
        t1 = NOW  # a live replay never shows the future
        z = V[(V.zone_id == tz) & V.timestamp.between(t0, t1)]
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=.035,
                            row_heights=[.16, .16, .16, .16, .36],
                            subplot_titles=[f"{cc.NAMES[f]} ({cc.UNITS[f]})" for f in cc.FEATURES]
                            + ["Combined anomaly score and alert levels"])
        for i, f in enumerate(cc.FEATURES, 1):
            lo = (z[f + "_typical"] - 2 * z[f + "_sigma"]).clip(lower=0)
            hi = z[f + "_typical"] + 2 * z[f + "_sigma"]
            fig.add_trace(go.Scatter(x=z.timestamp, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"), i, 1)
            fig.add_trace(go.Scatter(x=z.timestamp, y=lo, fill="tonexty", fillcolor="rgba(166,159,199,.12)",
                                     line=dict(width=0), name="usual range", showlegend=(i == 1), hoverinfo="skip"), i, 1)
            fig.add_trace(go.Scatter(x=z.timestamp, y=z[f + "_typical"], line=dict(color=ui.MUTED, dash="dot", width=1),
                                     name="typical for the hour", showlegend=(i == 1)), i, 1)
            dim = f == outage
            fig.add_trace(go.Scatter(x=z.timestamp, y=z[f], line=dict(color=ui.MUTED if dim else FEED_COL[f], width=2,
                                                                      dash="dash" if dim else "solid"),
                                     name=cc.NAMES[f] + (" (offline)" if dim else ""), showlegend=False), i, 1)
        for lo_, hi_, stt in [(THR["watch"], THR["confirmed"], "Watch"), (THR["confirmed"], THR["critical"], "Confirmed"),
                              (THR["critical"], 100, "Critical")]:
            fig.add_hrect(y0=lo_, y1=hi_, row=5, col=1, fillcolor=ui.STATE[stt], opacity=.08, line_width=0)
        fig.add_trace(go.Scatter(x=z.timestamp, y=z.h, line=dict(color=ui.TEXT, width=2), name="combined score",
                                 showlegend=False), 5, 1)
        flagged = z[z.st_v != "Normal"]
        fig.add_trace(go.Scatter(x=flagged.timestamp, y=flagged.h, mode="markers", showlegend=False,
                                 marker=dict(color=[ui.STATE[s] for s in flagged.st_v], size=7),
                                 text=flagged.st_v, hovertemplate="%{text} %{y:.0f}<extra></extra>"), 5, 1)
        shade_events(fig, tz, t0, t1, rows=range(1, 6))
        for r in range(1, 6):
            fig.add_vline(x=NOW, line=dict(color=ui.MARIGOLD, width=1.5), row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=5, col=1)
        ui.plotly_theme(fig, 760).update_layout(legend=dict(orientation="h", y=1.06), hovermode="x unified")
        fig.update_annotations(font=dict(size=13, color=ui.MUTED), x=0, xanchor="left")
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)

    st.markdown("#### Every neighbourhood across the week")
    hm = V.pivot_table(index="zone_id", columns="timestamp", values="pulse_v").loc[list(ZONES.zone_id[::-1])]
    hm.loc[:, hm.columns > NOW] = np.nan  # the future is not known yet
    fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=[ZNAME[z] for z in hm.index], zmin=0, zmax=100,
                               colorscale=[[0, ui.STATE["Critical"]], [.25, ui.STATE["Confirmed"]],
                                           [.5, ui.STATE["Watch"]], [.7, "#2E6B61"], [1, "#1B2F3A"]],
                               colorbar=dict(title="pulse", thickness=10),
                               hovertemplate="%{y}<br>%{x|%a %d %b %H:%M}<br>pulse %{z:.0f}<extra></extra>"))
    fig.add_vline(x=NOW, line=dict(color=ui.MARIGOLD, width=2))
    val_end = pd.Timestamp(META["split"]["val_end"])
    fig.add_vrect(x0=val_end, x1=TS[-1], fillcolor="rgba(0,0,0,0)", line=dict(color=ui.ROSE, width=1, dash="dot"),
                  annotation_text="unseen test period", annotation_position="top left",
                  annotation_font_color=ui.ROSE)
    st.plotly_chart(ui.plotly_theme(fig, 260), width="stretch", config=PLOT_CFG)


# =============================================================================== 3. events
def scenario_for(e):
    lab = e.get("matches_injected_scenario")
    if not lab:
        return None
    for s in META["scenarios"]:
        if lab.startswith(s["id"] + "_"):
            return s
    return None


def event_evidence(e, lang="en"):
    out = []
    for ev in e["evidence"]:
        f = ev["feed"]
        out.append(f"{cc.NAMES[f]} reached {fmt_feed(f, ev['value'])}, when it is usually "
                   f"{fmt_feed(f, ev['typical'])} at that hour")
    ch = e["channels"]
    agree = [n for n, k in [("the statistics check", "stat"), ("the Isolation Forest", "iso"),
                            ("the LSTM sequence model", "lstm")] if ch[k] >= cc.STRONG]
    if agree:
        out.append(f"Flagged independently by {join_names(agree)}")
    return out


def event_markdown(e):
    s = scenario_for(e)
    lines = [f"# {e['event_id']}: {e['pattern']}", "",
             f"- Where: {join_names(e['zone_names'])}",
             f"- When: {e['start']:%a %d %b %Y %H:%M} to {e['end']:%H:%M} ({e['duration_min']} min)",
             f"- Peak state: {ui.STATE_WORD[e['peak_state']]} (combined score {e['peak_hybrid']})",
             f"- Severity: {e['severity']:.0f}/100", "", "## Evidence"]
    lines += [f"- {x}" for x in event_evidence(e)]
    if s:
        lines += ["", f"Matches simulated scenario {s['id']}: {s['description']}."]
    lines += ["", "Signals rising together are a possible link, not a confirmed cause."]
    return "\n".join(lines)


with tabs[1]:
    c1, c2 = st.columns([2, 1])
    show_all = c2.toggle("Show the whole week", value=False, help="Off: only events up to the replay time.")
    evs = [e for e in EVENTS if show_all or e["start"] <= NOW]
    c1.markdown(f'<div class="big-note">{len(evs)} event{"s" if len(evs) != 1 else ""} '
                f'{"this week" if show_all else "so far"}</div>', unsafe_allow_html=True)
    if not evs:
        st.markdown('<div class="panel">Nothing unusual has been confirmed yet. Press <b>Play the week</b> in the '
                    'sidebar, or jump straight to an event.</div>', unsafe_allow_html=True)
    for e in reversed(evs):
        s = scenario_for(e)
        ongoing = (not show_all) and e["start"] <= NOW < e["end"]
        if ongoing:  # never reveal how an event ends before the replay gets there
            seen = V[V.zone_id.isin(e["zone_ids"]) & V.timestamp.between(e["start"], NOW)]
            pk = seen.loc[seen.h.idxmax()]
            e = dict(e, end=NOW, peak_state=max(seen.st_v, key=STATE_RANK.get),
                     duration_min=int((NOW - e["start"]).total_seconds() // 60) + 5,
                     evidence=[dict(feed=f, value=float(pk[f]), typical=float(pk[f + "_typical"]), z=float(pk[f"z_{f}"]))
                               for f in sorted(cc.FEATURES, key=lambda f: -pk[f"z_{f}"]) if pk[f"z_{f}"] >= 2],
                     channels=dict(stat=pk.stat_score, iso=pk.iso_score, lstm=pk.lstm_score),
                     confirmed_at=e["confirmed_at"] if e["confirmed_at"] and e["confirmed_at"] <= NOW else None)
        col = ui.STATE[e["peak_state"]]
        lag = ""
        if s and e.get("confirmed_at") is not None:
            mins = int((e["confirmed_at"] - pd.Timestamp(s["start"])).total_seconds() // 60)
            lag = f"Confirmed {mins} min after the disruption began. "
        origin = (f"Matches simulated scenario {s['id']}: {s['description']}." if s else
                  "Not one of the planned scenarios: the detector found this on its own in the "
                  + ("training period (real weather)." if e["split"] == "train" else "data."))
        ev_html = "".join(f"<li>{ui.esc(x)}</li>" for x in event_evidence(e))
        st.markdown(
            f'<div class="event" style="--accent:{col}"><div class="top"><div>'
            f'<h3>{ui.esc(e["pattern"])}</h3><div class="meta">{ui.esc(join_names(e["zone_names"]))}. '
            + (f'{e["start"]:%a %d %b, %H:%M}, still ongoing ({e["duration_min"]} min so far). ' if ongoing else
               f'{e["start"]:%a %d %b, %H:%M} to {e["end"]:%H:%M}, {e["duration_min"]} min. ') + f'{ui.esc(lag)}</div></div>'
            + (f'<div class="sev">live<small>in progress</small></div></div>' if ongoing else
               f'<div class="sev">{e["severity"]:.0f}<small>severity</small></div></div>')
            + f'<div style="margin-top:.4rem">{ui.chip(e["peak_state"])}</div><ul>{ev_html}</ul>'
            f'<div class="muted" style="font-size:.9rem">{ui.esc(origin)}</div>'
            f'<div class="caveat">Signals that rise together are shown as a possible link, not a confirmed cause.</div>'
            f'</div>', unsafe_allow_html=True)
        b1, b2, b3, _ = st.columns([1, 1.2, 1, 2.5])
        b1.button("Replay", key=f"rp_{e['event_id']}", on_click=jump_to,
                  args=(e["start"] - pd.Timedelta(minutes=30),))
        b3.download_button("Download", event_markdown(e), file_name=f"citypulse_{e['event_id']}.md",
                           key=f"dl_{e['event_id']}")
        if b2.button("Explain in plain words", key=f"ex_{e['event_id']}", disabled=not (use_ai and LLM)):
            facts = {k: e[k] for k in ["pattern", "zone_names", "duration_min", "peak_state", "severity"]}
            facts.update(start=f"{e['start']:%a %d %b %H:%M}", end=f"{e['end']:%H:%M}", evidence=event_evidence(e))
            txt, status = ai_text("event", "Explain this event to a resident in 3 short sentences: what happened, "
                                           "where and when, and what it may mean for getting around. "
                                           "Use a possible-link framing.", facts)
            st.success(txt) if txt else st.warning(f"AI explanation unavailable ({status}). The evidence above is "
                                                   f"generated directly from the data.")
        if ops:
            with st.expander("Detector detail"):
                comp = e["components"]
                cA, cB = st.columns(2)
                f1 = go.Figure(go.Bar(x=[comp[k] for k in ["magnitude", "spread", "duration", "agreement"]],
                                      y=["peak score", "area affected", "duration", "channel agreement"],
                                      orientation="h", marker_color=col))
                f1.update_layout(title="What drives the severity", xaxis=dict(range=[0, 1]))
                cA.plotly_chart(ui.plotly_theme(f1, 220), width="stretch", config=PLOT_CFG,
                                key=f"sev_{e['event_id']}")
                share = e["lstm_feed_share"]
                f2 = go.Figure(go.Bar(x=list(share.values()), y=[cc.NAMES[f] for f in share], orientation="h",
                                      marker_color=[FEED_COL[f] for f in share]))
                f2.update_layout(title="Which feed surprised the LSTM most", xaxis=dict(range=[0, 1]))
                cB.plotly_chart(ui.plotly_theme(f2, 220), width="stretch", config=PLOT_CFG,
                                key=f"lstm_{e['event_id']}")
                st.json({"channels_at_peak": e["channels"], "z_scores_at_peak": e["z_scores"],
                         "peak_time": str(e["peak_time"]), "confirmed_at": str(e["confirmed_at"])})

# =============================================================================== 4. why this score
with tabs[2]:
    wz = st.selectbox("Neighbourhood", ZONES.zone_id, format_func=ZNAME.get,
                      index=list(ZONES.zone_id).index(snap.h.idxmax()), key="why_zone")
    r = snap.loc[wz]
    st.markdown(f'<div class="big-note">{ZNAME[wz]} has a pulse of {int(r.pulse_v)}: '
                f'{ui.STATE_WORD[r.st_v].lower()} at {NOW:%H:%M}.</div>', unsafe_allow_html=True)
    cA, cB = st.columns([1.3, 1], gap="medium")
    with cA:
        if outage:
            st.caption("Channel split shown for the full feed set; the headline score above excludes the offline feed.")
        parts = [("Statistics vs typical", W["stat"] * r.stat_score * 100, ui.MARIGOLD),
                 ("Isolation Forest", W["iso"] * r.iso_score * 100, "#7CB7FF"),
                 ("LSTM sequence model", W["lstm"] * r.lstm_score * 100, ui.ROSE)]
        fig = go.Figure()
        for name, v, c in parts:
            fig.add_trace(go.Bar(x=[v], y=["score"], orientation="h", name=f"{name}  +{v:.0f}", marker_color=c))
        for k, stt, pos in [("watch", "Watch", "bottom left"), ("confirmed", "Confirmed", "top right"),
                            ("critical", "Critical", "top right")]:
            fig.add_vline(x=THR[k], line=dict(color=ui.STATE[stt], dash="dot"), annotation_position=pos,
                          annotation_text=ui.STATE_WORD[stt], annotation_font_color=ui.STATE[stt])
        fig.update_layout(barmode="stack", xaxis=dict(range=[0, 100], title="combined score"),
                          yaxis=dict(visible=False), legend=dict(orientation="h", y=-0.45),
                          title="How the three detectors add up")
        st.plotly_chart(ui.plotly_theme(fig, 230), width="stretch", config=PLOT_CFG)
        vals = [(f, float(r[f"z_{f}"])) for f in cc.FEATURES]
        fig = go.Figure(go.Bar(x=[v for _, v in vals], y=[cc.NAMES[f] for f, _ in vals], orientation="h",
                               marker_color=[ui.MUTED if f == outage else FEED_COL[f] for f, _ in vals],
                               text=[f"{fmt_feed(f, r[f])} vs usual {fmt_feed(f, r[f + '_typical'])}" for f, _ in vals],
                               textposition="auto"))
        fig.add_vline(x=2, line=dict(color=ui.STATE["Watch"], dash="dot"), annotation_text="clearly unusual")
        fig.update_layout(title="How far each feed is from its usual level (in usual-spreads)",
                          xaxis=dict(range=[min(-2, min(v for _, v in vals) - .5), max(6, max(v for _, v in vals) + 1)]))
        st.plotly_chart(ui.plotly_theme(fig, 260), width="stretch", config=PLOT_CFG)
    with cB:
        checks = [
            (r.h >= THR["watch"], f"Combined score {r.h:.0f} is above the watch level of {THR['watch']:.0f}"),
            (r.h >= THR["confirmed"], f"It is above the disrupted level of {THR['confirmed']:.0f}"),
            (r.strong_v >= 2, f"{int(r.strong_v)} of 3 detectors independently agree (2 needed)"),
            (r.warning_run >= 2, f"Unusual for {int(r.warning_run)} reading(s) in a row (2 needed)"),
            (r.h >= THR["critical"], f"Above the critical level of {THR['critical']:.0f}"),
        ]
        st.markdown('<div class="panel"><h4>Checks before anyone is alerted</h4>' + "".join(
            f'<div class="check"><span class="{"ok" if ok else "no"}">{"✓" if ok else "–"}</span><span>{ui.esc(t)}</span></div>'
            for ok, t in checks) + '<p class="muted" style="margin:.6rem 0 0;font-size:.9rem">One detector alone '
                                   'never raises an alert. That is what keeps false alarms at zero on the test week.</p></div>',
                    unsafe_allow_html=True)
        hot = [f for f in feeds_live if r[f"z_{f}"] >= 1.5]
        if hot:
            st.markdown("#### What would bring it back to calm")
            for f in hot:
                target = r[f + "_typical"] + r[f + "_sigma"]
                st.markdown(f"- {cc.NAMES[f]} falling from **{fmt_feed(f, r[f])}** to about **{fmt_feed(f, target)}**")
        lf = {f: float(r[f"lstm_err_{f}"]) for f in cc.FEATURES}
        if r.lstm_score >= 0.3 and sum(lf.values()) > 0:
            top_f = max(lf, key=lf.get)
            st.markdown(f"The LSTM, which learned how each neighbourhood normally changes over an hour, was most "
                        f"surprised by **{cc.NAMES[top_f].lower()}** "
                        f"({100 * lf[top_f] / sum(lf.values()):.0f}% of its reconstruction error).")

# =============================================================================== 5. what if
with tabs[3]:
    st.markdown('<div class="big-note">Move the readings and watch the detectors respond</div>',
                unsafe_allow_html=True)
    wz2 = st.selectbox("Neighbourhood", ZONES.zone_id, format_func=ZNAME.get,
                       index=list(ZONES.zone_id).index(my_zone), key="wi_zone")
    base = D[(D.zone_id == wz2) & (D.timestamp == NOW)].iloc[0]
    prev = D[(D.zone_id == wz2) & (D.timestamp <= NOW - pd.Timedelta(minutes=15))]
    prev = prev.iloc[-1] if len(prev) else base
    c = st.columns(4)
    ranges = {"rainfall": (0.0, 60.0, .5), "traffic_congestion": (0.0, 100.0, 1.0),
              "incidents_30m": (0.0, 20.0, 1.0), "aqi": (0.0, 400.0, 5.0)}
    new = {}
    for i, f in enumerate(cc.FEATURES):
        lo, hi, step = ranges[f]
        new[f] = c[i].slider(f"{cc.NAMES[f]} ({cc.UNITS[f]})", lo, hi, float(np.clip(round(base[f] / step) * step, lo, hi)),
                             step, key=f"wi_{f}_{wz2}")
        c[i].caption(f"usual at {NOW:%H:%M}: {fmt_feed(f, base[f + '_typical'])}")
    row = base.copy()
    for f in cc.FEATURES:
        row[f] = new[f]
        row[f"z_{f}"] = (new[f] - base[f + "_typical"]) / base[f + "_sigma"]
        row[f"dz_{f}"] = row[f"z_{f}"] - prev[f"z_{f}"]
    rr = rescore(pd.DataFrame([row])).iloc[0]
    inst_state = ("Critical" if rr.hybrid >= THR["critical"] and rr.strong >= 2 else
                  "Confirmed" if rr.hybrid >= THR["confirmed"] and rr.strong >= 2 else
                  "Watch" if rr.hybrid >= THR["watch"] else "Normal")
    pulse_new = int(cc.pulse_from_hybrid([rr.hybrid], THR)[0])
    k = st.columns(4)
    k[0].metric("Pulse", pulse_new, int(pulse_new - base.pulse))
    k[1].metric("Would show as", ui.STATE_WORD[inst_state])
    k[2].metric("Detectors agreeing", f"{int(rr.strong)} of 3")
    k[3].metric("Combined score", f"{rr.hybrid:.0f}", f"{rr.hybrid - base.hybrid:+.0f}", delta_color="inverse")
    fig = go.Figure()
    for nm, key_, col in [("Statistics", "stat_score", ui.MARIGOLD), ("Isolation Forest", "iso_score", "#7CB7FF"),
                          ("LSTM", "lstm_score", ui.ROSE)]:
        fig.add_trace(go.Bar(name=nm, x=["now (actual)", "your scenario"], y=[base[key_], rr[key_]], marker_color=col))
    fig.add_hline(y=cc.STRONG, line=dict(color=ui.MUTED, dash="dot"), annotation_text="counts as strong")
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.05], title="evidence (0 to 1)"),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(ui.plotly_theme(fig, 300), width="stretch", config=PLOT_CFG)
    st.caption("Statistics and the Isolation Forest are recomputed live with the same models the pipeline used. "
               "The LSTM needs the last hour of readings, so it keeps its actual value here; a real alert also needs "
               "the change to last for two readings.")

# =============================================================================== 6. report
CATS = {"Accident or crash": 2, "Waterlogging or flooded road": 2, "Traffic signal not working": 1,
        "Smoke, burning or bad air": 1, "Fallen tree or blocked road": 2, "Something else": 1}
PII = re.compile(r"(\b\d{10}\b|\+91[\s-]?\d{5}[\s-]?\d{5}|[\w.+-]+@[\w-]+\.[\w.]+)")

with tabs[4]:
    st.markdown('<div class="big-note">Seen something? Tell the city.</div>', unsafe_allow_html=True)
    st.markdown("Your report is added to the road-incident feed for that neighbourhood at the current replay time, "
                "and all detectors re-score it. You can watch it move the pulse.")
    with st.form("report", clear_on_submit=True):
        c1, c2, c3 = st.columns([1, 1.3, .7])
        rz = c1.selectbox("Where", ZONES.zone_id, format_func=ZNAME.get, index=list(ZONES.zone_id).index(my_zone))
        rc = c2.selectbox("What", list(CATS))
        rn = c3.number_input("How many", 1, 5, 1)
        note = st.text_input("Details (optional)", placeholder="e.g. water on the road near the market",
                             max_chars=160)
        sent = st.form_submit_button("Send report")
    if sent:
        before = int(D[(D.zone_id == rz) & (D.timestamp == NOW)].pulse.iloc[0])
        ss.reports.append(dict(zone=rz, time=NOW, n=int(rn * CATS[rc]), category=rc,
                               note=PII.sub("[removed]", note or "")))
        D2 = with_reports(reports_key())
        after = int(D2[(D2.zone_id == rz) & (D2.timestamp == NOW)].pulse.iloc[0])
        ss.report_msg = f"Report added. {ZNAME[rz]} pulse at {NOW:%H:%M}: {before} before, {after} after."
        st.rerun()
    if ss.get("report_msg"):
        st.success(ss.pop("report_msg"))
    if ss.reports:
        st.markdown("#### Reports in this session")
        for i, rp in enumerate(ss.reports):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f'<div class="tick"><b>{ZNAME[rp["zone"]]}</b>, {rp["time"]:%a %H:%M}: {ui.esc(rp["category"])}'
                        f'{": " + ui.esc(rp["note"]) if rp["note"] else ""}</div>', unsafe_allow_html=True)
            if c2.button("Remove", key=f"rm_{i}"):
                ss.reports.pop(i)
                st.rerun()
    st.caption("Privacy: no names or accounts. Phone numbers and email addresses are removed before a report is "
               "stored, and reports live only in this browser session.")

# =============================================================================== 7. ask
def fallback_answer(q: str) -> str:
    ql = q.lower()
    zs = [z for z in ZONES.zone_id if ZNAME[z].lower() in ql or ZNAME[z].split()[0].lower() in ql]
    if any(w in ql for w in ["event", "happened", "yesterday", "week", "last night", "history"]):
        evs = [e for e in EVENTS if e["start"] <= NOW and (not zs or set(zs) & set(e["zone_ids"]))]
        if not evs:
            return "No events have been confirmed there up to this replay time."
        return "Confirmed so far:\n" + "\n".join(
            f"- {e['start']:%a %d %b %H:%M}: {e['pattern'].lower()} in {join_names(e['zone_names'])} "
            f"({ui.STATE_WORD[e['peak_state']].lower()})" for e in evs[-6:])
    if any(w in ql for w in ["how", "work", "model", "lstm", "isolation", "accurate", "accuracy"]):
        m = META["metrics"]["test"]["events"]
        return (f"Three detectors look at every reading: a statistics check against what is typical for that "
                f"neighbourhood and hour, an Isolation Forest, and an LSTM that learned normal hour-long patterns. "
                f"An alert needs two of them to agree. On the unseen test period it caught {m['detected']} of "
                f"{m['true_events']} simulated events with {m['false_alarm_episodes']} false alarms.")
    target = zs or [snap.h.idxmax()]
    out = []
    for z in target:
        r = snap.loc[z]
        out.append(f"{ZNAME[z]} is {ui.STATE_WORD[r.st_v].lower()} (pulse {int(r.pulse_v)}) at {NOW:%H:%M}. "
                   f"{zone_reason(r, 'en', 3)}.")
    if any(STATE_RANK[snap.loc[z].st_v] >= 2 for z in target):
        out.append("If you are heading there, allow extra time and follow official Jaipur advisories.")
    return " ".join(out)


with tabs[5]:
    st.markdown('<div class="big-note">Ask about any neighbourhood, event or the method</div>', unsafe_allow_html=True)
    sugg = ["Is it a good time to drive through Sindhi Camp?", "What happened this week?",
            "Why is the top neighbourhood flagged?", "How does CityPulse decide?"]
    cols = st.columns(4)
    for i, s in enumerate(sugg):
        if cols[i].button(s, key=f"sg_{i}"):
            ss.pending_q = s
    for m in ss.chat[-12:]:
        with st.chat_message(m["role"], avatar="💗" if m["role"] == "assistant" else None):
            st.markdown(m["content"])
    q = st.chat_input("Ask CityPulse") or ss.pop("pending_q", None)
    if q:
        ss.chat.append(dict(role="user", content=q))
        facts = dict(now=snapshot_facts(snap),
                     events_so_far=[dict(when=f"{e['start']:%a %d %b %H:%M}-{e['end']:%H:%M}",
                                         where=e["zone_names"], what=e["pattern"],
                                         level=ui.STATE_WORD[e["peak_state"]], severity=e["severity"],
                                         evidence=event_evidence(e)) for e in EVENTS if e["start"] <= NOW],
                     method=dict(detectors=["statistics vs typical for zone and hour", "Isolation Forest",
                                            "LSTM autoencoder"], alert_rule="2 of 3 detectors agree and it lasts "
                                                                            "2 readings",
                                 test_events_caught=META["metrics"]["test"]["events"]["detected"],
                                 test_events_total=META["metrics"]["test"]["events"]["true_events"],
                                 test_false_alarms=META["metrics"]["test"]["events"]["false_alarm_episodes"]))
        txt, status = ai_text("ask", f"Question from a resident: {q}\nAnswer in at most 4 sentences"
                                     f"{' in Hindi' if lang == 'hi' else ''}.", facts, 350)
        ans = txt or fallback_answer(q)
        if not txt and use_ai and LLM and status.startswith("ungrounded"):
            ans += "\n\n_The AI answer quoted figures that are not in the data, so this answer comes from the data directly._"
        ss.chat.append(dict(role="assistant", content=ans))
        st.rerun()
    if ss.chat and st.button("Clear conversation"):
        ss.chat = []
        st.rerun()

# =============================================================================== 8. alerts
with tabs[6]:
    st.markdown('<div class="big-note">Get told before you get stuck</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    subs = c1.multiselect("Neighbourhoods to follow", ZONES.zone_id, default=[my_zone], format_func=ZNAME.get)
    lvl = c2.selectbox("Alert me from", ["Watch", "Confirmed", "Critical"], index=1, key="alert_level",
                       format_func=ui.STATE_WORD.get)
    log_rows = []
    for zid in subs:
        zz = V[V.zone_id == zid].sort_values("timestamp")
        on = zz.st_v.map(STATE_RANK) >= STATE_RANK[lvl]
        starts = zz[on & ~on.shift(fill_value=False)]
        for _, r in starts.iterrows():
            log_rows.append(dict(time=r.timestamp, zone=ZNAME[zid], level=ui.STATE_WORD[r.st_v],
                                 message=f"{ZNAME[zid]} is {ui.STATE_WORD[r.st_v].lower()}: {zone_reason(r, 'en', 2)}"))
    log = pd.DataFrame(log_rows)
    if log.empty:
        st.markdown('<div class="panel">No alerts at this level for these neighbourhoods this week.</div>',
                    unsafe_allow_html=True)
    else:
        log = log.sort_values("time")
        sent_ = log[log.time <= NOW]
        st.markdown(f"**{len(sent_)} alert(s) sent by {NOW:%a %H:%M}**, {len(log) - len(sent_)} more later in the week.")
        for _, a in sent_.iloc[::-1].iterrows():
            st.markdown(f'<div class="tick"><b>{a.time:%a %d %b, %H:%M}</b>&nbsp; {ui.esc(a.message)}</div>',
                        unsafe_allow_html=True)
        st.download_button("Download alert log (CSV)", log.to_csv(index=False), "citypulse_alerts.csv")
    st.caption("While the replay plays, a notification pops up whenever your neighbourhood reaches the level you "
               "chose. In production this would be an SMS, WhatsApp or push message.")

# =============================================================================== 9. how it works
with tabs[7]:
    mt = META["metrics"]["test"]
    ev = mt["events"]
    st.markdown('<div class="flow"><span>4 live feeds</span><i>→</i><span>one table, every 5 min, per neighbourhood</span>'
                '<i>→</i><span>typical for this place and hour</span><i>→</i><span>statistics</span><span>Isolation Forest</span>'
                '<span>LSTM autoencoder</span><i>→</i><span>2 of 3 must agree, and it must last</span><i>→</i>'
                '<span>events, pulse and plain-language summary</span></div>', unsafe_allow_html=True)
    lag = ev["mean_minutes_to_confirm"]
    st.markdown(f'<div class="big-note">On a test period the models never saw, CityPulse caught '
                f'{ev["detected"]} of {ev["true_events"]} simulated disruptions, raised '
                f'{ev["false_alarm_episodes"]} false alarm{"s" if ev["false_alarm_episodes"] != 1 else ""}, '
                f'and confirmed each one on average {lag:.0f} minutes after it began.</div>', unsafe_allow_html=True)
    k = st.columns(4)
    k[0].metric("Events caught (test)", f"{ev['detected']} / {ev['true_events']}")
    k[1].metric("False alarms per day", f"{ev['false_alarms_per_day']:.1f}")
    k[2].metric("Decoys ignored", f"{ev['decoys_total'] - len(ev['decoys_fired'])} / {ev['decoys_total']}")
    k[3].metric("PR-AUC (rows)", f"{mt['rows'].get('pr_auc_hybrid', 0):.2f}")
    cA = cB = st.container()
    ab = mt["ablation"]
    names = {"stat": "Statistics alone", "iso": "Isolation Forest alone", "lstm": "LSTM alone",
             "hybrid": "All three, 2 must agree"}
    fig = go.Figure(go.Bar(x=[ab[k]["false_alarms_per_day"] for k in names], y=list(names.values()), orientation="h",
                           marker_color=[ui.MARIGOLD, "#7CB7FF", ui.ROSE, ui.STATE["Normal"]],
                           text=[f"{ab[k]['false_alarms_per_day']:.1f} per day, catches {ab[k]['event_recall']:.0%} of events"
                                 for k in names], textposition="outside", cliponaxis=False))
    xmax = max(ab[k]["false_alarms_per_day"] for k in names)
    fig.update_layout(title="False alarms per day on the unseen test period", yaxis=dict(autorange="reversed"),
                      xaxis=dict(range=[0, xmax * 1.6 + 0.5]))
    cA.plotly_chart(ui.plotly_theme(fig, 260), width="stretch", config=PLOT_CFG)
    cA.caption("Each detector alone would either miss events or cry wolf. Requiring two of three to agree removes "
               "the false alarms without losing a single event.")
    rows = [dict(scenario=p["label"].split("_", 1)[1].replace("_", " "), where=", ".join(ZNAME[z] for z in p["zones"]),
                 began=pd.Timestamp(p["start"]).strftime("%a %H:%M"), caught="yes" if p["detected"] else "no",
                 **{"warned after (min)": p["minutes_to_first_warning"], "confirmed after (min)": p["minutes_to_confirm"]})
            for p in ev["per_event"]]
    cB.markdown("#### Every test scenario")
    cB.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(f"""
#### Method
- **Fusion.** Rainfall, temperature and AQI (hourly) and traffic and incidents (5-minute) are aligned into one table
  per neighbourhood every 5 minutes. Incidents become a rolling 30-minute count.
- **Typical, per place and hour.** For each neighbourhood and hour of day, the median and a robust spread
  (1.4826 × MAD) are learned from the training days only. Every reading becomes "how many usual spreads above normal".
- **Three detectors.** A statistics check (rain on its own is down-weighted, because rain is weather, not a civic
  problem); an Isolation Forest on the deviations and their 15-minute change; and an LSTM autoencoder
  ({META['lstm']['architecture']}) trained on clean hour-long windows, scored on the most recent 15 minutes.
- **Calibration without peeking.** Both models are calibrated on normal validation rows only. Fusion weights
  (statistics {W['stat']:.2f}, Isolation Forest {W['iso']:.2f}, LSTM {W['lstm']:.2f}) and the thresholds
  (watch {THR['watch']}, disrupted {THR['confirmed']}, critical {THR['critical']}) were chosen on the validation
  period, frozen, then run once on the test period.
- **Decision.** Disrupted needs the combined score above threshold, at least 2 of 3 detectors strong, and two
  readings in a row. Nearby neighbourhoods disrupted at the same time (within {cc.MERGE_KM:.0f} km) merge into one event.
- **Graceful degradation.** Every reading was also re-scored by all three detectors with each feed removed, so the
  outage switch in the sidebar shows real results.
- **Honesty.** {META['note']}
""")
    with st.expander("Validation period results and full model card"):
        st.json({"thresholds": THR, "weights": W, "calibration": CAL, "split": META["split"],
                 "validation": META["metrics"]["val"], "lstm": META["lstm"]})
    st.download_button("Download model card (JSON)", json.dumps(
        {k: META[k] for k in ["version", "data_source", "weights", "thresholds", "calibration", "split", "metrics",
                              "scenarios", "note"]}, indent=2, default=str), "citypulse_model_card.json")

