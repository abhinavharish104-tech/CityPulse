"""Visual layer for CityPulse: theme CSS and HTML components (vital-sign monitor rows, hero, chips)."""
from __future__ import annotations

import html
import time

import numpy as np

# ---------------------------------------------------------------- tokens (Jaipur at dusk)
INK = "#15122E"        # page
PANEL = "#1E1A3D"      # surfaces
RAISED = "#28234F"
LINE = "#3A3466"
TEXT = "#EDE8F7"
MUTED = "#A69FC7"
MARIGOLD = "#F2B544"   # interactive accent
ROSE = "#E99BBE"       # Pink City brand
STATE = {"Normal": "#5FD3B3", "Watch": "#F2C14E", "Confirmed": "#F2884B", "Critical": "#FF5A6A"}
STATE_WORD = {"Normal": "Calm", "Watch": "Watch", "Confirmed": "Disrupted", "Critical": "Critical"}
STATE_ICON = {"Normal": "●", "Watch": "▲", "Confirmed": "◆", "Critical": "✚"}

JAALI = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='44' height='44' viewBox='0 0 44 44'>"
         "<g fill='none' stroke='%23E99BBE' stroke-opacity='0.10' stroke-width='1.2'>"
         "<path d='M22 2 L42 22 L22 42 L2 22 Z'/><circle cx='22' cy='22' r='7'/>"
         "<path d='M22 15 Q27 22 22 29 Q17 22 22 15'/></g></svg>")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');

:root {{
  --ink:{INK}; --panel:{PANEL}; --raised:{RAISED}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED};
  --marigold:{MARIGOLD}; --rose:{ROSE};
  --calm:{STATE['Normal']}; --watch:{STATE['Watch']}; --disrupted:{STATE['Confirmed']}; --critical:{STATE['Critical']};
  --display:'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif;
  --body:'Atkinson Hyperlegible', 'Segoe UI', system-ui, sans-serif;
}}
html, body, [data-testid="stAppViewContainer"], .stMarkdown, .stText, p, li, label, input, textarea, button, select {{
  font-family: var(--body) !important;
}}
[data-testid="stAppViewContainer"] {{ background: radial-gradient(1200px 500px at 85% -10%, #2A2158 0%, var(--ink) 60%) fixed; }}
[data-testid="stHeader"] {{ background: transparent; }}
/* live board re-runs every tick: never fade elements while they refresh */
[data-testid="stElementContainer"][data-stale="true"], [data-stale="true"] {{ opacity: 1 !important; filter: none !important; transition: none !important; }}
[data-testid="stStatusWidget"] {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; max-width: 1380px; }}
h1, h2, h3, h4 {{ font-family: var(--display) !important; letter-spacing: -0.01em; color: var(--text); }}
h3 {{ font-weight: 700 !important; }}
h4 {{ font-size: 1.12rem !important; font-weight: 700 !important; margin-top: .4rem !important; }}
a {{ color: var(--rose); }}

/* sidebar */
[data-testid="stSidebar"] {{ background: #110E26; border-right: 1px solid var(--line); }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ font-size: 1.02rem; color: var(--muted); font-weight: 600 !important; margin-top: .6rem; }}
.brand {{ display:flex; align-items:center; gap:.6rem; margin: .2rem 0 1rem; }}
.brand svg {{ flex: none; }}
.brand b {{ font-family: var(--display); font-size: 1.45rem; color: var(--text); letter-spacing: -0.02em; }}
.brand span {{ display:block; color: var(--muted); font-size: .82rem; line-height: 1.25; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ font-family: var(--display); font-size: 1rem; color: var(--muted); padding: .5rem .85rem; }}
.stTabs [aria-selected="true"] {{ color: var(--text) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--marigold); height: 3px; border-radius: 3px; }}

/* metrics */
[data-testid="stMetric"] {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: .8rem 1rem; }}
[data-testid="stMetricValue"] {{ font-family: var(--display) !important; font-weight: 700; }}
[data-testid="stMetricLabel"] p {{ color: var(--muted) !important; }}

/* buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: 999px; border: 1px solid var(--line); background: var(--raised); color: var(--text);
  font-weight: 700; padding: .35rem 1rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: var(--marigold); color: var(--marigold); }}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button {{ background: var(--marigold); color: #241a00; border-color: var(--marigold); }}
:focus-visible {{ outline: 2px solid var(--marigold) !important; outline-offset: 2px; }}
[data-testid="stExpander"] {{ border: 1px solid var(--line); border-radius: 14px; background: var(--panel); }}

.stale-element {{ opacity: 1 !important; transition: none !important; }}

/* playback tape (replaces the slider while the week plays) */
.tape {{ display:flex; align-items:center; gap:1rem; margin: .1rem 0 .9rem; }}
.tape .rail {{ position: relative; flex: 1; height: 10px; border-radius: 999px; background: var(--raised); border: 1px solid var(--line); }}
.tape .fill {{ position:absolute; left:0; top:0; bottom:0; border-radius: 999px; background: var(--marigold); transition: width .9s linear; }}
.tape .head {{ position:absolute; top:50%; width:16px; height:16px; margin:-8px 0 0 -8px; border-radius:50%; background: var(--marigold);
  box-shadow: 0 0 0 4px rgba(242,181,68,.25); transition: left .9s linear; }}
.tape .mark {{ position:absolute; top:-5px; width:4px; height:20px; margin-left:-2px; border-radius:2px; opacity:.9; }}
.tape .clock {{ font-family: var(--display); font-weight: 800; font-size: 1.15rem; min-width: 12.5rem; text-align:right; font-variant-numeric: tabular-nums; }}
.tape .live {{ color: var(--critical); font-weight:700; font-size:.85rem; }}
.tape .live::before {{ content:"●"; margin-right:.3rem; animation: blink 1s steps(2) infinite; }}
@keyframes blink {{ 50% {{ opacity: .2; }} }}
@media (prefers-reduced-motion: reduce) {{ .tape .fill, .tape .head {{ transition: none; }} .tape .live::before {{ animation: none; }} }}

/* hero */
.hero {{ position: relative; overflow: hidden; border-radius: 22px; padding: 1.6rem 1.8rem 1.4rem;
  background: linear-gradient(115deg, rgba(30,26,61,.96) 0%, rgba(30,26,61,.82) 55%, rgba(40,35,79,.70) 100%), url("{JAALI}");
  border: 1px solid var(--line); display: grid; grid-template-columns: minmax(0,1fr) 220px; gap: 1.2rem; align-items: end; }}
.hero .bar {{ position:absolute; left:0; top:0; bottom:0; width:6px; background: var(--accent); }}
.hero h1 {{ font-size: clamp(1.7rem, 3.2vw, 2.7rem); line-height: 1.08; margin: 0 0 .55rem; font-weight: 800 !important; max-width: 22ch; }}
.hero p.sub {{ font-size: 1.1rem; color: #D8D1EE; margin: 0; max-width: 68ch; line-height: 1.5; }}
.hero .when {{ color: var(--muted); font-size: .92rem; margin-top: .9rem; }}
.hero .ai {{ display:inline-block; margin-left:.4rem; font-size:.8rem; color: var(--rose); border:1px solid rgba(233,155,190,.45); border-radius:999px; padding:0 .5rem; }}
.citypulse {{ text-align: right; }}
.citypulse .num {{ font-family: var(--display); font-size: 4.2rem; line-height: .9; font-weight: 800; color: var(--accent); font-variant-numeric: tabular-nums; }}
.citypulse .lbl {{ color: var(--muted); font-size: .92rem; }}
.citypulse .conf {{ color: var(--text); font-size: .92rem; margin-top: .35rem; }}
@media (max-width: 760px) {{ .hero {{ grid-template-columns: 1fr; }} .citypulse {{ text-align:left; }} }}

.source {{ color: var(--muted); font-size: .86rem; margin: .55rem 0 1.1rem .2rem; }}
.source b {{ color: var(--text); font-weight: 700; }}

/* vital monitor */
.monitor {{ background: #0F0C24; border: 1px solid var(--line); border-radius: 18px; overflow: hidden; }}
.vrow {{ display: grid; grid-template-columns: 180px minmax(0,1fr) 84px 118px; align-items: center; gap: 1rem;
  padding: .7rem 1.1rem; border-bottom: 1px solid rgba(58,52,102,.6); }}
.vrow:last-child {{ border-bottom: none; }}
.vrow.mine {{ background: rgba(242,181,68,.06); }}
.vrow .zn {{ font-family: var(--display); font-size: 1.15rem; font-weight: 700; color: var(--text); line-height: 1.15; }}
.vrow .why {{ color: var(--muted); font-size: .86rem; line-height: 1.3; margin-top: .15rem; }}
.vrow .pn {{ font-family: var(--display); font-size: 2.3rem; font-weight: 800; text-align: right; font-variant-numeric: tabular-nums; }}
.vrow .pn small {{ display:block; font-family: var(--body); font-size: .72rem; font-weight: 400; color: var(--muted); margin-top: -.2rem; }}
.chip {{ display: inline-flex; align-items: center; gap: .35rem; border-radius: 999px; padding: .2rem .7rem;
  font-weight: 700; font-size: .9rem; border: 1px solid currentColor; white-space: nowrap; }}
.ecg {{ height: 54px; overflow: hidden; position: relative;
  -webkit-mask-image: linear-gradient(90deg, transparent 0%, #000 18%, #000 100%); mask-image: linear-gradient(90deg, transparent 0%, #000 18%, #000 100%); }}
.ecg svg {{ position: absolute; left: 0; top: 0; height: 54px; animation: sweep var(--dur) linear infinite; animation-delay: var(--delay, 0s); }}
.ecg .grid {{ position:absolute; inset:0; background-image: linear-gradient(rgba(95,211,179,.06) 1px, transparent 1px), linear-gradient(90deg, rgba(95,211,179,.06) 1px, transparent 1px); background-size: 12px 12px; }}
@keyframes sweep {{ from {{ transform: translateX(0); }} to {{ transform: translateX(calc(-1 * var(--period))); }} }}
.flat svg {{ animation: none; }}
@media (max-width: 760px) {{ .vrow {{ grid-template-columns: 1fr 80px; }} .vrow .ecg {{ grid-column: 1 / -1; order: 3; }} .vrow .st {{ display:none; }} }}
@media (prefers-reduced-motion: reduce) {{ .ecg svg {{ animation: none; }} }}
/* panels & cards */
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 1.1rem 1.2rem; }}
.panel h4 {{ margin: 0 0 .5rem; font-size: 1.1rem; }}
.muted {{ color: var(--muted); }}
.event {{ background: var(--panel); border: 1px solid var(--line); border-left: 5px solid var(--accent); border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: .2rem; }}
.event .top {{ display:flex; justify-content: space-between; gap: 1rem; align-items: baseline; flex-wrap: wrap; }}
.event h3 {{ margin: 0; font-size: 1.35rem; }}
.event .meta {{ color: var(--muted); font-size: .92rem; margin-top: .2rem; }}
.event ul {{ margin: .6rem 0 .3rem 1.1rem; padding: 0; }}
.event li {{ margin: .15rem 0; }}
.sev {{ font-family: var(--display); font-weight: 800; font-size: 2rem; color: var(--accent); line-height: 1; text-align:right; }}
.sev small {{ display:block; font-family: var(--body); font-size: .75rem; color: var(--muted); font-weight: 400; }}
.caveat {{ font-size: .88rem; color: #CFC6EA; background: rgba(233,155,190,.08); border: 1px dashed rgba(233,155,190,.45); border-radius: 10px; padding: .45rem .7rem; margin-top: .5rem; }}
.check {{ display:flex; gap:.55rem; align-items:flex-start; margin:.3rem 0; }}
.check .ok {{ color: var(--calm); font-weight: 700; }} .check .no {{ color: var(--muted); font-weight: 700; }}
.tick {{ background: #0F0C24; border: 1px solid var(--line); border-radius: 12px; padding: .5rem .8rem; margin: .3rem 0; }}
.tick b {{ font-family: var(--display); }}
.flow {{ display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; margin:.4rem 0 1rem; }}
.flow span {{ background: var(--raised); border:1px solid var(--line); border-radius: 10px; padding: .35rem .7rem; font-size: .92rem; }}
.flow i {{ color: var(--muted); font-style: normal; }}
.big-note {{ font-family: var(--display); font-size: 1.5rem; line-height: 1.25; font-weight: 700; margin: .2rem 0 .8rem; max-width: 40ch; }}
</style>
"""

BRAND = f"""
<div class="brand">
  <svg width="38" height="38" viewBox="0 0 38 38" aria-hidden="true">
    <rect x="1" y="1" width="36" height="36" rx="10" fill="{RAISED}" stroke="{LINE}"/>
    <path d="M5 21 H12 L15 13 L19 27 L23 9 L26 21 H33" fill="none" stroke="{ROSE}" stroke-width="2.4"
      stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <div><b>CityPulse</b><span>Jaipur's neighbourhoods,<br>one heartbeat each</span></div>
</div>"""


def esc(s) -> str:
    return html.escape(str(s))


# ---------------------------------------------------------------- ECG trace
def _beat(x0: float, w: float, amp: float, rng) -> list[tuple[float, float]]:
    """One PQRST complex inside [x0, x0+w]. y=0 is baseline, positive is up."""
    j = lambda s: rng.normal(0, s)
    pts = [(0.00, 0), (0.10, 0.08 + j(.01)), (0.16, 0.0), (0.30, 0), (0.34, -0.12 * amp),
           (0.38, 1.0 * amp + j(.03)), (0.43, -0.32 * amp), (0.48, 0), (0.62, 0.18 + 0.05 * amp),
           (0.72, 0), (1.00, 0)]
    return [(x0 + a * w, b) for a, b in pts]


def ecg_svg(pulse: float, state: str, seed: int = 0, width: int = 520) -> tuple[str, float, float]:
    """Scrolling ECG trace. Calmer zones beat slower and smaller; stressed zones faster, taller,
    and (when critical) irregular. Returns (svg, period_px, duration_s)."""
    rng = np.random.default_rng(seed)
    stress = float(np.clip((100 - pulse) / 100, 0, 1))
    beat_w = 130 - 70 * stress                    # px per beat
    amp = 0.55 + 0.45 * stress
    n = int(np.ceil(width / beat_w)) + 2
    pts = []
    for k in range(n * 2):                        # two copies -> seamless loop
        a = amp * (1 + (rng.normal(0, .18) if state == "Critical" else 0))
        pts += _beat(k * beat_w, beat_w, a, rng)
    h, mid = 54, 34
    d = "M" + " L".join(f"{x:.1f},{mid - y * 26:.1f}" for x, y in pts)
    col = STATE.get(state, STATE["Normal"])
    period = n * beat_w
    dur = period / (120 + 260 * stress)          # px per second
    svg = (f'<svg width="{period * 2:.0f}" height="{h}" viewBox="0 0 {period * 2:.0f} {h}" aria-hidden="true">'
           f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.2" stroke-linejoin="round" '
           f'style="filter: drop-shadow(0 0 4px {col}88)"/></svg>')
    return svg, period, dur


def chip(state: str, word: str | None = None) -> str:
    c = STATE.get(state, MUTED)
    return f'<span class="chip" style="color:{c}">{STATE_ICON.get(state, "●")} {esc(word or STATE_WORD.get(state, state))}</span>'


def monitor(rows: list[dict]) -> str:
    """rows: dicts with zone_name, pulse, state, state_word, why, mine, seed"""
    out = ['<div class="monitor" role="list" aria-label="Neighbourhood health">']
    for r in rows:
        svg, period, dur = ecg_svg(r["pulse"], r["state"], r["seed"])
        col = STATE.get(r["state"], MUTED)
        out.append(
            f'<div class="vrow{" mine" if r.get("mine") else ""}" role="listitem">'
            f'<div><div class="zn">{esc(r["zone_name"])}{" ★" if r.get("mine") else ""}</div>'
            f'<div class="why">{esc(r["why"])}</div></div>'
            f'<div class="ecg" style="--period:{period:.1f}px; --dur:{dur:.2f}s; --delay:{-(time.time() % dur):.2f}s">'
            f'<div class="grid"></div>{svg}</div>'
            f'<div class="pn" style="color:{col}">{int(r["pulse"])}<small>pulse</small></div>'
            f'<div class="st">{chip(r["state"], r.get("state_word"))}</div></div>')
    out.append("</div>")
    return "".join(out)


def hero(headline: str, sub: str, when: str, pulse: int, state: str, confidence: str, ai: bool = False) -> str:
    col = STATE.get(state, STATE["Normal"])
    tag = '<span class="ai">written by AI, checked against the data</span>' if ai else ""
    return (f'<div class="hero" style="--accent:{col}"><div class="glow"></div>{HAWA_MAHAL}<div class="bar"></div>'
            f'<div><h1>{esc(headline)}</h1><p class="sub">{esc(sub)}{tag}</p><div class="when">{esc(when)}</div></div>'
            f'<div class="citypulse"><div class="num">{pulse}</div><div class="lbl">average pulse across the city</div>'
            f'<div class="conf">{esc(confidence)}</div></div></div>')


def plotly_theme(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Atkinson Hyperlegible, Segoe UI, sans-serif", color=TEXT, size=13),
        margin=dict(l=8, r=8, t=36, b=8), hoverlabel=dict(bgcolor=RAISED, bordercolor=LINE,
                                                          font=dict(color=TEXT)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(166,159,199,.12)", zerolinecolor="rgba(166,159,199,.2)", linecolor=LINE)
    fig.update_yaxes(gridcolor="rgba(166,159,199,.12)", zerolinecolor="rgba(166,159,199,.2)", linecolor=LINE)
    if height:
        fig.update_layout(height=height)
    return fig


def tape(frac: float, clock: str, marks: list[tuple[float, str]]) -> str:
    """Moving playback bar shown while the week plays."""
    pct = f"{100 * max(0.0, min(1.0, frac)):.2f}%"
    mk = "".join(f'<span class="mark" style="left:{100 * f:.2f}%; background:{c}"></span>' for f, c in marks)
    return (f'<div class="tape"><span class="live">Playing</span><div class="rail" role="progressbar" '
            f'aria-valuenow="{pct}" aria-label="Replay position">{mk}<div class="fill" style="width:{pct}"></div>'
            f'<div class="head" style="left:{pct}"></div></div><div class="clock">{esc(clock)}</div></div>')


# ---------------------------------------------------------------- Jaipur motifs (original line art)
def _hawa_mahal_paths() -> str:
    """Stylised Hawa Mahal facade: five stepped tiers of arched jharokha windows topped with chhatri domes."""
    cx, base = 180.0, 312.0
    tiers = [(344, 54), (300, 50), (238, 47), (168, 45), (98, 43)]
    out, y = [], base
    out.append(f'<path d="M{cx - 176} {base + 4} H{cx + 176}"/>')                         # plinth
    for t, (w, h) in enumerate(tiers):
        top, x0, x1 = y - h, cx - w / 2, cx + w / 2
        out.append(f'<rect x="{x0:.1f}" y="{top:.1f}" width="{w}" height="{h}" rx="2"/>')
        n = max(3, int(w / 24))
        nxt = tiers[t + 1][0] if t + 1 < len(tiers) else 0
        for k in range(n):
            wx = x0 + (k + 0.5) * w / n
            r, wy0, wy1 = 5.0, top + 12, y - 8
            out.append(f'<path d="M{wx - r:.1f} {wy1:.1f} V{wy0 + r:.1f} A{r} {r} 0 0 1 {wx + r:.1f} {wy0 + r:.1f} V{wy1:.1f}"/>')
            out.append(f'<path d="M{wx - r - 2:.1f} {wy1:.1f} H{wx + r + 2:.1f}"/>')      # balcony ledge
            if abs(wx - cx) > nxt / 2 - 2:                                                  # chhatri on exposed roof
                out.append(f'<path d="M{wx - 7:.1f} {top:.1f} A7 9 0 0 1 {wx + 7:.1f} {top:.1f}"/>')
                out.append(f'<path d="M{wx:.1f} {top - 9:.1f} V{top - 14:.1f}"/>')
        y = top
    out.append(f'<path d="M{cx - 16} {y:.1f} A16 20 0 0 1 {cx + 16} {y:.1f}"/><path d="M{cx} {y - 20:.1f} V{y - 30:.1f}"/>')
    out.append(f'<path d="M{cx - 14} {base} V{base - 26} A14 14 0 0 1 {cx + 14} {base - 26} V{base}"/>')  # gateway
    return "".join(out)


HAWA_MAHAL = (f'<svg class="hawa" viewBox="0 0 360 324" aria-hidden="true" preserveAspectRatio="xMidYMax meet">'
              f'<g fill="none" stroke="{ROSE}" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round">'
              f'{_hawa_mahal_paths()}</g></svg>')


def _skyline_tile() -> str:
    """Repeating city-wall horizon: crenellated wall, chhatri pavilions and a domed gate."""
    p = ['<path d="M0 60 H600"/><path d="M0 40']
    for x in range(0, 600, 20):                      # merlons
        p.append(f' H{x + 6} V34 H{x + 14} V40')
    p.append(' H600"/>')
    extra = []
    for x in (70, 250, 430):                         # chhatri pavilions
        extra.append(f'<path d="M{x} 40 V26 H{x + 24} V40"/><path d="M{x - 3} 26 H{x + 27}"/>'
                     f'<path d="M{x} 26 A12 11 0 0 1 {x + 24} 26"/><path d="M{x + 12} 15 V10"/>')
    extra.append('<path d="M320 40 V18 H372 V40"/><path d="M324 18 A22 18 0 0 1 368 18"/>'
                 '<path d="M346 0 V-2"/><path d="M336 40 V28 A10 10 0 0 1 356 28 V40"/>')
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='600' height='64' viewBox='0 -4 600 64'>"
           f"<g fill='none' stroke='{ROSE}' stroke-width='1.2' stroke-opacity='0.55'>{''.join(p)}{''.join(extra)}</g></svg>")
    return "data:image/svg+xml;utf8," + svg.replace("#", "%23").replace('"', "'")


SKYLINE = _skyline_tile()
ARCH = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 180 210'>"
        "<path d='M8 208 V96 C8 58 40 40 62 26 C76 17 86 10 90 2 C94 10 104 17 118 26 C140 40 172 58 172 96 V208' "
        "fill='none' stroke='%23E99BBE' stroke-opacity='0.35' stroke-width='1.6'/>"
        "<path d='M22 208 V100 C22 68 48 52 68 40 C80 33 87 26 90 20 C93 26 100 33 112 40 C132 52 158 68 158 100 V208' "
        "fill='none' stroke='%23E99BBE' stroke-opacity='0.16' stroke-width='1.2'/></svg>")

JAIPUR_CSS = f"""
<style>
.hero {{ grid-template-columns: minmax(0,1fr) 300px; min-height: 290px; align-items: end; }}
.hero > div:not(.bar):not(.glow) {{ position: relative; z-index: 2; }}
.hero .hawa {{ position: absolute; z-index: 1; right: 14px; bottom: -4px; width: 300px; height: auto; opacity: .42;
  -webkit-mask-image: linear-gradient(to top, #000 60%, transparent 100%); mask-image: linear-gradient(to top, #000 60%, transparent 100%);
  pointer-events: none; }}
.hero .glow {{ position:absolute; z-index:0; right: -40px; bottom: -120px; width: 420px; height: 340px; border-radius: 50%;
  background: radial-gradient(closest-side, rgba(233,155,190,.18), transparent 72%); pointer-events:none; }}
.citypulse {{ text-align: center; align-self: end; padding: 0 0 .1rem; }}
.citypulse .num {{ font-size: 4.6rem; text-shadow: 0 0 22px {INK}, 0 0 10px {INK}; }}
.citypulse .lbl, .citypulse .conf {{ display: inline-block; background: rgba(21,18,46,.82); border-radius: 999px; padding: .05rem .6rem; margin-top: .15rem; }}
.skyline {{ height: 64px; margin: 2.2rem 0 .4rem; background: url("{SKYLINE}") repeat-x left bottom / 600px 64px; opacity: .9;
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 12%, #000 88%, transparent);
  mask-image: linear-gradient(90deg, transparent, #000 12%, #000 88%, transparent); }}
.foot {{ text-align: center; color: var(--muted); font-size: .88rem; margin-bottom: 1rem; }}
.foot b {{ font-family: var(--display); color: var(--text); }}
[data-testid="stSidebar"] > div:first-child {{ background: url("{SKYLINE}") repeat-x left bottom / 600px 64px; }}
@media (max-width: 900px) {{ .hero {{ grid-template-columns: minmax(0,1fr) 220px; }} .hero .hawa {{ width: 230px; }} }}
@media (max-width: 760px) {{ .hero {{ grid-template-columns: 1fr; }} .hero .hawa {{ width: 200px; opacity: .2; }}
  .citypulse {{ text-align: left; }} }}
</style>"""


def footer() -> str:
    return ('<div class="skyline" aria-hidden="true"></div>'
            '<div class="foot">Made in Jaipur by <b>Team Neutral Navigators</b>, for AmiHacks</div>')
