"""Plain-language interpretation of CityPulse readings.

Presentation only: nothing here changes a score, a threshold or a model. It translates the numbers the
detector already produced into what they mean for a resident.
"""
from __future__ import annotations

import math

import citypulse_core as cc

ICON = {"rainfall": "🌧️", "traffic_congestion": "🚗", "incidents_30m": "⚠️", "aqi": "🌫️"}
SHORT = {"rainfall": "Rain", "traffic_congestion": "Traffic", "incidents_30m": "Road incidents", "aqi": "Air quality"}

# Absolute categories. AQI uses the US AQI scale reported by Open-Meteo; rain uses standard intensity bands.
AQI_BANDS = [(50, "Good", "Air is clean."),
             (100, "Moderate", "Fine for most people."),
             (150, "Unhealthy for sensitive groups", "Children, older people and anyone with asthma should limit long outdoor activity."),
             (200, "Unhealthy", "Everyone may feel it. Limit time outdoors and avoid exercising near traffic."),
             (300, "Very unhealthy", "Avoid outdoor activity if you can."),
             (math.inf, "Hazardous", "Stay indoors and follow official advisories.")]
RAIN_BANDS = [(0.1, "No rain", ""), (2.5, "Light rain", "Roads may be slippery."),
              (7.6, "Moderate rain", "Expect slower traffic."),
              (50, "Heavy rain", "Waterlogging is possible on low-lying roads."),
              (math.inf, "Extreme rain", "Flooding is likely. Avoid travel if you can.")]
TRAFFIC_BANDS = [(30, "Free-flowing", ""), (60, "Moderate traffic", ""),
                 (80, "Heavy traffic", "Journeys will be slower."), (math.inf, "Gridlock likely", "Expect long delays.")]

# How unusual, from the z-score (distance from typical for this place and hour, in usual spreads)
LEVELS = [(1.0, "normal for this hour", "normal", "#5FD3B3"),
          (2.0, "a little higher than usual", "a little high", "#9ED9A0"),
          (3.0, "higher than usual", "high", "#F2C14E"),
          (5.0, "much higher than usual", "very high", "#F2884B"),
          (math.inf, "far above anything usual", "extreme", "#FF5A6A")]


def band(f: str, v: float):
    tbl = {"aqi": AQI_BANDS, "rainfall": RAIN_BANDS, "traffic_congestion": TRAFFIC_BANDS}.get(f)
    if not tbl:
        return None, ""
    for hi, name, advice in tbl:
        if v < hi:
            return name, advice
    return tbl[-1][1], tbl[-1][2]


def level(z: float):
    """(phrase, short word, colour) for how unusual a reading is."""
    for hi, phrase, word, col in LEVELS:
        if z < hi:
            return phrase, word, col
    return LEVELS[-1][1:]


def times_usual(v: float, typ: float) -> str:
    """'about 2× the usual' when it is a meaningful multiple, else ''."""
    if typ <= 0.5 or v <= typ:
        return ""
    r = v / typ
    if r < 1.3:
        return ""
    return f"about {r:.1f}× the usual" if r < 3 else f"about {r:.0f}× the usual"


def fmt(f: str, v: float) -> str:
    s = cc.FMT[f].format(v)
    return s + ("%" if f == "traffic_congestion" else " mm/h" if f == "rainfall" else "")


def feed_story(f: str, row) -> dict:
    """Everything a resident needs to read one feed: value, usual, how unusual, category, meaning."""
    v, typ, sig, z = float(row[f]), float(row[f + "_typical"]), float(row[f + "_sigma"]), float(row[f"z_{f}"])
    phrase, word, col = level(z)
    cat, advice = band(f, v)
    mult = times_usual(v, typ)
    meaning = {
        "traffic_congestion": "Journeys through here will take longer than usual." if z >= 2 else "Traffic is as expected for this time.",
        "incidents_30m": "More accidents and breakdowns are being reported than usual. Expect blockages." if z >= 2
        else "Road incidents are at their usual level.",
        "aqi": advice if cat else "",
        "rainfall": advice or "No rain to worry about.",
    }[f]
    if f == "aqi" and z < 2 and cat in ("Good", "Moderate"):
        meaning = "Air quality is as usual for this hour. " + advice
    return dict(feed=f, icon=ICON[f], name=SHORT[f], value=fmt(f, v), usual=fmt(f, typ),
                low=max(typ - 2 * sig, 0.0), high=typ + 2 * sig, v=v, typ=typ, z=z,
                phrase=phrase, word=word, colour=col, category=cat, multiple=mult, meaning=meaning,
                unusual=z >= 2)


STATE_MEANING = {
    "Normal": ("Calm", "Everything is within its usual range for this time of day.", "No action needed."),
    "Watch": ("Watch", "Something is a little unusual, but not enough to confirm a problem yet.",
              "Nothing to do yet. Keep an eye on it."),
    "Confirmed": ("Disrupted", "Several readings are clearly unusual at the same time, and it has lasted.",
                  "Allow extra time if you are travelling through this area."),
    "Critical": ("Critical", "A severe disruption: several readings are far above normal.",
                 "Avoid non-essential travel through this area if you can, and follow official Jaipur advisories."),
}

PULSE_SCALE = [  # (from, to, state) on the 0-100 pulse
    (70, 100, "Normal"), (50, 70, "Watch"), (25, 50, "Confirmed"), (0, 25, "Critical")]

GLOSSARY = [
    ("Pulse (0 to 100)", "One number per neighbourhood. 100 means everything is exactly as usual for this time of "
                         "day; lower means more unusual. Think of it like a health score."),
    ("Usual for this hour", "What this neighbourhood normally looks like at this time of day, learned from past "
                            "days. Traffic at 6 pm is normally high, so high traffic at 6 pm is not a problem."),
    ("Usual range", "The grey band on charts. Readings inside it are normal; readings above it are unusual."),
    ("Calm, Watch, Disrupted, Critical", "Calm: all normal. Watch: a little unusual, being watched. Disrupted: "
                                         "clearly unusual and confirmed. Critical: severe."),
    ("Three detectors", "Three independent checks: one compares each reading with normal, one looks for unusual "
                        "combinations of readings, and one looks for unusual patterns over the last hour. An alert "
                        "needs two of them to agree."),
    ("Possible link", "When two things rise together, such as air quality and accidents, they may be related, "
                      "but CityPulse never claims one caused the other."),
    ("Air quality (AQI)", "US AQI scale: 0 to 50 good, 51 to 100 moderate, 101 to 150 unhealthy for sensitive "
                          "groups, 151 to 200 unhealthy, above 200 very unhealthy."),
]


EVENT_MEANING = {
    "Heavy rain with traffic disruption": "Rain and waterlogging slowed the roads: journeys took much longer than usual.",
    "Air-quality spike": "The air became unhealthy: people with breathing problems should have limited time outdoors.",
    "Air-quality spike with road disruption": "Poor air and road trouble at the same time: a bad moment to be out on the roads.",
    "Traffic and incident cluster": "Accidents and heavy traffic together: expect blocked roads and long delays.",
    "Unusual congestion": "Roads were much busier than usual for that time of day.",
    "Cluster of road incidents": "Several accidents or breakdowns happened close together.",
    "Heavy rain": "Heavy rain: waterlogging was possible on low-lying roads.",
}
