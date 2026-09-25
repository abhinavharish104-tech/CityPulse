<div align="center">

# 💗 CityPulse

### Jaipur's neighbourhoods, one heartbeat each

A live civic health dashboard that fuses weather, air quality, traffic and road incidents into **one pulse per neighbourhood**, and shows the evidence behind every alert.

[![Live app](https://img.shields.io/badge/Live%20app-open-F2B544?style=for-the-badge)](https://citypulse-asp.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)(https://citypulse-asp.streamlit.app/)
![AmiHacks](https://img.shields.io/badge/AmiHacks-Track%20B-E99BBE?style=for-the-badge)

<img src="docs/screenshots/live.png" alt="CityPulse live board: Sindhi Camp flagged as critical while four other neighbourhoods are calm" width="820">

</div>

> **Live app:**(https://citypulse-asp.streamlit.app/)

---

## Why CityPulse

City data already exists, but residents still find out too late. Weather, air quality, traffic and incident feeds live in separate apps. A raw number like "AQI 175" means little unless you know what is normal for that place at that hour. Single-signal alerts fire on every rain shower, so people stop trusting them.

CityPulse answers the question a resident actually has, in about ten seconds:

> **"Is my neighbourhood okay right now, and why?"**

It watches five Jaipur neighbourhoods: Malviya Nagar, Vaishali Nagar, Mansarovar, C-Scheme and Sindhi Camp. For each one it gives a 0–100 pulse, a plain-language headline in English or Hindi, and the evidence behind it. Signals that rise together are always described as a *possible link, not a confirmed cause*.

---

## Features

### 🫀 Live pulse board
Every neighbourhood gets a scrolling ECG-style heartbeat whose speed and colour reflect its health: calm, watch, disrupted or critical. Above it, one headline explains what is unusual and where, for example *"Air quality (AQI) 175 (usually 89) and road incidents 17 (usually 3) in Sindhi Camp at this hour."* A dark city map, a "last three hours" list and an "allow extra time in" list sit alongside.

### 📈 Live timeline recorder
Like a hospital monitor, readings are drawn one by one with a moving pen, each against its usual range for that hour. Events shade in as the pen reaches them. You can pause, change speed, go back six hours, or switch between 3 h, 6 h and 24 h windows. A full static chart and a week-long heat strip for all neighbourhoods are also available.

<img src="docs/screenshots/recorder.png" alt="Live timeline recorder drawing rainfall, traffic, incidents, AQI and the combined score" width="820">

### ⏪ Time machine
Play back the whole week and watch the city's heartbeat change. You can also jump straight to any detected event, or share a link to any moment with `?t=2026-09-23T20:30`. The replay never reveals the future: an ongoing event shows only what is known so far.

### 🔎 Why this score
For any neighbourhood and moment, this view shows:
- how the three detectors add up to the combined score
- how far each feed is from its usual level
- the checks an alert must pass
- what would bring the neighbourhood back to calm
- which feed surprised the sequence model most

<img src="docs/screenshots/why.png" alt="Why this score: detector breakdown, checks before alerting, and what would bring it back to calm" width="820">

### 🚨 Events with evidence
Every event card shows where and when it happened, its severity, how many minutes it took to confirm, the readings that were unusual, and which detectors independently agreed. You can replay it, get a plain-words explanation, or download it as a report.

<img src="docs/screenshots/event.png" alt="Event card: heavy rain with traffic disruption, with evidence and severity" width="820">

### 🎛️ What-if simulator
Drag rainfall, traffic, incidents or AQI and watch the detectors re-score live, with the new pulse, state and agreement shown against the actual reading.

### 📣 Citizen reports
Residents can report an accident, waterlogging, a broken signal, smoke or a blocked road. Each report flows into that neighbourhood's incident feed, and you can watch it move the pulse. Phone numbers and email addresses are stripped automatically, and reports stay in the browser session.

### 💬 Ask CityPulse
Ask a question in plain language, such as *"Is it a good time to drive through Sindhi Camp?"*, *"What happened this week?"* or *"How does CityPulse decide?"*. Answers are grounded in the live data. When an AI model is connected, every number in its reply is checked against the data; if anything doesn't match, CityPulse answers from the data directly.

<img src="docs/screenshots/ask.png" alt="Ask CityPulse chat listing confirmed events" width="820">

### 🔔 Alerts
Follow the neighbourhoods you care about and pick the level to be alerted from. During playback a notification pops up when your area crosses that level, and the full alert log can be downloaded as CSV.

### 📡 Feed-outage test
Switch off any feed from the sidebar. Every reading was also scored with that feed removed, so the pulse keeps working with lower stated confidence, for example "3 of 4 feeds live, confidence 75%".

### 🌐 Built for everyone
- **Resident view** for a clean, plain-language experience.
- **City operations view** adds detector scores, severity drivers and per-event model detail.
- **Hindi summaries** are available from the sidebar.
- **Accessibility:** colour is never the only signal, since every state also has a word and an icon, and animations respect reduced-motion settings.

---

## How it works

<img src="docs/architecture.png" alt="Architecture: data sources, fusion, three detectors, decision layer, experience; offline pipeline, data contract, online app" width="900">

1. **Fuse.** Hourly rainfall, temperature and AQI (Open-Meteo) and 5-minute traffic and incident feeds are aligned into one table per neighbourhood every 5 minutes. Incidents become a rolling 30-minute count.
2. **Compare with typical.** For every neighbourhood and hour of day, a robust baseline (median and 1.4826 × MAD) is learned from training days only. Each reading becomes "how far above normal is this, for this place, at this hour".
3. **Detect three ways.**
   - A **statistical check** against the baseline. Rain on its own is down-weighted, because rain is weather, not a civic problem.
   - An **Isolation Forest** on the deviations and their 15-minute change.
   - An **LSTM autoencoder** that learned how each neighbourhood normally changes over an hour, scored on the most recent 15 minutes.
4. **Decide carefully.** A neighbourhood is flagged only when the combined score is high, **at least two of the three detectors agree**, and the disruption lasts **two readings in a row**. Nearby neighbourhoods disrupted at the same time merge into one city event with a 0–100 severity.
5. **Explain.** The dashboard turns the evidence into plain language, framing co-occurring signals as a possible link. An optional AI layer rewrites summaries, but only with figures that pass a number-by-number check against the data.

### Engineering decisions

| Decision | Why it matters |
|---|---|
| Baseline per place **and** hour | "17 incidents" is alarming at night and ordinary at rush hour. |
| Two of three detectors must agree | A single detector either misses events or cries wolf. |
| Calibrated without peeking | Chronological 60/20/20 split. Weights and thresholds are chosen on validation days, frozen, then run **once** on unseen test days. |
| Heavy work offline, light app online | The app reads a small data contract, so it needs no TensorFlow and starts fast. |
| Every feed also scored "switched off" | The outage switch shows real re-scored results, not a cosmetic change. |
| AI figures verified | Any number an AI model writes must exist in the data, or the answer falls back to a data-generated one. |

---

## Results

Measured on test days the models never saw, with injected scenarios and decoys:

| | Statistics alone | Isolation Forest alone | LSTM alone | **CityPulse (2 of 3 agree)** |
|---|:-:|:-:|:-:|:-:|
| Events caught | 100% | 75% | 75% | **4 / 4** |
| False alarms per day | 5.8 | 2.2 | 1.4 | **0** |

| Metric | Value |
|---|---|
| Decoys ignored (busy but normal moments) | 2 / 2 |
| Average time to confirm an event | about 18 minutes |
| First warning | 0–15 minutes after onset |
| Row-level PR-AUC (combined score) | 0.84 |

Each detector alone either misses events or raises false alarms. Requiring agreement keeps every event and removes the false alarms.

> Metrics measure detection of injected scenarios and are not a claim of real-world accuracy. Re-run the pipeline on your own `city_data.csv` and update this table with your numbers.

---

## Quick start

```bash
git clone <your-repo-url> citypulse
cd citypulse
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. To try it:
1. In the sidebar, click **Wed 23 Sep, 20:00 Sindhi Camp**.
2. Turn on **Play the week**.
3. Open the **Timeline** tab.

### Rebuild the detector on real data
The shipped `data/` folder lets the app run immediately. To retrain on the real Open-Meteo dataset:
- **Colab (recommended):** open `pipeline/CityPulse_v3_Pipeline.ipynb`, upload the project zip and `city_data.csv`, then **Run all**. Download `citypulse_data.zip` and replace `data/` with its contents.
- **Local:**
  ```bash
  pip install -r pipeline/requirements.txt
  python pipeline/citypulse_pipeline.py --input city_data.csv --out data
  ```

### Optional: AI summaries
Add one key to `.streamlit/secrets.toml` locally, or to **App settings → Secrets** on Streamlit Cloud. Claude, Gemini, Groq and OpenAI-compatible providers are supported:

```toml
GEMINI_API_KEY = "your-key"   # free tier at aistudio.google.com
```

Without a key, everything works with data-generated summaries.

### Deploy on Streamlit Community Cloud
Push the repository to GitHub, go to [share.streamlit.io](https://share.streamlit.io), and click **Create app**. Select the repo, set the main file to `app.py`, choose Python 3.12, and deploy.

---

## Project structure

```
app.py                      Streamlit dashboard
citypulse_core.py           Shared scoring maths (single source of truth for pipeline and app)
cp_ui.py                    Theme, heartbeat monitor, hero and card components
cp_recorder.py              Live timeline recorder (runs in the browser)
cp_llm.py                   Grounded AI summaries with number checking
data/                       Detector output: scored.csv, events.json, model_meta.json, iso_train.csv
pipeline/
  citypulse_pipeline.py     Train, calibrate on validation, evaluate on test, export data/
  CityPulse_v3_Pipeline.ipynb   Colab notebook for the pipeline
  make_sample_data.py       Offline stand-in generator for city_data.csv
docs/                       Architecture diagram and screenshots
.streamlit/                 Theme and secrets template
```

## Tech stack

Python, pandas, NumPy, scikit-learn (Isolation Forest), TensorFlow/Keras (LSTM autoencoder, offline only), Streamlit, Plotly, pydeck (deck.gl + CARTO map), Open-Meteo APIs, and optional LLM providers (Claude, Gemini, Groq, OpenAI).

## Future scope

- Replace the simulated traffic and incident feeds with live city sensors and Rajasthan Sampark (181) complaints.
- Cover every Jaipur ward, then other Rajasthan cities.
- Send alerts over WhatsApp, SMS and push, in Hindi and English.
- Add power cuts, water supply, waste collection and heat stress as civic signals.
- Forecast the next hour's pulse from weather forecasts.
- Build a city operations console to assign, acknowledge and resolve events.

## Data and honesty

Rainfall, temperature and AQI come from **Open-Meteo** (real) when the pipeline is run on the project's `city_data.csv`. Traffic congestion, road incidents, the labelled scenarios and the decoys are **simulated**, and the dashboard says so on screen. No personal data is collected.

---

<div align="center">

### Created by Team Neutral Navigators

**Abhinav Harish** (Team Leader) · **Samdrisht** · **Paridhi**

| Member | Role | Contributions |
|---|---|---|
| Abhinav Harish | Team Leader | Project lead and system architecture, data-fusion layer and Open-Meteo integration, GitHub and Streamlit Cloud deployment, demo coordination |
| Samdrisht | Member | Dashboard UI design and build: heartbeat monitor, live map, timeline recorder, replay, alerts and citizen reports |
| Paridhi | Member | Problem research, demo scenarios and end-to-end testing, plain-language and Hindi text, documentation and presentation |

Built for **AmiHacks**, Track B: Live Civic Health Dashboard.

</div>
