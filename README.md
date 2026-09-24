# CityPulse — Jaipur's neighbourhoods, one heartbeat each

A live civic health dashboard. Rainfall, temperature and air quality (Open-Meteo), traffic congestion and road
incidents are fused every 5 minutes for five Jaipur neighbourhoods. Three detectors (statistics, Isolation Forest,
LSTM autoencoder) must agree before anyone is alerted. Every alert comes with its evidence, in plain language,
framed as a *possible link, not a confirmed cause*.

## Results on the unseen test period (stand-in data; rerun on your real `city_data.csv`)

| | Statistics alone | Isolation Forest alone | LSTM alone | **CityPulse (2 of 3 agree)** |
|---|---|---|---|---|
| Events caught | 100% | 75% | 75% | **100% (4 / 4)** |
| False alarms per day | 5.8 | 2.2 | 1.4 | **0** |

Decoys ignored: 2 / 2. Average time to confirm: 18 min (first warning 0–15 min). Row PR-AUC 0.84.

## Project layout

```
app.py                      Streamlit dashboard (no TensorFlow needed)
citypulse_core.py           Scoring maths shared by pipeline and app (single source of truth)
cp_ui.py                    Theme CSS + ECG monitor, hero and card components
cp_llm.py                   Grounded AI summaries (Claude / Gemini / Groq / OpenAI) with number checking
data/                       Output of the pipeline: scored.csv, events.json, model_meta.json, iso_train.csv
pipeline/
  citypulse_pipeline.py     Train -> calibrate on validation -> evaluate on test -> export data/
  CityPulse_v3_Pipeline.ipynb   Colab wrapper for the pipeline, with plots
  make_sample_data.py       Offline stand-in for city_data.csv (same generator as the Core Engine)
.streamlit/config.toml      Theme
.streamlit/secrets.toml.example
```

## 1. Rebuild `data/` from your real `city_data.csv` (do this first)

The `data/` folder shipped here was built from a simulated stand-in so the app runs out of the box; the dashboard
says so in its source line. To use your real Open-Meteo data:

- **Colab:** open `pipeline/CityPulse_v3_Pipeline.ipynb`, run all, upload `city_data.csv`, download
  `citypulse_data.zip`, and replace `data/` with its contents.
- **Local:** `pip install -r pipeline/requirements.txt`, then
  `python pipeline/citypulse_pipeline.py --input city_data.csv --out data`

Thresholds and weights are re-learned from your data, so your numbers will differ slightly from the table above.
Put the new numbers on your slides.

## 2. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open a specific moment directly: `http://localhost:8501/?t=2026-09-23T20:30`

## 3. Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repo, including `data/`. Never commit `secrets.toml`; `.gitignore`
   already excludes it.
2. Go to share.streamlit.io, click **Create app**, pick the repo and branch, and set the main file to `app.py`.
   Under *Advanced settings*, choose Python 3.12.
3. (Optional, for AI summaries) Under *Advanced settings → Secrets*, paste one key, for example
   `GEMINI_API_KEY = "..."`. See `.streamlit/secrets.toml.example`.
4. Deploy. Open the app once before judging starts so it is awake; free apps sleep after inactivity.

Without a key, everything works using data-grounded template summaries.

## Features

- **Right now:** a vital-signs monitor. Each neighbourhood gets a live ECG trace whose speed and colour reflect its
  health, plus a 0–100 pulse, the reason in one line, a dark map, the last three hours, and places to avoid.
- **Plain-language headline:** in English or Hindi. It uses a template, or an AI summary whose every number is
  checked against the data.
- **Time machine:** play the week, jump to any event, or share a link to any moment. The replay never shows the
  future: ongoing events show only what is known so far.
- **Timeline:** each feed against its usual range for that hour, the combined score with alert bands, and a
  week-long heat strip for every neighbourhood.
- **Events:** evidence, severity, minutes to confirm, which detectors agreed, a replay button, an AI explanation,
  and a Markdown download.
- **Why this score:** how the three detectors add up, each feed's distance from usual, the checks an alert must
  pass, what would bring the neighbourhood back to calm, and the feed that surprised the LSTM most.
- **What if:** drag readings and watch statistics and the Isolation Forest re-score live.
- **Report a problem:** citizen reports enter the incident feed and move the pulse. Phone numbers and emails are
  stripped, and nothing is stored beyond the session.
- **Ask CityPulse:** a chat grounded in current readings, events and the method, with a rule-based fallback.
- **Alerts:** follow neighbourhoods at a chosen level, with pop-up notifications during replay and a CSV log.
- **Feed outage:** switch off any feed and every score is replaced by one precomputed by all three detectors
  without it.
- **City operations mode:** detector channels, severity drivers and LSTM attribution per event.
- **How it works:** the ablation chart, per-scenario detection times, and a downloadable model card.

## Demo script (3 minutes)

1. **Problem (20 s).** Feeds are siloed, so residents find out after they are stuck.
2. **Glance test (30 s).** Open the app on a calm moment and read the headline aloud. Then press the sidebar event
   *Wed 23 Sep, 20:00 Sindhi Camp* and **Play the week**. The Sindhi Camp trace speeds up and turns red, the
   headline changes, and the alert toast fires.
3. **Why (40 s).** Open the *Why this score* tab. Walk through the three detectors adding up, the checks, and what
   would bring it back to calm. Then open *Events*: evidence, "confirmed 5 min after it began", and the
   possible-link caveat.
4. **Robustness (30 s).** In the sidebar set *Traffic congestion offline*. The pulse keeps working and the
   confidence drops. Then file a citizen report in a calm neighbourhood and watch its pulse move.
5. **Proof (40 s).** Open *How it works*. Each detector alone raises 1.4–5.8 false alarms a day; requiring agreement
   brings that to zero while still catching every event. Calibrated on validation, run once on test.
6. **Impact (20 s).** Resident view, city-ops mode, and Hindi. It can be extended to real 311, transit or
   power-outage feeds by adding a column.

## Honesty notes

Rainfall, temperature and AQI come from Open-Meteo (real) when you run the pipeline on your `city_data.csv`.
Traffic, incidents, the 7 labelled scenarios and the 4 decoys are simulated and labelled as such. Metrics measure
detection of injected scenarios and do not claim real-world accuracy. No personal data is used.
