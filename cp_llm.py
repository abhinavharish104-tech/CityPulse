"""Grounded LLM layer for CityPulse.

* The model only ever sees a small JSON of facts computed by the detector.
* Every number in its reply is checked against those facts; if any figure is not in the data
  the reply is rejected and the dashboard shows the deterministic template instead.
* Works with Anthropic (Claude), Google Gemini, Groq, or any OpenAI-compatible endpoint.
  Configure in .streamlit/secrets.toml (see secrets.toml.example). No key -> template mode.
"""
from __future__ import annotations

import json
import os
import re

import requests

SYSTEM = (
    "You are CityPulse, a civic dashboard assistant for residents of Jaipur. "
    "Use ONLY the facts in the JSON you are given. Never invent numbers, places, causes or times. "
    "Write for a non-technical resident: short sentences, plain words, no jargon such as z-score, "
    "Isolation Forest or LSTM unless the user asks how the system works. "
    "When two signals rise together, call it a possible link, never a confirmed cause. "
    "If the facts do not answer the question, say so and suggest what the dashboard can show. "
    "Do not give instructions that belong to emergency services; for danger, tell people to follow "
    "official Jaipur advisories and call 112."
)

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
}


def _get(secrets, key, default=None):
    try:
        if secrets is not None and key in secrets:
            return secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


def config(secrets=None) -> dict | None:
    """Pick the first provider with a key. LLM_PROVIDER forces one."""
    forced = (_get(secrets, "LLM_PROVIDER") or "").lower().strip()
    keys = {"anthropic": _get(secrets, "ANTHROPIC_API_KEY"), "gemini": _get(secrets, "GEMINI_API_KEY"),
            "groq": _get(secrets, "GROQ_API_KEY"), "openai": _get(secrets, "OPENAI_API_KEY")}
    order = [forced] if forced in keys else ["anthropic", "gemini", "groq", "openai"]
    for p in order:
        if keys.get(p):
            return dict(provider=p, key=keys[p], model=_get(secrets, "LLM_MODEL") or DEFAULT_MODELS[p],
                        base_url=_get(secrets, "OPENAI_BASE_URL", "https://api.openai.com/v1"))
    return None


def _call(cfg: dict, prompt: str, max_tokens: int = 350) -> str:
    p = cfg["provider"]
    if p == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=25,
                          headers={"x-api-key": cfg["key"], "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json={"model": cfg["model"], "max_tokens": max_tokens, "system": SYSTEM,
                                "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text").strip()
    if p == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['model']}:generateContent"
        r = requests.post(url, params={"key": cfg["key"]}, timeout=25, json={
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2}})
        r.raise_for_status()
        return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"]).strip()
    base = "https://api.groq.com/openai/v1" if p == "groq" else cfg["base_url"].rstrip("/")
    r = requests.post(f"{base}/chat/completions", timeout=25,
                      headers={"Authorization": f"Bearer {cfg['key']}", "Content-Type": "application/json"},
                      json={"model": cfg["model"], "max_tokens": max_tokens, "temperature": 0.2,
                            "messages": [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


_NUM = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def unsupported_numbers(reply: str, facts: dict) -> list[float]:
    """Numbers in the reply that are not (within rounding) in the facts. Small counting words
    like 'two feeds' are fine; times/dates are checked too because they appear in the facts."""
    blob = json.dumps(facts, default=str)
    allowed = set(_numbers(blob)) | {0, 1, 2, 3, 4, 5, 100, 112}
    bad = []
    for x in _numbers(reply):
        if any(abs(x - a) <= max(0.51, abs(a) * 0.01) for a in allowed):
            continue
        bad.append(x)
    return bad


def grounded(cfg: dict | None, task: str, facts: dict, max_tokens: int = 350) -> tuple[str | None, str]:
    """Returns (text or None, status). status in {'ok','no_key','ungrounded','error:<msg>'}."""
    if not cfg:
        return None, "no_key"
    prompt = (f"{task}\n\nFACTS (the only source you may use):\n```json\n"
              f"{json.dumps(facts, indent=1, default=str)}\n```")
    try:
        text = _call(cfg, prompt, max_tokens)
    except Exception as e:  # network, quota, bad key, model name...
        return None, f"error:{type(e).__name__}: {str(e)[:120]}"
    bad = unsupported_numbers(text, facts)
    if bad:
        return None, f"ungrounded:{bad[:5]}"
    return text, "ok"
