#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive interactive test suite for Mike (sports betting assistant).

- Keeps a requests.Session to preserve cookies (conversation + last teams memory).
- Covers fixtures, predictions, odds, follow-ups (corners/cards/fouls), team-only follow-ups,
  news, weekend value board, small-odds bankers, and general Q&A.
- Includes robustness tests (typos, nicknames, casing).
- Prints compact PASS/FAIL lines plus an optional debug snippet.
- Base URL can be overridden by env var MIKE_BASE_URL (default http://127.0.0.1:8000).

Run:
    python test_mike_full.py
"""

import os
import time
import json
import requests

BASE_URL = os.getenv("MIKE_BASE_URL", "http://127.0.0.1:8000")

# If your endpoint is not /ask/, change here
ASK_ENDPOINT = "/ask/"

# Utility: send a message and return (ok, text)
def send(session, question, extra_payload=None, timeout=30):
    payload = {"question": question}
    if extra_payload:
        payload.update(extra_payload)
    try:
        r = session.post(BASE_URL + ASK_ENDPOINT, json=payload, timeout=timeout)
        ok = (r.status_code == 200)
        try:
            data = r.json()
            text = data.get("response") or data.get("error") or r.text
        except Exception:
            text = r.text
        return ok, text
    except Exception as e:
        return False, str(e)

# Checking helper: does any of the expected keywords appear in the response?
def matches_any(text, expectations):
    low = (text or "").lower()
    for exp in expectations:
        if exp.lower() in low:
            return True
    return False

# Pretty print helpers
def short(s, n=160):
    s = s.replace("\\n", " ").replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s

def line(title, status, snippet):
    pad = 40 - len(title)
    pad = max(1, pad)
    return f"{title}{' ' * pad}{status}  {short(snippet)}"

def run_block(title, steps):
    """
    steps: list of dicts with keys:
        - q: question (str)
        - expect: list[str] substrings to search for (any match passes)
        - payload: optional extra dict to send
        - sleep: optional wait seconds between steps (default 0.3)
    """
    s = requests.Session()
    print(f"\n=== {title} ===")
    passed = 0
    for i, step in enumerate(steps, 1):
        time.sleep(step.get("sleep", 0.3))
        ok, text = send(s, step["q"], extra_payload=step.get("payload"))
        if not ok:
            print(line(f"{i}. {step['q']}", "FAIL", f"HTTP fail: {text}"))
            continue
        expect = step.get("expect", [])
        if expect and matches_any(text, expect):
            print(line(f"{i}. {step['q']}", "PASS", text))
            passed += 1
        else:
            print(line(f"{i}. {step['q']}", "FAIL", text))
    print(f"Subtotal: {passed}/{len(steps)} passed")
    return passed, len(steps)

if __name__ == "__main__":
    total_ok = 0
    total_n = 0

    # 1) Fixtures + weekend flow
    ok, n = run_block("Weekend fixtures & quick picks", [
        {"q": "Show EPL fixtures this weekend", "expect": ["fixtures", "📅", "epl"]},
        {"q": "best bets this weekend", "expect": ["value board", "value", "best bets", "📋"]},
        {"q": "give me another random betting pick", "expect": ["pick", "odds", "model", "edge"]},
    ])
    total_ok += ok; total_n += n

    # 2) Single match + follow-ups (memory of teams)
    ok, n = run_block("Match analysis + follow‑ups", [
        {"q": "Predict Liverpool vs Bournemouth and show odds", "expect": ["best", "final bet", "odds", "📈"]},
        {"q": "what about corners", "expect": ["corner", "corners", "team avg", "🎯"]},
        {"q": "and cards?", "expect": ["card", "cards", "yellow", "📒"]},
        {"q": "and fouls then", "expect": ["foul", "fouls", "avg", "🧤"]},
    ])
    total_ok += ok; total_n += n

    # 3) Team-only props
    ok, n = run_block("Single‑team props", [
        {"q": "corner bet for Liverpool", "expect": ["corners", "team avg", "🎯"]},
        {"q": "cards for Arsenal", "expect": ["cards", "yellow", "📒"]},
        {"q": "fouls for Chelsea", "expect": ["fouls", "avg", "🧤"]},
    ])
    total_ok += ok; total_n += n

    # 4) Robustness: nicknames, casing, typos
    ok, n = run_block("Robust team parsing", [
        {"q": "Predict spurs vs man utd", "expect": ["final bet", "best props", "odds"]},
        {"q": "predict Li verpool vs Bournemonth", "expect": ["final bet", "best props", "odds"]},
        {"q": "Tottenham v Manchester United odds", "expect": ["odds", "📈"]},
    ])
    total_ok += ok; total_n += n

    # 5) News
    ok, n = run_block("News flow", [
        {"q": "Give me latest EPL news", "expect": ["news", "•"]},
        {"q": "news on Chelsea", "expect": ["news", "•", "chelsea"]},
    ])
    total_ok += ok; total_n += n

    # 6) La Liga cross‑league (LLM fallback allowed but should be structured)
    ok, n = run_block("La Liga cross‑league", [
        {"q": "Predict Barcelona vs Real Madrid and show odds", "expect": ["final bet", "best props", "odds"]},
        {"q": "what about corners", "expect": ["corner", "corners", "team avg", "🎯"]},
    ])
    total_ok += ok; total_n += n

    # 7) General Q&A like ChatGPT
    ok, n = run_block("General Q&A", [
        {"q": "Who won the last EPL season?", "expect": ["city", "arsenal", "liverpool", "champion", "premier league"]},
        {"q": "Explain xG in simple terms", "expect": ["expected goals", "xg"]},
        {"q": "What is offside?", "expect": ["offside"]},
    ])
    total_ok += ok; total_n += n

    print(f"\n=== TOTAL: {total_ok}/{total_n} passed ===")
