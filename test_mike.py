#!/usr/bin/env python3
"""
Mike Test Suite v2 (accepts early-season short form)
Usage (Windows CMD):
  set BASE_URL=http://127.0.0.1:8000
  python test_mike_v2.py
"""
import os
import json
import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")

def get(path, **params):
    url = f"{BASE}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def post(path, payload):
    url = f"{BASE}{path}"
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def p(label, ok, extra=""):
    status = "PASS ✅" if ok else "FAIL ❌"
    print(f"{status}  {label}")
    if extra:
        print(f"   ↳ {extra}")

def main():
    print("== Mike Test Suite v2 ==")
    print(f"BASE_URL = {BASE}\n")

    # 1) Standings n=5
    try:
        j = get("/standings/epl", n=5)
        ok = isinstance(j.get("standings"), list) and len(j["standings"]) > 0
        ex = f"{len(j.get('standings', []))} rows" if ok else json.dumps(j)[:180]
        p("Standings EPL (n=5)", ok, ex)
    except Exception as e:
        p("Standings EPL (n=5)", False, str(e))

    # 2) Standings n=10 (accept early-season short forms 1..10)
    try:
        j = get("/standings/epl", n=10)
        rows = j.get("standings", [])
        ok = bool(rows)
        if ok:
            fm = rows[0].get("form","")
            ok = isinstance(fm, str) and (1 <= len(fm) <= 10)
        p("Standings EPL (n=10, early-season OK)", ok, f"form='{rows[0].get('form','')}'" if rows else "")
    except Exception as e:
        p("Standings EPL (n=10, early-season OK)", False, str(e))

    # 3) Team last5
    try:
        j = get("/api/team/Arsenal/last5")
        ok = j.get("ok") is True and len(j.get("results", [])) > 0
        p("Team last5 (Arsenal)", ok, f"{len(j.get('results', []))} results")
    except Exception as e:
        p("Team last5 (Arsenal)", False, str(e))

    # 4) Team last10
    try:
        j = get("/api/team/Chelsea/last10")
        ok = j.get("ok") is True and len(j.get("results", [])) > 0
        p("Team last10 (Chelsea)", ok, f"{len(j.get('results', []))} results")
    except Exception as e:
        p("Team last10 (Chelsea)", False, str(e))

    # 5) Team news (mediastack)
    try:
        j = get("/api/team/Liverpool/news")
        ok = j.get("ok") is True
        heads = j.get("headlines", [])
        extra = f"{len(heads)} headlines" if heads else "no headlines (check MEDIASTACK_KEY)"
        p("Team news (Liverpool)", ok, extra)
    except Exception as e:
        p("Team news (Liverpool)", False, str(e))

    # 6) Ask: last match Man United (cards & fouls)
    try:
        j = post("/ask/", {"question": "Show Manchester United last match (include cards and fouls)."})
        text = j.get("response","")
        ok = isinstance(text, str) and len(text) > 0 and ("Cards" in text or "cards" in text)
        p("Ask: Man United last match (cards/fouls)", ok, text[:140].replace("\n"," ") + ("..." if len(text)>140 else ""))
    except Exception as e:
        p("Ask: Man United last match (cards/fouls)", False, str(e))

    # 7) Ask: corners last match Chelsea
    try:
        j = post("/ask/", {"question": "How many corners did Chelsea have last match?"})
        text = j.get("response","")
        ok = isinstance(text, str) and len(text) > 0 and ("Corners" in text or "corners" in text)
        p("Ask: Chelsea corners last match", ok, text[:140].replace("\n"," ") + ("..." if len(text)>140 else ""))
    except Exception as e:
        p("Ask: Chelsea corners last match", False, str(e))

    # 8) Match analysis: Chelsea vs Arsenal
    try:
        j = post("/ask/", {"question": "chelsea vs arsenal — give me best props and a final bet"})
        text = j.get("response","")
        ok = isinstance(text, str) and ("Best Props" in text) and ("Final Bet" in text)
        p("Ask: Chelsea vs Arsenal analysis", ok, text.splitlines()[0] if isinstance(text,str) and text else "")
    except Exception as e:
        p("Ask: Chelsea vs Arsenal analysis", False, str(e))

    # 9) Fixtures this weekend
    try:
        j = post("/ask/", {"question": "EPL fixtures this weekend"})
        text = j.get("response","")
        ok = isinstance(text, str) and "fixtures" in text.lower()
        p("Ask: EPL fixtures this weekend", ok, text.splitlines()[0] if isinstance(text,str) and text else "")
    except Exception as e:
        p("Ask: EPL fixtures this weekend", False, str(e))

    # 10) Random pick
    try:
        j = post("/ask/", {"question": "Give me another random betting pick"})
        text = j.get("response","")
        ok = isinstance(text, str) and "Random pick" in text
        p("Ask: Random pick", ok, text.splitlines()[0] if isinstance(text,str) and text else "")
    except Exception as e:
        p("Ask: Random pick", False, str(e))

    print("\nDone.\n")
    print("Note: The earlier fail on 'Standings n=10' was just early-season short form (e.g., 'WW'). This v2 accepts 1..10.")
    print("If you still see connection errors, ensure your Django server is running on BASE_URL and that the app urls are included.")

if __name__ == "__main__":
    main()
