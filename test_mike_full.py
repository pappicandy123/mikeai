# test_mike_full.py
# End-to-end tester for "Mike" (EPL-focused)
# - Hits REST endpoints and the /ask/ chat intent
# - Prints PASS / FAIL lines with short context


import os, sys, json, time
from datetime import datetime
import requests

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000").rstrip("/")

API_FOOTBALL = bool(os.getenv("API_FOOTBALL_KEY"))
SPORTMONKS = bool(os.getenv("SPORTMONKS_API_KEY"))
MEDIASTACK = bool(os.getenv("MEDIASTACK_KEY"))

PASS = 0
FAIL = 0
SOFT = 0

def out(ok, title, detail=""):
    global PASS, FAIL, SOFT
    if ok is True:
        PASS += 1
        print(f"PASS ✅  {title}")
        if detail: print(f"   ↳ {detail}")
    elif ok == "SOFT":
        SOFT += 1
        print(f"SOFT ⚠️  {title}")
        if detail: print(f"   ↳ {detail}")
    else:
        FAIL += 1
        print(f"FAIL ❌  {title}")
        if detail: print(f"   ↳ {detail}")

def get_json(path, params=None, timeout=25):
    url = f"{BASE_URL}{path}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def post_json(path, payload=None, timeout=25):
    url = f"{BASE_URL}{path}"
    r = requests.post(url, json=payload or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def expect_substring(text, *needles):
    t = (text or "").lower()
    return all(n.lower() in t for n in needles)

print("== Mike Full Test v3 ==")
print(f"BASE_URL = {BASE_URL}\n")
hint = []
if not API_FOOTBALL: hint.append("API_FOOTBALL_KEY")
if not SPORTMONKS: hint.append("SPORTMONKS_API_KEY")
if not MEDIASTACK: hint.append("MEDIASTACK_KEY")
if hint:
    print(f"⚠️  Environment hint: missing {', '.join(hint)} (some tests may soft-fail).")
print()

# 0) health
try:
    h = get_json("/health")
    k = h.get("keys", {})
    ok = isinstance(k, dict)
    out(ok, "Health endpoint", f"keys={k}")
except Exception as e:
    out(False, "Health endpoint", str(e))

# 1) standings
try:
    epl = get_json("/standings/epl", {"n": 5})
    rows = epl.get("standings") or []
    ok = bool(rows) and isinstance(rows, list)
    out(ok, "Standings EPL (n=5)", f"{len(rows)} rows")
except Exception as e:
    out(False, "Standings EPL (n=5)", str(e))

try:
    epl10 = get_json("/standings/epl", {"n": 10})
    rows = epl10.get("standings") or []
    # Early season can have short form like 'WW'; don't be strict
    ok = bool(rows) and isinstance(rows, list)
    out(ok, "Standings EPL (n=10, early-season OK)", f"{len(rows)} rows")
except Exception as e:
    out(False, "Standings EPL (n=10, early-season OK)", str(e))

# 2) team last5/last10
try:
    a5 = get_json("/api/team/Arsenal/last5")
    ok = a5.get("ok") and (a5.get("results") is not None)
    out(ok, "Team last5 (Arsenal)", f"{len(a5.get('results') or [])} results")
except Exception as e:
    out(False, "Team last5 (Arsenal)", str(e))

try:
    c10 = get_json("/api/team/Chelsea/last10")
    ok = c10.get("ok") and (c10.get("results") is not None)
    out(ok, "Team last10 (Chelsea)", f"{len(c10.get('results') or [])} results")
except Exception as e:
    out(False, "Team last10 (Chelsea)", str(e))

# 3) team summary averages (corners/cards/fouls)
try:
    tsm = get_json("/api/team/Tottenham/summary", {"n": 5})
    if tsm.get("ok") and (tsm.get("matches_used", 0) > 0) and isinstance(tsm.get("averages"), dict):
        out(True, "Team summary (Spurs, n=5)", f"avg={tsm['averages']}, used={tsm['matches_used']}")
    elif tsm.get("ok") and tsm.get("matches_used", 0) == 0:
        out("SOFT", "Team summary (Spurs, n=5)", "No stat feed found in last-N fixtures.")
    else:
        out(False, "Team summary (Spurs, n=5)", tsm.get("error") or "unknown")
except Exception as e:
    out(False, "Team summary (Spurs, n=5)", str(e))

# 4) team news
try:
    news = get_json("/api/team/Liverpool/news")
    heads = news.get("headlines") or []
    ok = isinstance(heads, list)
    # if no key, still pass as structure OK
    out(ok, "Team news (Liverpool)", f"{len(heads)} headlines")
except Exception as e:
    out(False, "Team news (Liverpool)", str(e))

# 5) fixtures endpoint (prefers SportMonks)
try:
    fx = get_json("/api/fixtures", {"league": "epl", "days": 8})
    src = fx.get("source")
    rows = fx.get("fixtures") or []
    if rows:
        out(True, "Fixtures API (EPL next 8 days)", f"source={src}, rows={len(rows)}")
    else:
        out("SOFT", "Fixtures API (EPL next 8 days)", f"source={src}, rows=0")
except Exception as e:
    out(False, "Fixtures API (EPL next 8 days)", str(e))

# 6) random pick and best bets
try:
    rnd = get_json("/api/random-pick", {"league": "epl"})
    if rnd.get("ok"):
        out(True, "Random pick API", f"{rnd.get('match')} | {rnd.get('final_pick')}")
    else:
        out("SOFT", "Random pick API", rnd.get("error") or "no pick")
except Exception as e:
    out(False, "Random pick API", str(e))

try:
    bb = get_json("/api/best-bets", {"league": "epl"})
    res = bb.get("results") or []
    ok = isinstance(res, list)
    detail = (f"{len(res)} results" if res else "empty board")
    # Empty board is OK (model cautious), so soft-pass
    out(True if res else "SOFT", "Best bets API", detail)
except Exception as e:
    out(False, "Best bets API", str(e))

# 7) team panel page (HTML)
try:
    r = requests.get(f"{BASE_URL}/team?name=Arsenal", timeout=15)
    ok = (r.status_code == 200) and ("<html" in r.text.lower())
    out(ok, "Team Panel HTML (/team?name=Arsenal)")
except Exception as e:
    out(False, "Team Panel HTML (/team?name=Arsenal)", str(e))

# 8) /ask/ chat intents (text responses)
def ask(q):
    return post_json("/ask/", {"question": q})

# fixtures this weekend
try:
    a = ask("Show EPL fixtures this weekend")
    txt = a.get("response","")
    ok = expect_substring(txt, "epl", "fixtures")
    out(ok, "Ask: EPL fixtures this weekend", txt[:100])
except Exception as e:
    out(False, "Ask: EPL fixtures this weekend", str(e))

# last weekend single-team
try:
    a = ask("What did Arsenal play last weekend?")
    txt = a.get("response","")
    ok = ("arsenal" in txt.lower()) and ("last" in txt.lower() or "corners" in txt.lower() or "cards" in txt.lower())
    out(ok, "Ask: Arsenal last weekend", txt[:120])
except Exception as e:
    out(False, "Ask: Arsenal last weekend", str(e))

# last match corners/cards (Chelsea)
try:
    a = ask("How many corners did Chelsea have last match?")
    txt = a.get("response","")
    ok = ("chelsea" in txt.lower()) and ("corners" in txt.lower())
    out(ok, "Ask: Chelsea corners last match", txt[:120])
except Exception as e:
    out(False, "Ask: Chelsea corners last match", str(e))

# match analysis (Chelsea vs Arsenal)
try:
    a = ask("Chelsea vs Arsenal — give me best props and a final bet")
    txt = a.get("response","")
    if "couldn’t fetch" in txt.lower():
        out("SOFT", "Ask: Chelsea vs Arsenal analysis", "model not available right now")
    else:
        ok = ("best props" in txt.lower()) and ("final bet" in txt.lower())
        out(ok, "Ask: Chelsea vs Arsenal analysis", txt[:120])
except Exception as e:
    out(False, "Ask: Chelsea vs Arsenal analysis", str(e))

# best bets (board)
try:
    a = ask("Best bets for EPL this weekend")
    txt = a.get("response","")
    ok = ("best bets" in txt.lower()) or ("value board" in txt.lower()) or ("results" in txt.lower())
    out(ok, "Ask: Best bets this weekend", txt[:120])
except Exception as e:
    out(False, "Ask: Best bets this weekend", str(e))

# random pick (chat)
try:
    a = ask("Give me a random pick")
    txt = a.get("response","")
    if "random pick" in txt.lower():
        out(True, "Ask: Random pick", txt[:120])
    elif "couldn’t fetch" in txt.lower():
        out("SOFT", "Ask: Random pick", "no suitable matchup right now")
    else:
        out(False, "Ask: Random pick", txt[:120])
except Exception as e:
    out(False, "Ask: Random pick", str(e))

# next opponent + preview
try:
    a = ask("Who do Manchester United face this weekend? Give a quick preview.")
    txt = a.get("response","")
    ok = ("fixtures" in txt.lower()) or ("manchester united" in txt.lower()) or ("preview" in txt.lower())
    out(ok, "Ask: Man United next opponent preview", txt[:120])
except Exception as e:
    out(False, "Ask: Man United next opponent preview", str(e))

print("\n== Done ==")
print(f"PASS={PASS}  SOFT={SOFT}  FAIL={FAIL}")
if FAIL:
    print("\nTips:")
    print(" - Make sure your Django server is running at BASE_URL.")
    print(" - If last-match stats fail, confirm API_FOOTBALL_KEY is valid.")
    print(" - If fixtures return empty, ensure SPORTMONKS_API_KEY (or fallback uses API-Football).")
    print(" - If news is empty, set MEDIASTACK_KEY.")
    print(" - You can change BASE_URL with:  $env:BASE_URL='http://localhost:8001'")
