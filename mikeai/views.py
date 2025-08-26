# assistant/views.py — Mike v5.2 (EPL-only enrichments)

import os
import re
import json
import random
import unicodedata
from typing import Optional, Dict, Any, List, Tuple
from datetime import timedelta, date

import requests
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from rest_framework.views import APIView
from rest_framework.response import Response

from . import api_football as af
from .standings import standings_epl, standings_laliga

# Optional OpenAI
try:
    from openai import OpenAI
    OPENAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    OPENAI_OK = True
except Exception:
    OPENAI, OPENAI_OK = None, False

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SPORTMONKS_KEY     = os.getenv("SPORTMONKS_API_KEY")
SM_LEAGUE_EPL      = os.getenv("SPORTMONKS_LEAGUE_EPL")
SM_LEAGUE_LALIGA   = os.getenv("SPORTMONKS_LEAGUE_LALIGA")
MEDIASTACK_KEY     = os.getenv("MEDIASTACK_KEY")
ODDS_API_KEY       = os.getenv("ODDS_API_KEY")

SM_BASE = "https://api.sportmonks.com/v3/football"
AF_LEAGUE = {"epl": 39, "laliga": 140}

def chat_page(request):
    return render(request, "index.html")

def team_panel(request):
    # simple template-less variant if you didn't create team.html yet
    try:
        return render(request, "team.html")
    except Exception:
        return HttpResponse("<h1>Team Panel</h1><p>Create templates/team.html to render this page.</p>")

def log_info(msg, **ctx):
    print(f"[INFO] {msg} {json.dumps(ctx, ensure_ascii=False)}" if ctx else f"[INFO] {msg}")

def log_err(msg, **ctx):
    print(f"[ERROR] {msg} {json.dumps(ctx, ensure_ascii=False)}" if ctx else f"[ERROR] {msg}")

# ------------ name helpers (for parsing match prompts) ------------
def _norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    for junk in [" football club"," afc"," fc"," cf",".",",","-","_","|","/","\\","(",")","[","]","{","}",":",";","•","—"]:
        s = s.replace(junk, " ")
    return " ".join(s.split())

NICKNAMES = {
    "spurs":"Tottenham Hotspur","tottenham":"Tottenham Hotspur",
    "man city":"Manchester City","manchester city":"Manchester City","city":"Manchester City",
    "man united":"Manchester United","man utd":"Manchester United","manchester united":"Manchester United","united":"Manchester United",
    "west ham":"West Ham United","westham":"West Ham United",
    "wolves":"Wolverhampton Wanderers","wolverhampton":"Wolverhampton Wanderers",
    "bournemouth":"AFC Bournemouth","afc bournemouth":"AFC Bournemouth",
    "newcastle":"Newcastle United","forest":"Nottingham Forest","nottingham forest":"Nottingham Forest",
    "villa":"Aston Villa","aston villa":"Aston Villa",
    "liverpool":"Liverpool","chelsea":"Chelsea","arsenal":"Arsenal","everton":"Everton",
    "fulham":"Fulham","brentford":"Brentford","crystal palace":"Crystal Palace","palace":"Crystal Palace",
    "ipswich":"Ipswich Town","leicester":"Leicester City","southampton":"Southampton",
    "brighton":"Brighton & Hove Albion","brighton and hove albion":"Brighton & Hove Albion",
}

def _resolve_team(raw: str) -> str:
    if not raw: return ""
    x = _norm(raw); x_nospace = x.replace(" ", "")
    for k, v in NICKNAMES.items():
        if x == _norm(k) or x_nospace == _norm(k).replace(" ","") or _norm(k) in x or _norm(k).replace(" ","") in x_nospace:
            return v
    return " ".join(w.capitalize() for w in raw.strip().split())

def _extract_teams(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text: return (None, None)
    q = _norm(text)
    m = re.search(r"(.+?)\s+(?:vs|v|versus|and|against|between)\s+(.+)", q, flags=re.I)
    if not m: return (None, None)
    return _resolve_team(m.group(1)), _resolve_team(m.group(2))

def _guess_team(text: str) -> Optional[str]:
    q = _norm(text)
    for k, v in NICKNAMES.items():
        if _norm(k) in q or _norm(k).replace(" ", "") in q.replace(" ", ""):
            return v
    # naive fallback: first 2 tokens
    words = [w for w in text.split() if w.strip()]
    if words:
        return _resolve_team(" ".join(words[:2]))
    return None

# ------------ time helpers ------------
def _next_window(days=8):
    now = timezone.now()
    return now, now + timedelta(days=days)

def _recent_weekend_dates() -> List[str]:
    """
    Return ISO dates for the Saturday and Sunday of the most recent *completed* weekend.
    """
    today = timezone.localtime().date()
    wd = today.weekday()  # Mon=0..Sun=6
    days_since_sat = (wd - 5) % 7
    # If it's Sat/Sun, go to previous weekend
    if wd >= 5:
        days_since_sat += 7
    saturday = today - timedelta(days=days_since_sat)
    sunday = saturday + timedelta(days=1)
    return [saturday.isoformat(), sunday.isoformat()]

# ------------ SportMonks (fixtures fetch) ------------
def sm_get(path, **params):
    if not SPORTMONKS_KEY:
        raise RuntimeError("SPORTMONKS_API_KEY missing")
    q = {k: v for k, v in params.items() if v is not None}
    q["api_token"] = SPORTMONKS_KEY
    r = requests.get(f"{SM_BASE}/{path.lstrip('/')}", params=q, timeout=20)
    r.raise_for_status()
    return r.json(), r.url

def _sm_league_id(slug: str):
    if slug == "epl": return SM_LEAGUE_EPL
    if slug == "laliga": return SM_LEAGUE_LALIGA
    return None

def sm_try_fixtures(league_slug: str, start_date: str, end_date: str):
    lid = _sm_league_id(league_slug)
    if not lid:
        return [], f"SM league id missing for {league_slug}"
    attempts = [
        ("fixtures between", lambda: sm_get(f"fixtures/between/{start_date}/{end_date}", leagues=lid, include="participants")),
        ("fixtures filters", lambda: sm_get("fixtures", filters=f"between:{start_date},{end_date}|league_id:{lid}", include="participants")),
    ]
    last_err = None
    for label, fn in attempts:
        try:
            j, url = fn()
            data = j.get("data") or []
            out = []
            for row in data:
                dt = row.get("starting_at") or row.get("starting_at_utc") or ""
                parts = row.get("participants") or []
                home = next((p for p in parts if (p.get("meta") or {}).get("location")=="home"), {})
                away = next((p for p in parts if (p.get("meta") or {}).get("location")=="away"), {})
                out.append({
                    "fixture": {"date": str(dt)},
                    "teams": {
                        "home": {"name": home.get("name") or home.get("short_code") or ""},
                        "away": {"name": away.get("name") or away.get("short_code") or ""},
                    }
                })
            log_info("SM fixtures ok", label=label, url=url, rows=len(out))
            return out, f"SM ok via {label}"
        except Exception as e:
            last_err = str(e)
            log_err("SM fixtures error", label=label, error=last_err)
    return [], f"SM failed: {last_err or 'unknown'}"

# ------------ AF fixtures fallback ------------
def af_fixtures(league_slug: str, start_date: str, end_date: str):
    lg = AF_LEAGUE.get(league_slug)
    if not lg: return [], "AF: unknown league"
    try:
        resp = af.get_fixtures_by_range(start_date, end_date, league_id=lg)
        arr = resp.get("response", []) if isinstance(resp, dict) else resp
        out = []
        for fx in arr:
            f = fx.get("fixture", {})
            t = fx.get("teams", {})
            out.append({
                "fixture": {"date": f.get("date","")},
                "teams": {
                    "home": {"name": (t.get("home") or {}).get("name","")},
                    "away": {"name": (t.get("away") or {}).get("name","")},
                }
            })
        return out, "AF ok"
    except Exception as e:
        log_err("AF fixtures error", error=str(e))
        return [], f"AF failed: {e}"

# ------------ simple totals model (unchanged core) ------------
def _safe_num(d: dict, path: list, default: float) -> float:
    cur = d
    for k in path:
        if not isinstance(cur, dict): return default
        cur = cur.get(k)
        if cur is None: return default
    try:
        return float(str(cur).replace(",", "."))
    except Exception:
        return default

def _form_score(form_str: str) -> float:
    if not form_str: return 0.5
    pts = 0
    for ch in str(form_str).upper():
        if ch == 'W': pts += 3
        elif ch == 'D': pts += 1
    return pts / (max(1, len(str(form_str))) * 3)

def _quick_probs_from_team_stats(stats_home: dict, stats_away: dict):
    gfh = _safe_num(stats_home, ['goals','for','average','home'], 1.4)
    gfa = _safe_num(stats_away, ['goals','for','average','away'], 1.2)
    gah = _safe_num(stats_home, ['goals','against','average','home'], 1.1)
    gaa = _safe_num(stats_away, ['goals','against','average','away'], 1.1)
    exp_total = max(1.2, (gfh + gfa + gah + gaa) / 2)
    p_over25 = max(0.3, min(0.9, (exp_total - 1.8) / 1.6))
    # win shares (coarse)
    formH = _form_score(stats_home.get("form",""))
    formA = _form_score(stats_away.get("form",""))
    p_home = 0.45*formH + 0.25*(gfh - gaa + 1)/2 + 0.15
    p_away = 0.45*formA + 0.25*(gfa - gah + 1)/2
    scale = (p_home + p_away) or 1
    draw_share = 0.25
    p_home = (p_home/scale)*(1-draw_share); p_away=(p_away/scale)*(1-draw_share); p_draw=draw_share
    p_home = max(0.05, min(0.85, p_home))
    p_away = max(0.05, min(0.85, p_away))
    p_draw = max(0.10, min(0.35, 1 - (p_home + p_away)))
    p_o25 = p_over25
    p_o15 = min(0.95, 0.80 + (p_o25-0.55)*0.6)
    p_o05 = min(0.99, 0.90 + (p_o25-0.55)*0.4)
    return {
        "home_win": round(p_home*100),
        "draw": round(p_draw*100),
        "away_win": round(p_away*100),
        "over25": round(p_o25*100),
        "over15": round(p_o15*100),
        "over05": round(p_o05*100),
    }

def _team_id_any(team: str, league_slug: str):
    lg = AF_LEAGUE.get(league_slug, 39)
    tid = af.resolve_team_id(team, league_id=lg)
    return tid, lg

def _model_for_pair(h, a, league_slug: str):
    tidH, lg = _team_id_any(h, league_slug); tidA, _ = _team_id_any(a, league_slug)
    if not (tidH and tidA and lg): return None
    try:
        sH = af.get_team_stats(tidH, league_id=lg).get("response", {})
        sA = af.get_team_stats(tidA, league_id=lg).get("response", {})
        return _quick_probs_from_team_stats(sH, sA)
    except Exception as e:
        log_err("model_for_pair error", error=str(e), home=h, away=a)
        return None

# ------------ news ------------
def get_team_headlines(team: str, limit: int = 3):
    if not MEDIASTACK_KEY: return []
    url = "http://api.mediastack.com/v1/news"
    params = {"access_key": MEDIASTACK_KEY, "languages": "en", "keywords": team, "limit": limit, "sort": "published_desc"}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json().get("data", [])
        out = []
        for a in data[:limit]:
            src = a.get("source")
            title = a.get("title")
            published = (a.get("published_at") or "")[:16].replace("T", " ")
            if title:
                out.append(f"• {title} — {src} ({published})")
        return out
    except Exception:
        return []

# ------------ public APIs ------------
@csrf_exempt
def api_fixtures(request):
    league = (request.GET.get("league") or "epl").lower()
    try:
        days = int(request.GET.get("days", "8"))
    except Exception:
        days = 8
    start, end = _next_window(days)
    sd = start.date().isoformat(); ed = end.date().isoformat()
    sm_rows, _ = sm_try_fixtures(league, sd, ed)
    if sm_rows:
        return JsonResponse({"source": "sportmonks", "rows": len(sm_rows), "fixtures": sm_rows})
    af_rows, note = af_fixtures(league, sd, ed)
    return JsonResponse({"source": "api-football" if af_rows else "none", "rows": len(af_rows), "fixtures": af_rows, "note": note})

def _collect_upcoming_pairs(league: str, days=8):
    def collect(days_window: int):
        start, end = _next_window(days_window)
        sd = start.date().isoformat(); ed = end.date().isoformat()
        sm_rows, _ = sm_try_fixtures(league, sd, ed)
        rows = sm_rows or af_fixtures(league, sd, ed)[0]
        pairs = []
        seen = set()
        for r in rows or []:
            t = r.get("teams", {})
            h_raw = (t.get("home") or {}).get("name") or ""
            a_raw = (t.get("away") or {}).get("name") or ""
            h = _resolve_team(h_raw); a = _resolve_team(a_raw)
            if not h or not a: continue
            key = (h, a)
            if key in seen: continue
            seen.add(key)
            pairs.append(key)
        return pairs
    pairs = collect(days)
    if not pairs: pairs = collect(14)
    return pairs

@csrf_exempt
def api_random_pick(request):
    league = (request.GET.get("league") or "epl").lower()
    pairs = _collect_upcoming_pairs(league, days=8)
    if not pairs:
        return JsonResponse({"ok": False, "error": "No upcoming fixtures found."})
    random.shuffle(pairs)
    chosen = None
    model = None
    for h, a in pairs[:10]:
        model = _model_for_pair(h, a, league)
        if model:
            chosen = (h, a)
            break
    if not model or not chosen:
        return JsonResponse({"ok": False, "error": "I couldn’t fetch data for that matchup right now."})
    h, a = chosen
    pick = "Over 1.5 Goals"; conf = model["over15"]
    if model["over25"] >= 58: pick, conf = "Over 2.5 Goals", model["over25"]
    elif model["over05"] >= 90: pick, conf = "Over 0.5 Goals", model["over05"]
    return JsonResponse({
        "ok": True,
        "match": f"{h} vs {a}",
        "props": {"over05": model["over05"], "over15": model["over15"], "over25": model["over25"]},
        "final_pick": {"market": pick, "confidence": conf}
    })

@csrf_exempt
def api_best_bets(request):
    league = (request.GET.get("league") or "epl").lower()
    board = []
    for h, a in _collect_upcoming_pairs(league, days=8)[:24]:
        model = _model_for_pair(h, a, league)
        if not model: continue
        if model["over25"] >= 60:
            board.append({"match": f"{h} vs {a}", "market": "Over 2.5", "model_p": model["over25"]})
        elif model["over15"] >= 75:
            board.append({"match": f"{h} vs {a}", "market": "Over 1.5", "model_p": model["over15"]})
    board.sort(key=lambda x: x["model_p"], reverse=True)
    return JsonResponse({"ok": True, "results": board[:8], "count": len(board)})

# -------- Team REST (last5/last10 + news + summary) --------
def _team_last_n_payload(team_name: str, n: int, league_id: int = 39) -> Dict[str, Any]:
    tid = af.resolve_team_id(team_name, league_id=league_id)
    if not tid:
        msg = "Team not found. Make sure API_FOOTBALL_KEY is set and valid."
        return {"ok": False, "error": msg, "team": team_name}
    rows = af.team_last_n_results(tid, n=n, league_id=league_id)
    form = af.team_form_last_n(tid, n=n, league_id=league_id)
    return {"ok": True, "team": team_name, "n": n, "form": form, "results": rows}

@csrf_exempt
def api_team_last5(request, name: str):
    team = _resolve_team(name)
    return JsonResponse(_team_last_n_payload(team, 5, league_id=39))

@csrf_exempt
def api_team_last10(request, name: str):
    team = _resolve_team(name)
    return JsonResponse(_team_last_n_payload(team, 10, league_id=39))

@csrf_exempt
def api_team_news(request, name: str):
    team = _resolve_team(name)
    headlines = get_team_headlines(team, limit=5)
    return JsonResponse({"ok": True, "team": team, "headlines": headlines})

@csrf_exempt
def api_team_summary(request, name: str):
    """
    Averages from last N EPL matches: corners, yellow cards, red cards, fouls.
    GET /api/team/<name>/summary?n=5
    """
    team = _resolve_team(name)
    tid = af.resolve_team_id(team, league_id=39)
    if not tid:
        return JsonResponse({"ok": False, "error": "Team not found.", "team": team})
    try:
        n = int(request.GET.get("n", "5"))
    except Exception:
        n = 5
    n = max(3, min(10, n))
    fixtures = af.team_last_n_fixtures(tid, n=n, league_id=39)
    used = 0
    sums = {"corners": 0, "yellow": 0, "red": 0, "fouls": 0}
    for fx in fixtures:
        fid = fx.get("fixture_id") or 0
        home = (fx.get("home") or {}).get("id")
        away = (fx.get("away") or {}).get("id")
        if not fid or not home or not away:
            continue
        stats = af.fixture_stats_both(fid)
        side = home if home == tid else away if away == tid else None
        if not side or side not in stats:
            continue
        s = stats[side]
        sums["corners"] += int(s.get("corner kicks") or 0)
        sums["yellow"]  += int(s.get("yellow cards") or 0)
        sums["red"]     += int(s.get("red cards") or 0)
        sums["fouls"]   += int(s.get("fouls") or 0)
        used += 1
    if used == 0:
        return JsonResponse({"ok": True, "team": team, "n": n, "matches_used": 0, "note": "No stat feed found in last-N fixtures."})
    avgs = {k: round(v / used, 2) for k, v in sums.items()}
    return JsonResponse({"ok": True, "team": team, "n": n, "matches_used": used, "averages": avgs})

# --------- Chat (Ask) ---------
MIKE_SYSTEM = (
    "You are Mike — an elite, friendly football betting analyst focused on the English Premier League. "
    "Be direct and numeric. Always explain your angle clearly.\n"
    "FORMAT:\n"
    "1) 🔐 Best Props — 2–5 props with confidence %\n"
    "2) 🧾 Final Bet — choose ONE with confidence\n"
    "3) 💡 Why — 3–8 bullets (use form, goals, injuries/news if available)\n"
)

def _ask_llm(messages):
    if not OPENAI_OK: return None
    try:
        out = OPENAI.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.7)
        return out.choices[0].message.content
    except Exception as e:
        log_err("OpenAI error", error=str(e))
        return None

def _fmt_last_n_rows(tid: int, n: int = 5) -> str:
    rows = af.team_last_n_results(tid, n=n, league_id=39)
    if not rows: return "No recent matches found."
    lines = [f"Last {len(rows)} matches:"]
    for r in rows:
        lines.append(f"- {r['date']}: {r['venue']} vs {r['opponent']} — {r['score']} ({r['result']})")
    return "\n".join(lines)

def _find_last_weekend_fixture(team_id: int) -> Optional[Dict[str, Any]]:
    for d in _recent_weekend_dates():
        rows = af.team_fixtures_on_date(team_id, d, league_id=39)
        if rows:
            # if multiple, take the one with later kickoff time
            rows.sort(key=lambda x: (x.get("fixture") or {}).get("date",""))
            return rows[-1]
    return None

def _team_last_match_summary(team: str, when_hint: str = "") -> str:
    if not os.getenv("API_FOOTBALL_KEY"):
        return "API-Football key not configured. Set API_FOOTBALL_KEY to enable last-match stats."
    tid = af.resolve_team_id(team, league_id=39)
    if not tid:
        return f"I couldn't resolve the team '{team}'."

    fx = None
    q = when_hint.lower()
    if "last weekend" in q:
        fx = _find_last_weekend_fixture(tid)
    elif any(w in q for w in ["yesterday","last night","lastnight"]):
        y = timezone.localtime() - timedelta(days=1)
        rows = af.team_fixtures_on_date(tid, y.date().isoformat(), league_id=39)
        fx = rows[0] if rows else None
    if not fx:
        fx = af.team_last_fixture(tid, league_id=39)
    if not fx:
        return "I couldn't find a recent match for that team."

    f = fx.get("fixture", {})
    t = fx.get("teams", {})
    home = t.get("home") or {}; away = t.get("away") or {}
    score_ft = (fx.get("score") or {}).get("fulltime") or {}
    h, a = score_ft.get("home"), score_ft.get("away")
    when = (f.get("date","")[:16]).replace("T"," ")
    fid = f.get("id")

    stats = af.fixture_stats_both(fid) if fid else {}
    s_home = stats.get(home.get("id"), {}); s_away = stats.get(away.get("id"), {})

    lines = [f"{home.get('name')} {h}–{a} {away.get('name')} ({when})"]
    if not s_home and not s_away:
        lines.append("ℹ️ No stat feed for corners/cards/fouls on this fixture; showing result only.")
    else:
        def g(d: dict, k: str) -> int: return int(d.get(k, 0) or 0)
        corners = (g(s_home, "corner kicks"), g(s_away, "corner kicks"))
        yc = (g(s_home, "yellow cards"), g(s_away, "yellow cards"))
        rc = (g(s_home, "red cards"), g(s_away, "red cards"))
        fouls = (g(s_home, "fouls"), g(s_away, "fouls"))
        lines.append(f"Corners: {corners[0]}–{corners[1]}  |  Cards: Y {yc[0]}–{yc[1]}, R {rc[0]}–{rc[1]}  |  Fouls: {fouls[0]}–{fouls[1]}")

    lines.append(_fmt_last_n_rows(tid, n=5))
    return "\n".join(lines)

def _match_explainer(home, away, league_slug: str):
    # EPL-only scope guard
    if league_slug != "epl":
        return "I currently cover the English Premier League only."

    model = _model_for_pair(home, away, league_slug)
    if not model:
        return "I couldn’t fetch data for that matchup right now."

    # Form context
    tidH, _ = _team_id_any(home, "epl")
    tidA, _ = _team_id_any(away, "epl")
    fH = af.team_form_last_n(tidH, n=5, league_id=39) if tidH else {"form": "", "win_rate": 0}
    fA = af.team_form_last_n(tidA, n=5, league_id=39) if tidA else {"form": "", "win_rate": 0}

    # Injuries (top 2 names each if available)
    inj_lines = []
    try:
        injH = (af.get_injuries(tidH).get("response") if tidH else []) or []
        injA = (af.get_injuries(tidA).get("response") if tidA else []) or []
        if injH:
            names = [i.get("player", {}).get("name") for i in injH if i.get("player", {}).get("name")]
            if names: inj_lines.append(f"🩹 {home}: " + ", ".join(names[:2]))
        if injA:
            names = [i.get("player", {}).get("name") for i in injA if i.get("player", {}).get("name")]
            if names: inj_lines.append(f"🩹 {away}: " + ", ".join(names[:2]))
    except Exception:
        pass

    props = [
        ("Over 0.5 Goals", model["over05"]),
        ("Over 1.5 Goals", model["over15"]),
        ("Over 2.5 Goals", model["over25"]),
    ]
    pick, conf = "Over 1.5 Goals", model["over15"]
    if model["over25"] >= 58: pick, conf = "Over 2.5 Goals", model["over25"]
    elif model["over05"] >= 90: pick, conf = "Over 0.5 Goals", model["over05"]

    news_lines = []
    for team in (home, away):
        ns = get_team_headlines(team, limit=2)
        if ns:
            news_lines.append(f"🗞️ {team} news:\n" + "\n".join(ns))

    lines = [f"### {home} vs {away} — Analyst Note", ""]
    lines.append("#### 🔐 Best Props (model)")
    for name, pct in props:
        lines.append(f"- {name} — **{pct}%**")
    lines.append("")
    lines.append("#### 🧾 Final Bet")
    lines.append(f"- **{pick}** — Confidence: **{conf}%**")
    lines.append("")
    lines.append("#### 💡 Why")
    lines.append(f"- Recent form: {home} **{fH.get('form','')}** ({fH.get('win_rate',0)}%) vs {away} **{fA.get('form','')}** ({fA.get('win_rate',0)}%).")
    lines.append("- Goals averages and concession rates support a totals angle (2+ goals most likely).")
    lines.append("- Market shape suggests fair value on goals rather than 1X2.")
    if inj_lines:
        lines.append("- Key availability watch:\n  " + "\n  ".join(inj_lines))
    if news_lines:
        lines.append("")
        lines.extend(news_lines)
    return "\n".join(lines)

@method_decorator(csrf_exempt, name='dispatch')
class AskAssistant(APIView):
    def post(self, request):
        user_text = (request.data.get("question") or "").strip()
        if not user_text:
            return Response({"response": "⚠️ Please type a message."})
        ql = user_text.lower()

        # Fixtures list
        if any(k in ql for k in ["fixtures","what games","who is playing","schedule","this weekend","opening weekend"]):
            now, end = _next_window(8)
            sd = now.date().isoformat(); ed = end.date().isoformat()
            def pack(rows, title):
                if not rows: return f"📅 {title} fixtures (next 8 days): none found."
                lines = [f"📅 {title} fixtures (next 8 days):"]
                for fx in rows[:24]:
                    dt = (fx.get("fixture") or {}).get("date","")
                    t  = fx.get("teams") or {}
                    h  = (t.get("home") or {}).get("name","")
                    a  = (t.get("away") or {}).get("name","")
                    if h and a: lines.append(f"- {str(dt)[:16].replace('T',' ')}: {h} vs {a}")
                return "\n".join(lines)
            e_rows, _ = sm_try_fixtures("epl", sd, ed); 
            if not e_rows: e_rows, _ = af_fixtures("epl", sd, ed)
            l_rows, _ = sm_try_fixtures("laliga", sd, ed); 
            if not l_rows: l_rows, _ = af_fixtures("laliga", sd, ed)
            reply = pack(e_rows, "EPL") + "\n\n" + pack(l_rows, "La Liga")
            return Response({"response": reply})

        # Best bets
        if "best bets" in ql or "value board" in ql:
            req = request._request; req.GET = req.GET.copy(); req.GET["league"] = "epl"
            data = json.loads(api_best_bets(req).content.decode("utf-8"))
            if not data.get("results"):
                return Response({"response": "📋 No strong model edges found right now."})
            lines = ["📋 Weekend Value Board — model-only edges"]
            for r in data["results"]:
                lines.append(f"- {r['match']}: {r['market']} (model {r['model_p']}%)")
            return Response({"response": "\n".join(lines)})

        # Random pick / banker
        if "random" in ql or "banker" in ql or "small odds" in ql:
            req = request._request; req.GET = req.GET.copy(); req.GET["league"] = "epl"
            r = json.loads(api_random_pick(req).content.decode("utf-8"))
            if not r.get("ok"):
                return Response({"response": "I couldn’t fetch data for that matchup right now."})
            text = (f"🎲 **Random pick** — {r['match']}\n\n"
                    f"🔐 **Best Props (model)**\n"
                    f"- Over 0.5 Goals — **{r['props']['over05']}%**\n"
                    f"- Over 1.5 Goals — **{r['props']['over15']}%**\n"
                    f"- Over 2.5 Goals — **{r['props']['over25']}%**\n\n"
                    f"🧾 **Final Bet**\n- {r['final_pick']['market']} — Confidence: **{r['final_pick']['confidence']}%**")
            return Response({"response": text})

        # Single-team realtime Qs (now includes 'last weekend')
        if any(w in ql for w in ["last weekend","last match","yesterday","last night","lastnight","result","score","what did","corners","cards","fouls"]):
            team = _guess_team(user_text)
            if team:
                return Response({"response": _team_last_match_summary(team, when_hint=ql)})

        # Explicit matchup (EPL only)
        h, a = _extract_teams(user_text)
        if h and a:
            return Response({"response": _match_explainer(h, a, "epl")})

        # Fallback: plain LLM (still EPL-oriented instructions)
        if OPENAI_OK:
            messages = [{"role":"system","content": MIKE_SYSTEM},
                        {"role":"user","content": user_text}]
            txt = _ask_llm(messages)
            if txt:
                return Response({"response": txt})

        return Response({"response": "Ask for EPL fixtures, a random pick, best bets, a specific match (e.g., “Liverpool vs Bournemouth”), or a team’s last match (e.g., “What did Arsenal play last weekend?”)."})
