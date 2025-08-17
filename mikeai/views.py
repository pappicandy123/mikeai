# assistant/views.py — Mike v3: multi-league (EPL+La Liga), real-time fixtures,
# improved best bets / random pick with Over 0.5, 1.5, 2.5, 1xBet-aware odds.

import os
import re
import random
import unicodedata
from datetime import timedelta

from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from rest_framework.views import APIView
from rest_framework.response import Response

import requests

from . import api_football as af

# -------- OpenAI client (optional for write-ups)
try:
    from openai import OpenAI
    OPENAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    OPENAI_OK = True
except Exception:
    OPENAI, OPENAI_OK = None, False

MODEL_NAME = "gpt-4o-mini"

# -------- External keys
ODDS_API_KEY = os.getenv("ODDS_API_KEY")       # The Odds API v4
MEDIASTACK_KEY = os.getenv("MEDIASTACK_KEY")   # mediastack.com

# -------- Leagues we support
LEAGUE_IDS = {
    "EPL": 39,
    "LaLiga": 140,
}
SPORT_KEYS = {
    39: "soccer_epl",
    140: "soccer_spain_la_liga",
}

# =========================================================
# UI
# =========================================================
def chat_page(request):
    return render(request, "index.html")


# =========================================================
# Session memory (short)
# =========================================================
def _push_history(request, role: str, content: str):
    hist = request.session.get("conv_hist", [])
    hist.append({"role": role, "content": content})
    if len(hist) > 24:
        hist = hist[-24:]
    request.session["conv_hist"] = hist
    request.session.modified = True

def _recent_history(request, limit=12):
    return request.session.get("conv_hist", [])[-limit:]

def _remember_teams(request, t1=None, t2=None):
    if t1: request.session["last_team1"] = t1
    if t2: request.session["last_team2"] = t2
    request.session.modified = True

def _last_teams(request):
    return request.session.get("last_team1"), request.session.get("last_team2")


# =========================================================
# Team parsing / normalization
# =========================================================
NICKNAMES = {
    # EPL + some LaLiga short forms
    "spurs":"Tottenham Hotspur","tottenham":"Tottenham Hotspur",
    "man city":"Manchester City","manchester city":"Manchester City","city":"Manchester City",
    "man united":"Manchester United","man utd":"Manchester United","manchester united":"Manchester United","united":"Manchester United",
    "west ham":"West Ham United","westham":"West Ham United",
    "wolves":"Wolverhampton Wanderers","wolverhampton":"Wolverhampton Wanderers","wolverhampton wanderers":"Wolverhampton Wanderers",
    "bournemouth":"AFC Bournemouth","afc bournemouth":"AFC Bournemouth",
    "newcastle":"Newcastle United",
    "forest":"Nottingham Forest","nottingham forest":"Nottingham Forest",
    "villa":"Aston Villa","aston villa":"Aston Villa",
    "liverpool":"Liverpool","chelsea":"Chelsea","arsenal":"Arsenal","everton":"Everton",
    "fulham":"Fulham","brentford":"Brentford","crystal palace":"Crystal Palace","palace":"Crystal Palace",
    "ipswich":"Ipswich Town","leicester":"Leicester City","southampton":"Southampton","brighton":"Brighton & Hove Albion","brighton and hove albion":"Brighton & Hove Albion","brighton":"Brighton & Hove Albion",
    "leeds":"Leeds United","burnley":"Burnley","sunderland":"Sunderland",
    # La Liga
    "barcelona":"Barcelona","real madrid":"Real Madrid","madrid":"Real Madrid",
    "atletico":"Atletico Madrid","atletico madrid":"Atletico Madrid",
    "real sociedad":"Real Sociedad","athletic club":"Athletic Club","athletic bilbao":"Athletic Club",
    "sevilla":"Sevilla","villarreal":"Villarreal","valencia":"Valencia","betis":"Real Betis",
    "osasuna":"Osasuna","celta":"Celta Vigo","celta vigo":"Celta Vigo",
    "girona":"Girona","getafe":"Getafe","alaves":"Alaves","mallorca":"Mallorca","rayo":"Rayo Vallecano","rayo vallecano":"Rayo Vallecano",
    "espanyol":"Espanyol","elche":"Elche","oviedo":"Oviedo","levante":"Levante",
}

def _norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    for junk in [" football club"," afc"," fc"," cf",".",",","-","_","|","/","\\","(",")","[","]","{","}",":",";","•","—"]:
        s = s.replace(junk, " ")
    return " ".join(s.split())

def _resolve_team(raw: str) -> str:
    if not raw: return ""
    x = _norm(raw)
    x_nospace = x.replace(" ", "")
    for k, v in NICKNAMES.items():
        kn = k.replace(" ", "")
        if x == k or x_nospace == kn or k in x or kn in x_nospace:
            return v
    for team_key, full in NICKNAMES.items():
        if team_key.replace(" ", "") in x_nospace:
            return full
    return " ".join(w.capitalize() for w in raw.strip().split())

def _extract_teams(text: str):
    if not text: return (None, None)
    q = _norm(text)
    m = re.search(r"(.+?)\s+(?:vs|v|versus|and|against|between)\s+(.+)", q, flags=re.I)
    if not m: return (None, None)
    return _resolve_team(m.group(1)), _resolve_team(m.group(2))

def _find_team_id_any(team_name: str, season: int | None = None):
    """
    Try all supported leagues for this season; return (team_id, league_id) or (None, None).
    """
    if not team_name:
        return None, None
    if season is None:
        season = af.current_season()
    # try league-specific first (EPL), then La Liga
    for lg in (LEAGUE_IDS["EPL"], LEAGUE_IDS["LaLiga"]):
        tid = af.search_team_id(team_name, league_id=lg, season=season)
        if tid:
            return tid, lg
    # final fallback: global search without league filter
    tid = af.search_team_id(team_name, league_id=LEAGUE_IDS["EPL"], season=season) or None
    if tid:
        return tid, LEAGUE_IDS["EPL"]
    return None, None


# =========================================================
# Persona / LLM
# =========================================================
MIKE_SYSTEM = (
    "You are Mike — an elite, friendly sports betting analyst who can also answer ANY question like ChatGPT.\n"
    "\n"
    "FORMAT for sports picks:\n"
    "1) 🔐 Best Props — 2–5 props with confidence %\n"
    "2) 💰 Value Bets — call out +EV edges if model prob > implied by >=3%\n"
    "3) 🧾 Final Bet — pick ONE (use Over 0.5/1.5/2.5 goals when totals mentioned)\n"
    "4) 💡 Why — 3–8 bullets (form, tactics, H2H, injuries, pace)\n"
    "\n"
    "DATA:\n"
    "- Use API-Football stats/H2H/injuries for 2025/26 when available.\n"
    "- Use The Odds API v4 odds; mention bookmaker names if present (e.g., 1xBet).\n"
    "- If exact numbers are missing, give directional guidance; do not fabricate specific unused stats.\n"
)

def _ask_llm(messages):
    if not OPENAI_OK or not OPENAI:
        return "Mike is currently offline. Try again shortly."
    out = OPENAI.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.7)
    return out.choices[0].message.content


# =========================================================
# Odds API helpers (v4)
# =========================================================
ODDS_BASE = "https://api.the-odds-api.com/v4"
BOOKMAKERS_DEFAULT = "1xbet,bet365,unibet,williamhill,pinnacle,paddypower"

def _iso_z(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def _oddsapi_fetch_odds(sport_key: str,
                        markets: str = "h2h,totals",
                        regions: str = "uk,eu",
                        bookmakers: str | None = BOOKMAKERS_DEFAULT,
                        t_from_iso: str | None = None,
                        t_to_iso: str | None = None):
    if not ODDS_API_KEY:
        return None, "Set ODDS_API_KEY."
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "dateFormat": "iso",
        "oddsFormat": "decimal",
        "includeLinks": "true",
        "includeSids": "true",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    if t_from_iso:
        params["commenceTimeFrom"] = t_from_iso
    if t_to_iso:
        params["commenceTimeTo"] = t_to_iso
    try:
        r = requests.get(f"{ODDS_BASE}/sports/{sport_key}/odds", params=params, timeout=20)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"Odds API error: {e}"

def _best_price_line(pairs):
    if not pairs: return None, []
    mx = max(p for p, _ in pairs if p)
    books = [bk for p, bk in pairs if p == mx]
    return mx, books

def _implied_prob(odds: float) -> float:
    return 0.0 if not odds or odds <= 1.0 else 1.0/odds

def _value_check(prob: float, odds: float, edge: float = 0.03) -> bool:
    return prob - _implied_prob(odds) >= edge


# =========================================================
# Time windows
# =========================================================
def _next_8_days_window():
    now = timezone.now()
    start = now
    end = now + timedelta(days=8)
    return start, end

def _weekend_window_local():
    now = timezone.localtime()
    fri = (now + timedelta(days=(4 - now.weekday()) % 7)).replace(hour=0, minute=0, second=0, microsecond=0)
    sun = fri + timedelta(days=2, hours=23, minutes=59)
    return fri, sun


# =========================================================
# News (optional)
# =========================================================
def _get_team_headlines(team: str, limit: int = 3):
    if not MEDIASTACK_KEY:
        return []
    url = "http://api.mediastack.com/v1/news"
    params = {
        "access_key": MEDIASTACK_KEY,
        "languages": "en",
        "keywords": team,
        "limit": limit,
        "sort": "published_desc",
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        out = []
        for a in data.get("data", [])[:limit]:
            src = a.get("source")
            title = a.get("title")
            published = a.get("published_at", "")[:16].replace("T", " ")
            if title:
                out.append(f"• {title} — {src} ({published})")
        return out
    except Exception:
        return []


# =========================================================
# Fixtures (both leagues, next 8 days)
# =========================================================
def fixtures_skill_both_leagues():
    start, end = _next_8_days_window()
    start_d = start.date().isoformat(); end_d = end.date().isoformat()
    out_lines = []

    for name, lgid in (("EPL", LEAGUE_IDS["EPL"]), ("La Liga", LEAGUE_IDS["LaLiga"])):
        try:
            data = af.get_fixtures_by_range(start_d, end_d, league_id=lgid)
            arr = data.get("response", [])
            if not arr:
                out_lines.append(f"📅 {name} fixtures (next 8 days): none found.")
                continue
            lines = [f"📅 {name} fixtures (next 8 days):"]
            for fx in arr:
                f = fx.get("fixture", {})
                t = fx.get("teams", {})
                dt_iso = (f.get("date") or "")[:16].replace("T", " ")
                home = (t.get("home") or {}).get("name")
                away = (t.get("away") or {}).get("name")
                if home and away:
                    lines.append(f"- {dt_iso}: {home} vs {away}")
            out_lines.append("\n".join(lines))
        except Exception:
            out_lines.append(f"📅 {name} fixtures: API error.")
    return "\n\n".join(out_lines)


# =========================================================
# Model helpers
# =========================================================
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
    formH = _form_score(stats_home.get("form",""))
    formA = _form_score(stats_away.get("form",""))

    gfh = _safe_num(stats_home, ['goals','for','average','home'], 1.4)
    gfa = _safe_num(stats_away, ['goals','for','average','away'], 1.2)
    gah = _safe_num(stats_home, ['goals','against','average','home'], 1.1)
    gaa = _safe_num(stats_away, ['goals','against','average','away'], 1.1)

    exp_total = max(1.2, (gfh + gfa + gah + gaa) / 2)
    p_over25 = max(0.3, min(0.9, (exp_total - 1.8) / 1.6))

    p_home = 0.45*formH + 0.25*(gfh - gaa + 1)/2 + 0.15
    p_away = 0.45*formA + 0.25*(gfa - gah + 1)/2
    scale = (p_home + p_away) or 1
    draw_share = 0.25
    p_home = (p_home/scale)*(1-draw_share); p_away=(p_away/scale)*(1-draw_share); p_draw=draw_share

    p_home = max(0.05, min(0.85, p_home))
    p_away = max(0.05, min(0.85, p_away))
    p_draw = max(0.10, min(0.35, 1 - (p_home + p_away)))

    # simple transforms for Over 0.5 / 1.5 using Over 2.5 anchor
    p_o25 = p_over25
    p_o15 = min(0.95, 0.80 + (p_o25-0.55)*0.6)   # heuristic
    p_o05 = min(0.99, 0.90 + (p_o25-0.55)*0.4)   # heuristic

    return {
        "home_win": round(p_home*100),
        "draw": round(p_draw*100),
        "away_win": round(p_away*100),
        "over25": round(p_o25*100),
        "over15": round(p_o15*100),
        "over05": round(p_o05*100),
    }


# =========================================================
# Direct Q&A helpers (multi-league aware)
# =========================================================
def _team_id_or_none_any(team_name: str):
    try:
        return _find_team_id_any(team_name)
    except Exception:
        return (None, None)

def avg_goals_last_season(team_name: str):
    tid, lg = _team_id_or_none_any(team_name)
    if not tid:
        return None, "I couldn't resolve the team."
    try:
        stats = af.get_team_stats(tid, league_id=lg).get("response", {})
        g_for_total = _safe_num(stats, ['goals','for','average','total'], None)
        if g_for_total is None:
            gh = _safe_num(stats, ['goals','for','average','home'], 0.0)
            ga = _safe_num(stats, ['goals','for','average','away'], 0.0)
            g_for_total = round((gh+ga)/2, 2)
        return round(g_for_total, 2), None
    except Exception:
        return None, "Stats unavailable right now."

def team_avg_corners_last_n(team_name: str, n=5):
    tid, lg = _team_id_or_none_any(team_name)
    if not tid:
        return None, None, None, "I couldn't resolve the team."
    try:
        season = af.current_season()
        rec = af.team_recent_corners(tid, league_id=lg, season=season, last_n=n)
        return rec.get("team_avg"), rec.get("total_avg"), rec.get("n"), None
    except Exception:
        return None, None, None, "Corners data unavailable."

def player_assists_this_season(player_name: str):
    try:
        # Try EPL first, then LaLiga
        for lg in (LEAGUE_IDS["EPL"], LEAGUE_IDS["LaLiga"]):
            data = af.player_assists_current(player_name, league_id=lg)
            if data:
                return data.get("assists"), data.get("team"), None
        return None, None, "I couldn't find current-season assists for that player."
    except Exception:
        return None, None, "Player stats unavailable right now."

def team_home_performance_last_n(team_name: str, n=10):
    tid, lg = _team_id_or_none_any(team_name)
    if not tid:
        return None, "I couldn't resolve the team."
    try:
        season = af.current_season()
        fixtures = af.team_last_home_fixtures(tid, last_n=n, league_id=lg, season=season)
        if not fixtures:
            return None, "No recent home fixtures found."
        w = d = l = gf = ga = 0
        for fx in fixtures:
            t = fx.get("teams", {})
            sc = fx.get("score", {}).get("fulltime", {})
            home_id = t.get("home", {}).get("id")
            if home_id != tid:
                continue
            h = sc.get("home") or 0
            a = sc.get("away") or 0
            gf += (h or 0); ga += (a or 0)
            if h > a: w += 1
            elif h == a: d += 1
            else: l += 1
        return {"W": w, "D": d, "L": l, "GF": gf, "GA": ga, "N": w+d+l}, None
    except Exception:
        return None, "Couldn’t compute home performance right now."


# =========================================================
# Q&A dispatcher
# =========================================================
def qa_skill_direct(question: str, request):
    q = question.lower().strip()

    # Average goals last season (team)
    if "average goals" in q and "last" in q and ("season" in q or "epl" in q or "la liga" in q):
        m = re.search(r"what(?:'s| is)? the average goals (?:scored )?by (.+?) (?:in|last)\b", q)
        team = _resolve_team(m.group(1)) if m else _last_teams(request)[0]
        if not team:
            return "Tell me the team (e.g., 'average goals by Manchester United last season')."
        avg, err = avg_goals_last_season(team)
        if err: return err
        return f"📊 {team} — average goals per match (last season): **{avg}**"

    # Average corners last 5 (team)
    if "average corners" in q and ("last 5" in q or "last five" in q):
        m = re.search(r"(?:for|by|of)\s+([a-zA-Z\s\.-]+)", question, flags=re.I)
        team = _resolve_team(m.group(1)) if m else _last_teams(request)[0]
        if not team:
            return "Tell me the team (e.g., 'Arsenal average corners last 5')."
        ta, tot, n, err = team_avg_corners_last_n(team, n=5)
        if err: return err
        return (f"📐 **{team} — corners (last {n})**\n"
                f"- Team avg: **{ta}** | Match total avg: **{tot}**")

    # Player assists this season
    if "assists" in q and ("this season" in q or "current season" in q):
        m = re.search(r"how many assists does\s+(.+?)\s+have", q)
        player = m.group(1).strip() if m else None
        if not player:
            return "Tell me the player (e.g., 'How many assists does Jude Bellingham have this season?')."
        assists, team_name, err = player_assists_this_season(player)
        if err: return err
        team_note = f" for **{team_name}**" if team_name else ""
        return f"🎯 **{player}** has **{assists}** assists this season{team_note}."

    # Home performance last 10 (team)
    if ("home performance" in q or "home form" in q) and ("last 10" in q or "last ten" in q):
        m = re.search(r"(?:show me|what is|what's)?\s*([a-zA-Z\s\.-]+?)\s+home performance", question, flags=re.I)
        team = _resolve_team(m.group(1)) if m else _last_teams(request)[0]
        if not team:
            return "Tell me the team (e.g., 'Show me Brighton home performance in the last 10 games')."
        perf, err = team_home_performance_last_n(team, n=10)
        if err: return err
        return (f"🏟️ **{team} — last {perf['N']} home games**\n"
                f"- Record: **{perf['W']}W {perf['D']}D {perf['L']}L**\n"
                f"- Goals: **{perf['GF']} for / {perf['GA']} against**")

    return None  # not handled


# =========================================================
# Match analysis (with odds + model). Multi-league aware.
# =========================================================
def _get_match_model_and_odds(team1: str, team2: str):
    """
    Returns (model_dict, odds_summary_dict, league_id) or (None,None,None)
    """
    season = af.current_season()
    id1, lg1 = _find_team_id_any(team1, season=season)
    id2, lg2 = _find_team_id_any(team2, season=season)
    if not (id1 and id2):
        return None, None, None
    # if they decoded to different leagues (rare), prefer EPL if any is EPL
    league_id = lg1 if lg1 == lg2 else (LEAGUE_IDS["EPL"] if (lg1 == LEAGUE_IDS["EPL"] or lg2 == LEAGUE_IDS["EPL"]) else lg1)

    sH = af.get_team_stats(id1, league_id=league_id).get("response", {})
    sA = af.get_team_stats(id2, league_id=league_id).get("response", {})
    model = _quick_probs_from_team_stats(sH, sA)

    # odds window (next 8 days)
    t_from, t_to = _next_8_days_window()
    sport_key = SPORT_KEYS.get(league_id, "soccer_epl")
    data, err = _oddsapi_fetch_odds(
        sport_key=sport_key,
        markets="h2h,totals",
        regions="uk,eu",
        bookmakers=BOOKMAKERS_DEFAULT,
        t_from_iso=_iso_z(t_from),
        t_to_iso=_iso_z(t_to),
    )
    odds_summary = {}
    if not err and isinstance(data, list):
        t1 = team1.lower(); t2 = team2.lower()
        for g in data:
            home = (g.get("home_team") or "").lower()
            away = (g.get("away_team") or "").lower()
            if not ((t1 in home and t2 in away) or (t1 in away and t2 in home)):
                continue
            odds_summary = {"1x2": {"home":[], "draw":[], "away":[]}, "totals_goals": {}}
            for b in g.get("bookmakers", []):
                book = b.get("title") or b.get("key") or "bookmaker"
                for m in b.get("markets", []):
                    key = (m.get("key") or "").lower()
                    if key == "h2h":
                        for o in m.get("outcomes", []):
                            nm = (o.get("name") or "")
                            pr = o.get("price")
                            nm_low = nm.lower()
                            if nm_low == home:
                                odds_summary["1x2"]["home"].append((pr, book))
                            elif nm_low == away:
                                odds_summary["1x2"]["away"].append((pr, book))
                            elif nm_low == "draw":
                                odds_summary["1x2"]["draw"].append((pr, book))
                    elif key == "totals":
                        for o in m.get("outcomes", []):
                            nm = (o.get("name") or "")
                            pr = o.get("price")
                            pt = o.get("point")
                            if pt is None: 
                                continue
                            label = f"Over {pt}" if nm.lower()=="over" else f"Under {pt}"
                            odds_summary["totals_goals"].setdefault(label, []).append((pr, book))
            break

    return model, odds_summary or {}, league_id


def match_skill(question: str, request):
    # detect teams
    t1p = request.data.get("team1"); t2p = request.data.get("team2")
    if t1p and t2p: t1, t2 = _resolve_team(t1p), _resolve_team(t2p)
    else: t1, t2 = _extract_teams(question)
    if not (t1 and t2):
        lt1, lt2 = _last_teams(request)
        t1, t2 = t1 or lt1, t2 or lt2
    if not (t1 and t2):
        return "Tell me two teams (e.g., 'Liverpool vs Bournemouth')."

    _remember_teams(request, t1, t2)

    model, odds, league_id = _get_match_model_and_odds(t1, t2)
    if not model:
        return "I couldn’t fetch data for that matchup right now."

    # Build analyst note
    lines = []
    lines.append(f"### {t1} vs {t2} — Analyst Note")
    lines.append("")
    lines.append("#### 🔐 Best Props (model)")
    lines.append(f"- Over 0.5 Goals — **{model['over05']}%**")
    lines.append(f"- Over 1.5 Goals — **{model['over15']}%**")
    lines.append(f"- Over 2.5 Goals — **{model['over25']}%**")

    # Odds best prices
    if odds:
        def pick_best(lst): 
            px, books = _best_price_line(lst)
            if not px: return None
            bks = ", ".join(books[:3])
            return f"{px} ({bks})"
        # prefer 1xBet if tied in bests? We'll just display books list already includes names.
        one = odds.get("1x2", {})
        tot = odds.get("totals_goals", {})
        best_o25 = None
        for lbl, pairs in tot.items():
            if "Over 2.5" in lbl:
                px, books = _best_price_line(pairs)
                if px:
                    best_o25 = (px, books)
                    break
        lines.append("")
        lines.append("#### 📈 Odds (best prices)")
        if one:
            for side, lab in (("home","Home"),("draw","Draw"),("away","Away")):
                s = pick_best(one.get(side, []))
                if s: lines.append(f"- {lab}: {s}")
        if best_o25:
            px, books = best_o25
            lines.append(f"- Over 2.5: {px} ({', '.join(books[:3])})")

    # Value check (loose threshold so we surface picks)
    values = []
    if odds and odds.get("1x2"):
        for side, key in (("home","home_win"), ("draw","draw"), ("away","away_win")):
            best, _ = _best_price_line(odds["1x2"].get(side, []))
            if best:
                p = model[key] / 100.0
                if _value_check(p, best, edge=0.03):
                    values.append(f"1X2 {side.title()} is +EV (model {round(p*100)}% vs implied {round(_implied_prob(best)*100)}%)")
    if odds and odds.get("totals_goals"):
        for lbl, pairs in odds["totals_goals"].items():
            if "Over 2.5" in lbl:
                best, _ = _best_price_line(pairs)
                if best:
                    p = model["over25"]/100.0
                    if _value_check(p, best, edge=0.03):
                        values.append(f"Over 2.5 Goals is +EV (model {round(p*100)}% vs implied {round(_implied_prob(best)*100)}%)")

    lines.append("")
    if values:
        lines.append("#### 💰 Value Bets")
        for v in values[:4]:
            lines.append(f"- {v}")
    else:
        lines.append("#### 💰 Value Bets")
        lines.append("- No clear +EV edges by price right now — using model-only props above.")

    # Final bet: choose the best of Over 0.5 / 1.5 / 2.5 by a simple rule with odds presence
    final_pick = "Over 1.5 Goals"
    if model["over25"] >= 58:
        final_pick = "Over 2.5 Goals"
    elif model["over05"] >= 90:
        final_pick = "Over 0.5 Goals"
    final_conf = max(model["over25"], model["over15"], model["over05"])
    lines.append("")
    lines.append("#### 🧾 Final Bet")
    lines.append(f"- **{final_pick}** — Confidence: **{final_conf}%**")

    # Short “why”
    lines.append("")
    lines.append("#### 💡 Why")
    lines.append("- Form and goals averages support a goals-friendly script.")
    lines.append("- Both sides create chances; defenses concede at league-average or worse under pressure.")
    lines.append("- H2H trend leans to goals in recent meetings.")
    lines.append("- Market prices aren’t aggressively shading totals (modest edges where noted).")

    return "\n".join(lines)


# =========================================================
# Quick skills
# =========================================================
def fixtures_skill_api_only():
    return fixtures_skill_both_leagues()

def random_pick_skill():
    """
    Pick a random upcoming EPL/LaLiga game in the next 8 days.
    Output Over 0.5/1.5/2.5 model props and best price if available.
    """
    start, end = _next_8_days_window()
    start_d = start.date().isoformat(); end_d = end.date().isoformat()
    fixtures = []
    for lg in (LEAGUE_IDS["EPL"], LEAGUE_IDS["LaLiga"]):
        try:
            data = af.get_fixtures_by_range(start_d, end_d, league_id=lg)
            for fx in data.get("response", []):
                t = fx.get("teams", {})
                home = (t.get("home") or {}).get("name")
                away = (t.get("away") or {}).get("name")
                if home and away:
                    fixtures.append((home, away, lg))
        except Exception:
            continue
    if not fixtures:
        return "No priced markets found for EPL/La Liga in the next few days."

    home, away, lg = random.choice(fixtures)
    model, odds, _ = _get_match_model_and_odds(home, away)
    if not model:
        return "Random pick error — no model."

    lines = [f"🎲 **Random pick** — {home} vs {away} ({'EPL' if lg==LEAGUE_IDS['EPL'] else 'La Liga'})",
             "",
             "🔐 **Best Props (model)**",
             f"- Over 0.5 Goals — **{model['over05']}%**",
             f"- Over 1.5 Goals — **{model['over15']}%**",
             f"- Over 2.5 Goals — **{model['over25']}%**"]

    if odds:
        def best_label(key):
            px, bks = _best_price_line(key)
            if not px: return None
            return f"{px} ({', '.join(bks[:3])})"
        one = odds.get("1x2", {})
        tot = odds.get("totals_goals", {})
        best25 = None
        for lbl, pairs in tot.items():
            if "Over 2.5" in lbl:
                best25 = best_label(pairs)
                break
        lines.append("")
        lines.append("📈 **Odds (best prices)**")
        if one:
            for side, lab in (("home","Home"),("draw","Draw"),("away","Away")):
                s = best_label(one.get(side, []))
                if s: lines.append(f"- {lab}: {s}")
        if best25:
            lines.append(f"- Over 2.5: {best25}")

    # Final pick: choose strongest of over lines
    pick = "Over 1.5 Goals"
    conf = model["over15"]
    if model["over25"] >= 58:
        pick, conf = "Over 2.5 Goals", model["over25"]
    elif model["over05"] >= 90:
        pick, conf = "Over 0.5 Goals", model["over05"]

    lines.append("")
    lines.append("🧾 **Final Bet**")
    lines.append(f"- {pick} — Confidence: **{conf}%**")

    return "\n".join(lines)


# =========================================================
# Weekend Value Board (both leagues, with softer threshold + model fallback)
# =========================================================
def weekend_value_board():
    fri, sun = _weekend_window_local()
    start = fri.date().isoformat(); end = sun.date().isoformat()

    board = []

    for lg in (LEAGUE_IDS["EPL"], LEAGUE_IDS["LaLiga"]):
        try:
            fx = af.get_fixtures_by_range(start, end, league_id=lg).get("response", [])
            if not fx:
                continue
            sport_key = SPORT_KEYS.get(lg, "soccer_epl")
            oddsdata, err = _oddsapi_fetch_odds(
                sport_key=sport_key,
                markets="h2h,totals",
                regions="uk,eu",
                bookmakers=BOOKMAKERS_DEFAULT,
                t_from_iso=_iso_z(fri),
                t_to_iso=_iso_z(sun),
            )
            # index odds by normalized pair
            odds_index = {}
            if not err and isinstance(oddsdata, list):
                for g in oddsdata:
                    h = (g.get("home_team") or "").lower()
                    a = (g.get("away_team") or "").lower()
                    odds_index[(h,a)] = g

            for row in fx:
                teams = row.get("teams", {})
                home = (teams.get("home") or {}).get("name")
                away = (teams.get("away") or {}).get("name")
                if not (home and away): 
                    continue

                idH, _ = _find_team_id_any(home)
                idA, _ = _find_team_id_any(away)
                if not (idH and idA): 
                    continue

                sH = af.get_team_stats(idH, league_id=lg).get("response", {})
                sA = af.get_team_stats(idA, league_id=lg).get("response", {})
                model = _quick_probs_from_team_stats(sH, sA)

                # build summary from odds_index if present
                g = odds_index.get((home.lower(), away.lower())) or odds_index.get((away.lower(), home.lower()))
                lines = []
                bests = []
                if g:
                    out = {"1x2":{"home":[],"draw":[],"away":[]}, "totals":{}}
                    for b in g.get("bookmakers", []):
                        book = b.get("title") or b.get("key")
                        for m in b.get("markets", []):
                            key = (m.get("key") or "").lower()
                            if key == "h2h":
                                for o in m.get("outcomes", []):
                                    nm = (o.get("name") or "").lower(); pr = o.get("price")
                                    if nm == (g.get("home_team") or "").lower():
                                        out["1x2"]["home"].append((pr, book))
                                    elif nm == (g.get("away_team") or "").lower():
                                        out["1x2"]["away"].append((pr, book))
                                    elif nm == "draw":
                                        out["1x2"]["draw"].append((pr, book))
                            elif key == "totals":
                                for o in m.get("outcomes", []):
                                    nm = (o.get("name") or "")
                                    pr = o.get("price"); pt = o.get("point")
                                    if pt is None: continue
                                    lbl = f"Over {pt}" if nm.lower()=="over" else f"Under {pt}"
                                    out["totals"].setdefault(lbl, []).append((pr, book))

                    # check +EV for 1x2 and Over 2.5
                    rows = []
                    for side, key in (("home","home_win"), ("draw","draw"), ("away","away_win")):
                        best, _ = _best_price_line(out["1x2"].get(side, []))
                        if best:
                            p = model[key] / 100.0
                            edge = (p - _implied_prob(best))*100
                            rows.append(("1X2", side, best, round(p*100), round(edge,1)))
                    for lbl, pairs in (out.get("totals") or {}).items():
                        if "Over 2.5" in lbl:
                            best, _ = _best_price_line(pairs)
                            if best:
                                p = model["over25"]/100.0
                                edge = (p - _implied_prob(best))*100
                                rows.append(("Over 2.5", lbl, best, round(p*100), round(edge,1)))
                    rows.sort(key=lambda x: x[4], reverse=True)
                    if rows and rows[0][4] >= 3.0:  # relaxed to +3%
                        bests.append({
                            "match": f"{home} vs {away}",
                            "market": rows[0][0],
                            "selection": rows[0][1],
                            "odds": rows[0][2],
                            "model_p": rows[0][3],
                            "edge_pct": rows[0][4],
                            "league": "EPL" if lg==LEAGUE_IDS["EPL"] else "La Liga",
                        })

                # If no odds edges, still allow a model-led Over pick (to avoid empty board)
                if not bests:
                    if model["over25"] >= 60:
                        bests.append({"match": f"{home} vs {away}","market":"Over 2.5","selection":"Goals","odds":"—","model_p":model["over25"],"edge_pct":0,"league": "EPL" if lg==LEAGUE_IDS["EPL"] else "La Liga"})
                    elif model["over15"] >= 75:
                        bests.append({"match": f"{home} vs {away}","market":"Over 1.5","selection":"Goals","odds":"—","model_p":model["over15"],"edge_pct":0,"league": "EPL" if lg==LEAGUE_IDS["EPL"] else "La Liga"})

                board.extend(bests)

        except Exception:
            continue

    if not board:
        return "No clear +EV edges found in EPL or La Liga for the upcoming period."

    board.sort(key=lambda x: (x["odds"] if isinstance(x["odds"], (int,float)) else 0, -x["model_p"]), reverse=True)
    lines = [f"📋 Weekend Value Board — picks ({timezone.now().date()})"]
    for b in board[:8]:
        oddst = b["odds"] if isinstance(b["odds"], (int,float)) else "—"
        lines.append(f"- [{b['league']}] {b['match']}: {b['market']} {b['selection']} @ {oddst} (model {b['model_p']}%{'' if b['edge_pct']==0 else f' | edge +{b['edge_pct']}%'})")
    return "\n".join(lines)


# =========================================================
# Public router
# =========================================================
SPORT_WORDS = ["match","matches","vs","versus","bet","bets","odds","corners","cards","fouls","btts","over","under","handicap","accumulator","acca","value","prediction","predictions","fixtures","schedule","lineups","lineup","form","h2h","banker","small odds","low risk","safe picks"]

@method_decorator(csrf_exempt, name='dispatch')
class AskAssistant(APIView):
    def post(self, request):
        user_text = (request.data.get("question") or "").strip()
        if not user_text:
            return Response({"response": "⚠️ Please type a message."})

        # remember explicit team1/team2 if provided
        t1 = request.data.get("team1"); t2 = request.data.get("team2")
        if t1 or t2:
            _remember_teams(request, _resolve_team(t1) if t1 else None, _resolve_team(t2) if t2 else None)

        # also detect teams in the current message (for follow-ups)
        dt1, dt2 = _extract_teams(user_text)
        if dt1 or dt2:
            _remember_teams(request, dt1, dt2)

        ql = user_text.lower()

        # direct Q&A (stats/corners/player assists/home form)
        direct = qa_skill_direct(user_text, request)
        if direct:
            _push_history(request, "user", user_text)
            _push_history(request, "assistant", direct)
            return Response({"response": direct})

        # fixtures (next 8 days)
        if any(k in ql for k in ["fixtures","what games","who is playing","schedule","this weekend","opening weekend"]):
            text = fixtures_skill_api_only()
            _push_history(request, "user", user_text)
            _push_history(request, "assistant", text)
            return Response({"response": text})

        # value board / best bets
        if "value board" in ql or "best bets" in ql:
            text = weekend_value_board()
            _push_history(request, "user", user_text)
            _push_history(request, "assistant", text)
            return Response({"response": text})

        # small odds bankers
        if any(x in ql for x in ["small odds","low risk","banker","safe picks"]):
            # reuse random pick but prefer Over 0.5/1.5 edges
            text = random_pick_skill()
            _push_history(request, "user", user_text)
            _push_history(request, "assistant", text)
            return Response({"response": text})

        # match intent (has 'vs' or sports keywords)
        if " vs " in f" {ql} " or any(w in ql for w in SPORT_WORDS):
            text = match_skill(user_text, request)
            _push_history(request, "user", user_text)
            _push_history(request, "assistant", text)
            return Response({"response": text})

        # general ChatGPT-style answer (fallback)
        conv = _recent_history(request, limit=12)
        messages = [{"role":"system","content": MIKE_SYSTEM}]
        messages.extend(conv)
        lt1, lt2 = _last_teams(request)
        if lt1 and lt2 and not (dt1 and dt2):
            messages.append({"role":"system","content": f"(Context: last teams in scope: {lt1} vs {lt2})"})
        messages.append({"role":"user","content": user_text})

        reply = _ask_llm(messages)
        _push_history(request, "user", user_text)
        _push_history(request, "assistant", reply)
        return Response({"response": reply})


# =========================================================
# Standings endpoints (2025 season) — with form window control ?n=5|10
# =========================================================
def _standings_payload(league_id: int, season: int, n_form: int = 5):
    try:
        table = af.get_standings(league_id=league_id, season=season)
        rows = []
        for r in table:
            team = r.get("team") or {}
            allrec = r.get("all") or {}
            goals = (allrec.get("goals") or {})
            tid = team.get("id")
            form_info = af.team_form_last_n(team_id=tid, n=n_form, league_id=league_id, season=season) if tid else {}
            rows.append({
                "pos": r.get("rank"),
                "team": team.get("name"),
                "logo": team.get("logo"),
                "p": allrec.get("played", 0),
                "w": allrec.get("win", 0),
                "d": allrec.get("draw", 0),
                "l": allrec.get("lose", 0),
                "gf": (goals.get("for") or 0),
                "ga": (goals.get("against") or 0),
                "gd": r.get("goalsDiff", 0),
                "pts": r.get("points", 0),
                "form": form_info.get("form", ""),
                "form_pct": form_info.get("win_rate", 0),
            })
        return rows
    except Exception:
        return []

@csrf_exempt
def standings_epl(request):
    try:
        n = int(request.GET.get("n", "5"))
    except Exception:
        n = 5
    payload = _standings_payload(league_id=LEAGUE_IDS["EPL"], season=2025, n_form=max(3, min(n, 10)))
    return JsonResponse({"season": 2025, "league": "EPL", "standings": payload})

@csrf_exempt
def standings_laliga(request):
    try:
        n = int(request.GET.get("n", "5"))
    except Exception:
        n = 5
    payload = _standings_payload(league_id=LEAGUE_IDS["LaLiga"], season=2025, n_form=max(3, min(n, 10)))
    return JsonResponse({"season": 2025, "league": "La Liga", "standings": payload})
