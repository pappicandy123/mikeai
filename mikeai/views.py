# assistant/views.py — Mike: elite analyst with memory, odds/bookmakers, low-risk picks
import os
import re
import unicodedata
from datetime import timedelta

import requests
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.utils import timezone

from rest_framework.views import APIView
from rest_framework.response import Response

from . import api_football as af

# ---- OpenAI client
try:
    from openai import OpenAI
    OPENAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    OPENAI_OK = True
except Exception:
    OPENAI, OPENAI_OK = None, False

MODEL_NAME = "gpt-4o-mini"

# ---- External keys
ODDS_API_KEY = os.getenv("ODDS_API_KEY")       # The Odds API
MEDIASTACK_KEY = os.getenv("MEDIASTACK_KEY")   # mediastack.com


# ======================== UI ========================
def chat_page(request):
    return render(request, "index.html")


# ======================== Session memory ========================
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


# ======================== Team parsing helpers ========================
NICKNAMES = {
    "spurs":"Tottenham Hotspur","tottenham":"Tottenham Hotspur",
    "man city":"Manchester City","manchester city":"Manchester City",
    "man united":"Manchester United","man utd":"Manchester United","manchester united":"Manchester United",
    "west ham":"West Ham United","westham":"West Ham United",
    "wolves":"Wolverhampton Wanderers","wolverhampton":"Wolverhampton Wanderers",
    "bournemouth":"AFC Bournemouth","newcastle":"Newcastle United",
    "forest":"Nottingham Forest","nottingham":"Nottingham Forest","nottingham forest":"Nottingham Forest",
    "villa":"Aston Villa","aston villa":"Aston Villa",
    "liverpool":"Liverpool","chelsea":"Chelsea","arsenal":"Arsenal","everton":"Everton",
    "fulham":"Fulham","brentford":"Brentford","crystal palace":"Crystal Palace","palace":"Crystal Palace",
    "ipswich":"Ipswich Town","leicester":"Leicester City","southampton":"Southampton","brighton":"Brighton",
}

def _norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    # soften punctuation / fillers
    for junk in [" football club"," afc"," fc"," cf",".",",","-","_","|","/","\\","(",")","[","]","{","}",":",";","•","—"]:
        s = s.replace(junk, " ")
    return " ".join(s.split())

def _resolve_team(raw: str) -> str:
    """
    Robust resolver:
    - normalizes
    - matches nicknames also without spaces (fixes 'li erpool' -> 'liverpool')
    - falls back to title-cased raw
    """
    if not raw: return ""
    x = _norm(raw)
    x_nospace = x.replace(" ", "")
    for k, v in NICKNAMES.items():
        kn = k.replace(" ", "")
        if x == k or x_nospace == kn or k in x or kn in x_nospace:
            return v
    # common EPL teams often typed in segments
    for team_key, full in NICKNAMES.items():
        if team_key.replace(" ", "") in x_nospace:
            return full
    # fallback
    return " ".join(w.capitalize() for w in raw.strip().split())

def _extract_teams(text: str):
    if not text: return (None, None)
    q = _norm(text)
    m = re.search(r"(.+?)\s+(?:vs|v|versus|and|against|between)\s+(.+)", q, flags=re.I)
    if not m: return (None, None)
    return _resolve_team(m.group(1)), _resolve_team(m.group(2))


# ======================== Persona ========================
MIKE_SYSTEM = (
    "You are Mike — an elite, friendly sports betting analyst who can also answer ANY question like ChatGPT.\n"
    "\n"
    "SPORTS/BETTING OUTPUT FORMAT:\n"
    "1) 🔐 Best Props — 2–4 props with confidence %\n"
    "2) 💰 Value Bets — explain mispricing (model probability vs implied odds, or matchup-based)\n"
    "3) 🧾 Final Bet — one pick with confidence %\n"
    "4) 💡 Why — 2–6 bullets on tactics, form, matchups, pace, injuries, news\n"
    "Confidence usually 45–75%.\n"
    "\n"
    "DATA RULES:\n"
    "- App may provide API-Football stats/injuries/H2H, TheOddsAPI odds (with bookmaker names), and Mediastack headlines — use them faithfully.\n"
    "- If exact numbers missing, give directional guidance; avoid fabricating precise stats.\n"
    "- For non-sports topics, behave like standard ChatGPT.\n"
)

def _ask_llm(messages):
    if not OPENAI_OK or not OPENAI:
        return "Mike is currently offline. Try again shortly."
    out = OPENAI.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.7)
    return out.choices[0].message.content


# ======================== Small numeric model ========================
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

    p_btts = max(0.25, min(0.85, ((gfh>0.9)+(gfa>0.9)+(gah>1.0)+(gaa>1.0))/4))

    return {
        "home_win": round(p_home*100),
        "draw": round(p_draw*100),
        "away_win": round(p_away*100),
        "over25": round(p_over25*100),
        "btts": round(p_btts*100),
    }


# ======================== Odds + bookmaker names (The Odds API) ========================
def _get_theoddsapi_for_match(team1: str, team2: str):
    """
    Returns a dict aggregating *all bookmaker quotes* so we can show who has the best price.
    Structure:
      {
        '1x2': {'home': [(1.65,'Bet365'), (1.62,'Pinnacle')], 'draw': [...], 'away': [...]},
        'totals_goals': {'Over 2.5': [(1.90,'Pinnacle')], 'Under 2.5': [...]},
        'btts': {'Yes': [(1.85,'bet365')], 'No': [...]}
      }
    """
    if not ODDS_API_KEY:
        return {}

    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "uk,eu", "markets": "h2h,totals,btts"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        games = r.json()
    except Exception:
        return {}

    t1 = team1.lower(); t2 = team2.lower()
    for g in games:
        home = (g.get("home_team") or "").lower()
        away = (g.get("away_team") or "").lower()
        if not ((t1 in home and t2 in away) or (t1 in away and t2 in home)):
            continue

        out = {"1x2": {"home":[], "draw":[], "away":[]}, "totals_goals": {}, "btts": {}}
        for b in g.get("bookmakers", []):
            book = b.get("title") or b.get("key") or "bookmaker"
            for m in b.get("markets", []):
                key = (m.get("key") or "").lower()
                for o in m.get("outcomes", []):
                    name = (o.get("name") or "")
                    price = o.get("price")
                    point = o.get("point")

                    if key == "h2h":
                        nm = name.lower()
                        if nm == home: out["1x2"]["home"].append((price, book))
                        elif nm == away: out["1x2"]["away"].append((price, book))
                        elif nm == "draw": out["1x2"]["draw"].append((price, book))

                    elif key == "totals" and point is not None:
                        label = "Over 2.5" if (name.lower() == "over" and abs(point-2.5)<1e-6) else \
                                "Under 2.5" if (name.lower() == "under" and abs(point-2.5)<1e-6) else \
                                (f"Over {point}" if name.lower()=="over" else f"Under {point}")
                        out["totals_goals"].setdefault(label, []).append((price, book))

                    elif key == "btts":
                        label = name.title()  # Yes / No
                        out["btts"].setdefault(label, []).append((price, book))

        return out
    return {}

def _best_price_line(pairs):
    """pairs = [(price, book), ...] -> (max_price, [books..])"""
    if not pairs: return None, []
    valid = [p for p, _ in pairs if p]
    if not valid:
        return None, []
    mx = max(valid)
    books = [bk for p, bk in pairs if p == mx]
    return mx, books

def _implied_prob(odds: float) -> float:
    return 0.0 if not odds or odds <= 1.0 else 1.0/odds

def _value_check(prob: float, odds: float, edge: float = 0.08) -> bool:
    return prob - _implied_prob(odds) >= edge

def _format_odds_table(od):
    """
    Make a human-friendly odds section with bookmaker names,
    focusing on best prices.
    """
    if not od:
        return "Odds unavailable."

    lines = ["\n📈 **Odds (best prices & books)**"]
    if "1x2" in od:
        for key, label in (("home","Home"),("draw","Draw"),("away","Away")):
            px, books = _best_price_line(od["1x2"].get(key, []))
            if px:
                lines.append(f"- {label}: {px} ({', '.join(books[:3])})")
    if "totals_goals" in od:
        for label in ("Over 2.5","Under 2.5"):
            px, books = _best_price_line(od["totals_goals"].get(label, []))
            if px:
                lines.append(f"- {label}: {px} ({', '.join(books[:3])})")
    if "btts" in od:
        for label in ("Yes","No"):
            px, books = _best_price_line(od["btts"].get(label, []))
            if px:
                lines.append(f"- BTTS {label}: {px} ({', '.join(books[:3])})")
    return "\n".join(lines)


# ======================== News (Mediastack) ========================
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


# ======================== Fixtures (API-Football only) ========================
def _weekend_window_lagos():
    now = timezone.localtime()
    fri = (now + timedelta(days=(4 - now.weekday()) % 7)).replace(hour=0, minute=0, second=0, microsecond=0)
    sun = fri + timedelta(days=2, hours=23, minutes=59)
    return fri.date().isoformat(), sun.date().isoformat()

def fixtures_skill_api_only():
    start, end = _weekend_window_lagos()
    try:
        resp = af.get_fixtures_by_range(start, end, league_id=39)  # EPL
        arr = resp.get("response", [])
        if not arr:
            return "No EPL fixtures found in the weekend window (API-Football)."
        lines = [f"📅 EPL Fixtures ({start} → {end}):"]
        for fx in arr:
            f = fx.get("fixture", {}); t = fx.get("teams", {})
            dt = f.get("date","")
            lines.append(f"- {dt[:10]} {dt[11:16]}: {t.get('home',{}).get('name')} vs {t.get('away',{}).get('name')}")
        return "\n".join(lines)
    except Exception:
        return "⚠️ I couldn’t fetch fixtures right now (API error). Please try again shortly."


# ======================== Match analysis (API + Odds + News + LLM) ========================
def match_skill(question: str, request):
    t1p = request.data.get("team1"); t2p = request.data.get("team2")
    if t1p and t2p: t1, t2 = _resolve_team(t1p), _resolve_team(t2p)
    else: t1, t2 = _extract_teams(question)
    if not (t1 and t2):
        lt1, lt2 = _last_teams(request)
        t1, t2 = t1 or lt1, t2 or lt2
    if not (t1 and t2):
        return "Tell me two teams (e.g., 'Liverpool vs Bournemouth')."

    _remember_teams(request, t1, t2)

    context = {"teams": {"home": t1, "away": t2}}
    used_api = False

    try:
        id1 = af.search_team_id(t1, league_id=39)
        id2 = af.search_team_id(t2, league_id=39)
        if id1 and id2:
            used_api = True
            sH = af.get_team_stats(id1, league_id=39).get("response", {})
            sA = af.get_team_stats(id2, league_id=39).get("response", {})
            model = _quick_probs_from_team_stats(sH, sA)

            inj1 = af.get_injuries(id1).get("response", [])
            inj2 = af.get_injuries(id2).get("response", [])
            inj_names1 = ", ".join(sorted({ (p.get("player") or {}).get("name","") for p in inj1 if p.get("player") })) or "None"
            inj_names2 = ", ".join(sorted({ (p.get("player") or {}).get("name","") for p in inj2 if p.get("player") })) or "None"

            h2h = af.get_head_to_head(id1, id2, last=5).get("response", [])
            h2h_lines = []
            for m in h2h:
                f = m.get("fixture", {}); t = m.get("teams", {}); sc = (m.get("score") or {}).get("fulltime", {})
                h2h_lines.append(f"{f.get('date','')[:10]}: {t.get('home',{}).get('name')} {sc.get('home')}–{sc.get('away')} {t.get('away',{}).get('name')}")

            # Odds + value
            odds_summary = _get_theoddsapi_for_match(t1, t2)
            values = []
            if odds_summary.get("1x2"):
                ow = odds_summary["1x2"]
                for lbl, prob_key in (("home","home_win"), ("draw","draw"), ("away","away_win")):
                    if ow.get(lbl):
                        best, _ = _best_price_line(ow[lbl])
                        if best:
                            p = model[prob_key] / 100.0
                            if _value_check(p, best, edge=0.08):
                                values.append(f"1X2 {lbl.title()} is +EV (model {round(p*100)}% vs implied {round(_implied_prob(best)*100)}%)")
            for lbl, pairs in (odds_summary.get("totals_goals") or {}).items():
                if "Over 2.5" == lbl:
                    best, _ = _best_price_line(pairs)
                    if best:
                        p = model["over25"]/100.0
                        if _value_check(p, best, edge=0.08):
                            values.append(f"Over 2.5 Goals is +EV (model {round(p*100)}% vs implied {round(_implied_prob(best)*100)}%)")

            news_home = _get_team_headlines(t1, limit=3)
            news_away = _get_team_headlines(t2, limit=3)

            context.update({
                "model": model,
                "injuries": {t1: inj_names1, t2: inj_names2},
                "h2h": "\n".join(h2h_lines) if h2h_lines else "N/A",
                "odds": _format_odds_table(odds_summary),
                "value_spots": values,
                "news": {t1: news_home, t2: news_away},
                "form": {t1: sH.get("form","N/A"), t2: sA.get("form","N/A")},
            })
    except Exception:
        used_api = False

    messages = [
        {"role":"system","content": MIKE_SYSTEM},
        {"role":"user","content": f"Question: {question}\n\nContext: {context if used_api else 'LLM-only (no API context)'}\n\nWrite your analyst note now with props, value bets (if any), final pick, confidence, and a short 'why' section. If news is present, mention key headlines briefly."}
    ]
    return _ask_llm(messages)


# ======================== Single-team props (corners/cards/fouls) ========================
def team_props_skill(question: str, fallback_team: str | None = None):
    ql = question.lower()
    metric = "corners"
    if "card" in ql: metric = "cards"
    elif "foul" in ql: metric = "fouls"

    m = re.search(r"(?:for|on|of)\s+([a-zA-Z\s\.-]+)$", question.strip(), flags=re.I)
    team_guess = m.group(1).strip() if m else None
    team = _resolve_team(team_guess) if team_guess else (fallback_team or "")
    if not team:
        return "Tell me the team, e.g., 'corner bet for Liverpool'."

    tid = af.search_team_id(team, league_id=39)
    if not tid:
        return f"I couldn't resolve the team. Try: 'corner bet for Liverpool'."

    season = getattr(af, "current_season", lambda: timezone.now().year)()
    if metric == "corners":
        rec = af.team_recent_corners(tid, league_id=39, season=season, last_n=6)
        ta, tot, n = rec.get("team_avg"), rec.get("total_avg"), rec.get("n")
        if not n:
            return f"I don't have enough recent corners data for {team}."
        suggestions = []
        if ta and ta >= 5.5:
            suggestions.append(f"{team} Over 5.5 Corners — ~{min(85, int(ta/7.5*100))}%")
        elif ta and ta >= 4.5:
            suggestions.append(f"{team} Over 4.5 Corners — ~{min(80, int(ta/7.0*100))}%")
        if tot and tot >= 10.0:
            suggestions.append(f"Total Corners Over 9.5 — ~{min(80, int(tot/12.0*100))}%")

        text = [f"📊 **{team} — corners (last {n} matches)**",
                f"- Team avg: {ta} | Match total avg: {tot}",
                "🎯 Suggestions:"]
        text.extend([f"- {s}" for s in suggestions] or ["- No strong angle — consider live corners if pace is high."])
        return "\n".join(text)

    if metric == "cards":
        rec = af.team_recent_cards(tid, league_id=39, season=season, last_n=6)
        ya, ra, n = rec.get("yellow_avg"), rec.get("red_avg"), rec.get("n")
        if not n:
            return f"I don't have enough recent cards data for {team}."
        conf = min(75, int((ya or 0)/3.0*100))
        return (f"📒 **{team} — cards (last {n})**\n"
                f"- Yellow avg: {ya} | Red avg: {ra}\n"
                f"🎯 Suggestion: {team} Over 1.5 Team Cards — ~{max(45, conf)}%\n"
                f"Tip: Derbies/high-press games trend higher on cards.")

    if metric == "fouls":
        rec = af.team_recent_fouls(tid, league_id=39, season=season, last_n=6)
        fa, n = rec.get("fouls_avg"), rec.get("n")
        if not n:
            return f"I don't have enough recent fouls data for {team}."
        conf = min(75, int((fa or 0)/16.0*100))
        return (f"🧤 **{team} — fouls (last {n})**\n"
                f"- Fouls avg: {fa}\n"
                f"🎯 Suggestion: {team} Over 11.5 Team Fouls — ~{max(45, conf)}%\n"
                f"Note: Referee leniency and opponent ball retention can swing foul counts.")


# ======================== Match-level props follow-up (last teams) ========================
def match_props_skill(request, metric: str):
    t1, t2 = _last_teams(request)
    if not (t1 and t2):
        return "Tell me two teams (e.g., 'Liverpool vs Bournemouth')."

    id1 = af.search_team_id(t1, league_id=39)
    id2 = af.search_team_id(t2, league_id=39)
    if not (id1 and id2):
        return "I couldn't resolve one of the teams. Try: 'Liverpool vs Bournemouth'."

    season = getattr(af, "current_season", lambda: timezone.now().year)()

    lines = [f"📊 **{t1} vs {t2} — {metric.title()} (recent form)**"]
    if metric == "corners":
        c1 = af.team_recent_corners(id1, league_id=39, season=season, last_n=6)
        c2 = af.team_recent_corners(id2, league_id=39, season=season, last_n=6)
        totals = []
        if c1.get("total_avg"): totals.append(c1["total_avg"])
        if c2.get("total_avg"): totals.append(c2["total_avg"])
        avg_total = round(sum(totals)/len(totals), 2) if totals else None

        lines.append(f"- {t1}: team avg corners {c1.get('team_avg')} (n={c1.get('n')})")
        lines.append(f"- {t2}: team avg corners {c2.get('team_avg')} (n={c2.get('n')})")
        lines.append(f"- Match total avg (blend): {avg_total}")

        suggestions = []
        if avg_total and avg_total >= 10.0:
            suggestions.append("Total Corners Over 9.5")
        if c1.get("team_avg") and c1["team_avg"] >= 5.5:
            suggestions.append(f"{t1} Over 5.5 Team Corners")
        if c2.get("team_avg") and c2["team_avg"] >= 5.0:
            suggestions.append(f"{t2} Over 4.5 Team Corners")

        if suggestions:
            lines.append("🎯 Suggestions:")
            for s in suggestions:
                lines.append(f"- {s}")
        else:
            lines.append("🎯 No strong pre-match angle — consider live corners if pace is high.")
        return "\n".join(lines)

    if metric == "cards":
        a1 = af.team_recent_cards(id1, league_id=39, season=season, last_n=6)
        a2 = af.team_recent_cards(id2, league_id=39, season=season, last_n=6)
        lines.append(f"- {t1}: yellow {a1.get('yellow_avg')} / red {a1.get('red_avg')} (n={a1.get('n')})")
        lines.append(f"- {t2}: yellow {a2.get('yellow_avg')} / red {a2.get('red_avg')} (n={a2.get('n')})")
        lines.append("🎯 Suggestion: Consider Over 3.5 Total Cards if both teams trend ≥1.6 yellows.")
        return "\n".join(lines)

    if metric == "fouls":
        f1 = af.team_recent_fouls(id1, league_id=39, season=season, last_n=6)
        f2 = af.team_recent_fouls(id2, league_id=39, season=season, last_n=6)
        lines.append(f"- {t1}: fouls avg {f1.get('fouls_avg')} (n={f1.get('n')})")
        lines.append(f"- {t2}: fouls avg {f2.get('fouls_avg')} (n={f2.get('n')})")
        lines.append("🎯 Suggestion: Team with higher average vs possession-heavy opponent → team fouls over.")
        return "\n".join(lines)

    return "Unknown metric."


# ======================== Weekend Value Board (EPL) ========================
def weekend_value_board():
    start, end = _weekend_window_lagos()
    try:
        fx = af.get_fixtures_by_range(start, end, league_id=39).get("response", [])
        if not fx:
            return "No EPL fixtures found this weekend."

        board = []
        for row in fx:
            teams = row.get("teams", {})
            home = teams.get("home", {}).get("name")
            away = teams.get("away", {}).get("name")
            if not (home and away):
                continue

            idH = af.search_team_id(home, league_id=39)
            idA = af.search_team_id(away, league_id=39)
            if not (idH and idA):
                continue

            sH = af.get_team_stats(idH, league_id=39).get("response", {})
            sA = af.get_team_stats(idA, league_id=39).get("response", {})
            model = _quick_probs_from_team_stats(sH, sA)

            odds = _get_theoddsapi_for_match(home, away)
            if not odds:
                continue

            rows = []
            one = odds.get("1x2", {})
            for side, key in (("home","home_win"), ("draw","draw"), ("away","away_win")):
                pairs = one.get(side, [])
                best, _ = _best_price_line(pairs)
                if best:
                    p = model[key] / 100.0
                    edge = p - _implied_prob(best)
                    rows.append(("1X2", side, best, round(p*100), round(edge*100,1)))
            for lbl, pairs in (odds.get("totals_goals") or {}).items():
                if lbl == "Over 2.5":
                    best, _ = _best_price_line(pairs)
                    if best:
                        p = model["over25"]/100.0
                        edge = p - _implied_prob(best)
                        rows.append(("Over 2.5", lbl, best, round(p*100), round(edge*100,1)))

            if not rows:
                continue
            rows.sort(key=lambda x: x[4], reverse=True)
            best = rows[0]
            if best[4] >= 8.0:  # +8% edge threshold
                board.append({
                    "match": f"{home} vs {away}",
                    "market": best[0],
                    "selection": best[1],
                    "odds": best[2],
                    "model_p": best[3],
                    "edge_pct": best[4],
                })

        if not board:
            return f"📋 Weekend Value Board ({start}→{end}): No clear +EV edges found."

        board.sort(key=lambda x: x["edge_pct"], reverse=True)
        lines = [f"📋 Weekend Value Board ({start}→{end}) — +EV picks"]
        for b in board:
            lines.append(f"- {b['match']}: {b['market']} {b['selection']} @ {b['odds']} "
                         f"(model {b['model_p']}% | edge +{b['edge_pct']}%)")
        return "\n".join(lines)

    except Exception:
        return "⚠️ Couldn’t build the value board right now. Try again shortly."


# ======================== Low-risk 'small odds' bankers ========================
def small_odds_bankers():
    """
    Scan Odds API EPL board; pick favorites with small odds where our model probability is high
    and still shows some positive edge. Useful for 'find me safe small odds'.
    """
    if not ODDS_API_KEY:
        return "Set ODDS_API_KEY to use low-risk picks."

    # Pull whole board
    try:
        url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
        params = {"apiKey": ODDS_API_KEY, "regions": "uk,eu", "markets": "h2h"}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        games = r.json()
    except Exception:
        return "⚠️ Couldn’t read odds right now."

    picks = []
    for g in games:
        home = g.get("home_team"); away = g.get("away_team")
        if not (home and away):
            continue
        try:
            idH = af.search_team_id(home, league_id=39); idA = af.search_team_id(away, league_id=39)
            if not (idH and idA):
                continue
            sH = af.get_team_stats(idH, league_id=39).get("response", {})
            sA = af.get_team_stats(idA, league_id=39).get("response", {})
            model = _quick_probs_from_team_stats(sH, sA)
        except Exception:
            continue

        # aggregate best h2h prices
        all_pairs_home, all_pairs_away = [], []
        for b in g.get("bookmakers", []):
            book = b.get("title") or b.get("key")
            for m in b.get("markets", []):
                if (m.get("key") or "").lower() != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    nm = (o.get("name") or "").lower()
                    pr = o.get("price")
                    if nm == (home or "").lower():
                        all_pairs_home.append((pr, book))
                    elif nm == (away or "").lower():
                        all_pairs_away.append((pr, book))

        best_home, books_home = _best_price_line(all_pairs_home)
        best_away, books_away = _best_price_line(all_pairs_away)

        # choose favorite side with odds <= 1.40, model >= 60%, positive edge >= 3%
        cand = []
        if best_home:
            p = model["home_win"]/100.0
            edge = p - _implied_prob(best_home)
            if best_home <= 1.40 and p >= 0.60 and edge >= 0.03:
                cand.append(("home", best_home, p, edge, books_home))
        if best_away:
            p = model["away_win"]/100.0
            edge = p - _implied_prob(best_away)
            if best_away <= 1.40 and p >= 0.60 and edge >= 0.03:
                cand.append(("away", best_away, p, edge, books_away))

        if cand:
            cand.sort(key=lambda x: (x[1], -x[2]))  # prefer smaller odds but higher model prob
            side, odds, p, edge, bks = cand[0]
            picks.append({
                "match": f"{home} vs {away}",
                "pick": f"{'Home' if side=='home' else 'Away'} to Win",
                "odds": odds,
                "books": ", ".join(bks[:3]),
                "model_p": round(p*100),
                "edge_pct": round(edge*100, 1),
            })

    if not picks:
        return "No clear low-risk bankers found right now."

    picks.sort(key=lambda x: (-x["edge_pct"], x["odds"]))
    lines = ["🎯 **Low-risk small-odds picks** (model-supported)"]
    for p in picks[:6]:
        lines.append(f"- {p['match']}: {p['pick']} @ {p['odds']} ({p['books']}) — model {p['model_p']}% | edge +{p['edge_pct']}%")
    lines.append("\nTip: Treat these as singles or small doubles; avoid big accas even for bankers.")
    return "\n".join(lines)


# ======================== Router ========================
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

        # quick commands
        if "value board" in ql or ("best bets" in ql and "weekend" in ql):
            text = weekend_value_board()
            _push_history(request, "user", user_text)
            _push_history(request, "assistant", text)
            return Response({"response": text})

        # low-risk small odds / bankers
        if any(x in ql for x in ["small odds","low risk","banker","safe picks"]):
            text = small_odds_bankers()
            _push_history(request, "user", user_text)
            _push_history(request, "assistant", text)
            return Response({"response": text})

        # fixtures (weekend/opening weekend)
        if any(k in ql for k in ["fixtures","what games","who is playing","schedule","this weekend","opening weekend"]):
            text = fixtures_skill_api_only()
            _push_history(request, "user", user_text)
            _push_history(request, "assistant", text)
            return Response({"response": text})

        # follow-up props “what about corners/cards/fouls?”
        if any(w in ql for w in ["corners","cards","fouls"]) and " vs " not in f" {ql} ":
            metric = "corners" if "corner" in ql else ("cards" if "card" in ql else "fouls")
            lt1, lt2 = _last_teams(request)
            if lt1 and lt2:
                text = match_props_skill(request, metric)
            else:
                text = team_props_skill(user_text, fallback_team=(lt1 or lt2))
            _push_history(request, "user", user_text)
            _push_history(request, "assistant", text)
            return Response({"response": text})

        # match intent (has 'vs' or sports keywords)
        if " vs " in f" {ql} " or any(w in ql for w in SPORT_WORDS):
            text = match_skill(user_text, request)
            _push_history(request, "user", user_text)
            _push_history(request, "assistant", text)
            return Response({"response": text})

        # general ChatGPT-style answer
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
