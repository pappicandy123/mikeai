import os
import re
import json
import random
import unicodedata
import difflib
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import timedelta

import requests
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.views import APIView
from rest_framework.response import Response

from .models import UserContext
from . import api_football as af
from .sportmonks import get_fixtures_by_range as sm_get_fixtures
from .standings import standings_epl, standings_laliga
from .grok_utils import ask_grok

# Config
SPORTMONKS_KEY = os.getenv("SPORTMONKS_API_KEY")
SM_LEAGUE_EPL = os.getenv("SPORTMONKS_LEAGUE_EPL", "8")  # Default to 8 for EPL
SM_LEAGUE_LALIGA = os.getenv("SPORTMONKS_LEAGUE_LALIGA", "140")
MEDIASTACK_KEY = os.getenv("MEDIASTACK_KEY")
AF_LEAGUE = {"epl": 39, "laliga": 140}

MIKE_SYSTEM = """
You’re Mike, the user’s best mate and sports guru. Chat like we’re chilling with a pint—super casual, fun, emojis aplenty (⚽🏀). Answer any sports question (EPL, NBA, NFL, cricket, etc.) using provided stats. Always do a deep search on web/X first to get the most current, accurate data (injuries, form, stats) before answering. Reference past chats naturally. If data’s missing, say 'Lemme check, mate!' and search deeper. Always end with a hook like 'What’s your vibe, pal?' For predictions, give quick reasoning (goals, corners, cards, fouls) with confidence %.
"""

# ---------- pages ----------
def chat_page(request):
    """Render the chat interface."""
    return render(request, "index.html")

def team_panel(request):
    try:
        return render(request, "team.html")
    except Exception:
        return HttpResponse("<h1>Team Panel</h1><p>Create templates/team.html to render this page.</p>")

def log_info(msg, **ctx):
    print(f"[INFO] {msg} {json.dumps(ctx, ensure_ascii=False)}" if ctx else f"[INFO] {msg}")

def log_err(msg, **ctx):
    print(f"[ERROR] {msg} {json.dumps(ctx, ensure_ascii=False)}" if ctx else f"[ERROR] {msg}")

# ---------- cache (3 minutes) ----------
_CACHE: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL = 180  # seconds

def cache_get(key: str):
    now = time.time()
    hit = _CACHE.get(key)
    if not hit: return None
    t, val = hit
    if now - t > CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return val

def cache_set(key: str, val: Any):
    _CACHE[key] = (time.time(), val)
    return val

# ---------- text helpers ----------
def _norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    for junk in [" football club"," afc"," fc"," cf",".",",","-","_","|","/","\\","(",")","[","]","{","}",":",";","•","—","'"]:
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
CANON_BY_NORM: Dict[str, str] = {}
for k, v in NICKNAMES.items():
    CANON_BY_NORM[_norm(k)] = v
for v in set(NICKNAMES.values()):
    CANON_BY_NORM[_norm(v)] = v

def _fuzzy_team(raw: str) -> Optional[str]:
    cand = _norm(raw or "")
    if not cand:
        return None
    names = list(CANON_BY_NORM.keys())
    best = difflib.get_close_matches(cand, names, n=1, cutoff=0.75)
    if best:
        return CANON_BY_NORM.get(best[0])
    return None

def _resolve_team(raw: str) -> str:
    if not raw: return ""
    x = _norm(raw); x_nospace = x.replace(" ", "")
    for k, v in NICKNAMES.items():
        if x == _norm(k) or x_nospace == _norm(k).replace(" ","") or _norm(k) in x or _norm(k).replace(" ","") in x_nospace:
            return v
    fz = _fuzzy_team(raw)
    if fz:
        return fz
    return " ".join(w.capitalize() for w in raw.strip().split())

def _extract_teams(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text: return (None, None)
    q = _norm(text)
    m = re.search(r"(.+?)\s+(?:vs|v|versus|against)\s+(.+)", q)
    if m:
        h, a = m.groups()
        return _resolve_team(h), _resolve_team(a)
    return None, None

def _guess_team(text: str) -> Optional[str]:
    q = _norm(text)
    for k, v in CANON_BY_NORM.items():
        if k in q or k.replace(" ", "") in q.replace(" ", ""):
            return v
    return _fuzzy_team(text)

# ---------- session helpers ----------
def _get_profile(request) -> Dict[str, Any]:
    return request.session.get("profile", {"fav_team": None})

def _set_profile(request, **kwargs):
    prof = _get_profile(request)
    prof.update(kwargs)
    request.session["profile"] = prof
    request.session.modified = True

def _get_last_matchup(request) -> Tuple[Optional[str], Optional[str]]:
    return request.session.get("last_matchup", (None, None))

def _set_last_matchup(request, home: str, away: str):
    request.session["last_matchup"] = (home, away)
    request.session.modified = True

# ---------- data helpers ----------
def _next_window(days: int = 7) -> Tuple[timezone.datetime, timezone.datetime]:
    now = timezone.now()
    return now, now + timedelta(days=days)

def sm_try_fixtures(league: str, start_date: str, end_date: str) -> Tuple[List, bool]:
    if not SPORTMONKS_KEY or not league in AF_LEAGUE:
        return [], False
    try:
        lid = SM_LEAGUE_EPL if league == "epl" else SM_LEAGUE_LALIGA
        fixtures = sm_get_fixtures(lid, start_date, end_date)
        return fixtures, True
    except Exception as e:
        log_err("SportMonks fixtures failed", error=str(e), league=league)
        return [], False

def af_fixtures(league: str, start_date: str, end_date: str) -> Tuple[List, bool]:
    if not league in AF_LEAGUE:
        return [], False
    try:
        lid = AF_LEAGUE[league]
        fixtures = af.get_fixtures_by_range(start_date, end_date, lid)["response"]
        return fixtures, True
    except Exception as e:
        log_err("API-Football fixtures failed", error=str(e), league=league)
        return [], False

def _team_last_match_summary(team: str, when_hint: str = "") -> str:
    tid = af.resolve_team_id(team)
    if not tid:
        return f"Sorry, mate, couldn’t find {team}. Try another? 😎"
    last = af.team_form_last_n(tid, n=1, league_id=39)
    if not last or not isinstance(last, list) or len(last) == 0:
        log_err("No recent matches for team", team=team)
        return f"No recent matches found for {team}, pal. Maybe their season’s just kicking off? Wanna check their next game? ⚽"
    fx = last[0]
    prompt = f"""
    Do a deep search on web/X for details (e.g., corners, cards, injuries) from {team}’s last EPL match: {json.dumps(fx)}. Analyze the stats and give a friendly summary.
    Keep it short, fun, emojis, and ask a follow-up.
    """
    return ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])

def _preview_this_weekend_for_team(team: str) -> str:
    tid = af.resolve_team_id(team)
    if not tid:
        return f"Whoops, couldn’t find {team}. Another team? 😊"
    now, end = _next_window(8)
    sd, ed = now.date().isoformat(), end.date().isoformat()
    rows, ok = sm_try_fixtures("epl", sd, ed)
    if not ok or not rows:
        rows, ok = af_fixtures("epl", sd, ed)
    if not ok or not rows:
        prompt = f"""
        Do a deep search on web/X for {team}’s next EPL match between {sd} and {ed}. Predict outcome, corners, cards, fouls if found.
        Keep it short, fun, emojis, ask a question.
        """
        return ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
    for fx in rows:
        t = fx.get("teams") or {}
        h = (t.get("home") or {}).get("name")
        a = (t.get("away") or {}).get("name")
        if h == team or a == team:
            prompt = f"""
            Do a deep search on web/X for injuries, form, news on {team}’s next EPL match: {json.dumps(fx)}. Predict outcome, corners, cards, fouls using API stats and search data.
            Keep it short, fun, emojis, ask a question.
            """
            return ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
    return f"No EPL matches for {team} soon, mate. Check their last game? ⚽"

def _match_explainer(home: str, away: str, league: str, detailed: bool = False) -> str:
    h_id, a_id = af.resolve_team_id(home), af.resolve_team_id(away)
    if not (h_id and a_id):
        return f"Sorry, mate, couldn’t find {home} or {away}. Try again? 😅"
    stats = af.get_head_to_head(h_id, a_id)
    prompt = f"""
    Do a deep search on web/X for news, injuries, form on {home} vs {away} in {league}. Use stats: {json.dumps(stats)}. Analyze and predict outcome, corners, cards, fouls.
    Be fun, use emojis, give confidence %, ask a follow-up.
    {"Explain in detail" if detailed else "Keep it snappy"}
    """
    return ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])

# ---------- REST endpoints ----------
@method_decorator(csrf_exempt, name="dispatch")
class AskAssistant(APIView):
    def post(self, request):
        session_key = request.session.session_key or request.session.create()
        context, _ = UserContext.objects.get_or_create(
            session_key=session_key, defaults={"chat_history": [], "recent_leagues": "epl"}
        )
        history = context.chat_history[-10:]  # Last 10 turns
        user_text = request.data.get("question", "").strip()
        ql = user_text.lower()

        # Save user question
        history.append({"role": "user", "content": user_text})

        # 0a) Set favorite team
        m = re.search(r"my team is ([a-z\s]+)", ql, re.IGNORECASE)
        if m:
            team = _resolve_team(m.group(1))
            if team:
                _set_profile(request, fav_team=team)
                context.recent_leagues = "epl"
                context.save()
                reply = f"Got it, {team}’s your squad! 🙌 Ask me anything about ‘em. What’s up?"
                history.append({"role": "assistant", "content": reply})
                context.chat_history = history[-10:]
                context.save()
                return Response({"response": reply})

        # 0b) Replace 'we' with favorite team
        prof = _get_profile(request)
        fav = prof.get("fav_team")
        if fav:
            ql = re.sub(r"\bwe\b", fav.lower(), ql)

        # 1) Analysis/preview for last matchup or chip-based (e.g., "Vs Arsenal")
        if any(k in ql for k in ["explain", "analysis", "preview", "vs arsenal"]):
            h_prev, a_prev = _get_last_matchup(request)
            if "vs" in ql.lower():
                h, a = _extract_teams(ql)
                if h and a:
                    _set_last_matchup(request, h, a)
                    reply = _match_explainer(h, a, "epl", detailed=True)
                    history.append({"role": "assistant", "content": reply})
                    context.chat_history = history[-10:]
                    context.save()
                    return Response({"response": reply})
            if h_prev and a_prev:
                reply = _match_explainer(h_prev, a_prev, "epl", detailed=True)
                history.append({"role": "assistant", "content": reply})
                context.chat_history = history[-10:]
                context.save()
                return Response({"response": reply})

        # 2) Explicit matchup
        h, a = _extract_teams(ql)
        if h and a:
            _set_last_matchup(request, h, a)
            reply = _match_explainer(h, a, "epl", detailed="detail" in ql)
            history.append({"role": "assistant", "content": reply})
            context.chat_history = history[-10:]
            context.save()
            return Response({"response": reply})

        # 3) Team + next match
        if any(k in ql for k in ["weekend", "this weekend", "next match", "next game", "who do", "who's next"]):
            team = _guess_team(ql) or fav
            if team:
                reply = _preview_this_weekend_for_team(team)
                if reply:
                    hh, aa = _extract_teams(reply)
                    if hh and aa: _set_last_matchup(request, hh, aa)
                    history.append({"role": "assistant", "content": reply})
                    context.chat_history = history[-10:]
                    context.save()
                    return Response({"response": reply})

        # 4) Single-team last match + stats
        if any(w in ql for w in ["last weekend","last match","yesterday","last night","result","score","corners","cards","fouls"]):
            team = _guess_team(ql) or fav
            if team:
                reply = _team_last_match_summary(team, ql)
                history.append({"role": "assistant", "content": reply})
                context.chat_history = history[-10:]
                context.save()
                return Response({"response": reply})

        # 5) General fixtures
        if any(k in ql for k in ["fixtures","what games","who is playing","schedule","this weekend"]):
            now, end = _next_window(8)
            sd, ed = now.date().isoformat(), end.date().isoformat()
            def pack(rows, title):
                if not rows: return f"📅 {title} fixtures (next 8 days): none found."
                lines = [f"📅 {title} fixtures (next 8 days):"]
                for fx in rows[:24]:
                    dt = (fx.get("fixture") or {}).get("date","")
                    t = (fx.get("teams") or {})
                    h = (t.get("home") or {}).get("name","")
                    a = (t.get("away") or {}).get("name","")
                    if h and a: lines.append(f"- {str(dt)[:16].replace('T',' ')}: {h} vs {a}")
                return "\n".join(lines)
            e_rows, _ = sm_try_fixtures("epl", sd, ed)
            if not e_rows: e_rows, _ = af_fixtures("epl", sd, ed)
            l_rows, _ = sm_try_fixtures("laliga", sd, ed)
            if not l_rows: l_rows, _ = af_fixtures("laliga", sd, ed)
            reply = pack(e_rows, "EPL") + "\n\n" + pack(l_rows, "La Liga") + "\n\nWhat match you hyped for, mate? 😎"
            history.append({"role": "assistant", "content": reply})
            context.chat_history = history[-10:]
            context.save()
            return Response({"response": reply})

        # 6) Best bets
        if "best bets" in ql or "value board" in ql:
            req = request._request; req.GET = req.GET.copy(); req.GET["league"] = "epl"
            data = json.loads(api_best_bets(req).content.decode("utf-8"))
            if not data.get("results"):
                reply = "📋 No hot bets right now, pal. Wanna try a random pick? 😎"
            else:
                prompt = f"""
                Do a deep search on web/X for extra context (e.g., injuries, form) on EPL best bets: {json.dumps(data['results'])}.
                Keep it fun, emojis, ask a follow-up.
                """
                reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
            history.append({"role": "assistant", "content": reply})
            context.chat_history = history[-10:]
            context.save()
            return Response({"response": reply})

        # 7) Random pick / banker
        if "random" in ql or "banker" in ql or "small odds" in ql:
            req = request._request; req.GET = req.GET.copy(); req.GET["league"] = "epl"
            r = json.loads(api_random_pick(req).content.decode("utf-8"))
            if not r.get("ok"):
                reply = "Couldn’t grab a pick, mate. Another try? 😅"
            else:
                prompt = f"""
                Do a deep search on web/X for match context (e.g., news, form) on this EPL pick: {json.dumps(r)}.
                Keep it short, fun, emojis, ask a follow-up.
                """
                reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
            history.append({"role": "assistant", "content": reply})
            context.chat_history = history[-10:]
            context.save()
            return Response({"response": reply})

        # 8) Non-EPL sports or general questions
        sports = {"nba": "basketball", "nfl": "football", "cricket": "cricket", "tennis": "tennis"}
        sport = next((v for k, v in sports.items() if k in ql), None)
        if sport:
            prompt = f"""
            Do a deep search on web/X for recent {sport} updates on: {user_text}. Use context: {context.recent_leagues or 'EPL'}. Be fun, emojis, ask a follow-up.
            """
            reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
            history.append({"role": "assistant", "content": reply})
            context.chat_history = history[-10:]
            context.save()
            return Response({"response": reply})

        # 9) Fallback with context (e.g., "latest on Liverpool" + prediction)
        if any(k in ql for k in ["latest", "update", "news"]) and "predict" in ql:
            team = _guess_team(ql) or fav
            if team:
                last_summary = _team_last_match_summary(team)
                next_preview = _preview_this_weekend_for_team(team)
                if "No recent matches" in last_summary:
                    last_summary = "No recent match data, pal—season’s just kicking off maybe! ⚽"
                if "No EPL matches" in next_preview:
                    next_preview = _preview_this_weekend_for_team(team)  # Retry with deep search
                reply = f"{last_summary}\n\n{next_preview}"
                history.append({"role": "assistant", "content": reply})
                context.chat_history = history[-10:]
                context.save()
                return Response({"response": reply})

        # 10) Fallback with context
        prompt = f"""
        Do a deep search on web/X to get accurate data for: {user_text}.
        """
        reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}] + history + [{"role": "user", "content": prompt}])
        history.append({"role": "assistant", "content": reply})
        context.chat_history = history[-10:]
        context.save()
        if fav:
            return Response({"response": f"{reply}\n\nYo, {fav} fan! Got a match or stat you wanna dive into? 😎"})
        return Response({"response": f"{reply}\n\nYo! Tell me a match or say ‘my team is Arsenal’! 😎"})

# ---------- Other endpoints ----------
def api_fixtures(request):
    league = request.GET.get("league", "epl").lower()
    now, end = _next_window(8)
    sd, ed = now.date().isoformat(), end.date().isoformat()
    rows, ok = sm_try_fixtures(league, sd, ed)
    if not ok: rows, ok = af_fixtures(league, sd, ed)
    return JsonResponse({"league": league, "fixtures": rows})

def api_random_pick(request):
    # Mocked; enhance with Grok later
    return JsonResponse({
        "ok": True,
        "match": "Arsenal vs Tottenham",
        "props": {"over05": 95, "over15": 80, "over25": 60},
        "final_pick": {"market": "Over 1.5 Goals", "confidence": 80}
    })

def api_best_bets(request):
    # Mocked; enhance with Grok later
    return JsonResponse({
        "results": [
            {"match": "Man Utd vs Liverpool", "market": "Over 2.5 Goals", "model_p": 65},
            {"match": "Chelsea vs Arsenal", "market": "BTTS", "model_p": 70}
        ]
    })

def api_team_last5(request, name):
    tid = af.resolve_team_id(name)
    if not tid:
        return JsonResponse({"error": f"Team {name} not found"})
    matches = af.team_form_last_n(tid, n=5)
    if not matches:
        log_err("No last 5 matches for team", team=name)
        return JsonResponse({"matches": [], "message": f"No recent matches for {name}"})
    return JsonResponse({"matches": matches})

def api_team_last10(request, name):
    tid = af.resolve_team_id(name)
    if not tid:
        return JsonResponse({"error": f"Team {name} not found"})
    matches = af.team_form_last_n(tid, n=10)
    if not matches:
        log_err("No last 10 matches for team", team=name)
        return JsonResponse({"matches": [], "message": f"No recent matches for {name}"})
    return JsonResponse({"matches": matches})

def api_team_news(request, name):
    team = _resolve_team(name)
    prompt = f"""
    Do a deep search on web/X for injuries, transfers, fan buzz on {team} in the EPL.
    Summarize in a friendly way, use emojis, ask a follow-up.
    """
    reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
    return JsonResponse({"team": team, "news": reply})

def api_team_summary(request, name):
    tid = af.resolve_team_id(name)
    if not tid:
        return JsonResponse({"error": f"Team {name} not found"})
    stats = af.get_team_stats(tid)
    if not stats:
        log_err("No stats for team", team=name)
        prompt = f"""
        Do a deep search on web/X for form, key players, news on {name}’s EPL season.
        Summarize in a friendly way, use emojis, ask a question.
        """
    else:
        prompt = f"""
        Do a deep search on web/X for form, key players, news on {name}’s EPL season: {json.dumps(stats)}.
        Analyze and summarize in a friendly way, use emojis, ask a question.
        """
    reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
    return JsonResponse({"team": name, "summary": reply})

def health(request):
    return JsonResponse({"status": "ok"})

def me(request):
    prof = _get_profile(request)
    return JsonResponse({"profile": prof})