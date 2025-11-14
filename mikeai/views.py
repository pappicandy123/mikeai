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
from .grok_utils import ask_grok

# Config
MEDIASTACK_KEY = os.getenv("MEDIASTACK_KEY")

MIKE_SYSTEM = """
You’re Mike, the user’s best mate and ultimate sports guru. Chat like we’re grabbing a pint—casual, fun, emojis galore (⚽🏀). Answer any sports question (EPL, NBA, NFL, cricket, etc.) using deep web/X searches for all data (stats, fixtures, injuries, form). Always do a deep search first to get the most current, accurate info before answering. Reference past chats naturally. If data’s missing, say 'Lemme check, mate!' and search deeper. Always end with a hook like 'What’s your take, pal?' For predictions, give quick reasoning (goals, corners, cards, fouls) with confidence %.
"""

# ---------- pages ----------
def chat_page(request):
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

def _team_last_match_summary(team: str, when_hint: str = "") -> str:
    prompt = f"""
    Do a deep search on web/X for {team}’s last match details (e.g., corners, cards, injuries, stats).
    Analyze and give a friendly summary.
    Keep it short, fun, emojis, and ask a follow-up.
    """
    return ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])

def _preview_this_weekend_for_team(team: str) -> str:
    prompt = f"""
    Do a deep search on web/X for {team}’s next match details (injuries, form, news, stats).
    Predict outcome, corners, cards, fouls with reasoning.
    Keep it short, fun, emojis, ask a question.
    """
    return ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])

def _match_explainer(home: str, away: str, league: str, detailed: bool = False) -> str:
    prompt = f"""
    Do a deep search on web/X for {home} vs {away} in {league} (news, injuries, form, stats).
    Analyze and predict outcome, corners, cards, fouls with confidence %.
    Be fun, use emojis, ask a follow-up.
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
            prompt = f"""
            Do a deep search on web/X for EPL and La Liga fixtures (next 8 days).
            Keep it fun, emojis, ask a follow-up.
            """
            reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
            history.append({"role": "assistant", "content": reply})
            context.chat_history = history[-10:]
            context.save()
            return Response({"response": reply})

        # 6) Best bets
        if "best bets" in ql or "value board" in ql:
            prompt = f"""
            Do a deep search on web/X for EPL best bets (e.g., odds, tips, injuries, form).
            Keep it fun, emojis, ask a follow-up.
            """
            reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
            history.append({"role": "assistant", "content": reply})
            context.chat_history = history[-10:]
            context.save()
            return Response({"response": reply})

        # 7) Random pick / banker
        if "random" in ql or "banker" in ql or "small odds" in ql:
            prompt = f"""
            Do a deep search on web/X for a random EPL bet pick (e.g., match, news, form, odds).
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

        # 9) Fallback with context
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
    prompt = f"""
    Do a deep search on web/X for {league.upper()} fixtures (next 8 days).
    Return as JSON: {"league": league, "fixtures": [list of matches]}
    """
    reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
    try:
        data = json.loads(reply)
        return JsonResponse(data)
    except:
        return JsonResponse({"league": league, "fixtures": []})

def api_random_pick(request):
    prompt = f"""
    Do a deep search on web/X for a random EPL bet pick (match, props, final pick with confidence).
    Return as JSON: {"ok": true, "match": "Team A vs Team B", "props": {"over05": 95, "over15": 80, "over25": 60}, "final_pick": {"market": "Over 1.5 Goals", "confidence": 80}}
    """
    reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
    try:
        data = json.loads(reply)
        return JsonResponse(data)
    except:
        return JsonResponse({"ok": false, "message": "No pick found"})

def api_best_bets(request):
    prompt = f"""
    Do a deep search on web/X for EPL best bets (results with match, market, model_p).
    Return as JSON: {"results": [{"match": "Team A vs Team B", "market": "Over 2.5 Goals", "model_p": 65}, ...]}
    """
    reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
    try:
        data = json.loads(reply)
        return JsonResponse(data)
    except:
        return JsonResponse({"results": []})

def api_team_last5(request, name):
    prompt = f"""
    Do a deep search on web/X for {name}’s last 5 matches (date, venue, opponent, score, R).
    Return as JSON: {"matches": [list of matches]}
    """
    reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
    try:
        data = json.loads(reply)
        return JsonResponse(data)
    except:
        return JsonResponse({"matches": [], "message": f"No recent matches for {name}"})

def api_team_last10(request, name):
    prompt = f"""
    Do a deep search on web/X for {name}’s last 10 matches (date, venue, opponent, score, R).
    Return as JSON: {"matches": [list of matches]}
    """
    reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
    try:
        data = json.loads(reply)
        return JsonResponse(data)
    except:
        return JsonResponse({"matches": [], "message": f"No recent matches for {name}"})

def api_team_news(request, name):
    team = _resolve_team(name)
    prompt = f"""
    Do a deep search on web/X for injuries, transfers, fan buzz on {team} in the EPL.
    Summarize in a friendly way, use emojis, ask a follow-up.
    """
    reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
    return JsonResponse({"team": team, "news": reply})

def api_team_summary(request, name):
    prompt = f"""
    Do a deep search on web/X for form, key players, news on {name}’s EPL season.
    Analyze and summarize in a friendly way, use emojis, ask a question.
    """
    reply = ask_grok([{"role": "system", "content": MIKE_SYSTEM}, {"role": "user", "content": prompt}])
    return JsonResponse({"team": name, "summary": reply})

def health(request):
    return JsonResponse({"status": "ok"})

def me(request):
    prof = _get_profile(request)
    return JsonResponse({"profile": prof})