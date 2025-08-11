# assistant/api_football.py
import os
import time
import unicodedata
from difflib import get_close_matches
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_FOOTBALL_KEY}

# ---------- utils ----------
def _req(path: str, params: dict | None = None, timeout: int = 18):
    """GET wrapper with headers, timeouts, and safe JSON."""
    url = f"{BASE_URL}{path if path.startswith('/') else '/' + path}"
    try:
        r = requests.get(url, headers=HEADERS, params=params or {}, timeout=timeout)
        r.raise_for_status()
        return r.json() or {}
    except Exception:
        return {}

def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    for junk in [" football club", " afc", " fc", " cf", ".", ",", "-", "_", "|", "/", "\\",
                 "(", ")", "[", "]", "{", "}", ":", ";", "•", "—"]:
        s = s.replace(junk, " ")
    return " ".join(s.split())

NICKNAMES = {
    "spurs":"Tottenham Hotspur","tottenham":"Tottenham Hotspur",
    "man city":"Manchester City","manchester city":"Manchester City",
    "man united":"Manchester United","man utd":"Manchester United","manchester united":"Manchester United",
    "west ham":"West Ham United","westham":"West Ham United",
    "wolves":"Wolverhampton Wanderers","wolverhampton":"Wolverhampton Wanderers",
    "bournemouth":"AFC Bournemouth","newcastle":"Newcastle United",
    "forest":"Nottingham Forest","nottingham forest":"Nottingham Forest","nottingham":"Nottingham Forest",
    "villa":"Aston Villa","aston villa":"Aston Villa",
    "liverpool":"Liverpool","chelsea":"Chelsea","arsenal":"Arsenal","everton":"Everton",
    "fulham":"Fulham","brentford":"Brentford","crystal palace":"Crystal Palace","palace":"Crystal Palace",
    "ipswich":"Ipswich Town","leicester":"Leicester City","southampton":"Southampton","brighton":"Brighton",
}

def _resolve_team_name(raw: str) -> str:
    if not raw:
        return ""
    x = _norm(raw)
    x_nospace = x.replace(" ", "")
    for k, v in NICKNAMES.items():
        kn = k.replace(" ", "")
        if x == k or x_nospace == kn or k in x or kn in x_nospace:
            return v
    # fallback: title-case the input
    return " ".join(w.capitalize() for w in raw.strip().split())

def _safe(d: dict, path: list, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur

# ---------- season / ids ----------
def current_season() -> int:
    """
    API-Football uses the START year of the season.
    After July → current year; before July → previous year.
    """
    now = datetime.now()
    return now.year if now.month >= 7 else now.year - 1

def search_team_id(team_name: str, league_id: int = 39, season: int | None = None):
    """Fuzzy search team id (search + league scan + difflib)."""
    if not team_name:
        return None
    season = season or current_season()
    term = _resolve_team_name(team_name)

    # 1) direct search
    data = _req("/teams", {"search": term})
    for row in data.get("response", []):
        nm = _safe(row, ["team", "name"], "")
        if _norm(nm) == _norm(term) or _norm(term) in _norm(nm):
            return _safe(row, ["team", "id"])

    # 2) league scan
    league = _req("/teams", {"league": league_id, "season": season})
    teams = league.get("response", []) or []
    names = [_safe(t, ["team", "name"]) for t in teams if _safe(t, ["team"])]
    # contains
    tnorm = _norm(term)
    for nm in names:
        if nm and tnorm in _norm(nm):
            for t in teams:
                if _safe(t, ["team", "name"]) == nm:
                    return _safe(t, ["team", "id"])
    # difflib
    close = get_close_matches(term, names, n=1, cutoff=0.5)
    if close:
        best = close[0]
        for t in teams:
            if _safe(t, ["team", "name"]) == best:
                return _safe(t, ["team", "id"])
    return None

def get_all_team_names(league_id: int = 39, season: int | None = None):
    season = season or current_season()
    data = _req("/teams", {"league": league_id, "season": season})
    out = []
    for t in data.get("response", []) or []:
        tid = _safe(t, ["team", "id"])
        nm = _safe(t, ["team", "name"])
        if tid and nm:
            out.append((tid, nm))
    return out

# ---------- stats & info ----------
def get_team_stats(team_id: int, season: int | None = None, league_id: int = 39):
    season = season or current_season()
    return _req("/teams/statistics", {"team": team_id, "season": season, "league": league_id})

def get_head_to_head(team1_id: int, team2_id: int, last: int = 5):
    return _req("/fixtures/headtohead", {"h2h": f"{team1_id}-{team2_id}", "last": last})

def get_injuries(team_id: int, season: int | None = None):
    season = season or current_season()
    return _req("/injuries", {"team": team_id, "season": season})

def get_league_standings(league_id: int = 39, season: int | None = None):
    season = season or current_season()
    return _req("/standings", {"league": league_id, "season": season})

# ---------- fixtures ----------
def get_fixtures_by_date_and_league(date: str, league_id: int, season: int | None = None):
    season = season or current_season()
    return _req("/fixtures", {"date": date, "league": league_id, "season": season})

def get_fixtures_by_range(start_date: str, end_date: str, league_id: int, season: int | None = None):
    """Merge daily fixtures into one API-Football-shaped dict {'response': [...]}."""
    season = season or current_season()
    cur = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    out = []
    while cur <= end:
        ds = cur.strftime("%Y-%m-%d")
        daily = get_fixtures_by_date_and_league(ds, league_id, season)
        out.extend(daily.get("response", []) or [])
        cur += timedelta(days=1)
        # tiny courtesy delay to be gentle on free tiers
        time.sleep(0.15)
    return {"response": out}

def get_fixtures_by_team(team_id: int, league_id: int = 39, season: int | None = None, last: int = 10):
    season = season or current_season()
    return _req("/fixtures", {"team": team_id, "league": league_id, "season": season, "last": last})

def get_fixture_statistics(fixture_id: int):
    return _req("/fixtures/statistics", {"fixture": fixture_id})

# ---------- recent aggregates (corners / cards / fouls) ----------
def _avg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals)/len(vals), 2) if vals else None

def team_recent_corners(team_id: int, league_id: int = 39, season: int | None = None, last_n: int = 6):
    """
    Average team corners (and match total corners) over last N league fixtures.
    We call /fixtures(last=N) then /fixtures/statistics per fixture and take:
      - statistics[type == 'Corner Kicks'] for team_id
      - sum of both teams' corners for match total
    """
    season = season or current_season()
    fx = get_fixtures_by_team(team_id, league_id, season, last=last_n).get("response", []) or []
    if not fx:
        return {"team_avg": None, "total_avg": None, "n": 0}

    team_vals, total_vals, n = [], [], 0
    for row in fx:
        fid = _safe(row, ["fixture", "id"])
        if not fid:
            continue
        st = get_fixture_statistics(fid).get("response", []) or []
        team_c = None; sum_c = 0
        for block in st:
            tinfo = block.get("team") or {}
            stats = block.get("statistics") or []
            corners = None
            for s in stats:
                if s.get("type") == "Corner Kicks":
                    corners = s.get("value")
                    break
            if isinstance(corners, (int, float)):
                sum_c += corners
                if tinfo.get("id") == team_id:
                    team_c = corners
        if team_c is not None and sum_c > 0:
            team_vals.append(team_c)
            total_vals.append(sum_c)
            n += 1
        # light throttle
        time.sleep(0.12)
    return {"team_avg": _avg(team_vals), "total_avg": _avg(total_vals), "n": n}

def team_recent_cards(team_id: int, league_id: int = 39, season: int | None = None, last_n: int = 6):
    """
    Average yellow/red cards over last N fixtures using /fixtures/statistics -> types 'Yellow Cards'/'Red Cards'.
    """
    season = season or current_season()
    fx = get_fixtures_by_team(team_id, league_id, season, last=last_n).get("response", []) or []
    if not fx:
        return {"yellow_avg": None, "red_avg": None, "n": 0}

    yvals, rvals, n = [], [], 0
    for row in fx:
        fid = _safe(row, ["fixture", "id"])
        if not fid:
            continue
        st = get_fixture_statistics(fid).get("response", []) or []
        y, r = None, None
        for block in st:
            if _safe(block, ["team", "id"]) != team_id:
                continue
            for s in (block.get("statistics") or []):
                if s.get("type") == "Yellow Cards":
                    y = s.get("value")
                elif s.get("type") == "Red Cards":
                    r = s.get("value")
        if isinstance(y, (int, float)) and isinstance(r, (int, float)):
            yvals.append(y); rvals.append(r); n += 1
        time.sleep(0.12)
    return {"yellow_avg": _avg(yvals), "red_avg": _avg(rvals), "n": n}

def team_recent_fouls(team_id: int, league_id: int = 39, season: int | None = None, last_n: int = 6):
    """
    Average fouls committed over last N fixtures using /fixtures/statistics -> type 'Fouls'.
    """
    season = season or current_season()
    fx = get_fixtures_by_team(team_id, league_id, season, last=last_n).get("response", []) or []
    if not fx:
        return {"fouls_avg": None, "n": 0}

    fvals, n = [], 0
    for row in fx:
        fid = _safe(row, ["fixture", "id"])
        if not fid:
            continue
        st = get_fixture_statistics(fid).get("response", []) or []
        val = None
        for block in st:
            if _safe(block, ["team", "id"]) != team_id:
                continue
            for s in (block.get("statistics") or []):
                if s.get("type") == "Fouls":
                    val = s.get("value")
                    break
        if isinstance(val, (int, float)):
            fvals.append(val); n += 1
        time.sleep(0.12)
    return {"fouls_avg": _avg(fvals), "n": n}
