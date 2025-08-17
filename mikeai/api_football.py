# assistant/api_football.py
# Lightweight API-Football v3 wrapper used by views.py
# Endpoints: https://v3.football.api-sports.io

# import os
# import time
# from typing import Any, Dict, List, Optional, Tuple

# import requests
# from django.utils import timezone

# API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
# BASE_URL = "https://v3.football.api-sports.io"
# DEFAULT_TIMEOUT = 20

# # ----- simple in-memory cache (tiny TTL to reduce rate hits)
# _CACHE: Dict[str, Tuple[float, Any]] = {}
# CACHE_TTL = 60  # seconds

# # Common league ids
# LEAGUES = {
#     "EPL": 39,
#     "LA_LIGA": 140,
# }


# def _cache_get(key: str):
#     now = time.time()
#     item = _CACHE.get(key)
#     if not item:
#         return None
#     ts, val = item
#     if now - ts > CACHE_TTL:
#         _CACHE.pop(key, None)
#         return None
#     return val


# def _cache_set(key: str, val: Any):
#     _CACHE[key] = (time.time(), val)


# def _headers():
#     return {
#         "x-apisports-key": API_FOOTBALL_KEY or "",
#         "Accept": "application/json",
#     }


# def _get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     GET wrapper with tiny caching and error-hardening.
#     Returns parsed JSON dict (or {} on failure).
#     """
#     url = f"{BASE_URL}{path}"
#     key = f"{url}|{sorted(params.items())}"
#     try:
#         if API_FOOTBALL_KEY:
#             cached = _cache_get(key)
#             if cached is not None:
#                 return cached
#         r = requests.get(url, headers=_headers(), params=params, timeout=DEFAULT_TIMEOUT)
#         r.raise_for_status()
#         data = r.json()
#         if API_FOOTBALL_KEY:
#             _cache_set(key, data)
#         return data
#     except Exception:
#         return {}


# # ====================== Utilities ======================
# def current_season() -> int:
#     """
#     Returns the current 'football season' year (e.g., 2024 for 2024/25).
#     Simple rule: if month >= August -> use this year, else year-1.
#     """
#     now = timezone.now()
#     year = now.year
#     return year if now.month >= 8 else (year - 1)


# def last_season() -> int:
#     return current_season() - 1


# def _first_response_list(data: Dict[str, Any]) -> List[Any]:
#     """
#     API returns {"response": [...]}
#     """
#     try:
#         arr = data.get("response")
#         return arr if isinstance(arr, list) else []
#     except Exception:
#         return []


# def _first_response_object(data: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Some endpoints return {"response": {...}} or {"response": [obj]}
#     We coerce to a flat dict for convenience.
#     """
#     try:
#         resp = data.get("response")
#         if isinstance(resp, dict):
#             return resp
#         if isinstance(resp, list) and resp:
#             return resp[0]
#     except Exception:
#         pass
#     return {}


# # ====================== Core lookups used by views.py ======================
# def search_team_id(team_name: str, league_id: int = LEAGUES["EPL"], season: Optional[int] = None) -> Optional[int]:
#     """
#     Returns a team id for a given league/season by searching name.
#     """
#     if not team_name:
#         return None
#     if season is None:
#         season = current_season()
#     q = _get("/teams", {"search": team_name, "league": league_id, "season": season})
#     for item in _first_response_list(q):
#         t = item.get("team") or {}
#         if t.get("id"):
#             return t["id"]
#     # fallback: try without league filter
#     q2 = _get("/teams", {"search": team_name, "season": season})
#     for item in _first_response_list(q2):
#         t = item.get("team") or {}
#         if t.get("id"):
#             return t["id"]
#     return None


# def get_team_stats(team_id: int, league_id: int = LEAGUES["EPL"], season: Optional[int] = None) -> Dict[str, Any]:
#     """
#     Wrapper for /teams/statistics
#     views.py expects {"response": {...}}
#     """
#     if season is None:
#         season = current_season()
#     data = _get("/teams/statistics", {"league": league_id, "season": season, "team": team_id})
#     stats = _first_response_object(data)
#     return {"response": stats}


# def get_injuries(team_id: int, season: Optional[int] = None) -> Dict[str, Any]:
#     """
#     Wrapper for /injuries
#     Returns {"response": [ ... ]}
#     """
#     if season is None:
#         season = current_season()
#     data = _get("/injuries", {"team": team_id, "season": season})
#     arr = _first_response_list(data)
#     cleaned = []
#     for it in arr:
#         try:
#             cleaned.append({
#                 "player": {"name": ((it.get("player") or {}).get("name") or "")},
#                 "team": it.get("team"),
#                 "fixture": it.get("fixture"),
#                 "league": it.get("league"),
#                 "player_reason": (it.get("player") or {}).get("reason"),
#             })
#         except Exception:
#             continue
#     return {"response": cleaned}


# def get_head_to_head(team1_id: int, team2_id: int, last: int = 5) -> Dict[str, Any]:
#     """
#     Wrapper for /fixtures/headtohead
#     Returns {"response": [fixtures...]}
#     """
#     data = _get("/fixtures/headtohead", {"h2h": f"{team1_id}-{team2_id}", "last": last})
#     return {"response": _first_response_list(data)}


# def get_fixtures_by_range(start_date: str, end_date: str, league_id: int = LEAGUES["EPL"]) -> Dict[str, Any]:
#     """
#     Wrapper for /fixtures with date range
#     Returns {"response": [fixtures...]}
#     Dates: YYYY-MM-DD
#     """
#     data = _get("/fixtures", {"league": league_id, "from": start_date, "to": end_date})
#     return {"response": _first_response_list(data)}


# # ====================== Last-N team props (corners/cards/fouls) ======================
# def _last_fixtures_ids(team_id: int, league_id: int, season: int, last_n: int = 6, venue: Optional[str] = None) -> List[int]:
#     """
#     Get last N fixture ids for a team (optionally venue=home/away).
#     """
#     params = {"team": team_id, "league": league_id, "season": season, "last": last_n}
#     if venue in ("home", "away"):
#         params["venue"] = venue
#     data = _get("/fixtures", params)
#     ids = []
#     for fx in _first_response_list(data):
#         fid = (fx.get("fixture") or {}).get("id")
#         if fid:
#             ids.append(fid)
#     return ids


# def _fixture_stats_for_team(fixture_id: int, team_id: int) -> Dict[str, Any]:
#     """
#     Use /fixtures/statistics?fixture=... and find the team's row.
#     """
#     data = _get("/fixtures/statistics", {"fixture": fixture_id})
#     for row in _first_response_list(data):
#         team = row.get("team") or {}
#         if team.get("id") == team_id:
#             stats = {}
#             for it in (row.get("statistics") or []):
#                 t = (it.get("type") or "").lower()
#                 v = it.get("value")
#                 stats[t] = (v if isinstance(v, (int, float)) else (int(v) if (isinstance(v, str) and v.isdigit()) else 0))
#             return stats
#     return {}


# def team_recent_corners(team_id: int, league_id: int = LEAGUES["EPL"], season: Optional[int] = None, last_n: int = 6) -> Dict[str, Any]:
#     """
#     Returns: {"team_avg": float, "total_avg": float, "n": int}
#     We pull last N fixtures, grab both teams' "Corner Kicks".
#     """
#     if season is None:
#         season = current_season()
#     fids = _last_fixtures_ids(team_id, league_id, season, last_n)
#     if not fids:
#         return {}
#     team_sum = 0
#     total_sum = 0
#     count = 0
#     for fid in fids:
#         s_team = _fixture_stats_for_team(fid, team_id)
#         fx = _get("/fixtures", {"id": fid})
#         resp = _first_response_list(fx)
#         if not resp:
#             continue
#         teams = (resp[0].get("teams") or {})
#         home = teams.get("home") or {}
#         away = teams.get("away") or {}
#         opp_id = away.get("id") if home.get("id") == team_id else home.get("id")
#         s_opp = _fixture_stats_for_team(fid, opp_id) if opp_id else {}

#         t_c = s_team.get("corner kicks", 0)
#         o_c = s_opp.get("corner kicks", 0)
#         team_sum += (t_c or 0)
#         total_sum += ((t_c or 0) + (o_c or 0))
#         count += 1
#     if count == 0:
#         return {}
#     return {"team_avg": round(team_sum / count, 2), "total_avg": round(total_sum / count, 2), "n": count}


# def team_recent_cards(team_id: int, league_id: int = LEAGUES["EPL"], season: Optional[int] = None, last_n: int = 6) -> Dict[str, Any]:
#     """
#     Returns: {"yellow_avg": float, "red_avg": float, "n": int}
#     """
#     if season is None:
#         season = current_season()
#     fids = _last_fixtures_ids(team_id, league_id, season, last_n)
#     if not fids:
#         return {}
#     y_sum = r_sum = count = 0
#     for fid in fids:
#         s_team = _fixture_stats_for_team(fid, team_id)
#         y = s_team.get("yellow cards", 0) or 0
#         r = s_team.get("red cards", 0) or 0
#         y_sum += y
#         r_sum += r
#         count += 1
#     if count == 0:
#         return {}
#     return {"yellow_avg": round(y_sum / count, 2), "red_avg": round(r_sum / count, 2), "n": count}


# def team_recent_fouls(team_id: int, league_id: int = LEAGUES["EPL"], season: Optional[int] = None, last_n: int = 6) -> Dict[str, Any]:
#     """
#     Returns: {"fouls_avg": float, "n": int}
#     """
#     if season is None:
#         season = current_season()
#     fids = _last_fixtures_ids(team_id, league_id, season, last_n)
#     if not fids:
#         return {}
#     f_sum = count = 0
#     for fid in fids:
#         s_team = _fixture_stats_for_team(fid, team_id)
#         f = s_team.get("fouls", 0) or 0
#         f_sum += f
#         count += 1
#     if count == 0:
#         return {}
#     return {"fouls_avg": round(f_sum / count, 2), "n": count}


# # ====================== Extra helpers for views' direct Q&A ======================
# def player_assists_current(player_name: str, league_id: int = LEAGUES["EPL"], season: Optional[int] = None) -> Dict[str, Any]:
#     """
#     Returns: {"assists": int, "team": "Name"} for the current season.
#     Tries the specified league first, then falls back to cross-league search.
#     """
#     if not player_name:
#         return {}
#     if season is None:
#         season = current_season()

#     # Try with league filter first
#     data = _get("/players", {"search": player_name, "league": league_id, "season": season})
#     arr = _first_response_list(data)

#     # Fallback: without league (player might be in another league)
#     if not arr:
#         data = _get("/players", {"search": player_name, "season": season})
#         arr = _first_response_list(data)

#     best = None
#     for p in arr:
#         stats_list = (p.get("statistics") or [])
#         for st in stats_list:
#             team = (st.get("team") or {}).get("name")
#             goals = st.get("goals") or {}
#             assists = goals.get("assists")
#             if assists is None:
#                 continue
#             try:
#                 ai = int(assists)
#             except Exception:
#                 continue
#             # take the highest assists entry for the season
#             if best is None or ai > best["assists"]:
#                 best = {"assists": ai, "team": team}
#     return best or {}


# def player_current_club(player_name: str, season: Optional[int] = None) -> Dict[str, Any]:
#     """
#     Tries to infer the player's current club for the given season (defaults to current season)
#     using /players search (statistics team for this season). If not found, tries /transfers.
#     Returns {"team": "Club Name"} or {}.
#     """
#     if not player_name:
#         return {}
#     if season is None:
#         season = current_season()

#     # Primary: players statistics for the season
#     data = _get("/players", {"search": player_name, "season": season})
#     for p in _first_response_list(data):
#         stats = p.get("statistics") or []
#         for st in stats:
#             team = (st.get("team") or {}).get("name")
#             if team:
#                 return {"team": team}

#     # Fallback: transfers (latest incoming team)
#     data = _get("/transfers", {"player": player_name})
#     # When searching by player name the API may not match; try resolving player id first
#     if not _first_response_list(data):
#         psearch = _get("/players", {"search": player_name})
#         for cand in _first_response_list(psearch):
#             pid = (cand.get("player") or {}).get("id")
#             if not pid:
#                 continue
#             data = _get("/transfers", {"player": pid})
#             break

#     latest_team = None
#     latest_ts = None
#     for row in _first_response_list(data):
#         transfers = row.get("transfers") or []
#         for tr in transfers:
#             date = tr.get("date")  # "2024-08-20"
#             team_in = (tr.get("teams") or {}).get("in") or {}
#             name = team_in.get("name")
#             if name and date:
#                 # take most recent
#                 if latest_ts is None or str(date) > latest_ts:
#                     latest_ts = str(date)
#                     latest_team = name
#     return {"team": latest_team} if latest_team else {}


# def team_last_home_fixtures(team_id: int, last_n: int = 10, league_id: int = LEAGUES["EPL"], season: Optional[int] = None) -> List[Dict[str, Any]]:
#     """
#     Fetch last N home fixtures for a team.
#     Returns the raw fixtures list like /fixtures would provide.
#     """
#     if season is None:
#         season = current_season()
#     data = _get("/fixtures", {"team": team_id, "league": league_id, "season": season, "venue": "home", "last": last_n})
#     return _first_response_list(data)


# # ====================== High-level summaries for common Qs ======================
# def team_avg_goals_last_season(team_id: int, league_id: int = LEAGUES["EPL"]) -> Dict[str, Any]:
#     """
#     Average goals scored/conceded per match for LAST season using /teams/statistics.
#     Returns {"scored_per_match": float, "conceded_per_match": float, "matches": int, "season": int}
#     """
#     season = last_season()
#     data = _get("/teams/statistics", {"league": league_id, "season": season, "team": team_id})
#     stats = _first_response_object(data)
#     fixtures = (stats.get("fixtures") or {}).get("played") or {}
#     played = int((fixtures.get("total") or 0) or 0)

#     goals = stats.get("goals") or {}
#     gf_total = ((goals.get("for") or {}).get("total") or {}).get("total") or 0
#     ga_total = ((goals.get("against") or {}).get("total") or {}).get("total") or 0
#     try:
#         gf_total = int(gf_total)
#         ga_total = int(ga_total)
#     except Exception:
#         gf_total = int(gf_total or 0)
#         ga_total = int(ga_total or 0)

#     scored_pm = round(gf_total / played, 2) if played else 0.0
#     conceded_pm = round(ga_total / played, 2) if played else 0.0
#     return {
#         "scored_per_match": scored_pm,
#         "conceded_per_match": conceded_pm,
#         "matches": played,
#         "season": season,
#     }


# def home_performance_summary(team_id: int, last_n: int = 10, league_id: int = LEAGUES["EPL"], season: Optional[int] = None) -> Dict[str, Any]:
#     """
#     Summarize last-N home fixtures: W-D-L, avg GF/GA.
#     Returns {"W":int,"D":int,"L":int,"gf_avg":float,"ga_avg":float,"n":int}
#     """
#     fixtures = team_last_home_fixtures(team_id, last_n=last_n, league_id=league_id, season=season)
#     if not fixtures:
#         return {}
#     w = d = l = 0
#     gf_sum = ga_sum = 0
#     n = 0
#     for fx in fixtures:
#         t = fx.get("teams") or {}
#         home = t.get("home") or {}
#         away = t.get("away") or {}
#         if not home.get("id") or not away.get("id"):
#             continue
#         score = (fx.get("score") or {}).get("fulltime") or {}
#         sh = score.get("home")
#         sa = score.get("away")
#         # fallback to goals
#         if sh is None or sa is None:
#             goals = (fx.get("goals") or {})
#             sh = goals.get("home")
#             sa = goals.get("away")
#         try:
#             sh_i = int(sh if sh is not None else 0)
#             sa_i = int(sa if sa is not None else 0)
#         except Exception:
#             sh_i = int(sh or 0)
#             sa_i = int(sa or 0)

#         # result
#         if sh_i > sa_i:
#             w += 1
#         elif sh_i == sa_i:
#             d += 1
#         else:
#             l += 1

#         gf_sum += sh_i
#         ga_sum += sa_i
#         n += 1

#     if n == 0:
#         return {}
#     return {
#         "W": w, "D": d, "L": l,
#         "gf_avg": round(gf_sum / n, 2),
#         "ga_avg": round(ga_sum / n, 2),
#         "n": n,
#     }
# # ====== Standings + recent form helpers (add to api_football.py) ======

# def get_standings(league_id: int, season: int) -> List[Dict[str, Any]]:
#     """
#     Returns a flat list of table rows from API-Football:
#     [
#       {
#         "rank": 1, "team": {"id":..,"name":"...","logo":"..."},
#         "points": 0, "goalsDiff": 0, "form": None,
#         "all": {"played":0,"win":0,"draw":0,"lose":0,"goals":{"for":0,"against":0}},
#         ...
#       }, ...
#     ]
#     """
#     params = {"league": league_id, "season": int(season)}
#     data = _get("/standings", params)
#     try:
#         resp = data.get("response") or []
#         if not resp:
#             return []
#         league_obj = resp[0].get("league") or {}
#         grid = league_obj.get("standings") or []
#         # API shape: standings is a 2D array; first group is the main table
#         table = grid[0] if grid and isinstance(grid[0], list) else []
#         return table
#     except Exception:
#         return []


# def team_form_last_n(team_id: int, n: int = 5, league_id: int = 39, season: Optional[int] = None) -> Dict[str, Any]:
#     """
#     Compute recent form string (e.g., 'WWDLW') and simple win-rate % over last N fixtures.
#     Uses /fixtures?team=&league=&season=&last=N
#     """
#     if season is None:
#         season = current_season()
#     try:
#         data = _get("/fixtures", {
#             "team": team_id, "league": league_id, "season": int(season), "last": int(n)
#         })
#         rows = _first_response_list(data)
#         if not rows:
#             return {"form": "", "win_rate": 0}
#         form_chars: List[str] = []
#         wins = 0
#         total = 0
#         for fx in rows:
#             score = (fx.get("score") or {}).get("fulltime") or {}
#             teams = fx.get("teams") or {}
#             home = teams.get("home") or {}
#             away = teams.get("away") or {}
#             h = score.get("home")
#             a = score.get("away")
#             if h is None or a is None:
#                 # fallback: try overall "goals" if fulltime missing
#                 g = (fx.get("goals") or {})
#                 h = g.get("home", 0)
#                 a = g.get("away", 0)
#             total += 1
#             # detect which side is the team
#             if home.get("id") == team_id:
#                 if h > a:
#                     form_chars.append('W'); wins += 1
#                 elif h == a:
#                     form_chars.append('D')
#                 else:
#                     form_chars.append('L')
#             else:
#                 if a > h:
#                     form_chars.append('W'); wins += 1
#                 elif a == h:
#                     form_chars.append('D')
#                 else:
#                     form_chars.append('L')
#         form_str = "".join(form_chars)
#         win_rate = int(round((wins / max(1, total)) * 100))
#         return {"form": form_str, "win_rate": win_rate}
#     except Exception:
#         return {"form": "", "win_rate": 0}


# def get_fixtures_next(league_id: int, n: int = 20, season: Optional[int] = None) -> Dict[str, Any]:
#     if season is None:
#         season = current_season()
#     data = _get("/fixtures", {"league": league_id, "season": season, "next": n})
#     return {"response": _first_response_list(data)}


# assistant/api_football.py
# Lightweight API-Football v3 wrapper used by views.py
# Endpoints: https://v3.football.api-sports.io

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from django.utils import timezone

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
DEFAULT_TIMEOUT = 20

# ----- simple in-memory cache (tiny TTL to reduce rate hits)
_CACHE: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL = 60  # seconds


def _cache_get(key: str):
    now = time.time()
    item = _CACHE.get(key)
    if not item:
        return None
    ts, val = item
    if now - ts > CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return val


def _cache_set(key: str, val: Any):
    _CACHE[key] = (time.time(), val)


def _headers():
    return {
        "x-apisports-key": API_FOOTBALL_KEY or "",
        "Accept": "application/json",
    }


def _clean_season(season: Optional[int | str]) -> Optional[int]:
    """Coerce season into a clean 4-digit int (e.g., 2025)."""
    if season is None:
        return None
    try:
        s = str(season).strip()
        if len(s) > 4 and s.isdigit():
            s = s[:4]
        return int(s)
    except Exception:
        return None


def _get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET wrapper with tiny caching and error-hardening.
    Returns parsed JSON dict (or {} on failure).
    """
    url = f"{BASE_URL}{path}"
    # normalize season in params
    if "season" in params:
        cs = _clean_season(params["season"])
        if cs is not None:
            params = {**params, "season": cs}

    # build a poor-man cache key
    key = f"{url}|{sorted(params.items())}"
    try:
        if API_FOOTBALL_KEY:
            cached = _cache_get(key)
            if cached is not None:
                return cached
        r = requests.get(url, headers=_headers(), params=params, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if API_FOOTBALL_KEY:
            _cache_set(key, data)
        return data
    except Exception:
        return {}


# ====================== Utilities ======================
def current_season() -> int:
    """
    Returns the current 'football season' year (e.g., 2025 for 2025/26).
    Simple rule: if month >= August -> use this year, else year-1.
    """
    now = timezone.now()
    year = now.year
    return year if now.month >= 8 else (year - 1)


def _first_response_list(data: Dict[str, Any]) -> List[Any]:
    """
    API returns {"response": [...]}
    """
    try:
        arr = data.get("response")
        return arr if isinstance(arr, list) else []
    except Exception:
        return []


def _first_response_object(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Some endpoints return {"response": {...}} or {"response": [obj]}
    We coerce to a flat dict for convenience.
    """
    try:
        resp = data.get("response")
        if isinstance(resp, dict):
            return resp
        if isinstance(resp, list) and resp:
            return resp[0]
    except Exception:
        pass
    return {}


# ====================== Core lookups used by views.py ======================
def search_team_id(team_name: str, league_id: int = 39, season: Optional[int] = None) -> Optional[int]:
    """
    Returns a team id for a given league/season by searching name.
    """
    if not team_name:
        return None
    if season is None:
        season = current_season()
    q = _get("/teams", {"search": team_name, "league": league_id, "season": season})
    for item in _first_response_list(q):
        t = item.get("team") or {}
        if t.get("id"):
            return t["id"]
    # fallback: try without league filter
    q2 = _get("/teams", {"search": team_name, "season": season})
    for item in _first_response_list(q2):
        t = item.get("team") or {}
        if t.get("id"):
            return t["id"]
    return None


def get_team_stats(team_id: int, league_id: int = 39, season: Optional[int] = None) -> Dict[str, Any]:
    """
    Wrapper for /teams/statistics
    views.py expects {"response": {...}}
    """
    if season is None:
        season = current_season()
    data = _get("/teams/statistics", {"league": league_id, "season": season, "team": team_id})
    stats = _first_response_object(data)
    return {"response": stats}


def get_injuries(team_id: int, season: Optional[int] = None) -> Dict[str, Any]:
    """
    Wrapper for /injuries
    Returns {"response": [ ... ]}
    """
    if season is None:
        season = current_season()
    data = _get("/injuries", {"team": team_id, "season": season})
    arr = _first_response_list(data)
    # normalize player names inline (API varies)
    cleaned = []
    for it in arr:
        try:
            cleaned.append({
                "player": {"name": ((it.get("player") or {}).get("name") or "")},
                "team": it.get("team"),
                "fixture": it.get("fixture"),
                "league": it.get("league"),
                "player_reason": (it.get("player") or {}).get("reason"),
            })
        except Exception:
            continue
    return {"response": cleaned}


def get_head_to_head(team1_id: int, team2_id: int, last: int = 5) -> Dict[str, Any]:
    """
    Wrapper for /fixtures/headtohead
    Returns {"response": [fixtures...]}
    """
    data = _get("/fixtures/headtohead", {"h2h": f"{team1_id}-{team2_id}", "last": last})
    return {"response": _first_response_list(data)}


def get_fixtures_by_range(start_date: str, end_date: str, league_id: int = 39) -> Dict[str, Any]:
    """
    Wrapper for /fixtures with date range
    Returns {"response": [fixtures...]}
    Dates: YYYY-MM-DD
    """
    data = _get("/fixtures", {"league": league_id, "from": start_date, "to": end_date})
    return {"response": _first_response_list(data)}


# ====================== Standings & Form ======================
def get_standings(league_id: int, season: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Wrapper for /standings
    Returns the *first table array* (flattened list of rows).
    Shape per row matches API-Football: rank, team{name,logo,id}, points, goalsDiff, all{played,win,draw,lose,goals{for,against}}, etc.
    """
    if season is None:
        season = current_season()
    data = _get("/standings", {"league": league_id, "season": season})
    # Structure: {"response":[{"league":{"standings":[[ rows ]]}}]}
    resp = _first_response_object(data)
    league = (resp.get("league") or {})
    standings = league.get("standings") or []
    if isinstance(standings, list) and standings:
        table = standings[0]
        if isinstance(table, list):
            return table
    return []


def team_form_last_n(team_id: int, n: int = 5, league_id: int = 39, season: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute simple form over last N fixtures for this team (W/D/L) + win rate %.
    Uses /fixtures?team=&league=&season=&last=n
    """
    if season is None:
        season = current_season()
    n = max(3, min(10, int(n or 5)))
    # Last N fixtures (any venue). API returns most-recent first.
    data = _get("/fixtures", {"team": team_id, "league": league_id, "season": season, "last": n})
    arr = _first_response_list(data)
    form_chars: List[str] = []
    w = 0
    for fx in arr[:n]:
        teams = fx.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        score_ft = (fx.get("score") or {}).get("fulltime") or {}
        h, a = score_ft.get("home"), score_ft.get("away")
        if h is None or a is None:
            # fall back to 'winner' flag if no FT score
            win_home = home.get("winner")
            win_away = away.get("winner")
            if win_home is True and home.get("id") == team_id:
                form_chars.append("W"); w += 1
            elif win_away is True and away.get("id") == team_id:
                form_chars.append("W"); w += 1
            elif win_home is False and home.get("id") == team_id:
                form_chars.append("L")
            elif win_away is False and away.get("id") == team_id:
                form_chars.append("L")
            else:
                form_chars.append("D")
            continue

        # compare FT score from the perspective of team_id
        if home.get("id") == team_id:
            if h > a:
                form_chars.append("W"); w += 1
            elif h == a:
                form_chars.append("D")
            else:
                form_chars.append("L")
        elif away.get("id") == team_id:
            if a > h:
                form_chars.append("W"); w += 1
            elif a == h:
                form_chars.append("D")
            else:
                form_chars.append("L")
    # Trim to N just in case
    form = "".join(form_chars[:n])
    win_rate = int(round((w / max(1, min(n, len(form_chars)))) * 100))
    return {"form": form, "win_rate": win_rate}


# ====================== Last-N team props (corners/cards/fouls) ======================
def _last_fixtures_ids(team_id: int, league_id: int, season: int, last_n: int = 6, venue: Optional[str] = None) -> List[int]:
    """
    Get last N fixture ids for a team (optionally venue=home/away).
    """
    params = {"team": team_id, "league": league_id, "season": season, "last": last_n}
    if venue in ("home", "away"):
        params["venue"] = venue
    data = _get("/fixtures", params)
    ids = []
    for fx in _first_response_list(data):
        fid = (fx.get("fixture") or {}).get("id")
        if fid:
            ids.append(fid)
    return ids


def _fixture_stats_for_team(fixture_id: int, team_id: int) -> Dict[str, Any]:
    """
    Use /fixtures/statistics?fixture=... and find the team's row.
    """
    data = _get("/fixtures/statistics", {"fixture": fixture_id})
    for row in _first_response_list(data):
        team = row.get("team") or {}
        if team.get("id") == team_id:
            # statistics is a list of {"type": "Total Shots", "value": 10} etc
            stats = {}
            for it in (row.get("statistics") or []):
                t = (it.get("type") or "").lower()
                v = it.get("value")
                # normalize None -> 0
                stats[t] = (v if isinstance(v, (int, float)) else (int(v) if (isinstance(v, str) and v.isdigit()) else 0))
            return stats
    return {}


def team_recent_corners(team_id: int, league_id: int = 39, season: Optional[int] = None, last_n: int = 6) -> Dict[str, Any]:
    """
    Returns: {"team_avg": float, "total_avg": float, "n": int}
    We pull last N fixtures, grab both teams' "Corner Kicks".
    """
    if season is None:
        season = current_season()
    fids = _last_fixtures_ids(team_id, league_id, season, last_n)
    if not fids:
        return {}
    team_sum = 0
    total_sum = 0
    count = 0
    for fid in fids:
        # team stats
        s_team = _fixture_stats_for_team(fid, team_id)
        # opponent stats (need opponent id: fetch fixture once)
        fx = _get("/fixtures", {"id": fid})
        resp = _first_response_list(fx)
        if not resp:
            continue
        teams = (resp[0].get("teams") or {})
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        opp_id = away.get("id") if home.get("id") == team_id else home.get("id")
        s_opp = _fixture_stats_for_team(fid, opp_id) if opp_id else {}

        t_c = s_team.get("corner kicks", 0)
        o_c = s_opp.get("corner kicks", 0)
        team_sum += (t_c or 0)
        total_sum += ((t_c or 0) + (o_c or 0))
        count += 1
    if count == 0:
        return {}
    return {"team_avg": round(team_sum / count, 2), "total_avg": round(total_sum / count, 2), "n": count}


def team_recent_cards(team_id: int, league_id: int = 39, season: Optional[int] = None, last_n: int = 6) -> Dict[str, Any]:
    """
    Returns: {"yellow_avg": float, "red_avg": float, "n": int}
    """
    if season is None:
        season = current_season()
    fids = _last_fixtures_ids(team_id, league_id, season, last_n)
    if not fids:
        return {}
    y_sum = r_sum = count = 0
    for fid in fids:
        s_team = _fixture_stats_for_team(fid, team_id)
        y = s_team.get("yellow cards", 0) or 0
        r = s_team.get("red cards", 0) or 0
        y_sum += y
        r_sum += r
        count += 1
    if count == 0:
        return {}
    return {"yellow_avg": round(y_sum / count, 2), "red_avg": round(r_sum / count, 2), "n": count}


def team_recent_fouls(team_id: int, league_id: int = 39, season: Optional[int] = None, last_n: int = 6) -> Dict[str, Any]:
    """
    Returns: {"fouls_avg": float, "n": int}
    """
    if season is None:
        season = current_season()
    fids = _last_fixtures_ids(team_id, league_id, season, last_n)
    if not fids:
        return {}
    f_sum = count = 0
    for fid in fids:
        s_team = _fixture_stats_for_team(fid, team_id)
        f = s_team.get("fouls", 0) or 0
        f_sum += f
        count += 1
    if count == 0:
        return {}
    return {"fouls_avg": round(f_sum / count, 2), "n": count}


# ====================== Extra helpers for views' direct Q&A ======================
def player_assists_current(player_name: str, league_id: int = 39, season: Optional[int] = None) -> Dict[str, Any]:
    """
    Returns: {"assists": int, "team": "Name"} for the current season.
    Uses /players?search=Name&league=&season=
    """
    if not player_name:
        return {}
    if season is None:
        season = current_season()
    data = _get("/players", {"search": player_name, "league": league_id, "season": season})
    # response is list of players with statistics list
    for p in _first_response_list(data):
        stats = (p.get("statistics") or [])
        if not stats:
            continue
        # Usually first statistics entry corresponds to that league/season team
        st = stats[0]
        team = (st.get("team") or {}).get("name")
        goals = st.get("goals") or {}
        assists = goals.get("assists")
        if assists is not None:
            try:
                return {"assists": int(assists), "team": team}
            except Exception:
                pass
    return {}


def team_last_home_fixtures(team_id: int, last_n: int = 10, league_id: int = 39, season: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch last N home fixtures for a team.
    Returns the raw fixtures list like /fixtures would provide.
    """
    if season is None:
        season = current_season()
    data = _get("/fixtures", {"team": team_id, "league": league_id, "season": season, "venue": "home", "last": last_n})
    return _first_response_list(data)

