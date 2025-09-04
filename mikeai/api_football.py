# assistant/api_football.py
# API-Football v3 helpers with robust EPL team resolver
# Docs: https://v3.football.api-sports.io

import os
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import requests
from django.utils import timezone

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
DEFAULT_TIMEOUT = 20

# ----- tiny in-memory cache
_CACHE: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes default

def _cache_get(key: str):
    item = _CACHE.get(key)
    if not item:
        return None
    ts, val = item
    if time.time() - ts > CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return val

def _cache_set(key: str, val: Any):
    _CACHE[key] = (time.time(), val)

def _headers():
    return {"x-apisports-key": API_FOOTBALL_KEY or "", "Accept": "application/json"}

def _clean_season(season: Optional[int | str]) -> Optional[int]:
    if season is None: return None
    try:
        s = str(season).strip()
        if len(s) > 4 and s.isdigit(): s = s[:4]
        return int(s)
    except Exception:
        return None

def _get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    if "season" in params:
        cs = _clean_season(params["season"])
        if cs is not None:
            params = {**params, "season": cs}
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

def current_season() -> int:
    now = timezone.now()
    return now.year if now.month >= 8 else (now.year - 1)

def _first_response_list(data: Dict[str, Any]) -> List[Any]:
    try:
        arr = data.get("response")
        return arr if isinstance(arr, list) else []
    except Exception:
        return []

def _first_response_object(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        resp = data.get("response")
        if isinstance(resp, dict):
            return resp
        if isinstance(resp, list) and resp:
            return resp[0]
    except Exception:
        pass
    return {}

# -------------- Normalization --------------
def _norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    for junk in [" football club"," afc"," fc"," cf",".",",","-","_","|","/","\\","(",")","[","]","{","}",":",";","•","—"]:
        s = s.replace(junk, " ")
    return " ".join(s.split())

# -------------- Core lookups --------------
def search_team_id(team_name: str, league_id: int = 39, season: Optional[int] = None) -> Optional[int]:
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
    if season is None:
        season = current_season()
    data = _get("/teams/statistics", {"league": league_id, "season": season, "team": team_id})
    stats = _first_response_object(data)
    return {"response": stats}

def get_injuries(team_id: int, season: Optional[int] = None) -> Dict[str, Any]:
    if season is None: season = current_season()
    data = _get("/injuries", {"team": team_id, "season": season})
    arr = _first_response_list(data)
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
    data = _get("/fixtures/headtohead", {"h2h": f"{team1_id}-{team2_id}", "last": last})
    return {"response": _first_response_list(data)}

def get_fixtures_by_range(start_date: str, end_date: str, league_id: int = 39) -> Dict[str, Any]:
    data = _get("/fixtures", {"league": league_id, "from": start_date, "to": end_date})
    return {"response": _first_response_list(data)}

def get_standings(league_id: int, season: Optional[int] = None) -> List[Dict[str, Any]]:
    if season is None: season = current_season()
    data = _get("/standings", {"league": league_id, "season": season})
    resp = _first_response_object(data)
    league = (resp.get("league") or {})
    standings = league.get("standings") or []
    if isinstance(standings, list) and standings:
        table = standings[0]
        if isinstance(table, list):
            return table
    return []

def team_form_last_n(team_id: int, n: int = 5, league_id: int = 39, season: Optional[int] = None) -> Dict[str, Any]:
    if season is None: season = current_season()
    n = max(3, min(10, int(n or 5)))
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
        if home.get("id") == team_id:
            if h > a: form_chars.append("W"); w += 1
            elif h == a: form_chars.append("D")
            else: form_chars.append("L")
        elif away.get("id") == team_id:
            if a > h: form_chars.append("W"); w += 1
            elif a == h: form_chars.append("D")
            else: form_chars.append("L")
    form = "".join(form_chars[:n])
    win_rate = int(round((w / max(1, min(n, len(form_chars)))) * 100))
    return {"form": form, "win_rate": win_rate}

def team_last_fixture(team_id: int, league_id: int = 39, season: Optional[int] = None) -> Dict[str, Any]:
    if season is None: season = current_season()
    data = _get("/fixtures", {"team": team_id, "league": league_id, "season": season, "last": 1})
    arr = _first_response_list(data)
    return arr[0] if arr else {}

def team_fixtures_on_date(team_id: int, date: str, league_id: int = 39, season: Optional[int] = None) -> List[Dict[str, Any]]:
    if season is None: season = current_season()
    data = _get("/fixtures", {"team": team_id, "league": league_id, "season": season, "date": date})
    return _first_response_list(data)

def team_next_fixture(team_id: int, league_id: int = 39, season: Optional[int] = None) -> Dict[str, Any]:
    if season is None: season = current_season()
    data = _get("/fixtures", {"team": team_id, "league": league_id, "season": season, "next": 1})
    arr = _first_response_list(data)
    return arr[0] if arr else {}

def fixture_stats_both(fixture_id: int) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    data = _get("/fixtures/statistics", {"fixture": fixture_id})
    for row in _first_response_list(data):
        team = (row.get("team") or {}).get("id")
        if not team: continue
        stats: Dict[str, Any] = {}
        for it in (row.get("statistics") or []):
            t = (it.get("type") or "").lower()
            v = it.get("value")
            stats[t] = v if isinstance(v, (int, float)) else (int(v) if (isinstance(v, str) and v.isdigit()) else 0)
        out[team] = stats
    return out

def team_last_n_results(team_id: int, n: int = 5, league_id: int = 39, season: Optional[int] = None) -> List[Dict[str, Any]]:
    if season is None: season = current_season()
    n = max(1, min(10, int(n or 5)))
    data = _get("/fixtures", {"team": team_id, "league": league_id, "season": season, "last": n})
    rows = _first_response_list(data)
    out = []
    for fx in rows:
        f = fx.get("fixture") or {}
        dt = f.get("date") or ""
        teams = fx.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        score_ft = (fx.get("score") or {}).get("fulltime") or {}
        h, a = score_ft.get("home"), score_ft.get("away")
        if h is None or a is None:
            goals = fx.get("goals") or {}
            h, a = goals.get("home", 0), goals.get("away", 0)
        is_home = home.get("id") == team_id
        opponent = (away if is_home else home).get("name")
        venue = "H" if is_home else "A"
        r = "D"
        if (is_home and h > a) or ((not is_home) and a > h): r = "W"
        elif (is_home and h < a) or ((not is_home) and a < h): r = "L"
        out.append({
            "date": dt[:16].replace("T", " "),
            "opponent": opponent,
            "venue": venue,
            "score": f"{h}-{a}" if is_home else f"{a}-{h}",
            "result": r,
        })
    return out

# ---------- New: raw last-N fixtures (with fixture IDs) ----------
def team_last_n_fixtures(team_id: int, n: int = 5, league_id: int = 39, season: Optional[int] = None) -> List[Dict[str, Any]]:
    """Returns last-N league fixtures for a team with fixture id, date, and home/away info."""
    if season is None: season = current_season()
    n = max(1, min(10, int(n or 5)))
    data = _get("/fixtures", {"team": team_id, "league": league_id, "season": season, "last": n})
    rows = _first_response_list(data)
    out = []
    for fx in rows:
        f = fx.get("fixture") or {}
        t = fx.get("teams") or {}
        out.append({
            "fixture_id": (f.get("id") or 0),
            "date": (f.get("date") or ""),
            "home": t.get("home") or {},
            "away": t.get("away") or {},
        })
    return out

# ---------------- League team map & robust resolver ----------------
_LEAGUE_TEAM_MAP: Dict[str, Dict[str, int]] = {}  # key: f"{league_id}:{season}" -> {normalized_name: id}

def _league_key(league_id: int, season: Optional[int]) -> str:
    return f"{league_id}:{season or current_season()}"

def list_league_teams_map(league_id: int = 39, season: Optional[int] = None) -> Dict[str, int]:
    """Build a name->id map for a league season so we don't rely on brittle 'search'."""
    if season is None: season = current_season()
    key = _league_key(league_id, season)
    cached = _cache_get(f"TEAMMAP|{key}")
    if cached is not None:
        return cached
    data = _get("/teams", {"league": league_id, "season": season})
    out: Dict[str, int] = {}
    for item in _first_response_list(data):
        team = item.get("team") or {}
        tid = team.get("id"); name = (team.get("name") or "").strip()
        code = (team.get("code") or "").strip()
        short = (team.get("shortName") or "").strip()
        if not tid or not name: continue
        names = {name, name.replace("&", "and")}
        if code: names.add(code)
        if short: names.add(short)
        for n in names:
            out[_norm(n)] = tid
    _cache_set(f"TEAMMAP|{key}", out)
    return out

# Common EPL nicknames -> canonical names (so they match map keys)
NICK_TO_CANON = {
    "spurs":"tottenham hotspur",
    "tottenham":"tottenham hotspur",
    "man city":"manchester city",
    "manchester city":"manchester city",
    "city":"manchester city",
    "man united":"manchester united",
    "man utd":"manchester united",
    "manchester united":"manchester united",
    "united":"manchester united",
    "west ham":"west ham united",
    "westham":"west ham united",
    "wolves":"wolverhampton wanderers",
    "wolverhampton":"wolverhampton wanderers",
    "bournemouth":"afc bournemouth",
    "afc bournemouth":"afc bournemouth",
    "newcastle":"newcastle united",
    "forest":"nottingham forest",
    "nottingham forest":"nottingham forest",
    "villa":"aston villa",
    "aston villa":"aston villa",
    "liverpool":"liverpool",
    "chelsea":"chelsea",
    "arsenal":"arsenal",
    "everton":"everton",
    "fulham":"fulham",
    "brentford":"brentford",
    "crystal palace":"crystal palace",
    "palace":"crystal palace",
    "ipswich":"ipswich town",
    "leicester":"leicester city",
    "southampton":"southampton",
    "brighton":"brighton & hove albion",
    "brighton and hove albion":"brighton & hove albion",
}

def resolve_team_id(team_query: str, league_id: int = 39, season: Optional[int] = None) -> Optional[int]:
    """Robust resolver:
      1) use league team map (fast + reliable)
      2) try nickname->canonical, then map
      3) fallback to API /teams?search=
    """
    if not team_query:
        return None
    if season is None: season = current_season()
    name_norm = _norm(team_query)
    # 1) direct map
    m = list_league_teams_map(league_id, season)
    if name_norm in m:
        return m[name_norm]
    # 2) nickname -> canonical -> map
    canon = NICK_TO_CANON.get(name_norm)
    if not canon:
        # try contains match
        for k, v in NICK_TO_CANON.items():
            if k in name_norm:
                canon = v; break
    if canon:
        cn = _norm(canon)
        if cn in m:
            return m[cn]
    # 3) fallback search
    return search_team_id(team_query, league_id=league_id, season=season)
