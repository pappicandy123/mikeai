# mikeai/sportmonks.py — tiny wrappers; implement with your real SportMonks v3 calls.
import os, requests
BASE = "https://api.sportmonks.com/v3/football"
KEY = os.getenv("SPORTMONKS_API_KEY")

def _get(path, **params):
    params = {k:v for k,v in params.items() if v is not None}
    params["api_token"] = KEY
    r = requests.get(f"{BASE}/{path.lstrip('/')}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_fixtures_by_range(league_id: int, start_date: str, end_date: str):
    # Example: filter by date & league; adjust to your plan/filters
    j = _get("fixtures", filters=f"between:{start_date},{end_date}|league_id:{league_id}", include="participants")
    out = []
    for row in (j.get("data") or []):
        # Normalize into AF-like shape: fixture.date, teams.home/away.name
        dt = row.get("starting_at") or row.get("starting_at_timestamp")
        parts = row.get("participants") or []
        home = next((p for p in parts if p.get("meta",{}).get("location")=="home"), {})
        away = next((p for p in parts if p.get("meta",{}).get("location")=="away"), {})
        out.append({
            "fixture": {"date": str(dt)},
            "teams": {"home": {"name": home.get("name")}, "away": {"name": away.get("name")}},
        })
    return out

def search_team_id(name: str):
    j = _get("teams/search/"+name)
    # Return first match + league if available
    d0 = (j.get("data") or [None])[0]
    if not d0: return None, None
    return d0.get("id"), d0.get("league_id")

def get_team_stats(team_id: int, league_id: int):
    # You can compute a simple summary from last N matches, or use provided endpoints if available.
    # For now return AF-like minimal dict with 'form' and goals averages set to safe defaults.
    return {"form": "WWDLW", "goals": {"for": {"average": {"home": 1.5, "away": 1.3}},
                                       "against": {"average": {"home": 1.0, "away": 1.1}}}}
