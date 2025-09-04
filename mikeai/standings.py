# assistant/standings.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from . import api_football as af

EPL_ID = 39
LALIGA_ID = 140

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
    payload = _standings_payload(league_id=EPL_ID, season=2025, n_form=max(3, min(n, 10)))
    return JsonResponse({"season": 2025, "league": "EPL", "standings": payload})

@csrf_exempt
def standings_laliga(request):
    try:
        n = int(request.GET.get("n", "5"))
    except Exception:
        n = 5
    payload = _standings_payload(league_id=LALIGA_ID, season=2025, n_form=max(3, min(n, 10)))
    return JsonResponse({"season": 2025, "league": "La Liga", "standings": payload})
