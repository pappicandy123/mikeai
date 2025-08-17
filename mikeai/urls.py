from django.urls import path
from .views import AskAssistant, chat_page, standings_epl, standings_laliga
from .standings import standings_epl, standings_laliga

urlpatterns = [
    path('', chat_page, name='chat'),
    path('ask/', AskAssistant.as_view(), name='ask'),

    # NEW — tables
    path('standings/epl', standings_epl, name='standings_epl'),
    path('standings/laliga', standings_laliga, name='standings_laliga'),
    # path("odds/match", odds_match),
    # path("players/compare", players_compare),
    # path("teams/h2h", teams_h2h),
]

