from django.urls import path
from .views import (
    chat_page, AskAssistant,
    api_fixtures, api_random_pick, api_best_bets,
    api_team_last5, api_team_last10, api_team_news
)

from . import views
from .standings import standings_epl, standings_laliga  # use richer standings module

urlpatterns = [
    path('', chat_page, name='chat'),

    # Chat
    path('ask/', AskAssistant.as_view(), name='ask'),

    # Tables
    path('standings/epl', standings_epl, name='standings_epl'),
    path('standings/laliga', standings_laliga, name='standings_laliga'),

    # Data APIs
    path('api/fixtures', api_fixtures, name='api_fixtures'),
    path('api/random-pick', api_random_pick, name='api_random_pick'),
    path('api/best-bets', api_best_bets, name='api_best_bets'),

    # NEW: Team REST endpoints
    path('api/team/<str:name>/last5', api_team_last5, name='api_team_last5'),
    path('api/team/<str:name>/last10', api_team_last10, name='api_team_last10'),
    path('api/team/<str:name>/news', api_team_news, name='api_team_news'),
    path('team', views.team_panel),  # /team?name=Manchester%20United

    path('api/team/<str:name>/summary', views.api_team_summary),
    path('api/team/<str:name>/last5', views.api_team_last5),
    path('api/team/<str:name>/last10', views.api_team_last10),
    path('api/team/<str:name>/news', views.api_team_news),

    path('ask/', views.AskAssistant.as_view()),
    path('standings/epl', standings_epl),
    path('standings/laliga', standings_laliga),
    path('api/fixtures', views.api_fixtures),
    path('api/random-pick', views.api_random_pick),
]
