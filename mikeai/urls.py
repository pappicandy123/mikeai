from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_page, name='chat'),
    path('team', views.team_panel, name='team_panel'),

    # Chat + profile
    path('ask/', views.AskAssistant.as_view(), name='ask'),
    path('me', views.me, name='me'),

    # REST data
    path('api/fixtures', views.api_fixtures, name='api_fixtures'),
    path('api/random-pick', views.api_random_pick, name='api_random_pick'),
    path('api/best-bets', views.api_best_bets, name='api_best_bets'),

    path('api/team/<str:name>/last5', views.api_team_last5, name='api_team_last5'),
    path('api/team/<str:name>/last10', views.api_team_last10, name='api_team_last10'),
    path('api/team/<str:name>/news', views.api_team_news, name='api_team_news'),
    path('api/team/<str:name>/summary', views.api_team_summary, name='api_team_summary'),

    # Standings for index.html
    # path('standings/epl', views.standings_epl, name='standings_epl'),
    # path('standings/laliga', views.standings_laliga, name='standings_laliga'),

    path('health', views.health, name='health'),
]