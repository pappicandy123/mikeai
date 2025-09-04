from django.db import models

class Match(models.Model):
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    date = models.DateTimeField()
    league = models.CharField(max_length=100, blank=True, null=True)

class Odds(models.Model):
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="odds")
    market = models.CharField(max_length=50)  # e.g., "Over 2.5", "BTTS"
    value = models.DecimalField(max_digits=5, decimal_places=2)
    bookmaker = models.CharField(max_length=100)

class PredictionHistory(models.Model):
    question = models.TextField()
    match = models.ForeignKey(Match, on_delete=models.SET_NULL, null=True, blank=True)
    recommendation = models.TextField()
    confidence = models.IntegerField(null=True, blank=True)
    result = models.CharField(max_length=10, choices=[("won", "Won"), ("lost", "Lost")], blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

class UserContext(models.Model):
    session_key = models.CharField(max_length=40, unique=True, null=True, blank=True)  # Tie to Django session
    recent_leagues = models.TextField(blank=True)
    recent_bet_types = models.TextField(blank=True)
    history_summary = models.TextField(blank=True)
    chat_history = models.JSONField(default=list)  # [{"role": "user", "content": "..."}, ...]