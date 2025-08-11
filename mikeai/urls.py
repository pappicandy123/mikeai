from django.urls import path
from .views import AskAssistant, chat_page

urlpatterns = [
    path('', chat_page, name='chat'),
    path('ask/', AskAssistant.as_view(), name='ask'),
]

