import os
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import json

def ask_grok(messages, enable_web_search=True, enable_x_search=True):
    """
    Call Grok API with messages. Use search_parameters for live search (web/X).
    Disable searches if retries fail.
    """
    try:
        session = requests.Session()
        retries = Retry(total=1, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504, 408])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        payload = {
            "model": os.getenv("GROK_MODEL", "grok-4"),
            "messages": messages,
            "search_parameters": {"mode": "on"} if (enable_web_search or enable_x_search) else {"mode": "off"},
        }
        response = session.post(
            os.getenv("GROK_API_URL", "https://api.x.ai/v1/chat/completions"),
            headers={"Authorization": f"Bearer {os.getenv('GROK_API_KEY')}"},
            json=payload,
            timeout=90
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        print("[ERROR] Grok API timeout - disabling searches and failing over")
        if enable_web_search or enable_x_search:
            return ask_grok(messages, enable_web_search=False, enable_x_search=False)
        print("[ERROR] Grok API timeout - max retries exceeded")
        return "Timeout on that one, mate—Grok's taking a breather. Try again? 😅"
    except Exception as e:
        print(f"[ERROR] Grok API failed: {str(e)}")
        return "Lemme try that again, mate—something’s off! 😅 What’s your question?"