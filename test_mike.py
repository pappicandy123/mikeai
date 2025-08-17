import os
import requests

BASE_URL = os.getenv("MIKE_BASE_URL", "http://127.0.0.1:8000")

tests = [
    {"desc": "Get EPL weekend fixtures", "q": "Show EPL fixtures this weekend", "expect": "fixture"},
    {"desc": "Match prediction", "q": "Give me your prediction of the match Chelsea vs Liverpool and their odds", "expect": "odds"},
    {"desc": "Follow-up: corners", "q": "what about corners for the match", "expect": "corner"},
    {"desc": "Low-risk small odds", "q": "Give me best bets this weekend", "expect": "odds"},
    {"desc": "Follow-up low risk", "q": "Give me another random betting pick", "expect": "odds"},
    {"desc": "News test", "q": "Give me latest EPL news", "expect": "news"},
    {"desc": "Team corners stat", "q": "How many corners does Arsenal average?", "expect": "corner"},
    {"desc": "Match cards stat", "q": "How many yellow cards in Chelsea vs Man City last season?", "expect": "card"},
    {"desc": "General Q&A", "q": "Who won the last EPL season?", "expect": "winner"},
    {"desc": "La Liga prediction", "q": "Predict Barcelona vs Real Madrid and show odds", "expect": "odds"}
]

def run_test(question, expect):
    try:
        resp = requests.post(f"{BASE_URL}/ask/", json={"question": question}, timeout=30)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        text = resp.text.lower()
        if expect.lower() in text:
            return True, text[:120] + "..."
        return False, text[:120] + "..."
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    print(f"Testing Mike at {BASE_URL}\n")
    passed = 0
    for i, t in enumerate(tests, 1):
        ok, snippet = run_test(t["q"], t["expect"])
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"{i}. {t['desc']}: {status} → {snippet}")
    print(f"\nFinal: {passed}/{len(tests)} passed")
