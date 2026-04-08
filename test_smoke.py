"""Smoke tests for the GitHub Issue Triage environment."""

import json
import requests

BASE = "http://127.0.0.1:7860"


def main():
    # Test GET /
    r = requests.get(f"{BASE}/")
    assert r.status_code == 200
    print("GET /:", r.status_code, r.json()["name"])

    # Test GET /health
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    print("GET /health:", r.status_code, r.json())

    # ── task_easy: perfect action ───────────────────────────────────────
    r = requests.post(f"{BASE}/reset", json={"task_id": "task_easy"})
    assert r.status_code == 200
    obs = r.json()
    print("POST /reset task_easy:", r.status_code, "issue:", obs["current_issue"]["id"])

    action = {
        "labels": ["bug", "needs-reproduction"],
        "priority": "P1",
        "is_duplicate": False,
        "duplicate_of": None,
        "needs_info": False,
        "comment": None,
        "is_security": False,
        "close": False,
    }
    r = requests.post(f"{BASE}/step", json=action)
    assert r.status_code == 200
    result = r.json()
    print("POST /step (perfect):", r.status_code,
          "reward:", result["reward"]["total"],
          "done:", result["done"])
    assert result["reward"]["total"] == 1.0, f"Expected 1.0, got {result['reward']['total']}"
    assert result["done"] is True
    print("  => Perfect score confirmed!")

    # ── task_medium: step through all 5 ─────────────────────────────────
    r = requests.post(f"{BASE}/reset", json={"task_id": "task_medium"})
    assert r.status_code == 200
    obs = r.json()
    print("\nPOST /reset task_medium:", r.status_code,
          "issues_remaining:", obs["issues_remaining"])

    default_action = {
        "labels": ["bug"],
        "priority": "P2",
        "is_duplicate": False,
        "needs_info": False,
        "is_security": False,
        "close": False,
    }
    for i in range(5):
        r = requests.post(f"{BASE}/step", json=default_action)
        assert r.status_code == 200
        result = r.json()
        rw = result["reward"]["total"]
        done = result["done"]
        print(f"  step {i+1}: reward={rw}, done={done}")

    # ── task_hard: step through all 10 ──────────────────────────────────
    r = requests.post(f"{BASE}/reset", json={"task_id": "task_hard"})
    assert r.status_code == 200
    obs = r.json()
    print("\nPOST /reset task_hard:", r.status_code,
          "issues_remaining:", obs["issues_remaining"])

    for i in range(10):
        r = requests.post(f"{BASE}/step", json=default_action)
        assert r.status_code == 200
        result = r.json()
        rw = result["reward"]["total"]
        done = result["done"]
        sec = result["reward"]["security_score"]
        print(f"  step {i+1}: reward={rw}, security_score={sec}, done={done}")

    # ── Loop guard test ─────────────────────────────────────────────────
    print("\n--- Loop guard test ---")
    r = requests.post(f"{BASE}/reset", json={"task_id": "task_easy"})
    assert r.status_code == 200

    for i in range(4):
        r = requests.post(f"{BASE}/step", json=default_action)
        if r.status_code == 200:
            result = r.json()
            rw = result["reward"]["total"]
            done = result["done"]
            reason = result["reward"]["breakdown"].get("reason", "")
            print(f"  attempt {i+1}: reward={rw}, done={done}, reason={reason}")
        else:
            detail = r.json().get("detail", r.text)
            print(f"  attempt {i+1}: HTTP {r.status_code} - {detail}")

    # ── Error handling ──────────────────────────────────────────────────
    print("\n--- Error handling test ---")
    r = requests.post(f"{BASE}/reset", json={"task_id": "nonexistent"})
    assert r.status_code == 400
    print(f"  Invalid task_id: HTTP {r.status_code} (expected 400)")

    # GET /state
    r = requests.post(f"{BASE}/reset", json={"task_id": "task_easy"})
    r = requests.get(f"{BASE}/state")
    assert r.status_code == 200
    state = r.json()
    print(f"  GET /state: task_id={state['task_id']}, step={state['step']}")

    print("\n=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    main()
