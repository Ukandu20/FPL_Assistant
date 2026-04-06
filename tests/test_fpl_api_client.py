from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fpl_pipeline.scrape import api_client


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_request_json_retries_then_succeeds(monkeypatch):
    calls = {"count": 0}

    def fake_get(url, headers, timeout):
        calls["count"] += 1
        if calls["count"] < 3:
            raise requests.exceptions.ConnectionError("temporary failure")
        assert url == f"{api_client.FPL_API_BASE}/bootstrap-static/"
        assert timeout == api_client.DEFAULT_TIMEOUT
        assert headers == api_client.DEFAULT_HEADERS
        return _DummyResponse({"events": []})

    monkeypatch.setattr(api_client.requests, "get", fake_get)
    monkeypatch.setattr(api_client.time, "sleep", lambda _: None)

    data = api_client.get_data()

    assert data == {"events": []}
    assert calls["count"] == 3


def test_request_json_raises_clean_error_after_retry_exhaustion(monkeypatch):
    def fake_get(url, headers, timeout):
        raise requests.exceptions.ConnectionError("dns failure")

    monkeypatch.setattr(api_client.requests, "get", fake_get)
    monkeypatch.setattr(api_client.time, "sleep", lambda _: None)

    try:
        api_client.get_data()
    except api_client.FPLApiError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected FPLApiError to be raised")

    assert "bootstrap-static" in message
    assert "DNS resolution" in message
