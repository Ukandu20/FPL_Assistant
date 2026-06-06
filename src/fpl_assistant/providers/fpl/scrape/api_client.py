import json
import time
from typing import Any

import requests


FPL_API_BASE = "https://fantasy.premierleague.com/api"
DEFAULT_TIMEOUT = 20
DEFAULT_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 2
DEFAULT_HEADERS = {
    "User-Agent": "FPL-Assistant/1.0 (+https://fantasy.premierleague.com/)",
    "Accept": "application/json",
}


class FPLApiError(RuntimeError):
    """Raised when the FPL API cannot be reached or returns an invalid response."""


def _request_json(url: str, *, attempts: int = DEFAULT_ATTEMPTS, timeout: int = DEFAULT_TIMEOUT) -> Any:
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(RETRY_SLEEP_SECONDS)
        except ValueError as exc:
            raise FPLApiError(f"FPL API returned invalid JSON from {url}") from exc

    raise FPLApiError(
        f"Failed to fetch {url} after {attempts} attempts. "
        "Check internet access and DNS resolution for fantasy.premierleague.com."
    ) from last_error


def get_data():
    """Retrieve the FPL bootstrap data."""
    return _request_json(f"{FPL_API_BASE}/bootstrap-static/")


def get_individual_player_data(player_id):
    """Retrieve player-specific detailed data."""
    return _request_json(f"{FPL_API_BASE}/element-summary/{player_id}/")


def get_entry_data(entry_id):
    """Retrieve the summary/history data for a specific entry/team."""
    return _request_json(f"{FPL_API_BASE}/entry/{entry_id}/history/")


def get_entry_personal_data(entry_id):
    """Retrieve the personal data for a specific entry/team."""
    return _request_json(f"{FPL_API_BASE}/entry/{entry_id}/")


def get_entry_gws_data(entry_id, num_gws, start_gw=1):
    """Retrieve GW-by-GW data for a specific entry/team."""
    gw_data = []
    for i in range(start_gw, num_gws + 1):
        gw_data.append(_request_json(f"{FPL_API_BASE}/entry/{entry_id}/event/{i}/picks/"))
    return gw_data


def get_entry_transfers_data(entry_id):
    """Retrieve transfer data for a specific entry/team."""
    return _request_json(f"{FPL_API_BASE}/entry/{entry_id}/transfers/")


def get_fixtures_data():
    """Retrieve fixtures data for the season."""
    return _request_json(f"{FPL_API_BASE}/fixtures/")


def main():
    data = get_data()
    with open("raw.json", "w") as outf:
        json.dump(data, outf)


if __name__ == "__main__":
    main()
