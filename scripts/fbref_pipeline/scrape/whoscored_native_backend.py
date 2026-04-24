from __future__ import annotations

import json
import logging
import re
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import HTTPCookieProcessor, Request, build_opener

import pandas as pd

try:  # pragma: no cover - optional runtime dependency
    from lxml import html as lxml_html
except ImportError:  # pragma: no cover - optional runtime dependency
    lxml_html = None  # type: ignore[assignment]

try:  # pragma: no cover - optional runtime dependency
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
except ImportError:  # pragma: no cover - optional runtime dependency
    webdriver = None  # type: ignore[assignment]
    ChromeOptions = None  # type: ignore[assignment]


LOG = logging.getLogger("whoscored.native")
WHOSCORED_URL = "https://www.whoscored.com"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class CompetitionConfig:
    key: str
    source_name: str
    region_id: Optional[int]
    tournament_id: Optional[int]
    competition_type: str
    season_mode: str


@dataclass(frozen=True)
class SeasonRecord:
    league: str
    season: str
    season_id: int
    region_id: int
    tournament_id: int
    season_label: str


@dataclass(frozen=True)
class StageRecord:
    stage_id: int
    stage: Optional[str]


def load_native_competitions(path: Path) -> Dict[str, CompetitionConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, CompetitionConfig] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        out[str(key)] = CompetitionConfig(
            key=str(key),
            source_name=str(value.get("source_name") or value.get("WhoScored") or key),
            region_id=_coerce_int(value.get("region_id")),
            tournament_id=_coerce_int(value.get("tournament_id") or value.get("league_id")),
            competition_type=str(value.get("competition_type") or "club"),
            season_mode=str(value.get("season_mode") or "split-year"),
        )
    return out


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_url_ids(url: str) -> Dict[str, Optional[int]]:
    return {
        "region_id": _coerce_int(_first_group(url, r"regions/(\d+)")),
        "tournament_id": _coerce_int(_first_group(url, r"tournaments/(\d+)")),
        "season_id": _coerce_int(_first_group(url, r"seasons/(\d+)")),
        "stage_id": _coerce_int(_first_group(url, r"stages/(\d+)")),
        "match_id": _coerce_int(_first_group(url, r"matches/(\d+)")),
    }


def _first_group(text: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None


def _to_season_int(label: str) -> int:
    label = str(label).strip()
    for pattern in (r"(\d{4})[/-](\d{4})", r"(\d{4})[/-](\d{2})", r"(\d{2})[/-](\d{2})"):
        match = re.fullmatch(pattern, label)
        if match:
            start = match.group(1)
            return int(start) if len(start) == 4 else 2000 + int(start)
    match = re.fullmatch(r"(\d{4})", label)
    if match:
        return int(match.group(1))
    raise ValueError(f"Unrecognized season label: {label!r}")


def _balanced_fragment(text: str, start: int) -> Optional[str]:
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    quote = ""
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue
        if ch in ('"', "'"):
            in_string = True
            quote = ch
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def extract_json_after_marker(text: str, marker: str) -> Optional[Any]:
    idx = text.find(marker)
    if idx < 0:
        return None
    brace_positions = [pos for pos in (text.find("{", idx), text.find("[", idx)) if pos >= 0]
    if not brace_positions:
        return None
    start = min(brace_positions)
    fragment = _balanced_fragment(text, start)
    if not fragment:
        return None
    try:
        return json.loads(fragment)
    except Exception:
        return None


def parse_tiers_json(data: Any) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    if not isinstance(data, list):
        return out
    for region in data:
        if not isinstance(region, Mapping):
            continue
        region_id = _coerce_int(region.get("id"))
        region_name = str(region.get("name") or "")
        for tournament in region.get("tournaments", []):
            if not isinstance(tournament, Mapping):
                continue
            tournament_id = _coerce_int(tournament.get("id"))
            if region_id is None or tournament_id is None:
                continue
            source_name = f"{region_name} - {tournament.get('name')}"
            out[source_name] = {"region_id": region_id, "tournament_id": tournament_id}
    return out


def parse_season_options(
    html_text: str,
    *,
    competition: CompetitionConfig,
) -> List[SeasonRecord]:
    if lxml_html is None:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("lxml is required for native WhoScored parsing.")
    root = lxml_html.fromstring(html_text)
    records: List[SeasonRecord] = []
    for node in root.xpath("//select[contains(@id,'seasons')]/option"):
        season_url = node.get("value") or ""
        ids = _parse_url_ids(season_url)
        season_id = ids.get("season_id")
        if season_id is None:
            continue
        label = " ".join(node.itertext()).strip()
        try:
            season = str(_to_season_int(label))
        except ValueError:
            continue
        region_id = ids.get("region_id") or competition.region_id
        tournament_id = ids.get("tournament_id") or competition.tournament_id
        if region_id is None or tournament_id is None:
            continue
        records.append(
            SeasonRecord(
                league=competition.key,
                season=season,
                season_id=season_id,
                region_id=region_id,
                tournament_id=tournament_id,
                season_label=label,
            )
        )
    return records


def parse_stage_options(html_text: str) -> List[StageRecord]:
    if lxml_html is None:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("lxml is required for native WhoScored parsing.")
    root = lxml_html.fromstring(html_text)
    seen: set[int] = set()
    out: List[StageRecord] = []

    fixture_links = root.xpath("//a[normalize-space(text())='Fixtures']/@href")
    if fixture_links:
        stage_id = _parse_url_ids(fixture_links[0]).get("stage_id")
        if stage_id is not None:
            out.append(StageRecord(stage_id=stage_id, stage=None))
            seen.add(stage_id)

    for node in root.xpath("//select[contains(@id,'stages')]/option"):
        ids = _parse_url_ids(node.get("value") or "")
        stage_id = ids.get("stage_id")
        if stage_id is None or stage_id in seen:
            continue
        out.append(StageRecord(stage_id=stage_id, stage=" ".join(node.itertext()).strip() or None))
        seen.add(stage_id)
    return out


def parse_calendar_mask(html_text: str) -> Dict[str, List[str]]:
    payload = extract_json_after_marker(html_text, "wsCalendar")
    if isinstance(payload, Mapping) and isinstance(payload.get("mask"), Mapping):
        return {
            str(year): [str(month) for month in months]
            for year, months in payload["mask"].items()
            if isinstance(months, list)
        }
    mask_match = re.search(r"mask\s*:\s*\{", html_text)
    if not mask_match:
        return {}
    start = html_text.find("{", mask_match.start())
    if start < 0:
        return {}
    fragment = _balanced_fragment(html_text, start)
    if not fragment:
        return {}
    try:
        normalized = re.sub(r'([{\[,])\s*(\d+)\s*:', r'\1"\2":', fragment)
        decoded = json.loads(normalized)
    except Exception:
        return {}
    if not isinstance(decoded, Mapping):
        return {}
    out: Dict[str, List[str]] = {}
    for year, months in decoded.items():
        if not isinstance(months, Mapping):
            continue
        out[str(year)] = [str(month) for month, days in months.items() if isinstance(days, Mapping)]
    return out


def make_game_label(date_value: Any, home_team: Any, away_team: Any) -> str:
    date_text = str(date_value or "")[:10]
    return f"{date_text} {home_team}-{away_team}".strip()


def parse_schedule_month_payload(
    payload: Mapping[str, Any],
    *,
    league: str,
    season: str,
    stage_id: int,
    stage_name: Optional[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for tournament in payload.get("tournaments", []):
        if not isinstance(tournament, Mapping):
            continue
        for match in tournament.get("matches", []):
            if not isinstance(match, Mapping):
                continue
            date_value = match.get("startTimeUtc") or match.get("startDate") or match.get("date")
            home_team = match.get("homeTeamName") or match.get("home") or match.get("homeTeam")
            away_team = match.get("awayTeamName") or match.get("away") or match.get("awayTeam")
            home_score = match.get("homeScore")
            away_score = match.get("awayScore")
            rows.append(
                {
                    "league": league,
                    "season": season,
                    "stage_id": stage_id,
                    "stage": stage_name,
                    "game_id": _coerce_int(match.get("id") or match.get("matchId")),
                    "status": match.get("status"),
                    "start_time": date_value,
                    "date": str(date_value)[:10] if date_value is not None else None,
                    "home_team_id": _coerce_int(match.get("homeTeamId") or match.get("homeId")),
                    "home_team": home_team,
                    "away_team_id": _coerce_int(match.get("awayTeamId") or match.get("awayId")),
                    "away_team": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                    "score": match.get("score")
                    or (
                        f"{home_score}-{away_score}"
                        if home_score is not None and away_score is not None
                        else None
                    ),
                    "game": make_game_label(date_value, home_team, away_team),
                    "incidents": match.get("incidents"),
                }
            )
    return pd.DataFrame(rows)


def parse_embedded_tournament_fixtures(
    html_text: str,
    *,
    league: str,
    season: str,
    stage_id: int,
    stage_name: Optional[str],
) -> pd.DataFrame:
    match = re.search(
        r'<script type="application/json" data-hypernova-key="tournamentfixtures"[^>]*><!--(.*?)--></script>',
        html_text,
        re.S,
    )
    if not match:
        return pd.DataFrame()
    try:
        payload = json.loads(match.group(1))
    except Exception:
        return pd.DataFrame()
    tournaments = payload.get("tournaments")
    if not isinstance(tournaments, list):
        return pd.DataFrame()
    wrapped = {"tournaments": tournaments}
    return parse_schedule_month_payload(
        wrapped,
        league=league,
        season=season,
        stage_id=stage_id,
        stage_name=stage_name,
    )


def parse_missing_players_html(
    html_text: str,
    *,
    schedule_row: Mapping[str, Any],
    league: str,
    season: str,
) -> pd.DataFrame:
    if lxml_html is None:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("lxml is required for native WhoScored parsing.")
    root = lxml_html.fromstring(html_text)
    rows: List[Dict[str, Any]] = []
    sections = [
        ("home_team", "//div[@id='missing-players']/div[2]/table/tbody/tr"),
        ("away_team", "//div[@id='missing-players']/div[3]/table/tbody/tr"),
    ]
    for team_key, xpath in sections:
        for node in root.xpath(xpath):
            anchors = node.xpath("./td[contains(@class,'pn')]/a")
            if not anchors:
                continue
            href = anchors[0].get("href") or ""
            player_id = _parse_url_ids(href).get("match_id")
            if player_id is None:
                parts = href.strip("/").split("/")
                if len(parts) >= 2:
                    player_id = _coerce_int(parts[1])
            reason = None
            reason_nodes = node.xpath("./td[contains(@class,'reason')]/span")
            if reason_nodes:
                reason = reason_nodes[0].get("title") or " ".join(reason_nodes[0].itertext()).strip()
            status_nodes = node.xpath("./td[contains(@class,'confirmed')]")
            rows.append(
                {
                    "league": league,
                    "season": season,
                    "game": schedule_row.get("game"),
                    "team": schedule_row.get(team_key),
                    "player": " ".join(anchors[0].itertext()).strip(),
                    "game_id": _coerce_int(schedule_row.get("game_id")),
                    "player_id": player_id,
                    "reason": reason,
                    "status": " ".join(status_nodes[0].itertext()).strip() if status_nodes else None,
                }
            )
    return pd.DataFrame(rows)


def extract_match_centre_payload(html_text: str) -> Optional[Dict[str, Any]]:
    for marker in (
        "matchCentreData",
        "require.config.params['args'].matchCentreData",
        'require.config.params["args"].matchCentreData',
    ):
        payload = extract_json_after_marker(html_text, marker)
        if isinstance(payload, Mapping):
            return dict(payload)
    args_payload = extract_json_after_marker(html_text, "require.config.params['args']")
    if isinstance(args_payload, Mapping) and isinstance(args_payload.get("matchCentreData"), Mapping):
        return dict(args_payload["matchCentreData"])
    return None


def extract_report_json_blobs(html_text: str) -> Dict[str, Any]:
    blobs: Dict[str, Any] = {}
    for marker in (
        "matchCentreData",
        "matchStatistics",
        "playerStatistics",
        "statistics",
        "stats",
        "matchFacts",
    ):
        payload = extract_json_after_marker(html_text, marker)
        if payload is not None:
            blobs[marker] = payload
    return blobs


class NativeWhoScoredBackend:
    def __init__(
        self,
        *,
        competitions: Mapping[str, CompetitionConfig],
        cache_dir: Path,
        no_cache: bool = False,
        no_store: bool = False,
        browser_fallback: bool = True,
        path_to_browser: Optional[str] = None,
        headless: bool = True,
    ) -> None:
        self.competitions = dict(competitions)
        self.cache_dir = cache_dir
        self.no_cache = no_cache
        self.no_store = no_store
        self.browser_fallback = browser_fallback
        self.path_to_browser = path_to_browser
        self.headless = headless
        self._opener = build_opener(HTTPCookieProcessor())

    def resolve_seasons(self, league: str, explicit_seasons: Optional[Sequence[str]]) -> pd.DataFrame:
        competition = self.competitions[league]
        region_id, tournament_id = self._resolve_competition_ids(competition)
        url = f"{WHOSCORED_URL}/Regions/{region_id}/Tournaments/{tournament_id}"
        html_text = self._fetch_text(
            url,
            self.cache_dir / "pages" / league / "tournament.html",
        )
        records = parse_season_options(html_text, competition=competition)
        if explicit_seasons:
            wanted = {str(_to_season_int(value)) for value in explicit_seasons}
            records = [record for record in records if record.season in wanted]
        if not records:
            raise RuntimeError(f"Could not resolve seasons for {league}")
        df = pd.DataFrame([record.__dict__ for record in records]).set_index(["league", "season"])
        return df.sort_index()

    def read_schedule(
        self,
        *,
        league: str,
        season_record: Mapping[str, Any],
        match_ids: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        season_id = int(season_record["season_id"])
        region_id = int(season_record["region_id"])
        tournament_id = int(season_record["tournament_id"])
        season = str(season_record["season"])

        season_url = (
            f"{WHOSCORED_URL}/Regions/{region_id}/Tournaments/{tournament_id}/Seasons/{season_id}"
        )
        season_html = self._fetch_text(
            season_url,
            self.cache_dir / "pages" / league / season / "season.html",
        )
        stages = parse_stage_options(season_html) or [StageRecord(stage_id=tournament_id, stage=None)]

        parts: List[pd.DataFrame] = []
        for stage_record in stages:
            stage_url = (
                f"{WHOSCORED_URL}/Regions/{region_id}/Tournaments/{tournament_id}"
                f"/Seasons/{season_id}/Stages/{stage_record.stage_id}"
            )
            stage_html = self._fetch_text(
                stage_url,
                self.cache_dir / "pages" / league / season / f"stage_{stage_record.stage_id}.html",
            )
            mask = parse_calendar_mask(stage_html)
            for year, months in mask.items():
                for month in months:
                    url = f"{WHOSCORED_URL}/tournaments/{stage_record.stage_id}/data/?d={year}{int(month)+1:02d}"
                    try:
                        payload = self._fetch_json(
                            url,
                            self.cache_dir
                            / "json"
                            / league
                            / season
                            / f"schedule_{stage_record.stage_id}_{year}_{month}.json",
                        )
                    except Exception as exc:
                        LOG.debug(
                            "Monthly schedule endpoint failed for %s %s stage=%s year=%s month=%s: %s",
                            league,
                            season,
                            stage_record.stage_id,
                            year,
                            month,
                            exc,
                        )
                        continue
                    if isinstance(payload, Mapping):
                        parts.append(
                            parse_schedule_month_payload(
                                payload,
                                league=league,
                                season=season,
                                stage_id=stage_record.stage_id,
                                stage_name=stage_record.stage,
                            )
                        )
            if not parts:
                fallback_df = parse_embedded_tournament_fixtures(
                    stage_html,
                    league=league,
                    season=season,
                    stage_id=stage_record.stage_id,
                    stage_name=stage_record.stage,
                )
                if not fallback_df.empty:
                    parts.append(fallback_df)
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        if match_ids:
            wanted = {int(match_id) for match_id in match_ids}
            df = df[df["game_id"].isin(wanted)]
        return df.drop_duplicates(subset=["game_id"]).reset_index(drop=True)

    def read_missing_players(
        self,
        *,
        league: str,
        season: str,
        schedule_df: pd.DataFrame,
        raw_match_root: Optional[Path] = None,
        match_limit: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[int, str]]:
        rows: List[pd.DataFrame] = []
        artifacts: Dict[int, str] = {}
        iterator = schedule_df.head(match_limit) if match_limit else schedule_df
        for _, schedule_row in iterator.iterrows():
            game_id = _coerce_int(schedule_row.get("game_id"))
            if game_id is None:
                continue
            html_text = self._fetch_text(
                f"{WHOSCORED_URL}/Matches/{game_id}/Preview",
                self.cache_dir / "pages" / league / str(season) / "preview" / f"{game_id}.html",
            )
            rows.append(
                parse_missing_players_html(
                    html_text,
                    schedule_row=schedule_row.to_dict(),
                    league=league,
                    season=str(season),
                )
            )
            artifacts[game_id] = html_text
            if raw_match_root is not None:
                match_dir = raw_match_root / str(game_id)
                match_dir.mkdir(parents=True, exist_ok=True)
                (match_dir / "preview.html").write_text(html_text, encoding="utf-8")
        return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), artifacts

    def read_match_payloads(
        self,
        *,
        league: str,
        season: str,
        schedule_df: pd.DataFrame,
        raw_match_root: Optional[Path] = None,
        match_limit: Optional[int] = None,
        retry_failed_matches: bool = False,
        stats_mode: str = "all-visible",
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[int, str], List[int]]:
        payloads: Dict[int, Dict[str, Any]] = {}
        report_blobs: Dict[int, Dict[str, Any]] = {}
        report_html: Dict[int, str] = {}
        failures: List[int] = []

        iterator = schedule_df.head(match_limit) if match_limit else schedule_df
        match_rows = [row.to_dict() for _, row in iterator.iterrows()]
        for attempt in range(2 if retry_failed_matches else 1):
            pending = match_rows if attempt == 0 else [
                row for row in match_rows if _coerce_int(row.get("game_id")) in failures
            ]
            failures = []
            for schedule_row in pending:
                game_id = _coerce_int(schedule_row.get("game_id"))
                if game_id is None:
                    continue
                try:
                    html_text = self._fetch_text(
                        f"{WHOSCORED_URL}/Matches/{game_id}/Live",
                        self.cache_dir / "pages" / league / str(season) / "live" / f"{game_id}.html",
                    )
                    payload = extract_match_centre_payload(html_text)
                    if not isinstance(payload, Mapping):
                        raise RuntimeError("matchCentreData not found")
                    payloads[game_id] = dict(payload)
                    if raw_match_root is not None:
                        match_dir = raw_match_root / str(game_id)
                        match_dir.mkdir(parents=True, exist_ok=True)
                        (match_dir / "match_center.json").write_text(
                            json.dumps(payload, ensure_ascii=True, indent=2),
                            encoding="utf-8",
                        )
                    if stats_mode != "none":
                        report_text = self._fetch_report_html(game_id, league, str(season))
                        if report_text:
                            report_html[game_id] = report_text
                            report_blobs[game_id] = extract_report_json_blobs(report_text)
                            if raw_match_root is not None:
                                match_dir = raw_match_root / str(game_id)
                                match_dir.mkdir(parents=True, exist_ok=True)
                                (match_dir / "report.html").write_text(
                                    report_text,
                                    encoding="utf-8",
                                )
                except Exception as exc:
                    LOG.debug("Failed to fetch match payload for %s: %s", game_id, exc)
                    failures.append(game_id)
        return payloads, report_blobs, report_html, failures

    def _resolve_competition_ids(self, competition: CompetitionConfig) -> Tuple[int, int]:
        if competition.region_id is not None and competition.tournament_id is not None:
            return competition.region_id, competition.tournament_id
        tiers = self._tiers_map()
        ids = tiers.get(competition.source_name)
        if not ids:
            raise RuntimeError(
                f"Competition {competition.key} ({competition.source_name}) not found in WhoScored tiers."
            )
        return ids["region_id"], ids["tournament_id"]

    def _tiers_map(self) -> Dict[str, Dict[str, int]]:
        data = self._fetch_json(
            WHOSCORED_URL,
            self.cache_dir / "json" / "tiers.json",
            marker="allRegions",
        )
        return parse_tiers_json(data)

    def _fetch_report_html(self, game_id: int, league: str, season: str) -> Optional[str]:
        candidates = [
            f"{WHOSCORED_URL}/Matches/{game_id}",
            f"{WHOSCORED_URL}/Matches/{game_id}/MatchReport",
            f"{WHOSCORED_URL}/Matches/{game_id}/LiveStatistics",
        ]
        for idx, url in enumerate(candidates):
            try:
                return self._fetch_text(
                    url,
                    self.cache_dir / "pages" / league / season / "report" / f"{game_id}_{idx}.html",
                )
            except Exception:
                continue
        return None

    def _fetch_json(self, url: str, cache_path: Path, marker: Optional[str] = None) -> Any:
        text = self._fetch_text(url, cache_path)
        if marker:
            payload = extract_json_after_marker(text, marker)
            if payload is None:
                raise RuntimeError(f"Could not extract {marker} from {url}")
            return payload
        return json.loads(text)

    def _fetch_text(self, url: str, cache_path: Path) -> str:
        if not self.no_cache and cache_path.is_file():
            return cache_path.read_text(encoding="utf-8")

        try:
            req = Request(url, headers=DEFAULT_HEADERS)
            with self._opener.open(req, context=ssl.create_default_context(), timeout=30) as response:
                data = response.read().decode("utf-8", errors="replace")
        except TypeError:
            try:
                req = Request(url, headers=DEFAULT_HEADERS)
                with self._opener.open(req, timeout=30) as response:
                    data = response.read().decode("utf-8", errors="replace")
            except (HTTPError, URLError) as exc:
                if self.browser_fallback:
                    data = self._fetch_text_with_browser(url)
                else:
                    raise RuntimeError(f"Request failed for {url}: {exc}") from exc
        except (HTTPError, URLError) as exc:
            if self.browser_fallback:
                data = self._fetch_text_with_browser(url)
            else:
                raise RuntimeError(f"Request failed for {url}: {exc}") from exc
        if not self.no_store:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(data, encoding="utf-8")
        return data

    def _fetch_text_with_browser(self, url: str) -> str:
        if webdriver is None or ChromeOptions is None:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("Browser fallback requested but selenium is not installed.")
        options = ChromeOptions()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1600,1200")
        if self.path_to_browser:
            options.binary_location = self.path_to_browser
        driver = webdriver.Chrome(options=options)
        try:
            driver.get(url)
            return driver.page_source
        finally:
            driver.quit()
