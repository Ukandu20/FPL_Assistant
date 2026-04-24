from __future__ import annotations

import io
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import pandas as pd
import selenium
import soccerdata as sd
import undetected_chromedriver as uc
from lxml import etree, html
from selenium.common.exceptions import WebDriverException
from soccerdata._common import SeasonCode, add_alt_team_names, make_game_id, standardize_colnames
from soccerdata._config import DATA_DIR, NOCACHE, NOSTORE, TEAMNAME_REPLACEMENTS

FBREF_DATADIR = DATA_DIR / "FBref"
FBREF_HEADERS = {
    "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
}

BIG_FIVE_DICT = {
    "Serie A": "ITA-Serie A",
    "Ligue 1": "FRA-Ligue 1",
    "La Liga": "ESP-La Liga",
    "Premier League": "ENG-Premier League",
    "Fußball-Bundesliga": "GER-Bundesliga",
    "Bundesliga": "GER-Bundesliga",
}

DEFAULT_BROWSER_CANDIDATES = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
)

FBREF_API = "https://fbref.com"


def resolve_browser_path(browser_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    if browser_path:
        path = Path(browser_path)
        if not path.is_file():
            raise FileNotFoundError(f"Browser executable not found: {path}")
        return path
    for candidate in DEFAULT_BROWSER_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def _parse_table(html_table: html.HtmlElement) -> pd.DataFrame:
    for elem in html_table.xpath(".//span[contains(@class, 'f-i')]"):
        parent = elem.getparent()
        if parent is not None:
            etree.strip_elements(parent, "span", with_tail=False)
    for elem in html_table.xpath("//tbody/tr[contains(@class, 'spacer')]"):
        elem.getparent().remove(elem)
    for elem in html_table.xpath("//tbody/tr[contains(@class, 'thead')]"):
        elem.getparent().remove(elem)
    (df_table,) = pd.read_html(
        io.StringIO(html.tostring(html_table, encoding="unicode")),
        flavor="lxml",
    )
    return df_table.convert_dtypes()


def _concat(dfs: list[pd.DataFrame], key: list[str]) -> pd.DataFrame:
    all_columns = []
    dfs = sorted(dfs, key=lambda x: len(x.columns), reverse=True)

    for df in dfs:
        columns = pd.DataFrame(df.columns.tolist())
        columns.replace(to_replace=r"^Unnamed:.*", value=None, regex=True, inplace=True)
        if columns.shape[1] == 2:
            columns.replace(to_replace="", value=None, inplace=True)
            mask = pd.isnull(columns[1])
            columns.loc[mask, [0, 1]] = columns.loc[mask, [1, 0]].values
            all_columns.append(columns.copy())
            mask = pd.isnull(columns[0])
            columns.loc[mask, [0, 1]] = columns.loc[mask, [1, 0]].values
            columns.loc[mask, 1] = ""
            df.columns = pd.MultiIndex.from_tuples(columns.to_records(index=False).tolist())

    if len(all_columns) and all_columns[0].shape[1] == 2:
        columns = all_columns[0].copy()
        for other in all_columns[1:]:
            columns = columns.combine_first(other)

        mask = pd.isnull(columns[0])
        columns.loc[mask, [0, 1]] = columns.loc[mask, [1, 0]].values
        columns.loc[mask, 1] = ""
        column_idx = pd.MultiIndex.from_tuples(columns.to_records(index=False).tolist())

        for i, df in enumerate(dfs):
            if df.columns.equals(column_idx):
                continue
            if len(df.columns) == len(column_idx):
                df.columns = column_idx
            else:
                dfs[i] = df.reindex(columns=column_idx, fill_value=None)

    return pd.concat(dfs)


def _fix_nation_col(df_table: pd.DataFrame) -> pd.DataFrame:
    if "Nation" not in df_table.columns.get_level_values(1):
        df_table.loc[:, (slice(None), "Squad")] = (
            df_table.xs("Squad", axis=1, level=1)
            .squeeze()
            .apply(lambda x: x if isinstance(x, str) and x != "Squad" else None)
        )
        df_table.insert(
            2,
            ("Unnamed: nation", "Nation"),
            df_table.xs("Squad", axis=1, level=1).squeeze(),
        )
    return df_table


class PatchedFBref(sd.FBref):
    @classmethod
    def _all_leagues(cls) -> dict[str, str]:
        res = sd.FBref._all_leagues().copy()
        res.update({"Big 5 European Leagues Combined": "Big 5 European Leagues Combined"})
        return res

    def __init__(
        self,
        leagues: Optional[Union[str, list[str]]] = None,
        seasons: Optional[Union[str, int, list[Any]]] = None,
        proxy: Optional[Union[str, list[str], Callable[[], str]]] = None,
        no_cache: bool = NOCACHE,
        no_store: bool = NOSTORE,
        data_dir: Path = FBREF_DATADIR,
        browser_path: Optional[Union[str, Path]] = None,
        headless: bool = False,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.browser_path = resolve_browser_path(browser_path)
        self.headless = headless
        self.headers = {**FBREF_HEADERS, **(headers or {})}
        self._driver: Any = None
        super().__init__(
            leagues=leagues,
            seasons=seasons,
            proxy=proxy,
            no_cache=no_cache,
            no_store=no_store,
            data_dir=data_dir,
        )

    def close(self) -> None:
        driver = getattr(self, "_driver", None)
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
            self._driver = None

    def __del__(self) -> None:
        self.close()

    def _proxy_value(self) -> Optional[str]:
        proxy = getattr(self, "proxy", None)
        if proxy == dict:
            return None
        if callable(proxy):
            value = proxy()
            if isinstance(value, dict):
                return value.get("https") or value.get("http")
            return value
        return None

    def _init_driver(self) -> Any:
        options = uc.ChromeOptions()
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-first-run")
        options.add_argument("--lang=en-US")
        if self.headless:
            options.add_argument("--headless=new")
        if self.browser_path is not None:
            options.binary_location = str(self.browser_path)
        proxy_value = self._proxy_value()
        if proxy_value:
            if proxy_value == "tor":
                proxy_value = "socks5://127.0.0.1:9050"
            options.add_argument(f"--proxy-server={proxy_value}")
        driver = uc.Chrome(options=options, use_subprocess=True)
        self._install_network_headers(driver)
        return driver

    def _install_network_headers(self, driver: Any) -> None:
        try:
            driver.execute_cdp_cmd("Network.enable", {})
            user_agent = self.headers.get("User-Agent")
            accept_language = self.headers.get("Accept-Language", "en-US,en;q=0.9")
            if user_agent:
                driver.execute_cdp_cmd(
                    "Network.setUserAgentOverride",
                    {
                        "userAgent": user_agent,
                        "acceptLanguage": accept_language,
                        "platform": "Windows",
                    },
                )
            extra_headers = {k: v for k, v in self.headers.items() if k != "User-Agent"}
            if extra_headers:
                driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {"headers": extra_headers})
        except Exception:
            logging.getLogger("fbref").debug("Could not install CDP headers for browser session.")

    def _ensure_driver(self) -> Any:
        if self._driver is None:
            self._driver = self._init_driver()
        return self._driver

    def _validate_page(self, url: str) -> bytes:
        start = time.time()
        timeout = 20.0
        page_html = ""
        driver = self._ensure_driver()
        while time.time() - start < timeout:
            try:
                ready_state = driver.execute_script("return document.readyState")
            except Exception:
                ready_state = None
            page_html = driver.page_source or ""
            if any(token in page_html for token in ("Incapsula incident ID", "Access denied", "403 Forbidden")):
                raise ConnectionError(f"Blocked while loading {url}")
            if ready_state == "complete" and ("<table" in page_html or "Scores & Fixtures" in page_html):
                return page_html.encode("utf-8")
            time.sleep(0.5)
        raise TimeoutError(f"Timed out waiting for FBref page content at {url}")

    def _download_and_save(
        self,
        url: str,
        filepath: Optional[Path] = None,
        var: Optional[Union[str, Iterable[str]]] = None,
    ) -> io.BytesIO:
        for attempt in range(5):
            try:
                driver = self._ensure_driver()
                driver.get(url)
                time.sleep(self.rate_limit)
                if var is not None:
                    if not isinstance(var, str):
                        raise NotImplementedError("Only single JavaScript variables are supported.")
                    payload = io.BytesIO(str(driver.execute_script(f"return {var}")).encode("utf-8"))
                else:
                    payload = io.BytesIO(self._validate_page(url))
                if not self.no_store and filepath is not None:
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath.write_bytes(payload.getvalue())
                payload.seek(0)
                return payload
            except Exception as exc:
                logging.getLogger("fbref").exception(
                    "Error while scraping %s with browser transport. Retrying... (attempt %d of 5).",
                    url,
                    attempt + 1,
                )
                self.close()
                if attempt == 4:
                    raise ConnectionError(f"Could not download {url}.") from exc
                time.sleep(attempt + 1)
        raise ConnectionError(f"Could not download {url}.")


def build_fbref_reader(
    *,
    leagues: Optional[Union[str, list[str]]] = None,
    seasons: Optional[Union[str, int, list[Any]]] = None,
    proxy: Optional[Union[str, list[str], Callable[[], str]]] = None,
    no_cache: bool = NOCACHE,
    no_store: bool = NOSTORE,
    data_dir: Path = FBREF_DATADIR,
    browser_path: Optional[Union[str, Path]] = None,
    headless: bool = False,
    headers: Optional[dict[str, str]] = None,
) -> PatchedFBref:
    return PatchedFBref(
        leagues=leagues,
        seasons=seasons,
        proxy=proxy,
        no_cache=no_cache,
        no_store=no_store,
        data_dir=data_dir,
        browser_path=browser_path,
        headless=headless,
        headers=headers,
    )
