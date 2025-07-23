#!/usr/bin/env python3
import os, json, time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select as WebSelect
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ─── Helper: activate tab & read grid ─────────────────────────────────────────
def read_grid(driver, wait, tab_text):
    link = driver.find_element(
        By.XPATH,
        f'//a[contains(@class,"nav-link") and normalize-space()="{tab_text}"]'
    )
    href = link.get_attribute("href")
    pane_id = "#" + href.split("#")[-1]

    if "active" not in link.get_attribute("class"):
        link.click()

    wait.until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                f'//a[contains(@class,"nav-link active") '
                f'and @href="{href}" and @aria-selected="true"]',
            )
        )
    )
    wait.until(
        EC.presence_of_element_located(
            (
                By.CSS_SELECTOR,
                f"{pane_id}.tab-pane.active "
                f".ui-grid-render-container-body .ui-grid-row",
            )
        )
    )

    headers = [
        h.text.strip()
        for h in driver.find_elements(
            By.CSS_SELECTOR,
            f"{pane_id} .ui-grid-header-cell .ui-grid-cell-contents",
        )
    ]

    data = []
    for r in driver.find_elements(
        By.CSS_SELECTOR,
        f"{pane_id} .ui-grid-render-container-body .ui-grid-row",
    ):
        cells = [
            c.text.strip()
            for c in r.find_elements(
                By.CSS_SELECTOR, ".ui-grid-cell .ui-grid-cell-contents"
            )
        ]
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        data.append(cells)

    return pd.DataFrame(data, columns=headers)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    BASE_URL = "https://www.fantasynutmeg.com/history"
    OUT_DIR = "data/raw/fantasy"
    MANIFEST = os.path.join(OUT_DIR, "manifest.json")
    os.makedirs(OUT_DIR, exist_ok=True)

    done = (
        set(json.load(open(MANIFEST))["seasons"])
        if os.path.exists(MANIFEST)
        else set()
    )

    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=opts
    )
    wait = WebDriverWait(driver, 20)

    driver.get(BASE_URL)
    wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#seasonSelect option"))
    )
    select = WebSelect(driver.find_element(By.ID, "seasonSelect"))

    # Build clean [(label, value)] list, skipping blanks
    season_opts = [
        (opt.text.strip(), opt.get_attribute("value").strip())
        for opt in select.options
        if opt.text.strip()
    ]

    for label, value in season_opts:
        if label in done:
            print(f"✔️  {label} done; skip")
            continue

        print(f"▶️  Scraping {label}")
        select.select_by_value(value)  # ← FIX: only one select call
        driver.find_element(
            By.CSS_SELECTOR, 'button[ng-click="loadHistory()"]'
        ).click()

        # ---- Points & xG tables --------------------------------------------
        pts_df = read_grid(driver, wait, "Points Table")
        pts_df["Season"] = label       # ← FIX: use scalar label, not list

        xg_df = read_grid(driver, wait, "Team xG Data")
        xg_df["Season"] = label        # ← FIX

        # ---- Player links --------------------------------------------------
        driver.find_element(
            By.XPATH, '//a[normalize-space()="Points Table"]'
        ).click()
        wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    '//a[normalize-space()="Points Table" '
                    'and contains(@class,"active")]',
                )
            )
        )

        players = []
        for row in driver.find_elements(
            By.CSS_SELECTOR, ".tab-pane.active .ui-grid-row"
        ):
            try:
                a = row.find_element(
                    By.CSS_SELECTOR, ".ui-grid-cell-contents a"
                )
                players.append((a.text.strip(), a.get_attribute("href")))
            except:
                continue

        # ---- Per-GW breakdown ---------------------------------------------
        gw_frames = []
        for name, url in players:
            try:
                driver.get(url)
                wait.until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//table//th[contains(text(),'GW')]")
                    )
                )
                gw = pd.read_html(driver.page_source, match="GW")[0]
                gw["Player"], gw["Season"] = name, label   # ← FIX
                gw_frames.append(gw)
            except Exception as e:
                print(f"   ⚠️  {name}: {e}")
            time.sleep(0.25)

        gw_df = (
            pd.concat(gw_frames, ignore_index=True)
            if gw_frames
            else pd.DataFrame()
        )

        # ---- Save ----------------------------------------------------------
        for tag, df in (("points", pts_df), ("xg", xg_df), ("gw", gw_df)):
            df.to_csv(f"{OUT_DIR}/{tag}_{label}.csv", index=False)      # ← FIX
            df.to_parquet(f"{OUT_DIR}/{tag}_{label}.parquet", index=False)

        done.add(label)   # ← FIX
        with open(MANIFEST, "w") as fh:
            json.dump({"seasons": sorted(done)}, fh)

    driver.quit()
    print("✅ All seasons scraped.")


if __name__ == "__main__":
    main()
