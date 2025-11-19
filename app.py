import json
import asyncio
import csv
import subprocess
import re
from datetime import date as dt_date
from io import StringIO

import streamlit as st
import requests
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright


BASE_URL = "https://www.sofascore.com/api/v1/sport/table-tennis"
PROVIDER_ID = 226
MAX_SETS = 7
PLAYWRIGHT_CMD = ["playwright", "install", "chromium"]


@st.cache_resource(show_spinner=False)
def install_playwright():
    """
    Ensures the Playwright Chromium browser is installed once per Streamlit session.
    """
    try:
        subprocess.run(PLAYWRIGHT_CMD, check=True, timeout=300)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Failed to install Playwright Chromium.") from exc
    return True


async def get_sofascore_cookies(headless=True):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()

        page = await context.new_page()

        await page.goto("https://www.sofascore.com/table-tennis", timeout=60000)

        # Wait long enough for Cloudflare + React requests
        await page.wait_for_timeout(5000)

        # Force API calls by opening a match list item
        try:
            await page.locator("a[href*='/match']").first.click(timeout=5000)
        except:
            pass

        await page.wait_for_timeout(5000)

        # Trigger internal stats API calls
        try:
            await page.get_by_text("H2H").click(timeout=3000)
            await page.wait_for_timeout(3000)
        except:
            pass

        cookies = await context.cookies()

        await browser.close()
        return cookies


def cookies_to_header(cookie_list):
    return "; ".join([f"{c['name']}={c['value']}" for c in cookie_list])


def fractional_to_decimal(frac):
    if not frac: return None
    frac = frac.strip().upper()
    if frac in ("E", "EVS"): return 2.0
    if "/" in frac:
        a, b = frac.split("/")
        return 1 + float(a) / float(b)
    try:
        return float(frac)
    except:
        return None


def _normalize_period_value(value, preferred_key=None):
    if value is None:
        return ""
    if isinstance(value, dict):
        if preferred_key and value.get(preferred_key) not in (None, ""):
            return value[preferred_key]
        for key in ("current", "value", "display", "home", "away"):
            if value.get(key) not in (None, ""):
                return value[key]
        return ""
    return value


def extract_set_scores(home_score, away_score, max_sets=MAX_SETS):
    home_score = home_score or {}
    away_score = away_score or {}
    scores = {}

    for idx in range(1, max_sets + 1):
        period_key = f"period{idx}"
        scores[f"s{idx}h"] = _normalize_period_value(home_score.get(period_key), preferred_key="home")
        scores[f"s{idx}a"] = _normalize_period_value(away_score.get(period_key), preferred_key="away")

    return scores


def rows_to_csv_bytes(rows):
    if not rows:
        return b""

    serialized_rows = []
    for row in rows:
        serialized_row = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                serialized_row[key] = json.dumps(value, ensure_ascii=False)
            else:
                serialized_row[key] = value
        serialized_rows.append(serialized_row)

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=serialized_rows[0].keys())
    writer.writeheader()
    writer.writerows(serialized_rows)
    return buffer.getvalue().encode("utf-8")


def rows_to_json_bytes(rows):
    return json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")


def _safe_int(value):
    if value in (None, "", "-"):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match.group())
    except ValueError:
        return None


def compute_score_metrics(row):
    home_points = 0
    away_points = 0
    total_sets = 0
    first_set_points = None

    for idx in range(1, MAX_SETS + 1):
        home_val = _safe_int(row.get(f"s{idx}h"))
        away_val = _safe_int(row.get(f"s{idx}a"))

        if home_val is None and away_val is None:
            continue

        total_sets += 1

        if home_val is not None:
            home_points += home_val
        if away_val is not None:
            away_points += away_val

        if idx == 1 and first_set_points is None:
            first_set_points = (home_val or 0) + (away_val or 0)

    total_points = home_points + away_points
    points_margin = home_points - away_points

    return {
        "totalSets": total_sets,
        "totalPoints": total_points,
        "firstSetPoints": first_set_points if first_set_points is not None else "",
        "homePoints": home_points,
        "awayPoints": away_points,
        "pointsMargin": points_margin,
    }


def compute_odds_normalization(row):
    home_odds = row.get("oddsHome")
    away_odds = row.get("oddsAway")

    try:
        home_odds = float(home_odds) if home_odds not in (None, "") else None
        away_odds = float(away_odds) if away_odds not in (None, "") else None
    except (TypeError, ValueError):
        home_odds = away_odds = None

    if not home_odds or not away_odds or home_odds <= 0 or away_odds <= 0:
        return "", ""

    implied_home = 1 / home_odds
    implied_away = 1 / away_odds
    total = implied_home + implied_away

    if total <= 0:
        return "", ""

    normalized_home = round(implied_home / total, 4)
    normalized_away = round(implied_away / total, 4)
    return normalized_home, normalized_away


def fetch_json_with_playwright(url, headers):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=headers.get("User-Agent", "Mozilla/5.0"),
            extra_http_headers=headers
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=45000)
        page.wait_for_timeout(2000)

        raw = None
        pre = page.locator("pre")
        if pre.count():
            raw = pre.first.inner_text()
        else:
            raw = page.inner_text("body")

        browser.close()

    if not raw:
        raise ValueError("Unable to extract JSON payload from Sofascore response.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Received non-JSON payload from Sofascore.") from exc


def fetch_json(url, headers):
    r = requests.get(url, headers=headers)
    if r.status_code == 403:
        print("403 - cookies invalid or missing, retrying with Playwright rendering...")
        return fetch_json_with_playwright(url, headers)
    r.raise_for_status()
    return r.json()


def merge_events_and_odds(date, cookies_header):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/",
        "X-Fsign": "SW9D1eZo",
        "Cookie": cookies_header
    }

    events_url = f"{BASE_URL}/scheduled-events/{date}"
    odds_url = f"{BASE_URL}/odds/{PROVIDER_ID}/{date}"

    events = fetch_json(events_url, headers).get("events", [])
    odds = fetch_json(odds_url, headers).get("odds", {})

    merged = []

    for ev in events:
        eid = str(ev["id"])
        if eid not in odds:
            continue

        market = odds[eid]
        choices = market.get("choices", [])

        home_choice = next((c for c in choices if c.get("name") == "1"), None)
        away_choice = next((c for c in choices if c.get("name") == "2"), None)

        row = {
            "eventId": eid,
            "tournament": ev.get("tournament", {}).get("name", ""),
            "homePlayer": ev.get("homeTeam", {}).get("name", ""),
            "awayPlayer": ev.get("awayTeam", {}).get("name", ""),
            "homeScore": ev.get("homeScore", {}).get("display", ""),
            "awayScore": ev.get("awayScore", {}).get("display", ""),
            "oddsHome": fractional_to_decimal(home_choice.get("fractionalValue") if home_choice else None),
            "oddsAway": fractional_to_decimal(away_choice.get("fractionalValue") if away_choice else None),
        }

        row.update(extract_set_scores(ev.get("homeScore"), ev.get("awayScore")))
        row["startTimestamp"] = ev.get("startTimestamp")

        row.update(compute_score_metrics(row))
        norm_home, norm_away = compute_odds_normalization(row)
        row["oddsNormalizationHome"] = norm_home
        row["oddsNormalizationAway"] = norm_away

        merged.append(row)

    return merged


def fetch_rows_for_date(date_str, cookies_header):
    rows = merge_events_and_odds(date_str, cookies_header)
    for row in rows:
        row["eventDate"] = date_str
    return rows


SINGLE_FILE_MODE = "Single file (combined)"
PER_DATE_MODE = "Separate file per date"


def build_filename_for_dates(dates, extension):
    if not dates:
        return f"tt_export.{extension}"

    sorted_dates = sorted(dates)
    if len(sorted_dates) == 1:
        return f"tt_{sorted_dates[0]}.{extension}"
    return f"tt_{sorted_dates[0]}_{sorted_dates[-1]}_{len(sorted_dates)}dates.{extension}"


def render_download_section(prepared_exports):
    if not prepared_exports:
        return

    fmt = prepared_exports["format"]
    mode = prepared_exports["mode"]
    results = prepared_exports["results"]

    selected_dates = [item["date"] for item in results]
    total_rows = sum(len(item["rows"]) for item in results)

    st.subheader("Prepared files")
    st.write(f"{total_rows} rows across {len(selected_dates)} date(s).")

    if total_rows == 0:
        st.info("No events were returned for the selected dates.")
        return

    encoder = rows_to_csv_bytes if fmt == "CSV" else rows_to_json_bytes
    extension = "csv" if fmt == "CSV" else "json"
    mime = "text/csv" if fmt == "CSV" else "application/json"

    if mode == SINGLE_FILE_MODE:
        combined = []
        for item in results:
            combined.extend(item["rows"])
        data_bytes = encoder(combined)
        filename = build_filename_for_dates(selected_dates, extension)
        st.download_button(
            label=f"Download {filename}",
            data=data_bytes,
            file_name=filename,
            mime=mime,
            type="primary",
            key=f"download_combined_{fmt}",
        )
        st.caption("Preview (first 50 rows)")
        st.dataframe(combined[:50])
        return

    # Per-date files
    for item in results:
        date_str = item["date"]
        rows = item["rows"]
        if not rows:
            st.info(f"No matches found for {date_str}.")
            continue

        filename = f"tt_{date_str}.{extension}"
        data_bytes = encoder(rows)
        st.download_button(
            label=f"Download {filename}",
            data=data_bytes,
            file_name=filename,
            mime=mime,
            key=f"download_{date_str}_{fmt}",
        )
        st.caption(f"{date_str} preview (first 25 rows)")
        st.dataframe(rows[:25])


def run_streamlit_app():
    st.set_page_config(page_title="Sofascore TT Exporter", layout="wide")
    st.title("Sofascore Table Tennis Exporter")
    st.caption("Pick one or more dates and export Sofascore data as CSV or JSON.")

    try:
        with st.spinner("Ensuring Playwright Chromium is available..."):
            install_playwright()
    except RuntimeError as exc:
        st.error(f"{exc}")
        st.stop()

    if "selected_dates" not in st.session_state:
        st.session_state["selected_dates"] = []

    date_col, action_col = st.columns([3, 1])
    with date_col:
        selected_date = st.date_input(
            "Pick a match date",
            value=dt_date.today(),
            format="YYYY-MM-DD",
        )
    with action_col:
        if st.button("Add date", use_container_width=True):
            date_str = selected_date.strftime("%Y-%m-%d")
            if date_str in st.session_state["selected_dates"]:
                st.info(f"{date_str} is already in the queue.")
            else:
                st.session_state["selected_dates"].append(date_str)
                st.session_state["selected_dates"].sort()

    if st.session_state["selected_dates"]:
        st.success(
            "Dates queued: "
            + ", ".join(st.session_state["selected_dates"])
        )
        if st.button("Clear dates"):
            st.session_state["selected_dates"] = []
            st.session_state.pop("prepared_exports", None)
    else:
        st.info("Add at least one date to continue.")

    export_format = st.radio(
        "Export format",
        options=["CSV", "JSON"],
        horizontal=True,
    )
    export_mode = st.radio(
        "File mode",
        options=[SINGLE_FILE_MODE, PER_DATE_MODE],
        horizontal=True,
    )

    if st.button("Fetch & prepare files", type="primary"):
        if not st.session_state["selected_dates"]:
            st.warning("Please add at least one date first.")
        else:
            try:
                with st.spinner("Collecting Cloudflare cookies..."):
                    cookies = asyncio.run(get_sofascore_cookies())
                cookies_header = cookies_to_header(cookies)

                results = []
                for date_str in st.session_state["selected_dates"]:
                    with st.spinner(f"Scraping {date_str} data..."):
                        rows = fetch_rows_for_date(date_str, cookies_header)
                    results.append({"date": date_str, "rows": rows})

                st.session_state["prepared_exports"] = {
                    "format": export_format,
                    "mode": export_mode,
                    "results": results,
                }

                if any(item["rows"] for item in results):
                    st.success("Data fetched. Use the download section below.")
                else:
                    st.warning("No events returned for the selected dates.")
            except Exception as exc:
                st.error(f"Failed to fetch data: {exc}")

    prepared_exports = st.session_state.get("prepared_exports")
    render_download_section(prepared_exports)


if __name__ == "__main__":
    run_streamlit_app()
