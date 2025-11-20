import json
import asyncio
import csv
import subprocess
import re
import concurrent.futures
from datetime import date as dt_date, timedelta
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


async def get_sofascore_session_data(headless=True):
    """
    Launches a browser to get valid Cloudflare cookies and the specific User-Agent.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()

        try:
            # Go to the main TT page
            await page.goto("https://www.sofascore.com/table-tennis", timeout=60000, wait_until="domcontentloaded")

            # Wait for network to settle (handles Cloudflare challenges better than sleep)
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except:
                pass

            # Attempt to close cookie consent if it exists (improves behavior)
            try:
                await page.locator("button:has-text('Agree')").click(timeout=2000)
            except:
                pass

            # Force some interaction to prove humanity
            try:
                await page.wait_for_selector("a[href*='/match']", timeout=5000)
            except:
                pass

            cookies = await context.cookies()
            # CRITICAL: Get the actual UA used by Playwright to match headers later
            user_agent = await page.evaluate("navigator.userAgent")
            
            return cookies, user_agent

        finally:
            await browser.close()


def cookies_to_header(cookie_list):
    return "; ".join([f"{c['name']}={c['value']}" for c in cookie_list])


def generate_date_range(start_date, end_date):
    """Yields dates between start_date and end_date (inclusive)."""
    delta = timedelta(days=1)
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += delta


def fractional_to_decimal(frac):
    if not frac: return None
    frac = str(frac).strip().upper()
    if frac in ("E", "EVS"): return 2.0
    if "/" in frac:
        try:
            a, b = frac.split("/")
            return 1 + float(a) / float(b)
        except ValueError:
            return None
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
    """
    Fallback mechanism: Launches a sync browser to bypass stubborn WAFs.
    Uses the same User-Agent as the session to avoid fingerprint mismatch.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=headers.get("User-Agent", "Mozilla/5.0"),
            extra_http_headers=headers
        )
        page = context.new_page()
        
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
        except:
            # sometimes networkidle fails on APIs, try load
            page.goto(url, wait_until="load", timeout=30000)
            
        page.wait_for_timeout(1000) # Short stabilization

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
    """
    Primary fetch method using Requests. Falls back to Playwright on 403.
    """
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 403:
            # print("403 - cookies invalid or missing, retrying with Playwright rendering...")
            return fetch_json_with_playwright(url, headers)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        # Network error, try fallback
        return fetch_json_with_playwright(url, headers)


def merge_events_and_odds(date, cookies_header, user_agent):
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/",
        "X-Fsign": "SW9D1eZo",
        "Cookie": cookies_header
    }

    events_url = f"{BASE_URL}/scheduled-events/{date}"
    odds_url = f"{BASE_URL}/odds/{PROVIDER_ID}/{date}"

    # Fetch Events
    events_data = fetch_json(events_url, headers)
    events = events_data.get("events", [])
    
    # Fetch Odds
    odds_data = fetch_json(odds_url, headers)
    odds = odds_data.get("odds", {})

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


def fetch_rows_for_date_task(args):
    """Wrapper for threading that unpacks arguments"""
    date_str, cookies_header, user_agent = args
    try:
        rows = merge_events_and_odds(date_str, cookies_header, user_agent)
        for row in rows:
            row["eventDate"] = date_str
        return date_str, rows, None
    except Exception as e:
        return date_str, [], str(e)


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

    st.markdown("---")
    st.subheader("Prepared files")
    st.write(f"**{total_rows}** matches found across **{len(selected_dates)}** dates.")

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
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                label=f"Download {filename}",
                data=data_bytes,
                file_name=filename,
                mime=mime,
                type="primary",
                key=f"download_combined_{fmt}",
            )
        with col2:
             st.success("Ready for download")
             
        with st.expander("Preview Data (First 50 rows)"):
            st.dataframe(combined[:50])
        return

    # Per-date files
    for item in results:
        date_str = item["date"]
        rows = item["rows"]
        if not rows:
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


def run_streamlit_app():
    st.set_page_config(page_title="Sofascore TT Exporter", layout="wide")
    st.title("Sofascore Table Tennis Exporter")
    st.caption("Multi-threaded scraper with date-range support and anti-bot evasion.")

    try:
        with st.spinner("Ensuring Playwright Chromium is available..."):
            install_playwright()
    except RuntimeError as exc:
        st.error(f"{exc}")
        st.stop()

    if "selected_dates" not in st.session_state:
        st.session_state["selected_dates"] = []

    # --- DATE SELECTION UI ---
    st.subheader("1. Select Dates")
    
    col_date, col_btn = st.columns([2, 1])
    with col_date:
        date_selection = st.date_input(
            "Pick a date OR a range (click start, then end)",
            value=[],
            min_value=dt_date(2020, 1, 1),
            max_value=dt_date.today() + timedelta(days=30),
            format="YYYY-MM-DD",
            help="To select a range: click the first date, then click the last date."
        )

    with col_btn:
        st.write("") # Spacing
        st.write("") 
        if st.button("Add to Queue", use_container_width=True):
            new_dates = []
            if len(date_selection) == 2:
                start, end = date_selection
                if start > end:
                    st.error("Start date must be before end date.")
                else:
                    for d in generate_date_range(start, end):
                        new_dates.append(d.strftime("%Y-%m-%d"))
                    st.toast(f"Added {len(new_dates)} dates to queue.", icon="✅")
            elif len(date_selection) == 1:
                new_dates.append(date_selection[0].strftime("%Y-%m-%d"))
                st.toast("Added 1 date to queue.", icon="✅")
            else:
                st.warning("Please pick a date or range first.")

            if new_dates:
                current_set = set(st.session_state["selected_dates"])
                for d_str in new_dates:
                    if d_str not in current_set:
                        st.session_state["selected_dates"].append(d_str)
                st.session_state["selected_dates"].sort()

    # --- QUEUE DISPLAY ---
    if st.session_state["selected_dates"]:
        st.markdown(f"**Queue:** `{len(st.session_state['selected_dates'])} dates selected`")
        with st.expander("View/Edit Queue"):
            st.write(", ".join(st.session_state["selected_dates"]))
            if st.button("Clear Queue"):
                st.session_state["selected_dates"] = []
                st.session_state.pop("prepared_exports", None)
                st.rerun()
    else:
        st.info("Queue is empty. Add dates to proceed.")

    st.divider()

    # --- EXPORT SETTINGS ---
    st.subheader("2. Fetch Data")
    
    c1, c2 = st.columns(2)
    with c1:
        export_format = st.radio("Format", ["CSV", "JSON"], horizontal=True)
    with c2:
        export_mode = st.radio("Mode", [SINGLE_FILE_MODE, PER_DATE_MODE], horizontal=True)

    if st.button("Start Scraping", type="primary", disabled=not st.session_state["selected_dates"]):
        
        # 1. GET COOKIES (Cached in Session State if possible)
        if "sofascore_cookies" not in st.session_state or "sofascore_ua" not in st.session_state:
            try:
                with st.spinner("Initializing Browser Session (Anti-Bot Check)..."):
                    cookies, ua = asyncio.run(get_sofascore_session_data())
                    st.session_state["sofascore_cookies"] = cookies
                    st.session_state["sofascore_ua"] = ua
                    # st.write(f"Debug: Acquired User-Agent: {ua[:30]}...")
            except Exception as exc:
                st.error(f"Failed to initialize session: {exc}")
                st.stop()
        
        cookies_header = cookies_to_header(st.session_state["sofascore_cookies"])
        user_agent = st.session_state["sofascore_ua"]

        # 2. PARALLEL FETCH
        results = []
        dates_to_fetch = st.session_state["selected_dates"]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Prepare arguments for threading
        # We use a tuple (date, cookies, ua) for each task
        tasks = [(d, cookies_header, user_agent) for d in dates_to_fetch]
        
        completed_count = 0
        total_count = len(tasks)

        # Use ThreadPoolExecutor for concurrency
        # Max workers = 4 is usually safe for this API without getting rate-limited too hard
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_date = {executor.submit(fetch_rows_for_date_task, task): task[0] for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_date):
                d_str, rows, err = future.result()
                completed_count += 1
                
                progress = completed_count / total_count
                progress_bar.progress(progress)
                status_text.write(f"Finished {d_str} ({completed_count}/{total_count})")
                
                if err:
                    st.error(f"Error scraping {d_str}: {err}")
                else:
                    results.append({"date": d_str, "rows": rows})

        # Sort results by date again because threading finishes out of order
        results.sort(key=lambda x: x["date"])

        st.session_state["prepared_exports"] = {
            "format": export_format,
            "mode": export_mode,
            "results": results,
        }
        
        status_text.success("Scraping Complete!")
        progress_bar.empty()

    # --- DOWNLOAD SECTION ---
    prepared_exports = st.session_state.get("prepared_exports")
    render_download_section(prepared_exports)


if __name__ == "__main__":
    run_streamlit_app()
