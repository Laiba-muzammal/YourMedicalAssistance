"""
tracker_utils.py
Shared logic: (1) check a webpage for medicine availability, (2) send email.
Used by both LLM.py (immediate check) and scheduler.py (periodic recheck).

CHANGE LOG:
- check_availability() now renders the page with a headless browser (Playwright)
  instead of a plain `requests.get()`. Sites like Dvago build their product pages
  with client-side JS (React/Vue/Next.js), so the price, stock badge, and
  "Add to Cart" button don't exist in the raw HTML response `requests` gets —
  they're injected into the DOM after JS runs. Playwright launches a real
  (headless) browser, lets that JS execute, and then hands us the FINAL
  rendered HTML — which is what BeautifulSoup should have been parsing all along.
"""

import os
import re
import smtplib
import sys
import json
import time
from email.mime.text import MIMEText
from bs4 import BeautifulSoup
from pathlib import Path
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def extract_first_url(search_result_str: str) -> str | None:
    """Helper: Text/Search payload se pehla valid HTTPS/HTTP URL extract karta hai."""
    urls = re.findall(r'https?://[^\s\'"\]\)]+', str(search_result_str))
    return urls[0] if urls else None


def resolve_product_url(medicine_name: str, url_or_pharmacy: str, web_search_tool=None) -> str:
    """
    Agar user ne direct URL diya hai to wahi use karta hai.
    Agar homepage ya site name (e.g., 'dwatson') diya hai, to Web Search se direct PDP link fetch karta hai.
    """
    target = url_or_pharmacy.strip()

    if (target.startswith("http://") or target.startswith("https://")) and len(target.split("/")) > 3:
        return target

    if web_search_tool:
        query = f"{medicine_name} buy online {target}"
        try:
            search_res = web_search_tool.invoke(query)
            found_url = extract_first_url(search_res)
            if found_url:
                return found_url
        except Exception:
            pass

    if not target.startswith("http"):
        return f"https://{target}.pk" if not ("." in target) else f"https://{target}"

    return target

def _fetch_rendered_html(url: str, timeout_ms: int = 20000) -> str:
    """
    Launch a headless browser, navigate to the URL, wait for the page's
    background stock-check API calls to actually finish (not just for a
    placeholder keyword to appear), and return the fully-rendered HTML.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
            )
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")

            # Wait for network activity (the real stock-check API call) to settle,
            # instead of grabbing the page the instant any placeholder text appears.
            try:
                page.wait_for_load_state("networkidle", timeout=timeout_ms)
            except PlaywrightTimeoutError:
                pass  # some sites keep long-poll/analytics connections open forever

            # Small buffer for React/Vue to re-paint after that data arrives.
            page.wait_for_timeout(1000)

            html = page.content()
            return html
        finally:
            browser.close()

BLOCK_PAGE_MARKERS = [
    "captcha", "verify you are human", "access denied",
    "unusual traffic", "are you a robot", "cloudflare",
]

def is_real_product_page(page_text: str, url: str) -> tuple[bool, str]:
    """
    Sanity-check that we actually landed on a live product page,
    not a 404, redirect, or bot-block shell. Returns (ok, reason_if_not).
    """
    if any(marker in page_text for marker in BLOCK_PAGE_MARKERS):
        return False, "Page appears to be a bot-check/CAPTCHA wall, not real content."

    NOT_FOUND_MARKERS = ["page not found", "404", "doesn't exist", "no results found"]
    if any(marker in page_text for marker in NOT_FOUND_MARKERS):
        return False, "Page looks like a 404 / not-found page."

    if len(page_text.strip()) < 200:
        return False, "Rendered page is suspiciously empty (likely failed to load)."

    return True, ""

def check_availability(url: str, medicine_name: str, timeout: int = 15) -> tuple[bool, str]:
    try:
        html = _fetch_rendered_html(url, timeout_ms=timeout * 1000)
    except Exception as e:
        raise ConnectionError(f"Could not fetch {url}: {e}")

    soup = BeautifulSoup(html, "html.parser")

    # --- PRIMARY SIGNAL: schema.org JSON-LD availability (reliable, static, no JS needed) ---
    for script_tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script_tag.string)
        except (json.JSONDecodeError, TypeError):
            continue

        if data.get("@type") == "Product":
            offers = data.get("offers", {})
            availability = offers.get("availability", "")
            if "InStock" in availability:
                return True, f"Found '{medicine_name}' - schema.org marks it InStock."
            if "OutOfStock" in availability:
                return False, f"Found '{medicine_name}' - schema.org marks it OutOfStock."
            # availability present but unrecognized value - fall through to text-based check

    # --- FALLBACK: keyword text match, only used if no schema.org data was found ---
    raw_text = soup.get_text(separator=" ", strip=True)
    page_text = re.sub(r'\s+', ' ', raw_text).lower()

    clean_med_input = re.sub(r'[^\w\s]', '', medicine_name).lower()
    raw_words = clean_med_input.split()
    form_words = {"tablet","tablets","capsule","capsules","caplet","caplets",
                  "syrup","bottle","bottles","pack","packs","mg","ml"}
    core_keywords = [w for w in raw_words if w not in form_words] or raw_words

    if not all(word in page_text for word in core_keywords):
        return False, f"'{medicine_name}' core terms not found on page."

    NEGATIVE_KEYWORDS = ["out of stock", "sold out", "currently unavailable", "no stock"]
    POSITIVE_KEYWORDS = ["add to cart", "add to bag", "in stock", "buy now", "order now", "instock", "add to basket"]
    has_positive = any(pos in page_text for pos in POSITIVE_KEYWORDS)
    has_negative = any(neg in page_text for neg in NEGATIVE_KEYWORDS)

    if has_negative:
        return False, f"Found '{medicine_name}' but page indicates Out of Stock."
    if has_positive:
        return True, f"Found '{medicine_name}' - item is Available (Add to Cart / In Stock detected)."
    return False, f"Found '{medicine_name}' but no clear availability signal."

def send_email(to_email: str, subject: str, body: str) -> None:
    """
    Send a plain-text email via Gmail SMTP using an App Password
    (never the real account password - see EMAIL_APP_PASSWORD in .env).
    """
    load_dotenv(BASE_DIR / ".env")
    sender = os.getenv("EMAIL_SENDER")
    app_password = os.getenv("EMAIL_APP_PASSWORD")

    if not sender or not app_password:
        raise RuntimeError("EMAIL_SENDER / EMAIL_APP_PASSWORD missing from .env")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email

    max_retries = 3
    backoff_base = 2

    for attempt in range(1, max_retries + 1):
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
                server.login(sender, app_password)
                server.sendmail(sender, [to_email], msg.as_string())
            return
        except smtplib.SMTPAuthenticationError as exc:
            raise RuntimeError(
                "Gmail authentication failed. Use a valid Gmail App Password in "
                "EMAIL_APP_PASSWORD, and make sure EMAIL_SENDER is the same Gmail "
                "address that generated it. Regular Gmail passwords are not accepted."
            ) from exc
        except smtplib.SMTPRecipientsRefused as exc:
            raise RuntimeError(f"Recipient refused by SMTP server: {exc.recipients}") from exc
        except smtplib.SMTPSenderRefused as exc:
            raise RuntimeError(f"Sender address refused by SMTP server: {exc.smtp_error}") from exc
        except smtplib.SMTPDataError as exc:
            raise RuntimeError(f"SMTP data error when sending email: {exc}") from exc
        except (smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected, smtplib.SMTPHeloError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Could not connect to SMTP server: {exc}") from exc
            wait = backoff_base ** (attempt - 1)
            sys.stderr.write(f"SMTP connection issue (attempt {attempt}), retrying in {wait}s...\n")
            time.sleep(wait)
            continue
        except smtplib.SMTPException as exc:
            raise RuntimeError(f"Could not send email via Gmail SMTP: {exc}") from exc
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Unexpected error while sending email: {exc}") from exc
            wait = backoff_base ** (attempt - 1)
            sys.stderr.write(f"Unexpected error (attempt {attempt}), retrying in {wait}s...\n")
            time.sleep(wait)
            continue