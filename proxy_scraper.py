import requests
import json
import xml.etree.ElementTree as ET
import re
import ipaddress
import time
import logging
import argparse
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Any, List, Optional, Set
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Logging (thread-safe, replaces tqdm.set_description for status messages)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
shutdown_event = threading.Event()

# ---------------------------------------------------------------------------
# Regex: valid IP + port 1-65535
# ---------------------------------------------------------------------------
PROXY_REGEX = re.compile(
    r'\b'
    r'(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'
    r':'
    r'(?:[1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])'
    r'\b'
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_HEADERS = HEADERS.copy()
if GITHUB_TOKEN:
    API_HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"


# ---------------------------------------------------------------------------
# Session with retry strategy
# ---------------------------------------------------------------------------
def build_session() -> requests.Session:
    """Creates a session with automatic retry on transient errors."""
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,                          # 1 s, 2 s, 4 s
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,           # honours GitHub's Retry-After
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


session = build_session()


# ---------------------------------------------------------------------------
# IP helpers
# ---------------------------------------------------------------------------
def is_public_ip(ip: str) -> bool:
    """Returns True only for routable, public IP addresses."""
    try:
        addr = ipaddress.ip_address(ip)
        return not (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_unspecified
            or addr.is_reserved
        )
    except ValueError:
        return False


def filter_proxies(proxies: List[str]) -> List[str]:
    """Drops proxies with private / loopback / reserved IPs."""
    result = []
    for proxy in proxies:
        ip = proxy.rsplit(":", 1)[0]
        if is_public_ip(ip):
            result.append(proxy)
    return result


# ---------------------------------------------------------------------------
# GitHub rate-limit guard
# ---------------------------------------------------------------------------
_rate_limit_lock = threading.Lock()


def check_and_wait_rate_limit() -> None:
    """Checks GitHub API rate limit and sleeps if nearly exhausted."""
    with _rate_limit_lock:
        try:
            r = session.get(
                "https://api.github.com/rate_limit",
                headers=API_HEADERS,
                timeout=5,
            )
            data = r.json()
            remaining = data["rate"]["remaining"]
            reset_ts = data["rate"]["reset"]
            if remaining < 5:
                wait = max(0, reset_ts - time.time()) + 5
                logger.warning(
                    "GitHub rate limit: %d requests left. "
                    "Sleeping %.0f s until reset…",
                    remaining,
                    wait,
                )
                time.sleep(wait)
            elif remaining < 20:
                logger.warning(
                    "GitHub rate limit low: %d requests remaining.", remaining
                )
        except Exception as exc:
            logger.debug("Rate-limit check failed: %s", exc)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def find_proxies_in_text(text: str) -> List[str]:
    return PROXY_REGEX.findall(text)


def parse_json_recursively(element: Any, found: List[str]) -> None:
    if isinstance(element, dict):
        for v in element.values():
            parse_json_recursively(v, found)
    elif isinstance(element, list):
        for item in element:
            parse_json_recursively(item, found)
    elif isinstance(element, str):
        found.extend(find_proxies_in_text(element))


def parse_xml_recursively(element: ET.Element, found: List[str]) -> None:
    if element.text:
        found.extend(find_proxies_in_text(element.text))
    for child in element:
        parse_xml_recursively(child, found)


# ---------------------------------------------------------------------------
# File fetching
# ---------------------------------------------------------------------------
def fetch_and_parse_file(file_url: str, timeout: int) -> List[str]:
    """Downloads a file and extracts proxy strings from it."""
    if shutdown_event.is_set():
        return []
    try:
        response = session.get(file_url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        content = response.text

        if file_url.endswith(".json"):
            proxies: List[str] = []
            try:
                parse_json_recursively(json.loads(content), proxies)
            except json.JSONDecodeError:
                proxies = find_proxies_in_text(content)
            return proxies

        if file_url.endswith(".xml"):
            proxies = []
            try:
                parse_xml_recursively(ET.fromstring(content), proxies)
            except ET.ParseError:
                proxies = find_proxies_in_text(content)
            return proxies

        return find_proxies_in_text(content)

    except requests.RequestException as exc:
        logger.debug("Error fetching %s: %s", file_url, exc)
        return []


# ---------------------------------------------------------------------------
# GitHub repo helpers
# ---------------------------------------------------------------------------
def get_default_branch(user: str, repo: str) -> Optional[str]:
    check_and_wait_rate_limit()
    try:
        r = session.get(
            f"https://api.github.com/repos/{user}/{repo}",
            headers=API_HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("default_branch")
    except Exception as exc:
        logger.debug("Could not get default branch for %s/%s: %s", user, repo, exc)
        return None


def get_files_from_repo(repo_url: str, timeout: int) -> List[str]:
    """Returns raw-content URLs for .txt/.json/.xml files in the repo."""
    if shutdown_event.is_set():
        return []

    parts = repo_url.strip("/").split("/")
    if len(parts) < 2:
        logger.warning("Invalid repository URL: %s", repo_url)
        return []

    user, repo = parts[-2], parts[-1]
    branch = get_default_branch(user, repo)
    if not branch:
        logger.warning("Skipping %s/%s: cannot determine default branch.", user, repo)
        return []

    check_and_wait_rate_limit()
    try:
        r = session.get(
            f"https://api.github.com/repos/{user}/{repo}/git/trees/{branch}?recursive=1",
            headers=API_HEADERS,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()

        if data.get("truncated"):
            logger.warning("File list for %s/%s is truncated by GitHub.", user, repo)

        urls = []
        for item in data.get("tree", []):
            if shutdown_event.is_set():
                break
            path = item.get("path", "")
            if item.get("type") == "blob" and path.endswith((".txt", ".json", ".xml")):
                urls.append(
                    f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
                )
        return urls

    except Exception as exc:
        logger.debug("Error listing files for %s/%s: %s", user, repo, exc)
        return []


# ---------------------------------------------------------------------------
# Repository processing
# ---------------------------------------------------------------------------
def process_repository(repo_url: str, timeout: int) -> Set[str]:
    """Scans one repository and returns a set of unique public proxies."""
    if shutdown_event.is_set():
        return set()

    user, repo = repo_url.strip("/").split("/")[-2:]
    logger.info("Scanning %s/%s …", user, repo)

    files = get_files_from_repo(repo_url, timeout)
    if not files:
        logger.info("No eligible files found in %s/%s.", user, repo)
        return set()

    repo_proxies: Set[str] = set()
    with tqdm(total=len(files), desc=f"{user}/{repo}", leave=False, unit="file") as fpbar:
        for file_url in files:
            if shutdown_event.is_set():
                break
            raw = fetch_and_parse_file(file_url, timeout)
            repo_proxies.update(filter_proxies(raw))
            fpbar.update(1)

    logger.info(
        "Finished %s/%s — found %d unique public proxies.", user, repo, len(repo_proxies)
    )
    return repo_proxies


# ---------------------------------------------------------------------------
# Proxy validation (optional)
# ---------------------------------------------------------------------------
def validate_proxy(proxy: str, timeout: int) -> bool:
    """Returns True if the proxy forwards traffic successfully."""
    proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
    try:
        r = requests.get(
            "http://httpbin.org/ip",
            proxies=proxies,
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        return False


def validate_proxies(proxies: Set[str], workers: int, timeout: int) -> Set[str]:
    """Concurrently validates all proxies and returns the live ones."""
    live: Set[str] = set()
    proxy_list = list(proxies)
    logger.info("Validating %d proxies (this may take a while)…", len(proxy_list))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(validate_proxy, p, timeout): p for p in proxy_list
        }
        with tqdm(total=len(proxy_list), desc="Validating", unit="proxy") as pbar:
            for future in as_completed(future_map):
                if shutdown_event.is_set():
                    break
                proxy = future_map[future]
                try:
                    if future.result():
                        live.add(proxy)
                except Exception:
                    pass
                pbar.update(1)

    logger.info("Validation complete: %d / %d proxies are live.", len(live), len(proxy_list))
    return live


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape free proxies from GitHub repositories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repos", default="repositories.txt",
        help="Path to file with repository URLs (one per line).",
    )
    parser.add_argument(
        "--output", default="proxies_output.txt",
        help="Output file for collected proxies.",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Number of parallel worker threads.",
    )
    parser.add_argument(
        "--timeout", type=int, default=10,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Validate each proxy before saving (slow).",
    )
    parser.add_argument(
        "--check-workers", type=int, default=50,
        help="Threads used during proxy validation.",
    )
    parser.add_argument(
        "--check-timeout", type=int, default=7,
        help="Timeout for each proxy validation request.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Merge new proxies with existing output file instead of overwriting.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # -- Load repository list ------------------------------------------------
    if not os.path.exists(args.repos):
        logger.error("'%s' not found. Create it and add repository URLs, one per line.", args.repos)
        return

    with open(args.repos, "r") as f:
        repo_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not repo_urls:
        logger.error("'%s' is empty.", args.repos)
        return

    # -- Optionally seed with existing proxies (--append mode) ---------------
    all_proxies: Set[str] = set()
    if args.append and os.path.exists(args.output):
        with open(args.output, "r") as f:
            existing = {line.strip() for line in f if line.strip()}
        all_proxies.update(existing)
        logger.info("Loaded %d existing proxies from '%s'.", len(existing), args.output)

    # -- Auth hint -----------------------------------------------------------
    if not GITHUB_TOKEN:
        logger.warning(
            "GITHUB_TOKEN is not set. API rate limit: 60 req/h. "
            "Set it for 5 000 req/h: "
            "https://docs.github.com/en/authentication/keeping-your-account-and-data-secure"
            "/creating-a-personal-access-token"
        )

    logger.info("Starting proxy collection from %d repositories…", len(repo_urls))
    if args.check:
        logger.info("Proxy validation is ENABLED (--check).")

    # -- Scrape --------------------------------------------------------------
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            with tqdm(total=len(repo_urls), desc="Repositories", unit="repo") as pbar:
                future_map = {
                    executor.submit(process_repository, url, args.timeout): url
                    for url in repo_urls
                }
                for future in as_completed(future_map):
                    if shutdown_event.is_set():
                        for f in future_map:
                            f.cancel()
                        break
                    try:
                        result = future.result()
                        all_proxies.update(result)
                    except Exception as exc:
                        logger.debug("Unhandled error for %s: %s", future_map[future], exc)
                    finally:
                        pbar.update(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user — shutting down…")
        shutdown_event.set()

    # -- Validate (optional) -------------------------------------------------
    if args.check and all_proxies and not shutdown_event.is_set():
        all_proxies = validate_proxies(all_proxies, args.check_workers, args.check_timeout)

    # -- Save ----------------------------------------------------------------
    if all_proxies:
        logger.info("Saving %d unique proxies to '%s'…", len(all_proxies), args.output)
        sorted_proxies = sorted(all_proxies)
        with open(args.output, "w") as f:
            f.write("\n".join(sorted_proxies) + "\n")
        logger.info("Done.")
    elif not shutdown_event.is_set():
        logger.info("No proxies found.")


if __name__ == "__main__":
    main()
