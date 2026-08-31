"""
Read-only client for the Prime Intellect Environments Hub.

Pure stdlib urllib + certifi SSL, per-host token-bucket throttle, retry with
backoff, transparent gzip, content-addressed disk cache. Returns None on
failure so one bad environment never kills a corpus run.

Same idiom as solana_screener/http_client.py, specialised to the three Hub
endpoints and to a cache keyed on the server-supplied content hash rather than
on wall-clock age -- the Hub hands us `sha256` (per version) and `content_hash`
(per file), which makes cache invalidation exact and free.

Nothing here imports or executes environment code. This module is safe to point
at the whole public corpus: it only ever reads source over HTTPS.

Endpoints (verified live 2026-08-06):
    GET /api/v1/environmentshub/?limit=&offset=          -> {total_count, data: [...]}
    GET /api/v1/environmentshub/{owner}/{name}/@{ver}    -> {data: {sha256, wheel_url, metadata}}
    GET /api/v1/environmentshub/{owner}/{name}/@{ver}/inspect[?path=]
            -> directory: {data: {kind: "directory", entries: [...]}}
            -> file:      {data: {kind: "file", content, encoding, truncated}}
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover - certifi is a hard dep in pyproject
    _SSL = ssl.create_default_context()

API_ROOT = "https://api.primeintellect.ai/api/v1/environmentshub"
USER_AGENT = "rlverify/0.1 (+https://github.com/rlverify/rlverify)"

DEFAULT_RATE_HZ = 4.0     # polite: the Hub publishes no documented limit
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
PAGE_SIZE = 100
# Per-environment file-listing ceiling. Generous because the benchmark
# environments that matter most carry one task directory per graded task
# (skillsbench ships 101); breaching it is recorded, never silent.
MAX_WALK_ENTRIES = 4000

# ...and a second, independent ceiling on *requests*, which is the one that
# actually governs wall-clock. A file cap does not bound cost: the walk pays one
# request per directory whether or not that directory contains files, so a repo
# with 3000 sparse directories costs 3000 round trips under any file cap.
# Measured: the Hub's inspect endpoint serves ~0.14-0.4 listings/sec regardless
# of our 4 Hz throttle, so 5410 requests bought 11 hours and fewer than 25
# environments. 400 requests is ~15-45 min worst case per environment, and
# hitting it is reported as truncation exactly like the file cap.
MAX_WALK_REQUESTS = 400


def _seg(value: Any) -> str:
    """Percent-encode one URL path segment.

    Environment names may contain spaces (`bernardwolf/Trading Strategy RL`),
    which urllib rejects outright -- the environment is simply unreachable
    without this. `safe=""` also escapes any `/` so a segment can never inject
    an extra path level.

    Cache-safe: ordinary names contain only unreserved characters, so `quote`
    returns them byte-identical and every URL-keyed cache entry stays valid.
    """
    return urllib.parse.quote(str(value), safe="")


class HubError(Exception):
    """Raised only for programming errors (bad arguments), never for network faults."""


class HubClient:
    """Read-only, caching client for the Environments Hub.

    Every fetch method returns None on network failure rather than raising, so a
    corpus sweep degrades gracefully. Failures are recorded in `self.failures`
    so a run can report exactly what it missed -- silent partial coverage is the
    thing we most want to avoid when publishing a corpus finding.
    """

    def __init__(
        self,
        cache_dir: str = "~/.cache/rlverify/hub",
        rate_hz: float = DEFAULT_RATE_HZ,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        offline: bool = False,
    ) -> None:
        self.cache_dir = os.path.expanduser(cache_dir)
        self.rate_hz = rate_hz
        self.timeout = timeout
        self.retries = retries
        self.offline = offline
        self._last_call = 0.0
        self.failures: List[Dict[str, str]] = []
        os.makedirs(os.path.join(self.cache_dir, "blobs"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "meta"), exist_ok=True)

    # ---------------------------------------------------------------- internals

    def _throttle(self) -> None:
        gap = 1.0 / self.rate_hz
        dt = time.monotonic() - self._last_call
        if dt < gap:
            time.sleep(gap - dt)
        self._last_call = time.monotonic()

    def _blob_path(self, key: str) -> str:
        """Content-addressed path. Two-level fanout keeps directories small at 1.4k+ envs."""
        return os.path.join(self.cache_dir, "blobs", key[:2], key[2:4], key)

    def read_blob(self, key: str) -> Optional[str]:
        path = self._blob_path(key)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except OSError:
                return None
        return None

    def write_blob(self, key: str, text: str) -> None:
        path = self._blob_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)          # atomic; a killed run never leaves a torn blob
        except OSError:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _get_json(self, url: str, cache_key: Optional[str] = None) -> Optional[Any]:
        """GET a JSON document, optionally memoised under an exact content key."""
        if cache_key:
            cached = self.read_blob(cache_key + ".json")
            if cached is not None:
                try:
                    return json.loads(cached)
                except ValueError:
                    pass

        if self.offline:
            self.failures.append({"url": url, "error": "offline and not cached"})
            return None

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }
        for attempt in range(self.retries):
            try:
                self._throttle()
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL) as resp:
                    raw = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    text = raw.decode("utf-8")
                data = json.loads(text)
                if cache_key:
                    self.write_blob(cache_key + ".json", text)
                return data
            except urllib.error.HTTPError as exc:
                # 404/410 are terminal facts about the corpus, not transient faults.
                if exc.code in (404, 410):
                    self.failures.append({"url": url, "error": "HTTP %d" % exc.code})
                    return None
                if attempt == self.retries - 1:
                    self.failures.append({"url": url, "error": "HTTP %d" % exc.code})
                    return None
                time.sleep(1.5 * (attempt + 1))
            except Exception as exc:
                if attempt == self.retries - 1:
                    self.failures.append({"url": url, "error": repr(exc)})
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    @staticmethod
    def _unwrap(payload: Optional[Any]) -> Optional[Any]:
        """Hub responses are {"data": ..., "status": ...}; unwrap to the payload."""
        if payload is None:
            return None
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        return payload

    # ------------------------------------------------------------------- public

    def total_count(self) -> Optional[int]:
        page = self._get_json("%s/?limit=1&offset=0" % API_ROOT)
        if isinstance(page, dict):
            return page.get("total_count")
        return None

    def iter_environments(self, page_size: int = PAGE_SIZE) -> Iterator[Dict[str, Any]]:
        """Yield every public environment record, paginating until exhausted.

        Deliberately does not cache: the catalog is the one thing that genuinely
        changes, and it is cheap (15 requests for the whole corpus).
        """
        offset = 0
        seen = 0
        total = None
        while True:
            url = "%s/?limit=%d&offset=%d" % (API_ROOT, page_size, offset)
            page = self._get_json(url)
            if not isinstance(page, dict):
                break
            if total is None:
                total = page.get("total_count")
            items = page.get("data") or []
            if not items:
                break
            for item in items:
                seen += 1
                yield item
            offset += len(items)
            if total is not None and seen >= total:
                break

    def get_detail(self, owner: str, name: str, version: str) -> Optional[Dict[str, Any]]:
        """Version detail: sha256, wheel_url, metadata.dependencies, requires_python."""
        url = "%s/%s/%s/@%s" % (API_ROOT, _seg(owner), _seg(name), _seg(version))
        key = "detail_" + hashlib.sha256(url.encode()).hexdigest()[:32]
        return self._unwrap(self._get_json(url, cache_key=key))

    def list_files(
        self, owner: str, name: str, version: str, path: str = ""
    ) -> Optional[List[Dict[str, Any]]]:
        """Directory listing. Entries carry `content_hash`, so we can cache exactly."""
        node = self._inspect(owner, name, version, path)
        if not isinstance(node, dict):
            return None
        if node.get("kind") != "directory":
            return None
        return node.get("entries") or []

    def walk_files_ex(
        self, owner: str, name: str, version: str,
        max_entries: int = MAX_WALK_ENTRIES,
        max_requests: int = MAX_WALK_REQUESTS,
    ) -> Tuple[List[Dict[str, Any]], Optional[str], bool]:
        """Recursively enumerate every file. Returns (entries, truncated_by, incomplete).

        `truncated_by` is None for a complete listing, else "entries" or
        "requests" -- the bound that stopped it. The bound is *reported*: the
        previous silent limit of 500 cut `benchflow/skillsbench` from 101 task
        directories to 26 and `proximal/frontier-swe` from 17 to 12, and nothing
        downstream could tell a truncated listing from a complete one. A file
        set that is known-incomplete is a coverage gap, exactly like a file we
        could not parse, and must never be counted as a clean result.

        `incomplete` is a SEPARATE signal from `truncated_by`, and the two must
        never be collapsed: `truncated_by` reports a deterministic cap we chose
        to stop at (a genuine, complete-up-to-the-cap measurement); `incomplete`
        is True when at least one directory's listing could not be read at all
        (an HTTP failure, a timeout, a malformed response) -- we do not know
        what that directory contained, so nothing this call returns can be
        trusted as a full picture. Previously `entries is None` for a failed
        directory was silently skipped (`continue`) with no signal at all,
        which is indistinguishable downstream from "that directory had no
        children" -- the exact distinction this codebase's docstrings insist
        must never be collapsed. A caller (`sync_files`/`record_files`) that
        cannot see this must not write a coverage claim it did not earn.

        Two independent bounds, because they limit different things. The entry
        cap bounds how much we analyse; the request cap bounds what it costs.
        Only the second governs wall-clock -- the walk pays one request per
        directory whether or not that directory holds files, so a repo with
        thousands of sparse directories is expensive under any file cap.

        Breadth-first, so the shallow files that actually matter (the top-level
        env module, `tasks/<task>/tests/...`) are reached before the request
        budget is spent deep inside a vendored subtree.
        """
        out: List[Dict[str, Any]] = []
        queue = [""]
        requests = 0
        truncated_by: Optional[str] = None
        incomplete = False
        while queue:
            if len(out) >= max_entries:
                truncated_by = "entries"
                break
            if requests >= max_requests:
                truncated_by = "requests"
                break
            here = queue.pop(0)               # FIFO: breadth-first
            entries = self.list_files(owner, name, version, here)
            requests += 1
            if entries is None:
                incomplete = True
                continue
            for e in entries:
                if e.get("is_directory"):
                    queue.append(e.get("path") or "")
                else:
                    out.append(e)
                    if len(out) >= max_entries:
                        truncated_by = "entries"
                        break
            if truncated_by:
                break
        # Unvisited directories remain: the listing is short, not complete.
        if truncated_by is None and queue:
            truncated_by = "requests"
        return out, truncated_by, incomplete

    def walk_files(
        self, owner: str, name: str, version: str,
        max_entries: int = MAX_WALK_ENTRIES,
    ) -> List[Dict[str, Any]]:
        """Entries only. Prefer `walk_files_ex`, which also reports truncation."""
        return self.walk_files_ex(owner, name, version, max_entries)[0]

    def read_file(
        self,
        owner: str,
        name: str,
        version: str,
        path: str,
        content_hash: Optional[str] = None,
    ) -> Optional[str]:
        """Fetch one file's source as text.

        When `content_hash` is supplied (it comes free from the directory
        listing) the blob cache is exact: identical content across versions or
        across environments is fetched once, and a changed file is never served
        stale.
        """
        if content_hash:
            cached = self.read_blob(content_hash)
            if cached is not None:
                return cached

        node = self._inspect(owner, name, version, path)
        if not isinstance(node, dict) or node.get("kind") != "file":
            return None

        content = node.get("content")
        if content is None:
            return None
        if node.get("encoding") == "base64":
            import base64
            try:
                content = base64.b64decode(content).decode("utf-8", errors="replace")
            except Exception:
                return None

        if node.get("truncated"):
            # Record it: a truncated read must never be silently analysed as whole.
            self.failures.append(
                {"url": "%s/%s/%s@%s:%s" % (API_ROOT, owner, name, version, path),
                 "error": "truncated"}
            )

        if content_hash:
            self.write_blob(content_hash, content)
        return content

    def _inspect(
        self, owner: str, name: str, version: str, path: str = ""
    ) -> Optional[Dict[str, Any]]:
        base = "%s/%s/%s/@%s/inspect" % (
            API_ROOT, _seg(owner), _seg(name), _seg(version)
        )
        url = base + ("?path=" + urllib.parse.quote(path) if path else "")
        # Directory listings are cheap and change with the version; key on the URL.
        key = "inspect_" + hashlib.sha256(url.encode()).hexdigest()[:32]
        return self._unwrap(self._get_json(url, cache_key=key))


if __name__ == "__main__":
    import tempfile

    # Deterministic, offline, runs first: `incomplete` must fire when a
    # directory listing fails, and must NOT fire -- nor claim a false
    # truncation -- when the walk completes cleanly. No network involved, so
    # this passes with no connectivity, unlike the live smoke test below.
    _smoke_cache = tempfile.mkdtemp(prefix="rlv-hubclient-smoke-")

    class _FlakyDirClient(HubClient):
        """A directory listing that fails exactly once, on a chosen path."""

        def __init__(self, fail_on: str) -> None:
            super().__init__(cache_dir=_smoke_cache)
            self._fail_on = fail_on

        def list_files(self, owner, name, version, path=""):
            if path == self._fail_on:
                self.failures.append({"url": "fake://" + path, "error": "simulated"})
                return None
            if path == "":
                return [{"name": "sub", "path": self._fail_on, "is_directory": True},
                        {"name": "top.py", "path": "top.py", "size": 1, "content_hash": "h"}]
            return []

    flaky = _FlakyDirClient(fail_on="broken")
    entries, truncated_by, incomplete = flaky.walk_files_ex("o", "n", "v")
    assert incomplete is True, "a failed directory listing must set incomplete=True"
    assert truncated_by is None, (
        "a directory read failure is not a deterministic cap -- truncated_by must stay None")
    assert [e["path"] for e in entries] == ["top.py"], (
        "entries collected before the failure must still be returned: %r" % entries)
    print("ok    incomplete=True on a failed directory listing, truncated_by unaffected")

    class _CleanWalkClient(HubClient):
        def __init__(self) -> None:
            super().__init__(cache_dir=_smoke_cache)

        def list_files(self, owner, name, version, path=""):
            if path == "":
                return [{"name": "top.py", "path": "top.py", "size": 1, "content_hash": "h"}]
            return []

    clean = _CleanWalkClient()
    entries, truncated_by, incomplete = clean.walk_files_ex("o", "n", "v")
    assert incomplete is False, "a walk with no failed directory reads must not be incomplete"
    assert truncated_by is None, "a complete walk under the caps must not be reported truncated"
    print("ok    incomplete=False on a walk with no failed directory reads")

    # Smoke test: prove all three endpoints, the recursive walk, and the blob cache.
    c = HubClient()

    total = c.total_count()
    print("total public environments : %s" % total)

    first = None
    for i, env in enumerate(c.iter_environments(page_size=5)):
        if i == 0:
            first = env
        if i >= 4:
            break
    assert first is not None, "list endpoint returned nothing"
    owner = first["owner"]["name"]
    name = first["name"]
    ver = first["latest_version"]
    print("sample env                : %s/%s@%s" % (owner, name, ver))

    detail = c.get_detail(owner, name, ver)
    print("detail sha256             : %s" % (detail or {}).get("sha256", "MISSING"))
    print("dependencies              : %s" % ((detail or {}).get("metadata", {}) or {}).get("dependencies"))

    files = c.walk_files(owner, name, ver)
    py = [f for f in files if f["name"].endswith(".py")]
    print("files / python files      : %d / %d" % (len(files), len(py)))

    if py:
        target = py[0]
        t0 = time.monotonic()
        src = c.read_file(owner, name, ver, target["path"], target.get("content_hash"))
        cold = time.monotonic() - t0
        t0 = time.monotonic()
        src2 = c.read_file(owner, name, ver, target["path"], target.get("content_hash"))
        warm = time.monotonic() - t0
        ok = src is not None and src == src2
        print("read %-24s: %s (%d chars, cold %.0fms, warm %.1fms)"
              % (target["name"], "ok" if ok else "FAILED", len(src or ""), cold * 1000, warm * 1000))
        assert warm < cold, "blob cache did not short-circuit the second read"

    print("failures                  : %d" % len(c.failures))
    for f in c.failures[:5]:
        print("   %s -> %s" % (f["url"][-60:], f["error"]))
