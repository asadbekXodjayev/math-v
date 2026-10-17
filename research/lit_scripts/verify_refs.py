"""
M2 reference verification — checks every candidate in candidates.json against an
authoritative source and records what that source actually says.

- arXiv entries: export.arxiv.org API (title, authors, first-submission date, journal_ref,
  DOI, comment). A title that does not match what we believed is flagged, never silently fixed.
- DOI entries: Crossref (title, authors, container, year).
- URL entries (venue papers with no arXiv/DOI record): the official venue page; the title
  must appear on it, and authors are read from it. (The OpenReview API was tried first; it
  now requires a browser challenge.)
- arXiv entries: Semantic Scholar batch lookup (venue, year) — the scripted venue evidence
  used in REFERENCES.bib. (DBLP was tried first; its API now sits behind a bot challenge.)
  Venues S2 does not know are added by hand in candidates.json ("venue_override") with the
  URL that confirms them.
- Citation counts are deliberately not collected: the free sources disagree by an order of
  magnitude between preprint and venue records, so they would only add noise.

Output: research/artifacts/m2_reference_metadata.json (+ a summary on stdout).

Usage (from mathvis-model/):
    .venv/Scripts/python.exe research/lit_scripts/verify_refs.py [--skip-s2]
"""

import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = Path(__file__).with_name("candidates.json")
OUT = ROOT / "research" / "artifacts" / "m2_reference_metadata.json"

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
UA = {"User-Agent": "mathgen-m2-litreview/1.0 (reference verification script)"}
TITLE_MATCH = 0.90


def get(url: str, retries: int = 4, pause: float = 3.0) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after {e!r}")
            time.sleep(pause * (attempt + 1))
    raise RuntimeError("unreachable")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def arxiv_batch(ids: list[str]) -> dict[str, dict]:
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": ",".join(ids), "max_results": len(ids)}
    )
    root = ET.fromstring(get(url))
    out = {}
    for e in root.findall(f"{ATOM}entry"):
        abs_url = e.findtext(f"{ATOM}id", "").strip()
        m = re.search(r"abs/(.+?)(v\d+)?$", abs_url)
        if not m:
            continue
        base = m.group(1)
        doi = e.findtext(f"{ARXIV}doi")
        out[base] = {
            "abs_url": abs_url,
            "latest_version": m.group(2),
            "title": re.sub(r"\s+", " ", e.findtext(f"{ATOM}title", "")).strip(),
            "authors": [a.findtext(f"{ATOM}name", "").strip() for a in e.findall(f"{ATOM}author")],
            "published": e.findtext(f"{ATOM}published", "")[:10],
            "updated": e.findtext(f"{ATOM}updated", "")[:10],
            "journal_ref": (e.findtext(f"{ARXIV}journal_ref") or "").strip() or None,
            "comment": re.sub(r"\s+", " ", e.findtext(f"{ARXIV}comment") or "").strip() or None,
            "doi": doi.strip() if doi else None,
            "primary_category": (e.find(f"{ARXIV}primary_category").get("term")
                                 if e.find(f"{ARXIV}primary_category") is not None else None),
            "abstract": re.sub(r"\s+", " ", e.findtext(f"{ATOM}summary", "")).strip(),
        }
    return out


def crossref(doi: str) -> dict:
    msg = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    issued = msg.get("issued", {}).get("date-parts", [[None]])[0]
    return {
        "title": " ".join(msg.get("title", [])),
        "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in msg.get("author", [])],
        "container": " ".join(msg.get("container-title", [])),
        "year": issued[0] if issued else None,
        "volume": msg.get("volume"),
        "issue": msg.get("issue"),
        "page": msg.get("page"),
        "publisher": msg.get("publisher"),
        "type": msg.get("type"),
    }


def official_page(url: str, title: str) -> dict | None:
    """Venue page for a paper with no arXiv/DOI record. Confirms the title is on the page and
    reads the author line that follows it (icml.cc lists authors separated by '⋅')."""
    page = get(url).decode("utf-8", "replace")
    lines = [ln.strip() for ln in re.sub(r"<[^>]+>", "\n", page).splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if similarity(ln, title) >= 0.98 and i + 1 < len(lines) and "⋅" in lines[i + 1]:
            return {
                "title": ln,
                "authors": [a.strip() for a in lines[i + 1].split("⋅")],
                "listed_under": lines[i - 2] if i >= 2 else None,
                "url": url,
            }
    return None


def semantic_scholar(arxiv_ids: list[str]) -> dict[str, dict]:
    """One batch POST for all arXiv IDs (the per-paper endpoint is rate-limited without a key)."""
    url = ("https://api.semanticscholar.org/graph/v1/paper/batch?"
           "fields=title,venue,year,publicationVenue,publicationTypes,externalIds")
    body = json.dumps({"ids": [f"ARXIV:{i}" for i in arxiv_ids]}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, data=body, headers={**UA, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                papers = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            print(f"  S2 429, retry {attempt + 1}")
            time.sleep(10 * (attempt + 1))
    out = {}
    for i, p in zip(arxiv_ids, papers):
        if not p:
            continue
        pv = p.get("publicationVenue") or {}
        out[i] = {
            "title": p.get("title"),
            "venue": p.get("venue") or None,
            "publication_venue": pv.get("name"),
            "publication_venue_type": pv.get("type"),
            "year": p.get("year"),
            "publication_types": p.get("publicationTypes"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-s2", action="store_true")
    args = ap.parse_args()

    refs = json.loads(CANDIDATES.read_text(encoding="utf-8"))["refs"]
    keys = [r["key"] for r in refs]
    assert len(keys) == len(set(keys)), "duplicate bibtex keys in candidates.json"

    results: dict[str, dict] = {}

    arxiv_refs = [r for r in refs if r.get("arxiv")]
    for i in range(0, len(arxiv_refs), 20):
        chunk = arxiv_refs[i:i + 20]
        print(f"arXiv batch {i // 20 + 1}: {len(chunk)} ids")
        meta = arxiv_batch([r["arxiv"] for r in chunk])
        for r in chunk:
            m = meta.get(r["arxiv"])
            sim = similarity(r["title"], m["title"]) if m else 0.0
            results[r["key"]] = {
                "source": "arxiv",
                "id": r["arxiv"],
                "believed_title": r["title"],
                "found": m is not None,
                "title_similarity": round(sim, 3),
                "title_ok": m is not None and sim >= TITLE_MATCH,
                "metadata": m,
                "themes": r["themes"],
            }
        time.sleep(3.0)

    for r in refs:
        if r.get("doi"):
            print(f"Crossref: {r['doi']}")
            try:
                m = crossref(r["doi"])
            except Exception as e:  # recorded as a failure, not hidden
                m = None
                print(f"  FAILED: {e!r}")
            sim = similarity(r["title"], m["title"]) if m else 0.0
            results[r["key"]] = {
                "source": "crossref",
                "id": r["doi"],
                "believed_title": r["title"],
                "found": m is not None,
                "title_similarity": round(sim, 3),
                "title_ok": m is not None and sim >= TITLE_MATCH,
                "metadata": m,
                "themes": r["themes"],
            }
            time.sleep(1.0)

    for r in refs:
        if r.get("url"):
            print(f"Official page: {r['url']}")
            try:
                m = official_page(r["url"], r["title"])
            except Exception as e:
                m = None
                print(f"  FAILED: {e!r}")
            sim = similarity(r["title"], m["title"]) if m else 0.0
            results[r["key"]] = {
                "source": "url",
                "id": r["url"],
                "believed_title": r["title"],
                "found": m is not None,
                "title_similarity": round(sim, 3),
                "title_ok": m is not None and sim >= TITLE_MATCH,
                "metadata": m,
                "themes": r["themes"],
            }

    if not args.skip_s2:
        print("Semantic Scholar batch")
        try:
            s2 = semantic_scholar([r["arxiv"] for r in arxiv_refs])
        except Exception as e:  # a hint only; never worth aborting verification for
            print(f"  S2 unavailable, venue hints skipped: {e!r}")
            s2 = {}
        for r in arxiv_refs:
            results[r["key"]]["semantic_scholar"] = s2.get(r["arxiv"])

    for r in refs:
        if r.get("venue_override"):
            results[r["key"]]["venue_override"] = r["venue_override"]

    bad = [k for k, v in results.items() if not v["title_ok"]]
    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_candidates": len(refs),
        "n_found": sum(v["found"] for v in results.values()),
        "n_title_ok": sum(v["title_ok"] for v in results.values()),
        "flagged": bad,
    }
    OUT.write_text(json.dumps({"summary": summary, "refs": results}, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print("\n" + json.dumps(summary, indent=2))
    for k in bad:
        v = results[k]
        found = (v["metadata"] or {}).get("title")
        print(f"FLAG {k}: believed '{v['believed_title']}' | source says '{found}' "
              f"(sim {v['title_similarity']})")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
