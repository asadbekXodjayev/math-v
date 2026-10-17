"""
Merges the official-page venue checks (research/artifacts/m2_venue_checks.json, produced by
the M2 venue-verification pass) into candidates.json as "venue_override" entries, which
build_bib.py then uses. Where the published title differs from the arXiv title, the
published title is fetched from Crossref by DOI (never typed by hand) and stored as
"venue_title".

Crossref-only promotions (publisher-deposited record, official page bot-walled) are listed
in CROSSREF_PROMOTIONS with the DOI; they are held to the same standard as the two
Crossref-sourced references.

Usage (from mathvis-model/):
    .venv/Scripts/python.exe research/lit_scripts/merge_venues.py
"""

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKS = ROOT / "research" / "artifacts" / "m2_venue_checks.json"
CANDIDATES = Path(__file__).with_name("candidates.json")
UA = {"User-Agent": "mathgen-m2-litreview/1.0 (venue merge)"}

# Keys whose published title differs from the arXiv title (reported by the venue check).
TITLE_FROM_DOI = {"hoffmann2022chinchilla", "hupkes2023taxonomy", "razeghi2022frequencies"}
CROSSREF_PROMOTIONS = {"chen2024posthinking": "10.1109/tnnls.2025.3591266"}


def crossref(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.loads(r.read())["message"]


def main() -> None:
    checks = json.loads(CHECKS.read_text(encoding="utf-8"))
    confirmed = {k: v for k, v in checks.items() if not k.startswith("_")}
    confirmed.update(checks.get("_out_of_scope_confirmed", {}))

    for key, doi in CROSSREF_PROMOTIONS.items():
        msg = crossref(doi)
        confirmed[key] = {
            "venue": " ".join(msg["container-title"]),
            "year": msg["issued"]["date-parts"][0][0],
            "type": "article",
            "volume": msg.get("volume"),
            "number": msg.get("issue"),
            "pages": msg.get("page"),
            "doi": doi,
            "venue_title": " ".join(msg["title"]),
            "evidence": f"https://api.crossref.org/works/{doi} (publisher-deposited Crossref record; "
                        "IEEE Xplore is bot-walled)",
        }

    for key in TITLE_FROM_DOI:
        confirmed[key]["venue_title"] = re.sub(r"\s+", " ", " ".join(crossref(confirmed[key]["doi"])["title"]))

    data = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    n = 0
    for ref in data["refs"]:
        v = confirmed.get(ref["key"])
        if v:
            ref["venue_override"] = {f: v[f] for f in ("venue", "year", "type", "volume", "number", "pages",
                                                      "doi", "venue_title", "evidence") if v.get(f)}
            n += 1
    CANDIDATES.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    missing = sorted(set(confirmed) - {r["key"] for r in data["refs"]})
    print(f"merged {n} venue overrides into {CANDIDATES.name}; unknown keys: {missing or 'none'}")
    for key in sorted(TITLE_FROM_DOI | set(CROSSREF_PROMOTIONS)):
        print(f"  {key}: published title = {confirmed[key]['venue_title']!r}")


if __name__ == "__main__":
    main()
