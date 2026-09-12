#!/usr/bin/env python3
"""Pre-publish checks for the built site.

Catches the mechanical failures that are embarrassing in front of a recruiter and
invisible to whoever built the page: dead internal links, missing alt text, heavy
images, absent link-preview metadata, leftover placeholders, and any term from the
confidentiality denylist appearing in the output.

    python scripts/check_site.py dist/ --denylist content/denylist.txt

Exit code 1 if anything at ERROR level fails. Warnings are judgment calls.
"""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

PLACEHOLDERS = [r"\bTODO\b", r"\bTBD\b", r"lorem ipsum", r"\bFIXME\b", r"\bXXX\b",
                r"\[your ", r"placeholder", r"example\.com"]
WEAK_WORDS = ["passionate about", "leveraged", "cutting-edge", "seamless", "synergy",
              "in today's fast-paced", "results-driven", "think outside the box"]
MAX_IMG_KB = 600
MAX_PAGE_MB = 2.5


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.imgs, self.links, self.videos = [], [], []
        self.h1 = 0
        self.headings = []
        self.title = ""
        self.metas = {}
        self.props = {}
        self.lang_seen = False
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "img":
            self.imgs.append(a)
        elif tag == "video":
            self.videos.append(a)
        elif tag == "a" and a.get("href"):
            self.links.append(a["href"])
        elif tag == "html":
            self.lang_seen = bool(a.get("lang"))
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            if a.get("name"):
                self.metas[a["name"]] = a.get("content", "")
            if a.get("property"):
                self.props[a["property"]] = a.get("content", "")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.headings.append(int(tag[1]))
            if tag == "h1":
                self.h1 += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def text_of(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    return re.sub(r"<[^>]+>", " ", html)


def check_page(path: Path, root: Path, errors: list, warns: list):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    p = Page()
    p.feed(raw)
    where = path.relative_to(root)

    if not p.title.strip():
        errors.append(f"{where}: no <title> — this is the LinkedIn/Google headline")
    if not p.metas.get("description"):
        errors.append(f"{where}: no meta description")
    for prop in ("og:title", "og:description"):
        if not p.props.get(prop):
            warns.append(f"{where}: missing {prop} — link previews will look bare")
    if not p.props.get("og:image"):
        warns.append(f"{where}: no og:image — LinkedIn shares will have no thumbnail")
    if not p.lang_seen:
        warns.append(f"{where}: <html> has no lang attribute")
    if p.h1 != 1:
        errors.append(f"{where}: {p.h1} <h1> elements — there should be exactly one")

    last = 0
    for lvl in p.headings:
        if last and lvl > last + 1:
            warns.append(f"{where}: heading jumps h{last} -> h{lvl}; screen readers follow this")
            break
        last = lvl

    for img in p.imgs:
        src = img.get("src", "")
        if not img.get("alt", "").strip():
            errors.append(f"{where}: <img src={src}> has no alt text")
        elif len(img["alt"].split()) < 3:
            warns.append(f"{where}: alt text for {src} is thin — say what it shows technically")
        if not (img.get("width") and img.get("height")):
            warns.append(f"{where}: {src} has no width/height — causes layout shift")
    for vid in p.videos:
        if not vid.get("poster"):
            warns.append(f"{where}: <video src={vid.get('src')}> has no poster frame")

    for href in p.links:
        if href.startswith(("http://", "https://", "mailto:", "tel:", "#")):
            continue
        bare = href.split("#")[0]
        # Root-relative hrefs resolve against the published root, not the page's
        # directory — this is a user Pages site, so "/" is the docs/ root.
        if bare.startswith("/"):
            target = (root / bare.lstrip("/")).resolve()
        else:
            target = (path.parent / bare).resolve()
        if bare and not target.exists():
            errors.append(f"{where}: broken internal link -> {href}")

    body = text_of(raw)
    for pat in PLACEHOLDERS:
        if re.search(pat, body, re.I):
            errors.append(f"{where}: unfilled placeholder matching /{pat}/")
    for w in WEAK_WORDS:
        if w in body.lower():
            warns.append(f'{where}: filler phrase "{w}" — see the copywriting rules')

    words = len(body.split())
    if path.name == "index.html" and words < 250:
        warns.append(f"{where}: only {words} words — likely too thin to be convincing")

    has_mailto = any(l.startswith("mailto:") for l in p.links)
    has_linkedin = any("linkedin.com" in l for l in p.links)
    if path.name == "index.html":
        if not has_mailto:
            errors.append("index.html: no mailto: link — a recruiter cannot contact him")
        if not has_linkedin:
            errors.append("index.html: no LinkedIn link")


def check_weights(root: Path, errors: list, warns: list):
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        kb = f.stat().st_size / 1024
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"} and kb > MAX_IMG_KB:
            warns.append(f"{f.relative_to(root)}: {kb:.0f} KB — run extract_media.py optimize")
    index_assets = sum(f.stat().st_size for f in root.rglob("*")
                       if f.is_file() and f.suffix.lower() != ".pdf")
    mb = index_assets / 1024 / 1024
    if mb > MAX_PAGE_MB * 3:
        warns.append(f"site payload {mb:.1f} MB — heavy for a first visit on mobile")


def check_denylist(root: Path, denylist: Path, errors: list):
    terms = [t.strip() for t in denylist.read_text(encoding="utf-8").splitlines()
             if t.strip() and not t.startswith("#")]
    if not terms:
        return
    for f in root.rglob("*.html"):
        body = f.read_text(encoding="utf-8", errors="ignore").lower()
        for t in terms:
            if t.lower() in body:
                errors.append(f"CONFIDENTIAL: {f.relative_to(root)} contains a denylisted term")
    for f in root.rglob("*"):
        if f.is_file():
            name = f.name.lower()
            for t in terms:
                if t.lower() in name:
                    errors.append(f"CONFIDENTIAL: filename {f.relative_to(root)} contains a denylisted term")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dist", nargs="?", default="dist")
    ap.add_argument("--denylist")
    args = ap.parse_args()

    root = Path(args.dist)
    if not root.exists():
        print(f"! {root} does not exist — run build_site.py first", file=sys.stderr)
        return 1

    errors, warns = [], []
    pages = sorted(root.rglob("*.html"))
    if not pages:
        print("! no HTML pages found", file=sys.stderr)
        return 1
    for page in pages:
        check_page(page, root, errors, warns)
    check_weights(root, errors, warns)

    if args.denylist:
        dl = Path(args.denylist)
        if dl.exists():
            check_denylist(root, dl, errors)
        else:
            warns.append(f"denylist {dl} not found — confidentiality scan skipped")
    else:
        warns.append("no --denylist given — run the confidentiality scan before publishing")

    def dedupe(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    errors, warns = dedupe(errors), dedupe(warns)
    print(f"checked {len(pages)} page(s)\n")
    for w in warns:
        print(f"  warn   {w}")
    for e in errors:
        print(f"  ERROR  {e}")
    print(f"\n{len(errors)} error(s), {len(warns)} warning(s)")
    if not errors:
        print("\nMechanical checks pass. Still to do by hand: read it as the thirty-second\n"
              "recruiter, and get explicit approval on every employer-related sentence.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
