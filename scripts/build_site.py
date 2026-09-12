#!/usr/bin/env python3
"""Render content/site.json into a static multi-page site.

Standard library only, no build toolchain — the site must still build in three
years without a dependency archaeology session.

    python3 scripts/build_site.py --out docs

Emits:
    docs/index.html              home: hero, videos, featured work, about teaser
    docs/projects.html           every project, filterable by tag
    docs/about.html              bio + experience & education timeline
    docs/projects/<slug>.html    a detail page per project
    docs/site.css, docs/site.js  copied from assets/
    docs/assets/media/...        copied from assets/media/
    docs/.nojekyll               so Pages serves files starting with _

Paths in content are root-relative ("/assets/..."). This is a user Pages site
served at the domain root, so they resolve identically from every page depth.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path

E = html.escape
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")



# ------------------------------------------------- intrinsic image dimensions
#
# Declared width/height let the browser reserve the right box before the image
# arrives. Read from the file header rather than hardcoded, so a replaced asset
# never silently disagrees with its markup. Stdlib only, by design.

_DIM_CACHE: dict = {}


def image_size(src: str, assets: Path) -> tuple:
    """(width, height) for a local /assets/... image, or (None, None)."""
    if not src.startswith("/assets/"):
        return (None, None)
    if src in _DIM_CACHE:
        return _DIM_CACHE[src]
    path = assets / src.split("/assets/", 1)[1]
    dim = (None, None)
    try:
        b = path.read_bytes()
        if b[:8] == b"\x89PNG\r\n\x1a\n":
            dim = (int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big"))
        elif b[:3] == b"GIF":
            dim = (int.from_bytes(b[6:8], "little"), int.from_bytes(b[8:10], "little"))
        elif b[:2] == b"\xff\xd8":
            i = 2
            while i < len(b) - 9:
                if b[i] != 0xFF:
                    i += 1
                    continue
                m = b[i + 1]
                if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                         0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    dim = (int.from_bytes(b[i + 7:i + 9], "big"),
                           int.from_bytes(b[i + 5:i + 7], "big"))
                    break
                if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
                    i += 2
                else:
                    i += 2 + int.from_bytes(b[i + 2:i + 4], "big")
    except Exception:
        pass
    _DIM_CACHE[src] = dim
    return dim


ASSETS = Path("assets")


# ---------------------------------------------------------------- text helpers

def rich(text: str) -> str:
    return LINK_RE.sub(r'<a href="\2">\1</a>', E(text or ""))


def paras(text: str, cls: str = "") -> str:
    if not text:
        return ""
    attr = f' class="{cls}"' if cls else ""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    return "\n".join(f"<p{attr}>{rich(b)}</p>" for b in blocks)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# ---------------------------------------------------------------- page scaffold

NAV = [("/", "Home"), ("/projects.html", "Projects"), ("/about.html", "About")]


def head(title: str, desc: str, person: dict, og_image: str | None,
         canonical: str = "", jsonld: str | None = None) -> str:
    parts = [
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{E(title)}</title>",
        f'<meta name="description" content="{E(desc)}">',
        f'<meta property="og:title" content="{E(title)}">',
        f'<meta property="og:description" content="{E(desc)}">',
        '<meta property="og:type" content="website">',
        '<meta name="twitter:card" content="summary_large_image">',
    ]
    if canonical:
        parts.append(f'<meta property="og:url" content="{E(canonical)}">')
        parts.append(f'<link rel="canonical" href="{E(canonical)}">')
    if og_image:
        parts.append(f'<meta property="og:image" content="{E(og_image)}">')
    if person.get("favicon"):
        parts.append(f'<link rel="icon" href="{E(person["favicon"])}">')
    parts.append('<link rel="stylesheet" href="/site.css">')
    parts.append('<script src="/site.js" defer></script>')
    if jsonld:
        parts.append(f'<script type="application/ld+json">{jsonld}</script>')
    return "\n  ".join(parts)


def person_jsonld(person: dict) -> str:
    same_as = [u for u in (person.get("linkedin"), person.get("github"),
                           person.get("scholar")) if u]
    data = {
        "@context": "https://schema.org",
        "@type": "Person",
        "name": person.get("name", ""),
        "jobTitle": person.get("job_title", ""),
        "url": person.get("site_url", ""),
        "email": f"mailto:{person['email']}" if person.get("email") else "",
        "sameAs": same_as,
    }
    return json.dumps({k: v for k, v in data.items() if v}, ensure_ascii=False)


def site_header(active: str, person: dict) -> str:
    links = []
    for href, lbl in NAV:
        cls = "site-nav__link" + (" is-active" if href == active else "")
        links.append(f'<a class="{cls}" href="{href}">{E(lbl)}</a>')
    if person.get("resume"):
        links.append(f'<a class="site-nav__link" href="{E(person["resume"])}">CV</a>')
    return (
        '<header class="site-header">'
        f'<a class="site-header__name" href="/">{E(person.get("name", ""))}</a>'
        f'<nav class="site-nav" aria-label="Sections">{"".join(links)}</nav>'
        "</header>"
    )


def site_footer(person: dict) -> str:
    bits = []
    if person.get("email"):
        bits.append(f'<a href="mailto:{E(person["email"])}">{E(person["email"])}</a>')
    for key, lbl in (("linkedin", "LinkedIn"), ("github", "GitHub"), ("scholar", "Scholar")):
        if person.get(key):
            bits.append(f'<a href="{E(person[key])}" rel="noopener">{E(lbl)}</a>')
    if person.get("resume"):
        bits.append(f'<a href="{E(person["resume"])}">CV</a>')
    return (
        '<footer class="site-footer">'
        f'<div class="site-footer__links">{"".join(bits)}</div>'
        f'<p class="site-footer__note">{E(person.get("name", ""))} &middot; '
        f'{E(person.get("location", ""))}</p>'
        "</footer>"
    )


def page(body: str, *, title: str, desc: str, person: dict, active: str,
         og_image: str | None = None, canonical: str = "", body_class: str = "",
         jsonld: str | None = None, lang: str = "en") -> str:
    h = head(title, desc, person, og_image, canonical, jsonld)
    return (f'<!doctype html>\n<html lang="{E(lang)}">\n<head>\n  {h}\n</head>\n'
            f'<body class="{E(body_class)}">\n'
            '<a class="skip-link" href="#main">Skip to content</a>\n'
            f'{site_header(active, person)}\n'
            f'<main id="main">{body}</main>\n'
            f'{site_footer(person)}\n</body>\n</html>\n')


# ---------------------------------------------------------------- components

def project_card(pr: dict, *, reveal: bool = True, eager: bool = False) -> str:
    slug = pr.get("slug", "")
    href = f"/projects/{slug}.html"
    thumb = pr.get("thumb")
    tags = pr.get("tags", [])
    data_tags = " ".join(slugify(t) for t in tags)

    if thumb:
        # The first cards are above the fold; lazy-loading them only delays the
        # largest paint.
        load = "" if eager else ' loading="lazy"'
        w, h = image_size(thumb, ASSETS)
        dims = f' width="{w}" height="{h}"' if w else ""
        media = (f'<img class="card__img" src="{E(thumb)}" '
                 f'alt="{E(pr.get("thumb_alt", ""))}"{dims}{load} decoding="async">')
    else:
        # Employer-tier projects carry no imagery by disclosure rule. Give the card
        # a deliberate typographic face rather than a broken-looking gap.
        initials = "".join(w[0] for w in pr.get("title", "?").split()[:2]).upper()
        media = f'<span class="card__mark" aria-hidden="true">{E(initials)}</span>'

    tag_html = "".join(f'<span class="tag">{E(t)}</span>' for t in tags)
    cls = "card reveal" if reveal else "card"
    return (
        f'<article class="{cls}" data-tags="{E(data_tags)}">'
        f'<a class="card__link" href="{href}">'
        f'<span class="card__media">{media}</span>'
        '<span class="card__body">'
        f'<span class="card__period">{E(pr.get("period", ""))}</span>'
        f'<span class="card__title">{E(pr.get("title", ""))}</span>'
        f'<span class="card__blurb">{E(pr.get("blurb", ""))}</span>'
        f'<span class="card__tags">{tag_html}</span>'
        "</span></a></article>"
    )


def video_block(v: dict) -> str:
    """A local clip, or a YouTube id rendered as a click-through facade so the
    page never ships a third-party player it does not need."""
    title = v.get("title", "")
    if v.get("youtube"):
        yid = v["youtube"]
        poster = v.get("poster") or f"https://i.ytimg.com/vi/{yid}/hqdefault.jpg"
        return (
            f'<a class="vid vid--yt" href="https://www.youtube.com/watch?v={E(yid)}" '
            f'rel="noopener">'
            f'<img class="vid__img" src="{E(poster)}" alt="{E(title)}" loading="lazy">'
            '<span class="vid__play" aria-hidden="true"></span>'
            f'<span class="vid__title">{E(title)}</span></a>'
        )
    src = v.get("src", "")
    poster = f' poster="{E(v["poster"])}"' if v.get("poster") else ""
    return (
        '<figure class="vid">'
        f'<video class="vid__video" src="{E(src)}"{poster} autoplay loop muted '
        f'playsinline preload="metadata" aria-label="{E(title)}"></video>'
        f'<figcaption class="vid__title">{E(title)}</figcaption>'
        "</figure>"
    )


def video_placeholder(v: dict) -> str:
    return (
        '<div class="vid vid--empty">'
        f'<span class="vid__slot">{E(v.get("title", "clip"))}</span>'
        f'<span class="vid__hint">{E(v.get("hint", ""))}</span>'
        "</div>"
    )


def media_block(m: dict) -> str:
    src = E(m.get("src", ""))
    alt = m.get("alt") or m.get("caption", "")
    dims = ""
    if m.get("width"):
        dims += f' width="{E(str(m["width"]))}"'
    if m.get("height"):
        dims += f' height="{E(str(m["height"]))}"'
    if not dims:
        w, h = image_size(m.get("src", ""), ASSETS)
        if w:
            dims = f' width="{w}" height="{h}"'
    if m.get("type") == "video" or src.endswith((".mp4", ".webm")):
        poster = f' poster="{E(m["poster"])}"' if m.get("poster") else ""
        inner = (f'<video class="media__video" src="{src}"{poster}{dims} autoplay loop '
                 f'muted playsinline preload="metadata" aria-label="{E(alt)}"></video>')
    else:
        inner = (f'<img class="media__img" src="{src}" alt="{E(alt)}"{dims} '
                 f'loading="lazy" decoding="async">')
    cap = (f'<figcaption class="media__caption">{rich(m["caption"])}</figcaption>'
           if m.get("caption") else "")
    return f'<figure class="media">{inner}{cap}</figure>'


# ---------------------------------------------------------------- pages

def build_home(site: dict) -> str:
    p = site["person"]
    hero = site.get("hero", {})
    projects = site.get("projects", [])
    featured = [pr for pr in projects if pr.get("featured")][:3]

    bits = ['<section class="hero">',
            f'<h1 class="hero__headline">{rich(hero.get("headline", ""))}</h1>']
    if hero.get("sub"):
        bits.append(f'<p class="hero__sub">{rich(hero["sub"])}</p>')
    if hero.get("buttons"):
        bits.append('<div class="hero__links">' + "".join(
            f'<a class="btn{" btn--primary" if b.get("primary") else ""}" '
            f'href="{E(b["url"])}">{E(b["label"])}</a>' for b in hero["buttons"]) + "</div>")
    bits.append("</section>")

    if site.get("videos"):
        cells = [video_block(v) if (v.get("src") or v.get("youtube")) else video_placeholder(v)
                 for v in site["videos"][:4]]
        bits.append(f'<section class="section reveal"><div class="vid-grid">'
                    f'{"".join(cells)}</div></section>')

    bits.append(
        '<section class="section reveal">'
        '<div class="section__head"><h2 class="section__title">Featured projects</h2>'
        '<a class="section__more" href="/projects.html">View all &rarr;</a></div>'
        f'<div class="card-grid">'
        + "".join(project_card(pr, eager=(i < 2))
                   for i, pr in enumerate(featured)) + "</div>"
        "</section>"
    )

    about = site.get("about", {})
    photo = ""
    if p.get("photo"):
        photo = (f'<img class="portrait" src="{E(p["photo"])}" '
                 f'alt="Portrait of {E(p.get("name", ""))}">')
    teaser = about.get("teaser") or (about.get("bio", [""])[0] if about.get("bio") else "")
    bits.append(
        f'<section class="section reveal about-teaser{" about-teaser--photo" if photo else ""}">'
        f'{photo}<div class="about-teaser__text">'
        '<h2 class="section__title">About</h2>'
        f'{paras(teaser)}'
        '<a class="section__more" href="/about.html">More about me &rarr;</a>'
        "</div></section>"
    )

    return page("".join(bits), title=f'{p.get("name", "")} — {p.get("job_title", "")}',
                desc=site.get("meta_description", ""), person=p, active="/",
                og_image=site.get("og_image"), canonical=p.get("site_url", ""),
                body_class="page-home", jsonld=person_jsonld(p))


def build_projects(site: dict) -> str:
    p = site["person"]
    projects = site.get("projects", [])
    tags = []
    for pr in projects:
        for t in pr.get("tags", []):
            if t not in tags:
                tags.append(t)
    tags.sort()

    filters = ['<button class="filter is-on" data-filter="all" type="button">All</button>']
    filters += [f'<button class="filter" data-filter="{E(slugify(t))}" type="button">{E(t)}</button>'
                for t in tags]

    body = (
        '<section class="section">'
        '<h1 class="page-title">Projects</h1>'
        f'<p class="page-lede">{E(site.get("projects_lede", ""))}</p>'
        '<div class="filters" role="group" aria-label="Filter projects by topic">'
        f'{"".join(filters)}</div>'
        '<p class="filter-count" aria-live="polite"></p>'
        f'<div class="card-grid" id="project-grid">'
        + "".join(project_card(pr, reveal=False, eager=(i < 4))
                    for i, pr in enumerate(projects)) + "</div>"
        "</section>"
    )
    return page(body, title=f'Projects — {p.get("name", "")}',
                desc=site.get("projects_lede", ""), person=p, active="/projects.html",
                og_image=site.get("og_image"),
                canonical=p.get("site_url", "").rstrip("/") + "/projects.html",
                body_class="page-projects")


def timeline_entry(e: dict) -> str:
    logo = e.get("logo")
    if logo:
        mark = f'<img class="tl__logo" src="{E(logo)}" alt="">'
    else:
        initials = "".join(w[0] for w in e.get("org", "?").split()[:2]).upper()
        mark = f'<span class="tl__logo tl__logo--mono" aria-hidden="true">{E(initials)}</span>'
    note = f'<p class="tl__note">{rich(e["note"])}</p>' if e.get("note") else ""
    return (
        f'<li class="tl__item reveal" data-kind="{E(e.get("kind", "work"))}">'
        f'{mark}<div class="tl__body">'
        f'<p class="tl__period">{E(e.get("period", ""))}</p>'
        f'<h3 class="tl__role">{E(e.get("role", ""))}</h3>'
        f'<p class="tl__org">{E(e.get("org", ""))}</p>{note}</div></li>'
    )


def build_about(site: dict) -> str:
    p = site["person"]
    about = site.get("about", {})
    photo = ""
    if p.get("photo"):
        photo = (f'<img class="portrait portrait--lg" src="{E(p["photo"])}" '
                 f'alt="Portrait of {E(p.get("name", ""))}">')

    bio = about.get("bio", [])
    bio_html = paras("\n\n".join(bio) if isinstance(bio, list) else bio)

    interests = ""
    if about.get("interests"):
        interests = ('<p class="about__interests"><span>Interested in</span> '
                     + ", ".join(E(i) for i in about["interests"]) + ".</p>")

    tl_html = ""
    if site.get("timeline"):
        tl_html = ('<section class="section">'
                   '<h2 class="section__title">Experience &amp; education</h2>'
                   f'<ol class="tl">{"".join(timeline_entry(e) for e in site["timeline"])}</ol>'
                   "</section>")

    body = ('<section class="section about">'
            '<h1 class="page-title">About me</h1>'
            f'<div class="about__grid{" about__grid--photo" if photo else ""}">{photo}'
            f'<div class="about__text">{bio_html}{interests}</div></div>'
            "</section>"
            f"{tl_html}")
    return page(body, title=f'About — {p.get("name", "")}',
                desc=(bio[0] if bio else site.get("meta_description", "")),
                person=p, active="/about.html", og_image=site.get("og_image"),
                canonical=p.get("site_url", "").rstrip("/") + "/about.html",
                body_class="page-about")


def build_case(pr: dict, site: dict) -> str:
    p = site["person"]
    out = ['<a class="back" href="/projects.html">Back to all projects</a>',
           '<header class="case__hero">',
           f'<p class="case__period">{E(pr.get("period", ""))}</p>',
           f'<h1 class="case__title">{E(pr.get("title", ""))}</h1>']
    if pr.get("hook"):
        out.append(f'<p class="case__hook">{rich(pr["hook"])}</p>')
    if pr.get("tags"):
        out.append('<ul class="case__tags">'
                   + "".join(f'<li class="tag">{E(t)}</li>' for t in pr["tags"]) + "</ul>")
    if pr.get("role"):
        out.append(f'<p class="case__role"><span>My part</span> {rich(pr["role"])}</p>')
    if pr.get("links"):
        out.append('<div class="case__links">' + "".join(
            f'<a class="btn{" btn--primary" if l.get("primary") else ""}" href="{E(l["url"])}"'
            + (' rel="noopener"' if l["url"].startswith("http") else "")
            + f'>{E(l["label"])}</a>' for l in pr["links"]) + "</div>")
    out.append("</header>")

    lead = pr.get("lead_media")
    if not lead and pr.get("thumb"):
        # the card figure doubles as the page's lead image rather than appearing
        # only in the grid the reader has just left
        lead = {"src": pr["thumb"], "alt": pr.get("thumb_alt", ""),
                "caption": pr.get("thumb_caption", "")}
    if lead:
        out.append(f'<div class="case__lead">{media_block(lead)}</div>')

    blocks = []
    for lbl, key in (("Problem", "problem"), ("Approach", "approach"), ("Outcome", "outcome")):
        if pr.get(key):
            blocks.append(f'<div class="pao__row"><h2 class="pao__label">{lbl}</h2>'
                          f'<div class="pao__body">{paras(pr[key])}</div></div>')
    if blocks:
        out.append(f'<section class="section pao">{"".join(blocks)}</section>')

    for sec in pr.get("sections", []):
        inner = [f'<h2 class="case__section-title">{E(sec.get("heading", ""))}</h2>',
                 paras(sec.get("body", ""))]
        for m in sec.get("media", []):
            inner.append(media_block(m))
        out.append(f'<section class="case__section reveal">{"".join(inner)}</section>')

    og = pr.get("thumb")
    if not og:
        for m in [pr.get("lead_media")] + [mm for s in pr.get("sections", [])
                                           for mm in s.get("media", [])]:
            if m and m.get("type") != "video":
                og = m.get("src")
                break
    og = og or site.get("og_image")
    base = p.get("site_url", "").rstrip("/")
    if og and og.startswith("/") and base:
        og = base + og

    return page("".join(out), title=f'{pr.get("title", "")} — {p.get("name", "")}',
                desc=pr.get("blurb") or pr.get("hook", ""), person=p,
                active="/projects.html", og_image=og,
                canonical=base + f'/projects/{pr["slug"]}.html',
                body_class="page-case")


# ---------------------------------------------------------------- main

def validate(site: dict) -> list:
    errs = []
    p = site.get("person", {})
    for f in ("name", "email", "linkedin"):
        if not p.get(f):
            errs.append(f"person.{f} is required — the site is useless without a contact path")
    slugs = set()
    for pr in site.get("projects", []):
        s = pr.get("slug")
        if not s:
            errs.append(f"project {pr.get('title', '?')!r} has no slug")
        elif s in slugs:
            errs.append(f"duplicate project slug: {s}")
        else:
            slugs.add(s)
        if not pr.get("blurb"):
            errs.append(f"project {s} has no blurb — that is the card's most-read line")
    if not [pr for pr in site.get("projects", []) if pr.get("featured")]:
        errs.append("no project is marked featured — the home page needs three")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", default="content/site.json")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    global ASSETS
    ASSETS = Path(args.assets)

    site = json.loads(Path(args.content).read_text(encoding="utf-8"))

    # A portrait that has not been supplied yet must not ship as a broken image.
    # Drop the reference; the pages fall back to a text-only About automatically,
    # and it comes back the moment the file exists.
    photo = site.get("person", {}).get("photo", "")
    if photo.startswith("/"):
        local = Path(args.assets) / photo.split("/assets/", 1)[-1]
        if not local.exists():
            print(f"  . no portrait at {local} — rendering About without a photo",
                  file=sys.stderr)
            site["person"].pop("photo", None)

    errs = validate(site)
    for e in errs:
        print(f"  ! {e}", file=sys.stderr)
    if errs and args.strict:
        return 1

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "projects").mkdir(parents=True, exist_ok=True)

    (out / "index.html").write_text(build_home(site), encoding="utf-8")
    (out / "projects.html").write_text(build_projects(site), encoding="utf-8")
    (out / "about.html").write_text(build_about(site), encoding="utf-8")

    n = 0
    for pr in site.get("projects", []):
        (out / "projects" / f'{pr["slug"]}.html').write_text(build_case(pr, site), encoding="utf-8")
        n += 1

    assets = Path(args.assets)
    for name in ("site.css", "site.js"):
        src = assets / name
        if src.exists():
            shutil.copy(src, out / name)
        else:
            print(f"  ! missing {src}", file=sys.stderr)

    if (assets / "media").exists():
        shutil.copytree(assets / "media", out / "assets" / "media")
    for extra in ("resume.pdf", "favicon.ico", "CNAME"):
        if (assets / extra).exists():
            shutil.copy(assets / extra, out / extra)

    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"built {out}: home + projects + about + {n} project pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
