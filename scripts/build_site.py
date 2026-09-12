#!/usr/bin/env python3
"""Render content/site.json into a static site in dist/.

Standard library only, no build toolchain — the site must still build in three years
without a dependency archaeology session.

Usage:
    python scripts/build_site.py [--content content/site.json] [--out dist]
                                 [--css assets/site.css] [--media assets/media]

Emits:
    dist/index.html
    dist/projects/<slug>.html   for each project with "deep_dive": true
    dist/site.css               copied from --css
    dist/assets/media/...       copied from --media
    dist/.nojekyll              so GitHub Pages serves files starting with _
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------- small helpers

E = html.escape
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def rich(text: str) -> str:
    """Escape text, then re-enable [label](url) inline links."""
    out = E(text or "")
    return LINK_RE.sub(r'<a href="\2">\1</a>', out)


def paras(text: str, cls: str = "") -> str:
    """Blank-line separated text -> <p> blocks."""
    if not text:
        return ""
    attr = f' class="{cls}"' if cls else ""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    return "\n".join(f"<p{attr}>{rich(b)}</p>" for b in blocks)


def tag(name: str, inner: str, **attrs) -> str:
    bits = "".join(f' {k.rstrip("_").replace("_", "-")}="{E(str(v))}"' for k, v in attrs.items() if v)
    return f"<{name}{bits}>{inner}</{name}>"


def chips(items, cls="chip") -> str:
    if not items:
        return ""
    return "".join(f'<li class="{cls}">{E(i)}</li>' for i in items)


def links_list(items, cls="link-list") -> str:
    if not items:
        return ""
    out = []
    for item in items:
        li_cls = "link-list__item"
        if item.get("primary"):
            li_cls += " link-list__item--primary"
        rel = ' rel="noopener"' if item.get("url", "").startswith("http") else ""
        out.append(
            f'<li class="{li_cls}"><a href="{E(item["url"])}"{rel}>{E(item["label"])}</a></li>'
        )
    return f'<ul class="{cls}">' + "".join(out) + "</ul>"


def media_block(m: dict, lazy: bool = True) -> str:
    src = E(m.get("src", ""))
    caption = m.get("caption", "")
    alt = m.get("alt") or caption
    dims = ""
    if m.get("width"):
        dims += f' width="{E(str(m["width"]))}"'
    if m.get("height"):
        dims += f' height="{E(str(m["height"]))}"'

    if m.get("type") == "video" or src.endswith((".mp4", ".webm")):
        poster = f' poster="{E(m["poster"])}"' if m.get("poster") else ""
        inner = (
            f'<video class="media__video" src="{src}"{poster}{dims} '
            f'autoplay loop muted playsinline preload="metadata" '
            f'aria-label="{E(alt)}"></video>'
        )
    else:
        loading = ' loading="lazy" decoding="async"' if lazy else ""
        inner = f'<img class="media__img" src="{src}" alt="{E(alt)}"{dims}{loading}>'

    cap = f'<figcaption class="media__caption">{rich(caption)}</figcaption>' if caption else ""
    return f'<figure class="media">{inner}{cap}</figure>'


# ---------------------------------------------------------------- page scaffold

def head(title: str, description: str, person: dict, page_url: str, og_image: str | None,
         css_href: str, jsonld: str | None = None) -> str:
    parts = [
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{E(title)}</title>",
        f'<meta name="description" content="{E(description)}">',
        f'<meta property="og:title" content="{E(title)}">',
        f'<meta property="og:description" content="{E(description)}">',
        '<meta property="og:type" content="website">',
        '<meta name="twitter:card" content="summary_large_image">',
    ]
    if page_url:
        parts.append(f'<meta property="og:url" content="{E(page_url)}">')
        parts.append(f'<link rel="canonical" href="{E(page_url)}">')
    if og_image:
        parts.append(f'<meta property="og:image" content="{E(og_image)}">')
    if person.get("favicon"):
        parts.append(f'<link rel="icon" href="{E(person["favicon"])}">')
    parts.append(f'<link rel="stylesheet" href="{E(css_href)}">')
    if jsonld:
        parts.append(f'<script type="application/ld+json">{jsonld}</script>')
    return "\n  ".join(parts)


def person_jsonld(person: dict) -> str:
    same_as = [u for u in (person.get("linkedin"), person.get("github"),
                           person.get("huggingface"), person.get("scholar")) if u]
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


def contact_links(person: dict, primary_first: bool = True) -> list:
    out = []
    if person.get("email"):
        out.append({"label": person["email"], "url": f"mailto:{person['email']}", "primary": True})
    for key, label in (("linkedin", "LinkedIn"), ("github", "GitHub"),
                       ("huggingface", "Hugging Face"), ("scholar", "Scholar")):
        if person.get(key):
            out.append({"label": label, "url": person[key]})
    if person.get("resume"):
        out.append({"label": "Résumé (PDF)", "url": person["resume"], "primary": primary_first})
    return out


# ---------------------------------------------------------------- sections

def render_hero(site: dict) -> str:
    p = site["person"]
    bits = [f'<h1 class="hero__name">{E(p.get("name", ""))}</h1>']
    if p.get("tagline"):
        bits.append(f'<p class="hero__tagline">{rich(p["tagline"])}</p>')
    if site.get("intro"):
        intro = site["intro"]
        text = "\n\n".join(intro) if isinstance(intro, list) else intro
        bits.append(f'<div class="hero__intro">{paras(text)}</div>')
    bits.append(f'<nav class="hero__links" aria-label="Contact">{links_list(contact_links(p))}</nav>')
    hero_media = ""
    if site.get("hero_media"):
        hero_media = f'<div class="hero__media">{media_block(site["hero_media"], lazy=False)}</div>'
    return f'<header class="hero">{"".join(bits)}{hero_media}</header>'


def render_project_card(pr: dict) -> str:
    head_bits = [f'<h3 class="project__title">{E(pr.get("title", ""))}</h3>']
    if pr.get("period"):
        head_bits.append(f'<p class="project__period">{E(pr["period"])}</p>')
    out = [f'<div class="project__head">{"".join(head_bits)}</div>']

    if pr.get("hook"):
        out.append(f'<p class="project__hook">{rich(pr["hook"])}</p>')

    if pr.get("lead_media"):
        out.append(f'<div class="project__media">{media_block(pr["lead_media"])}</div>')

    blocks = []
    for label, key in (("Problem", "problem"), ("Approach", "approach"), ("Outcome", "outcome")):
        if pr.get(key):
            blocks.append(
                '<div class="project__block">'
                f'<h4 class="project__block-label">{label}</h4>'
                f'<div class="project__block-body">{paras(pr[key])}</div>'
                "</div>"
            )
    if blocks:
        out.append(f'<div class="project__blocks">{"".join(blocks)}</div>')

    if pr.get("stack"):
        out.append(f'<ul class="project__stack">{chips(pr["stack"])}</ul>')

    link_items = list(pr.get("links", []))
    if pr.get("deep_dive"):
        link_items = [{"label": "Read the case study", "url": f'projects/{pr["slug"]}.html',
                       "primary": True}] + link_items
    if link_items:
        out.append(f'<div class="project__links">{links_list(link_items)}</div>')

    return f'<article class="project" id="{E(pr.get("slug", ""))}">{"".join(out)}</article>'


def render_role(role: dict) -> str:
    head_bits = [
        f'<h3 class="role__org">{E(role.get("org", ""))}</h3>',
        f'<p class="role__title">{E(role.get("title", ""))}</p>',
    ]
    if role.get("period"):
        head_bits.append(f'<p class="role__period">{E(role["period"])}</p>')
    out = [f'<div class="role__head">{"".join(head_bits)}</div>']
    if role.get("summary"):
        out.append(f'<div class="role__summary">{paras(role["summary"])}</div>')
    if role.get("highlights"):
        items = "".join(f"<li>{rich(h)}</li>" for h in role["highlights"])
        out.append(f'<ul class="role__highlights">{items}</ul>')
    if role.get("stack"):
        out.append(f'<ul class="role__stack">{chips(role["stack"])}</ul>')
    return f'<article class="role">{"".join(out)}</article>'


def render_skills(groups: list) -> str:
    out = []
    for g in groups:
        out.append(
            '<div class="skill-group">'
            f'<h3 class="skill-group__name">{E(g.get("group", ""))}</h3>'
            f'<ul class="skill-group__items">{chips(g.get("items", []))}</ul>'
            "</div>"
        )
    return f'<div class="skills">{"".join(out)}</div>'


def render_edu(items: list) -> str:
    out = []
    for e in items:
        bits = [f'<h3 class="edu__institution">{E(e.get("institution", ""))}</h3>']
        if e.get("degree"):
            bits.append(f'<p class="edu__degree">{E(e["degree"])}</p>')
        if e.get("period"):
            bits.append(f'<p class="edu__period">{E(e["period"])}</p>')
        if e.get("note"):
            bits.append(f'<p class="edu__note">{rich(e["note"])}</p>')
        out.append(f'<article class="edu">{"".join(bits)}</article>')
    return "".join(out)


def section(title: str, body: str, sid: str) -> str:
    if not body:
        return ""
    return (
        f'<section class="section" id="{E(sid)}">'
        f'<h2 class="section__title">{E(title)}</h2>'
        f'<div class="section__body">{body}</div>'
        "</section>"
    )


# ---------------------------------------------------------------- pages

def build_index(site: dict, css_href: str) -> str:
    p = site["person"]
    labels = site.get("section_titles", {})
    projects = site.get("projects", [])

    nav_targets = [("work", labels.get("projects", "Selected work"))]
    if site.get("experience"):
        nav_targets.append(("experience", labels.get("experience", "Experience")))
    if site.get("skills"):
        nav_targets.append(("skills", labels.get("skills", "Skills")))
    nav_targets.append(("contact", labels.get("contact", "Contact")))
    nav = "".join(f'<a class="site-nav__link" href="#{sid}">{E(lbl)}</a>' for sid, lbl in nav_targets)

    body = [
        '<a class="skip-link" href="#work">Skip to work</a>',
        '<header class="site-header">'
        f'<span class="site-header__name">{E(p.get("name", ""))}</span>'
        f'<nav class="site-nav" aria-label="Sections">{nav}</nav>'
        "</header>",
        "<main>",
        render_hero(site),
        section(labels.get("projects", "Selected work"),
                "".join(render_project_card(pr) for pr in projects), "work"),
        section(labels.get("experience", "Experience"),
                "".join(render_role(r) for r in site.get("experience", [])), "experience"),
        section(labels.get("skills", "Skills"),
                render_skills(site.get("skills", [])) if site.get("skills") else "", "skills"),
        section(labels.get("education", "Education"),
                render_edu(site.get("education", [])), "education"),
        section(labels.get("contact", "Contact"),
                f'<div class="contact"><div class="contact__links">'
                f'{links_list(contact_links(p), cls="link-list contact__list")}</div></div>',
                "contact"),
        "</main>",
        f'<footer class="site-footer"><p>{E(site.get("footer", ""))}</p></footer>',
    ]

    desc = site.get("meta_description") or p.get("tagline", "")
    h = head(f'{p.get("name", "")} — {p.get("job_title", "")}'.strip(" —"), desc, p,
             p.get("site_url", ""), site.get("og_image"), css_href, person_jsonld(p))
    return (f'<!doctype html>\n<html lang="{E(site.get("lang", "en"))}">\n<head>\n  {h}\n</head>\n'
            f'<body class="page-index">\n{"".join(body)}\n</body>\n</html>\n')


def build_case(pr: dict, site: dict) -> str:
    p = site["person"]
    out = [
        '<a class="case__back" href="../index.html">Back to all work</a>',
        '<header class="case__hero">',
        f'<h1 class="case__title">{E(pr.get("title", ""))}</h1>',
    ]
    if pr.get("hook"):
        out.append(f'<p class="case__hook">{rich(pr["hook"])}</p>')
    meta = []
    if pr.get("period"):
        meta.append(f'<span class="case__period">{E(pr["period"])}</span>')
    if pr.get("role"):
        meta.append(f'<span class="case__role">{E(pr["role"])}</span>')
    if meta:
        out.append(f'<p class="case__meta">{"".join(meta)}</p>')
    if pr.get("stack"):
        out.append(f'<ul class="project__stack">{chips(pr["stack"])}</ul>')
    if pr.get("links"):
        out.append(f'<div class="project__links">{links_list(pr["links"])}</div>')
    out.append("</header><main>")

    for sec in pr.get("sections", []):
        inner = [f'<h2 class="case__section-title">{E(sec.get("heading", ""))}</h2>',
                 paras(sec.get("body", ""))]
        for m in sec.get("media", []):
            inner.append(media_block(m))
        out.append(f'<section class="case__section">{"".join(inner)}</section>')

    out.append("</main>")
    footer_links = [{"label": "More work", "url": "../index.html#work"}]
    if p.get("email"):
        footer_links.append({"label": "Get in touch", "url": f'mailto:{p["email"]}'})
    out.append(f'<footer class="site-footer">{links_list(footer_links)}</footer>')

    desc = pr.get("hook", "")
    if len(desc) > 180:
        desc = desc[:180].rsplit(" ", 1)[0] + "\u2026"
    og = None
    for m in [pr.get("lead_media")] + [mm for s in pr.get("sections", []) for mm in s.get("media", [])]:
        if m and m.get("type") != "video":
            og = m.get("src")
            break
    h = head(f'{pr.get("title", "")} — {p.get("name", "")}', desc, p, "", og, "../site.css")
    return (f'<!doctype html>\n<html lang="{E(site.get("lang", "en"))}">\n<head>\n  {h}\n</head>\n'
            f'<body class="page-case">\n{"".join(out)}\n</body>\n</html>\n')


# ---------------------------------------------------------------- main

def validate(site: dict) -> list:
    errs = []
    p = site.get("person", {})
    for field in ("name", "email", "linkedin"):
        if not p.get(field):
            errs.append(f"person.{field} is required — the site is useless without a contact path")
    slugs = set()
    for pr in site.get("projects", []):
        if not pr.get("slug"):
            errs.append(f"project {pr.get('title', '?')!r} has no slug")
        elif pr["slug"] in slugs:
            errs.append(f"duplicate project slug: {pr['slug']}")
        else:
            slugs.add(pr["slug"])
        if not pr.get("hook"):
            errs.append(f"project {pr.get('slug', '?')} has no hook — that's the most-read line")
    n = len(site.get("projects", []))
    if n and not 3 <= n <= 6:
        errs.append(f"{n} projects — the curated range is 4-6 (3 acceptable); see structure.md")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", default="content/site.json")
    ap.add_argument("--out", default="dist")
    ap.add_argument("--css", default="assets/site.css")
    ap.add_argument("--media", default="assets/media")
    ap.add_argument("--strict", action="store_true", help="fail on validation warnings")
    args = ap.parse_args()

    site = json.loads(Path(args.content).read_text(encoding="utf-8"))

    errs = validate(site)
    for e in errs:
        print(f"  ! {e}", file=sys.stderr)
    if errs and args.strict:
        return 1

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "projects").mkdir(parents=True, exist_ok=True)

    (out / "index.html").write_text(build_index(site, "site.css"), encoding="utf-8")
    n_case = 0
    for pr in site.get("projects", []):
        if pr.get("deep_dive"):
            (out / "projects" / f'{pr["slug"]}.html').write_text(build_case(pr, site), encoding="utf-8")
            n_case += 1

    css = Path(args.css)
    if css.exists():
        shutil.copy(css, out / "site.css")
    else:
        print(f"  ! no stylesheet at {css} — write one per the design plan", file=sys.stderr)

    media = Path(args.media)
    if media.exists():
        shutil.copytree(media, out / "assets" / "media")

    for extra in ("resume.pdf", "favicon.ico", "favicon.svg", "CNAME"):
        src = Path("assets") / extra
        if src.exists():
            shutil.copy(src, out / extra)

    (out / ".nojekyll").write_text("", encoding="utf-8")

    print(f"built {out}/index.html + {n_case} case study page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
