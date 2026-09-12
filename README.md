# shubhamagr281999.github.io

Source for my personal site: <https://shubhamagr281999.github.io>

Static HTML, no build toolchain. Content lives in `content/site.json`; the design lives
in `assets/site.css`; `docs/` is generated output and is what GitHub Pages serves.

```bash
python scripts/build_site.py            # content/site.json -> docs/
python scripts/check_site.py docs/      # pre-publish checks
```
