"""
Bundle the site into ONE html file (data and team logos inlined) that opens with a double-click,
no Terminal or Python needed on the other end. It's a snapshot: rerun after each data update.

Usage:  .venv/bin/python make_standalone.py [output.html]
        (default output: site/NFL-EPA.html)
"""
import base64
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SITE = ROOT / "site"


def main() -> None:
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else SITE / "NFL-EPA.html"
    js = (SITE / "data.js").read_text()
    data = json.loads(js[js.index("=") + 1:].strip().rstrip(";"))
    for t in data["teams"]:
        if t.get("logo"):
            png = (SITE / t["logo"]).read_bytes()
            t["logo"] = "data:image/png;base64," + base64.b64encode(png).decode()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = (SITE / "index.html").read_text()
    tag = '<script src="data.js"></script>'
    assert html.count(tag) == 1, "index.html no longer loads data.js the expected way"
    html = html.replace(tag, f"<script>window.EPA_DATA = {blob};</script>")
    out.write_text(html)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB), data built {data['meta']['built']}")


if __name__ == "__main__":
    main()
