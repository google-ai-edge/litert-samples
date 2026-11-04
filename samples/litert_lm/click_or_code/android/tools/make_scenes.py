#!/usr/bin/env python3
"""Draw the example screens of the click-or-code sample: every window, icon and picture here is written for this sample
(no third-party screen), text in DejaVu (Bitstream Vera / DejaVu license, the fonts of matplotlib's bundled copy).

  python make_scenes.py OUT_DIR --fonts DIR_WITH_DEJAVU_TTFS [--only NAME]

For each screen it writes OUT_DIR/<name>.html, OUT_DIR/<name>.png (1920 x 1080, or 1080 x 2400 for a phone screen) and
OUT_DIR/<name>.boxes.json: every element with data-box="..." as [x, y, w, h] in pixels of that PNG. Needs Google Chrome
(headless, for the drawing) and Pillow (to cut the window to the exact size).
"""
import argparse
import json
import os
import subprocess
import sys

from PIL import Image

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

BASE_CSS = """
@font-face{font-family:'DV';src:url('file://%(fonts)s/DejaVuSans.ttf')}
@font-face{font-family:'DV';font-weight:bold;src:url('file://%(fonts)s/DejaVuSans-Bold.ttf')}
@font-face{font-family:'DVM';src:url('file://%(fonts)s/DejaVuSansMono.ttf')}
*{box-sizing:border-box}
html,body{margin:0;padding:0;overflow:hidden;background:#000}
#s{position:relative;width:1920px;height:1080px;font-family:'DV';overflow:hidden;
   background:linear-gradient(135deg,#2b1b3d,#5a3a72 55%%,#c2693f)}
.top{position:absolute;left:0;top:0;right:0;height:32px;background:#1c1c1c;color:#eee;font-size:14px;display:flex;
     justify-content:space-between;align-items:center;padding:0 16px}
.dock{position:absolute;left:0;top:32px;bottom:0;width:70px;background:rgba(28,28,28,.93);display:flex;
      flex-direction:column;align-items:center;gap:12px;padding-top:14px}
.dock i{display:block;width:46px;height:46px;border-radius:10px}
.win{position:absolute;left:110px;top:60px;width:1760px;height:990px;background:#fff;border-radius:10px 10px 0 0;
     box-shadow:0 8px 28px rgba(0,0,0,.55);overflow:hidden;display:flex;flex-direction:column;color:#222}
.tb{height:44px;flex:none;background:#ebebeb;border-bottom:1px solid #c9c9c9;text-align:center;line-height:44px;
    font-weight:bold;font-size:15px;color:#333;position:relative}
.tb .x{position:absolute;right:14px;top:12px;display:flex;gap:10px}
.tb .x b{display:block;width:18px;height:18px;border-radius:50%%;background:#b9b9b9}
.tb .x b:last-child{background:#e2624a}
.menu{height:32px;flex:none;background:#f5f5f5;border-bottom:1px solid #ddd;font-size:14px;line-height:32px;
      padding-left:14px;color:#333;word-spacing:18px}
.body{flex:1;display:flex;min-height:0}
svg{display:block}
"""
DOCK_COLORS = ["#e95420", "#3f8ae0", "#57a773", "#d9b23c", "#8e6bbf", "#4a4a4a"]


def desktop(title, menu, body, extra_css=""):
  dock = "".join(f'<i style="background:{c}"></i>' for c in DOCK_COLORS)
  m = f'<div class="menu">{menu}</div>' if menu else ""
  return (f'<style>{extra_css}</style><div id="s"><div class="top"><span>Activities</span><span>Tue 14:05</span>'
          f'<span>EN &nbsp; 82 %</span></div><div class="dock">{dock}</div><div class="win"><div class="tb">{title}'
          f'<div class="x"><b></b><b></b><b></b></div></div>{m}<div class="body">{body}</div></div></div>')


# ---------------------------------------------------------------- pictures (thumbnails drawn as small scenes)
def art(name, w=210, h=150):
  sky = '<rect width="210" height="150" fill="url(#sky)"/>'
  defs = ('<defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8fc6ee"/>'
          '<stop offset="1" stop-color="#dff1fb"/></linearGradient></defs>')
  parts = {
      "sailboat": sky + '<rect y="100" width="210" height="50" fill="#2f7fb5"/><polygon points="105,28 105,98 58,98" fill="#fff"/>'
                  '<polygon points="110,40 110,98 150,98" fill="#f25c54"/><rect x="104" y="28" width="3" height="76" fill="#5b4636"/>'
                  '<path d="M60 102 h96 l-14 18 h-68 z" fill="#7a4b2a"/>',
      "dog": '<rect width="210" height="150" fill="#b7dd98"/><ellipse cx="108" cy="104" rx="52" ry="26" fill="#a9682f"/>'
             '<circle cx="66" cy="74" r="26" fill="#a9682f"/><ellipse cx="50" cy="60" rx="9" ry="20" fill="#6e3f17"/>'
             '<ellipse cx="82" cy="58" rx="9" ry="20" fill="#6e3f17"/><circle cx="58" cy="72" r="3.5" fill="#111"/>'
             '<circle cx="74" cy="72" r="3.5" fill="#111"/><ellipse cx="66" cy="84" rx="6" ry="4.5" fill="#111"/>'
             '<rect x="76" y="118" width="11" height="26" fill="#a9682f"/><rect x="130" y="118" width="11" height="26" fill="#a9682f"/>'
             '<path d="M158 96 q24 -22 26 -40" stroke="#a9682f" stroke-width="9" fill="none"/>',
      "mountain": sky + '<polygon points="10,132 78,36 146,132" fill="#6c7a89"/><polygon points="78,36 60,62 96,62" fill="#fff"/>'
                  '<polygon points="90,132 150,62 208,132" fill="#55626f"/><rect y="130" width="210" height="20" fill="#5f9e5a"/>',
      "city": '<rect width="210" height="150" fill="#27304d"/><rect x="18" y="58" width="36" height="92" fill="#454e6e"/>'
              '<rect x="62" y="30" width="40" height="120" fill="#5a6488"/><rect x="110" y="72" width="34" height="78" fill="#454e6e"/>'
              '<rect x="150" y="46" width="44" height="104" fill="#5a6488"/>'
              + "".join(f'<rect x="{x}" y="{y}" width="7" height="9" fill="#f6d55c"/>' for x, y in
                        ((26, 70), (40, 70), (26, 92), (70, 44), (86, 44), (70, 70), (86, 92), (118, 86), (160, 60), (176, 60), (160, 88), (176, 116))),
      "bicycle": '<rect width="210" height="150" fill="#f0e6d2"/><circle cx="62" cy="102" r="30" fill="none" stroke="#333" stroke-width="5"/>'
                 '<circle cx="150" cy="102" r="30" fill="none" stroke="#333" stroke-width="5"/>'
                 '<path d="M62 102 L92 62 L132 62 L150 102 M92 62 L108 102 L62 102 M108 102 L132 62" stroke="#c0392b" stroke-width="5" fill="none"/>'
                 '<rect x="80" y="50" width="26" height="6" fill="#333"/><path d="M132 62 l-3 -14 h16" stroke="#333" stroke-width="5" fill="none"/>',
      "flower": '<rect width="210" height="150" fill="#cfe8c4"/><rect x="102" y="82" width="6" height="66" fill="#3f8f45"/>'
                + "".join(f'<circle cx="{105 + 22 * dx}" cy="{62 + 22 * dy}" r="17" fill="#e97bb1"/>' for dx, dy in
                          ((1, 0), (-1, 0), (0, 1), (0, -1), (.7, .7), (-.7, .7), (.7, -.7), (-.7, -.7)))
                + '<circle cx="105" cy="62" r="15" fill="#f4c542"/>',
      "car": sky + '<rect y="118" width="210" height="32" fill="#777"/><path d="M28 112 l18 -32 h86 l28 32 h22 v16 h-164 v-16 z" fill="#d7263d"/>'
             '<polygon points="56,84 84,84 84,108 42,108" fill="#cfe7f7"/><polygon points="92,84 128,84 148,108 92,108" fill="#cfe7f7"/>'
             '<circle cx="62" cy="128" r="14" fill="#222"/><circle cx="150" cy="128" r="14" fill="#222"/>',
      "cat": '<rect width="210" height="150" fill="#e6dff2"/><ellipse cx="108" cy="110" rx="42" ry="36" fill="#555"/>'
             '<circle cx="108" cy="62" r="30" fill="#555"/><polygon points="82,44 84,12 106,36" fill="#555"/><polygon points="134,44 132,12 110,36" fill="#555"/>'
             '<ellipse cx="96" cy="60" rx="5" ry="7" fill="#d3e86a"/><ellipse cx="120" cy="60" rx="5" ry="7" fill="#d3e86a"/>'
             '<polygon points="108,70 103,76 113,76" fill="#f2a0a0"/>',
      "beach": sky + '<rect y="100" width="210" height="50" fill="#f1dca7"/><rect y="88" width="210" height="16" fill="#3a9bc1"/>'
               '<circle cx="176" cy="30" r="16" fill="#ffd34d"/><rect x="88" y="52" width="5" height="78" fill="#6b4a2b"/>'
               '<path d="M42 62 a48 44 0 0 1 98 0 z" fill="#f28b30"/>',
      "lighthouse": sky + '<rect y="118" width="210" height="32" fill="#3a78a8"/><polygon points="120,126 96,126 102,46 114,46" fill="#fff"/>'
                    '<rect x="100" y="66" width="17" height="12" fill="#d7263d"/><rect x="98" y="94" width="21" height="12" fill="#d7263d"/>'
                    '<rect x="98" y="34" width="20" height="14" fill="#333"/><polygon points="96,34 120,34 108,22" fill="#d7263d"/>'
                    '<ellipse cx="108" cy="130" rx="44" ry="9" fill="#6a6a6a"/>',
      "coffee": '<rect width="210" height="150" fill="#d9c2a6"/><path d="M62 54 h78 v46 a30 30 0 0 1 -30 30 h-18 a30 30 0 0 1 -30 -30 z" fill="#fff"/>'
                '<path d="M140 66 h14 a14 14 0 0 1 0 30 h-14" fill="none" stroke="#fff" stroke-width="7"/><ellipse cx="101" cy="56" rx="39" ry="7" fill="#5b3a21"/>'
                '<ellipse cx="102" cy="136" rx="62" ry="7" fill="#c9ad8c"/>',
      "tree": '<rect width="210" height="150" fill="#fbe9c9"/><rect x="96" y="84" width="18" height="62" fill="#7a4b2a"/><circle cx="105" cy="58" r="42" fill="#3d9148"/>'
              '<circle cx="72" cy="76" r="26" fill="#357f3f"/><circle cx="138" cy="76" r="26" fill="#357f3f"/>',
  }
  return f'<svg width="{w}" height="{h}" viewBox="0 0 210 150">{defs}{parts[name]}</svg>'


# ---------------------------------------------------------------- screens
def files():
  css = """
  .side{width:230px;background:#f2f2f2;border-right:1px solid #ddd;padding:14px 0;font-size:15px}
  .side div{padding:9px 22px}.side .on{background:#d7e6f7}
  .main{flex:1;display:flex;flex-direction:column}
  .path{height:46px;flex:none;border-bottom:1px solid #ddd;display:flex;align-items:center;padding:0 16px;font-size:15px;gap:8px;background:#fafafa}
  .path span{background:#e8e8e8;border-radius:5px;padding:5px 10px}
  .grid{flex:1;display:grid;grid-template-columns:repeat(6,1fr);gap:14px 8px;padding:18px 14px;align-content:start}
  .cell{display:flex;flex-direction:column;align-items:center;font-size:14px;gap:8px;padding:6px 0}
  .btn{margin-left:auto;display:flex;gap:6px}
  .ib{width:36px;height:32px;border:1px solid #ccc;border-radius:5px;background:#fff;display:flex;align-items:center;justify-content:center}
  .status{height:30px;flex:none;border-top:1px solid #ddd;background:#f5f5f5;font-size:13px;line-height:30px;padding-left:14px}
  """
  names = ["lighthouse", "coffee", "city", "dog", "mountain", "bicycle", "sailboat", "tree", "flower", "car", "cat", "beach"]
  cells = ""
  for i, n in enumerate(names):
    label = f"IMG_{2041 + i}.JPG"
    cells += f'<div class="cell" data-box="thumb_{n}">{art(n)}<div data-box="label_{n}">{label}</div></div>'
  grid_icon = '<svg width="18" height="18"><rect x="1" y="1" width="7" height="7" fill="#555"/><rect x="10" y="1" width="7" height="7" fill="#555"/><rect x="1" y="10" width="7" height="7" fill="#555"/><rect x="10" y="10" width="7" height="7" fill="#555"/></svg>'
  list_icon = '<svg width="18" height="18"><rect x="1" y="2" width="16" height="3" fill="#555"/><rect x="1" y="8" width="16" height="3" fill="#555"/><rect x="1" y="14" width="16" height="3" fill="#555"/></svg>'
  body = (f'<div class="side"><div>Home</div><div>Documents</div><div>Downloads</div><div class="on">Pictures</div><div>Music</div><div>Trash</div></div>'
          f'<div class="main"><div class="path"><span>home</span>/<span>user</span>/<span>Pictures</span>/<span>trip</span>'
          f'<div class="btn"><div class="ib" data-box="btn_list">{list_icon}</div><div class="ib" data-box="btn_grid">{grid_icon}</div></div></div>'
          f'<div class="grid">{cells}</div><div class="status">12 items, 41.8 MB &nbsp;&nbsp; /home/user/Pictures/trip</div></div>')
  return desktop("trip - Files", "", body, css)


def sheet():
  css = """
  .tools{height:44px;flex:none;background:#f5f5f5;border-bottom:1px solid #ddd;display:flex;align-items:center;gap:6px;padding-left:12px}
  .tools b{display:flex;width:32px;height:32px;border:1px solid #d0d0d0;border-radius:4px;background:#fff;align-items:center;justify-content:center;font-size:14px}
  .fx{height:32px;flex:none;background:#fff;border-bottom:1px solid #ddd;font-size:14px;line-height:32px;padding-left:12px;font-family:'DVM'}
  .col{display:flex;flex-direction:column;flex:1;min-width:0}
  table{border-collapse:collapse;font-size:15px;font-family:'DV'}
  td,th{border:1px solid #d4d4d4;height:34px;width:132px;padding:0 8px;text-align:right}
  th{background:#efefef;font-weight:normal;text-align:center;color:#555}
  td.l{text-align:left}th.r{width:44px}
  tr.hi td{background:#fff27a}
  .area{flex:1;display:flex;min-height:0}
  .chart{width:560px;margin:28px;padding:18px;border:1px solid #bbb;background:#fff;height:430px;display:flex;flex-direction:column}
  .chart h4{margin:0 0 14px;font-size:15px;font-weight:normal;text-align:center;color:#333}
  .bars{flex:1;display:flex;align-items:flex-end;gap:26px;padding:0 18px;border-bottom:2px solid #666;border-left:2px solid #666}
  .bars i{display:block;flex:1;background:#4a86c8}
  .xl{display:flex;gap:26px;padding:6px 18px 0 20px;font-size:13px;color:#444}.xl span{flex:1;text-align:center}
  .tabs{height:34px;flex:none;background:#efefef;border-top:1px solid #ccc;display:flex;font-size:14px}
  .tabs div{padding:8px 22px;border-right:1px solid #ccc}.tabs .on{background:#fff;font-weight:bold}
  """
  rows = [("Rent", 1200, 1200, 1200), ("Payroll", 8450, 8600, 8700), ("Software", 640, 710, 695), ("Travel", 320, 915, 410),
          ("Marketing", 2100, 1850, 2400), ("Utilities", 280, 310, 295), ("Insurance", 450, 450, 450), ("Supplies", 190, 240, 165),
          ("Training", 500, 0, 750), ("Misc", 75, 130, 60)]
  head = '<tr><th class="r"></th>' + "".join(f"<th>{c}</th>" for c in "ABCDE") + "</tr>"
  trs = '<tr><th class="r">1</th><td class="l" colspan="1">Budget</td><td></td><td></td><td></td><td></td></tr>'
  trs += '<tr><th class="r">2</th><td class="l"><b>Item</b></td><td><b>Oct</b></td><td><b>Nov</b></td><td><b>Dec</b></td><td><b>Total</b></td></tr>'
  for i, (n, a, b, c) in enumerate(rows):
    hi = ' class="hi"' if n == "Travel" else ""
    trs += f'<tr{hi}><th class="r">{i + 3}</th><td class="l">{n}</td><td>{a:,}</td><td>{b:,}</td><td>{c:,}</td><td>{a + b + c:,}</td></tr>'
  for k in range(13, 20):
    trs += f'<tr><th class="r">{k}</th><td></td><td></td><td></td><td></td><td></td></tr>'
  bars = [("Oct", 54), ("Nov", 92), ("Dec", 71), ("Jan", 38)]
  bar_html = "".join(f'<i data-box="bar_{n}" style="height:{p}%"></i>' for n, p in bars)
  tools = "".join(f"<b>{t}</b>" for t in ("&#128190;", "&#9112;", "&#8630;", "&#8631;", "<u>B</u>", "<i>I</i>", "&#9776;", "%", "&#931;"))
  body = (f'<div class="col"><div class="tools">{tools}</div><div class="fx">A1 &nbsp; | &nbsp; Budget</div>'
          f'<div class="area"><div><table>{head}{trs}</table></div>'
          f'<div class="chart"><h4>Travel by month</h4><div class="bars">{bar_html}</div>'
          f'<div class="xl">{"".join(f"<span>{n}</span>" for n, _ in bars)}</div></div></div>'
          f'<div class="tabs"><div class="on">Summary</div><div data-box="tab_forecast">Forecast</div><div>Notes</div></div></div>')
  return desktop("/home/user/Documents/finance/budget.xlsx - Sheets", "File Edit View Insert Format Data Tools Help", body, css)


def tool_svg(kind):
  return {
      "brush": '<path d="M20 4 L8 18 l-3 7 l7 -3 L24 8 z" fill="#c0702c"/>',
      "eraser": '<path d="M6 19 L16 7 l9 7 L15 26 H10 z" fill="#ef8fa3"/><path d="M6 19 l9 7" stroke="#333" stroke-width="2"/>',
      "text": '<text x="7" y="24" font-size="24" font-weight="bold" fill="#333" font-family="DV">T</text>',
      "bucket": '<path d="M7 13 l9 -8 l10 10 l-9 9 z" fill="#3f8ae0"/><circle cx="26" cy="24" r="3" fill="#3f8ae0"/>',
      "crop": '<path d="M9 3 V23 H29 M3 9 H23 V29" stroke="#333" stroke-width="3" fill="none"/>',
      "move": '<path d="M16 3 l5 6 h-10 z M16 29 l5 -6 h-10 z M3 16 l6 -5 v10 z M29 16 l-6 -5 v10 z" fill="#333"/><rect x="14.5" y="8" width="3" height="16" fill="#333"/><rect x="8" y="14.5" width="16" height="3" fill="#333"/>',
      "zoom": '<circle cx="13" cy="13" r="8" fill="none" stroke="#333" stroke-width="3"/><path d="M19 19 L28 28" stroke="#333" stroke-width="4"/>',
      "dropper": '<path d="M24 4 l4 4 l-5 5 l-2 -2 l-11 11 l-5 2 l2 -5 l11 -11 l-2 -2 z" fill="#555"/>',
      "rect": '<rect x="5" y="8" width="22" height="16" fill="none" stroke="#333" stroke-width="2" stroke-dasharray="4 3"/>',
      "ellipse": '<ellipse cx="16" cy="16" rx="11" ry="8" fill="none" stroke="#333" stroke-width="2" stroke-dasharray="4 3"/>',
      "line": '<path d="M5 27 L27 5" stroke="#333" stroke-width="3"/>',
      "gradient": '<defs><linearGradient id="g"><stop offset="0" stop-color="#222"/><stop offset="1" stop-color="#eee"/></linearGradient></defs><rect x="4" y="9" width="24" height="14" fill="url(#g)"/>',
  }[kind]


def editor():
  css = """
  .tb2{width:118px;background:#3b3b3b;display:grid;grid-template-columns:repeat(2,50px);gap:8px;padding:12px 6px;align-content:start;justify-content:center}
  .tool{width:50px;height:50px;background:#e9e9e9;border-radius:6px;display:flex;align-items:center;justify-content:center}
  .canvas{flex:1;background:#555;display:flex;align-items:center;justify-content:center}
  .right{width:360px;background:#3b3b3b;color:#eee;font-size:15px;display:flex;flex-direction:column}
  .right h4{margin:0;padding:12px 14px;font-size:14px;background:#2f2f2f;font-weight:normal}
  .lay{display:flex;align-items:center;gap:12px;padding:10px 12px;border-bottom:1px solid #2a2a2a}
  .lay.on{background:#55708f}
  .eye{width:40px;height:40px;background:#4a4a4a;border-radius:6px;display:flex;align-items:center;justify-content:center}
  .th{width:72px;height:54px;background:#fff;display:flex;align-items:center;justify-content:center}
  .small{padding:10px 14px;font-size:13px;color:#ccc}
  """
  eye = '<svg width="28" height="18"><path d="M1 9 Q14 -5 27 9 Q14 23 1 9 z" fill="none" stroke="#eee" stroke-width="2"/><circle cx="14" cy="9" r="4" fill="#eee"/></svg>'
  thumbs = [("Layer 4", '<svg width="48" height="48"><polygon points="24,4 46,44 2,44" fill="#3fa34d"/></svg>', "tri"),
            ("Layer 3", '<svg width="48" height="48"><rect x="6" y="6" width="36" height="36" fill="#2d6cdf"/></svg>', "sq"),
            ("Layer 2", '<svg width="48" height="48"><circle cx="24" cy="24" r="20" fill="#e0312d"/></svg>', "cir"),
            ("Layer 1", '<svg width="60" height="48"><rect width="60" height="48" fill="#f3e7c7"/></svg>', "bg")]
  rows = ""
  for i, (name, svg, key) in enumerate(thumbs):
    on = " on" if i == 0 else ""
    rows += (f'<div class="lay{on}" data-box="row_{key}"><div class="eye" data-box="eye_{key}">{eye}</div>'
             f'<div class="th" data-box="thumb_{key}">{svg}</div><div>{name}</div></div>')
  tools = "".join(f'<div class="tool" data-box="tool_{t}"><svg width="32" height="32">{tool_svg(t)}</svg></div>' for t in
                  ("move", "rect", "ellipse", "crop", "brush", "eraser", "bucket", "gradient", "line", "text", "dropper", "zoom"))
  canvas = ('<svg width="760" height="760"><rect width="760" height="760" fill="#f3e7c7"/><circle cx="240" cy="290" r="150" fill="#e0312d"/>'
            '<rect x="400" y="230" width="240" height="240" fill="#2d6cdf"/><polygon points="370,720 560,440 730,720" fill="#3fa34d"/></svg>')
  body = (f'<div class="tb2">{tools}</div><div class="canvas">{canvas}</div>'
          f'<div class="right"><h4>Layers</h4>{rows}<div class="small">Normal &nbsp; Opacity 100 %</div></div>')
  return desktop("/home/user/Projects/poster/poster.xcf - Image Editor", "File Edit Select View Image Layer Colors Tools Filters Windows Help", body, css)


def settings():
  css = """
  .side{width:330px;background:#f2f2f2;border-right:1px solid #ddd;padding:12px 0;font-size:16px}
  .side div{padding:12px 24px}.side .on{background:#d7e6f7}
  .main{flex:1;padding:34px 80px;background:#fafafa}
  .main h2{margin:0 0 24px;font-size:24px;font-weight:normal}
  .row{display:flex;align-items:center;justify-content:space-between;background:#fff;border:1px solid #ddd;padding:20px 26px;font-size:17px;margin-bottom:-1px}
  .row small{display:block;color:#777;font-size:14px;margin-top:4px}
  .sw{width:64px;height:32px;border-radius:16px;position:relative}
  .sw i{position:absolute;top:3px;width:26px;height:26px;border-radius:50%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.4)}
  .sw.on{background:#3f8ae0}.sw.on i{left:35px}.sw.off{background:#bbb}.sw.off i{left:3px}
  """
  rows = [("Wi-Fi", "Connected to home-5G", "wifi", True), ("Bluetooth", "Visible as user-laptop", "bluetooth", True),
          ("Airplane mode", "Turns off all wireless connections", "airplane", False), ("Night light", "Warmer colors after sunset", "night", False),
          ("Do not disturb", "Hide notifications", "dnd", False), ("Automatic updates", "Download updates when available", "updates", True)]
  html = ""
  for t, s, k, on in rows:
    html += (f'<div class="row" data-box="row_{k}"><div>{t}<small>{s}</small></div>'
             f'<div class="sw {"on" if on else "off"}" data-box="sw_{k}"><i></i></div></div>')
  side = "".join(f'<div class="{"on" if n == "Connections" else ""}">{n}</div>' for n in
                 ("Connections", "Display", "Sound", "Notifications", "Power", "Privacy", "Users", "About"))
  body = f'<div class="side">{side}</div><div class="main"><h2>Connections</h2>{html}</div>'
  return desktop("Settings", "", body, css)


def terminal():
  css = """
  .term{flex:1;background:#141820;color:#d6dde6;font-family:'DVM';font-size:17px;line-height:27px;padding:18px 22px;white-space:pre}
  .g{color:#7fd17f}.r{color:#ff7b72}.b{color:#7aa7e0}
  """
  text = ('<span class="g">user@laptop</span>:<span class="b">~/Projects/report</span>$ ls\n'
          'build_report.py  data.csv  out  README.md\n'
          '<span class="g">user@laptop</span>:<span class="b">~/Projects/report</span>$ python3 build_report.py\n'
          'Traceback (most recent call last):\n'
          '  File "/home/user/Projects/report/build_report.py", line 4, in &lt;module&gt;\n'
          '    import yaml\n'
          '<span class="r">ModuleNotFoundError: No module named \'yaml\'</span>\n'
          '<span class="g">user@laptop</span>:<span class="b">~/Projects/report</span>$ <span style="background:#d6dde6">&nbsp;</span>')
  return desktop("user@laptop: ~/Projects/report", "", f'<div class="term">{text}</div>', css)


PHONE_CSS = """
#s{position:relative;width:1080px;height:2400px;font-family:'DV';overflow:hidden;background:#fbfbfb;color:#1c1c1c}
.st{height:84px;display:flex;justify-content:space-between;align-items:center;padding:0 44px;font-size:34px;color:#333}
.ab{height:150px;display:flex;align-items:center;padding:0 44px;font-size:52px}
.nav{position:absolute;left:0;right:0;bottom:0;height:90px;display:flex;align-items:center;justify-content:center}
.nav i{display:block;width:320px;height:10px;border-radius:5px;background:#888}
"""


def phone(body, css=""):
  return f'<style>{PHONE_CSS}{css}</style><div id="s"><div class="st"><span>14:05</span><span>5G &nbsp; 82 %</span></div>{body}<div class="nav"><i></i></div></div>'


def phone_settings():
  css = """
  .it{display:flex;align-items:center;gap:40px;padding:34px 44px;font-size:42px}
  .it small{display:block;font-size:32px;color:#6d6d6d;margin-top:8px}
  .ic{width:84px;height:84px;border-radius:50%;flex:none}
  .search{margin:0 44px 24px;height:100px;border-radius:50px;background:#eceff1;font-size:36px;line-height:100px;padding-left:44px;color:#666}
  """
  rows = [("Network &amp; internet", "Wi-Fi, mobile, hotspot", "#3f8ae0", "network"), ("Connected devices", "Bluetooth, pairing", "#57a773", "devices"),
          ("Apps", "Recent apps, default apps", "#d9792c", "apps"), ("Notifications", "History, conversations", "#c94f7c", "notifications"),
          ("Battery", "82 %", "#6aa84f", "battery"), ("Storage", "63 % used", "#8e6bbf", "storage"),
          ("Sound &amp; vibration", "Volume, haptics", "#2aa6a6", "sound"), ("Display", "Dark theme, font size", "#d9b23c", "display"),
          ("Privacy", "Permissions, account activity", "#5c7cbd", "privacy"), ("Location", "On", "#c0504d", "location")]
  html = "".join(f'<div class="it" data-box="row_{k}"><div class="ic" style="background:{c}"></div><div>{t}<small>{s}</small></div></div>'
                 for t, s, c, k in rows)
  return phone(f'<div class="ab">Settings</div><div class="search">Search settings</div>{html}', css)


def phone_files():
  css = """
  .crumb{padding:0 44px 20px;font-size:34px;color:#555}
  .it{display:flex;align-items:center;gap:32px;padding:26px 44px;font-size:38px;border-bottom:1px solid #e6e6e6}
  .it small{display:block;font-size:30px;color:#6d6d6d;margin-top:6px}
  .ic{width:96px;height:96px;border-radius:14px;flex:none;color:#fff;font-size:26px;display:flex;align-items:center;justify-content:center;font-weight:bold}
  """
  files_ = [("maps_offline_1.apk", "APK", "#3ddc84", "84 MB, Oct 28"), ("boarding_pass.pdf", "PDF", "#e0443e", "412 kB, Oct 27"), ("calculator_old.apk", "APK", "#3ddc84", "12 MB, Oct 21"),
            ("receipt_0931.jpg", "JPG", "#5c7cbd", "2.1 MB, Oct 19"), ("notes_export.txt", "TXT", "#888888", "18 kB, Oct 18"), ("reader_beta.apk", "APK", "#3ddc84", "47 MB, Oct 11"),
            ("lease_2026.pdf", "PDF", "#e0443e", "1.3 MB, Oct 3"), ("photo_backup.zip", "ZIP", "#d9b23c", "640 MB, Sep 30"), ("game_patch.apk", "APK", "#3ddc84", "212 MB, Sep 24"),
            ("ticket_0812.pdf", "PDF", "#e0443e", "96 kB, Aug 12")]
  html = "".join(f'<div class="it"><div class="ic" style="background:{c}">{e}</div><div>{n}<small>{s}</small></div></div>' for n, e, c, s in files_)
  return phone(f'<div class="ab">Download</div><div class="crumb">Internal storage &gt; Download</div>{html}', css)


def phone_home():
  css = """
  .g{display:grid;grid-template-columns:repeat(4,1fr);gap:70px 0;padding:560px 40px 0}
  .a{display:flex;flex-direction:column;align-items:center;gap:18px;font-size:34px;color:#fff}
  .a svg{border-radius:28px}
  #s{background:linear-gradient(170deg,#23395b,#406e8e 55%,#8ea8c3)}
  .st,.nav i{color:#fff}
  """
  def ic(c, inner):
    return f'<svg width="156" height="156"><rect width="156" height="156" fill="{c}"/>{inner}</svg>'
  apps = [("Camera", "camera", ic("#444", '<rect x="28" y="50" width="100" height="70" rx="10" fill="#ddd"/><rect x="58" y="36" width="40" height="18" fill="#ddd"/><circle cx="78" cy="85" r="24" fill="#444"/><circle cx="78" cy="85" r="15" fill="#6ab0e0"/>')),
          ("Clock", "clock", ic("#fff", '<circle cx="78" cy="78" r="52" fill="none" stroke="#333" stroke-width="7"/><path d="M78 78 V44 M78 78 L104 92" stroke="#333" stroke-width="7"/>')),
          ("Calendar", "calendar", ic("#e8eaf6", '<rect x="28" y="36" width="100" height="90" fill="#fff" stroke="#3f51b5" stroke-width="5"/><rect x="28" y="36" width="100" height="24" fill="#3f51b5"/><text x="60" y="108" font-size="42" font-family="DV" fill="#333">4</text>')),
          ("Notes", "notes", ic("#f6d55c", '<path d="M44 50 H112 M44 76 H112 M44 102 H90" stroke="#6b5a12" stroke-width="7"/>')),
          ("Phone", "phone", ic("#2e9f5b", '<path d="M50 36 q-14 8 -8 30 q14 44 52 58 q18 6 26 -8 l-20 -24 l-14 8 q-20 -10 -28 -30 l10 -12 z" fill="#fff"/>')),
          ("Messages", "messages", ic("#3f8ae0", '<path d="M30 42 H126 V104 H70 L46 126 V104 H30 z" fill="#fff"/>')),
          ("Music", "music", ic("#c94f7c", '<path d="M64 42 L116 32 V100 a16 12 0 1 1 -10 -11 V54 L74 60 V110 a16 12 0 1 1 -10 -11 z" fill="#fff"/>')),
          ("Maps", "maps", ic("#b7e0a2", '<path d="M78 30 a30 30 0 0 1 30 30 q0 22 -30 66 q-30 -44 -30 -66 a30 30 0 0 1 30 -30 z" fill="#e0443e"/><circle cx="78" cy="60" r="11" fill="#fff"/>')),
          ]
  html = "".join(f'<div class="a" data-box="app_{k}">{svg}<div>{n}</div></div>' for n, k, svg in apps)
  return phone(f'<div class="g">{html}</div>', css)


def mail():
  css = """
  .side{width:250px;background:#f2f2f2;border-right:1px solid #ddd;padding:12px 0;font-size:15px}
  .side div{padding:10px 22px}.side .on{background:#d7e6f7}
  .list{width:620px;border-right:1px solid #ddd;font-size:14px;overflow:hidden}
  .m{padding:14px 18px;border-bottom:1px solid #e8e8e8}.m b{display:block;font-size:15px}.m span{color:#666}
  .read{flex:1;padding:30px;font-size:16px;line-height:26px}
  .read h3{margin:0 0 8px;font-size:21px;font-weight:normal}
  """
  mails = [("Tomás Herrera", "Team offsite dates", "Can everyone make the 18th?", "herrera"), ("Noor Haddad", "Invoice 2231 is overdue", "Please see the attached reminder.", "noor"),
           ("Priya Raman", "Lunch on Thursday?", "The new place on 4th looks good.", "priya_lunch"), ("Build bot", "Nightly build failed", "report-service: 2 tests failed", "bot"),
           ("Priya Raman", "Invoice 2240, corrected total", "Here is the invoice with the right total.", "priya_invoice"), ("Lena Fischer", "Slides for Monday", "Draft attached, comments welcome.", "lena"),
           ("Omar Sidibe", "Printer on floor 3", "It is back, toner replaced.", "omar"), ("Noor Haddad", "Parking permits", "Renewal forms are due Friday.", "noor2"),
           ("IT Desk", "Password expires soon", "Change it within 5 days.", "it"), ("Lena Fischer", "Re: Q4 numbers", "Thanks, those look right.", "lena2")]
  rows = "".join(f'<div class="m" data-box="mail_{k}"><b>{a}</b>{s}<br><span>{p}</span></div>' for a, s, p, k in mails)
  side = "".join(f'<div class="{"on" if n.startswith("Inbox") else ""}">{n}</div>' for n in ("Inbox (10)", "Starred", "Sent", "Drafts", "Spam", "Trash"))
  body = (f'<div class="side">{side}</div><div class="list">{rows}</div><div class="read"><h3>Team offsite dates</h3>Tomás Herrera, 09:12<br><br>'
          'Hi all,<br>Can everyone make the 18th? We can book the room by the river if we confirm this week.<br><br>Tomás</div>')
  return desktop("Inbox - Mail", "", body, css)


def browser():
  css = """
  .col{flex:1;display:flex;flex-direction:column}
  .tabs{height:44px;flex:none;background:#dfe1e5;display:flex;align-items:flex-end;padding:0 8px;gap:2px;font-size:14px}
  .tab{width:250px;height:38px;background:#eef0f2;border-radius:8px 8px 0 0;display:flex;align-items:center;justify-content:space-between;padding:0 10px 0 14px}
  .tab.on{background:#fff}.xb{width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-size:17px;color:#555}
  .url{height:44px;flex:none;background:#fff;border-bottom:1px solid #ddd;display:flex;align-items:center;padding:0 14px;font-size:15px;gap:14px;color:#444}
  .url span{flex:1;background:#f1f3f4;border-radius:16px;padding:6px 16px}
  .pg{flex:1;padding:36px 200px;font-size:18px;line-height:30px}
  .pg h1{font-weight:normal;font-size:34px;margin:0 0 16px}
  """
  tabs = [("Weather forecast", "1"), ("City library catalog", "2"), ("Train timetable", "3"), ("Recipe: lentil soup", "4")]
  t = "".join(f'<div class="tab{" on" if i == 0 else ""}" data-box="tab_{k}"><span>{n}</span><div class="xb" data-box="close_{k}">&#10005;</div></div>' for i, (n, k) in enumerate(tabs))
  body = (f'<div class="col"><div class="tabs">{t}</div><div class="url">&#8592; &#8594; &#8635;<span>weather.example.org/today</span></div>'
          '<div class="pg"><h1>Today: light rain, 14 °C</h1>Rain until the early afternoon, then clearing from the west. Wind from the south at 12 km/h.<br>Tomorrow: sunny, 17 °C.</div></div>')
  return desktop("Weather forecast - Browser", "", body, css)


def dialog():
  css = """
  .doc{flex:1;padding:40px 90px;font-size:18px;line-height:30px;color:#444;background:#fff}
  .shade{position:absolute;inset:0;background:rgba(0,0,0,.28)}
  .dlg{position:absolute;left:620px;top:360px;width:680px;background:#f7f7f7;border-radius:10px;box-shadow:0 8px 28px rgba(0,0,0,.5);padding:30px 34px;font-size:17px}
  .dlg h3{margin:0 0 12px;font-size:20px}
  .bs{display:flex;justify-content:flex-end;gap:14px;margin-top:28px}
  .bs div{padding:11px 26px;border:1px solid #bbb;border-radius:6px;background:#fff;font-size:16px}
  .bs .p{background:#3f8ae0;color:#fff;border-color:#3f8ae0}
  """
  body = ('<div class="doc">Meeting notes, November<br><br>1. Move the review to the 12th.<br>2. Ask finance for the corrected totals.<br>3. Book the room by the river.<br>4. Send the agenda on Friday.</div>'
          '<div class="shade"></div>')
  inner = desktop("meeting_notes.odt - Writer", "File Edit View Insert Format Styles Table Tools Help", body, css)
  dlg = ('<div class="dlg"><h3>Save changes to "meeting_notes.odt"?</h3>Your changes will be lost if you close this document without saving.'
         '<div class="bs"><div data-box="btn_discard">Don\'t Save</div><div data-box="btn_cancel">Cancel</div><div class="p" data-box="btn_save">Save</div></div></div>')
  return inner[:-len("</div>")] + dlg + "</div>"


def music():
  css = """
  .pl{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#222;color:#eee;font-size:20px;gap:18px}
  .cover{width:380px;height:380px;background:linear-gradient(140deg,#c94f7c,#3f51b5)}
  .ctl{display:flex;gap:30px;align-items:center;margin-top:14px}
  .ctl div{width:76px;height:76px;border-radius:50%;background:#3a3a3a;display:flex;align-items:center;justify-content:center}
  .ctl .big{width:96px;height:96px;background:#e8e8e8}
  .bar{width:620px;height:6px;background:#555}.bar i{display:block;width:38%;height:6px;background:#e8e8e8}
  """
  ic = {
      "shuffle": '<path d="M4 8 h6 l14 16 h6 M4 24 h6 l4 -5 M19 13 l5 -5 h6" stroke="#ddd" stroke-width="3" fill="none"/>',
      "prev": '<rect x="6" y="6" width="4" height="20" fill="#ddd"/><polygon points="28,6 12,16 28,26" fill="#ddd"/>',
      "pause": '<rect x="7" y="4" width="7" height="24" fill="#222"/><rect x="18" y="4" width="7" height="24" fill="#222"/>',
      "next": '<polygon points="4,6 20,16 4,26" fill="#ddd"/><rect x="22" y="6" width="4" height="20" fill="#ddd"/>',
      "repeat": '<path d="M6 12 a8 8 0 0 1 8 -6 h10 M26 20 a8 8 0 0 1 -8 6 h-10" stroke="#ddd" stroke-width="3" fill="none"/><polygon points="22,1 29,6 22,11" fill="#ddd"/><polygon points="10,21 3,26 10,31" fill="#ddd"/>',
  }
  ctl = "".join(f'<div class="{"big" if k == "pause" else ""}" data-box="btn_{k}"><svg width="32" height="32">{ic[k]}</svg></div>' for k in ("shuffle", "prev", "pause", "next", "repeat"))
  body = (f'<div class="pl"><div class="cover"></div><div>Harbor Lights</div><div style="font-size:16px;color:#aaa">Quiet Tide</div>'
          f'<div class="bar"><i></i></div><div class="ctl">{ctl}</div></div>')
  return desktop("Music", "", body, css)


def pdfview():
  css = """
  .col{flex:1;display:flex;flex-direction:column;background:#6b6b6b}
  .tl{height:46px;flex:none;background:#f5f5f5;border-bottom:1px solid #ddd;display:flex;align-items:center;gap:10px;padding:0 14px;font-size:14px}
  .tl div{width:36px;height:32px;border:1px solid #ccc;border-radius:4px;background:#fff;display:flex;align-items:center;justify-content:center}
  .pg{margin:24px auto;width:640px;height:860px;background:#fff;padding:60px 70px;font-size:15px;line-height:25px;color:#333}
  .pg h1{font-size:25px;font-weight:normal;margin:0 0 18px}
  """
  up = '<svg width="16" height="16"><polygon points="8,3 15,13 1,13" fill="#444"/></svg>'
  down = '<svg width="16" height="16"><polygon points="8,13 15,3 1,3" fill="#444"/></svg>'
  body = (f'<div class="col"><div class="tl"><div data-box="btn_prev">{up}</div><div data-box="btn_next">{down}</div><span>Page 1</span></div>'
          '<div class="pg"><h1>Annual report</h1>This report summarizes the work of the year: what was built, what it cost and what comes next. '
          'The first section reviews the budget, the second the projects that shipped, and the third the plan for the coming year.<br><br>'
          'Spending stayed within the plan in every quarter except the third, when travel for the field study ran over by a small margin.</div></div>')
  return desktop("/home/user/Documents/report_final.pdf - Document Viewer", "", body, css)


def notes():
  css = """
  .ed{flex:1;font-family:'DVM';font-size:17px;line-height:28px;padding:22px 30px;white-space:pre-wrap;color:#222;background:#fff}
  .ln{width:52px;background:#f2f2f2;color:#888;font-family:'DVM';font-size:17px;line-height:28px;text-align:right;padding:22px 8px 0 0}
  """
  lines = ["Style guide notes", "", "The colour of the header should match the logo.", "Use one accent colour on each page.", "Tables: light grey rows, no colour in the borders.",
           "Links keep the default colour unless the page is dark.", "", "Check with the print team before changing a colour."]
  body = (f'<div class="ln">{"<br>".join(str(i + 1) for i in range(len(lines)))}</div><div class="ed">{chr(10).join(lines)}</div>')
  return desktop("/home/user/Documents/style_notes.txt - Text Editor", "File Edit View Search Tools Help", body, css)


def deck():
  css = """
  .side{width:280px;background:#efefef;border-right:1px solid #ccc;padding:16px;display:flex;flex-direction:column;gap:14px;font-size:13px;color:#555}
  .side div{background:#fff;border:1px solid #bbb;height:136px;padding:8px}
  .sl{flex:1;background:#8c8c8c;display:flex;align-items:center;justify-content:center}
  .page{width:1120px;height:630px;background:#fff;padding:70px;font-size:26px;line-height:40px;color:#333}
  .page h1{margin:0 0 30px;font-size:46px;font-weight:normal}
  """
  sides = "".join(f"<div>{i + 1} &nbsp; {n}</div>" for i, n in enumerate(("Q3 review", "Revenue", "Hiring", "Risks")))
  body = (f'<div class="side">{sides}</div><div class="sl"><div class="page"><h1>Q3 review</h1>Revenue up 8 %<br>Two products shipped<br>Hiring on plan</div></div>')
  return desktop("/home/user/Documents/quarterly.pptx - Slides", "File Edit View Insert Format Slide Tools Help", body, css)


def downloads():
  css = """
  .side{width:230px;background:#f2f2f2;border-right:1px solid #ddd;padding:14px 0;font-size:15px}
  .side div{padding:9px 22px}.side .on{background:#d7e6f7}
  .main{flex:1;display:flex;flex-direction:column}
  .path{height:46px;flex:none;border-bottom:1px solid #ddd;display:flex;align-items:center;padding:0 16px;font-size:15px;gap:8px;background:#fafafa}
  .path span{background:#e8e8e8;border-radius:5px;padding:5px 10px}
  .grid{flex:1;display:grid;grid-template-columns:repeat(6,1fr);gap:26px 8px;padding:26px 14px;align-content:start}
  .cell{display:flex;flex-direction:column;align-items:center;font-size:14px;gap:10px}
  """
  def doc(c):
    return f'<svg width="96" height="120"><path d="M6 2 H66 L92 28 V118 H6 z" fill="{c}"/><path d="M66 2 V28 H92 z" fill="rgba(255,255,255,.55)"/><path d="M22 58 H76 M22 74 H76 M22 90 H60" stroke="rgba(255,255,255,.8)" stroke-width="5"/></svg>'
  items = [("minutes_oct.txt", "#5c7cbd"), ("q3_figures.txt", "#5c7cbd"), ("draft_a.txt", "#57a773"), ("draft_b.txt", "#57a773"), ("contract_v2.txt", "#d7342f"), ("agenda.txt", "#5c7cbd"),
           ("reading_list.txt", "#d9b23c"), ("packing.txt", "#d9b23c"), ("old_notes.txt", "#888888"), ("ideas.txt", "#57a773"), ("contacts.txt", "#5c7cbd"), ("todo.txt", "#888888")]
  cells = "".join(f'<div class="cell"><div>{doc(c)}</div><div>{n}</div></div>' for n, c in items)
  body = (f'<div class="side"><div>Home</div><div>Documents</div><div class="on">Downloads</div><div>Pictures</div><div>Music</div><div>Trash</div></div>'
          f'<div class="main"><div class="path"><span>home</span>/<span>user</span>/<span>Downloads</span></div><div class="grid">{cells}</div></div>')
  return desktop("Downloads - Files", "", body, css)


def code():
  css = """
  .act{width:60px;background:#2c2c32}
  .scm{width:400px;background:#f3f3f3;border-right:1px solid #ddd;font-size:14px}
  .scm h4{margin:0;padding:12px 14px;font-size:13px;color:#555;font-weight:normal;letter-spacing:.5px}
  .scm .i{padding:6px 18px;display:flex;justify-content:space-between}.scm .m{color:#b5810a}
  .inp{margin:6px 14px;height:40px;border:1px solid #ccc;background:#fff;font-size:13px;line-height:40px;padding-left:10px;color:#888}
  .ed{flex:1;font-family:'DVM';font-size:16px;line-height:26px;padding:16px 22px;white-space:pre;background:#fff;color:#222}
  .branch{position:absolute;left:0;right:0;bottom:0;height:26px;background:#3f6fb5;color:#fff;font-size:13px;line-height:26px;padding-left:14px}
  """
  src = ('def total(rows):\n    """Add up the amounts of all rows."""\n    return sum(r["amount"] for r in rows)\n\n\n'
         'def report(rows):\n    print("Total:", total(rows))\n')
  body = ('<div class="act"></div><div class="scm"><h4>SOURCE CONTROL</h4><div class="inp">Message</div><h4>CHANGES</h4>'
          '<div class="i">report.py <span class="m">M</span></div><div class="i">README.md <span class="m">M</span></div></div>'
          f'<div class="ed">{src}</div><div class="branch">main &nbsp; ~/Projects/ledger</div>')
  return desktop("report.py - ledger - Code", "File Edit Selection View Go Run Terminal Help", body, css)


SCREENS = {"files": files, "sheet": sheet, "editor": editor, "settings": settings, "terminal": terminal, "phone_settings": phone_settings,
           "phone_files": phone_files, "phone_home": phone_home, "mail": mail, "browser": browser, "dialog": dialog, "music": music,
           "pdfview": pdfview, "notes": notes, "deck": deck, "downloads": downloads, "code": code}
BOX_JS = """<script>
(function(){const o={};document.querySelectorAll('[data-box]').forEach(e=>{const r=e.getBoundingClientRect();
o[e.dataset.box]=[Math.round(r.left),Math.round(r.top),Math.round(r.width),Math.round(r.height)];});
const p=document.createElement('pre');p.id='__boxes';p.style.display='none';p.textContent=JSON.stringify(o);document.body.appendChild(p);})();
</script>"""


def render(name, html, out, fonts):
  portrait = name.startswith("phone_")
  w, h = (1080, 2400) if portrait else (1920, 1080)
  doc = (f'<!doctype html><html><head><meta charset="utf-8"><style>{BASE_CSS % {"fonts": fonts}}</style></head><body>{html}'
         f'{BOX_JS}</body></html>')
  path = os.path.join(out, name + ".html")
  with open(path, "w", encoding="utf-8") as f:
    f.write(doc)
  url = "file://" + os.path.abspath(path)
  png = os.path.join(out, name + ".png")
  base = [CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
          "--allow-file-access-from-files", f"--window-size={w},{h + 200}", "--virtual-time-budget=1500",
          f"--user-data-dir={out}/.chrome"]
  subprocess.run(base + [f"--screenshot={png}", url], check=True, capture_output=True, timeout=120)
  im = Image.open(png).convert("RGB")
  im.crop((0, 0, w, h)).save(png, format="PNG", optimize=True)
  dom = subprocess.run(base + ["--dump-dom", url], check=True, capture_output=True, timeout=120).stdout.decode("utf-8")
  a = dom.index('id="__boxes"')
  a = dom.index(">", a) + 1
  b = dom.index("</pre>", a)
  boxes = json.loads(dom[a:b].replace("&amp;", "&"))
  with open(os.path.join(out, name + ".boxes.json"), "w") as f:
    json.dump(boxes, f, indent=0, sort_keys=True)
  return boxes


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("out")
  ap.add_argument("--fonts", required=True)
  ap.add_argument("--only", default=None)
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  for name, fn in SCREENS.items():
    if a.only and name != a.only:
      continue
    boxes = render(name, fn(), a.out, os.path.abspath(a.fonts))
    print(name, len(boxes), "boxes", file=sys.stderr)


if __name__ == "__main__":
  main()
