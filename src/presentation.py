"""Shared dark presentation; ordinary HTML tables preserve browser page zoom."""
from html import escape
import json

import pandas as pd
import streamlit as st

GITHUB_URL = "https://github.com/okada-kou-dev/sound-insight"

STYLE = """
<style>
.stApp {background:#0b1120;color:#e6edf7;color-scheme:dark;}
[data-testid="stHeader"] {background:rgba(11,17,32,.94);}
.block-container {max-width:88rem;padding-top:1.5rem;padding-bottom:2rem;}
h1,h2,h3 {letter-spacing:-.025em;}
[data-testid="stCaptionContainer"] {color:#9baec7;}
[data-testid="stMetric"] {padding:.85rem 1rem;background:#111b2c;border:1px solid #24344f;border-radius:.9rem;}
[data-testid="stMetricValue"] {color:#e9f3ff;}
[data-testid="stTabs"] [role="tablist"] {gap:1.6rem;border-bottom:1px solid #26334b;}
[data-testid="stTabs"] [role="tab"] {font-size:1rem;padding:.6rem .2rem;}
button {border-radius:7px;font-weight:600;}
button[kind="primary"] {background:#38bdf8;color:#071522;border:0;box-shadow:0 0 1.2rem #38bdf81a;}
button[kind="secondary"] {background:#162338;color:#e6edf7;border-color:#2e425e;}
[data-testid="stExpander"] {border-color:#26334b;background:#0e1829;border-radius:.8rem;}
.si-hero {display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:.6rem;}
.si-kicker {color:#38bdf8;font-size:.72rem;letter-spacing:.19em;font-weight:700;margin:0 0 .35rem;}
.si-hero h1 {font-size:clamp(1.6rem,2.5vw,2.1rem);margin:0;line-height:1.2;color:#f2f7ff;}
.si-hero p {margin:.7rem 0 0;color:#9baec7;font-size:.95rem;}
.si-tag {white-space:nowrap;border:1px solid #32445f;padding:.5rem .8rem;border-radius:2rem;font-size:.75rem;color:#b7cce3;}
.si-table-wrap {width:100%;overflow-x:auto;border:1px solid #26354c;border-radius:.75rem;}
.si-table {border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums;background:#0e192a;}
.si-table th {text-align:left;color:#a5bbd6;background:#152239;font-weight:600;}
.si-table td,.si-table th {padding:.65rem .8rem;border-bottom:1px solid #223047;white-space:nowrap;}
.si-table tr:last-child td {border-bottom:0;}
.si-table tr:hover td {background:#15253b;}
.si-table .good {color:#69e4c5;}.si-table .bad {color:#ff9c99;}
.si-record {white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem;color:#b9cbe1;line-height:1.7;}
.si-source {border-top:1px solid #26354c;padding-top:1.2rem;margin-top:2rem;color:#859bb5;font-size:.78rem;}
.si-source a {color:#88cfc9;}
@media(max-width:640px){.si-hero{align-items:flex-start;flex-direction:column}.block-container{padding-top:1.5rem;}}
</style>
"""


def app_header(*, online: bool) -> None:
    st.html(STYLE)
    badge = "SAMPLE DEMO · 50 CLIPS" if online else "LOCAL · 50 CLIPS"
    st.html(f'''<section class="si-hero"><div><div class="si-kicker">ACOUSTIC ANOMALY LAB</div>
    <h1>SOUND INSIGHT</h1></div>
    <span class="si-tag">{badge}</span></section>''')


def html_table(frame: pd.DataFrame, *, index: bool = False) -> None:
    """No canvas grid or wheel listener: all contents use native browser zoom."""
    if frame.empty:
        st.caption("該当する記録はありません。")
        return
    table = frame.to_html(index=index, border=0, classes="si-table", escape=True,
                          float_format=lambda value: f"{value:.6g}")
    # Fixed literal substitutions only; all values were escaped by pandas.
    for value in ("検出成功", "正常を正しく判定"):
        table = table.replace(f"<td>{value}</td>", f'<td class="good">{value}</td>')
    for value in ("見逃し", "誤警報", "判定不能"):
        table = table.replace(f"<td>{value}</td>", f'<td class="bad">{value}</td>')
    st.html(f'<div class="si-table-wrap">{table}</div>')


def json_record(value) -> None:
    st.html('<pre class="si-record">' + escape(json.dumps(value, ensure_ascii=False, indent=2)) + '</pre>')
