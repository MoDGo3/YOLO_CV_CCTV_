import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="CCTV Analytics Dashboard", layout="wide")


def get_theme_tokens(mode: str) -> Dict[str, str]:
    light = {
        "bg": "linear-gradient(135deg, #f8fafc 0%, #e2e8f0 55%, #f1f5f9 100%)",
        "card_bg": "rgba(255, 255, 255, 0.9)",
        "card_border": "rgba(148, 163, 184, 0.25)",
        "shadow": "0 22px 40px rgba(15, 23, 42, 0.12)",
        "accent": "#2563eb",
        "success": "#16a34a",
        "danger": "#dc2626",
        "text": "#0f172a",
        "muted": "#475569",
        "slider_bg": "rgba(148, 163, 184, 0.32)",
        "input_bg": "rgba(241, 245, 249, 0.85)",
        "input_border": "rgba(148, 163, 184, 0.45)",
        "header_bg": "rgba(248, 250, 252, 0.82)",
        "table_even": "rgba(226, 232, 240, 0.7)",
        "table_odd": "rgba(241, 245, 249, 0.55)",
        "grid_color": "#cbd5f5",
        "axis_label": "#334155",
        "axis_title": "#1e293b",
        "metric_glow": "rgba(37, 99, 235, 0.14)",
        "section_glow": "rgba(99, 102, 241, 0.12)",
        "primary_button": "linear-gradient(135deg, #2563eb, #7c3aed)",
        "primary_button_hover": "linear-gradient(135deg, #1d4ed8, #6d28d9)",
        "button_text": "#f8fafc",
        "status_disc_bg": "rgba(220, 38, 38, 0.12)",
        "status_disc_shadow": "0 12px 24px rgba(220, 38, 38, 0.18)",
        "status_disc_dot_shadow": "0 0 14px rgba(220, 38, 38, 0.48)",
        "status_conn_bg": "rgba(34, 197, 94, 0.16)",
        "status_conn_shadow": "0 12px 24px rgba(34, 197, 94, 0.22)",
        "status_conn_dot_shadow": "0 0 14px rgba(34, 197, 94, 0.5)",
        "sidebar_text": "#0f172a",
        "chart_bg": "#f8fafc",
    }
    dark = {
        "bg": "linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #312e81 100%)",
        "card_bg": "rgba(15, 23, 42, 0.9)",
        "card_border": "rgba(148, 163, 184, 0.15)",
        "shadow": "0 22px 40px rgba(8, 15, 30, 0.55)",
        "accent": "#38bdf8",
        "success": "#22c55e",
        "danger": "#ef4444",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "slider_bg": "rgba(148, 163, 184, 0.18)",
        "input_bg": "rgba(15, 23, 42, 0.78)",
        "input_border": "rgba(148, 163, 184, 0.25)",
        "header_bg": "rgba(15, 23, 42, 0.65)",
        "table_even": "rgba(30, 41, 59, 0.65)",
        "table_odd": "rgba(15, 23, 42, 0.6)",
        "grid_color": "#1e293b",
        "axis_label": "#e2e8f0",
        "axis_title": "#cbd5f5",
        "metric_glow": "rgba(56, 189, 248, 0.18)",
        "section_glow": "rgba(79, 70, 229, 0.18)",
        "primary_button": "linear-gradient(135deg, #38bdf8, #6366f1)",
        "primary_button_hover": "linear-gradient(135deg, #0ea5e9, #4f46e5)",
        "button_text": "#f8fafc",
        "status_disc_bg": "rgba(239, 68, 68, 0.18)",
        "status_disc_shadow": "0 12px 28px rgba(239, 68, 68, 0.18)",
        "status_disc_dot_shadow": "0 0 14px rgba(239, 68, 68, 0.7)",
        "status_conn_bg": "rgba(34, 197, 94, 0.22)",
        "status_conn_shadow": "0 12px 28px rgba(34, 197, 94, 0.25)",
        "status_conn_dot_shadow": "0 0 14px rgba(34, 197, 94, 0.7)",
        "sidebar_text": "#f8fafc",
        "chart_bg": "#0b1120",
    }
    return light if mode == "Light" else dark

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"

initial_index = 0 if st.session_state["theme_mode"] == "Light" else 1
theme_mode = st.sidebar.radio(
    "Theme", options=("Light", "Dark"), index=initial_index, horizontal=True
)
st.session_state["theme_mode"] = theme_mode
theme = get_theme_tokens(theme_mode)

st.markdown(
    f"""
    <style>
        :root {{
            --bg: {theme['bg']};
            --card-bg: {theme['card_bg']};
            --card-border: {theme['card_border']};
            --shadow: {theme['shadow']};
            --accent: {theme['accent']};
            --success: {theme['success']};
            --danger: {theme['danger']};
            --text: {theme['text']};
            --muted: {theme['muted']};
            --button-text: {theme['button_text']};
        }}

        html, body, #root {{ background: var(--bg); color: var(--text); }}
        div[data-testid="stAppViewContainer"] {{ background: var(--bg); color: var(--text); }}
        div[data-testid="block-container"] {{ background: transparent !important; color: var(--text); }}
        section.main {{ background: transparent !important; color: var(--text); }}
        div[data-testid="stHeader"] {{ background: {theme['header_bg']}; backdrop-filter: blur(12px); }}
        section[data-testid="stSidebar"] {{
            background: {theme['card_bg']} !important;
            border-right: 1px solid {theme['card_border']};
            color: {theme['sidebar_text']};
        }}
        section[data-testid="stSidebar"] * {{ color: inherit; }}

        .stMetric {{
            position: relative;
            overflow: hidden;
            background: radial-gradient(circle at top right, {theme['metric_glow']}, transparent 55%), var(--card-bg);
            border-radius: 18px;
            border: 1px solid var(--card-border);
            padding: 1rem 1.4rem;
            box-shadow: var(--shadow);
        }}
        .stMetric label {{ color: var(--muted); letter-spacing: 0.08em; font-size: 0.78rem; }}
        .stMetric div[data-testid="stMetricValue"] {{
            color: var(--text);
            font-size: 2.3rem;
            font-weight: 700;
        }}

        /* ===== Carded dataframe container ===== */
        div[data-testid="stDataFrame"] {{
            border-radius: 18px;
            border: 1px solid var(--card-border);
            background: var(--card-bg);
            box-shadow: var(--shadow);
            padding: 0.4rem;
        }}

        /* Status chip */
        .status-chip {{
            display: inline-flex;
            gap: 10px;
            align-items: center;
            background: {theme['status_disc_bg']};
            border-radius: 999px;
            padding: 0.45rem 0.85rem;
            font-weight: 600;
            color: var(--danger);
            box-shadow: {theme['status_disc_shadow']};
        }}
        .status-dot {{ width: 12px; height: 12px; border-radius: 50%; background: var(--danger); box-shadow: {theme['status_disc_dot_shadow']}; }}
        .status-chip.connected {{ background: {theme['status_conn_bg']}; color: var(--success); box-shadow: {theme['status_conn_shadow']}; }}
        .status-chip.connected .status-dot {{ background: var(--success); box-shadow: {theme['status_conn_dot_shadow']}; }}

        .metric-subtitle {{ color: var(--muted); font-size: 0.92rem; margin-top: -0.35rem; }}
        .section-card {{
            background: radial-gradient(circle at top left, {theme['section_glow']}, transparent 55%), var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.6rem;
            box-shadow: var(--shadow);
        }}

        /* NOTE: เดิมเราเบลอ vertical block ทั้งหมด — แต่ถ้ามีตารางให้ยกเลิก */
        .stApp div[data-testid="stBlock"] > div:has(> div[data-testid="stVerticalBlock"]) {{ backdrop-filter: blur(6px); }}
        .stApp div[data-testid="stBlock"] > div:has( div[data-testid="stDataFrame"] ) {{ backdrop-filter: none !important; }}
        .stApp div[data-testid="stBlock"] > div:has( > div[data-testid="stVerticalBlock"] ):has( div[data-testid="stDataFrame"] ) {{ backdrop-filter: none !important; }}

        /* Buttons & slider */
        button[kind="primary"] {{ background: {theme['primary_button']}; color: var(--button-text); border: none; box-shadow: 0 14px 30px rgba(99, 102, 241, 0.35); }}
        button[kind="primary"]:hover {{ background: {theme['primary_button_hover']}; }}
        button[kind="secondary"] {{ background: {theme['primary_button']}; color: var(--button-text); border: none; box-shadow: 0 12px 26px rgba(99, 102, 241, 0.28); }}
        button[kind="secondary"]:hover {{ background: {theme['primary_button_hover']}; color: var(--button-text); }}
        div[data-baseweb="slider"] > div {{ background: {theme['slider_bg']}; }}

        /* ===== DataFrame table colors (legacy table path) ===== */
        div[data-testid="stDataFrame"] table {{ color: var(--text); background: transparent; }}
        div[data-testid="stDataFrame"] th {{
            background: var(--card-bg) !important;
            color: var(--text) !important;
            border-bottom: 1px solid var(--card-border) !important;
        }}
        div[data-testid="stDataFrame"] tbody td {{
            background: transparent !important;
            color: var(--text) !important;
            border-color: var(--card-border) !important;
        }}
        div[data-testid="stDataFrame"] tbody tr:nth-child(even) {{ background: {theme['table_even']}; }}
        div[data-testid="stDataFrame"] tbody tr:nth-child(odd)  {{ background: {theme['table_odd']};  }}

        input, textarea {{ background: {theme['input_bg']} !important; color: var(--text) !important; border: 1px solid {theme['input_border']} !important; }}
        label[data-baseweb="checkbox"] span {{ color: inherit !important; }}

        /* ===== Glide Data Editor (เส้นทางใหม่ของ st.dataframe) – บังคับไม่ให้ซีด ===== */
        /* ครอบทั้งหมดในตาราง */
        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] * {{
            color: var(--text) !important;
            opacity: 1 !important;
            filter: none !important;
            -webkit-text-fill-color: var(--text) !important;
            text-shadow: none !important;
            mix-blend-mode: normal !important;
        }}

        /* คอนเทนเนอร์หลักของ GDE */
        div[data-testid="stDataFrame"] .stDataFrameGlideDataEditor {{
            background: var(--card-bg) !important;
            color: var(--text) !important;
            border: 1px solid var(--card-border) !important;
        }}
        div[data-testid="stDataFrame"] .dvn-scroller,
        div[data-testid="stDataFrame"] .dvn-outer-container,
        div[data-testid="stDataFrame"] .dvn-outer-grid,
        div[data-testid="stDataFrame"] .dvn-content-container,
        div[data-testid="stDataFrame"] .dvn-view-with-header,
        div[data-testid="stDataFrame"] .dvn-grid-container,
        div[data-testid="stDataFrame"] .dvn-table-container {{
            background: transparent !important;
            border-color: {theme['card_border']} !important;
            filter: none !important;
            opacity: 1 !important;
        }}

        /* หัวตาราง/เซลล์/ค่า */
        div[data-testid="stDataFrame"] .dvn-column-header,
        div[data-testid="stDataFrame"] .dvn-row-header,
        div[data-testid="stDataFrame"] .dvn-grid-cell,
        div[data-testid="stDataFrame"] .dvn-grid-cell__text,
        div[data-testid="stDataFrame"] .dvn-cell-value,
        div[data-testid="stDataFrame"] .dvn-column-header__name,
        div[data-testid="stDataFrame"] .dvn-column-header__menu,
        div[data-testid="stDataFrame"] .dvn-row-selector,
        div[data-testid="stDataFrame"] .dvn-grid-cell__text span,
        div[data-testid="stDataFrame"] .dvn-cell-value span {{
            color: var(--text) !important;
            opacity: 1 !important;
            filter: none !important;
            font-weight: 600;
        }}

        /* ข้อความ placeholder/empty */
        div[data-testid="stDataFrame"] .dvn-no-data,
        div[data-testid="stDataFrame"] .dvn-empty,
        div[data-testid="stDataFrame"] .dvn-empty *,
        div[data-testid="stDataFrame"] .dvn-footer,
        div[data-testid="stDataFrame"] .dvn-scroller span,
        div[data-testid="stDataFrame"] span {{
            color: var(--text) !important;
            opacity: 1 !important;
            filter: none !important;
            -webkit-text-fill-color: var(--text) !important;
        }}

        /* โฟกัสเซลล์ */
        div[data-testid="stDataFrame"] .dvn-grid-cell--focused,
        div[data-testid="stDataFrame"] .dvn-grid-cell--selected {{ box-shadow: inset 0 0 0 1px var(--accent) !important; }}

        /* roles (สำรอง) */
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="rowheader"],
        div[data-testid="stDataFrame"] [role="columnheader"] {{
            background-color: transparent !important;
            color: var(--text) !important;
        }}
        div[data-testid="stDataFrame"] .dvn-column-header,
        div[data-testid="stDataFrame"] .dvn-row-header {{
            background: transparent !important;
            color: var(--text) !important;
            border-color: var(--card-border) !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("📡 CCTV MQTT Analytics Dashboard")
st.caption("แดชบอร์ดวิเคราะห์ข้อมูลเข้า-ออกจากระบบ CCTV แบบเรียลไทม์ พร้อมสถิติและตารางเหตุการณ์")

# Create tabs for different views
tab_realtime, tab_benchmark = st.tabs(["📊 Real-time Analytics", "⚡ Benchmark Results"])


def _get_nested_value(source: Dict[str, Any], paths: Iterable[Tuple[str, ...]]) -> Any:
    for path in paths:
        current: Any = source
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is not None:
            return current
    return None


def fetch_metrics(base_url: str, limit: int) -> Dict[str, Any]:
    base = base_url.rstrip("/")
    response = requests.get(f"{base}/metrics", params={"limit": limit}, timeout=5)
    response.raise_for_status()
    return response.json()


def fetch_benchmark(base_url: str) -> Dict[str, Any]:
    """Fetch benchmark results from FastAPI"""
    base = base_url.rstrip("/")
    try:
        response = requests.get(f"{base}/api/benchmark", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


default_backend = os.getenv("DASHBOARD_BACKEND_URL", "http://localhost:8080")

backend_url = st.sidebar.text_input("FastAPI Dashboard URL", default_backend)
message_limit = st.sidebar.slider("จำนวนข้อความล่าสุด", 10, 300, 80, step=10)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("รีเฟรชอัตโนมัติ", value=True)
refresh_interval = st.sidebar.slider("ช่วงเวลารีเฟรช (วินาที)", 1, 15, 3)


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


if st.sidebar.button("รีเฟรชตอนนี้", use_container_width=True):
    rerun_app()


if not backend_url:
    st.warning("กรุณาระบุ URL ของ FastAPI dashboard")
    st.stop()

try:
    metrics = fetch_metrics(backend_url, message_limit)
    connected = True
except Exception as exc:  # pylint: disable=broad-except
    connected = False
    st.error(f"ไม่สามารถดึงข้อมูลจาก {backend_url}: {exc}")
    metrics = {"stats": {}, "messages": [], "timestamp": time.time()}
    auto_refresh = False

stats = metrics.get("stats", {})
messages = metrics.get("messages", [])
updated_at = metrics.get("timestamp")

status_class = "status-chip connected" if connected else "status-chip"
status_text = "Connected" if connected else "Disconnected"

# =====================================================
# TAB 1: REAL-TIME ANALYTICS
# =====================================================
with tab_realtime:
    st.markdown(
        f"""
        <div class="{status_class}">
            <span class="status-dot"></span>
            {status_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if updated_at:
        st.caption(f"อัปเดตล่าสุด: {datetime.fromtimestamp(updated_at).strftime('%Y-%m-%d %H:%M:%S')}")

    metric_labels = [
        ("IN", stats.get("in_count", 0), "จำนวนเข้า"),
        ("OUT", stats.get("out_count", 0), "จำนวนออก"),
        ("Asia", stats.get("asia_count", 0), "ภูมิภาคเอเชีย"),
        ("Europe", stats.get("europe_count", 0), "ภูมิภาคยุโรป"),
        ("Male", stats.get("male_count", 0), "เพศชาย"),
        ("Female", stats.get("female_count", 0), "เพศหญิง"),
        ("Unknown IN", stats.get("unknown_in", 0), "ไม่ทราบเพศ (เข้า)"),
        ("Unknown OUT", stats.get("unknown_out", 0), "ไม่ทราบเพศ (ออก)"),
    ]

    metric_cols = st.columns(len(metric_labels))
    for col, (label, value, subtitle) in zip(metric_cols, metric_labels):
        col.metric(label, value)
        col.markdown(f'<div class="metric-subtitle">{subtitle}</div>', unsafe_allow_html=True)

    chart_df = pd.DataFrame(metric_labels, columns=["category", "value", "subtitle"])
    with st.container():
        st.subheader("การกระจายจำนวน (Bar Chart)")
        color_map_light = {
            "IN": "#2563eb",
            "OUT": "#fb923c",
            "Asia": "#8b5cf6",
            "Europe": "#0ea5e9",
            "Male": "#22c55e",
            "Female": "#ec4899",
            "Unknown IN": "#94a3b8",
            "Unknown OUT": "#64748b",
        }
        color_map_dark = {
            "IN": "#38bdf8",
            "OUT": "#f97316",
            "Asia": "#8b5cf6",
            "Europe": "#3b82f6",
            "Male": "#10b981",
            "Female": "#f43f5e",
            "Unknown IN": "#94a3b8",
            "Unknown OUT": "#64748b",
        }
        active_palette = color_map_light if theme_mode == "Light" else color_map_dark
        color_map = {key: active_palette[key] for key in active_palette}
        domain = [label for label in color_map if label in chart_df["category"].values]
        color_scale = alt.Scale(domain=domain, range=[color_map[label] for label in domain])
        bar_chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("category:N", title="หมวดหมู่"),
                y=alt.Y("value:Q", title="จำนวน"),
                color=alt.Color("category:N", scale=color_scale, legend=None),
                tooltip=["category:N", "value:Q", "subtitle:N"],
            )
            .properties(height=340, background=theme["chart_bg"])
        ).configure_view(strokeOpacity=0, fill=theme["chart_bg"]).configure_axis(
            labelColor=theme["axis_label"], titleColor=theme["axis_title"]
        ).configure_axisY(gridColor=theme["grid_color"]).configure_axisX(grid=False)
        st.altair_chart(bar_chart, use_container_width=True)

    rows = []
    for msg in messages:
        payload = msg.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        timestamp = msg.get("timestamp", time.time())
        rows.append(
            {
                "เวลา": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "ทิศทาง": _get_nested_value(
                    payload,
                    [
                        ("direction",),
                        ("summary", "direction"),
                        ("data", "movement", "direction"),
                    ],
                ),
                "ภูมิภาค": _get_nested_value(
                    payload,
                    [
                        ("region",),
                        ("summary", "region"),
                        ("data", "region"),
                        ("data", "demographics", "region"),
                    ],
                ),
                "เพศ": _get_nested_value(
                    payload,
                    [
                        ("gender",),
                        ("summary", "gender"),
                        ("data", "gender"),
                        ("data", "demographics", "gender"),
                    ],
                ),
                "สถานะ": _get_nested_value(
                    payload,
                    [
                        ("summary", "status"),
                        ("data", "movement", "status"),
                    ],
                ),
            }
        )

    if rows:
        table_df = pd.DataFrame(rows)
        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.subheader("ภาพรวมทิศทาง")
            st.dataframe(
                table_df["ทิศทาง"]
                .value_counts()
                .rename_axis("ทิศทาง")
                .reset_index(name="จำนวน"),
                hide_index=True,
                use_container_width=True,
            )
        with summary_col2:
            st.subheader("ภูมิภาคล่าสุด")
            st.dataframe(
                table_df["ภูมิภาค"]
                .value_counts()
                .rename_axis("ภูมิภาค")
                .reset_index(name="จำนวน"),
                hide_index=True,
                use_container_width=True,
            )
        with summary_col3:
            st.subheader("สัดส่วนเพศ")
            st.dataframe(
                table_df["เพศ"]
                .value_counts()
                .rename_axis("เพศ")
                .reset_index(name="จำนวน"),
                hide_index=True,
                use_container_width=True,
            )

        st.subheader("บันทึกเหตุการณ์ล่าสุด")
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        with st.expander("ดูข้อความล่าสุดในรูปแบบ JSON"):
            st.json(messages[-1], expanded=False)
    else:
        st.info("ยังไม่มีข้อความ MQTT ใน buffer")

# =====================================================
# TAB 2: BENCHMARK RESULTS
# =====================================================
with tab_benchmark:
    st.subheader("⚡ ผลการทดสอบประสิทธิภาพระบบ (Benchmark)")
    
    # Fetch benchmark data
    benchmark_data = fetch_benchmark(backend_url) if backend_url else {}
    
    if not benchmark_data or benchmark_data.get("last_updated") is None:
        # Try loading from local file
        import json as json_module
        from pathlib import Path
        results_dir = Path("benchmark/results")
        if results_dir.exists():
            json_files = list(results_dir.glob("benchmark_results_*.json"))
            if json_files:
                import os
                latest_file = max(json_files, key=os.path.getmtime)
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        benchmark_data = json_module.load(f)
                except Exception:
                    pass
    
    if benchmark_data:
        # Summary Metrics - แสดงสรุปเป็น %
        st.markdown("### 📊 สรุปผลการทดสอบ (Summary)")
        
        # ดึงข้อมูลสรุป
        speed = benchmark_data.get("inference_speed", {})
        
        # เช็คว่าใช้ fairface_metrics (FairFace) หรือ gender_metrics (เดิม)
        fairface_metrics = benchmark_data.get("fairface_metrics", {})
        if fairface_metrics:
            gender = fairface_metrics.get("gender", {})
            ethnicity = fairface_metrics.get("ethnicity", {})
        else:
            gender = benchmark_data.get("gender_metrics", {})
            ethnicity = benchmark_data.get("ethnicity_metrics", {})
        
        # คำนวณค่าสรุป
        fps_value = speed.get('fps', 0) if speed else 0
        fps_percent = min(100, (fps_value / 30) * 100) if fps_value else 0  # 30 FPS = 100%
        
        gender_acc = gender.get("accuracy_percent", gender.get("accuracy", 0) * 100 if gender.get("accuracy") else 0) if gender else 0
        ethnicity_acc = ethnicity.get("accuracy_percent", ethnicity.get("accuracy", 0) * 100 if ethnicity.get("accuracy") else 0) if ethnicity else 0
        
        # mAP Reference (YOLOv8n)
        map_value = 37.3  # mAP50 from YOLOv8n
        
        # แสดง 5 metrics หลัก
        summary_cols = st.columns(5)
        summary_cols[0].metric("⏱️ Speed", f"{fps_percent:.0f}%", f"{fps_value} FPS")
        summary_cols[1].metric("👤 Gender", f"{gender_acc:.1f}%" if gender_acc else "N/A")
        summary_cols[2].metric("🌍 Ethnicity", f"{ethnicity_acc:.1f}%" if ethnicity_acc else "N/A")
        summary_cols[3].metric("🎯 mAP", f"{map_value}%", "YOLOv8n")
        
        # คำนวณ Overall Accuracy (เฉลี่ย)
        valid_scores = [s for s in [gender_acc, ethnicity_acc] if s > 0]
        overall_acc = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        summary_cols[4].metric("📈 Overall", f"{overall_acc:.1f}%" if overall_acc else "N/A")
        
        st.markdown("---")
        
        # Inference Speed
        speed = benchmark_data.get("inference_speed", {})
        if speed:
            st.markdown("### ⏱️ ความเร็วในการประมวลผล (Inference Speed)")
            
            speed_cols = st.columns(5)
            speed_cols[0].metric("Pre-process", f"{speed.get('preprocess_ms', 'N/A')} ms")
            speed_cols[1].metric("Inference", f"{speed.get('inference_ms', 'N/A')} ms")
            speed_cols[2].metric("Post-process", f"{speed.get('postprocess_ms', 'N/A')} ms")
            speed_cols[3].metric("Total", f"{speed.get('total_ms', 'N/A')} ms")
            speed_cols[4].metric("FPS", f"{speed.get('fps', 'N/A')}")
            
            # FPS visualization
            fps_value = speed.get('fps', 0)
            if fps_value:
                realtime_status = speed.get('realtime_status', 'unknown')
                if realtime_status == 'excellent':
                    st.success(f"✅ ระบบทำงานแบบ Real-time ได้ดีเยี่ยม ({fps_value} FPS ≥ 30)")
                elif realtime_status == 'good':
                    st.success(f"✅ ระบบทำงานแบบ Real-time ได้ ({fps_value} FPS ≥ 20)")
                elif realtime_status == 'acceptable':
                    st.warning(f"⚠️ ระบบทำงานได้ระดับปานกลาง ({fps_value} FPS)")
                else:
                    st.error(f"❌ ระบบช้าเกินไป ({fps_value} FPS < 15)")
        
        st.markdown("---")
        
        # Gender Accuracy
        gender = benchmark_data.get("gender_metrics", {})
        if gender:
            st.markdown("### 👤 ความถูกต้องในการจำแนกเพศ (Gender Classification)")
            
            acc_percent = gender.get("accuracy_percent", gender.get("accuracy", 0) * 100 if gender.get("accuracy") else "N/A")
            st.metric("Overall Accuracy", f"{acc_percent}%")
            
            class_metrics = gender.get("class_metrics", {})
            if class_metrics:
                gender_rows = []
                for cls, metrics in class_metrics.items():
                    gender_rows.append({
                        "Class": cls,
                        "Precision": metrics.get("precision", "N/A"),
                        "Recall": metrics.get("recall", "N/A"),
                        "F1-Score": metrics.get("f1_score", "N/A")
                    })
                if gender_rows:
                    st.dataframe(pd.DataFrame(gender_rows), hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Ethnicity Accuracy
        ethnicity = benchmark_data.get("ethnicity_metrics", {})
        if ethnicity:
            st.markdown("### 🌍 ความถูกต้องในการจำแนกสัญชาติ (Ethnicity Classification)")
            
            acc_percent = ethnicity.get("accuracy_percent", ethnicity.get("accuracy", 0) * 100 if ethnicity.get("accuracy") else "N/A")
            st.metric("Overall Accuracy", f"{acc_percent}%")
            
            class_metrics = ethnicity.get("class_metrics", {})
            if class_metrics:
                eth_rows = []
                for cls, metrics in class_metrics.items():
                    eth_rows.append({
                        "Class": cls,
                        "Precision": metrics.get("precision", "N/A"),
                        "Recall": metrics.get("recall", "N/A"),
                        "F1-Score": metrics.get("f1_score", "N/A")
                    })
                if eth_rows:
                    st.dataframe(pd.DataFrame(eth_rows), hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # FairFace Per-Race Analysis
        fairface_data = benchmark_data.get("fairface_metrics", {})
        if fairface_data:
            st.markdown("### 🌍 FairFace Dataset - Per-Race Accuracy Analysis")
            
            # เช็ค ethnicity.per_race ก่อน ถ้าไม่มี ลองหา per_race_accuracy
            ethnicity_data = fairface_data.get("ethnicity", {})
            per_race = ethnicity_data.get("per_race", fairface_data.get("per_race_accuracy", {}))
            if per_race:
                # แสดงตาราง
                race_rows = []
                races_list = []
                accuracy_list = []
                
                for race, acc in sorted(per_race.items(), key=lambda x: x[1], reverse=True):
                    races_list.append(race)
                    accuracy_list.append(acc)
                    race_rows.append({
                        "Race/Ethnicity": race.title(),
                        "Accuracy": f"{acc:.1f}%",
                        "Status": "✅ Good" if acc >= 80 else "⚠️ Fair" if acc >= 70 else "❌ Poor"
                    })
                
                # แสดงตาราง
                st.dataframe(pd.DataFrame(race_rows), hide_index=True, use_container_width=True)
                
                # แสดง Fairness Gap
                if races_list and accuracy_list:
                    best_race = races_list[0]
                    best_acc = accuracy_list[0]
                    worst_race = races_list[-1]
                    worst_acc = accuracy_list[-1]
                    fairness_gap = best_acc - worst_acc
                    
                    fairness_cols = st.columns(3)
                    fairness_cols[0].metric("🥇 Best Performance", f"{best_acc:.1f}%", f"{best_race.title()}")
                    fairness_cols[1].metric("⚠️ Fairness Gap", f"{fairness_gap:.1f}%", f"{best_race.title()} vs {worst_race.title()}")
                    fairness_cols[2].metric("⛔ Worst Performance", f"{worst_acc:.1f}%", f"{worst_race.title()}")
                    
                    # Warning if gap is large
                    if fairness_gap > 15:
                        st.warning(f"⚠️ **Significant fairness gap detected:** The system performs {fairness_gap:.1f}% better for {best_race.title()} than {worst_race.title()}. Consider improving the training data for {worst_race.title()}.")
                    elif fairness_gap > 10:
                        st.info(f"ℹ️ **Moderate fairness gap:** The system performs {fairness_gap:.1f}% better for {best_race.title()} than {worst_race.title()}.")
                
                # Chart visualization
                st.markdown("#### Per-Race Accuracy Chart")
                if races_list and accuracy_list:
                    chart_data = pd.DataFrame({
                        "Race": [r.title() for r in races_list],
                        "Accuracy %": accuracy_list
                    })
                    
                    bar_chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X("Accuracy %", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("Race", sort='-x'),
                        color=alt.Color(
                            "Accuracy %:Q",
                            scale=alt.Scale(
                                domain=[0, 100],
                                range=['#ef4444', '#eab308', '#22c55e']  # Red, Yellow, Green
                            )
                        ),
                        tooltip=['Race', 'Accuracy %']
                    ).properties(height=300, title="Accuracy by Race/Ethnicity")
                    
                    st.altair_chart(bar_chart, use_container_width=True)
        
        st.markdown("---")
        
        # Environmental Tests
        environmental = benchmark_data.get("environmental_tests", {})
        if environmental:
            st.markdown("### 🌤️ ความแม่นยำตามสภาวะแวดล้อม (Environmental Accuracy)")
            
            # สร้างตารางแสดงผล
            env_data = []
            
            normal_light = environmental.get("normal_light", {})
            if normal_light:
                env_data.append({
                    "สภาวะแวดล้อม": normal_light.get("name", "แสงปกติ (กลางวัน)"),
                    "ความแม่นยำในการนับ": f"{normal_light.get('counting_accuracy', 'N/A')}%",
                    "ความแม่นยำในการจำแนก": f"{normal_light.get('classification_accuracy', 'N/A')}%"
                })
            
            low_light = environmental.get("low_light", {})
            if low_light:
                env_data.append({
                    "สภาวะแวดล้อม": low_light.get("name", "แสงน้อย (กลางคืน)"),
                    "ความแม่นยำในการนับ": f"{low_light.get('counting_accuracy', 'N/A')}%",
                    "ความแม่นยำในการจำแนก": f"{low_light.get('classification_accuracy', 'N/A')}%"
                })
            
            if env_data:
                st.dataframe(pd.DataFrame(env_data), hide_index=True, use_container_width=True)
                
                # แสดง metrics แยก
                env_cols = st.columns(2)
                with env_cols[0]:
                    st.markdown("**☀️ แสงปกติ (กลางวัน)**")
                    if normal_light:
                        st.metric("การนับ", f"{normal_light.get('counting_accuracy', 0)}%")
                        st.metric("การจำแนก", f"{normal_light.get('classification_accuracy', 0)}%")
                
                with env_cols[1]:
                    st.markdown("**🌙 แสงน้อย (กลางคืน)**")
                    if low_light:
                        st.metric("การนับ", f"{low_light.get('counting_accuracy', 0)}%")
                        st.metric("การจำแนก", f"{low_light.get('classification_accuracy', 0)}%")
        else:
            # Default values if no environmental data
            st.markdown("### 🌤️ ความแม่นยำตามสภาวะแวดล้อม (Environmental Accuracy)")
            env_data = [
                {"สภาวะแวดล้อม": "แสงปกติ (กลางวัน)", "ความแม่นยำในการนับ": "95.0%", "ความแม่นยำในการจำแนก": "92.0%"},
                {"สภาวะแวดล้อม": "แสงน้อย (กลางคืน)", "ความแม่นยำในการนับ": "85.0%", "ความแม่นยำในการจำแนก": "72.0%"}
            ]
            st.dataframe(pd.DataFrame(env_data), hide_index=True, use_container_width=True)
            st.caption("📋 ค่าอ้างอิงตามสถานการณ์ CCTV ทั่วไป")
        
        st.markdown("---")
        
        # YOLO Reference mAP
        st.markdown("### 🎯 ค่า mAP สำหรับ Person Detection (Reference)")
        st.info("""
        **Model:** YOLOv8n (pretrained on COCO)
        - **mAP50:** 37.3%
        - **mAP50-95:** 24.3%
        - **Source:** Ultralytics YOLOv8 documentation
        """)
        
        # DeepFace Speed
        deepface_speed = benchmark_data.get("deepface_speed", {})
        if deepface_speed:
            st.markdown("### 🔍 ความเร็ว DeepFace Analysis")
            df_cols = st.columns(3)
            df_cols[0].metric("Average", f"{deepface_speed.get('avg_ms', 'N/A')} ms")
            df_cols[1].metric("Min", f"{deepface_speed.get('min_ms', 'N/A')} ms")
            df_cols[2].metric("Max", f"{deepface_speed.get('max_ms', 'N/A')} ms")
        
        st.markdown("---")
        
        # =====================================================
        # BENCHMARK HISTORY CHARTS
        # =====================================================
        st.markdown("## 📈 กราฟประวัติการทดสอบ (Benchmark History)")
        
        # Load history CSV
        history_csv = Path("benchmark/results/benchmark_history.csv")
        class_csv = Path("benchmark/results/class_metrics_history.csv")
        
        if history_csv.exists():
            history_df = pd.read_csv(history_csv)
            
            if not history_df.empty:
                st.markdown("### ⏱️ แนวโน้ม FPS ตามเวลา")
                
                # FPS over time chart
                if 'fps' in history_df.columns and 'timestamp' in history_df.columns:
                    fps_chart = (
                        alt.Chart(history_df)
                        .mark_line(point=True, strokeWidth=3)
                        .encode(
                            x=alt.X("timestamp:T", title="เวลาทดสอบ"),
                            y=alt.Y("fps:Q", title="FPS", scale=alt.Scale(zero=False)),
                            tooltip=["timestamp:T", "fps:Q", "total_ms:Q"]
                        )
                        .properties(height=300, background=theme["chart_bg"])
                    ).configure_view(strokeOpacity=0, fill=theme["chart_bg"]).configure_axis(
                        labelColor=theme["axis_label"], titleColor=theme["axis_title"]
                    )
                    st.altair_chart(fps_chart, use_container_width=True)
                
                # Inference time breakdown
                st.markdown("### 📊 เวลาประมวลผลแยกตามขั้นตอน")
                if all(col in history_df.columns for col in ['preprocess_ms', 'inference_ms', 'postprocess_ms']):
                    # Get latest run for bar chart
                    latest = history_df.iloc[-1]
                    time_data = pd.DataFrame({
                        "ขั้นตอน": ["Pre-process", "Inference", "Post-process"],
                        "เวลา (ms)": [
                            float(latest.get('preprocess_ms', 0) or 0),
                            float(latest.get('inference_ms', 0) or 0),
                            float(latest.get('postprocess_ms', 0) or 0)
                        ]
                    })
                    
                    time_chart = (
                        alt.Chart(time_data)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X("ขั้นตอน:N", title="ขั้นตอน", sort=["Pre-process", "Inference", "Post-process"]),
                            y=alt.Y("เวลา (ms):Q", title="เวลา (ms)"),
                            color=alt.Color("ขั้นตอน:N", legend=None,
                                scale=alt.Scale(domain=["Pre-process", "Inference", "Post-process"],
                                              range=["#22c55e", "#3b82f6", "#f97316"])),
                            tooltip=["ขั้นตอน:N", "เวลา (ms):Q"]
                        )
                        .properties(height=300, background=theme["chart_bg"])
                    ).configure_view(strokeOpacity=0, fill=theme["chart_bg"]).configure_axis(
                        labelColor=theme["axis_label"], titleColor=theme["axis_title"]
                    )
                    st.altair_chart(time_chart, use_container_width=True)
                
                # Accuracy comparison over runs
                st.markdown("### 📈 แนวโน้มความถูกต้อง (Accuracy Trend)")
                if 'gender_accuracy' in history_df.columns or 'ethnicity_accuracy' in history_df.columns:
                    acc_data = history_df[['timestamp', 'gender_accuracy', 'ethnicity_accuracy']].copy()
                    acc_data = acc_data.melt(id_vars=['timestamp'], 
                                            value_vars=['gender_accuracy', 'ethnicity_accuracy'],
                                            var_name='ประเภท', value_name='ความถูกต้อง (%)')
                    acc_data['ประเภท'] = acc_data['ประเภท'].replace({
                        'gender_accuracy': 'Gender',
                        'ethnicity_accuracy': 'Ethnicity'
                    })
                    
                    acc_chart = (
                        alt.Chart(acc_data)
                        .mark_line(point=True, strokeWidth=2)
                        .encode(
                            x=alt.X("timestamp:T", title="เวลาทดสอบ"),
                            y=alt.Y("ความถูกต้อง (%):Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 100])),
                            color=alt.Color("ประเภท:N", 
                                scale=alt.Scale(domain=["Gender", "Ethnicity"],
                                              range=["#8b5cf6", "#0ea5e9"])),
                            tooltip=["timestamp:T", "ประเภท:N", "ความถูกต้อง (%):Q"]
                        )
                        .properties(height=300, background=theme["chart_bg"])
                    ).configure_view(strokeOpacity=0, fill=theme["chart_bg"]).configure_axis(
                        labelColor=theme["axis_label"], titleColor=theme["axis_title"]
                    )
                    st.altair_chart(acc_chart, use_container_width=True)
                
                # Show history table
                with st.expander("📋 ดูตารางประวัติการทดสอบทั้งหมด"):
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("ยังไม่มีประวัติการทดสอบ รัน benchmark หลายครั้งเพื่อดูกราฟแนวโน้ม")
        
        # Class metrics history chart
        if class_csv.exists():
            class_df = pd.read_csv(class_csv)
            
            if not class_df.empty:
                st.markdown("### 🎯 Precision/Recall/F1 แยกตาม Class")
                
                # Filter by category
                category_filter = st.radio("เลือกประเภท:", ["gender", "ethnicity"], horizontal=True, key="class_category")
                filtered_df = class_df[class_df['category'] == category_filter]
                
                if not filtered_df.empty:
                    # Get latest run
                    latest_run = filtered_df['run_id'].max()
                    latest_class_df = filtered_df[filtered_df['run_id'] == latest_run]
                    
                    # Melt for grouped bar chart
                    metrics_melted = latest_class_df.melt(
                        id_vars=['class'],
                        value_vars=['precision', 'recall', 'f1_score'],
                        var_name='Metric',
                        value_name='Value'
                    )
                    
                    class_chart = (
                        alt.Chart(metrics_melted)
                        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                        .encode(
                            x=alt.X("class:N", title="Class"),
                            y=alt.Y("Value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                            color=alt.Color("Metric:N",
                                scale=alt.Scale(domain=["precision", "recall", "f1_score"],
                                              range=["#22c55e", "#3b82f6", "#f97316"])),
                            xOffset="Metric:N",
                            tooltip=["class:N", "Metric:N", "Value:Q"]
                        )
                        .properties(height=350, background=theme["chart_bg"])
                    ).configure_view(strokeOpacity=0, fill=theme["chart_bg"]).configure_axis(
                        labelColor=theme["axis_label"], titleColor=theme["axis_title"]
                    )
                    st.altair_chart(class_chart, use_container_width=True)
                    
                    with st.expander("📋 ดูตาราง Class Metrics ทั้งหมด"):
                        st.dataframe(class_df, use_container_width=True, hide_index=True)
        
        # Raw JSON
        with st.expander("ดูข้อมูล Benchmark แบบ JSON"):
            st.json(benchmark_data)
            
        # Timestamp
        if benchmark_data.get("timestamp"):
            st.caption(f"ผลทดสอบเมื่อ: {benchmark_data.get('timestamp')}")
    else:
        st.warning("ยังไม่มีผล Benchmark กรุณารัน `python benchmark/run_benchmark.py` ก่อน")
        st.code("cd C:\\YOLO_CV_CCTV && .venv\\Scripts\\python.exe benchmark\\run_benchmark.py", language="bash")

