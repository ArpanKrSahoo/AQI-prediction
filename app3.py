import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0A0E1A;
    color: #E8EDF5;
  }

  /* ── Hide default Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1400px; }

  /* ── Hero Banner ── */
  .hero-wrap {
    background: linear-gradient(135deg, #0D1B3E 0%, #0A0E1A 50%, #061025 100%);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 20px;
    padding: 2.8rem 3rem 2.2rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #60A5FA, #34D399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
  }
  .hero-sub {
    color: #94A3B8;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #60A5FA;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
    letter-spacing: 1px;
  }

  /* ── Cards ── */
  .card {
    background: linear-gradient(145deg, #111827, #0F172A);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
  }
  .card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 1rem;
  }

  /* ── Number Inputs ── */
  .stNumberInput > div > div > input {
    background: #0F172A !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E8EDF5 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.1rem !important;
    padding: 0.6rem 1rem !important;
  }
  .stNumberInput > div > div > input:focus {
    border-color: rgba(59,130,246,0.5) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
  }
  label { color: #94A3B8 !important; font-size: 0.85rem !important; font-weight: 500 !important; }

  /* ── Predict Button ── */
  .stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #0EA5E9) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(29,78,216,0.35) !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(29,78,216,0.5) !important;
    background: linear-gradient(135deg, #2563EB, #0EA5E9) !important;
  }

  /* ── AQI Result Box ── */
  .aqi-result-box {
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
  }
  .aqi-value {
    font-family: 'Space Mono', monospace;
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.2rem 0;
  }
  .aqi-label {
    font-size: 1.2rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 0.5rem;
  }
  .aqi-desc { font-size: 0.9rem; color: rgba(255,255,255,0.65); margin-top: 0.4rem; }

  /* ── Stat Tiles ── */
  .stat-tile {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
    margin-bottom: 0.8rem;
  }
  .stat-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #60A5FA;
  }
  .stat-key { font-size: 0.75rem; color: #64748B; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.8px; }

  /* ── Section Divider ── */
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
    padding: 1.2rem 0 0.6rem 0;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 0.5rem;
  }

  /* ── Plotly override ── */
  .js-plotly-plot .plotly { background: transparent !important; }

  /* ── Tooltip pill ── */
  .pollutant-pill {
    display: inline-block;
    background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.2);
    color: #93C5FD;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    margin: 2px;
  }
  .stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─── AQI CATEGORY HELPER ────────────────────────────────────────────────────
def get_aqi_info(val):
    if val <= 50:
        return {"label": "Good", "emoji": "✅", "color": "#10B981", "bg": "rgba(16,185,129,0.12)",
                "border": "rgba(16,185,129,0.3)", "desc": "Air quality is satisfactory. Enjoy outdoor activities.",
                "health": "No health implications for the general population."}
    elif val <= 100:
        return {"label": "Satisfactory", "emoji": "🟡", "color": "#F59E0B", "bg": "rgba(245,158,11,0.12)",
                "border": "rgba(245,158,11,0.3)", "desc": "Acceptable quality. Sensitive groups may notice minor effects.",
                "health": "People with respiratory/heart conditions should reduce prolonged exertion."}
    elif val <= 200:
        return {"label": "Moderate", "emoji": "🟠", "color": "#F97316", "bg": "rgba(249,115,22,0.12)",
                "border": "rgba(249,115,22,0.3)", "desc": "Breathing discomfort for asthma/heart disease patients.",
                "health": "Sensitive groups should limit outdoor activity. Others may notice mild irritation."}
    elif val <= 300:
        return {"label": "Poor", "emoji": "🔴", "color": "#EF4444", "bg": "rgba(239,68,68,0.12)",
                "border": "rgba(239,68,68,0.3)", "desc": "Breathing discomfort for most people on prolonged exposure.",
                "health": "Everyone should reduce outdoor exercise. Wear N95 mask if outdoors."}
    elif val <= 400:
        return {"label": "Very Poor", "emoji": "🟣", "color": "#8B5CF6", "bg": "rgba(139,92,246,0.12)",
                "border": "rgba(139,92,246,0.3)", "desc": "Respiratory illness risk on prolonged exposure.",
                "health": "Avoid outdoor activities. Keep windows closed. Use air purifiers indoors."}
    else:
        return {"label": "Severe / Hazardous", "emoji": "☠️", "color": "#DC2626", "bg": "rgba(220,38,38,0.15)",
                "border": "rgba(220,38,38,0.5)", "desc": "Hazardous — affects healthy people; serious health risk.",
                "health": "Stay indoors. Seek medical attention if experiencing symptoms. Avoid all outdoor exposure."}

# ─── PLOTLY THEME ───────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#94A3B8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),  # default
)

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("top_8_aqi_features.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

# ─── HERO BANNER ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">🛰 REAL-TIME ENVIRONMENTAL INTELLIGENCE</div>
  <div class="hero-title">🌍 AQI Predictor</div>
  <p class="hero-sub">AI-powered Air Quality Index estimation using Random Forest Regression — <br>
  trained on CPCB data from 26 Indian cities (2015–2020).</p>
</div>
""", unsafe_allow_html=True)

# ─── MAIN LAYOUT: LEFT INPUTS | RIGHT RESULTS ───────────────────────────────
left_col, right_col = st.columns([1, 1.3], gap="large")

# ── LEFT: INPUT PANEL ──────────────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-title">◈ Pollutant Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=50.0, step=1.0, format="%.2f")
        no2  = st.number_input("NO2 (µg/m³)",   min_value=0.0, value=3.0,  step=0.5, format="%.2f")
        co   = st.number_input("CO (mg/m³)",     min_value=0.0, value=2.0,  step=0.1, format="%.2f")
        nh3  = st.number_input("NH3 (µg/m³)",   min_value=0.0, value=34.0, step=1.0, format="%.2f")
    with c2:
        pm10 = st.number_input("PM10 (µg/m³)",  min_value=0.0, value=80.0, step=1.0, format="%.2f")
        so2  = st.number_input("SO2 (µg/m³)",   min_value=0.0, value=14.0, step=0.5, format="%.2f")
        o3   = st.number_input("O3 (µg/m³)",    min_value=0.0, value=35.0, step=1.0, format="%.2f")
        temp = st.number_input("Temperature (°C)", value=30.0, step=0.5, format="%.2f")

    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("⚡  PREDICT AQI", use_container_width=True)

    # Quick reference guide
    st.markdown('<div class="section-title">◈ AQI Scale Reference</div>', unsafe_allow_html=True)
    scale_data = {
        "Range": ["0–50", "51–100", "101–200", "201–300", "301–400", "401+"],
        "Category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
        "Color": ["#10B981", "#F59E0B", "#F97316", "#EF4444", "#8B5CF6", "#DC2626"]
    }
    fig_scale = go.Figure()
    for i, (rng, cat, col) in enumerate(zip(scale_data["Range"], scale_data["Category"], scale_data["Color"])):
        fig_scale.add_trace(go.Bar(
            x=[1], y=[i], orientation="h", name=cat,
            marker_color=col, marker_line_width=0,
            text=f"  {rng}  {cat}", textposition="inside",
            textfont=dict(size=11, color="white", family="DM Sans"),
            showlegend=False, hoverinfo="skip"
        ))
    _sl = {**PLOT_LAYOUT, "margin": dict(l=0, r=0, t=8, b=8)}
    fig_scale.update_layout(
        **_sl,
        barmode="overlay",
        height=200,
        xaxis=dict(visible=False, range=[0, 1.2]),
        yaxis=dict(visible=False, autorange="reversed"),
    )
    st.plotly_chart(fig_scale, use_container_width=True, config={"displayModeBar": False})

# ── RIGHT: RESULT PANEL ────────────────────────────────────────────────────
with right_col:
    if predict_clicked:
        if model is None:
            st.error("⚠️ Model file **top_8_aqi_features.pkl** not found. Place it in the same directory as app.py.")
        else:
            features = np.array([[pm25, pm10, no2, so2, co, o3, nh3, temp]])
            pred = model.predict(features)[0]
            info = get_aqi_info(pred)

            # ── AQI Result Card ──
            st.markdown(f"""
            <div class="aqi-result-box" style="background:{info['bg']}; border:1px solid {info['border']};">
              <div style="font-size:0.75rem; letter-spacing:2px; text-transform:uppercase;
                          color:rgba(255,255,255,0.45); font-family:'Space Mono',monospace; margin-bottom:0.5rem;">
                PREDICTED AQI
              </div>
              <div class="aqi-value" style="color:{info['color']};">{pred:.1f}</div>
              <div class="aqi-label" style="color:{info['color']};">{info['emoji']} {info['label']}</div>
              <div class="aqi-desc">{info['desc']}</div>
              <div style="margin-top:1rem; padding:0.6rem 1rem; background:rgba(0,0,0,0.2);
                          border-radius:8px; font-size:0.82rem; color:rgba(255,255,255,0.55); text-align:left;">
                🩺 <strong>Health Advisory:</strong> {info['health']}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Stats Row ──
            s1, s2, s3 = st.columns(3)
            dominant = {"PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2,
                        "CO×10": co*10, "O3": o3, "NH3": nh3}
            dom_poll = max(dominant, key=dominant.get)
            with s1:
                st.markdown(f'<div class="stat-tile"><div class="stat-val">{pred:.0f}</div><div class="stat-key">AQI Value</div></div>', unsafe_allow_html=True)
            with s2:
                st.markdown(f'<div class="stat-tile"><div class="stat-val">{info["label"][:4]}.</div><div class="stat-key">Category</div></div>', unsafe_allow_html=True)
            with s3:
                st.markdown(f'<div class="stat-tile"><div class="stat-val">{dom_poll}</div><div class="stat-key">Dominant Pollutant</div></div>', unsafe_allow_html=True)

            # ── Gauge Chart ──
            st.markdown('<div class="section-title">◈ AQI Gauge</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"font": {"size": 48, "family": "Space Mono", "color": info["color"]},
                        "suffix": " AQI"},
                gauge={
                    "axis": {"range": [0, 500], "tickcolor": "#475569",
                             "tickfont": {"size": 10, "color": "#64748B"},
                             "dtick": 100},
                    "bar": {"color": info["color"], "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50],   "color": "rgba(16,185,129,0.15)"},
                        {"range": [50, 100], "color": "rgba(245,158,11,0.15)"},
                        {"range": [100, 200],"color": "rgba(249,115,22,0.15)"},
                        {"range": [200, 300],"color": "rgba(239,68,68,0.15)"},
                        {"range": [300, 400],"color": "rgba(139,92,246,0.15)"},
                        {"range": [400, 500],"color": "rgba(220,38,38,0.2)"},
                    ],
                    "threshold": {"line": {"color": info["color"], "width": 3},
                                  "thickness": 0.75, "value": pred}
                }
            ))
            fig_gauge.update_layout(**PLOT_LAYOUT, height=260)
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            # ── Pollutant Breakdown Radar ──
            st.markdown('<div class="section-title">◈ Pollutant Profile Radar</div>', unsafe_allow_html=True)
            poll_names  = ["PM2.5", "PM10", "NO2", "SO2", "CO×10", "O3", "NH3"]
            poll_vals   = [pm25, pm10, no2, so2, co*10, o3, nh3]
            # Normalize to 0-100 for radar
            max_vals    = [200, 300, 150, 80, 200, 150, 100]
            norm_vals   = [min(v/m*100, 100) for v, m in zip(poll_vals, max_vals)]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_vals + [norm_vals[0]],
                theta=poll_names + [poll_names[0]],
                fill="toself",
                fillcolor=f"rgba({','.join(str(int(info['color'].lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15)",
                line=dict(color=info["color"], width=2),
                marker=dict(size=6, color=info["color"]),
                name="Pollutant Levels"
            ))
            fig_radar.update_layout(
                **PLOT_LAYOUT,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    angularaxis=dict(tickcolor="#475569", gridcolor="rgba(255,255,255,0.06)",
                                    tickfont=dict(size=11, color="#94A3B8")),
                    radialaxis=dict(visible=True, range=[0, 100],
                                   gridcolor="rgba(255,255,255,0.05)",
                                   tickfont=dict(size=9, color="#475569"),
                                   ticksuffix="%")
                ),
                showlegend=False,
                height=290,
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    else:
        # ── Placeholder state ──
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.08);
                    border-radius:20px; padding:4rem 2rem; text-align:center; margin-bottom:1.5rem;">
          <div style="font-size:3.5rem; margin-bottom:1rem;">🌬️</div>
          <div style="font-family:'Space Mono',monospace; font-size:1.1rem; color:#475569; letter-spacing:1px;">
            AWAITING INPUT
          </div>
          <div style="color:#334155; font-size:0.85rem; margin-top:0.5rem;">
            Enter pollutant values and click Predict
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─── BOTTOM CHARTS SECTION ───────────────────────────────────────────────────
if predict_clicked and model is not None:
    st.markdown('<div class="section-title">◈ Detailed Analysis</div>', unsafe_allow_html=True)

    b1, b2 = st.columns(2, gap="large")

    with b1:
        # ── Horizontal Bar: Pollutant Contributions ──
        poll_df = pd.DataFrame({
            "Pollutant": ["PM2.5", "PM10", "NO₂", "SO₂", "CO", "O₃", "NH₃", "Temp"],
            "Value":     [pm25, pm10, no2, so2, co, o3, nh3, temp],
            "Limit":     [60, 100, 80, 80, 4, 100, 100, 45],  # NAAQS safe limits
            "Unit":      ["µg/m³","µg/m³","µg/m³","µg/m³","mg/m³","µg/m³","µg/m³","°C"]
        })
        poll_df["% of Limit"] = (poll_df["Value"] / poll_df["Limit"] * 100).clip(upper=200)
        poll_df["Color"] = poll_df["% of Limit"].apply(
            lambda x: "#10B981" if x < 60 else "#F59E0B" if x < 100 else "#EF4444"
        )

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=poll_df["% of Limit"],
            y=poll_df["Pollutant"],
            orientation="h",
            marker_color=poll_df["Color"],
            marker_line_width=0,
            text=[f"{v:.1f} {u}" for v, u in zip(poll_df["Value"], poll_df["Unit"])],
            textposition="outside",
            textfont=dict(size=10.5, color="#94A3B8"),
        ))
        fig_bar.add_vline(x=100, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                          annotation_text="SAFE LIMIT", annotation_font_size=9,
                          annotation_font_color="#475569")
        fig_bar.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="Pollutants vs Safe Limits (%)", font=dict(size=13, color="#94A3B8"), x=0),
            xaxis=dict(title="% of NAAQS Safe Limit", gridcolor="rgba(255,255,255,0.04)",
                       tickfont=dict(color="#64748B"), range=[0, max(poll_df["% of Limit"].max() * 1.25, 130)]),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#CBD5E1", size=12)),
            height=320,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with b2:
        # ── Donut: AQI Breakdown by contribution estimate ──
        contrib_labels = ["PM2.5", "PM10", "NO₂", "SO₂", "CO", "O₃", "NH₃"]
        weights = [0.30, 0.22, 0.12, 0.10, 0.10, 0.08, 0.08]
        raw_vals = [pm25, pm10, no2, so2, co*10, o3, nh3]
        contrib_vals = [w * v for w, v in zip(weights, raw_vals)]
        total = sum(contrib_vals) or 1
        contrib_pct = [v/total*100 for v in contrib_vals]

        donut_colors = ["#60A5FA","#34D399","#F59E0B","#F97316","#EF4444","#8B5CF6","#EC4899"]
        fig_donut = go.Figure(go.Pie(
            labels=contrib_labels,
            values=contrib_pct,
            hole=0.62,
            marker=dict(colors=donut_colors, line=dict(color="#0A0E1A", width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, color="#E2E8F0"),
            hovertemplate="<b>%{label}</b><br>Contribution: %{percent}<extra></extra>",
            showlegend=False,
        ))
        fig_donut.add_annotation(
            text=f"<b>{pred:.0f}</b><br><span style='font-size:10px'>AQI</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=22, color=info["color"], family="Space Mono"),
            align="center"
        )
        fig_donut.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="Estimated Pollutant Contribution", font=dict(size=13, color="#94A3B8"), x=0),
            height=320,
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    # ── Comparison: Your values vs national average ──
    st.markdown('<div class="section-title">◈ Your Input vs National Average (India 2020)</div>', unsafe_allow_html=True)

    nat_avg = {"PM2.5": 67.5, "PM10": 118.1, "NO2": 28.6, "SO2": 14.5, "CO": 2.2, "O3": 34.5, "NH3": 23.5}
    user_vals = {"PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2, "CO": co, "O3": o3, "NH3": nh3}

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Your Input",
        x=list(nat_avg.keys()),
        y=list(user_vals.values()),
        marker_color=info["color"],
        marker_line_width=0,
        opacity=0.9,
    ))
    fig_comp.add_trace(go.Bar(
        name="National Average",
        x=list(nat_avg.keys()),
        y=list(nat_avg.values()),
        marker_color="rgba(148,163,184,0.3)",
        marker_line_width=0,
        opacity=0.9,
    ))
    fig_comp.update_layout(
        **PLOT_LAYOUT,
        barmode="group",
        bargap=0.2,
        bargroupgap=0.05,
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#CBD5E1", size=12)),
        yaxis=dict(title="Concentration", gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748B")),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#94A3B8", size=12)),
        height=280,
    )
    st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding-top:1.5rem; border-top:1px solid rgba(255,255,255,0.06);
            text-align:center; color:#334155; font-size:0.78rem; font-family:'Space Mono',monospace;">
  Built by <span style="color:#60A5FA;">Arpan Kumar Sahoo</span> · Roll-28 · CSE IoT · 
  Enrollment 12024052019023 &nbsp;|&nbsp; 
  Model: Random Forest · Trained on CPCB India Dataset (2015–2020)
</div>
""", unsafe_allow_html=True)
