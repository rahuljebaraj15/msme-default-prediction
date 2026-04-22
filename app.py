import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import json
import time
import math

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MSME Risk Predictor",
    page_icon="🏦",
    layout="centered",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  — dark terminal aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── Reset & root ── */
:root {
  --bg:       #03080F;
  --surface:  #080F1A;
  --surface2: #0D1826;
  --border:   rgba(0,200,255,0.12);
  --cyan:     #00C8FF;
  --green:    #00FFB2;
  --amber:    #FFB800;
  --red:      #FF3B5C;
  --text:     #E8F4FF;
  --muted:    #5A7A9A;
}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* grid overlay */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(0,200,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,200,255,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
}

/* glow orbs */
[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed; top: -100px; right: -100px;
  width: 400px; height: 400px; border-radius: 50%;
  background: rgba(0,200,255,0.06);
  filter: blur(120px); pointer-events: none; z-index: 0;
}

[data-testid="stMain"] > div { position: relative; z-index: 1; }

/* sidebar hidden if not used */
[data-testid="stSidebar"] { display: none; }

/* remove default top padding */
.block-container { padding-top: 2rem !important; max-width: 860px !important; }

/* ── Headings ── */
h1, h2, h3 {
  font-family: 'Syne', sans-serif !important;
  color: var(--text) !important;
}

/* ── Streamlit divider ── */
hr { border-color: var(--border) !important; margin: 1.4rem 0 !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
  background: linear-gradient(90deg, var(--cyan), var(--green)) !important;
}
[data-testid="stSlider"] [role="slider"] {
  background: var(--cyan) !important;
  box-shadow: 0 0 10px var(--cyan) !important;
  border: none !important;
}
[data-testid="stSlider"] p {
  font-family: 'DM Mono', monospace !important;
  color: var(--muted) !important;
  font-size: 11px !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
}

/* ── Number input ── */
[data-testid="stNumberInput"] label {
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
[data-testid="stNumberInput"] input {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
}
[data-testid="stNumberInput"] input:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px rgba(0,200,255,0.15) !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
  width: 100% !important;
  padding: 14px !important;
  background: linear-gradient(135deg, #00C8FF, #00FFB2) !important;
  color: #03080F !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 16px !important;
  font-weight: 700 !important;
  letter-spacing: 1px !important;
  border: none !important;
  border-radius: 14px !important;
  transition: transform 0.2s, box-shadow 0.2s !important;
  cursor: pointer !important;
}
[data-testid="stButton"] > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px rgba(0,200,255,0.4) !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'DM Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 28px !important;
  font-weight: 800 !important;
  color: var(--cyan) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
  font-family: 'DM Mono', monospace !important;
  color: var(--muted) !important;
  font-size: 12px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
}

/* ── Caption / footer ── */
[data-testid="stCaptionContainer"] p {
  font-family: 'DM Mono', monospace !important;
  font-size: 10px !important;
  color: var(--muted) !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  text-align: center !important;
  opacity: 0.6 !important;
}

/* ── Toast / success / warning / error boxes ── */
[data-testid="stAlert"] {
  border-radius: 12px !important;
  font-family: 'DM Sans', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS — reusable HTML components
# ─────────────────────────────────────────────

def section_label(num: str, title: str):
    st.markdown(f"""
    <div style="
      font-family:'DM Mono',monospace; font-size:10px; letter-spacing:3px;
      text-transform:uppercase; color:#00C8FF; margin:24px 0 14px;
      display:flex; align-items:center; gap:10px;">
      {num} · {title}
      <span style="flex:1; height:1px;
        background:linear-gradient(90deg,rgba(0,200,255,0.15),transparent);
        display:inline-block; vertical-align:middle;"></span>
    </div>""", unsafe_allow_html=True)


def badge_row():
    st.markdown("""
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin:14px 0 28px;">
      <span style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:1.5px;
        padding:5px 12px; border-radius:20px; border:1px solid #00C8FF44;
        color:#00C8FF; background:#00C8FF0A; text-transform:uppercase;">
        Logistic Regression</span>
      <span style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:1.5px;
        padding:5px 12px; border-radius:20px; border:1px solid #00FFB244;
        color:#00FFB2; background:#00FFB20A; text-transform:uppercase;">
        SMOTE Balanced</span>
      <span style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:1.5px;
        padding:5px 12px; border-radius:20px; border:1px solid #FFB80044;
        color:#FFB800; background:#FFB8000A; text-transform:uppercase;">
        Threshold 0.48</span>
    </div>""", unsafe_allow_html=True)


def arc_gauge(prob: float):
    """
    Renders the arc gauge inside a components.html() iframe — the only way
    to guarantee SVG renders correctly in Streamlit (st.markdown strips SVG).
    """
    pct   = max(0.0, min(1.0, prob))
    angle = pct * 180
    r, cx, cy = 90, 110, 110

    def to_xy(deg):
        rad = math.radians(deg - 180)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    ex, ey = to_xy(angle)
    large  = 1 if angle > 180 else 0
    color  = "#FF3B5C" if pct > 0.6 else ("#FFB800" if pct > 0.4 else "#00FFB2")

    dots_svg = ""
    for t, dc in [(0, "#00FFB2"), (0.4, "#FFB800"), (0.6, "#FFB800"), (1.0, "#FF3B5C")]:
        dx, dy = to_xy(t * 180)
        dots_svg += f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="3" fill="{dc}" opacity="0.7"/>\n'

    arc_path_d = f"M 20 110 A 90 90 0 {large} 1 {ex:.2f} {ey:.2f}" if pct > 0.001 else ""
    arc_el = (
        f'<path d="{arc_path_d}" fill="none" stroke="{color}" stroke-width="12" '
        f'stroke-linecap="round" filter="url(#glow)"/>'
        if arc_path_d else ""
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <link href="https://fonts.googleapis.com/css2?family=Syne:wght@800&family=DM+Mono:wght@400&display=swap" rel="stylesheet">
      <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ background:#080F1A; display:flex; flex-direction:column;
                align-items:center; justify-content:center;
                height:240px; overflow:hidden; }}
        .label {{ font-family:'DM Mono',monospace; font-size:10px;
                  letter-spacing:3px; text-transform:uppercase;
                  color:#5A7A9A; margin-bottom:14px; }}
        .sub   {{ font-family:'DM Mono',monospace; font-size:11px;
                  letter-spacing:1px; color:#5A7A9A; margin-top:10px; }}
        .gauge-wrap {{ position:relative; width:220px; height:120px; }}
        .big-num {{
          position:absolute; bottom:0; left:50%; transform:translateX(-50%);
          font-family:'Syne',sans-serif; font-size:46px; font-weight:800;
          color:{color}; line-height:1; white-space:nowrap;
          text-shadow: 0 0 20px {color}66;
        }}
        .pct-unit {{ font-size:20px; color:#5A7A9A; font-weight:400; }}
      </style>
    </head>
    <body>
      <p class="label">Probability of Default</p>
      <div class="gauge-wrap">
        <svg width="220" height="120" viewBox="0 0 220 120" style="overflow:visible">
          <defs>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="blur"/>
              <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
          </defs>
          <!-- track -->
          <path d="M 20 110 A 90 90 0 0 1 200 110"
            fill="none" stroke="#0D1826" stroke-width="12" stroke-linecap="round"/>
          <!-- filled arc -->
          {arc_el}
          <!-- needle tip -->
          <circle cx="{ex:.2f}" cy="{ey:.2f}" r="7" fill="{color}" filter="url(#glow)"/>
          <!-- zone markers -->
          {dots_svg}
        </svg>
        <div class="big-num">
          {pct*100:.1f}<span class="pct-unit">%</span>
        </div>
      </div>
      <p class="sub">Default Probability Score</p>
    </body>
    </html>
    """
    components.html(html, height=240)


def status_card(level: str, prob: float):
    cfg = {
        "low":  ("#00FFB2", "#00FFB208", "#00FFB244", "🟢",
                 "LOW RISK BORROWER",
                 "Borrower demonstrates stable financial behavior with low default probability."),
        "med":  ("#FFB800", "#FFB80008", "#FFB80044", "🟡",
                 "MODERATE RISK BORROWER",
                 "Some indicators suggest caution. Conditional approval recommended."),
        "high": ("#FF3B5C", "#FF3B5C08", "#FF3B5C44", "🔴",
                 "HIGH RISK BORROWER",
                 "Significant financial stress indicators detected. High default probability."),
    }
    c, bg, border, icon, title, desc = cfg[level]
    st.markdown(f"""
    <div style="
      padding:20px 24px; border-radius:14px;
      border:1px solid {border}; background:{bg};
      display:flex; align-items:center; gap:16px; margin:16px 0;">
      <span style="font-size:32px;">{icon}</span>
      <div>
        <div style="font-family:'Syne',sans-serif; font-size:18px;
          font-weight:700; color:{c};">{title}</div>
        <div style="font-family:'DM Sans',sans-serif; font-size:13px;
          color:#5A7A9A; margin-top:3px;">{desc}</div>
      </div>
    </div>""", unsafe_allow_html=True)


def factor_bar(name: str, val: float):
    pct = val * 100
    color = ("#FF3B5C" if val > 0.6 else ("#FFB800" if val > 0.35 else "#00FFB2"))
    grad  = f"linear-gradient(90deg, {color}88, {color})"
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
      <span style="font-family:'DM Mono',monospace; font-size:11px;
        color:#5A7A9A; width:160px; flex-shrink:0; overflow:hidden;
        text-overflow:ellipsis; white-space:nowrap; letter-spacing:0.5px;">
        {name}</span>
      <div style="flex:1; height:6px; background:#0D1826; border-radius:3px; overflow:hidden;">
        <div style="width:{pct:.0f}%; height:100%; border-radius:3px;
          background:{grad}; transition:width 1s ease;"></div>
      </div>
      <span style="font-family:'DM Mono',monospace; font-size:11px;
        color:#5A7A9A; width:36px; text-align:right;">{pct:.0f}</span>
    </div>""", unsafe_allow_html=True)


def rec_grid(level: str):
    recs = {
        "low": [
            ("✅", "Safe to approve — standard lending terms apply"),
            ("📉", "Low monitoring cadence required"),
            ("💼", "Eligible for premium rate packages"),
            ("🔓", "Minimal collateral requirements"),
        ],
        "med": [
            ("⚙️", "Approve with conditional clauses"),
            ("🔒", "Collateral or guarantor recommended"),
            ("📊", "Reduce initial loan exposure amount"),
            ("🔁", "Quarterly credit review schedule"),
        ],
        "high": [
            ("🚫", "Reject or require strong collateral"),
            ("🔍", "Initiate full credit investigation"),
            ("📉", "Limit maximum loan exposure"),
            ("⚠️", "Flag for enhanced due diligence"),
        ],
    }
    border_color = {"low": "#00FFB288", "med": "#FFB80088", "high": "#FF3B5C88"}[level]
    items = recs[level]
    cols = st.columns(2)
    for i, (icon, text) in enumerate(items):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="
              background:#0D1826; border-radius:12px; padding:14px 16px;
              border-left:3px solid {border_color};
              display:flex; gap:10px; align-items:flex-start;
              font-size:13px; font-family:'DM Sans',sans-serif;
              color:#B8D0E8; margin-bottom:10px;">
              <span style="font-size:16px; margin-top:1px; flex-shrink:0;">{icon}</span>
              <span>{text}</span>
            </div>""", unsafe_allow_html=True)


def card_open():
    st.markdown("""
    <div style="
      background:#080F1A; border:1px solid rgba(0,200,255,0.12);
      border-radius:16px; padding:24px; margin-bottom:16px;
      position:relative; overflow:hidden;">
      <div style="position:absolute; top:0; left:0; right:0; height:1px;
        background:linear-gradient(90deg,transparent,#00C8FF,transparent);
        opacity:0.3;"></div>
    """, unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL ASSETS
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model  = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("features.json") as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, features = load_assets()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:6px;">
  <div style="
    width:52px; height:52px; border-radius:14px;
    background:linear-gradient(135deg,#00C8FF22,#00FFB222);
    border:1px solid rgba(0,200,255,0.12);
    display:flex; align-items:center; justify-content:center; font-size:24px;">
    🏦
  </div>
  <div>
    <h1 style="
      font-family:'Syne',sans-serif; font-size:clamp(22px,4vw,30px);
      font-weight:800; margin:0; line-height:1.1;
      background:linear-gradient(90deg,#E8F4FF,#00C8FF);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
      MSME Risk Predictor
    </h1>
    <p style="
      font-family:'DM Mono',monospace; font-size:11px; letter-spacing:2px;
      text-transform:uppercase; color:#5A7A9A; margin:4px 0 0;">
      Decision Support System · Credit Risk Evaluation
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

badge_row()

# ─────────────────────────────────────────────
# SECTION 01 — BORROWER PROFILE
# ─────────────────────────────────────────────
section_label("01", "Borrower Profile")

user_input = {}

feature_labels = {
    "RevolvingUtilizationOfUnsecuredLines": ("Credit Usage (%)",      "slider",  0,   100, 30),
    "age":                                  ("Age",                    "slider",  18,   80, 35),
    "MonthlyIncome":                        ("Monthly Income (₹)",     "number",  None, None, 20000),
    "DebtRatio":                            ("Debt Ratio (%)",         "slider",  0,   200, 50),
    "NumberOfOpenCreditLinesAndLoans":      ("Total Active Loans",     "number",  None, None, 1),
    "NumberOfTime30-59DaysPastDueNotWorse": ("Late Payments 30–59 days","number", None, None, 0),
    "NumberOfTimes90DaysLate":              ("Late Payments 90+ days", "number",  None, None, 0),
    "NumberOfTime60-89DaysPastDueNotWorse": ("Late Payments 60–89 days","number", None, None, 0),
    "NumberRealEstateLoansOrLines":         ("Real Estate Loans",      "number",  None, None, 0),
    "NumberOfDependents":                   ("Dependents",             "number",  None, None, 0),
}

# Two-column layout
col1, col2 = st.columns(2, gap="medium")
cols = [col1, col2]

for idx, feat_key in enumerate(features):
    label, kind, mn, mx, default = feature_labels.get(
        feat_key, (feat_key, "number", None, None, 0)
    )
    with cols[idx % 2]:
        if kind == "slider":
            raw = st.slider(label, min_value=mn, max_value=mx, value=default, key=feat_key)
            # Convert slider % values back to model scale
            if feat_key == "RevolvingUtilizationOfUnsecuredLines":
                user_input[feat_key] = raw / 100.0
            elif feat_key == "DebtRatio":
                user_input[feat_key] = raw / 100.0
            else:
                user_input[feat_key] = raw
        else:
            step = 1000 if feat_key == "MonthlyIncome" else 1
            user_input[feat_key] = st.number_input(
                label, min_value=0, value=default, step=step, key=feat_key
            )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ASSESS BUTTON
# ─────────────────────────────────────────────
assess = st.button("🔍  Assess Credit Risk", use_container_width=True)

# ─────────────────────────────────────────────
# PREDICTION & RESULTS
# ─────────────────────────────────────────────
if assess:
    with st.spinner("Running Risk Model..."):
        time.sleep(1.4)

    input_df     = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prob         = model.predict_proba(input_scaled)[0][1]

    level = "high" if prob > 0.6 else ("med" if prob > 0.4 else "low")

    # ── 02  Risk Assessment ──────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_label("02", "Risk Assessment")

    # Arc gauge — rendered in sandboxed iframe (components.html handles SVG safely)
    arc_gauge(prob)

    # Probability metric row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Default Probability", f"{prob*100:.2f}%")
    with m2:
        st.metric("Risk Tier", level.upper())
    with m3:
        st.metric("Decision Threshold", "0.48")

    # Status card
    status_card(level, prob)

    # ── 03  Factor Analysis ──────────────────
    section_label("03", "Factor Analysis")

    util_val  = user_input.get("RevolvingUtilizationOfUnsecuredLines", 0)
    debt_val  = user_input.get("DebtRatio", 0)
    late90    = user_input.get("NumberOfTimes90DaysLate", 0)
    late30    = user_input.get("NumberOfTime30-59DaysPastDueNotWorse", 0)
    income    = user_input.get("MonthlyIncome", 20000)
    loans     = user_input.get("NumberOfOpenCreditLinesAndLoans", 0)

    factors = [
        ("Credit Utilization",   min(util_val, 1.0)),
        ("Debt Burden",          min(debt_val, 1.0)),
        ("Late Payment History", min((late90 + late30) / 10, 1.0)),
        ("Income Stability",     1 - min(income / 100_000, 1.0)),
        ("Loan Exposure",        min(loans / 15, 1.0)),
    ]

    st.markdown("""
    <div style="background:#080F1A; border:1px solid rgba(0,200,255,0.12);
      border-radius:16px; padding:24px; margin-bottom:16px; position:relative;">
      <div style="position:absolute;top:0;left:0;right:0;height:1px;
        background:linear-gradient(90deg,transparent,#00C8FF,transparent);opacity:0.3;"></div>
    """, unsafe_allow_html=True)
    for name, val in factors:
        factor_bar(name, val)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 04  Bank Decision ────────────────────
    section_label("04", "Bank Decision")
    rec_grid(level)

    # Toast-style feedback
    st.markdown("<br>", unsafe_allow_html=True)
    if level == "low":
        st.success("✅  Low risk profile — safe to proceed with standard terms.")
    elif level == "med":
        st.warning("🟡  Moderate risk detected — conditional approval recommended.")
    else:
        st.error("🔴  High risk alert — detailed investigation required before proceeding.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("MODEL: LOGISTIC REGRESSION + SMOTE  ·  THRESHOLD = 0.48  ·  FOR DECISION SUPPORT ONLY")
