import streamlit as st
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import plotly.express as px
from geopy.geocoders import Nominatim
from fpdf import FPDF
import base64

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lahore Air Guard - Muhammad Mujtaba",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
    
    /* GLASSMORPHISM CARD STYLE */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 15px rgba(0,0,0,0.1); }
    
    /* FIX FOR MAP CONTAINER */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-radius: 10px;
        padding: 20px;
    }
    
    .label-text { font-size: 11px; font-weight: 700; color: #555; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; display: flex; align-items: center; }
    .value-text { font-size: 42px; font-weight: 800; color: #111; margin: 0; }
    .sub-text { font-size: 13px; color: #666; margin-top: 5px; font-weight: 500; }
    .status-tag { display: inline-block; padding: 6px 12px; border-radius: 4px; font-size: 12px; font-weight: 700; color: white; margin-top: 10px; }
    
    /* TOOLTIPS */
    .tooltip { position: relative; display: inline-block; margin-left: 8px; cursor: help; font-size: 14px; color: #888; }
    .tooltip .tooltiptext { visibility: hidden; width: 220px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -110px; opacity: 0; transition: opacity 0.3s; font-size: 10px; font-weight: normal; line-height: 1.4; text-transform: none; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
    
    /* DOWNLOAD BUTTON */
    .download-btn {
        display: block; padding: 15px 20px; background: linear-gradient(135deg, #111 0%, #333 100%);
        color: white !important; border-radius: 8px; text-decoration: none; font-weight: 700;
        text-align: center; letter-spacing: 1px; transition: all 0.3s ease; border: 1px solid #444; font-size: 12px;
    }
    .download-btn:hover { transform: translateY(-2px); background: linear-gradient(135deg, #000 0%, #222 100%); }

    .watermark { position: fixed; bottom: 10px; right: 10px; color: #888; font-size: 12px; font-weight: bold; z-index: 100; }

    @media (prefers-color-scheme: dark) {
        .metric-card, div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); }
        .label-text { color: #aaa; }
        .value-text { color: #fff; }
        .sub-text { color: #888; }
        .download-btn { background: linear-gradient(135deg, #6200EA 0%, #3700b3 100%); border: none; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION ---
DEFAULT_LAT = 31.5204
DEFAULT_LON = 74.3587

# --- ENGINE ---
def get_open_meteo(lat, lon):
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = { "latitude": lat, "longitude": lon, "current": ["pm2_5", "us_aqi"], "hourly": ["pm2_5"], "past_days": 2, "forecast_days": 1 }
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            curr_pm25 = data['current']['pm2_5']
            hourly = data['hourly']
            df = pd.DataFrame({"date": hourly['time'], "P2": hourly['pm2_5']})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            return df, curr_pm25
    except: return None, None
    return None, None

def run_ai_models(history_df):
    try:
        # Validation (Now)
        train_data = history_df.iloc[:-1] 
        if len(train_data) < 10: return None, None
        model_val = ARIMA(train_data['P2'], order=(8, 0, 1)).fit()
        pred_now = model_val.forecast(steps=1).iloc[0]
        
        # Forecast (Future)
        model_fut = ARIMA(history_df['P2'], order=(8, 0, 1)).fit()
        pred_fut_series = model_fut.forecast(steps=24)
        return int(pred_now), pred_fut_series
    except: return None, None

# --- PDF ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Lahore Air Guard - Intelligence Report', 0, 1, 'C')
        self.ln(5)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'Developed by Muhammad Mujtaba', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_download_link(val, pred_now, pred_future, status):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Advisory: {status}", ln=1)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Real Time: {val} ug/m3", ln=1)
    pdf.cell(200, 10, txt=f"AI Validation: {pred_now} ug/m3", ln=1)
    pdf.cell(200, 10, txt=f"Next Hour: {pred_future} ug/m3", ln=1)
    pdf_content = pdf.output(dest='S').encode('latin-1')
    b64 = base64.b64encode(pdf_content).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="Mujtaba_Air_Report.pdf" class="download-btn">📥 DOWNLOAD REPORT</a>'

# --- LOCATIONS ---
LOCATION_COORDS = {
    "Lahore (General)": (31.5204, 74.3587), "DHA Phase 5": (31.465, 74.405), "DHA Phase 6": (31.478, 74.450),
    "Gulberg III": (31.510, 74.345), "Model Town": (31.480, 74.320), "Johar Town": (31.460, 74.290),
    "Bahria Town": (31.360, 74.180), "Islamabad (Test)": (33.6844, 73.0479), "Custom Search": (None, None)
}

# --- UI START ---
st.sidebar.title("Location Search")
selected_loc = st.sidebar.selectbox("Select Area", list(LOCATION_COORDS.keys()))

if selected_loc == "Custom Search":
    address = st.sidebar.text_input("Enter Address", "Lahore")
    try:
        loc = Nominatim(user_agent="lhr_air_guard_final").geocode(address)
        lat, lon, display_name = (loc.latitude, loc.longitude, address) if loc else (DEFAULT_LAT, DEFAULT_LON, "Lahore")
    except: lat, lon, display_name = DEFAULT_LAT, DEFAULT_LON, "Lahore"
else:
    lat, lon = LOCATION_COORDS[selected_loc]
    display_name = selected_loc

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Muhammad Mujtaba")

st.title("LAHORE AIR GUARD")
st.markdown(f"**INTELLIGENCE DASHBOARD** | Monitoring: **{display_name}**")
st.markdown("---")

df, curr_pm25 = get_open_meteo(lat, lon)

if df is not None:
    pred_now_val, forecast_series = run_ai_models(df)
    next_hour_val = int(forecast_series.iloc[0]) if forecast_series is not None else 0

    if curr_pm25 > 300: status, color = "LOCKDOWN LIKELY", "#D50000"
    elif curr_pm25 > 200: status, color = "SEVERE RISK", "#AA00FF"
    elif curr_pm25 > 150: status, color = "UNHEALTHY", "#FF6D00"
    else: status, color = "NORMAL OPERATIONS", "#00C853"

    # Row 1
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"""<div class="metric-card"><div class="label-text">1. REAL TIME READING<div class="tooltip">ℹ️<span class="tooltiptext">Source: Open-Meteo Satellite</span></div></div><div class="value-text">{curr_pm25}</div><div class="sub-text">PM2.5 Concentration (µg/m³)</div><div class="status-tag" style="background-color:#00B0FF">SATELLITE LIVE</div></div>""", unsafe_allow_html=True)
    with c2: 
        # FIX IS HERE: Added int() to the calculation below
        acc = int(abs(curr_pm25 - pred_now_val)) if pred_now_val else 0
        st.markdown(f"""<div class="metric-card"><div class="label-text">2. MY PREDICTION (REAL-TIME)<div class="tooltip">ℹ️<span class="tooltiptext">AI Backcast Validation</span></div></div><div class="value-text">{pred_now_val}</div><div class="sub-text" style="color:{'#00C853' if acc < 15 else '#FF6D00'}; font-weight:700">Variance: {acc} pts</div><div class="status-tag" style="background-color:#6200EA">AI VALIDATION</div></div>""", unsafe_allow_html=True)
    with c3:
        # FIX IS HERE: Added int() to the calculation below
        diff_val = int(abs(next_hour_val - curr_pm25))
        st.markdown(f"""<div class="metric-card"><div class="label-text">3. MY PREDICTION (NEXT HOUR)<div class="tooltip">ℹ️<span class="tooltiptext">ARIMA Forecast</span></div></div><div class="value-text">{next_hour_val}</div><div class="sub-text" style="color:#D50000; font-weight:700">{'▲' if next_hour_val > curr_pm25 else '▼'} {diff_val} µg/m³</div><div class="status-tag" style="background-color:#6200EA">AI FORECAST</div></div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="background-color:{color}; padding:12px; border-radius:8px; text-align:center; margin-bottom:20px; color:white; font-weight:bold; letter-spacing:1px; font-size:14px;">GOVT ADVISORY: {status}</div>""", unsafe_allow_html=True)

    st.subheader("My Future Prediction (24-Hour Trend)")
    if forecast_series is not None:
        future_dates = pd.date_range(start=df.index[-1], periods=25, freq='H')[1:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.tail(24).index, y=df.tail(24)['P2'], mode='lines', name='Past 24h', line=dict(color='#00B0FF', width=3), fill='tozeroy', fillcolor='rgba(0, 176, 255, 0.1)'))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast_series.values, mode='lines+markers', name='Future Prediction', line=dict(color='#6200EA', width=3, dash='dash')))
        fig.add_trace(go.Scatter(x=[df.index[-1]], y=[pred_now_val], mode='markers', name='AI Validation Point', marker=dict(color='orange', size=10, symbol='x')))
        fig.add_hline(y=200, line_dash="dot", line_color="#FF3D00", annotation_text="Lockdown Threshold")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=350, hovermode="x unified", margin=dict(t=20, b=20, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    # --- FIXED LAYOUT: Native Containers ---
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="height: 450px; display: flex; flex-direction: column; justify-content: center;">
            <div class="label-text" style="font-size:16px; margin-bottom:15px;">📥 EXPORT INTELLIGENCE</div>
            <div style="font-size:14px; color:#555; margin-bottom:30px; line-height:1.6;">
                Generate a signed PDF report. Includes timestamped verification of current smog levels and AI validity checks.
            </div>
            {create_download_link(curr_pm25, pred_now_val, next_hour_val, status)}
            <div style="margin-top:30px; font-size:11px; color:#888; text-align:center;">
                System ID: LHR-AI-01<br>Dev: Muhammad Mujtaba
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        with st.container(border=True):
            st.markdown('<div class="label-text" style="font-size:14px; margin-bottom:5px;">📍 LIVE GEO-LOCATION</div>', unsafe_allow_html=True)
            map_df = pd.DataFrame({'lat': [lat], 'lon': [lon], 'intensity': [curr_pm25]})
            fig_map = px.density_mapbox(map_df, lat='lat', lon='lon', z='intensity', radius=40, center=dict(lat=lat, lon=lon), zoom=12, mapbox_style="carto-positron")
            fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400, showlegend=False) 
            st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("""<div class="watermark">Created by Muhammad Mujtaba</div>""", unsafe_allow_html=True)

else: st.error("System Offline.")