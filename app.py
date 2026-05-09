
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import os
from datetime import datetime

BASE = "/content/xai_trading"

STOCKS = {
    "HDFCBANK_NS": {"name": "HDFC Bank", "ticker": "HDFCBANK.NS", "sector": "Banking", "desc": "India's largest private bank. Benchmark for banking sector performance."},
    "ICICIBANK_NS": {"name": "ICICI Bank", "ticker": "ICICIBANK.NS", "sector": "Banking", "desc": "Fastest growing large private bank with strong retail and corporate book."},
    "SBIN_NS": {"name": "State Bank of India", "ticker": "SBIN.NS", "sector": "Banking", "desc": "India's largest bank. Proxy for entire Indian economy and credit growth."},
    "AXISBANK_NS": {"name": "Axis Bank", "ticker": "AXISBANK.NS", "sector": "Banking", "desc": "Third largest private bank with strong turnaround story."},
    "KOTAKBANK_NS": {"name": "Kotak Mahindra Bank", "ticker": "KOTAKBANK.NS", "sector": "Banking", "desc": "Premium private bank known for quality assets and conservative lending."},
    "TCS_NS": {"name": "Tata Consultancy Services", "ticker": "TCS.NS", "sector": "IT", "desc": "India's largest IT exporter. Consistent compounder with global revenues."},
    "INFY_NS": {"name": "Infosys", "ticker": "INFY.NS", "sector": "IT", "desc": "Second largest IT company with strong US client base."},
    "HCLTECH_NS": {"name": "HCL Technologies", "ticker": "HCLTECH.NS", "sector": "IT", "desc": "Fastest growing large IT company. Strong in infrastructure services."},
    "WIPRO_NS": {"name": "Wipro", "ticker": "WIPRO.NS", "sector": "IT", "desc": "Top IT company with strong cloud and AI services growth."},
    "RELIANCE_NS": {"name": "Reliance Industries", "ticker": "RELIANCE.NS", "sector": "Energy", "desc": "India's largest company by market cap. Energy to retail conglomerate."},
    "NTPC_NS": {"name": "NTPC", "ticker": "NTPC.NS", "sector": "Power", "desc": "India's largest power producer. Key player in energy transition."},
    "POWERGRID_NS": {"name": "Power Grid Corporation", "ticker": "POWERGRID.NS", "sector": "Power", "desc": "Monopoly transmission utility with stable regulated returns."},
    "SUNPHARMA_NS": {"name": "Sun Pharmaceutical", "ticker": "SUNPHARMA.NS", "sector": "Pharma", "desc": "Largest pharma company in India with strong US generics business."},
    "DRREDDY_NS": {"name": "Dr. Reddys Laboratories", "ticker": "DRREDDY.NS", "sector": "Pharma", "desc": "Global generics leader with strong R&D pipeline."},
    "MARUTI_NS": {"name": "Maruti Suzuki", "ticker": "MARUTI.NS", "sector": "Automobile", "desc": "Largest passenger car maker with dominant 40% market share."},
    "BAJAJ-AUTO_NS": {"name": "Bajaj Auto", "ticker": "BAJAJ-AUTO.NS", "sector": "Automobile", "desc": "Leading two-wheeler exporter with strong domestic and global brands."},
    "HINDUNILVR_NS": {"name": "Hindustan Unilever", "ticker": "HINDUNILVR.NS", "sector": "FMCG", "desc": "India's largest FMCG company. Defensive stock with consistent dividends."},
    "ITC_NS": {"name": "ITC Limited", "ticker": "ITC.NS", "sector": "FMCG", "desc": "Diversified conglomerate across FMCG, hotels and agribusiness."},
    "BRITANNIA_NS": {"name": "Britannia Industries", "ticker": "BRITANNIA.NS", "sector": "FMCG", "desc": "Market leader in biscuits with strong rural distribution."},
    "TATASTEEL_NS": {"name": "Tata Steel", "ticker": "TATASTEEL.NS", "sector": "Metals", "desc": "Largest integrated steel company with global operations."},
    "HINDALCO_NS": {"name": "Hindalco Industries", "ticker": "HINDALCO.NS", "sector": "Metals", "desc": "Global aluminium and copper leader. Novelis adds premium value."},
    "TITAN_NS": {"name": "Titan Company", "ticker": "TITAN.NS", "sector": "Consumer", "desc": "Leader in jewellery and watches. Strong brand with premiumisation play."},
    "ASIANPAINT_NS": {"name": "Asian Paints", "ticker": "ASIANPAINT.NS", "sector": "Consumer", "desc": "Market leader in paints. Proxy for urban consumption and housing."},
    "BAJFINANCE_NS": {"name": "Bajaj Finance", "ticker": "BAJFINANCE.NS", "sector": "NBFC", "desc": "India's premier consumer lending NBFC. High growth, high quality."},
    "LT_NS": {"name": "Larsen & Toubro", "ticker": "LT.NS", "sector": "Infrastructure", "desc": "India's largest engineering conglomerate. Key infra beneficiary."},
}

FEATURE_INFO = {
    "Close": ("Closing Price", "The final price at which the stock traded today. Most referenced price point for analysis."),
    "Open": ("Opening Price", "First traded price of the day. Gap from previous close signals overnight market sentiment."),
    "High": ("Day High", "Highest price reached during trading. Acts as a short-term resistance level."),
    "Low": ("Day Low", "Lowest price during trading. Acts as a short-term support level."),
    "Volume": ("Trading Volume", "Total shares traded. High volume confirms price moves. Low volume = weak signal."),
    "SMA_10": ("10-Day Moving Average", "Average closing price over 10 days. Captures short-term trend direction."),
    "SMA_50": ("50-Day Moving Average", "Average closing price over 50 days. Most watched by fund managers for medium-term trend."),
    "EMA_12": ("Fast Exponential Average", "Gives more weight to recent prices. Reacts quickly to news and price changes."),
    "EMA_26": ("Slow Exponential Average", "Slower reaction average. Paired with EMA-12 to detect momentum shifts."),
    "MACD": ("MACD — Momentum Indicator", "Difference between fast and slow EMA. Positive = upward momentum building."),
    "MACD_Signal": ("MACD Signal Line", "9-day average of MACD. When MACD crosses above this line = bullish signal."),
    "RSI": ("RSI — Strength Meter (0-100)", "Above 70 = overbought, may fall. Below 30 = oversold, may rise. 30-70 = healthy zone."),
    "Stoch_K": ("Stochastic %K", "Shows where price sits within recent high-low range. Measures momentum strength."),
    "Stoch_D": ("Stochastic %D", "Smoothed %K. More reliable. Crossovers signal momentum shifts."),
    "ROC": ("Rate of Change", "% price change over 10 days. High positive ROC = strong upward momentum."),
    "Williams_R": ("Williams %R", "Ranges -100 to 0. Above -20 = overbought. Below -80 = oversold. Like inverted RSI."),
    "BB_Upper": ("Bollinger Upper Band", "Price ceiling 2 standard deviations above average. Price touching = potential reversal down."),
    "BB_Middle": ("Bollinger Middle Band", "20-day moving average. Center of price channel. Price gravitates back here."),
    "BB_Lower": ("Bollinger Lower Band", "Price floor 2 standard deviations below average. Price touching = potential reversal up."),
    "ATR": ("Volatility — Average True Range", "Measures daily price swings. High ATR = volatile stock. Helps set stop losses."),
    "OBV": ("On Balance Volume", "Running volume total. Rising OBV with rising price confirms healthy uptrend."),
    "CMF": ("Chaikin Money Flow", "Positive = institutional buying. Negative = institutional selling. Shows smart money flow.")
}

st.set_page_config(page_title="XAI Trading System", layout="wide", page_icon="📊", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #060D1F; }
.main-title { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #2563EB, #60A5FA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1; margin-bottom: 4px; }
.subtitle { color: #475569; font-size: 0.9rem; letter-spacing: 1px; text-transform: uppercase; }
.card { background: #0D1B3E; border: 1px solid #1E3A5F; border-radius: 14px; padding: 18px; margin: 6px 0; transition: border-color 0.2s; }
.card:hover { border-color: #3B82F6; }
.tag { background: #1E3A5F; color: #60A5FA; padding: 2px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.5px; }
.buy-tag { background: #052e16; color: #4ade80; border: 1px solid #166534; padding: 2px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }
.sell-tag { background: #450a0a; color: #f87171; border: 1px solid #991b1b; padding: 2px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }
.mbox { background: #0D1B3E; border: 1px solid #1E3A5F; border-radius: 12px; padding: 16px 12px; text-align: center; }
.mval { font-size: 1.25rem; font-weight: 700; color: #F8FAFC; }
.mlbl { font-size: 0.65rem; color: #64748B; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.8px; }
.sec { font-size: 1rem; font-weight: 700; color: #60A5FA; border-left: 3px solid #2563EB; padding-left: 10px; margin: 24px 0 14px; }
.icard { background: #0D1B3E; border: 1px solid #1E3A5F; border-radius: 10px; padding: 14px; margin: 6px 0; }
.ititle { font-size: 0.88rem; font-weight: 600; color: #93C5FD; }
.idesc { font-size: 0.78rem; color: #64748B; margin-top: 4px; line-height: 1.55; }
.ritem { border-left: 3px solid #2563EB; border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 5px 0; background: #0D1B3E; font-size: 0.84rem; color: #CBD5E1; }
.ritem.pos { border-left-color: #16a34a; }
.ritem.neg { border-left-color: #dc2626; }
.disc { background: #0D1B3E; border: 1px solid #1E3A5F; border-radius: 10px; padding: 12px; color: #475569; font-size: 0.75rem; text-align: center; margin-top: 24px; }
hr { border-color: #1E3A5F; margin: 20px 0; }
button[kind="primary"] { background: linear-gradient(135deg, #2563EB, #1d4ed8) !important; }
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected" not in st.session_state:
    st.session_state.selected = None

@st.cache_data(ttl=3600)
def load_signals():
    return pd.read_csv("predictions/signals_summary.csv", index_col=0)

@st.cache_data
def load_processed(key):
    return pd.read_csv(f"data/processed/{key}_processed.csv", index_col="Date", parse_dates=True).dropna()

@st.cache_data(ttl=300)
def get_live_price(ticker):
    try:
        h = yf.Ticker(ticker).history(period="2d")
        p = round(h["Close"].iloc[-1], 2)
        c = round(((h["Close"].iloc[-1]-h["Close"].iloc[-2])/h["Close"].iloc[-2])*100, 2)
        return p, c
    except:
        return None, None

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    try:
        i = yf.Ticker(ticker).info
        return {
            "Market Cap": i.get("marketCap"),
            "P/E": i.get("trailingPE"),
            "EPS": i.get("trailingEps"),
            "ROE": i.get("returnOnEquity"),
            "D/E": i.get("debtToEquity"),
            "Div Yield": i.get("dividendYield"),
            "52W High": i.get("fiftyTwoWeekHigh"),
            "52W Low": i.get("fiftyTwoWeekLow"),
            "Rev": i.get("totalRevenue"),
            "Margin": i.get("profitMargins")
        }
    except:
        return {}

def fmt(n, prefix="₹"):
    if n is None: return "N/A"
    if n >= 1e12: return f"{prefix}{n/1e12:.1f}T"
    if n >= 1e9: return f"{prefix}{n/1e9:.1f}B"
    if n >= 1e7: return f"{prefix}{n/1e7:.1f}Cr"
    return str(round(n, 2))

def get_reasons(row, signal):
    r = []
    try:
        rsi = float(row.get("rsi", 50))
        macd = float(row.get("macd", 0))
        ms = float(row.get("macd_signal", 0))
        s10 = float(row.get("sma10", 0))
        s50 = float(row.get("sma50", 0))
        sk = float(row.get("stoch_k", 50))
        wr = float(row.get("williams_r", -50))
        if signal == "BUY":
            if rsi < 55: r.append(f"RSI at {rsi:.1f} — not overbought, has room to grow")
            if macd > ms: r.append("MACD crossed above signal line — upward momentum building")
            if s10 > s50: r.append("10-day average above 50-day — short-term uptrend confirmed")
            if sk < 75: r.append(f"Stochastic at {sk:.1f} — momentum not yet exhausted")
            if wr < -50: r.append("Williams %R shows potential upside from current levels")
            if not r: r.append("LSTM detected bullish pattern in 60-day price sequence")
        else:
            if rsi > 60: r.append(f"RSI at {rsi:.1f} — approaching overbought, pullback risk")
            if macd < ms: r.append("MACD below signal line — bearish momentum detected")
            if s10 < s50: r.append("10-day average below 50-day — short-term downtrend active")
            if sk > 70: r.append(f"Stochastic at {sk:.1f} — momentum may be reversing")
            if wr > -25: r.append("Williams %R in overbought zone — reversal likely")
            if not r: r.append("LSTM detected bearish pattern in 60-day price sequence")
    except:
        r.append("Signal generated from LSTM analysis of 60-day price and indicator sequence")
    return r[:4]

signals_df = load_signals()

# ═══════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════
if st.session_state.page == "home":
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<div class="main-title">XAI Trading System</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Explainable AI for Indian Equity Markets &nbsp;·&nbsp; Nifty 50 &nbsp;·&nbsp; LSTM + PPO + SHAP</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div style="text-align:right;color:#475569;font-size:0.8rem;padding-top:20px;">🕐 {datetime.now().strftime("%d %b %Y, %H:%M")}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 25 Handpicked Stocks", "🔍 Screener", "ℹ️ About"])

    with tab1:
        st.markdown("""
        <div class="card">
        <div style="color:#60A5FA;font-weight:700;font-size:0.95rem;margin-bottom:10px;">🏦 Why these 25 stocks from the Nifty 50 basket?</div>
        <div style="color:#94A3B8;font-size:0.84rem;line-height:1.8;">
        The <strong style="color:#F1F5F9;">Nifty 50</strong> is India's benchmark index — top 50 companies on NSE by free-float market cap, 
        representing ~65% of India's total market value. From this basket, we handpicked <strong style="color:#F1F5F9;">25 stocks</strong> 
        representing <strong style="color:#F1F5F9;">9 diverse sectors</strong> — Banking, IT, Energy, Pharma, Auto, FMCG, Metals, Consumer and Infrastructure. 
        Selection criteria: <strong style="color:#F1F5F9;">10+ years of clean data</strong>, high liquidity, strong institutional participation, 
        and sector diversity for robust AI model training.
        </div>
        </div>
        """, unsafe_allow_html=True)

        sectors = sorted(set(v["sector"] for v in STOCKS.values()))
        c1, c2 = st.columns([3,1])
        with c1:
            search = st.text_input("🔍 Search stocks", placeholder="Company name, ticker or sector...")
        with c2:
            sec_f = st.selectbox("Sector", ["All Sectors"] + sectors)

        filtered = {k:v for k,v in STOCKS.items()
                    if (sec_f=="All Sectors" or v["sector"]==sec_f)
                    and (search=="" or search.lower() in v["name"].lower() 
                         or search.lower() in k.lower().replace("_ns","")
                         or search.lower() in v["sector"].lower())}

        st.caption(f"Showing {len(filtered)} of 25 stocks")
        cols = st.columns(3)
        for i, (key, info) in enumerate(filtered.items()):
            with cols[i%3]:
                sig_html = ""
                if not signals_df.empty and key in signals_df.index:
                    sig = signals_df.loc[key, "signal"]
                    conf = float(signals_df.loc[key, "confidence"])
                    cls = "buy-tag" if sig=="BUY" else "sell-tag"
                    sig_html = f'<span class="{cls}">{sig} {conf:.0%}</span>'
                ticker_display = info["ticker"].replace(".NS","")
                st.markdown(f"""
                <div class="card">
                    <div style="font-weight:700;color:#F8FAFC;font-size:0.95rem;">{info["name"]}</div>
                    <div style="color:#475569;font-size:0.72rem;margin:2px 0;">{ticker_display} · NSE</div>
                    <div style="margin:6px 0;display:flex;gap:6px;flex-wrap:wrap;">
                        <span class="tag">{info["sector"]}</span> {sig_html}
                    </div>
                    <div style="color:#475569;font-size:0.76rem;line-height:1.5;">{info["desc"]}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Analyse →", key=f"b_{key}"):
                    st.session_state.selected = key
                    st.session_state.page = "stock"
                    st.rerun()

    with tab2:
        st.markdown('<div class="sec">AI Stock Screener</div>', unsafe_allow_html=True)
        st.caption("Filter all 25 stocks by AI signal, technical conditions and sector — instantly.")

        sc1,sc2,sc3,sc4 = st.columns(4)
        with sc1: sig_f = st.selectbox("AI Signal", ["Any","BUY only","SELL only"])
        with sc2: rsi_f = st.selectbox("RSI Zone", ["Any","Oversold <30","Neutral 30-70","Overbought >70"])
        with sc3: conf_f = st.selectbox("Min Confidence", ["Any",">55%",">60%",">65%",">70%"])
        with sc4: sec_sf = st.selectbox("Sector ", ["All"]+sectors)

        results = []
        for key, info in STOCKS.items():
            if key not in signals_df.index: continue
            row = signals_df.loc[key]
            sig = row["signal"]; conf = float(row["confidence"]); rsi = float(row.get("rsi",50))
            macd = float(row.get("macd",0)); ms = float(row.get("macd_signal",0))
            if sig_f=="BUY only" and sig!="BUY": continue
            if sig_f=="SELL only" and sig!="SELL": continue
            if rsi_f=="Oversold <30" and rsi>=30: continue
            if rsi_f=="Neutral 30-70" and not (30<=rsi<=70): continue
            if rsi_f=="Overbought >70" and rsi<=70: continue
            if conf_f==">55%" and conf<0.55: continue
            if conf_f==">60%" and conf<0.60: continue
            if conf_f==">65%" and conf<0.65: continue
            if conf_f==">70%" and conf<0.70: continue
            if sec_sf!="All" and info["sector"]!=sec_sf: continue
            results.append({"key":key,"name":info["name"],"sector":info["sector"],"sig":sig,"conf":conf,"rsi":rsi,"macd_trend":"Bullish" if macd>ms else "Bearish"})

        st.markdown(f'<div style="color:#60A5FA;font-weight:600;margin-bottom:10px;">{len(results)} stocks matched</div>', unsafe_allow_html=True)
        for r in results:
            c1,c2 = st.columns([5,1])
            with c1:
                sc = "#4ade80" if r["sig"]=="BUY" else "#f87171"
                mc = "#4ade80" if r["macd_trend"]=="Bullish" else "#f87171"
                st.markdown(f"""<div class="icard">
                    <strong style="color:#F8FAFC;">{r["name"]}</strong>
                    <span class="tag" style="margin-left:8px;">{r["sector"]}</span>
                    <span style="color:{sc};font-weight:700;margin-left:10px;">{r["sig"]}</span>
                    <span style="color:#64748B;font-size:0.8rem;margin-left:6px;">{r["conf"]:.0%} confidence</span>
                    <span style="color:#64748B;font-size:0.8rem;margin-left:10px;">RSI: {r["rsi"]:.1f}</span>
                    <span style="color:{mc};font-size:0.8rem;margin-left:10px;">MACD: {r["macd_trend"]}</span>
                </div>""", unsafe_allow_html=True)
            with c2:
                if st.button("View", key=f"s_{r['key']}"):
                    st.session_state.selected = r["key"]
                    st.session_state.page = "stock"
                    st.rerun()

    with tab3:
        st.markdown("""
        <div class="card">
        <div style="color:#60A5FA;font-weight:700;font-size:0.95rem;margin-bottom:12px;">About XAI Trading System</div>
        <div style="color:#94A3B8;font-size:0.84rem;line-height:1.9;">
        <strong style="color:#F8FAFC;">XAI Trading System</strong> — Open-source explainable AI for Indian equity markets.<br>
        
        <strong style="color:#60A5FA;">What it does:</strong><br>
        Predicts next-day price direction (UP/DOWN) for 25 Nifty 50 stocks using deep learning 
        and explains the reasoning behind every signal using explainable AI techniques.<br><br>
        <strong style="color:#60A5FA;">Models Used:</strong><br>
        • <strong style="color:#F8FAFC;">LSTM</strong> — Long Short-Term Memory neural network. Analyzes 60-day price sequences.<br>
        • <strong style="color:#F8FAFC;">PPO</strong> — Proximal Policy Optimization RL agent for portfolio management.<br>
        • <strong style="color:#F8FAFC;">SHAP</strong> — Feature importance for explainability and transparency.<br><br>
        <strong style="color:#60A5FA;">Data:</strong> 10 years (2015–2026) | 22 Technical Indicators | Real-time prices via NSE<br><br>
        <strong style="color:#f87171;">⚠️ Disclaimer: For research and educational purposes only. Not financial advice.</strong>
        </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════
# STOCK DETAIL PAGE
# ═══════════════════════════════════════
elif st.session_state.page == "stock":
    key = st.session_state.selected
    info = STOCKS.get(key, {})

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    try:
        df = load_processed(key)
        row = signals_df.loc[key] if key in signals_df.index else {}
        signal = str(row.get("signal","N/A"))
        confidence = float(row.get("confidence", 0.5))

        live_price, live_change = get_live_price(info["ticker"])
        price = live_price or round(df["Close"].iloc[-1],2)
        change = live_change or 0
        change_color = "#4ade80" if change>=0 else "#f87171"
        fund = get_fundamentals(info["ticker"])

        # ── Header
        st.markdown(f'<div style="font-size:1.7rem;font-weight:800;color:#F8FAFC;margin:8px 0 4px;">{info["name"]}</div>', unsafe_allow_html=True)
        ticker_d = info["ticker"].replace(".NS","")
        st.markdown(f'<span class="tag">{info["sector"]}</span> <span style="color:#475569;font-size:0.8rem;margin-left:8px;">{ticker_d} · NSE · Nifty 50</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Live Price Row
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.markdown(f'<div class="mbox"><div class="mval">₹{price:,}</div><div class="mlbl">Live Price</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="mbox"><div class="mval" style="color:{change_color};">{change:+.2f}%</div><div class="mlbl">Day Change</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="mbox"><div class="mval">{fmt(fund.get("Market Cap"))}</div><div class="mlbl">Market Cap</div></div>', unsafe_allow_html=True)
        pe = fund.get("P/E"); m4.markdown(f'<div class="mbox"><div class="mval">{round(pe,1) if pe else "N/A"}</div><div class="mlbl">P/E Ratio</div></div>', unsafe_allow_html=True)
        wh = fund.get("52W High"); m5.markdown(f'<div class="mbox"><div class="mval">{f"₹{round(wh,0)}" if wh else "N/A"}</div><div class="mlbl">52W High</div></div>', unsafe_allow_html=True)
        wl = fund.get("52W Low"); m6.markdown(f'<div class="mbox"><div class="mval">{f"₹{round(wl,0)}" if wl else "N/A"}</div><div class="mlbl">52W Low</div></div>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── AI Signal
        st.markdown('<div class="sec">🤖 AI Trading Signal</div>', unsafe_allow_html=True)
        s1,s2 = st.columns([1,2])
        with s1:
            color = "#4ade80" if signal=="BUY" else "#f87171"
            bg = "#052e16" if signal=="BUY" else "#450a0a"
            bd = "#16a34a" if signal=="BUY" else "#dc2626"
            emoji = "🟢" if signal=="BUY" else "🔴"
            st.markdown(f'<div style="background:{bg};border:2px solid {bd};border-radius:16px;padding:28px;text-align:center;"><div style="font-size:2.2rem;font-weight:800;color:{color};">{emoji} {signal}</div><div style="color:#94A3B8;margin-top:10px;font-size:0.9rem;">Confidence<br><strong style="color:{color};font-size:1.4rem;">{confidence:.1%}</strong></div><div style="color:#475569;font-size:0.72rem;margin-top:8px;">Based on 60-day LSTM analysis<br>of price + 22 indicators</div></div>', unsafe_allow_html=True)
        with s2:
            direction = "rise" if signal=="BUY" else "fall"
            st.markdown(f'<div class="icard" style="margin-bottom:12px;"><div style="color:#94A3B8;font-size:0.86rem;line-height:1.7;">Our <strong style="color:#60A5FA;">LSTM neural network</strong> analyzed the last <strong style="color:#F8FAFC;">60 trading days</strong> of price history alongside <strong style="color:#F8FAFC;">22 technical indicators</strong> and predicts that <strong style="color:{color};">{info["name"]} is likely to {direction} tomorrow</strong> with {confidence:.1%} confidence.</div></div>', unsafe_allow_html=True)
            reasons = get_reasons(row, signal)
            st.markdown('<div style="color:#475569;font-size:0.72rem;font-weight:600;letter-spacing:0.5px;margin-bottom:6px;">KEY REASONS BEHIND THIS SIGNAL:</div>', unsafe_allow_html=True)
            for r in reasons:
                icon = "✅" if signal=="BUY" else "⚠️"
                cls = "pos" if signal=="BUY" else "neg"
                st.markdown(f'<div class="ritem {cls}">{icon} {r}</div>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Fundamentals
        st.markdown('<div class="sec">📋 Fundamental Snapshot</div>', unsafe_allow_html=True)
        st.caption("Key financial health metrics. Updated from NSE quarterly disclosures.")
        f1,f2,f3,f4,f5,f6 = st.columns(6)
        roe=fund.get("ROE"); eps=fund.get("EPS"); de=fund.get("D/E"); div=fund.get("Div Yield"); pm=fund.get("Margin"); rev=fund.get("Rev")
        f1.markdown(f'<div class="mbox"><div class="mval">{round(pe,1) if pe else "N/A"}</div><div class="mlbl">P/E Ratio</div></div>', unsafe_allow_html=True)
        f2.markdown(f'<div class="mbox"><div class="mval">{str(round(roe*100,1))+"%" if roe else "N/A"}</div><div class="mlbl">ROE</div></div>', unsafe_allow_html=True)
        f3.markdown(f'<div class="mbox"><div class="mval">{round(eps,1) if eps else "N/A"}</div><div class="mlbl">EPS ₹</div></div>', unsafe_allow_html=True)
        f4.markdown(f'<div class="mbox"><div class="mval">{round(de,1) if de else "N/A"}</div><div class="mlbl">Debt/Equity</div></div>', unsafe_allow_html=True)
        f5.markdown(f'<div class="mbox"><div class="mval">{str(round(div*100,2))+"%" if div else "N/A"}</div><div class="mlbl">Div Yield</div></div>', unsafe_allow_html=True)
        f6.markdown(f'<div class="mbox"><div class="mval">{fmt(rev)}</div><div class="mlbl">Revenue</div></div>', unsafe_allow_html=True)

        with st.expander("📖 What do these mean?"):
            st.markdown("""
| Metric | What it tells you | Healthy range |
|--------|------------------|---------------|
| **P/E Ratio** | How much you pay per ₹1 of earnings. Lower = cheaper. | IT: 20-35, Banks: 10-20 |
| **ROE** | Profit per ₹1 of shareholder money. Higher = more efficient. | >15% is good |
| **EPS** | Profit per share. Growing EPS = healthy business. | Higher & growing |
| **Debt/Equity** | How leveraged. Lower = safer balance sheet. | <1 safe, >2 risky |
| **Div Yield** | Annual dividend as % of share price. Income indicator. | Higher = more income |
| **Revenue** | Total sales. Growing revenue = expanding business. | Year-on-year growth |
            """)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Price Chart
        st.markdown('<div class="sec">📊 Price History</div>', unsafe_allow_html=True)
        st.caption("🕯️ Green = price up | Red = price down | Orange = 10-day avg | Blue = 50-day avg | Dotted = Bollinger Bands")
        period = st.radio("Period", ["1M","3M","6M","1Y","5Y","All"], horizontal=True)
        pm2 = {"1M":22,"3M":63,"6M":126,"1Y":252,"5Y":1260,"All":len(df)}
        n = pm2[period]
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index[-n:], open=df["Open"].iloc[-n:], high=df["High"].iloc[-n:], low=df["Low"].iloc[-n:], close=df["Close"].iloc[-n:], name="Price", increasing_line_color="#22c55e", decreasing_line_color="#ef4444", increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444"))
        fig.add_trace(go.Scatter(x=df.index[-n:], y=df["SMA_10"].iloc[-n:], name="10D Avg", line=dict(color="#f59e0b",width=1.5)))
        fig.add_trace(go.Scatter(x=df.index[-n:], y=df["SMA_50"].iloc[-n:], name="50D Avg", line=dict(color="#3b82f6",width=1.5)))
        fig.add_trace(go.Scatter(x=df.index[-n:], y=df["BB_Upper"].iloc[-n:], name="BB Upper", line=dict(color="#475569",width=1,dash="dot"), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index[-n:], y=df["BB_Lower"].iloc[-n:], name="BB Lower", line=dict(color="#475569",width=1,dash="dot"), fill="tonexty", fillcolor="rgba(59,130,246,0.04)", showlegend=False))
        fig.update_layout(height=440, xaxis_rangeslider_visible=False, plot_bgcolor="#060D1F", paper_bgcolor="#060D1F", font_color="#94A3B8", legend=dict(bgcolor="#0D1B3E",bordercolor="#1E3A5F",font=dict(size=11)))
        fig.update_xaxes(gridcolor="#1E3A5F",showgrid=True)
        fig.update_yaxes(gridcolor="#1E3A5F",showgrid=True,title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Technical Indicators
        st.markdown('<div class="sec">📉 Technical Indicators</div>', unsafe_allow_html=True)
        t1,t2 = st.columns(2)
        with t1:
            rsi_v = df["RSI"].iloc[-1]
            if rsi_v>70: rc,rs="#f87171",f"🔴 Overbought ({rsi_v:.1f}) — Potential pullback ahead"
            elif rsi_v<30: rc,rs="#4ade80",f"🟢 Oversold ({rsi_v:.1f}) — Potential bounce ahead"
            else: rc,rs="#f59e0b",f"🟡 Neutral ({rsi_v:.1f}) — Healthy trading zone"
            st.markdown(f'<div class="icard"><div class="ititle">RSI — Relative Strength Index</div><div style="color:{rc};font-weight:600;margin:6px 0;font-size:0.88rem;">{rs}</div><div class="idesc">Think of RSI like a battery. Above 70 means overcharged and may drop. Below 30 means depleted and may bounce. 30-70 is the healthy zone.</div></div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index[-252:], y=df["RSI"].iloc[-252:], fill="tozeroy", line=dict(color="#8b5cf6",width=2), fillcolor="rgba(139,92,246,0.1)", name="RSI"))
            fig2.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought 70", annotation_font=dict(color="#ef4444",size=10))
            fig2.add_hline(y=30, line_dash="dash", line_color="#22c55e", annotation_text="Oversold 30", annotation_font=dict(color="#22c55e",size=10))
            fig2.add_hrect(y0=30, y1=70, fillcolor="#3b82f6", opacity=0.03)
            fig2.update_layout(height=200, plot_bgcolor="#060D1F", paper_bgcolor="#060D1F", font_color="#94A3B8", showlegend=False, margin=dict(t=5,b=5,l=5,r=5))
            fig2.update_yaxes(range=[0,100], gridcolor="#1E3A5F")
            fig2.update_xaxes(gridcolor="#1E3A5F")
            st.plotly_chart(fig2, use_container_width=True)

        with t2:
            macd_v = df["MACD"].iloc[-1]; ms_v = df["MACD_Signal"].iloc[-1]
            mc2 = "#4ade80" if macd_v>ms_v else "#f87171"
            ms2 = "🟢 Bullish — Blue above orange, upward momentum" if macd_v>ms_v else "🔴 Bearish — Blue below orange, downward pressure"
            st.markdown(f'<div class="icard"><div class="ititle">MACD — Moving Average Convergence Divergence</div><div style="color:{mc2};font-weight:600;margin:6px 0;font-size:0.88rem;">{ms2}</div><div class="idesc">When the blue MACD line crosses above the orange signal line = buying opportunity. Green histogram bars = growing momentum. Red = fading momentum. Zero line crossing = major trend shift.</div></div>', unsafe_allow_html=True)
            macd_hist = df["MACD"].iloc[-252:] - df["MACD_Signal"].iloc[-252:]
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=df.index[-252:], y=macd_hist, marker_color=["#22c55e" if v>0 else "#ef4444" for v in macd_hist], name="Histogram", opacity=0.7))
            fig3.add_trace(go.Scatter(x=df.index[-252:], y=df["MACD"].iloc[-252:], name="MACD", line=dict(color="#3b82f6",width=2)))
            fig3.add_trace(go.Scatter(x=df.index[-252:], y=df["MACD_Signal"].iloc[-252:], name="Signal", line=dict(color="#f59e0b",width=2)))
            fig3.update_layout(height=200, plot_bgcolor="#060D1F", paper_bgcolor="#060D1F", font_color="#94A3B8", legend=dict(bgcolor="#0D1B3E",font=dict(size=10)), margin=dict(t=5,b=5,l=5,r=5))
            fig3.update_xaxes(gridcolor="#1E3A5F")
            fig3.update_yaxes(gridcolor="#1E3A5F")
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Backtest
        st.markdown('<div class="sec">📈 Backtest Performance</div>', unsafe_allow_html=True)
        st.caption("Simulated trading using AI signals on historical data. Starting capital: ₹1,00,000 | Transaction cost: 0.1% per trade")
        try:
            summary = pd.read_csv("backtest_results/summary.csv")
            stock_bt = summary[summary["Stock"]==key]
            if not stock_bt.empty:
                ptabs = st.tabs(["6 Months","1 Year","5 Years","10 Years"])
                pkeys = ["6M","1Y","5Y","10Y"]
                for pt,ptab in zip(pkeys,ptabs):
                    with ptab:
                        prow = stock_bt[stock_bt["Period"]==pt]
                        if prow.empty: st.info(f"Not enough data for {pt} backtest."); continue
                        prow = prow.iloc[0]
                        ret=float(prow["Return(%)"]); bh=float(prow["BuyHold(%)"]); alpha=float(prow["Alpha(%)"])
                        sd=prow["StartDate"]; ed=prow["EndDate"]; days=int(prow["Days"]); fv=float(prow["FinalValue"])
                        pos=ret>0; rc2="#4ade80" if pos else "#f87171"; ac="#4ade80" if alpha>0 else "#f87171"
                        st.markdown(f'<div style="color:#64748B;font-size:0.78rem;margin-bottom:14px;">📅 Period: <strong style="color:#94A3B8;">{sd}</strong> → <strong style="color:#94A3B8;">{ed}</strong> &nbsp;·&nbsp; {days} trading days &nbsp;·&nbsp; Starting capital: <strong style="color:#94A3B8;">₹1,00,000</strong> &nbsp;·&nbsp; Transaction cost: <strong style="color:#94A3B8;">0.1%</strong></div>', unsafe_allow_html=True)
                        b1,b2,b3,b4 = st.columns(4)
                        b1.markdown(f'<div class="mbox"><div class="mval">₹1,00,000</div><div class="mlbl">Starting Capital</div></div>', unsafe_allow_html=True)
                        b2.markdown(f'<div class="mbox"><div class="mval">₹{fv:,.0f}</div><div class="mlbl">Final Value</div></div>', unsafe_allow_html=True)
                        b3.markdown(f'<div class="mbox"><div class="mval" style="color:{rc2};">{ret:+.1f}%</div><div class="mlbl">AI Strategy</div></div>', unsafe_allow_html=True)
                        b4.markdown(f'<div class="mbox"><div class="mval" style="color:{ac};">{alpha:+.1f}%</div><div class="mlbl">Alpha vs B&H</div></div>', unsafe_allow_html=True)
                        bt_file=f"backtest_results/{key}_{pt}_backtest.csv"
                        if os.path.exists(bt_file):
                            bt=pd.read_csv(bt_file)
                            lc="#22c55e" if pos else "#ef4444"; fc="rgba(34,197,94,0.08)" if pos else "rgba(239,68,68,0.08)"
                            fig5=go.Figure()
                            fig5.add_trace(go.Scatter(x=list(range(len(bt))), y=bt["portfolio"], fill="tozeroy", line=dict(color=lc,width=2), fillcolor=fc))
                            fig5.add_hline(y=100000, line_dash="dash", line_color="#475569", annotation_text="₹1,00,000 start")
                            fig5.update_layout(height=250, plot_bgcolor="#060D1F", paper_bgcolor="#060D1F", font_color="#94A3B8", showlegend=False, margin=dict(t=10,b=10))
                            fig5.update_xaxes(gridcolor="#1E3A5F", title="Trading Days")
                            fig5.update_yaxes(gridcolor="#1E3A5F", title="Portfolio ₹")
                            st.plotly_chart(fig5, use_container_width=True)
                        if pos:
                            st.success(f"**Why did the AI profit in this {pt} period?**\n\n- LSTM detected recurring bullish price patterns in {info['name']}\n- Technical momentum signals aligned with actual upward moves\n- Model timed entries before major rallies and exits before drops\n- This period likely had clear directional trends the AI could follow")
                        else:
                            st.error(f"**Why did the AI lose in this {pt} period?**\n\n- This period may have been sideways or highly volatile — hard for any model to predict\n- Market events (rate hikes, global shocks) can override technical signals\n- Sudden news-driven moves are outside the scope of technical analysis\n- **Note:** Even professional fund managers fail to beat markets consistently. Negative backtests are honest and realistic — not a model failure. Use this signal alongside fundamental research.")
        except Exception as e:
            st.warning(f"Backtest data unavailable: {e}")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── SHAP
        st.markdown('<div class="sec">🔍 Why Did the AI Decide This?</div>', unsafe_allow_html=True)
        st.caption("Each bar shows how much that indicator influenced the AI prediction. Red = highest impact. Longer bar = stronger influence.")
        shap_path = f"shap_plots/{key}_shap.csv"
        if os.path.exists(shap_path):
            fi = pd.read_csv(shap_path).sort_values("Importance", ascending=True).tail(15)
            fi["Label"] = fi["Feature"].map(lambda x: FEATURE_INFO[x][0] if x in FEATURE_INFO else x)
            fi["Label"] = fi["Feature"].map(lambda x: FEATURE_INFO[x][0] if x in FEATURE_INFO else x)
            n_bars = len(fi)
            bar_colors = []
            for i in range(n_bars):
                if i >= n_bars-3: bar_colors.append("#ef4444")
                elif i >= n_bars-8: bar_colors.append("#3b82f6")
                else: bar_colors.append("#1E3A5F")
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=fi["Importance"], y=fi["Label"], orientation="h",
                marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)", width=0)),
                text=[f"{v:.4f}" for v in fi["Importance"]], textposition="outside",
                textfont=dict(color="#64748B", size=10), hovertemplate="%{y}<br>Influence: %{x:.4f}<extra></extra>"
            ))
            fig4.update_layout(
                height=500, plot_bgcolor="#060D1F", paper_bgcolor="#060D1F",
                font_color="#94A3B8", xaxis_title="Influence Level",
                margin=dict(l=10,r=80,t=10,b=10),
                xaxis=dict(gridcolor="#1E3A5F", showgrid=True),
                yaxis=dict(gridcolor="#1E3A5F", showgrid=False)
            )
            st.plotly_chart(fig4, use_container_width=True)

            st.markdown('<div style="color:#60A5FA;font-size:0.82rem;font-weight:600;margin:14px 0 8px;letter-spacing:0.5px;">TOP 5 FACTORS — WHAT THEY MEAN FOR THIS PREDICTION:</div>', unsafe_allow_html=True)
            top5 = fi.tail(5)[::-1]
            for _,frow in top5.iterrows():
                feat = frow["Feature"]
                if feat in FEATURE_INFO:
                    title, desc = FEATURE_INFO[feat]
                    imp = frow["Importance"]
                    imp_color = "#ef4444" if imp == fi["Importance"].max() else "#3b82f6"
                    st.markdown(f'<div class="icard"><div style="display:flex;justify-content:space-between;align-items:center;"><div class="ititle">📊 {title}</div><div style="color:{imp_color};font-size:0.75rem;font-weight:600;">Influence: {imp:.4f}</div></div><div class="idesc" style="margin-top:6px;">{desc}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="disc">⚠️ XAI Trading System is for educational and research purposes only. AI predictions are based on historical patterns and do not guarantee future returns. This is NOT financial advice. Always consult a SEBI-registered investment advisor before making investment decisions.</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        if st.button("← Go Back"):
            st.session_state.page = "home"
            st.rerun()
