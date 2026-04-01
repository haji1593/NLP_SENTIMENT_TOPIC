"""
app.py — Combined NLP Dashboard + Pipeline
==========================================
Dual-mode file:

  streamlit run app.py          →  Launch the interactive dashboard
  python app.py                 →  Run full NLP pipeline (generate data)
  python app.py --notebooks     →  Execute Jupyter notebooks then launch app
  python app.py --notebooks --no-app  →  Notebooks only, no app
"""

# ── Streamlit context detection (must be first) ───────────────
def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

_STREAMLIT_MODE = _in_streamlit()


# ═════════════════════════════════════════════════════════════
#  DASHBOARD MODE  —  streamlit run app.py
# ═════════════════════════════════════════════════════════════
if _STREAMLIT_MODE:

    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import json
    import time
    from datetime import datetime
    from pathlib import Path

    st.set_page_config(
        page_title="War Sentiment Intelligence",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Master CSS ──────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    :root {
      --bg:#07090f; --navy:#0b0f1a; --navy2:#10172a; --navy3:#161f36;
      --surface:#1a2035; --red:#f03e5a; --red-dim:#7b1d2e;
      --gold:#f5a623; --green:#00d68f; --blue:#4c8ef7;
      --purple:#8b5cf6; --teal:#06b6d4;
      --text:#e8edf5; --muted:#7a8699; --border:#1e2d4a;
      --card-bg:rgba(16,23,42,0.9);
    }
    html,body,.stApp{font-family:'Inter',sans-serif;background:var(--bg)!important;color:var(--text);}
    ::-webkit-scrollbar{width:5px;height:5px;}
    ::-webkit-scrollbar-track{background:var(--navy);}
    ::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px;}
    ::-webkit-scrollbar-thumb:hover{background:#2e4070;}
    .main .block-container{padding:0 2rem 3rem 2rem!important;max-width:1400px;}
    section[data-testid="stSidebar"]{background:linear-gradient(180deg,var(--navy) 0%,var(--navy2) 100%)!important;border-right:1px solid var(--border)!important;}
    section[data-testid="stSidebar"] *{color:var(--text)!important;}
    section[data-testid="stSidebar"] .stButton>button{background:transparent!important;border:none!important;border-radius:8px!important;color:var(--muted)!important;font-size:0.84rem!important;font-weight:500!important;text-align:left!important;padding:0.55rem 0.85rem!important;margin-bottom:2px!important;transition:all 0.15s ease!important;width:100%!important;justify-content:flex-start!important;box-shadow:none!important;}
    section[data-testid="stSidebar"] .stButton>button:hover{background:rgba(76,142,247,0.1)!important;color:#c9d8f5!important;border-left:2px solid var(--blue)!important;padding-left:calc(0.85rem - 2px)!important;}
    .hero{background:linear-gradient(135deg,var(--navy) 0%,var(--navy2) 40%,#150a14 100%);border:1px solid var(--border);border-radius:16px;padding:2.5rem 2.5rem 2rem;margin:1rem 0 2rem;position:relative;overflow:hidden;box-shadow:0 4px 40px rgba(0,0,0,0.5);}
    .hero::before{content:'';position:absolute;top:-60px;right:-60px;width:300px;height:300px;background:radial-gradient(circle,rgba(233,69,96,0.12) 0%,transparent 70%);pointer-events:none;}
    .hero::after{content:'';position:absolute;bottom:-80px;left:30%;width:400px;height:200px;background:radial-gradient(circle,rgba(59,130,246,0.07) 0%,transparent 70%);pointer-events:none;}
    .hero-eyebrow{font-size:0.72rem;font-weight:600;letter-spacing:0.15em;text-transform:uppercase;color:var(--red);margin-bottom:0.6rem;}
    .hero-title{font-size:2.1rem;font-weight:800;color:#fff;line-height:1.15;margin:0 0 0.5rem;letter-spacing:-0.03em;}
    .hero-title span{color:var(--red);}
    .hero-subtitle{font-size:0.95rem;color:var(--muted);max-width:620px;line-height:1.6;margin-bottom:1.4rem;}
    .hero-badges{display:flex;flex-wrap:wrap;gap:0.5rem;}
    .badge{display:inline-flex;align-items:center;gap:0.35rem;padding:0.3rem 0.75rem;border-radius:99px;font-size:0.75rem;font-weight:600;border:1px solid;}
    .badge-red{background:rgba(233,69,96,0.12);color:#f87171;border-color:rgba(233,69,96,0.3);}
    .badge-blue{background:rgba(59,130,246,0.12);color:#93c5fd;border-color:rgba(59,130,246,0.3);}
    .badge-green{background:rgba(16,185,129,0.12);color:#6ee7b7;border-color:rgba(16,185,129,0.3);}
    .badge-gold{background:rgba(245,158,11,0.12);color:#fcd34d;border-color:rgba(245,158,11,0.3);}
    .page-crumb{font-size:0.72rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:var(--muted);margin:1.2rem 0 0.2rem;}
    .page-crumb span{color:var(--red);}
    .sec-header{display:flex;align-items:center;gap:0.6rem;margin:2rem 0 0.3rem;}
    .sec-icon{width:36px;height:36px;background:linear-gradient(135deg,rgba(233,69,96,0.2),rgba(233,69,96,0.05));border:1px solid rgba(233,69,96,0.3);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1rem;}
    .sec-title{font-size:1.2rem;font-weight:700;color:#fff;letter-spacing:-0.02em;}
    .sec-bar{height:3px;background:linear-gradient(90deg,var(--red),transparent);border-radius:2px;margin-bottom:0.4rem;}
    .sec-sub{font-size:0.85rem;color:var(--muted);margin-bottom:1.2rem;line-height:1.5;}
    .kpi-card{background:var(--card-bg);border:1px solid var(--border);border-radius:14px;padding:1.4rem 1.5rem 1.2rem;position:relative;overflow:hidden;transition:transform 0.2s ease,box-shadow 0.2s ease,border-color 0.2s ease;backdrop-filter:blur(8px);}
    .kpi-card:hover{transform:translateY(-4px);box-shadow:0 16px 40px rgba(0,0,0,0.5);border-color:rgba(240,62,90,0.4);}
    .kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:12px 12px 0 0;}
    .kpi-card.red::before{background:linear-gradient(90deg,#f03e5a,#7b1d2e);}
    .kpi-card.green::before{background:linear-gradient(90deg,#00d68f,#065f46);}
    .kpi-card.blue::before{background:linear-gradient(90deg,#4c8ef7,#1e3a6e);}
    .kpi-card.gold::before{background:linear-gradient(90deg,#f5a623,#78350f);}
    .kpi-card.purple::before{background:linear-gradient(90deg,#8b5cf6,#4c1d95);}
    .kpi-card.teal::before{background:linear-gradient(90deg,#06b6d4,#164e63);}
    .kpi-icon{font-size:1.5rem;margin-bottom:0.6rem;}
    .kpi-value{font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;color:#fff;line-height:1;margin-bottom:0.25rem;}
    .kpi-label{font-size:0.75rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;}
    .kpi-delta{font-size:0.78rem;margin-top:0.5rem;padding:0.2rem 0.6rem;border-radius:99px;display:inline-block;}
    .kpi-delta.neg{background:rgba(239,68,68,0.15);color:#f87171;}
    .kpi-delta.pos{background:rgba(16,185,129,0.15);color:#6ee7b7;}
    .kpi-delta.neu{background:rgba(245,158,11,0.15);color:#fcd34d;}
    .stTabs [data-baseweb="tab-list"]{background:var(--navy2);border-radius:10px;padding:4px;gap:2px;border:1px solid var(--border);}
    .stTabs [data-baseweb="tab"]{background:transparent;border-radius:7px;color:var(--muted);font-family:'Inter',sans-serif;font-weight:500;font-size:0.84rem;padding:0.45rem 1rem;border:none;transition:all 0.15s ease;}
    .stTabs [data-baseweb="tab"]:hover{color:var(--text)!important;background:rgba(76,142,247,0.08)!important;}
    .stTabs [aria-selected="true"]{background:linear-gradient(135deg,rgba(240,62,90,0.2),rgba(76,142,247,0.12))!important;color:#fff!important;box-shadow:0 1px 6px rgba(0,0,0,0.4);}
    .stTabs [data-baseweb="tab-panel"]{padding-top:1.2rem;}
    .stDataFrame,.stDataFrame table{background:var(--navy2)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:8px;}
    .stAlert{background:rgba(22,27,34,0.9)!important;border-radius:10px!important;border-left:4px solid var(--gold)!important;color:var(--text)!important;}
    .clf-card{background:linear-gradient(135deg,var(--navy2),var(--navy3));border:1px solid var(--border);border-radius:14px;padding:1.8rem;text-align:center;margin-top:0.8rem;position:relative;overflow:hidden;}
    .clf-card::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0;}
    .clf-card.pos::after{background:linear-gradient(90deg,#10b981,#064e3b);}
    .clf-card.neg::after{background:linear-gradient(90deg,#e94560,#7f1d2e);}
    .clf-card.neu::after{background:linear-gradient(90deg,#f59e0b,#78350f);}
    .clf-label{font-size:0.72rem;font-weight:600;letter-spacing:0.12em;color:var(--muted);text-transform:uppercase;}
    .clf-value{font-family:'JetBrains Mono',monospace;font-size:2.2rem;font-weight:700;color:#fff;margin:0.3rem 0 0.1rem;}
    .clf-score{font-size:0.85rem;color:var(--muted);}
    .fancy-divider{height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:2rem 0;}
    .stSelectbox label,.stToggle label{color:var(--muted)!important;font-size:0.82rem!important;}
    .stButton>button[kind="primary"]{background:linear-gradient(135deg,var(--red),#c0392b)!important;border:none!important;color:white!important;font-weight:600!important;border-radius:8px!important;transition:all 0.2s ease!important;box-shadow:0 4px 14px rgba(233,69,96,0.35)!important;}
    .stButton>button[kind="primary"]:hover{transform:translateY(-1px)!important;box-shadow:0 6px 20px rgba(233,69,96,0.5)!important;}
    .stButton>button[kind="secondary"]{background:var(--navy2)!important;border:1px solid var(--border)!important;color:var(--muted)!important;border-radius:8px!important;}
    .stTextArea textarea{background:var(--navy2)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--text)!important;font-family:'Inter',sans-serif!important;font-size:0.88rem!important;}
    .stTextArea textarea:focus{border-color:var(--red)!important;box-shadow:0 0 0 3px rgba(233,69,96,0.15)!important;}
    .stChatInput>div{background:var(--navy2)!important;border:1px solid var(--border)!important;border-radius:10px!important;}
    .streamlit-expanderHeader{background:var(--navy2)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--muted)!important;font-size:0.82rem!important;}
    .streamlit-expanderContent{background:var(--navy2)!important;border:1px solid var(--border)!important;border-top:none!important;}
    [data-testid="metric-container"]{background:var(--navy2);border:1px solid var(--border);border-radius:10px;padding:0.8rem 1rem;}
    [data-testid="stMetricValue"]{color:#fff!important;font-family:'JetBrains Mono',monospace!important;}
    [data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:0.78rem!important;}
    .stSpinner>div{border-top-color:var(--red)!important;}
    [data-testid="stChatMessage"]{background:var(--navy2)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:0.7rem 1rem!important;margin-bottom:0.4rem!important;}
    [data-testid="stChatMessageContent"] p{color:var(--text)!important;font-size:0.88rem!important;line-height:1.6!important;}
    [data-testid="stChatMessageAvatarUser"]{background:linear-gradient(135deg,var(--blue),#1e3a8a)!important;}
    [data-testid="stChatMessageAvatarAssistant"]{background:linear-gradient(135deg,var(--red),var(--red-dim))!important;}
    #MainMenu,footer,header{visibility:hidden;}
    </style>
    """, unsafe_allow_html=True)

    # ── Helpers ─────────────────────────────────────────────────
    def section(icon, title, subtitle=""):
        st.markdown(
            f'<div class="sec-header"><div class="sec-icon">{icon}</div>'
            f'<div class="sec-title">{title}</div></div>'
            f'<div class="sec-bar"></div>'
            + (f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""),
            unsafe_allow_html=True)

    def kpi(icon, value, label, delta="", color="blue"):
        dcls  = "neg" if "negative" in delta.lower() else "neu"
        dhtml = f'<div class="kpi-delta {dcls}">{delta}</div>' if delta else ""
        st.markdown(
            f'<div class="kpi-card {color}"><div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-label">{label}</div>'
            f'{dhtml}</div>', unsafe_allow_html=True)

    def divider():
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    def page_crumb(label):
        st.markdown(
            f'<div class="page-crumb">War Sentiment Hub &nbsp;/&nbsp; <span>{label}</span></div>',
            unsafe_allow_html=True)

    CHART_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#8b949e", size=12),
        margin=dict(l=10, r=20, t=44, b=30),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#8b949e"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#8b949e"),
        title_font=dict(size=13, color="#e6edf3", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
    )
    COLORS = {"red":"#e94560","green":"#10b981","gold":"#f59e0b","blue":"#3b82f6","muted":"#8b949e"}
    BASE_DIR = Path(__file__).parent

    # Download NLTK data once (needed on Streamlit Cloud)
    import nltk
    for _res, _kind in [("vader_lexicon","sentiment"),("stopwords","corpora"),
                        ("wordnet","corpora"),("punkt_tab","tokenizers"),
                        ("averaged_perceptron_tagger_eng","taggers")]:
        try: nltk.data.find(f"{_kind}/{_res}")
        except LookupError: nltk.download(_res, quiet=True)

    def _chart(fig, h=380):
        fig.update_layout(height=h, **CHART_LAYOUT)
        return fig

    def load_nlp_results():
        """Read JSON outputs fresh on every rerun — files are small, no caching needed."""
        sp = BASE_DIR / "data" / "sentiment_results.json"
        tp = BASE_DIR / "data" / "topic_results.json"
        if not sp.exists() or not tp.exists():
            return None, None
        try:
            s = json.loads(sp.read_text(encoding="utf-8"))
            t = json.loads(tp.read_text(encoding="utf-8"))
            return s, t
        except Exception:
            return None, None

    # ── Chatbot ─────────────────────────────────────────────────
    def get_chatbot_response(msg_raw, sd, td):
        msg = msg_raw.lower()

        def _angle(name):
            if not sd: return None
            for a in sd.get("by_angle", []):
                if a["angle"].lower().startswith(name.lower()): return a
            return None

        if any(w in msg for w in ["sentiment","vader","score","overall"]):
            if sd:
                s = sd["summary"]
                angs = sorted(sd.get("by_angle",[]), key=lambda x: x["avg_sentiment"])
                tail = (f"\n\n**{angs[0]['angle']}** most negative ({angs[0]['avg_sentiment']:+.4f}), "
                        f"**{angs[-1]['angle']}** least negative ({angs[-1]['avg_sentiment']:+.4f}).") if angs else ""
                return (f"Across **{s['total_docs']:,}** docs from **{s['sources']}** platforms, "
                        f"avg VADER = **{s['avg_sentiment']:+.4f}**.\n\n"
                        f"**{s['pct_positive']}%** Positive · **{s['pct_neutral']}%** Neutral · **{s['pct_negative']}%** Negative." + tail)
            return "Run `python app.py` to generate data."

        if any(w in msg for w in ["military","drone","missile","strike","idf","irgc","weapon"]):
            a = _angle("Military")
            return (f"**Military Operations** avg VADER **{a['avg_sentiment']:+.4f}**, only **{a['pct_positive']}%** positive."
                    if a else "Run `python app.py` first.")

        if any(w in msg for w in ["media","propaganda","bias","misinform","trust","fake"]):
            a = _angle("Media")
            return (f"**Media / Propaganda** avg **{a['avg_sentiment']:+.4f}** ({a['pct_positive']}% positive)."
                    if a else "Run `python app.py` first.")

        if any(w in msg for w in ["economic","economy","oil","gas","hormuz","energy","market","price","sanction"]):
            a = _angle("Economic")
            return (f"**Economic Impact** avg **{a['avg_sentiment']:+.4f}** ({a['pct_positive']}% positive)."
                    if a else "Run `python app.py` first.")

        if any(w in msg for w in ["geopolit","tension","middle east","proxy","hezbollah","hamas"]):
            a = _angle("Geopolit")
            return (f"**Geopolitical Tensions** avg **{a['avg_sentiment']:+.4f}** ({a['pct_positive']}% positive)."
                    if a else "Run `python app.py` first.")

        if any(w in msg for w in ["support","protest","peace","oppose","ceasefire","condemn"]):
            a = _angle("War Support")
            return (f"**War Support / Opposition** avg **{a['avg_sentiment']:+.4f}** ({a['pct_positive']}% positive)."
                    if a else "Run `python app.py` first.")

        if any(w in msg for w in ["topic","lda","model","theme"]):
            if td:
                lines = "\n".join(f"**{t['id']+1}.** {t['label']} — {t['percentage']}%" for t in td["topics"])
                return f"LDA found **6 topics**:\n\n{lines}"
            return "Run `python app.py` to generate topic data."

        if any(w in msg for w in ["source","platform","facebook","twitter","reddit","youtube","tiktok","instagram"]):
            if sd:
                rows = sorted(sd["by_source"], key=lambda x: x["avg_sentiment"])
                lines = "\n".join(f"**{r['source']}**: {r['avg_sentiment']:+.4f}" for r in rows)
                return f"By platform (most → least negative):\n\n{lines}"
            return "Run `python app.py` first."

        if any(w in msg for w in ["textblob","polarity","blob"]):
            if sd:
                tb = sd.get("textblob_summary", {})
                if tb:
                    return (f"**TextBlob** avg polarity **{tb.get('avg_sentiment',0):+.4f}**\n"
                            f"Positive: {tb.get('pct_positive',0)}% · Neutral: {tb.get('pct_neutral',0)}% · Negative: {tb.get('pct_negative',0)}%")
            return "Run `python app.py` first."

        if any(w in msg for w in ["classif","logistic","tfidf","accuracy","supervised","f1"]):
            if sd:
                sup = sd.get("supervised", {})
                parts = [f"**{'VADER' if m=='vader' else 'TextBlob'}**: {r.get('accuracy',0)*100:.1f}%"
                         for m, r in sup.items() if r]
                if parts: return "TF-IDF + Logistic Regression:\n\n" + "\n".join(f"- {p}" for p in parts)
            return "Run `python app.py` first."

        if any(w in msg for w in ["cluster","kmeans","k-means","pca","elbow"]):
            if sd:
                cl = sd.get("clustering", {})
                if cl.get("clusters"):
                    lines = "\n".join(f"- **Cluster {c['cluster']}** ({c['count']:,} docs): {', '.join(c['top_words'][:5])}"
                                      for c in cl["clusters"])
                    return f"K-Means (k=5) clusters:\n\n{lines}"
            return "Run `python app.py` first."

        if any(w in msg for w in ["preprocess","clean","lemma","stopword"]):
            return ("Pipeline: lowercase → emoji/URL/HTML strip → custom stopwords → "
                    "POS-tagged WordNet lemmatization → language filter (English only).")

        if any(w in msg for w in ["hi","hello","hey"]):
            return "Hello! Ask me about sentiment, topics, platforms, classifier accuracy, or clusters."

        if any(w in msg for w in ["help","what can","how"]):
            return ("Ask about: **sentiment** · **textblob** · **military/economic/media/geopolitical** "
                    "angles · **platform** comparisons · **topics** · **classifier** · **clusters**")

        return "Try: sentiment, topics, classifier accuracy, platform comparisons, or clustering."

    # ── Session state ────────────────────────────────────────────
    if "page"              not in st.session_state: st.session_state.page              = "overview"
    if "chat_history"      not in st.session_state: st.session_state.chat_history      = []
    if "classifier_result" not in st.session_state: st.session_state.classifier_result = None

    sd, td     = load_nlp_results()
    pipeline_ok = sd is not None and td is not None
    page        = st.session_state.page

    # ── Sidebar ──────────────────────────────────────────────────
    NAV = [
        ("📊","Overview & KPIs",   "overview"),
        ("🎭","Sentiment Analysis","sentiment"),
        ("🗂","Topic Modeling",    "topics"),
        ("🤖","ML Results",        "ml_results"),
        ("📋","Data Sources",      "data_sources"),
        ("💬","Chat Assistant",    "chat"),
        ("🔬","Live Classifier",   "classifier"),
    ]
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0 0.5rem 0;">
          <div style="font-size:0.65rem;letter-spacing:0.18em;text-transform:uppercase;color:#e94560;font-weight:700;margin-bottom:4px;">NLP Intelligence</div>
          <div style="font-size:1.15rem;font-weight:800;color:#fff;letter-spacing:-0.02em;">War Sentiment Hub</div>
          <div style="font-size:0.75rem;color:#8b949e;margin-top:3px;">US / Israel – Iran Conflict</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div style="height:1px;background:linear-gradient(90deg,#e94560,transparent);margin:0.5rem 0 1rem;"></div>', unsafe_allow_html=True)

        if st.button("↺  Reload Data", use_container_width=True, type="secondary", key="reload_btn"):
            st.rerun()

        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#8b949e;margin-bottom:0.5rem;">Navigation</div>', unsafe_allow_html=True)

        for emoji, label, key in NAV:
            if st.session_state.page == key:
                st.markdown(
                    f'<div style="background:linear-gradient(90deg,rgba(240,62,90,0.18),rgba(76,142,247,0.08));'
                    f'border-left:3px solid #f03e5a;border-radius:8px;padding:0.55rem calc(0.85rem - 3px);'
                    f'margin-bottom:2px;font-size:0.84rem;font-weight:600;color:#fff;width:100%;box-sizing:border-box;">'
                    f'{emoji}&nbsp;&nbsp;{label}</div>', unsafe_allow_html=True)
            else:
                if st.button(f"{emoji}  {label}", key=f"nav_{key}", use_container_width=True):
                    st.session_state.page = key; st.rerun()

        st.markdown('<div style="height:1px;background:#1e2d4a;margin:1rem 0;"></div>', unsafe_allow_html=True)
        theme    = st.selectbox("Chart theme", ["plotly_dark","plotly","plotly_white"])
        show_raw = st.toggle("Show data tables", value=False)

        if pipeline_ok:
            s = sd["summary"]
            st.markdown('<div style="height:1px;background:#1e2d4a;margin:1rem 0;"></div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#8b949e;margin-bottom:0.7rem;">Dataset</div>', unsafe_allow_html=True)
            for lbl, val in [("Documents",f"{s['total_docs']:,}"),("Platforms",str(s['sources'])),
                              ("Avg Sentiment",f"{s['avg_sentiment']:+.4f}"),("% Negative",f"{s['pct_negative']}%")]:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:0.28rem 0;border-bottom:1px solid #1e2d4a;">'
                    f'<span style="font-size:0.78rem;color:#8b949e;">{lbl}</span>'
                    f'<span style="font-size:0.82rem;font-weight:600;color:#e8edf5;font-family:JetBrains Mono,monospace;">{val}</span>'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);border-radius:8px;padding:0.7rem;font-size:0.78rem;color:#fcd34d;margin-top:1rem;">Run: python app.py</div>', unsafe_allow_html=True)

        st.markdown(f'<div style="margin-top:2rem;font-size:0.68rem;color:#484f58;text-align:center;">NLP Project · {datetime.now().strftime("%Y")}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    #  PAGE: OVERVIEW
    # ════════════════════════════════════════════════════════════
    if page == "overview":
        if pipeline_ok:
            s = sd["summary"]
            status = '<span class="badge badge-red">● Live Data</span>'
            docs   = f'<span class="badge badge-blue">📄 {s["total_docs"]:,} Documents</span>'
            srcs   = f'<span class="badge badge-gold">📡 {s["sources"]} Platforms</span>'
        else:
            status = '<span class="badge badge-gold">⚠ Awaiting Pipeline</span>'
            docs   = '<span class="badge badge-blue">📄 13,046 Documents</span>'
            srcs   = '<span class="badge badge-gold">📡 7 Platforms</span>'

        st.markdown(f"""
        <div class="hero">
          <div class="hero-eyebrow">NLP Sentiment Intelligence Dashboard</div>
          <div class="hero-title">US / Israel – <span>Iran War</span><br>Coverage Analysis</div>
          <div class="hero-subtitle">Multi-source sentiment analysis and topic modeling across news outlets and social media.
          Powered by VADER &amp; TextBlob, Logistic Regression, K-Means, and LDA.</div>
          <div class="hero-badges">{status}{docs}{srcs}
            <span class="badge badge-green">🧠 VADER + TextBlob + LDA</span>
            <span class="badge badge-red">🎯 5 Angles</span>
            <span class="badge badge-blue">🗂 6 Topics</span>
            <span class="badge badge-gold">📐 Supervised + Unsupervised ML</span>
          </div>
        </div>""", unsafe_allow_html=True)

        if not pipeline_ok:
            st.markdown("""
            <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);border-radius:10px;padding:1rem 1.2rem;margin-bottom:1.5rem;">
              <div style="font-weight:700;color:#fcd34d;margin-bottom:0.3rem;">⚠ Pipeline outputs not found</div>
              <div style="font-size:0.83rem;color:#8b949e;">Run <code style="background:#10172a;padding:1px 6px;border-radius:4px;">python app.py</code> to generate all data.</div>
            </div>""", unsafe_allow_html=True)

        section("📊","Overview","Key metrics from the full NLP pipeline")
        if pipeline_ok:
            s = sd["summary"]; tb = sd.get("textblob_summary",{}); sup = sd.get("supervised",{})
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: kpi("📄",f"{s['total_docs']:,}","Total Documents",f"{s['sources']} platforms","blue")
            with c2: kpi("🎭",f"{s['avg_sentiment']:+.4f}","Avg VADER","Predominantly negative","red")
            with c3: kpi("📉",f"{s['pct_negative']}%","VADER Negative",f"{s['pct_positive']}% positive","red")
            with c4:
                tb_val = f"{tb.get('avg_sentiment',0):+.4f}" if tb else "—"
                kpi("🔵",tb_val,"Avg TextBlob",f"{tb.get('pct_negative','—')}% negative" if tb else "Run pipeline","blue")
            with c5:
                acc = sup.get("vader",{}).get("accuracy")
                kpi("🤖",f"{acc*100:.1f}%" if acc else "—","Classifier Accuracy","VADER TF-IDF LogReg","gold")
        else:
            cols = st.columns(5)
            for col,ico,lbl in zip(cols,["📄","🎭","📉","🔵","🤖"],["Total Docs","Avg VADER","% Negative","TextBlob","Classifier"]):
                with col: kpi(ico,"—",lbl,"Run pipeline","blue")
        divider()

        section("🧭","Explore the Dashboard","Click a section to jump straight to the analysis")
        c1,c2,c3 = st.columns(3)
        for col,em,ttl,nk,desc in [
            (c1,"🎭","Sentiment Analysis","sentiment","VADER & TextBlob across 7 platforms and 5 angles"),
            (c2,"🗂","Topic Modeling",   "topics",   "6 LDA topics with keywords, word clouds, heatmaps"),
            (c3,"🤖","ML Results",       "ml_results","Logistic Regression accuracy, confusion matrix, K-Means"),
        ]:
            with col:
                st.markdown(f'<div style="background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:1.2rem;height:130px;"><div style="font-size:1.4rem;margin-bottom:0.4rem;">{em}</div><div style="font-weight:700;color:#fff;font-size:0.95rem;margin-bottom:0.3rem;">{ttl}</div><div style="font-size:0.78rem;color:var(--muted);">{desc}</div></div>', unsafe_allow_html=True)
                if st.button(f"Open {ttl}", key=f"q_{nk}", use_container_width=True):
                    st.session_state.page = nk; st.rerun()
        c4,c5,c6,c7 = st.columns(4)
        for col,em,ttl,nk in [(c4,"📋","Data Sources","data_sources"),(c5,"💬","Chat","chat"),(c6,"🔬","Classifier","classifier")]:
            with col:
                st.markdown(f'<div style="background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:1rem;height:90px;"><div style="font-size:1.2rem;margin-bottom:0.3rem;">{em}</div><div style="font-weight:700;color:#fff;font-size:0.88rem;">{ttl}</div></div>', unsafe_allow_html=True)
                if st.button(f"Go →", key=f"q2_{nk}", use_container_width=True):
                    st.session_state.page = nk; st.rerun()
        with c7:
            st.markdown('<div style="background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:1rem;height:90px;"><div style="font-size:1.2rem;margin-bottom:0.3rem;">↺</div><div style="font-weight:700;color:#fff;font-size:0.88rem;">Reload</div></div>', unsafe_allow_html=True)
            if st.button("Reload", key="q2_reload", use_container_width=True):
                st.rerun()

    # ════════════════════════════════════════════════════════════
    #  PAGE: SENTIMENT ANALYSIS
    # ════════════════════════════════════════════════════════════
    elif page == "sentiment":
        page_crumb("Sentiment Analysis")
        section("🎭","Sentiment Analysis","VADER & TextBlob across 7 platforms and 5 thematic angles")
        t_src,t_ang,t_dst,t_tb,t_med,t_det = st.tabs(["📡 By Platform","🔍 By Angle","📊 Distribution","🔵 TextBlob","📺 News vs Social","📋 Data"])

        if pipeline_ok:
            with t_src:
                df_s = pd.DataFrame(sd["by_source"]).sort_values("avg_sentiment")
                bc   = [COLORS["red"] if v<-0.1 else COLORS["gold"] if v<0.05 else COLORS["green"] for v in df_s["avg_sentiment"]]
                fig  = go.Figure(go.Bar(x=df_s["avg_sentiment"],y=df_s["source"],orientation="h",
                    marker=dict(color=bc,line=dict(color="rgba(0,0,0,0)")),
                    text=[f"{v:+.4f}" for v in df_s["avg_sentiment"]],textposition="outside",textfont=dict(color="#8b949e",size=11)))
                fig.add_vline(x=0,line_dash="dot",line_color="#30363d",line_width=1.5)
                fig.update_layout(title="Average VADER Score by Platform",xaxis_title="Compound Score")
                st.plotly_chart(_chart(fig), use_container_width=True)
                sd2 = pd.DataFrame(sd["source_detail"])
                if all(c in sd2.columns for c in ["Negative","Neutral","Positive"]):
                    fig2 = go.Figure()
                    for col,color,nm in [("Negative",COLORS["red"],"Negative"),("Neutral",COLORS["gold"],"Neutral"),("Positive",COLORS["green"],"Positive")]:
                        fig2.add_trace(go.Bar(name=nm,x=sd2["source"],y=sd2[col],marker_color=color,marker_line=dict(color="rgba(0,0,0,0)")))
                    fig2.update_layout(barmode="stack",title="Sentiment Counts by Platform")
                    st.plotly_chart(_chart(fig2), use_container_width=True)

            with t_ang:
                df_a = pd.DataFrame(sd["by_angle"]).sort_values("avg_sentiment")
                cl,cr = st.columns(2)
                with cl:
                    bc2 = [COLORS["red"] if v<-0.3 else COLORS["gold"] if v<0 else COLORS["green"] for v in df_a["avg_sentiment"]]
                    fig = go.Figure(go.Bar(x=df_a["avg_sentiment"],y=df_a["angle"],orientation="h",
                        marker=dict(color=bc2,line=dict(color="rgba(0,0,0,0)")),
                        text=[f"{v:+.4f}" for v in df_a["avg_sentiment"]],textposition="outside",textfont=dict(color="#8b949e",size=11)))
                    fig.add_vline(x=0,line_dash="dot",line_color="#30363d",line_width=1.5)
                    fig.update_layout(title="Avg Sentiment by Angle")
                    st.plotly_chart(_chart(fig), use_container_width=True)
                with cr:
                    fig2 = go.Figure(go.Bar(x=df_a["pct_positive"],y=df_a["angle"],orientation="h",
                        marker=dict(color=COLORS["green"],opacity=0.85,line=dict(color="rgba(0,0,0,0)")),
                        text=[f"{v:.1f}%" for v in df_a["pct_positive"]],textposition="outside",textfont=dict(color="#8b949e",size=11)))
                    fig2.update_layout(title="% Positive by Angle")
                    st.plotly_chart(_chart(fig2), use_container_width=True)
                disp = df_a.rename(columns={"angle":"Angle","avg_sentiment":"Avg VADER","pct_positive":"% Positive","doc_count":"Docs"})
                disp["Avg VADER"] = disp["Avg VADER"].map("{:+.4f}".format)
                disp["% Positive"] = disp["% Positive"].map("{:.1f}%".format)
                st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

            with t_dst:
                dist  = sd["sentiment_distribution"]
                lbls  = list(dist.keys()); vals = list(dist.values())
                clr   = {"Positive":COLORS["green"],"Neutral":COLORS["gold"],"Negative":COLORS["red"]}
                pc    = [clr.get(l,COLORS["muted"]) for l in lbls]
                c1,c2 = st.columns(2)
                with c1:
                    fig = go.Figure(go.Pie(labels=lbls,values=vals,marker_colors=pc,hole=0.5,textinfo="label+percent",textfont=dict(size=12,color="#e6edf3")))
                    fig.update_layout(title="Sentiment Distribution",legend=dict(orientation="h",y=-0.1))
                    st.plotly_chart(_chart(fig), use_container_width=True)
                with c2:
                    fig2 = go.Figure(go.Bar(x=lbls,y=vals,marker_color=pc,marker_line=dict(color="rgba(0,0,0,0)"),
                        text=[f"{v:,}" for v in vals],textposition="outside",textfont=dict(color="#8b949e")))
                    fig2.update_layout(title="Absolute Counts",yaxis_title="Posts / Articles")
                    st.plotly_chart(_chart(fig2), use_container_width=True)

            with t_tb:
                tb_d = sd.get("textblob_summary",{})
                if tb_d:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Avg Polarity",f"{tb_d.get('avg_sentiment',0):+.4f}")
                    c2.metric("Positive",f"{tb_d.get('pct_positive',0)}%")
                    c3.metric("Negative",f"{tb_d.get('pct_negative',0)}%")
                tb_src = sd.get("textblob_by_source",[])
                if tb_src:
                    df_ts = pd.DataFrame(tb_src).sort_values("avg_sentiment")
                    bc3   = [COLORS["red"] if v<-0.05 else COLORS["gold"] if v<0.05 else COLORS["green"] for v in df_ts["avg_sentiment"]]
                    fig   = go.Figure(go.Bar(x=df_ts["avg_sentiment"],y=df_ts["source"],orientation="h",
                        marker=dict(color=bc3,line=dict(color="rgba(0,0,0,0)")),
                        text=[f"{v:+.4f}" for v in df_ts["avg_sentiment"]],textposition="outside",textfont=dict(color="#8b949e",size=11)))
                    fig.add_vline(x=0,line_dash="dot",line_color="#30363d",line_width=1.5)
                    fig.update_layout(title="TextBlob Polarity by Platform",xaxis_title="Polarity Score")
                    st.plotly_chart(_chart(fig), use_container_width=True)
                tb_ang = sd.get("textblob_by_angle",[])
                if tb_ang:
                    df_ta = pd.DataFrame(tb_ang).sort_values("avg_sentiment")
                    bc4   = [COLORS["red"] if v<-0.05 else COLORS["gold"] if v<0.05 else COLORS["green"] for v in df_ta["avg_sentiment"]]
                    fig   = go.Figure(go.Bar(x=df_ta["avg_sentiment"],y=df_ta["angle"],orientation="h",
                        marker=dict(color=bc4,line=dict(color="rgba(0,0,0,0)")),
                        text=[f"{v:+.4f}" for v in df_ta["avg_sentiment"]],textposition="outside",textfont=dict(color="#8b949e",size=11)))
                    fig.add_vline(x=0,line_dash="dot",line_color="#30363d",line_width=1.5)
                    fig.update_layout(title="TextBlob Polarity by Angle",xaxis_title="Polarity Score")
                    st.plotly_chart(_chart(fig), use_container_width=True)
                tb_dist = sd.get("textblob_distribution",{})
                if tb_dist:
                    fig = go.Figure(go.Pie(labels=list(tb_dist.keys()),values=list(tb_dist.values()),
                        marker_colors=[clr.get(l,COLORS["muted"]) for l in tb_dist],hole=0.5,
                        textinfo="label+percent",textfont=dict(size=12,color="#e6edf3")))
                    fig.update_layout(title="TextBlob Distribution",legend=dict(orientation="h",y=-0.1))
                    st.plotly_chart(_chart(fig), use_container_width=True)
                if not tb_d:
                    st.info("Install textblob and re-run `python app.py` for TextBlob results.")

            with t_med:
                p = BASE_DIR/"data"/"vader_news_vs_social.png"
                if p.exists(): st.image(str(p), caption="VADER: News vs Social Media", width="stretch")
                else: st.info("Run `python app.py` to generate this chart.")

            with t_det:
                if show_raw:
                    st.markdown("**VADER By Platform**")
                    st.dataframe(pd.DataFrame(sd["by_source"]), use_container_width=True, hide_index=True)
                    st.markdown("**VADER By Angle**")
                    st.dataframe(pd.DataFrame(sd["by_angle"]),  use_container_width=True, hide_index=True)
                    if sd.get("textblob_by_source"):
                        st.markdown("**TextBlob By Platform**")
                        st.dataframe(pd.DataFrame(sd["textblob_by_source"]), use_container_width=True, hide_index=True)
                else:
                    st.info("Enable **Show data tables** in the sidebar.")
        else:
            for tab in [t_src,t_ang,t_dst,t_tb,t_med,t_det]:
                with tab: st.info("Run `python app.py` to populate this section.")

    # ════════════════════════════════════════════════════════════
    #  PAGE: TOPIC MODELING
    # ════════════════════════════════════════════════════════════
    elif page == "topics":
        page_crumb("Topic Modeling")
        section("🗂","Topic Modeling — LDA","6 topics · 9,109 documents · top-10 keywords per topic")
        t_td,t_kw,t_wc,t_hm,t_bo = st.tabs(["📊 Distribution","🔑 Keywords","☁ Word Clouds","🗺 Heatmap","📡 By Source"])

        if pipeline_ok:
            topics = td["topics"]; df_t = pd.DataFrame(topics); pal = px.colors.qualitative.Bold
            with t_td:
                fig = go.Figure(go.Bar(x=df_t["count"],y=df_t["label"],orientation="h",
                    marker=dict(color=pal[:len(topics)],line=dict(color="rgba(0,0,0,0)")),
                    text=[f"{r['count']:,} ({r['percentage']}%)" for _,r in df_t.iterrows()],
                    textposition="outside",textfont=dict(color="#8b949e",size=11)))
                fig.update_layout(title="Topic Distribution — LDA",xaxis_title="Articles / Posts")
                st.plotly_chart(_chart(fig,420), use_container_width=True)
            with t_kw:
                cols = st.columns(2)
                for i,tp in enumerate(topics):
                    with cols[i%2]:
                        st.markdown(
                            f'<div style="background:var(--navy2);border:1px solid var(--border);border-radius:10px;padding:1rem;margin-bottom:0.8rem;">'
                            f'<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;color:{pal[i%len(pal)]};margin-bottom:0.3rem;">Topic {tp["id"]+1} · {tp["percentage"]}%</div>'
                            f'<div style="font-weight:600;color:#e6edf3;margin-bottom:0.6rem;font-size:0.88rem;">{tp["label"]}</div>'
                            f'<div style="display:flex;flex-wrap:wrap;gap:5px;">'
                            + "".join(f'<span style="background:var(--navy3);border:1px solid var(--border);border-radius:99px;padding:2px 10px;font-size:0.75rem;color:#c9d8f5;">{w}</span>' for w in tp["words"])
                            + "</div></div>", unsafe_allow_html=True)
            with t_wc:
                p = BASE_DIR/"data"/"wordclouds_topics.png"
                if p.exists(): st.image(str(p), caption="Word Clouds per LDA Topic", width="stretch")
                else: st.info("Run `python app.py` to generate word clouds.")
            with t_hm:
                p = BASE_DIR/"data"/"heatmap_document_topic.png"
                if p.exists(): st.image(str(p), caption="Document-Topic Heatmap", width="stretch")
                else: st.info("Run `python app.py` to generate the heatmap.")
            with t_bo:
                p = BASE_DIR/"data"/"topic_by_outlet.png"
                if p.exists():
                    st.image(str(p), caption="Topic Distribution by Source", width="stretch")
                elif td.get("topic_by_source"):
                    df_tbs = pd.DataFrame(td["topic_by_source"])
                    tcols  = [c for c in df_tbs.columns if c != "Source"]
                    fig2   = go.Figure()
                    for idx,tc in enumerate(tcols):
                        fig2.add_trace(go.Bar(name=tc[:35],x=df_tbs["Source"],y=df_tbs[tc],
                            marker_color=pal[idx%len(pal)],marker_line=dict(color="rgba(0,0,0,0)")))
                    fig2.update_layout(barmode="group",title="Topic Distribution by Source",xaxis_tickangle=-30)
                    st.plotly_chart(_chart(fig2,420), use_container_width=True)
        else:
            for tab in [t_td,t_kw,t_wc,t_hm,t_bo]:
                with tab: st.info("Run `python app.py` to populate this section.")

    # ════════════════════════════════════════════════════════════
    #  PAGE: ML RESULTS
    # ════════════════════════════════════════════════════════════
    elif page == "ml_results":
        page_crumb("ML Results")
        section("🤖","Machine Learning Results","Supervised: TF-IDF + Logistic Regression · Unsupervised: K-Means (k=5) + PCA")
        t_sup,t_cm,t_cl,t_pca,t_elb = st.tabs(["📐 Accuracy","🗺 Confusion Matrix","📦 Clusters","📍 PCA","📈 Elbow"])

        if pipeline_ok:
            sup = sd.get("supervised",{}); cl = sd.get("clustering",{})
            with t_sup:
                if sup:
                    c1,c2 = st.columns(2)
                    for col,method in [(c1,"vader"),(c2,"textblob")]:
                        with col:
                            res = sup.get(method,{}); lbl = "VADER" if method=="vader" else "TextBlob"
                            if res:
                                acc = res.get("accuracy",0)
                                st.markdown(
                                    f'<div style="background:var(--navy2);border:1px solid var(--border);border-radius:10px;padding:1.2rem;text-align:center;">'
                                    f'<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;color:#8b949e;margin-bottom:0.4rem;">{lbl} Classifier</div>'
                                    f'<div style="font-family:JetBrains Mono,monospace;font-size:2.4rem;font-weight:700;color:#10b981;">{acc*100:.1f}%</div>'
                                    f'<div style="font-size:0.8rem;color:#8b949e;margin-top:0.3rem;">TF-IDF + Logistic Regression</div>'
                                    f'<div style="font-size:0.78rem;color:#8b949e;margin-top:0.3rem;">Train: {res.get("n_train",0):,} · Test: {res.get("n_test",0):,}</div>'
                                    f'</div>', unsafe_allow_html=True)
                                rpt = res.get("report",{})
                                if rpt:
                                    rows = [{"Class":k,"Precision":round(v.get("precision",0),3),
                                             "Recall":round(v.get("recall",0),3),
                                             "F1-Score":round(v.get("f1-score",0),3),
                                             "Support":int(v.get("support",0))}
                                            for k,v in rpt.items() if isinstance(v,dict)]
                                    if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                            else: st.info(f"{lbl} results not available.")
                else: st.info("Run `python app.py` to compute accuracy.")
            with t_cm:
                c1,c2 = st.columns(2)
                for col,m in [(c1,"vader"),(c2,"textblob")]:
                    with col:
                        p = BASE_DIR/"data"/f"confusion_matrix_{m}.png"
                        lbl = "VADER" if m=="vader" else "TextBlob"
                        if p.exists(): st.image(str(p), caption=f"Confusion Matrix — {lbl}", width="stretch")
                        else: st.info(f"{lbl} confusion matrix not found.")
            with t_cl:
                for fn,cap in [("cluster_distribution.png","K-Means Cluster Distribution (k=5)"),
                               ("cluster_vs_vader.png","Cluster vs VADER Sentiment")]:
                    p = BASE_DIR/"data"/fn
                    if p.exists(): st.image(str(p), caption=cap, width="stretch")
                if cl.get("clusters"):
                    st.markdown("**Top keywords per cluster:**")
                    cols3 = st.columns(3)
                    for i,ci in enumerate(cl["clusters"]):
                        with cols3[i%3]:
                            st.markdown(
                                f'<div style="background:var(--navy2);border:1px solid var(--border);border-radius:8px;padding:0.8rem;margin-bottom:0.6rem;">'
                                f'<div style="font-size:0.7rem;font-weight:700;color:#3b82f6;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem;">Cluster {ci["cluster"]} · {ci["count"]:,} docs</div>'
                                f'<div style="font-size:0.8rem;color:#c9d8f5;">{", ".join(ci["top_words"][:6])}</div>'
                                f'</div>', unsafe_allow_html=True)
            with t_pca:
                p = BASE_DIR/"data"/"cluster_pca.png"
                if p.exists(): st.image(str(p), caption="PCA Cluster Visualization", width="stretch")
                else: st.info("Run `python app.py` to generate PCA chart.")
            with t_elb:
                p = BASE_DIR/"data"/"elbow_method.png"
                if p.exists(): st.image(str(p), caption="Elbow Method — Optimal k", width="stretch")
                else: st.info("Run `python app.py` to generate elbow chart.")
        else:
            for tab in [t_sup,t_cm,t_cl,t_pca,t_elb]:
                with tab: st.info("Run `python app.py` to populate this section.")

    # ════════════════════════════════════════════════════════════
    #  PAGE: DATA SOURCES
    # ════════════════════════════════════════════════════════════
    elif page == "data_sources":
        page_crumb("Data Sources")
        section("📋","Data Sources & Collection Log","13,046 raw records across 7 platforms")
        SMETA = [
            {"Platform":"News Outlet","Type":"News Media",    "Examples":"BBC, Al Jazeera, Reuters, RT News","Raw Records":"~4,200","Notes":"Full articles; highest editorial quality"},
            {"Platform":"Twitter/X",  "Type":"Social Media",  "Examples":"Conflict hashtags & accounts",     "Raw Records":"~2,800","Notes":"Short-form; high emotional intensity"},
            {"Platform":"YouTube",    "Type":"Video/Comments","Examples":"War-coverage comment sections",    "Raw Records":"~2,100","Notes":"Mixed: video descriptions + comments"},
            {"Platform":"Facebook",   "Type":"Social Media",  "Examples":"News page posts & public groups", "Raw Records":"~1,500","Notes":"Community discussions"},
            {"Platform":"Reddit",     "Type":"Forum",         "Examples":"r/worldnews, r/geopolitics",      "Raw Records":"~1,100","Notes":"Long-form threads; nuanced debate"},
            {"Platform":"Instagram",  "Type":"Social Media",  "Examples":"News account posts",              "Raw Records":"~800",  "Notes":"Short captions"},
            {"Platform":"TikTok",     "Type":"Video/Comments","Examples":"War-related video captions",      "Raw Records":"~546",  "Notes":"Short-form; youth-dominated"},
        ]
        df_s2 = pd.DataFrame(SMETA)
        if pipeline_ok:
            lkp = {r["source"]: r["avg_sentiment"] for r in sd["by_source"]}
            df_s2["Avg Sentiment"] = df_s2["Platform"].map(lambda p: f"{lkp[p]:+.4f}" if p in lkp else "—")
        ct,ci = st.columns([3,1],gap="large")
        with ct: st.dataframe(df_s2, use_container_width=True, hide_index=True)
        with ci:
            st.markdown("""
            <div style="background:var(--navy2);border:1px solid var(--border);border-radius:10px;padding:1.2rem;">
            <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;color:#8b949e;margin-bottom:0.8rem;">Collection Info</div>
            <div style="font-size:0.82rem;color:#c9d8f5;line-height:1.7;">
            <strong style="color:#e8edf5;">Format:</strong> Excel (.xlsx)<br>
            <strong style="color:#e8edf5;">Language:</strong> English<br>
            <strong style="color:#e8edf5;">Time frame:</strong> 2024–2025<br>
            <strong style="color:#e8edf5;">Pipeline:</strong> VADER + LDA<br>
            <strong style="color:#e8edf5;">After filter:</strong> 12,207 docs
            </div></div>""", unsafe_allow_html=True)
        divider()
        section("🕷","Data Collection","Unified scraper — run: python app.py scraper info")
        st.markdown("""
        <div style="background:var(--navy2);border:1px solid var(--border);border-radius:10px;padding:1.2rem;font-size:0.84rem;color:#c9d8f5;line-height:1.8;">
        <strong style="color:#e8edf5;">Sources:</strong> NewsAPI · Apify (Facebook/TikTok/Instagram) · Tweepy v2 · yt-dlp · Reddit JSON API<br>
        <strong style="color:#e8edf5;">Output:</strong> all_sources_combined.xlsx + per-source Excel files<br>
        <strong style="color:#e8edf5;">CLI:</strong> <code style="background:var(--navy3);padding:1px 7px;border-radius:4px;">python scraper.py</code>
        or <code style="background:var(--navy3);padding:1px 7px;border-radius:4px;">python scraper.py --source news</code>
        </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    #  PAGE: CHAT ASSISTANT
    # ════════════════════════════════════════════════════════════
    elif page == "chat":
        page_crumb("Chat Assistant")
        section("💬","Chat Assistant","Ask natural-language questions about the war coverage data")
        chat_col, tip_col = st.columns([2,1])
        with chat_col:
            box = st.container(height=500)
            with box:
                if not st.session_state.chat_history:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown("Hello! Ask me about:\n- *'What is the overall sentiment?'*\n- *'Which platform is most negative?'*\n- *'What topics were found?'*\n- *'What is the classifier accuracy?'*")
                for entry in st.session_state.chat_history:
                    with st.chat_message(entry["role"], avatar="🤖" if entry["role"]=="assistant" else "👤"):
                        st.markdown(entry["content"])
            user_in = st.chat_input("Ask a question…", key="chat_input")
            if user_in:
                st.session_state.chat_history.append({"role":"user","content":user_in})
                st.session_state.chat_history.append({"role":"assistant","content":get_chatbot_response(user_in,sd,td)})
                st.rerun()
            if st.session_state.chat_history:
                if st.button("Clear conversation", type="secondary", use_container_width=True):
                    st.session_state.chat_history = []; st.rerun()
        with tip_col:
            st.markdown("""
            <div style="background:var(--navy2);border:1px solid var(--border);border-radius:10px;padding:1.2rem;margin-top:2rem;">
            <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;color:#8b949e;margin-bottom:0.8rem;">Try asking about</div>
            <div style="font-size:0.82rem;color:#c9d8f5;line-height:2.1;">
            • Overall sentiment<br>• Military operations<br>• Economic impact<br>• Geopolitical tensions<br>
            • Media & propaganda<br>• War support / opposition<br>• LDA topics<br>• Platform comparisons<br>
            • TextBlob analysis<br>• Classifier accuracy<br>• K-Means clusters<br>• Preprocessing pipeline
            </div></div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    #  PAGE: LIVE CLASSIFIER
    # ════════════════════════════════════════════════════════════
    elif page == "classifier":
        page_crumb("Live Classifier")
        section("🔬","Live Sentiment Classifier","Paste any text — VADER analyses it in real-time")
        clf_col,res_col = st.columns([1,1],gap="large")
        with clf_col:
            with st.container(border=True):
                user_text = st.text_area("Paste a headline, comment, or excerpt:",
                    placeholder="e.g. 'Iran launched a barrage of ballistic missiles toward Tel Aviv...'",
                    height=160, key="clf_input")
                run_clf = st.button("⚡  Analyse Sentiment", type="primary", use_container_width=True)
            if run_clf and user_text.strip():
                try:
                    from nltk.sentiment.vader import SentimentIntensityAnalyzer
                    import nltk
                    try: nltk.data.find("sentiment/vader_lexicon.zip")
                    except LookupError: nltk.download("vader_lexicon", quiet=True)
                    with st.spinner("Running VADER…"):
                        time.sleep(0.15)
                        scores   = SentimentIntensityAnalyzer().polarity_scores(user_text)
                    compound = scores["compound"]
                    if   compound >= 0.05:  lbl,ccls,eico = "Positive","pos","🟢"
                    elif compound <= -0.05: lbl,ccls,eico = "Negative","neg","🔴"
                    else:                   lbl,ccls,eico = "Neutral","neu","🟡"
                    st.session_state.classifier_result = {"label":lbl,"card_cls":ccls,"emoji":eico,"compound":compound,"scores":scores}
                    st.rerun()
                except ImportError:
                    st.error("Run: `pip install nltk`")
            elif run_clf:
                st.warning("Please enter some text first.")
        with res_col:
            if st.session_state.classifier_result:
                r = st.session_state.classifier_result; sc = r["scores"]
                st.markdown(
                    f'<div class="clf-card {r["card_cls"]}"><div class="clf-label">VADER Sentiment Result</div>'
                    f'<div class="clf-value">{r["emoji"]} {r["label"]}</div>'
                    f'<div class="clf-score">Compound: <strong>{r["compound"]:+.4f}</strong></div></div>',
                    unsafe_allow_html=True)
                gc = COLORS["green"] if r["compound"]>=0.05 else (COLORS["red"] if r["compound"]<=-0.05 else COLORS["gold"])
                fig_g = go.Figure(go.Indicator(mode="gauge+number",value=round(r["compound"]*100,1),
                    title={"text":"Compound × 100","font":{"size":12,"color":"#8b949e"}},
                    number={"font":{"color":"#e8edf5","family":"JetBrains Mono"}},
                    gauge={"axis":{"range":[-100,100],"tickcolor":"#8b949e"},"bar":{"color":gc,"thickness":0.25},
                           "bgcolor":"#10172a","bordercolor":"#1e2d4a",
                           "steps":[{"range":[-100,-5],"color":"rgba(233,69,96,0.1)"},
                                    {"range":[-5,5],   "color":"rgba(245,158,11,0.1)"},
                                    {"range":[5,100],  "color":"rgba(16,185,129,0.1)"}]}))
                fig_g.update_layout(height=220,margin=dict(l=20,r=20,t=30,b=10),paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#8b949e"))
                st.plotly_chart(fig_g, use_container_width=True)
                with st.expander("Full VADER breakdown"):
                    ca,cb,cc,cd = st.columns(4)
                    ca.metric("Positive", f"{sc['pos']:.3f}")
                    cb.metric("Neutral",  f"{sc['neu']:.3f}")
                    cc.metric("Negative", f"{sc['neg']:.3f}")
                    cd.metric("Compound", f"{sc['compound']:+.4f}")
                if pipeline_ok:
                    s = sd["summary"]
                    st.markdown(f'<div style="background:var(--navy2);border:1px solid var(--border);border-radius:10px;padding:1rem;font-size:0.8rem;color:#8b949e;margin-top:1rem;"><strong style="color:#e8edf5;">Corpus baseline:</strong> avg VADER = {s["avg_sentiment"]:+.4f} ({s["pct_negative"]}% negative across {s["total_docs"]:,} docs)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background:var(--navy2);border:1px solid var(--border);border-radius:14px;padding:3rem 2rem;text-align:center;color:#8b949e;"><div style="font-size:2.5rem;margin-bottom:0.8rem;">🔬</div><div style="font-size:0.9rem;">Enter text on the left and click<br><strong style="color:#e8edf5;">Analyse Sentiment</strong>.</div></div>', unsafe_allow_html=True)

    # ── Footer (all pages) ───────────────────────────────────────
    st.markdown(
        f'<div style="margin-top:3rem;padding:1.5rem;background:linear-gradient(135deg,var(--navy2),var(--navy));'
        f'border:1px solid var(--border);border-radius:12px;text-align:center;">'
        f'<div style="font-size:0.85rem;font-weight:700;color:#e8edf5;margin-bottom:0.3rem;">US / Israel – Iran War · Sentiment Intelligence Dashboard</div>'
        f'<div style="font-size:0.75rem;color:#484f58;">Built with Streamlit · VADER · LDA · {datetime.now().strftime("%Y")}</div>'
        f'</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  CLI MODE  —  python app.py
# ═════════════════════════════════════════════════════════════
else:
    import os, sys, json, re, string, warnings, html as html_mod, shutil, subprocess, argparse
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import seaborn as sns
    warnings.filterwarnings("ignore")

    # ── Notebook runner (from pipeline.py) ──────────────────────
    NOTEBOOKS = [
        ("Final_Load.ipynb",     "Step 1/3 — Preprocessing"),
        ("VADER (3).ipynb",      "Step 2/3 — Sentiment Analysis (VADER)"),
        ("Topic Modeling.ipynb", "Step 3/3 — Topic Modeling (LDA)"),
    ]

    def run_notebook(path, label):
        print(f"\n{'='*60}\n  {label}\n  {path}\n{'='*60}")
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert",
             "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=900", path],
            capture_output=False)
        if result.returncode != 0:
            print(f"\n[ERR] '{path}' failed.")
            return False
        print(f"\n[OK] Done: {path}")
        return True

    # ── Full NLP pipeline (from generate_data.py) ───────────────
    def run_pipeline():
        SAVE_DIR  = "nlp_visualisations"
        DATA_FILE = "consolidated_war_data1.xlsx"
        os.makedirs(SAVE_DIR, exist_ok=True)
        os.makedirs("data",   exist_ok=True)

        print("=" * 60)
        print("  US/Israel-Iran War - NLP Data Generator")
        print("  (combined app.py — pipeline mode)")
        print("=" * 60)

        if not os.path.exists(DATA_FILE):
            print(f"\n[ERR] {DATA_FILE} not found."); sys.exit(1)

        print(f"\n[1/7] Loading {DATA_FILE}...")
        df = pd.read_excel(DATA_FILE)
        print(f"      {len(df):,} rows.  Columns: {list(df.columns)}")

        rename = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("source","platform","outlet") and "Source" not in rename.values(): rename[col] = "Source"
            if cl in ("content","text","body","post","comment","message") and "Content" not in rename.values(): rename[col] = "Content"
        if rename: df = df.rename(columns=rename)
        if "Source"  not in df.columns: df["Source"] = "Unknown"
        if "Content" not in df.columns: raise ValueError("Cannot find a text column.")
        df["Content"] = df["Content"].fillna("").astype(str)
        df = df[df["Content"].str.strip().str.len() > 10].reset_index(drop=True)
        print(f"      {len(df):,} rows after removing empties.")

        # ── Preprocessing ──────────────────────────────────────
        print("\n[2/7] Preprocessing text...")
        import nltk
        for res,kind in [("vader_lexicon","sentiment"),("stopwords","corpora"),("wordnet","corpora"),
                         ("punkt_tab","tokenizers"),("averaged_perceptron_tagger_eng","taggers")]:
            try: nltk.data.find(f"{kind}/{res}")
            except LookupError: nltk.download(res, quiet=True)

        from nltk.corpus import stopwords as sw, wordnet
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        from nltk import pos_tag

        try:
            import emoji as emoji_mod; HAS_EMOJI = True
        except ImportError:
            HAS_EMOJI = False; print("      [WARN] emoji package not found")
        try:
            from langdetect import detect, LangDetectException; HAS_LANGDETECT = True
        except ImportError:
            HAS_LANGDETECT = False; print("      [WARN] langdetect not found")

        STOPWORDS = sw.words("english")
        CUSTOM_SW = {
            "rt","amp","via","like","lol","lmao","omg","smh","gonna","gotta","wanna",
            "dont","doesnt","didnt","cant","wont","isnt","wasnt","wouldnt","shouldnt",
            "couldnt","aint","im","ive","youre","theyre","theyve","weve","hes","shes",
            "thats","whats","lets","youve","youll","theyll","hadnt","hasnt",
            "said","reported","according","says","told","added","noted","sticker",
            "reply","retweet","share","comment","subscribe","video","watch","click",
            "link","bio","follow","page","also","even","still","much","many","really",
            "just","thing","things","something","everything","nothing","anyone",
            "everyone","someone","people","guy","guys","got","get","gets","getting",
            "go","going","gone","went","come","came","coming","back","well","way",
            "know","think","make","made","take","took","see","look","put","use","used",
            "one","two","would","could","may","might","need","want","right","good",
            "new","first","last","long","great","little","big","old","year","years",
            "time","day","days","bbc","cnn","reuters","associated","press","read",
            "source","updated","published","copyright","reserved","rights","news",
            "article","report",
        }
        ALL_SW = set(STOPWORDS).union(CUSTOM_SW)

        def preprocess_text(text):
            if pd.isna(text): return ""
            text = str(text).lower()
            text = re.sub(r"\[sticker\]","",text)
            if HAS_EMOJI: text = emoji_mod.replace_emoji(text,replace="")
            text = re.sub(r"http\S+|www\S+|t\.co/\S+","",text)
            text = re.sub(r"<.*?>","",text)
            text = html_mod.unescape(text)
            text = re.sub(r"@\w+","",text)
            text = re.sub(r"#(\w+)",r"\1",text)
            text = re.sub(r"[^\x00-\x7F]+","",text)
            text = text.translate(str.maketrans("","",string.punctuation))
            text = re.sub(r"\d+","",text)
            text = re.sub(r"\b[^ia\s]\b","",text)
            text = re.sub(r"\s+"," ",text).strip()
            return " ".join(w for w in text.split() if w not in ALL_SW)

        lemmatizer = WordNetLemmatizer()
        def get_wn_pos(tag):
            if tag.startswith("J"): return wordnet.ADJ
            if tag.startswith("V"): return wordnet.VERB
            if tag.startswith("R"): return wordnet.ADV
            return wordnet.NOUN
        def lemmatize_text(text):
            if not isinstance(text,str) or not text.strip(): return ""
            tokens = word_tokenize(text)
            return " ".join(lemmatizer.lemmatize(w,get_wn_pos(t)) for w,t in pos_tag(tokens))
        def is_english(text):
            if not HAS_LANGDETECT: return True
            if not isinstance(text,str) or len(text.strip())<3: return False
            try:
                from langdetect import detect
                return detect(text) == "en"
            except: return False

        df["clean_text"]      = df["Content"].apply(preprocess_text)
        df["lemmatized_text"] = df["clean_text"].apply(lemmatize_text)
        before = len(df)
        df["is_english"] = df["clean_text"].apply(is_english)
        df = df[df["is_english"]].drop(columns=["is_english"]).reset_index(drop=True)
        print(f"      Language filter: kept {len(df):,} / {before:,} rows")

        # ── VADER ──────────────────────────────────────────────
        print("\n[3/7] VADER sentiment analysis...")
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        df["vader_compound"] = df["clean_text"].apply(lambda t: sid.polarity_scores(str(t))["compound"] if t else 0.0)
        df["vader_label"]    = df["vader_compound"].apply(lambda s: "Positive" if s>=0.05 else ("Negative" if s<=-0.05 else "Neutral"))
        vader_dist = df["vader_label"].value_counts().to_dict()
        print(f"      VADER: {vader_dist}  avg={df['vader_compound'].mean():+.4f}")

        by_source_vader = (df.groupby("Source")["vader_compound"].mean().reset_index()
                           .rename(columns={"Source":"source","vader_compound":"avg_sentiment"}))
        by_source_vader["avg_sentiment"] = by_source_vader["avg_sentiment"].round(4)
        source_detail = (df.groupby("Source")["vader_label"].value_counts().unstack(fill_value=0).reset_index()
                         .rename(columns={"Source":"source"}))
        source_detail.columns.name = None
        for lbl in ["Positive","Neutral","Negative"]:
            if lbl not in source_detail.columns: source_detail[lbl] = 0

        ANGLE_KW = {
            "Military Operations":     ["drone","missile","attack","fighter","military","defense","force","weapon"],
            "Geopolitical Tensions":   ["middle east","africa","asia","iran","israel","allies","international","diplomacy"],
            "Economic Impact":         ["oil","energy","gas","strait","hormuz","market","price","economy","trade"],
            "Media / Propaganda":      ["bbc","reuters","media","misinformation","bias","propaganda","news","trust","youtube","facebook"],
            "War Support / Opposition":["support","protest","peace","against","ally","sanction","condemn"],
        }
        angle_rows = []
        for angle,kws in ANGLE_KW.items():
            mask = df["clean_text"].str.contains("|".join(kws),case=False,na=False)
            sub  = df[mask]
            if len(sub)==0: continue
            angle_rows.append({"angle":angle,"avg_sentiment":round(float(sub["vader_compound"].mean()),4),
                                "pct_positive":round(float((sub["vader_label"]=="Positive").mean()*100),1),
                                "doc_count":int(len(sub))})
        by_angle_vader = pd.DataFrame(angle_rows)

        # VADER charts
        sc_colors = {"Positive":"#2ecc71","Neutral":"#f39c12","Negative":"#e74c3c"}
        vc = df["vader_label"].value_counts()
        fig,ax = plt.subplots(figsize=(10,6))
        bars = ax.bar(vc.index,vc.values,color=[sc_colors.get(l,"#3498db") for l in vc.index],edgecolor="white",linewidth=1.5,width=0.6)
        for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+15,f"{int(bar.get_height()):,}",ha="center",va="bottom",fontsize=12,fontweight="bold")
        ax.set_title("VADER Sentiment Distribution",fontsize=14,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/vader_sentiment_distribution.png",dpi=150,bbox_inches="tight"); plt.close()

        palette = {"News Outlet":"#2166ac","Twitter":"#1DA1F2","Reddit":"#FF4500","YouTube":"#FF0000",
                   "Facebook":"#1877F2","TikTok":"#010101","Instagram":"#C13584"}
        fig,ax = plt.subplots(figsize=(12,6))
        sns.barplot(data=df,x="Source",y="vader_compound",
                    palette={s:palette.get(s,"#888") for s in df["Source"].unique()},
                    edgecolor="white",linewidth=1.5,ax=ax,errorbar=None)
        ax.axhline(0,color="black",linewidth=0.8,linestyle="--",alpha=0.5)
        ax.set_title("VADER Sentiment by Source",fontsize=14,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
        plt.xticks(rotation=45,ha="right"); plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/vader_sentiment_by_source.png",dpi=150,bbox_inches="tight"); plt.close()

        angles_dict = {r["angle"]:r["avg_sentiment"] for r in angle_rows}
        cmap_d = cm.get_cmap("tab10",len(angles_dict))
        fig,ax = plt.subplots(figsize=(12,6))
        bars = ax.bar(list(angles_dict.keys()),list(angles_dict.values()),
                      color=[cmap_d(i) for i in range(len(angles_dict))],edgecolor="white",linewidth=1.5,width=0.6)
        for bar in bars:
            yv = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2,yv+(0.003 if yv>=0 else -0.015),f"{yv:+.4f}",
                    ha="center",va="bottom" if yv>=0 else "top",fontsize=10,fontweight="bold")
        ax.axhline(0,color="black",linewidth=0.8,linestyle="--",alpha=0.5)
        ax.set_title("VADER Sentiment Across Conflict Dimensions",fontsize=14,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
        plt.xticks(rotation=15,ha="right"); plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/vader_conflict_dimensions.png",dpi=150,bbox_inches="tight"); plt.close()

        SOCIAL = ["Twitter","Reddit","YouTube","Facebook","TikTok","Instagram"]
        df["media_type"] = df["Source"].apply(lambda s: "Social Media" if s in SOCIAL else "Traditional News")
        overall_media    = df.groupby("media_type")["vader_compound"].mean().round(4)
        fig,ax = plt.subplots(figsize=(8,6))
        types  = overall_media.index.tolist(); scores = overall_media.values.tolist()
        clrs2  = ["#d73027" if t=="Social Media" else "#2166ac" for t in types]
        bars   = ax.bar(types,scores,color=clrs2,width=0.45,edgecolor="white",linewidth=1.5)
        for bar,score in zip(bars,scores): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,f"{score:+.4f}",ha="center",va="bottom",fontsize=12,fontweight="bold")
        ax.axhline(0,color="black",linewidth=0.8,linestyle="--"); ax.set_title("VADER: News vs Social Media",fontsize=14,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/vader_news_vs_social.png",dpi=150,bbox_inches="tight"); plt.close()
        print("      [OK] VADER charts saved")

        # ── TextBlob ───────────────────────────────────────────
        print("\n[4/7] TextBlob sentiment analysis...")
        try:
            from textblob import TextBlob
            df["tb_score"] = df["clean_text"].apply(lambda t: TextBlob(str(t)).sentiment.polarity if t else 0.0)
            df["tb_label"] = df["tb_score"].apply(lambda s: "Positive" if s>0 else ("Negative" if s<0 else "Neutral"))
            tb_dist = df["tb_label"].value_counts().to_dict()
            print(f"      TextBlob: {tb_dist}  avg={df['tb_score'].mean():+.4f}")
            by_source_tb = (df.groupby("Source")["tb_score"].mean().reset_index()
                            .rename(columns={"Source":"source","tb_score":"avg_sentiment"}))
            by_source_tb["avg_sentiment"] = by_source_tb["avg_sentiment"].round(4)
            tb_angle_rows = []
            for angle,kws in ANGLE_KW.items():
                mask = df["clean_text"].str.contains("|".join(kws),case=False,na=False)
                sub  = df[mask]
                if len(sub)==0: continue
                tb_angle_rows.append({"angle":angle,"avg_sentiment":round(float(sub["tb_score"].mean()),4),
                                      "pct_positive":round(float((sub["tb_label"]=="Positive").mean()*100),1),
                                      "doc_count":int(len(sub))})
            by_angle_tb = pd.DataFrame(tb_angle_rows)
            for fn,col,ttl in [("textblob_sentiment_distribution.png","tb_label","TextBlob Sentiment Distribution"),
                               ("textblob_sentiment_by_source.png","tb_score","TextBlob Sentiment by Source")]:
                fig,ax = plt.subplots(figsize=(10,6) if "dist" in fn else (12,6))
                if "dist" in fn:
                    tc = df[col].value_counts()
                    bars = ax.bar(tc.index,tc.values,color=[sc_colors.get(l,"#3498db") for l in tc.index],edgecolor="white",linewidth=1.5,width=0.6)
                    for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+15,f"{int(bar.get_height()):,}",ha="center",va="bottom",fontsize=12,fontweight="bold")
                else:
                    sns.barplot(data=df,x="Source",y=col,palette={s:palette.get(s,"#888") for s in df["Source"].unique()},edgecolor="white",linewidth=1.5,ax=ax,errorbar=None)
                    ax.axhline(0,color="black",linewidth=0.8,linestyle="--",alpha=0.5); plt.xticks(rotation=45,ha="right")
                ax.set_title(ttl,fontsize=14,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/{fn}",dpi=150,bbox_inches="tight"); plt.close()
            tb_a = {r["angle"]:r["avg_sentiment"] for r in tb_angle_rows}
            fig,ax = plt.subplots(figsize=(12,6))
            bars = ax.bar(list(tb_a.keys()),list(tb_a.values()),color=[cmap_d(i) for i in range(len(tb_a))],edgecolor="white",linewidth=1.5,width=0.6)
            for bar in bars:
                yv = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2,yv+(0.002 if yv>=0 else -0.008),f"{yv:+.4f}",ha="center",va="bottom" if yv>=0 else "top",fontsize=10,fontweight="bold")
            ax.axhline(0,color="black",linewidth=0.8,linestyle="--",alpha=0.5)
            ax.set_title("TextBlob Sentiment Across Conflict Dimensions",fontsize=14,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
            plt.xticks(rotation=15,ha="right"); plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/textblob_conflict_dimensions.png",dpi=150,bbox_inches="tight"); plt.close()
            HAS_TB = True; print("      [OK] TextBlob charts saved")
        except ImportError:
            print("      [WARN] textblob not installed — run: pip install textblob")
            HAS_TB = False
            df["tb_score"] = 0.0; df["tb_label"] = "Neutral"
            by_source_tb = pd.DataFrame(columns=["source","avg_sentiment"])
            tb_angle_rows = []; tb_dist = {}; by_angle_tb = pd.DataFrame()

        # ── Supervised ML ──────────────────────────────────────
        print("\n[5/7] Supervised learning (TF-IDF + Logistic Regression)...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

        df["clean_text"] = df["clean_text"].fillna("")
        supervised_results = {}
        for method,lcol in [("vader","vader_label"),("textblob","tb_label")]:
            if method=="textblob" and not HAS_TB: continue
            try:
                vec = TfidfVectorizer(max_features=5000)
                X   = vec.fit_transform(df["clean_text"]); y = df[lcol]
                Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
                clf = LogisticRegression(max_iter=10000)
                clf.fit(Xtr,ytr); yp = clf.predict(Xte)
                acc = round(accuracy_score(yte,yp),4)
                rpt = classification_report(yte,yp,output_dict=True)
                print(f"      {method.upper()} accuracy: {acc:.4f}")
                cm_arr = confusion_matrix(yte,yp,labels=sorted(y.unique()))
                fig,ax = plt.subplots(figsize=(8,6))
                sns.heatmap(cm_arr,annot=True,fmt="d",cmap="YlOrRd",
                            xticklabels=sorted(y.unique()),yticklabels=sorted(y.unique()),
                            linewidths=1,linecolor="white",cbar_kws={"shrink":0.8},ax=ax)
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix ({'VADER' if method=='vader' else 'TextBlob'} Supervised)",fontsize=14,fontweight="bold")
                plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/confusion_matrix_{method}.png",dpi=150,bbox_inches="tight"); plt.close()
                supervised_results[method] = {"accuracy":acc,"n_train":int(Xtr.shape[0]),"n_test":int(Xte.shape[0]),
                    "classes":sorted(y.unique().tolist()),
                    "report":{k:v for k,v in rpt.items() if k not in ("accuracy","macro avg","weighted avg")}}
            except Exception as e: print(f"      [WARN] {method} classifier: {e}")
        print("      [OK] Supervised learning done")

        # ── K-Means ────────────────────────────────────────────
        print("\n[6/7] K-Means clustering (k=5)...")
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA as skPCA
        cluster_results = {}
        try:
            vec_cl = TfidfVectorizer(max_features=10000,stop_words="english")
            X_cl   = vec_cl.fit_transform(df["clean_text"]); terms = vec_cl.get_feature_names_out()
            km     = KMeans(n_clusters=5,random_state=42,n_init=10); df["cluster"] = km.fit_predict(X_cl)
            cl_kw  = []
            for i in range(5):
                ctr  = km.cluster_centers_[i]; tw = [terms[j] for j in ctr.argsort()[-10:]]
                cl_kw.append({"cluster":i,"top_words":tw,"count":int((df["cluster"]==i).sum())})
                print(f"      Cluster {i} ({cl_kw[-1]['count']:,} docs): {', '.join(tw[:6])}")
            cc = df["cluster"].value_counts().sort_index(); cmap_cl = cm.get_cmap("tab10",len(cc))
            fig,ax = plt.subplots(figsize=(10,6))
            bars = ax.bar(cc.index.astype(str),cc.values,color=[cmap_cl(i) for i in range(len(cc))],edgecolor="white",linewidth=1.5,width=0.6)
            for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+10,f"{int(bar.get_height()):,}",ha="center",va="bottom",fontsize=11,fontweight="bold")
            ax.set_title("K-Means Cluster Distribution (k=5)",fontsize=14,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/cluster_distribution.png",dpi=150,bbox_inches="tight"); plt.close()
            pca   = skPCA(n_components=2,random_state=42); Xr = pca.fit_transform(X_cl.toarray())
            cmap_p = cm.get_cmap("tab10",5)
            fig,ax = plt.subplots(figsize=(12,8))
            for i,color in zip(range(5),[cmap_p(i) for i in range(5)]):
                idx = df["cluster"]==i; ax.scatter(Xr[idx,0],Xr[idx,1],color=color,label=f"Cluster {i}",alpha=0.6,edgecolors="white",linewidth=0.3,s=30)
            ax.set_title("Cluster Visualization — PCA",fontsize=14,fontweight="bold"); ax.legend(title="Clusters",fontsize=10)
            ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/cluster_pca.png",dpi=150,bbox_inches="tight"); plt.close()
            ct_norm = pd.crosstab(df["cluster"],df["vader_label"],normalize="index")
            fig,ax  = plt.subplots(figsize=(10,6))
            sns.heatmap(ct_norm,annot=True,fmt=".2f",cmap="YlGnBu",linewidths=1,linecolor="white",cbar_kws={"shrink":0.8},ax=ax)
            ax.set_title("Cluster vs VADER Sentiment (Proportion)",fontsize=14,fontweight="bold"); plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/cluster_vs_vader.png",dpi=150,bbox_inches="tight"); plt.close()
            inertia = []
            for k in range(2,10): km2 = KMeans(n_clusters=k,random_state=42,n_init=10); km2.fit(X_cl); inertia.append(km2.inertia_)
            plt.figure(figsize=(8,5)); plt.plot(range(2,10),inertia,marker="o",color="#8e44ad",markerfacecolor="#f39c12",markeredgecolor="white")
            plt.title("Elbow Method — Optimal k"); plt.xlabel("Number of Clusters"); plt.ylabel("Inertia"); plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/elbow_method.png",dpi=150,bbox_inches="tight"); plt.close()
            cluster_results = {"n_clusters":5,"clusters":cl_kw}
            print("      [OK] Clustering done")
        except Exception as e: print(f"      [ERR] Clustering: {e}"); cluster_results = {}

        # ── LDA Topic Modeling ─────────────────────────────────
        print("\n[7/7] LDA Topic Modeling...")
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        from wordcloud import WordCloud

        TOPIC_LABELS = {
            0:"State Power, Governance & Political Leadership",
            1:"Regional Conflict & Proxy Warfare (Iran-Israel-Hezbollah)",
            2:"Online Discourse & Nuclear Conflict Narratives",
            3:"Military Operations & Weapons Systems",
            4:"Global Energy Markets & Strait of Hormuz",
            5:"Diplomatic Negotiations & International Response",
        }
        topic_results = None
        try:
            cv   = CountVectorizer(min_df=5,max_df=0.5,max_features=5000,ngram_range=(1,2),stop_words="english")
            cm2  = cv.fit_transform(df["lemmatized_text"].fillna("").tolist()); vocab = cv.get_feature_names_out()
            print(f"      DTM shape: {cm2.shape}")
            lda  = LatentDirichletAllocation(n_components=6,random_state=122,max_iter=100); lda.fit(cm2)
            dt   = lda.transform(cm2)
            df["topic_id"]    = dt.argmax(axis=1)
            df["topic_label"] = df["topic_id"].map(TOPIC_LABELS)
            df["topic_score"] = dt.max(axis=1)
            topics_export = []
            for i,comp in enumerate(lda.components_):
                ti  = comp.argsort()[-10:][::-1]; tw = [vocab[j] for j in ti]
                cnt = int((df["topic_id"]==i).sum()); pct = round(float((df["topic_id"]==i).mean()*100),1)
                topics_export.append({"id":i,"label":TOPIC_LABELS[i],"words":tw,"count":cnt,"percentage":pct})
                print(f"      Topic {i+1} ({pct}%): {', '.join(tw[:6])}")
            outlet_topic      = pd.crosstab(df["Source"],df["topic_label"]).reset_index()
            outlet_topic_list = outlet_topic.to_dict("records")
            fig,axes = plt.subplots(2,3,figsize=(18,10)); axes = axes.flatten()
            for i,comp in enumerate(lda.components_):
                ti2 = comp.argsort()[-10:][::-1]; ws = " ".join(vocab[j] for j in ti2)
                wc  = WordCloud(width=800,height=400,background_color="black",colormap="Set2",prefer_horizontal=0.7,relative_scaling=0).generate(ws)
                axes[i].imshow(wc); axes[i].set_title(f"Topic {i+1}\n{TOPIC_LABELS[i]}",fontsize=11,fontweight="bold"); axes[i].axis("off")
            plt.suptitle("Word Clouds Per Topic — LDA",fontsize=14,fontweight="bold"); plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/wordclouds_topics.png",dpi=150,bbox_inches="tight"); plt.close()
            tc2 = df["topic_label"].value_counts().reset_index(); tc2.columns = ["topic_label","article_count"]
            tc2["percentage"] = (tc2["article_count"]/tc2["article_count"].sum()*100).round(1)
            clrs_tc = cm.Set2(np.linspace(0,1,len(tc2)))
            fig,ax = plt.subplots(figsize=(12,6))
            bars = ax.barh(tc2["topic_label"],tc2["article_count"],color=clrs_tc,edgecolor="white",height=0.6)
            for bar,(cnt,pct) in zip(bars,zip(tc2["article_count"],tc2["percentage"])):
                ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,f"{cnt} ({pct}%)",va="center",fontsize=10)
            ax.set_title("Topic Distribution — LDA",fontsize=14,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/topic_distribution.png",dpi=150,bbox_inches="tight"); plt.close()
            pd.crosstab(df["Source"],df["topic_label"]).plot(kind="bar",figsize=(14,6),colormap="Set2",edgecolor="white",width=0.7)
            plt.title("Topic Distribution by Source",fontsize=14,fontweight="bold"); plt.xticks(rotation=30,ha="right")
            plt.legend(title="Topic",bbox_to_anchor=(1.05,1),loc="upper left",fontsize=8); plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/topic_by_outlet.png",dpi=150,bbox_inches="tight"); plt.close()
            df_dt = pd.DataFrame(dt[:20],columns=[f"Topic {i+1}" for i in range(6)])
            fig,ax = plt.subplots(figsize=(12,8))
            sns.heatmap(df_dt,cmap="viridis",annot=True,fmt=".2f",linewidths=0.5,ax=ax)
            ax.set_title("Document-Topic Distribution Heatmap (First 20)",fontsize=14,fontweight="bold"); plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/heatmap_document_topic.png",dpi=150,bbox_inches="tight"); plt.close()
            topic_results = {"n_topics":6,"topics":topics_export,"topic_by_source":outlet_topic_list}
            print("      [OK] LDA done")
        except Exception as e: print(f"      [ERR] LDA: {e}"); topic_results = None

        # ── Copy PNGs to data/ ─────────────────────────────────
        for fname in ["wordclouds_topics.png","topic_distribution.png","topic_by_outlet.png",
                      "heatmap_document_topic.png","vader_sentiment_distribution.png",
                      "vader_sentiment_by_source.png","vader_conflict_dimensions.png",
                      "vader_news_vs_social.png","textblob_sentiment_distribution.png",
                      "textblob_sentiment_by_source.png","textblob_conflict_dimensions.png",
                      "confusion_matrix_vader.png","confusion_matrix_textblob.png",
                      "cluster_distribution.png","cluster_pca.png","cluster_vs_vader.png","elbow_method.png"]:
            src = os.path.join(SAVE_DIR,fname)
            if os.path.exists(src): shutil.copy(src,os.path.join("data",fname))

        # ── Save JSON ──────────────────────────────────────────
        sentiment_results = {
            "summary":{"total_docs":int(len(df)),"sources":int(df["Source"].nunique()),
                       "avg_sentiment":round(float(df["vader_compound"].mean()),4),
                       "pct_positive":round(float((df["vader_label"]=="Positive").mean()*100),1),
                       "pct_neutral": round(float((df["vader_label"]=="Neutral").mean()*100),1),
                       "pct_negative":round(float((df["vader_label"]=="Negative").mean()*100),1)},
            "by_source":by_source_vader.to_dict("records"),
            "source_detail":source_detail.to_dict("records"),
            "by_angle":by_angle_vader.to_dict("records"),
            "sentiment_distribution":vader_dist,
            "textblob_summary":{"avg_sentiment":round(float(df["tb_score"].mean()),4),
                                "pct_positive":round(float((df["tb_label"]=="Positive").mean()*100),1),
                                "pct_neutral": round(float((df["tb_label"]=="Neutral").mean()*100),1),
                                "pct_negative":round(float((df["tb_label"]=="Negative").mean()*100),1)} if HAS_TB else {},
            "textblob_by_source":by_source_tb.to_dict("records"),
            "textblob_by_angle":tb_angle_rows,
            "textblob_distribution":tb_dist,
            "supervised":supervised_results,
            "clustering":cluster_results,
        }
        with open("data/sentiment_results.json","w",encoding="utf-8") as f:
            json.dump(sentiment_results,f,indent=2,ensure_ascii=False)
        print("\n[OK] Saved data/sentiment_results.json")
        if topic_results:
            with open("data/topic_results.json","w",encoding="utf-8") as f:
                json.dump(topic_results,f,indent=2,ensure_ascii=False)
            print("[OK] Saved data/topic_results.json")

        # ── Summary ────────────────────────────────────────────
        s2 = sentiment_results["summary"]
        print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
        print(f"  Documents : {s2['total_docs']:,}  |  Sources: {s2['sources']}")
        print(f"  VADER     : avg={s2['avg_sentiment']:+.4f}  pos={s2['pct_positive']}%  neg={s2['pct_negative']}%")
        if HAS_TB:
            tb2 = sentiment_results["textblob_summary"]
            print(f"  TextBlob  : avg={tb2['avg_sentiment']:+.4f}  pos={tb2['pct_positive']}%  neg={tb2['pct_negative']}%")
        for m,res in supervised_results.items():
            print(f"  {m.upper():<10}: {res['accuracy']:.4f} accuracy")
        print(f"\n  PNGs  -> {os.path.abspath(SAVE_DIR)}/")
        print(f"  JSON  -> {os.path.abspath('data')}/")
        print(f"\n{'='*60}")
        print("  [OK] Done.  Launch the dashboard:")
        print("       streamlit run app.py")
        print(f"{'='*60}\n")

    # ── CLI entry point ──────────────────────────────────────────
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description="US/Israel-Iran War NLP — combined pipeline + dashboard",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python app.py                         Run full NLP pipeline (generate data)
  python app.py --notebooks             Run Jupyter notebooks then launch app
  python app.py --notebooks --no-app    Run Jupyter notebooks only
  streamlit run app.py                  Launch interactive dashboard
""")
        parser.add_argument("--notebooks", action="store_true",
                            help="Execute Jupyter notebooks instead of generate_data pipeline")
        parser.add_argument("--no-app",    action="store_true",
                            help="(with --notebooks) Do not launch Streamlit after notebooks")
        args = parser.parse_args()

        if args.notebooks:
            os.makedirs("data", exist_ok=True)
            for nb_path, label in NOTEBOOKS:
                if not os.path.exists(nb_path):
                    print(f"\n[WARN] Notebook not found, skipping: {nb_path}"); continue
                if not run_notebook(nb_path, label):
                    print("\nPipeline aborted."); sys.exit(1)
            print("\n" + "="*60 + "\n  Notebooks complete.\n" + "="*60)
            if not args.no_app:
                print("\nLaunching dashboard…")
                subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
        else:
            run_pipeline()
