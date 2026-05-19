
"""
NAUTIQ | Yat Üretiminde Akıllı Tedarik Zekası
Production Dashboard — MIUUL Capstone Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import json
import os
from pathlib import Path

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="NAUTIQ | Akıllı Tedarik Zekası",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS — Professional styling
# ========================================
st.markdown("""
<style>
    /* Main color scheme */
    :root {
        --primary: #1B3A6B;
        --secondary: #2E74B5;
        --accent: #10b981;
        --warning: #f59e0b;
        --danger: #dc2626;
        --neutral: #94a3b8;
    }
    
    /* Custom header */
    .nautiq-header {
        background: linear-gradient(135deg, #1B3A6B 0%, #2E74B5 100%);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        color: white;
    }
    .nautiq-header h1 {
        color: white !important;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    .nautiq-header p {
        color: #d1e2f5 !important;
        font-size: 1.1rem;
        margin-top: 10px;
        margin-bottom: 0;
    }
    
    /* Section dividers */
    .section-divider {
        border-top: 2px solid #e5e7eb;
        margin: 30px 0;
    }
    
    /* KPI cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        text-align: left;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        color: #1F2937 !important;
    }
    .kpi-card * {
        color: inherit;
    }
    .kpi-label {
        color: #6b7280;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        color: #1B3A6B;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 8px 0 4px 0;
    }
    .kpi-delta {
        color: #6b7280;
        font-size: 0.85rem;
    }
    
    /* Finding cards */
    .finding-card {
        background: #FFFFFF !important;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid;
        margin: 10px 0;
        color: #1F2937 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    .finding-card * {
        color: #1F2937;
    }
    .finding-card strong {
        color: #111827 !important;
    }
    .finding-critical { border-left-color: #dc2626; }
    .finding-warning  { border-left-color: #f59e0b; }
    .finding-info     { border-left-color: #2E74B5; }
    .finding-success  { border-left-color: #10b981; }
    
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ========================================
# PATHS
# ========================================
BASE = Path(__file__).parent
DB_PATH = BASE / "data" / "nautiq_miuul.db"
MODELS_DIR = BASE / "models"
GENAI_PATH = BASE / "data" / "genai_outputs.json"
IMG_DIR = BASE / "images"

# ========================================
# CACHED LOADERS
# ========================================
@st.cache_data
def load_sf():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM supplier_features", conn)
    conn.close()
    return df

@st.cache_data
def load_yb():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM yacht_benchmark", conn)
    conn.close()
    return df

@st.cache_data
def load_ss():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM stage_supplier_value", conn)
    conn.close()
    return df

@st.cache_data
def load_cp():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM control_panel LIMIT 5000", conn)
    conn.close()
    return df

@st.cache_data
def load_genai():
    with open(GENAI_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_models():
    with open(MODELS_DIR / "classifier_model.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(MODELS_DIR / "classifier_meta.pkl", "rb") as f:
        clf_meta = pickle.load(f)
    with open(MODELS_DIR / "similarity_matrix.pkl", "rb") as f:
        sim_data = pickle.load(f)
    return clf, clf_meta, sim_data

# Load
sf = load_sf()
yb = load_yb()
ss = load_ss()
cp = load_cp()
genai = load_genai()
clf, clf_meta, sim_data = load_models()

# Build similarity DataFrame
sim_df = pd.DataFrame(
    sim_data["sim_matrix"],
    index=sim_data["supplier_index"],
    columns=sim_data["supplier_index"]
)
np.fill_diagonal(sim_df.values, np.nan)

# ========================================
# HELPER: KPI card
# ========================================
def kpi_card(label, value, delta=None, color="#1B3A6B"):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        {delta_html}
    </div>
    """

# ========================================
# SIDEBAR NAVIGATION
# ========================================
st.sidebar.markdown("""
<div style='text-align: center; padding: 15px 0;'>
    <h1 style='color: #1B3A6B; margin: 0; font-size: 2rem;'>⚓ NAUTIQ</h1>
    <p style='color: #6b7280; margin: 0; font-size: 0.85rem;'>Akıllı Tedarik Zekası</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📂 Bölümler",
    [
        "🏠 Anasayfa",
        "📊 EDA & Trendler",
        "🎯 Segmentasyon",
        "🤖 ML Modelleri",
        "🤝 Öneri Motoru",
        "🚢 Yat Sektörü",
        "🧠 GenAI Asistan",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**📋 Proje Bilgileri**

- 🗄️ **355** Tedarikçi
- 🚢 **21** Yat Projesi
- 💰 **732M TRY** İş Hacmi
- 🤖 **3** ML Modeli
- 📊 **10** Üretim Aşaması

---
**🎓 MIUUL Data Science Bootcamp**  
*Capstone Project | 2026*
""")


# ========================================
# PAGE 1: HOMEPAGE
# ========================================
if page == "🏠 Anasayfa":
    # Hero header
    st.markdown("""
    <div class="nautiq-header">
        <h1>⚓ NAUTIQ | Yat Üretiminde Akıllı Tedarik Zekası</h1>
        <p>End-to-end ML-powered supply chain decision support system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Executive summary
    st.markdown("### 📌 Executive Summary")
    st.markdown("""
    **NAUTIQ**, yat üretiminde tedarik zinciri karar destek sistemi olarak geliştirildi. 
    355 tedarikçinin 732 milyon TRY'lik iş hacmini analiz ederek **kritik riskleri**, 
    **fırsatları** ve **somut aksiyonları** otomatik tespit ediyor.
    """)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # 6 KPI cards
    st.markdown("### 📊 Anahtar Performans Göstergeleri")
    
    total_value_M = sf["total_due_try"].sum() / 1e6
    n_suppliers = len(sf)
    n_projects = len(yb)
    n_at_risk = (sf["segment"] == "⚠️ At Risk").sum()
    at_risk_value = sf[sf["segment"] == "⚠️ At Risk"]["total_due_try"].sum() / 1e6
    n_champions = (sf["segment"] == "🏆 Champions").sum()
    eur_pct = (sf["total_due_try"] * sf["pct_EUR"].fillna(0) / 100).sum() / sf["total_due_try"].sum() * 100
    
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card("Toplam İş Hacmi", f"{total_value_M:,.0f}M TRY", "355 tedarikçi"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Aktif Yat Projesi", f"{n_projects}", "URSA, CYGNUS, AQUILA..."), unsafe_allow_html=True)
    c3.markdown(kpi_card("Champion Tedarikçi", f"{n_champions}", "Stratejik elit", color="#10b981"), unsafe_allow_html=True)
    
    c4, c5, c6 = st.columns(3)
    c4.markdown(kpi_card("At Risk Tedarikçi", f"{n_at_risk}", f"{at_risk_value:.0f}M TRY taşıyor", color="#f59e0b"), unsafe_allow_html=True)
    c5.markdown(kpi_card("EUR Bağımlılığı", f"%{eur_pct:.1f}", "Kur kırılganlığı", color="#dc2626"), unsafe_allow_html=True)
    c6.markdown(kpi_card("ML Model Sayısı", "3", "Classifier + KMeans + Similarity", color="#2E74B5"), unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Problem & Solution columns
    col_p, col_s = st.columns(2)
    
    with col_p:
        st.markdown("### 🎯 Sektör Problemleri")
        st.markdown("""
        <div class="finding-card finding-critical">
        <strong>🔴 Konsantrasyon Riski</strong><br>
        Az sayıda tedarikçide yüksek iş hacmi → tek noktadan başarısızlık riski
        </div>
        
        <div class="finding-card finding-warning">
        <strong>🟡 Kur Kırılganlığı</strong><br>
        Alımların büyük çoğunluğu yabancı para cinsinden → TRY dalgalanması = milyonlarca TRY etki
        </div>
        
        <div class="finding-card finding-info">
        <strong>🔵 Manuel Analiz</strong><br>
        Yöneticiler sayıları görür ama içgörü çıkaramaz — karar gecikir
        </div>
        """, unsafe_allow_html=True)
    
    with col_s:
        st.markdown("### ✨ NAUTIQ Çözümü")
        st.markdown("""
        <div class="finding-card finding-success">
        <strong>🟢 5-Segment Stratejisi</strong><br>
        Champions, Loyal, At Risk, Lost, Occasional — her segment için somut aksiyon
        </div>
        
        <div class="finding-card finding-success">
        <strong>🟢 3 ML Modeli</strong><br>
        Classifier %85 accuracy + KMeans clustering + Cosine similarity öneri
        </div>
        
        <div class="finding-card finding-success">
        <strong>🟢 GenAI Asistan</strong><br>
        30 dakikalık manuel rapor → 3 saniyede otomatik üretim (Claude API)
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Top findings
    st.markdown("### 💎 Üç Kritik Bulgu")
    
    f1, f2, f3 = st.columns(3)
    
    with f1:
        st.markdown("""
        <div class="finding-card finding-critical">
        <strong style="font-size:1.1rem;">256M TRY Paradoksu</strong>
        <p style="font-size: 1.8rem; color: #dc2626; margin: 8px 0;"><strong>18 tedarikçi → 35% iş</strong></p>
        <p style="margin:0; color:#6b7280;">At Risk segmenti, toplam iş hacminin üçte birini taşıyor. Bu küçük grup motor sistemleri ve uzmanlık gerektiren bileşenler tedarik ediyor.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with f2:
        st.markdown("""
        <div class="finding-card finding-warning">
        <strong style="font-size:1.1rem;">Döviz Bağımlılığı</strong>
        <p style="font-size: 1.8rem; color: #f59e0b; margin: 8px 0;"><strong>%94.7 yabancı para</strong></p>
        <p style="margin:0; color:#6b7280;">EUR %74, USD %20, TRY sadece %5. TRY %10 değer kaybetse, +69M TRY ek maliyet çıkıyor.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with f3:
        st.markdown("""
        <div class="finding-card finding-info">
        <strong style="font-size:1.1rem;">Gordion Çoklu Hakimiyet</strong>
        <p style="font-size: 1.8rem; color: #2E74B5; margin: 8px 0;"><strong>3 aşamada #1</strong></p>
        <p style="margin:0; color:#6b7280;">Tek tedarikçi yat üretiminin Sistemler + Elektrik + Dış Donanım aşamalarında baskın. Toplam 84M TRY konsantrasyon.</p>
        </div>
        """, unsafe_allow_html=True)

# ========================================
# PAGE 2: EDA & TRENDS
# ========================================
elif page == "📊 EDA & Trendler":
    st.markdown("""
    <div class="nautiq-header">
        <h1>📊 EDA | Veri Keşfi & Trendler</h1>
        <p>11,954 satınalma talebi | 75,879 transaction | 21 yat projesi</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data overview
    st.markdown("### 🗂️ Veri Genel Bakış")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Toplam Talep</div><div class="kpi-value">11,954</div><div class=\"kpi-delta\">11.5K aktif</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Transaction</div><div class="kpi-value">75,879</div><div class=\"kpi-delta\">75K hareket</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Tedarikçi</div><div class="kpi-value">{len(sf)}</div><div class=\"kpi-delta\">355 firma</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="kpi-card"><div class="kpi-label">Item Çeşidi</div><div class="kpi-value">9,322</div><div class=\"kpi-delta\">~9K unique item</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Pareto analysis
    st.markdown("### 📈 Pareto Analizi (80/20 Kuralı)")
    
    sf_sorted = sf.sort_values("total_due_try", ascending=False).reset_index(drop=True)
    sf_sorted["cumulative_pct"] = (sf_sorted["total_due_try"].cumsum() / sf_sorted["total_due_try"].sum() * 100)
    
    # Find supplier index at 80%
    n_for_80 = (sf_sorted["cumulative_pct"] <= 80).sum() + 1
    pct_of_suppliers = n_for_80 / len(sf_sorted) * 100
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        # Bar chart for cumulative %
        fig.add_trace(go.Bar(
            x=sf_sorted.index[:100],
            y=sf_sorted["total_due_try"][:100] / 1e6,
            name="Tedarikçi Değeri (M TRY)",
            marker_color="#2E74B5",
            yaxis="y1",
        ))
        # Line for cumulative
        fig.add_trace(go.Scatter(
            x=sf_sorted.index[:100],
            y=sf_sorted["cumulative_pct"][:100],
            name="Kümülatif %",
            yaxis="y2",
            line=dict(color="#dc2626", width=3),
        ))
        # 80% reference line
        fig.add_hline(y=80, line_dash="dash", line_color="gray", yref="y2",
                     annotation_text="80% Eşiği", annotation_position="right")
        
        fig.update_layout(
            xaxis=dict(title="Tedarikçi Sırası"),
            yaxis=dict(title="Değer (M TRY)", side="left"),
            yaxis2=dict(title="Kümülatif %", overlaying="y", side="right", range=[0, 105]),
            hovermode="x unified",
            height=400,
            showlegend=True,
            legend=dict(x=0.5, y=1.1, orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown("#### 🎯 Pareto Bulgusu")
        st.markdown(f"""
        <div class="finding-card finding-critical">
        <strong style="font-size:1.2rem;">{n_for_80} tedarikçi = %80 iş</strong>
        <p style="margin:8px 0; color:#6b7280;">
        Toplam {len(sf)} tedarikçinin sadece <strong>{pct_of_suppliers:.1f}%</strong>'i 
        iş hacminin <strong>%80'ini</strong> taşıyor.
        </p>
        <p style="margin:0; color:#1B3A6B;">
        <strong>Stratejik anlam:</strong> Bu küçük grupla derin ilişki yönetimi, 
        toplam riskinin yönetilmesinde anahtar.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Top 20 suppliers
    st.markdown("### 🏆 En Büyük 20 Tedarikçi")
    
    top20 = sf.nlargest(20, "total_due_try")[
        ["Supplier", "segment", "total_due_try", "supplier_hybrid_score", 
         "avg_delay", "on_time_rate", "country"]
    ].copy()
    top20["total_M_TRY"] = (top20["total_due_try"] / 1e6).round(1)
    top20["Skor"] = top20["supplier_hybrid_score"].round(1)
    top20["Gecikme"] = top20["avg_delay"].round(1)
    top20["On-time %"] = top20["on_time_rate"].round(1)
    
    display = top20[["Supplier", "segment", "total_M_TRY", "Skor", 
                      "Gecikme", "On-time %", "country"]].rename(columns={
        "Supplier": "Tedarikçi", "segment": "Segment",
        "total_M_TRY": "Değer (M TRY)", "country": "Ülke"
    })
    
    st.dataframe(display, use_container_width=True, hide_index=True, height=500)

# ========================================
# PAGE 3: SEGMENTATION
# ========================================
elif page == "🎯 Segmentasyon":
    st.markdown("""
    <div class="nautiq-header">
        <h1>🎯 Tedarikçi Segmentasyonu</h1>
        <p>355 tedarikçi | 5 stratejik grup | Hibrit yaklaşım (kural + K-Means)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 5 segment cards
    st.markdown("### 📊 Segment Dağılımı")
    
    seg_data = [
        ("🏆 Champions", 48, 172, "#10b981", "Stratejik elit | Long-term contracts"),
        ("💚 Loyal", 104, 190, "#3b82f6", "Güvenilir orta | Develop more business"),
        ("⚠️ At Risk", 18, 256, "#f59e0b", "Yüksek değer + sorun | Rescue plan"),
        ("🔴 Lost", 14, 8, "#dc2626", "Performans krizi | Replace strategy"),
        ("🌫️ Occasional", 171, 107, "#94a3b8", "Tek seferlik | Passive management"),
    ]
    
    cols = st.columns(5)
    for col, (seg, n, val_M, color, action) in zip(cols, seg_data):
        with col:
            st.markdown(f"""
            <div style='background:{color}; padding:18px; border-radius:10px; color:white; height:180px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='font-size:0.85rem; opacity:0.9;'>{seg}</div>
                <div style='font-size:2.5rem; font-weight:700; margin:8px 0;'>{n}</div>
                <div style='font-size:1.1rem; font-weight:500;'>{val_M}M TRY</div>
                <div style='font-size:0.75rem; margin-top:8px; opacity:0.85;'>{action}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # The paradox spotlight
    st.markdown("### 🚨 256M TRY Paradoksu")
    
    st.markdown("""
    <div class="finding-card finding-critical" style="padding: 25px;">
    <h3 style="margin-top:0; color: #dc2626;">⚠️ Konsantrasyon Riski Alarmı</h3>
    <p style="font-size: 1.05rem;">
    <strong>At Risk segmenti sadece 18 tedarikçi (%5.1) ama 256M TRY taşıyor — toplam işin %35'i!</strong>
    </p>
    <p style="margin-bottom: 0; color: #6b7280;">
    Bu paradoks değil — yat üretim sektörünün doğal yapısı. Motor sistemleri, navigasyon ekipmanları gibi 
    kritik bileşenler sınırlı sayıda uzmanlaşmış tedarikçi üzerinden geliyor. Bu yapı onlara fiyatlandırma 
    kaldıracı veriyor.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Interactive segment explorer
    st.markdown("### 🔍 Segment Detayları (interaktif filtre)")
    
    selected_seg = st.selectbox(
        "Segment seç:",
        ["🏆 Champions", "💚 Loyal", "⚠️ At Risk", "🔴 Lost", "🌫️ Occasional"]
    )
    
    seg_df = sf[sf["segment"] == selected_seg].copy()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tedarikçi", len(seg_df))
    c2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Toplam Değer</div><div class="kpi-value">{seg_df['total_due_try'].sum()/1e6:.1f}M TRY</div></div>""", unsafe_allow_html=True)
    c3.metric("Avg Skor", f"{seg_df['supplier_hybrid_score'].mean():.1f}" if seg_df['supplier_hybrid_score'].notna().any() else "N/A")
    c4.metric("Avg Gecikme", f"{seg_df['avg_delay'].mean():.1f}g" if seg_df['avg_delay'].notna().any() else "N/A")
    
    # Show top suppliers in selected segment
    st.markdown(f"#### {selected_seg} — Top 10 (değere göre)")
    top_seg = seg_df.nlargest(10, "total_due_try")
    display = top_seg[["Supplier", "supplier_hybrid_score", "avg_delay", 
                       "on_time_rate", "payment_completion_rate", 
                       "total_due_try", "country"]].round(1).copy()
    display["total_M_TRY"] = (display["total_due_try"] / 1e6).round(1)
    display = display.drop(columns=["total_due_try"])
    display.columns = ["Tedarikçi", "Skor", "Gecikme (g)", "On-time %", "Ödeme %", "Ülke", "Değer (M TRY)"]
    st.dataframe(display, use_container_width=True, hide_index=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Show segmentation image
    seg_img = IMG_DIR / "nb04_01_segmentation_overview.png"
    if seg_img.exists():
        st.markdown("### 📊 Segmentasyon Görseli")
        st.image(str(seg_img), caption="4-panel görselleştirme: Dağılım | Stratejik Harita | Aktivite | İş Değeri",
                 use_container_width=True)

# ========================================
# PAGE 4: ML MODELS
# ========================================
elif page == "🤖 ML Modelleri":
    st.markdown("""
    <div class="nautiq-header">
        <h1>🤖 Machine Learning Modelleri</h1>
        <p>3 model | 2 production-ready | 1 bilinçli başarısız (öğretici)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model overview cards
    st.markdown("### 🎯 Model Karnesi")
    
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown("""
        <div class="finding-card finding-success">
        <h4 style="margin-top:0; color:#10b981;">✅ Classifier</h4>
        <p style="font-size: 2.2rem; font-weight: 700; color: #1B3A6B; margin: 5px 0;">%84.8</p>
        <p style="margin:0; color:#6b7280;">Test Accuracy | F1: 0.78 | CV: %79.8</p>
        <hr style="margin:12px 0;">
        <p style="margin:0;"><strong>At Risk Precision: %100</strong></p>
        <p style="margin:0; color:#6b7280; font-size:0.9rem;">False alarm sıfır — direkt aksiyon!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        st.markdown("""
        <div class="finding-card finding-success">
        <h4 style="margin-top:0; color:#10b981;">✅ KMeans</h4>
        <p style="font-size: 2.2rem; font-weight: 700; color: #1B3A6B; margin: 5px 0;">k=4</p>
        <p style="margin:0; color:#6b7280;">Silhouette: 0.314 | Pipeline: 2.4 KB</p>
        <hr style="margin:12px 0;">
        <p style="margin:0;"><strong>Production-ready</strong></p>
        <p style="margin:0; color:#6b7280; font-size:0.9rem;">Bonus keşif: Uyumakta Loyal (n=29)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        st.markdown("""
        <div class="finding-card finding-warning">
        <h4 style="margin-top:0; color:#f59e0b;">⚠️ Regressor</h4>
        <p style="font-size: 2.2rem; font-weight: 700; color: #1B3A6B; margin: 5px 0;">R² -1.38</p>
        <p style="margin:0; color:#6b7280;">MAE: 3.3 gün | CV mean: -3.08</p>
        <hr style="margin:12px 0;">
        <p style="margin:0;"><strong>Bilinçli başarısız</strong></p>
        <p style="margin:0; color:#6b7280; font-size:0.9rem;">Bimodal target — sınırı kabul ettik</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Classifier deep dive
    st.markdown("### 🌳 RandomForest Classifier — Detaylı Analiz")
    
    classifier_img = IMG_DIR / "nb05_01_classifier_results.png"
    if classifier_img.exists():
        st.image(str(classifier_img), caption="Confusion Matrix + Feature Importance",
                 use_container_width=True)
    
    # Feature importance table
    st.markdown("#### 📊 En Önemli 5 Özellik")
    
    importance_data = [
        ("total_volume", 0.166, "Tedarikçinin toplam alım sayısı"),
        ("total_paid_try", 0.148, "Ödenen tutar (finansal ilişki gücü)"),
        ("payment_completion_rate", 0.140, "Ödeme tamamlama oranı"),
        ("n_delivered", 0.109, "Teslim edilen sayı"),
        ("on_time_rate", 0.107, "Zamanında teslim oranı"),
    ]
    
    imp_df = pd.DataFrame(importance_data, columns=["Feature", "Önem", "Açıklama"])
    imp_df["Önem"] = imp_df["Önem"].apply(lambda x: f"{x:.3f}")
    st.dataframe(imp_df, use_container_width=True, hide_index=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Regressor honest reporting
    st.markdown("### 📈 Regressor Hikayesi — Dürüst Veri Bilimi")
    
    st.warning("""
    **🎓 Bilinçli Başarısızlık:** İlk versiyonda MAE 2.9 gün çıktı (görünüş iyi) ama R² negatif (-1.87). 
    Feature importance'a baktığımda anladım: `delay_consistency` tek başına %92 ağırlık almıştı. 
    Bu feature `avg_delay`'in std sapması — **dolaylı data leakage**. Düzelttim, gerçek tabloyu gördüm: 
    target dağılımı **aşırı çarpık**, Random Forest median'a çekiliyor.
    
    **Sonuç:** Regressor başarısız oldu, classifier'ın "At Risk" tahmini regresyon yerine geçti. 
    **Her başarısızlık öğretici** — sınırı kabul etmek de bilim.
    """)


# ========================================
# PAGE 5: RECOMMENDATION ENGINE
# ========================================
elif page == "🤝 Öneri Motoru":
    st.markdown("""
    <div class="nautiq-header">
        <h1>🤝 Akıllı Öneri Motoru</h1>
        <p>3 farklı senaryo | Cosine similarity + Hybrid scoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([
        "🔄 Alternatif Bul", 
        "🎯 Yeni Alım Önerisi",
        "🛡️ Yedek Plan (Backup)"
    ])
    
    # ---- TAB 1: Alternative Finder ----
    with tab1:
        st.markdown("### Problemli tedarikçiye Champion alternatif")
        st.caption("At Risk veya Lost segmentindeki tedarikçi için cosine similarity ile en uygun yedek bul")
        
        problematic = sf[sf["segment"].isin(["⚠️ At Risk", "🔴 Lost"])]["Supplier"].tolist()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected = st.selectbox(
                "🔍 Problemli tedarikçi seç:",
                problematic,
                index=problematic.index("Phaselis Hardware - (Turkey)") if "Phaselis Hardware - (Turkey)" in problematic else 0
            )
        with col2:
            min_score = st.slider("Min hibrit skor", 60, 95, 75)
        
        if selected and selected in sim_df.index:
            target = sf[sf["Supplier"] == selected].iloc[0]
            
            # Target info
            st.markdown("#### 📋 Mevcut Tedarikçi Durumu")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Segment", target["segment"])
            c2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Hibrit Skor</div><div class="kpi-value">{target['supplier_hybrid_score']:.1f}</div></div>""", unsafe_allow_html=True)
            c3.metric("Avg Gecikme", f"{target['avg_delay']:.0f}g" if pd.notna(target['avg_delay']) else "N/A")
            c4.metric("On-time %", f"{target['on_time_rate']:.0f}" if pd.notna(target['on_time_rate']) else "N/A")
            c5.markdown(f"""<div class="kpi-card"><div class="kpi-label">Değer</div><div class="kpi-value">{target['total_due_try']/1e6:.1f}M TRY</div></div>""", unsafe_allow_html=True)
            
            # Find alternatives
            sims = sim_df[selected].dropna().sort_values(ascending=False)
            candidates = pd.DataFrame({
                "Supplier": sims.index, 
                "similarity": sims.values
            })
            candidates = candidates.merge(
                sf[["Supplier", "segment", "supplier_hybrid_score", 
                    "on_time_rate", "country", "total_due_try"]],
                on="Supplier", how="left"
            )
            candidates = candidates[
                candidates["supplier_hybrid_score"] >= min_score
            ].head(5)
            
            st.markdown(f"#### 🎯 NAUTIQ Önerisi — En İyi 5 Alternatif")
            
            rec_df = candidates[["Supplier", "segment", "similarity", 
                                  "supplier_hybrid_score", "on_time_rate", "country"]].copy()
            rec_df["similarity"] = rec_df["similarity"].round(3)
            rec_df["supplier_hybrid_score"] = rec_df["supplier_hybrid_score"].round(1)
            rec_df["on_time_rate"] = rec_df["on_time_rate"].round(1)
            rec_df.columns = ["Tedarikçi", "Segment", "Benzerlik", "Skor", "On-time %", "Ülke"]
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
            
            # Insight
            top1 = candidates.iloc[0]
            st.success(f"""
            **💡 NAUTIQ Tavsiyesi:** En iyi alternatif **{top1['Supplier']}** 
            (skor: {top1['supplier_hybrid_score']:.1f}, segment: {top1['segment']}, 
            on-time: %{top1['on_time_rate']:.0f}). Benzerlik {top1['similarity']:.2f} ile 
            profil eşleşmesi sağlıyor.
            """)
    
    # ---- TAB 2: New Purchase ----
    with tab2:
        st.markdown("### Yeni alım için en iyi tedarikçi önerisi")
        st.caption("Hybrid scoring: %40 segment + %30 skor + %20 filtre + %10 on-time")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            currency = st.selectbox(
                "💱 Para birimi tercihi:",
                ["EUR", "USD", "TRY", "Fark etmez"],
                index=0
            )
        with col2:
            country_pref = st.selectbox(
                "🌍 Ülke tercihi:",
                ["Fark etmez", "Turkey", "Sadece yabancı"]
            )
        with col3:
            min_score_new = st.slider("📊 Min hibrit skor", 60, 95, 80, key="min_new")
        
        # Filter
        candidates2 = sf[sf["supplier_hybrid_score"] >= min_score_new].copy()
        if country_pref == "Turkey":
            candidates2 = candidates2[candidates2["country"] == "Turkey"]
        elif country_pref == "Sadece yabancı":
            candidates2 = candidates2[candidates2["country"] != "Turkey"]
        
        # Currency filter
        if currency != "Fark etmez":
            pct_col = f"pct_{currency}"
            if pct_col in candidates2.columns:
                candidates2 = candidates2[candidates2[pct_col] > 30]
        
        # Composite scoring
        SEG_SCORES = {"🏆 Champions": 100, "💚 Loyal": 70, "⚠️ At Risk": 30, 
                      "🔴 Lost": 10, "🌫️ Occasional": 40}
        candidates2["seg_score"] = candidates2["segment"].map(SEG_SCORES).fillna(40)
        
        filter_match = candidates2.get(f"pct_{currency}", pd.Series([50] * len(candidates2), index=candidates2.index)) if currency != "Fark etmez" else 50
        
        candidates2["composite"] = (
            0.4 * candidates2["seg_score"] +
            0.3 * candidates2["supplier_hybrid_score"].fillna(50) +
            0.2 * (filter_match if isinstance(filter_match, pd.Series) else pd.Series([filter_match]*len(candidates2), index=candidates2.index)) +
            0.1 * candidates2["on_time_rate"].fillna(50)
        )
        
        if len(candidates2) > 0:
            top5 = candidates2.nlargest(5, "composite")[
                ["Supplier", "segment", "supplier_hybrid_score", 
                 "on_time_rate", "pct_EUR", "pct_USD", "pct_TRY", "country", "composite"]
            ].round(1).copy()
            top5.columns = ["Tedarikçi", "Segment", "Skor", "On-time %", 
                            "EUR %", "USD %", "TRY %", "Ülke", "Hybrid Skor"]
            st.markdown("#### 🏆 NAUTIQ'in Top 5 Önerisi")
            st.dataframe(top5, use_container_width=True, hide_index=True)
            
            # Show why #1 was chosen
            top1 = candidates2.nlargest(1, "composite").iloc[0]
            st.info(f"""
            **🎯 #1 Önerim:** **{top1['Supplier']}** — Hybrid skor: {top1['composite']:.1f}/100
            
            **Niye?**
            - {top1['segment']} segmenti (segment puanı: {SEG_SCORES.get(top1['segment'], 40)})
            - Hibrit skor: {top1['supplier_hybrid_score']:.1f}/100
            - On-time rate: %{top1['on_time_rate']:.1f}
            - {currency} alım oranı: %{top1.get('pct_' + currency, 'N/A') if currency != 'Fark etmez' else '—'}
            """)
        else:
            st.warning("Bu kriterlere uyan tedarikçi bulunamadı. Filtreleri gevşetin.")
    
    # ---- TAB 3: Backup Plan ----
    with tab3:
        st.markdown("### Kritik tedarikçi için Champion peer'leri (yedek plan)")
        st.caption("Tek bir Champion durursa hazırda 5 alternatif")
        
        champions = sf[sf["segment"] == "🏆 Champions"]["Supplier"].tolist()
        
        selected_champ = st.selectbox(
            "🏆 Kritik Champion seç:",
            champions,
            index=champions.index("Pala Industrial - (Turkey)") if "Pala Industrial - (Turkey)" in champions else 0
        )
        
        if selected_champ and selected_champ in sim_df.index:
            sims = sim_df[selected_champ].dropna().sort_values(ascending=False)
            backups = pd.DataFrame({"Supplier": sims.index, "similarity": sims.values})
            backups = backups.merge(
                sf[["Supplier", "segment", "supplier_hybrid_score", "on_time_rate", "country"]],
                on="Supplier", how="left"
            )
            # Only Champions
            backups = backups[backups["segment"] == "🏆 Champions"].head(5)
            
            target_c = sf[sf["Supplier"] == selected_champ].iloc[0]
            
            st.markdown(f"#### 🛡️ {selected_champ} İçin Yedek Plan")
            st.markdown(f"**Mevcut profil:** Skor {target_c['supplier_hybrid_score']:.1f} | Volume {int(target_c['total_volume'])} | {target_c['country']}")
            
            display_b = backups[["Supplier", "similarity", "supplier_hybrid_score", "on_time_rate"]].round(2)
            display_b.columns = ["Yedek Tedarikçi", "Benzerlik", "Skor", "On-time %"]
            st.dataframe(display_b, use_container_width=True, hide_index=True)
            
            top_backup = backups.iloc[0]
            st.success(f"""
            **🥇 En İyi Yedek:** **{top_backup['Supplier']}** 
            (benzerlik {top_backup['similarity']:.3f} — neredeyse aynı profil!)
            
            Eğer {selected_champ} herhangi bir sebeple iş yapmazsa, bu tedarikçi 
            **doğrudan geçişe hazır**.
            """)

# ========================================
# PAGE 6: YACHT SECTOR
# ========================================
elif page == "🚢 Yat Sektörü":
    st.markdown("""
    <div class="nautiq-header">
        <h1>🚢 Yat Sektörü Analizi</h1>
        <p>3 sektörel modül | Kur Riski | Üretim Aşamaları | Yat Benchmark</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([
        "💱 Kur Riski Analizi",
        "🚢 Üretim Aşamaları",
        "⛵ Yat Benchmark"
    ])
    
    # ---- TAB 1: Currency ----
    with tab1:
        st.markdown("### Multi-Currency Exposure & Stres Testi")
        
        eur_total = (sf["total_due_try"] * sf["pct_EUR"].fillna(0) / 100).sum() / 1e6
        usd_total = (sf["total_due_try"] * sf["pct_USD"].fillna(0) / 100).sum() / 1e6
        try_total = (sf["total_due_try"] * sf["pct_TRY"].fillna(0) / 100).sum() / 1e6
        total = eur_total + usd_total + try_total
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi_card("EUR Bağımlılığı", f"{eur_total:.0f}M TRY", f"%{eur_total/total*100:.1f}", color="#3b82f6"), unsafe_allow_html=True)
        c2.markdown(kpi_card("USD Bağımlılığı", f"{usd_total:.0f}M TRY", f"%{usd_total/total*100:.1f}", color="#10b981"), unsafe_allow_html=True)
        c3.markdown(kpi_card("TRY Bağımlılığı", f"{try_total:.0f}M TRY", f"%{try_total/total*100:.1f}", color="#f59e0b"), unsafe_allow_html=True)
        c4.markdown(kpi_card("Yabancı Para Toplam", f"{eur_total+usd_total:.0f}M TRY", f"%{(eur_total+usd_total)/total*100:.1f}", color="#dc2626"), unsafe_allow_html=True)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Stress test interactive
        st.markdown("### 🧪 Stres Testi — TRY Devalüasyon Simülatörü")
        
        deval_pct = st.slider("TRY değer kaybı simülasyonu (%)", 0, 30, 10, 1)
        extra_cost = (eur_total + usd_total) * deval_pct / 100
        
        col_a, col_b = st.columns([2, 1])
        with col_a:
            import plotly.graph_objects as go
            
            devals = list(range(0, 31))
            costs = [(eur_total + usd_total) * d / 100 for d in devals]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=devals, y=costs, mode="lines+markers",
                line=dict(color="#dc2626", width=3),
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(220,38,38,0.1)",
                name="Ek Maliyet"
            ))
            # Highlight current selection
            fig.add_trace(go.Scatter(
                x=[deval_pct], y=[extra_cost], mode="markers",
                marker=dict(size=20, color="#1B3A6B", symbol="star"),
                name=f"Seçilen: %{deval_pct}",
                showlegend=False
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="orange",
                         annotation_text="50M Eşiği")
            fig.add_hline(y=100, line_dash="dash", line_color="red",
                         annotation_text="100M Kritik Eşik")
            
            fig.update_layout(
                xaxis_title="TRY Devalüasyon %",
                yaxis_title="Ek Maliyet (M TRY)",
                height=400,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            st.markdown(f"""
            <div class="finding-card finding-critical" style="padding: 20px;">
            <p style="margin:0; color:#6b7280;">Eğer TRY <strong>%{deval_pct}</strong> değer kaybetse:</p>
            <p style="font-size: 2.5rem; color: #dc2626; font-weight: 700; margin: 10px 0;">
            +{extra_cost:.0f}M TRY
            </p>
            <p style="margin:0; color:#6b7280;">ek maliyet çıkacak</p>
            <hr style="margin:15px 0;">
            <p style="margin:0; font-size: 0.9rem;">
            Bu tutar toplam iş hacminin <strong>%{extra_cost/total*100:.1f}</strong>'i
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Top exposed suppliers
        st.markdown("### 🚨 En Çok Yabancı Para Bağımlısı Tedarikçiler")
        
        sf_ce = sf.copy()
        sf_ce["foreign_M"] = (sf_ce["total_due_try"] * 
                              (sf_ce["pct_EUR"].fillna(0) + sf_ce["pct_USD"].fillna(0)) / 100 / 1e6).round(1)
        top_ce = sf_ce.nlargest(10, "foreign_M")[
            ["Supplier", "segment", "foreign_M", "pct_EUR", "pct_USD", "country"]
        ].round(1)
        top_ce.columns = ["Tedarikçi", "Segment", "Döviz Bağımlı (M TRY)", "EUR %", "USD %", "Ülke"]
        st.dataframe(top_ce, use_container_width=True, hide_index=True)
        
        # Image
        img = IMG_DIR / "nb08_01_currency_exposure.png"
        if img.exists():
            st.image(str(img), caption="4-panel currency exposure analysis", use_container_width=True)
    
    # ---- TAB 2: Production Stages ----
    with tab2:
        st.markdown("### Üretim Aşaması × Tedarikçi Matrix")
        
        st.markdown("""
        <div class="finding-card finding-critical" style="padding: 20px;">
        <h4 style="margin-top:0; color:#dc2626;">🚨 Gordion Components — 3 Aşamada Birden #1</h4>
        <ul style="margin: 10px 0;">
            <li><strong>4. Systems & Mooring:</strong> 19.6M TRY</li>
            <li><strong>5. Electrical System:</strong> 13.2M TRY</li>
            <li><strong>9. Exterior Fitting Out:</strong> 50.6M TRY</li>
        </ul>
        <p style="margin:0;"><strong>Toplam: 84M TRY</strong> — yat üretiminin yarısı tek tedarikçide</p>
        </div>
        """, unsafe_allow_html=True)
        
        img = IMG_DIR / "nb08_02_stage_supplier_matrix.png"
        if img.exists():
            st.image(str(img), caption="Top 12 supplier × 9 production stage dependency matrix",
                     use_container_width=True)
        
        # Stage-specific drill-down
        st.markdown("### 🔍 Aşama Detayları (interaktif)")
        
        if len(ss) > 0:
            stages_list = sorted(ss["stage_clean"].dropna().unique())
            selected_stage = st.selectbox("Aşama seç:", stages_list,
                                          index=stages_list.index("9.Exterior Fitting Out") if "9.Exterior Fitting Out" in stages_list else 0)
            
            stage_top = ss[ss["stage_clean"] == selected_stage].nlargest(10, "total_value")
            stage_top["value_M"] = (stage_top["total_value"] / 1e6).round(2)
            display_s = stage_top[["Supplier", "value_M", "n_requests"]].copy()
            display_s.columns = ["Tedarikçi", "Değer (M TRY)", "Talep Sayısı"]
            
            st.dataframe(display_s, use_container_width=True, hide_index=True)
    
    # ---- TAB 3: Yacht Benchmark ----
    with tab3:
        st.markdown("### 21 Yacht Project Benchmark")
        
        # Highlights
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="finding-card finding-success">
            <h4 style="margin-top:0; color:#10b981;">🥇 URSA-025 — Model Yat</h4>
            <p style="margin:0;">1,586 talep, 150 tedarikçi</p>
            <p style="font-size: 1.8rem; font-weight: 700; color: #10b981; margin: 5px 0;">%78.4 on-time</p>
            <p style="margin:0; color:#6b7280;">Şirketin en başarılı projesi</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="finding-card finding-critical">
            <h4 style="margin-top:0; color:#dc2626;">🚨 ORION-01 — Saklı Felaket</h4>
            <p style="margin:0;">65M TRY (büyük yat)</p>
            <p style="font-size: 1.8rem; font-weight: 700; color: #dc2626; margin: 5px 0;">%12.9 on-time</p>
            <p style="margin:0; color:#6b7280;">Sadece 13 tedarikçi — bağımlılık riski</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="finding-card finding-warning">
            <h4 style="margin-top:0; color:#f59e0b;">⏰ LEPUS-003 — Gecikme Kralı</h4>
            <p style="margin:0;">21M TRY, 356 gün</p>
            <p style="font-size: 1.8rem; font-weight: 700; color: #f59e0b; margin: 5px 0;">+16.2 gün</p>
            <p style="margin:0; color:#6b7280;">En kötü ortalama gecikme</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Yacht table — full
        st.markdown("### 📊 Tüm 21 Yat — Tam Karşılaştırma")
        
        yb_sorted = yb.sort_values("total_value_M", ascending=False)
        show_df = yb_sorted[["Project", "duration_days", "n_requests", "n_suppliers", 
                              "total_value_M", "avg_delay", "ontime_pct"]].round(1)
        show_df.columns = ["Yat", "Süre (gün)", "Talep", "Tedarikçi", 
                            "Değer (M TRY)", "Avg Gecikme", "On-time %"]
        st.dataframe(show_df, use_container_width=True, hide_index=True, height=600)
        
        img = IMG_DIR / "nb08_03_yacht_benchmark.png"
        if img.exists():
            st.image(str(img), caption="4-panel yacht benchmark analysis",
                     use_container_width=True)

# ========================================
# PAGE 7: GENAI ASSISTANT
# ========================================
elif page == "🧠 GenAI Asistan":
    st.markdown("""
    <div class="nautiq-header">
        <h1>🧠 GenAI Asistan</h1>
        <p>Claude API tarafından üretilen yönetici raporları | Hibrit mimari (API + Mock fallback)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Value proposition
    st.markdown("""
    <div class="finding-card finding-info" style="padding: 20px; background: #FFFFFF;">
    <h4 style="margin-top:0; color: #1F2937;">⚡ Değer Önermesi</h4>
    <table style="width: 100%; border-collapse: collapse; color: #1F2937;">
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; color: #1F2937;">📝 Geleneksel manuel rapor</td>
            <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: right;"><strong>30 dakika</strong></td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; color: #1F2937;">🤖 NAUTIQ GenAI üretimi</td>
            <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: right;"><strong style="color:#10b981;">3 saniye</strong></td>
        </tr>
        <tr>
            <td style="padding: 8px; color: #1F2937;">⚡ Hız çarpanı</td>
            <td style="padding: 8px; text-align: right;"><strong style="color:#10b981;">600×</strong></td>
        </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # 3 categories
    st.markdown("### 📚 Hazır Rapor Kategorileri")
    
    category_groups = {
        "📊 Tedarikçi Raporu": {
            "Phaselis Hardware (At Risk)": "report_at_risk_problem",
            "Pala Industrial (Champion - Top)": "report_champion_top",
            "Cappadocia Works (Champion - Local)": "report_champion_local",
        },
        "🚨 Anomaly Açıklaması": {
            "Gordion Konsantrasyon Riski": "anomaly_concentration_gordion",
            "256M TRY At Risk Paradoksu": "anomaly_at_risk_paradox",
            "Uyumakta Olan Loyal Tedarikçiler": "anomaly_sleeping_loyal",
        },
        "🎯 Öneri Mantığı": {
            "Cappadocia Works — Yerli Alım Önerisi": "rec_cappadocia",
            "Sharruma Materials — EUR Önerisi": "rec_sharruma",
        }
    }
    
    selected_group = st.radio(
        "Kategori:",
        list(category_groups.keys()),
        horizontal=True
    )
    
    selected_item = st.selectbox(
        "İçerik seç:",
        list(category_groups[selected_group].keys())
    )
    
    key = category_groups[selected_group][selected_item]
    
    if key in genai:
        item = genai[key]
        source = "🟢 Claude API (cached)" if item["source"] == "api" else "🟡 Mock"
        
        st.markdown(f"<div style='background:#f3f4f6; padding:8px 15px; border-radius:6px; margin: 15px 0; font-size:0.85rem; color:#6b7280;'>"
                    f"Source: {source} | Generated by NAUTIQ GenAI Assistant"
                    f"</div>", unsafe_allow_html=True)
        
        # Container for the report
        st.markdown(f"<div style='background:#FFFFFF; padding:25px; border-radius:10px; border:1px solid #e5e7eb; color:#1F2937;'>{item['content']}</div>",
                    unsafe_allow_html=True)
    else:
        st.error(f"İçerik bulunamadı: {key}")
