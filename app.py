# app.py
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from model import UNet
from fpdf import FPDF
import io
import re

# ============================================================
#  BRILLIANT UI - DARK THEME WITH GLASSMORPHISM & GRADIENTS
# ============================================================

plt.rcParams.update({
    'text.color': '#E2E8F0',
    'axes.labelcolor': '#E2E8F0',
    'axes.edgecolor': '#334155',
    'axes.facecolor': '#0F172A',
    'figure.facecolor': '#0F172A',
    'xtick.color': '#94A3B8',
    'ytick.color': '#94A3B8',
    'grid.color': '#1E293B',
    'grid.alpha': 0.3
})

st.set_page_config(
    page_title="Urban Change Detector | AI Satellite Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS: Brilliant Dark Theme with Glassmorphism ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif !important; }
    
    #MainMenu, footer, header, [data-testid="stToolbar"] { 
        visibility: hidden; 
        display: none !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0B0F1E 0%, #0F172A 50%, #1E1B4B 100%);
        background-attachment: fixed;
    }
    
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0F172A; }
    ::-webkit-scrollbar-thumb { background: #3B82F6; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #60A5FA; }
    
    .hero-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem !important;
    }
    .hero-subtitle {
        font-size: 1.1rem !important;
        color: #94A3B8 !important;
        font-weight: 400 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .hero-glow {
        width: 120px;
        height: 4px;
        background: linear-gradient(90deg, transparent, #60A5FA, #A78BFA, transparent);
        margin: 1rem auto;
        border-radius: 2px;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: 
            0 4px 6px -1px rgba(0, 0, 0, 0.3),
            0 2px 4px -1px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(96, 165, 250, 0.3) !important;
        box-shadow: 
            0 20px 25px -5px rgba(0, 0, 0, 0.4),
            0 10px 10px -5px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transform: translateY(-2px);
    }
    
    .section-header {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #E2E8F0 !important;
        margin-bottom: 1.5rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(96, 165, 250, 0.5), transparent);
        margin-left: 1rem;
    }
    
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border: none !important;
        width: 100% !important;
        letter-spacing: 0.02em;
        box-shadow: 
            0 4px 15px rgba(59, 130, 246, 0.4),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset !important;
        transition: all 0.3s ease !important;
        position: relative;
        overflow: hidden;
    }
    div.stButton > button:first-child::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    div.stButton > button:first-child:hover::before {
        left: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 8px 25px rgba(59, 130, 246, 0.6),
            0 0 0 1px rgba(255, 255, 255, 0.15) inset !important;
    }
    div.stButton > button:first-child:active {
        transform: translateY(0);
    }
    
    div.stButton > button:not(:first-child) {
        background: rgba(30, 41, 59, 0.6) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(96, 165, 250, 0.3) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:not(:first-child):hover {
        background: rgba(59, 130, 246, 0.2) !important;
        border-color: rgba(96, 165, 250, 0.6) !important;
    }
    
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #E2E8F0 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #94A3B8 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #E2E8F0 !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2)) !important;
        color: #60A5FA !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
    }
    
    div[data-testid="stNumberInput"] input {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(96, 165, 250, 0.2) !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    }
    
    div[data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
    }
    div[data-testid="stFileUploader"] > section {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 2px dashed rgba(96, 165, 250, 0.3) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"] > section:hover {
        border-color: rgba(96, 165, 250, 0.8) !important;
        background: rgba(15, 23, 42, 0.8) !important;
    }
    div[data-testid="stFileUploader"] > section > div > span {
        color: #60A5FA !important;
        font-weight: 600 !important;
    }
    div[data-testid="stFileUploader"] > section > div > small {
        color: #64748B !important;
    }
    
    img {
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .stSpinner > div {
        border-top-color: #3B82F6 !important;
        border-right-color: #8B5CF6 !important;
    }
    
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(96, 165, 250, 0.3), transparent) !important;
        margin: 2rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIGURATION & MODEL ---
DEVICE = torch.device('cpu') 
IMAGE_SIZE = 256
PIXEL_RESOLUTION = 10 

@st.cache_resource
def load_model():
    model = UNet(in_ch=3, out_ch=2)
    model.load_state_dict(torch.load('best.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --- 3. HELPERS ---
def get_year_from_file(file):
    if file is None: return None
    try:
        img = Image.open(file)
        exif = img._getexif()
        if exif and 36867 in exif:
            date_str = exif[36867]
            year = int(date_str.split(':')[0])
            if 1900 < year < 2100: return year
    except: pass
    match = re.search(r'(19|20)\d{2}', file.name)
    if match: return int(match.group(0))
    return None

def preprocess(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = np.array(img).astype(np.float32)
    if img.max() > 1: img /= 255.0
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return np.transpose(img_resized, (2, 0, 1)), img_resized

def predict(img_chw):
    t = torch.FloatTensor(img_chw).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)
        prob = torch.softmax(out, 1)[0, 1].numpy()
    return (prob > 0.5).astype(np.uint8)

def calc_stats(m1, m2, y1, y2):
    px_km2 = (PIXEL_RESOLUTION**2) / 1e6
    u1, u2 = m1.sum() * px_km2, m2.sum() * px_km2
    cm = np.zeros_like(m1, dtype=np.uint8)
    cm[(m1==0)&(m2==1)] = 1
    cm[(m1==1)&(m2==0)] = 2
    cm[(m1==1)&(m2==1)] = 3
    new = (cm==1).sum() * px_km2
    loss = (cm==2).sum() * px_km2
    net = u2 - u1
    yrs = max(y2 - y1, 1)
    gr = (net/u1*100) if u1 > 0 else 0
    return cm, u1, u2, new, loss, net, gr, yrs

# --- 4. COMPREHENSIVE PDF GENERATOR ---
def generate_comprehensive_pdf(stats, change_map_img, orig1_pil, orig2_pil, mask1_pil, mask2_pil):
    pdf = FPDF()
    
    trend = "expansion" if stats['net'] > 0 else "contraction" if stats['net'] < 0 else "stability"
    if stats['new'] > stats['loss']:
        dev_note = "New urban developments outweighed urban loss, indicating active infrastructure growth."
    else:
        dev_note = "Urban loss exceeded new developments, suggesting potential redevelopment or environmental reclamation."
    
    conclusion = (f"Between {stats['y1']} and {stats['y2']}, the analyzed region experienced significant urban {trend}. "
                  f"The total urban area changed by {stats['net']:+.2f} km2, representing a {stats['gr']:+.1f}% variation over the period. "
                  f"Specifically, {stats['new']:.2f} km2 of new urban areas were detected, while {stats['loss']:.2f} km2 of previously urbanized land was lost. "
                  f"{dev_note} This data is critical for future urban planning and resource allocation.")

    orig1_pil.save("temp_orig1.png")
    orig2_pil.save("temp_orig2.png")
    mask1_pil.save("temp_mask1.png")
    mask2_pil.save("temp_mask2.png")
    change_map_img.save("temp_map.png")

    # PAGE 1: EXECUTIVE SUMMARY
    pdf.add_page()
    pdf.set_fill_color(59, 130, 246)
    pdf.rect(0, 0, 220, 40, style='F')
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 28, "Urban Change Detection Report", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 10, f"Analysis Period: {stats['y1']} to {stats['y2']}", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(226, 232, 240)
    pdf.cell(0, 10, "1. Key Metrics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 8, f"  - Urban Area ({stats['y1']}): {stats['u1']:.2f} km2", ln=True)
    pdf.cell(0, 8, f"  - Urban Area ({stats['y2']}): {stats['u2']:.2f} km2", ln=True)
    pdf.cell(0, 8, f"  - Net Change: {stats['net']:+.2f} km2 ({stats['gr']:+.1f}%)", ln=True)
    pdf.cell(0, 8, f"  - New Urban Expansion: +{stats['new']:.2f} km2", ln=True)
    pdf.cell(0, 8, f"  - Urban Loss: -{stats['loss']:.2f} km2", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(226, 232, 240)
    pdf.cell(0, 10, "2. Executive Conclusion", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(0, 6, conclusion)

    # PAGE 2: VISUAL ANALYSIS
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 15, "Visual Analysis: Input & AI Detection", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(95, 8, f"Original Satellite ({stats['y1']})", ln=False, align="C")
    pdf.cell(95, 8, f"Original Satellite ({stats['y2']})", ln=True, align="C")
    pdf.image("temp_orig1.png", x=10, w=90)
    pdf.image("temp_orig2.png", x=110, w=90)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(95, 8, f"AI Urban Mask ({stats['y1']})", ln=False, align="C")
    pdf.cell(95, 8, f"AI Urban Mask ({stats['y2']})", ln=True, align="C")
    pdf.image("temp_mask1.png", x=10, w=90)
    pdf.image("temp_mask2.png", x=110, w=90)

    # PAGE 3: CHANGE MAPPING
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 15, "Change Detection Mapping", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(0, 6, "The map illustrates spatial distribution of urban changes. Red = new expansion, Orange = urban loss, Dark red = stable urban, Light green = stable non-urban.")
    pdf.ln(5)
    pdf.image("temp_map.png", x=10, w=190)
    
    pdf_output = pdf.output()
    return bytes(pdf_output) if not isinstance(pdf_output, bytes) else pdf_output

# ============================================================
#  UI LAYOUT - BRILLIANT REDESIGN
# ============================================================

# Hero Section
st.markdown('''
<div class="hero-container">
    <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">🛰️</div>
    <h1 class="hero-title">Urban Change Detector</h1>
    <div class="hero-glow"></div>
    <p class="hero-subtitle">AI-Powered Satellite Imagery Analysis</p>
</div>
''', unsafe_allow_html=True)

# Upload Section with Glassmorphism
st.markdown('<div class="section-header">📡 Upload Satellite Images</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">'
                '<span style="font-size: 1.5rem;">📸</span>'
                '<span style="font-size: 1.2rem; font-weight: 700; color: #E2E8F0;">Period 1 (T1)</span>'
                '</div>', unsafe_allow_html=True)
    
    img1_file = st.file_uploader("Drop T1 satellite image here", type=["png", "jpg", "jpeg"], key="img1")
    year1_default = get_year_from_file(img1_file) if img1_file else 2015
    year1 = st.number_input("Year T1", value=year1_default, step=1, key="year1_input")
    
    if img1_file:
        st.markdown('<div style="margin-top: 1rem; border-radius: 12px; overflow: hidden;">', unsafe_allow_html=True)
        st.image(img1_file, caption=f"T1 Preview - {year1}", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="text-align: center; padding: 2rem; color: #475569;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📁</div>
            <p style="font-size: 0.9rem;">Drag & drop or click to upload</p>
            <p style="font-size: 0.75rem; color: #334155;">Supports PNG, JPG, JPEG</p>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">'
                '<span style="font-size: 1.5rem;">🛸</span>'
                '<span style="font-size: 1.2rem; font-weight: 700; color: #E2E8F0;">Period 2 (T2)</span>'
                '</div>', unsafe_allow_html=True)
    
    img2_file = st.file_uploader("Drop T2 satellite image here", type=["png", "jpg", "jpeg"], key="img2")
    year2_default = get_year_from_file(img2_file) if img2_file else 2023
    year2 = st.number_input("Year T2", value=year2_default, step=1, key="year2_input")
    
    if img2_file:
        st.markdown('<div style="margin-top: 1rem; border-radius: 12px; overflow: hidden;">', unsafe_allow_html=True)
        st.image(img2_file, caption=f"T2 Preview - {year2}", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="text-align: center; padding: 2rem; color: #475569;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📁</div>
            <p style="font-size: 0.9rem;">Drag & drop or click to upload</p>
            <p style="font-size: 0.75rem; color: #334155;">Supports PNG, JPG, JPEG</p>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Analysis Button
st.markdown('<div style="max-width: 400px; margin: 2rem auto;">', unsafe_allow_html=True)
analyze_btn = st.button("🚀 Launch Analysis", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if analyze_btn:
    if not img1_file or not img2_file:
        st.error("⚠️ Please upload both satellite images to proceed with the analysis.")
    else:
        with st.spinner('🔬 Processing satellite imagery with AI models...'):
            chw1, disp1 = preprocess(img1_file)
            chw2, disp2 = preprocess(img2_file)
            mask1 = predict(chw1)
            mask2 = predict(chw2)
            cm, u1, u2, new, loss, net, gr, yrs = calc_stats(mask1, mask2, year1, year2)
            
            stats = {'y1': year1, 'y2': year2, 'u1': u1, 'u2': u2, 'new': new, 'loss': loss, 'net': net, 'gr': gr}
            
            mask1_pil = Image.fromarray((mask1 * 255).astype(np.uint8))
            mask2_pil = Image.fromarray((mask2 * 255).astype(np.uint8))
            orig1_pil = Image.fromarray((disp1 * 255).astype(np.uint8))
            orig2_pil = Image.fromarray((disp2 * 255).astype(np.uint8))
            
            # Results Section
            st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"Urban Area {year1}", f"{u1:.2f} km2", "Baseline")
            with col2:
                st.metric(f"Urban Area {year2}", f"{u2:.2f} km2", f"{net:+.2f} km2")
            with col3:
                st.metric("Net Change", f"{net:+.2f} km2", f"{gr:+.1f}%")
            with col4:
                st.metric("Annual Growth", f"{gr/yrs:+.2f} %/yr", f"over {yrs} years")
            
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🗺️ Visual Analysis</div>', unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["📸 Original Imagery", "🤖 AI Detection Masks", "🔄 Change Map"])
            
            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(disp1, caption=f"T1 Satellite - {year1}", use_column_width=True)
                with c2:
                    st.image(disp2, caption=f"T2 Satellite - {year2}", use_column_width=True)
            
            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(mask1*255, caption=f"AI Urban Mask - {year1}", use_column_width=True, clamp=True)
                with c2:
                    st.image(mask2*255, caption=f"AI Urban Mask - {year2}", use_column_width=True, clamp=True)
            
            with tab3:
                fig, ax = plt.subplots(figsize=(12, 7))
                cmap = ListedColormap(['#90EE90', '#FF0000', '#FFA500', '#800000'])
                ax.imshow(cm, cmap=cmap, vmin=0, vmax=3)
                ax.axis('off')
                
                legend_elements = [
                    Patch(facecolor='#90EE90', label='Stable Non-Urban'),
                    Patch(facecolor='#800000', label='Stable Urban'),
                    Patch(facecolor='#FF0000', label=f'New Urban (+{new:.2f} km2)'),
                    Patch(facecolor='#FFA500', label=f'Urban Loss (-{loss:.2f} km2)')
                ]
                leg = ax.legend(
                    handles=legend_elements, 
                    loc='lower center', 
                    bbox_to_anchor=(0.5, -0.08), 
                    ncol=4, 
                    frameon=True,
                    fontsize=10,
                    fancybox=True,
                    shadow=True
                )
                for text in leg.get_texts(): 
                    text.set_color('#E2E8F0')
                    text.set_fontweight('600')
                leg.get_frame().set_facecolor('#0F172A')
                leg.get_frame().set_edgecolor('#334155')
                leg.get_frame().set_alpha(0.95)
                
                st.pyplot(fig)
                
                st.markdown('<hr>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📄 Export Report</div>', unsafe_allow_html=True)
                
                fig.canvas.draw()
                width, height = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(height, width, 4)
                map_img = Image.fromarray(buf[:, :, :3], 'RGB')
                
                pdf_bytes = generate_comprehensive_pdf(stats, map_img, orig1_pil, orig2_pil, mask1_pil, mask2_pil)
                
                st.download_button(
                    label="⬇️ Download Full PDF Report",
                    data=pdf_bytes,
                    file_name=f"Urban_Change_Report_{year1}_{year2}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
else:
    st.markdown('''
    <div style="text-align: center; padding: 3rem; color: #475569;">
        <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;">🛰️</div>
        <p style="font-size: 1.1rem; color: #64748B;">Upload two satellite images to begin AI-powered urban change detection</p>
        <p style="font-size: 0.85rem; color: #334155; margin-top: 0.5rem;">Supports temporal analysis across multiple years</p>
    </div>
    ''', unsafe_allow_html=True)
