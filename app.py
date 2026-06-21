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
#  MAGICAL 3D UI - DARK THEME WITH ANIMATIONS & DEPTH
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
    page_title='Urban Change Detector | AI Satellite Analysis',
    page_icon='🛰',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# --- CSS: Magical 3D Dark Theme with Animations ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    * { font-family: 'Inter', 'Space Grotesk', sans-serif !important; }

    #MainMenu, footer, header, [data-testid='stToolbar'] {
        visibility: hidden;
        display: none !important;
    }

    .stApp {
        background: linear-gradient(-45deg, #0a0e27, #0f172a, #1a103c, #0d1b2a, #1e1b4b);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        background-attachment: fixed;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            radial-gradient(2px 2px at 20px 30px, rgba(96,165,250,0.3), transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(139,92,246,0.2), transparent),
            radial-gradient(1px 1px at 90px 40px, rgba(244,114,182,0.3), transparent),
            radial-gradient(2px 2px at 160px 120px, rgba(96,165,250,0.2), transparent),
            radial-gradient(1px 1px at 230px 80px, rgba(139,92,246,0.3), transparent),
            radial-gradient(2px 2px at 300px 150px, rgba(244,114,182,0.2), transparent),
            radial-gradient(1px 1px at 350px 50px, rgba(96,165,250,0.3), transparent),
            radial-gradient(2px 2px at 450px 180px, rgba(139,92,246,0.2), transparent);
        background-size: 500px 250px;
        animation: particleFloat 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes particleFloat {
        0% { transform: translateY(0); }
        100% { transform: translateY(-250px); }
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0F172A; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #3B82F6, #8B5CF6); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #60A5FA, #A78BFA); }

    .hero-container {
        text-align: center;
        padding: 3rem 0 2rem 0;
        margin-bottom: 2rem;
        position: relative;
        z-index: 1;
    }
    .hero-emoji {
        font-size: 4rem !important;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 10px 20px rgba(59,130,246,0.4));
    }
    @keyframes float {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(5deg); }
    }
    .hero-title {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 30%, #F472B6 60%, #FB923C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 0 60px rgba(96,165,250,0.3);
        animation: titlePulse 4s ease-in-out infinite;
    }
    @keyframes titlePulse {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.2); }
    }
    .hero-subtitle {
        font-size: 1rem !important;
        color: #94A3B8 !important;
        font-weight: 400 !important;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        animation: fadeInUp 1s ease-out 0.3s both;
    }
    .hero-glow {
        width: 200px;
        height: 3px;
        background: linear-gradient(90deg, transparent, #60A5FA, #A78BFA, #F472B6, transparent);
        margin: 1.5rem auto;
        border-radius: 2px;
        animation: glowExpand 3s ease-in-out infinite;
    }
    @keyframes glowExpand {
        0%, 100% { width: 120px; opacity: 0.6; }
        50% { width: 200px; opacity: 1; }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.8); }
        to { opacity: 1; transform: scale(1); }
    }

    /* Glass card styling for Streamlit columns */
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        background: rgba(30, 41, 59, 0.3) !important;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 24px !important;
        padding: 1.5rem !important;
        margin: 0.5rem !important;
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.4),
            0 0 0 1px rgba(255, 255, 255, 0.05) inset,
            0 20px 60px rgba(59, 130, 246, 0.1) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeInUp 0.8s ease-out both;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:hover {
        border-color: rgba(96, 165, 250, 0.4) !important;
        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.5),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset,
            0 0 40px rgba(59, 130, 246, 0.2) !important;
        transform: translateY(-4px);
    }

    /* Legacy glass-card class for other uses */
    .glass-card {
        background: rgba(30, 41, 59, 0.3) !important;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 24px !important;
        padding: 2rem !important;
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.4),
            0 0 0 1px rgba(255, 255, 255, 0.05) inset,
            0 20px 60px rgba(59, 130, 246, 0.1) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        transform-style: preserve-3d;
        perspective: 1000px;
        animation: fadeInUp 0.8s ease-out both;
    }
    .glass-card:hover {
        border-color: rgba(96, 165, 250, 0.4) !important;
        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.5),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset,
            0 0 40px rgba(59, 130, 246, 0.2) !important;
        transform: translateY(-8px) rotateX(2deg);
    }

    .section-header {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #E2E8F0 !important;
        margin-bottom: 1.5rem !important;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        animation: fadeInUp 0.6s ease-out both;
    }
    .section-header::after {
        content: '';
        flex: 1;
        height: 2px;
        background: linear-gradient(90deg, rgba(96, 165, 250, 0.6), rgba(139, 92, 246, 0.4), transparent);
        margin-left: 1rem;
        border-radius: 1px;
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #3B82F6 0%, #6366F1 50%, #8B5CF6 100%) !important;
        color: white !important;
        border-radius: 16px !important;
        padding: 1.2rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        border: none !important;
        width: 100% !important;
        letter-spacing: 0.03em;
        box-shadow:
            0 4px 15px rgba(59, 130, 246, 0.4),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset,
            0 0 30px rgba(59, 130, 246, 0.2) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        position: relative;
        overflow: hidden;
        animation: scaleIn 0.6s ease-out both;
    }
    div.stButton > button:first-child::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent 30%,
            rgba(255,255,255,0.15) 50%,
            transparent 70%
        );
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    @keyframes shimmer {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow:
            0 12px 40px rgba(59, 130, 246, 0.6),
            0 0 0 1px rgba(255, 255, 255, 0.15) inset,
            0 0 50px rgba(139, 92, 246, 0.3) !important;
    }
    div.stButton > button:first-child:active {
        transform: translateY(-1px) scale(0.98) !important;
    }

    div.stButton > button:not(:first-child) {
        background: rgba(30, 41, 59, 0.6) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(96, 165, 250, 0.3) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:not(:first-child):hover {
        background: rgba(59, 130, 246, 0.2) !important;
        border-color: rgba(96, 165, 250, 0.6) !important;
        transform: translateY(-2px);
    }

    div[data-testid='stMetric'] {
        background: rgba(30, 41, 59, 0.3) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.05) inset !important;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out both;
    }
    div[data-testid='stMetric']:hover {
        transform: translateY(-4px);
        box-shadow:
            0 12px 40px rgba(0, 0, 0, 0.4),
            0 0 20px rgba(59, 130, 246, 0.15) !important;
    }
    div[data-testid='stMetricLabel'] {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid='stMetricValue'] {
        color: #E2E8F0 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    div[data-testid='stMetricDelta'] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }

    .stTabs [data-baseweb='tab-list'] {
        gap: 0.5rem;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 16px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb='tab'] {
        background: transparent !important;
        color: #94A3B8 !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.75rem !important;
        font-weight: 500 !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stTabs [data-baseweb='tab']:hover {
        color: #E2E8F0 !important;
        background: rgba(255, 255, 255, 0.05) !important;
        transform: translateY(-2px);
    }
    .stTabs [aria-selected='true'] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(139, 92, 246, 0.25)) !important;
        color: #60A5FA !important;
        font-weight: 600 !important;
        box-shadow:
            0 4px 15px rgba(59, 130, 246, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        transform: translateY(-2px);
    }

    div[data-testid='stNumberInput'] input {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(96, 165, 250, 0.2) !important;
        border-radius: 12px !important;
        color: #E2E8F0 !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) inset;
        transition: all 0.3s ease;
    }
    div[data-testid='stNumberInput'] input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2), 0 2px 8px rgba(0,0,0,0.2) inset !important;
    }

    div[data-testid='stFileUploader'] {
        background: transparent !important;
        border: none !important;
    }
    div[data-testid='stFileUploader'] > section {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 2px dashed rgba(96, 165, 250, 0.25) !important;
        border-radius: 20px !important;
        padding: 2.5rem !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    div[data-testid='stFileUploader'] > section:hover {
        border-color: rgba(96, 165, 250, 0.8) !important;
        background: rgba(15, 23, 42, 0.7) !important;
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }
    div[data-testid='stFileUploader'] > section > div > span {
        color: #60A5FA !important;
        font-weight: 600 !important;
    }
    div[data-testid='stFileUploader'] > section > div > small {
        color: #64748B !important;
    }

    img {
        border-radius: 16px !important;
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.4),
            0 0 0 1px rgba(255, 255, 255, 0.05) inset !important;
        transition: all 0.4s ease;
    }
    img:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow:
            0 16px 48px rgba(0, 0, 0, 0.5),
            0 0 20px rgba(59, 130, 246, 0.1) !important;
    }

    .stAlert {
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
        backdrop-filter: blur(10px);
    }

    .stSpinner > div {
        border-top-color: #3B82F6 !important;
        border-right-color: #8B5CF6 !important;
        border-bottom-color: transparent !important;
        border-left-color: transparent !important;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Rotating analysis text animation (single definition — duplicates removed) */
    .stSpinner p {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #60A5FA !important;
        animation: textPulse 1.5s ease-in-out infinite;
        text-align: center;
        margin-top: 1rem;
    }
    @keyframes textPulse {
        0%, 100% { opacity: 1; transform: scale(1); text-shadow: 0 0 10px rgba(96,165,250,0.5); }
        50% { opacity: 0.6; transform: scale(0.95); text-shadow: 0 0 20px rgba(96,165,250,0.8); }
    }

    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, rgba(96, 165, 250, 0.4), rgba(139, 92, 246, 0.3), rgba(244, 114, 182, 0.2), transparent) !important;
        margin: 2.5rem 0 !important;
        border-radius: 1px;
    }

    .status-pulse {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: statusPulse 2s infinite;
    }
    @keyframes statusPulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .animate-1 { animation-delay: 0.1s; }
    .animate-2 { animation-delay: 0.2s; }
    .animate-3 { animation-delay: 0.3s; }
    .animate-4 { animation-delay: 0.4s; }

    .download-btn-wrapper button {
        background: linear-gradient(135deg, #10B981 0%, #059669 50%, #047857 100%) !important;
        box-shadow:
            0 4px 15px rgba(16, 185, 129, 0.4),
            0 0 30px rgba(16, 185, 129, 0.2) !important;
    }
    .download-btn-wrapper button:hover {
        box-shadow:
            0 8px 25px rgba(16, 185, 129, 0.6),
            0 0 40px rgba(16, 185, 129, 0.3) !important;
        transform: translateY(-3px) scale(1.02) !important;
    }

    /* Native Streamlit elements inside glass cards */
    .glass-card h3 {
        color: #E2E8F0 !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.25rem !important;
    }
    .glass-card p[data-testid="stCaption"] {
        color: #64748B !important;
        font-size: 0.8rem !important;
        margin-bottom: 1rem !important;
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

# --- 4. COMPREHENSIVE PDF GENERATOR - BLACK TEXT ON WHITE ---
class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Page {self.page_no()}/3 | Urban Change Detection Report | Generated by AI', ln=True, align='C')

def generate_comprehensive_pdf(stats, change_map_img, orig1_pil, orig2_pil, mask1_pil, mask2_pil):
    pdf = PDF()

    trend = 'expansion' if stats['net'] > 0 else 'contraction' if stats['net'] < 0 else 'stability'
    if stats['new'] > stats['loss']:
        dev_note = 'New urban developments outweighed urban loss, indicating active infrastructure growth.'
    else:
        dev_note = 'Urban loss exceeded new developments, suggesting potential redevelopment or environmental reclamation.'

    y1_str = str(stats['y1'])
    y2_str = str(stats['y2'])
    u1_str = f"{stats['u1']:.2f}"
    u2_str = f"{stats['u2']:.2f}"
    net_str = f"{stats['net']:+.2f}"
    gr_str = f"{stats['gr']:+.1f}"
    new_str = f"{stats['new']:.2f}"
    loss_str = f"{stats['loss']:.2f}"

    conclusion = (
        'Between ' + y1_str + ' and ' + y2_str + ', the analyzed region experienced significant urban ' + trend + '. '
        'The total urban area changed by ' + net_str + ' km2, representing a ' + gr_str + '% variation over the period. '
        'Specifically, ' + new_str + ' km2 of new urban areas were detected, while ' + loss_str + ' km2 of previously urbanized land was lost. '
        + dev_note + ' This data is critical for future urban planning and resource allocation.'
    )

    orig1_pil.save('temp_orig1.png')
    orig2_pil.save('temp_orig2.png')
    mask1_pil.save('temp_mask1.png')
    mask2_pil.save('temp_mask2.png')
    change_map_img.save('temp_map.png')

    # PAGE 1: EXECUTIVE SUMMARY - WHITE BACKGROUND, BLACK TEXT
    pdf.add_page()
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, 220, 297, style='F')

    # Header bar - gradient blue
    pdf.set_fill_color(37, 99, 235)
    pdf.rect(0, 0, 220, 35, style='F')
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 25, 'Urban Change Detection Report', ln=True, align='C')
    pdf.ln(5)

    # Subtitle
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 10, 'Analysis Period: ' + y1_str + ' to ' + y2_str, ln=True, align='C')
    pdf.ln(8)

    # Decorative line
    pdf.set_draw_color(37, 99, 235)
    pdf.set_line_width(0.5)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(8)

    # Metrics Section - BLACK TEXT
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, '1. Key Metrics', ln=True)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 8, '   Urban Area (' + y1_str + '): ' + u1_str + ' km2', ln=True)
    pdf.cell(0, 8, '   Urban Area (' + y2_str + '): ' + u2_str + ' km2', ln=True)
    pdf.cell(0, 8, '   Net Change: ' + net_str + ' km2 (' + gr_str + '%)', ln=True)
    pdf.cell(0, 8, '   New Urban Expansion: +' + new_str + ' km2', ln=True)
    pdf.cell(0, 8, '   Urban Loss: -' + loss_str + ' km2', ln=True)
    pdf.ln(10)

    # Decorative line
    pdf.set_draw_color(139, 92, 246)
    pdf.set_line_width(0.5)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(8)

    # Conclusion Section - BLACK TEXT
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, '2. Executive Conclusion', ln=True)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 6, conclusion)

    # PAGE 2: VISUAL ANALYSIS - WHITE BACKGROUND
    pdf.add_page()
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, 220, 297, style='F')

    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 15, 'Visual Analysis: Input & AI Detection', ln=True, align='C')
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(95, 8, 'Original Satellite (' + y1_str + ')', ln=False, align='C')
    pdf.cell(95, 8, 'Original Satellite (' + y2_str + ')', ln=True, align='C')
    pdf.image('temp_orig1.png', x=10, w=90)
    pdf.image('temp_orig2.png', x=110, w=90)
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(95, 8, 'AI Urban Mask (' + y1_str + ')', ln=False, align='C')
    pdf.cell(95, 8, 'AI Urban Mask (' + y2_str + ')', ln=True, align='C')
    pdf.image('temp_mask1.png', x=10, w=90)
    pdf.image('temp_mask2.png', x=110, w=90)

    # PAGE 3: CHANGE MAPPING - WHITE BACKGROUND
    pdf.add_page()
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, 220, 297, style='F')

    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 15, 'Change Detection Mapping', ln=True, align='C')
    pdf.ln(5)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 6, 'The map illustrates spatial distribution of urban changes. Red = new expansion, Orange = urban loss, Dark red = stable urban, Light green = stable non-urban.')
    pdf.ln(5)
    pdf.image('temp_map.png', x=10, w=190)

    pdf_output = pdf.output()
    return bytes(pdf_output) if not isinstance(pdf_output, bytes) else pdf_output

# ============================================================
#  UI LAYOUT - MAGICAL 3D REDESIGN
# ============================================================

# Hero Section with 3D effects
st.markdown('''
<div class="hero-container">
    <div class="hero-emoji">🛰</div>
    <h1 class="hero-title">Urban Change Detector</h1>
    <div class="hero-glow"></div>
    <p class="hero-subtitle">AI-Powered Satellite Imagery Analysis</p>
    <div style="margin-top: 1rem;">
        <span class="status-pulse"></span>
        <span style="color: #64748B; font-size: 0.85rem; margin-left: 0.5rem;">System Ready</span>
    </div>
</div>
''', unsafe_allow_html=True)

# Upload Section with Glassmorphism
st.markdown('<div class="section-header">📡 Upload Satellite Images</div>', unsafe_allow_html=True)

# Use Streamlit columns with CSS-targeted styling (no HTML wrappers around widgets)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Period 1 (T1)")
    st.caption("Baseline satellite imagery")
    img1_file = st.file_uploader("Drop T1 image", type=["png", "jpg", "jpeg"], key="img1")
    year1_default = get_year_from_file(img1_file) if img1_file else 2015
    year1 = st.number_input("Year T1", value=year1_default, step=1, key="year1_input")
    if img1_file:
        st.image(img1_file, caption=f"T1 Preview — {year1}", use_column_width=True)

with col2:
    st.subheader("🛸 Period 2 (T2)")
    st.caption("Comparison satellite imagery")
    img2_file = st.file_uploader("Drop T2 image", type=["png", "jpg", "jpeg"], key="img2")
    year2_default = get_year_from_file(img2_file) if img2_file else 2023
    year2 = st.number_input("Year T2", value=year2_default, step=1, key="year2_input")
    if img2_file:
        st.image(img2_file, caption=f"T2 Preview — {year2}", use_column_width=True)

# Analysis Button - Centered with 3D effect
st.markdown('<div style="max-width: 450px; margin: 2.5rem auto;">', unsafe_allow_html=True)
analyze_btn = st.button("🚀 Launch Analysis", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if analyze_btn:
    if not img1_file or not img2_file:
        st.error('⚠ Please upload both satellite images to proceed with the analysis.')
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

            # Results Section with animations
            st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(f'Urban Area {year1}', f'{u1:.2f} km2', 'Baseline')
            with col2:
                st.metric(f'Urban Area {year2}', f'{u2:.2f} km2', f'{net:+.2f} km2')
            with col3:
                st.metric('Net Change', f'{net:+.2f} km2', f'{gr:+.1f}%')
            with col4:
                st.metric('Annual Growth', f'{gr/yrs:+.2f} %/yr', f'over {yrs} years')

            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🗺 Visual Analysis</div>', unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(['📸 Original Imagery', '🤖 AI Detection Masks', '🔄 Change Map'])

            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(disp1, caption=f'T1 Satellite - {year1}', use_column_width=True)
                with c2:
                    st.image(disp2, caption=f'T2 Satellite - {year2}', use_column_width=True)

            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(mask1*255, caption=f'AI Urban Mask - {year1}', use_column_width=True, clamp=True)
                with c2:
                    st.image(mask2*255, caption=f'AI Urban Mask - {year2}', use_column_width=True, clamp=True)

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

                st.markdown('<div class="download-btn-wrapper">', unsafe_allow_html=True)
                st.download_button(
                    label='⬇ Download Full PDF Report',
                    data=pdf_bytes,
                    file_name=f'Urban_Change_Report_{year1}_{year2}.pdf',
                    mime='application/pdf',
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('''
    <div style="text-align: center; padding: 4rem; color: #475569;">
        <div style="font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.4; animation: float 3s ease-in-out infinite;">🛰</div>
        <p style="font-size: 1.2rem; color: #64748B; font-weight: 500;">Upload two satellite images to begin AI-powered urban change detection</p>
        <p style="font-size: 0.9rem; color: #475569; margin-top: 0.75rem;">Supports temporal analysis across multiple years with deep learning segmentation</p>
        <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1rem;">
            <span style="background: rgba(59,130,246,0.15); color: #60A5FA; padding: 0.5rem 1rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;">🧠 AI Segmentation</span>
            <span style="background: rgba(139,92,246,0.15); color: #A78BFA; padding: 0.5rem 1rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;">📊 Change Detection</span>
            <span style="background: rgba(16,185,129,0.15); color: #34D399; padding: 0.5rem 1rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;">📄 PDF Export</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
