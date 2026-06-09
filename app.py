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

# --- 1. GLOBAL SETTINGS ---
plt.rcParams.update({
    'text.color': '#F8FAFC',
    'axes.labelcolor': '#F8FAFC',
    'axes.edgecolor': '#334155',
    'axes.facecolor': '#1E293B',
    'figure.facecolor': '#0F172A',
    'xtick.color': '#F8FAFC',
    'ytick.color': '#F8FAFC'
})

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    label, p, div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"], h1, h2, h3, h4 {
        color: #F8FAFC !important;
    }
    
    div.stButton > button {
        background-color: #10B981;
        color: white !important;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #059669;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed #475569;
        border-radius: 8px;
        padding: 10px;
    }
    
    .upload-container {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIGURATION & MODEL ---
st.set_page_config(page_title="Urban Change Detector", page_icon="🏙️", layout="wide")

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

def create_pdf_report(stats, change_map_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(248, 250, 252)
    pdf.set_fill_color(15, 23, 42)
    pdf.cell(0, 15, "Urban Change Detection Report", ln=True, align="C", fill=True)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Analysis Period: {stats['y1']} to {stats['y2']}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Key Metrics:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"- Urban Area {stats['y1']}: {stats['u1']:.2f} km²", ln=True)
    pdf.cell(0, 8, f"- Urban Area {stats['y2']}: {stats['u2']:.2f} km²", ln=True)
    pdf.cell(0, 8, f"- Net Change: {stats['net']:+.2f} km² ({stats['gr']:+.1f}%)", ln=True)
    pdf.cell(0, 8, f"- New Urban Expansion: +{stats['new']:.2f} km²", ln=True)
    pdf.cell(0, 8, f"- Urban Loss: -{stats['loss']:.2f} km²", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Change Map:", ln=True)
    temp_path = "temp_map.png"
    change_map_img.save(temp_path)
    pdf.image(temp_path, x=10, w=190)
    
    # FIX: Force output to be bytes
    pdf_output = pdf.output()
    if isinstance(pdf_output, str):
        pdf_bytes = pdf_output.encode('latin-1')
    else:
        pdf_bytes = bytes(pdf_output)
        
    return pdf_bytes

# --- 4. UI LAYOUT ---
st.markdown("<h1 style='text-align: center; color: #10B981;'>️ Urban Change Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8;'>AI-powered satellite imagery analysis</p>", unsafe_allow_html=True)
st.divider()

st.subheader("📥 Upload Satellite Images", divider="gray")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.subheader("Period 1 (T1)")
    img1_file = st.file_uploader("Upload T1 Image", type=["png", "jpg", "jpeg"], key="img1")
    
    year1_default = get_year_from_file(img1_file) if img1_file else 2015
    year1 = st.number_input("Year T1", value=year1_default, step=1, key="year1_input")
    
    if img1_file:
        st.image(img1_file, caption=f"Preview T1 ({year1})", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.subheader("Period 2 (T2)")
    img2_file = st.file_uploader("Upload T2 Image", type=["png", "jpg", "jpeg"], key="img2")
    
    year2_default = get_year_from_file(img2_file) if img2_file else 2023
    year2 = st.number_input("Year T2", value=year2_default, step=1, key="year2_input")
    
    if img2_file:
        st.image(img2_file, caption=f"Preview T2 ({year2})", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

analyze_btn = st.button("🚀 Run Analysis", type="primary")

if analyze_btn:
    if not img1_file or not img2_file:
        st.error("️ Please upload both images to proceed.")
    else:
        with st.spinner('Processing satellite imagery...'):
            chw1, disp1 = preprocess(img1_file)
            chw2, disp2 = preprocess(img2_file)
            mask1 = predict(chw1)
            mask2 = predict(chw2)
            cm, u1, u2, new, loss, net, gr, yrs = calc_stats(mask1, mask2, year1, year2)
            
            stats = {'y1': year1, 'y2': year2, 'u1': u1, 'u2': u2, 'new': new, 'loss': loss, 'net': net, 'gr': gr}
            
            st.subheader("📊 Analytics Summary", divider="gray")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"Urban Area {year1}", f"{u1:.2f} km²")
            col2.metric(f"Urban Area {year2}", f"{u2:.2f} km²")
            col3.metric("Net Change", f"{net:+.2f} km²", f"{gr:+.1f}%")
            col4.metric("Annual Growth", f"{gr/yrs:+.2f} %/yr")
            
            st.divider()
            
            st.subheader("🗺️ Visual Analysis", divider="gray")
            tab1, tab2, tab3 = st.tabs(["📸 Original", "🤖 AI Masks", "🔄 Change Map"])
            
            with tab1:
                c1, c2 = st.columns(2)
                c1.image(disp1, caption=f"T1 ({year1})", use_column_width=True)
                c2.image(disp2, caption=f"T2 ({year2})", use_column_width=True)
                
            with tab2:
                c1, c2 = st.columns(2)
                c1.image(mask1*255, caption=f"Mask {year1}", use_column_width=True, clamp=True)
                c2.image(mask2*255, caption=f"Mask {year2}", use_column_width=True, clamp=True)
                
            with tab3:
                fig, ax = plt.subplots(figsize=(10, 6))
                cmap = ListedColormap(['#90EE90', '#FF0000', '#FFA500', '#800000'])
                ax.imshow(cm, cmap=cmap, vmin=0, vmax=3)
                ax.axis('off')
                
                legend_elements = [
                    Patch(facecolor='#90EE90', label='Stable Non-Urban'),
                    Patch(facecolor='#800000', label='Stable Urban'),
                    Patch(facecolor='#FF0000', label=f'New Urban (+{new:.2f} km²)'),
                    Patch(facecolor='#FFA500', label=f'Urban Loss (-{loss:.2f} km²)')
                ]
                leg = ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=True)
                
                for text in leg.get_texts():
                    text.set_color('#F8FAFC')
                leg.get_frame().set_facecolor('#1E293B')
                leg.get_frame().set_edgecolor('#334155')
                
                st.pyplot(fig)
                
                st.divider()
                st.subheader("📄 Export Report")
                
                fig.canvas.draw()
                width, height = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(height, width, 4)
                map_img = Image.fromarray(buf[:, :, :3], 'RGB')
                
                pdf_bytes = create_pdf_report(stats, map_img)
                
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"Urban_Change_Report_{year1}_{year2}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
else:
    st.info("👆 Upload your satellite images above and click **Run Analysis** to begin.")
