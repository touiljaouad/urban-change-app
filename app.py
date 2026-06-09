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

# --- 1. CUSTOM CSS FOR AESTHETICS ---
st.markdown("""
<style>
    /* Hide default Streamlit footer and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Style the main button */
    div.stButton > button {
        background-color: #10B981;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #059669;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }

    /* Style the sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    
    /* Style file uploaders */
    div[data-testid="stFileUploader"] {
        border: 1px dashed #475569;
        border-radius: 8px;
        padding: 10px;
    }

    /* Custom Metric styling */
    .metric-container {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIGURATION & MODEL LOADING ---
st.set_page_config(page_title="Urban Change Detector", page_icon="️", layout="wide")

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

# --- 3. HELPER FUNCTIONS ---
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

# --- 4. MODERN UI LAYOUT ---
# Header
st.markdown("<h1 style='text-align: center; color: #10B981;'>🏙️ Urban Change Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8;'>AI-powered satellite imagery analysis for urban expansion tracking</p>", unsafe_allow_html=True)
st.divider()

# Sidebar
with st.sidebar:
    st.header("📥 Data Input", divider="gray")
    year1 = st.number_input("Year of Image 1 (T1)", value=2015, step=1)
    img1_file = st.file_uploader("Upload T1 Image", type=["png", "jpg", "jpeg"])
    
    st.divider()
    
    year2 = st.number_input("Year of Image 2 (T2)", value=2023, step=1)
    img2_file = st.file_uploader("Upload T2 Image", type=["png", "jpg", "jpeg"])
    
    st.divider()
    analyze_btn = st.button("🚀 Run Analysis", type="primary")

# Main Content Area
if analyze_btn:
    if not img1_file or not img2_file:
        st.error("⚠️ Please upload both images to proceed.")
    else:
        with st.spinner('Processing satellite imagery...'):
            chw1, disp1 = preprocess(img1_file)
            chw2, disp2 = preprocess(img2_file)
            mask1 = predict(chw1)
            mask2 = predict(chw2)
            cm, u1, u2, new, loss, net, gr, yrs = calc_stats(mask1, mask2, year1, year2)
            
            # --- METRICS DASHBOARD ---
            st.subheader("📊 Analytics Summary", divider="gray")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(label=f"Urban Area {year1}", value=f"{u1:.2f} km²", delta=None)
            with col2:
                st.metric(label=f"Urban Area {year2}", value=f"{u2:.2f} km²", delta=None)
            with col3:
                st.metric(label="Net Change", value=f"{net:+.2f} km²", delta=f"{gr:+.1f}%")
            with col4:
                st.metric(label="Annual Growth", value=f"{gr/yrs:+.2f} %/yr", delta=None)
            
            st.divider()
            
            # --- VISUALIZATION TABS ---
            st.subheader("🗺️ Visual Analysis", divider="gray")
            tab1, tab2, tab3 = st.tabs(["📸 Original Imagery", " AI Urban Masks", "🔄 Change Map"])
            
            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(disp1, caption=f"Period T1 ({year1})", use_column_width=True)
                with c2:
                    st.image(disp2, caption=f"Period T2 ({year2})", use_column_width=True)
                    
            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(mask1*255, caption=f"Detected Urban Area ({year1})", use_column_width=True, clamp=True)
                with c2:
                    st.image(mask2*255, caption=f"Detected Urban Area ({year2})", use_column_width=True, clamp=True)
                    
            with tab3:
                # Create a nice container for the map
                with st.container(border=True):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    cmap = ListedColormap(['#90EE90', '#FF0000', '#FFA500', '#800000'])
                    ax.imshow(cm, cmap=cmap, vmin=0, vmax=3)
                    ax.axis('off')
                    
                    legend_elements = [
                        Patch(facecolor='#90EE90', label='Stable Non-Urban'),
                        Patch(facecolor='#800000', label='Stable Urban'),
                        Patch(facecolor='#FF0000', label=f'New Urban Expansion (+{new:.2f} km²)'),
                        Patch(facecolor='#FFA500', label=f'Urban Loss (-{loss:.2f} km²)')
                    ]
                    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')
                    
                    st.pyplot(fig)
else:
    # Empty state
    st.info("👈 Upload your satellite images in the sidebar and click **Run Analysis** to begin.")
