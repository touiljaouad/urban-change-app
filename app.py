# app.py
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from model import UNet # Imports the model from model.py

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Urban Change Detector", page_icon="🏙️", layout="wide")

# Force CPU for Streamlit free tier deployment
DEVICE = torch.device('cpu') 
IMAGE_SIZE = 256
PIXEL_RESOLUTION = 10 # meters per pixel

# --- 2. LOAD MODEL (CACHED) ---
@st.cache_resource
def load_model():
    model = UNet(in_ch=3, out_ch=2)
    # Load weights, map to CPU
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
    # Return (C, H, W) for PyTorch and (H, W, C) for displaying
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
    cm[(m1==0)&(m2==1)] = 1 # Expansion
    cm[(m1==1)&(m2==0)] = 2 # Loss
    cm[(m1==1)&(m2==1)] = 3 # Stable Urban
    
    new = (cm==1).sum() * px_km2
    loss = (cm==2).sum() * px_km2
    net = u2 - u1
    yrs = max(y2 - y1, 1)
    gr = (net/u1*100) if u1 > 0 else 0
    return cm, u1, u2, new, loss, net, gr, yrs

# --- 4. STREAMLIT UI ---
st.title("🏙️ Satellite Urban Change Detection")
st.markdown("Upload satellite imagery from two periods to instantly analyze urban expansion.")

# Sidebar
with st.sidebar:
    st.header("Upload Imagery")
    year1 = st.number_input("Year of Image 1 (T1)", value=2015, step=1)
    img1_file = st.file_uploader(f"Upload Image 1", type=["png", "jpg", "jpeg"])
    
    st.divider()
    
    year2 = st.number_input("Year of Image 2 (T2)", value=2023, step=1)
    img2_file = st.file_uploader(f"Upload Image 2", type=["png", "jpg", "jpeg"])
    
    analyze_btn = st.button("🚀 Analyze Urban Change", use_container_width=True, type="primary")

# Main Page
if analyze_btn:
    if not img1_file or not img2_file:
        st.error("Please upload both images!")
    else:
        with st.spinner('AI is analyzing the satellite imagery...'):
            # Run Inference
            chw1, disp1 = preprocess(img1_file)
            chw2, disp2 = preprocess(img2_file)
            
            mask1 = predict(chw1)
            mask2 = predict(chw2)
            
            cm, u1, u2, new, loss, net, gr, yrs = calc_stats(mask1, mask2, year1, year2)
            
            # Display Metrics
            st.subheader("📊 Analytics Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"Urban Area {year1}", f"{u1:.2f} km²")
            col2.metric(f"Urban Area {year2}", f"{u2:.2f} km²")
            col3.metric("Net Change", f"{net:+.2f} km²", f"{gr:+.1f}%")
            col4.metric("Annual Growth", f"{gr/yrs:+.2f} %/yr")
            
            st.divider()
            
            # Display Images in Tabs
            st.subheader("🗺️ Visual Maps")
            tab1, tab2, tab3 = st.tabs(["Original Images", "AI Detected Urban Masks", "Change Map"])
            
            with tab1:
                c1, c2 = st.columns(2)
                c1.image(disp1, caption=f"T1 ({year1})", use_column_width=True)
                c2.image(disp2, caption=f"T2 ({year2})", use_column_width=True)
                
            with tab2:
                c1, c2 = st.columns(2)
                c1.image(mask1*255, caption=f"Urban Mask ({year1})", use_column_width=True, clamp=True)
                c2.image(mask2*255, caption=f"Urban Mask ({year2})", use_column_width=True, clamp=True)
                
            with tab3:
                fig, ax = plt.subplots(figsize=(8, 6))
                cmap = ListedColormap(['#90EE90', '#FF0000', '#FFA500', '#800000'])
                ax.imshow(cm, cmap=cmap, vmin=0, vmax=3)
                ax.axis('off')
                
                # Legend
                legend_elements = [
                    Patch(facecolor='#90EE90', label='Stable Non-Urban'),
                    Patch(facecolor='#800000', label='Stable Urban'),
                    Patch(facecolor='#FF0000', label=f'New Urban (+{new:.2f} km²)'),
                    Patch(facecolor='#FFA500', label=f'Urban Loss (-{loss:.2f} km²)')
                ]
                ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
                
                st.pyplot(fig)
else:
    st.info("👈 Upload your satellite images in the sidebar and click Analyze.")