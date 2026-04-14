import streamlit as st
import os
import json
import time
import tempfile
import numpy as np
from PIL import Image
from utils import pdf_to_images, pil_to_base64, crop_region, scale_to_unit_eight
from api_handler import OpenRouterClient, check_openrouter_status
from forensics import analyze_region_consistency, calculate_ela, get_noise_map, calculate_diff_map
from metadata_audit import audit_pdf_metadata, audit_image_metadata, finalize_report, finalize_pdf_report

REF_PDF_PATH = r"d:\MD-II\bill-pdf\digital_bill.pdf"

# --- UI CONFIG ---
st.set_page_config(page_title="Forensic Region Consistency", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Minimalist Dark Monochrome Look
st.markdown("""
<style>
    .main {
        background-color: #111111;
        color: #f0f0f0;
    }
    .stApp {
        background-color: #111111;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
        border-right: 1px solid #333333;
    }
    .f-card {
        background: #1e1e1e;
        border-radius: 4px;
        padding: 20px;
        border: 1px solid #333333;
        margin-bottom: 20px;
    }
    .metric-card {
        text-align: center;
        padding: 15px;
        background: #1a1a1a;
        border-radius: 4px;
        border: 1px solid #444444;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton>button {
        background-color: #ffffff;
        color: #000000;
        border-radius: 2px;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #cccccc;
        color: #000000;
    }
    /* Simple tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent ;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
def load_templates():
    if os.path.exists("templates.json"):
        with open("templates.json", "r") as f:
            return json.load(f)
    return {}

# --- SIDEBAR ---
with st.sidebar:
    st.title(" Forensic Lab")
    st.markdown("Advanced Region Consistency Analysis")
    
    # api_key = st.text_input("OpenRouter API Key", type="password")
    api_key = "YOUR_API_KEY"
    model_choice = st.selectbox("Vision Model", [
        "google/gemini-2.0-flash-001",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-mini"
    ])
    
    poppler_path = st.text_input("Poppler Path", value=r"C:\Users\ZAINAB\Downloads\poppler\poppler-25.12.0\Library\bin")
    
    st.markdown("---")
    if st.button("Check API Status"):
        if api_key:
            status = check_openrouter_status(api_key)
            st.json(status)
        else:
            st.error("Enter API Key")

# --- MAIN UI ---
st.title("Bill Region Consistency Detector")
st.markdown("Detecting high-tech forgery through structural and pixel-level inconsistencies.")

uploaded_file = st.file_uploader("Upload Bill (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    # Save temp file
    suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Load Images
        if suffix == ".pdf":
            pages = pdf_to_images(tmp_path, poppler_path=poppler_path)
            page_img = pages[0]
        else:
            page_img = Image.open(tmp_path)

        # 1. API Analysis
        if st.button("Run Full Forensic Analysis"):
            if not api_key:
                st.warning("Please provide an OpenRouter API key in the sidebar.")
            else:
                with st.spinner("Executing AI Perception and Forensic Scan..."):
                    # Call OpenRouter
                    b64 = pil_to_base64(page_img)
                    client = OpenRouterClient(api_key, model=model_choice)
                    ai_result = client.call_vision(b64)
                    
                    if "error" in ai_result:
                        st.error(f"API Error: {ai_result['error']}")
                    else:
                        provider = ai_result.get("provider_company", "Unknown")
                        detected_regions = ai_result.get("regions", {})
                        ai_forgery = ai_result.get("ai_forgery_data", {})
                        
                        # Load local templates for fallback/validation
                        templates = load_templates()
                        template_regions = templates.get(provider, {})
                        
                        # Merge regions (AI detection prioritized)
                        final_regions = {**template_regions, **detected_regions}
                        
                        # 2. Metadata Audit
                        if suffix == ".pdf":
                            meta_results = audit_pdf_metadata(tmp_path)
                        else:
                            meta_results = audit_image_metadata(tmp_path)

                        # 3. Load Reference (Golden Standard)
                        ref_crops = None
                        if provider == "K-Electric" and os.path.exists(REF_PDF_PATH):
                            try:
                                ref_pages = pdf_to_images(REF_PDF_PATH, poppler_path=poppler_path)
                                ref_img = ref_pages[0]
                                ref_crops = {name: crop_region(ref_img, box) for name, box in final_regions.items()}
                            except:
                                pass

                        # 4. Forensic Processing
                        crops = {name: crop_region(page_img, box) for name, box in final_regions.items()}
                        consistency = analyze_region_consistency(page_img, crops, reference_regions=ref_crops)
                        
                        # --- DISPLAY RESULTS ---
                        st.divider()
                        st.subheader(f"Forensic Audit: {provider} Document")
                        
                        # Metadata Red Flags
                        if meta_results.get("red_flags"):
                            for flag in meta_results["red_flags"]:
                                st.warning(f"🚩 Metadata Alert: {flag}")

                        # Combined Intelligence Verdict
                        score = consistency['texture_variance_score']
                        ai_score = ai_forgery.get("score", 0)
                        
                        # Weighted Verdict: AI reasoning (60%) + Texture math (40%)
                        overall_risk = (ai_score * 0.6) + (min(1.0, score / 4.0) * 0.4)
                        
                        if overall_risk < 0.3:
                            verdict, v_color = "Genuine Background", "green"
                        elif overall_risk < 0.6:
                            verdict, v_color = "Anomaly Detected (Review)", "orange"
                        else:
                            verdict, v_color = "High Risk: Tampering Detected", "red"
                            
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 10px solid {v_color}; text-align: left;">
                            <h3>Verdict: {verdict}</h3>
                            <p><b>AI Reasoning:</b> {ai_forgery.get("reasoning", "No detailed reasoning provided.")}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Texture Variance", f"{score:.2f}", help="Measures if the background fingerprint is uniform across regions.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("AI Confidence", f"{100 - (ai_score * 100):.1f}%", help="AI vision confidence in document integrity.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Metadata Risk", f"{'HIGH' if meta_results.get('is_tampered_hint') else 'LOW'}")
                            st.markdown('</div>', unsafe_allow_html=True)

                        st.divider()

                        # --- FORENSIC SUMMARY DASHBOARD ---
                        st.subheader("📊 Forensic Audit Summary")
                        
                        s_col1, s_col2 = st.columns([2, 1])
                        with s_col1:
                            st.markdown("#### Regional Scoring Matrix")
                            score_data = []
                            for name, data in consistency.get('regions', {}).items():
                                status = "🚩 Suspicious" if data.get('is_suspicious') else "✅ Consistent"
                                score_data.append({
                                    "Region": name,
                                    "Z-Score": f"{data.get('z_score', 0):.4f}",
                                    "Diff": f"{data.get('diff_score', 0):.4f}",
                                    "Status": status
                                })
                            st.table(score_data)
                            
                        with s_col2:
                            st.markdown("#### Full Audit Metadata")
                            info = meta_results.get('info', {})
                            
                            full_meta = []
                            for k, v in info.items():
                                full_meta.append({"Property": k, "Value": str(v)})
                            
                            # Use a taller container for all details
                            st.dataframe(full_meta, use_container_width=True, hide_index=True)
                            
                            if meta_results.get('red_flags'):
                                with st.expander("🚩 Metadata Alerts", expanded=True):
                                    for flag in meta_results['red_flags']:
                                        st.error(flag)

                        st.divider()
                        # Final Report Section
                        pdf_report = finalize_pdf_report(ai_result, consistency, meta_results, model_name=model_choice)
                        st.download_button(
                            label="⚖️ Download Official Forensic PDF Report",
                            data=pdf_report,
                            file_name=f"Forensic_Audit_{provider}_{time.strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                        )
                        
                        # Multi-Region Dashboard
                        tabs = st.tabs(list(crops.keys()))
                        
                        for i, (name, crop) in enumerate(crops.items()):
                            with tabs[i]:
                                c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
                                meta = consistency['regions'].get(name, {})
                                
                                with c1:
                                    st.image(crop, caption=f"Audit Region: {name}", use_container_width=True)
                                    st.write(f"Z-Score: {meta.get('z_score', 0):.2f}")
                                    if meta.get('is_suspicious'):
                                        st.error("🚨 Suspicious")
                                    else:
                                        st.success("✅ Consistent")
                                        
                                with c2:
                                    ela = calculate_ela(crop)
                                    st.image(ela, caption="ELA map", use_container_width=True)
                                    
                                with c3:
                                    noise = get_noise_map(crop)
                                    st.image(scale_to_unit_eight(noise), caption="Noise map", use_container_width=True)

                                with c4:
                                    if ref_crops and name in ref_crops:
                                        diff = calculate_diff_map(crop, ref_crops[name])
                                        st.image(scale_to_unit_eight(diff), caption="Pixel Diff (vs Gold)", use_container_width=True)
                                        st.caption("Highlights bitwise changes from the original template.")
                                    else:
                                        st.info("No Ref Available")

                        # Full Page Map
                        with st.expander("View Full Document Forensic Overlays"):
                            full_ela = calculate_ela(page_img)
                            st.image(full_ela, caption="Full Page ELA", use_container_width=True)

    except Exception as e:
        st.error(f"Processing Error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    # Landing state
    st.info("Upload a document to begin the forensic audit.")
