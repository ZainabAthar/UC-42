import streamlit as st
import os
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import fitz  # PyMuPDF
from checks import (
    call_screenshot_api, call_forgery_api, analyze_document_tampering_fast_ela, 
    analyze_metadata, call_trufor_analysis, analyze_document_structure_local,
    analyze_pdf_native_images
)
import numpy as np
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Forensic API Checks", layout="wide")
st.title("Forensic API Checks - Modularity Test")
st.markdown("This standalone app is dedicated only to API checks: Screenshot Detection, Deepfake/Forgery API, and Fast ELA for Tampering.")

uploaded_file = st.file_uploader("Upload Document to Analyze", type=["jpg", "png", "jpeg", "pdf"])

if uploaded_file:
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension == '.pdf':
        with st.spinner("Processing PDF..."):
            pdf_bytes = uploaded_file.getbuffer()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            num_pages = len(pdf_document)
            
            if num_pages > 1:
                page_num = st.selectbox("Select Page to Analyze", range(1, num_pages + 1), index=0)
            else:
                page_num = 1
                
            page = pdf_document.load_page(page_num - 1)
            pix = page.get_pixmap(dpi=200)
            
            img_path = Path("temp_input") / f"{Path(uploaded_file.name).stem}_page_{page_num}.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(img_path))
            pdf_document.close()
            
            orig_path = Path("temp_input") / uploaded_file.name
            with open(orig_path, "wb") as f:
                f.write(pdf_bytes)
    else:
        # Save uploaded file temporarily for processing
        img_path = Path("temp_input") / uploaded_file.name
        img_path.parent.mkdir(parents=True, exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        orig_path = img_path
            
    st.image(Image.open(img_path), caption="Parsed Document Representation", use_column_width=True)

    st.subheader("Run Checks")
    
    run_trufor = st.checkbox("🔍 Enable Deep Copy-Paste/Splicing Detection (TruFor AI)", value=True, help="Highly accurate deep neural tracing of exact pasted regions (Takes 15-30s).")
    
    if st.button("Run Full Enterprise Forensic Suite", use_container_width=True, type="primary"):
        st.session_state.final_results = {}
        
        with st.status("Executing Forensic Analysis Pipeline...", expanded=True) as status:
            st.write("🔍 Extracting Deep Document Metadata...")
            meta_result = analyze_metadata(orig_path)
            st.session_state.final_results['metadata'] = meta_result
            
            st.write("📸 Checking Screenshot / Origin (Sightengine Type Model)...")
            is_screen_data = call_screenshot_api(orig_path)
            st.session_state.final_results['screenshot'] = is_screen_data
            
            st.write("🔬 Verifying Content Structure (Local Engine)...")
            struct_result = analyze_document_structure_local(orig_path)
            st.session_state.final_results['edenai'] = struct_result
            
            if file_extension == '.pdf':
                st.write("📄 Analyzing Native PDF Embedded Images...")
                native_img_res = analyze_pdf_native_images(orig_path)
                st.session_state.final_results['native_pdf'] = native_img_res
            
            st.write("🛡️ Evaluating Synthesis & GenAI Score (Sightengine)...")
            api_results = call_forgery_api(img_path)
            st.session_state.final_results['api'] = api_results
            
            st.write("🗺️ Processing Pixel-Level Error Level Analysis (ELA)...")
            ela_result = analyze_document_tampering_fast_ela(str(img_path))
            st.session_state.final_results['ela'] = ela_result
            
            if run_trufor:
                st.write("🧬 Tracing Spliced Pixels & Copy-Move Signatures (Deep Model)...")
                trufor_res = call_trufor_analysis(str(img_path))
                st.session_state.final_results['trufor'] = trufor_res
            
            status.update(label="Forensic Suite Execution Complete", state="complete", expanded=False)
            
        st.markdown("---")
        st.subheader("Final Result Report")
        
        results = st.session_state.final_results
        
        score_ela = results['ela'].get('score', 0) if 'score' in results['ela'] else 0
        is_edited_meta = results['metadata'].get('is_edited', False)
        
        forgery_confidence = 0
        if is_edited_meta:
            forgery_confidence += 40
        if score_ela > 0.55:
            forgery_confidence += 40
        if results['screenshot'].get('is_screenshot', False):
            forgery_confidence += 20
            
        if run_trufor and 'trufor' in results:
            t_score = results['trufor'].get('score', 0)
            if t_score > 0.6:
                forgery_confidence += 50
                
        if 'native_pdf' in results:
            npdf_score = results['native_pdf'].get('score', 0)
            if npdf_score > 0.6:
                forgery_confidence += 50
            
        col1, col2 = st.columns([2, 1])
            
        with col1:
            if forgery_confidence > 60:
                st.error(f"🔥 **Status: REJECTED (High Tampering Probability)** | Total Confidence Factor: {min(forgery_confidence, 100)}%")
            elif forgery_confidence > 20:
                st.warning(f"⚠️ **Status: SUSPICIOUS (Manual Review Recommended)** | Total Confidence Factor: {forgery_confidence}%")
            else:
                st.success(f"🛡️ **Status: VERIFIED (Appears Authentic)** | Total Confidence Factor: {forgery_confidence}%")
                
            st.markdown("### Executive Forensic Report")
            
            # 1. Metadata
            meta_status = "❌ **SUSPICIOUS:** The document's internal metadata indicates manipulation. Evidence of explicit image-editing software or deliberately stripped origination tags was detected." if is_edited_meta else "✅ **AUTHENTIC:** The document maintains secure formatting metadata consistent with standard generation. No manipulation software footprints found."
            st.markdown(f"**1. Metadata Signature Analysis**\n> {meta_status}")
            
            # 2. ELA
            ela_status = "❌ **ANOMALOUS:** Pixel-level compression differentials detected. The surface shows structural inconsistencies indicative of localized blurring, cloning, or text erasure." if score_ela > 0.85 else "✅ **VERIFIED:** Pixel compression matrices appear uniform across the document surface. No overt surface-level edits (e.g., eraser marks) detected."
            st.markdown(f"**2. Surface Compression Variability (ELA)**\n> {ela_status}")
            
            # 3. Origin
            is_ss = results['screenshot'].get('is_screenshot', False)
            ss_details = results['screenshot'].get('details', '')
            orig_status = "❌ **UNVERIFIED ORIGIN:** Computer vision models classify this document as a digital screenshot or artificial illustration construct rather than a native file/photograph." if is_ss else f"✅ **NATIVE ORIGIN:** Optical properties confirm native origination. ({ss_details})"
            st.markdown(f"**3. Document Origination Check**\n> {orig_status}")
            
            # 4. TruFor
            if run_trufor:
                t_score = results.get('trufor', {}).get('score', 0)
                trufor_status = "❌ **SEVERE FORGERY DETECTED:** Advanced localization models have traced definitive boundaries of spliced imagery, digital alterations, or copy-pasted overlays (e.g., pasted QR codes or signatures)." if t_score > 0.85 else "✅ **CLEAN:** Deep neural pixel tracing analyzed the entire canvas and found no evidence of copy-pasted elements or spliced artifact overlays."
                st.markdown(f"**4. Deep Neural Splicing Detection (TruFor AI)**\n> {trufor_status}")
                
            # 5. Structure
            struct_ok = results['edenai'].get('coherent')
            struct_status = "✅ **COHERENT:** The document successfully passed heuristic text-layer verification, natively exhibiting standard financial/billing structural indicators (Amounts, Dates, Currencies)." if struct_ok else f"⚠️ **WARNING:** The document failed structural evaluation. {results['edenai'].get('info', 'Lacks expected mathematical text structure for documents.')}"
            st.markdown(f"**5. Layout & Structural Coherence**\n> {struct_status}")
            
            # 6. Native PDF Images
            if 'native_pdf' in results:
                np_score = results['native_pdf'].get('score', 0)
                np_status = f"❌ **STRUCTURAL ANOMALY DETECTED:** {results['native_pdf'].get('message')} This strongly suggests a composite or copy-pasted object." if np_score > 0.6 else "✅ **UNIFORM STRUCTURE:** Native PDF object analysis found no explicitly suspicious pasted images/layers."
                st.markdown(f"**6. Native PDF Object Layer (Copy/Paste Check)**\n> {np_status}")
            
            with st.expander("View Deep API Logs"):
                st.write("**Sightengine AI Gen Model:**")
                st.json(results['api'])
                st.write("**Sightengine Type Model (Screenshot Check):**")
                st.json(results['screenshot'])
                st.write("**Local Structural Parser Info:**")
                st.json(results['edenai'])
                st.write("**Detailed Metadata Extraction:**")
                st.json(results['metadata'])
                if run_trufor and 'trufor' in results:
                    st.write("**TruFor Local Analysis Details:**")
                    st.json({
                        "score": results['trufor'].get('score'),
                        "bounding_box": results['trufor'].get('bbox'),
                        "message": results['trufor'].get('message')
                    })
                if 'native_pdf' in results:
                    st.write("**Native PDF Structural Image Analysis:**")
                    st.json(results['native_pdf'])
                
        with col2:
            if run_trufor and 'trufor' in results and 'map' in results['trufor']:
                st.write("**Copy-Paste/Splice Region Map (TruFor):**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(Image.open(img_path))
                
                loc_map = results['trufor']['map']
                ax.imshow(loc_map, cmap='jet', alpha=0.55)
                
                bbox = results['trufor'].get('bbox')
                if bbox and results['trufor'].get('score', 0) > 0.85:
                    import matplotlib.patches as patches
                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=3, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    ax.set_title("⚠️ RED ALARM: SUSPECTED SPLICING ⚠️", color='red')
                else:
                    ax.set_title("Clean: No High-Confidence Splices")
                    
                ax.axis('off')
                st.pyplot(fig)
            elif 'map' in results['ela']:
                st.write("**Pixel Difference Heatmap (ELA):**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ela_map = results['ela']['map']
                im = ax.imshow(ela_map, cmap='magma')
                ax.set_title("Compression Variance Surface Map")
                ax.axis('off')
                fig.colorbar(im, ax=ax)
                st.pyplot(fig)
