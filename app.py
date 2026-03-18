import streamlit as st
import subprocess
import os
import sys
from pathlib import Path
import time
import numpy as np
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Helper Functions for API and Screenshot Detection ---
def is_screenshot(img_path):
    """
    Detects if an image is a screenshot based on filename and EXIF data.
    """
    name = str(img_path).lower()
    if "screenshot" in name or "screen_shot" in name or "capture" in name:
        return True
    
    try:
        img = Image.open(img_path)
        exif = img.getexif()
        if not exif:
            return True
            
        # 271 is Make, 272 is Model (Standard EXIF tags for cameras)
        has_camera_info = 271 in exif or 272 in exif
        if not has_camera_info:
            return True
    except Exception:
        return True
        
    return False

def call_forgery_api(img_path):
    """
    Calls the external Screenshot Forgery/Deepfake API (Sightengine).
    Uses SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET from .env.
    """
    api_user = os.getenv("SIGHTENGINE_API_USER")
    api_secret = os.getenv("SIGHTENGINE_API_SECRET")
    
    if not api_user or not api_secret:
        return {"error": "Sightengine API User or Secret missing in .env", "score": None}
        
    url = "https://api.sightengine.com/1.0/check.json"
    
    try:
        data = {
            'models': 'genai',
            'api_user': api_user.strip(),
            'api_secret': api_secret.strip()
        }
        with open(img_path, 'rb') as f:
            files = {'media': f}
            response = requests.post(url, files=files, data=data)
            
        json_res = response.json()
        
        if json_res.get("status") == "success":
            # Extract probability of AI generation
            score = 0.0
            if "type" in json_res and "ai_generated" in json_res["type"]:
                score = json_res["type"]["ai_generated"]
            return {"status": "success", "score": f"{score:.3f} (AI Gen)", "message": "Sightengine Analysis Complete"}
        else:
            err_msg = json_res.get("error", {}).get("message", "Unknown API error")
            return {"error": f"API Error: {err_msg}", "score": None}
            
    except Exception as e:
        return {"error": str(e), "score": None}
# ---------------------------------------------------------

# Import forensic report generation
sys.path.insert(0, str(Path(__file__).parent / "Report generation" / "core"))
from forensic_report_integration import streamlit_forensic_ui

st.set_page_config(page_title="TruFor Global Trace", layout="wide")
st.title("Analysis & Visualization")

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'output_file' not in st.session_state:
    st.session_state.output_file = None
if 'viz_output' not in st.session_state:
    st.session_state.viz_output = None
if 'img_path' not in st.session_state:
    st.session_state.img_path = None
if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'model_output' not in st.session_state:
    st.session_state.model_output = None

BASE_DIR = Path(os.getcwd()).absolute()
UI_OUTPUT = BASE_DIR / "temp_ui" / "output"
MODEL_PATH = BASE_DIR / "trained_models" / "trufor.pth.tar"

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Use a unique name to make it easier to find
    img_path = BASE_DIR / "temp_ui" / "input" / uploaded_file.name
    img_path.parent.mkdir(parents=True, exist_ok=True)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    analysis_method = st.radio(
        "Select Analysis Method",
        options=[
            "Fast Local ELA Heuristic (Instantly detects Photoshop/Overlays)", 
            "Deep Local Analysis (TruFor Model)"
        ],
        index=0,
        help="ELA is instant and catches most manual edits. TruFor is thorough but slow."
    )

    if st.button("Run Analysis & Trace File"):
        with st.spinner("Model/API is running..."):
            # Create output directory and define output file
            UI_OUTPUT.mkdir(parents=True, exist_ok=True)
            output_file = UI_OUTPUT / f"{img_path.stem}.npz"
            
            # --- API RESULTS DICTIONARY ---
            screenshot_detected = is_screenshot(img_path)
            api_results = {}
            if screenshot_detected:
                with st.spinner("Screenshot detected! Extracting screenshot properties..."):
                    api_results = call_forgery_api(img_path)

            if "TruFor" in analysis_method:
                # Execute Model
                test_cmd = [
                    sys.executable, str(BASE_DIR / "test.py"),
                    "-in", str(img_path),
                    "-out", str(output_file),
                    "-exp", "trufor_ph3",
                    "TEST.MODEL_FILE", str(MODEL_PATH)
                ]
                
                result = subprocess.run(test_cmd, capture_output=True, text=True, cwd=str(BASE_DIR))

                # --- THE TRACE LOGIC ---
                if output_file.exists():
                    
                    # Load the NPZ file
                    result_data = np.load(output_file)
                    
                    # Create visualization
                    with st.spinner("Generating visualization..."):
                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                        
                        if screenshot_detected:
                            api_score = api_results.get('score')
                            score_text = f"[API - SCREENSHOT] Score: {api_score}" if api_score is not None else "[API] " + api_results.get('error', 'Error')
                            fig.suptitle(score_text)
                        else:
                            fig.suptitle(f"[TRUFOR] Score: {result_data.get('score', 'N/A'):.3f}" if 'score' in result_data else "Analysis Results")
                        
                        for ax in axs:
                            ax.axis('off')
                        
                        # Image
                        axs[0].imshow(Image.open(img_path))
                        axs[0].set_title('Input Image')
                        
                        # Localization map
                        axs[1].imshow(result_data['map'], cmap='RdBu_r', clim=[0, 1])
                        axs[1].set_title('Localization Map')
                        
                        # Confidence map
                        if 'conf' in result_data:
                            axs[2].imshow(result_data['conf'], cmap='gray', clim=[0, 1])
                            axs[2].set_title('Confidence Map')
                        else:
                            axs[2].axis('off')
                        
                        plt.tight_layout()
                        
                        # Save visualization
                        viz_output = UI_OUTPUT / f"{img_path.stem}_visualization.png"
                        plt.savefig(viz_output, dpi=150, bbox_inches='tight')
                        
                        # Store in session state to persist across interactions
                        st.session_state.analysis_complete = True
                        st.session_state.output_file = output_file
                        st.session_state.viz_output = viz_output
                        st.session_state.img_path = img_path
                        st.session_state.fig = fig
                else:
                    st.error(f"Output file not found at: {output_file}")
                    if result.stderr:
                        st.error(f"**Error details:** {result.stderr}")
            elif "Fast Local ELA Heuristic" in analysis_method:
                import api_integrations
                st.info("Running Fast Local ELA to detect manual document tampering...")
                fast_api_result = api_integrations.analyze_document_tampering_fast_ela(str(img_path))
                
            if "TruFor" not in analysis_method:
                
                has_map = "map" in fast_api_result
                
                # Create Visualization for Fast API
                if has_map:
                     fig, axs = plt.subplots(1, 2, figsize=(14, 6))
                     ax1, ax2 = axs
                else:
                     fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
                
                # Title based on API results
                if "error" in fast_api_result:
                    title_text = f"API Error: {fast_api_result['error']}"
                    fig.suptitle(title_text, color='red', fontsize=16)
                    score_to_save = 0.0
                else:
                    score = fast_api_result.get('score', 0)
                    msg = fast_api_result.get('message', '')
                    title_text = f"Document Tampering Score: {score:.3f} | {msg}"
                    color = 'red' if score > 0.5 else 'green'
                    score_to_save = score
                    
                    if screenshot_detected and 'score' in api_results:
                         title_text += f"\n[Screenshot GenAI Score: {api_results['score']}]"
                         
                    fig.suptitle(title_text, color=color, fontsize=16)
                
                # Original Image
                ax1.imshow(Image.open(img_path))
                ax1.set_title("Original Document")
                ax1.axis('off')
                
                dummy_map = np.zeros((100, 100))
                
                # Heatmap
                if has_map:
                    ela_map = fast_api_result['map']
                    im = ax2.imshow(ela_map, cmap='magma') # Magma makes the edits glow brightly
                    ax2.set_title("ELA Tampering Heatmap")
                    ax2.axis('off')
                    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                    
                    # Normalize for NPZ downstream
                    dummy_map = ela_map
                    if np.max(dummy_map) > 1.0:
                        dummy_map = dummy_map / 255.0
                
                plt.tight_layout()
                
                viz_output = UI_OUTPUT / f"{img_path.stem}_visualization.png"
                plt.savefig(viz_output, dpi=150, bbox_inches='tight')
                
                # Save NPZ carrying the normalized map
                np.savez(output_file, score=score_to_save, map=dummy_map, conf=dummy_map)
                
                st.session_state.analysis_complete = True
                st.session_state.output_file = output_file
                st.session_state.viz_output = viz_output
                st.session_state.img_path = img_path
                st.session_state.fig = fig
    
    # Display visualization if analysis is complete
    if st.session_state.analysis_complete and st.session_state.fig is not None:
        st.pyplot(st.session_state.fig)
    
    # Display download section if analysis is complete
    if st.session_state.analysis_complete:
        st.subheader("Download Results")
        
        with open(st.session_state.viz_output, "rb") as f:
            st.download_button(
                label="Download Visualization",
                data=f.read(),
                file_name=st.session_state.viz_output.name,
                mime="image/png"
                )
        

        # ===== FORENSIC REPORT GENERATION =====
        st.markdown("---")
        st.subheader("Generate Forensic Report")
        
        # col_left, col_right = st.columns(2)
        
        # with col_left:
        #     api_provider = st.selectbox(
        #         "Analysis Provider",
        #         options=["local", "openai", "anthropic", "gemini", "groq"],
        #         key="forensic_provider",
        #         help="Local: Free, no API key\nOpenAI: GPT-4\nAnthropic: Claude-3\nGemini: Google AI\nGroq: Fast LLM"
        #     )
        
        # with col_right:
        #     if api_provider != "local":
        #         api_key = st.text_input(
        #             f"{api_provider.upper()} API Key",
        #             type="password",
        #             placeholder=f"Enter your {api_provider.upper()} API key",
        #             key=f"{api_provider}_key"
        #         )
        #     else:
        #         api_key = None
        
        # Generate report button
        if st.button("Generate Forensic Report", type="primary", use_container_width=True):
            # if api_provider != "local" and not api_key:
            #     st.error(f"Please enter your {api_provider.upper()} API key")
            # else:
                with st.spinner(f"Generating forensic report"):
                    try:
                        # Import here to avoid issues
                        from forensic_report_integration import generate_forensic_report_from_analysis
                        
                        # Generate report
                        report_output = UI_OUTPUT / f"{st.session_state.output_file.stem}_forensic_report.pdf"
                        
                        success, message = generate_forensic_report_from_analysis(
                            npz_file_path=str(st.session_state.output_file),
                            # api_key=api_key,
                            # api_provider=api_provider,
                            output_path=str(report_output),
                            image_path=str(st.session_state.img_path)
                        )
                        
                        if success:
                            # Download button
                            with open(report_output, "rb") as f:
                                st.download_button(
                                    label="Download Forensic Report (PDF)",
                                    data=f.read(),
                                    file_name=report_output.name,
                                    mime="application/pdf",
                                    use_container_width=True,
                                    type="primary"
                                )
                        else:
                            st.error(f"{message}")
                    
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        # st.info("Make sure all dependencies are installed: `pip install -r 'Report generation/core/requirements_forensic.txt'`")
#AUDIOS.                        
# audio_paths=[]#paths to files in input cache.
# audio_files=st.file_uploader('Upload Audio (max. 30s)',type=['wav','mp3'],accept_multiple_files=True)
# if audio_files:
#     #Require 2 input files.
#     if len(audio_files)!=2:
#         st.error('expect 2 audio valid clips as input.')
#     else:
#         #Save input audios.
#         for audio_file in audio_files:
#             audio_path=BASE_DIR/"temp_ui"/"input"/audio_file.name
    #         audio_path.parent.mkdir(parents=True,exist_ok=True)
    #         with open(audio_path,"wb")as f:
    #             f.write(audio_file.getbuffer())
    #         audio_paths.append(audio_path)
    # #Run analysis.
    # if st.button('Analyze Audio'):
    #     if len(audio_paths) < 2:
    #         st.error("Please upload 2 audio files before analyzing.")
    #     else:
    #         with st.spinner('Processing...'):
    #             from audioproc import interface as audio_proc
    #             audio_proc.IMG_DIR=BASE_DIR/'temp_ui'/'output'
    #             model,feature_extractor,best_thresh=audio_proc.init_model(BASE_DIR/'audioproc'/'checkpoints_new'/'best_model.pth')
    #             waveform_path=audio_proc.visualize_waveform_similarity(model,feature_extractor,audio_paths[0],audio_paths[1])
    #             spectrogram_path=audio_proc.visualize_spectrogram_similarity(model,feature_extractor,audio_paths[0],audio_paths[1])
    #             res=audio_proc.compute_similarity(model,feature_extractor,audio_paths[0],audio_paths[1],best_thresh)
                
    #             # Store in session state
    #             st.session_state.audio_analysis_complete = True
    #             st.session_state.audio_results = {
    #                 'res': res,
    #                 'waveform_path': waveform_path,
    #                 'spectrogram_path': spectrogram_path,
    #                 'audio_paths': [str(p) for p in audio_paths]
    #             }

    # # # Display results if complete
    # if st.session_state.audio_analysis_complete and st.session_state.audio_results:
    #     results = st.session_state.audio_results
    #     res = results['res']
    #     decision = 'Same Speaker' if res['decision'] else 'Different Speakers'
        
    #     st.markdown("---")
    #     st.subheader("Audio Analysis Results")
    #     st.text(f"Similarity Score: {res['similarity']:.4f}\nDecision: {decision}")
        
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.image(results['waveform_path'], caption="Waveform Similarity Saliency")
    #     with col2:
    #         st.image(results['spectrogram_path'], caption="Spectrogram Similarity Saliency")
        
    #     # PDF Report Generation
    #     st.markdown("---")
    #     if st.button("Generate Audio Forensic Report", key="gen_audio_report"):
    #         with st.spinner("Generating PDF report..."):
    #             from audio_report_generator import AudioForensicReportGenerator
    #             report_gen = AudioForensicReportGenerator()
    #             report_output = BASE_DIR / "temp_ui" / "output" / f"audio_forensic_report_{int(time.time())}.pdf"
                
    #             audio_filenames = [os.path.basename(p) for p in results['audio_paths']]
    #             success, msg = report_gen.create_pdf_report(
    #                 report_output,
    #                 res['similarity'],
    #                 decision,
    #                 results['waveform_path'],
    #                 results['spectrogram_path'],
    #                 audio_names=audio_filenames
    #             )
                
    #             if success:
    #                 st.success("Report generated!")
    #                 with open(report_output, "rb") as f:
    #                     st.download_button(
    #                         label="Download Audio Forensic Report (PDF)",
    #                         data=f.read(),
    #                         file_name=report_output.name,
    #                         mime="application/pdf",
    #                         use_container_width=True,
    #                         type="primary"
    #                     )
    #             else:
                    # st.error(f"Failed to generate report: {msg}")
