"""
Streamlit Integration Module for Forensic Report Generator
Provides simple wrapper functions for easy integration into Streamlit apps
"""

import os
import streamlit as st
from pathlib import Path
from generate_forensic_report import ForensicReportGenerator


def get_api_provider_options():
    """Get list of available API providers"""
    return ["local", "openai", "anthropic", "gemini", "groq"]


def validate_api_key(provider: str, api_key: str) -> tuple[bool, str]:
    """
    Validate API key format
    
    Args:
        provider: LLM provider name
        api_key: API key to validate
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if provider == "local":
        return True, "Local provider - no key needed"
    
    if not api_key or len(api_key.strip()) < 10:
        return False, f"{provider.upper()}: API key is required and should be at least 10 characters"
    
    if provider == "openai" and not api_key.startswith('sk-'):
        return False, "OpenAI keys typically start with 'sk-'. Please verify your key."
    
    if provider == "anthropic" and not api_key.startswith('sk-ant-'):
        return False, "Anthropic keys typically start with 'sk-ant-'. Please verify your key."
    
    if provider == "gemini" and len(api_key) < 20:
        return False, "Gemini keys are typically longer. Please verify your key."
    
    if provider == "groq" and not api_key.startswith('gsk_'):
        return False, "Groq keys typically start with 'gsk_'. Please verify your key."
    
    return True, f"{provider.upper()}: Key format looks good"


def generate_forensic_report_from_analysis(
    npz_file_path: str,
    api_key: str = None,
    api_provider: str = "local",
    output_path: str = "forensic_report.pdf",
    image_path: str = None
) -> tuple[bool, str]:
    """
    Generate a forensic report from TruFor analysis results
    
    Args:
        npz_file_path: Path to the .npz file from TruFor analysis
        api_key: API key for LLM provider (optional for local provider)
        api_provider: LLM provider: "local", "openai", "anthropic", "gemini", "groq"
        output_path: Where to save the report
        image_path: Optional path to the original image
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Validate inputs
        if not Path(npz_file_path).exists():
            return False, f"Error: File not found: {npz_file_path}"
        
        # Validate API key if needed
        if api_provider != "local":
            is_valid, msg = validate_api_key(api_provider, api_key)
            if not is_valid:
                return False, f"API Key Error: {msg}"
        
        # Create generator
        generator = ForensicReportGenerator(
            api_key=api_key,
            api_provider=api_provider
        )
        
        # Load analysis results with image path
        if not generator.load_analysis_results(npz_file_path, image_path=image_path):
            return False, "Error: Could not load analysis results from .npz file"
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        print(f"📁 Output directory: {output_dir}")
        print(f"📁 Creating directory: {output_dir.mkdir(parents=True, exist_ok=True)}")
        
        # Generate report
        if not generator.create_pdf_report(output_path):
            return False, "Error: Report generation failed. Check logs."
        
        return True, f"✅ Success! Report saved"
        
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def generate_report_from_npz_file(npz_path: str, output_path: str, api_provider: str = "local", api_key: str = None, image_path: str = None) -> tuple[bool, str, str]:
    """
    Quick function to generate report from .npz file
    
    Args:
        npz_path: Path to .npz analysis file
        output_path: Where to save the PDF report
        api_provider: LLM provider to use
        api_key: API key (optional for local)
        image_path: Optional path to original image
        
    Returns:
        Tuple of (success: bool, message: str, output_file: str)
    """
    success, message = generate_forensic_report_from_analysis(
        npz_file_path=npz_path,
        api_key=api_key,
        api_provider=api_provider,
        output_path=output_path,
        image_path=image_path
    )
    return success, message, output_path if success else ""


# Streamlit UI Components (for direct import into Streamlit apps)
def streamlit_forensic_ui(npz_file_path: str = None, image_path: str = None, output_dir: str = "./reports"):
    """
    Add forensic report generation UI to a Streamlit app
    
    Usage (with file upload):
        from Report_generation.core.forensic_report_integration import streamlit_forensic_ui
        streamlit_forensic_ui()
    
    Usage (with pre-loaded .npz file and image):
        streamlit_forensic_ui(npz_file_path="path/to/analysis.npz", image_path="path/to/original.jpg")
    
    Args:
        npz_file_path: Optional path to .npz file (if None, shows file uploader)
        image_path: Optional path to original image
        output_dir: Directory to save reports
    """
    st.header("📊 Generate Forensic Report")
    
    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine input source
    if npz_file_path and Path(npz_file_path).exists():
        # Use provided file
        working_npz = npz_file_path
        file_name = Path(npz_file_path).name
        using_provided_file = True
    else:
        # Show file uploader
        uploaded_file = st.file_uploader("Upload TruFor .npz analysis results", type=['npz'])
        if not uploaded_file:
            st.info("📁 Please upload a .npz file from TruFor analysis")
            return
        
        # Save uploaded file temporarily
        temp_dir = Path(output_dir) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        working_npz = str(temp_dir / uploaded_file.name)
        with open(working_npz, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        file_name = uploaded_file.name
        using_provided_file = False
    
    # Provider selection
    col1, col2 = st.columns(2)
    with col1:
        api_provider = st.selectbox(
            "Analysis Provider",
            options=get_api_provider_options(),
            help="Local: free, no API key\nOpenAI: GPT-4 analysis\nAnthropic: Claude-3 analysis\nGemini: Google AI\nGroq: Fast LLM"
        )
    
    with col2:
        if api_provider != "local":
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder=f"Enter {api_provider.upper()} API key",
                help=f"Get from: {_get_provider_url(api_provider)}"
            )
        else:
            api_key = None
            st.info("✅ Local analysis - no API key needed")
    
    # Generate button
    if st.button("🚀 Generate Forensic Report", type="primary", use_container_width=True):
        # Validate API key if needed
        if api_provider != "local" and not api_key:
            st.error(f"❌ Please enter your {api_provider.upper()} API key")
            return
        
        with st.spinner(f"🔄 Generating report using {api_provider.upper()}..."):
            report_name = f"forensic_report_{Path(file_name).stem}.pdf"
            output_file = str(Path(output_dir) / report_name)
            
            success, message = generate_forensic_report_from_analysis(
                npz_file_path=working_npz,
                api_key=api_key,
                api_provider=api_provider,
                output_path=output_file,
                image_path=image_path
            )
            
            if success:
                st.success(f"✅ {message}")
                st.balloons()
                
                # Download button
                with open(output_file, 'rb') as f:
                    st.download_button(
                        label="📥 Download Forensic Report (PDF)",
                        data=f.read(),
                        file_name=report_name,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )
            else:
                st.error(f"❌ {message}")


def _get_provider_url(provider: str) -> str:
    """Get API key URL for provider"""
    urls = {
        "openai": "https://platform.openai.com/api-keys",
        "anthropic": "https://console.anthropic.com/",
        "gemini": "https://ai.google.dev/",
        "groq": "https://console.groq.com/"
    }
    return urls.get(provider, "")
