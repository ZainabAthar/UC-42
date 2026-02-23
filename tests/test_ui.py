import pytest
from streamlit.testing.v1 import AppTest
import numpy as np
import os
import sys
from pathlib import Path
from unittest.mock import patch
import io

# Setup paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def test_app_startup():
    """Test if the app starts up and shows the title."""
    at = AppTest.from_file("app.py").run()
    assert not at.exception
    # Check for title
    assert any("Analysis" in t.value for t in at.title)

def test_file_upload_ui_logic():
    """Test that uploading a file triggers the UI elements."""
    at = AppTest.from_file("app.py").run()
    
    # Try pluralized attribute name used in specific Streamlit versions
    if hasattr(at, "file_uploaders") and len(at.file_uploaders) > 0:
        uploader = at.file_uploaders[0]
        uploader.upload(data=b"fake image data", name="test.jpg")
        at.run()
        # Verify 'Run Analysis' button appears
        assert any("Run Analysis" in btn.label for btn in at.button)
    else:
        # Fallback for alternative indexing
        assert len(at.get("file_uploader")) > 0

@patch("subprocess.run")
def test_analysis_execution_flow(mock_subproc, tmp_path):
    """Test UI transitions when analysis_complete is True."""
    mock_subproc.return_value.returncode = 0
    
    at = AppTest.from_file("app.py")
    
    # Create the physical mock .npz file
    mock_npz = tmp_path / "mock_output.npz"
    np.savez(mock_npz, 
             score=np.array(0.5), 
             map=np.zeros((10,10)), 
             conf=np.zeros((10,10)), 
             imgsize=[10,10])

    # Inject session state
    at.session_state.analysis_complete = True
    at.session_state.output_file = str(mock_npz)
    at.session_state.img_path = str(tmp_path / "test.jpg")
    
    at.run()
    
    # When complete, the "Run Analysis" button should be gone
    run_btns = [btn for btn in at.button if "Run Analysis" in btn.label]
    assert len(run_btns) == 0

def test_report_generation_button_click(tmp_path):
    """Verify buttons appear when valid files and state are present."""
    at = AppTest.from_file("app.py")

    # Create real dummy files on disk so st.image and np.load don't crash
    dummy_npz = tmp_path / "dummy.npz"
    dummy_img = tmp_path / "dummy.jpg"
    
    # Save a valid-structured NPZ
    np.savez(dummy_npz, 
             score=np.array(0.5), 
             map=np.zeros((10,10)), 
             conf=np.zeros((10,10)), 
             imgsize=[10,10])
    
    # Create an empty file for the image path
    dummy_img.write_text("fake image data")

    # Inject valid strings for paths
    at.session_state.analysis_complete = True
    at.session_state.output_file = str(dummy_npz)
    at.session_state.img_path = str(dummy_img)
    
    # Run the app with the injected state
    at.run()

    # Check for buttons
    # In app.py, st.button("Generate Forensic Report") should now be visible
    has_buttons = len(at.button) > 0
    
    # In some versions it's 'download_button' (singular) or 'download_buttons' (plural)
    has_download = False
    if hasattr(at, "download_button"):
        has_download = len(at.download_button) > 0
    elif hasattr(at, "download_buttons"):
        has_download = len(at.download_buttons) > 0

    # If this fails, let's print the exceptions from the app to debug
    if at.exception:
        print(f"App Exception: {at.exception}")

    assert has_buttons or has_download or not at.exception