import pytest
import numpy as np
import os
import sys
from pathlib import Path

# --- Path Configuration ---
# Current file: Application/tests/test.py
# Target: Application/Report generation/core/
current_dir = Path(__file__).resolve().parent
application_root = current_dir.parent
forensic_core_path = application_root / "Report generation" / "core"

# Add the core directory to sys.path so Python can find the modules
sys.path.insert(0, str(forensic_core_path))

# Now import the forensic modules
from generate_forensic_report import ForensicReportGenerator
from forensic_report_integration import generate_forensic_report_from_analysis

# --- Fixtures ---
@pytest.fixture
def sample_npz(tmp_path):
    """Creates a mock .npz file that matches TruFor output structure."""
    npz_path = tmp_path / "test_analysis.npz"
    data = {
        'map': np.random.rand(100, 100),
        'conf': np.random.rand(100, 100),
        'score': np.array(0.75),
        'imgsize': np.array([100, 100])
    }
    np.savez(npz_path, **data)
    return str(npz_path)

# --- Unit Tests ---

def test_local_analysis_logic():
    """Verify the generator uses local rule-based logic when no API is provided."""
    gen = ForensicReportGenerator()
    
    # Test high tampering score (Expected: Very low confidence)
    gen.analysis_results = {'score': 0.85}
    analysis = gen._generate_local_analysis()
    assert "Very low confidence" in analysis
    
    # Test authentic score (Expected: Very high confidence)
    gen.analysis_results = {'score': 0.1}
    analysis = gen._generate_local_analysis()
    assert "Very high confidence" in analysis

def test_load_analysis_results_valid(sample_npz):
    """Ensure valid .npz files load correctly into the class."""
    gen = ForensicReportGenerator()
    success = gen.load_analysis_results(sample_npz)
    assert success is True
    assert gen.analysis_results['score'] == 0.75

def test_integration_wrapper_success(sample_npz, tmp_path):
    """Test the full report generation flow using the integration layer."""
    output_pdf = tmp_path / "report.pdf"
    
    success, message = generate_forensic_report_from_analysis(
        npz_file_path=sample_npz,
        output_path=str(output_pdf)
    )
    
    assert success is True
    assert os.path.exists(output_pdf)

def test_missing_npz_file():
    """Test behavior when the .npz file is missing."""
    success, message = generate_forensic_report_from_analysis(
        npz_file_path="non_existent.npz",
        output_path="failed_report.pdf"
    )
    assert success is False
    assert "File not found" in message

def test_report_generation_without_image(sample_npz, tmp_path):
    """Verify report generation still works even if the original image path is missing."""
    output_pdf = tmp_path / "no_image_report.pdf"
    gen = ForensicReportGenerator()
    gen.load_analysis_results(sample_npz)
    
    # Explicitly ensure image_path is None
    gen.image_path = None
    success = gen.create_pdf_report(str(output_pdf))
    
    assert success is True
    assert os.path.exists(output_pdf)

def test_score_threshold_boundaries():
    """Test the local analysis text for specific boundary scores."""
    gen = ForensicReportGenerator(api_provider="local")
    
    # Test boundary for 'Moderate confidence' (Score 0.5)
    gen.analysis_results = {'score': 0.5}
    analysis = gen._generate_local_analysis()
    assert "Moderate confidence" in analysis
    
    # Test boundary for 'Low confidence' (Score 0.7)
    gen.analysis_results = {'score': 0.7}
    analysis = gen._generate_local_analysis()
    assert "Low confidence" in analysis

def test_heatmap_generation_logic(sample_npz, tmp_path):
    """Verify that heatmap images are created during the PDF process."""
    gen = ForensicReportGenerator()
    gen.load_analysis_results(sample_npz)
    
    # Test internal heatmap creator
    heatmap_path = tmp_path / "test_heat.png"
    result_path = gen._create_heatmap_image(
        gen.analysis_results['map'], 
        str(heatmap_path), 
        title="Test Map"
    )
    
    assert result_path is not None
    assert os.path.exists(heatmap_path)