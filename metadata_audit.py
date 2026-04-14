import os
import time
from pypdf import PdfReader
from PIL import Image
from PIL.ExifTags import TAGS
from fpdf import FPDF

RED_FLAG_SOFTWARE = [
    "photoshop", "illustrator", "nitro", "smallpdf", "ilovepdf", 
    "pdfescape", "inkscape", "gimp", "canvas", "expert pdf"
]

def sanitize_text(text):
    """
    Replaces non-Latin-1 characters to avoid FPDF encoding errors.
    """
    if not text:
        return ""
    # Map common unicode to latin-1 equivalents or just replace
    return "".join(c if ord(c) < 256 else "?" for c in str(text))

def audit_pdf_metadata(pdf_path):
    """
    Extracts metadata from a PDF and checks for red flags.
    """
    try:
        reader = PdfReader(pdf_path)
        meta = reader.metadata
        
        # High-level metadata
        info = {k[1:] if k.startswith('/') else k: v for k, v in meta.items()}
        
        # Check for red flags
        red_flags = []
        for key, val in info.items():
            if isinstance(val, str):
                v_low = val.lower()
                for flag in RED_FLAG_SOFTWARE:
                    if flag in v_low:
                        red_flags.append(f"Suspicious {key}: {val}")
        
        # Check for "XMP" history
        xmp = reader.xmp_metadata
        if xmp:
            # Add raw XMP text to info so the user sees everything
            info["Raw_XMP_Metadata"] = str(xmp)
            xmp_str = str(xmp).lower()
            for flag in RED_FLAG_SOFTWARE:
                if flag in xmp_str:
                    red_flags.append(f"XMP History Flag: {flag}")

        return {
            "info": info,
            "red_flags": red_flags,
            "is_tampered_hint": len(red_flags) > 0
        }
    except Exception as e:
        return {"error": str(e)}

def audit_image_metadata(img_path):
    """
    Extracts EXIF metadata from an image.
    """
    try:
        image = Image.open(img_path)
        exifdata = image.getexif()
        
        info = {}
        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            if isinstance(data, bytes):
                data = data.decode()
            info[tag] = str(data)
            
        red_flags = []
        software = info.get("Software", "").lower()
        for flag in RED_FLAG_SOFTWARE:
            if flag in software:
                red_flags.append(f"Suspicious Software: {info['Software']}")
                
        return {
            "info": info,
            "red_flags": red_flags,
            "is_tampered_hint": len(red_flags) > 0
        }
    except Exception as e:
        return {"error": str(e)}

def finalize_pdf_report(ai_data, local_forensics, metadata_results, model_name="Unknown"):
    """
    Generates a professional PDF forensic report.
    """
    ai_forgery = ai_data.get("ai_forgery_data", {})
    provider = ai_data.get("provider_company", "Unknown")
    texture_score = local_forensics.get('texture_variance_score', 0)
    meta_hint = metadata_results.get('is_tampered_hint', False)
    
    # Verdict Logic
    verdict = "CLEAN"
    v_color = (0, 128, 0) # Green
    if meta_hint or texture_score > 2.5:
        verdict = "SUSPICIOUS - POTENTIAL TAMPERING"
        v_color = (255, 140, 0) # Orange
    if ai_forgery.get("score", 0) > 0.6:
        verdict = "HIGH RISK - FORGERY DETECTED"
        v_color = (200, 0, 0) # Red

    pdf = FPDF()
    pdf.add_page()
    
    # 1. Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, sanitize_text("FORENSIC AUDIT REPORT"), ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 5, sanitize_text(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"), ln=True, align='C')
    pdf.ln(10)
    
    # 2. Executive Summary
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, sanitize_text(" 1. EXECUTIVE SUMMARY"), ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(40, 10, sanitize_text("Document Provider:"))
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, sanitize_text(provider), ln=True)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(40, 10, sanitize_text("Overall Verdict:"))
    pdf.set_text_color(*v_color)
    pdf.cell(0, 10, sanitize_text(verdict), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    # 3. AI Analysis
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, sanitize_text(" 2. AI VISUAL ANALYSIS"), ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(40, 7, sanitize_text("Model:"))
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 7, sanitize_text(model_name), ln=True)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(40, 7, sanitize_text("AI Confidence:"))
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 7, sanitize_text(f"{100 - (ai_forgery.get('score', 0) * 100):.1f}%"), ln=True)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 7, sanitize_text("Reasoning:"), ln=True)
    pdf.set_font("Arial", '', 9)
    pdf.multi_cell(0, 5, sanitize_text(ai_forgery.get('reasoning', 'No reasoning provided.')))
    pdf.ln(5)
    
    # 4. Forensic Scores Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, sanitize_text(" 3. REGIONAL FORENSIC SCORES"), ln=True, fill=True)
    pdf.ln(2)
    
    # Table Header
    pdf.set_font("Arial", 'B', 9)
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(45, 8, sanitize_text("Region"), 1, 0, 'C', True)
    pdf.cell(35, 8, sanitize_text("Z-Score (Noise)"), 1, 0, 'C', True)
    pdf.cell(35, 8, sanitize_text("Pixel Density"), 1, 0, 'C', True)
    pdf.cell(35, 8, sanitize_text("Diff Score"), 1, 0, 'C', True)
    pdf.cell(35, 8, sanitize_text("Status"), 1, 1, 'C', True)
    
    pdf.set_font("Arial", '', 8)
    for name, data in local_forensics.get('regions', {}).items():
        pdf.cell(45, 7, sanitize_text(name), 1)
        pdf.cell(35, 7, sanitize_text(f"{data.get('z_score', 0):.4f}"), 1, 0, 'C')
        pdf.cell(35, 7, sanitize_text(f"{data.get('density', 0):.4f}"), 1, 0, 'C')
        pdf.cell(35, 7, sanitize_text(f"{data.get('diff_score', 0):.4f}"), 1, 0, 'C')
        status = "SUSPICIOUS" if data.get('is_suspicious') else "CONSISTENT"
        if status == "SUSPICIOUS": pdf.set_text_color(200, 0, 0)
        pdf.cell(35, 7, sanitize_text(status), 1, 1, 'C')
        pdf.set_text_color(0, 0, 0)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 7, sanitize_text(f"Global Texture Variance: {texture_score:.4f}"), ln=True)
    pdf.ln(5)

    # 5. Metadata Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, sanitize_text(" 4. METADATA PROPERTIES"), ln=True, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(60, 8, sanitize_text("Property"), 1, 0, 'C', True)
    pdf.cell(125, 8, sanitize_text("Value"), 1, 1, 'C', True)
    
    pdf.set_font("Arial", '', 8)
    for k, v in metadata_results.get('info', {}).items():
        v_str = str(v).replace("\n", " ")
        if len(v_str) > 80: v_str = v_str[:77] + "..."
        
        pdf.cell(60, 6, sanitize_text(k), 1)
        pdf.cell(125, 6, sanitize_text(v_str), 1, 1)

    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, sanitize_text("DISCLAIMER: This report is generated by an automated forensic system. High-risk results should be verified by a human forensic expert."))
    
    return bytes(pdf.output())

def finalize_report(ai_data, local_forensics, metadata_results, model_name="Unknown"):
    """
    Combines all findings into a professional, structured markdown report.
    """
    ai_forgery = ai_data.get("ai_forgery_data", {})
    provider = ai_data.get("provider_company", "Unknown")
    texture_score = local_forensics.get('texture_variance_score', 0)
    meta_hint = metadata_results.get('is_tampered_hint', False)
    
    # Verdict Logic
    verdict = "✅ CLEAN"
    if meta_hint or texture_score > 2.5:
        verdict = "⚠️ SUSPICIOUS - POTENTIAL TAMPERING"
    if ai_forgery.get("score", 0) > 0.6:
        verdict = "🚨 HIGH RISK - FORGERY DETECTED"

    # 1. Header
    report = f"""# Forensic Audit Report: {provider}
Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
Overall Verdict: **{verdict}**

---

## 1. Executive Summary
This report provides a multi-layered forensic analysis of the uploaded document, combining AI visual reasoning, metadata auditing, and pixel-level consistency checks.

## 2. AI Visual Perception
- **Model Used**: `{model_name}`
- **Integrity Confidence**: {100 - (ai_forgery.get('score', 0) * 100):.1f}%
- **AI Reasoning**: {ai_forgery.get('reasoning', 'No detailed reasoning provided.')}
- **Suspicious Regions Identified by AI**: {", ".join(ai_forgery.get('suspicious_regions', [])) if ai_forgery.get('suspicious_regions') else "None"}

## 3. Forensic Region Analysis
| Region Name | Z-Score (Noise) | Density | Diff Score | Status |
| :--- | :--- | :--- | :--- | :--- |
"""
    # Region table
    for name, data in local_forensics.get('regions', {}).items():
        status = "🚩 SUSPICIOUS" if data.get('is_suspicious') else "✅ CONSISTENT"
        report += f"| {name} | {data.get('z_score', 0):.4f} | {data.get('density', 0):.4f} | {data.get('diff_score', 0):.4f} | {status} |\n"

    report += f"""
**Global Texture Variance**: {texture_score:.4f}
*High variance suggests background manipulation or inconsistent compression artifacts.*

## 4. Metadata Audit
"""
    # Metadata Red Flags
    if metadata_results.get('red_flags'):
        report += "### 🚩 Metadata Red Flags\n"
        for flag in metadata_results['red_flags']:
            report += f"- {flag}\n"
        report += "\n"

    # Full Metadata Table
    report += "### Complete Metadata Properties\n"
    report += "| Property | Value |\n| :--- | :--- |\n"
    for k, v in metadata_results.get('info', {}).items():
        v_str = str(v).replace("\n", " ")
        if len(v_str) > 100: v_str = v_str[:97] + "..."
        report += f"| {k} | {v_str} |\n"

    report += """
---
*Disclaimer: This report is generated by an automated forensic system. High-risk results should be verified by a human forensic expert.*
"""
    return report
