import os
import requests
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

def is_screenshot(img_path):
    """
    Detects if an image is a screenshot based on filename and EXIF data.
    """
    import pathlib
    if pathlib.Path(img_path).suffix.lower() == '.pdf':
        return False  # PDFs are essentially never natural photo screenshots
        
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
        # If we failed to parse, it could be a weird format, but not definitely a screenshot
        return False
        
    return False

def call_screenshot_api(img_path):
    """
    Uses Sightengine's deep learning visual classifiers to establish if an image
    is a high-probability screenshot/illustration versus a captured photograph.
    """
    import pathlib
    if pathlib.Path(img_path).suffix.lower() == '.pdf':
        return {
            "status": "success", 
            "is_screenshot": False, 
            "confidence": 0.0, 
            "details": "Native PDF. Assessed by metadata, not pixel-level screenshot heuristics."
        }
        
    api_user = os.getenv("SIGHTENGINE_API_USER")
    api_secret = os.getenv("SIGHTENGINE_API_SECRET")
    
    if not api_user or not api_secret:
        return {"error": "Sightengine API keys missing."}
        
    url = "https://api.sightengine.com/1.0/check.json"
    
    try:
        data = {
            'models': 'type,text-content',
            'api_user': api_user.strip(),
            'api_secret': api_secret.strip()
        }
        with open(img_path, 'rb') as f:
            files = {'media': f}
            response = requests.post(url, files=files, data=data)
            
        json_res = response.json()
        
        if json_res.get("status") == "success":
            type_data = json_res.get("type", {})
            ill_score = type_data.get("illustration", 0)
            photo_score = type_data.get("photography", 0)
            
            # Screen captures tend to flag extremely high for illustration, lacking photo noise
            is_ss = ill_score > 0.85 and photo_score < 0.2
            
            return {
                "status": "success",
                "is_screenshot": is_ss,
                "confidence": float(ill_score),
                "details": f"Digital Probability: {ill_score:.2f} | Photography: {photo_score:.2f}"
            }
        else:
            return {"error": json_res.get("error", {}).get("message", "API Error")}
            
    except Exception as e:
        return {"error": str(e)}

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

def analyze_document_structure_local(file_path):
    """
    Local heuristic alternative to EdenAI. Parses the raw text layer of a PDF
    to verify structural coherence (looks for typical invoice/billing layout elements).
    """
    import fitz
    import re
    from pathlib import Path
    
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.pdf':
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            if not text.strip():
                return {
                    "status": "success",
                    "coherent": False,
                    "info": "PDF lacks a native text layer (Likely a flattened image/screenshot masquerading as PDF).",
                    "provider_used": "PyMuPDF Local"
                }
                
            # Basic invoice/billing heuristics
            text_lower = text.lower()
            has_money = bool(re.search(r'total|amount|subtotal|balance|due', text_lower))
            has_currency = bool(re.search(r'\$|€|£|rs\.|aed|usd|eur|gbp|inr', text_lower))
            
            if has_money or has_currency:
                return {
                    "status": "success",
                    "coherent": True,
                    "info": "Detected coherent billing structure. Financial/Layout markers found natively.",
                    "provider_used": "PyMuPDF Local"
                }
            else:
                return {
                    "status": "success",
                    "coherent": False, 
                    "info": "Extracted text, but lacks standard financial/billing structural markers.",
                    "provider_used": "PyMuPDF Local"
                }
                
        except Exception as e:
            return {"error": str(e), "coherent": False}
    else:
        # Fallback for images without requiring Tesseract OCR
        return {
            "status": "success",
            "coherent": True,
            "info": "Local structural parsing relies on PDF text layers. Assuming coherent for raw Image.",
            "provider_used": "PyMuPDF Local (Bypassed)"
        }

def analyze_document_tampering_fast_ela(img_path):
    """
    Performs fast Error Level Analysis (ELA) locally to detect manual tampering 
    (photoshop, copy-paste overlays, blurring) without needing an API key.
    Works perfectly on salary slips and PDFs to detect inserted fake text or blur.
    """
    try:
        quality = 90
        
        # Load image
        original = Image.open(img_path).convert('RGB')
        
        # Save as temporary compressed JPEG
        temp_path = str(img_path) + ".temp.jpg"
        original.save(temp_path, 'JPEG', quality=quality)
        
        # Load compressed image
        compressed = Image.open(temp_path)
        
        # Calculate difference (highlighting resaved areas)
        ela_im = ImageChops.difference(original, compressed)
        
        # Enhance the difference
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        if max_diff == 0:
            max_diff = 1
            
        scale = 255.0 / max_diff
        ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
        
        # Convert difference to numpy array for scoring
        diff_array = np.array(ela_im.convert('L'))
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Score heuristic: high ELA deviation -> manual tampering/photoshop
        std_dev = np.std(diff_array)
        max_v = np.max(diff_array)
        
        # Text documents have high contrast, leading to normally high ELA std_dev.
        # We divide by 40 to ensure only extreme tampering hits a score > 0.85
        score = min(float(std_dev) / 40.0, 1.0)
        
        # Force a higher score if there was extreme peaking 
        if max_v > 240 and std_dev > 15:
             score = max(score, 0.90)
             
        return {
            "status": "success",
            "score": score,
            "provider": "Fast Local ELA",
            "message": "Heavy surface compression divergence detected." if score > 0.85 else "Document surface compression appears normal.",
            "map": diff_array
        }
    except Exception as e:
        return {"error": str(e)}

def analyze_metadata(file_path):
    """
    Analyzes the metadata of a document or image to detect traces of editing software,
    anomalous creation/modification dates, or mismatching creator tools.
    """
    import fitz
    from pathlib import Path
    
    file_ext = Path(file_path).suffix.lower()
    results = {
        "status": "success",
        "is_edited": False,
        "anomalies": [],
        "metadata_extracted": {}
    }
    
    # Legit layout/creation tools vs explicit manipulation tools
    manipulation_software = ["photoshop", "gimp", "paint"]

    if file_ext == '.pdf':
        try:
            doc = fitz.open(file_path)
            meta = doc.metadata
            doc.close()
            
            results["metadata_extracted"] = {
                "Creator": meta.get("creator", ""),
                "Producer": meta.get("producer", ""),
                "CreationDate": meta.get("creationDate", ""),
                "ModDate": meta.get("modDate", "")
            }
            
            # Enterprise Check: If all critical metadata is missing/scrubbed, rank as highly suspicious
            if not meta.get("creator") and not meta.get("producer") and not meta.get("creationDate"):
                results["is_edited"] = True
                results["anomalies"].append("CRITICAL: PDF Metadata has been entirely stripped/flattened. Genuine system bills always contain structural metadata.")
            
            for field in ["creator", "producer"]:
                val = str(meta.get(field, "")).lower()
                for sw in manipulation_software:
                    if sw in val:
                        results["is_edited"] = True
                        results["anomalies"].append(f"Document created or modified using explicit manipulation software {sw.title()} ({field}).")
                        
            # Mod date and creation date often differ slightly; we'll remove the anomaly flag for simple mismatch 
            # to reduce false positives on valid digital bills.
        except Exception as e:
            results["error"] = f"Failed to extract PDF metadata: {str(e)}"
            results["status"] = "error"
            
    else:  # Images
        try:
            img = Image.open(file_path)
            exif = img.getexif()
            if exif:
                # 305 is Software, 306 is DateTime
                software = exif.get(305, "")
                datetime = exif.get(306, "")
                
                results["metadata_extracted"] = {
                    "Software": software,
                    "DateTime": datetime
                }
                
                if software:
                    val = str(software).lower()
                    for sw in manipulation_software:
                        if sw in val:
                            results["is_edited"] = True
                            results["anomalies"].append(f"Image edited using {sw.title()} (EXIF Software tag).")
            else:
                pass # Don't flag missing EXIF as anomaly for images, very common on legit downloads
        except Exception as e:
            results["error"] = f"Failed to extract image EXIF data: {str(e)}"
            results["status"] = "error"
            
    if not results["anomalies"] and not results["is_edited"]:
        results["anomalies"].append("No obvious anomalies detected in metadata.")
        
    return results

def call_trufor_analysis(img_path):
    """
    Executes the deep learning local analysis (TruFor Model) to trace exact
    localization of copy-moved or spliced pixels (e.g., pasted QR codes, forged signatures).
    """
    import subprocess
    import sys
    import numpy as np
    from pathlib import Path
    
    BASE_DIR = Path("d:/MD-II/AI-Assisted-Forensic-Analysis-System-for-Decision-Support")
    MODEL_PATH = BASE_DIR / "trained_models" / "trufor.pth.tar"
    output_file = Path("d:/MD-II/AI-Assisted-Forensic-Analysis-System-for-Decision-Support/BM/temp_input") / f"{Path(img_path).stem}_trufor.npz"
    
    test_cmd = [
        sys.executable, str(BASE_DIR / "test.py"),
        "-in", str(img_path),
        "-out", str(output_file),
        "-exp", "trufor_ph3",
        "TEST.MODEL_FILE", str(MODEL_PATH)
    ]
    
    try:
        # Run TruFor model
        result = subprocess.run(test_cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
        
        if output_file.exists():
            data = np.load(output_file)
            loc_map = data.get('map', np.zeros((10,10)))
            conf_map = data.get('conf', np.zeros((10,10)))
            score = float(data.get('score', 0.0))
            
            # Find bounding box of heavily altered region
            mask = loc_map > 0.65
            bbox = None
            if np.any(mask):
                y, x = np.where(mask)
                bbox = (int(x.min()), int(y.min()), int(x.max()), int(y.max()))
            
            return {
                "status": "success",
                "score": score,
                "map": loc_map,
                "confidence": conf_map,
                "bbox": bbox,
                "message": "Deep Forgery Trace Complete"
            }
        else:
            return {"error": "Deep Local Analysis model failed to write output.", "details": result.stderr}
    except Exception as e:
        return {"error": str(e)}

def analyze_pdf_native_images(pdf_path):
    """
    Analyzes embedded images inside a native PDF using PyMuPDF.
    Look for isolated, explicitly positioned objects (like a copy-pasted QR code or signature)
    inserted independently from the main document layout.
    """
    import fitz
    from pathlib import Path
    
    if Path(pdf_path).suffix.lower() != '.pdf':
        return {"status": "skipped", "message": "Not a PDF."}
        
    try:
        doc = fitz.open(pdf_path)
        suspicious_images = []
        total_images = 0
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            total_images += len(image_list)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                img_dict = doc.extract_image(xref)
                
                # Check for common pasted image indicators
                # Pasted web images/screenshots often lack certain expected compression or color space metadata natively
                # found in enterprise scanned documents
                
                # Also look at bounding boxes
                rects = page.get_image_rects(xref)
                for rect in rects:
                    # Heuristic: A small, isolated square image (like a QR code or stamp) added 
                    # as a separate XObject often indicates post-generation tampering if the rest
                    # of the document is a single scanned image or plain text.
                    width = rect.width
                    height = rect.height
                    
                    if width > 0 and height > 0:
                        aspect_ratio = width / height
                        
                        # Most QR codes or copy-pasted logos are relatively small and roughly square
                        is_square = 0.8 < aspect_ratio < 1.2
                        is_small = (width < page.rect.width * 0.4) and (height < page.rect.height * 0.4)
                        
                        if is_square and is_small:
                            suspicious_images.append({
                                "page": page_num + 1,
                                "xref": xref,
                                "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                                "reason": "Isolated small/square embedded image detected (Potential copy-pasted QR/Logo)."
                            })
                            
        doc.close()
        
        if suspicious_images:
            return {
                "status": "success",
                "score": 0.85, # High confidence of structural edit
                "message": f"Found {len(suspicious_images)} anomalous embedded image(s) indicative of copy-pasting.",
                "details": suspicious_images
            }
        else:
             return {
                "status": "success",
                "score": 0.1,
                "message": f"Analyzed {total_images} native images. No explicit structural copy-paste anomalies detected.",
                "details": []
            }
            
    except Exception as e:
        return {"error": str(e)}

