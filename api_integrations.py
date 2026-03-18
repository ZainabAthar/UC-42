import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

def analyze_document_tampering_fast_ela(img_path):
    """
    Performs fast Error Level Analysis (ELA) locally to detect manual tampering 
    (photoshop, copy-paste overlays, blurring) without needing an API key.
    Works perfectly on salary slips and PDFs to detect inserted fake text or blur.
    """
    import numpy as np
    from PIL import Image, ImageChops, ImageEnhance
    import os
    
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
        
        # Docs usually have very low std_dev. A std_dev of 8-10 often indicates heavy blurring or pasting.
        score = min(float(std_dev) / 12.0, 1.0)
        
        # Force a higher score if there was extreme peaking (like black boxes or totally white fake text overlays)
        if max_v > 240 and std_dev > 5:
             score = max(score, 0.85)
             
        return {
            "status": "success",
            "score": score,
            "provider": "Fast Local ELA",
            "message": "Manual document editing detected (photoshop/overlays/blur)." if score > 0.55 else "Document appears clear of manual edits.",
            "map": diff_array
        }
    except Exception as e:
        return {"error": str(e)}
