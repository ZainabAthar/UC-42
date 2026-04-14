import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
from skimage.metrics import structural_similarity as ssim
import io

def get_noise_map(image):
    """
    Creates a noise map by subtracting a blurred version of the image from itself.
    """
    gray = image.convert('L')
    img_arr = np.array(gray, dtype=np.float32)
    
    # Simple high-pass filter: Image - Gaussian Blur
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
    blur_arr = np.array(blurred, dtype=np.float32)
    
    noise = np.abs(img_arr - blur_arr)
    return noise

def calculate_ela(image, quality=90):
    """
    Error Level Analysis (ELA)
    Saves image as JPEG and calculates the difference with original.
    """
    original = image.convert('RGB')
    
    # Save to buffer
    tmp_buffer = io.BytesIO()
    original.save(tmp_buffer, format='JPEG', quality=quality)
    tmp_buffer.seek(0)
    
    recompressed = Image.open(tmp_buffer)
    
    # Calculate difference
    ela_image = ImageChops.difference(original, recompressed)
    
    # Enhance for visualization
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    
    scale = 255.0 / max_diff
    ela_enhanced = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_enhanced

def calculate_ssim(img1, img2):
    """
    Calculates Structural Similarity Index between two images.
    Images are resized to match if necessary.
    """
    # Convert to grayscale
    gray1 = np.array(img1.convert('L'))
    gray2 = np.array(img2.convert('L'))
    
    # Resize gray2 to match gray1
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    
    if (h1, w1) != (h2, w2):
        img2_resized = img2.convert('L').resize((w1, h1), Image.Resampling.LANCZOS)
        gray2 = np.array(img2_resized)
        
    score, diff = ssim(gray1, gray2, full=True)
    return score, diff

def calculate_diff_map(img_uploaded, img_reference):
    """
    Calculates the bitwise difference between an uploaded region 
    and a 'Golden Standard' reference region.
    Returns a heatmap of changes.
    """
    # 1. Convert to Grayscale
    gray_up = np.array(img_uploaded.convert('L'), dtype=np.float32)
    
    # 2. Resize reference to match uploaded if necessary
    h, w = gray_up.shape
    img_ref_resized = img_reference.convert('L').resize((w, h), Image.Resampling.LANCZOS)
    gray_ref = np.array(img_ref_resized, dtype=np.float32)
    
    # 3. Calculate absolute difference
    diff = np.abs(gray_up - gray_ref)
    
    # 4. Threshold to remove minor noise/interpolation artifacts
    diff[diff < 30] = 0 
    
    # 5. Return normalized heatmap
    return diff

def analyze_region_consistency(full_image, region_images, reference_regions=None):
    """
    Compares noise levels across multiple regions to check for global consistency.
    Also performs bitwise diff if reference regions are provided.
    """
    results = {}
    noise_levels = []
    
    # Global noise baseline
    global_noise = get_noise_map(full_image)
    
    # Estimate global density roughly
    global_arr = np.array(full_image.convert('L'))
    g_density = max(0.01, np.sum(global_arr < 250) / global_arr.size)
    global_norm_std = np.std(global_noise) / (g_density ** 0.5)
    
    for name, reg_img in region_images.items():
        # 1. Calculate Pixel Density
        reg_arr = np.array(reg_img.convert('L'))
        content_pixels = np.sum(reg_arr < 250)
        total_pixels = reg_arr.size
        density = max(0.01, content_pixels / total_pixels)
        
        # 2. Extract Noise
        reg_noise = get_noise_map(reg_img)
        reg_std = np.std(reg_noise)
        
        # 3. Normalize noise by density
        norm_std = reg_std / (density ** 0.5)
        
        # 4. Bitwise Diff (if reference available)
        diff_score = 0
        if reference_regions and name in reference_regions:
            diff = calculate_diff_map(reg_img, reference_regions[name])
            diff_score = float(np.mean(diff))
        
        results[name] = {
            "noise_std": float(reg_std),
            "norm_std": float(norm_std),
            "density": float(density),
            "diff_score": diff_score
        }
        noise_levels.append(norm_std)
    
    # 5. Calculate Z-scores
    for name in results:
        z_score = abs(results[name]["norm_std"] - global_norm_std) / (global_norm_std + 1e-6)
        results[name]["z_score"] = float(z_score)
        # Suspicious if either noise z-score is high OR bitwise diff is high
        results[name]["is_suspicious"] = (z_score > 3.0) or (results[name]["diff_score"] > 5.0)
    
    # Internal consistency
    texture_variance = np.std(noise_levels) if noise_levels else 0
    
    return {
        "regions": results,
        "global_std": np.std(global_noise),
        "texture_variance_score": float(texture_variance)
    }
