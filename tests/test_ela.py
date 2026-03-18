"""
KE Bill Tampering Detection - Full Test Harness
Tests: Arithmetic consistency, OCR confidence, Name format, Area code vs address
"""

import sys
import os
import re
import json
import traceback
from pathlib import Path

# ── install deps quietly ──────────────────────────────────────────────────────
def install(pkg, import_as=None):
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q", "--break-system-packages"])
    if import_as:
        return __import__(import_as)

try:
    import easyocr
except ImportError:
    install("easyocr", "easyocr")
    import easyocr

try:
    import cv2
except ImportError:
    install("opencv-python-headless", "cv2")
    import cv2

try:
    import numpy as np
except ImportError:
    install("numpy", "numpy")
    import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    install("pymupdf", "fitz")
    import fitz

try:
    from PIL import Image
except ImportError:
    install("Pillow", "PIL")
    from PIL import Image

import io

# ── OCR reader (loaded once) ──────────────────────────────────────────────────
print("Loading EasyOCR (first run downloads model ~100MB)...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("OCR ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load JPG or first page of PDF as numpy array."""
    ext = Path(path).suffix.lower()
    if ext == '.pdf':
        doc = fitz.open(path)
        page = doc[0]
        mat = fitz.Matrix(2.0, 2.0)          # 2× zoom for better OCR
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        doc.close()
        return img
    else:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img


def ocr_full(img: np.ndarray):
    """Run EasyOCR and return list of (bbox, text, confidence)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb, detail=1, paragraph=False)
    return results


def extract_text_blocks(ocr_results):
    """Flatten OCR results to list of (text, confidence) tuples."""
    return [(text.strip(), conf) for (_, text, conf) in ocr_results if text.strip()]


def find_field(blocks, keywords, window=3):
    """
    Find a value near a keyword label.
    Returns (value_text, confidence) or (None, 0).
    """
    texts = [t for t, _ in blocks]
    confs = [c for _, c in blocks]
    for i, t in enumerate(texts):
        if any(kw.lower() in t.lower() for kw in keywords):
            # Return the next non-empty block within window
            for j in range(i+1, min(i+1+window, len(texts))):
                if texts[j].strip():
                    return texts[j], confs[j]
    return None, 0.0


def extract_number(text):
    """Pull first float from a string."""
    if not text:
        return None
    m = re.search(r'[\d,]+\.?\d*', text.replace(',', ''))
    if m:
        try:
            return float(m.group().replace(',', ''))
        except:
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Meter Reading Arithmetic Consistency
# ─────────────────────────────────────────────────────────────────────────────

KE_TARIFF_SLABS = [
    (100,  7.00),   # 0–100 units
    (200,  10.50),  # 101–200
    (300,  16.00),  # 201–300
    (700,  19.00),  # 301–700
    (float('inf'), 22.00),  # 700+
]

def estimate_energy_charge(units):
    """Rough KE slab-based energy charge estimate."""
    if units is None:
        return None
    charge = 0.0
    prev = 0
    for limit, rate in KE_TARIFF_SLABS:
        if units <= prev:
            break
        consumed_in_slab = min(units, limit) - prev
        charge += consumed_in_slab * rate
        prev = limit
        if units <= limit:
            break
    return charge


def test_arithmetic(blocks, label):
    results = {}

    prev_r,  prev_c  = find_field(blocks, ["previous reading", "prev read", "previous meter"])
    curr_r,  curr_c  = find_field(blocks, ["current reading", "curr read", "present reading"])
    units_t, units_c = find_field(blocks, ["units consumed", "consumption", "kwh"])
    amount_t, amt_c  = find_field(blocks, ["amount payable", "total amount", "net payable", "payable amount"])
    energy_t, ene_c  = find_field(blocks, ["energy charges", "energy charge", "units charge"])

    prev   = extract_number(prev_r)
    curr   = extract_number(curr_r)
    units  = extract_number(units_t)
    amount = extract_number(amount_t)
    energy = extract_number(energy_t)

    results['raw'] = {
        'prev_reading': prev, 'curr_reading': curr,
        'units_consumed': units, 'amount': amount, 'energy_charges': energy
    }

    flags = []
    score = 0

    # Check 1: reading delta vs units consumed
    if prev is not None and curr is not None and units is not None:
        delta = curr - prev
        if delta < 0:
            flags.append(f"❌ Negative delta ({delta:.0f}) — curr < prev reading")
            score += 40
        elif abs(delta - units) > max(5, units * 0.05):
            flags.append(f"❌ Reading delta ({delta:.0f}) ≠ units consumed ({units:.0f})")
            score += 40
        else:
            flags.append(f"✅ Reading delta {delta:.0f} ≈ units consumed {units:.0f}")
    else:
        flags.append(f"⚠️  Could not extract readings (prev={prev}, curr={curr}, units={units})")

    # Check 2: units vs estimated energy charge
    if units is not None and energy is not None:
        estimated = estimate_energy_charge(units)
        if estimated:
            ratio = energy / estimated
            if ratio < 0.5 or ratio > 2.0:
                flags.append(f"❌ Energy charge {energy:.0f} far from slab estimate {estimated:.0f} (ratio={ratio:.2f})")
                score += 20
            else:
                flags.append(f"✅ Energy charge {energy:.0f} vs slab estimate {estimated:.0f} (ratio={ratio:.2f})")

    results['flags'] = flags
    results['risk_score'] = score
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — OCR Confidence in Key Fields
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.80

def test_ocr_confidence(blocks, label):
    results = {'field_confidences': {}, 'flags': [], 'risk_score': 0}

    field_keywords = {
        'name':    ["customer name", "consumer name", "name"],
        'address': ["address", "installation address"],
        'account': ["account no", "consumer no", "reference no", "ref no"],
        'amount':  ["amount payable", "total amount", "net payable"],
    }

    score = 0
    for field, keywords in field_keywords.items():
        val, conf = find_field(blocks, keywords)
        results['field_confidences'][field] = {'value': val, 'confidence': round(conf, 3)}
        if val is not None:
            if conf < CONFIDENCE_THRESHOLD:
                results['flags'].append(
                    f"❌ Low OCR confidence on '{field}': {conf:.3f} ('{val[:40]}')"
                )
                score += 15
            else:
                results['flags'].append(
                    f"✅ '{field}' confidence OK: {conf:.3f}"
                )
        else:
            results['flags'].append(f"⚠️  Field '{field}' not found in OCR")

    results['risk_score'] = score
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Name Format Validation
# ─────────────────────────────────────────────────────────────────────────────

INVALID_NAME_PATTERNS = [
    (r'\d',                     "Contains digits"),
    (r'[<>{}[\]\\|@#$%^&*]',    "Contains special characters"),
    (r'(pvt|ltd|co\.|inc|llc)', "Looks like company name"),
    (r'^[A-Z\s]+$',             "ALL CAPS (possible Paint edit)"),
    (r'^\S+$',                  "Single word only (suspicious)"),
]

def test_name_format(blocks, label):
    results = {'flags': [], 'risk_score': 0}
    name, conf = find_field(blocks, ["customer name", "consumer name", "name"])

    if name is None:
        results['flags'].append("⚠️  Name field not found")
        return results

    results['name_found'] = name
    score = 0

    # Length check
    if len(name) < 4:
        results['flags'].append(f"❌ Name too short: '{name}'")
        score += 10
    elif len(name) > 60:
        results['flags'].append(f"❌ Name too long ({len(name)} chars)")
        score += 10
    else:
        results['flags'].append(f"✅ Name length OK: {len(name)} chars")

    # Word count
    words = name.split()
    if len(words) < 2:
        results['flags'].append(f"❌ Only one word in name: '{name}'")
        score += 10
    else:
        results['flags'].append(f"✅ Name has {len(words)} words")

    # Pattern checks
    for pattern, reason in INVALID_NAME_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            results['flags'].append(f"❌ Name pattern fail — {reason}: '{name}'")
            score += 10

    results['risk_score'] = score
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Consumer Number → Area Consistency
# ─────────────────────────────────────────────────────────────────────────────

# KE Division codes → area keywords (extend as you learn more codes)
KE_DIVISION_MAP = {
    '01': ['clifton', 'saddar', 'defence', 'dha'],
    '02': ['gulshan', 'nazimabad', 'north nazimabad'],
    '03': ['korangi', 'landhi', 'malir'],
    '04': ['site', 'baldia', 'orangi'],
    '05': ['north karachi', 'surjani', 'liaquatabad'],
    '06': ['gulberg', 'fb area', 'federal b area'],
    '07': ['shahrah-e-faisal', 'pechs', 'tariq road'],
    '08': ['lyari', 'kemari', 'mauripur'],
}

def test_area_consistency(blocks, label):
    results = {'flags': [], 'risk_score': 0}

    consumer_val, _ = find_field(blocks, ["consumer no", "account no", "reference no", "ref no", "consumer number"])
    address_val, _  = find_field(blocks, ["address", "installation address", "premises"])

    results['consumer_no'] = consumer_val
    results['address'] = address_val

    if not consumer_val:
        results['flags'].append("⚠️  Consumer number not found for area check")
        return results

    if not address_val:
        results['flags'].append("⚠️  Address not found for area check")
        return results

    # Try to extract division code (first 2 digits of consumer no)
    digits = re.sub(r'\D', '', consumer_val)
    if len(digits) < 2:
        results['flags'].append(f"⚠️  Consumer number too short to parse: '{consumer_val}'")
        return results

    div_code = digits[:2]
    expected_areas = KE_DIVISION_MAP.get(div_code)

    if not expected_areas:
        results['flags'].append(f"⚠️  Division code '{div_code}' not in lookup table (extend KE_DIVISION_MAP)")
        return results

    address_lower = address_val.lower()
    match = any(area in address_lower for area in expected_areas)

    if match:
        results['flags'].append(f"✅ Address area matches division code '{div_code}'")
    else:
        results['flags'].append(
            f"❌ Address '{address_val[:50]}' doesn't match division '{div_code}' "
            f"(expected areas: {expected_areas})"
        )
        results['risk_score'] = 30

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests(path, label):
    print(f"\n{'═'*70}")
    print(f"  FILE : {Path(path).name}")
    print(f"  LABEL: {label}")
    print(f"{'═'*70}")

    try:
        img = load_image(path)
        print(f"  Image loaded: {img.shape[1]}×{img.shape[0]}px")
    except Exception as e:
        print(f"  ❌ LOAD FAILED: {e}")
        return

    try:
        ocr_results = ocr_full(img)
        blocks = extract_text_blocks(ocr_results)
        print(f"  OCR complete: {len(blocks)} text blocks detected")
    except Exception as e:
        print(f"  ❌ OCR FAILED: {e}")
        traceback.print_exc()
        return

    total_risk = 0

    # ── TEST 1: Arithmetic ──
    print(f"\n  ┌─ TEST 1: Meter Reading Arithmetic")
    t1 = test_arithmetic(blocks, label)
    for f in t1['flags']:
        print(f"  │  {f}")
    print(f"  │  Raw fields: {t1['raw']}")
    print(f"  └─ Risk contribution: +{t1['risk_score']}")
    total_risk += t1['risk_score']

    # ── TEST 2: OCR Confidence ──
    print(f"\n  ┌─ TEST 2: OCR Confidence on Key Fields")
    t2 = test_ocr_confidence(blocks, label)
    for f in t2['flags']:
        print(f"  │  {f}")
    print(f"  └─ Risk contribution: +{t2['risk_score']}")
    total_risk += t2['risk_score']

    # ── TEST 3: Name Format ──
    print(f"\n  ┌─ TEST 3: Name Format Validation")
    t3 = test_name_format(blocks, label)
    for f in t3['flags']:
        print(f"  │  {f}")
    print(f"  └─ Risk contribution: +{t3['risk_score']}")
    total_risk += t3['risk_score']

    # ── TEST 4: Area Consistency ──
    print(f"\n  ┌─ TEST 4: Consumer No → Area Consistency")
    t4 = test_area_consistency(blocks, label)
    for f in t4['flags']:
        print(f"  │  {f}")
    print(f"  └─ Risk contribution: +{t4['risk_score']}")
    total_risk += t4['risk_score']

    # ── VERDICT ──
    print(f"\n  {'─'*50}")
    print(f"  TOTAL RISK SCORE : {total_risk}")
    if total_risk >= 80:
        verdict = "🔴 REJECT"
    elif total_risk >= 40:
        verdict = "🟡 MANUAL REVIEW"
    else:
        verdict = "🟢 PASS"
    print(f"  VERDICT          : {verdict}")
    print(f"  {'─'*50}")

    return {
        'file': Path(path).name,
        'label': label,
        'total_risk': total_risk,
        'verdict': verdict,
        'tests': {'arithmetic': t1, 'ocr_confidence': t2, 'name_format': t3, 'area': t4}
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

files = [
    (r'C:\Users\DELL\PycharmProjects\uc-42-v2\test_data\ke_bills\genuine\digital_bill.pdf',      'GENUINE pdf'),
    (r'C:\Users\DELL\PycharmProjects\uc-42-v2\test_data\ke_bills\genuine\ke_bill_paid1.jpg',     'GENUINE photo'),
    (r'C:\Users\DELL\PycharmProjects\uc-42-v2\test_data\ke_bills\genuine\New Document(1)_1.jpg', 'GENUINE scanned'),
    (r'C:\Users\DELL\PycharmProjects\uc-42-v2\test_data\ke_bills\forged\forged2.jpg',            'FORGED2 amount edit'),
    (r'C:\Users\DELL\PycharmProjects\uc-42-v2\test_data\ke_bills\forged\forged.jpg',             'FORGED'),
]

print("KE BILL TAMPERING DETECTION — FULL TEST SUITE")
print("=" * 70)

all_results = []
for path, label in files:
    result = run_all_tests(path, label)
    if result:
        all_results.append(result)

# ── SUMMARY TABLE ──
print(f"\n\n{'═'*70}")
print("  SUMMARY")
print(f"{'═'*70}")
print(f"  {'File':<35} {'Label':<25} {'Score':>5}  {'Verdict'}")
print(f"  {'─'*35} {'─'*25} {'─'*5}  {'─'*15}")
for r in all_results:
    print(f"  {r['file']:<35} {r['label']:<25} {r['total_risk']:>5}  {r['verdict']}")
print(f"{'═'*70}\n")

# Save JSON results
out_path = Path(__file__).parent / "ke_tamper_results.json"
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Full results saved to: {out_path}")