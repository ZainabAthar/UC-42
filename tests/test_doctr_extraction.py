"""
test_doctr_extraction.py
------------------------
Test doctr-based field extraction on KE bills.
Tests the address extraction strategy using footer name as anchor.

USAGE:
    cd C:\\Users\\DELL\\PycharmProjects\\uc-42-v2
    python test_doctr_extraction.py
    python test_doctr_extraction.py test_data/ke_bills/genuine/digital_bill.pdf
"""

import re
import sys
import cv2
import numpy as np
from pathlib import Path


def load_image(file_path: str):
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path
        pages = convert_from_path(str(path), dpi=200, first_page=1, last_page=1)
        if not pages:
            return None
        img = np.array(pages[0])
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imread(str(path))


def run_doctr(img: np.ndarray) -> list:
    """Run doctr OCR and return list of text lines."""
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    from PIL import Image as PILImage
    import tempfile
    import os

    # Convert to PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)

    # Resize to 1200px wide for speed
    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h))

    # Save to temp file - doctr requires file path, not numpy array
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    pil_img.save(tmp.name, 'JPEG', quality=95)
    tmp.close()

    # Run OCR
    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_images([tmp.name])
    result = model(doc)

    # Clean up temp file
    os.unlink(tmp.name)

    # Extract lines
    lines = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            text = ' '.join([w.value for w in line.words]).strip()
            if text:
                lines.append(text)
    return lines


def extract_fields(lines: list) -> dict:
    """
    Extract all fields from doctr OCR lines.
    Key strategy: use footer name (clean) to extract address from body.
    """
    fields = {}
    full_text = " ".join(lines)
    n = len(lines)

    # ── Step 1: Extract clean name from footer ──
    # Footer has: 'Customer Name' label then 'MUSTT RABIA BEGUM' on next line
    # or 'MUSTT RABIA BEGUM' directly
    footer_name = None
    for i, line in enumerate(lines):
        if re.search(r'customer\s*name', line, re.IGNORECASE):
            # Check next 2 lines for name
            for j in range(1, 3):
                if i + j < n:
                    candidate = lines[i + j].strip()
                    # Clean name: all caps, letters and spaces only
                    # Remove any trailing punctuation
                    candidate = re.sub(r'[^A-Z\s\.]', '', candidate).strip()
                    if re.match(r'^[A-Z][A-Z\s\.]{3,40}$', candidate):
                        footer_name = candidate
                        fields["consumer_name"] = footer_name
                        break
            if footer_name:
                break

    # Fallback: scan all lines for name pattern
    if not footer_name:
        for line in reversed(lines):  # footer is at bottom
            clean = re.sub(r'[^A-Z\s\.]', '', line.upper()).strip()
            if re.match(r'^[A-Z]{2,}(\s[A-Z]{2,}){1,3}$', clean) and len(clean) > 8:
                footer_name = clean
                fields["consumer_name"] = footer_name
                break

    # ── Step 2: Extract address using name as anchor ──
    # Strategy: find line in body containing the name
    # Remove name from that line → partial address
    # Next line(s) contain rest of address
    if footer_name:
        name_words = set(footer_name.upper().split())

        for i, line in enumerate(lines[:-5]):  # skip footer lines
            line_upper = line.upper()
            line_words = set(re.sub(r'[^A-Z\s]', ' ', line_upper).split())

            # Check if this line contains significant name overlap
            overlap = name_words & line_words
            if len(overlap) >= 2:  # at least 2 name words found
                # Remove name words from line to get address part
                remaining = line_upper
                for word in name_words:
                    remaining = re.sub(r'\b' + word + r'\b', '', remaining)
                remaining = re.sub(r'^\W+', '', remaining).strip()
                remaining = re.sub(r'\s+', ' ', remaining)

                # Collect next line(s) as more address
                address_parts = []
                if len(remaining) > 5:
                    address_parts.append(remaining)

                # Next 1-2 lines may have more address
                for j in range(1, 3):
                    if i + j < n:
                        next_line = lines[i + j].strip()
                        # Stop if it looks like a different field
                        if re.search(
                            r'User Name|CNIC No|Account No|Consumer No|'
                            r'Meter No|Schd No|Amount|Due Date|Rs\.',
                            next_line, re.IGNORECASE):
                            break
                        # Stop if it's too short or garbled
                        if len(next_line) > 5:
                            address_parts.append(next_line)

                if address_parts:
                    # Clean up the combined address
                    address = " ".join(address_parts)
                    address = re.sub(r'\s+', ' ', address).strip()
                    # Remove leading digits/special chars
                    address = re.sub(r'^[\d\W]+', '', address).strip()
                    if len(address) > 10:
                        fields["address"] = address
                        break

    # ── GST Number ──
    m = re.search(
        r'[GC6][S5][T1]\s*N[Oo]\.?\s*'
        r'([0-9]{2}[-\s][0-9]{2}[-\s][0-9]{4}[-\s][0-9]{3}[-\s][0-9a-zA-Z]{2})',
        full_text, re.IGNORECASE)
    if m:
        raw = re.sub(r'\s', '', m.group(1))
        fields["gst_number"] = raw

    # ── NTN Number ──
    m = re.search(r'NTN\s*N[Oo]\.?[\s;:.]*([0-9]+[-\.][0-9]+)',
                  full_text, re.IGNORECASE)
    if m:
        fields["ntn_number"] = m.group(1).strip().replace('.', '-')

    # ── Account Number ──
    m = re.search(r'Account\s*N[o0]\.?\s*(\d{13,})', full_text, re.IGNORECASE)
    if m:
        fields["account_number"] = m.group(1).strip()
    else:
        hits = re.findall(r'\b04\d{11}\b', full_text)
        if hits:
            fields["account_number"] = hits[0]

    # ── Issue Date ──
    # Doctr gives: 'Bill Month: Feb-26 Issue Date: 19-Feb-2026' on same line
    m = re.search(
        r'Issue\s*Date[:\s]*(\d{1,2}[-/]\w{3}[-/]\d{2,4})',
        full_text, re.IGNORECASE)
    if m:
        fields["issue_date_raw"] = m.group(1)

    # ── Bill Month ──
    m = re.search(
        r'Bill\s*Month[:\s]*(\w{3}-\d{2,4})',
        full_text, re.IGNORECASE)
    if m:
        fields["bill_month_raw"] = m.group(1)

    # ── Due Date ──
    m = re.search(
        r'(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})',
        full_text)
    if m:
        fields["due_date_raw"] = m.group(1)

    # ── Amount ──
    # Look for Amount Payable within Due Date
    m = re.search(
        r'Amount\s*Payable\s*with[i1]n[^0-9]*([0-9,]+)',
        full_text, re.IGNORECASE)
    if m:
        try:
            fields["amount_payable"] = float(m.group(1).replace(',', ''))
        except ValueError:
            pass

    # ── Invoice Number ──
    m = re.search(r'(\d{12,})', full_text)
    if m:
        fields["invoice_number"] = m.group(1)

    # ── CNIC on bill ──
    cnic = re.findall(r'\d{5}-\d{7}-\d', full_text)
    if cnic:
        fields["cnic_on_bill"] = cnic[0]

    return fields


def test_file(file_path: str):
    print(f"\n{'='*60}")
    print(f"  File: {Path(file_path).name}")
    print(f"{'='*60}")

    img = load_image(file_path)
    if img is None:
        print("  ERROR: Could not load file")
        return

    print("  Running doctr OCR...")
    lines = run_doctr(img)

    print(f"  Lines extracted: {len(lines)}")
    print(f"\n  ── All OCR Lines ──────────────────────────────────")
    for i, line in enumerate(lines[:20]):  # Show first 20 lines
        print(f"  [{i:03d}] {repr(line)}")
    if len(lines) > 20:
        print(f"  ... and {len(lines)-20} more lines")

    print(f"\n  ── Extracted Fields ───────────────────────────────")
    fields = extract_fields(lines)
    for k, v in fields.items():
        print(f"  {k:<25}: {v}")

    # Address quality check
    addr = fields.get("address", "")
    if addr:
        print(f"\n  ── Address Analysis ───────────────────────────────")
        print(f"  Extracted : {addr}")
        print(f"  Length    : {len(addr)} chars")
        has_plot    = bool(re.search(r'plot|house|flat|b-|h-', addr, re.IGNORECASE))
        has_block   = bool(re.search(r'block|sector|phase|town', addr, re.IGNORECASE))
        has_city    = bool(re.search(r'karachi|lahore|nazimabad|gulshan|clifton|defence',
                                      addr, re.IGNORECASE))
        print(f"  Has plot/house no : {'✅' if has_plot else '❌'}")
        print(f"  Has block/area    : {'✅' if has_block else '❌'}")
        print(f"  Has city/area     : {'✅' if has_city else '❌'}")
    else:
        print(f"\n  ❌ Address NOT extracted")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file(sys.argv[1])
    else:
        # Test all available bills
        test_files = [
            r"C:\Users\DELL\PycharmProjects\uc-42-v2\test_data\ke_bills\genuine\ke_bill_paid1.jpg",
            r"C:\Users\DELL\PycharmProjects\uc-42-v2\test_data\ke_bills\genuine\New Document(1)_1.jpg",
            r"C:\Users\DELL\PycharmProjects\uc-42-v2\test_data\ke_bills\genuine\digital_bill.pdf",
            r"E:\Downloads\WhatsApp Image 2026-02-13 at 8.05.45 PM (1).jpeg",
            r"E:\Downloads\WhatsApp Image 2026-02-13 at 8.05.45 PM.jpeg",
        ]
        for f in test_files:
            if Path(f).exists():
                test_file(f)
            else:
                print(f"Skipping (not found): {f}")