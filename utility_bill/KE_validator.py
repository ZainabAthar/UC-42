"""
modules/utility_bill/ke_validator.py
--------------------------------------
K-Electric Bill Validation Module — UC-42 KYC Forgery Detection

APPROACH:
    1. Detect input quality (resolution check)
    2. Decode barcode — extract account, amount, due date, contract
    3. Run OCR — extract all text fields
    4. Cross-check barcode vs OCR fields (tamper detection)
    5. Validate fixed KE identifiers (GST, NTN)
    6. Validate data logic (recency, dates, amounts)
    7. Return structured verdict

VERDICTS:
    PASS      — all critical checks passed, document looks genuine
    FAIL      — critical check failed, document likely fake/tampered
    REVIEW    — soft failures, needs human review
    RESUBMIT  — image quality too low, ask for better photo

BARCODE CROSS-CHECK (primary tamper detection):
    If someone edits the printed amount → barcode amount stays same → MISMATCH → FAIL
    If someone edits account number     → barcode account stays same → MISMATCH → FAIL
    Barcode requires specialist tools to regenerate → strong tamper signal

INPUT:
    Phone photo (JPG/PNG) — needs full resolution, not WhatsApp compressed
    Scanned PDF           — always works
    Scanned image         — works if resolution > 1500px wide

LATER INTEGRATION:
    cnic_name param → cross-check consumer name on bill vs NADRA verified name
"""

import re
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ── Fixed KE constants (verified from genuine bills) ──────────────────────
KE_GST_NUMBER     = "12-00-2716-007-28"
KE_NTN_NUMBER     = "1543137-1"
KE_ACCOUNT_PREFIX = "04"
KE_ACCOUNT_LENGTH = 13
BILL_MAX_AGE_DAYS = 90
MIN_WIDTH_FOR_BARCODE = 1500   # below this → RESUBMIT

KE_REQUIRED_KEYWORDS = [
    "K-Electric",
    "CORRUPTION",
    "Amount Payable",
    "Due Date",
    "Account",
    "Invoice",
]

CNIC_PATTERN    = re.compile(r'\d{5}-\d{7}-\d')
ACCOUNT_PATTERN = re.compile(r'\b04\d{11}\b')

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,  'may': 5,  'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    'january': 1, 'february': 2, 'march': 3,  'april': 4, 'june': 6,
    'july': 7, 'august': 8, 'september': 9,  'october': 10,
    'november': 11, 'december': 12,
}


class KEBillValidator:

    def __init__(self):
        self._ocr_reader = None

    # ── Loading ────────────────────────────────────────────────────────────

    def _load_image(self, file_path: str) -> Optional[np.ndarray]:
        path = Path(file_path)
        if not path.exists():
            return None
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(file_path)
        return cv2.imread(str(path))

    def _load_pdf(self, file_path: str) -> Optional[np.ndarray]:
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(
                file_path, dpi=200, first_page=1, last_page=1)
            if not pages:
                return None
            return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"PDF load failed: {e}")
            return None

    # ── Quality check ──────────────────────────────────────────────────────

    def _check_image_quality(self, img: np.ndarray) -> dict:
        """
        Check if image resolution is sufficient for barcode reading.
        WhatsApp compresses photos to ~1000px wide — barcodes become unreadable.
        Full resolution phone photos are 3000-4000px wide — barcodes work fine.
        """
        h, w   = img.shape[:2]
        passed = w >= MIN_WIDTH_FOR_BARCODE
        return {
            "width":   w,
            "height":  h,
            "passed":  passed,
            "detail":  f"{w}x{h}px — " + (
                "✓ sufficient resolution" if passed
                else f"✗ too low ({w}px < {MIN_WIDTH_FOR_BARCODE}px minimum) — "
                     f"barcode unreadable. Ask customer to upload original photo.")
        }

    # ── Barcode decoding ───────────────────────────────────────────────────

    def _decode_barcode(self, img: np.ndarray) -> Optional[dict]:
        """
        Decode CODE128 barcode from KE bill.
        Tries multiple scales and preprocessing until barcode is found.

        KE barcode format (verified from 3 genuine bills):
        XXXXXXXX AAAAAAAAAAAAA DDMMYY NNNNNNNNNN XXXXXXXXXX CCCCCCC
         prefix   account(13)  date    amount(10)   ref(11)  contract

        Example:
        0000004000000911170202260000011362000001231030171072
        390000 4000000091117 056526 0000011663 00000126373 0171072
        """
        from pyzbar import pyzbar

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
               if len(img.shape) == 3 else img

        # Try at different scales
        for scale in [1, 1.5, 2, 3]:
            resized = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC)
            codes = pyzbar.decode(resized)
            if codes:
                for c in codes:
                    if c.type == "CODE128":
                        parsed = self._parse_barcode(c.data.decode("utf-8"))
                        if parsed:
                            parsed["raw"]   = c.data.decode("utf-8")
                            parsed["scale"] = scale
                            return parsed

            # Try with Otsu threshold
            _, otsu = cv2.threshold(
                resized, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            codes = pyzbar.decode(otsu)
            if codes:
                for c in codes:
                    if c.type == "CODE128":
                        parsed = self._parse_barcode(c.data.decode("utf-8"))
                        if parsed:
                            parsed["raw"]   = c.data.decode("utf-8")
                            parsed["scale"] = scale
                            return parsed

        return None

    def _parse_barcode(self, data: str) -> Optional[dict]:
        """
        Parse KE CODE128 barcode data into fields.

        Verified format from 3 genuine bills:
          digital PDF:  0000004000000911170202260000011362000001231030171072
          natural photo:3900004000000911170503260000011663000001263730171072
          scanned:      3900004000000911170565260000011663000001263730171072

        Pattern: [prefix 6] [account 13] [DDMMYY 6] [amount 10] [ref 11] [contract 7]
        Total length: 53 chars
        """
        if len(data) < 40:
            return None

        try:
            # Account number is always 13 digits starting with 04
            # Find it in the barcode data
            acct_match = re.search(r'(04\d{11})', data)
            if not acct_match:
                return None

            acct_start = acct_match.start()
            account    = acct_match.group(1)

            # After account: DDMMYY (6 digits)
            date_start = acct_start + 13
            date_raw   = data[date_start:date_start + 6]

            # After date: amount (10 digits)
            amt_start = date_start + 6
            amt_raw   = data[amt_start:amt_start + 10]

            # Contract number — last 7 digits
            contract = data[-7:].lstrip('0')

            # Parse due date
            due_date = None
            if len(date_raw) == 6 and date_raw.isdigit():
                dd = int(date_raw[0:2])
                mm = int(date_raw[2:4])
                yy = int(date_raw[4:6]) + 2000
                try:
                    due_date = datetime(yy, mm, dd)
                except ValueError:
                    pass

            # Parse amount
            amount = None
            if amt_raw.isdigit():
                amount = int(amt_raw.lstrip('0') or '0')

            return {
                "account":  account,
                "due_date": due_date,
                "amount":   amount,
                "contract": contract,
                "date_raw": date_raw,
                "amt_raw":  amt_raw,
            }

        except Exception as e:
            logger.warning(f"Barcode parse failed: {e}")
            return None

    # ── OCR ────────────────────────────────────────────────────────────────

    def _get_ocr(self):
        if self._ocr_reader is None:
            import easyocr
            print("[KEValidator] Loading EasyOCR...")
            self._ocr_reader = easyocr.Reader(['en'], gpu=False)
        return self._ocr_reader

    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Scale to 1200px wide for OCR — optimal for EasyOCR."""
        h, w = img.shape[:2]
        if w > 1200:
            scale = 1200 / w
            img   = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        elif w < 1000:
            scale = 1200 / w
            img   = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)
        return img

    def _run_ocr(self, img: np.ndarray) -> tuple:
        reader    = self._get_ocr()
        lines     = reader.readtext(img, detail=0, paragraph=False)
        full_text = " ".join(lines)
        return full_text, lines

    # ── Field Extraction ───────────────────────────────────────────────────

    def _extract_fields(self, full_text: str, lines: list) -> dict:
        fields = {}
        n = len(lines)

        def next_line(i):
            return lines[i + 1].strip() if i + 1 < n else ""

        def find_label(pattern):
            for i, line in enumerate(lines):
                if re.search(pattern, line, re.IGNORECASE):
                    return i
            return -1

        # GST Number
        # OCR sometimes reads 'GST' as 'CST' or '6ST'
        # Pattern: XX-XX-XXXX-XXX-XX
        m = re.search(
            r'[GC6][S5][T1]\s*N[oc]\.?[\s;:]*'
            r'([0-9]{2}[-\s][0-9]{2}[-\s][0-9]{4}[-\s][0-9]{3}[-\s][0-9a-zA-Z]{2})',
            full_text, re.IGNORECASE)
        if m:
            # Normalize: remove spaces, fix common OCR substitutions
            raw = m.group(1).strip()
            raw = raw.replace(' ', '')
            fields["gst_number"] = raw
            # Also check if it matches expected (allow last char OCR error e→8)
            normalized = raw[:-1] + '8' if raw.endswith('e') else raw
            fields["gst_number_normalized"] = normalized

        # NTN Number
        # OCR reads semicolon instead of space: 'NTN No; 1543137-1'
        # OCR reads period instead of dash:     'NTN No; 1543137.1'
        m = re.search(
            r'NTN\s*N[oc][\s;:.]*([0-9]+[-\.][0-9]+)',
            full_text, re.IGNORECASE)
        if m:
            # Normalize: replace period with dash
            fields["ntn_number"] = m.group(1).strip().replace('.', '-')

        # Account Number
        m = re.search(r'Account\s*No\.?\s*(\d{13,})',
                      full_text, re.IGNORECASE)
        if m:
            fields["account_number"] = m.group(1).strip()
        else:
            hits = ACCOUNT_PATTERN.findall(full_text)
            if hits:
                fields["account_number"] = hits[0]

        # Consumer Name
        idx = find_label(r'customer\s*name|consumer\s*name')
        if idx >= 0:
            candidate = next_line(idx)
            if re.match(r'^[A-Z][A-Z\s\.]{3,50}$', candidate):
                fields["consumer_name"] = candidate

        # CNIC on bill
        hits = CNIC_PATTERN.findall(full_text)
        if hits:
            fields["cnic_on_bill"] = hits[0]

        # Issue Date
        # OCR: 'Issue Date: 19-Fob-2026' — date on same line as label
        # Try same line first, then next line
        idx = find_label(r'issue\s*[Dd]a[lt]e')
        if idx >= 0:
            # Check same line and next line combined
            for check_line in [lines[idx], next_line(idx),
                                lines[idx] + " " + next_line(idx)]:
                m = re.search(
                    r'(\d{1,2}[-/]\w{3}[-/]\d{2,4}'
                    r'|\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                    check_line)
                if m:
                    raw = self._fix_ocr_month(m.group(1))
                    fields["issue_date_raw"] = raw
                    fields["issue_date"]     = self._parse_date(raw)
                    break

        # Bill Month fallback
        # OCR: 'BilI Monlh: Fob-26' — various OCR errors on label
        idx = find_label(r'[Bb]il[lI]\s*[Mm][o0]n[lt]h')
        if idx >= 0:
            for check_line in [lines[idx], next_line(idx)]:
                m = re.search(r'([A-Za-z]{3}-\d{2,4})', check_line)
                if m:
                    raw = self._fix_ocr_month(m.group(1))
                    fields["bill_month_raw"] = raw
                    fields["bill_month"]     = self._parse_date(raw)
                    break

        # Due Date
        m = re.search(
            r'Due\s*Date[^0-9]{0,30}?'
            r'(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}'
            r'|\d{1,2}[-/]\w{3}[-/]\d{2,4}'
            r'|\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            full_text, re.IGNORECASE)
        if m:
            fields["due_date_raw"] = m.group(1)
            fields["due_date"]     = self._parse_date(m.group(1))

        # Primary Amount — "Amount Payable within Due Date"
        # This is the field most commonly forged
        # Extract it specifically, not just any amount on the bill
        primary_amount = None

        # Pattern 1: Amount Payable within Due Date
        m = re.search(
            r'Amount\s*Payable\s*wi?thin\s*Due\s*Date[^\d]*(\d[\d,]+)',
            full_text, re.IGNORECASE)
        if m:
            try:
                primary_amount = float(m.group(1).replace(',', ''))
                fields["primary_amount"] = primary_amount
            except ValueError:
                pass

        # Pattern 2: Amount Due box — 'Rs. XXXXX' near 'Amount Due'
        if not primary_amount:
            m = re.search(
                r'Amount\s*Due[^\d]*Rs\.?\s*(\d[\d,]+)',
                full_text, re.IGNORECASE)
            if m:
                try:
                    primary_amount = float(m.group(1).replace(',', ''))
                    fields["primary_amount"] = primary_amount
                except ValueError:
                    pass

        # Pattern 3: Proximity — find amount on line after Amount Due label
        if not primary_amount:
            idx = find_label(r'amount\s*due|amount\s*payable')
            if idx >= 0:
                for j in range(1, 4):
                    if idx + j < n:
                        m = re.search(r'Rs\.?\s*(\d[\d,]+)', lines[idx + j])
                        if m:
                            try:
                                primary_amount = float(
                                    m.group(1).replace(',', ''))
                                fields["primary_amount"] = primary_amount
                                break
                            except ValueError:
                                pass

        # All amounts (for amount_logic check)
        amounts = []
        for a in re.findall(r'Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)', full_text):
            try:
                amounts.append(float(a.replace(",", "")))
            except ValueError:
                pass
        for a in re.findall(r'Amount[:\s]+([0-9,]+(?:\.[0-9]+)?)',
                            full_text, re.IGNORECASE):
            try:
                amounts.append(float(a.replace(",", "")))
            except ValueError:
                pass
        if amounts:
            fields["amounts"]    = amounts
            fields["min_amount"] = min(amounts)
            fields["max_amount"] = max(amounts)

        # Address
        # KE bill address is always on left panel below consumer name
        # OCR: 'PLOT NO, B-43 HUSSAIN D SILVA TOWN BLOCK P, NORTH NAZ'
        # Strategy: find name line, take next 1-2 lines as address
        idx = find_label(r'customer\s*name|consumer\s*name')
        if idx >= 0:
            # Name is on next line after label
            name_line = idx + 1
            address_parts = []
            for j in range(name_line + 1, min(name_line + 4, n)):
                candidate = lines[j].strip()
                # Address line: contains numbers, commas, or location words
                # Stop at lines that look like other fields
                if re.search(r'CNIC|Account|User|Schd|Consumer No|Meter',
                             candidate, re.IGNORECASE):
                    break
                if len(candidate) > 5:
                    address_parts.append(candidate)
            if address_parts:
                fields["address"] = " ".join(address_parts)

        # Also try: address is always line starting with PLOT/HOUSE/FLAT/SHOP
        if "address" not in fields:
            for line in lines:
                if re.match(
                    r'^(PLOT|HOUSE|FLAT|SHOP|H\.?NO|FLOOR|BLOCK|SECTOR|'
                    r'STREET|LANE|ROAD|AVENUE|PHASE|DHA|GULSHAN|CLIFTON)',
                    line.strip(), re.IGNORECASE):
                    fields["address"] = line.strip()
                    break
        if m:
            fields["invoice_number"] = m.group(1)

        # Contract Number
        m = re.search(r'Contract\s*N(?:o|umber)\.?\s*(\d{6,})',
                      full_text, re.IGNORECASE)
        if m:
            fields["contract_number"] = m.group(1)

        return fields

    def _fix_ocr_month(self, date_str: str) -> str:
        """
        Fix common OCR errors in month names.
        OCR often confuses: Fob→Feb, Jon→Jan, Mor→Mar, Aor→Apr etc.
        """
        OCR_MONTH_FIXES = {
            'fob': 'Feb', 'fob': 'Feb', 'feb': 'Feb',
            'jon': 'Jan', 'jan': 'Jan',
            'mor': 'Mar', 'mar': 'Mar',
            'aor': 'Apr', 'apr': 'Apr',
            'moy': 'May', 'may': 'May',
            'jun': 'Jun', 'juo': 'Jun',
            'jul': 'Jul', 'jol': 'Jul',
            'aug': 'Aug', 'aog': 'Aug',
            'sop': 'Sep', 'sep': 'Sep',
            'oct': 'Oct', 'oc1': 'Oct',
            'nov': 'Nov', 'noy': 'Nov',
            'doc': 'Dec', 'dec': 'Dec',
        }
        # Find 3-letter month part and fix it
        def fix_match(m):
            word = m.group(0).lower()
            return OCR_MONTH_FIXES.get(word, m.group(0))

        return re.sub(r'[A-Za-z]{3}', fix_match, date_str)

    def _parse_date(self, s: str) -> Optional[datetime]:
        if not s:
            return None
        s = re.sub(r'(?<=\d)(st|nd|rd|th)', '', s.strip(),
                   flags=re.IGNORECASE).strip()
        for fmt in ["%d-%b-%Y", "%d-%b-%y", "%d/%m/%Y", "%d-%m-%Y",
                    "%d %B %Y", "%d %b %Y", "%d %b %y",
                    "%d/%m/%y", "%d-%m-%y"]:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        m = re.match(r'(\w{3})-(\d{2,4})$', s)
        if m:
            mon = MONTH_MAP.get(m.group(1).lower())
            yr  = int(m.group(2))
            if yr < 100:
                yr += 2000
            if mon:
                return datetime(yr, mon, 1)
        return None

    # ── Individual checks ──────────────────────────────────────────────────

    def _check_gst_number(self, fields, text):
        gst        = fields.get("gst_number", "")
        gst_norm   = fields.get("gst_number_normalized", "")
        expected   = KE_GST_NUMBER.replace("-", "")

        # Check 1: exact string in text
        if KE_GST_NUMBER in text:
            return {"passed": True, "detail": f"✓ {KE_GST_NUMBER}"}

        # Check 2: CST variant in text (OCR reads G as C)
        if "CST" in text.upper() and "12-00-2716-007" in text:
            return {"passed": True,
                    "detail": f"✓ GST found (OCR read as CST): {KE_GST_NUMBER}"}

        # Check 3: extracted and normalized matches
        for candidate in [gst, gst_norm]:
            if not candidate:
                continue
            candidate_digits = re.sub(r'\W', '', candidate)
            expected_digits  = re.sub(r'\W', '', KE_GST_NUMBER)
            # Allow 1 character difference (OCR error on last char e→8)
            if candidate_digits == expected_digits:
                return {"passed": True, "detail": f"✓ GST matches: {candidate}"}
            if len(candidate_digits) == len(expected_digits):
                diffs = sum(a != b for a, b in
                            zip(candidate_digits, expected_digits))
                if diffs <= 1:
                    return {"passed": True,
                            "detail": f"✓ GST matches (1 OCR error tolerated): {candidate}"}

        return {
            "passed": False,
            "detail": f"Found: '{gst}' | Expected: '{KE_GST_NUMBER}'"
        }

    def _check_ntn_number(self, fields, text):
        ntn    = fields.get("ntn_number", "")
        # Direct string check
        if KE_NTN_NUMBER in text:
            return {"passed": True, "detail": f"✓ {KE_NTN_NUMBER}"}
        # Extracted and normalized
        if ntn == KE_NTN_NUMBER:
            return {"passed": True, "detail": f"✓ {KE_NTN_NUMBER}"}
        # Fuzzy: digits only comparison (handles 1543137.1 vs 1543137-1)
        ntn_digits      = re.sub(r'\D', '', ntn)
        expected_digits = re.sub(r'\D', '', KE_NTN_NUMBER)
        if ntn_digits == expected_digits:
            return {"passed": True,
                    "detail": f"✓ NTN matches (separator normalized): {ntn}"}
        return {
            "passed": False,
            "detail": f"Found: '{ntn}' | Expected: '{KE_NTN_NUMBER}'"
        }

    def _check_account_format(self, fields):
        acct = fields.get("account_number", "")
        if not acct:
            return {"passed": False, "detail": "Account number not found"}
        valid = (len(acct) == KE_ACCOUNT_LENGTH and
                 acct.startswith(KE_ACCOUNT_PREFIX) and
                 acct.isdigit())
        return {
            "passed": valid,
            "detail": f"{acct} — " + (
                "✓ valid format" if valid
                else f"✗ invalid format")
        }

    def _check_required_keywords(self, text):
        missing = [kw for kw in KE_REQUIRED_KEYWORDS
                   if kw.lower() not in text.lower()]
        passed  = len(missing) == 0
        return {
            "passed": passed,
            "detail": "✓ All keywords present" if passed
                      else f"✗ Missing: {missing}"
        }

    def _check_bill_recency(self, fields, barcode=None):
        # Priority: issue_date → bill_month → infer from barcode due date
        date = fields.get("issue_date") or fields.get("bill_month")

        # Fallback: use barcode due date - 20 days as approximate issue date
        if not date and barcode:
            barcode_due = barcode.get("due_date")
            if barcode_due:
                from datetime import timedelta
                date = barcode_due - timedelta(days=20)
                fields["issue_date_inferred"] = str(date.date())

        if not date:
            return {"passed": False,
                    "detail": "Could not extract issue date"}
        age    = (datetime.now() - date).days
        passed = 0 <= age <= BILL_MAX_AGE_DAYS
        return {
            "passed": passed,
            "detail": f"Issued ~{age} days ago — " + (
                "✓ within 3 months" if passed
                else f"✗ older than {BILL_MAX_AGE_DAYS} days")
        }

    def _check_date_logic(self, fields):
        issue = fields.get("issue_date") or fields.get("bill_month")
        due   = fields.get("due_date")
        if not issue or not due:
            return {"passed": None,
                    "detail": "Could not extract both dates"}
        passed = due >= issue
        return {
            "passed": passed,
            "detail": f"Issue:{issue.date()} Due:{due.date()} — " + (
                "✓ logical" if passed else "✗ impossible dates")
        }

    def _check_amount_logic(self, fields):
        amounts = fields.get("amounts", [])
        if not amounts:
            return {"passed": None, "detail": "No amounts found"}
        positive = [a for a in amounts if a > 0]
        passed   = len(positive) >= 1
        return {
            "passed": passed,
            "detail": f"{len(amounts)} amounts found, max={max(amounts):.0f} — " + (
                "✓ OK" if passed else "✗ suspicious")
        }

    def _check_barcode_crosscheck(self, barcode: Optional[dict],
                                   fields: dict) -> dict:
        """
        PRIMARY TAMPER DETECTION.
        Cross-check barcode decoded values vs OCR extracted values.
        Mismatch = document has been tampered after barcode was generated.
        """
        if not barcode:
            return {
                "passed": None,
                "detail": "Barcode not decoded — cannot cross-check"
            }

        issues   = []
        matches  = []

        # Check 1: Account number
        ocr_acct      = fields.get("account_number", "")
        barcode_acct  = barcode.get("account", "")
        if ocr_acct and barcode_acct:
            if ocr_acct == barcode_acct:
                matches.append(f"account={ocr_acct}")
            else:
                issues.append(
                    f"account MISMATCH: printed={ocr_acct} barcode={barcode_acct}")

        # Check 2: Primary Amount
        # Use primary_amount (Amount Payable within Due Date) not max amount
        # This is the field most commonly forged
        barcode_amount  = barcode.get("amount")
        primary_amount  = fields.get("primary_amount")
        ocr_amounts     = fields.get("amounts", [])

        if barcode_amount and primary_amount:
            diff = abs(primary_amount - barcode_amount)
            if diff <= 5:
                matches.append(f"amount={barcode_amount}")
            else:
                issues.append(
                    f"amount MISMATCH: printed={primary_amount:.0f} "
                    f"barcode={barcode_amount} — POSSIBLE FORGERY")
        elif barcode_amount and ocr_amounts:
            # Fallback: flag only if amount is 9x+ barcode amount
            # 11663 → 111663 = 9.57x (forgery) → flagged ✅
            # 9737.73 → 97177 (OCR misread) = 8.33x → NOT flagged ✅
            suspicious = [
                a for a in ocr_amounts
                if a > barcode_amount * 9
            ]
            if suspicious:
                issues.append(
                    f"amount SUSPICIOUS: found {max(suspicious):.0f} "
                    f"but barcode={barcode_amount} — digit likely added")
            else:
                closest = min(ocr_amounts,
                              key=lambda x: abs(x - barcode_amount))
                diff = abs(closest - barcode_amount)
                if diff <= 5:
                    matches.append(f"amount={barcode_amount}")

        # Check 3: Due date
        barcode_due = barcode.get("due_date")
        ocr_due     = fields.get("due_date")
        if barcode_due and ocr_due:
            if barcode_due.date() == ocr_due.date():
                matches.append(f"due_date={barcode_due.date()}")
            else:
                issues.append(
                    f"due_date MISMATCH: printed={ocr_due.date()} "
                    f"barcode={barcode_due.date()}")

        # Check 4: Contract number
        barcode_contract = barcode.get("contract", "")
        ocr_contract     = fields.get("contract_number", "")
        if barcode_contract and ocr_contract:
            if barcode_contract in ocr_contract or ocr_contract in barcode_contract:
                matches.append(f"contract={barcode_contract}")
            else:
                issues.append(
                    f"contract MISMATCH: printed={ocr_contract} "
                    f"barcode={barcode_contract}")

        if issues:
            return {
                "passed": False,
                "detail": f"✗ TAMPER DETECTED — {'; '.join(issues)}"
            }
        if matches:
            return {
                "passed": True,
                "detail": f"✓ Barcode matches: {', '.join(matches)}"
            }
        return {
            "passed": None,
            "detail": "Barcode decoded but could not compare fields"
        }

    def _check_address_match(self, fields, stated_address: Optional[str]) -> dict:
        """
        Cross-check address on bill vs address stated during KYC.
        Note: Bill may be on landlord's name — address match is what matters.
        """
        bill_address = fields.get("address", "")
        if not bill_address:
            return {"passed": None,
                    "detail": "Could not extract address from bill"}
        if not stated_address:
            return {"passed": None,
                    "detail": f"Address on bill: '{bill_address}' — no stated address to compare"}

        def normalize(s):
            # Uppercase, remove punctuation, collapse spaces
            s = re.sub(r'[^\w\s]', ' ', s.upper())
            return re.sub(r'\s+', ' ', s).strip()

        bill_norm    = normalize(bill_address)
        stated_norm  = normalize(stated_address)

        bill_words   = set(bill_norm.split())
        stated_words = set(stated_norm.split())

        # Remove common stop words
        stop = {'NO', 'THE', 'AND', 'OF', 'ST', 'RD', 'TH', 'HOUSE', 'FLAT'}
        bill_words   -= stop
        stated_words -= stop

        if not bill_words or not stated_words:
            return {"passed": None, "detail": "Address too short to compare"}

        overlap  = bill_words & stated_words
        shorter  = min(len(bill_words), len(stated_words))
        score    = len(overlap) / shorter if shorter > 0 else 0

        if score >= 0.6:
            return {"passed": True,
                    "detail": f"✓ Address matches ({score*100:.0f}%): '{bill_address}'"}
        elif score >= 0.3:
            return {"passed": None,
                    "detail": f"⚪ Partial address match ({score*100:.0f}%) — review: "
                              f"Bill='{bill_address}' Stated='{stated_address}'"}
        return {"passed": False,
                "detail": f"✗ Address mismatch: Bill='{bill_address}' "
                          f"Stated='{stated_address}'"}
        """Cross-check consumer name on bill vs CNIC name (NADRA verified later)."""
        if not cnic_name:
            return {"passed": None,
                    "detail": "No CNIC name provided (will be added after NADRA integration)"}
        bill_name = fields.get("consumer_name", "")
        if not bill_name:
            return {"passed": None,
                    "detail": "Could not extract name from bill"}

        def norm(s):
            return re.sub(r'\s+', ' ', s.upper().strip())

        b, c = norm(bill_name), norm(cnic_name)
        if b == c:
            return {"passed": True, "detail": f"✓ Exact match: {bill_name}"}

        bw, cw  = set(b.split()), set(c.split())
        overlap = bw & cw
        shorter = min(len(bw), len(cw))
        if shorter > 0 and len(overlap) / shorter >= 0.7:
            return {"passed": True,
                    "detail": f"✓ Partial match: '{bill_name}' ~ '{cnic_name}'"}
        return {"passed": False,
                "detail": f"✗ Mismatch: Bill='{bill_name}' CNIC='{cnic_name}'"}

    def _check_name_match(self, fields, cnic_name):
        """Cross-check consumer name on bill vs CNIC name (NADRA verified later)."""
        if not cnic_name:
            return {"passed": None,
                    "detail": "No CNIC name provided (will be added after NADRA integration)"}
        bill_name = fields.get("consumer_name", "")
        if not bill_name:
            return {"passed": None,
                    "detail": "Could not extract name from bill"}

        def norm(s):
            import re
            return re.sub(r'\s+', ' ', s.upper().strip())

        b, c = norm(bill_name), norm(cnic_name)
        if b == c:
            return {"passed": True, "detail": f"✓ Exact match: {bill_name}"}

        bw, cw  = set(b.split()), set(c.split())
        overlap = bw & cw
        shorter = min(len(bw), len(cw))
        if shorter > 0 and len(overlap) / shorter >= 0.7:
            return {"passed": True,
                    "detail": f"✓ Partial match: '{bill_name}' ~ '{cnic_name}'"}
        return {"passed": False,
                "detail": f"✗ Mismatch: Bill='{bill_name}' CNIC='{cnic_name}'"}

    def _check_cnic_format(self, fields):
        cnic = fields.get("cnic_on_bill", "")
        if not cnic:
            return {"passed": None,
                    "detail": "No CNIC on bill (not always present)"}
        valid = bool(CNIC_PATTERN.fullmatch(cnic))
        return {
            "passed": valid,
            "detail": f"{cnic} — " + ("✓ valid" if valid else "✗ invalid format")
        }

    # ── Verdict ────────────────────────────────────────────────────────────

    def _calculate_verdict(self, checks: dict) -> tuple:
        CRITICAL = {
            "gst_number",
            "ntn_number",
            "account_format",
            "bill_recency",
            "barcode_crosscheck",   # tamper detection
        }
        flags, warnings, score = [], [], 100

        for name, res in checks.items():
            p = res.get("passed")
            if p is False:
                if name in CRITICAL:
                    flags.append(
                        f"CRITICAL — {name}: {res.get('detail','')}")
                    score -= 30
                else:
                    warnings.append(
                        f"WARNING — {name}: {res.get('detail','')}")
                    score -= 10
            elif p is None:
                score -= 2

        score   = max(0, min(100, score))
        verdict = ("FAIL"   if flags else
                   "REVIEW" if (len(warnings) >= 3 or score < 70) else
                   "PASS")
        return verdict, score, flags, warnings

    # ── Public API ─────────────────────────────────────────────────────────

    def validate(self, file_path: str,
                 cnic_name: Optional[str] = None,
                 stated_address: Optional[str] = None) -> dict:
        """
        Validate a K-Electric bill.

        Args:
            file_path:  Path to bill image (JPG/PNG) or PDF
            cnic_name:  Consumer name from CNIC for cross-check
                        (leave None until NADRA API integrated)

        Returns:
            {
                verdict:          PASS | FAIL | REVIEW | RESUBMIT | ERROR
                confidence:       0-100
                checks:           { name: {passed, detail} }
                extracted_fields: { field: value }
                barcode_data:     { account, amount, due_date, contract }
                flags:            [ critical failures ]
                warnings:         [ soft failures ]
            }
        """
        result = {
            "file":             file_path,
            "verdict":          "ERROR",
            "confidence":       0,
            "checks":           {},
            "extracted_fields": {},
            "barcode_data":     {},
            "flags":            [],
            "warnings":         [],
        }

        # ── Load ──
        img = self._load_image(file_path)
        if img is None:
            result["flags"].append("Could not load file")
            return result

        # ── Quality check ──
        quality = self._check_image_quality(img)
        result["image_quality"] = quality
        if not quality["passed"]:
            result["verdict"] = "RESUBMIT"
            result["flags"].append(
                f"Image resolution too low: {quality['detail']}")
            result["flags"].append(
                "Please upload the original photo (not via WhatsApp) "
                "or submit the digital PDF from the K-Electric app.")
            return result

        # ── Barcode decode (on full resolution image) ──
        barcode = self._decode_barcode(img)
        if barcode:
            result["barcode_data"] = {
                k: str(v) for k, v in barcode.items()
                if k not in ("raw", "scale")
            }
            result["barcode_data"]["raw"]   = barcode.get("raw", "")
            result["barcode_data"]["scale"] = barcode.get("scale", 1)

        # ── OCR (on resized image for speed) ──
        ocr_img           = self._preprocess_for_ocr(img)
        full_text, lines  = self._run_ocr(ocr_img)

        if not full_text.strip():
            result["flags"].append("OCR returned no text")
            result["verdict"] = "REVIEW"
            return result

        # ── Extract fields ──
        fields = self._extract_fields(full_text, lines)
        result["extracted_fields"] = {
            k: str(v) for k, v in fields.items()
            if k not in ("amounts",)
        }
        if "amounts" in fields:
            result["extracted_fields"]["amounts_count"] = len(fields["amounts"])
            result["extracted_fields"]["max_amount"]    = fields.get("max_amount", 0)

        # ── Run all checks ──
        checks = {
            "gst_number":        self._check_gst_number(fields, full_text),
            "ntn_number":        self._check_ntn_number(fields, full_text),
            "account_format":    self._check_account_format(fields),
            "required_keywords": self._check_required_keywords(full_text),
            "bill_recency":      self._check_bill_recency(fields, barcode),
            "date_logic":        self._check_date_logic(fields),
            "amount_logic":      self._check_amount_logic(fields),
            "barcode_crosscheck":self._check_barcode_crosscheck(barcode, fields),
            "address_match":     self._check_address_match(fields, stated_address),
            "name_match":        self._check_name_match(fields, cnic_name),
            "cnic_format":       self._check_cnic_format(fields),
        }
        result["checks"] = checks

        verdict, confidence, flags, warnings = self._calculate_verdict(checks)
        result.update({
            "verdict":    verdict,
            "confidence": confidence,
            "flags":      flags,
            "warnings":   warnings,
        })
        return result

    def validate_and_print(self, file_path: str,
                           cnic_name: Optional[str] = None,
                           stated_address: Optional[str] = None) -> dict:
        r    = self.validate(file_path, cnic_name, stated_address)
        icon = {"PASS": "✅", "FAIL": "❌", "REVIEW": "⚠️",
                "RESUBMIT": "🔄", "ERROR": "💥"}.get(r["verdict"], "?")

        print(f"\n{'='*65}")
        print(f"  K-ELECTRIC BILL VALIDATION REPORT")
        print(f"{'='*65}")
        print(f"  File      : {Path(file_path).name}")
        print(f"  Verdict   : {icon} {r['verdict']}")
        print(f"  Score     : {r['confidence']}/100")

        if r.get("image_quality"):
            q = r["image_quality"]
            print(f"  Resolution: {q['width']}x{q['height']}px")

        if r.get("barcode_data") and r["barcode_data"].get("account"):
            b = r["barcode_data"]
            print(f"\n  ── Barcode Decoded ───────────────────────────────────")
            print(f"  Account  : {b.get('account')}")
            print(f"  Amount   : {b.get('amount')}")
            print(f"  Due Date : {b.get('due_date')}")
            print(f"  Contract : {b.get('contract')}")

        print(f"\n  ── Extracted Fields ──────────────────────────────────")
        for k, v in r["extracted_fields"].items():
            print(f"  {k:<28}: {v}")

        print(f"\n  ── Checks ────────────────────────────────────────────")
        for name, res in r["checks"].items():
            p  = res.get("passed")
            ic = "✅" if p is True else ("❌" if p is False else "⚪")
            print(f"  {ic} {name:<32} {res.get('detail','')}")

        if r["flags"]:
            print(f"\n  ── Failures ──────────────────────────────────────────")
            for f in r["flags"]:
                print(f"  • {f}")
        if r["warnings"]:
            print(f"\n  ── Warnings ──────────────────────────────────────────")
            for w in r["warnings"]:
                print(f"  • {w}")

        print(f"{'='*65}\n")
        return r