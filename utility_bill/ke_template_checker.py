"""
modules/utility_bill/ke_template_checker.py
--------------------------------------------
K-Electric Bill Visual Template Verification — UC-42

CHECKS (13 total):
    Structural:
        1.  aspect_ratio          — portrait document
        2.  teal_header           — full-width teal band at top
        3.  ke_green_logo         — KE green logo top-left
        4.  right_panel_header    — green "Reach K-Electric Limited" bar
        5.  left_panel_content    — consumer details zone not blank
        6.  right_panel_content   — billing info zone not blank

    Color/Branding:
        7.  light_blue_sections   — section background colors
        8.  star_badge_orange     — orange STAR consumer badge center-left
        9.  amount_payable_box    — highlighted amount box right panel

    Content Zones:
        10. qr_code_zone          — QR code top-right (high contrast)
        11. usage_graph_zone      — 13-month bar chart left panel
        12. corruption_zone       — SAY NO TO CORRUPTION bottom-left
        13. footer_zone           — barcode strip at bottom

DESIGN:
    - Fully local, no API, no OCR
    - Works on phone photos and scanned PDFs
    - BaseTemplateChecker is extensible for LESCO, SSGC, CNIC etc.

USAGE:
    from modules.utility_bill.ke_template_checker import KETemplateChecker
    checker = KETemplateChecker()
    result  = checker.check("path/to/bill.jpg")
    result  = checker.check_and_print("path/to/bill.jpg")
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# BASE CLASS
# ══════════════════════════════════════════════════════════════════════════

class BaseTemplateChecker:
    """
    Base visual template checker.
    Subclass for each document type:
        KETemplateChecker        ← built
        LESCOTemplateChecker     ← future
        SSGCTemplateChecker      ← future
        CNICTemplateChecker      ← future
    """

    def _load_image(self, file_path: str) -> Optional[np.ndarray]:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(file_path)
        img = cv2.imread(str(path))
        if img is None:
            logger.error(f"Could not read: {file_path}")
        return img

    def _load_pdf(self, file_path: str) -> Optional[np.ndarray]:
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(
                file_path, dpi=150, first_page=1, last_page=1)
            if not pages:
                return None
            return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"PDF load failed: {e}")
            return None

    def _normalize(self, img: np.ndarray,
                   target_width: int = 1000) -> np.ndarray:
        """Resize to standard width for consistent zone percentages."""
        h, w = img.shape[:2]
        if w != target_width:
            scale = target_width / w
            img   = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        return img

    def _crop_zone(self, img: np.ndarray,
                   x_pct: float, y_pct: float,
                   w_pct: float, h_pct: float) -> np.ndarray:
        """Crop zone by percentage of image dimensions."""
        h, w = img.shape[:2]
        x1 = max(0, int(x_pct * w))
        y1 = max(0, int(y_pct * h))
        x2 = min(w, int((x_pct + w_pct) * w))
        y2 = min(h, int((y_pct + h_pct) * h))
        return img[y1:y2, x1:x2]

    def _color_ratio(self, img: np.ndarray,
                     lower_hsv: np.ndarray,
                     upper_hsv: np.ndarray) -> float:
        """Fraction of pixels matching HSV color range."""
        if img.size == 0:
            return 0.0
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        return float(np.count_nonzero(mask)) / float(mask.size)

    def _color_present(self, img: np.ndarray,
                       lower_hsv: np.ndarray,
                       upper_hsv: np.ndarray,
                       min_pct: float = 0.05) -> bool:
        return self._color_ratio(img, lower_hsv, upper_hsv) >= min_pct

    def _contrast(self, img: np.ndarray) -> float:
        """Standard deviation of grayscale — measure of content richness."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return float(np.std(gray))

    def check(self, file_path: str) -> dict:
        raise NotImplementedError("Subclass must implement check()")


# ══════════════════════════════════════════════════════════════════════════
# K-ELECTRIC CHECKER
# ══════════════════════════════════════════════════════════════════════════

class KETemplateChecker(BaseTemplateChecker):
    """
    Visual template checker for K-Electric utility bills.

    KE Bill Layout (percentage coords, works at any resolution):

    ┌─────────────────────────────────────────────────┐ 0%
    │  [TEAL BAND — full width]                       │ 0–8%
    │  [KE green logo]        [QR] [comm channels]   │ 0–15%
    ├──────────────────────┬──────────────────────────┤ 8%
    │                      │  [green header bar]      │
    │  Consumer details    │  Account No.             │
    │  Name, address       │  Invoice No.             │
    │  Tariff category     │  Issue Date / Bill Month │
    │                      │                          │
    │  [STAR orange badge] │  [Amount Payable box]    │ 25–45%
    │                      │  Rs. XXXX                │
    │  Usage graph ████    │  [Due Date box]          │ 35–55%
    │  (bar chart)         │                          │
    │                      │  [Barcode — right side]  │ 30–55%
    │  [light blue section │                          │
    │   backgrounds]       │                          │
    │  Billing history     │  Bill calculation        │ 55–75%
    │                      │                          │
    │  [SAY NO TO          │                          │ 63–83%
    │   CORRUPTION]        │                          │
    ├──────────────────────┴──────────────────────────┤ 83%
    │  [KE footer logo]    [footer barcode strip]     │ 83–100%
    └─────────────────────────────────────────────────┘ 100%

    Left panel:  0–48% width
    Right panel: 48–100% width
    """

    name = "K-Electric Bill"

    # ── HSV Color Ranges (verified from genuine KE bills) ─────────────────

    # KE green — logo bird, section headers, right panel header bar
    _GREEN_L  = np.array([35,  35,  35])
    _GREEN_U  = np.array([85, 255, 255])

    # Teal/cyan — top header band
    _TEAL_L   = np.array([80,  70,  70])
    _TEAL_U   = np.array([105, 255, 255])

    # Light blue — section backgrounds in left panel
    _LBLUE_L  = np.array([95,  15, 140])
    _LBLUE_U  = np.array([130, 200, 255])

    # Orange — STAR consumer badge
    _ORANGE_L = np.array([8,  100, 100])
    _ORANGE_U = np.array([28, 255, 255])

    # ── Checks ────────────────────────────────────────────────────────────

    def _check_aspect_ratio(self, img: np.ndarray) -> dict:
        """Portrait document: height/width = 1.2 to 2.0."""
        h, w   = img.shape[:2]
        ratio  = h / w
        passed = 1.2 <= ratio <= 2.0
        return {
            "passed": passed,
            "detail": f"ratio={ratio:.2f} — " + (
                "✓ portrait document" if passed
                else "✗ not a portrait document")
        }

    def _check_teal_header(self, img: np.ndarray) -> dict:
        """Full-width teal band: top 8% of bill."""
        zone  = self._crop_zone(img, 0.0, 0.0, 1.0, 0.08)
        ratio = self._color_ratio(zone, self._TEAL_L, self._TEAL_U)
        passed = ratio >= 0.04
        return {
            "passed": passed,
            "detail": f"teal={ratio*100:.1f}% — " + (
                "✓ header band present" if passed
                else "✗ teal header missing")
        }

    def _check_ke_green_logo(self, img: np.ndarray) -> dict:
        """KE green logo: top-left 28% width, top 18% height."""
        zone  = self._crop_zone(img, 0.0, 0.0, 0.28, 0.18)
        ratio = self._color_ratio(zone, self._GREEN_L, self._GREEN_U)
        passed = ratio >= 0.02
        return {
            "passed": passed,
            "detail": f"green={ratio*100:.1f}% in logo zone — " + (
                "✓ KE logo present" if passed
                else "✗ KE logo color missing")
        }

    def _check_right_panel_header(self, img: np.ndarray) -> dict:
        """Green 'Reach K-Electric Limited' bar: right 52%, top 10%."""
        zone  = self._crop_zone(img, 0.48, 0.0, 0.52, 0.10)
        ratio = self._color_ratio(zone, self._GREEN_L, self._GREEN_U)
        passed = ratio >= 0.03
        return {
            "passed": passed,
            "detail": f"green={ratio*100:.1f}% in right header — " + (
                "✓ right panel header present" if passed
                else "✗ right panel header missing")
        }

    def _check_left_panel_content(self, img: np.ndarray) -> dict:
        """Left panel must not be blank: left 48%, 15–80% height."""
        zone   = self._crop_zone(img, 0.0, 0.15, 0.48, 0.65)
        std    = self._contrast(zone)
        passed = std > 15
        return {
            "passed": passed,
            "detail": f"contrast={std:.0f} — " + (
                "✓ left panel has content" if passed
                else "✗ left panel appears blank")
        }

    def _check_right_panel_content(self, img: np.ndarray) -> dict:
        """Right panel must not be blank: right 52%, 10–85% height."""
        zone   = self._crop_zone(img, 0.48, 0.10, 0.52, 0.75)
        std    = self._contrast(zone)
        passed = std > 15
        return {
            "passed": passed,
            "detail": f"contrast={std:.0f} — " + (
                "✓ right panel has content" if passed
                else "✗ right panel appears blank")
        }

    def _check_light_blue_sections(self, img: np.ndarray) -> dict:
        """Light blue section backgrounds: left 48%, middle 20–65%."""
        zone  = self._crop_zone(img, 0.0, 0.20, 0.48, 0.45)
        ratio = self._color_ratio(zone, self._LBLUE_L, self._LBLUE_U)
        passed = ratio >= 0.03
        return {
            "passed": passed,
            "detail": f"light_blue={ratio*100:.1f}% — " + (
                "✓ section backgrounds present" if passed
                else "✗ expected section colors missing")
        }

    def _check_star_badge(self, img: np.ndarray) -> dict:
        """
        Orange STAR consumer badge: center-left panel, 18–35% height.
        'You are a STAR Consumer' — orange star graphic always present.
        """
        zone  = self._crop_zone(img, 0.05, 0.18, 0.40, 0.18)
        ratio = self._color_ratio(zone, self._ORANGE_L, self._ORANGE_U)
        passed = ratio >= 0.01
        return {
            "passed": passed,
            "detail": f"orange={ratio*100:.1f}% in STAR zone — " + (
                "✓ STAR badge present" if passed
                else "✗ STAR consumer badge color missing")
        }

    def _check_qr_code_zone(self, img: np.ndarray) -> dict:
        """
        QR code: top-right area, approximately 48–65% width, 0–15% height.
        QR code = very high contrast black/white pattern.
        """
        zone   = self._crop_zone(img, 0.48, 0.0, 0.20, 0.15)
        std    = self._contrast(zone)
        # QR code has very high contrast — black squares on white
        passed = std > 40
        return {
            "passed": passed,
            "detail": f"QR zone contrast={std:.0f} — " + (
                "✓ QR code zone has content" if passed
                else "✗ QR code zone appears missing")
        }

    def _check_usage_graph(self, img: np.ndarray) -> dict:
        """
        13-month usage bar chart: left panel, approximately 38–58% height.
        Bar chart = vertical dark stripes — detectable as high contrast.
        """
        zone   = self._crop_zone(img, 0.02, 0.38, 0.44, 0.20)
        std    = self._contrast(zone)
        passed = std > 20
        return {
            "passed": passed,
            "detail": f"graph zone contrast={std:.0f} — " + (
                "✓ usage graph zone present" if passed
                else "✗ 13-month graph zone appears missing")
        }

    def _check_amount_payable_box(self, img: np.ndarray) -> dict:
        """
        'Amount Payable' highlighted box: right panel, 28–42% height.
        Teal/green highlighted box always present with Rs. amount.
        """
        zone  = self._crop_zone(img, 0.48, 0.28, 0.52, 0.14)
        ratio = self._color_ratio(zone, self._TEAL_L, self._TEAL_U)
        # Also check green
        ratio_g = self._color_ratio(zone, self._GREEN_L, self._GREEN_U)
        passed  = (ratio + ratio_g) >= 0.03
        return {
            "passed": passed,
            "detail": f"amount box color={( ratio+ratio_g)*100:.1f}% — " + (
                "✓ Amount Payable box present" if passed
                else "✗ Amount Payable highlighted box missing")
        }

    def _check_corruption_zone(self, img: np.ndarray) -> dict:
        """
        SAY NO TO CORRUPTION: left panel, 63–83% height.
        Always has text + box graphics.
        """
        zone   = self._crop_zone(img, 0.0, 0.63, 0.48, 0.20)
        std    = self._contrast(zone)
        passed = std > 15
        return {
            "passed": passed,
            "detail": f"corruption zone contrast={std:.0f} — " + (
                "✓ bottom content zone present" if passed
                else "✗ SAY NO TO CORRUPTION zone appears blank")
        }

    def _check_footer_zone(self, img: np.ndarray) -> dict:
        """
        Footer barcode strip: bottom 15%.
        High contrast = dark barcode lines on light background.
        """
        zone   = self._crop_zone(img, 0.0, 0.85, 1.0, 0.15)
        std    = self._contrast(zone)
        passed = std > 25
        return {
            "passed": passed,
            "detail": f"footer contrast={std:.0f} — " + (
                "✓ footer barcode strip present" if passed
                else "✗ footer zone appears blank")
        }

    # ── Verdict ────────────────────────────────────────────────────────────

    def _calculate_verdict(self, checks: dict) -> tuple:
        # Must pass — wrong document type if these fail
        CRITICAL = {
            "aspect_ratio",
            "teal_header",
            "ke_green_logo",
        }
        # Should pass — fake/tampered template if these fail
        IMPORTANT = {
            "right_panel_header",
            "left_panel_content",
            "right_panel_content",
            "light_blue_sections",
            "star_badge",
            "qr_code_zone",
            "usage_graph",
            "amount_payable_box",
            "corruption_zone",
            "footer_zone",
        }

        flags, warnings, score = [], [], 100

        for name, res in checks.items():
            p = res.get("passed")
            if p is False:
                if name in CRITICAL:
                    flags.append(
                        f"CRITICAL — {name}: {res.get('detail','')}")
                    score -= 35
                else:
                    warnings.append(
                        f"WARNING — {name}: {res.get('detail','')}")
                    score -= 8
            elif p is None:
                score -= 2

        score   = max(0, min(100, score))
        verdict = ("FAIL"   if flags else
                   "REVIEW" if (len(warnings) >= 3 or score < 60) else
                   "PASS")
        return verdict, score, flags, warnings

    # ── Public API ─────────────────────────────────────────────────────────

    def check(self, file_path: str) -> dict:
        """
        Run all 13 visual checks on a KE bill.

        Args:
            file_path: Path to bill image (JPG/PNG) or PDF

        Returns:
            {
                verdict:    PASS | FAIL | REVIEW | ERROR
                confidence: 0-100
                checks:     { name: {passed, detail} }
                flags:      [ critical failures ]
                warnings:   [ soft failures ]
            }
        """
        result = {
            "file":       file_path,
            "checker":    self.name,
            "verdict":    "ERROR",
            "confidence": 0,
            "checks":     {},
            "flags":      [],
            "warnings":   [],
        }

        img = self._load_image(file_path)
        if img is None:
            result["flags"].append("Could not load file")
            return result

        img = self._normalize(img)

        checks = {
            # Structural
            "aspect_ratio":        self._check_aspect_ratio(img),
            "teal_header":         self._check_teal_header(img),
            "ke_green_logo":       self._check_ke_green_logo(img),
            "right_panel_header":  self._check_right_panel_header(img),
            "left_panel_content":  self._check_left_panel_content(img),
            "right_panel_content": self._check_right_panel_content(img),
            # Color/branding
            "light_blue_sections": self._check_light_blue_sections(img),
            "star_badge":          self._check_star_badge(img),
            "amount_payable_box":  self._check_amount_payable_box(img),
            # Content zones
            "qr_code_zone":        self._check_qr_code_zone(img),
            "usage_graph":         self._check_usage_graph(img),
            "corruption_zone":     self._check_corruption_zone(img),
            "footer_zone":         self._check_footer_zone(img),
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

    def check_and_print(self, file_path: str) -> dict:
        r    = self.check(file_path)
        icon = {"PASS": "✅", "FAIL": "❌", "REVIEW": "⚠️",
                "ERROR": "💥"}.get(r["verdict"], "?")

        print(f"\n{'='*65}")
        print(f"  KE VISUAL TEMPLATE CHECK  ({self.name})")
        print(f"{'='*65}")
        print(f"  File    : {Path(file_path).name}")
        print(f"  Verdict : {icon} {r['verdict']}")
        print(f"  Score   : {r['confidence']}/100")

        print(f"\n  ── Structural ────────────────────────────────────────")
        structural = ["aspect_ratio", "teal_header", "ke_green_logo",
                      "right_panel_header", "left_panel_content",
                      "right_panel_content"]
        for name in structural:
            res = r["checks"].get(name, {})
            p   = res.get("passed")
            ic  = "✅" if p is True else ("❌" if p is False else "⚪")
            print(f"  {ic} {name:<30} {res.get('detail','')}")

        print(f"\n  ── Color / Branding ──────────────────────────────────")
        color = ["light_blue_sections", "star_badge", "amount_payable_box"]
        for name in color:
            res = r["checks"].get(name, {})
            p   = res.get("passed")
            ic  = "✅" if p is True else ("❌" if p is False else "⚪")
            print(f"  {ic} {name:<30} {res.get('detail','')}")

        print(f"\n  ── Content Zones ─────────────────────────────────────")
        content = ["qr_code_zone", "usage_graph",
                   "corruption_zone", "footer_zone"]
        for name in content:
            res = r["checks"].get(name, {})
            p   = res.get("passed")
            ic  = "✅" if p is True else ("❌" if p is False else "⚪")
            print(f"  {ic} {name:<30} {res.get('detail','')}")

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