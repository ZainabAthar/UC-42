"""
tests/test_ke_template.py
--------------------------
Test KE visual template checker on genuine and forged bills.

USAGE:
    cd C:\\Users\\DELL\\PycharmProjects\\uc-42-v2
    python tests/test_ke_template.py

PUT TEST IMAGES IN:
    test_data/ke_bills/genuine/   — should PASS
    test_data/ke_bills/forged/    — should FAIL or REVIEW
"""

import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.utility_bill.ke_template_checker import KETemplateChecker

GENUINE_DIR = PROJECT_ROOT / "test_data" / "ke_bills" / "genuine"
FORGED_DIR  = PROJECT_ROOT / "test_data" / "ke_bills" / "forged"
OUTPUT_CSV  = PROJECT_ROOT / "outputs" / "ke_template_results.csv"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".pdf", ".bmp"}


def run():
    checker = KETemplateChecker()

    genuine = [p for p in GENUINE_DIR.rglob("*")
               if p.suffix.lower() in IMAGE_EXTS] if GENUINE_DIR.exists() else []
    forged  = [p for p in FORGED_DIR.rglob("*")
               if p.suffix.lower() in IMAGE_EXTS] if FORGED_DIR.exists() else []

    print(f"[Test] Genuine: {len(genuine)} | Forged: {len(forged)}")

    results = []
    fp = fn = tp = tn = 0

    # ── Genuine bills ──
    if genuine:
        print(f"\n{'='*65}")
        print(f"  GENUINE BILLS (expect PASS or REVIEW)")
        print(f"{'='*65}")
        for p in genuine:
            r       = checker.check(str(p))
            correct = r["verdict"] != "FAIL"
            status  = "✅ CORRECT" if correct else "🚨 FALSE POSITIVE"
            if correct: tn += 1
            else:        fp += 1
            print(f"  {status} | {r['verdict']:6} | Score:{r['confidence']:3} | {p.name}")
            for f in r["flags"]:
                print(f"           └─ {f}")
            results.append({"file": p.name, "label": "GENUINE",
                            "verdict": r["verdict"], "score": r["confidence"],
                            "correct": correct,
                            "flags": "; ".join(r["flags"]),
                            "warnings": "; ".join(r["warnings"])})

    # ── Forged bills ──
    if forged:
        print(f"\n{'='*65}")
        print(f"  FORGED BILLS (expect FAIL or REVIEW)")
        print(f"{'='*65}")
        for p in forged:
            r       = checker.check(str(p))
            correct = r["verdict"] != "PASS"
            status  = "✅ DETECTED" if correct else "❌ MISSED"
            if correct: tp += 1
            else:        fn += 1
            print(f"  {status} | {r['verdict']:6} | Score:{r['confidence']:3} | {p.name}")
            for f in r["flags"]:
                print(f"           └─ {f}")
            results.append({"file": p.name, "label": "FORGED",
                            "verdict": r["verdict"], "score": r["confidence"],
                            "correct": correct,
                            "flags": "; ".join(r["flags"]),
                            "warnings": "; ".join(r["warnings"])})

    # ── Summary ──
    total = len(results)
    if total > 0:
        accuracy = sum(1 for r in results if r["correct"]) / total * 100
        print(f"\n{'='*65}")
        print(f"  SUMMARY")
        print(f"{'='*65}")
        print(f"  Total      : {total}")
        print(f"  Accuracy   : {accuracy:.1f}%")
        if genuine:
            fpr = fp / len(genuine) * 100
            print(f"  FPR        : {fpr:.1f}%  (genuine wrongly failed)")
        if forged:
            tpr = tp / len(forged) * 100
            print(f"  Detection  : {tpr:.1f}%  (forged correctly caught)")
        print(f"{'='*65}")

    # ── Save CSV ──
    if results:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    run()