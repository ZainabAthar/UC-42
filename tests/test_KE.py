import sys
import csv
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix the import path - the file is ke_validator.py not KE_validator.py
from modules.utility_bill.KE_validator import KEBillValidator

GENUINE_DIR = PROJECT_ROOT / "test_data" / "ke_bills" / "genuine"
FORGED_DIR  = PROJECT_ROOT / "test_data" / "ke_bills" / "forged"
OUTPUT_CSV  = PROJECT_ROOT / "outputs" / "ke_validation_results.csv"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".pdf", ".bmp"}


def test_single(file_path: str, cnic_name: str = None, stated_address: str = None):
    """Quick test on a single file - prints full report."""
    validator = KEBillValidator()
    # Fixed: validate_and_print expects stated_address parameter
    validator.validate_and_print(file_path, cnic_name=cnic_name, stated_address=stated_address)


def test_all():
    """Run full test suite on genuine and forged folders."""
    validator = KEBillValidator()

    genuine = sorted([p for p in GENUINE_DIR.rglob("*")
                      if p.suffix.lower() in IMAGE_EXTS]) \
              if GENUINE_DIR.exists() else []

    forged  = sorted([p for p in FORGED_DIR.rglob("*")
                      if p.suffix.lower() in IMAGE_EXTS]) \
              if FORGED_DIR.exists() else []

    if not genuine and not forged:
        print("\n[Test] No images found.")
        print(f"  Put genuine bills in: {GENUINE_DIR}")
        print(f"  Put forged bills in:  {FORGED_DIR}")
        return

    print(f"[Test] Genuine: {len(genuine)} | Forged: {len(forged)}")

    results = []
    fp = fn = tp = tn = 0

    # Genuine bills
    if genuine:
        print(f"\n{'='*65}")
        print(f"  GENUINE BILLS (expect PASS or REVIEW or RESUBMIT)")
        print(f"{'='*65}")
        for p in genuine:
            r = validator.validate(str(p))

            # Genuine bill should NOT be FAIL
            correct = r["verdict"] != "FAIL"
            if correct:
                tn += 1
                status = "[CORRECT]"
            else:
                fp += 1
                status = "[FALSE POSITIVE]"

            print(f"  {status} | {r['verdict']:8} | "
                  f"Score:{r['confidence']:3} | {p.name}")

            # Show barcode result if decoded
            bd = r.get("barcode_data", {})
            if bd.get("account"):
                print(f"           Barcode: acct={bd.get('account')} "
                      f"amt={bd.get('amount')} due={bd.get('due_date')}")

            for f in r["flags"]:
                print(f"           >> {f}")

            results.append({
                "file":     p.name,
                "label":    "GENUINE",
                "verdict":  r["verdict"],
                "score":    r["confidence"],
                "correct":  correct,
                "barcode":  bd.get("account", ""),
                "flags":    "; ".join(r["flags"]),
                "warnings": "; ".join(r["warnings"]),
            })

    # Forged bills
    if forged:
        print(f"\n{'='*65}")
        print(f"  FORGED BILLS (expect FAIL or REVIEW)")
        print(f"{'='*65}")
        for p in forged:
            r = validator.validate(str(p))

            correct = r["verdict"] != "PASS"
            if correct:
                tp += 1
                status = "[DETECTED]"
            else:
                fn += 1
                status = "[MISSED]"

            print(f"  {status} | {r['verdict']:8} | "
                  f"Score:{r['confidence']:3} | {p.name}")
            for f in r["flags"]:
                print(f"           >> {f}")

            bd = r.get("barcode_data", {})
            results.append({
                "file":     p.name,
                "label":    "FORGED",
                "verdict":  r["verdict"],
                "score":    r["confidence"],
                "correct":  correct,
                "barcode":  bd.get("account", ""),
                "flags":    "; ".join(r["flags"]),
                "warnings": "; ".join(r["warnings"]),
            })

    # Summary
    total = len(results)
    if total > 0:
        accuracy = sum(1 for r in results if r["correct"]) / total * 100
        print(f"\n{'='*65}")
        print(f"  SUMMARY")
        print(f"{'='*65}")
        print(f"  Total tested : {total}")
        print(f"  Accuracy     : {accuracy:.1f}%")

        if genuine:
            fpr = fp / len(genuine) * 100
            print(f"  FPR          : {fpr:.1f}%  (genuine wrongly failed)")
            if fpr == 0:
                rating = "EXCELLENT"
            elif fpr <= 5:
                rating = "GOOD"
            elif fpr <= 10:
                rating = "ACCEPTABLE"
            else:
                rating = "TOO HIGH - adjust thresholds"
            print(f"  Rating       : {rating}")

        if forged:
            tpr = tp / len(forged) * 100
            print(f"  Detection    : {tpr:.1f}%  (forged correctly caught)")
        print(f"{'='*65}")

    # Save CSV
    if results:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="Test a single file")
    parser.add_argument("--cnic", type=str, default=None,
                        help="CNIC name for cross-check")
    parser.add_argument("--address", type=str, default=None,
                        help="Stated address for cross-check")
    args = parser.parse_args()

    if args.file:
        test_single(args.file, cnic_name=args.cnic, stated_address=args.address)
    else:
        test_all()