import re


def verify_ux_changes():
    try:
        with open("dashboard/templates/trades.html", encoding="utf-8") as f:
            content = f.read()

        checks = [
            {
                "name": "Label for filter-days",
                "pattern": r'<label\s+for="filter-days">Période</label>',
                "description": "Label 'Période' should have for='filter-days'",
            },
            {
                "name": "Label for filter-action",
                "pattern": r'<label\s+for="filter-action">Action</label>',
                "description": "Label 'Action' should have for='filter-action'",
            },
            {
                "name": "Label for filter-symbol",
                "pattern": r'<label\s+for="filter-symbol">Symbole</label>',
                "description": "Label 'Symbole' should have for='filter-symbol'",
            },
            {
                "name": "Keyboard support on symbol input",
                "pattern": r'id="filter-symbol"[^>]*onkeydown="if\(event\.key\s*===\s*\'Enter\'\)\s*applyFilters\(\)"',
                "description": "Symbol input should trigger applyFilters on Enter key",
            },
            {
                "name": "ARIA label on button",
                "pattern": r'<button[^>]*aria-label="Appliquer les filtres"[^>]*>',
                "description": "Filter button should have aria-label",
            },
        ]

        print("🔍 Verifying UX improvements in dashboard/templates/trades.html...\n")

        passed_count = 0
        for check in checks:
            if re.search(check["pattern"], content):
                print(f"✅ PASS: {check['name']}")
                passed_count += 1
            else:
                print(f"❌ FAIL: {check['name']}")
                print(f"   Expected pattern not found: {check['description']}")

        print(f"\nResults: {passed_count}/{len(checks)} checks passed.")

        if passed_count == len(checks):
            print("\n🎉 All UX improvements verified successfully!")
        else:
            print("\n⚠️  Some checks failed. Please review the implementation.")

    except FileNotFoundError:
        print("❌ Error: dashboard/templates/trades.html not found.")
    except Exception as e:
        print(f"❌ Error during verification: {str(e)}")


if __name__ == "__main__":
    verify_ux_changes()
