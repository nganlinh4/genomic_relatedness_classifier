import sys

print("This script is deprecated. Please use 'scripts/build_report.py <dataset>' instead to generate consolidated results under reports/<dataset>.")
if len(sys.argv) >= 2:
    ds = sys.argv[1]
    print(f"Try: ./.venv/Scripts/python.exe scripts/build_report.py {ds}")