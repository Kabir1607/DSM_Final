import json
import os
import pandas as pd

catalog_path = "data/master_data_catalog.json"
with open(catalog_path, 'r') as f:
    catalog = json.load(f)

keywords = {
    "Autonomy": ["autonom", "status", "type"],
    "Enrollment": ["enroll", "student", "intake"],
    "Diploma/Exits": ["diploma", "certificate", "dropout", "exit"],
    "Vocational": ["vocation", "skill"],
    "Placements": ["placement", "company", "salary"],
    "Infrastructure/Digital": ["infra", "digital", "library", "computer", "internet"],
    "Equity": ["sc", "st", "obc", "pwd", "minority", "category"],
    "Gender/PhD": ["female", "gender", "phd", "doctorate", "women"]
}

# Scan AISHE files specifically
aishe_files = []
for root, dirs, files in os.walk("data/kaggle/aishe"):
    for f in files:
        if f.endswith(".csv"):
            aishe_files.append(os.path.join(root, f))

print("Scanning AISHE tables for keywords...")
found_mapping = {k: set() for k in keywords}

for fpath in aishe_files:
    fname = os.path.basename(fpath).lower()
    for cat, kws in keywords.items():
        if any(kw in fname for kw in kws):
            found_mapping[cat].add(os.path.basename(fpath))

for cat, files in found_mapping.items():
    print(f"\n[{cat}]")
    for f in sorted(list(files))[:10]:
        print(f"  - {f}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")

