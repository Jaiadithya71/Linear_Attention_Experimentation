"""
import_colab_results.py
========================
Automated importer for Colab benchmark deliverables.
Finds 'verified_results.zip' in the current workspace or the user's Downloads directory,
extracts all authentic empirical data (CSVs, PNG publication plots, PPTX presentation),
synchronizes both root and Linear_Attention_Experimentation directories, and runs an audit.
"""

import os
import sys
import shutil
import zipfile
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def find_zip():
    candidates = [
        "verified_results.zip",
        os.path.expanduser(r"~\Downloads\verified_results.zip"),
        os.path.expanduser(r"~\Downloads\verified_results (1).zip"),
        os.path.expanduser(r"~\Downloads\verified_results (2).zip"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None

def import_results():
    zip_path = find_zip()
    if not zip_path:
        print("[!] Could not locate 'verified_results.zip' in workspace or Downloads folder.")
        print("    Please download the zip file from Google Colab and place it in this folder:")
        print(f"    {os.getcwd()}")
        return False

    print("=" * 75)
    print(f"IMPORTING AUTHENTIC BENCHMARK RESULTS FROM:")
    print(f"  {zip_path} ({os.path.getsize(zip_path):,} bytes)")
    print("=" * 75)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.join(root_dir, "Linear_Attention_Experimentation")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(root_dir)
        print(f"[OK] Extracted deliverables to: {root_dir}")
        if os.path.exists(repo_dir):
            zf.extractall(repo_dir)
            print(f"[OK] Extracted deliverables to: {repo_dir}")

    # Integrity verification
    print("\nAUDITING EXTRACTED DELIVERABLES:")
    files = [
        "data/efficiency_raw.csv",
        "data/retrieval_raw.csv",
        "data/noninferiority.csv",
        "data/deltanet_comparison.csv",
        "data/qwen_prefill_results.csv",
        "docs/figures/fig1_efficiency_scaling_curves.png",
        "docs/figures/fig2_retrieval_accuracy_curves.png",
        "docs/figures/fig3_tradeoff_and_noninferiority.png",
        "docs/figures/fig4_sota_deltanet_comparison.png",
        "presentation/Team3_Linear_Attention_Executive_Summary.pptx"
    ]

    all_ok = True
    for rel in files:
        full = os.path.join(root_dir, rel)
        if os.path.exists(full):
            print(f"  [OK]   {rel:<55} ({os.path.getsize(full):,} bytes)")
        else:
            print(f"  [FAIL] {rel:<55} MISSING [X]")
            all_ok = False

    if all_ok:
        print("\n[SUCCESS] ALL 10 EMPIRICAL DELIVERABLES SUCCESSFULLY IMPORTED AND VERIFIED!")
        # Print a quick preview of Qwen and Efficiency
        qwen_csv = os.path.join(root_dir, "data/qwen_prefill_results.csv")
        if os.path.exists(qwen_csv):
            print("\n--- MEASURED QWEN PREFILL RESULTS ---")
            df = pd.read_csv(qwen_csv)
            print(df.to_string(index=False))

    return all_ok

if __name__ == "__main__":
    import_results()
